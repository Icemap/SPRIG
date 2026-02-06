from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

from sprig.data.hotpotqa import load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.eval import evaluate
from sprig.retrieval.dense import DenseRetriever
from sprig.retrieval.graph import build_bipartite_graph, ppr_search
from sprig.utils.text import normalize_whitespace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed-source", choices=["dense", "bm25"], default="dense")
    p.add_argument("--seed-k", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--seed-weighting", nargs="+", default=["raw"])
    p.add_argument("--seed-temp", type=float, nargs="+", default=[1.0])
    p.add_argument("--ner", nargs="+", default=["regex", "spacy"])
    p.add_argument("--entity-normalize", nargs="+", default=["none"])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.15, 0.2])
    p.add_argument("--iter", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--ppr-mode", nargs="+", default=["power"])
    p.add_argument("--ppr-tol", type=float, nargs="+", default=[0.0])
    p.add_argument("--ppr-push-eps", type=float, default=1e-4)
    p.add_argument("--ppr-push-max-steps", type=int, default=200000)
    p.add_argument("--top-ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--graph-min-df", type=int, default=1)
    p.add_argument("--graph-max-df-ratio", type=float, default=1.0)
    p.add_argument("--graph-hub-penalty", type=float, nargs="+", default=[0.0])
    p.add_argument("--graph-norm", default="row", choices=["row", "sym", "none"])
    p.add_argument("--seed-entity-df-power", type=float, nargs="+", default=[0.0])
    p.add_argument("--cache-root", default="outputs/_cache")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_dataset(dataset: str, split: str, max_samples: int | None, seed: int):
    if dataset == "hotpotqa":
        return load_hotpotqa(split=split, max_samples=max_samples, seed=seed)
    return load_twowiki(split=split, max_samples=max_samples, seed=seed)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs, queries = load_dataset(args.dataset, args.split, args.max_samples, args.seed)
    docs = [
        (d.doc_id, normalize_whitespace(d.text))
        for d in docs
    ]
    queries = [(q.qid, normalize_whitespace(q.question), list(q.gold_titles)) for q in queries]

    if args.limit_queries is not None and args.limit_queries < len(queries):
        import random

        random.seed(args.seed)
        queries = random.sample(queries, args.limit_queries)

    doc_ids = [d for d, _ in docs]
    doc_texts = [t for _, t in docs]
    gold = {qid: gold_titles for qid, _, gold_titles in queries}

    top_k_max = max(args.top_ks)
    dense_hits = {}
    bm25_hits = {}
    index_time_base = 0.0

    if args.seed_source == "dense":
        cache_root = Path(args.cache_root) / "dense" / args.dataset / args.split / (
            f"n{args.max_samples}" if args.max_samples is not None else "nfull"
        ) / f"s{args.seed}"
        t0 = time.perf_counter()
        dense = DenseRetriever(args.dense_model, cache_dir=cache_root, use_hnsw=True)
        dense_index = dense.build_index(doc_ids, doc_texts)
        index_time_base = time.perf_counter() - t0
        for qid, question, _ in queries:
            dense_hits[qid] = dense.search(dense_index, question, top_k=top_k_max)
    else:
        from sprig.retrieval.bm25 import BM25Index

        t0 = time.perf_counter()
        bm25 = BM25Index.build(doc_texts)
        index_time_base = time.perf_counter() - t0
        for qid, question, _ in queries:
            bm25_hits[qid] = bm25.search(question, top_k=top_k_max)

    # Graph indexes per NER / normalization / hub penalty
    graph_indexes = {}
    graph_index_times = {}
    for ner in args.ner:
        for normalize in args.entity_normalize:
            for hub_penalty in args.graph_hub_penalty:
                key = (ner, normalize, hub_penalty)
                t1 = time.perf_counter()
                graph_indexes[key] = build_bipartite_graph(
                    doc_ids,
                    doc_texts,
                    ner_mode=ner,
                    entity_normalize=normalize,
                    min_entity_df=args.graph_min_df,
                    max_entity_df_ratio=args.graph_max_df_ratio,
                    hub_penalty=hub_penalty,
                    normalization=args.graph_norm,
                )
                graph_index_times[key] = time.perf_counter() - t1

    rows = []
    csv_path = out_dir / "ablation_results.csv"
    done_keys = set()
    if args.resume and csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (
                    r.get("ner"),
                    r.get("seed_k"),
                    r.get("alpha"),
                    r.get("max_iter"),
                    r.get("seed_weighting"),
                    r.get("seed_temp"),
                    r.get("ppr_mode"),
                    r.get("ppr_tol"),
                    r.get("entity_normalize"),
                    r.get("graph_hub_penalty"),
                    r.get("seed_entity_df_power"),
                    r.get("seed_source"),
                )
                done_keys.add(key)
    write_header = not csv_path.exists()

    from itertools import product

    for (
        ner,
        normalize,
        hub_penalty,
        alpha,
        max_iter,
        seed_k,
        seed_weighting,
        seed_temp,
        ppr_mode,
        ppr_tol,
        seed_df_power,
    ) in product(
        args.ner,
        args.entity_normalize,
        args.graph_hub_penalty,
        args.alpha,
        args.iter,
        args.seed_k,
        args.seed_weighting,
        args.seed_temp,
        args.ppr_mode,
        args.ppr_tol,
        args.seed_entity_df_power,
    ):
        key = (
            str(ner),
            str(seed_k),
            str(alpha),
            str(max_iter),
            str(seed_weighting),
            str(seed_temp),
            str(ppr_mode),
            str(ppr_tol),
            str(normalize),
            str(hub_penalty),
            str(seed_df_power),
            str(args.seed_source),
        )
        if key in done_keys:
            continue
        graph_key = (ner, normalize, hub_penalty)
        graph_index = graph_indexes[graph_key]
        t2 = time.perf_counter()
        retrieved = {}
        tol = None if ppr_tol <= 0 else ppr_tol
        for qid, question, _ in queries:
            if args.seed_source == "dense":
                seed_docs = dense_hits[qid][:seed_k]
            else:
                seed_docs = bm25_hits[qid][:seed_k]
            hits = ppr_search(
                graph_index,
                question,
                top_k=top_k_max,
                ner_mode=ner,
                entity_normalize=normalize,
                alpha=alpha,
                max_iter=max_iter,
                seed_docs=seed_docs,
                seed_weighting=seed_weighting,
                seed_temp=seed_temp,
                mode=ppr_mode,
                tol=tol,
                push_eps=args.ppr_push_eps,
                push_max_steps=args.ppr_push_max_steps,
                seed_entity_df_power=seed_df_power,
            )
            retrieved[qid] = [doc_ids[i] for i, _ in hits]
        query_time = time.perf_counter() - t2
        metrics = evaluate(gold, retrieved, ks=args.top_ks)

        row = {
            "dataset": args.dataset,
            "split": args.split,
            "max_samples": args.max_samples,
            "docs": len(docs),
            "queries": len(queries),
            "method": f"graph_{args.seed_source}",
            "seed_source": args.seed_source,
            "ner": ner,
            "entity_normalize": normalize,
            "graph_hub_penalty": hub_penalty,
            "seed_k": seed_k,
            "seed_weighting": seed_weighting,
            "seed_temp": seed_temp,
            "seed_entity_df_power": seed_df_power,
            "alpha": alpha,
            "max_iter": max_iter,
            "ppr_mode": ppr_mode,
            "ppr_tol": tol,
            "ppr_push_eps": args.ppr_push_eps,
            "ppr_push_max_steps": args.ppr_push_max_steps,
            "index_time_sec": index_time_base + graph_index_times[graph_key],
            "query_time_sec": query_time,
            "mrr": metrics.mrr,
        }
        for k, v in metrics.recall_at_k.items():
            row[f"recall@{k}"] = v
        for k, v in metrics.hit_at_k.items():
            row[f"hit@{k}"] = v
        rows.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

    # CSV already updated incrementally above.

    # Reload all rows from CSV so resume runs are merged.
    all_rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append(r)

    # Save JSON for programmatic use
    with (out_dir / "ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # Best config summary
    best = sorted(all_rows, key=lambda r: float(r.get("recall@10", 0.0)), reverse=True)[:10]
    with (out_dir / "ablation_top10.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # Simple LaTeX table with top 10
    tex_path = out_dir / "ablation_top10.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\\n")
        f.write("Dataset & NER & k & $\\alpha$ & it & R@5 & R@10 & MRR & QTime \\\\\\n")
        f.write("\\hline\\n")
        for r in best:
            f.write(
                f"{r['dataset']} & {r['ner']} & {r['seed_k']} & {float(r['alpha']):.2f} & {r['max_iter']} "
                f"& {float(r.get('recall@5',0)):.3f} & {float(r.get('recall@10',0)):.3f} "
                f"& {float(r.get('mrr',0)):.3f} & {float(r.get('query_time_sec',0)):.1f} \\\\\n"
            )
        f.write("\\end{tabular}\\n")

    print(f"Ablation done. Wrote {csv_path}")


if __name__ == "__main__":
    main()
