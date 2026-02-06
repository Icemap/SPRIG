from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from sprig.data.hotpotqa import load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.eval import evaluate
from sprig.retrieval.graph import build_term_bipartite_graph, ppr_search
from sprig.utils.text import normalize_whitespace, tokenize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-df", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument("--max-df-ratio", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--ppr-alpha", type=float, default=0.15)
    p.add_argument("--ppr-max-iter", type=int, default=5)
    p.add_argument("--limit-queries", type=int, default=500)
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
    docs = [(d.doc_id, normalize_whitespace(d.text)) for d in docs]
    queries = [(q.qid, normalize_whitespace(q.question), list(q.gold_titles)) for q in queries]

    if args.limit_queries is not None and args.limit_queries < len(queries):
        import random

        random.seed(args.seed)
        queries = random.sample(queries, args.limit_queries)

    doc_ids = [d for d, _ in docs]
    doc_texts = [t for _, t in docs]
    gold = {qid: gold_titles for qid, _, gold_titles in queries}
    top_k_max = 10

    csv_path = out_dir / "ablation_results.csv"
    done_keys = set()
    if args.resume and csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                done_keys.add((r.get("min_df"), r.get("max_df_ratio")))

    write_header = not csv_path.exists()
    rows = []
    for min_df in args.min_df:
        for max_df_ratio in args.max_df_ratio:
            key = (str(min_df), str(max_df_ratio))
            if key in done_keys:
                continue
            t0 = time.perf_counter()
            term_graph = build_term_bipartite_graph(
                doc_ids,
                doc_texts,
                min_df=min_df,
                max_df_ratio=max_df_ratio,
                normalization="row",
            )
            index_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            retrieved = {}
            for qid, question, _ in queries:
                seed_terms = tokenize(question)
                hits = ppr_search(
                    term_graph,
                    question,
                    top_k=top_k_max,
                    seed_terms=seed_terms,
                    alpha=args.ppr_alpha,
                    max_iter=args.ppr_max_iter,
                    mode="power",
                )
                retrieved[qid] = [doc_ids[i] for i, _ in hits]
            query_time = time.perf_counter() - t1
            metrics = evaluate(gold, retrieved, ks=[1, 3, 5, 10])
            row = {
                "dataset": args.dataset,
                "split": args.split,
                "max_samples": args.max_samples,
                "docs": len(docs),
                "queries": len(queries),
                "min_df": min_df,
                "max_df_ratio": max_df_ratio,
                "ppr_alpha": args.ppr_alpha,
                "ppr_max_iter": args.ppr_max_iter,
                "index_time_sec": index_time,
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

    all_rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append(r)

    with (out_dir / "ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    best = sorted(all_rows, key=lambda r: float(r.get("recall@10", 0.0)), reverse=True)[:10]
    with (out_dir / "ablation_top10.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"Ablation done. Wrote {csv_path}")


if __name__ == "__main__":
    main()
