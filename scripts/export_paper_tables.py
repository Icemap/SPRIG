from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _pick_main_runs(tag: str) -> dict[str, Path]:
    outputs = Path("outputs")
    candidates: dict[str, list[Path]] = {"hotpotqa": [], "2wikimultihopqa": []}
    for metrics_path in outputs.glob("*/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if config.get("tag") != tag:
            continue
        dataset = config.get("dataset")
        if dataset in candidates:
            candidates[dataset].append(metrics_path.parent)

    runs: dict[str, Path] = {}
    for dataset, dirs in candidates.items():
        if not dirs:
            continue
        best = None
        best_score = (-1, -1.0)
        for run_dir in dirs:
            score = 0
            for extra in ["graph_stats.json", "ner_proxy.json", "hub_pruning.json"]:
                if (run_dir / extra).exists():
                    score += 1
            mtime = run_dir.stat().st_mtime
            key = (score, mtime)
            if key > best_score:
                best_score = key
                best = run_dir
        if best is not None:
            runs[dataset] = best
    return runs


def _write_table(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{%s}\n" % header)
        for row in rows:
            f.write(row + "\n")
        f.write("\\end{tabular}\n")


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


_METHOD_ORDER = [
    "bm25",
    "rm3",
    "bm25_2step",
    "dense",
    "rrf",
    "rerank",
    "tfidf_graph",
    "graph",
    "graph_bm25_fallback",
    "graph_hybrid",
    "graph_dense",
]

_METHOD_LABELS = {
    "bm25": "BM25",
    "rm3": "BM25+RM3",
    "bm25_2step": "BM25-2step",
    "dense": "Dense",
    "rrf": "RRF",
    "rerank": "BM25+CE",
    "rrf_ppr_fusion": "RRF+PPR (fusion)",
    "rrf_rerank": "RRF+CE",
    "tfidf_graph": "TF-IDF Graph",
    "graph": "Graph",
    "graph_bm25_fallback": "Graph+BM25 fallback",
    "graph_hybrid": "GraphHybrid",
    "graph_rrf": "GraphRRF",
    "graph_dense": "GraphDense",
}

_DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
}


def _order_methods(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(_METHOD_ORDER)}
    return df.assign(_order=df["method"].map(order).fillna(9999)).sort_values("_order")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="rev2")
    p.add_argument("--supp-tag", default=None)
    p.add_argument("--ann-tag", default=None)
    p.add_argument("--dense-tag", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    supp_tag = args.supp_tag or args.tag
    ann_tag = args.ann_tag or supp_tag
    dense_tag = args.dense_tag or supp_tag
    paper_tables = Path("paper") / "tables"
    main = pd.read_csv("outputs/summary/main_results.csv")
    if "tag" in main.columns:
        main = main[main["tag"] == args.tag]
    all_results = pd.read_csv("outputs/summary/all_results.csv")
    if "run_mtime" in all_results.columns:
        all_results["run_mtime"] = pd.to_numeric(all_results["run_mtime"], errors="coerce")

    # Main retrieval tables
    for dataset in ["hotpotqa", "2wikimultihopqa"]:
        sub = main[(main["dataset"] == dataset) & (main["method"].isin(_METHOD_ORDER))].copy()
        sub = _order_methods(sub)
        rows = ["Method & R@5 & R@10 & Hit@10 & MRR & QTime \\\\", "\\hline"]
        for _, r in sub.iterrows():
            label = _METHOD_LABELS.get(r["method"], r["method"])
            rows.append(
                f"{_tex_escape(label)} & {r.get('recall@5',0):.3f} & {r.get('recall@10',0):.3f} "
                f"& {r.get('hit@10',0):.3f} & {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
            )
        _write_table(
            paper_tables / f"main_results_{dataset}.tex",
            "lrrrrr",
            rows,
        )

    # Latency stats table (p50/p95)
    rows = ["Dataset & Method & p50 (s) & p95 (s) & p99 (s) \\\\", "\\hline"]
    for _, r in main.iterrows():
        label = _METHOD_LABELS.get(r["method"], r["method"])
        rows.append(
            f"{_tex_escape(_DATASET_LABELS.get(r['dataset'], r['dataset']))} & {_tex_escape(label)} "
            f"& {r.get('latency_p50',0):.3f} "
            f"& {r.get('latency_p95',0):.3f} & {r.get('latency_p99',0):.3f} \\\\"
        )
    _write_table(paper_tables / "latency_stats.tex", "llrrr", rows)

    # Memory breakdown table (index/query RSS)
    rows = ["Dataset & Method & RSS index (MB) & RSS query (MB) \\\\", "\\hline"]
    for _, r in main.iterrows():
        label = _METHOD_LABELS.get(r["method"], r["method"])
        rows.append(
            f"{_tex_escape(_DATASET_LABELS.get(r['dataset'], r['dataset']))} & {_tex_escape(label)} "
            f"& {r.get('rss_index_mb',0):.1f} & {r.get('rss_query_mb',0):.1f} \\\\"
        )
    _write_table(paper_tables / "rss_stats.tex", "llrr", rows)

    # Graph timing breakdown (seed vs PPR)
    timing_methods = {"graph_hybrid", "graph_rrf", "graph_dense", "rrf_ppr_fusion"}
    timing = main[main["method"].isin(timing_methods)].copy()
    if not timing.empty:
        rows = ["Dataset & Method & Seed (s) & PPR (s) & QTime (s) \\\\", "\\hline"]
        for _, r in timing.iterrows():
            label = _METHOD_LABELS.get(r["method"], r["method"])
            rows.append(
                f"{_tex_escape(_DATASET_LABELS.get(r['dataset'], r['dataset']))} & {_tex_escape(label)} "
                f"& {r.get('seed_time_sec',0):.1f} & {r.get('ppr_time_sec',0):.1f} "
                f"& {r.get('query_time_sec',0):.1f} \\\\"
            )
        _write_table(paper_tables / "graph_timing_breakdown.tex", "llrrr", rows)

    # Graph stats
    runs = _pick_main_runs(tag=args.tag)
    rows = [
        "Dataset & Graph & Nodes & Edges & Deg$_{p95}$ (E) & Deg$_{p95}$ (D) \\\\",
        "\\hline",
    ]
    for dataset, run_dir in runs.items():
        stats_path = run_dir / "graph_stats.json"
        if not stats_path.exists():
            alias = {"hotpotqa": "hotpot", "2wikimultihopqa": "2wiki"}.get(dataset, dataset)
            fallback = Path(f"outputs/rev2_{alias}_graph_only/graph_stats.json")
            if fallback.exists():
                stats_path = fallback
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for graph_name, g in stats.items():
            rows.append(
                f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {_tex_escape(graph_name)} "
                f"& {g['nodes']} & {g['edges']} "
                f"& {g['entity_degree']['p95']:.0f} & {g['doc_degree']['p95']:.0f} \\\\"
            )
    _write_table(paper_tables / "graph_stats.tex", "llrrrr", rows)

    # NER proxy table (query entity coverage buckets)
    rows = [
        "Dataset & NER & Bucket & Method & \\%Queries & R@10 & MRR \\\\",
        "\\hline",
    ]
    for dataset, run_dir in runs.items():
        ner_path = run_dir / "ner_proxy.json"
        if not ner_path.exists():
            continue
        payload = json.loads(ner_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if row.get("method") not in {"graph", "graph_hybrid"}:
                continue
            rows.append(
                f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {_tex_escape(row.get('ner_mode',''))} "
                f"& {_tex_escape(row.get('bucket',''))} & {_tex_escape(_METHOD_LABELS.get(row.get('method'), row.get('method')))} "
                f"& {row.get('rate',0)*100:.1f} & {row.get('recall@10',0):.3f} & {row.get('mrr',0):.3f} \\\\"
            )
    if len(rows) > 2:
        _write_table(paper_tables / "ner_proxy_stats.tex", "llllrrr", rows)

    # Hub pruning coverage table
    rows = [
        "Dataset & Hub Top\\% & Gold Removed \\% & Queries Affected \\% \\\\",
        "\\hline",
    ]
    for dataset, run_dir in runs.items():
        hub_path = run_dir / "hub_pruning.json"
        if not hub_path.exists():
            continue
        payload = json.loads(hub_path.read_text(encoding="utf-8"))
        rows.append(
            f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} "
            f"& {payload.get('graph_hub_top_ratio',0)*100:.1f} "
            f"& {payload.get('gold_removed_rate',0)*100:.2f} "
            f"& {payload.get('queries_affected_rate',0)*100:.2f} \\\\"
        )
    if len(rows) > 2:
        _write_table(paper_tables / "hub_pruning_stats.tex", "lrrr", rows)

    # QA results table
    rows = ["Dataset & Method & EM & F1 & N \\\\", "\\hline"]
    for dataset, run_dir in runs.items():
        qa_path = run_dir / "qa_metrics.json"
        if not qa_path.exists():
            alias = {"hotpotqa": "hotpot", "2wikimultihopqa": "2wiki"}.get(dataset, dataset)
            merged = Path(f"outputs/rev2_{alias}_merged/qa_metrics.json")
            if merged.exists():
                qa_path = merged
        if not qa_path.exists():
            continue
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        for method, vals in qa.items():
            rows.append(
                f"{_tex_escape(dataset)} & {_tex_escape(method)} & {vals.get('em',0):.3f} & {vals.get('f1',0):.3f} & {vals.get('n',0)} \\\\"
            )
    _write_table(paper_tables / "qa_results.tex", "llrrr", rows)

    # Exact vs ANN dense seeding (GraphDense, 2k subset)
    subset_default = all_results[
        (all_results["tag"] == args.tag)
        & (all_results["method"] == "graph_dense")
        & (all_results["max_samples"] == 2000)
    ]
    dense_use_hnsw = all_results.get("dense_use_hnsw")
    if dense_use_hnsw is not None:
        dense_use_hnsw = dense_use_hnsw.astype(str).str.lower() == "true"
    else:
        dense_use_hnsw = pd.Series([False] * len(all_results))

    subset_tuned = all_results[
        (all_results["tag"] == ann_tag)
        & (all_results["method"] == "graph_dense")
        & (all_results["max_samples"] == 2000)
        & dense_use_hnsw
    ]
    if not subset_default.empty:
        rows = ["Dataset & Seeding & R@10 & Hit@10 & MRR & QTime \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            dataset_rows = []
            sub = subset_default[subset_default["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("dense_use_hnsw", ascending=False)
            for _, r in sub.iterrows():
                seeding = "ANN (default)" if str(r.get("dense_use_hnsw", True)).lower() == "true" else "Exact"
                dataset_rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {seeding} "
                    f"& {r.get('recall@10',0):.3f} "
                    f"& {r.get('hit@10',0):.3f} & {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
                )
            tuned = subset_tuned[subset_tuned["dataset"] == dataset].copy()
            if not tuned.empty:
                tuned = tuned.sort_values("recall@10", ascending=False).head(1)
                for _, r in tuned.iterrows():
                    dataset_rows.insert(
                        1 if len(dataset_rows) > 1 else 0,
                        f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & ANN (tuned) "
                        f"& {r.get('recall@10',0):.3f} "
                        f"& {r.get('hit@10',0):.3f} & {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\",
                    )
            rows.extend(dataset_rows)
        _write_table(paper_tables / "graph_dense_seeding.tex", "llrrrr", rows)

    # ANN tuning grid (GraphDense, 2k subset)
    ann_grid = all_results[
        (all_results["tag"] == ann_tag)
        & (all_results["method"] == "graph_dense")
        & (all_results["max_samples"] == 2000)
        & dense_use_hnsw
    ]
    if not ann_grid.empty:
        rows = ["Dataset & M & efSearch & R@10 & QTime \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            sub = ann_grid[ann_grid["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["hnsw_m", "hnsw_ef_search"])
            for _, r in sub.iterrows():
                rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {int(r.get('hnsw_m',0) or 0)} "
                    f"& {int(r.get('hnsw_ef_search',0) or 0)} & {r.get('recall@10',0):.3f} "
                    f"& {r.get('query_time_sec',0):.1f} \\\\"
                )
        _write_table(paper_tables / "graph_dense_ann_tuning.tex", "lrrrr", rows)

    # GraphRRF ablation (full validation)
    graph_rrf = all_results[
        (all_results["tag"] == supp_tag)
        & (all_results["method"].isin(["graph_rrf", "rrf_ppr_fusion", "rrf_rerank"]))
    ]
    if not graph_rrf.empty:
        rows = ["Dataset & Method & R@10 & Hit@10 & MRR & QTime \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            base = main[
                (main["dataset"] == dataset)
                & (main["method"].isin(["rrf", "graph_hybrid", "graph_dense"]))
            ].copy()
            for _, r in base.iterrows():
                label = _METHOD_LABELS.get(r["method"], r["method"])
                rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {_tex_escape(label)} "
                    f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                    f"& {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
                )
            sub = graph_rrf[graph_rrf["dataset"] == dataset].copy()
            if not sub.empty:
                sub = sub.sort_values("run_mtime").tail(3)
                for _, r in sub.iterrows():
                    label = _METHOD_LABELS.get(r["method"], r["method"])
                    rows.append(
                        f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {_tex_escape(label)} "
                        f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                        f"& {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
                    )
        _write_table(paper_tables / "graph_rrf_ablation.tex", "llrrrr", rows)

    # Graph fallback ablation (full validation)
    graph_fallback = all_results[
        (all_results["tag"] == supp_tag) & (all_results["method"] == "graph_bm25_fallback")
    ]
    if not graph_fallback.empty:
        rows = ["Dataset & Method & R@10 & Hit@10 & MRR & Fallback \\% \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            base = main[(main["dataset"] == dataset) & (main["method"] == "graph")].copy()
            for _, r in base.iterrows():
                rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & Graph "
                    f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                    f"& {r.get('mrr',0):.3f} & -- \\\\"
                )
            sub = graph_fallback[graph_fallback["dataset"] == dataset].copy()
            if not sub.empty:
                sub = sub.sort_values("run_mtime").tail(1)
                for _, r in sub.iterrows():
                    rate = r.get("fallback_rate", 0.0) * 100.0
                    rows.append(
                        f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & Graph+BM25 fallback "
                        f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                        f"& {r.get('mrr',0):.3f} & {rate:.1f} \\\\"
                    )
        _write_table(paper_tables / "graph_fallback.tex", "llrrrl", rows)

    # Dense model sensitivity (2k subset)
    dense_sens = all_results[
        (all_results["tag"] == dense_tag)
        & (all_results["max_samples"] == 2000)
        & (all_results["method"].isin(["dense", "graph_dense"]))
    ]
    if not dense_sens.empty:
        rows = ["Dataset & Model & Method & R@10 & MRR & QTime \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            sub = dense_sens[dense_sens["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["dense_model", "method"])
            for _, r in sub.iterrows():
                model = str(r.get("dense_model", ""))
                model = model.split("/")[-1] if model else model
                rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & {_tex_escape(model)} "
                    f"& {_tex_escape(_METHOD_LABELS.get(r['method'], r['method']))} "
                    f"& {r.get('recall@10',0):.3f} & {r.get('mrr',0):.3f} "
                    f"& {r.get('query_time_sec',0):.1f} \\\\"
                )
        _write_table(paper_tables / "dense_model_sensitivity.tex", "lllrrr", rows)

    # Tuned ANN vs default on full validation (GraphDense)
    tuned_full = all_results[
        (all_results["tag"] == supp_tag)
        & (all_results["method"] == "graph_dense")
        & (all_results["max_samples"].fillna(0) >= 7000)
    ]
    default_full = main[main["method"] == "graph_dense"].copy()
    if not tuned_full.empty and not default_full.empty:
        rows = ["Dataset & Seeding & R@10 & Hit@10 & MRR & QTime \\\\", "\\hline"]
        for dataset in ["hotpotqa", "2wikimultihopqa"]:
            base = default_full[default_full["dataset"] == dataset].copy()
            for _, r in base.iterrows():
                rows.append(
                    f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & ANN (default) "
                    f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                    f"& {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
                )
            sub = tuned_full[tuned_full["dataset"] == dataset].copy()
            if not sub.empty:
                sub = sub.sort_values("run_mtime").tail(1)
                for _, r in sub.iterrows():
                    rows.append(
                        f"{_tex_escape(_DATASET_LABELS.get(dataset, dataset))} & ANN (tuned) "
                        f"& {r.get('recall@10',0):.3f} & {r.get('hit@10',0):.3f} "
                        f"& {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\"
                    )
        _write_table(paper_tables / "graph_dense_tuned_full.tex", "llrrrr", rows)

    print(f"Wrote tables to {paper_tables}")


if __name__ == "__main__":
    main()
