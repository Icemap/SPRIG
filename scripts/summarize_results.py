from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def collect_metrics(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    results = payload.get("results", [])
    run_mtime = run_dir.stat().st_mtime
    rows = []
    for r in results:
        latency = r.get("latency_stats", {}) or {}
        row = {
            "run_dir": str(run_dir),
            "run_mtime": run_mtime,
            **config,
            "method": r.get("method"),
            "index_time_sec": r.get("index_time_sec"),
            "query_time_sec": r.get("query_time_sec"),
            "docs": r.get("docs"),
            "queries": r.get("queries"),
            "mrr": r.get("metrics", {}).get("mrr"),
            "latency_mean": latency.get("mean"),
            "latency_p50": latency.get("p50"),
            "latency_p95": latency.get("p95"),
            "latency_p99": latency.get("p99"),
            "rss_peak_mb": r.get("rss_peak_mb"),
            "rss_index_mb": r.get("rss_index_mb"),
            "rss_query_mb": r.get("rss_query_mb"),
            "seed_time_sec": r.get("seed_time_sec"),
            "ppr_time_sec": r.get("ppr_time_sec"),
            "fallback_count": r.get("fallback_count"),
            "fallback_rate": r.get("fallback_rate"),
            "fallback_k": r.get("fallback_k"),
        }
        for k, v in r.get("metrics", {}).get("recall_at_k", {}).items():
            row[f"recall@{k}"] = v
        for k, v in r.get("metrics", {}).get("hit_at_k", {}).items():
            row[f"hit@{k}"] = v
        rows.append(row)
    return rows


def main() -> None:
    outputs = Path("outputs")
    summary_dir = outputs / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in outputs.glob("*/metrics.json"):
        rows.extend(collect_metrics(metrics_path.parent) or [])

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found.")
        return

    df.to_csv(summary_dir / "all_results.csv", index=False)

    # Prefer latest tagged runs (rev3 > rev2 > rev1 > main3 > main2 > main)
    tag_pref = None
    for tag in ["rev3", "rev2", "rev1", "main3", "main2", "main"]:
        if (df["tag"] == tag).any():
            tag_pref = tag
            break

    if tag_pref:
        df_main = df[df["tag"] == tag_pref]
    else:
        df_main = df

    # Main results: full hotpot validation (queries >= 7000) + 2wiki n10000
    main = df_main[
        ((df_main["dataset"] == "hotpotqa") & (df_main["split"] == "validation") & (df_main["queries"] >= 7000))
        | ((df_main["dataset"] == "2wikimultihopqa") & (df_main["max_samples"] == 10000))
    ]
    if "run_mtime" in main.columns:
        main = (
            main.sort_values("run_mtime")
            .groupby(["dataset", "method"], as_index=False)
            .tail(1)
        )
    main.to_csv(summary_dir / "main_results.csv", index=False)

    # LaTeX table for main
    latex_path = summary_dir / "main_results.tex"
    with latex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrrr}\\n")
        f.write("Dataset & Method & R@5 & R@10 & Hit@10 & MRR & QTime \\\\\n")
        f.write("\\hline\\n")
        for _, r in main.iterrows():
            f.write(
                f"{r['dataset']} & {r['method']} & {r.get('recall@5',0):.3f} & {r.get('recall@10',0):.3f} "
                f"& {r.get('hit@10',0):.3f} & {r.get('mrr',0):.3f} & {r.get('query_time_sec',0):.1f} \\\\\n"
            )
        f.write("\\end{tabular}\\n")

    print(f"Wrote {summary_dir}")


if __name__ == "__main__":
    main()
