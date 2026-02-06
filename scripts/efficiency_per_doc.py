from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", required=True, help="Path to all_results.csv")
    p.add_argument("--tag", default="eff2")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "graph", "graph_dense"],
    )
    p.add_argument("--output", required=True, help="Output LaTeX table path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    if "tag" in df.columns:
        df = df[df["tag"] == args.tag]

    rows = []
    for dataset in sorted(df["dataset"].dropna().unique()):
        for method in args.methods:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if sub.empty:
                continue
            per_doc = (sub["index_time_sec"] / sub["docs"]) * 1000.0
            rows.append((dataset, method, per_doc.median(), per_doc.min(), per_doc.max()))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrr}\n")
        f.write("Dataset & Method & Median ms/doc & Min--Max ms/doc \\\\ \n")
        f.write("\\hline\n")
        for dataset, method, med, lo, hi in rows:
            f.write(f"{dataset} & {method} & {med:.3f} & [{lo:.3f}, {hi:.3f}] \\\\ \n")
        f.write("\\end{tabular}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
