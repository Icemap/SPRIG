from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    results = payload.get("results", [])
    rows = []
    for r in results:
        rows.append(
            {
                "dataset": config.get("dataset"),
                "split": config.get("split"),
                "max_samples": config.get("max_samples"),
                "method": r.get("method"),
                "docs": r.get("docs"),
                "queries": r.get("queries"),
                "index_time_sec": r.get("index_time_sec"),
                "query_time_sec": r.get("query_time_sec"),
                "tag": config.get("tag"),
            }
        )
    return rows


def main() -> None:
    outputs = Path("outputs")
    rows = []
    for metrics_path in outputs.glob("*/metrics.json"):
        rows.extend(load_metrics(metrics_path.parent))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for efficiency plots")
        return

    if (df["tag"] == "eff2").any():
        df = df[df["tag"] == "eff2"]

    out_dir = outputs / "summary" / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in df["dataset"].dropna().unique():
        sub = df[df["dataset"] == dataset]
        for metric in ["index_time_sec", "query_time_sec"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in sorted(sub["method"].unique()):
                m = sub[sub["method"] == method].sort_values("docs")
                ax.plot(m["docs"], m[metric], marker="o", label=method)
            ax.set_xlabel("#docs")
            ax.set_ylabel(metric)
            ax.set_title(f"{dataset} {metric}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{dataset}_{metric}.png", dpi=200)
            fig.savefig(out_dir / f"{dataset}_{metric}.pdf")
            plt.close(fig)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
