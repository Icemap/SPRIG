from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_metrics(metrics_path: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not results:
        return

    out_dir = metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recall@K bar chart for each method
    ks = sorted({k for r in results for k in r["metrics"]["recall_at_k"].keys()})
    ks = [int(k) for k in ks]

    for metric_name in ["recall_at_k", "hit_at_k"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(ks))
        width = 0.25
        for idx, r in enumerate(results):
            vals = [r["metrics"][metric_name][str(k)] for k in ks]
            ax.bar([i + idx * width for i in x], vals, width=width, label=r["method"])
        ax.set_xticks([i + width for i in x])
        ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.set_title(f"{metric_name} comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_name}.png", dpi=200)
        plt.close(fig)

    # Latency bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = [r["method"] for r in results]
    query_times = [r["query_time_sec"] for r in results]
    ax.bar(methods, query_times)
    ax.set_ylabel("query_time_sec")
    ax.set_title("Total query time (lower is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "query_time.png", dpi=200)
    plt.close(fig)

    # Index time bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    index_times = [r["index_time_sec"] for r in results]
    ax.bar(methods, index_times)
    ax.set_ylabel("index_time_sec")
    ax.set_title("Index time")
    fig.tight_layout()
    fig.savefig(out_dir / "index_time.png", dpi=200)
    plt.close(fig)
