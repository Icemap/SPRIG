from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Ablation CSVs. Defaults to outputs/ablation_*/ablation_results.csv",
    )
    p.add_argument("--output", default="outputs/summary/ablation")
    return p.parse_args()


def load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_dir"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_seed_k(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if "ner" not in df.columns or "seed_k" not in df.columns:
        return
    df = df.dropna(subset=["ner", "seed_k", "recall@10"])
    if df.empty:
        return
    df["ner"] = df["ner"].astype(str)
    df["seed_k"] = df["seed_k"].astype(int)
    best = (
        df.groupby(["ner", "seed_k"], as_index=False)["recall@10"]
        .max()
        .sort_values(["ner", "seed_k"])
    )
    if best.empty:
        return
    plt.figure(figsize=(6.5, 4.2))
    for ner in sorted(best["ner"].unique()):
        sub = best[best["ner"] == ner]
        plt.plot(
            sub["seed_k"],
            sub["recall@10"],
            marker="o",
            label=f"NER={ner}",
        )
    plt.xlabel("Seed k")
    plt.ylabel("Recall@10")
    plt.title(f"Ablation: Seed-k vs Recall@10 ({dataset})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"ablation_{dataset}_seedk_recall10.{ext}", bbox_inches="tight")
    plt.close()


def plot_alpha_iter_heatmap(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if "ner" not in df.columns or "seed_k" not in df.columns:
        return
    df = df.dropna(subset=["ner", "seed_k", "recall@10", "alpha", "max_iter"])
    if df.empty:
        return
    df["ner"] = df["ner"].astype(str)
    df["seed_k"] = df["seed_k"].astype(int)
    df["alpha"] = df["alpha"].astype(float)
    df["max_iter"] = df["max_iter"].astype(int)
    for ner in sorted(df["ner"].unique()):
        sub = df[df["ner"] == ner]
        if sub.empty:
            continue
        # Choose the seed_k that yields best recall@10 for this NER.
        seed_best = (
            sub.groupby("seed_k", as_index=False)["recall@10"].max().sort_values("recall@10", ascending=False)
        )
        if seed_best.empty:
            continue
        best_seed_k = int(seed_best.iloc[0]["seed_k"])
        grid = sub[sub["seed_k"] == best_seed_k]
        if grid.empty:
            continue
        pivot = (
            grid.pivot_table(
                index="alpha",
                columns="max_iter",
                values="recall@10",
                aggfunc="max",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        if pivot.empty:
            continue
        data = pivot.values
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{a:.2f}" for a in pivot.index])
        ax.set_xlabel("PPR max_iter")
        ax.set_ylabel("PPR alpha")
        ax.set_title(f"Ablation: Alpha/Iter (NER={ner}, seed_k={best_seed_k}) ({dataset})")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Recall@10")
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "pdf"]:
            fig.savefig(
                out_dir / f"ablation_{dataset}_alpha_iter_ner-{ner}_seedk-{best_seed_k}.{ext}",
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_seed_weighting(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if "seed_weighting" not in df.columns:
        return
    df = df.dropna(subset=["seed_weighting", "recall@10"])
    if df.empty:
        return
    best = (
        df.groupby(["seed_weighting"], as_index=False)["recall@10"]
        .max()
        .sort_values("recall@10", ascending=False)
    )
    if best.empty:
        return
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(best["seed_weighting"], best["recall@10"])
    plt.xlabel("Seed weighting")
    plt.ylabel("Recall@10 (best)")
    plt.title(f"Ablation: Seed weighting ({dataset})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"ablation_{dataset}_seed_weighting_recall10.{ext}", bbox_inches="tight")
    plt.close()


def plot_hub_penalty(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if "graph_hub_penalty" not in df.columns:
        return
    df = df.dropna(subset=["graph_hub_penalty", "recall@10"])
    if df.empty:
        return
    best = (
        df.groupby(["graph_hub_penalty"], as_index=False)["recall@10"]
        .max()
        .sort_values("graph_hub_penalty")
    )
    if best.empty:
        return
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(best["graph_hub_penalty"], best["recall@10"], marker="o")
    plt.xlabel("Hub penalty (df^-p)")
    plt.ylabel("Recall@10 (best)")
    plt.title(f"Ablation: Hub penalty ({dataset})")
    plt.grid(True, linestyle="--", alpha=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"ablation_{dataset}_hub_penalty_recall10.{ext}", bbox_inches="tight")
    plt.close()


def plot_ppr_mode(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if "ppr_mode" not in df.columns:
        return
    df = df.dropna(subset=["ppr_mode", "recall@10"])
    if df.empty:
        return
    best = (
        df.groupby(["ppr_mode"], as_index=False)["recall@10"]
        .max()
        .sort_values("recall@10", ascending=False)
    )
    if best.empty:
        return
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(best["ppr_mode"], best["recall@10"])
    plt.xlabel("PPR mode")
    plt.ylabel("Recall@10 (best)")
    plt.title(f"Ablation: PPR mode ({dataset})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"ablation_{dataset}_ppr_mode_recall10.{ext}", bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        paths = list(Path("outputs").glob("ablation_*/ablation_results.csv"))
    df = load_inputs(paths)
    if df.empty:
        raise SystemExit("No ablation_results.csv found.")
    out_dir = Path(args.output)
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        plot_seed_k(sub, out_dir, dataset)
        plot_alpha_iter_heatmap(sub, out_dir, dataset)
        plot_seed_weighting(sub, out_dir, dataset)
        plot_hub_penalty(sub, out_dir, dataset)
        plot_ppr_mode(sub, out_dir, dataset)
    print(f"Saved ablation plots to {out_dir}")


if __name__ == "__main__":
    main()
