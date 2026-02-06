from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from sprig.data.hotpotqa import load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.eval import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hotpot-run", required=True)
    p.add_argument("--twowiki-run", required=True)
    p.add_argument("--hotpot-n", type=int, default=7405)
    p.add_argument("--twowiki-n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune-n", type=int, default=500)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "rrf", "graph_hybrid", "graph_dense"],
    )
    p.add_argument("--output", default=None)
    return p.parse_args()


def _load(dataset: str, max_samples: int, seed: int):
    if dataset == "hotpotqa":
        return load_hotpotqa(split="validation", max_samples=max_samples, seed=seed)
    return load_twowiki(split="validation", max_samples=max_samples, seed=seed)


def _compute_metrics(
    dataset: str,
    run_dir: str,
    max_samples: int,
    seed: int,
    tune_n: int,
    methods: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    _docs, queries = _load(dataset, max_samples=max_samples, seed=seed)
    qids = [q.qid for q in queries]
    random.seed(seed)
    tune = set(random.sample(qids, min(tune_n, len(qids))))
    gold = {q.qid: list(q.gold_titles) for q in queries if q.qid not in tune}
    retrieved = json.loads(Path(run_dir, "retrieved.json").read_text(encoding="utf-8"))

    metrics: Dict[str, Dict[str, float]] = {}
    for method in methods:
        m = evaluate(gold, retrieved.get(method, {}), ks=[10])
        metrics[method] = {"recall@10": m.recall_at_k[10], "mrr": m.mrr}
    return metrics


def main() -> None:
    args = parse_args()
    payload = {
        "hotpotqa": _compute_metrics(
            "hotpotqa",
            args.hotpot_run,
            args.hotpot_n,
            args.seed,
            args.tune_n,
            args.methods,
        ),
        "2wikimultihopqa": _compute_metrics(
            "2wikimultihopqa",
            args.twowiki_run,
            args.twowiki_n,
            args.seed,
            args.tune_n,
            args.methods,
        ),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
