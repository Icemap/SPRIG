from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--method-a", required=True)
    p.add_argument("--method-b", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_gold(run_dir: Path) -> Dict[str, List[str]]:
    gold = {}
    queries_path = run_dir / "queries.jsonl"
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        gold[q["qid"]] = list(dict.fromkeys(q.get("gold_titles", [])))
    return gold


def per_query_recall(gold: Dict[str, List[str]], retrieved: Dict[str, List[str]], k: int) -> Dict[str, float]:
    out = {}
    for qid, gold_docs in gold.items():
        preds = retrieved.get(qid, [])[:k]
        if not gold_docs:
            continue
        covered = sum(1 for d in gold_docs if d in preds)
        out[qid] = covered / len(gold_docs)
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    gold = load_gold(run_dir)
    retrieved = json.loads((run_dir / "retrieved.json").read_text(encoding="utf-8"))
    a = per_query_recall(gold, retrieved.get(args.method_a, {}), args.k)
    b = per_query_recall(gold, retrieved.get(args.method_b, {}), args.k)

    qids = [q for q in a.keys() if q in b]
    diffs = [a[q] - b[q] for q in qids]
    base = sum(diffs) / max(1, len(diffs))

    random.seed(args.seed)
    boot = []
    for _ in range(args.iters):
        sample = [diffs[random.randrange(len(diffs))] for _ in range(len(diffs))]
        boot.append(sum(sample) / max(1, len(sample)))

    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot))]

    # Sign test: count wins
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)

    out = {
        "method_a": args.method_a,
        "method_b": args.method_b,
        "k": args.k,
        "mean_diff": base,
        "ci_95": [lo, hi],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n": len(diffs),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
