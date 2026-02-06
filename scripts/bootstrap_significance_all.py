from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--baseline", nargs="+", default=["bm25", "rrf"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
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


def per_query_metrics(
    gold: Dict[str, List[str]],
    retrieved: Dict[str, List[str]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for qid, gold_docs in gold.items():
        if not gold_docs:
            continue
        preds = retrieved.get(qid, [])[:k]
        covered = sum(1 for d in gold_docs if d in preds)
        recall = covered / len(gold_docs)
        hit = 1.0 if any(d in gold_docs for d in preds) else 0.0
        rr = 0.0
        for rank, doc_id in enumerate(preds, start=1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break
        out[qid] = {"recall": recall, "hit": hit, "mrr": rr}
    return out


def bootstrap_ci(diffs: List[float], iters: int, seed: int) -> List[float]:
    if not diffs:
        return [0.0, 0.0]
    random.seed(seed)
    boot = []
    n = len(diffs)
    for _ in range(iters):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot))]
    return [float(lo), float(hi)]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    gold = load_gold(run_dir)
    retrieved = json.loads((run_dir / "retrieved.json").read_text(encoding="utf-8"))
    methods = sorted(retrieved.keys())

    metrics = {m: per_query_metrics(gold, retrieved.get(m, {}), args.k) for m in methods}
    out = {"k": args.k, "iters": args.iters, "baseline": args.baseline, "comparisons": []}

    for base in args.baseline:
        if base not in metrics:
            continue
        base_m = metrics[base]
        qids = list(base_m.keys())
        for method in methods:
            if method == base:
                continue
            m = metrics[method]
            common = [q for q in qids if q in m]
            if not common:
                continue
            for metric in ["recall", "hit", "mrr"]:
                diffs = [m[q][metric] - base_m[q][metric] for q in common]
                mean_diff = sum(diffs) / len(diffs)
                ci = bootstrap_ci(diffs, args.iters, args.seed)
                wins = sum(1 for d in diffs if d > 0)
                ties = sum(1 for d in diffs if d == 0)
                losses = sum(1 for d in diffs if d < 0)
                out["comparisons"].append(
                    {
                        "method": method,
                        "baseline": base,
                        "metric": metric,
                        "mean_diff": mean_diff,
                        "ci_95": ci,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "n": len(diffs),
                    }
                )

    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
