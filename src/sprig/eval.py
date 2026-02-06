from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class Metrics:
    recall_at_k: Dict[int, float]
    hit_at_k: Dict[int, float]
    mrr: float


def evaluate(
    gold: Dict[str, Sequence[str]],
    retrieved: Dict[str, Sequence[str]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Metrics:
    ks = list(sorted(set(ks)))
    recall_sums = {k: 0.0 for k in ks}
    hit_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n = 0

    for qid, gold_docs in gold.items():
        gold_set = list(dict.fromkeys(gold_docs))
        if not gold_set:
            continue
        preds = retrieved.get(qid, [])
        n += 1

        # MRR: rank of first relevant
        rr = 0.0
        for rank, doc_id in enumerate(preds, start=1):
            if doc_id in gold_set:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        for k in ks:
            topk = preds[:k]
            hit = 1.0 if any(d in gold_set for d in topk) else 0.0
            hit_sums[k] += hit
            # recall: fraction of gold covered in top-k
            if gold_set:
                covered = sum(1 for d in gold_set if d in topk)
                recall_sums[k] += covered / len(gold_set)

    if n == 0:
        return Metrics(recall_at_k={k: 0.0 for k in ks}, hit_at_k={k: 0.0 for k in ks}, mrr=0.0)

    recall_at_k = {k: recall_sums[k] / n for k in ks}
    hit_at_k = {k: hit_sums[k] / n for k in ks}
    mrr = mrr_sum / n
    return Metrics(recall_at_k=recall_at_k, hit_at_k=hit_at_k, mrr=mrr)
