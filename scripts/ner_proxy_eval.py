from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


from sprig.utils.entity_linking import apply_alias_map, build_title_alias_map
from sprig.utils.ner import extract_entities
from sprig.utils.text import normalize_entity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--method", nargs="+", required=True)
    p.add_argument("--ner-mode", nargs="+", default=["regex"])
    p.add_argument("--entity-normalize", default="simple", choices=["none", "lower", "simple"])
    p.add_argument("--use-aliases", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def _bucket_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    arr = sorted(values)
    n = len(arr)

    def _pct(p: float) -> float:
        if n == 0:
            return 0.0
        idx = min(n - 1, int(round(p * (n - 1))))
        return float(arr[idx])

    avg = float(sum(arr)) / n
    return {"avg": avg, "p50": _pct(0.50), "p95": _pct(0.95)}


def _metrics_for_bucket(
    bucket_qids: Iterable[str],
    gold: Dict[str, List[str]],
    retrieved: Dict[str, List[str]],
    k: int = 10,
) -> Tuple[int, float, float, float]:
    n = 0
    recall_sum = 0.0
    hit_sum = 0.0
    mrr_sum = 0.0
    for qid in bucket_qids:
        gold_docs = gold.get(qid, [])
        if not gold_docs:
            continue
        preds = retrieved.get(qid, [])
        n += 1
        topk = preds[:k]
        hit_sum += 1.0 if any(d in gold_docs for d in topk) else 0.0
        covered = sum(1 for d in gold_docs if d in topk)
        recall_sum += covered / max(1, len(gold_docs))
        rr = 0.0
        for rank, doc_id in enumerate(preds, start=1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    return n, recall_sum / n, hit_sum / n, mrr_sum / n


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    queries_path = run_dir / "queries.jsonl"
    docs_path = run_dir / "docs.jsonl"
    retrieved_path = run_dir / "retrieved.json"

    queries = []
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(json.loads(line))

    doc_ids = []
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_ids.append(doc["doc_id"])

    retrieved_payload = json.loads(retrieved_path.read_text(encoding="utf-8"))
    gold = {str(q["qid"]): list(q.get("gold_titles", [])) for q in queries}

    alias_map = build_title_alias_map(doc_ids) if args.use_aliases else None

    results = []
    entity_counts = {}
    for mode in args.ner_mode:
        per_query_entities: Dict[str, List[str]] = {}
        count_list = []
        gold_match = {}
        for q in queries:
            qid = str(q["qid"])
            ents = extract_entities(q["question"], mode=mode, dedup=True, normalize="none")
            ents = apply_alias_map(ents, alias_map, normalize_mode=args.entity_normalize)
            per_query_entities[qid] = ents
            count_list.append(len(ents))
            gold_norm = {
                normalize_entity(t, mode=args.entity_normalize) for t in q.get("gold_titles", []) if t
            }
            gold_match[qid] = bool(set(ents) & gold_norm)

        entity_counts[mode] = _bucket_stats(count_list)
        total_q = len(queries)
        buckets = {
            "no_entity": [qid for qid, ents in per_query_entities.items() if not ents],
            "entity_no_gold": [
                qid for qid, ents in per_query_entities.items() if ents and not gold_match[qid]
            ],
            "entity_gold": [qid for qid, ents in per_query_entities.items() if ents and gold_match[qid]],
        }

        for method in args.method:
            retrieved = retrieved_payload.get(method, {}) or {}
            for bucket_name, bucket_qids in buckets.items():
                n, recall, hit, mrr = _metrics_for_bucket(bucket_qids, gold, retrieved)
                results.append(
                    {
                        "ner_mode": mode,
                        "bucket": bucket_name,
                        "method": method,
                        "count": n,
                        "rate": (n / total_q) if total_q else 0.0,
                        "recall@10": recall,
                        "hit@10": hit,
                        "mrr": mrr,
                    }
                )

    out_path = Path(args.output) if args.output else run_dir / "ner_proxy.json"
    payload = {
        "entity_normalize": args.entity_normalize,
        "use_aliases": args.use_aliases,
        "entity_counts": entity_counts,
        "rows": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
