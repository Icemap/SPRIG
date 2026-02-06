from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--method", nargs="+", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="distilbert-base-cased-distilled-squad")
    p.add_argument("--max-context-chars", type=int, default=4000)
    return p.parse_args()


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^0-9a-z\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def load_answers(dataset: str, split: str) -> Dict[str, str]:
    if dataset == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor", split=split)
    else:
        ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
    answers = {}
    for sample in ds:
        qid = sample.get("id")
        ans = sample.get("answer", "")
        if qid is not None:
            answers[str(qid)] = str(ans)
    return answers


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

    docs = {}
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        docs[doc["doc_id"]] = doc["text"]

    retrieved = json.loads(retrieved_path.read_text(encoding="utf-8"))
    answers = load_answers(args.dataset, args.split)

    random.seed(args.seed)
    if args.limit_queries is not None and args.limit_queries < len(queries):
        queries = random.sample(queries, args.limit_queries)

    qa = pipeline("question-answering", model=args.model, tokenizer=args.model, device=-1)

    metrics = {}
    per_query_out = {}
    for method in args.method:
        em_sum = 0.0
        f1_sum = 0.0
        n = 0
        per_query = {}
        for q in queries:
            qid = str(q["qid"])
            question = q["question"]
            gold = answers.get(qid)
            if gold is None:
                continue
            doc_ids = retrieved.get(method, {}).get(qid, [])[: args.top_k]
            context_parts: List[str] = []
            for did in doc_ids:
                text = docs.get(did, "")
                if text:
                    context_parts.append(text)
            context = "\n".join(context_parts)
            if len(context) > args.max_context_chars:
                context = context[: args.max_context_chars]
            if not context:
                pred = ""
            else:
                pred = qa(question=question, context=context).get("answer", "")
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            em_sum += em
            f1_sum += f1
            n += 1
            per_query[qid] = {"pred": pred, "gold": gold, "em": em, "f1": f1}
        metrics[method] = {"em": em_sum / max(1, n), "f1": f1_sum / max(1, n), "n": n}
        per_query_out[method] = per_query

    out_path = run_dir / "qa_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "qa_per_query.json").write_text(
        json.dumps(per_query_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
