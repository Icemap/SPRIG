from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from sprig.data.hotpotqa import load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.utils.entity_linking import apply_alias_map, build_title_alias_map
from sprig.utils.ner import extract_entities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph-ner", default="spacy", choices=["spacy", "regex"])
    p.add_argument("--graph-entity-normalize", default="none", choices=["none", "lower", "simple"])
    p.add_argument("--graph-use-aliases", action="store_true")
    p.add_argument("--graph-min-df", type=int, default=1)
    p.add_argument("--graph-max-df-ratio", type=float, default=1.0)
    p.add_argument("--graph-hub-top-ratio", type=float, default=0.01)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _iter_entity_df(
    docs: Iterable[str],
    ner_mode: str,
    entity_normalize: str,
    alias_map: Dict[str, str] | None,
    min_entity_len: int = 2,
) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for text in docs:
        ents = extract_entities(text, mode=ner_mode, dedup=False, normalize="none")
        ents = apply_alias_map(ents, alias_map, normalize_mode=entity_normalize)
        counts: Dict[str, int] = {}
        for ent in ents:
            if len(ent) < min_entity_len:
                continue
            counts[ent] = counts.get(ent, 0) + 1
        for ent in counts:
            df[ent] = df.get(ent, 0) + 1
    return df


def main() -> None:
    args = parse_args()
    if args.dataset == "hotpotqa":
        docs, queries = load_hotpotqa(split=args.split, max_samples=args.max_samples, seed=args.seed)
    else:
        docs, queries = load_twowiki(split=args.split, max_samples=args.max_samples, seed=args.seed)

    doc_ids = [d.doc_id for d in docs]
    doc_texts = [d.text for d in docs]
    alias_map = build_title_alias_map(doc_ids) if args.graph_use_aliases else None

    entity_df = _iter_entity_df(
        doc_texts,
        ner_mode=args.graph_ner,
        entity_normalize=args.graph_entity_normalize,
        alias_map=alias_map,
    )
    n_docs = len(doc_ids)
    max_df = max(1, int(args.graph_max_df_ratio * max(1, n_docs)))
    filtered = [e for e, df in entity_df.items() if df >= args.graph_min_df and df <= max_df]
    removed = set()
    if args.graph_hub_top_ratio > 0 and filtered:
        ratio = min(1.0, max(0.0, float(args.graph_hub_top_ratio)))
        cut = int(len(filtered) * ratio)
        if cut > 0:
            sorted_ents = sorted(filtered, key=lambda e: entity_df[e], reverse=True)
            removed = set(sorted_ents[:cut])

    gold_titles = []
    for q in queries:
        gold_titles.extend(list(q.gold_titles))
    gold_titles = list(dict.fromkeys(gold_titles))
    gold_mapped = apply_alias_map(
        gold_titles,
        alias_map,
        normalize_mode=args.graph_entity_normalize,
    )
    gold_set = set(gold_mapped)
    removed_gold = {t for t in gold_set if t in removed}
    kept_gold = {t for t in gold_set if t in filtered and t not in removed}

    # Query-level coverage
    affected_queries = 0
    for q in queries:
        mapped = apply_alias_map(
            list(q.gold_titles),
            alias_map,
            normalize_mode=args.graph_entity_normalize,
        )
        if any(t in removed for t in mapped):
            affected_queries += 1

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "graph_ner": args.graph_ner,
        "graph_entity_normalize": args.graph_entity_normalize,
        "graph_use_aliases": args.graph_use_aliases,
        "graph_min_df": args.graph_min_df,
        "graph_max_df_ratio": args.graph_max_df_ratio,
        "graph_hub_top_ratio": args.graph_hub_top_ratio,
        "entities_total": len(entity_df),
        "entities_filtered": len(filtered),
        "entities_removed": len(removed),
        "gold_titles": len(gold_set),
        "gold_titles_removed": len(removed_gold),
        "gold_titles_kept": len(kept_gold),
        "gold_removed_rate": (len(removed_gold) / len(gold_set)) if gold_set else 0.0,
        "queries": len(queries),
        "queries_affected": affected_queries,
        "queries_affected_rate": affected_queries / max(1, len(queries)),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
