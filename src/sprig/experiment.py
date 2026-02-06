from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sprig.data.hotpotqa import Doc, Query, load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.eval import evaluate
from sprig.retrieval.bm25 import BM25Index
from sprig.retrieval.graph import build_bipartite_graph, build_term_bipartite_graph, ppr_search
from sprig.utils.entity_linking import build_title_alias_map
from sprig.utils.graph_stats import compute_graph_stats
from sprig.utils.ner import extract_entities
from sprig.utils.text import normalize_whitespace, tokenize


def _seed_everything(seed: int) -> None:
    random.seed(seed)


def _latency_stats(times: List[float]) -> Dict[str, float]:
    if not times:
        return {"total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = sorted(times)
    n = len(arr)

    def _pct(p: float) -> float:
        if n == 0:
            return 0.0
        idx = min(n - 1, int(round(p * (n - 1))))
        return arr[idx]

    total = float(sum(arr))
    return {
        "total": total,
        "mean": total / n,
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


def _max_rss_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes
        if sys.platform == "darwin":
            return float(usage) / (1024 * 1024)
        return float(usage) / 1024.0
    except Exception:
        return None


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _rrf_fuse(rankings: Dict[str, List[str]], top_k: int, rrf_k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for docs in rankings.values():
        for rank, doc_id in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def _rrf_scores(rankings: Dict[str, List[int]], rrf_k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for docs in rankings.values():
        for rank, doc_id in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    max_val = max(scores.values())
    if max_val <= 0:
        return scores
    return {k: v / max_val for k, v in scores.items()}


def _rm3_weights(
    query: str,
    fb_docs: List[Tuple[int, float]],
    doc_tokens: List[List[str]],
    fb_terms: int,
    orig_weight: float,
) -> Dict[str, float]:
    q_tokens = tokenize(query)
    q_tf = Counter(q_tokens)
    q_total = float(sum(q_tf.values()))
    q_model = {t: c / q_total for t, c in q_tf.items()} if q_total > 0 else {}

    if not fb_docs:
        return q_model

    scores = [max(0.0, s) for _, s in fb_docs]
    score_sum = sum(scores)
    if score_sum <= 0:
        doc_weights = [1.0 / len(fb_docs)] * len(fb_docs)
    else:
        doc_weights = [s / score_sum for s in scores]

    fb_model: Dict[str, float] = {}
    for (doc_idx, _), weight in zip(fb_docs, doc_weights):
        if doc_idx < 0 or doc_idx >= len(doc_tokens):
            continue
        toks = doc_tokens[doc_idx]
        if not toks:
            continue
        tf = Counter(toks)
        doc_len = float(sum(tf.values()))
        if doc_len == 0:
            continue
        for term, cnt in tf.items():
            fb_model[term] = fb_model.get(term, 0.0) + weight * (cnt / doc_len)

    if not fb_model:
        return q_model

    fb_top = sorted(fb_model.items(), key=lambda x: x[1], reverse=True)[:fb_terms]
    fb_total = sum(w for _, w in fb_top)
    fb_norm = {t: (w / fb_total if fb_total > 0 else 0.0) for t, w in fb_top}

    combined: Dict[str, float] = {}
    for term, prob in q_model.items():
        combined[term] = combined.get(term, 0.0) + orig_weight * prob
    for term, prob in fb_norm.items():
        combined[term] = combined.get(term, 0.0) + (1.0 - orig_weight) * prob
    return combined


def _bm25_2step_weights(
    query: str,
    seed_docs: List[Tuple[int, float]],
    doc_texts: List[str],
    expand_per_doc: int,
    max_terms: int,
    entity_mode: str,
    entity_normalize: str,
    query_weight: float,
) -> Dict[str, float]:
    weights: Counter[str] = Counter()
    for rank, (doc_idx, _) in enumerate(seed_docs, start=1):
        if doc_idx < 0 or doc_idx >= len(doc_texts):
            continue
        ents = extract_entities(
            doc_texts[doc_idx],
            mode=entity_mode,
            dedup=True,
            normalize=entity_normalize,
        )
        if expand_per_doc > 0:
            ents = ents[:expand_per_doc]
        for ent in ents:
            for tok in tokenize(ent):
                if tok:
                    weights[tok] += 1.0 / rank

    for tok in tokenize(query):
        if tok:
            weights[tok] += query_weight

    if not weights:
        return {}
    if max_terms > 0:
        weights = Counter(dict(weights.most_common(max_terms)))
    return dict(weights)


def run_experiment(
    output_dir: Path,
    dataset: str = "hotpotqa",
    split: str = "validation",
    max_samples: int | None = 200,
    seed: int = 42,
    methods: Sequence[str] = ("bm25", "dense", "graph"),
    top_ks: Sequence[int] = (1, 3, 5, 10),
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    graph_ner: str = "spacy",
    graph_entity_normalize: str = "none",
    graph_use_aliases: bool = False,
    graph_seed_docs_k: int = 5,
    graph_seed_docs_k_dense: int | None = None,
    graph_seed_docs_k_bm25: int | None = None,
    graph_seed_docs_k_rrf: int | None = None,
    graph_fallback_k: int = 1,
    dense_use_hnsw: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
    ppr_alpha: float = 0.15,
    ppr_max_iter: int = 10,
    ppr_mode: str = "power",
    ppr_tol: float | None = None,
    ppr_push_eps: float = 1e-4,
    ppr_push_max_steps: int = 200000,
    graph_min_df: int = 1,
    graph_max_df_ratio: float = 1.0,
    graph_hub_penalty: float = 0.0,
    graph_hub_top_ratio: float = 0.0,
    graph_entity_edge_cap: int | None = None,
    graph_norm: str = "row",
    graph_seed_entity_df_power: float = 0.0,
    graph_seed_weighting: str = "raw",
    graph_seed_temp: float = 1.0,
    graph_seed_mix_mode: str = "l1",
    graph_seed_mix_alpha: float | None = None,
    term_min_df: int = 5,
    term_max_df_ratio: float = 0.2,
    term_norm: str = "row",
    rrf_k: int = 60,
    rm3_fb_docs: int = 10,
    rm3_fb_terms: int = 20,
    rm3_orig_weight: float = 0.5,
    bm25_2step_k1: int = 10,
    bm25_2step_expand_per_doc: int = 5,
    bm25_2step_max_terms: int = 30,
    bm25_2step_entity_mode: str = "regex",
    bm25_2step_entity_normalize: str = "simple",
    bm25_2step_query_weight: float = 1.0,
    rrf_ppr_fusion_weight: float = 1.0,
    rerank_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    rerank_candidates: int = 100,
    rerank_batch_size: int = 32,
    cache_root: Path | None = None,
    tag: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(seed)
    if dataset == "hotpotqa":
        docs, queries = load_hotpotqa(split=split, max_samples=max_samples, seed=seed)
    elif dataset == "2wikimultihopqa":
        docs, queries = load_twowiki(split=split, max_samples=max_samples, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Basic cleaning
    docs = [Doc(doc_id=d.doc_id, title=d.title, text=normalize_whitespace(d.text)) for d in docs]
    queries = [
        Query(qid=q.qid, question=normalize_whitespace(q.question), gold_titles=q.gold_titles)
        for q in queries
    ]

    doc_ids = [d.doc_id for d in docs]
    doc_texts = [d.text for d in docs]
    entity_alias_map = build_title_alias_map(doc_ids) if graph_use_aliases else None

    gold = {q.qid: list(q.gold_titles) for q in queries}

    # Persist processed corpus and queries for reproducibility
    with (output_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {"doc_id": d.doc_id, "title": d.title, "text": d.text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(
                json.dumps(
                    {"qid": q.qid, "question": q.question, "gold_titles": list(q.gold_titles)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    results = []
    per_query = {}
    latency = {}
    graph_stats = {}

    top_k_max = max(top_ks)

    need_bm25 = any(
        m in methods
        for m in [
            "bm25",
            "bm25_2step",
            "graph_hybrid",
            "graph_bm25_fallback",
            "graph_rrf",
            "rrf",
            "rrf_ppr_fusion",
            "rrf_rerank",
            "tfidf_graph",
            "rm3",
            "rerank",
        ]
    )
    need_dense = any(
        m in methods for m in ["dense", "graph_dense", "graph_rrf", "rrf", "rrf_ppr_fusion", "rrf_rerank"]
    )
    bm25 = None
    bm25_retrieved = None
    bm25_hits = None
    bm25_times = None
    dense_retriever = None
    dense_index = None
    dense_hits_cache = None
    dense_times = None
    doc_tokens = None

    if need_bm25:
        t0 = time.perf_counter()
        bm25 = BM25Index.build(doc_texts)
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()

        retrieved = {}
        per_times = []
        bm25_hits = {}
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            hits = bm25.search(q.question, top_k=top_k_max)
            bm25_hits[q.qid] = hits
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1
        bm25_times = per_times
        bm25_retrieved = retrieved

        if "bm25" in methods:
            metrics = evaluate(gold, retrieved, ks=top_ks)
            lat_stats = _latency_stats(per_times)
            results.append(
                {
                    "method": "bm25",
                    "index_time_sec": index_time,
                    "query_time_sec": query_time,
                    "metrics": asdict(metrics),
                    "docs": len(docs),
                    "queries": len(queries),
                    "latency_stats": lat_stats,
                    "rss_peak_mb": _max_rss_mb(),
                    "rss_index_mb": index_rss,
                    "rss_query_mb": rss_query_peak,
                }
            )
            per_query["bm25"] = retrieved
            latency["bm25"] = per_times

    if "rm3" in methods:
        if bm25 is None:
            raise RuntimeError("rm3 requires BM25")
        if doc_tokens is None:
            doc_tokens = [tokenize(text) for text in doc_texts]
        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            fb_hits = bm25.search(q.question, top_k=rm3_fb_docs)
            term_weights = _rm3_weights(
                q.question,
                fb_hits,
                doc_tokens,
                fb_terms=rm3_fb_terms,
                orig_weight=rm3_orig_weight,
            )
            if term_weights:
                hits = bm25.search_weighted(term_weights, top_k=top_k_max)
            else:
                hits = bm25.search(q.question, top_k=top_k_max)
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "rm3",
                "index_time_sec": 0.0,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": _rss_mb(),
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["rm3"] = retrieved
        latency["rm3"] = per_times

    if "bm25_2step" in methods:
        if bm25 is None:
            raise RuntimeError("bm25_2step requires BM25")
        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            seed_hits = bm25.search(q.question, top_k=bm25_2step_k1)
            term_weights = _bm25_2step_weights(
                q.question,
                seed_hits,
                doc_texts,
                expand_per_doc=bm25_2step_expand_per_doc,
                max_terms=bm25_2step_max_terms,
                entity_mode=bm25_2step_entity_mode,
                entity_normalize=bm25_2step_entity_normalize,
                query_weight=bm25_2step_query_weight,
            )
            if term_weights:
                hits = bm25.search_weighted(term_weights, top_k=top_k_max)
            else:
                hits = seed_hits[:top_k_max]
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "bm25_2step",
                "index_time_sec": 0.0,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": _rss_mb(),
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["bm25_2step"] = retrieved
        latency["bm25_2step"] = per_times

    if need_dense:
        from sprig.retrieval.dense import DenseRetriever

        cache_dir = output_dir / "cache"
        if cache_root is not None:
            cache_dir = (
                cache_root
                / "dense"
                / dataset
                / split
                / (f"n{max_samples}" if max_samples is not None else "nfull")
                / f"s{seed}"
            )
        t0 = time.perf_counter()
        dense_retriever = DenseRetriever(
            dense_model,
            cache_dir=cache_dir,
            use_hnsw=dense_use_hnsw,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
        dense_index = dense_retriever.build_index(doc_ids, doc_texts)
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()

        dense_hits_cache = {}
        dense_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            hits = dense_retriever.search(dense_index, q.question, top_k=top_k_max)
            dense_hits_cache[q.qid] = hits
            dense_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        dense_query_time = time.perf_counter() - t1

        if "dense" in methods:
            retrieved = {qid: [doc_ids[i] for i, _ in hits] for qid, hits in dense_hits_cache.items()}
            per_times = dense_times
            query_time = dense_query_time

            metrics = evaluate(gold, retrieved, ks=top_ks)
            lat_stats = _latency_stats(per_times)
            results.append(
                {
                    "method": "dense",
                    "index_time_sec": index_time,
                    "query_time_sec": query_time,
                    "metrics": asdict(metrics),
                    "docs": len(docs),
                    "queries": len(queries),
                    "latency_stats": lat_stats,
                    "rss_peak_mb": _max_rss_mb(),
                    "rss_index_mb": index_rss,
                    "rss_query_mb": rss_query_peak,
                }
            )
            per_query["dense"] = retrieved
            latency["dense"] = per_times

    if "rrf" in methods:
        if bm25_retrieved is None or dense_hits_cache is None:
            raise RuntimeError("rrf requires BM25 and Dense")
        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for idx, q in enumerate(queries):
            q_start = time.perf_counter()
            bm25_docs = bm25_retrieved[q.qid]
            dense_hits = [doc_ids[i] for i, _ in dense_hits_cache[q.qid]]
            fused = _rrf_fuse({"bm25": bm25_docs, "dense": dense_hits}, top_k=top_k_max, rrf_k=rrf_k)
            retrieved[q.qid] = fused
            fusion_time = time.perf_counter() - q_start
            bm25_t = bm25_times[idx] if bm25_times is not None else 0.0
            dense_t = dense_times[idx] if dense_times is not None else 0.0
            per_times.append(fusion_time + bm25_t + dense_t)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)
        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "rrf",
                "index_time_sec": 0.0,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": _rss_mb(),
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["rrf"] = retrieved
        latency["rrf"] = per_times

    reranker = None
    if "rerank" in methods or "rrf_rerank" in methods:
        from sprig.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=rerank_model, batch_size=rerank_batch_size)

    if "rerank" in methods:
        if bm25 is None:
            raise RuntimeError("rerank requires BM25")
        if reranker is None:
            raise RuntimeError("rerank requires CrossEncoder reranker")
        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            bm25_candidates = bm25.search(q.question, top_k=rerank_candidates)
            candidate_ids = [doc_ids[i] for i, _ in bm25_candidates]
            candidate_texts = [doc_texts[i] for i, _ in bm25_candidates]
            ranked = reranker.rerank(q.question, candidate_texts, top_k=top_k_max)
            reranked_docs = [candidate_ids[i] for i, _ in ranked]
            retrieved[q.qid] = reranked_docs
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "rerank",
                "index_time_sec": 0.0,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": _rss_mb(),
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["rerank"] = retrieved
        latency["rerank"] = per_times

    if "rrf_rerank" in methods:
        if reranker is None:
            raise RuntimeError("rrf_rerank requires CrossEncoder reranker")
        if bm25_hits is None or dense_hits_cache is None or bm25_times is None or dense_times is None:
            raise RuntimeError("rrf_rerank requires BM25 and Dense seeds")
        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        for idx, q in enumerate(queries):
            seed_time = float(bm25_times[idx]) + float(dense_times[idx])
            rerank_start = time.perf_counter()
            bm25_docs = [doc_idx for doc_idx, _ in bm25_hits[q.qid]]
            dense_docs = [doc_idx for doc_idx, _ in dense_hits_cache[q.qid]]
            fused = _rrf_scores({"bm25": bm25_docs, "dense": dense_docs}, rrf_k=rrf_k)
            candidate_docs = [doc_idx for doc_idx, _ in fused[:rerank_candidates]]
            candidate_texts = [doc_texts[i] for i in candidate_docs]
            ranked = reranker.rerank(q.question, candidate_texts, top_k=top_k_max)
            reranked_docs = [candidate_docs[i] for i, _ in ranked]
            retrieved[q.qid] = [doc_ids[i] for i in reranked_docs]
            rerank_time = time.perf_counter() - rerank_start
            per_times.append(seed_time + rerank_time)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "rrf_rerank",
                "index_time_sec": 0.0,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": _rss_mb(),
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["rrf_rerank"] = retrieved
        latency["rrf_rerank"] = per_times

    if "graph" in methods:
        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                ner_mode=graph_ner,
                entity_normalize=graph_entity_normalize,
                entity_alias_map=entity_alias_map,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
            )
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "graph",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["graph"] = retrieved
        latency["graph"] = per_times

    if "graph_bm25_fallback" in methods:
        if bm25_hits is None:
            raise RuntimeError("graph_bm25_fallback requires BM25 seeds")

        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        fallback_count = 0
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        fallback_k = max(1, int(graph_fallback_k))
        for q in queries:
            q_start = time.perf_counter()
            entities = extract_entities(
                q.question,
                mode=graph_ner,
                dedup=True,
                normalize="none",
            )
            seed_terms = entities
            seed_docs = None
            if not entities:
                fallback_count += 1
                seed_docs = bm25_hits[q.qid][:fallback_k]
            hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                seed_terms=seed_terms,
                seed_docs=seed_docs,
                seed_weighting=graph_seed_weighting,
                seed_temp=graph_seed_temp,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
                entity_alias_map=entity_alias_map,
            )
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        fallback_rate = fallback_count / max(1, len(queries))
        results.append(
            {
                "method": "graph_bm25_fallback",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_rate,
                "fallback_k": fallback_k,
            }
        )
        per_query["graph_bm25_fallback"] = retrieved
        latency["graph_bm25_fallback"] = per_times

    if "tfidf_graph" in methods:
        t0 = time.perf_counter()
        term_graph = build_term_bipartite_graph(
            doc_ids,
            doc_texts,
            min_df=term_min_df,
            max_df_ratio=term_max_df_ratio,
            normalization=term_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["term_graph"] = compute_graph_stats(term_graph)

        retrieved = {}
        per_times = []
        rss_query_peak = _rss_mb() or 0.0
        t1 = time.perf_counter()
        for q in queries:
            q_start = time.perf_counter()
            seed_terms = tokenize(q.question)
            hits = ppr_search(
                term_graph,
                q.question,
                top_k=top_k_max,
                seed_terms=seed_terms,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
            )
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(time.perf_counter() - q_start)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = time.perf_counter() - t1

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "tfidf_graph",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
            }
        )
        per_query["tfidf_graph"] = retrieved
        latency["tfidf_graph"] = per_times

    if "graph_hybrid" in methods:
        if bm25_hits is None:
            raise RuntimeError("graph_hybrid requires BM25 seeds")

        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        seed_times = []
        ppr_times = []
        rss_query_peak = _rss_mb() or 0.0
        for idx, q in enumerate(queries):
            seed_time = float(bm25_times[idx]) if bm25_times is not None else 0.0
            hits_with_scores = bm25_hits[q.qid]
            seed_k = graph_seed_docs_k_bm25 or graph_seed_docs_k
            seed_docs = hits_with_scores[:seed_k]
            ppr_start = time.perf_counter()
            hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                ner_mode=graph_ner,
                entity_normalize=graph_entity_normalize,
                entity_alias_map=entity_alias_map,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                seed_docs=seed_docs,
                seed_weighting=graph_seed_weighting,
                seed_temp=graph_seed_temp,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
            )
            ppr_time = time.perf_counter() - ppr_start
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            seed_times.append(seed_time)
            ppr_times.append(ppr_time)
            per_times.append(seed_time + ppr_time)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "graph_hybrid",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
                "seed_time_sec": sum(seed_times),
                "ppr_time_sec": sum(ppr_times),
            }
        )
        per_query["graph_hybrid"] = retrieved
        latency["graph_hybrid"] = per_times

    if "graph_rrf" in methods:
        if bm25_hits is None or dense_hits_cache is None:
            raise RuntimeError("graph_rrf requires BM25 and Dense seeds")

        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        seed_times = []
        ppr_times = []
        rss_query_peak = _rss_mb() or 0.0
        seed_k = graph_seed_docs_k_rrf or graph_seed_docs_k
        for idx, q in enumerate(queries):
            seed_time = float(bm25_times[idx]) + float(dense_times[idx]) if bm25_times and dense_times else 0.0
            bm25_docs = [doc_idx for doc_idx, _ in bm25_hits[q.qid]]
            dense_docs = [doc_idx for doc_idx, _ in dense_hits_cache[q.qid]]
            fused = _rrf_scores({"bm25": bm25_docs, "dense": dense_docs}, rrf_k=rrf_k)
            seed_docs = fused[:seed_k]
            ppr_start = time.perf_counter()
            hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                ner_mode=graph_ner,
                entity_normalize=graph_entity_normalize,
                entity_alias_map=entity_alias_map,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                seed_docs=seed_docs,
                seed_weighting=graph_seed_weighting,
                seed_temp=graph_seed_temp,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
            )
            ppr_time = time.perf_counter() - ppr_start
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            seed_times.append(seed_time)
            ppr_times.append(ppr_time)
            per_times.append(seed_time + ppr_time)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "graph_rrf",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
                "seed_time_sec": sum(seed_times),
                "ppr_time_sec": sum(ppr_times),
            }
        )
        per_query["graph_rrf"] = retrieved
        latency["graph_rrf"] = per_times

    if "rrf_ppr_fusion" in methods:
        if bm25_hits is None or dense_hits_cache is None or bm25_times is None or dense_times is None:
            raise RuntimeError("rrf_ppr_fusion requires BM25 and Dense seeds")

        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        seed_times = []
        ppr_times = []
        rss_query_peak = _rss_mb() or 0.0
        for idx, q in enumerate(queries):
            seed_time = float(bm25_times[idx]) + float(dense_times[idx])
            bm25_docs = [doc_idx for doc_idx, _ in bm25_hits[q.qid]]
            dense_docs = [doc_idx for doc_idx, _ in dense_hits_cache[q.qid]]
            rrf_scores = dict(_rrf_scores({"bm25": bm25_docs, "dense": dense_docs}, rrf_k=rrf_k))
            ppr_start = time.perf_counter()
            ppr_hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                ner_mode=graph_ner,
                entity_normalize=graph_entity_normalize,
                entity_alias_map=entity_alias_map,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
            )
            ppr_time = time.perf_counter() - ppr_start
            ppr_scores = {doc_idx: score for doc_idx, score in ppr_hits}
            rrf_norm = _normalize_scores(rrf_scores)
            ppr_norm = _normalize_scores(ppr_scores)
            combined: Dict[int, float] = {}
            for doc_idx, score in rrf_norm.items():
                combined[doc_idx] = combined.get(doc_idx, 0.0) + score
            for doc_idx, score in ppr_norm.items():
                combined[doc_idx] = combined.get(doc_idx, 0.0) + (rrf_ppr_fusion_weight * score)
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k_max]
            retrieved[q.qid] = [doc_ids[i] for i, _ in ranked]
            seed_times.append(seed_time)
            ppr_times.append(ppr_time)
            per_times.append(seed_time + ppr_time)
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "rrf_ppr_fusion",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
                "seed_time_sec": sum(seed_times),
                "ppr_time_sec": sum(ppr_times),
            }
        )
        per_query["rrf_ppr_fusion"] = retrieved
        latency["rrf_ppr_fusion"] = per_times

    if "graph_dense" in methods:
        if dense_hits_cache is None or dense_times is None:
            raise RuntimeError("graph_dense requires Dense seeds")

        t0 = time.perf_counter()
        graph_index = build_bipartite_graph(
            doc_ids,
            doc_texts,
            ner_mode=graph_ner,
            entity_normalize=graph_entity_normalize,
            entity_alias_map=entity_alias_map,
            min_entity_df=graph_min_df,
            max_entity_df_ratio=graph_max_df_ratio,
            hub_penalty=graph_hub_penalty,
            hub_top_ratio=graph_hub_top_ratio,
            entity_edge_cap=graph_entity_edge_cap,
            normalization=graph_norm,
        )
        index_time = time.perf_counter() - t0
        index_rss = _rss_mb()
        graph_stats["entity_graph"] = compute_graph_stats(graph_index)

        retrieved = {}
        per_times = []
        seed_times = []
        ppr_times = []
        rss_query_peak = _rss_mb() or 0.0
        for idx, q in enumerate(queries):
            seed_start = time.perf_counter()
            dense_hits = dense_hits_cache[q.qid]
            seed_k = graph_seed_docs_k_dense or graph_seed_docs_k
            seed_docs = dense_hits[:seed_k]
            seed_time = dense_times[idx]
            seed_times.append(seed_time)
            ppr_start = time.perf_counter()
            hits = ppr_search(
                graph_index,
                q.question,
                top_k=top_k_max,
                ner_mode=graph_ner,
                entity_normalize=graph_entity_normalize,
                entity_alias_map=entity_alias_map,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                seed_docs=seed_docs,
                seed_weighting=graph_seed_weighting,
                seed_temp=graph_seed_temp,
                mode=ppr_mode,
                tol=ppr_tol,
                push_eps=ppr_push_eps,
                push_max_steps=ppr_push_max_steps,
                seed_entity_df_power=graph_seed_entity_df_power,
                seed_mix_mode=graph_seed_mix_mode,
                seed_mix_alpha=graph_seed_mix_alpha,
            )
            ppr_times.append(time.perf_counter() - ppr_start)
            retrieved[q.qid] = [doc_ids[i] for i, _ in hits]
            per_times.append(seed_time + ppr_times[-1])
            rss_now = _rss_mb()
            if rss_now is not None:
                rss_query_peak = max(rss_query_peak, rss_now)
        query_time = sum(per_times)

        metrics = evaluate(gold, retrieved, ks=top_ks)
        lat_stats = _latency_stats(per_times)
        results.append(
            {
                "method": "graph_dense",
                "index_time_sec": index_time,
                "query_time_sec": query_time,
                "metrics": asdict(metrics),
                "docs": len(docs),
                "queries": len(queries),
                "latency_stats": lat_stats,
                "rss_peak_mb": _max_rss_mb(),
                "rss_index_mb": index_rss,
                "rss_query_mb": rss_query_peak,
                "seed_time_sec": sum(seed_times),
                "ppr_time_sec": sum(ppr_times),
            }
        )
        per_query["graph_dense"] = retrieved
        latency["graph_dense"] = per_times
        latency["graph_dense_seed"] = seed_times
        latency["graph_dense_ppr"] = ppr_times

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "dataset": dataset,
                    "split": split,
                    "max_samples": max_samples,
                "methods": list(methods),
                "top_ks": list(top_ks),
                "dense_model": dense_model,
                "graph_ner": graph_ner,
                "graph_seed_docs_k": graph_seed_docs_k,
                "graph_seed_docs_k_dense": graph_seed_docs_k_dense,
                "graph_seed_docs_k_bm25": graph_seed_docs_k_bm25,
                "graph_seed_docs_k_rrf": graph_seed_docs_k_rrf,
                "graph_fallback_k": graph_fallback_k,
                "dense_use_hnsw": dense_use_hnsw,
                "hnsw_m": hnsw_m,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search": hnsw_ef_search,
                "ppr_alpha": ppr_alpha,
                "ppr_max_iter": ppr_max_iter,
                "ppr_mode": ppr_mode,
                "ppr_tol": ppr_tol,
                "ppr_push_eps": ppr_push_eps,
                "ppr_push_max_steps": ppr_push_max_steps,
                "graph_min_df": graph_min_df,
                "graph_max_df_ratio": graph_max_df_ratio,
                "graph_hub_penalty": graph_hub_penalty,
                "graph_hub_top_ratio": graph_hub_top_ratio,
                "graph_entity_edge_cap": graph_entity_edge_cap,
                "graph_norm": graph_norm,
                "graph_entity_normalize": graph_entity_normalize,
                "graph_use_aliases": graph_use_aliases,
                "graph_seed_entity_df_power": graph_seed_entity_df_power,
                "graph_seed_weighting": graph_seed_weighting,
                "graph_seed_temp": graph_seed_temp,
                "graph_seed_mix_mode": graph_seed_mix_mode,
                "graph_seed_mix_alpha": graph_seed_mix_alpha,
                "term_min_df": term_min_df,
                "term_max_df_ratio": term_max_df_ratio,
                "term_norm": term_norm,
                "rrf_k": rrf_k,
                "rm3_fb_docs": rm3_fb_docs,
                "rm3_fb_terms": rm3_fb_terms,
                "rm3_orig_weight": rm3_orig_weight,
                "bm25_2step_k1": bm25_2step_k1,
                "bm25_2step_expand_per_doc": bm25_2step_expand_per_doc,
                "bm25_2step_max_terms": bm25_2step_max_terms,
                "bm25_2step_entity_mode": bm25_2step_entity_mode,
                "bm25_2step_entity_normalize": bm25_2step_entity_normalize,
                "bm25_2step_query_weight": bm25_2step_query_weight,
                "rrf_ppr_fusion_weight": rrf_ppr_fusion_weight,
                "rerank_model": rerank_model,
                "rerank_candidates": rerank_candidates,
                "rerank_batch_size": rerank_batch_size,
                "tag": tag,
                "seed": seed,
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "retrieved.json").open("w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False)

    if latency:
        with (output_dir / "latency.json").open("w", encoding="utf-8") as f:
            json.dump(latency, f, ensure_ascii=False)

    if graph_stats:
        with (output_dir / "graph_stats.json").open("w", encoding="utf-8") as f:
            json.dump(graph_stats, f, ensure_ascii=False, indent=2)

    # Run metadata + manifest
    run_meta = {
        "dataset": dataset,
        "split": split,
        "max_samples": max_samples,
        "methods": list(methods),
        "top_ks": list(top_ks),
        "dense_model": dense_model,
        "graph_ner": graph_ner,
        "graph_seed_docs_k": graph_seed_docs_k,
        "graph_seed_docs_k_dense": graph_seed_docs_k_dense,
        "graph_seed_docs_k_bm25": graph_seed_docs_k_bm25,
        "graph_seed_docs_k_rrf": graph_seed_docs_k_rrf,
        "graph_fallback_k": graph_fallback_k,
        "dense_use_hnsw": dense_use_hnsw,
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
        "hnsw_ef_search": hnsw_ef_search,
        "seed": seed,
        "ppr_alpha": ppr_alpha,
        "ppr_max_iter": ppr_max_iter,
        "ppr_mode": ppr_mode,
        "ppr_tol": ppr_tol,
        "ppr_push_eps": ppr_push_eps,
        "ppr_push_max_steps": ppr_push_max_steps,
        "graph_min_df": graph_min_df,
        "graph_max_df_ratio": graph_max_df_ratio,
        "graph_hub_penalty": graph_hub_penalty,
        "graph_hub_top_ratio": graph_hub_top_ratio,
        "graph_entity_edge_cap": graph_entity_edge_cap,
        "graph_norm": graph_norm,
        "term_min_df": term_min_df,
        "term_max_df_ratio": term_max_df_ratio,
        "term_norm": term_norm,
        "rrf_k": rrf_k,
        "rm3_fb_docs": rm3_fb_docs,
        "rm3_fb_terms": rm3_fb_terms,
        "rm3_orig_weight": rm3_orig_weight,
        "bm25_2step_k1": bm25_2step_k1,
        "bm25_2step_expand_per_doc": bm25_2step_expand_per_doc,
        "bm25_2step_max_terms": bm25_2step_max_terms,
        "bm25_2step_entity_mode": bm25_2step_entity_mode,
        "bm25_2step_entity_normalize": bm25_2step_entity_normalize,
        "bm25_2step_query_weight": bm25_2step_query_weight,
        "rrf_ppr_fusion_weight": rrf_ppr_fusion_weight,
        "rerank_model": rerank_model,
        "rerank_candidates": rerank_candidates,
        "rerank_batch_size": rerank_batch_size,
        "graph_entity_normalize": graph_entity_normalize,
        "graph_use_aliases": graph_use_aliases,
        "graph_seed_entity_df_power": graph_seed_entity_df_power,
        "graph_seed_weighting": graph_seed_weighting,
        "graph_seed_temp": graph_seed_temp,
        "graph_seed_mix_mode": graph_seed_mix_mode,
        "graph_seed_mix_alpha": graph_seed_mix_alpha,
        "cache_root": str(cache_root) if cache_root is not None else None,
        "tag": tag,
        "docs": len(docs),
        "queries": len(queries),
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    manifest_path = output_dir.parent / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")

    return output_dir
