from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from sprig.utils.entity_linking import apply_alias_map
from sprig.utils.ner import extract_entities
from sprig.utils.text import tokenize


@dataclass
class GraphIndex:
    doc_ids: List[str]
    entity_ids: List[str]
    entity_df: List[int]
    transition: csr_matrix
    edge_count: int


def build_bipartite_graph(
    doc_ids: List[str],
    docs: Iterable[str],
    min_entity_len: int = 2,
    ner_mode: str = "regex",
    entity_normalize: str = "none",
    entity_alias_map: Dict[str, str] | None = None,
    min_entity_df: int = 1,
    max_entity_df_ratio: float = 1.0,
    hub_penalty: float = 0.0,
    hub_top_ratio: float = 0.0,
    entity_edge_cap: int | None = None,
    normalization: str = "row",
) -> GraphIndex:
    # Build entity <-> doc bipartite graph with tf-idf edge weights
    entity_df: Dict[str, int] = {}
    doc_texts = list(docs)
    doc_entity_counts: List[Dict[str, int]] = []

    if ner_mode == "spacy":
        try:
            from sprig.utils.ner import _SPACY_LABELS, _nlp

            nlp = _nlp()
            for doc in nlp.pipe(doc_texts, batch_size=64):
                ents = [ent.text.strip() for ent in doc.ents if ent.label_ in _SPACY_LABELS]
                ents = apply_alias_map(ents, entity_alias_map, normalize_mode=entity_normalize)
                counts: Dict[str, int] = {}
                for ent in ents:
                    if len(ent) < min_entity_len:
                        continue
                    counts[ent] = counts.get(ent, 0) + 1
                doc_entity_counts.append(counts)
                for ent in counts:
                    entity_df[ent] = entity_df.get(ent, 0) + 1
        except Exception:
            for text in doc_texts:
                ents = extract_entities(text, mode=ner_mode, dedup=False, normalize="none")
                ents = apply_alias_map(ents, entity_alias_map, normalize_mode=entity_normalize)
                counts: Dict[str, int] = {}
                for ent in ents:
                    if len(ent) < min_entity_len:
                        continue
                    counts[ent] = counts.get(ent, 0) + 1
                doc_entity_counts.append(counts)
                for ent in counts:
                    entity_df[ent] = entity_df.get(ent, 0) + 1
    else:
        for text in doc_texts:
            ents = extract_entities(text, mode=ner_mode, dedup=False, normalize="none")
            ents = apply_alias_map(ents, entity_alias_map, normalize_mode=entity_normalize)
            counts: Dict[str, int] = {}
            for ent in ents:
                if len(ent) < min_entity_len:
                    continue
                counts[ent] = counts.get(ent, 0) + 1
            doc_entity_counts.append(counts)
            for ent in counts:
                entity_df[ent] = entity_df.get(ent, 0) + 1

    n_docs = len(doc_ids)
    max_df = max(1, int(max_entity_df_ratio * max(1, n_docs)))
    filtered_entities = [
        e for e, df in entity_df.items() if df >= min_entity_df and df <= max_df
    ]
    if hub_top_ratio > 0 and filtered_entities:
        ratio = min(1.0, max(0.0, float(hub_top_ratio)))
        cut = int(len(filtered_entities) * ratio)
        if cut > 0:
            sorted_ents = sorted(filtered_entities, key=lambda e: entity_df[e], reverse=True)
            removed = set(sorted_ents[:cut])
            filtered_entities = [e for e in filtered_entities if e not in removed]
    entity_to_idx = {e: i for i, e in enumerate(filtered_entities)}
    entity_df_list = [entity_df[e] for e in filtered_entities]
    n_entities = len(entity_to_idx)
    n_nodes = n_entities + n_docs

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    n_docs_f = float(max(1, n_docs))
    edge_count = 0

    if entity_edge_cap is not None and entity_edge_cap > 0:
        edges_by_entity: Dict[int, List[Tuple[int, float, float]]] = {}
        for d_idx, counts in enumerate(doc_entity_counts):
            for ent, tf in counts.items():
                if ent not in entity_to_idx:
                    continue
                e_idx = entity_to_idx[ent]
                df = entity_df[ent]
                idf = math.log((n_docs_f + 1.0) / (df + 1.0)) + 1.0
                weight = float(tf) * idf
                hub_weight = 1.0
                if hub_penalty > 0:
                    hub_weight = (float(df) + 1e-9) ** (-hub_penalty)
                edges_by_entity.setdefault(e_idx, []).append((d_idx, weight, hub_weight))

        for e_idx, edges in edges_by_entity.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            for d_idx, weight, hub_weight in edges[: int(entity_edge_cap)]:
                rows.append(e_idx)
                cols.append(n_entities + d_idx)
                data.append(weight)
                rows.append(n_entities + d_idx)
                cols.append(e_idx)
                data.append(weight * hub_weight)
                edge_count += 1
    else:
        for d_idx, counts in enumerate(doc_entity_counts):
            for ent, tf in counts.items():
                if ent not in entity_to_idx:
                    continue
                e_idx = entity_to_idx[ent]
                df = entity_df[ent]
                idf = math.log((n_docs_f + 1.0) / (df + 1.0)) + 1.0
                weight = float(tf) * idf
                # entity -> doc
                rows.append(e_idx)
                cols.append(n_entities + d_idx)
                data.append(weight)
                # doc -> entity (optionally downweight hubs by df)
                hub_weight = 1.0
                if hub_penalty > 0:
                    hub_weight = (float(df) + 1e-9) ** (-hub_penalty)
                rows.append(n_entities + d_idx)
                cols.append(e_idx)
                data.append(weight * hub_weight)
                edge_count += 1

    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    if normalization not in {"row", "sym", "none"}:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if normalization == "sym":
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        inv_sqrt = 1.0 / np.sqrt(deg)
        adj = adj.multiply(inv_sqrt[:, None])
        adj = adj.multiply(inv_sqrt[None, :])

    if normalization == "none":
        transition = adj.tocsr()
    else:
        # row-normalize to get stochastic transition
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv = 1.0 / row_sums
        transition = adj.multiply(inv[:, None]).tocsr()

    return GraphIndex(
        doc_ids=doc_ids,
        entity_ids=list(entity_to_idx.keys()),
        entity_df=entity_df_list,
        transition=transition,
        edge_count=edge_count,
    )


def build_term_bipartite_graph(
    doc_ids: List[str],
    docs: Iterable[str],
    min_df: int = 5,
    max_df_ratio: float = 0.2,
    normalization: str = "row",
) -> GraphIndex:
    # Build term <-> doc bipartite graph with tf-idf edge weights
    doc_texts = list(docs)
    term_df: Dict[str, int] = {}
    doc_term_counts: List[Dict[str, int]] = []

    for text in doc_texts:
        toks = tokenize(text)
        counts: Dict[str, int] = {}
        for tok in toks:
            if not tok:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        doc_term_counts.append(counts)
        for tok in counts:
            term_df[tok] = term_df.get(tok, 0) + 1

    n_docs = len(doc_ids)
    max_df = max(1, int(max_df_ratio * max(1, n_docs)))
    filtered_terms = [t for t, df in term_df.items() if df >= min_df and df <= max_df]
    term_to_idx = {t: i for i, t in enumerate(filtered_terms)}
    term_df_list = [term_df[t] for t in filtered_terms]
    n_terms = len(term_to_idx)
    n_nodes = n_terms + n_docs

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    n_docs_f = float(max(1, n_docs))
    edge_count = 0

    for d_idx, counts in enumerate(doc_term_counts):
        for tok, tf in counts.items():
            if tok not in term_to_idx:
                continue
            t_idx = term_to_idx[tok]
            df = term_df[tok]
            idf = math.log((n_docs_f + 1.0) / (df + 1.0)) + 1.0
            weight = float(tf) * idf
            rows.append(t_idx)
            cols.append(n_terms + d_idx)
            data.append(weight)
            rows.append(n_terms + d_idx)
            cols.append(t_idx)
            data.append(weight)
            edge_count += 1

    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    if normalization not in {"row", "sym", "none"}:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if normalization == "sym":
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        inv_sqrt = 1.0 / np.sqrt(deg)
        adj = adj.multiply(inv_sqrt[:, None])
        adj = adj.multiply(inv_sqrt[None, :])

    if normalization == "none":
        transition = adj.tocsr()
    else:
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv = 1.0 / row_sums
        transition = adj.multiply(inv[:, None]).tocsr()

    return GraphIndex(
        doc_ids=doc_ids,
        entity_ids=list(term_to_idx.keys()),
        entity_df=term_df_list,
        transition=transition,
        edge_count=edge_count,
    )


def ppr_search(
    index: GraphIndex,
    query: str,
    top_k: int = 5,
    alpha: float = 0.15,
    max_iter: int = 10,
    ner_mode: str = "regex",
    entity_normalize: str = "none",
    entity_alias_map: Dict[str, str] | None = None,
    seed_entity_df_power: float = 0.0,
    seed_terms: List[str] | None = None,
    seed_docs: List[Tuple[int, float]] | None = None,
    seed_weighting: str = "raw",
    seed_temp: float = 1.0,
    seed_mix_mode: str = "l1",
    seed_mix_alpha: float | None = None,
    mode: str = "power",
    tol: float | None = None,
    push_eps: float = 1e-4,
    push_max_steps: int = 200000,
) -> List[Tuple[int, float]]:
    n_entities = len(index.entity_ids)
    n_docs = len(index.doc_ids)
    n_nodes = n_entities + n_docs

    ent_to_idx = {e: i for i, e in enumerate(index.entity_ids)}
    entity_df = index.entity_df if index.entity_df else [1] * n_entities
    if seed_terms is None:
        seeds = extract_entities(
            query,
            mode=ner_mode,
            dedup=True,
            normalize="none",
        )
    else:
        seeds = seed_terms
    seeds = apply_alias_map(seeds, entity_alias_map, normalize_mode=entity_normalize)
    s_ent = np.zeros(n_nodes, dtype=np.float32)
    s_doc = np.zeros(n_nodes, dtype=np.float32)
    for ent in seeds:
        idx = ent_to_idx.get(ent)
        if idx is not None:
            weight = 1.0
            if seed_entity_df_power > 0 and idx < len(entity_df):
                df = max(1.0, float(entity_df[idx]))
                weight = df ** (-seed_entity_df_power)
            s_ent[idx] = max(s_ent[idx], weight)

    if seed_docs:
        weighted_docs = _normalize_seed_docs(seed_docs, seed_weighting, seed_temp)
        for doc_idx, weight in weighted_docs:
            if 0 <= doc_idx < n_docs:
                s_doc[n_entities + doc_idx] += max(0.0, float(weight))

    s = _mix_seed_vectors(s_ent, s_doc, seed_mix_mode, seed_mix_alpha)
    if s is None or s.sum() == 0:
        # fallback: uniform over docs
        doc_scores = np.ones(n_docs, dtype=np.float32)
        doc_scores /= doc_scores.sum()
        ranked = sorted(enumerate(doc_scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    r = s.copy()
    if mode not in {"power", "push"}:
        raise ValueError(f"Unsupported PPR mode: {mode}")
    if mode == "push":
        r = _ppr_push(
            index.transition,
            s,
            alpha=alpha,
            eps=push_eps,
            max_steps=push_max_steps,
        )
    else:
        transition_t = index.transition.transpose().tocsr()
        for _ in range(max_iter):
            next_r = alpha * s + (1 - alpha) * (transition_t @ r)
            if tol is not None:
                delta = float(np.abs(next_r - r).sum())
                r = next_r
                if delta < tol:
                    break
            else:
                r = next_r

    doc_scores = r[n_entities:]
    ranked = sorted(enumerate(doc_scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _normalize_seed_docs(
    seed_docs: List[Tuple[int, float]],
    mode: str = "raw",
    temp: float = 1.0,
) -> List[Tuple[int, float]]:
    if not seed_docs:
        return []
    if mode == "rank":
        return [(doc_idx, 1.0 / rank) for rank, (doc_idx, _) in enumerate(seed_docs, start=1)]

    scores = np.array([max(0.0, float(score)) for _, score in seed_docs], dtype=np.float32)
    if mode == "softmax":
        scale = max(1e-6, float(temp))
        scores = scores / scale
        scores = scores - scores.max()
        weights = np.exp(scores)
    elif mode == "raw":
        weights = scores
    else:
        raise ValueError(f"Unsupported seed weighting: {mode}")
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    return [(doc_idx, float(weights[i])) for i, (doc_idx, _) in enumerate(seed_docs)]


def _l1_normalize(vec: np.ndarray) -> np.ndarray:
    total = float(vec.sum())
    if total <= 0:
        return vec
    return vec / total


def _mix_seed_vectors(
    s_ent: np.ndarray,
    s_doc: np.ndarray,
    mode: str = "l1",
    alpha: float | None = None,
) -> np.ndarray | None:
    ent_sum = float(s_ent.sum())
    doc_sum = float(s_doc.sum())
    if ent_sum <= 0 and doc_sum <= 0:
        return None
    if mode == "l1":
        return _l1_normalize(s_ent + s_doc)
    if ent_sum <= 0:
        return _l1_normalize(s_doc)
    if doc_sum <= 0:
        return _l1_normalize(s_ent)

    ent_norm = _l1_normalize(s_ent)
    doc_norm = _l1_normalize(s_doc)
    if mode == "fixed":
        mix = 0.5 if alpha is None else float(alpha)
    elif mode == "auto":
        n_ent = int((s_ent > 0).sum())
        n_doc = int((s_doc > 0).sum())
        mix = (n_ent + 1.0) / (n_ent + n_doc + 2.0)
    else:
        raise ValueError(f"Unsupported seed mix mode: {mode}")
    mix = min(1.0, max(0.0, mix))
    return (mix * ent_norm) + ((1.0 - mix) * doc_norm)


def _ppr_push(
    transition: csr_matrix,
    seed: np.ndarray,
    alpha: float = 0.15,
    eps: float = 1e-4,
    max_steps: int = 200000,
) -> np.ndarray:
    n_nodes = seed.shape[0]
    p = np.zeros(n_nodes, dtype=np.float32)
    r = seed.astype(np.float32).copy()

    active: List[int] = [int(i) for i in np.where(r > eps)[0]]
    in_queue = set(active)
    steps = 0
    indptr = transition.indptr
    indices = transition.indices
    data = transition.data

    while active and steps < max_steps:
        v = active.pop()
        in_queue.discard(v)
        rv = float(r[v])
        if rv <= eps:
            continue
        p[v] += alpha * rv
        residual = (1.0 - alpha) * rv
        r[v] = 0.0

        start = indptr[v]
        end = indptr[v + 1]
        for idx in range(start, end):
            u = indices[idx]
            r[u] += residual * data[idx]
            if r[u] > eps and u not in in_queue:
                active.append(int(u))
                in_queue.add(int(u))
        steps += 1

    return p
