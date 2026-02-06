from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Any, Optional

import numpy as np


@dataclass
class DenseIndex:
    embeddings: np.ndarray
    doc_ids: List[str]
    hnsw: Optional[Any] = None


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        use_hnsw: bool = True,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name, device="cpu")
        self.use_hnsw = use_hnsw
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

    def build_index(self, doc_ids: List[str], docs: Iterable[str]) -> DenseIndex:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        docs_list = list(docs)
        cache_key = _cache_key(doc_ids, docs_list, self.model_name)
        cache_path = self.cache_dir / f"dense_{cache_key}.npz"
        hnsw_path = self.cache_dir / f"dense_{cache_key}_m{self.hnsw_m}_efc{self.hnsw_ef_construction}.hnsw"
        hnsw = None
        if cache_path.exists():
            data = np.load(cache_path)
            embeddings = data["embeddings"]
            doc_ids = json.loads(data["doc_ids"].item())
            if self.use_hnsw:
                hnsw = _load_hnsw(hnsw_path, dim=embeddings.shape[1])
                if hnsw is not None:
                    hnsw.set_ef(self.hnsw_ef_search)
                if hnsw is None:
                    hnsw = _build_hnsw(embeddings, hnsw_path, self)
            return DenseIndex(embeddings=embeddings, doc_ids=doc_ids, hnsw=hnsw)

        embeddings = self.model.encode(docs_list, show_progress_bar=True, batch_size=32)
        embeddings = _l2_normalize(embeddings)
        np.savez_compressed(cache_path, embeddings=embeddings, doc_ids=json.dumps(doc_ids))
        if self.use_hnsw:
            hnsw = _build_hnsw(embeddings, hnsw_path, self)
        return DenseIndex(embeddings=embeddings, doc_ids=doc_ids, hnsw=hnsw)

    def search(self, index: DenseIndex, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], show_progress_bar=False)
        q_emb = _l2_normalize(q_emb)[0]
        if index.hnsw is not None:
            labels, distances = index.hnsw.knn_query(q_emb, k=top_k)
            labels = labels[0].tolist()
            distances = distances[0].tolist()
            # For cosine in hnswlib: distance in [0, 2], smaller is better
            sims = [1.0 - d / 2.0 for d in distances]
            return list(zip(labels, sims))

        scores = index.embeddings @ q_emb
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def _cache_key(doc_ids: List[str], docs: Iterable[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for doc_id, doc in zip(doc_ids, docs):
        h.update(doc_id.encode("utf-8"))
        h.update(doc.encode("utf-8"))
    return h.hexdigest()[:16]


def _load_hnsw(hnsw_path: Path, dim: int):
    if not hnsw_path.exists():
        return None
    try:
        import hnswlib

        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(hnsw_path))
        return index
    except Exception:
        return None


def _build_hnsw(embeddings: np.ndarray, hnsw_path: Path, retriever: DenseRetriever):
    try:
        import hnswlib

        dim = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=embeddings.shape[0],
            ef_construction=retriever.hnsw_ef_construction,
            M=retriever.hnsw_m,
        )
        index.add_items(embeddings, list(range(embeddings.shape[0])))
        index.set_ef(retriever.hnsw_ef_search)
        index.save_index(str(hnsw_path))
        return index
    except Exception:
        return None
