from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from sprig.utils.text import tokenize


@dataclass
class BM25Index:
    docs: List[str]
    doc_len: List[int]
    doc_freq: Dict[str, int]
    postings: Dict[str, List[Tuple[int, int]]]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: Iterable[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        docs_list = list(docs)
        doc_tokens = [tokenize(d) for d in docs_list]
        doc_freq: Dict[str, int] = {}
        postings: Dict[str, List[Tuple[int, int]]] = {}
        doc_len: List[int] = []
        total_len = 0
        for i, toks in enumerate(doc_tokens):
            dl = len(toks)
            doc_len.append(dl)
            total_len += dl
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                doc_freq[t] = doc_freq.get(t, 0) + 1
                postings.setdefault(t, []).append((i, count))
        avgdl = total_len / max(1, len(doc_tokens))
        return cls(docs_list, doc_len, doc_freq, postings, avgdl, k1=k1, b=b)

    def _idf(self, term: str, n_docs: int) -> float:
        # BM25+ style idf smoothing
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_terms = tokenize(query)
        n_docs = len(self.docs)
        scores = [0.0] * n_docs
        for term in q_terms:
            idf = self._idf(term, n_docs)
            if idf == 0:
                continue
            for i, tf in self.postings.get(term, []):
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1)) / (denom + 1e-9)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_weighted(self, term_weights: Dict[str, float], top_k: int = 5) -> List[Tuple[int, float]]:
        n_docs = len(self.docs)
        scores = [0.0] * n_docs
        for term, weight in term_weights.items():
            if weight == 0:
                continue
            idf = self._idf(term, n_docs)
            if idf == 0:
                continue
            for i, tf in self.postings.get(term, []):
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                scores[i] += weight * idf * (tf * (self.k1 + 1)) / (denom + 1e-9)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
