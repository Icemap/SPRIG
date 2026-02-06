from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class CrossEncoderReranker:
    model_name: str
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install with optional deps: pip install 'sprig[vector]'."
            ) from exc
        self._model = CrossEncoder(self.model_name, device="cpu")

    def rerank(self, query: str, docs: Iterable[str], top_k: int) -> List[Tuple[int, float]]:
        pairs = [[query, doc] for doc in docs]
        if not pairs:
            return []
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
