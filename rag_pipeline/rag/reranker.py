"""Cross-encoder reranker for post-retrieval precision improvement."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Re-rank retrieved documents using a cross-encoder relevance model.

    Design notes:
    - Lazy model loading avoids startup overhead when reranking is disabled.
    - CPU-compatible by default, but uses CUDA if available.
    - Supports optional blending with vector similarity for stability.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            import torch

            actual_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading reranker model %s on %s ...", self.model_name, actual_device)
            self._model = CrossEncoder(self.model_name, device=actual_device)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        blend_alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return documents sorted by reranker score (or blended score).

        Args:
            query: User query.
            documents: Retrieved docs (expects 'text' and optional vector 'score').
            top_n: Number of reranked docs to keep.
            blend_alpha: Optional blend weight for reranker score.
                final_score = alpha * reranker_score + (1 - alpha) * vector_score
        """
        if not documents:
            return []

        effective_top_n = min(top_n or len(documents), len(documents))
        pairs = [(query, (doc.get("text") or "")) for doc in documents]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        scored_docs: List[Dict[str, Any]] = []
        for idx, (doc, score) in enumerate(zip(documents, scores), start=1):
            reranker_score = float(score)
            vector_score = doc.get("score")

            if blend_alpha is not None and isinstance(vector_score, (int, float)):
                alpha = max(0.0, min(1.0, float(blend_alpha)))
                final_score = alpha * reranker_score + (1.0 - alpha) * float(vector_score)
            else:
                final_score = reranker_score

            enriched = dict(doc)
            enriched["vector_score"] = float(vector_score) if isinstance(vector_score, (int, float)) else None
            enriched["reranker_score"] = reranker_score
            enriched["final_score"] = final_score
            enriched["vector_rank"] = idx
            scored_docs.append(enriched)

        scored_docs.sort(key=lambda item: item.get("final_score", float("-inf")), reverse=True)
        for rank, doc in enumerate(scored_docs, start=1):
            doc["rerank_rank"] = rank
        return scored_docs[:effective_top_n]

