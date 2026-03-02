"""BGE-M3 compatible dense embedder using sentence-transformers.

Uses BAAI/bge-small-en-v1.5 (384-dim, ~133MB) by default.
Swap MODEL_NAME to BAAI/bge-m3 (1024-dim) when resources allow.

Design: BGEM3Embedder is pure — it only produces vectors.
It is deliberately decoupled from the vector DB layer so the
storage backend (Chroma, Qdrant, Milvus) can be swapped freely.
"""

from __future__ import annotations

import logging
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — swap to BAAI/bge-m3 when resources allow (1024-dim, ~2.3GB)
# ---------------------------------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 16


class BGEM3Embedder:
    """Dense embedder backed by sentence-transformers.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Number of texts encoded per GPU/CPU pass.
        device: 'cuda' or 'cpu', detected automatically.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        import torch # Lazy load
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading — avoids loading the model until first use
    # ------------------------------------------------------------------

    @property
    def model(self):
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            logger.info("Loading %s on %s ...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim = self._model.get_sentence_embedding_dimension()
            logger.info("Model ready — dim=%d", dim)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings and return normalized float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float lists, one per input text. Length matches input.
        """
        if not texts:
            return []

        logger.info("Embedding %d texts (batch_size=%d) ...", len(texts), self.batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine similarity = dot product
            show_progress_bar=len(texts) > 50,
        )
        # embeddings is a numpy ndarray — convert to plain Python lists
        return embeddings.tolist()

    @property
    def embedding_dim(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        return self.model.get_sentence_embedding_dimension()
