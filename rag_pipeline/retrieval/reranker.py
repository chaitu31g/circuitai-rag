import logging
import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BGEM3Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        # Explicitly use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker {model_name} on {self.device}...")
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)
        logger.info("Reranker loaded successfully.")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Scores the retrieved chunks against the query and returns the top_k most relevant chunks.
        """
        if not chunks:
            return []

        # Prepare pairs of (query, chunk_text)
        pairs = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            pairs.append([query, text])

        # Compute relevance scores
        logger.info(f"Scoring {len(pairs)} chunks with the reranker...")
        scores = self.model.predict(pairs)

        # Attach scores to chunks
        for chunk, score in zip(chunks, scores):
            # We can store the score directly in the dict
            chunk["reranker_score"] = float(score)

        # Sort chunks by score descending
        sorted_chunks = sorted(chunks, key=lambda x: x.get("reranker_score", 0.0), reverse=True)

        # Return top-k
        return sorted_chunks[:top_k]
