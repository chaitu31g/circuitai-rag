"""End-to-end offline RAG orchestration for CircuitAI."""

from __future__ import annotations

import json
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.rag.prompt_builder import DatasheetPromptBuilder
from rag_pipeline.rag.reranker import CrossEncoderReranker
from rag_pipeline.rag.retriever import Retriever, RetrieverConfig
from rag_pipeline.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    top_k: int = 5
    max_context_chars: int = 6000
    default_trimmed_chunks: int = 2
    deduplicate_context: bool = True
    temperature: float = 0.2
    ollama_model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    request_timeout_s: int = 120
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 16
    reranker_blend_alpha: Optional[float] = None


class OllamaClient:
    """Small local Ollama API client using stdlib HTTP only."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        url: str = "http://localhost:11434/api/generate",
        timeout_s: int = 120,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout_s = timeout_s

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        timeout_s: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
            with urlopen(request, timeout=effective_timeout) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                text = parsed.get("response", "")
                if not text:
                    raise RuntimeError("Ollama returned empty response text.")
                return text.strip()
        except socket.timeout as exc:
            raise RuntimeError(
                "Ollama request timed out during local inference. "
                "Increase timeout (for CPU inference), reduce context, or use a smaller top-k."
            ) from exc
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise RuntimeError(
                    "Ollama request timed out during local inference. "
                    "Increase timeout (for CPU inference), reduce context, or use a smaller top-k."
                ) from exc
            raise RuntimeError(
                "Could not reach Ollama at http://localhost:11434. "
                "Ensure `ollama serve` is running locally."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON received from Ollama API.") from exc


class RAGPipeline:
    """Modular RAG pipeline: retrieve -> assemble context -> prompt -> generate."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[DatasheetPromptBuilder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[OllamaClient] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.retriever = retriever or Retriever(
            vector_store=ChromaStore(),
            embedder=BGEM3Embedder(),
            config=RetrieverConfig(top_k=self.config.top_k),
        )
        self.prompt_builder = prompt_builder or DatasheetPromptBuilder()
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranker_model,
            batch_size=self.config.reranker_batch_size,
        )
        self.llm_client = llm_client or OllamaClient(
            model=self.config.ollama_model,
            url=self.config.ollama_url,
            timeout_s=self.config.request_timeout_s,
        )
        # Future extension point: inject reranker here before context assembly.

    def assemble_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
        deduplicate: Optional[bool] = None,
        max_context_chunks: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Build bounded context preserving retrieval ranking and chunk atomicity."""
        if not retrieved_docs:
            return {"context": "", "used_docs": [], "trimmed_count": 0, "debug_events": []}

        budget = max_chars if max_chars is not None else self.config.max_context_chars
        do_dedupe = deduplicate if deduplicate is not None else self.config.deduplicate_context
        debug_events: List[str] = []

        seen_texts = set()
        candidate_docs: List[Dict[str, Any]] = []
        candidate_parts: List[str] = []
        total_candidate_chars = 0

        for doc in retrieved_docs:
            text = (doc.get("text") or "").strip()
            if not text:
                continue

            if do_dedupe:
                key = " ".join(text.split()).lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)

            formatted = self._format_context_chunk(doc)
            candidate_docs.append(doc)
            candidate_parts.append(formatted)
            total_candidate_chars += len(formatted) + (2 if len(candidate_parts) > 1 else 0)

        effective_max_chunks = max_context_chunks
        if effective_max_chunks is None and total_candidate_chars > budget:
            # Default small-LLM optimization: if context is oversized, force top-2 chunks.
            effective_max_chunks = self.config.default_trimmed_chunks
            if debug:
                event = (
                    f"context_chars={total_candidate_chars} exceeded max_context_chars={budget}; "
                    f"trimming to top {effective_max_chunks} chunks"
                )
                debug_events.append(event)
                logger.debug(event)

        used_docs: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        total = 0

        for doc, formatted in zip(candidate_docs, candidate_parts):
            if effective_max_chunks is not None and len(used_docs) >= effective_max_chunks:
                break

            # Preserve atomic chunks (tables/graphs) by all-or-nothing inclusion.
            next_len = len(formatted) + (2 if context_parts else 0)
            context_parts.append(formatted)
            total += next_len
            used_docs.append(doc)

        trimmed_count = len(candidate_docs) - len(used_docs)
        if debug and trimmed_count > 0 and not debug_events:
            event = (
                f"trimmed {trimmed_count} chunks via max_context_chunks={effective_max_chunks}"
            )
            debug_events.append(event)
            logger.debug(event)

        return {
            "context": "\n\n".join(context_parts),
            "used_docs": used_docs,
            "trimmed_count": max(trimmed_count, 0),
            "debug_events": debug_events,
        }

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
        max_context_chunks: Optional[int] = None,
        debug: bool = False,
        mode: str = "qa",
        use_reranker: bool = False,
        rerank_top_n: Optional[int] = None,
        rerank_blend_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a full offline RAG pass and return answer + retrieval diagnostics."""
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k or self.config.top_k,
            filters=filters,
        )
        vector_ranking = self._build_vector_ranking(retrieved_docs)
        rerank_error: Optional[str] = None
        dropped_after_rerank: List[str] = []
        reranked_docs = retrieved_docs

        if use_reranker and retrieved_docs:
            effective_rerank_top_n = min(rerank_top_n or len(retrieved_docs), len(retrieved_docs))
            blend_alpha = (
                rerank_blend_alpha
                if rerank_blend_alpha is not None
                else self.config.reranker_blend_alpha
            )
            try:
                reranked_docs = self.reranker.rerank(
                    query=query,
                    documents=retrieved_docs,
                    top_n=effective_rerank_top_n,
                    blend_alpha=blend_alpha,
                )
                kept_ids = {doc.get("id") for doc in reranked_docs}
                dropped_after_rerank = [
                    str(doc.get("id"))
                    for doc in retrieved_docs
                    if doc.get("id") not in kept_ids and doc.get("id") is not None
                ]
                if debug:
                    logger.debug("reranker enabled: kept=%d dropped=%d", len(reranked_docs), len(dropped_after_rerank))
            except Exception as exc:
                # Keep pipeline robust: fallback to vector ranking if reranker is unavailable.
                rerank_error = str(exc)
                reranked_docs = retrieved_docs
                if debug:
                    logger.warning("reranker failed; falling back to vector ranking: %s", rerank_error)

        reranked_order = self._build_reranked_order(reranked_docs)
        reranker_scores = self._build_reranker_scores(reranked_docs)

        assembled = self.assemble_context(
            reranked_docs,
            max_context_chunks=max_context_chunks,
            debug=debug,
        )
        context = assembled["context"]
        used_docs = assembled["used_docs"]

        if not used_docs:
            empty_json = [] if mode == "json_spec" else None
            return {
                "query": query,
                "answer": "[]" if mode == "json_spec" else "I could not find relevant datasheet context for this query.",
                "retrieved_docs": retrieved_docs,
                "reranked_docs": reranked_docs,
                "used_docs": [],
                "prompt": "",
                "trimmed_count": assembled.get("trimmed_count", 0),
                "debug_events": assembled.get("debug_events", []),
                "mode": mode,
                "chunk_ids_used": [],
                "prompt_length_chars": 0,
                "extracted_numeric_tokens": [],
                "parsed_json": empty_json,
                "parsed_json_count": len(empty_json) if empty_json is not None else 0,
                "use_reranker": use_reranker,
                "rerank_error": rerank_error,
                "original_vector_ranking": vector_ranking,
                "reranked_order": reranked_order,
                "reranker_scores": reranker_scores,
                "dropped_after_rerank": dropped_after_rerank,
            }

        chunk_ids_used = [d.get("id", "") for d in used_docs if d.get("id")]
        extracted_numeric_tokens: List[str] = []
        if mode in {"spec", "json_spec", "rag_answer"}:
            raw_context_text = "\n".join((d.get("text") or "") for d in used_docs)
            extracted_numeric_tokens = self._extract_numeric_tokens(raw_context_text)
            if debug:
                logger.debug("%s_mode numeric_tokens=%s", mode, extracted_numeric_tokens)
                logger.debug("%s_mode chunk_ids_used=%s", mode, chunk_ids_used)

        if mode == "spec":
            prompt = self.prompt_builder.build_spec_extraction_prompt(
                query=query,
                context=context,
                context_chunk_ids=chunk_ids_used,
            )
        elif mode == "json_spec":
            prompt = self.prompt_builder.build_json_spec_prompt(
                query=query,
                context=context,
                context_chunk_ids=chunk_ids_used,
            )
        elif mode == "rag_answer":
            # rag_answer mode is the primary grounded summarization path for UI use.
            prompt = self.prompt_builder.build_rag_answer_prompt(
                query=query,
                context=context,
                context_chunk_ids=chunk_ids_used,
            )
        else:
            prompt = self.prompt_builder.build(
                query=query,
                context=context,
                context_chunk_ids=chunk_ids_used,
            )

        if debug and mode in {"spec", "json_spec", "rag_answer"}:
            logger.debug("%s_mode prompt_length=%d", mode, len(prompt.full_prompt))

        llm_error: Optional[str] = None
        try:
            answer = self.llm_client.generate(
                prompt=prompt.full_prompt,
                temperature=self.config.temperature,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            llm_error = str(exc)
            answer = (
                "Retrieved relevant datasheet context, but local LLM generation failed. "
                f"Error: {llm_error}"
            )

        if mode == "rag_answer" and llm_error is None and used_docs and self._looks_unspecified(answer):
            # Guardrail for small local models: when retrieved context exists, force
            # grounded summarization instead of an "unspecified" response.
            answer = self._build_grounded_constraint_summary(query, used_docs)
            if debug:
                logger.debug("rag_answer_mode fallback activated due to unspecified response")

        parsed_json: Optional[List[Dict[str, Any]]] = None
        if mode == "json_spec" and llm_error is None:
            # Parse-and-normalize model output so downstream consumers can rely on JSON.
            parsed_json, parse_error = self._parse_json_spec_output(answer)
            if parse_error:
                logger.warning("json_spec parse failed; returning raw text. error=%s", parse_error)
            else:
                # Canonicalized JSON string for deterministic machine consumption.
                answer = json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))

        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "used_docs": used_docs,
            "prompt": prompt.full_prompt,
            "context": context,
            "llm_error": llm_error,
            "trimmed_count": assembled.get("trimmed_count", 0),
            "debug_events": assembled.get("debug_events", []),
            "timeout_used_s": timeout_s if timeout_s is not None else self.config.request_timeout_s,
            "mode": mode,
            "chunk_ids_used": chunk_ids_used,
            "prompt_length_chars": len(prompt.full_prompt),
            "extracted_numeric_tokens": extracted_numeric_tokens,
            "parsed_json": parsed_json,
            "parsed_json_count": len(parsed_json) if parsed_json is not None else 0,
            "use_reranker": use_reranker,
            "rerank_error": rerank_error,
            "original_vector_ranking": vector_ranking,
            "reranked_order": reranked_order,
            "reranker_scores": reranker_scores,
            "dropped_after_rerank": dropped_after_rerank,
        }

    @staticmethod
    def _build_vector_ranking(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranking: List[Dict[str, Any]] = []
        for rank, doc in enumerate(docs, start=1):
            ranking.append(
                {
                    "rank": rank,
                    "id": doc.get("id"),
                    "vector_score": doc.get("score"),
                }
            )
        return ranking

    @staticmethod
    def _build_reranked_order(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        order: List[Dict[str, Any]] = []
        for rank, doc in enumerate(docs, start=1):
            order.append(
                {
                    "rank": rank,
                    "id": doc.get("id"),
                    "vector_rank": doc.get("vector_rank"),
                    "vector_score": doc.get("vector_score", doc.get("score")),
                    "reranker_score": doc.get("reranker_score"),
                    "final_score": doc.get("final_score", doc.get("score")),
                }
            )
        return order

    @staticmethod
    def _build_reranker_scores(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores: List[Dict[str, Any]] = []
        for doc in docs:
            if doc.get("reranker_score") is None:
                continue
            scores.append(
                {
                    "id": doc.get("id"),
                    "reranker_score": doc.get("reranker_score"),
                    "final_score": doc.get("final_score"),
                }
            )
        return scores

    @staticmethod
    def _format_context_chunk(doc: Dict[str, Any]) -> str:
        meta = doc.get("metadata") or {}
        section = meta.get("section_name", "unknown_section")
        chunk_type = meta.get("chunk_type") or meta.get("chunkType") or "unknown_type"
        component = meta.get("componentId", "unknown_component")
        score = doc.get("score", 0.0)
        chunk_id = doc.get("id", "unknown_id")
        text = doc.get("text", "").strip()
        return (
            f"[Chunk {chunk_id}] component={component} section={section} "
            f"type={chunk_type} score={score}\n{text}"
        )

    @staticmethod
    def _extract_numeric_tokens(text: str) -> List[str]:
        """Extract numeric-like tokens for spec-mode debug visibility."""
        pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[A-Za-z\u00b5\u03bc\u00b0%]+)?"
        matches = re.findall(pattern, text or "")
        cleaned = [m.strip() for m in matches if m.strip()]
        # Preserve order while de-duplicating.
        seen = set()
        deduped: List[str] = []
        for token in cleaned:
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(token)
        return deduped

    @staticmethod
    def _looks_unspecified(answer: str) -> bool:
        text = (answer or "").lower()
        return (
            "does not specify this value" in text
            or "not specified" in text
            or "unspecified" in text
        )

    @staticmethod
    def _build_grounded_constraint_summary(query: str, used_docs: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for doc in used_docs:
            text = (doc.get("text") or "").strip()
            if text:
                lines.append(f"- {text}")
        if not lines:
            return "The datasheet context does not specify this value."
        return (
            f"Based on the retrieved datasheet constraints for '{query}', the relevant limits are:\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _parse_json_spec_output(text: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Validate JSON-spec output; fallback to raw text if parsing fails."""
        candidate = (text or "").strip()
        if not candidate:
            return None, "empty model output"

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Some local models still emit surrounding text; salvage array when possible.
            start = candidate.find("[")
            end = candidate.rfind("]")
            if start < 0 or end <= start:
                return None, "no JSON array found in output"
            snippet = candidate[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError as exc:
                return None, f"invalid JSON array: {exc}"

        if not isinstance(parsed, list):
            return None, "JSON root is not an array"

        normalized: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "parameter": str(item.get("parameter", "")),
                    "value": str(item.get("value", "")),
                    "unit": str(item.get("unit", "")),
                    "limit_type": str(item.get("limit_type", "")),
                    "conditions": str(item.get("conditions", "")),
                    "source_text": str(item.get("source_text", "")),
                }
            )
        return normalized, None
