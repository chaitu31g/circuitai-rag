import json
import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.config import config
from backend.models import create_job, get_job, update_job, jobs_db, IngestionJob, JobStatus
from backend.llm.hf_llm import load_model_once, generate_response, stream_response, build_prompt, build_synthesis_prompt
from rag_pipeline.services.ingest_service import ingest_pdf_pipeline, get_embedder
from rag_pipeline.vectordb.chroma_store import ChromaStore
from rag_pipeline.rag.rag_pipeline import RAGPipeline, RAGConfig
from rag_pipeline.rag.retriever import (
    Retriever,
    RetrieverConfig,
    classify_query_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CircuitAI RAG API")

# Allow all origins — required for Google Colab + ngrok tunnels where
# the frontend URL changes every session.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-load LLM in background so first request isn't slow ───────────

@app.on_event("startup")
def preload_hf_model():
    import threading
    def _load():
        try:
            load_model_once(config.hf_model)
        except Exception as exc:
            logger.warning("HF model pre-load failed (will retry on first query): %s", exc)
    threading.Thread(target=_load, daemon=True, name="hf-model-loader").start()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    """Serialize a dict as a single SSE data line."""
    return f"data: {json.dumps(obj)}\n\n"


def _keepalive() -> str:
    """SSE comment — invisible to the browser but resets Cloudflare's 100s proxy timeout."""
    return ": keep-alive\n\n"


# Keywords that indicate a counting intent
_COUNT_KEYWORDS = {"how many", "number of", "count", "total"}


def _is_graph_query(query: str) -> bool:
    """Return True if the query is asking about a graph, chart, or characteristic curve."""
    return classify_query_type(query) == "graph_query"


def _is_count_graph_query(query: str) -> bool:
    """Return True when user is asking HOW MANY graphs/figures exist."""
    q = query.lower()
    has_count = any(kw in q for kw in _COUNT_KEYWORDS)
    has_graph = any(kw in q for kw in {"graph", "graphs", "figure", "figures", "plot", "plots", "chart", "charts"})
    return has_count and has_graph


def _get_all_figure_chunks(
    store: "ChromaStore", component_filter: str | None
) -> list[dict]:
    """Retrieve ALL figure chunks from ChromaDB using metadata filtering.

    Tries two filter strategies to handle both old ingestions (chunk_type=figure)
    and new ingestions (type=figure) without requiring a re-index.

    Strategy A: $or  type=figure  OR  chunk_type=figure   (single call)
    Strategy B: two separate .get() calls merged          (fallback)
    """
    collection = store.collection

    def _build_base_filter(field: str) -> dict:
        if component_filter:
            return {
                "$and": [
                    {"part_number": {"$eq": component_filter}},
                    {field: {"$eq": "figure"}},
                ]
            }
        return {field: {"$eq": "figure"}}

    seen_ids: set[str] = set()
    results: list[dict] = []

    # Strategy A: try $or (supported in ChromaDB ≥ 0.4.x)
    try:
        if component_filter:
            where = {
                "$and": [
                    {"part_number": {"$eq": component_filter}},
                    {"$or": [{"type": {"$eq": "figure"}}, {"chunk_type": {"$eq": "figure"}}]},
                ]
            }
        else:
            where = {"$or": [{"type": {"$eq": "figure"}}, {"chunk_type": {"$eq": "figure"}}]}

        raw = collection.get(where=where, include=["documents", "metadatas", "ids"])
        ids   = raw.get("ids", []) or []
        docs  = raw.get("documents", []) or []
        metas = raw.get("metadatas") or [{}] * len(ids)
        for did, dtxt, dmeta in zip(ids, docs, metas):
            if did not in seen_ids:
                seen_ids.add(did)
                results.append({"id": did, "text": dtxt, "metadata": dmeta, "score": 1.0})
        logger.info("_get_all_figure_chunks ($or): found %d chunks", len(results))
        return results
    except Exception as exc_a:
        logger.debug("$or filter not supported, falling back to two-pass: %s", exc_a)

    # Strategy B: two separate passes (backward compat)
    for field in ("type", "chunk_type"):
        try:
            raw   = collection.get(
                where=_build_base_filter(field),
                include=["documents", "metadatas", "ids"],
            )
            ids   = raw.get("ids",       []) or []
            docs  = raw.get("documents", []) or []
            metas = raw.get("metadatas")    or [{}] * len(ids)
            for did, dtxt, dmeta in zip(ids, docs, metas):
                if did not in seen_ids:
                    seen_ids.add(did)
                    results.append({"id": did, "text": dtxt, "metadata": dmeta, "score": 1.0})
        except Exception as exc_b:
            logger.debug("Figure filter on field '%s' failed: %s", field, exc_b)

    logger.info("_get_all_figure_chunks (two-pass): found %d chunks", len(results))
    return results


def _build_retrieval(query: str, top_k: int, component_filter: str | None):
    """Retrieve context for a user query with graph-aware strategy.

    Retrieval modes
    ───────────────
    • Count query  ("how many graphs"):
          Uses _get_all_figure_chunks() to fetch ALL figure chunks
          deterministically — no vector similarity, no LLM guessing.
          Prepends a synthetic summary chunk with the exact count.

    • Graph query  ("explain the graphs", "list the graphs"):
          Fetches ALL figure chunks via metadata filter so every figure is
          surfaced, then also runs a standard semantic pass and merges.

    • Normal query:
          Standard top-k vector similarity retrieval.
    """
    store = ChromaStore(
        persist_dir=Path(config.chroma_persist_dir),
        collection_name=config.chroma_collection,
    )
    embedder = get_embedder()
    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
        config=RetrieverConfig(top_k=top_k),
    )

    # ── Deterministic graph-count shortcut ───────────────────────────────────
    if _is_count_graph_query(query):
        try:
            retrieved = _get_all_figure_chunks(store, component_filter)
            count = len(retrieved)
            logger.info("Count-graph query: found %d figure chunks", count)

            # Only count actual 'graph' figure_type chunks if possible;
            # fall back to total figure count when figure_type metadata is absent.
            graph_only = [
                r for r in retrieved
                if (r.get("metadata") or {}).get("figure_type") == "graph"
            ]
            if graph_only:
                graph_count = len(graph_only)
                comp_label = f" for '{component_filter}'" if component_filter else ""
                direct_answer = (
                    f"There are {graph_count} graph{'' if graph_count == 1 else 's'} "
                    f"in the datasheet{comp_label} "
                    f"(plus {count - graph_count} other figure(s) such as circuit diagrams and package drawings)."
                )
            else:
                comp_label = f" for '{component_filter}'" if component_filter else ""
                direct_answer = (
                    f"There are {count} figure{'' if count == 1 else 's'}/graph{'' if count == 1 else 's'} "
                    f"in the datasheet{comp_label}."
                )

            summary_text = (
                f"Figure Count Summary: {direct_answer}\n\n"
                "Figure list:\n" +
                "\n".join(
                    f"- Figure {i+1} (page {(r.get('metadata') or {}).get('page', '?')}): "
                    f"{(r.get('text') or '')[:120]}"
                    for i, r in enumerate(retrieved[:30])
                )
            )
            summary_doc = {
                "id": "__figure_count_summary__",
                "text": summary_text,
                "metadata": {"section_name": "figure_count", "chunk_type": "summary"},
                "score": 1.0,
            }
            retrieved = [summary_doc] + retrieved
        except Exception as exc:
            logger.warning("Count-graph retrieval failed, falling back to vector search: %s", exc)
            direct_answer = None
            retrieved = retriever.retrieve(query=query, top_k=top_k, filters=None)
        else:
            # Return early with direct answer — no LLM needed for counting
            sources = [
                {
                    "id":        d.get("id", ""),
                    "text":      d.get("text", "")[:400],
                    "score":     round(d.get("score", 0), 3),
                    "component": (d.get("metadata") or {}).get("part_number", "unknown"),
                    "section":   (d.get("metadata") or {}).get("section_name", ""),
                    "type":      (d.get("metadata") or {}).get("chunk_type", ""),
                }
                for d in retrieved
            ]
            return direct_answer, sources, True  # True = direct_answer, skip LLM


    # ── Pass 1: standard semantic retrieval ──────────────────────────────────
    part_filter = {"part_number": component_filter} if component_filter else None
    retrieved = retriever.retrieve(query=query, top_k=top_k, filters=part_filter)

    # ── Pass 2: full figure retrieval for graph queries ───────────────────────
    if _is_graph_query(query):
        try:
            figure_chunks = _get_all_figure_chunks(store, component_filter)
            seen_ids = {d.get("id") for d in retrieved}
            added = 0
            for chunk in figure_chunks:
                if chunk.get("id") not in seen_ids:
                    retrieved.append(chunk)
                    seen_ids.add(chunk.get("id"))
                    added += 1
            logger.info(
                "Graph query detected — added %d figure chunk(s) via metadata filter", added
            )
        except Exception as exc:
            logger.warning("Figure metadata retrieval failed (non-fatal): %s", exc)

    # ── Context assembly ──────────────────────────────────────────────────────
    # No hard chunk cap here; rely on max_context_chars=12000 to bound size.
    # The section-summarization path (_build_retrieval_and_summarize) handles
    # its own budget via the reranker + SectionSummarizer.
    pipeline = RAGPipeline(
        retriever=retriever,
        config=RAGConfig(
            top_k=top_k,
            max_context_chars=20000,    # raised: full elec-char table can be ~15k chars
            default_trimmed_chunks=20,  # raised from 2: stops silent row-drops
        ),
    )
    max_ctx = None  # let assemble_context use char budget instead of hard chunk cap
    if _is_graph_query(query):
        max_ctx = 20  # keep graph path bounded since it injects ALL figures
    assembled = pipeline.assemble_context(retrieved, max_context_chunks=max_ctx)

    sources = [
        {
            "id":        d.get("id", ""),
            "text":      d.get("text", "")[:400],
            "score":     round(d.get("score", 0), 3),
            "component": (d.get("metadata") or {}).get("part_number", "unknown"),
            "section":   (d.get("metadata") or {}).get("section_name", ""),
            "type":      (d.get("metadata") or {}).get("chunk_type", ""),
        }
        for d in assembled["used_docs"]
    ]
    return assembled["context"], sources, False


def _build_retrieval_and_summarize(
    query: str,
    top_k: int,
    component_filter: str | None,
) -> tuple:
    """New-architecture pipeline: 50-chunk retrieval → reranker → section summarization.

    Steps
    ─────
    1. Retrieve top-50 chunks (wide pool).
    2. Rerank with CrossEncoderReranker → top-20 most relevant chunks.
    3. Group chunks by section (features, electrical_characteristics, etc.).
    4. Summarize each section independently via LLM.
    5. Return structured multi-section context + sources.
    """
    from functools import partial
    from rag_pipeline.rag.rag_pipeline import SectionSummarizer
    from rag_pipeline.rag.reranker import CrossEncoderReranker

    store = ChromaStore(
        persist_dir=Path(config.chroma_persist_dir),
        collection_name=config.chroma_collection,
    )
    embedder = get_embedder()
    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
        config=RetrieverConfig(top_k=top_k),
    )

    # ── Deterministic graph-count shortcut (unchanged) ────────────────────────
    if _is_count_graph_query(query):
        # Delegate to original path — no summarization needed for counting
        return _build_retrieval(query, top_k, component_filter)

    # ── Step 1: Retrieve top-50 ───────────────────────────────────────────────
    part_filter = {"part_number": component_filter} if component_filter else None
    retrieved = retriever.retrieve(query=query, top_k=top_k, filters=part_filter)

    # Inject ALL figure chunks for graph queries
    if _is_graph_query(query):
        try:
            figure_chunks = _get_all_figure_chunks(store, component_filter)
            seen_ids = {d.get("id") for d in retrieved}
            for chunk in figure_chunks:
                if chunk.get("id") not in seen_ids:
                    retrieved.append(chunk)
                    seen_ids.add(chunk.get("id"))
        except Exception as exc:
            logger.warning("Figure metadata retrieval failed (non-fatal): %s", exc)

    if not retrieved:
        return "No relevant context found in the knowledge base for this query.", [], False

    # ── Step 2: Rerank → top-20 ───────────────────────────────────────────────
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    try:
        reranked = reranker.rerank(
            query=query,
            documents=retrieved,
            top_n=min(20, len(retrieved)),
        )
        logger.info(
            "Reranker: %d → %d chunks", len(retrieved), len(reranked)
        )
    except Exception as exc:
        logger.warning("Reranker unavailable, using retrieval order: %s", exc)
        reranked = retrieved[:20]

    # Sources built from reranked set (before summarization)
    sources = [
        {
            "id":        d.get("id", ""),
            "text":      d.get("text", "")[:400],
            "score":     round(d.get("score", d.get("final_score", 0)), 3),
            "component": (d.get("metadata") or {}).get("part_number", "unknown"),
            "section":   (d.get("metadata") or {}).get("section_name", ""),
            "type":      (d.get("metadata") or {}).get("chunk_type", ""),
        }
        for d in reranked
    ]

    # ── Step 3+4: Section grouping + LLM summarization ───────────────────────
    def _llm_fn(prompt_str: str) -> str:
        """Wrap generate_response for a plain-string prompt in the section summarizer."""
        from backend.llm.hf_llm import generate_response as _gen
        # build_prompt expects (context, query) but summarization prompts are self-contained;
        # pass the full prompt as context with an empty query sentinel.
        msgs = [
            {"role": "system", "content": "You are an expert electronics engineer."},
            {"role": "user",   "content": prompt_str},
        ]
        return _gen(
            prompt=msgs,
            model_id=config.hf_model,
            max_new_tokens=200,
            temperature=0.3,
        )

    summarizer = SectionSummarizer(
        llm_fn=_llm_fn,
        max_new_tokens=200,
        temperature=0.3,
    )
    section_context, summaries = summarizer.build_summarized_context(reranked)
    logger.info(
        "Section summarization complete: sections=%s", list(summaries.keys())
    )

    if not section_context.strip():
        # Fallback: assemble raw context
        raw_ctx = "\n\n".join((d.get("text") or "") for d in reranked[:8])
        return raw_ctx, sources, False

    return section_context, sources, False


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "CircuitAI RAG backend running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    query: str
    component_filter: str | None = None
    top_k: int = 50
    max_new_tokens: int = 1200
    temperature: float = 0.1
    use_section_summary: bool = False

def format_exact_match_table(query: str, sources: list) -> str:
    import re
    def normalize_param(p: str) -> str:
        if not p:
            return ""
        p = str(p).lower().replace("_", " ")
        p = re.sub(r"\s+", " ", p).strip()
        return p

    user_query_norm = normalize_param(query)
    print(f"DEBUG: Normalized user query: '{user_query_norm}'", flush=True)
    
    exact_matches = []
    for doc in sources:
        meta = doc.get("metadata", {})
        param_norm = normalize_param(meta.get("parameter", ""))
        print(f"DEBUG: Stored param: '{param_norm}'", flush=True)
        if param_norm == user_query_norm:
            exact_matches.append(meta)
            
    if not exact_matches:
        print(f"DEBUG: No exact match found for '{user_query_norm}'. Trying substring.", flush=True)
        for doc in sources:
            meta = doc.get("metadata", {})
            param_norm = normalize_param(meta.get("parameter", ""))
            if param_norm and (user_query_norm in param_norm or param_norm in user_query_norm):
                exact_matches.append(meta)
                
    if not exact_matches:
        return ""
        
    all_columns = []
    for m in exact_matches:
        for k in m.keys():
            k_lower = k.lower()
            if k_lower not in ["component", "type", "source", "chunk_type", "id", "score", "part_number", "page", "table_index"]:
                if k not in all_columns and k_lower not in [c.lower() for c in all_columns]:
                    all_columns.append(k)
                    
    ordered_cols = []
    for pref in ["parameter", "symbol", "condition", "min", "typ", "max", "value", "unit"]:
        for c in all_columns:
             if c.lower() == pref and c not in ordered_cols:
                 ordered_cols.append(c)
    for c in all_columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
            
    def extract_temp(m):
        for k, v in m.items():
            if "condition" in k.lower() or "test" in k.lower():
                t_match = re.search(r"(-?\d+)\s*(?:°|deg)?C", str(v), re.IGNORECASE)
                if t_match:
                    return float(t_match.group(1))
        return 0.0
        
    exact_matches.sort(key=extract_temp)
    
    header_row = "| " + " | ".join([c.capitalize() for c in ordered_cols]) + " |"
    sep_row = "| " + " | ".join(["---"] * len(ordered_cols)) + " |"
    
    rows = []
    for m in exact_matches:
        row_vals = []
        for c in ordered_cols:
            val = m.get(c, m.get(c.lower(), "-"))
            row_vals.append(str(val))
        rows.append("| " + " | ".join(row_vals) + " |")
        
    return "\n".join([header_row, sep_row] + rows)



@app.post("/chat")
def chat(req: ChatRequest):
    """RAG query — returns the full answer in one JSON response (non-streaming)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        if req.use_section_summary:
            result = _build_retrieval_and_summarize(req.query, req.top_k, req.component_filter)
        else:
            result = _build_retrieval(req.query, req.top_k, req.component_filter)
        context_or_answer, sources, is_direct = result

        # Deterministic direct-answer path (e.g., graph count queries)
        if is_direct:
            return {"answer": context_or_answer, "sources": sources}

        context = context_or_answer
        if not context:
            answer = "No relevant context found in the knowledge base for this query."
        else:
            table_answer = format_exact_match_table(req.query, sources)
            if table_answer:
                answer = table_answer
            else:
                answer = "No exact match found for parameter: " + req.query

        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """RAG query with real SSE token streaming via Cloudflare Tunnel.

    SSE event format:
      data: {"type": "sources", "sources": [...]}
      data: {"type": "token",   "token":   "..."}
      data: {"type": "done"}
      data: {"type": "error",   "message": "..."}

    Keep-alive comments (`: keep-alive`) are sent periodically to prevent
    Cloudflare's 100-second idle-connection timeout from killing long
    LLM inference runs.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def event_generator():
        try:
            # ── 1. Keep connection alive during retrieval (can take a few seconds) ─
            yield _keepalive()

            # ── 2. Retrieve & assemble / summarize context ────────────────────────
            if req.use_section_summary:
                result = _build_retrieval_and_summarize(req.query, req.top_k, req.component_filter)
            else:
                result = _build_retrieval(req.query, req.top_k, req.component_filter)
            context_or_answer, sources, is_direct = result

            # ── 3. Send sources immediately so the UI shows them before generation
            yield _sse({"type": "sources", "sources": sources})

            # ── 4a. Deterministic direct answer (e.g., graph count) ───────────────
            if is_direct:
                yield _sse({"type": "token", "token": context_or_answer})
                yield _sse({"type": "done"})
                return

            # ── 4b. No context → short-circuit ───────────────────────────────────
            context = context_or_answer
            if not context:
                yield _sse({"type": "token", "token": "No relevant context found in the knowledge base for this query."})
                yield _sse({"type": "done"})
                return

            # ── 5. Bypass LLM for Exact Extraction ───────────────────────────────
            table_answer = format_exact_match_table(req.query, sources)
            if table_answer:
                answer = table_answer
            else:
                answer = "No exact match found for parameter: " + req.query
                
            yield _sse({"type": "token", "token": answer})
            yield _keepalive()

            yield _sse({"type": "done"})

        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",  # no-transform stops CF compressing SSE
            "X-Accel-Buffering": "no",                  # disable nginx / CF buffering
            "Access-Control-Allow-Origin": "*",          # explicit CORS for SSE preflight
        },
    )



# ── Figure / graph metadata endpoints ─────────────────────────────────────────

@app.get("/figures")
def get_figures(component: str | None = None):
    """Return all figure chunks with structured metadata.

    Query params
    ────────────
    component : Optional part_number to filter by component.

    Response
    ────────
    {
        "total_figures": int,
        "graph_count":   int,
        "diagram_count": int,
        "figures": [
            {
                "id":          str,
                "figure_type": str,   # "graph" | "diagram" | "unknown"
                "page":        int | None,
                "caption":     str,
                "component":   str,
                "text":        str,   # first 300 chars of chunk text
            }
        ]
    }
    """
    try:
        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
        chunks = _get_all_figure_chunks(store, component)
        figures = []
        for ch in chunks:
            meta = ch.get("metadata") or {}
            figures.append({
                "id":          ch.get("id", ""),
                "figure_type": meta.get("figure_type", "unknown"),
                "page":        meta.get("page"),
                "caption":     meta.get("caption", ""),
                "component":   meta.get("part_number", "unknown"),
                "text":        (ch.get("text") or "")[:300],
            })
        graph_count   = sum(1 for f in figures if f["figure_type"] == "graph")
        diagram_count = sum(1 for f in figures if f["figure_type"] == "diagram")
        return {
            "total_figures": len(figures),
            "graph_count":   graph_count,
            "diagram_count": diagram_count,
            "figures":       figures,
        }
    except Exception as exc:
        logger.error("Figures endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Library endpoints ─────────────────────────────────────────────────────────

@app.get("/library")
def get_library():
    """Return all unique components stored in ChromaDB with chunk stats + file metadata."""
    try:
        try:
            store = ChromaStore(
                persist_dir=Path(config.chroma_persist_dir),
                collection_name=config.chroma_collection,
            )
            logger.info("✅ ChromaStore initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Vector store initialization failed: {e}")
            
        components = store.get_library()

        knowledge_dir = Path(config.knowledge_dir)
        pdfs_dir = Path(config.pdfs_dir)
        from datetime import datetime, timezone

        for comp in components:
            cid = comp["component_id"]
            knowledge_file = knowledge_dir / f"{cid}_knowledge.json"
            comp["ingested_at"] = (
                datetime.fromtimestamp(knowledge_file.stat().st_mtime, tz=timezone.utc).isoformat()
                if knowledge_file.exists() else None
            )
            pdf_file = next(
                (p for p in pdfs_dir.glob("*") if p.stem.replace(" ", "_").lower() == cid.lower()),
                None,
            )
            comp["file_size_kb"] = round(pdf_file.stat().st_size / 1024, 1) if pdf_file else None

        return {
            "total_components": len(components),
            "total_vectors": store.count(),
            "components": components,
        }
    except Exception as e:
        logger.error("Library fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Upload & job management endpoints ─────────────────────────────────────────

@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    jobs = []
    for file in files:
        job_id = str(uuid.uuid4())

        if not file.filename.lower().endswith(".pdf"):
            job = create_job(job_id=job_id, file_name=file.filename)
            update_job(job_id, status="failed", error_message=f"{file.filename} is not a PDF")
            jobs.append(job)
            continue

        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        if file_size_mb > config.max_file_size_mb:
            job = create_job(job_id=job_id, file_name=file.filename)
            update_job(job_id, status="failed", error_message=f"Exceeds {config.max_file_size_mb} MB limit")
            jobs.append(job)
            continue

        safe_filename = file.filename.replace(" ", "_").replace("/", "")
        file_path = Path(config.upload_dir) / f"{job_id}_{safe_filename}"
        file_path.write_bytes(file_content)

        job = create_job(job_id=job_id, file_name=file.filename)
        jobs.append(job)
        background_tasks.add_task(ingest_pdf_pipeline, str(file_path), job_id)

    return {"status": "success", "jobs": jobs}


@app.get("/jobs", response_model=List[IngestionJob])
def list_jobs():
    return sorted(list(jobs_db.values()), key=lambda j: j.created_at, reverse=True)


@app.get("/jobs/{job_id}", response_model=IngestionJob)
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "logs": job.logs}


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_filename = job.file_name.replace(" ", "_").replace("/", "")
    file_path = Path(config.upload_dir) / f"{job_id}_{safe_filename}"
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Original PDF no longer exists on server.")
    create_job(job_id, job.file_name)
    background_tasks.add_task(ingest_pdf_pipeline, str(file_path), job_id)
    return {"status": "retrying", "job_id": job_id}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Remove a single non-processing job from history."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot delete a job that is currently processing.")
    del jobs_db[job_id]
    return {"status": "deleted", "job_id": job_id}


@app.delete("/jobs")
def delete_all_jobs():
    """Remove all finished (done/failed) jobs from history."""
    to_delete = [jid for jid, j in jobs_db.items() if j.status in (JobStatus.DONE, JobStatus.FAILED)]
    for jid in to_delete:
        del jobs_db[jid]
    return {"status": "cleared", "deleted_count": len(to_delete)}


@app.delete("/library/{component_id}")
def delete_library_component(component_id: str):
    """Remove all vectors for a component from ChromaDB and delete associated files."""
    try:
        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=component_id,
        )
        deleted_count = store.delete_component(component_id)

        deleted_files = {"knowledge": False, "pdf": False, "docling_json": False}

        knowledge_file = Path(config.knowledge_dir) / f"{component_id}_knowledge.json"
        if knowledge_file.exists():
            knowledge_file.unlink()
            deleted_files["knowledge"] = True

        docling_file = Path(config.docling_dir) / f"{component_id}.json"
        if docling_file.exists():
            docling_file.unlink()
            deleted_files["docling_json"] = True

        pdfs_dir = Path(config.pdfs_dir)
        for pdf_path in pdfs_dir.glob("*.pdf"):
            if pdf_path.stem.replace(" ", "_").replace("/", "").lower() == component_id.lower():
                pdf_path.unlink()
                deleted_files["pdf"] = True
                break

        uploads_dir = Path(config.upload_dir)
        for upload_path in uploads_dir.glob("*"):
            if component_id.lower() in upload_path.stem.lower():
                try:
                    upload_path.unlink()
                except Exception:
                    pass

        to_delete = [jid for jid, j in jobs_db.items() if j.component_id == component_id]
        for jid in to_delete:
            del jobs_db[jid]

        return {
            "status": "deleted",
            "component_id": component_id,
            "vectors_removed": deleted_count,
            "files_removed": deleted_files,
        }
    except Exception as e:
        logger.error("Failed to delete component %s: %s", component_id, e)
        raise HTTPException(status_code=500, detail=str(e))
