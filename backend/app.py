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
from backend.llm.hf_llm import load_model_once, generate_response, stream_response, build_prompt
from rag_pipeline.services.ingest_service import ingest_pdf_pipeline, get_embedder
from rag_pipeline.vectordb.chroma_store import ChromaStore
from rag_pipeline.rag.rag_pipeline import RAGPipeline, RAGConfig
from rag_pipeline.rag.retriever import Retriever, RetrieverConfig

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


def _build_retrieval(query: str, top_k: int, component_filter: str | None):
    """Shared retrieval + context assembly for both endpoints."""
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
    filters = {"part_number": component_filter} if component_filter else None
    retrieved = retriever.retrieve(query=query, top_k=top_k, filters=filters)

    pipeline = RAGPipeline(
        retriever=retriever,
        config=RAGConfig(top_k=top_k, max_context_chars=4000, default_trimmed_chunks=6),
    )
    assembled = pipeline.assemble_context(retrieved, max_context_chunks=6)

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
    return assembled["context"], sources


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
    top_k: int = 10
    max_new_tokens: int = 512
    temperature: float = 0.2


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG query — returns the full answer in one JSON response (non-streaming)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        context, sources = _build_retrieval(req.query, req.top_k, req.component_filter)

        if not context:
            answer = "No relevant context found in the knowledge base for this query."
        else:
            prompt = build_prompt(context=context, query=req.query)
            answer = generate_response(
                prompt,
                model_id=config.hf_model,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            )

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

            # ── 2. Retrieve & assemble context ───────────────────────────────────
            context, sources = _build_retrieval(req.query, req.top_k, req.component_filter)

            # ── 3. Send sources immediately so the UI shows them before generation
            yield _sse({"type": "sources", "sources": sources})

            # ── 4. No context → short-circuit ────────────────────────────────────
            if not context:
                yield _sse({"type": "token", "token": "No relevant context found in the knowledge base for this query."})
                yield _sse({"type": "done"})
                return

            # ── 5. Build prompt ──────────────────────────────────────────────────
            prompt = build_prompt(context=context, query=req.query)

            # ── 6. Stream tokens — send a keep-alive every 30 tokens so Cloudflare
            #       doesn't drop the connection during slow GPU/CPU inference ──────
            token_count = 0
            for token in stream_response(
                prompt,
                model_id=config.hf_model,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            ):
                yield _sse({"type": "token", "token": token})
                token_count += 1
                if token_count % 30 == 0:
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



# ── Library endpoints ─────────────────────────────────────────────────────────

@app.get("/library")
def get_library():
    """Return all unique components stored in ChromaDB with chunk stats + file metadata."""
    try:
        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
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
            collection_name=config.chroma_collection,
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
