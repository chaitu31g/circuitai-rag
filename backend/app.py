import os
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import config
from backend.models import create_job, get_job, jobs_db, IngestionJob, JobStatus
from backend.llm.hf_llm import load_model_once, generate_response, build_prompt
from rag_pipeline.services.ingest_service import ingest_pdf_pipeline, get_embedder
from rag_pipeline.vectordb.chroma_store import ChromaStore
from rag_pipeline.rag.rag_pipeline import RAGPipeline, RAGConfig
from rag_pipeline.rag.retriever import Retriever, RetrieverConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CircuitAI RAG Ingestion API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def preload_hf_model():
    """Pre-load the HuggingFace model into GPU memory at startup.
    
    This runs in a background thread so it doesn't block FastAPI from
    accepting requests while the (large) model weights are downloaded/loaded.
    """
    import threading
    def _load():
        try:
            load_model_once(config.hf_model)
        except Exception as exc:
            logger.warning("HF model pre-load failed (will retry on first query): %s", exc)
    threading.Thread(target=_load, daemon=True, name="hf-model-loader").start()

@app.get("/")
def root():
    return {"message": "CircuitAI RAG backend running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/library")
def get_library():
    """Return all unique components stored in ChromaDB with chunk stats + file metadata."""
    try:
        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
        components = store.get_library()

        # Enrich each component with file metadata for sorting
        knowledge_dir = Path(config.knowledge_dir)
        pdfs_dir = Path(config.pdfs_dir)
        from datetime import datetime, timezone

        for comp in components:
            cid = comp["component_id"]
            # Try to find the knowledge JSON file for this component
            knowledge_file = knowledge_dir / f"{cid}_knowledge.json"
            if knowledge_file.exists():
                mtime = knowledge_file.stat().st_mtime
                comp["ingested_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            else:
                comp["ingested_at"] = None

            # Try to find the source PDF — could have spaces or underscores
            pdf_file = None
            for candidate in pdfs_dir.glob("*"):
                if candidate.stem.replace(" ", "_").lower() == cid.lower():
                    pdf_file = candidate
                    break
            if pdf_file and pdf_file.exists():
                comp["file_size_kb"] = round(pdf_file.stat().st_size / 1024, 1)
            else:
                comp["file_size_kb"] = None

        return {
            "total_components": len(components),
            "total_vectors": store.count(),
            "components": components,
        }
    except Exception as e:
        logger.error("Library fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    query: str
    component_filter: str | None = None   # optional: restrict to one component
    top_k: int = 10                        # retrieve from ALL component files
    timeout_s: int = 360                   # default 6 min; plenty for CPU qwen2.5:3b


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG query — retrieve → embed → ChromaDB → HuggingFace LLM → return answer + sources."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # ── 1. Retrieval ──────────────────────────────────────────────────────
        store     = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
        embedder  = get_embedder()
        retriever = Retriever(
            vector_store=store,
            embedder=embedder,
            config=RetrieverConfig(top_k=req.top_k),
        )
        filters = {"part_number": req.component_filter} if req.component_filter else None
        retrieved = retriever.retrieve(query=req.query, top_k=req.top_k, filters=filters)

        # ── 2. Context assembly ───────────────────────────────────────────────
        pipeline  = RAGPipeline(
            retriever=retriever,
            config=RAGConfig(
                top_k=req.top_k,
                max_context_chars=4000,
                default_trimmed_chunks=6,
            ),
        )
        assembled = pipeline.assemble_context(retrieved, max_context_chunks=6)
        used_docs = assembled["used_docs"]
        context   = assembled["context"]

        # ── 3. Build sources list ─────────────────────────────────────────────
        sources = [
            {
                "id":        d.get("id", ""),
                "text":      d.get("text", "")[:400],
                "score":     round(d.get("score", 0), 3),
                "component": (d.get("metadata") or {}).get("part_number", "unknown"),
                "section":   (d.get("metadata") or {}).get("section_name", ""),
                "type":      (d.get("metadata") or {}).get("chunk_type", ""),
            }
            for d in used_docs
        ]

        # ── 4. LLM generation (HuggingFace Mistral-7B) ───────────────────────
        if not context:
            answer = "No relevant context found in the knowledge base for this query."
        else:
            prompt = build_prompt(context=context, query=req.query)
            answer = generate_response(prompt)

        return {
            "answer":  answer,
            "sources": sources,
        }
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """RAG query via the /chat/stream endpoint.

    Streaming is not used with HuggingFace Transformers generation, so this
    endpoint now returns a standard JSON response identical to /chat.
    The endpoint is kept for frontend compatibility.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # ── 1. Retrieval ──────────────────────────────────────────────────────
        store     = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
        embedder  = get_embedder()
        retriever = Retriever(
            vector_store=store,
            embedder=embedder,
            config=RetrieverConfig(top_k=req.top_k),
        )
        filters   = {"part_number": req.component_filter} if req.component_filter else None
        retrieved = retriever.retrieve(query=req.query, top_k=req.top_k, filters=filters)

        # ── 2. Context assembly ───────────────────────────────────────────────
        pipeline  = RAGPipeline(
            retriever=retriever,
            config=RAGConfig(top_k=req.top_k, max_context_chars=4000),
        )
        assembled = pipeline.assemble_context(retrieved, max_context_chunks=6)
        used_docs = assembled["used_docs"]
        context   = assembled["context"]

        # ── 3. Sources ────────────────────────────────────────────────────────
        sources = [
            {
                "id":        d.get("id", ""),
                "text":      d.get("text", "")[:400],
                "score":     round(d.get("score", 0), 3),
                "component": (d.get("metadata") or {}).get("part_number", "unknown"),
                "section":   (d.get("metadata") or {}).get("section_name", ""),
                "type":      (d.get("metadata") or {}).get("chunk_type", ""),
            }
            for d in used_docs
        ]

        # ── 4. LLM generation (HuggingFace Mistral-7B) ───────────────────────
        if not context:
            answer = "No relevant context found in the knowledge base for this query."
        else:
            prompt = build_prompt(context=context, query=req.query)
            answer = generate_response(prompt)

        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error("Chat stream error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    jobs = []
    
    for file in files:
        job_id = str(uuid.uuid4())
        
        if not file.filename.lower().endswith(".pdf"):
            job = create_job(job_id=job_id, file_name=file.filename)
            job = update_job(job_id, status="failed", error_message=f"File {file.filename} is not a PDF")
            jobs.append(job)
            continue
            
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > config.max_file_size_mb:
            job = create_job(job_id=job_id, file_name=file.filename)
            job = update_job(job_id, status="failed", error_message=f"File {file.filename} exceeds {config.max_file_size_mb}MB limit")
            jobs.append(job)
            continue
            
        # Duplicate detection (optional but requested)
        existing_job = next((j for j in jobs_db.values() if j.file_name == file.filename and j.status == "done"), None)
        if existing_job:
             logger.info(f"File {file.filename} was already ingested successfully as job {existing_job.job_id}")

        safe_filename = file.filename.replace(" ", "_").replace("/", "")
        file_path = Path(config.upload_dir) / f"{job_id}_{safe_filename}"
        
        # Save file chunks to disk
        with file_path.open("wb") as f:
            f.write(file_content)
            
        # Create job in memory
        job = create_job(job_id=job_id, file_name=file.filename)
        jobs.append(job)
        
        # Trigger background task
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
    
    # Find the file on disk. We saved it as {job_id}_{safe_filename}
    # But wait, if we want to retry, we need to know the original file path.
    # In my upload, I used config.upload_dir / f"{job_id}_{safe_filename}"
    
    safe_filename = job.file_name.replace(" ", "_").replace("/", "")
    file_path = Path(config.upload_dir) / f"{job_id}_{safe_filename}"
    
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Original PDF file no longer exists on server.")
    
    # Update status and restart
    create_job(job_id, job.file_name) # Resets fields
    background_tasks.add_task(ingest_pdf_pipeline, str(file_path), job_id)
    
    return {"status": "retrying", "job_id": job_id}

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Remove a single job from history. Only non-processing jobs can be deleted."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot delete a job that is currently processing.")
    del jobs_db[job_id]
    return {"status": "deleted", "job_id": job_id}

@app.delete("/jobs")
def delete_all_jobs():
    """Remove only finished (done/failed) jobs from history. Pending and processing jobs are preserved."""
    to_delete = [jid for jid, j in jobs_db.items() if j.status in (JobStatus.DONE, JobStatus.FAILED)]
    for jid in to_delete:
        del jobs_db[jid]
    return {"status": "cleared", "deleted_count": len(to_delete)}

@app.delete("/library/{component_id}")
def delete_library_component(component_id: str):
    """Remove all vectors for a component from ChromaDB and delete its knowledge JSON."""
    try:
        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=config.chroma_collection,
        )
        
        # 1. Delete from ChromaDB
        deleted_count = store.delete_component(component_id)
        
        # 2. Delete associated file artifacts
        deleted_files = {
            "knowledge": False,
            "pdf": False,
            "docling_json": False
        }

        # Knowledge JSON
        knowledge_file = Path(config.knowledge_dir) / f"{component_id}_knowledge.json"
        if knowledge_file.exists():
            knowledge_file.unlink()
            deleted_files["knowledge"] = True

        # Docling JSON
        docling_file = Path(config.docling_dir) / f"{component_id}.json"
        if docling_file.exists():
            docling_file.unlink()
            deleted_files["docling_json"] = True
            
        # PDF File (may have spaces or variations, need to find it)
        # Search by stem (without extension) matching the component_id
        pdfs_dir = Path(config.pdfs_dir)
        for pdf_path in pdfs_dir.glob("*.pdf"):
            if pdf_path.stem.replace(" ", "_").replace("/", "").lower() == component_id.lower():
                pdf_path.unlink()
                deleted_files["pdf"] = True
                break
        for pdf_path in pdfs_dir.glob("*.PDF"):
            if pdf_path.stem.replace(" ", "_").replace("/", "").lower() == component_id.lower():
                pdf_path.unlink()
                deleted_files["pdf"] = True
                break

        # Also delete the original file from the uploads directory if it's there
        uploads_dir = Path(config.upload_dir)
        for upload_path in uploads_dir.glob("*"):
            if component_id.lower() in upload_path.stem.lower():
                try:
                    upload_path.unlink()
                except Exception:
                    pass

        # 3. Optional: Clear pipeline history for this component if it was recently added
        to_delete = [jid for jid, j in jobs_db.items() if j.component_id == component_id]
        for jid in to_delete:
            del jobs_db[jid]

        return {
            "status": "deleted",
            "component_id": component_id,
            "vectors_removed": deleted_count,
            "files_removed": deleted_files
        }
    except Exception as e:
        logger.error("Failed to delete component %s: %s", component_id, e)
        raise HTTPException(status_code=500, detail=str(e))
