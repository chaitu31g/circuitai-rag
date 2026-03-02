import os
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from urllib.request import Request as URLRequest, urlopen

from backend.config import config
from backend.models import create_job, get_job, jobs_db, IngestionJob, JobStatus
from rag_pipeline.services.ingest_service import ingest_pdf_pipeline, get_embedder
from rag_pipeline.vectordb.chroma_store import ChromaStore
from rag_pipeline.rag.rag_pipeline import RAGPipeline, RAGConfig, OllamaClient
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
def warmup_ollama():
    """Load qwen2.5:3b into RAM at server start so first query is instant."""
    import threading, http.client, json as _j
    def _warm():
        try:
            body = _j.dumps({
                "model": config.ollama_model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1},
            }).encode()
            conn = http.client.HTTPConnection("localhost", 11434, timeout=None)
            conn.request("POST", "/api/generate", body,
                         {"Content-Type": "application/json", "Content-Length": str(len(body))})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            logger.info("Ollama model warm-up complete ✓")
        except Exception as e:
            logger.warning("Ollama warm-up skipped: %s", e)
    threading.Thread(target=_warm, daemon=True, name="ollama-warmup").start()

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
    """RAG query — retrieve → assemble context → Ollama → return answer + sources."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
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
        pipeline  = RAGPipeline(
            retriever=retriever,
            llm_client=OllamaClient(timeout_s=req.timeout_s),
            config=RAGConfig(
                top_k=req.top_k,
                max_context_chars=4000,       # allow more context from multiple datasheets
                default_trimmed_chunks=6,     # allow up to 6 chunks for multi-datasheet coverage
                request_timeout_s=req.timeout_s,
            ),
        )
        filters = {"part_number": req.component_filter} if req.component_filter else None
        result  = pipeline.answer(
            query=req.query,
            filters=filters,
            mode="rag_answer",
            timeout_s=req.timeout_s,
            max_context_chunks=6,             # allow chunks from multiple datasheets
        )
        sources = [
            {
                "id":        d.get("id", ""),
                "text":      d.get("text", "")[:400],
                "score":     round(d.get("score", 0), 3),
                "component": (d.get("metadata") or {}).get("part_number", "unknown"),
                "section":   (d.get("metadata") or {}).get("section_name", ""),
                "type":      (d.get("metadata") or {}).get("chunk_type", ""),
            }
            for d in result.get("used_docs", [])
        ]
        return {
            "answer":    result["answer"],
            "sources":   sources,
            "llm_error": result.get("llm_error"),
        }
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streaming RAG — retrieval is instant, then Ollama tokens stream via SSE.

    SSE event types:
        {"type": "sources", "sources": [...]}   # sent first, before any tokens
        {"type": "token",   "token":   "..."}   # one per Ollama token
        {"type": "error",   "message": "..."}   # on failure
        {"type": "done"}                         # stream finished
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── 1. Retrieval + context assembly (milliseconds) ────────────────────
    try:
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
        
        # Apply component filter if provided
        filters = {"part_number": req.component_filter} if req.component_filter else None
        
        retrieved = retriever.retrieve(query=req.query, top_k=req.top_k, filters=filters)
        
        pipeline  = RAGPipeline(
            retriever=retriever,
            config=RAGConfig(top_k=req.top_k, max_context_chars=4000),
        )
        assembled = pipeline.assemble_context(retrieved, max_context_chunks=6)
        used_docs = assembled["used_docs"]
        context   = assembled["context"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    chunk_ids   = [d.get("id", "") for d in used_docs if d.get("id")]
    prompt_obj  = pipeline.prompt_builder.build(
        query=req.query, context=context, context_chunk_ids=chunk_ids
    )
    prompt_text = prompt_obj.full_prompt

    # ── Helper: format retrieved docs as a structured answer (no LLM needed) ──
    def _direct_answer() -> str:
        """Extract and format structured specs grouped by datasheet."""
        import re as _re

        if not used_docs:
            return "*No relevant datasheet content found for this query.*"

        # Group docs by component (datasheet)
        by_component = {}
        for doc in used_docs:
            meta = doc.get("metadata") or {}
            comp = meta.get("part_number", "unknown")
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(doc)

        lines = [
            f"**📋 Datasheet Results for:** *{req.query}*",
            f"*Found relevant data in {len(by_component)} datasheet(s)*\n",
        ]

        for comp_idx, (comp, docs) in enumerate(by_component.items()):
            comp_display = comp.replace('_', ' ').title()
            lines.append(f"### 📄 Datasheet: {comp_display}")
            lines.append("---")

            for doc in docs:
                meta    = doc.get("metadata") or {}
                section = meta.get("section_name", "").replace("_", " ").title()
                ctype   = meta.get("chunk_type", "")
                text    = doc.get("text", "").strip()
                score   = doc.get("score", 0)

                # Strip the "section_name of component:" prefix if present
                text = _re.sub(r'^[\w\s]+ of [\w\d_\-]+:\s*', '', text)

                if ctype == "table":
                    # Extract key-value pairs from table data
                    rows = [r.strip() for r in text.split('\n') if r.strip()]
                    spec_pairs = []
                    for row in rows[:25]:
                        # Try to extract "Parameter: Value" or "Parameter | Value" patterns
                        parts = _re.split(r'[:|\t]+', row, maxsplit=1)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            param = parts[0].strip()
                            val = parts[1].strip()
                            spec_pairs.append(f"**{param}:** {val}")
                        else:
                            spec_pairs.append(row)
                    lines.append(f"*Section: {section}*")
                    lines.extend(spec_pairs)
                else:
                    # Format text section — extract specs like "Max Voltage: 50V"
                    spec_pattern = _re.findall(
                        r'((?:Max|Min|Typ|Nom|Rating|Range|Limit|Operating|Storage|Power|Voltage|Current|'
                        r'Resistance|Temperature|Frequency|Impedance|Capacitance|Tolerance|TCR|'
                        r'Rated|Working|Breakdown|Surge|Overload|Derating|Dissipation|Weight|Size)'
                        r'[^:.,;\n]{0,40}?)\s*[:=]\s*([^\n,;]{1,60})',
                        text, _re.I
                    )
                    if spec_pattern:
                        lines.append(f"*Section: {section}*")
                        for param, val in spec_pattern:
                            lines.append(f"**{param.strip()}:** {val.strip()}")
                    else:
                        # Fall back to showing the text with section label
                        lines.append(f"*Section: {section}*")
                        # Break into readable sentences
                        sentences = _re.split(r'(?<=[.!?])\s+', text)
                        query_words = set(_re.findall(r'\w+', req.query.lower())) - {
                            'what', 'is', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'and', 'or', 'are', 'how', 'does'
                        }
                        for sent in sentences[:10]:
                            words = set(_re.findall(r'\w+', sent.lower()))
                            if query_words & words:
                                lines.append(f"• **{sent.strip()}**")
                            else:
                                lines.append(f"• {sent.strip()}")

                lines.append(f"*Relevance: {score:.3f}*\n")

            lines.append("")  # spacing between datasheets

        lines.append("---")
        lines.append("*⚡ Direct context mode — LLM was unavailable. Showing raw datasheet data.*")

        return '\n'.join(lines)

    # ── Check if Ollama is responsive (2-second timeout) ─────────────────
    def _ollama_alive() -> bool:
        import http.client as _hc
        try:
            c = _hc.HTTPConnection("localhost", 11434, timeout=2)
            c.request("GET", "/api/tags")
            r = c.getresponse()
            r.read()
            c.close()
            return r.status == 200
        except Exception:
            return False

    # ── 2. SSE event stream ───────────────────────────────────────────────
    def event_stream():
        import json as _j
        import http.client

        # Always send sources first
        yield f"data: {_j.dumps({'type': 'sources', 'sources': sources})}\n\n"

        if not context:
            msg = "No relevant context found in the knowledge base for this query."
            yield f"data: {_j.dumps({'type': 'token', 'token': msg})}\n\n"
            yield f"data: {_j.dumps({'type': 'done'})}\n\n"
            return

        # ── Mode A: Ollama available → stream tokens ──────────────────────
        # Give Ollama up to 60 seconds to start generating an answer before fallback
        if _ollama_alive():
            body = _j.dumps({
                "model":   config.ollama_model,
                "prompt":  prompt_text,
                "stream":  True,
                "options": {"temperature": 0.2, "num_predict": 400},
            }).encode("utf-8")

            try:
                # 8-second wait before failing over to Mode B. Your PC is struggling.
                # If it takes more than 8s to generate the FIRST token after the alive check, 
                # we don't want to make you wait a whole minute holding the stream open.
                conn = http.client.HTTPConnection("localhost", 11434, timeout=8)
                conn.request(
                    "POST", "/api/generate", body,
                    {"Content-Type": "application/json", "Content-Length": str(len(body))}
                )
                resp = conn.getresponse()
                tokens_yielded = 0
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = _j.loads(line)
                        tok   = chunk.get("response", "")
                        if tok:
                            tokens_yielded += 1
                            yield f"data: {_j.dumps({'type': 'token', 'token': tok})}\n\n"
                        if chunk.get("done"):
                            yield f"data: {_j.dumps({'type': 'done'})}\n\n"
                            conn.close()
                            return
                    except _j.JSONDecodeError:
                        continue
                conn.close()
            except Exception as e:
                logger.warning("Ollama stream error, falling back: %s", e)
                if tokens_yielded > 0:
                    yield f"data: {_j.dumps({'type': 'error', 'message': f' Generation interrupted: {str(e)}'})}\n\n"
                    yield f"data: {_j.dumps({'type': 'done'})}\n\n"
                    return
                # Fall through to direct-context mode below if nothing was sent

        # ── Mode B: Direct context answer (no LLM, always works) ─────────
        yield f"data: {_j.dumps({'type': 'mode', 'mode': 'direct'})}\n\n"
        answer = _direct_answer()
        # Stream it word-by-word so UI typewriter effect still works
        words = answer.split(' ')
        chunk_size = 4  # words per SSE event
        for i in range(0, len(words), chunk_size):
            piece = ' '.join(words[i:i+chunk_size]) + ' '
            yield f"data: {_j.dumps({'type': 'token', 'token': piece})}\n\n"
        yield f"data: {_j.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



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
