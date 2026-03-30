import io
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from dataclasses import asdict
from typing import Any

from backend.models import update_job, append_log, JobStatus, PipelineStage
from backend.config import config
from scripts.parse_pdf import parse_pdf
from ingestion.datasheet_chunker import chunk_document
from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.embeddings.embed_pipeline import EmbeddingPipeline
from rag_pipeline.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# Shared embedder — loaded once on first use
_SHARED_EMBEDDER = None

# Persistent storage dirs (from config)
PDFS_DIR      = Path(config.pdfs_dir)
DOCLING_DIR   = Path(config.docling_dir)
KNOWLEDGE_DIR = Path(config.knowledge_dir)


def get_embedder() -> BGEM3Embedder:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = BGEM3Embedder()
    return _SHARED_EMBEDDER


# ── Log capture helpers ────────────────────────────────────────────────────────

class _StdoutCapture(io.TextIOBase):
    """Tee: forward writes to the real stdout AND append to the job log buffer."""
    def __init__(self, job_id: str, original):
        self.job_id   = job_id
        self.original = original
        self._buf     = ""

    def write(self, s: str) -> int:
        self.original.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                append_log(self.job_id, line)
        return len(s)

    def flush(self):
        self.original.flush()


class _JobLogHandler(logging.Handler):
    """Forward Python logger records to the job's in-memory log buffer."""
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        self.setFormatter(logging.Formatter("%(name)s — %(message)s"))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Skip the root-level backend noise to keep terminal readable
            if "uvicorn" in record.name or "httpx" in record.name:
                return
            append_log(self.job_id, msg)
        except Exception:
            pass


def _log(job_id: str, msg: str) -> None:
    """Convenience: log to Python logger + job buffer directly."""
    logger.info("[%s] %s", job_id, msg)
    append_log(job_id, msg)


def _flatten_metadata(meta: dict) -> dict[str, Any]:
    """Flatten metadata so every value is a ChromaDB-safe scalar."""
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        elif isinstance(v, (dict, list, set, tuple)):
            flat[k] = json.dumps(v, default=str)
        else:
            flat[k] = str(v)
    return flat


# ── Main pipeline ──────────────────────────────────────────────────────────────

def ingest_pdf_pipeline(pdf_path: str, job_id: str) -> None:
    start_time = time.time()
    pdf_file   = Path(pdf_path)
    temp_json  = Path(config.temp_dir) / f"{job_id}.json"

    # Derive clean original filename from the job's stored file_name
    from backend.models import get_job
    job = get_job(job_id)
    
    # If the job is missing, use the filename stem as the ultimate source of truth
    # to avoid 'unknown' collections or metadata.
    original_name = job.file_name if job else pdf_file.name
    clean_stem    = pdf_file.stem.replace(" ", "_").replace("/", "")
    part_number   = clean_stem

    # Install stdout capture + logging handler so all output goes to job terminal
    stdout_cap  = _StdoutCapture(job_id, sys.stdout)
    log_handler = _JobLogHandler(job_id)
    log_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    sys.stdout = stdout_cap

    try:
        update_job(job_id, status=JobStatus.PROCESSING)
        _log(job_id, f"━━━ Pipeline started for '{original_name}' ━━━")

        # ── STAGE 0: PRE-STAGING ──────────────────────────────────────────
        # Ensure target directories exist (Safety guard for Drive)
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        DOCLING_DIR.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        # Save PDF → pdfs/<original_name> BEFORE parsing starts
        dest_pdf = PDFS_DIR / original_name
        shutil.copy2(pdf_file, dest_pdf)
        _log(job_id, f"│  ✔ PDF   staged → pdfs/{original_name}")

        # Use the staged PDF as the source for the rest of the pipeline
        pdf_file = dest_pdf

        # ── STAGE 1: PARSING ──────────────────────────────────────────────
        update_job(job_id, current_stage=PipelineStage.PARSING)
        _log(job_id, "")
        _log(job_id, "┌─ Stage 1/4 · PDF Parsing")
        _log(job_id, f"│  Input  : {pdf_file.name}")
        _log(job_id, f"│  Output : {clean_stem}.json")

        parse_pdf(str(pdf_file), str(temp_json))   # prints to stdout → captured above

        if not temp_json.exists():
            raise FileNotFoundError(f"PDF parse failed — no JSON at {temp_json}")

        # Save Docling JSON → docling_output/<clean_stem>.json
        dest_docling = DOCLING_DIR / f"{clean_stem}.json"
        shutil.copy2(temp_json, dest_docling)
        _log(job_id, f"│  ✔ JSON  saved → docling_output/{clean_stem}.json")

        with temp_json.open("r", encoding="utf-8") as f:
            docling_data = json.load(f)

        n_texts  = len(docling_data.get("texts",    []))
        n_tables = len(docling_data.get("tables",   []))
        n_pics   = len(docling_data.get("pictures", []))
        _log(job_id, f"│  Parsed : {n_texts} text blocks, {n_tables} tables, {n_pics} figures")

        update_job(job_id, stage_completed=PipelineStage.PARSING)
        _log(job_id, "└─ Parsing complete ✔")

        # ── STAGE 2: CHUNKING ─────────────────────────────────────────────
        update_job(job_id, current_stage=PipelineStage.CHUNKING)
        _log(job_id, "")
        _log(job_id, "┌─ Stage 2/4 · Structure-Aware Chunking")
        part_number = clean_stem
        update_job(job_id, component_id=part_number)

        chunks = chunk_document(docling_data, part_number=part_number, pdf_path=pdf_file)

        if not chunks:
            raise ValueError("No chunks generated from document.")

        # Breakdown by type for visibility
        by_type: dict[str, int] = {}
        for c in chunks:
            by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        semantic = sum(v for k, v in by_type.items() if k not in ("table", "figure", "raw_text"))
        _log(job_id, f"│  Total    : {len(chunks)} chunks")
        _log(job_id, f"│  Semantic : {semantic}  |  Tables: {by_type.get('table', 0)}  |  Figures: {by_type.get('figure', 0)}  |  Coverage windows: {by_type.get('raw_text', 0)}")

        # Save knowledge JSON → knowledge_json/<clean_stem>_knowledge.json
        knowledge_data = [asdict(c) for c in chunks]
        dest_knowledge = KNOWLEDGE_DIR / f"{clean_stem}_knowledge.json"
        with dest_knowledge.open("w", encoding="utf-8") as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        _log(job_id, f"│  ✔ Knowledge JSON saved → knowledge_json/{clean_stem}_knowledge.json")

        update_job(job_id, stage_completed=PipelineStage.CHUNKING)
        _log(job_id, "└─ Chunking complete ✔")

        # ── STAGE 3: EMBEDDING ────────────────────────────────────────────
        update_job(job_id, current_stage=PipelineStage.EMBEDDING)
        _log(job_id, "")
        _log(job_id, "┌─ Stage 3/4 · BGE-M3 Embedding")
        _log(job_id, f"│  Embedding model : BAAI/bge-m3")
        _log(job_id, f"│  Encoding {len(chunks)} chunks…")

        embedder = get_embedder()
        pipeline = EmbeddingPipeline(embedder=embedder)
        raw_chunks_dicts = [asdict(c) for c in chunks]
        embedded = pipeline.run(raw_chunks_dicts)

        dim = len(embedded[0]["embedding"]) if embedded else 0
        _log(job_id, f"│  Embedded : {len(embedded)} vectors  (dim={dim})")
        _log(job_id, f"│  Vector dimension : {dim}")

        update_job(job_id, stage_completed=PipelineStage.EMBEDDING)
        _log(job_id, "└─ Embedding complete ✔")

        # ── STAGE 4: STORING ──────────────────────────────────────────────
        update_job(job_id, current_stage=PipelineStage.STORING)
        _log(job_id, "")
        _log(job_id, "┌─ Stage 4/4 · ChromaDB Storage")

        # Flatten metadata before storing to avoid ChromaDB type errors
        for chunk in embedded:
            chunk["metadata"] = _flatten_metadata(chunk.get("metadata", {}))

        store = ChromaStore(
            persist_dir=Path(config.chroma_persist_dir),
            collection_name=clean_stem,
            expected_dim=dim or 1024,   # triggers auto-rebuild if old 384-dim collection exists
        )
        before = store.count()
        store.upsert_chunks(embedded)
        store.persist()
        after = store.count()

        _log(job_id, f"│  Collection : '{config.chroma_collection}'")
        _log(job_id, f"│  Before     : {before} vectors")
        _log(job_id, f"│  Upserted   : {len(embedded)} vectors")
        _log(job_id, f"│  After      : {after} vectors total")

        update_job(job_id, stage_completed=PipelineStage.STORING)
        _log(job_id, "└─ Storing complete ✔")

        # ── DONE ──────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        update_job(
            job_id,
            status=JobStatus.DONE,
            current_stage=None,
            chunks_created=len(embedded),
            processing_time_sec=elapsed,
        )
        _log(job_id, "")
        _log(job_id, f"━━━ ✅ Pipeline complete in {elapsed:.1f}s — {len(embedded)} vectors indexed ━━━")

    except Exception as e:
        logger.error("[%s] ❌ Error: %s", job_id, str(e), exc_info=True)
        _log(job_id, f"")
        _log(job_id, f"━━━ ❌ FAILED: {str(e)} ━━━")
        update_job(job_id, status=JobStatus.FAILED, current_stage=None, error_message=str(e))

    finally:
        # Restore stdout and remove log handler
        sys.stdout = stdout_cap.original
        root_logger.removeHandler(log_handler)

        try:
            if temp_json.exists():
                temp_json.unlink()
        except Exception as cleanup_err:
            logger.error("[%s] Cleanup error: %s", job_id, str(cleanup_err))
