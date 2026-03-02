from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

# Indian Standard Time  UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(_IST)

class JobStatus(str):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"

class PipelineStage(str):
    PARSING   = "parsing"
    CHUNKING  = "chunking"
    EMBEDDING = "embedding"
    STORING   = "storing"

class IngestionJob(BaseModel):
    job_id: str
    file_name: str
    status: str = JobStatus.PENDING
    current_stage: Optional[str] = None
    stages_done: List[str] = []
    chunks_created: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    component_id: Optional[str] = None
    processing_time_sec: Optional[float] = None
    logs: List[str] = []

# Simple in-memory job store
jobs_db: Dict[str, IngestionJob] = {}

def get_job(job_id: str) -> Optional[IngestionJob]:
    return jobs_db.get(job_id)

def create_job(job_id: str, file_name: str) -> IngestionJob:
    now = _now_ist()
    job = IngestionJob(
        job_id=job_id,
        file_name=file_name,
        status=JobStatus.PENDING,
        current_stage=None,
        stages_done=[],
        logs=[],
        created_at=now,
        updated_at=now,
    )
    jobs_db[job_id] = job
    return job

def append_log(job_id: str, message: str) -> None:
    """Append an IST-timestamped log line to the job's log buffer."""
    job = jobs_db.get(job_id)
    if not job:
        return
    ts = _now_ist().strftime("%H:%M:%S")
    job.logs = job.logs + [f"[{ts}] {message}"]
    job.updated_at = _now_ist()

def update_job(
    job_id: str,
    status: Optional[str] = None,
    current_stage: Optional[str] = None,
    stage_completed: Optional[str] = None,
    chunks_created: Optional[int] = None,
    error_message: Optional[str] = None,
    component_id: Optional[str] = None,
    processing_time_sec: Optional[float] = None,
) -> IngestionJob:
    job = jobs_db.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    if status is not None:
        job.status = status
    if current_stage is not None:
        job.current_stage = current_stage
    if stage_completed is not None and stage_completed not in job.stages_done:
        job.stages_done = job.stages_done + [stage_completed]
        job.current_stage = None          # clear active stage once it completes
    if chunks_created is not None:
        job.chunks_created = chunks_created
    if error_message is not None:
        job.error_message = error_message
    if component_id is not None:
        job.component_id = component_id
    if processing_time_sec is not None:
        job.processing_time_sec = processing_time_sec

    job.updated_at = _now_ist()
    return job
