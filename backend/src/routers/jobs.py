"""Job queue and pipeline focus endpoints."""
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import Job, TranscriptSegment, Video
from ..deps import get_ingestion_service, get_session
from ..job_utils import (
    PIPELINE_ACTIVE_STATUSES,
    PIPELINE_ACTIVE_STATUSES_CORE,
    _detach_job_from_transcript_campaign_items,
    _job_queue_name,
    _mark_job_cancelled,
    _sync_auxiliary_video_job_state,
)
from ..schemas import JobRead, PipelineFocusRead, PipelineFocusUpdate

router = APIRouter()

@router.post("/jobs/pause-all")
def pause_all_jobs(session: Session = Depends(get_session)):
    """Pause all queued jobs. Running jobs will complete."""
    statement = select(Job).where(Job.status == "queued")
    jobs = session.exec(statement).all()
    
    count = 0
    for job in jobs:
        job.status = "paused"
        _sync_auxiliary_video_job_state(session, job, "paused")
        session.add(job)
        count += 1
        
    session.commit()
    return {"paused_count": count}

@router.post("/jobs/resume-all")
def resume_all_jobs(session: Session = Depends(get_session)):
    """Resume all paused jobs. Worker will pick them up automatically."""
    statement = select(Job).where(Job.status == "paused")
    jobs = session.exec(statement).all()

    count = 0
    for job in jobs:
        job.status = "queued"
        _sync_auxiliary_video_job_state(session, job, "queued")
        session.add(job)
        count += 1

    session.commit()
    return {"resumed_count": count}


@router.get("/jobs", response_model=List[JobRead])
def read_jobs(
    status: Optional[str] = None,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    job_type: Optional[str] = None,
    limit: int = 500,
    sort_by: Optional[str] = None,
    sort_dir: Optional[str] = "desc",
    session: Session = Depends(get_session)
):
    from sqlalchemy import case, asc, desc
    from sqlalchemy.orm import selectinload
    
    query = select(Job).options(selectinload(Job.video))
    active_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    
    if status:
        statuses = [s.strip() for s in str(status).split(",") if s.strip()]
        if len(statuses) == 1:
            query = query.where(Job.status == statuses[0])
        elif len(statuses) > 1:
            query = query.where(Job.status.in_(statuses))
    else:
        query = query.where(Job.status != "waiting_diarize")
        
    if job_type:
        job_types = [jt.strip() for jt in str(job_type).split(",") if jt.strip()]
        if len(job_types) == 1:
            query = query.where(Job.job_type == job_types[0])
        elif len(job_types) > 1:
            query = query.where(Job.job_type.in_(job_types))
            
    if channel_id:
        query = query.join(Video, Job.video_id == Video.id).where(Video.channel_id == channel_id)
    if video_id:
        query = query.where(Job.video_id == video_id)

    # Determine sorting behavior
    s_dir = asc if sort_dir == "asc" else desc

    if sort_by == "duration":
        # we need the video join to sort by duration
        if not channel_id:
            query = query.join(Video, Job.video_id == Video.id)
        # Pin active jobs first, then sort by duration
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Video.duration)
            )
        else:
            query = query.order_by(s_dir(Video.duration))
            
    elif sort_by == "name":
        # we need the video join to sort by title
        if not channel_id:
            query = query.join(Video, Job.video_id == Video.id)
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Video.title)
            )
        else:
            query = query.order_by(s_dir(Video.title))
            
    else:
        # Default sort by created_at (Order Added)
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Job.created_at)
            )
        else:
            query = query.order_by(s_dir(Job.created_at))

    safe_limit = max(1, min(limit, 2000))
    return session.exec(query.limit(safe_limit)).all()


@router.get("/jobs/queues/summary")
def get_job_queues_summary(session: Session = Depends(get_session)):
    jobs = session.exec(select(Job.id, Job.job_type, Job.status)).all()
    summary = {
        "pipeline": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "funny": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "youtube": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "clip": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "other": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
    }
    for row in jobs:
        # sqlite row can come back as tuple in this select mode
        job_type = row[1] if isinstance(row, tuple) else getattr(row, "job_type", "")
        status = row[2] if isinstance(row, tuple) else getattr(row, "status", "")
        if status == "waiting_diarize":
            continue
        q = _job_queue_name(job_type)
        bucket = summary[q]
        bucket["total"] += 1
        if status in bucket:
            bucket[status] += 1
        elif status in {"downloading", "transcribing", "diarizing"}:
            bucket["running"] += 1
    return summary

@router.get("/jobs/status")
def get_queue_status(session: Session = Depends(get_session)):
    """Get summary of job queue status"""
    running_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    running = session.exec(select(Job).where(Job.status.in_(running_like_statuses))).all()
    queued = session.exec(select(Job).where(Job.status == "queued")).all()
    paused = session.exec(select(Job).where(Job.status == "paused")).all()
    
    return {
        "running": len(running),
        "queued": len(queued),
        "paused": len(paused),
        "total_active": len(running) + len(queued) + len(paused)
    }


def _compute_pipeline_focus_counts(session: Session) -> dict:
    """Shared helper for pipeline focus count queries."""
    transcribe_active_statuses = ["running", "downloading", "transcribing"]
    diarize_active_statuses = ["running", "diarizing"]
    transcribe_active = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "process",
                Job.status.in_(transcribe_active_statuses),
            )
        ).one() or 0
    )
    transcribe_queued = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "process",
                Job.status == "queued",
            )
        ).one() or 0
    )
    diarize_active = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "diarize",
                Job.status.in_(diarize_active_statuses),
            )
        ).one() or 0
    )
    diarize_queued = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "diarize",
                Job.status == "queued",
            )
        ).one() or 0
    )
    return {
        "transcribe_active": transcribe_active,
        "transcribe_queued": transcribe_queued,
        "diarize_active": diarize_active,
        "diarize_queued": diarize_queued,
        "auto_diarize_ready": transcribe_active == 0 and transcribe_queued == 0 and (diarize_active > 0 or diarize_queued > 0),
        "diarize_auto_start_threshold": int(os.getenv("DIARIZE_AUTO_START_THRESHOLD", "0")),
    }


@router.get("/jobs/pipeline/focus", response_model=PipelineFocusRead)
def get_pipeline_focus(session: Session = Depends(get_session)):
    counts = _compute_pipeline_focus_counts(session)
    return {
        "mode": get_ingestion_service().get_pipeline_focus_mode(),
        "execution_mode": get_ingestion_service().get_pipeline_execution_mode(),
        **counts,
    }


@router.post("/jobs/pipeline/focus", response_model=PipelineFocusRead)
def set_pipeline_focus(payload: PipelineFocusUpdate, session: Session = Depends(get_session)):
    mode = get_ingestion_service().set_pipeline_focus_mode(payload.mode)
    paused_active_count = 0
    if mode == "diarize" and payload.pause_active_transcription:
        active_process_jobs = session.exec(
            select(Job).where(
                Job.job_type == "process",
                Job.status.in_(["running", "downloading", "transcribing"]),
            )
        ).all()
        for job in active_process_jobs:
            job.status = "paused"
            session.add(job)
            paused_active_count += 1
        if paused_active_count:
            session.commit()

    counts = _compute_pipeline_focus_counts(session)
    return {
        "mode": mode,
        "execution_mode": get_ingestion_service().get_pipeline_execution_mode(),
        **counts,
        "active_transcription_paused": paused_active_count,
    }


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in PIPELINE_ACTIVE_STATUSES_CORE:
        raise HTTPException(status_code=400, detail="Cannot pause job in current status")
    
    job.status = "paused"
    _sync_auxiliary_video_job_state(session, job, "paused")
    session.add(job)
    session.commit()
    return {"status": "paused"}

@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")

    job.status = "queued"
    _sync_auxiliary_video_job_state(session, job, "queued")
    session.add(job)
    session.commit()

    # Worker will pick this up automatically
    return {"status": "resumed"}

@router.post("/jobs/{job_id}/move-to-top")
def move_job_to_top(job_id: int, session: Session = Depends(get_session)):
    """Move a queued/paused job to the front of the pending queue."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ["queued", "paused"]:
        raise HTTPException(status_code=400, detail="Only queued or paused jobs can be reordered")

    pending = session.exec(
        select(Job).where(
            Job.status.in_(["queued", "paused"]),
            Job.job_type == job.job_type,
            Job.id != job_id,
        ).order_by(Job.created_at.asc())
    ).all()

    oldest = pending[0].created_at if pending else datetime.now()
    # Queue worker consumes queued jobs in ascending created_at order.
    job.created_at = oldest - timedelta(microseconds=1)
    session.add(job)
    session.commit()
    session.refresh(job)
    return {"status": "moved_to_top", "job_id": job.id, "created_at": job.created_at.isoformat()}

@router.delete("/jobs/queue")
def clear_queue(session: Session = Depends(get_session)):
    """Delete all queued and paused jobs quickly and reliably."""
    from sqlalchemy.exc import OperationalError

    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            clearable_statuses = ["queued", "paused", "waiting_diarize"]
            jobs = session.exec(
                select(Job).where(Job.status.in_(clearable_statuses))
            ).all()
            deleted_count = len(jobs)
            if deleted_count <= 0:
                return {"deleted": 0}

            for job in jobs:
                _detach_job_from_transcript_campaign_items(session, job, deleted_from_queue=True)
                _sync_auxiliary_video_job_state(session, job, "cleared")
                session.delete(job)
            session.commit()
            return {"deleted": deleted_count}
        except OperationalError as e:
            session.rollback()
            is_locked = "database is locked" in str(e).lower()
            if (not is_locked) or attempt >= (max_attempts - 1):
                raise HTTPException(status_code=503, detail="Failed to clear queue: database busy.")
            time.sleep(min(0.15 * (attempt + 1), 1.2))

    raise HTTPException(status_code=503, detail="Failed to clear queue: database busy.")

@router.delete("/jobs/history")
def clear_history(session: Session = Depends(get_session)):
    """Delete completed, failed, cancelled, and waiting_diarize jobs.

    Skips waiting_diarize jobs that still have an active child diarize job
    (queued/running/diarizing) to avoid orphaning in-flight work.
    """
    active_child_statuses = {"queued", "running", "diarizing"}
    active_child_parent_ids: set[int] = set()
    active_children = session.exec(
        select(Job).where(Job.job_type == "diarize", Job.status.in_(active_child_statuses))
    ).all()
    for child in active_children:
        try:
            payload = json.loads(child.payload_json) if child.payload_json else {}
            pid = payload.get("parent_job_id")
            if pid:
                active_child_parent_ids.add(int(pid))
        except Exception:
            pass

    jobs = session.exec(
        select(Job).where(Job.status.in_(["completed", "failed", "cancelled", "waiting_diarize"]))
    ).all()
    count = 0
    for job in jobs:
        if job.status == "waiting_diarize" and job.id in active_child_parent_ids:
            continue
        _detach_job_from_transcript_campaign_items(session, job, deleted_from_queue=False)
        session.delete(job)
        count += 1
    session.commit()
    return {"deleted": count}

@router.post("/jobs/{job_id}/resubmit")
def resubmit_job(job_id: int, session: Session = Depends(get_session)):
    """Resubmit a completed or failed job — creates a new job for the same video"""
    old_job = session.get(Job, job_id)
    if not old_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if old_job.status not in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Can only resubmit completed, failed, or cancelled jobs")
    
    video_id = old_job.video_id
    
    # Check no active job already exists for this video
    active = session.exec(select(Job).where(
        Job.video_id == video_id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active:
        raise HTTPException(status_code=400, detail=f"Video already has an active job ({active.status})")
    
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Clear existing transcript data so re-processing starts fresh
    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    for s in segments:
        session.delete(s)
    
    # Purge temp checkpoint files
    get_ingestion_service().purge_artifacts(video_id)
    
    # Reset video status
    try:
        audio_path = get_ingestion_service().get_audio_path(video)
        video.status = "downloaded" if audio_path.exists() else "pending"
    except:
        video.status = "pending"
    video.processed = False
    session.add(video)
    
    # Create fresh job
    new_job = Job(video_id=video_id, job_type="process", status="queued")
    session.add(new_job)
    session.commit()
    session.refresh(new_job)
    
    return {"status": "resubmitted", "new_job_id": new_job.id}

@router.delete("/jobs/{job_id}")
def cancel_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    jt = str(job.job_type or "").strip().lower()
    _sync_auxiliary_video_job_state(session, job, "cleared")

    # Reset video status only for the main transcript pipeline jobs.
    if job.video and jt in {"process", "diarize"}:
        video = job.video
        # Check if audio exists
        try:
            audio_path = get_ingestion_service().get_audio_path(video)
            if audio_path.exists():
                video.status = "downloaded"
            else:
                video.status = "pending"
        except Exception as e:
            print(f"Error checking audio path during cancel: {e}")
            video.status = "pending"
            
        video.processed = False # Ensure processed flag is reset
        session.add(video)

    payload = json.loads(job.payload_json) if job.payload_json else {}
    if job.job_type == "diarize":
        parent_job_id = int(payload.get("parent_job_id") or 0)
        if parent_job_id:
            parent_job = session.get(Job, parent_job_id)
            if parent_job:
                _mark_job_cancelled(session, parent_job)
    elif job.job_type == "process" and job.status == "waiting_diarize":
        child_job_id = int(payload.get("diarize_job_id") or 0)
        if child_job_id:
            child_job = session.get(Job, child_job_id)
            if child_job:
                _mark_job_cancelled(session, child_job)

    _mark_job_cancelled(session, job)
    session.commit()
    return {"status": "cancelled", "video_status": job.video.status if job.video else "unknown"}
