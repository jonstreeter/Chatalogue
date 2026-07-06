"""Job-state helpers shared by main.py and the jobs router."""
from datetime import datetime

from sqlmodel import Session, select

from .db.database import Job, TranscriptOptimizationCampaignItem, Video

# Canonical list of job statuses considered "active" in the pipeline.
PIPELINE_ACTIVE_STATUSES = ["queued", "running", "downloading", "transcribing", "diarizing", "waiting_diarize"]
# Subset without waiting_diarize, for contexts where that status isn't relevant.
PIPELINE_ACTIVE_STATUSES_CORE = ["queued", "running", "downloading", "transcribing", "diarizing"]


def _job_queue_name(job_type: str) -> str:
    jt = (job_type or "").strip().lower()
    if jt in {"process", "diarize", "voicefixer_cleanup", "conversation_reconstruct", "transcript_repair"}:
        return "pipeline"
    if jt in {"funny_detect", "funny_explain"}:
        return "funny"
    if jt in {"youtube_metadata", "episode_clone"}:
        return "youtube"
    if jt in {"clip_export_mp4", "clip_export_captions"}:
        return "clip"
    return "other"


def _sync_auxiliary_video_job_state(session: Session, job: Job, state: str) -> None:
    video_id = int(getattr(job, "video_id", 0) or 0)
    if video_id <= 0:
        return
    video = session.get(Video, video_id)
    if not video:
        return

    jt = str(getattr(job, "job_type", "") or "").strip().lower()
    normalized = str(state or "").strip().lower()

    if jt == "voicefixer_cleanup":
        if normalized == "paused":
            video.voicefixer_status = "paused"
            video.voicefixer_error = None
        elif normalized == "queued":
            video.voicefixer_status = "queued"
            video.voicefixer_error = None
        elif normalized == "cleared":
            if str(getattr(video, "voicefixer_cleaned_path", "") or "").strip():
                apply_scope = str(getattr(video, "voicefixer_apply_scope", "none") or "none").strip().lower()
                video.voicefixer_status = "ready" if apply_scope != "none" else "disabled"
            else:
                video.voicefixer_status = None
            video.voicefixer_error = None
        session.add(video)
        return

    if jt == "conversation_reconstruct":
        if normalized == "paused":
            video.reconstruction_status = "paused"
            video.reconstruction_error = None
        elif normalized == "queued":
            video.reconstruction_status = "queued"
            video.reconstruction_error = None
        elif normalized == "cleared":
            if str(getattr(video, "reconstruction_audio_path", "") or "").strip():
                video.reconstruction_status = "ready"
            else:
                video.reconstruction_status = None
            video.reconstruction_error = None
        session.add(video)
        return


def _detach_job_from_transcript_campaign_items(session: Session, job: Job, *, deleted_from_queue: bool = False) -> None:
    job_id = int(getattr(job, "id", 0) or 0)
    if job_id <= 0:
        return
    items = session.exec(
        select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.job_id == job_id)
    ).all()
    if not items:
        return
    for item in items:
        item.job_id = None
        if deleted_from_queue:
            status_value = str(getattr(item, "status", "") or "").strip().lower()
            if status_value in {"pending", "queued", "paused", "running"}:
                item.status = "cleared"
        item.updated_at = datetime.now()
        session.add(item)


def _mark_job_cancelled(session: Session, job: Job, *, detail: str = "Cancelled by user.") -> None:
    if not job:
        return
    if str(getattr(job, "status", "") or "").strip().lower() == "cancelled":
        return
    job.status = "cancelled"
    job.status_detail = None
    job.error = detail
    job.completed_at = datetime.now()
    session.add(job)
