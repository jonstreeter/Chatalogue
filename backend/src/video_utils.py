"""Video-domain helpers shared by main.py and the domain routers:
remote metadata fetch, description archiving, and unique-job queueing."""
import json
import os
import re
import secrets
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from .db.database import (
    FunnyMoment,
    Job,
    TranscriptSegment,
    Video,
    VideoDescriptionRevision,
)
from .deps import get_ingestion_service
from .job_utils import PIPELINE_ACTIVE_STATUSES, PIPELINE_ACTIVE_STATUSES_CORE


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from . import main

    return main


def _extract_publish_datetime(info: dict) -> Optional[datetime]:
    upload_date_str = info.get("upload_date")
    if upload_date_str:
        try:
            return datetime.strptime(str(upload_date_str), "%Y%m%d")
        except ValueError:
            pass

    timestamp_value = info.get("release_timestamp") or info.get("timestamp")
    if timestamp_value:
        try:
            return datetime.fromtimestamp(float(timestamp_value))
        except Exception:
            return None
    return None


def _extract_best_thumbnail_url(info: dict) -> Optional[str]:
    thumbnails = info.get("thumbnails") or []
    for thumb in reversed(thumbnails):
        url = str(thumb.get("url") or "").strip()
        if url:
            return url
    return None


def _make_unique_external_video_id(prefix: str, raw_id: str, session: Session) -> str:
    base = f"{prefix}_{raw_id}" if raw_id else f"{prefix}_{secrets.token_hex(8)}"
    candidate = base
    while session.exec(select(Video).where(Video.youtube_id == candidate)).first():
        candidate = f"{base}_{secrets.token_hex(3)}"
    return candidate


def _normalize_tiktok_video_url(url: str) -> str:
    text = " ".join((url or "").strip().split())
    match = re.search(r"tiktok\.com/@([^/?#]+)/video/(\d+)", text, re.IGNORECASE)
    if match:
        return f"https://www.tiktok.com/@{match.group(1)}/video/{match.group(2)}"
    vm_match = re.search(r"(https?://vm\.tiktok\.com/[A-Za-z0-9]+/?|https?://vt\.tiktok\.com/[A-Za-z0-9]+/?)", text, re.IGNORECASE)
    if vm_match:
        return vm_match.group(1)
    return text


def _fetch_remote_video_info(url: str) -> dict:
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True}
    ydl_opts = _main()._apply_ytdlp_auth_opts(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("Failed to extract remote video metadata.")
    return info

def _archive_video_description_if_needed(
    session: Session,
    video: Video,
    *,
    reason: str,
    ai_model: Optional[str] = None,
    note: Optional[str] = None,
) -> Optional[VideoDescriptionRevision]:
    current_text = (video.description or "").strip()
    if not current_text:
        return None

    latest = session.exec(
        select(VideoDescriptionRevision)
        .where(VideoDescriptionRevision.video_id == video.id)
        .order_by(VideoDescriptionRevision.created_at.desc(), VideoDescriptionRevision.id.desc())
    ).first()
    if latest and (latest.description_text or "").strip() == current_text:
        return None

    existing_count = len(session.exec(
        select(VideoDescriptionRevision.id).where(VideoDescriptionRevision.video_id == video.id)
    ).all())
    source = reason
    if existing_count == 0 and reason == "before_ai_publish":
        source = "ingest_original"

    rev = VideoDescriptionRevision(
        video_id=video.id,
        description_text=current_text,
        source=source,
        ai_model=ai_model,
        note=note,
    )
    session.add(rev)
    session.flush()
    return rev


def _enqueue_unique_job(
    session: Session,
    *,
    video_id: int,
    job_type: str,
    payload: Optional[dict] = None,
) -> Job:
    payload_text = json.dumps(payload or {}, sort_keys=True) if payload is not None else None
    existing = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type == job_type,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES_CORE),
        )
    ).all()
    for j in existing:
        if (j.payload_json or None) == (payload_text or None):
            return j
    job = Job(video_id=video_id, job_type=job_type, status="queued", payload_json=payload_text)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def _queue_diarization_rebuild_job(
    session: Session,
    *,
    video: Video,
    force: bool,
    note: Optional[str],
    queued_from: str,
    optimization_target: str = "diarization_rebuild",
    diarization_sensitivity_override: Optional[str] = None,
    speaker_match_threshold_override: Optional[float] = None,
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(select(Job).where(
        Job.video_id == video.id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before redoing diarization.")

    try:
        audio_path = get_ingestion_service().get_audio_path(video)
        safe_title = get_ingestion_service().sanitize_filename(video.title)
        raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
        if not raw_path.exists():
            raise HTTPException(status_code=400, detail="No raw transcript found. Use full redo instead.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not locate audio/transcript files. Use full redo instead.")

    quality = get_ingestion_service().evaluate_transcript_quality(session, int(video.id), source="diarization_queue_gate", persist_snapshot=False)
    if not force and str(quality.get("recommended_tier") or "") != "diarization_rebuild":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'diarization_rebuild'. Use force=true to queue anyway.",
        )

    segments = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
    ).all()
    funny_rows = session.exec(
        select(FunnyMoment).where(FunnyMoment.video_id == video.id).order_by(FunnyMoment.start_time)
    ).all()
    backup_payload = {
        "video_id": int(video.id),
        "video_status": video.status,
        "video_processed": bool(video.processed),
        "saved_at": datetime.now().isoformat(),
        "segments": [
            {
                "speaker_id": s.speaker_id,
                "matched_profile_id": s.matched_profile_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "words": s.words,
            }
            for s in segments
        ],
        "funny_moments": [
            {
                "start_time": fm.start_time,
                "end_time": fm.end_time,
                "score": fm.score,
                "source": fm.source,
                "snippet": fm.snippet,
                "humor_summary": fm.humor_summary,
                "humor_confidence": fm.humor_confidence,
                "humor_model": fm.humor_model,
                "humor_explained_at": fm.humor_explained_at.isoformat() if fm.humor_explained_at else None,
                "created_at": fm.created_at.isoformat() if fm.created_at else None,
            }
            for fm in funny_rows
        ],
    }
    backup_path = get_ingestion_service()._get_temp_redo_backup_path(int(video.id))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f, ensure_ascii=False)

    d_path = get_ingestion_service()._get_temp_diarization_path(video.id)
    if d_path.exists():
        try:
            os.unlink(d_path)
        except Exception:
            pass

    video.status = "downloaded"
    video.processed = False
    session.add(video)

    payload = {
        "mode": "redo_diarization",
        "optimization_target": str(optimization_target or "diarization_rebuild"),
        "queued_from": str(queued_from or "manual"),
        "redo_diarization_backup_file": str(backup_path),
        "quality_profile_before": quality.get("quality_profile"),
        "recommended_tier_before": quality.get("recommended_tier"),
        "quality_score_before": quality.get("quality_score"),
        "quality_metrics_before": quality.get("metrics"),
        "quality_reasons_before": quality.get("reasons"),
        "note": str(note or "").strip() or None,
        "diarization_sensitivity_override": str(diarization_sensitivity_override or "").strip() or None,
        "speaker_match_threshold_override": float(speaker_match_threshold_override) if speaker_match_threshold_override is not None else None,
        "benchmark_variant": str(optimization_target or "").strip().lower() == "diarization_benchmark",
    }

    last_process_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.job_type == "process",
            Job.status == "completed"
        ).order_by(Job.completed_at.desc())
    ).first()
    if last_process_job and last_process_job.payload_json:
        try:
            old_payload = json.loads(last_process_job.payload_json)
            for k, v in old_payload.items():
                if k.startswith("stage_transcribe") or k.startswith("parakeet_") or k.startswith("transcription_"):
                    payload[k] = v
        except Exception as e:
            print(f"Failed to inherit transcription stats for redo: {e}")

    job = _enqueue_unique_job(session, video_id=int(video.id), job_type="process", payload=payload)
    return job, quality, len(segments), len(funny_rows)


def _queue_full_retranscription_job(
    session: Session,
    *,
    video: Video,
    force: bool,
    note: Optional[str],
    queued_from: str,
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(select(Job).where(
        Job.video_id == video.id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before redoing transcription.")

    quality = get_ingestion_service().evaluate_transcript_quality(session, int(video.id), source="retranscription_queue_gate", persist_snapshot=False)
    if not force and str(quality.get("recommended_tier") or "") != "full_retranscription":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'full_retranscription'. Use force=true to queue anyway.",
        )

    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id)).all()
    funny_rows = session.exec(select(FunnyMoment).where(FunnyMoment.video_id == video.id)).all()
    backup_payload = {
        "video_id": int(video.id),
        "video_status": video.status,
        "video_processed": bool(video.processed),
        "saved_at": datetime.now().isoformat(),
        "segments": [
            {
                "speaker_id": s.speaker_id,
                "matched_profile_id": s.matched_profile_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "words": s.words,
            }
            for s in segments
        ],
        "funny_moments": [
            {
                "start_time": fm.start_time,
                "end_time": fm.end_time,
                "score": fm.score,
                "source": fm.source,
                "snippet": fm.snippet,
                "humor_summary": fm.humor_summary,
                "humor_confidence": fm.humor_confidence,
                "humor_model": fm.humor_model,
                "humor_explained_at": fm.humor_explained_at.isoformat() if fm.humor_explained_at else None,
                "created_at": fm.created_at.isoformat() if fm.created_at else None,
            }
            for fm in funny_rows
        ],
    }
    backup_path = get_ingestion_service()._get_temp_redo_backup_path(int(video.id))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f, ensure_ascii=False)

    get_ingestion_service().purge_artifacts(video.id, delete_raw_transcript=True, delete_audio=False)

    try:
        if get_ingestion_service().get_audio_path(video).exists():
            video.status = "downloaded"
        else:
            video.status = "pending"
    except Exception:
        video.status = "pending"
    video.transcript_source = None
    video.transcript_language = None
    video.transcript_is_placeholder = False
    video.processed = False
    session.add(video)

    payload = {
        "mode": "full_retranscription",
        "optimization_target": "full_retranscription",
        "queued_from": str(queued_from or "manual"),
        "redo_diarization_backup_file": str(backup_path),
        "force_retranscription": True,
        "quality_profile_before": quality.get("quality_profile"),
        "recommended_tier_before": quality.get("recommended_tier"),
        "quality_score_before": quality.get("quality_score"),
        "quality_metrics_before": quality.get("metrics"),
        "quality_reasons_before": quality.get("reasons"),
        "note": str(note or "").strip() or None,
    }
    job = _enqueue_unique_job(session, video_id=int(video.id), job_type="process", payload=payload)
    return job, quality, len(segments), len(funny_rows)
