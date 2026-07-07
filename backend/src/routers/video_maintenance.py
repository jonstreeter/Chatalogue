"""Video maintenance endpoints: purge, redo diarization/transcription, consolidate transcripts, mute."""
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db.database import Channel, FunnyMoment, Job, TranscriptSegment, Video
from ..deps import get_ingestion_service, get_session
from ..job_utils import PIPELINE_ACTIVE_STATUSES
from ..schemas import TranscriptRepairResultRead
from ..video_utils import (
    _enqueue_unique_job,
    _queue_diarization_rebuild_job,
    _queue_full_retranscription_job,
)
from ..services import speaker_queries as spk_q

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.post("/videos/{video_id}/purge")
def purge_video(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")

    # verify no active job
    active_job = session.exec(select(Job).where(Job.video_id == video.id, Job.status.in_(PIPELINE_ACTIVE_STATUSES))).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Cannot purge video with active job {active_job.id} ({active_job.status})")

    # 1. Delete segments
    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    for s in segments:
        session.delete(s)
    funny_moments = session.exec(select(FunnyMoment).where(FunnyMoment.video_id == video_id)).all()
    for fm in funny_moments:
        session.delete(fm)
    
    # 2. Purge temp files
    get_ingestion_service().purge_artifacts(video.id)

    # 3. Reset video status
    try:
        if get_ingestion_service().get_audio_path(video).exists():
            video.status = "downloaded"
        else:
            video.status = "pending"
    except:
        video.status = "pending"
    video.transcript_source = None
    video.transcript_language = None
    video.transcript_is_placeholder = False
    video.processed = False
    session.add(video)
    session.commit()
    spk_q._invalidate_speaker_query_caches()
    return {"status": "purged", "deleted_segments": len(segments), "deleted_funny_moments": len(funny_moments)}

@router.post("/videos/{video_id}/redo-diarization")
def redo_diarization(video_id: int, session: Session = Depends(get_session)):
    """Delete segments and re-run diarization only, reusing the existing raw transcript."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    new_job, _, segment_count, funny_count = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=True,
        note="Manual redo-diarization request",
        queued_from="redo_diarization",
    )
    spk_q._invalidate_speaker_query_caches()

    return {
        "status": "diarization_requeued",
        "deleted_segments": segment_count,
        "deleted_funny_moments": funny_count,
        "job_id": new_job.id,
    }


@router.post("/channels/{channel_id}/redo-diarization")
def redo_channel_diarization(
    channel_id: int,
    dry_run: bool = True,
    processed_only: bool = True,
    include_muted: bool = False,
    limit: int = 0,
    session: Session = Depends(get_session),
):
    """Bulk re-queue diarization across channel videos using existing raw transcripts."""
    from sqlalchemy import delete

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    query = select(Video).where(Video.channel_id == channel_id).order_by(Video.id.desc())
    videos = session.exec(query).all()
    if limit and limit > 0:
        videos = videos[:limit]

    active_statuses = ["queued", "running", "downloading", "transcribing", "diarizing", "paused"]
    result = {
        "channel_id": int(channel_id),
        "channel_name": channel.name,
        "dry_run": bool(dry_run),
        "counts": {
            "scanned": 0,
            "eligible": 0,
            "queued": 0,
            "skipped_active": 0,
            "skipped_no_raw_transcript": 0,
            "skipped_muted": 0,
            "skipped_unprocessed": 0,
            "errors": 0,
            "deleted_segments": 0,
            "deleted_funny_moments": 0,
        },
        "job_ids": [],
        "sample_skips": [],
    }

    for video in videos:
        result["counts"]["scanned"] += 1

        if not include_muted and bool(video.muted):
            result["counts"]["skipped_muted"] += 1
            continue

        if processed_only and not bool(video.processed):
            result["counts"]["skipped_unprocessed"] += 1
            continue

        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == video.id,
                Job.status.in_(active_statuses),
            )
        ).first()
        if active_job:
            result["counts"]["skipped_active"] += 1
            continue

        has_raw_transcript = False
        try:
            audio_path = get_ingestion_service().get_audio_path(video)
            safe_title = get_ingestion_service().sanitize_filename(video.title)
            raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
            has_raw_transcript = raw_path.exists()
        except Exception:
            has_raw_transcript = False

        if not has_raw_transcript:
            result["counts"]["skipped_no_raw_transcript"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": "no_raw_transcript",
                })
            continue

        result["counts"]["eligible"] += 1
        if dry_run:
            continue

        try:
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
            ).all()
            funny_rows = session.exec(
                select(FunnyMoment).where(FunnyMoment.video_id == video.id).order_by(FunnyMoment.start_time)
            ).all()
            seg_count = len(segments)
            funny_count = len(funny_rows)

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
            backup_path = get_ingestion_service()._get_temp_redo_backup_path(video.id)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_payload, f, ensure_ascii=False)

            if seg_count > 0:
                session.exec(delete(TranscriptSegment).where(TranscriptSegment.video_id == video.id))
            if funny_count > 0:
                session.exec(delete(FunnyMoment).where(FunnyMoment.video_id == video.id))

            d_path = get_ingestion_service()._get_temp_diarization_path(video.id)
            if d_path.exists():
                try:
                    d_path.unlink()
                except Exception:
                    pass

            video.status = "downloaded"
            video.processed = False
            session.add(video)

            payload = {
                "mode": "redo_diarization",
                "redo_diarization_backup_file": str(backup_path),
            }

            # Inherit transcription timestamps from the last completed pipeline job
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
                    for k, pval in old_payload.items():
                        if k.startswith("stage_transcribe") or k.startswith("parakeet_") or k.startswith("transcription_"):
                            payload[k] = pval
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to inherit transcription stats for channel redo: {e}")

            job = _enqueue_unique_job(
                session,
                video_id=video.id,
                job_type="process",
                payload=payload,
            )

            result["counts"]["queued"] += 1
            result["counts"]["deleted_segments"] += int(seg_count)
            result["counts"]["deleted_funny_moments"] += int(funny_count)
            result["job_ids"].append(int(job.id))
        except Exception as e:
            session.rollback()
            result["counts"]["errors"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": f"error: {e}",
                })

    if not dry_run and (result["counts"]["queued"] > 0 or result["counts"]["deleted_segments"] > 0):
        spk_q._invalidate_speaker_query_caches()

    return result


@router.post("/videos/{video_id}/redo-transcription")
def redo_transcription(video_id: int, session: Session = Depends(get_session)):
    """Purge transcript artifacts and re-run full transcription pipeline (transcribe + diarize)."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    new_job, _, segment_count, funny_count = _queue_full_retranscription_job(
        session,
        video=video,
        force=True,
        note="Manual redo-transcription request",
        queued_from="redo_transcription",
    )
    spk_q._invalidate_speaker_query_caches()
    return {
        "status": "transcription_requeued",
        "deleted_segments": segment_count,
        "deleted_funny_moments": funny_count,
        "job_id": new_job.id,
    }


@router.post("/videos/{video_id}/consolidate-transcript", response_model=TranscriptRepairResultRead)
def consolidate_video_transcript(video_id: int, session: Session = Depends(get_session)):
    """Post-process an existing transcript to smooth tiny speaker islands and merge same-speaker neighbors."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")

    try:
        result = get_ingestion_service().repair_existing_transcript(
            session,
            video_id,
            save_files=True,
            persist_run=True,
            persist_snapshot=True,
            source="manual_sync",
            note="Manual consolidate-transcript request",
            trigger_semantic_index=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to consolidate transcript: {e}")

    spk_q._invalidate_speaker_query_caches()
    return TranscriptRepairResultRead(**result)


@router.post("/channels/{channel_id}/consolidate-transcripts")
def consolidate_channel_transcripts(
    channel_id: int,
    processed_only: bool = True,
    include_muted: bool = False,
    limit: int = 0,
    session: Session = Depends(get_session),
):
    """Bulk post-process existing transcripts for a channel without re-transcribing or re-diarizing."""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    query = select(Video).where(Video.channel_id == channel_id).order_by(Video.id.desc())
    videos = session.exec(query).all()
    if limit and limit > 0:
        videos = videos[:limit]

    result = {
        "channel_id": int(channel_id),
        "channel_name": channel.name,
        "counts": {
            "scanned": 0,
            "eligible": 0,
            "changed": 0,
            "merged_segments": 0,
            "reassigned_islands": 0,
            "skipped_active": 0,
            "skipped_muted": 0,
            "skipped_unprocessed": 0,
            "skipped_no_segments": 0,
            "errors": 0,
        },
        "videos": [],
        "sample_skips": [],
    }

    for video in videos:
        result["counts"]["scanned"] += 1

        if not include_muted and bool(video.muted):
            result["counts"]["skipped_muted"] += 1
            continue

        if processed_only and not bool(video.processed):
            result["counts"]["skipped_unprocessed"] += 1
            continue

        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == video.id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            result["counts"]["skipped_active"] += 1
            continue

        seg_exists = session.exec(
            select(TranscriptSegment.id).where(TranscriptSegment.video_id == video.id).limit(1)
        ).first()
        if not seg_exists:
            result["counts"]["skipped_no_segments"] += 1
            continue

        result["counts"]["eligible"] += 1
        try:
            video_result = get_ingestion_service().repair_existing_transcript(
                session,
                int(video.id),
                save_files=True,
                persist_run=True,
                persist_snapshot=True,
                source="channel_bulk_sync",
                note=f"Bulk consolidate-transcripts for channel {channel_id}",
                trigger_semantic_index=True,
            )
            result["videos"].append(video_result)
            if video_result["changed"]:
                result["counts"]["changed"] += 1
            result["counts"]["merged_segments"] += int(video_result["merged_count"])
            result["counts"]["reassigned_islands"] += int(video_result["reassigned_islands"])
        except Exception as e:
            session.rollback()
            result["counts"]["errors"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": f"error: {e}",
                })

    if result["counts"]["eligible"] > 0:
        spk_q._invalidate_speaker_query_caches()
    return result

# --- Video Controls ---

@router.patch("/videos/{video_id}/mute", response_model=Video)
def toggle_video_mute(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")
    
    video.muted = not video.muted
    session.add(video)
    session.commit()
    session.refresh(video)
    return video

# --- Jobs ---
