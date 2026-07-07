"""Core video endpoints: listing, processing, funny moments, YouTube AI metadata."""
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    FunnyMoment,
    FunnyMomentRead,
    Job,
    Speaker,
    TranscriptSegment,
    TranscriptSegmentRead,
    Video,
    VideoDescriptionRevision,
    VideoDescriptionRevisionRead,
)
from ..deps import get_ingestion_service, get_session
from ..job_utils import PIPELINE_ACTIVE_STATUSES
from ..schemas import ChannelBatchPublishRequest, VideoListItemRead
from ..video_utils import _archive_video_description_if_needed, _enqueue_unique_job

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/videos", response_model=List[Video])
def read_videos(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    from sqlalchemy import case
    query = select(Video)
    if channel_id:
        query = query.where(Video.channel_id == channel_id)
    # NULLs first (newest videos often lack dates from flat extraction), then by date desc, then by id desc
    query = query.order_by(
        case((Video.published_at.is_(None), 0), else_=1),
        Video.published_at.desc(),
        Video.id.desc(),
    )
    return session.exec(query).all()

@router.get("/videos/list", response_model=List[VideoListItemRead])
def read_videos_list(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    from sqlalchemy import case
    pipeline_job_types = ["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct", "transcript_repair"]

    query = select(
        Video.id,
        Video.youtube_id,
        Video.channel_id,
        Video.title,
        Video.media_source_type,
        Video.source_url,
        Video.media_kind,
        Video.manual_media_path,
        Video.published_at,
        Video.description,
        Video.thumbnail_url,
        Video.duration,
        Video.view_count,
        Video.processed,
        Video.muted,
        Video.access_restricted,
        Video.access_restriction_reason,
        Video.status,
        Video.transcript_source,
        Video.transcript_language,
        Video.transcript_is_placeholder,
    )
    if channel_id:
        query = query.where(Video.channel_id == channel_id)
    query = query.order_by(
        case((Video.published_at.is_(None), 0), else_=1),
        Video.published_at.desc(),
        Video.id.desc(),
    )

    rows = session.exec(query).all()
    video_ids = [int(row[0]) for row in rows if row[0] is not None]
    latest_pipeline_job_by_video: dict[int, tuple[str, str]] = {}
    if video_ids:
        job_rows = session.exec(
            select(Job.video_id, Job.job_type, Job.status, Job.created_at, Job.id)
            .where(
                Job.video_id.in_(video_ids),
                Job.job_type.in_(pipeline_job_types),
            )
            .order_by(Job.video_id, Job.created_at.desc(), Job.id.desc())
        ).all()
        for job_row in job_rows:
            video_id = int(job_row[0])
            if video_id in latest_pipeline_job_by_video:
                continue
            latest_pipeline_job_by_video[video_id] = (
                str(job_row[2] or "").strip().lower() or "queued",
                str(job_row[1] or "").strip().lower() or "process",
            )

    return [
        VideoListItemRead(
            id=row[0],
            youtube_id=row[1],
            channel_id=row[2],
            title=row[3],
            media_source_type=row[4],
            source_url=row[5],
            media_kind=row[6],
            manual_media_path=row[7],
            published_at=row[8],
            description=row[9],
            thumbnail_url=row[10],
            duration=row[11],
            view_count=row[12],
            processed=bool(row[13]),
            muted=bool(row[14]),
            access_restricted=bool(row[15]),
            access_restriction_reason=row[16],
            status=row[17],
            transcript_source=row[18],
            transcript_language=row[19],
            transcript_is_placeholder=bool(row[20]),
            last_pipeline_job_status=latest_pipeline_job_by_video.get(int(row[0]), (None, None))[0] if row[0] is not None else None,
            last_pipeline_job_type=latest_pipeline_job_by_video.get(int(row[0]), (None, None))[1] if row[0] is not None else None,
        )
        for row in rows
    ]

@router.get("/videos/{video_id}", response_model=Video)
def read_video(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.post("/videos/{video_id}/process")
def process_video(video_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.access_restricted:
        raise HTTPException(
            status_code=409,
            detail=video.access_restriction_reason or "This video is not accessible with the current YouTube session.",
        )
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before processing.")
    job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if not job:
        job = Job(video_id=video_id, job_type="process", status="queued")
        session.add(job)
        session.commit()
    
    # Worker will pick this up automatically
    # background_tasks.add_task(get_ingestion_service().process_queue)
    return {"status": "processing_queued"}

@router.post("/channels/{channel_id}/process-all")
def process_all_videos(channel_id: int, session: Session = Depends(get_session)):
    """Queue all unmuted, unprocessed videos in a channel for processing"""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    queued = get_ingestion_service()._queue_channel_unprocessed_videos(session, channel_id)
    session.commit()
    return {"queued": int(queued)}

@router.get("/videos/{video_id}/segments", response_model=List[TranscriptSegmentRead])
def read_segments(video_id: int, session: Session = Depends(get_session)):
    # Join with Speaker to get name
    results = session.exec(
        select(TranscriptSegment, Speaker.name)
        .join(Speaker, isouter=True)
        .where(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
    ).all()
    
    segments = []
    for seg, speaker_name in results:
        # Use model_dump(exclude={"speaker"}) to avoid conflict with relationship if it exists in dump
        seg_dict = seg.model_dump(exclude={"speaker"})
        read_seg = TranscriptSegmentRead(**seg_dict, speaker=speaker_name)
        segments.append(read_seg)
    return segments

@router.get("/videos/{video_id}/funny-moments", response_model=List[FunnyMomentRead])
def read_funny_moments(video_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(FunnyMoment)
        .where(FunnyMoment.video_id == video_id)
        .order_by(FunnyMoment.start_time)
    ).all()

@router.post("/videos/{video_id}/funny-moments/detect", response_model=List[FunnyMomentRead])
def detect_funny_moments(video_id: int, force: bool = False):
    try:
        return get_ingestion_service().detect_funny_moments(video_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment detection failed: {e}")


@router.post("/videos/{video_id}/funny-moments/detect/queue")
def queue_detect_funny_moments(video_id: int, force: bool = True, session: Session = Depends(get_session)):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    job = _enqueue_unique_job(session, video_id=video_id, job_type="funny_detect", payload={"force": bool(force)})
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@router.get("/videos/{video_id}/funny-moments/progress")
def get_funny_moments_progress(video_id: int):
    try:
        if get_ingestion_service() is None:
            return {"video_id": video_id, "status": "idle"}
        return get_ingestion_service().get_funny_task_progress(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment progress unavailable: {e}")

@router.post("/videos/{video_id}/funny-moments/explain", response_model=List[FunnyMomentRead])
def explain_funny_moments(video_id: int, force: bool = False, limit: Optional[int] = None):
    try:
        return get_ingestion_service().explain_funny_moments(video_id, force=force, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Ollama disabled/unreachable/config issue
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment explanation failed: {e}")


@router.post("/videos/{video_id}/funny-moments/explain/queue")
def queue_explain_funny_moments(
    video_id: int,
    force: bool = True,
    limit: Optional[int] = None,
    session: Session = Depends(get_session),
):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    payload = {"force": bool(force)}
    if limit is not None:
        payload["limit"] = int(limit)
    job = _enqueue_unique_job(session, video_id=video_id, job_type="funny_explain", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@router.post("/videos/{video_id}/youtube-ai/generate", response_model=Video)
def generate_youtube_ai_metadata(video_id: int, force: bool = False):
    try:
        return get_ingestion_service().generate_youtube_metadata_suggestion(video_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube metadata generation failed: {e}")


@router.post("/videos/{video_id}/youtube-ai/generate/queue")
def queue_generate_youtube_ai_metadata(video_id: int, force: bool = True, session: Session = Depends(get_session)):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    job = _enqueue_unique_job(session, video_id=video_id, job_type="youtube_metadata", payload={"force": bool(force)})
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@router.get("/videos/{video_id}/description-history", response_model=List[VideoDescriptionRevisionRead])
def get_video_description_history(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return session.exec(
        select(VideoDescriptionRevision)
        .where(VideoDescriptionRevision.video_id == video_id)
        .order_by(VideoDescriptionRevision.created_at.desc(), VideoDescriptionRevision.id.desc())
    ).all()

def _publish_video_ai_description_internal(
    session: Session,
    video: Video,
    *,
    push_to_youtube: bool,
) -> dict:
    draft = (video.youtube_ai_description_text or "").strip()
    if not draft:
        raise ValueError("No generated YouTube description draft found. Generate it first.")

    current_desc = (video.description or "").strip()
    remote_pushed = False
    try:
        if current_desc != draft:
            _archive_video_description_if_needed(
                session,
                video,
                reason="before_ai_publish",
                ai_model=video.youtube_ai_model,
                note="Archived before applying AI-generated YouTube description draft",
            )

        if push_to_youtube:
            _main()._youtube_update_video_description_remote(video.youtube_id, draft)
            remote_pushed = True

        if current_desc != draft:
            video.description = draft
            session.add(video)

        session.commit()
        session.refresh(video)
        return {
            "video": video,
            "updated_local": current_desc != draft,
            "pushed_to_youtube": remote_pushed,
            "skipped_local_same_text": current_desc == draft,
        }
    except Exception:
        session.rollback()
        raise


@router.post("/videos/{video_id}/youtube-ai/publish-description", response_model=Video)
def publish_youtube_ai_description(video_id: int, push_to_youtube: Optional[bool] = None, session: Session = Depends(get_session)):
    """Archive current description and apply the AI-generated YouTube description draft.

    If YouTube publishing is enabled/configured, this also pushes the description to the
    actual YouTube video via `videos.update` while preserving snippet title/category/tags.
    """
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    is_youtube_video = str(getattr(video, "media_source_type", "") or "youtube").lower() == "youtube" and bool(video.youtube_id)
    should_push = (_main()._youtube_get_cfg()["push_enabled"] and is_youtube_video) if push_to_youtube is None else bool(push_to_youtube and is_youtube_video)
    try:
        result = _publish_video_ai_description_internal(session, video, push_to_youtube=should_push)
        return result["video"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Publish failed: {e}")

@router.post("/videos/{video_id}/description-history/{revision_id}/restore", response_model=Video)
def restore_video_description_from_history(video_id: int, revision_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    revision = session.get(VideoDescriptionRevision, revision_id)
    if not revision or revision.video_id != video_id:
        raise HTTPException(status_code=404, detail="Description revision not found for this video")

    target_text = (revision.description_text or "").strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="Selected revision has an empty description")

    current_desc = (video.description or "").strip()
    if current_desc == target_text:
        return video

    _archive_video_description_if_needed(
        session,
        video,
        reason="before_restore",
        ai_model=video.youtube_ai_model,
        note=f"Archived before restoring description revision #{revision_id}",
    )
    video.description = revision.description_text
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@router.post("/channels/{channel_id}/youtube-ai/publish-descriptions")
def batch_publish_channel_youtube_descriptions(channel_id: int, req: ChannelBatchPublishRequest, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    should_push = _main()._youtube_get_cfg()["push_enabled"] if req.push_to_youtube is None else bool(req.push_to_youtube)
    limit = None if req.limit is None else max(1, min(int(req.limit), 1000))
    ownership_check = None
    if should_push:
        ownership_check = _main()._youtube_channel_ownership_check_for_app_channel(channel)

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id, Video.processed == True, Video.muted == False)
        .order_by(Video.published_at.asc(), Video.id.asc())
    ).all()
    if limit is not None:
        videos = videos[:limit]

    items = []
    eligible_video_ids: list[int] = []
    for v in videos:
        draft = (v.youtube_ai_description_text or "").strip()
        current_desc = (v.description or "").strip()
        has_draft = bool(draft)
        matches = bool(has_draft and current_desc == draft)
        reason = None
        if not has_draft:
            reason = "missing_ai_draft"
        elif matches and not should_push:
            reason = "already_matches_draft"
        item = {
            "video_id": v.id,
            "youtube_id": v.youtube_id,
            "title": v.title,
            "processed": bool(v.processed),
            "has_ai_draft": has_draft,
            "current_matches_draft": matches,
            "eligible": reason is None,
            "reason": reason,
            "youtube_ai_model": v.youtube_ai_model,
            "published_at": v.published_at.isoformat() if v.published_at else None,
        }
        if reason is None:
            eligible_video_ids.append(v.id)
        items.append(item)

    if req.dry_run:
        return {
            "status": "dry_run",
            "channel_id": channel_id,
            "channel_name": channel.name,
            "push_to_youtube": should_push,
            "ownership_check": ownership_check,
            "confirm_required": True,
            "counts": {
                "scanned": len(videos),
                "eligible": len(eligible_video_ids),
                "missing_ai_draft": sum(1 for i in items if i["reason"] == "missing_ai_draft"),
                "already_matches_draft": sum(1 for i in items if i["reason"] == "already_matches_draft"),
            },
            "estimated_youtube_quota_units": (len(eligible_video_ids) * 51) if should_push else 0,
            "items": items,
        }

    if not req.confirm:
        raise HTTPException(status_code=400, detail="Batch publish requires confirm=true when dry_run=false")
    if should_push:
        ownership_check = ownership_check or _main()._youtube_channel_ownership_check_for_app_channel(channel)
        if ownership_check.get("status") != "owned":
            raise HTTPException(
                status_code=400,
                detail=f"Batch YouTube publish blocked: ownership check status is '{ownership_check.get('status')}'. "
                       "Connect the matching YouTube channel or run with push_to_youtube=false."
            )

    results = []
    success_count = 0
    error_count = 0
    for item in items:
        if not item["eligible"]:
            item["status"] = "skipped"
            results.append(item)
            continue
        try:
            video = session.get(Video, item["video_id"])
            if not video:
                raise RuntimeError("Video not found during batch publish")
            publish_result = _publish_video_ai_description_internal(session, video, push_to_youtube=should_push)
            item["status"] = "published"
            item["updated_local"] = bool(publish_result["updated_local"])
            item["pushed_to_youtube"] = bool(publish_result["pushed_to_youtube"])
            success_count += 1
        except Exception as e:
            session.rollback()
            item["status"] = "error"
            item["error"] = str(e)
            error_count += 1
        results.append(item)

    return {
        "status": "completed",
        "channel_id": channel_id,
        "channel_name": channel.name,
        "push_to_youtube": should_push,
        "ownership_check": ownership_check,
        "counts": {
            "scanned": len(videos),
            "eligible": len(eligible_video_ids),
            "published": success_count,
            "errors": error_count,
            "skipped": len(videos) - len(eligible_video_ids),
        },
        "estimated_youtube_quota_units": (len(eligible_video_ids) * 51) if should_push else 0,
        "items": results,
    }
