"""Channel CRUD, ingest, refresh, upload, export/import endpoints."""
import base64
import logging
import secrets
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    Clip,
    ClipExportArtifact,
    FunnyMoment,
    Job,
    Speaker,
    SpeakerEmbedding,
    TranscriptChunkEmbedding,
    TranscriptSegment,
    TranscriptSegmentRevision,
    Video,
    VideoDescriptionRevision,
    engine,
)
from ..deps import get_ingestion_service, get_session
from ..video_utils import (
    _extract_best_thumbnail_url,
    _fetch_remote_video_info,
    _make_unique_external_video_id,
    _normalize_tiktok_video_url,
)
from ..job_utils import PIPELINE_ACTIVE_STATUSES
from ..paths import MANUAL_MEDIA_DIR
from ..schemas import ChannelOverviewRead

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.post("/channels", response_model=Channel)
def create_channel(url: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    try:
        channel = get_ingestion_service().add_channel(url)
        managed_channel = session.get(Channel, channel.id)
        if managed_channel:
            managed_channel.status = "refreshing"
            managed_channel.sync_status_detail = "Starting channel scan..."
            managed_channel.sync_progress = 1
            managed_channel.sync_total_items = 0
            managed_channel.sync_completed_items = 0
            session.add(managed_channel)
            session.commit()
            session.refresh(managed_channel)
            channel = managed_channel
        # Auto-refresh on add
        background_tasks.add_task(get_ingestion_service().refresh_channel, channel.id)
        return channel
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/channels/manual", response_model=Channel)
def create_manual_channel(name: str, session: Session = Depends(get_session)):
    try:
        channel = get_ingestion_service().create_manual_channel(name)
        managed_channel = session.get(Channel, channel.id)
        return managed_channel or channel
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/channels/tiktok", response_model=Channel)
def create_tiktok_channel(
    name: Optional[str] = None,
    url: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session),
):
    try:
        channel = get_ingestion_service().create_tiktok_channel(name, url)
        managed_channel = session.get(Channel, channel.id)
        channel_obj = managed_channel or channel
        if "tiktok.com" in str(channel_obj.url or "").lower():
            channel_obj.status = "refreshing"
            channel_obj.sync_status_detail = "Starting TikTok profile scan..."
            channel_obj.sync_progress = 1
            channel_obj.sync_total_items = 0
            channel_obj.sync_completed_items = 0
            session.add(channel_obj)
            session.commit()
            session.refresh(channel_obj)
            if background_tasks is not None:
                background_tasks.add_task(get_ingestion_service().refresh_channel, channel_obj.id)
        return channel_obj
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/channels", response_model=List[Channel])
def read_channels(session: Session = Depends(get_session)):
    return session.exec(select(Channel)).all()

@router.get("/channels/overview", response_model=List[ChannelOverviewRead])
def read_channels_overview(session: Session = Depends(get_session)):
    from sqlalchemy import and_, case

    video_stats = (
        select(
            Video.channel_id.label("channel_id"),
            func.count(Video.id).label("video_count"),
            func.sum(case((Video.processed.is_(True), 1), else_=0)).label("processed_count"),
            func.sum(case((and_(Video.processed.is_(False), Video.muted.is_(False), Video.access_restricted.is_(False)), 1), else_=0)).label("pending_video_count"),
            func.sum(func.coalesce(Video.duration, 0)).label("total_duration_seconds"),
        )
        .group_by(Video.channel_id)
        .subquery()
    )
    speaker_stats = (
        select(
            Speaker.channel_id.label("channel_id"),
            func.count(Speaker.id).label("speaker_count"),
        )
        .group_by(Speaker.channel_id)
        .subquery()
    )
    active_job_stats = (
        select(
            Video.channel_id.label("channel_id"),
            func.count(Job.id).label("active_job_count"),
        )
        .select_from(Job)
        .join(Video, Video.id == Job.video_id)
        .where(
            Job.job_type.in_(["process", "diarize"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
        .group_by(Video.channel_id)
        .subquery()
    )

    rows = session.exec(
        select(
            Channel,
            func.coalesce(video_stats.c.video_count, 0).label("video_count"),
            func.coalesce(video_stats.c.processed_count, 0).label("processed_count"),
            func.coalesce(video_stats.c.pending_video_count, 0).label("pending_video_count"),
            func.coalesce(video_stats.c.total_duration_seconds, 0).label("total_duration_seconds"),
            func.coalesce(speaker_stats.c.speaker_count, 0).label("speaker_count"),
            func.coalesce(active_job_stats.c.active_job_count, 0).label("active_job_count"),
        )
        .outerjoin(video_stats, video_stats.c.channel_id == Channel.id)
        .outerjoin(speaker_stats, speaker_stats.c.channel_id == Channel.id)
        .outerjoin(active_job_stats, active_job_stats.c.channel_id == Channel.id)
        .order_by(Channel.id.asc())
    ).all()

    return [
        ChannelOverviewRead(
            id=channel.id,
            url=channel.url,
            name=channel.name,
            source_type=channel.source_type,
            icon_url=channel.icon_url,
            header_image_url=channel.header_image_url,
            last_updated=channel.last_updated,
            status=channel.status,
            actively_monitored=bool(getattr(channel, "actively_monitored", False)),
            sync_status_detail=channel.sync_status_detail,
            sync_progress=int(channel.sync_progress or 0),
            sync_total_items=int(channel.sync_total_items or 0),
            sync_completed_items=int(channel.sync_completed_items or 0),
            video_count=int(video_count or 0),
            processed_count=int(processed_count or 0),
            pending_video_count=int(pending_video_count or 0),
            active_job_count=int(active_job_count or 0),
            total_duration_seconds=int(total_duration_seconds or 0),
            speaker_count=int(speaker_count or 0),
        )
        for channel, video_count, processed_count, pending_video_count, total_duration_seconds, speaker_count, active_job_count in rows
    ]

@router.get("/channels/{channel_id}", response_model=Channel)
def read_channel(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel

@router.post("/channels/{channel_id}/refresh")
def refresh_channel(channel_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        raise HTTPException(status_code=409, detail="Manual channels do not support remote refresh.")
    if channel_source == "tiktok" and "tiktok.com" not in str(channel.url or "").lower():
        raise HTTPException(status_code=409, detail="This TikTok channel needs a real creator/profile URL before it can refresh.")
    channel.status = "refreshing"
    channel.sync_status_detail = "Starting channel scan..."
    channel.sync_progress = 1
    channel.sync_total_items = 0
    channel.sync_completed_items = 0
    session.add(channel)
    session.commit()
    background_tasks.add_task(get_ingestion_service().refresh_channel, channel_id)
    return {"status": "refresh_started"}


def _backfill_remote_channel_metadata_task(channel_ids: List[int]) -> None:
    for channel_id in channel_ids:
        with Session(engine) as session:
            channel = session.get(Channel, int(channel_id))
            if not channel:
                continue
            channel_source = (channel.source_type or "youtube").strip().lower()
            channel_name = str(channel.name or f"Channel {channel.id}")
            if channel_source == "manual":
                continue
            channel.status = "refreshing"
            channel.sync_status_detail = "Queued metadata backfill..."
            channel.sync_progress = 1
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()

        try:
            if channel_source == "tiktok":
                get_ingestion_service().refresh_channel(int(channel_id))
                continue

            get_ingestion_service()._backfill_dates(
                int(channel_id),
                max_items=None,
                progress_start=5,
                progress_end=95,
                detail_prefix="Backfilling popularity metadata",
                status="refreshing",
            )
            get_ingestion_service()._update_channel_sync_progress(
                int(channel_id),
                status="idle",
                detail="Metadata backfill complete.",
                progress=100,
            )
        except Exception as e:
            log_message = f"Bulk metadata backfill failed for channel {channel_id} ({channel_name}): {e}"
            logging.exception(log_message)
            get_ingestion_service()._update_channel_sync_progress(
                int(channel_id),
                status="failed",
                detail=str(e)[:240] or "Metadata backfill failed.",
                progress=0,
            )


@router.post("/channels/metadata/backfill-all")
def backfill_all_channel_metadata(background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    rows = session.exec(
        select(Channel.id, Channel.source_type)
        .where(Channel.source_type.is_(None) | (Channel.source_type != "manual"))
        .order_by(Channel.id.asc())
    ).all()
    channel_ids = [int(row[0]) for row in rows if row[0] is not None]
    if not channel_ids:
        return {"status": "nothing_to_do", "channels": 0}
    background_tasks.add_task(_backfill_remote_channel_metadata_task, channel_ids)
    return {"status": "started", "channels": len(channel_ids)}

@router.patch("/channels/{channel_id}/actively-monitored", response_model=Channel)
def set_channel_actively_monitored(
    channel_id: int,
    enabled: bool,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        if enabled:
            raise HTTPException(status_code=409, detail="Manual channels do not support active monitoring.")
        channel.actively_monitored = False
        session.add(channel)
        session.commit()
        session.refresh(channel)
        return channel
    if channel_source == "tiktok" and enabled and "tiktok.com" not in str(channel.url or "").lower():
        raise HTTPException(status_code=409, detail="This TikTok channel needs a real creator/profile URL before monitoring can be enabled.")
    channel.actively_monitored = enabled
    if not enabled and channel.status == "refreshing":
        channel.status = "active"
        channel.sync_status_detail = "Monitoring disabled."
        channel.sync_progress = 0
        channel.sync_total_items = 0
        channel.sync_completed_items = 0
    session.add(channel)
    session.commit()
    session.refresh(channel)
    if enabled:
        background_tasks.add_task(get_ingestion_service().sync_monitored_channel, channel_id)
    return channel

@router.get("/channels/{channel_id}/youtube-publish-ownership-check")
def check_channel_youtube_publish_ownership(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    try:
        check = _main()._youtube_channel_ownership_check_for_app_channel(channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ownership check failed: {e}")
    return {
        "channel_id": channel.id,
        "channel_name": channel.name,
        "channel_url": channel.url,
        **check,
    }

@router.post("/channels/{channel_id}/add-video")
def add_video_to_channel(channel_id: int, url: str, session: Session = Depends(get_session)):
    """Manually add a source video to a channel."""
    
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        raise HTTPException(status_code=409, detail="Use media upload for manual channels.")

    normalized_url = " ".join((url or "").strip().split())
    if not normalized_url:
        raise HTTPException(status_code=400, detail="A source URL is required.")

    if channel_source == "youtube":
        match = re.search(r'(?:v=|youtu\.be/|/v/)([a-zA-Z0-9_-]{11})', normalized_url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        source_id = match.group(1)
        lookup_url = f"https://www.youtube.com/watch?v={source_id}"
        existing = session.exec(select(Video).where(Video.youtube_id == source_id)).first()
        if existing:
            return existing
    elif channel_source == "tiktok":
        if "tiktok.com" not in normalized_url.lower():
            raise HTTPException(status_code=400, detail="Invalid TikTok URL")
        normalized_url = _normalize_tiktok_video_url(normalized_url)
        lookup_url = normalized_url
        existing = session.exec(select(Video).where(Video.source_url == normalized_url)).first()
        if existing:
            return existing
    else:
        raise HTTPException(status_code=409, detail="This channel type does not support remote add-video ingest yet.")

    try:
        info = _fetch_remote_video_info(lookup_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video info: {e}")

    youtube_api_meta = {}
    if channel_source == "youtube" and get_ingestion_service() is not None:
        try:
            youtube_api_meta = get_ingestion_service()._fetch_youtube_data_api_video_metadata_batch([source_id]).get(source_id) or {}
        except Exception as e:
            logging.warning("YouTube Data API metadata enrichment skipped for %s: %s", source_id, e)

    external_url = str(info.get("webpage_url") or normalized_url).strip() or normalized_url
    if channel_source == "tiktok":
        external_url = _normalize_tiktok_video_url(external_url)
    if channel_source == "tiktok":
        existing = session.exec(select(Video).where(Video.source_url == external_url)).first()
        if existing:
            return existing

    if channel_source == "youtube":
        external_id = str(info.get("id") or source_id).strip() or source_id
        unique_video_id = external_id
        media_source_type = "youtube"
        media_kind = None
    else:
        external_id = str(info.get("id") or "").strip()
        unique_video_id = _make_unique_external_video_id("tiktok", external_id, session)
        media_source_type = "tiktok"
        media_kind = "video"

    video = Video(
        youtube_id=unique_video_id,
        channel_id=channel.id,
        title=str(youtube_api_meta.get("title") or info.get("title") or "Unknown Title"),
        media_source_type=media_source_type,
        source_url=external_url,
        media_kind=media_kind,
        description=youtube_api_meta.get("description") or info.get("description"),
        published_at=get_ingestion_service()._extract_published_at_from_info(youtube_api_meta or info),
        duration=youtube_api_meta.get("duration") if youtube_api_meta.get("duration") is not None else info.get("duration"),
        view_count=youtube_api_meta.get("view_count") if youtube_api_meta.get("view_count") is not None else info.get("view_count"),
        thumbnail_url=youtube_api_meta.get("thumbnail") or _extract_best_thumbnail_url(info),
        status="pending",
    )
    session.add(video)
    session.flush()
    if channel_source == "youtube":
        try:
            get_ingestion_service().populate_placeholder_transcript(session, video, info=info)
        except Exception as e:
            logging.warning("Placeholder transcript fetch failed for %s: %s", unique_video_id, e)
    elif channel_source == "tiktok":
        try:
            get_ingestion_service().populate_placeholder_transcript(session, video, info=info)
        except Exception as e:
            logging.warning("TikTok placeholder transcript fetch failed for %s: %s", unique_video_id, e)
    session.commit()
    session.refresh(video)
    return video


def _classify_manual_media_kind(filename: str, content_type: str | None) -> Optional[str]:
    suffix = Path(filename or "").suffix.lower()
    if (content_type or "").startswith("audio/"):
        return "audio"
    if (content_type or "").startswith("video/"):
        return "video"
    if suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        return "video"
    return None


@router.post("/channels/{channel_id}/upload-media", response_model=Video)
async def upload_media_to_channel(
    channel_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    if (channel.source_type or "youtube") != "manual":
        raise HTTPException(status_code=409, detail="Media uploads are only supported for manual channels.")

    original_name = (file.filename or "").strip()
    if not original_name:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    media_kind = _classify_manual_media_kind(original_name, file.content_type)
    if media_kind is None:
        raise HTTPException(status_code=400, detail="Unsupported media type. Upload an audio or video file.")

    suffix = Path(original_name).suffix.lower()
    safe_base = get_ingestion_service().sanitize_filename(Path(original_name).stem)
    safe_name = f"{safe_base or 'episode'}{suffix}"
    storage_rel_dir = Path(f"channel_{channel.id}")
    storage_dir = MANUAL_MEDIA_DIR / storage_rel_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    unique_prefix = secrets.token_hex(6)
    stored_path = storage_dir / f"{unique_prefix}_{safe_name}"

    try:
        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    try:
        duration_raw = get_ingestion_service()._probe_media_duration_seconds(stored_path)
    except Exception:
        duration_raw = None
    duration = None
    if duration_raw and duration_raw > 0:
        duration = max(1, int(round(float(duration_raw))))

    resolved_title = " ".join((title or Path(original_name).stem).strip().split()) or Path(original_name).stem or "Uploaded Episode"

    youtube_id = f"upload_{secrets.token_hex(8)}"
    while session.exec(select(Video).where(Video.youtube_id == youtube_id)).first():
        youtube_id = f"upload_{secrets.token_hex(8)}"

    video = Video(
        youtube_id=youtube_id,
        channel_id=channel.id,
        title=resolved_title,
        media_source_type="upload",
        media_kind=media_kind,
        manual_media_path=(storage_rel_dir / stored_path.name).as_posix(),
        published_at=datetime.now(),
        duration=duration,
        status="pending",
        processed=False,
    )
    session.add(video)
    channel.last_updated = datetime.now()
    session.add(channel)
    session.commit()
    session.refresh(video)
    return video

@router.get("/channels/{channel_id}/stats")
def get_channel_stats(channel_id: int, session: Session = Depends(get_session)):
    """Get statistics for a channel"""
    from sqlalchemy import func
    
    video_count = session.exec(select(func.count(Video.id)).where(Video.channel_id == channel_id)).one()
    processed_count = session.exec(select(func.count(Video.id)).where(Video.channel_id == channel_id, Video.processed == True)).one()
    speaker_count = session.exec(select(func.count(Speaker.id)).where(Speaker.channel_id == channel_id)).one()
    transcript_count = session.exec(select(func.count(TranscriptSegment.id)).join(Video).where(Video.channel_id == channel_id)).one()
    
    return {
        "video_count": video_count,
        "processed_count": processed_count,
        "speaker_count": speaker_count,
        "transcript_count": transcript_count
    }


@router.get("/channels/{channel_id}/delete-preview")
def delete_channel_preview(channel_id: int, session: Session = Depends(get_session)):
    """Preview what will be deleted when a channel is removed."""
    from sqlalchemy import func

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    video_ids = [v.id for v in session.exec(select(Video).where(Video.channel_id == channel_id)).all()]
    video_count = len(video_ids)
    segment_count = 0
    job_count = 0
    clip_count = 0
    active_jobs = 0
    if video_ids:
        segment_count = session.exec(select(func.count(TranscriptSegment.id)).where(TranscriptSegment.video_id.in_(video_ids))).one()
        job_count = session.exec(select(func.count(Job.id)).where(Job.video_id.in_(video_ids))).one()
        clip_count = session.exec(select(func.count(Clip.id)).where(Clip.video_id.in_(video_ids))).one()
        active_jobs = session.exec(select(func.count(Job.id)).where(
            Job.video_id.in_(video_ids),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )).one()
    speaker_count = session.exec(select(func.count(Speaker.id)).where(Speaker.channel_id == channel_id)).one()

    return {
        "channel_name": channel.name,
        "video_count": video_count,
        "segment_count": segment_count,
        "speaker_count": speaker_count,
        "job_count": job_count,
        "clip_count": clip_count,
        "active_jobs": active_jobs,
    }


@router.delete("/channels/{channel_id}")
def delete_channel(channel_id: int, session: Session = Depends(get_session)):
    """Delete a channel and ALL associated data. Irreversible."""
    from sqlalchemy import delete as sa_delete, or_
    from sqlalchemy.exc import IntegrityError

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Gather related IDs
    videos = session.exec(select(Video).where(Video.channel_id == channel_id)).all()
    video_ids = [v.id for v in videos]
    speakers = session.exec(select(Speaker).where(Speaker.channel_id == channel_id)).all()
    speaker_ids = [s.id for s in speakers]

    # Block if active jobs
    if video_ids:
        active = session.exec(select(Job).where(
            Job.video_id.in_(video_ids),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )).first()
        if active:
            raise HTTPException(status_code=400, detail=f"Cannot delete channel with active job (job {active.id}, status: {active.status}). Cancel active jobs first.")

    # Cascade delete in FK-safe order.
    deleted = {
        "segment_revisions": 0,
        "segments": 0,
        "chunk_embeddings": 0,
        "funny_moments": 0,
        "clip_exports": 0,
        "clips": 0,
        "jobs": 0,
        "description_revisions": 0,
        "embeddings": 0,
        "speakers": 0,
        "videos": 0,
    }

    try:
        if video_ids:
            # Delete revisions first to avoid FK failures.
            res = session.exec(sa_delete(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id.in_(video_ids)))
            deleted["segment_revisions"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(FunnyMoment).where(FunnyMoment.video_id.in_(video_ids)))
            deleted["funny_moments"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(ClipExportArtifact).where(ClipExportArtifact.video_id.in_(video_ids)))
            deleted["clip_exports"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(Clip).where(Clip.video_id.in_(video_ids)))
            deleted["clips"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(Job).where(Job.video_id.in_(video_ids)))
            deleted["jobs"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(VideoDescriptionRevision).where(VideoDescriptionRevision.video_id.in_(video_ids)))
            deleted["description_revisions"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(TranscriptChunkEmbedding).where(TranscriptChunkEmbedding.video_id.in_(video_ids)))
            deleted["chunk_embeddings"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(TranscriptSegment).where(TranscriptSegment.video_id.in_(video_ids)))
            deleted["segments"] = int(res.rowcount or 0)

        if speaker_ids or video_ids:
            emb_conditions = []
            if speaker_ids:
                emb_conditions.append(SpeakerEmbedding.speaker_id.in_(speaker_ids))
            if video_ids:
                emb_conditions.append(SpeakerEmbedding.source_video_id.in_(video_ids))
            if emb_conditions:
                res = session.exec(sa_delete(SpeakerEmbedding).where(or_(*emb_conditions)))
                deleted["embeddings"] = int(res.rowcount or 0)

        if speaker_ids:
            res = session.exec(sa_delete(Speaker).where(Speaker.id.in_(speaker_ids)))
            deleted["speakers"] = int(res.rowcount or 0)

        if video_ids:
            res = session.exec(sa_delete(Video).where(Video.id.in_(video_ids)))
            deleted["videos"] = int(res.rowcount or 0)

        session.delete(channel)
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Delete failed due to dependent records: {e.orig}") from e

    # File cleanup (best-effort, don't fail if files are missing)
    try:
        safe_channel = get_ingestion_service().sanitize_filename(channel.name)
        audio_dir = Path(__file__).parent.parent / "data" / "audio" / safe_channel
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
    except Exception as e:
        print(f"Warning: failed to clean audio dir: {e}")

    # Clean temp files for each video
    temp_dir = Path(__file__).parent.parent / "data" / "temp"
    for vid in video_ids:
        for pattern in [f"transcript_{vid}_partial.json", f"diarization_{vid}.rttm"]:
            p = temp_dir / pattern
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Clean speaker thumbnails
    images_dir = Path(__file__).parent.parent / "data" / "images"
    for sid in speaker_ids:
        for f in images_dir.glob(f"speaker_{sid}_*"):
            try:
                f.unlink()
            except Exception:
                pass
    thumbs_dir = Path(__file__).parent.parent / "data" / "thumbnails" / "speakers"
    if thumbs_dir.exists():
        for sid in speaker_ids:
            for f in thumbs_dir.glob(f"speaker_{sid}_*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    return {"status": "deleted", "channel": channel.name, "deleted": deleted}


@router.get("/channels/{channel_id}/export")
def export_channel(
    channel_id: int,
    compact: bool = True,
    session: Session = Depends(get_session),
):
    """Export a channel archive as JSON with transcripts and speaker profiles."""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    backend_data_dir = Path(__file__).parent.parent / "data"

    def _load_speaker_thumbnail_from_path(thumbnail_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not thumbnail_path:
            return None, None
        p = thumbnail_path.strip()
        if not p.startswith("/"):
            return None, None
        if not (p.startswith("/images/") or p.startswith("/thumbnails/speakers/")):
            return None, None
        local = backend_data_dir / p.lstrip("/")
        try:
            if not local.exists() or not local.is_file():
                return None, None
            return base64.b64encode(local.read_bytes()).decode("ascii"), (local.suffix.lower() or None)
        except Exception:
            return None, None

    # Gather speakers with embeddings
    speakers = session.exec(select(Speaker).where(Speaker.channel_id == channel_id)).all()
    speakers_data = []
    for sp in speakers:
        sp_embeddings = session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.speaker_id == sp.id)).all()
        emb_data = []
        for emb in sp_embeddings:
            source_yt_id = None
            if emb.source_video_id:
                source_vid = session.get(Video, emb.source_video_id)
                if source_vid:
                    source_yt_id = source_vid.youtube_id
            emb_item = {
                "embedding_blob_b64": base64.b64encode(emb.embedding_blob).decode("ascii"),
                "source_video_youtube_id": source_yt_id,
                "sample_start_time": emb.sample_start_time,
                "sample_end_time": emb.sample_end_time,
                "sample_text": emb.sample_text,
            }
            if compact:
                emb_item.pop("sample_text", None)
            emb_data.append(emb_item)
        thumb_b64, thumb_ext = _load_speaker_thumbnail_from_path(sp.thumbnail_path)
        speakers_data.append({
            "name": sp.name,
            "embedding_blob_b64": base64.b64encode(sp.embedding_blob).decode("ascii"),
            "is_extra": sp.is_extra,
            "thumbnail_path": sp.thumbnail_path,
            "thumbnail_b64": thumb_b64,
            "thumbnail_ext": thumb_ext,
            "embeddings": emb_data,
        })

    # Gather videos with segments
    videos = session.exec(select(Video).where(Video.channel_id == channel_id)).all()
    videos_data = []
    for v in videos:
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == v.id).order_by(TranscriptSegment.start_time)
        ).all()
        seg_data = []
        for seg in segments:
            speaker_name = None
            if seg.speaker_id:
                sp = session.get(Speaker, seg.speaker_id)
                if sp:
                    speaker_name = sp.name
            seg_item = {
                "speaker_name": speaker_name,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "words": seg.words,
            }
            if compact:
                seg_item.pop("words", None)
            seg_data.append(seg_item)
        video_item = {
            "youtube_id": v.youtube_id,
            "title": v.title,
            "published_at": v.published_at.isoformat() if v.published_at else None,
            "description": v.description,
            "thumbnail_url": v.thumbnail_url,
            "duration": v.duration,
            "muted": v.muted,
            "segments": seg_data,
        }
        if compact:
            video_item.pop("description", None)
            video_item.pop("thumbnail_url", None)
        videos_data.append(video_item)

    archive = {
        "format_version": 2 if compact else 1,
        "exported_at": datetime.now().isoformat(),
        "compact": bool(compact),
        "channel": {
            "url": channel.url,
            "name": channel.name,
            "icon_url": getattr(channel, "icon_url", None),
            "header_image_url": getattr(channel, "header_image_url", None),
        },
        "speakers": speakers_data,
        "videos": videos_data,
    }

    from fastapi.responses import JSONResponse
    headers = {"Content-Disposition": f'attachment; filename="{get_ingestion_service().sanitize_filename(channel.name)}_archive.json"'}
    return JSONResponse(content=archive, headers=headers)


@router.post("/channels/import")
def import_channel(archive: dict, session: Session = Depends(get_session)):
    """Import a channel from an archive JSON. Restores speakers, transcripts, and video metadata."""
    if archive.get("format_version") not in {1, 2}:
        raise HTTPException(status_code=400, detail="Unsupported archive format version")

    ch_data = archive.get("channel", {})
    if not ch_data.get("url") or not ch_data.get("name"):
        raise HTTPException(status_code=400, detail="Archive missing channel url/name")

    # Check if channel already exists
    existing = session.exec(select(Channel).where(Channel.url == ch_data["url"])).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Channel '{existing.name}' already exists (id={existing.id}). Delete it first or use a different URL.")

    # Create channel
    channel = Channel(
        url=ch_data["url"],
        name=ch_data["name"],
        icon_url=ch_data.get("icon_url"),
        header_image_url=ch_data.get("header_image_url"),
        status="active",
        last_updated=datetime.now(),
    )
    session.add(channel)
    session.commit()
    session.refresh(channel)

    thumb_dir = Path(__file__).parent.parent / "data" / "thumbnails" / "speakers"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    def _restore_speaker_thumbnail(speaker_id: int, sp_data: dict) -> Optional[str]:
        thumb_b64 = sp_data.get("thumbnail_b64")
        if not thumb_b64:
            return None
        ext = (sp_data.get("thumbnail_ext") or ".jpg").strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        filename = f"speaker_{speaker_id}_import_{int(time.time() * 1000)}{ext}"
        out = thumb_dir / filename
        try:
            out.write_bytes(base64.b64decode(thumb_b64))
            return f"/thumbnails/speakers/{filename}"
        except Exception:
            return None

    # Create speakers and build name→id map
    speaker_map = {}  # name → speaker_id
    for sp_data in archive.get("speakers", []):
        blob = base64.b64decode(sp_data["embedding_blob_b64"])
        speaker = Speaker(
            channel_id=channel.id,
            name=sp_data["name"],
            embedding_blob=blob,
            thumbnail_path=None,
            is_extra=sp_data.get("is_extra", False),
        )
        session.add(speaker)
        session.commit()
        session.refresh(speaker)
        restored_thumbnail_path = _restore_speaker_thumbnail(speaker.id, sp_data)
        if restored_thumbnail_path:
            speaker.thumbnail_path = restored_thumbnail_path
            session.add(speaker)
            session.commit()
            session.refresh(speaker)
        speaker_map[sp_data["name"]] = speaker.id

        # Create speaker embeddings (deferred source_video_id linking)
        for emb_data in sp_data.get("embeddings", []):
            emb_blob = base64.b64decode(emb_data["embedding_blob_b64"])
            emb = SpeakerEmbedding(
                speaker_id=speaker.id,
                embedding_blob=emb_blob,
                source_video_id=None,  # Will link after videos are created
                sample_start_time=emb_data.get("sample_start_time"),
                sample_end_time=emb_data.get("sample_end_time"),
                sample_text=emb_data.get("sample_text"),
            )
            session.add(emb)
            # Store youtube_id for later linking
            emb._yt_id = emb_data.get("source_video_youtube_id")
        session.commit()

    # Create videos and segments
    yt_to_video_id = {}  # youtube_id → video.id
    imported = {"videos": 0, "segments": 0, "speakers": len(speaker_map)}
    for v_data in archive.get("videos", []):
        pub_at = None
        if v_data.get("published_at"):
            try:
                pub_at = datetime.fromisoformat(v_data["published_at"])
            except (ValueError, TypeError):
                pass

        video = Video(
            youtube_id=v_data["youtube_id"],
            channel_id=channel.id,
            title=v_data["title"],
            published_at=pub_at,
            description=v_data.get("description"),
            thumbnail_url=v_data.get("thumbnail_url"),
            duration=v_data.get("duration"),
            view_count=v_data.get("view_count"),
            muted=v_data.get("muted", False),
            status="completed" if v_data.get("segments") else "pending",
            processed=bool(v_data.get("segments")),
        )
        session.add(video)
        session.commit()
        session.refresh(video)
        yt_to_video_id[v_data["youtube_id"]] = video.id
        imported["videos"] += 1

        for seg_data in v_data.get("segments", []):
            sp_id = speaker_map.get(seg_data.get("speaker_name"))
            segment = TranscriptSegment(
                video_id=video.id,
                speaker_id=sp_id,
                start_time=seg_data["start_time"],
                end_time=seg_data["end_time"],
                text=seg_data["text"],
                words=seg_data.get("words"),
            )
            session.add(segment)
            imported["segments"] += 1
        session.commit()

    # Link speaker embeddings to video IDs now that videos exist
    for sp_data in archive.get("speakers", []):
        sp_id = speaker_map.get(sp_data["name"])
        if not sp_id:
            continue
        embeddings = session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.speaker_id == sp_id)).all()
        emb_idx = 0
        for emb_data in sp_data.get("embeddings", []):
            yt_id = emb_data.get("source_video_youtube_id")
            if yt_id and yt_id in yt_to_video_id and emb_idx < len(embeddings):
                embeddings[emb_idx].source_video_id = yt_to_video_id[yt_id]
                session.add(embeddings[emb_idx])
            emb_idx += 1
        session.commit()

    return {
        "status": "imported",
        "channel_id": channel.id,
        "channel_name": channel.name,
        "imported": imported,
    }


# --- Videos ---
