"""Speaker profile, sample, thumbnail, and merge endpoints."""
import base64
import pickle
import shutil
import time

import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    engine,
    Speaker,
    SpeakerEmbedding,
    TranscriptSegment,
    Video,
)
from ..deps import get_ingestion_service, get_session
from ..paths import IMAGES_DIR, THUMBNAILS_DIR
from ..services.speaker_merge import merge_speakers_in_session
from ..schemas import (
    ExtractThumbnailRequest,
    MergeRequest,
    MoveSpeakerProfileRequest,
    SpeakerCountsRead,
    SpeakerEpisodeAppearanceRead,
    SpeakerMergeSuggestionRead,
    SpeakerOverviewRead,
    SpeakerRead,
    SpeakerSample,
    SpeakerVoiceProfileRead,
)
from ..services import speaker_queries as spk_q

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


def _move_profile_between_speakers(
    session: Session,
    source_speaker: Speaker,
    profile: SpeakerEmbedding,
    *,
    target_speaker_id: Optional[int] = None,
    new_speaker_name: Optional[str] = None,
):
    from sqlalchemy import func

    has_target = target_speaker_id is not None
    has_new = bool((new_speaker_name or "").strip())
    if has_target == has_new:
        raise HTTPException(status_code=400, detail="Provide exactly one of target_speaker_id or new_speaker_name")

    profile_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == source_speaker.id)
    ).first() or 0
    if int(profile_count) <= 1:
        raise HTTPException(status_code=400, detail="Cannot move the last voice profile")

    target_speaker = None
    created_target = False

    if has_target:
        target_speaker = session.get(Speaker, int(target_speaker_id))
        if not target_speaker:
            raise HTTPException(status_code=404, detail="Target speaker not found")
        if target_speaker.channel_id != source_speaker.channel_id:
            raise HTTPException(status_code=400, detail="Target speaker must be in the same channel")
        if target_speaker.id == source_speaker.id:
            raise HTTPException(status_code=400, detail="Target speaker must be different from source speaker")
    else:
        new_name = (new_speaker_name or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="New speaker name is required")
        target_speaker = Speaker(
            channel_id=source_speaker.channel_id,
            name=new_name,
            embedding_blob=profile.embedding_blob,
            is_extra=False,
        )
        session.add(target_speaker)
        session.commit()
        session.refresh(target_speaker)
        created_target = True

    profile.speaker_id = target_speaker.id
    session.add(profile)

    # Keep legacy single-embedding blob fields aligned with current profiles.
    source_replacement = session.exec(
        select(SpeakerEmbedding)
        .where(SpeakerEmbedding.speaker_id == source_speaker.id, SpeakerEmbedding.id != profile.id)
        .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
    ).first()
    if source_replacement:
        source_speaker.embedding_blob = source_replacement.embedding_blob
        session.add(source_speaker)
    if not created_target:
        target_speaker.embedding_blob = profile.embedding_blob
        session.add(target_speaker)

    session.commit()

    remaining_source = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == source_speaker.id)
    ).first() or 0

    return {
        "profile_id": int(profile.id),
        "source_speaker_id": int(source_speaker.id),
        "target_speaker_id": int(target_speaker.id),
        "target_speaker_name": target_speaker.name,
        "created_target": created_target,
        "remaining_source_profiles": int(remaining_source),
    }


@router.get("/speakers", response_model=List[SpeakerRead])
def read_speakers(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: Session = Depends(get_session)
):
    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))
    page_rows = spk_q._query_speaker_page_rows(
        session=session,
        channel_id=channel_id,
        video_id=video_id,
        search=search,
        offset=safe_offset,
        limit=safe_limit,
    )
    return _build_speaker_reads(session, page_rows)


def _build_speaker_reads(session: Session, rows: list[dict]) -> List[SpeakerRead]:
    from sqlalchemy import func

    emb_counts: dict[int, int] = {}
    if rows:
        speaker_ids = [int(row["id"]) for row in rows]
        emb_query = select(
            SpeakerEmbedding.speaker_id,
            func.count(SpeakerEmbedding.id).label("cnt")
        ).where(
            SpeakerEmbedding.speaker_id.in_(speaker_ids)
        ).group_by(SpeakerEmbedding.speaker_id)
        for speaker_id, cnt in session.exec(emb_query).all():
            emb_counts[int(speaker_id)] = int(cnt or 0)

    speakers: List[SpeakerRead] = []
    for row in rows:
        speaker_id = int(row["id"])
        speakers.append(
            SpeakerRead(
                id=speaker_id,
                channel_id=int(row["channel_id"]),
                name=str(row["name"]),
                thumbnail_path=row.get("thumbnail_path"),
                is_extra=bool(row.get("is_extra")),
                total_speaking_time=float(row.get("total_speaking_time") or 0.0),
                embedding_count=int(emb_counts.get(speaker_id, 0)),
                created_at=row["created_at"],
            )
        )
    return speakers


@router.get("/speakers/overview", response_model=SpeakerOverviewRead)
def read_speaker_overview(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: Session = Depends(get_session)
):
    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))
    full_rows = spk_q._query_full_speaker_scope_rows(
        session=session,
        channel_id=channel_id,
        video_id=video_id,
        search=search,
    )
    page_rows = full_rows[safe_offset:] if safe_limit is None else full_rows[safe_offset:safe_offset + safe_limit]
    counts = SpeakerCountsRead(**spk_q._summarize_speaker_scope_rows(full_rows))
    return SpeakerOverviewRead(
        items=_build_speaker_reads(session, page_rows),
        counts=counts,
        total=len(full_rows),
        offset=safe_offset,
        limit=safe_limit,
    )


@router.get("/speakers/stats", response_model=SpeakerCountsRead)
def read_speaker_counts(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    cache_key = spk_q._speaker_scope_key(channel_id, video_id)
    cached = spk_q._get_speaker_counts_cache(cache_key)
    if cached is not None:
        return SpeakerCountsRead(**cached)
    summary = spk_q._query_speaker_count_summary(session=session, channel_id=channel_id, video_id=video_id)
    counts = SpeakerCountsRead(
        total=int(summary.get("total") or 0),
        identified=int(summary.get("identified") or 0),
        unknown=int(summary.get("unknown") or 0),
        main=int(summary.get("main") or 0),
        extras=int(summary.get("extras") or 0),
    )
    spk_q._set_speaker_counts_cache(cache_key, counts.model_dump())
    return counts

@router.get("/speakers/{speaker_id}", response_model=SpeakerRead)
def read_speaker(speaker_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func

    row = session.exec(
        select(
            Speaker.id,
            Speaker.channel_id,
            Speaker.name,
            Speaker.thumbnail_path,
            Speaker.is_extra,
            Speaker.created_at,
        ).where(Speaker.id == speaker_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Speaker not found")

    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == speaker_id)
    ).first()
    total_time = total_time_result or 0

    emb_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    return SpeakerRead(
        id=int(row[0]),
        channel_id=int(row[1]),
        name=str(row[2]),
        thumbnail_path=row[3],
        is_extra=bool(row[4]),
        total_speaking_time=round(total_time, 1),
        embedding_count=int(emb_count),
        created_at=row[5],
    )

@router.get("/speakers/{speaker_id}/appearances", response_model=List[SpeakerEpisodeAppearanceRead])
def read_speaker_appearances(
    speaker_id: int,
    offset: int = 0,
    limit: Optional[int] = None,
    response: Response = None,
    session: Session = Depends(get_session),
):
    from sqlalchemy import func, case

    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))

    base_query = (
        select(
            Video.id,
            Video.youtube_id,
            Video.title,
            Video.media_source_type,
            Video.media_kind,
            Video.published_at,
            Video.thumbnail_url,
            func.count(TranscriptSegment.id).label("segment_count"),
            func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time).label("total_time"),
            func.min(TranscriptSegment.start_time).label("first_start"),
            func.max(TranscriptSegment.end_time).label("last_end"),
        )
        .join(TranscriptSegment, TranscriptSegment.video_id == Video.id)
        .where(TranscriptSegment.speaker_id == speaker_id)
        .group_by(Video.id)
        .order_by(
            case((Video.published_at.is_(None), 1), else_=0),
            Video.published_at.desc(),
            Video.id.desc(),
        )
    )
    if response is not None:
        count_rows = session.exec(
            select(func.count()).select_from(base_query.subquery("speaker_appearance_rows"))
        ).first()
        response.headers["X-Total-Count"] = str(int(count_rows or 0))
    if safe_offset:
        base_query = base_query.offset(safe_offset)
    if safe_limit is not None:
        base_query = base_query.limit(safe_limit)

    rows = session.exec(base_query).all()

    appearances: List[SpeakerEpisodeAppearanceRead] = []
    for row in rows:
        appearances.append(
            SpeakerEpisodeAppearanceRead(
                video_id=row[0],
                youtube_id=row[1],
                title=row[2],
                media_source_type=row[3] or "youtube",
                media_kind=row[4],
                published_at=row[5],
                thumbnail_url=row[6],
                segment_count=int(row[7] or 0),
                total_speaking_time=round(float(row[8] or 0), 1),
                first_start_time=float(row[9] or 0),
                last_end_time=float(row[10] or 0),
            )
        )
    return appearances

@router.get("/speakers/{speaker_id}/profiles", response_model=List[SpeakerVoiceProfileRead])
def read_speaker_profiles(
    speaker_id: int,
    offset: int = 0,
    limit: Optional[int] = None,
    response: Response = None,
    session: Session = Depends(get_session),
):
    speaker_exists = session.exec(select(Speaker.id).where(Speaker.id == speaker_id)).first()
    if not speaker_exists:
        raise HTTPException(status_code=404, detail="Speaker not found")

    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))

    query = (
        select(
            SpeakerEmbedding.id,
            SpeakerEmbedding.speaker_id,
            SpeakerEmbedding.source_video_id,
            SpeakerEmbedding.sample_start_time,
            SpeakerEmbedding.sample_end_time,
            SpeakerEmbedding.sample_text,
            SpeakerEmbedding.created_at,
            Video.title,
            Video.youtube_id,
            Video.media_source_type,
            Video.media_kind,
            Video.published_at,
        )
        .join(Video, SpeakerEmbedding.source_video_id == Video.id, isouter=True)
        .where(SpeakerEmbedding.speaker_id == speaker_id)
        .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
    )
    if response is not None:
        total_profiles = session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
        ).first()
        response.headers["X-Total-Count"] = str(int(total_profiles or 0))
    if safe_offset:
        query = query.offset(safe_offset)
    if safe_limit is not None:
        query = query.limit(safe_limit)

    rows = session.exec(query).all()

    profiles: List[SpeakerVoiceProfileRead] = []
    for row in rows:
        profiles.append(
            SpeakerVoiceProfileRead(
                id=int(row[0]),
                speaker_id=int(row[1]),
                source_video_id=int(row[2]) if row[2] is not None else None,
                source_video_title=row[7],
                source_video_youtube_id=row[8],
                source_video_media_source_type=row[9],
                source_video_media_kind=row[10],
                source_video_published_at=row[11],
                sample_start_time=float(row[3]) if row[3] is not None else None,
                sample_end_time=float(row[4]) if row[4] is not None else None,
                sample_text=row[5],
                created_at=row[6],
            )
        )
    return profiles

@router.delete("/speakers/{speaker_id}/profiles/{profile_id}")
def delete_speaker_profile(speaker_id: int, profile_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func

    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile or profile.speaker_id != speaker_id:
        raise HTTPException(status_code=404, detail="Voice profile not found for this speaker")

    profile_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    if int(profile_count) <= 1:
        raise HTTPException(status_code=400, detail="Cannot remove the last voice profile")

    session.delete(profile)
    session.commit()
    spk_q._invalidate_speaker_query_caches()

    remaining = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    return {"status": "deleted", "profile_id": profile_id, "remaining_profiles": int(remaining)}

@router.post("/profiles/{profile_id}/reassign-segments")
def reassign_segments_for_profile(profile_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func, update

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    target_speaker = session.get(Speaker, profile.speaker_id)
    if not target_speaker:
        raise HTTPException(status_code=404, detail="Target speaker not found for this voice profile")

    total_matched = session.exec(
        select(func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(
            Video.channel_id == target_speaker.channel_id,
            TranscriptSegment.matched_profile_id == profile_id,
        )
    ).one() or 0

    to_update = session.exec(
        select(func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(
            Video.channel_id == target_speaker.channel_id,
            TranscriptSegment.matched_profile_id == profile_id,
            TranscriptSegment.speaker_id != target_speaker.id,
        )
    ).one() or 0

    if int(to_update) > 0:
        session.exec(
            update(TranscriptSegment)
            .where(
                TranscriptSegment.matched_profile_id == profile_id,
                TranscriptSegment.speaker_id != target_speaker.id,
                TranscriptSegment.video_id.in_(
                    select(Video.id).where(Video.channel_id == target_speaker.channel_id)
                ),
            )
            .values(speaker_id=target_speaker.id)
        )
        session.commit()
        spk_q._invalidate_speaker_query_caches()

    return {
        "status": "reassigned",
        "profile_id": int(profile_id),
        "target_speaker_id": int(target_speaker.id),
        "target_speaker_name": target_speaker.name,
        "channel_id": int(target_speaker.channel_id),
        "matched_segments": int(total_matched),
        "updated_segments": int(to_update),
    }


@router.post("/speakers/{speaker_id}/profiles/{profile_id}/move")
def move_speaker_profile(
    speaker_id: int,
    profile_id: int,
    req: MoveSpeakerProfileRequest,
    session: Session = Depends(get_session)
):
    source_speaker = session.get(Speaker, speaker_id)
    if not source_speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile or profile.speaker_id != speaker_id:
        raise HTTPException(status_code=404, detail="Voice profile not found for this speaker")
    result = _move_profile_between_speakers(
        session,
        source_speaker,
        profile,
        target_speaker_id=req.target_speaker_id,
        new_speaker_name=req.new_speaker_name,
    )
    spk_q._invalidate_speaker_query_caches()
    return {
        "status": "moved",
        **result,
    }

@router.get("/speakers/{speaker_id}/samples", response_model=List[SpeakerSample])
def get_speaker_samples(speaker_id: int, count: int = 3, strategy: str = "random", session: Session = Depends(get_session)):
    """
    Get audio samples for a speaker.
    strategy: 'random' (default) or 'longest'
    """
    from sqlalchemy.sql import func
    
    # Base query joining Video to get metadata
    query = select(
        TranscriptSegment,
        Video.channel_id,
        Video.youtube_id,
        Video.media_source_type,
        Video.media_kind,
    )\
        .join(Video)\
        .where(
            TranscriptSegment.speaker_id == speaker_id,
            (TranscriptSegment.end_time - TranscriptSegment.start_time) > 2.0
        )
    
    if strategy == "longest":
        # Get the longest segments, up to count
        query = query.order_by((TranscriptSegment.end_time - TranscriptSegment.start_time).desc()).limit(count)
    else:
        # Get random segments
        query = query.order_by(func.random()).limit(count)
    
    results = session.exec(query).all()
    
    # Convert to response model
    samples = []
    for segment, channel_id, youtube_id, media_source_type, media_kind in results:
        # Create a dictionary of the segment data and add the extra fields
        data = segment.model_dump()
        data["channel_id"] = channel_id
        data["youtube_id"] = youtube_id
        data["media_source_type"] = media_source_type or "youtube"
        data["media_kind"] = media_kind
        samples.append(SpeakerSample(**data))
        
    return samples

@router.post("/speakers/{speaker_id}/thumbnail/extract", response_model=SpeakerRead)
def extract_speaker_thumbnail(speaker_id: int, req: ExtractThumbnailRequest, session: Session = Depends(get_session)):
    # Validate existence quickly, then release the request session before the
    # long-running ffmpeg/yt-dlp extraction work to reduce SQLite lock time.
    if not session.get(Speaker, speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")

    try:
        # Use the ingestion service singleton owned by main's lifespan;
        # lazily create it if the lifespan has not run (e.g. in tests).
        app_main = _main()
        if app_main.ingestion_service is None:
            from ..services.ingestion import IngestionService
            app_main.ingestion_service = IngestionService()

        try:
            session.close()
        except Exception:
            pass

        # Extract and update
        thumb_path = get_ingestion_service().extract_frame_and_crop(req.video_id, req.timestamp, req.crop_coords)

        # SQLite can be briefly locked by the queue worker; retry a few times.
        from sqlalchemy.exc import OperationalError
        import time as _time

        last_error = None
        for attempt in range(5):
            try:
                with Session(engine) as write_session:
                    speaker = write_session.get(Speaker, speaker_id)
                    if not speaker:
                        raise HTTPException(status_code=404, detail="Speaker not found")
                    speaker.thumbnail_path = thumb_path
                    write_session.add(speaker)
                    write_session.commit()
                    spk_q._invalidate_speaker_query_caches()
                    write_session.refresh(speaker)
                    return read_speaker(speaker_id, write_session)
            except OperationalError as e:
                last_error = e
                if "database is locked" not in str(e).lower() or attempt == 4:
                    raise
                _time.sleep(0.2 * (attempt + 1))

        if last_error:
            raise last_error
        
    except Exception as e:
        # Don't print traceback to stderr as it causes issues in some environments
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.delete("/speakers/{speaker_id}/thumbnail", response_model=SpeakerRead)
def delete_speaker_thumbnail(speaker_id: int, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    old_path = speaker.thumbnail_path
    speaker.thumbnail_path = None
    session.add(speaker)
    session.commit()
    spk_q._invalidate_speaker_query_caches()
    session.refresh(speaker)

    # Best-effort cleanup of local image files we own.
    if old_path:
        try:
            path_str = str(old_path)
            local_file = None
            if path_str.startswith("/images/"):
                local_file = IMAGES_DIR / Path(path_str).name
            elif path_str.startswith("/thumbnails/"):
                local_file = THUMBNAILS_DIR / Path(path_str).relative_to("/thumbnails")
            if local_file and local_file.exists():
                local_file.unlink()
        except Exception:
            pass

    return read_speaker(speaker_id, session)

@router.post("/speakers/{speaker_id}/thumbnail", response_model=SpeakerRead)
async def upload_thumbnail(speaker_id: int, file: UploadFile = File(...), session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker: raise HTTPException(status_code=404, detail="Speaker not found")
    
    # Save file
    safe_name = f"speaker_{speaker_id}_{file.filename}"
    file_path = IMAGES_DIR / safe_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    speaker.thumbnail_path = f"/images/{safe_name}"
    session.add(speaker)
    session.commit()
    spk_q._invalidate_speaker_query_caches()
    session.refresh(speaker)
    return read_speaker(speaker_id, session)

@router.post("/speakers/{speaker_id}/thumbnail_base64", response_model=SpeakerRead)
def upload_thumbnail_base64(speaker_id: int, data: dict, session: Session = Depends(get_session)):
    # Expects {"image": "base64string..."}
    speaker = session.get(Speaker, speaker_id)
    if not speaker: raise HTTPException(status_code=404, detail="Speaker not found")
    
    b64_str = data.get("image")
    if not b64_str: raise HTTPException(status_code=400, detail="No image data")
    
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
        
    image_data = base64.b64decode(b64_str)
    filename = f"speaker_{speaker_id}_pasted.png"
    file_path = IMAGES_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(image_data)
        
    speaker.thumbnail_path = f"/images/{filename}"
    session.add(speaker)
    session.commit()
    spk_q._invalidate_speaker_query_caches()
    session.refresh(speaker)
    return read_speaker(speaker_id, session)

@router.patch("/speakers/{speaker_id}", response_model=SpeakerRead)
def update_speaker(speaker_id: int, data: dict, session: Session = Depends(get_session)):
    from sqlalchemy import func
    from sqlalchemy.exc import OperationalError
    wants_name = "name" in data
    wants_is_extra = "is_extra" in data
    new_name = None
    if wants_name:
        new_name = str(data.get("name") or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
    new_is_extra = bool(data.get("is_extra")) if wants_is_extra else None

    updated_speaker = None
    max_attempts = 24
    for attempt in range(max_attempts):
        speaker = session.get(Speaker, speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")

        if wants_name:
            speaker.name = new_name
        if wants_is_extra:
            speaker.is_extra = new_is_extra

        session.add(speaker)
        try:
            session.commit()
            session.refresh(speaker)
            updated_speaker = speaker
            break
        except OperationalError as e:
            session.rollback()
            is_locked = "database is locked" in str(e).lower()
            if not is_locked:
                raise HTTPException(status_code=500, detail=f"Failed to update speaker: {e}")
            if attempt >= (max_attempts - 1):
                raise HTTPException(status_code=503, detail="Database busy. Please retry in a moment.")
            time.sleep(min(0.2 * (attempt + 1), 1.5))

    if not updated_speaker:
        raise HTTPException(status_code=503, detail="Database busy. Please retry in a moment.")

    spk_q._invalidate_speaker_query_caches()
    
    # Calculate total speaking time for the response
    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == speaker_id)
    ).first()
    
    return SpeakerRead(
        id=updated_speaker.id,
        channel_id=updated_speaker.channel_id,
        name=updated_speaker.name,
        thumbnail_path=updated_speaker.thumbnail_path,
        is_extra=updated_speaker.is_extra,
        total_speaking_time=round(total_time_result or 0, 1),
        embedding_count=session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
        ).first() or 0,
        created_at=updated_speaker.created_at
    )

@router.post("/speakers/merge", response_model=SpeakerRead)
def merge_speakers(req: MergeRequest, session: Session = Depends(get_session)):
    """Merge multiple speakers into one target speaker.
    Reassigns all speaker-owned records and moves all embeddings to the target.
    Deletes the source speakers."""
    from sqlalchemy import func
    
    target = session.get(Speaker, req.target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target speaker not found")
    
    if req.target_id in req.source_ids:
        raise HTTPException(status_code=400, detail="Target speaker cannot be in source list")
    
    # Only merge speakers in the same channel as target
    source_id_rows = session.exec(
        select(Speaker.id).where(
            Speaker.id.in_(req.source_ids),
            Speaker.channel_id == target.channel_id,
        )
    ).all()
    source_ids = [row[0] if isinstance(row, tuple) else row for row in source_id_rows]

    if not source_ids:
        raise HTTPException(status_code=400, detail="No valid source speakers found in target channel")

    try:
        # Use SQL updates/deletes to avoid ORM relationship synchronization nulling child FKs
        # and to keep all speaker-owned rows consistent before deleting the sources.
        merge_speakers_in_session(session, target_id=req.target_id, source_ids=source_ids)
        session.commit()
        spk_q._invalidate_speaker_query_caches()
    except Exception:
        session.rollback()
        raise

    session.refresh(target)
    
    # Build response
    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == req.target_id)
    ).first()
    
    emb_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == req.target_id)
    ).first() or 0
    
    return SpeakerRead(
        id=target.id,
        channel_id=target.channel_id,
        name=target.name,
        thumbnail_path=target.thumbnail_path,
        is_extra=target.is_extra,
        total_speaking_time=round(total_time_result or 0, 1),
        embedding_count=emb_count,
        created_at=target.created_at
    )


@router.get("/channels/{channel_id}/speaker-merge-suggestions", response_model=List[SpeakerMergeSuggestionRead])
def get_channel_speaker_merge_suggestions(
    channel_id: int,
    threshold: float = 0.12,
    limit: int = 100,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    threshold = max(0.0, min(float(threshold), 1.0))
    limit = max(1, min(int(limit), 500))

    rows = session.exec(
        select(SpeakerEmbedding, Speaker)
        .join(Speaker, SpeakerEmbedding.speaker_id == Speaker.id)
        .where(Speaker.channel_id == channel_id)
    ).all()
    if not rows:
        return []

    vectors_by_speaker: dict[int, list[np.ndarray]] = {}
    speaker_names: dict[int, str] = {}
    for emb, speaker in rows:
        try:
            vec = np.asarray(pickle.loads(emb.embedding_blob), dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        speaker_id = int(speaker.id)
        vectors_by_speaker.setdefault(speaker_id, []).append((vec / norm).astype(np.float32, copy=False))
        speaker_names[speaker_id] = speaker.name

    speaker_ids = sorted(vectors_by_speaker.keys())
    if len(speaker_ids) < 2:
        return []

    centroids = []
    for speaker_id in speaker_ids:
        speaker_vectors = vectors_by_speaker[speaker_id]
        centroid = np.mean(np.stack(speaker_vectors, axis=0), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            continue
        centroids.append((speaker_id, (centroid / centroid_norm).astype(np.float32, copy=False)))

    if len(centroids) < 2:
        return []

    speaker_ids = [speaker_id for speaker_id, _ in centroids]
    matrix = np.stack([vec for _, vec in centroids], axis=0)

    emb_count_rows = session.exec(
        select(SpeakerEmbedding.speaker_id, func.count(SpeakerEmbedding.id))
        .join(Speaker, SpeakerEmbedding.speaker_id == Speaker.id)
        .where(Speaker.channel_id == channel_id)
        .group_by(SpeakerEmbedding.speaker_id)
    ).all()
    segment_count_rows = session.exec(
        select(TranscriptSegment.speaker_id, func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(Video.channel_id == channel_id, TranscriptSegment.speaker_id.is_not(None))
        .group_by(TranscriptSegment.speaker_id)
    ).all()
    embedding_counts = {int(speaker_id): int(count or 0) for speaker_id, count in emb_count_rows}
    segment_counts = {int(speaker_id): int(count or 0) for speaker_id, count in segment_count_rows}

    suggestions: list[SpeakerMergeSuggestionRead] = []
    distances = 1.0 - (matrix @ matrix.T)
    for i in range(len(speaker_ids)):
        for j in range(i + 1, len(speaker_ids)):
            dist = float(distances[i, j])
            if not np.isfinite(dist) or dist > threshold:
                continue

            left_id = int(speaker_ids[i])
            right_id = int(speaker_ids[j])
            left_seg_count = int(segment_counts.get(left_id, 0))
            right_seg_count = int(segment_counts.get(right_id, 0))
            left_emb_count = int(embedding_counts.get(left_id, 0))
            right_emb_count = int(embedding_counts.get(right_id, 0))

            target_id = left_id
            source_id = right_id
            if (
                right_seg_count > left_seg_count
                or (right_seg_count == left_seg_count and right_emb_count > left_emb_count)
                or (right_seg_count == left_seg_count and right_emb_count == left_emb_count and right_id < left_id)
            ):
                target_id = right_id
                source_id = left_id

            suggestions.append(
                SpeakerMergeSuggestionRead(
                    source_speaker_id=source_id,
                    source_speaker_name=speaker_names.get(source_id, f"Speaker {source_id}"),
                    target_speaker_id=target_id,
                    target_speaker_name=speaker_names.get(target_id, f"Speaker {target_id}"),
                    distance=round(dist, 6),
                    source_embedding_count=int(embedding_counts.get(source_id, 0)),
                    target_embedding_count=int(embedding_counts.get(target_id, 0)),
                    source_segment_count=int(segment_counts.get(source_id, 0)),
                    target_segment_count=int(segment_counts.get(target_id, 0)),
                )
            )

    suggestions.sort(
        key=lambda item: (
            item.distance,
            -item.target_segment_count,
            -item.target_embedding_count,
            item.source_speaker_id,
            item.target_speaker_id,
        )
    )
    return suggestions[:limit]
