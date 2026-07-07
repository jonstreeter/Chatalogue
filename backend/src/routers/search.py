"""Transcript search (keyword + semantic) and semantic-index endpoints."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import IS_POSTGRES, Channel, Speaker, TranscriptSegment, Video
from ..deps import get_session
from ..services import semantic_search as sem_svc
from ..schemas import (
    SemanticIndexRebuildResponse,
    SemanticIndexStatus,
    SemanticSearchHit,
    SemanticSearchRequest,
    SemanticSearchPage,
    TranscriptSearchItemRead,
    TranscriptSearchPage,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/search", response_model=TranscriptSearchPage)
def search_segments(
    q: str,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    sort: str = "newest",
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    from sqlalchemy import case

    q = (q or "").strip()
    if not q:
        return TranscriptSearchPage(items=[], total=0, limit=max(1, min(limit, 200)), offset=max(0, offset), has_more=False)

    if year is not None and (year < 2000 or year > 2100):
        raise HTTPException(status_code=400, detail="year must be between 2000 and 2100")
    if month is not None and (month < 1 or month > 12):
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")
    sort_mode = (sort or "newest").lower()
    if sort_mode == "chronological":
        sort_mode = "oldest"
    if sort_mode not in ["newest", "oldest"]:
        raise HTTPException(status_code=400, detail="sort must be 'newest' or 'oldest'")

    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)

    needs_video_join = bool(channel_id or year is not None or month is not None)

    q_like = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if IS_POSTGRES:
        text_match = TranscriptSegment.text.ilike(f"%{q_like}%", escape="\\")
    else:
        text_match = TranscriptSegment.text.contains(q)

    data_query = (
        select(
            TranscriptSegment.id,
            TranscriptSegment.video_id,
            TranscriptSegment.speaker_id,
            TranscriptSegment.matched_profile_id,
            TranscriptSegment.start_time,
            TranscriptSegment.end_time,
            TranscriptSegment.text,
            Speaker.name,
        )
        .join(Speaker, isouter=True)
        .where(text_match)
    )
    count_query = select(func.count(TranscriptSegment.id)).where(text_match)

    if video_id:
        data_query = data_query.where(TranscriptSegment.video_id == video_id)
        count_query = count_query.where(TranscriptSegment.video_id == video_id)

    if needs_video_join:
        data_query = data_query.join(Video, TranscriptSegment.video_id == Video.id)
        count_query = count_query.join(Video, TranscriptSegment.video_id == Video.id)

    if channel_id:
        data_query = data_query.where(Video.channel_id == channel_id)
        count_query = count_query.where(Video.channel_id == channel_id)

    if year is not None:
        if IS_POSTGRES:
            data_query = data_query.where(func.extract("year", Video.published_at) == year)
            count_query = count_query.where(func.extract("year", Video.published_at) == year)
        else:
            year_str = f"{year:04d}"
            data_query = data_query.where(func.strftime("%Y", Video.published_at) == year_str)
            count_query = count_query.where(func.strftime("%Y", Video.published_at) == year_str)

    if month is not None:
        if IS_POSTGRES:
            data_query = data_query.where(func.extract("month", Video.published_at) == month)
            count_query = count_query.where(func.extract("month", Video.published_at) == month)
        else:
            month_str = f"{month:02d}"
            data_query = data_query.where(func.strftime("%m", Video.published_at) == month_str)
            count_query = count_query.where(func.strftime("%m", Video.published_at) == month_str)

    # Stable ordering for pagination: newest videos first, then transcript order.
    if needs_video_join:
        nulls_last = case((Video.published_at.is_(None), 1), else_=0)
        if sort_mode == "oldest":
            data_query = data_query.order_by(
                nulls_last,
                Video.published_at.asc(),
                TranscriptSegment.video_id.asc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
        else:
            data_query = data_query.order_by(
                nulls_last,
                Video.published_at.desc(),
                TranscriptSegment.video_id.desc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
    else:
        if sort_mode == "oldest":
            data_query = data_query.order_by(
                TranscriptSegment.video_id.asc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
        else:
            data_query = data_query.order_by(TranscriptSegment.id.desc())

    total = session.exec(count_query).one()
    results = session.exec(data_query.offset(safe_offset).limit(safe_limit)).all()

    items: List[TranscriptSearchItemRead] = []
    for row in results:
        items.append(
            TranscriptSearchItemRead(
                id=int(row[0]),
                video_id=int(row[1]),
                speaker_id=row[2],
                matched_profile_id=row[3],
                start_time=float(row[4]),
                end_time=float(row[5]),
                text=row[6],
                speaker=row[7],
            )
        )

    return TranscriptSearchPage(
        items=items,
        total=int(total or 0),
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < int(total or 0),
    )

@router.post("/search/semantic", response_model=SemanticSearchPage)
def search_semantic(body: SemanticSearchRequest):
    """Semantic or hybrid transcript search using chunk embeddings."""
    q = (body.query or "").strip()
    if not q:
        return SemanticSearchPage(items=[], total=0, limit=body.limit, offset=body.offset)

    safe_limit = max(1, min(body.limit, 200))
    safe_offset = max(0, body.offset)

    try:
        if body.mode == "hybrid":
            result = sem_svc.hybrid_search(
                query=q,
                channel_id=body.channel_id,
                video_id=body.video_id,
                speaker_id=body.speaker_id,
                year=body.year,
                month=body.month,
                limit=safe_limit,
                offset=safe_offset,
            )
        else:
            result = sem_svc.semantic_search(
                query=q,
                channel_id=body.channel_id,
                video_id=body.video_id,
                speaker_id=body.speaker_id,
                year=body.year,
                month=body.month,
                limit=safe_limit,
                offset=safe_offset,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {exc}")

    items = [SemanticSearchHit(**hit) for hit in result["items"]]
    return SemanticSearchPage(
        items=items,
        total=result["total"],
        limit=safe_limit,
        offset=safe_offset,
    )


@router.get("/semantic-index/status", response_model=SemanticIndexStatus)
def get_semantic_index_status():
    """Return current semantic indexing job progress."""
    return SemanticIndexStatus(**sem_svc.get_indexing_status())


@router.post("/videos/{video_id}/semantic-index/rebuild", response_model=SemanticIndexRebuildResponse)
def rebuild_video_semantic_index(video_id: int, session: Session = Depends(get_session)):
    """Queue a semantic index rebuild for a single video."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    started = sem_svc.start_index_job([video_id])
    if not started:
        return SemanticIndexRebuildResponse(
            started=False,
            message="An indexing job is already running.",
            video_ids=[video_id],
        )
    return SemanticIndexRebuildResponse(
        started=True,
        message=f"Semantic indexing started for video {video_id}.",
        video_ids=[video_id],
    )


@router.post("/channels/{channel_id}/semantic-index/rebuild", response_model=SemanticIndexRebuildResponse)
def rebuild_channel_semantic_index(channel_id: int, session: Session = Depends(get_session)):
    """Queue a semantic index rebuild for all processed videos in a channel."""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .where(Video.processed == True)
    ).all()
    video_ids = [v.id for v in videos]

    if not video_ids:
        return SemanticIndexRebuildResponse(
            started=False,
            message="No processed videos found for this channel.",
            video_ids=[],
        )

    started = sem_svc.start_index_job(video_ids)
    if not started:
        return SemanticIndexRebuildResponse(
            started=False,
            message="An indexing job is already running.",
            video_ids=video_ids,
        )
    return SemanticIndexRebuildResponse(
        started=True,
        message=f"Semantic indexing started for {len(video_ids)} video(s) in channel {channel_id}.",
        video_ids=video_ids,
    )
