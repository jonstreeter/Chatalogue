"""Speaker list/count query builders with TTL caches, shared by speaker and segment routers."""
import os
import re
import threading
import time
from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import IS_POSTGRES, Speaker, TranscriptSegment

_SPEAKER_QUERY_CACHE_TTL_SECONDS = max(
    3,
    int(os.getenv("SPEAKER_QUERY_CACHE_TTL_SECONDS", "120"))
)
_speaker_list_cache_lock = threading.Lock()
_speaker_list_cache: dict[str, tuple[float, list[dict]]] = {}
_speaker_counts_cache_lock = threading.Lock()
_speaker_counts_cache: dict[str, tuple[float, dict]] = {}
_speaker_scope_cache_lock = threading.Lock()
_speaker_scope_cache: dict[str, tuple[float, list[dict]]] = {}


def _speaker_cache_fresh(ts: float) -> bool:
    return (time.time() - ts) < _SPEAKER_QUERY_CACHE_TTL_SECONDS


def _get_speaker_list_cache(key: str) -> Optional[list[dict]]:
    with _speaker_list_cache_lock:
        cached = _speaker_list_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_list_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_list_cache(key: str, value: list[dict]) -> None:
    with _speaker_list_cache_lock:
        _speaker_list_cache[key] = (time.time(), value)


def _get_speaker_counts_cache(key: str) -> Optional[dict]:
    with _speaker_counts_cache_lock:
        cached = _speaker_counts_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_counts_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_counts_cache(key: str, value: dict) -> None:
    with _speaker_counts_cache_lock:
        _speaker_counts_cache[key] = (time.time(), value)


def _get_speaker_scope_cache(key: str) -> Optional[list[dict]]:
    with _speaker_scope_cache_lock:
        cached = _speaker_scope_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_scope_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_scope_cache(key: str, value: list[dict]) -> None:
    with _speaker_scope_cache_lock:
        _speaker_scope_cache[key] = (time.time(), value)


def _is_unknown_speaker_name(name: Optional[str]) -> bool:
    normalized = (name or "").strip()
    if not normalized:
        return True
    if re.match(r"^unknown(\s+speaker)?$", normalized, re.IGNORECASE):
        return True
    if re.match(r"^speaker\s+\d+$", normalized, re.IGNORECASE):
        return True
    return False


def _speaker_scope_key(channel_id: Optional[int], video_id: Optional[int], search: Optional[str] = None) -> str:
    normalized_search = (search or "").strip().lower()
    return f"channel:{channel_id or 'all'}|video:{video_id or 'all'}|search:{normalized_search or 'all'}"


def _build_speaker_scope_totals_subquery(
    *,
    channel_id: Optional[int],
    video_id: Optional[int],
):

    seg_duration = (TranscriptSegment.end_time - TranscriptSegment.start_time)
    total_time = func.sum(seg_duration).label("total_time")
    query = (
        select(
            TranscriptSegment.speaker_id.label("speaker_id"),
            total_time,
        )
        .where(TranscriptSegment.speaker_id.is_not(None))
    )

    if video_id:
        query = query.where(TranscriptSegment.video_id == video_id)

    if channel_id:
        query = query.join(Speaker, Speaker.id == TranscriptSegment.speaker_id).where(Speaker.channel_id == channel_id)

    query = query.group_by(TranscriptSegment.speaker_id).having(total_time > 5.0)
    return query.subquery("speaker_scope_totals")


def _build_speaker_scope_list_query(
    *,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str] = None,
):

    totals = _build_speaker_scope_totals_subquery(channel_id=channel_id, video_id=video_id)
    total_time = totals.c.total_time
    query = (
        select(
            Speaker.id,
            Speaker.channel_id,
            Speaker.name,
            Speaker.thumbnail_path,
            Speaker.is_extra,
            Speaker.created_at,
            total_time,
        )
        .join(totals, totals.c.speaker_id == Speaker.id)
    )

    if channel_id:
        query = query.where(Speaker.channel_id == channel_id)

    normalized_search = (search or "").strip().lower()
    if normalized_search:
        query = query.where(func.lower(Speaker.name).like(f"%{normalized_search}%"))

    return query, total_time


def _query_speaker_page_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str],
    offset: int,
    limit: Optional[int],
) -> list[dict]:
    scope_key = _speaker_scope_key(channel_id, video_id, search)
    cache_key = f"{scope_key}|offset:{offset}|limit:{limit if limit is not None else 'all'}"
    cached_rows = _get_speaker_list_cache(cache_key)
    if cached_rows is not None:
        return cached_rows

    if limit is None:
        out = _query_full_speaker_scope_rows(
            session=session,
            channel_id=channel_id,
            video_id=video_id,
            search=search,
        )[offset:]
        _set_speaker_list_cache(cache_key, list(out))
        return list(out)

    query, total_time = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id, search=search)
    query = query.order_by(total_time.desc()).offset(max(0, offset)).limit(max(0, limit))

    rows = session.exec(query).all()
    out: list[dict] = []
    for speaker_id, speaker_channel_id, name, thumbnail_path, is_extra, created_at, total_time_value in rows:
        out.append(
            {
                "id": int(speaker_id),
                "channel_id": int(speaker_channel_id),
                "name": str(name),
                "thumbnail_path": thumbnail_path,
                "is_extra": bool(is_extra),
                "created_at": created_at,
                "total_speaking_time": round(float(total_time_value or 0.0), 1),
            }
        )
    _set_speaker_list_cache(cache_key, out)
    return out


def _query_speaker_count_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
) -> list[tuple[str, bool, float]]:
    query, _ = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id)
    rows_subquery = query.subquery("speaker_scope_count_rows")
    count_query = select(
        rows_subquery.c.name,
        rows_subquery.c.is_extra,
        rows_subquery.c.total_time,
    )
    return [
        (
            str(name or ""),
            bool(is_extra),
            float(total_time_value or 0.0),
        )
        for name, is_extra, total_time_value in session.exec(count_query).all()
    ]


def _query_speaker_count_summary(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
) -> dict[str, int]:
    cached_rows = _get_speaker_scope_cache(_speaker_scope_key(channel_id, video_id))
    if cached_rows is not None:
        return _summarize_speaker_scope_rows(cached_rows)

    from sqlalchemy import case, or_

    query, _ = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id)
    rows_subquery = query.subquery("speaker_scope_count_summary")
    name_col = rows_subquery.c.name
    is_extra_col = rows_subquery.c.is_extra
    total_time_col = rows_subquery.c.total_time

    if IS_POSTGRES:
        unknown_expr = or_(
            func.btrim(name_col) == "",
            func.lower(name_col).in_(["unknown", "unknown speaker"]),
            name_col.op("~*")(r"^speaker\s+\d+$"),
        )
    else:
        lowered_name = func.lower(name_col)
        unknown_expr = or_(
            func.trim(name_col) == "",
            lowered_name.in_(["unknown", "unknown speaker"]),
            lowered_name.like("speaker %"),
        )

    extras_expr = or_(is_extra_col.is_(True), total_time_col < 60.0)
    summary_query = select(
        func.count().label("total"),
        func.sum(case((unknown_expr, 1), else_=0)).label("unknown"),
        func.sum(case((~unknown_expr, 1), else_=0)).label("identified"),
        func.sum(case((~unknown_expr & extras_expr, 1), else_=0)).label("extras"),
        func.sum(case((~unknown_expr & ~extras_expr, 1), else_=0)).label("main"),
    )
    row = session.exec(summary_query).first()
    if not row:
        return {"total": 0, "identified": 0, "unknown": 0, "main": 0, "extras": 0}
    return {
        "total": int(row[0] or 0),
        "unknown": int(row[1] or 0),
        "identified": int(row[2] or 0),
        "extras": int(row[3] or 0),
        "main": int(row[4] or 0),
    }


def _query_full_speaker_scope_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str] = None,
) -> list[dict]:
    scope_key = _speaker_scope_key(channel_id, video_id, search)
    cached_rows = _get_speaker_scope_cache(scope_key)
    if cached_rows is not None:
        return cached_rows

    query, total_time = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id, search=search)
    query = query.order_by(total_time.desc())

    rows = session.exec(query).all()
    out: list[dict] = []
    for speaker_id, speaker_channel_id, name, thumbnail_path, is_extra, created_at, total_time_value in rows:
        out.append(
            {
                "id": int(speaker_id),
                "channel_id": int(speaker_channel_id),
                "name": str(name),
                "thumbnail_path": thumbnail_path,
                "is_extra": bool(is_extra),
                "created_at": created_at,
                "total_speaking_time": round(float(total_time_value or 0.0), 1),
            }
        )

    _set_speaker_scope_cache(scope_key, out)
    return out


def _summarize_speaker_scope_rows(rows: list[dict]) -> dict[str, int]:
    total = 0
    unknown = 0
    identified = 0
    extras = 0
    main = 0

    for row in rows:
        total += 1
        name = str(row.get("name") or "")
        is_extra = bool(row.get("is_extra"))
        total_time = float(row.get("total_speaking_time") or 0.0)
        row_is_unknown = _is_unknown_speaker_name(name)
        if row_is_unknown:
            unknown += 1
            continue
        identified += 1
        if is_extra or total_time < 60.0:
            extras += 1
        else:
            main += 1

    return {
        "total": int(total),
        "unknown": int(unknown),
        "identified": int(identified),
        "extras": int(extras),
        "main": int(main),
    }


def _invalidate_speaker_query_caches() -> None:
    with _speaker_list_cache_lock:
        _speaker_list_cache.clear()
    with _speaker_counts_cache_lock:
        _speaker_counts_cache.clear()
    with _speaker_scope_cache_lock:
        _speaker_scope_cache.clear()
