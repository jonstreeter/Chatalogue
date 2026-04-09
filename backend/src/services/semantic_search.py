"""Semantic search service.

Handles transcript chunking, embedding generation, indexing, and retrieval.

Storage: embeddings stored as float32 bytes (BYTEA in Postgres, BLOB in SQLite).
Similarity: cosine in Python/numpy — appropriate for local-first scale.
Model: BAAI/bge-small-en-v1.5 via sentence-transformers (384-dim, CPU-friendly).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    Speaker,
    TranscriptChunkEmbedding,
    TranscriptSegment,
    Video,
    engine,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Chunking heuristics
MAX_GAP_SECONDS = 10.0          # break chunk run on silence > this
TARGET_CHUNK_CHARS_MIN = 500
TARGET_CHUNK_CHARS_MAX = 1200
OVERLAP_SEGMENTS = 1            # segments to back up when starting next chunk

# RRF constant
RRF_K = 60

# ── Embedding model singleton ─────────────────────────────────────────────────

_model_lock = threading.Lock()
_embedding_model = None


def _get_model():
    global _embedding_model
    with _model_lock:
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer  # deferred import
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _embedding_model


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    return np.array(vecs, dtype=np.float32)


def _to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalised vectors."""
    return float(np.dot(a, b))


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _chunk_hash(video_id: int, segment_ids: List[int], text: str) -> str:
    key = f"{video_id}|{sorted(segment_ids)}|{text.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


# ── Background indexing state ─────────────────────────────────────────────────

_index_lock = threading.Lock()
_index_state: Dict[str, Any] = {
    "is_running": False,
    "current_video_id": None,
    "current_video_title": None,
    "videos_completed": 0,
    "videos_total": 0,
    "started_at": None,
    "last_finished_at": None,
}


def get_indexing_status() -> dict:
    with _index_lock:
        return dict(_index_state)


# ── Chunking ──────────────────────────────────────────────────────────────────

def build_chunks_for_video(session: Session, video_id: int) -> List[dict]:
    """Return speaker-pure transcript chunks for a video.

    Each chunk stays within one speaker's contiguous run and respects
    the target size window. Chunks overlap by OVERLAP_SEGMENTS at run
    boundaries so ideas that span adjacent segments aren't cut off.
    """
    segs = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
    ).all()

    if not segs:
        return []

    video = session.get(Video, video_id)
    if not video:
        return []
    channel_id = video.channel_id

    chunks: List[dict] = []

    i = 0
    while i < len(segs):
        # Collect a same-speaker run with no large gaps
        run: List[TranscriptSegment] = [segs[i]]
        j = i + 1
        while j < len(segs):
            prev, curr = segs[j - 1], segs[j]
            if curr.speaker_id != prev.speaker_id:
                break
            if (curr.start_time - prev.end_time) > MAX_GAP_SECONDS:
                break
            run.append(curr)
            j += 1
        i = j

        # Slice the run into target-sized chunks with overlap
        k = 0
        while k < len(run):
            chunk_segs = [run[k]]
            char_count = len(run[k].text)

            m = k + 1
            while m < len(run):
                next_chars = len(run[m].text)
                if char_count >= TARGET_CHUNK_CHARS_MIN and char_count + next_chars > TARGET_CHUNK_CHARS_MAX:
                    break
                chunk_segs.append(run[m])
                char_count += next_chars
                m += 1

            chunk_text = " ".join(s.text.strip() for s in chunk_segs if s.text.strip())
            if not chunk_text.strip():
                k = max(k + 1, m - OVERLAP_SEGMENTS)
                continue

            seg_ids = [s.id for s in chunk_segs]
            chunks.append({
                "channel_id": channel_id,
                "video_id": video_id,
                "speaker_id": chunk_segs[0].speaker_id,
                "start_time": chunk_segs[0].start_time,
                "end_time": chunk_segs[-1].end_time,
                "chunk_text": chunk_text,
                "chunk_token_estimate": _token_estimate(chunk_text),
                "segment_ids_json": json.dumps(seg_ids),
                "content_hash": _chunk_hash(video_id, seg_ids, chunk_text),
            })

            advance = max(1, len(chunk_segs) - OVERLAP_SEGMENTS)
            k += advance

    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_video(video_id: int) -> dict:
    """Chunk, embed, and upsert a single video. Skips unchanged chunks."""
    with Session(engine) as session:
        video = session.get(Video, video_id)
        if not video:
            return {"error": "Video not found"}

        chunks = build_chunks_for_video(session, video_id)
        if not chunks:
            return {"chunks_indexed": 0, "chunks_skipped": 0, "chunks_deleted": 0}

        # Existing rows for this video
        existing_rows = session.exec(
            select(TranscriptChunkEmbedding.content_hash, TranscriptChunkEmbedding.id)
            .where(TranscriptChunkEmbedding.video_id == video_id)
        ).all()
        existing_by_hash: dict[str, int] = {row[0]: row[1] for row in existing_rows}

        current_hashes = {c["content_hash"] for c in chunks}
        new_chunks = [c for c in chunks if c["content_hash"] not in existing_by_hash]
        stale_ids = [existing_by_hash[h] for h in existing_by_hash if h not in current_hashes]

        # Delete stale rows
        for chunk_id in stale_ids:
            obj = session.get(TranscriptChunkEmbedding, chunk_id)
            if obj:
                session.delete(obj)

        # Embed + insert new chunks
        if new_chunks:
            texts = [c["chunk_text"] for c in new_chunks]
            embeddings = _embed_texts(texts)
            now = datetime.now()
            for chunk_data, emb in zip(new_chunks, embeddings):
                session.add(TranscriptChunkEmbedding(
                    channel_id=chunk_data["channel_id"],
                    video_id=chunk_data["video_id"],
                    speaker_id=chunk_data["speaker_id"],
                    start_time=chunk_data["start_time"],
                    end_time=chunk_data["end_time"],
                    chunk_text=chunk_data["chunk_text"],
                    chunk_token_estimate=chunk_data["chunk_token_estimate"],
                    segment_ids_json=chunk_data["segment_ids_json"],
                    embedding_model=EMBEDDING_MODEL_NAME,
                    embedding_dim=EMBEDDING_DIM,
                    embedding_bytes=_to_bytes(emb),
                    content_hash=chunk_data["content_hash"],
                    created_at=now,
                    updated_at=now,
                ))

        session.commit()

    return {
        "chunks_indexed": len(new_chunks),
        "chunks_skipped": len(chunks) - len(new_chunks),
        "chunks_deleted": len(stale_ids),
    }


def delete_chunks_for_video(video_id: int) -> int:
    """Delete all chunk embeddings for a video. Returns rows deleted."""
    with Session(engine) as session:
        rows = session.exec(
            select(TranscriptChunkEmbedding)
            .where(TranscriptChunkEmbedding.video_id == video_id)
        ).all()
        for row in rows:
            session.delete(row)
        session.commit()
        return len(rows)


def _run_index_job(video_ids: List[int]) -> None:
    """Background thread: index a list of videos in order."""
    with _index_lock:
        _index_state.update({
            "is_running": True,
            "videos_total": len(video_ids),
            "videos_completed": 0,
            "started_at": datetime.now().isoformat(),
            "last_finished_at": None,
        })

    try:
        for i, vid_id in enumerate(video_ids):
            with Session(engine) as session:
                v = session.get(Video, vid_id)
                title = v.title if v else str(vid_id)
            with _index_lock:
                _index_state["current_video_id"] = vid_id
                _index_state["current_video_title"] = title
            try:
                index_video(vid_id)
            except Exception as exc:
                # Log but continue so one bad video doesn't block the rest
                print(f"[semantic_search] Failed to index video {vid_id}: {exc}", flush=True)
            with _index_lock:
                _index_state["videos_completed"] = i + 1
    finally:
        with _index_lock:
            _index_state.update({
                "is_running": False,
                "current_video_id": None,
                "current_video_title": None,
                "last_finished_at": datetime.now().isoformat(),
            })


def start_index_job(video_ids: List[int]) -> bool:
    """Enqueue a background index run. Returns False if already running."""
    with _index_lock:
        if _index_state["is_running"]:
            return False
    t = threading.Thread(target=_run_index_job, args=(video_ids,), daemon=True)
    t.start()
    return True


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _load_chunks(
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
    speaker_id: Optional[int],
    year: Optional[int],
    month: Optional[int],
) -> List[TranscriptChunkEmbedding]:
    """Load chunk rows matching the given filters."""
    stmt = (
        select(TranscriptChunkEmbedding)
        .where(TranscriptChunkEmbedding.embedding_bytes.is_not(None))
    )
    if channel_id is not None:
        stmt = stmt.where(TranscriptChunkEmbedding.channel_id == channel_id)
    if video_id is not None:
        stmt = stmt.where(TranscriptChunkEmbedding.video_id == video_id)
    if speaker_id is not None:
        stmt = stmt.where(TranscriptChunkEmbedding.speaker_id == speaker_id)

    chunks = list(session.exec(stmt).all())

    if (year is not None or month is not None) and chunks:
        video_ids = list({c.video_id for c in chunks})
        videos = session.exec(select(Video).where(Video.id.in_(video_ids))).all()
        valid_ids: set[int] = set()
        for v in videos:
            if not v.published_at:
                continue
            if year is not None and v.published_at.year != year:
                continue
            if month is not None and v.published_at.month != month:
                continue
            valid_ids.add(v.id)
        chunks = [c for c in chunks if c.video_id in valid_ids]

    return chunks


def _enrich_chunk(
    session: Session,
    chunk: TranscriptChunkEmbedding,
    score: float,
    video_cache: dict,
    speaker_cache: dict,
) -> dict:
    if chunk.video_id not in video_cache:
        video_cache[chunk.video_id] = session.get(Video, chunk.video_id)
    video = video_cache[chunk.video_id]

    if chunk.speaker_id and chunk.speaker_id not in speaker_cache:
        speaker_cache[chunk.speaker_id] = session.get(Speaker, chunk.speaker_id)
    speaker = speaker_cache.get(chunk.speaker_id) if chunk.speaker_id else None

    context_before = _context_neighbors(session, chunk, "before")
    context_after = _context_neighbors(session, chunk, "after")

    return {
        "id": chunk.id,
        "score": round(score, 6),
        "video_id": chunk.video_id,
        "video_title": video.title if video else None,
        "channel_id": chunk.channel_id,
        "speaker_id": chunk.speaker_id,
        "speaker_name": speaker.name if speaker else None,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "chunk_text": chunk.chunk_text,
        "segment_ids": json.loads(chunk.segment_ids_json or "[]"),
        "context_before": context_before,
        "context_after": context_after,
    }


def _context_neighbors(
    session: Session,
    chunk: TranscriptChunkEmbedding,
    direction: str,
    window_seconds: float = 60.0,
) -> List[dict]:
    """Return up to 2 neighboring chunks by time range."""
    if direction == "before":
        stmt = (
            select(TranscriptChunkEmbedding)
            .where(TranscriptChunkEmbedding.video_id == chunk.video_id)
            .where(TranscriptChunkEmbedding.end_time <= chunk.start_time)
            .where(TranscriptChunkEmbedding.end_time >= chunk.start_time - window_seconds)
            .order_by(TranscriptChunkEmbedding.end_time.desc())
            .limit(2)
        )
    else:
        stmt = (
            select(TranscriptChunkEmbedding)
            .where(TranscriptChunkEmbedding.video_id == chunk.video_id)
            .where(TranscriptChunkEmbedding.start_time >= chunk.end_time)
            .where(TranscriptChunkEmbedding.start_time <= chunk.end_time + window_seconds)
            .order_by(TranscriptChunkEmbedding.start_time.asc())
            .limit(2)
        )
    neighbors = session.exec(stmt).all()
    return [
        {
            "chunk_id": n.id,
            "speaker_id": n.speaker_id,
            "start_time": n.start_time,
            "end_time": n.end_time,
            "chunk_text": n.chunk_text,
        }
        for n in neighbors
    ]


# ── Search ────────────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    speaker_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Pure vector search ordered by cosine similarity."""
    query_vec = _embed_texts([query])[0]

    with Session(engine) as session:
        chunks = _load_chunks(session, channel_id, video_id, speaker_id, year, month)
        if not chunks:
            return {"items": [], "total": 0, "limit": limit, "offset": offset}

        scored = sorted(
            ((chunk, _cosine(query_vec, _from_bytes(chunk.embedding_bytes))) for chunk in chunks),
            key=lambda x: -x[1],
        )
        total = len(scored)
        page = scored[offset: offset + limit]

        video_cache: dict = {}
        speaker_cache: dict = {}
        items = [_enrich_chunk(session, c, s, video_cache, speaker_cache) for c, s in page]

    return {"items": items, "total": total, "limit": limit, "offset": offset}


def hybrid_search(
    query: str,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    speaker_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Hybrid search: RRF fusion of lexical keyword rank + semantic cosine rank."""
    query_vec = _embed_texts([query])[0]
    q_terms = [t for t in query.lower().split() if len(t) > 2]

    with Session(engine) as session:
        chunks = _load_chunks(session, channel_id, video_id, speaker_id, year, month)
        if not chunks:
            return {"items": [], "total": 0, "limit": limit, "offset": offset}

        chunk_by_id = {c.id: c for c in chunks}

        # Lexical rank (term frequency)
        def lex_score(c: TranscriptChunkEmbedding) -> int:
            if not q_terms:
                return 0
            t = c.chunk_text.lower()
            return sum(t.count(term) for term in q_terms)

        lex_ranked = sorted(chunks, key=lambda c: -lex_score(c))
        lex_rank = {c.id: i + 1 for i, c in enumerate(lex_ranked)}

        # Semantic rank
        sem_ranked = sorted(
            chunks,
            key=lambda c: -_cosine(query_vec, _from_bytes(c.embedding_bytes)),
        )
        sem_rank = {c.id: i + 1 for i, c in enumerate(sem_ranked)}

        # RRF fusion
        rrf: dict[int, float] = {
            cid: 1.0 / (RRF_K + lex_rank[cid]) + 1.0 / (RRF_K + sem_rank[cid])
            for cid in chunk_by_id
        }
        ranked_ids = sorted(rrf, key=lambda cid: -rrf[cid])
        total = len(ranked_ids)
        page_ids = ranked_ids[offset: offset + limit]

        video_cache: dict = {}
        speaker_cache: dict = {}
        items = [
            _enrich_chunk(session, chunk_by_id[cid], rrf[cid], video_cache, speaker_cache)
            for cid in page_ids
        ]

    return {"items": items, "total": total, "limit": limit, "offset": offset}
