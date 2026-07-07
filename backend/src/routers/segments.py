"""Transcript segment editing endpoints: speaker assignment, text edits, revisions, ad-hoc clip preview."""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..db.database import (
    Speaker,
    TranscriptSegment,
    TranscriptSegmentRead,
    TranscriptSegmentRevision,
    TranscriptSegmentRevisionRead,
)
from ..deps import get_ingestion_service, get_session
from ..schemas import AssignSpeakerRequest, SegmentTextUpdateRequest
from ..services import speaker_queries as spk_q

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.patch("/segments/{segment_id}/assign-speaker")
def assign_segment_speaker(segment_id: int, body: AssignSpeakerRequest, session: Session = Depends(get_session)):
    """Assign or reassign a speaker to a transcript segment."""
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    speaker = session.get(Speaker, body.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    segment.speaker_id = body.speaker_id
    # Manual reassignment overrides diarization profile provenance.
    segment.matched_profile_id = None
    session.add(segment)
    session.commit()
    spk_q._invalidate_speaker_query_caches()
    session.refresh(segment)
    return {"id": segment.id, "speaker_id": body.speaker_id, "speaker_name": speaker.name, "matched_profile_id": None}


@router.patch("/segments/{segment_id}/text", response_model=TranscriptSegmentRead)
def update_segment_text(segment_id: int, body: SegmentTextUpdateRequest, session: Session = Depends(get_session)):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    requested_words: Optional[List[str]] = None
    if body.words is not None:
        requested_words = []
        for raw in body.words:
            token = " ".join(str(raw or "").replace("\n", " ").split()).strip()
            if token:
                requested_words.append(token)

    normalized_text = " ".join((body.text or "").replace("\n", " ").split()).strip()
    if requested_words is not None and len(requested_words) > 0:
        new_text = " ".join(requested_words).strip()
    else:
        new_text = normalized_text

    if not new_text or not new_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")

    old_text = (segment.text or "").strip()
    if old_text == new_text and requested_words is None:
        # Return current row flattened
        speaker_name = None
        if segment.speaker_id:
            sp = session.get(Speaker, segment.speaker_id)
            speaker_name = sp.name if sp else None
        seg_dict = segment.model_dump(exclude={"speaker"})
        return TranscriptSegmentRead(**seg_dict, speaker=speaker_name)

    # If explicit word tokens are provided, preserve existing per-word timestamps when
    # lengths match; otherwise rebuild timings uniformly over the segment window.
    if requested_words is not None and len(requested_words) > 0:
        existing_words = []
        if segment.words:
            try:
                parsed = json.loads(segment.words)
                if isinstance(parsed, list):
                    for row in parsed:
                        s = float(row.get("start"))
                        e = float(row.get("end"))
                        if e > s:
                            existing_words.append({"start": s, "end": e, "word": str(row.get("word") or "").strip()})
            except Exception:
                existing_words = []

        rebuilt_words = []
        if existing_words and len(existing_words) == len(requested_words):
            for i, token in enumerate(requested_words):
                rebuilt_words.append({
                    "start": existing_words[i]["start"],
                    "end": existing_words[i]["end"],
                    "word": token,
                })
        else:
            seg_start = float(segment.start_time)
            seg_end = max(seg_start + 0.05, float(segment.end_time))
            count = max(1, len(requested_words))
            step = (seg_end - seg_start) / count
            for i, token in enumerate(requested_words):
                s = seg_start + (i * step)
                e = seg_start + ((i + 1) * step)
                rebuilt_words.append({
                    "start": round(s, 3),
                    "end": round(max(s + 0.01, e), 3),
                    "word": token,
                })

        segment.words = json.dumps(rebuilt_words, ensure_ascii=False)

    rev = TranscriptSegmentRevision(
        segment_id=segment.id,
        video_id=segment.video_id,
        old_text=segment.text or "",
        new_text=new_text,
        source="manual_edit_words" if requested_words is not None else "manual_edit",
    )
    session.add(rev)
    segment.text = new_text
    session.add(segment)
    session.commit()
    session.refresh(segment)
    speaker_name = None
    if segment.speaker_id:
        sp = session.get(Speaker, segment.speaker_id)
        speaker_name = sp.name if sp else None
    seg_dict = segment.model_dump(exclude={"speaker"})
    return TranscriptSegmentRead(**seg_dict, speaker=speaker_name)

@router.get("/segments/{segment_id}/revisions", response_model=List[TranscriptSegmentRevisionRead])
def get_segment_revisions(segment_id: int, session: Session = Depends(get_session)):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    return session.exec(
        select(TranscriptSegmentRevision)
        .where(TranscriptSegmentRevision.segment_id == segment_id)
        .order_by(TranscriptSegmentRevision.created_at.desc(), TranscriptSegmentRevision.id.desc())
    ).all()

@router.get("/videos/{video_id}/clip")
def get_clip(video_id: int, start: float, end: float, audio_only: bool = False):
    try:
        path = get_ingestion_service().create_clip(video_id, start, end, audio_only=audio_only)
        media_type = "audio/mp4" if audio_only else "video/mp4"
        return FileResponse(path, filename=Path(path).name, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Saved Clips ---

