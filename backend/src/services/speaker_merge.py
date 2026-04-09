from __future__ import annotations

from collections.abc import Sequence

from sqlmodel import Session
from sqlalchemy import update, delete

from ..db.database import Avatar, Speaker, SpeakerEmbedding, TranscriptChunkEmbedding, TranscriptSegment


def merge_speakers_in_session(session: Session, *, target_id: int, source_ids: Sequence[int]) -> None:
    normalized_source_ids = [int(sid) for sid in source_ids if int(sid) != int(target_id)]
    if not normalized_source_ids:
        return

    session.exec(
        update(TranscriptSegment)
        .where(TranscriptSegment.speaker_id.in_(normalized_source_ids))
        .values(speaker_id=target_id)
    )
    session.exec(
        update(SpeakerEmbedding)
        .where(SpeakerEmbedding.speaker_id.in_(normalized_source_ids))
        .values(speaker_id=target_id)
    )
    session.exec(
        update(TranscriptChunkEmbedding)
        .where(TranscriptChunkEmbedding.speaker_id.in_(normalized_source_ids))
        .values(speaker_id=target_id)
    )
    session.exec(
        update(Avatar)
        .where(Avatar.speaker_id.in_(normalized_source_ids))
        .values(speaker_id=target_id)
    )
    session.exec(
        delete(Speaker).where(Speaker.id.in_(normalized_source_ids))
    )
