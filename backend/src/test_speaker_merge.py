import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_speaker_merge_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Avatar, Channel, Speaker, SpeakerEmbedding, TranscriptChunkEmbedding, TranscriptSegment, Video
from src.services.speaker_merge import merge_speakers_in_session


def test_merge_speakers_reassigns_all_foreign_keys():
    engine = create_engine("sqlite://")

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@merge-test", name="Merge Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(youtube_id="merge-test-video", channel_id=channel.id, title="Merge Test Video")
        session.add(video)
        session.commit()
        session.refresh(video)

        target = Speaker(channel_id=channel.id, name="Target", embedding_blob=b"target")
        source = Speaker(channel_id=channel.id, name="Source", embedding_blob=b"source")
        session.add(target)
        session.add(source)
        session.commit()
        session.refresh(target)
        session.refresh(source)

        session.add(SpeakerEmbedding(speaker_id=target.id, embedding_blob=b"target-profile"))
        source_embedding = SpeakerEmbedding(speaker_id=source.id, embedding_blob=b"source-profile")
        session.add(source_embedding)
        session.add(TranscriptSegment(video_id=video.id, speaker_id=source.id, start_time=0.0, end_time=1.0, text="hello"))
        session.add(
            TranscriptChunkEmbedding(
                channel_id=channel.id,
                video_id=video.id,
                speaker_id=source.id,
                start_time=0.0,
                end_time=1.0,
                chunk_text="hello",
                content_hash="chunk-1",
            )
        )
        session.add(Avatar(channel_id=channel.id, speaker_id=source.id, name="Source Avatar"))
        session.commit()

        merge_speakers_in_session(session, target_id=target.id, source_ids=[source.id])
        session.commit()

        assert session.get(Speaker, source.id) is None
        assert session.exec(select(TranscriptSegment).where(TranscriptSegment.speaker_id == target.id)).first() is not None
        assert session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.id == source_embedding.id)).first().speaker_id == target.id
        assert session.exec(select(TranscriptChunkEmbedding).where(TranscriptChunkEmbedding.speaker_id == target.id)).first() is not None
        assert session.exec(select(Avatar).where(Avatar.speaker_id == target.id)).first() is not None
