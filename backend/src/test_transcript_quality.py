import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_quality_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Channel, Speaker, TranscriptQualitySnapshot, TranscriptRun, TranscriptSegment, Video
from src.services import ingestion as ingestion_mod


def test_transcript_quality_flags_fragmented_diarization_and_persists_snapshot(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@quality", name="Quality Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="quality-video-1",
            channel_id=channel.id,
            title="Quality Test Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=0.0, end_time=2.4, text="This is a complete sentence."))
        session.add(TranscriptSegment(video_id=video.id, start_time=2.45, end_time=2.7, text="uh"))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=2.72, end_time=5.0, text="This continues the same speaker smoothly."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=5.1, end_time=6.6, text="Another segment without punctuation"))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=6.8, end_time=8.1, text="Closing segment."))
        session.commit()

        result = service.evaluate_transcript_quality(session, video.id, source="test", persist_snapshot=True)

        assert result["quality_profile"] == "english_general"
        assert result["recommended_tier"] == "low_risk_repair"
        assert result["eligible_for_optimization"] is True
        assert result["metrics"]["same_speaker_interruptions"] == 1
        assert result["metrics"]["tiny_unknown_count"] == 1
        assert result["created_snapshot_id"] is not None

        snapshot = session.exec(select(TranscriptQualitySnapshot)).first()
        run = session.exec(select(TranscriptRun)).first()
        assert snapshot is not None
        assert run is not None
        assert snapshot.run_id == run.id
        assert snapshot.recommended_tier == "low_risk_repair"
        assert run.mode == "quality_assessment"
