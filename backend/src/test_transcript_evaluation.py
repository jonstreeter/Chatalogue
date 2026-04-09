import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_evaluation_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Channel, Speaker, TranscriptEvaluationResult, TranscriptGoldWindow, TranscriptSegment, Video
from src.services import ingestion as ingestion_mod


def test_gold_window_evaluation_scores_current_transcript(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@eval", name="Evaluation Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="John Dehlin", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="eval-video-1",
            channel_id=channel.id,
            title="ConneXions Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(
            TranscriptSegment(
                video_id=video.id,
                speaker_id=speaker.id,
                start_time=0.0,
                end_time=2.0,
                text="John Dehlin covered the ConneXions scandal.",
            )
        )
        session.add(
            TranscriptSegment(
                video_id=video.id,
                speaker_id=None,
                start_time=2.0,
                end_time=4.2,
                text="The story changed quickly.",
            )
        )
        session.commit()

        window = service.upsert_transcript_gold_window(
            session,
            video.id,
            label="proper nouns",
            quality_profile="english_general",
            language="en",
            start_time=0.0,
            end_time=4.5,
            reference_text="John Dehlin covered the ConneXions scandal. The story changed quickly.",
            entities=["John Dehlin", "ConneXions"],
            notes="Check proper nouns and punctuation",
            active=True,
        )

        result = service.evaluate_transcript_gold_windows(session, video.id, source="test")

        assert window.id is not None
        assert result["total_windows"] == 1
        assert result["average_wer"] == 0.0
        assert result["average_cer"] == 0.0
        assert result["average_entity_accuracy"] == 1.0
        assert result["average_unknown_speaker_rate"] == 0.5

        stored_window = session.exec(select(TranscriptGoldWindow)).first()
        stored_result = session.exec(select(TranscriptEvaluationResult)).first()
        assert stored_window is not None
        assert stored_result is not None
        assert stored_result.gold_window_id == stored_window.id
        assert stored_result.total_entity_count == 2
        assert stored_result.matched_entity_count == 2


def test_gold_window_evaluation_handles_transcript_miss(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@eval2", name="Evaluation Test 2")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="eval-video-2",
            channel_id=channel.id,
            title="Spanish Segment",
            processed=True,
            transcript_language="es",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(
            TranscriptSegment(
                video_id=video.id,
                start_time=0.0,
                end_time=2.8,
                text="Que paso eso",
            )
        )
        session.commit()

        service.upsert_transcript_gold_window(
            session,
            video.id,
            label="spanish punctuation",
            quality_profile="multilingual_or_non_english",
            language="es",
            start_time=0.0,
            end_time=3.0,
            reference_text="¿Qué pasó con eso?",
            entities=[],
            notes=None,
            active=True,
        )

        result = service.evaluate_transcript_gold_windows(session, video.id, source="test")

        assert result["total_windows"] == 1
        assert result["average_wer"] > 0.0
        assert result["average_cer"] > 0.0
