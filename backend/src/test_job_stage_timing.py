import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_job_stage_timing_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from src.db.database import Channel, Job, Video
from src.services import ingestion as ingestion_mod


def test_recording_diarize_stage_closes_transcribe_stage(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    monkeypatch.setattr(ingestion_mod, "engine", engine, raising=False)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@stage-timing", name="Stage Timing")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="stage-timing-video-1",
            channel_id=channel.id,
            title="Stage Timing Episode",
            processed=False,
            status="transcribing",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        job = Job(
            video_id=video.id,
            job_type="process",
            status="transcribing",
            payload_json='{"stage_transcribe_started_at":"2026-04-09T11:48:17.774629"}',
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        service._record_job_stage_start(job.id, "diarize")

        session.refresh(job)
        payload = ingestion_mod.IngestionService._load_job_payload(job.payload_json)

        assert payload.get("stage_diarize_started_at")
        assert payload.get("stage_transcribe_completed_at")
        assert float(payload.get("stage_transcribe_seconds") or 0.0) >= 0.0
