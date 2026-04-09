import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_parakeet_reload_guard_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from src.db.database import Channel, Job, Video
from src.services import ingestion as ingestion_mod


def test_load_parakeet_model_aborts_after_excessive_reload_thrash(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("PARAKEET_MAX_RELOADS_DURING_TRANSCRIBE", "2")
    service = ingestion_mod.IngestionService()
    monkeypatch.setattr(
        service,
        "_parakeet_dependencies_available",
        lambda: (_ for _ in ()).throw(AssertionError("reload guard did not trip before dependency load")),
        raising=False,
    )

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    monkeypatch.setattr(ingestion_mod, "engine", engine, raising=False)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@parakeet-guard", name="Parakeet Guard")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="parakeet-guard-video-1",
            channel_id=channel.id,
            title="Parakeet Guard Episode",
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
            payload_json='{"stage_transcribe_started_at":"2026-04-08T14:00:00","parakeet_model_reload_count":2,"cuda_soft_reset_count":5}',
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        try:
            service._load_parakeet_model(job.id)
            assert False, "Expected Parakeet reload guard to raise"
        except RuntimeError as exc:
            assert "reload-thrash" in str(exc)
            assert "soft_resets=5" in str(exc)
