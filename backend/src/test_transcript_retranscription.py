import json
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_retranscription_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Channel, Speaker, TranscriptQualitySnapshot, TranscriptRun, TranscriptSegment, Video
from src.services import ingestion as ingestion_mod


def test_create_full_retranscription_run_persists_run_and_snapshot(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@retranscribe", name="Retranscribe Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="retranscribe-video-1",
            channel_id=channel.id,
            title="Retranscribe Test Episode",
            processed=True,
            transcript_language="es",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        for idx in range(5):
            session.add(
                TranscriptSegment(
                    video_id=video.id,
                    speaker_id=speaker.id,
                    start_time=float(idx) * 1.4,
                    end_time=float(idx) * 1.4 + 1.2,
                    text=f"Segment {idx + 1}.",
                )
            )
        session.commit()

        payload = {
            "mode": "full_retranscription",
            "optimization_target": "full_retranscription",
            "quality_profile_before": "multilingual_or_non_english",
            "recommended_tier_before": "full_retranscription",
            "quality_score_before": 52.0,
            "quality_metrics_before": {
                "unknown_speaker_rate": 0.02,
                "word_timed_segment_rate": 0.35,
                "total_segments": 5,
            },
            "quality_reasons_before": ["Language profile suggests multilingual or non-English handling should be re-routed."],
            "redo_diarization_backup_file": "F:/tmp/retranscribe_backup.json",
            "queued_from": "test",
            "transcription_engine_requested": "parakeet",
            "transcription_engine_used": "whisper",
            "force_retranscription": True,
        }

        result = service.create_full_retranscription_run(
            session,
            video.id,
            payload=payload,
            note="retranscribe test",
        )

        assert result is not None
        assert result["run_id"] is not None
        assert result["snapshot_id"] is not None
        assert result["recommended_tier_before"] == "full_retranscription"

        run = session.exec(select(TranscriptRun).where(TranscriptRun.id == result["run_id"])).first()
        snapshot = session.exec(select(TranscriptQualitySnapshot).where(TranscriptQualitySnapshot.id == result["snapshot_id"])).first()
        assert run is not None
        assert snapshot is not None
        assert run.mode == "full_retranscription"

        artifact_refs = json.loads(run.artifact_refs_json or "{}")
        model_provenance = json.loads(run.model_provenance_json or "{}")
        assert artifact_refs["raw_transcript_reused"] is False
        assert artifact_refs["force_retranscription"] is True
        assert model_provenance["transcription_engine_requested"] == "parakeet"
        assert model_provenance["transcription_engine_used"] == "whisper"


def test_backup_path_helper_accepts_full_retranscription_mode(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()
    path = service._get_redo_backup_path_from_payload(
        json.dumps(
            {
                "mode": "full_retranscription",
                "redo_diarization_backup_file": "F:/tmp/retranscribe_backup.json",
            }
        )
    )
    assert path is not None
    assert str(path).replace("\\", "/").endswith("/tmp/retranscribe_backup.json")
