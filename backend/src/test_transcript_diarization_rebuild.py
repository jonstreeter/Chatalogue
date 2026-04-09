import json
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_diarization_rebuild_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Channel, Speaker, TranscriptQualitySnapshot, TranscriptRun, TranscriptSegment, Video
from src.services import ingestion as ingestion_mod


def test_create_diarization_rebuild_run_persists_run_and_snapshot(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@rebuild", name="Rebuild Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="rebuild-video-1",
            channel_id=channel.id,
            title="Rebuild Test Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=0.0, end_time=1.6, text="Opening sentence."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=1.7, end_time=3.1, text="Second sentence."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=3.2, end_time=4.5, text="Third sentence."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=4.6, end_time=5.9, text="Fourth sentence."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=6.0, end_time=7.2, text="Closing sentence."))
        session.commit()

        payload = {
            "mode": "redo_diarization",
            "optimization_target": "diarization_rebuild",
            "quality_profile_before": "english_longform",
            "recommended_tier_before": "diarization_rebuild",
            "quality_score_before": 61.5,
            "quality_metrics_before": {
                "unknown_speaker_rate": 0.18,
                "word_timed_segment_rate": 0.88,
                "total_segments": 5,
            },
            "quality_reasons_before": ["High unknown-speaker rate with usable timing."],
            "redo_diarization_backup_file": "F:/tmp/rebuild_backup.json",
            "queued_from": "test",
        }

        result = service.create_diarization_rebuild_run(
            session,
            video.id,
            payload=payload,
            note="rebuild test",
        )

        assert result is not None
        assert result["run_id"] is not None
        assert result["snapshot_id"] is not None
        assert result["recommended_tier_before"] == "diarization_rebuild"
        assert result["quality_profile_before"] == "english_longform"

        run = session.exec(select(TranscriptRun).where(TranscriptRun.id == result["run_id"])).first()
        snapshot = session.exec(select(TranscriptQualitySnapshot).where(TranscriptQualitySnapshot.id == result["snapshot_id"])).first()
        assert run is not None
        assert snapshot is not None
        assert run.mode == "diarization_rebuild"

        artifact_refs = json.loads(run.artifact_refs_json or "{}")
        assert artifact_refs["optimization_target"] == "diarization_rebuild"
        assert artifact_refs["redo_backup_file"] == "F:/tmp/rebuild_backup.json"
