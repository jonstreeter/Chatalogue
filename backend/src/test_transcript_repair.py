import json
import os
from pathlib import Path

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_repair_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import Channel, Speaker, TranscriptQualitySnapshot, TranscriptRun, TranscriptSegment, TranscriptSegmentRevision, Video
from src.services import ingestion as ingestion_mod


def test_repair_existing_transcript_creates_run_snapshot_and_backup(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@repair", name="Repair Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="repair-video-1",
            channel_id=channel.id,
            title="Repair Test Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=0.0, end_time=2.4, text="This is a complete sentence."))
        session.add(TranscriptSegment(video_id=video.id, start_time=2.45, end_time=2.7, text="uh"))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=2.72, end_time=5.0, text="This continues the same speaker smoothly."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=5.02, end_time=6.4, text="And this should merge cleanly."))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=6.6, end_time=7.6, text="Closing segment."))
        session.commit()

        result = service.repair_existing_transcript(
            session,
            video.id,
            save_files=False,
            persist_run=True,
            persist_snapshot=True,
            source="test",
            note="repair test",
            trigger_semantic_index=False,
        )

        assert result["changed"] is True
        assert result["after_count"] < result["before_count"]
        assert result["run_id"] is not None
        assert result["snapshot_id"] is not None
        assert result["recommended_tier_before"] == "low_risk_repair"

        remaining = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
        ).all()
        assert len(remaining) == result["after_count"]
        assert remaining
        assert all(seg.speaker_id == speaker.id for seg in remaining)

        run = session.exec(select(TranscriptRun).where(TranscriptRun.id == result["run_id"])).first()
        snapshot = session.exec(select(TranscriptQualitySnapshot).where(TranscriptQualitySnapshot.id == result["snapshot_id"])).first()
        assert run is not None
        assert snapshot is not None
        assert run.mode == "low_risk_repair"

        artifact_refs = json.loads(run.artifact_refs_json or "{}")
        backup_file = Path(str(artifact_refs.get("backup_file") or result["backup_file"]))
        assert backup_file.exists()
        backup_file.unlink()


def test_repair_existing_transcript_applies_entity_repair_and_revision_history(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@mormonstories", name="Mormon Stories")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="John Dehlin", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="repair-video-entity-1",
            channel_id=channel.id,
            title="ConneXions Scandal with John Dehlin",
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
                end_time=3.0,
                text="John DeLin covered the connections scandal.",
                words=json.dumps(
                    [
                        {"start": 0.0, "end": 0.5, "word": "John"},
                        {"start": 0.5, "end": 1.0, "word": "DeLin"},
                        {"start": 1.0, "end": 1.4, "word": "covered"},
                        {"start": 1.4, "end": 1.7, "word": "the"},
                        {"start": 1.7, "end": 2.3, "word": "connections"},
                        {"start": 2.3, "end": 3.0, "word": "scandal"},
                    ],
                    ensure_ascii=False,
                ),
            )
        )
        session.commit()

        result = service.repair_existing_transcript(
            session,
            video.id,
            save_files=False,
            persist_run=True,
            persist_snapshot=True,
            source="test",
            note="entity repair test",
            trigger_semantic_index=False,
        )

        repaired = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id)).first()
        revisions = session.exec(select(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id == video.id)).all()
        run = session.exec(select(TranscriptRun).where(TranscriptRun.id == result["run_id"])).first()

        assert repaired is not None
        assert repaired.text == "John Dehlin covered the ConneXions scandal."
        assert result["entity_replacement_count"] >= 2
        assert result["entity_segments_changed"] == 1
        assert any(rev.source == "entity_repair" for rev in revisions)

        artifact_refs = json.loads(run.artifact_refs_json or "{}") if run else {}
        assert artifact_refs.get("entity_replacement_count", 0) >= 2
        assert "speaker_name" in (artifact_refs.get("entity_sources") or [])


def test_repair_existing_transcript_applies_formatting_cleanup_and_revision_history(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@formatting", name="Formatting Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="repair-video-formatting-1",
            channel_id=channel.id,
            title="Formatting Test Episode",
            processed=True,
            transcript_language="es",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(
            TranscriptSegment(
                video_id=video.id,
                speaker_id=speaker.id,
                start_time=0.0,
                end_time=2.6,
                text="que paso con esto",
            )
        )
        session.add(
            TranscriptSegment(
                video_id=video.id,
                speaker_id=speaker.id,
                start_time=3.2,
                end_time=6.3,
                text="  this is   badly spaced  ",
            )
        )
        session.commit()

        result = service.repair_existing_transcript(
            session,
            video.id,
            save_files=False,
            persist_run=True,
            persist_snapshot=True,
            source="test",
            note="formatting repair test",
            trigger_semantic_index=False,
        )

        repaired = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
        ).all()
        revisions = session.exec(select(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id == video.id)).all()
        run = session.exec(select(TranscriptRun).where(TranscriptRun.id == result["run_id"])).first()

        assert len(repaired) == 2
        assert repaired[0].text == "¿Que paso con esto?"
        assert repaired[1].text == "This is badly spaced."
        assert result["formatting_segments_changed"] == 2
        assert result["formatting_steps"].get("terminal_punctuation", 0) >= 2
        assert any(rev.source == "formatting_cleanup" for rev in revisions)

        artifact_refs = json.loads(run.artifact_refs_json or "{}") if run else {}
        assert artifact_refs.get("formatting_segments_changed", 0) == 2


def test_consolidation_merges_same_speaker_into_turns_with_pause_and_continuation(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    segments = [
        TranscriptSegment(
            video_id=1,
            speaker_id=7,
            start_time=0.0,
            end_time=2.4,
            text="I wanted to make one point.",
        ),
        TranscriptSegment(
            video_id=1,
            speaker_id=7,
            start_time=3.0,
            end_time=5.0,
            text="and then keep developing the same idea.",
        ),
    ]

    result = service._consolidate_transcript_segments(segments)

    assert result["merged_count"] == 1
    assert len(result["segments"]) == 1
    merged = result["segments"][0]
    assert merged.start_time == 0.0
    assert merged.end_time == 5.0
    assert merged.text == "I wanted to make one point. and then keep developing the same idea."


def test_consolidation_preserves_same_speaker_sentence_break_after_meaningful_pause(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    segments = [
        TranscriptSegment(
            video_id=1,
            speaker_id=7,
            start_time=0.0,
            end_time=2.4,
            text="That was the first point.",
        ),
        TranscriptSegment(
            video_id=1,
            speaker_id=7,
            start_time=3.05,
            end_time=5.0,
            text="Here is the second point.",
        ),
    ]

    result = service._consolidate_transcript_segments(segments)

    assert result["merged_count"] == 0
    assert len(result["segments"]) == 2
