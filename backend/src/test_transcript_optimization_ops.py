import json
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcript_optimization_ops_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from src.db.database import (
    Channel,
    Job,
    Speaker,
    TranscriptEvaluationResult,
    TranscriptGoldWindow,
    TranscriptOptimizationCampaign,
    TranscriptOptimizationCampaignItem,
    TranscriptRun,
    TranscriptSegment,
    Video,
)
from src import main as main_mod
from src.services import ingestion as ingestion_mod


def test_restore_transcript_from_run_creates_restore_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()
    monkeypatch.setattr(service, "save_transcript_files", lambda video, session: None, raising=False)
    monkeypatch.setattr(service, "reindex_video_semantic_embeddings", lambda video_id: None, raising=False)

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    backup_path = tmp_path / "restore-backup.json"

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@restore", name="Restore Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        video = Video(
            youtube_id="restore-video-1",
            channel_id=channel.id,
            title="Restore Test Episode",
            processed=True,
            transcript_language="en",
            status="completed",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        session.add(TranscriptSegment(video_id=video.id, speaker_id=speaker.id, start_time=0.0, end_time=1.0, text="Current transcript text."))
        session.commit()

        backup_path.write_text(
            json.dumps(
                {
                    "video_id": int(video.id),
                    "video_status": "completed",
                    "video_processed": True,
                    "segments": [
                        {
                            "speaker_id": speaker.id,
                            "matched_profile_id": None,
                            "start_time": 0.0,
                            "end_time": 1.2,
                            "text": "Restored transcript text.",
                            "words": None,
                        }
                    ],
                    "funny_moments": [],
                }
            ),
            encoding="utf-8",
        )

        source_run = TranscriptRun(
            video_id=video.id,
            mode="diarization_rebuild",
            pipeline_version="diarization-rebuild-v1",
            status="completed",
            rollback_state=str(backup_path),
            metrics_before_json=json.dumps({"total_segments": 1}),
        )
        session.add(source_run)
        session.commit()
        session.refresh(source_run)

        result = service.restore_transcript_from_run(session, video.id, source_run.id, source="test")

        restored_segment = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id)).first()
        restore_run = session.exec(
            select(TranscriptRun).where(TranscriptRun.id == result["restore_run_id"])
        ).first()

        assert restored_segment is not None
        assert restored_segment.text == "Restored transcript text."
        assert restore_run is not None
        assert restore_run.mode == "rollback_restore"
        assert result["restored_from_run_id"] == source_run.id


def test_campaign_creation_and_diarization_benchmark_summary(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@campaign", name="Campaign Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        speaker = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        repair_video = Video(
            youtube_id="campaign-video-1",
            channel_id=channel.id,
            title="Repair Candidate",
            processed=True,
            transcript_language="en",
        )
        benchmark_video = Video(
            youtube_id="campaign-video-2",
            channel_id=channel.id,
            title="Benchmark Candidate",
            processed=True,
            transcript_language="en",
        )
        session.add(repair_video)
        session.add(benchmark_video)
        session.commit()
        session.refresh(repair_video)
        session.refresh(benchmark_video)

        session.add(TranscriptSegment(video_id=repair_video.id, speaker_id=speaker.id, start_time=0.0, end_time=2.4, text="This is a complete sentence."))
        session.add(TranscriptSegment(video_id=repair_video.id, start_time=2.45, end_time=2.7, text="uh"))
        session.add(TranscriptSegment(video_id=repair_video.id, speaker_id=speaker.id, start_time=2.71, end_time=4.8, text="This continues the same speaker."))
        session.add(TranscriptSegment(video_id=repair_video.id, speaker_id=speaker.id, start_time=4.9, end_time=6.2, text="Closing segment."))
        session.add(TranscriptSegment(video_id=repair_video.id, speaker_id=speaker.id, start_time=6.3, end_time=7.5, text="Fifth segment without punctuation"))
        session.add(TranscriptSegment(video_id=benchmark_video.id, speaker_id=speaker.id, start_time=0.0, end_time=1.5, text="One sentence."))
        session.commit()

        campaign = service.create_transcript_optimization_campaign(
            session,
            channel_id=channel.id,
            limit=20,
            tiers=["low_risk_repair"],
            force_non_eligible=False,
            note="campaign test",
        )
        items = session.exec(
            select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign.id)
        ).all()
        assert len(items) == 1
        assert items[0].video_id == repair_video.id
        assert items[0].action_tier == "low_risk_repair"

        window = TranscriptGoldWindow(
            video_id=benchmark_video.id,
            label="Window",
            start_time=0.0,
            end_time=1.5,
            reference_text="One sentence.",
        )
        session.add(window)
        session.commit()
        session.refresh(window)

        run = TranscriptRun(
            video_id=benchmark_video.id,
            mode="diarization_rebuild",
            pipeline_version="diarization-benchmark-v1",
            status="completed",
            model_provenance_json=json.dumps(
                {
                    "diarization_sensitivity": "conservative",
                    "speaker_match_threshold": 0.42,
                }
            ),
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        result = TranscriptEvaluationResult(
            gold_window_id=window.id,
            video_id=benchmark_video.id,
            run_id=run.id,
            source="test",
            candidate_text="One sentence.",
            reference_text="One sentence.",
            wer=0.12,
            cer=0.05,
            unknown_speaker_rate=0.02,
        )
        session.add(result)
        session.commit()

        summaries = service.summarize_diarization_benchmarks(session, channel_id=channel.id)
        assert len(summaries) == 1
        assert summaries[0]["diarization_sensitivity"] == "conservative"
        assert summaries[0]["speaker_match_threshold"] == 0.42
        assert summaries[0]["average_wer"] == 0.12


def test_clear_queue_detaches_campaign_item_job_references():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@queue-clear", name="Queue Clear Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="queue-clear-video-1",
            channel_id=channel.id,
            title="Queue Clear Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        campaign = TranscriptOptimizationCampaign(
            scope="library",
            status="executed",
            limit=10,
            queued_jobs=1,
        )
        session.add(campaign)
        session.commit()
        session.refresh(campaign)

        job = Job(video_id=video.id, job_type="transcript_repair", status="paused")
        session.add(job)
        session.commit()
        session.refresh(job)

        item = TranscriptOptimizationCampaignItem(
            campaign_id=campaign.id,
            video_id=video.id,
            recommended_tier="low_risk_repair",
            action_tier="low_risk_repair",
            quality_score=0.0,
            status="queued",
            job_id=job.id,
        )
        session.add(item)
        session.commit()
        session.refresh(item)

        result = main_mod.clear_queue(session=session)
        refreshed_item = session.exec(
            select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.id == item.id)
        ).first()
        remaining_job = session.exec(select(Job).where(Job.id == job.id)).first()

        assert result["deleted"] == 1
        assert remaining_job is None
        assert refreshed_item is not None
        assert refreshed_item.job_id is None
        assert refreshed_item.status == "cleared"


def test_delete_campaign_removes_items_but_keeps_existing_jobs():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@campaign-delete", name="Campaign Delete Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="campaign-delete-video-1",
            channel_id=channel.id,
            title="Campaign Delete Episode",
            processed=True,
            transcript_language="en",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        campaign = TranscriptOptimizationCampaign(
            scope="library",
            status="queued",
            limit=10,
            queued_jobs=1,
        )
        session.add(campaign)
        session.commit()
        session.refresh(campaign)

        job = Job(video_id=video.id, job_type="transcript_repair", status="queued")
        session.add(job)
        session.commit()
        session.refresh(job)

        item = TranscriptOptimizationCampaignItem(
            campaign_id=campaign.id,
            video_id=video.id,
            recommended_tier="low_risk_repair",
            action_tier="low_risk_repair",
            quality_score=0.0,
            status="queued",
            job_id=job.id,
        )
        session.add(item)
        session.commit()

        result = main_mod.delete_transcript_optimization_campaign(campaign.id, session=session)

        remaining_campaign = session.get(TranscriptOptimizationCampaign, campaign.id)
        remaining_item = session.exec(
            select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign.id)
        ).all()
        remaining_job = session.get(Job, job.id)

        assert result.campaign_id == campaign.id
        assert result.deleted_items == 1
        assert result.detached_job_refs == 1
        assert remaining_campaign is None
        assert remaining_item == []
        assert remaining_job is not None
