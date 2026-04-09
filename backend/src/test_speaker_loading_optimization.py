import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_speaker_loading_optimization_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from src.db.database import Channel, Speaker, SpeakerEmbedding, TranscriptSegment, Video
from src import main as main_mod


def test_read_speaker_overview_returns_counts_without_stats_query(monkeypatch):
    main_mod._invalidate_speaker_query_caches()
    monkeypatch.setattr(
        main_mod,
        "_query_speaker_count_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stats summary should not be used by speaker overview")),
    )

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@speaker-overview", name="Speaker Overview")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="speaker-overview-video-1",
            channel_id=channel.id,
            title="Speaker Overview Episode",
            processed=True,
            status="completed",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        host = Speaker(channel_id=channel.id, name="Host", embedding_blob=b"host")
        guest = Speaker(channel_id=channel.id, name="Guest", embedding_blob=b"guest", is_extra=True)
        unknown = Speaker(channel_id=channel.id, name="Speaker 7", embedding_blob=b"unknown")
        session.add(host)
        session.add(guest)
        session.add(unknown)
        session.commit()
        session.refresh(host)
        session.refresh(guest)
        session.refresh(unknown)

        session.add(SpeakerEmbedding(speaker_id=host.id, embedding_blob=b"emb-1"))
        session.add(SpeakerEmbedding(speaker_id=host.id, embedding_blob=b"emb-2"))

        session.add(TranscriptSegment(video_id=video.id, speaker_id=host.id, start_time=0.0, end_time=180.0, text="Host long segment"))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=guest.id, start_time=180.0, end_time=210.0, text="Guest short segment"))
        session.add(TranscriptSegment(video_id=video.id, speaker_id=unknown.id, start_time=210.0, end_time=260.0, text="Unknown segment"))
        session.commit()

        overview = main_mod.read_speaker_overview(
            channel_id=channel.id,
            offset=0,
            limit=1,
            session=session,
        )

        assert overview.total == 3
        assert overview.counts.total == 3
        assert overview.counts.identified == 2
        assert overview.counts.unknown == 1
        assert overview.counts.main == 1
        assert overview.counts.extras == 1
        assert len(overview.items) == 1
        assert overview.items[0].name == "Host"
        assert overview.items[0].embedding_count == 2


def test_query_speaker_count_summary_prefers_cached_scope_rows(monkeypatch):
    main_mod._invalidate_speaker_query_caches()
    monkeypatch.setattr(
        main_mod,
        "_build_speaker_scope_list_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached count summary should not rebuild query")),
    )

    cache_key = main_mod._speaker_scope_key(channel_id=5, video_id=None)
    main_mod._set_speaker_scope_cache(
        cache_key,
        [
            {"id": 1, "channel_id": 5, "name": "Host", "is_extra": False, "total_speaking_time": 120.0, "created_at": None},
            {"id": 2, "channel_id": 5, "name": "Guest", "is_extra": True, "total_speaking_time": 40.0, "created_at": None},
            {"id": 3, "channel_id": 5, "name": "Speaker 9", "is_extra": False, "total_speaking_time": 25.0, "created_at": None},
        ],
    )

    summary = main_mod._query_speaker_count_summary(
        session=None,
        channel_id=5,
        video_id=None,
    )

    assert summary == {
        "total": 3,
        "identified": 2,
        "unknown": 1,
        "main": 1,
        "extras": 1,
    }
