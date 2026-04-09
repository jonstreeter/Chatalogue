import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_youtube_download_notices_bootstrap.db")

from src.services import ingestion as ingestion_mod


def test_classify_ytdlp_download_notice_for_upcoming_premiere(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    service = ingestion_mod.IngestionService()

    exc = Exception("ERROR: [youtube] Y9Y5M9lSDzg: Premieres in 4 days")
    notice = service._classify_ytdlp_download_notice(exc)

    assert notice is not None
    assert notice["code"] == "youtube_premiere_scheduled"
    assert notice["video_status"] == "pending"
    assert notice["access_restricted"] is False
