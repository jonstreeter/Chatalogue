import os
from datetime import datetime

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_youtube_metadata_api_bootstrap.db")

from src.services import ingestion as ingestion_mod


def test_fetch_youtube_data_api_video_metadata_batch_parses_core_fields(monkeypatch):
    monkeypatch.setenv("YOUTUBE_DATA_API_KEY", "test-key")
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)

    service = ingestion_mod.IngestionService()

    def fake_request(path: str, *, query=None, timeout: int = 30):
        assert path == "/videos"
        assert query is not None
        assert query["part"] == "snippet,contentDetails,statistics"
        assert query["id"] == "abc123xyz99,def456uvw88"
        return {
            "items": [
                {
                    "id": "abc123xyz99",
                    "snippet": {
                        "title": "Episode One",
                        "description": "Alpha",
                        "publishedAt": "2024-06-01T12:34:56Z",
                        "thumbnails": {
                            "high": {"url": "https://img.example/high.jpg"},
                        },
                    },
                    "contentDetails": {"duration": "PT1H2M3S"},
                    "statistics": {"viewCount": "12345"},
                },
                {
                    "id": "def456uvw88",
                    "snippet": {
                        "title": "Episode Two",
                        "publishedAt": "2024-06-02T00:00:00Z",
                        "thumbnails": {
                            "default": {"url": "https://img.example/default.jpg"},
                        },
                    },
                    "contentDetails": {"duration": "PT45S"},
                    "statistics": {},
                },
            ]
        }

    monkeypatch.setattr(service, "_youtube_data_api_request", fake_request)

    metadata = service._fetch_youtube_data_api_video_metadata_batch(
        ["abc123xyz99", "def456uvw88", "abc123xyz99"]
    )

    assert set(metadata.keys()) == {"abc123xyz99", "def456uvw88"}
    assert metadata["abc123xyz99"]["title"] == "Episode One"
    assert metadata["abc123xyz99"]["description"] == "Alpha"
    assert metadata["abc123xyz99"]["duration"] == 3723
    assert metadata["abc123xyz99"]["view_count"] == 12345
    assert metadata["abc123xyz99"]["thumbnail"] == "https://img.example/high.jpg"
    assert metadata["abc123xyz99"]["published_at"] == datetime(2024, 6, 1, 12, 34, 56)
    assert metadata["def456uvw88"]["duration"] == 45
    assert metadata["def456uvw88"]["view_count"] is None
