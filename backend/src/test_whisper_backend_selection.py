import importlib.util
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_whisper_backend_selection_bootstrap.db")

from src.services import ingestion as ingestion_mod


def test_resolve_whisper_backend_uses_installed_insanely_fast_whisper_setting(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("WHISPER_BACKEND", "insanely_fast_whisper")

    service = ingestion_mod.IngestionService()
    backend = service._resolve_whisper_backend()

    assert backend["requested"] == "insanely_fast_whisper"
    assert backend["resolved"] == "insanely_fast_whisper"
    assert backend["fallback_used"] is False
    assert backend["available"] is True


def test_resolve_whisper_backend_falls_back_when_transformers_unavailable(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("WHISPER_BACKEND", "insanely_fast_whisper")

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "transformers":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    service = ingestion_mod.IngestionService()
    backend = service._resolve_whisper_backend()

    assert backend["requested"] == "insanely_fast_whisper"
    assert backend["resolved"] == "faster_whisper"
    assert backend["fallback_used"] is True
    assert "falling back to faster_whisper" in str(backend["fallback_reason"]).lower()


def test_transcription_engine_test_reports_whisper_backend_resolution(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("WHISPER_BACKEND", "insanely_fast_whisper")

    service = ingestion_mod.IngestionService()

    def fake_ensure_device():
        service.device = "cuda"

    def fake_load_whisper_model(*args, **kwargs):
        service._whisper_backend = "insanely_fast_whisper"
        service._whisper_compute_type = "float16"
        service._whisper_device = "cuda"

    monkeypatch.setattr(service, "_ensure_device", fake_ensure_device)
    monkeypatch.setattr(service, "_load_whisper_model", fake_load_whisper_model)

    result = service.test_transcription_engine("whisper")

    assert result["status"] == "ok"
    assert result["resolved_engine"] == "whisper"
    assert result["whisper_backend_requested"] == "insanely_fast_whisper"
    assert result["whisper_backend_resolved"] == "insanely_fast_whisper"
    assert result["whisper_backend_available"] is True
    assert result["fallback_used"] is False
