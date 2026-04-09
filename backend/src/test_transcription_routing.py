import os
from pathlib import Path

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_transcription_routing_bootstrap.db")

from src.db.database import Video
from src.services import ingestion as ingestion_mod


def test_multilingual_routing_uses_metadata_hints_for_spanish_episode(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("MULTILINGUAL_WHISPER_MODEL", "large-v3")

    service = ingestion_mod.IngestionService()
    video = Video(youtube_id="routing-meta-1", title="Entrevista en Español", description="Conversación completa")

    def fail_probe(*args, **kwargs):
        raise AssertionError("audio probe should not run when metadata already clearly indicates Spanish")

    monkeypatch.setattr(service, "_probe_language_with_whisper", fail_probe)

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"))

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "whisper"
    assert route["language"] == "es"
    assert route["multilingual_route_applied"] is True
    assert route["language_source"] == "video_metadata_text"
    assert route["whisper_model_override"] == "large-v3"


def test_multilingual_routing_uses_audio_probe_when_metadata_is_ambiguous(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("MULTILINGUAL_WHISPER_MODEL", "large-v3")
    monkeypatch.setenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.65")

    service = ingestion_mod.IngestionService()
    video = Video(youtube_id="routing-audio-1", title="Interview 42", description="Open discussion")

    monkeypatch.setattr(
        service,
        "_probe_language_with_whisper",
        lambda *args, **kwargs: {
            "language": "es",
            "confidence": 0.91,
            "source": "audio_probe",
            "reason": "Whisper probe heard Spanish in the opening sample.",
            "model": "large-v3",
        },
    )

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"))

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "whisper"
    assert route["language"] == "es"
    assert route["language_source"] == "audio_probe"
    assert route["multilingual_route_applied"] is True
    assert route["whisper_model_override"] == "large-v3"


def test_english_probe_keeps_default_engine(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("MULTILINGUAL_WHISPER_MODEL", "large-v3")
    monkeypatch.setenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.65")

    service = ingestion_mod.IngestionService()
    video = Video(youtube_id="routing-en-1", title="Interview 43", description="Open discussion")

    monkeypatch.setattr(
        service,
        "_probe_language_with_whisper",
        lambda *args, **kwargs: {
            "language": "en",
            "confidence": 0.95,
            "source": "audio_probe",
            "reason": "Whisper probe heard English in the opening sample.",
            "model": "large-v3",
        },
    )

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"))

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "parakeet"
    assert route["language"] == "en"
    assert route["multilingual_route_applied"] is False
    assert route["whisper_model_override"] is None


def test_sequential_backlog_reroutes_long_cold_parakeet_job_to_whisper(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "auto")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "sequential")
    monkeypatch.setenv("PARAKEET_QUEUE_THROUGHPUT_GUARD", "true")
    monkeypatch.setenv("PARAKEET_QUEUE_BACKLOG_WHISPER_THRESHOLD", "1")
    monkeypatch.setenv("PARAKEET_QUEUE_LONG_AUDIO_WHISPER_THRESHOLD_SECONDS", "1200")

    service = ingestion_mod.IngestionService()
    service.parakeet_model = None
    video = Video(youtube_id="routing-queue-1", title="English longform", description="Open discussion", duration=2360)

    monkeypatch.setattr(
        service,
        "_probe_language_with_whisper",
        lambda *args, **kwargs: {
            "language": "en",
            "confidence": 0.98,
            "source": "audio_probe",
            "reason": "Whisper probe heard English in the opening sample.",
            "model": "large-v3",
        },
    )
    monkeypatch.setattr(service, "_transcription_queue_pressure", lambda current_job_id=None: 12)

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"), job_id=123)

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "whisper"
    assert route["language"] == "en"
    assert route["multilingual_route_applied"] is False
    assert route["operational_route_applied"] is True
    assert "Sequential queue throughput guard" in (route["operational_route_reason"] or "")


def test_sequential_backlog_respects_explicit_parakeet_preference(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "sequential")
    monkeypatch.setenv("PARAKEET_QUEUE_THROUGHPUT_GUARD", "true")
    monkeypatch.setenv("PARAKEET_QUEUE_BACKLOG_WHISPER_THRESHOLD", "1")
    monkeypatch.setenv("PARAKEET_QUEUE_LONG_AUDIO_WHISPER_THRESHOLD_SECONDS", "1200")

    service = ingestion_mod.IngestionService()
    service.parakeet_model = None
    video = Video(youtube_id="routing-queue-3", title="English longform", description="Open discussion", duration=2360)

    monkeypatch.setattr(
        service,
        "_probe_language_with_whisper",
        lambda *args, **kwargs: {
            "language": "en",
            "confidence": 0.98,
            "source": "audio_probe",
            "reason": "Whisper probe heard English in the opening sample.",
            "model": "large-v3",
        },
    )
    monkeypatch.setattr(service, "_transcription_queue_pressure", lambda current_job_id=None: 12)

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"), job_id=123)

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "parakeet"
    assert route["operational_route_applied"] is False


def test_sequential_backlog_keeps_warm_parakeet_for_long_job(monkeypatch):
    monkeypatch.setattr(ingestion_mod, "create_db_and_tables", lambda: None)
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "sequential")
    monkeypatch.setenv("PARAKEET_QUEUE_THROUGHPUT_GUARD", "true")
    monkeypatch.setenv("PARAKEET_QUEUE_BACKLOG_WHISPER_THRESHOLD", "1")
    monkeypatch.setenv("PARAKEET_QUEUE_LONG_AUDIO_WHISPER_THRESHOLD_SECONDS", "1200")

    service = ingestion_mod.IngestionService()
    service.parakeet_model = object()
    video = Video(youtube_id="routing-queue-2", title="English longform", description="Open discussion", duration=2360)

    monkeypatch.setattr(
        service,
        "_probe_language_with_whisper",
        lambda *args, **kwargs: {
            "language": "en",
            "confidence": 0.98,
            "source": "audio_probe",
            "reason": "Whisper probe heard English in the opening sample.",
            "model": "large-v3",
        },
    )
    monkeypatch.setattr(service, "_transcription_queue_pressure", lambda current_job_id=None: 12)

    route = service._resolve_transcription_route(video, audio_path=Path("sample.wav"), job_id=123)

    assert route["requested_engine"] == "parakeet"
    assert route["engine"] == "parakeet"
    assert route["language"] == "en"
    assert route["operational_route_applied"] is False
