"""Settings, LLM-provider tests, and Ollama management endpoints."""
import os
import threading
import time
from datetime import datetime
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from dotenv import set_key
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, text
from sqlmodel import Session, select

from ..db.database import Job, get_db_metrics_snapshot
from ..deps import get_ingestion_service, get_session
from ..env_utils import ENV_PATH
from ..job_utils import _job_queue_name
from ..schemas import (
    OllamaPullRequest,
    Settings,
    TranscriptionEngineTestRequest,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py.

    Resolved at call time, so this module never imports main at import
    time (no circular import in any import order). As helpers move into
    services/, replace _main().<name> with direct imports.
    """
    from .. import main

    return main

@router.get("/settings/db-health")
def get_db_health(session: Session = Depends(get_session)):
    def _row_scalar(row) -> Optional[int]:
        if row is None:
            return None
        if isinstance(row, tuple):
            return int(row[0]) if row else None
        try:
            return int(row[0])  # RowMapping-like
        except Exception:
            try:
                return int(row)
            except Exception:
                return None

    db = get_db_metrics_snapshot()
    active_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    queued_like_statuses = ["queued", "paused"] + active_like_statuses

    queue_summary = {
        "pipeline": {"queued": 0, "running": 0, "paused": 0},
        "funny": {"queued": 0, "running": 0, "paused": 0},
        "youtube": {"queued": 0, "running": 0, "paused": 0},
        "clip": {"queued": 0, "running": 0, "paused": 0},
        "other": {"queued": 0, "running": 0, "paused": 0},
    }

    rows = session.exec(
        select(Job.job_type, Job.status, func.count(Job.id))
        .where(Job.status.in_(queued_like_statuses))
        .group_by(Job.job_type, Job.status)
    ).all()
    for row in rows:
        job_type = row[0] if isinstance(row, tuple) else getattr(row, "job_type", "")
        status = row[1] if isinstance(row, tuple) else getattr(row, "status", "")
        count = int(row[2] if isinstance(row, tuple) else getattr(row, "count_1", 0) or 0)
        queue_name = _job_queue_name(job_type)
        if status == "queued":
            queue_summary[queue_name]["queued"] += count
        elif status == "paused":
            queue_summary[queue_name]["paused"] += count
        elif status in active_like_statuses:
            queue_summary[queue_name]["running"] += count

    total_running = sum(v["running"] for v in queue_summary.values())
    total_queued = sum(v["queued"] for v in queue_summary.values())
    total_paused = sum(v["paused"] for v in queue_summary.values())

    connections = {
        "total": None,
        "active": None,
        "max": None,
    }

    if db.get("is_postgres"):
        try:
            total = session.exec(text("SELECT COUNT(*) FROM pg_stat_activity WHERE datname = current_database()")).one()
            active = session.exec(
                text(
                    "SELECT COUNT(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND state = 'active'"
                )
            ).one()
            max_conn = session.exec(text("SHOW max_connections")).one()
            connections = {
                "total": _row_scalar(total),
                "active": _row_scalar(active),
                "max": _row_scalar(max_conn),
            }
        except Exception:
            pass

    return {
        "timestamp": datetime.now().isoformat(),
        "database": {
            "provider": db.get("provider"),
            "database_url": db.get("database_url"),
            "is_postgres": db.get("is_postgres"),
            "pool": db.get("pool"),
            "connections": connections,
            "query_metrics": db.get("query_metrics"),
        },
        "queue_depth": {
            "running": total_running,
            "queued": total_queued,
            "paused": total_paused,
            "total_active": total_running + total_queued + total_paused,
            "by_queue": queue_summary,
        },
    }


@router.get("/settings", response_model=Settings)
def get_settings():
    llm_enabled = os.getenv("LLM_ENABLED")
    if llm_enabled is None:
        llm_enabled = os.getenv("OLLAMA_ENABLED", "false")
    pipeline_execution_mode = (os.getenv("PIPELINE_EXECUTION_MODE") or "sequential").strip().lower()
    if pipeline_execution_mode not in {"sequential", "staged"}:
        pipeline_execution_mode = "sequential"
    return Settings(
        hf_token=os.getenv("HF_TOKEN") or "",
        transcription_engine=(os.getenv("TRANSCRIPTION_ENGINE") or "auto"),
        pipeline_execution_mode=pipeline_execution_mode,
        whisper_backend=(os.getenv("WHISPER_BACKEND") or "faster_whisper"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL") or "medium",
        transcription_compute_type=os.getenv("TRANSCRIPTION_COMPUTE_TYPE") or "int8_float16",
        multilingual_routing_enabled=os.getenv("MULTILINGUAL_ROUTING_ENABLED", "true").lower() == "true",
        multilingual_whisper_model=os.getenv("MULTILINGUAL_WHISPER_MODEL") or "large-v3",
        language_detection_sample_seconds=max(15, min(int(os.getenv("LANGUAGE_DETECTION_SAMPLE_SECONDS", "45")), 180)),
        language_detection_confidence_threshold=max(0.3, min(float(os.getenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.65")), 0.99)),
        parakeet_model=os.getenv("PARAKEET_MODEL") or "nvidia/parakeet-tdt-0.6b-v2",
        parakeet_batch_size=int(os.getenv("PARAKEET_BATCH_SIZE", "16")),
        parakeet_batch_auto=os.getenv("PARAKEET_BATCH_AUTO", "true").lower() == "true",
        parakeet_require_word_timestamps=os.getenv("PARAKEET_REQUIRE_WORD_TIMESTAMPS", "true").lower() == "true",
        parakeet_allow_whisper_fallback=os.getenv("PARAKEET_ALLOW_WHISPER_FALLBACK", "true").lower() == "true",
        parakeet_unload_after_transcribe=os.getenv("PARAKEET_UNLOAD_AFTER_TRANSCRIBE", "false").lower() == "true",
        beam_size=int(os.getenv("TRANSCRIPTION_BEAM_SIZE", "1")),
        vad_filter=os.getenv("TRANSCRIPTION_VAD_FILTER", "true").lower() == "true",
        batched_transcription=os.getenv("TRANSCRIPTION_BATCHED", "true").lower() == "true",
        verbose_logging=os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
        llm_provider=_normalize_llm_provider(os.getenv("LLM_PROVIDER") or "ollama"),
        llm_enabled=str(llm_enabled).lower() == "true",
        ollama_url=os.getenv("OLLAMA_URL") or "http://localhost:11434",
        ollama_model=os.getenv("OLLAMA_MODEL") or "mistral",
        ollama_model_tier=os.getenv("OLLAMA_MODEL_TIER") or "medium",
        ollama_enabled=os.getenv("OLLAMA_ENABLED", "false").lower() == "true",
        nvidia_nim_base_url=os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com",
        nvidia_nim_model=os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5",
        nvidia_nim_api_key=os.getenv("NVIDIA_NIM_API_KEY") or "",
        nvidia_nim_thinking_mode=os.getenv("NVIDIA_NIM_THINKING_MODE", "false").lower() == "true",
        nvidia_nim_min_request_interval_seconds=float(os.getenv("NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS", "2.5")),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com",
        openai_model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY") or "",
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or "",
        gemini_base_url=os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com",
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
        gemini_api_key=os.getenv("GEMINI_API_KEY") or "",
        groq_base_url=os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai",
        groq_model=os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY") or "",
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api",
        openrouter_model=os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or "",
        xai_base_url=os.getenv("XAI_BASE_URL") or "https://api.x.ai",
        xai_model=os.getenv("XAI_MODEL") or "grok-2",
        xai_api_key=os.getenv("XAI_API_KEY") or "",
        youtube_data_api_key=os.getenv("YOUTUBE_DATA_API_KEY") or "",
        youtube_oauth_client_id=os.getenv("YOUTUBE_OAUTH_CLIENT_ID") or "",
        youtube_oauth_client_secret=os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET") or "",
        youtube_oauth_redirect_uri=os.getenv("YOUTUBE_OAUTH_REDIRECT_URI") or "http://localhost:8000/auth/youtube/callback",
        youtube_publish_push_enabled=os.getenv("YOUTUBE_PUBLISH_PUSH_ENABLED", "false").lower() == "true",
        ytdlp_cookies_file=os.getenv("YTDLP_COOKIES_FILE") or "",
        ytdlp_cookies_from_browser=os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "",
        diarization_sensitivity=os.getenv("DIARIZATION_SENSITIVITY") or "balanced",
        speaker_match_threshold=float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.35")),
        diarize_auto_start_threshold=int(os.getenv("DIARIZE_AUTO_START_THRESHOLD", "0")),
        funny_moments_max_saved=int(os.getenv("FUNNY_MOMENTS_MAX_SAVED", "25")),
        funny_moments_explain_batch_limit=int(os.getenv("FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", "12")),
        setup_wizard_completed=os.getenv("SETUP_WIZARD_COMPLETED", "false").lower() == "true",
    )

@router.post("/settings")
def update_settings(settings: Settings, session: Session = Depends(get_session)):
    """Update settings and reload models"""
    normalized_transcription_engine = (getattr(settings, "transcription_engine", "auto") or "auto").strip().lower()
    if normalized_transcription_engine not in {"auto", "whisper", "parakeet"}:
        normalized_transcription_engine = "auto"
    normalized_whisper_backend = (getattr(settings, "whisper_backend", "faster_whisper") or "faster_whisper").strip().lower().replace("-", "_")
    if normalized_whisper_backend not in {"faster_whisper", "insanely_fast_whisper"}:
        normalized_whisper_backend = "faster_whisper"
    pipeline_execution_mode = (getattr(settings, "pipeline_execution_mode", "sequential") or "sequential").strip().lower()
    if pipeline_execution_mode not in {"sequential", "staged"}:
        pipeline_execution_mode = "sequential"
    multilingual_routing_enabled = bool(getattr(settings, "multilingual_routing_enabled", True))
    multilingual_whisper_model = (getattr(settings, "multilingual_whisper_model", "") or "large-v3").strip() or "large-v3"
    language_detection_sample_seconds = max(15, min(int(getattr(settings, "language_detection_sample_seconds", 45)), 180))
    language_detection_confidence_threshold = max(0.30, min(float(getattr(settings, "language_detection_confidence_threshold", 0.65)), 0.99))
    parakeet_model = (getattr(settings, "parakeet_model", "") or "nvidia/parakeet-tdt-0.6b-v2").strip()
    parakeet_batch_size = max(1, min(int(getattr(settings, "parakeet_batch_size", 16)), 64))
    parakeet_batch_auto = bool(getattr(settings, "parakeet_batch_auto", True))
    parakeet_require_word_timestamps = bool(getattr(settings, "parakeet_require_word_timestamps", True))
    parakeet_allow_whisper_fallback = bool(getattr(settings, "parakeet_allow_whisper_fallback", True))
    parakeet_unload_after_transcribe = bool(getattr(settings, "parakeet_unload_after_transcribe", False))
    llm_enabled = bool(getattr(settings, "llm_enabled", False) or settings.ollama_enabled)
    normalized_provider = _normalize_llm_provider(getattr(settings, "llm_provider", "ollama"))
    allowed_providers = {"ollama", "nvidia_nim", "openai", "anthropic", "gemini", "groq", "openrouter", "xai"}
    if normalized_provider not in allowed_providers:
        normalized_provider = "ollama"
    diarize_auto_start_threshold = max(0, int(getattr(settings, "diarize_auto_start_threshold", 0)))
    funny_moments_max_saved = max(1, min(int(getattr(settings, "funny_moments_max_saved", 25)), 200))
    funny_moments_explain_batch_limit = max(1, min(int(getattr(settings, "funny_moments_explain_batch_limit", 12)), 200))
    nvidia_nim_min_interval = max(0.0, min(float(getattr(settings, "nvidia_nim_min_request_interval_seconds", 2.5)), 30.0))
    youtube_redirect_uri = (getattr(settings, "youtube_oauth_redirect_uri", "") or "http://localhost:8000/auth/youtube/callback").strip()
    normalized_ollama_model = _main()._normalize_ollama_model_ref(getattr(settings, "ollama_model", "") or "")

    # 1. Update .env file
    set_key(ENV_PATH, "HF_TOKEN", settings.hf_token)
    set_key(ENV_PATH, "TRANSCRIPTION_ENGINE", normalized_transcription_engine)
    set_key(ENV_PATH, "PIPELINE_EXECUTION_MODE", pipeline_execution_mode)
    set_key(ENV_PATH, "WHISPER_BACKEND", normalized_whisper_backend)
    set_key(ENV_PATH, "TRANSCRIPTION_MODEL", settings.transcription_model)
    set_key(ENV_PATH, "TRANSCRIPTION_COMPUTE_TYPE", settings.transcription_compute_type)
    set_key(ENV_PATH, "MULTILINGUAL_ROUTING_ENABLED", str(multilingual_routing_enabled).lower())
    set_key(ENV_PATH, "MULTILINGUAL_WHISPER_MODEL", multilingual_whisper_model)
    set_key(ENV_PATH, "LANGUAGE_DETECTION_SAMPLE_SECONDS", str(language_detection_sample_seconds))
    set_key(ENV_PATH, "LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", str(language_detection_confidence_threshold))
    set_key(ENV_PATH, "PARAKEET_MODEL", parakeet_model)
    set_key(ENV_PATH, "PARAKEET_BATCH_SIZE", str(parakeet_batch_size))
    set_key(ENV_PATH, "PARAKEET_BATCH_AUTO", str(parakeet_batch_auto).lower())
    set_key(ENV_PATH, "PARAKEET_REQUIRE_WORD_TIMESTAMPS", str(parakeet_require_word_timestamps).lower())
    set_key(ENV_PATH, "PARAKEET_ALLOW_WHISPER_FALLBACK", str(parakeet_allow_whisper_fallback).lower())
    set_key(ENV_PATH, "PARAKEET_UNLOAD_AFTER_TRANSCRIBE", str(parakeet_unload_after_transcribe).lower())
    set_key(ENV_PATH, "TRANSCRIPTION_BEAM_SIZE", str(settings.beam_size))
    set_key(ENV_PATH, "TRANSCRIPTION_VAD_FILTER", str(settings.vad_filter).lower())
    set_key(ENV_PATH, "TRANSCRIPTION_BATCHED", str(settings.batched_transcription).lower())
    set_key(ENV_PATH, "VERBOSE_LOGGING", str(settings.verbose_logging).lower())
    set_key(ENV_PATH, "LLM_PROVIDER", normalized_provider)
    set_key(ENV_PATH, "LLM_ENABLED", str(llm_enabled).lower())
    set_key(ENV_PATH, "OLLAMA_URL", settings.ollama_url)
    set_key(ENV_PATH, "OLLAMA_MODEL", normalized_ollama_model or settings.ollama_model)
    set_key(ENV_PATH, "OLLAMA_MODEL_TIER", (getattr(settings, "ollama_model_tier", "medium") or "medium"))
    set_key(ENV_PATH, "OLLAMA_ENABLED", str(llm_enabled).lower())
    set_key(ENV_PATH, "NVIDIA_NIM_BASE_URL", settings.nvidia_nim_base_url)
    set_key(ENV_PATH, "NVIDIA_NIM_MODEL", settings.nvidia_nim_model)
    set_key(ENV_PATH, "NVIDIA_NIM_API_KEY", settings.nvidia_nim_api_key)
    set_key(ENV_PATH, "NVIDIA_NIM_THINKING_MODE", str(settings.nvidia_nim_thinking_mode).lower())
    set_key(ENV_PATH, "NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS", str(nvidia_nim_min_interval))
    set_key(ENV_PATH, "OPENAI_BASE_URL", settings.openai_base_url)
    set_key(ENV_PATH, "OPENAI_MODEL", settings.openai_model)
    set_key(ENV_PATH, "OPENAI_API_KEY", settings.openai_api_key)
    set_key(ENV_PATH, "ANTHROPIC_BASE_URL", settings.anthropic_base_url)
    set_key(ENV_PATH, "ANTHROPIC_MODEL", settings.anthropic_model)
    set_key(ENV_PATH, "ANTHROPIC_API_KEY", settings.anthropic_api_key)
    set_key(ENV_PATH, "GEMINI_BASE_URL", settings.gemini_base_url)
    set_key(ENV_PATH, "GEMINI_MODEL", settings.gemini_model)
    set_key(ENV_PATH, "GEMINI_API_KEY", settings.gemini_api_key)
    set_key(ENV_PATH, "GROQ_BASE_URL", settings.groq_base_url)
    set_key(ENV_PATH, "GROQ_MODEL", settings.groq_model)
    set_key(ENV_PATH, "GROQ_API_KEY", settings.groq_api_key)
    set_key(ENV_PATH, "OPENROUTER_BASE_URL", settings.openrouter_base_url)
    set_key(ENV_PATH, "OPENROUTER_MODEL", settings.openrouter_model)
    set_key(ENV_PATH, "OPENROUTER_API_KEY", settings.openrouter_api_key)
    set_key(ENV_PATH, "XAI_BASE_URL", settings.xai_base_url)
    set_key(ENV_PATH, "XAI_MODEL", settings.xai_model)
    set_key(ENV_PATH, "XAI_API_KEY", settings.xai_api_key)
    set_key(ENV_PATH, "YOUTUBE_DATA_API_KEY", (getattr(settings, "youtube_data_api_key", "") or "").strip())
    set_key(ENV_PATH, "YOUTUBE_OAUTH_CLIENT_ID", settings.youtube_oauth_client_id)
    set_key(ENV_PATH, "YOUTUBE_OAUTH_CLIENT_SECRET", settings.youtube_oauth_client_secret)
    set_key(ENV_PATH, "YOUTUBE_OAUTH_REDIRECT_URI", youtube_redirect_uri)
    set_key(ENV_PATH, "YOUTUBE_PUBLISH_PUSH_ENABLED", str(bool(getattr(settings, "youtube_publish_push_enabled", False))).lower())
    set_key(ENV_PATH, "YTDLP_COOKIES_FILE", (getattr(settings, "ytdlp_cookies_file", "") or "").strip())
    set_key(ENV_PATH, "YTDLP_COOKIES_FROM_BROWSER", (getattr(settings, "ytdlp_cookies_from_browser", "") or "").strip())
    set_key(ENV_PATH, "DIARIZATION_SENSITIVITY", settings.diarization_sensitivity)
    set_key(ENV_PATH, "SPEAKER_MATCH_THRESHOLD", str(settings.speaker_match_threshold))
    set_key(ENV_PATH, "DIARIZE_AUTO_START_THRESHOLD", str(diarize_auto_start_threshold))
    set_key(ENV_PATH, "FUNNY_MOMENTS_MAX_SAVED", str(funny_moments_max_saved))
    set_key(ENV_PATH, "FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", str(funny_moments_explain_batch_limit))
    set_key(ENV_PATH, "SETUP_WIZARD_COMPLETED", str(bool(getattr(settings, "setup_wizard_completed", False))).lower())

    # 2. Update current environment
    os.environ["HF_TOKEN"] = settings.hf_token
    os.environ["TRANSCRIPTION_ENGINE"] = normalized_transcription_engine
    os.environ["PIPELINE_EXECUTION_MODE"] = pipeline_execution_mode
    os.environ["WHISPER_BACKEND"] = normalized_whisper_backend
    os.environ["TRANSCRIPTION_MODEL"] = settings.transcription_model
    os.environ["TRANSCRIPTION_COMPUTE_TYPE"] = settings.transcription_compute_type
    os.environ["MULTILINGUAL_ROUTING_ENABLED"] = str(multilingual_routing_enabled).lower()
    os.environ["MULTILINGUAL_WHISPER_MODEL"] = multilingual_whisper_model
    os.environ["LANGUAGE_DETECTION_SAMPLE_SECONDS"] = str(language_detection_sample_seconds)
    os.environ["LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD"] = str(language_detection_confidence_threshold)
    os.environ["PARAKEET_MODEL"] = parakeet_model
    os.environ["PARAKEET_BATCH_SIZE"] = str(parakeet_batch_size)
    os.environ["PARAKEET_BATCH_AUTO"] = str(parakeet_batch_auto).lower()
    os.environ["PARAKEET_REQUIRE_WORD_TIMESTAMPS"] = str(parakeet_require_word_timestamps).lower()
    os.environ["PARAKEET_ALLOW_WHISPER_FALLBACK"] = str(parakeet_allow_whisper_fallback).lower()
    os.environ["PARAKEET_UNLOAD_AFTER_TRANSCRIBE"] = str(parakeet_unload_after_transcribe).lower()
    os.environ["TRANSCRIPTION_BEAM_SIZE"] = str(settings.beam_size)
    os.environ["TRANSCRIPTION_VAD_FILTER"] = str(settings.vad_filter).lower()
    os.environ["TRANSCRIPTION_BATCHED"] = str(settings.batched_transcription).lower()
    os.environ["VERBOSE_LOGGING"] = str(settings.verbose_logging).lower()
    os.environ["LLM_PROVIDER"] = normalized_provider
    os.environ["LLM_ENABLED"] = str(llm_enabled).lower()
    os.environ["OLLAMA_URL"] = settings.ollama_url
    os.environ["OLLAMA_MODEL"] = normalized_ollama_model or settings.ollama_model
    os.environ["OLLAMA_MODEL_TIER"] = (getattr(settings, "ollama_model_tier", "medium") or "medium")
    os.environ["OLLAMA_ENABLED"] = str(llm_enabled).lower()
    os.environ["NVIDIA_NIM_BASE_URL"] = settings.nvidia_nim_base_url
    os.environ["NVIDIA_NIM_MODEL"] = settings.nvidia_nim_model
    os.environ["NVIDIA_NIM_API_KEY"] = settings.nvidia_nim_api_key
    os.environ["NVIDIA_NIM_THINKING_MODE"] = str(settings.nvidia_nim_thinking_mode).lower()
    os.environ["NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS"] = str(nvidia_nim_min_interval)
    os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
    os.environ["OPENAI_MODEL"] = settings.openai_model
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    os.environ["ANTHROPIC_BASE_URL"] = settings.anthropic_base_url
    os.environ["ANTHROPIC_MODEL"] = settings.anthropic_model
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    os.environ["GEMINI_BASE_URL"] = settings.gemini_base_url
    os.environ["GEMINI_MODEL"] = settings.gemini_model
    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    os.environ["GROQ_BASE_URL"] = settings.groq_base_url
    os.environ["GROQ_MODEL"] = settings.groq_model
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
    os.environ["OPENROUTER_BASE_URL"] = settings.openrouter_base_url
    os.environ["OPENROUTER_MODEL"] = settings.openrouter_model
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
    os.environ["XAI_BASE_URL"] = settings.xai_base_url
    os.environ["XAI_MODEL"] = settings.xai_model
    os.environ["XAI_API_KEY"] = settings.xai_api_key
    os.environ["YOUTUBE_DATA_API_KEY"] = (getattr(settings, "youtube_data_api_key", "") or "").strip()
    os.environ["YOUTUBE_OAUTH_CLIENT_ID"] = settings.youtube_oauth_client_id
    os.environ["YOUTUBE_OAUTH_CLIENT_SECRET"] = settings.youtube_oauth_client_secret
    os.environ["YOUTUBE_OAUTH_REDIRECT_URI"] = youtube_redirect_uri
    os.environ["YOUTUBE_PUBLISH_PUSH_ENABLED"] = str(bool(getattr(settings, "youtube_publish_push_enabled", False))).lower()
    os.environ["YTDLP_COOKIES_FILE"] = (getattr(settings, "ytdlp_cookies_file", "") or "").strip()
    os.environ["YTDLP_COOKIES_FROM_BROWSER"] = (getattr(settings, "ytdlp_cookies_from_browser", "") or "").strip()
    os.environ["DIARIZATION_SENSITIVITY"] = settings.diarization_sensitivity
    os.environ["SPEAKER_MATCH_THRESHOLD"] = str(settings.speaker_match_threshold)
    os.environ["DIARIZE_AUTO_START_THRESHOLD"] = str(diarize_auto_start_threshold)
    os.environ["FUNNY_MOMENTS_MAX_SAVED"] = str(funny_moments_max_saved)
    os.environ["FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT"] = str(funny_moments_explain_batch_limit)
    os.environ["SETUP_WIZARD_COMPLETED"] = str(bool(getattr(settings, "setup_wizard_completed", False))).lower()
    
    # 3. Reconfigure logging based on new verbose_logging setting
    _main().configure_logging()
    
    # 4. Reload models in ingestion service
    if get_ingestion_service():
        print("Reloading models with new settings...")
        _main()._purge_runtime_models(reason="settings_updated")
        
    return {"status": "updated"}

@router.post("/settings/test-transcription-engine")
def test_transcription_engine(request: TranscriptionEngineTestRequest):
    if not get_ingestion_service():
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    engine = (request.engine or os.getenv("TRANSCRIPTION_ENGINE") or "auto").strip().lower()
    return get_ingestion_service().test_transcription_engine(engine, whisper_backend_override=request.whisper_backend)


@router.post("/settings/validate-token")
def validate_hf_token():
    """
    Validate that the HF_TOKEN can access all required pyannote models.
    Returns detailed status for each model.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        return {
            "valid": False,
            "error": "No token configured",
            "models": {}
        }
    
    # Required models for diarization
    required_models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization-community-1",
        "pyannote/segmentation-3.0", 
        "pyannote/embedding"
    ]
    
    model_status = {}
    all_valid = True
    
    from huggingface_hub import HfApi
    api = HfApi()
    
    for model_id in required_models:
        try:
            # Try to get model info with the token
            api.model_info(model_id, token=token)
            model_status[model_id] = {"accessible": True, "error": None}
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "restricted" in error_msg.lower() or "gated" in error_msg.lower():
                model_status[model_id] = {
                    "accessible": False, 
                    "error": "Access denied - you need to accept the model agreement",
                    "url": f"https://huggingface.co/{model_id}"
                }
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                model_status[model_id] = {
                    "accessible": False, 
                    "error": "Invalid token"
                }
            else:
                model_status[model_id] = {
                    "accessible": False, 
                    "error": error_msg[:100]
                }
            all_valid = False
    
    return {
        "valid": all_valid,
        "token_set": True,
        "models": model_status
    }

@router.post("/settings/test-ollama")
def test_ollama_connection():
    """Test Ollama connectivity, list available models, and verify the selected model works."""
    import httpx
    import time

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = _main()._normalize_ollama_model_ref(os.getenv("OLLAMA_MODEL", "mistral"))

    # 1. Check if Ollama is reachable
    try:
        tags_start = time.perf_counter()
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        tags_latency_ms = int((time.perf_counter() - tags_start) * 1000)
        models_data = r.json()
        available_models = [m["name"] for m in models_data.get("models", [])]
    except httpx.ConnectError:
        return {"status": "error", "error": f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?", "available_models": []}
    except Exception as e:
        return {"status": "error", "error": str(e), "available_models": []}

    # 2. Check if selected model is available
    # Model names from /api/tags include the tag (e.g. "mistral:latest")
    model_found = any(_main()._ollama_model_name_matches(m, ollama_model) for m in available_models)

    if not model_found:
        return {
            "status": "model_not_found",
            "error": f"Model '{ollama_model}' not found. Pull it with: ollama pull {ollama_model}",
            "available_models": available_models,
            "tags_latency_ms": tags_latency_ms,
        }

    # 3. Quick generation test
    try:
        gen_start = time.perf_counter()
        r = httpx.post(f"{ollama_url}/api/generate", json={
            "model": ollama_model,
            "prompt": "Reply with only the word: OK",
            "stream": False,
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {"num_predict": 10}
        }, timeout=30)
        r.raise_for_status()
        generation_latency_ms = int((time.perf_counter() - gen_start) * 1000)
        body = r.json()
        response_text = str(body.get("response") or "").strip()
        if not response_text:
            thinking = str(body.get("thinking") or "").strip()
            if thinking:
                return {
                    "status": "generation_failed",
                    "error": "Model returned thinking-only output (empty final response). Disable reasoning mode for this model.",
                    "model": ollama_model,
                    "available_models": available_models,
                    "latency_ms": generation_latency_ms,
                    "generation_latency_ms": generation_latency_ms,
                    "tags_latency_ms": tags_latency_ms,
                }
            return {
                "status": "generation_failed",
                "error": "Model returned an empty response.",
                "model": ollama_model,
                "available_models": available_models,
                "latency_ms": generation_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "tags_latency_ms": tags_latency_ms,
            }
        return {
            "status": "ok",
            "model": ollama_model,
            "test_response": response_text,
            "available_models": available_models,
            "latency_ms": generation_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "tags_latency_ms": tags_latency_ms,
        }
    except httpx.HTTPStatusError as e:
        response_detail = ""
        try:
            body = e.response.json() if e.response is not None else {}
            if isinstance(body, dict):
                response_detail = str(body.get("error") or "").strip()
        except Exception:
            try:
                response_detail = (e.response.text or "").strip() if e.response is not None else ""
            except Exception:
                response_detail = ""
        detail_suffix = f" | Ollama error: {response_detail}" if response_detail else ""
        return {
            "status": "generation_failed",
            "error": f"Model loaded but generation failed: HTTP {getattr(e.response, 'status_code', 'error')}{detail_suffix}",
            "available_models": available_models,
            "tags_latency_ms": tags_latency_ms,
        }
    except Exception as e:
        return {
            "status": "generation_failed",
            "error": f"Model loaded but generation failed: {e}",
            "available_models": available_models,
            "tags_latency_ms": tags_latency_ms,
        }


@router.get("/settings/ollama/models")
def get_ollama_models(url: Optional[str] = None):
    """List locally downloaded Ollama models from /api/tags."""
    import httpx

    ollama_url = (url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=8)
        r.raise_for_status()
        data = r.json()
    except httpx.ConnectError:
        return {
            "status": "error",
            "error": f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?",
            "models": [],
            "current_model": _main()._normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to query Ollama models: {e}",
            "models": [],
            "current_model": _main()._normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
        }

    raw_models = data.get("models") or []
    models = []
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        details = m.get("details") or {}
        models.append({
            "name": str(m.get("name") or "").strip(),
            "size_bytes": int(m.get("size") or 0),
            "modified_at": m.get("modified_at"),
            "parameter_size": (details.get("parameter_size") if isinstance(details, dict) else None),
            "quantization_level": (details.get("quantization_level") if isinstance(details, dict) else None),
            "families": (details.get("families") if isinstance(details, dict) else None),
        })

    models = [m for m in models if m.get("name")]
    models.sort(key=lambda x: x["name"].lower())
    return {
        "status": "ok",
        "ollama_url": ollama_url,
        "models": models,
        "current_model": _main()._normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
    }


def _normalize_llm_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    aliases = {
        "chatgpt": "openai",
        "claude": "anthropic",
        "google": "gemini",
        "google_gemini": "gemini",
        "google-gemini": "gemini",
        "nvidia": "nvidia_nim",
        "nim": "nvidia_nim",
        "nvidia-nim": "nvidia_nim",
    }
    return aliases.get(p, p)


def _extract_openai_compatible_text(data: dict) -> str:
    try:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", "")).strip()
                for p in content
                if isinstance(p, dict) and p.get("text")
            )
        response_text = str(content or "").strip()
        if not response_text:
            response_text = str(message.get("reasoning_content") or "").strip()
        return response_text
    except Exception:
        return ""


def _test_openai_compatible_provider(
    *,
    provider_label: str,
    base_url: str,
    model: str,
    api_key: str,
    thinking_mode: bool = False,
):
    import httpx
    import time

    if not api_key:
        return {
            "status": "error",
            "error": f"{provider_label} API key is not configured. Add it in Settings first."
        }
    if not model:
        return {
            "status": "error",
            "error": f"{provider_label} model is not configured."
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with only the word: OK"}],
        "stream": False,
        "max_tokens": 16,
        "temperature": 0,
    }
    if thinking_mode:
        payload["chat_template_kwargs"] = {"thinking": True}

    try:
        req_start = time.perf_counter()
        normalized_base = base_url.rstrip("/")
        endpoint = f"{normalized_base}/chat/completions" if normalized_base.lower().endswith("/v1") else f"{normalized_base}/v1/chat/completions"
        r = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=45,
        )
        latency_ms = int((time.perf_counter() - req_start) * 1000)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        latency_ms = int((time.perf_counter() - req_start) * 1000) if 'req_start' in locals() else None
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return {
            "status": "error",
            "error": f"{provider_label} request failed ({e.response.status_code}): {detail[:400]}",
            "latency_ms": latency_ms,
        }
    except httpx.ConnectError:
        return {
            "status": "error",
            "error": f"Cannot connect to {provider_label} at {base_url}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

    response_text = _extract_openai_compatible_text(data)
    return {
        "status": "ok",
        "model": model,
        "test_response": response_text or "(empty)",
        "latency_ms": latency_ms,
    }


def _test_anthropic_provider(base_url: str, model: str, api_key: str):
    import httpx
    import time

    if not api_key:
        return {"status": "error", "error": "Anthropic API key is not configured. Add it in Settings first."}
    if not model:
        return {"status": "error", "error": "Anthropic model is not configured."}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 24,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with only the word: OK"}],
    }
    try:
        req_start = time.perf_counter()
        r = httpx.post(f"{base_url.rstrip('/')}/v1/messages", headers=headers, json=payload, timeout=45)
        latency_ms = int((time.perf_counter() - req_start) * 1000)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        latency_ms = int((time.perf_counter() - req_start) * 1000) if 'req_start' in locals() else None
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return {
            "status": "error",
            "error": f"Anthropic request failed ({e.response.status_code}): {detail[:400]}",
            "latency_ms": latency_ms,
        }
    except httpx.ConnectError:
        return {"status": "error", "error": f"Cannot connect to Anthropic at {base_url}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    try:
        response_text = " ".join(
            str(part.get("text") or "").strip()
            for part in (data.get("content") or [])
            if isinstance(part, dict) and str(part.get("type") or "").lower() == "text"
        ).strip()
    except Exception:
        response_text = ""

    return {
        "status": "ok",
        "model": model,
        "test_response": response_text or "(empty)",
        "latency_ms": latency_ms,
    }


def _test_gemini_provider(base_url: str, model: str, api_key: str):
    import httpx
    import time

    if not api_key:
        return {"status": "error", "error": "Gemini API key is not configured. Add it in Settings first."}
    if not model:
        return {"status": "error", "error": "Gemini model is not configured."}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with only the word: OK"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 24},
    }
    try:
        req_start = time.perf_counter()
        r = httpx.post(
            f"{base_url.rstrip('/')}/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=45,
        )
        latency_ms = int((time.perf_counter() - req_start) * 1000)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        latency_ms = int((time.perf_counter() - req_start) * 1000) if 'req_start' in locals() else None
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return {
            "status": "error",
            "error": f"Gemini request failed ({e.response.status_code}): {detail[:400]}",
            "latency_ms": latency_ms,
        }
    except httpx.ConnectError:
        return {"status": "error", "error": f"Cannot connect to Gemini at {base_url}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    try:
        candidates = data.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
        response_text = " ".join(
            str(p.get("text") or "").strip()
            for p in parts
            if isinstance(p, dict) and p.get("text")
        ).strip()
    except Exception:
        response_text = ""

    return {
        "status": "ok",
        "model": model,
        "test_response": response_text or "(empty)",
        "latency_ms": latency_ms,
    }


def _test_hosted_provider(provider: str):
    p = _normalize_llm_provider(provider)
    if p == "nvidia_nim":
        thinking_mode = os.getenv("NVIDIA_NIM_THINKING_MODE", "false").lower() == "true"
        result = _test_openai_compatible_provider(
            provider_label="NVIDIA NIM",
            base_url=(os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com").rstrip("/"),
            model=(os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip(),
            api_key=(os.getenv("NVIDIA_NIM_API_KEY") or "").strip(),
            thinking_mode=thinking_mode,
        )
        result["provider"] = "nvidia_nim"
        result["thinking_mode"] = thinking_mode
        return result
    if p == "openai":
        result = _test_openai_compatible_provider(
            provider_label="OpenAI",
            base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/"),
            model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
            api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        )
        result["provider"] = "openai"
        return result
    if p == "anthropic":
        result = _test_anthropic_provider(
            base_url=(os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/"),
            model=(os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip(),
            api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
        )
        result["provider"] = "anthropic"
        return result
    if p == "gemini":
        result = _test_gemini_provider(
            base_url=(os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/"),
            model=(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
            api_key=(os.getenv("GEMINI_API_KEY") or "").strip(),
        )
        result["provider"] = "gemini"
        return result
    if p == "groq":
        result = _test_openai_compatible_provider(
            provider_label="Groq",
            base_url=(os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai").rstrip("/"),
            model=(os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
            api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
        )
        result["provider"] = "groq"
        return result
    if p == "openrouter":
        result = _test_openai_compatible_provider(
            provider_label="OpenRouter",
            base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api").rstrip("/"),
            model=(os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
            api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        )
        result["provider"] = "openrouter"
        return result
    if p == "xai":
        result = _test_openai_compatible_provider(
            provider_label="xAI",
            base_url=(os.getenv("XAI_BASE_URL") or "https://api.x.ai").rstrip("/"),
            model=(os.getenv("XAI_MODEL") or "grok-2").strip(),
            api_key=(os.getenv("XAI_API_KEY") or "").strip(),
        )
        result["provider"] = "xai"
        return result
    return {"status": "error", "provider": p or "unknown", "error": f"Unsupported hosted provider '{provider}'"}

@router.post("/settings/test-nvidia-nim")
def test_nvidia_nim_connection():
    """Back-compat endpoint for NVIDIA NIM test."""
    return _test_hosted_provider("nvidia_nim")


@router.post("/settings/test-hosted-llm")
def test_hosted_llm_connection():
    """Test currently selected hosted LLM provider connectivity/model."""
    provider = _normalize_llm_provider(os.getenv("LLM_PROVIDER") or "")
    if provider in {"", "ollama"}:
        return {
            "status": "error",
            "provider": provider or "unknown",
            "error": "Current provider is local Ollama. Use /settings/test-ollama for local connectivity."
        }
    return _test_hosted_provider(provider)

@router.post("/settings/ollama/pull-model")
def pull_ollama_model(req: OllamaPullRequest):
    """Pull the configured Ollama model if it is not already installed.
    Default behavior runs asynchronously to avoid long request timeouts."""
    import httpx

    ollama_url = (req.url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    ollama_model = _main()._normalize_ollama_model_ref((req.model or os.getenv("OLLAMA_MODEL") or "mistral").strip())
    wait_for_completion = bool(getattr(req, "wait_for_completion", False))

    if not ollama_model:
        raise HTTPException(status_code=400, detail="No Ollama model specified")

    try:
        tags_resp = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        tags_resp.raise_for_status()
        models_data = tags_resp.json()
        available_models = [m.get("name", "") for m in models_data.get("models", [])]
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query Ollama models: {e}")

    model_found = any(_main()._ollama_model_name_matches(m, ollama_model) for m in available_models)
    if model_found:
        return {
            "status": "already_installed",
            "model": ollama_model,
            "available_models": available_models,
        }

    if wait_for_completion:
        _main()._run_ollama_pull_job(ollama_url, ollama_model)
        key = _main()._ollama_pull_job_key(ollama_url, ollama_model)
        with _main()._ollama_pull_jobs_lock:
            job = dict(_main()._ollama_pull_jobs.get(key) or {})
        status = str(job.get("status") or "")
        if status == "failed":
            raise HTTPException(status_code=500, detail=f"Failed to pull model '{ollama_model}': {job.get('error') or 'Unknown error'}")
        return {
            "status": "pulled" if status == "completed" else "pull_completed_unverified",
            "model": ollama_model,
            "available_models": job.get("available_models") or available_models,
            "ollama_response": job.get("ollama_response") or {},
        }

    key = _main()._ollama_pull_job_key(ollama_url, ollama_model)
    with _main()._ollama_pull_jobs_lock:
        existing = dict(_main()._ollama_pull_jobs.get(key) or {})
        if existing.get("status") in {"queued", "running"}:
            return {
                "status": "already_running",
                "model": ollama_model,
                "job": existing,
            }
        _main()._ollama_pull_jobs[key] = {
            "status": "queued",
            "started_at": time.time(),
            "updated_at": time.time(),
            "completed_at": None,
            "error": None,
            "available_models": available_models[:2000],
        }

    t = threading.Thread(
        target=_main()._run_ollama_pull_job,
        args=(ollama_url, ollama_model),
        daemon=True,
        name=f"ollama-pull-{int(time.time())}"
    )
    t.start()
    with _main()._ollama_pull_jobs_lock:
        job = dict(_main()._ollama_pull_jobs.get(key) or {})
    return {
        "status": "pulling_started",
        "model": ollama_model,
        "job": job,
    }


@router.get("/settings/ollama/pull-status")
def get_ollama_pull_status(url: Optional[str] = None, model: Optional[str] = None):
    ollama_url = (url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    if not model:
        with _main()._ollama_pull_jobs_lock:
            jobs = [
                {"key": k, **v}
                for k, v in _main()._ollama_pull_jobs.items()
                if k.startswith(f"{ollama_url.lower()}|")
            ]
        jobs.sort(key=lambda j: float(j.get("updated_at") or j.get("started_at") or 0), reverse=True)
        return {"status": "ok", "jobs": jobs[:20]}

    normalized_model = _main()._normalize_ollama_model_ref(model)
    key = _main()._ollama_pull_job_key(ollama_url, normalized_model)
    with _main()._ollama_pull_jobs_lock:
        job = dict(_main()._ollama_pull_jobs.get(key) or {})

    if not job:
        return {
            "status": "not_found",
            "model": normalized_model,
            "job": None,
        }

    job_status = str(job.get("status") or "")
    elapsed = None
    started = float(job.get("started_at") or 0)
    finished = float(job.get("completed_at") or 0)
    now_ts = time.time()
    if started > 0:
        elapsed = max(0.0, (finished if finished > 0 else now_ts) - started)

    return {
        "status": "ok",
        "model": normalized_model,
        "job": job,
        "job_status": job_status,
        "elapsed_seconds": round(float(elapsed), 1) if elapsed is not None else None,
    }


@router.get("/settings/ollama/hardware-recommendation")
def get_ollama_hardware_recommendation(objective: str = "balanced"):
    hardware = _main()._detect_gpu_hardware()
    recommendation = _main()._recommend_ollama_for_hardware(hardware.get("gpu_vram_gb"), objective=objective)
    return {
        "status": "ok",
        "hardware": hardware,
        "recommendation": recommendation,
    }
