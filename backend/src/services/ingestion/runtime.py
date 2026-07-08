"""Shared runtime configuration for the ingestion service package.

Holds the directory constants, environment helpers, and the rebindable DB
globals (``engine``, ``create_db_and_tables``). Service code must reach the
DB engine via ``runtime.engine`` (call-time attribute lookup) so tests can
monkeypatch it in exactly one place no matter how many modules the service
implementation is split across.
"""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

# Rebindable DB globals — tests monkeypatch these on this module.
from ...db.database import create_db_and_tables, engine  # noqa: F401

# runtime.py lives at backend/src/services/ingestion/ - four parents up is backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANUAL_MEDIA_DIR = DATA_DIR / "manual_media"
TEMP_DIR = DATA_DIR / "temp"
PYTHON_TEMP_DIR = TEMP_DIR / "python_runtime"
EXPORT_DIR = DATA_DIR / "exports"
HEARTBEAT_FILE = DATA_DIR / "worker_heartbeat"
RUNTIME_DIR = BACKEND_DIR / "runtime"
CUDA_RESTART_STATE_FILE = RUNTIME_DIR / "cuda_restart_state.json"
CUDA_MAX_AUTO_RESTARTS = int(os.getenv("CUDA_MAX_AUTO_RESTARTS", "3"))
CUDA_RESTART_WINDOW_SECONDS = int(os.getenv("CUDA_RESTART_WINDOW_SECONDS", "600"))

# Load .env from backend root
load_dotenv(BACKEND_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
YOUTUBE_DATA_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

PROCESS_JOB_TYPES = {"process"}
VOICEFIXER_JOB_TYPES = {"voicefixer_cleanup"}
RECONSTRUCTION_JOB_TYPES = {"conversation_reconstruct"}
DIARIZE_JOB_TYPES = {"diarize"}
FUNNY_JOB_TYPES = {"funny_detect", "funny_explain"}
YOUTUBE_JOB_TYPES = {"youtube_metadata", "episode_clone"}
CLIP_JOB_TYPES = {"clip_export_mp4", "clip_export_captions"}
TRANSCRIPT_REPAIR_JOB_TYPES = {"transcript_repair"}


def _env_float(name: str, default: str) -> float:
    """Parse a float from an environment variable with a fallback default."""
    try:
        return float((os.getenv(name) or default).strip() or default)
    except (ValueError, TypeError):
        return float(default)


def _truncate_error(error: str | None, max_len: int = 4000) -> str:
    """Truncate an error message to a safe DB storage length."""
    return (error or "Unknown error")[:max_len]


def ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    create_db_and_tables()


def configure_python_temp_dir():
    PYTHON_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = str(PYTHON_TEMP_DIR)
    os.environ["TMP"] = temp_path
    os.environ["TEMP"] = temp_path
    os.environ["TMPDIR"] = temp_path
    tempfile.tempdir = temp_path


configure_python_temp_dir()


@contextmanager
def temporary_disabled_blackhole_proxies():
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    removed: dict[str, str] = {}
    try:
        for key in proxy_keys:
            value = str(os.environ.get(key) or "").strip()
            lowered = value.lower()
            if "127.0.0.1:9" in lowered or "localhost:9" in lowered:
                removed[key] = value
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in removed.items():
            os.environ[key] = value
