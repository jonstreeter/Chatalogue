from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timedelta
from sqlmodel import Session, select
from sqlalchemy import func, text
from pathlib import Path
import shutil
import base64
import threading
import atexit
import os
import json
import pickle
import time
import html
import secrets
import re
import logging
import subprocess
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
from dotenv import set_key, load_dotenv
from .schemas import (
    ChannelOverviewRead, ChannelBatchPublishRequest,
    VideoListItemRead,
    TranscriptSearchPage, TranscriptSearchItemRead,
    AssignSpeakerRequest, SegmentTextUpdateRequest, SplitSegmentProfileRequest,
    ClipCreate, ClipRead, ChannelClipRead, ClipCaptionExportRequest, ClipExportPresetRequest,
    ClipYoutubeUploadRequest, ClipBatchYoutubeUploadRequest,
    SpeakerRead, SpeakerCountsRead, SpeakerEpisodeAppearanceRead, SpeakerVoiceProfileRead,
    MoveSpeakerProfileRequest, SpeakerSample, ExtractThumbnailRequest, MergeRequest,
    JobRead, PipelineFocusRead, PipelineFocusUpdate,
    Settings, OllamaPullRequest, TranscriptionEngineTestRequest,
)
from filelock import FileLock, Timeout as FileLockTimeout

# Load .env before configuring logging
load_dotenv()

# Configure logging based on VERBOSE_LOGGING setting
def configure_logging():
    verbose = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
    
    if not verbose:
        # Silence noisy loggers in clean mode
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        # In verbose mode, show everything
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)

configure_logging()

from .db.database import create_db_and_tables, engine, get_db_metrics_snapshot, IS_POSTGRES, Channel, Video, Speaker, SpeakerEmbedding, TranscriptSegment, TranscriptSegmentRead, TranscriptSegmentRevision, TranscriptSegmentRevisionRead, Job, Clip, ClipExportArtifact, ClipExportArtifactRead, FunnyMoment, FunnyMomentRead, VideoDescriptionRevision, VideoDescriptionRevisionRead
from .services.ingestion import IngestionService

ingestion_service = None
worker_threads: dict[str, threading.Thread] = {}
prefetch_thread = None

# Canonical list of job statuses considered "active" in the pipeline.
PIPELINE_ACTIVE_STATUSES = ["queued", "running", "downloading", "transcribing", "diarizing", "waiting_diarize"]
# Subset without waiting_diarize, for contexts where that status isn't relevant.
PIPELINE_ACTIVE_STATUSES_CORE = ["queued", "running", "downloading", "transcribing", "diarizing"]
youtube_oauth_pending_states: dict[str, float] = {}
backend_instance_lock: FileLock | None = None
BACKEND_RUNTIME_DIR = Path(__file__).parent.parent / "runtime"
BACKEND_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_INSTANCE_LOCK_PATH = BACKEND_RUNTIME_DIR / "backend.instance.lock"
BACKEND_INSTANCE_INFO_PATH = BACKEND_RUNTIME_DIR / "backend.instance.json"

# Ensure image directory exists
IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR = Path(__file__).parent.parent / "data" / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
ENV_PATH = Path(__file__).parent.parent / ".env"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_service, worker_threads, prefetch_thread, backend_instance_lock

    backend_instance_lock = FileLock(str(BACKEND_INSTANCE_LOCK_PATH))
    try:
        backend_instance_lock.acquire(timeout=0)
    except FileLockTimeout:
        detail = f"Another Chatalogue backend instance is already running for this install. Lock: {BACKEND_INSTANCE_LOCK_PATH}"
        try:
            if BACKEND_INSTANCE_INFO_PATH.exists():
                info = json.loads(BACKEND_INSTANCE_INFO_PATH.read_text(encoding="utf-8"))
                detail += f" | holder_pid={info.get('pid')} | holder_python={info.get('python')}"
        except Exception:
            pass
        print(detail)
        raise RuntimeError(detail)

    def _release_backend_lock():
        global backend_instance_lock
        try:
            if backend_instance_lock is not None and getattr(backend_instance_lock, "is_locked", False):
                backend_instance_lock.release()
        except Exception:
            pass
        try:
            if BACKEND_INSTANCE_INFO_PATH.exists():
                BACKEND_INSTANCE_INFO_PATH.unlink()
        except Exception:
            pass

    atexit.register(_release_backend_lock)
    BACKEND_INSTANCE_INFO_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "python": os.sys.executable,
                "started_at": datetime.now().isoformat(),
                "cwd": str(Path.cwd()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        create_db_and_tables()
        ingestion_service = IngestionService()
        try:
            ingestion_service.cleanup_orphaned_active_jobs()
        except Exception as e:
            print(f"Startup orphan-job cleanup failed: {e}")

        # Start queue workers by queue type to allow safe parallelism:
        # - process: download/transcribe stage
        # - diarize: speaker diarization stage (waits for transcribe drain)
        # - funny: laughter detection + humor explanation
        # - youtube: summary/chapter generation
        # - clip: rendering/export jobs
        worker_specs = {
            "process": ingestion_service.process_queue,
            "diarize": ingestion_service.process_diarize_queue,
            "funny": ingestion_service.process_funny_queue,
            "youtube": ingestion_service.process_youtube_queue,
            "clip": ingestion_service.process_clip_queue,
        }
        worker_threads = {}
        for name, target in worker_specs.items():
            t = threading.Thread(target=target, daemon=True, name=f"{name}-queue-worker")
            t.start()
            worker_threads[name] = t
        print(f"Queue workers started: {', '.join(worker_threads.keys())}")

        # Start a lightweight background worker that pre-downloads audio for queued jobs
        # so processing can begin immediately when jobs reach the front of the queue.
        prefetch_thread = threading.Thread(target=ingestion_service.prefetch_queue_audio, daemon=True)
        prefetch_thread.start()
        print("Audio prefetch worker thread started.")

        yield
    finally:
        _release_backend_lock()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")


def get_session():
    with Session(engine) as session:
        yield session


@app.get("/system/cuda-health")
def system_cuda_health():
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service not ready")
    try:
        return ingestion_service.get_cuda_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect CUDA health: {e}")


YOUTUBE_OAUTH_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
YOUTUBE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"


def _set_env_persist(key: str, value: str):
    set_key(ENV_PATH, key, value)
    os.environ[key] = value


def _apply_ytdlp_auth_opts(ydl_opts: dict) -> dict:
    opts = dict(ydl_opts or {})
    cookies_file = (os.getenv("YTDLP_COOKIES_FILE") or "").strip()
    cookies_from_browser = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip()

    if cookies_file:
        path = Path(cookies_file).expanduser()
        if path.exists():
            opts["cookiefile"] = str(path)

    if cookies_from_browser and "cookiefile" not in opts:
        if ":" in cookies_from_browser:
            browser, profile = cookies_from_browser.split(":", 1)
            browser = browser.strip()
            profile = profile.strip() or None
        else:
            browser = cookies_from_browser.strip()
            profile = None
        if browser:
            opts["cookiesfrombrowser"] = (browser, profile, None, None)

    return opts


def _detect_gpu_hardware() -> dict:
    gpu_name = None
    gpu_vram_gb = None
    gpu_vendor = None
    gpu_count = 0
    detection_method = None

    # 1) torch CUDA (most reliable for active compute device)
    try:
        import torch  # lazy import so backend still runs without torch
        if torch.cuda.is_available():
            gpu_count = int(torch.cuda.device_count() or 0)
            best_mem = -1
            best_name = None
            for idx in range(gpu_count):
                props = torch.cuda.get_device_properties(idx)
                mem = int(getattr(props, "total_memory", 0) or 0)
                name = str(torch.cuda.get_device_name(idx) or f"CUDA GPU {idx}")
                if mem > best_mem:
                    best_mem = mem
                    best_name = name
            if best_name and best_mem > 0:
                gpu_name = best_name
                gpu_vram_gb = round(best_mem / (1024 ** 3), 1)
                detection_method = "torch_cuda"
    except Exception:
        pass

    # 2) nvidia-smi fallback
    if not gpu_name:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode == 0:
                lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
                best_mem = -1
                best_name = None
                for ln in lines:
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    try:
                        mem_mb = float(parts[1])
                    except Exception:
                        continue
                    if mem_mb > best_mem:
                        best_mem = mem_mb
                        best_name = name
                if best_name and best_mem > 0:
                    gpu_name = best_name
                    gpu_vram_gb = round(best_mem / 1024.0, 1)
                    gpu_count = len(lines)
                    detection_method = "nvidia_smi"
        except Exception:
            pass

    # 3) Windows video controller fallback
    if not gpu_name and os.name == "nt":
        try:
            ps = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM | ConvertTo-Json -Compress"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                data = json.loads(proc.stdout)
                cards = data if isinstance(data, list) else [data]
                cards = [c for c in cards if isinstance(c, dict)]
                # Prefer likely discrete GPU rows over software adapters
                filtered = [
                    c for c in cards
                    if "microsoft basic" not in str(c.get("Name", "")).lower()
                ] or cards
                best = None
                best_mem = -1
                for card in filtered:
                    try:
                        mem_bytes = float(card.get("AdapterRAM") or 0)
                    except Exception:
                        mem_bytes = 0
                    if mem_bytes > best_mem:
                        best_mem = mem_bytes
                        best = card
                if best:
                    name = str(best.get("Name") or "").strip()
                    if name:
                        gpu_name = name
                        gpu_count = len(filtered)
                        if best_mem > 0:
                            gpu_vram_gb = round(best_mem / (1024 ** 3), 1)
                        detection_method = "win32_video_controller"
        except Exception:
            pass

    if gpu_name:
        low = gpu_name.lower()
        if "nvidia" in low:
            gpu_vendor = "nvidia"
        elif "amd" in low or "radeon" in low:
            gpu_vendor = "amd"
        elif "intel" in low:
            gpu_vendor = "intel"
        else:
            gpu_vendor = "unknown"
    else:
        gpu_vendor = "cpu_only"

    return {
        "gpu_name": gpu_name,
        "gpu_vendor": gpu_vendor,
        "gpu_vram_gb": gpu_vram_gb,
        "gpu_count": gpu_count,
        "detection_method": detection_method or "none",
    }


def _build_ollama_quant_tag(base_model: str, tier: str) -> str:
    base = (base_model or "").strip()
    if not base:
        return ""
    # Keep medium/default as canonical base tag.
    if tier == "medium":
        return base
    if tier == "lite":
        return f"{base}-q4_K_M"
    if tier == "q8":
        return f"{base}-q8_0"
    return base


def _normalize_ollama_model_ref(model_ref: str) -> str:
    """
    Normalize model refs for Ollama pull/generate.
    Supports:
    - direct Ollama tags (unchanged), e.g. qwen3.5:35b-a3b
    - HF refs, e.g. hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M
    - HF URLs, e.g. https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
    """
    raw = (model_ref or "").strip()
    if not raw:
        return ""

    lower_raw = raw.lower()
    if lower_raw.startswith("hf.co/"):
        return raw

    m = re.match(r"^https?://huggingface\.co/([^/\s]+)/([^/\s?#:]+)", raw, flags=re.IGNORECASE)
    if not m:
        return raw

    owner = m.group(1).strip()
    repo = m.group(2).strip()
    normalized = f"hf.co/{owner}/{repo}"

    quant_match = re.search(r"(?::|[?&](?:quant|gguf|q)=)([A-Za-z0-9_]+)", raw, flags=re.IGNORECASE)
    if quant_match:
        quant = quant_match.group(1).strip()
        if quant:
            normalized = f"{normalized}:{quant}"
    return normalized


def _ollama_model_name_matches(local_model_name: str, requested_model_ref: str) -> bool:
    local = (local_model_name or "").strip().lower()
    req = (requested_model_ref or "").strip().lower()
    if not local or not req:
        return False

    if local == req or local.startswith(f"{req}:"):
        return True

    def _sig(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    local_base = re.sub(r":latest$", "", local)
    req_base = re.sub(r":latest$", "", req)
    if local_base == req_base:
        return True

    local_sig = _sig(local_base)
    req_sig = _sig(req_base)
    if local_sig and req_sig and (local_sig == req_sig or local_sig.startswith(req_sig) or req_sig.startswith(local_sig)):
        return True

    # HF/Unsloth refs may be downloaded under normalized Ollama names
    # (e.g. qwen3.5:35b-a3b-q4_k_m or unsloth/Qwen...).
    if req_base.startswith("hf.co/"):
        no_hf = req_base[len("hf.co/"):]
        if ":" in no_hf:
            repo_path, req_quant = no_hf.rsplit(":", 1)
        else:
            repo_path, req_quant = no_hf, ""
        repo_tail = (repo_path.split("/")[-1] if repo_path else "").strip().lower()
        repo_tail_no_gguf = re.sub(r"-gguf$", "", repo_tail)
        req_quant_norm = _sig(req_quant)

        tail_sig = _sig(repo_tail_no_gguf)
        if tail_sig and tail_sig in local_sig:
            if not req_quant_norm or req_quant_norm in local_sig:
                return True

    return False


def _ollama_pull_job_key(ollama_url: str, model_ref: str) -> str:
    return f"{(ollama_url or '').rstrip('/').lower()}|{(model_ref or '').strip().lower()}"


def _set_ollama_pull_job(key: str, patch: dict) -> None:
    with _ollama_pull_jobs_lock:
        job = dict(_ollama_pull_jobs.get(key) or {})
        job.update(patch)
        _ollama_pull_jobs[key] = job


def _run_ollama_pull_job(ollama_url: str, model_ref: str) -> None:
    import httpx

    key = _ollama_pull_job_key(ollama_url, model_ref)
    started_ts = time.time()
    _set_ollama_pull_job(key, {
        "status": "running",
        "started_at": started_ts,
        "updated_at": started_ts,
        "completed_at": None,
        "error": None,
        "ollama_response": None,
        "pull_event_status": "starting",
        "pull_completed": None,
        "pull_total": None,
        "pull_percent": None,
    })

    try:
        pull_data: dict = {}
        with httpx.stream(
            "POST",
            f"{ollama_url.rstrip('/')}/api/pull",
            json={"name": model_ref, "stream": True},
            timeout=7200,
        ) as pull_resp:
            pull_resp.raise_for_status()
            for raw_line in pull_resp.iter_lines():
                line = (raw_line or "").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    err_text = str(event.get("error") or "").strip() or "Ollama pull failed"
                    now_ts = time.time()
                    _set_ollama_pull_job(key, {
                        "status": "failed",
                        "updated_at": now_ts,
                        "completed_at": now_ts,
                        "error": err_text[:1200],
                        "pull_event_status": "failed",
                    })
                    return
                pull_data = event
                status_text = str(event.get("status") or "").strip()
                completed = event.get("completed")
                total = event.get("total")
                percent = None
                try:
                    if completed is not None and total is not None:
                        c = float(completed)
                        t = float(total)
                        if t > 0:
                            percent = round(max(0.0, min(100.0, (c / t) * 100.0)), 1)
                except Exception:
                    percent = None
                _set_ollama_pull_job(key, {
                    "status": "running",
                    "updated_at": time.time(),
                    "pull_event_status": status_text or "downloading",
                    "pull_completed": completed if isinstance(completed, (int, float)) else None,
                    "pull_total": total if isinstance(total, (int, float)) else None,
                    "pull_percent": percent,
                })

        available_models: list[str] = []
        try:
            tags_resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
            tags_resp.raise_for_status()
            available_models = [m.get("name", "") for m in (tags_resp.json().get("models") or [])]
        except Exception:
            available_models = []

        model_found_after = any(_ollama_model_name_matches(m, model_ref) for m in available_models)
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "completed" if model_found_after else "completed_unverified",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": None,
            "ollama_response": pull_data,
            "available_models": available_models[:2000],
            "pull_event_status": "completed",
            "pull_percent": 100.0 if model_found_after else pull_data.get("pull_percent"),
        })
    except httpx.HTTPStatusError as e:
        now_ts = time.time()
        detail = ""
        try:
            body = (e.response.text or "").strip()
            if body:
                detail = f" | response: {body[:800]}"
        except Exception:
            detail = ""
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": f"HTTP {getattr(e.response, 'status_code', 'error')} from Ollama /api/pull{detail}"[:1200],
            "pull_event_status": "failed",
        })
    except httpx.ConnectError:
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": f"Cannot connect to Ollama at {ollama_url}",
            "pull_event_status": "failed",
        })
    except Exception as e:
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": str(e)[:1200],
            "pull_event_status": "failed",
        })


OLLAMA_KNOWN_TAG_SIZES_GB = {
    # Sourced from Ollama library tag pages (2026-02-28).
    "qwen2.5:3b": 1.9,
    "qwen2.5:7b": 4.7,
    "qwen2.5:14b": 9.0,
    "qwen3.5:27b": 17.0,
    "qwen3.5:27b-q4_k_m": 17.0,
    "qwen3.5:35b-a3b": 24.0,
    "qwen3.5:35b-a3b-q4_k_m": 24.0,
}
_ollama_size_cache_lock = threading.Lock()
_ollama_size_cache: dict[str, tuple[float, float]] = {}
_OLLAMA_SIZE_CACHE_TTL_SECONDS = 12 * 60 * 60
_ollama_pull_jobs_lock = threading.Lock()
_ollama_pull_jobs: dict[str, dict] = {}


def _get_ollama_exact_size_gb_for_tag(model_tag: str) -> Optional[float]:
    tag = (model_tag or "").strip().lower()
    if not tag:
        return None

    if tag in OLLAMA_KNOWN_TAG_SIZES_GB:
        return float(OLLAMA_KNOWN_TAG_SIZES_GB[tag])

    now_ts = time.time()
    with _ollama_size_cache_lock:
        cached = _ollama_size_cache.get(tag)
        if cached and (now_ts - cached[1]) < _OLLAMA_SIZE_CACHE_TTL_SECONDS:
            return float(cached[0]) if cached[0] >= 0 else None

    try:
        safe_tag = urllib.parse.quote(tag, safe=":-_.")
        url = f"https://ollama.com/library/{safe_tag}"
        html_text = urllib.request.urlopen(url, timeout=6).read().decode("utf-8", errors="ignore")
        m = re.search(r"·\s*(\d+(?:\.\d+)?)GB\s*·", html_text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)GB", html_text, flags=re.IGNORECASE)
        if m:
            size_gb = round(float(m.group(1)), 1)
            with _ollama_size_cache_lock:
                _ollama_size_cache[tag] = (size_gb, time.time())
            return size_gb
    except Exception:
        pass

    with _ollama_size_cache_lock:
        _ollama_size_cache[tag] = (-1.0, time.time())
    return None


def _ollama_quant_info(tier: str) -> tuple[str, float]:
    t = (tier or "").strip().lower()
    if t == "lite":
        return "Q4_K_M", 4.5
    if t == "q8":
        return "Q8_0", 8.5
    # Ollama default tag (no explicit quant suffix) varies by model release.
    return "Default (varies by model, typically Q5/Q6 class)", 5.5


def _estimate_ollama_model_size_gb(base_model: str, tier: str) -> tuple[Optional[float], str]:
    """
    Heuristic estimate of on-disk GGUF model size for the selected tag.
    This is informational only; actual size depends on exact upstream build/tag.
    """
    base = (base_model or "").strip().lower()
    if not base:
        return None, "unknown"

    # Prefer exact Ollama library tag size when available.
    tag = _build_ollama_quant_tag(base, tier)
    exact = _get_ollama_exact_size_gb_for_tag(tag)
    if exact is None and tier in {"medium", "lite"}:
        # Many models expose default q4-ish tags without explicit suffix.
        exact = _get_ollama_exact_size_gb_for_tag(base)
    if exact is not None:
        return round(float(exact), 1), "ollama_exact"

    # Extract first "<num>b" token from model tag (e.g. qwen2.5:14b, qwen3.5:35b-a3b)
    m = re.search(r"(\d+(?:\.\d+)?)b", base)
    if not m:
        return None, "unknown"
    try:
        params_b = float(m.group(1))
    except Exception:
        return None, "unknown"

    # Approximate effective bits-per-weight for displayed tiers.
    _label, bits = _ollama_quant_info(tier)

    # Convert params+quant to rough GB with format/index overhead factor.
    estimated_gb = params_b * (bits / 8.0) * 1.15
    return round(estimated_gb, 1), "estimated"


def _recommend_ollama_for_hardware(gpu_vram_gb: Optional[float], objective: str = "balanced") -> dict:
    """Recommend a single Ollama model tag based on VRAM and user tradeoff objective."""
    vram = float(gpu_vram_gb) if gpu_vram_gb is not None else None
    normalized = (objective or "balanced").strip().lower()
    if normalized not in {"speed", "balanced", "capability"}:
        normalized = "balanced"

    # Default fallback when GPU VRAM cannot be reliably detected.
    if vram is None:
        if normalized == "speed":
            base, tier = "qwen2.5:3b", "lite"
        elif normalized == "capability":
            base, tier = "qwen2.5:14b", "medium"
        else:
            base, tier = "qwen2.5:7b", "medium"
        reason = (
            "GPU VRAM could not be detected. "
            f'Using "{normalized}" objective fallback.'
        )
    else:
        # Speed-first: prioritize lower latency while staying useful.
        if normalized == "speed":
            if vram >= 36:
                base, tier = "qwen3.5:27b", "lite"
            elif vram >= 24:
                base, tier = "qwen2.5:14b", "lite"
            elif vram >= 12:
                base, tier = "qwen2.5:7b", "lite"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "lite"
            elif vram >= 4:
                base, tier = "qwen2.5:3b", "medium"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Speed objective favors smaller/faster quantized tags."
        # Capability-first: maximize output quality within practical VRAM targets.
        elif normalized == "capability":
            if vram >= 48:
                base, tier = "qwen3.5:35b-a3b", "q8"
            elif vram >= 30:
                base, tier = "qwen3.5:35b-a3b", "medium"
            elif vram >= 20:
                base, tier = "qwen3.5:27b", "medium"
            elif vram >= 12:
                base, tier = "qwen2.5:14b", "q8"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "q8"
            elif vram >= 4:
                base, tier = "qwen2.5:7b", "lite"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Capability objective favors stronger models and higher-quality quants."
        # Balanced: default compromise of latency and output quality.
        else:
            if vram >= 32:
                base, tier = "qwen3.5:35b-a3b", "medium"
            elif vram >= 20:
                base, tier = "qwen3.5:27b", "medium"
            elif vram >= 12:
                base, tier = "qwen2.5:14b", "medium"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "q8"
            elif vram >= 4:
                base, tier = "qwen2.5:7b", "lite"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Balanced objective targets speed/quality stability."

    model_tag = _build_ollama_quant_tag(base, tier)
    estimated_size_gb, size_source = _estimate_ollama_model_size_gb(base, tier)

    # If capability suggested q8 but tag size cannot be resolved from Ollama,
    # prefer a known-available medium tag for this model family.
    if tier == "q8" and size_source != "ollama_exact":
        tier = "medium"
        model_tag = _build_ollama_quant_tag(base, tier)
        estimated_size_gb, size_source = _estimate_ollama_model_size_gb(base, tier)

    quant_level, quant_bits_estimate = _ollama_quant_info(tier)

    # Explain why capability may still land on medium/default quant.
    if normalized == "capability" and tier != "q8" and vram is not None:
        q8_est, _q8_source = _estimate_ollama_model_size_gb(base, "q8")
        if q8_est is not None and q8_est > (vram * 0.9):
            reason = (
                f"{reason} "
                f"Q8 for {base} is estimated around {q8_est:.1f} GB, so this recommendation keeps a lower quant for fit/stability."
            )

    return {
        "objective": normalized,
        "base_model": base,
        "tier": tier,
        "model_tag": model_tag,
        "estimated_size_gb": estimated_size_gb,
        "size_source": size_source,
        "quant_level": quant_level,
        "quant_bits_estimate": round(float(quant_bits_estimate), 1),
        "fallback_tag": base,
        "reason": reason,
    }


def _youtube_get_cfg() -> dict:
    return {
        "client_id": (os.getenv("YOUTUBE_OAUTH_CLIENT_ID") or "").strip(),
        "client_secret": (os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET") or "").strip(),
        "redirect_uri": (os.getenv("YOUTUBE_OAUTH_REDIRECT_URI") or "http://localhost:8000/auth/youtube/callback").strip(),
        "access_token": (os.getenv("YOUTUBE_OAUTH_ACCESS_TOKEN") or "").strip(),
        "refresh_token": (os.getenv("YOUTUBE_OAUTH_REFRESH_TOKEN") or "").strip(),
        "token_expiry": (os.getenv("YOUTUBE_OAUTH_TOKEN_EXPIRY") or "").strip(),
        "channel_id": (os.getenv("YOUTUBE_OAUTH_CHANNEL_ID") or "").strip(),
        "channel_title": (os.getenv("YOUTUBE_OAUTH_CHANNEL_TITLE") or "").strip(),
        "push_enabled": (os.getenv("YOUTUBE_PUBLISH_PUSH_ENABLED", "false").lower() == "true"),
    }


def _youtube_oauth_is_configured() -> bool:
    cfg = _youtube_get_cfg()
    return bool(cfg["client_id"] and cfg["client_secret"] and cfg["redirect_uri"])


def _youtube_parse_expiry(expiry_text: str) -> Optional[datetime]:
    if not expiry_text:
        return None
    try:
        return datetime.fromisoformat(expiry_text)
    except Exception:
        return None


def _youtube_http_json(
    method: str,
    url: str,
    *,
    headers: Optional[dict] = None,
    body: Optional[dict] = None,
    form: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    req_headers = dict(headers or {})
    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data_bytes = urllib.parse.urlencode(form).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        try:
            detail_raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail_raw = str(e)
        try:
            detail_json = json.loads(detail_raw)
        except Exception:
            detail_json = {"raw": detail_raw}
        raise RuntimeError(f"HTTP {e.code}: {detail_json}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e


def _youtube_exchange_code_for_tokens(code: str) -> dict:
    cfg = _youtube_get_cfg()
    if not _youtube_oauth_is_configured():
        raise RuntimeError("YouTube OAuth client credentials are not configured in Settings.")
    return _youtube_http_json(
        "POST",
        YOUTUBE_OAUTH_TOKEN_URL,
        form={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": cfg["redirect_uri"],
        },
        timeout=30,
    )


def _youtube_refresh_access_token() -> str:
    cfg = _youtube_get_cfg()
    if not cfg["refresh_token"]:
        raise RuntimeError("No YouTube refresh token is stored. Connect your YouTube account in Settings first.")
    if not _youtube_oauth_is_configured():
        raise RuntimeError("YouTube OAuth client credentials are not configured in Settings.")

    token_data = _youtube_http_json(
        "POST",
        YOUTUBE_OAUTH_TOKEN_URL,
        form={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "refresh_token": cfg["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    access_token = str(token_data.get("access_token") or "").strip()
    if not access_token:
        raise RuntimeError("YouTube token refresh did not return an access token.")

    expires_in = int(token_data.get("expires_in") or 3600)
    expiry = datetime.now() + timedelta(seconds=max(60, expires_in - 30))
    _set_env_persist("YOUTUBE_OAUTH_ACCESS_TOKEN", access_token)
    _set_env_persist("YOUTUBE_OAUTH_TOKEN_EXPIRY", expiry.isoformat())
    return access_token


def _youtube_get_valid_access_token() -> str:
    cfg = _youtube_get_cfg()
    token = cfg["access_token"]
    expiry = _youtube_parse_expiry(cfg["token_expiry"])
    if token and expiry and expiry > datetime.now() + timedelta(seconds=30):
        return token
    return _youtube_refresh_access_token()


def _youtube_api_request(
    method: str,
    path: str,
    *,
    query: Optional[dict] = None,
    body: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    access_token = _youtube_get_valid_access_token()
    qs = urllib.parse.urlencode({k: v for k, v in (query or {}).items() if v is not None}, doseq=True)
    url = f"{YOUTUBE_API_BASE_URL}{path}"
    if qs:
        url = f"{url}?{qs}"
    return _youtube_http_json(
        method,
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        body=body,
        timeout=timeout,
    )


def _youtube_fetch_authenticated_channel_info() -> dict:
    data = _youtube_api_request("GET", "/channels", query={"part": "id,snippet", "mine": "true", "maxResults": 1})
    items = data.get("items") or []
    if not items:
        raise RuntimeError("Authenticated account does not appear to have a YouTube channel.")
    item = items[0]
    return {
        "channel_id": str(item.get("id") or "").strip(),
        "channel_title": str(((item.get("snippet") or {}).get("title")) or "").strip(),
        "raw": item,
    }


def _youtube_update_video_description_remote(youtube_video_id: str, new_description: str) -> dict:
    # Fetch current snippet so we preserve required/mutable fields on update.
    listed = _youtube_api_request("GET", "/videos", query={"part": "snippet", "id": youtube_video_id, "maxResults": 1})
    items = listed.get("items") or []
    if not items:
        raise RuntimeError(f"YouTube video {youtube_video_id} not found or not accessible.")
    item = items[0]
    snippet = item.get("snippet") or {}

    title = str(snippet.get("title") or "").strip()
    category_id = str(snippet.get("categoryId") or "").strip()
    if not title or not category_id:
        raise RuntimeError("YouTube video snippet is missing required title/categoryId for update.")

    update_snippet = {
        "title": title,
        "categoryId": category_id,
        "description": new_description,
    }
    for key in ("tags", "defaultLanguage", "defaultAudioLanguage"):
        if key in snippet and snippet.get(key) is not None:
            update_snippet[key] = snippet.get(key)

    updated = _youtube_api_request(
        "PUT",
        "/videos",
        query={"part": "snippet"},
        body={"id": youtube_video_id, "snippet": update_snippet},
        timeout=45,
    )
    return updated


def _format_seconds_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _youtube_upload_video_resumable(file_path: Path, snippet: dict, status: dict) -> dict:
    if not file_path.exists():
        raise RuntimeError(f"Clip export file not found: {file_path}")

    access_token = _youtube_get_valid_access_token()
    metadata = {"snippet": snippet, "status": status}
    init_url = f"{YOUTUBE_API_BASE_URL.replace('/youtube/v3', '/upload/youtube/v3')}/videos?uploadType=resumable&part=snippet,status"
    file_size = file_path.stat().st_size

    init_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size),
    }
    init_req = urllib.request.Request(
        init_url,
        data=json.dumps(metadata).encode("utf-8"),
        headers=init_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(init_req, timeout=45) as resp:
            upload_url = resp.headers.get("Location")
    except urllib.error.HTTPError as e:
        detail_raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"YouTube resumable upload init failed ({e.code}): {detail_raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"YouTube resumable upload init network error: {e}") from e

    if not upload_url:
        raise RuntimeError("YouTube resumable upload init did not return an upload location URL.")

    upload_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "video/mp4",
        "Content-Length": str(file_size),
    }

    try:
        payload = file_path.read_bytes()
        upload_req = urllib.request.Request(
            upload_url,
            data=payload,
            headers=upload_headers,
            method="PUT",
        )
        with urllib.request.urlopen(upload_req, timeout=1800) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail_raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"YouTube upload failed ({e.code}): {detail_raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"YouTube upload network error: {e}") from e


def _build_default_clip_upload_description(video: Video, clip: Clip) -> str:
    source_url = ""
    if video.youtube_id:
        source_url = f"https://www.youtube.com/watch?v={video.youtube_id}&t={max(0, int(clip.start_time))}s"
    lines = [
        f"Clip from: {video.title}",
        f"Clip range: {_format_seconds_hms(clip.start_time)} - {_format_seconds_hms(clip.end_time)}",
    ]
    if source_url:
        lines.append(f"Original episode: {source_url}")
    lines.extend(["", "Generated with Chatalogue"])
    return "\n".join(lines).strip()


def _extract_channel_identity_from_url(channel_url: str) -> dict:
    url = (channel_url or "").strip()
    lower = url.lower()
    out = {"url": url, "channel_id": None, "handle": None, "kind": None}

    m = re.search(r"/channel/(UC[\w-]{10,})", url, flags=re.IGNORECASE)
    if m:
        out.update({"channel_id": m.group(1), "kind": "channel_id"})
        return out

    m = re.search(r"/@([A-Za-z0-9._-]+)", url)
    if m:
        out.update({"handle": m.group(1), "kind": "handle"})
        return out

    if "/user/" in lower:
        out["kind"] = "legacy_user_url"
    elif "/c/" in lower:
        out["kind"] = "custom_url"
    else:
        out["kind"] = "unknown"
    return out


def _resolve_public_channel_identity_for_publish(channel: Channel) -> dict:
    """Resolve app channel -> YouTube channel identity (best effort).

    Returns a dict with fields:
      resolved(bool), method, channel_id, channel_title, handle, kind, error
    """
    parsed = _extract_channel_identity_from_url(channel.url or "")
    result = {
        "resolved": False,
        "method": None,
        "channel_id": None,
        "channel_title": None,
        "handle": parsed.get("handle"),
        "kind": parsed.get("kind"),
        "error": None,
    }

    try:
        if parsed.get("channel_id"):
            data = _youtube_api_request(
                "GET",
                "/channels",
                query={"part": "id,snippet", "id": parsed["channel_id"], "maxResults": 1},
            )
            items = data.get("items") or []
            if items:
                item = items[0]
                result.update({
                    "resolved": True,
                    "method": "youtube_api_channel_id",
                    "channel_id": str(item.get("id") or "").strip(),
                    "channel_title": str(((item.get("snippet") or {}).get("title")) or "").strip() or None,
                })
                return result
        elif parsed.get("handle"):
            handle = str(parsed["handle"]).lstrip("@")
            for candidate in (handle, f"@{handle}"):
                data = _youtube_api_request(
                    "GET",
                    "/channels",
                    query={"part": "id,snippet", "forHandle": candidate, "maxResults": 1},
                )
                items = data.get("items") or []
                if items:
                    item = items[0]
                    result.update({
                        "resolved": True,
                        "method": "youtube_api_handle",
                        "channel_id": str(item.get("id") or "").strip(),
                        "channel_title": str(((item.get("snippet") or {}).get("title")) or "").strip() or None,
                    })
                    return result
    except Exception as e:
        # Keep going to yt-dlp fallback (for /c/ or /user/ URLs etc.)
        result["error"] = str(e)

    # Fallback: use yt-dlp metadata resolution
    try:
        import yt_dlp
        opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "playlistend": 1,
        }
        opts = _apply_ytdlp_auth_opts(opts)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel.url, download=False)
        if info:
            channel_id = (
                info.get("channel_id")
                or info.get("uploader_id")
                or None
            )
            channel_title = info.get("channel") or info.get("uploader") or None
            handle = parsed.get("handle")
            if not handle:
                channel_url = str(info.get("channel_url") or info.get("uploader_url") or "")
                hm = re.search(r"/@([A-Za-z0-9._-]+)", channel_url)
                if hm:
                    handle = hm.group(1)
            if channel_id or channel_title:
                result.update({
                    "resolved": True,
                    "method": "yt_dlp",
                    "channel_id": str(channel_id).strip() if channel_id else None,
                    "channel_title": str(channel_title).strip() if channel_title else None,
                    "handle": handle,
                })
                return result
    except Exception as e:
        result["error"] = str(e)

    return result


def _youtube_channel_ownership_check_for_app_channel(channel: Channel) -> dict:
    cfg = _youtube_get_cfg()
    connected_channel_id = (cfg.get("channel_id") or "").strip()
    connected_channel_title = (cfg.get("channel_title") or "").strip()
    if not (cfg.get("access_token") or cfg.get("refresh_token")):
        return {
            "status": "not_connected",
            "can_publish_to_channel": False,
            "connected_channel_id": None,
            "connected_channel_title": None,
            "resolved_channel": None,
            "reason": "YouTube OAuth is not connected.",
        }

    resolved = _resolve_public_channel_identity_for_publish(channel)
    if not resolved.get("resolved"):
        return {
            "status": "unknown",
            "can_publish_to_channel": False,
            "connected_channel_id": connected_channel_id or None,
            "connected_channel_title": connected_channel_title or None,
            "resolved_channel": resolved,
            "reason": "Could not reliably resolve the app channel to a YouTube channel ID for ownership comparison.",
        }

    target_channel_id = (resolved.get("channel_id") or "").strip()
    owned = bool(connected_channel_id and target_channel_id and connected_channel_id == target_channel_id)
    return {
        "status": "owned" if owned else "not_owned",
        "can_publish_to_channel": owned,
        "connected_channel_id": connected_channel_id or None,
        "connected_channel_title": connected_channel_title or None,
        "resolved_channel": resolved,
        "reason": None if owned else "Connected YouTube OAuth channel does not match this app channel.",
    }

# --- Channels ---

@app.post("/channels", response_model=Channel)
def create_channel(url: str, background_tasks: BackgroundTasks):
    try:
        channel = ingestion_service.add_channel(url)
        # Auto-refresh on add
        background_tasks.add_task(ingestion_service.refresh_channel, channel.id)
        return channel
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/channels", response_model=List[Channel])
def read_channels(session: Session = Depends(get_session)):
    return session.exec(select(Channel)).all()

@app.get("/channels/overview", response_model=List[ChannelOverviewRead])
def read_channels_overview(session: Session = Depends(get_session)):
    from sqlalchemy import case

    video_stats = (
        select(
            Video.channel_id.label("channel_id"),
            func.count(Video.id).label("video_count"),
            func.sum(case((Video.processed.is_(True), 1), else_=0)).label("processed_count"),
            func.sum(func.coalesce(Video.duration, 0)).label("total_duration_seconds"),
        )
        .group_by(Video.channel_id)
        .subquery()
    )
    speaker_stats = (
        select(
            Speaker.channel_id.label("channel_id"),
            func.count(Speaker.id).label("speaker_count"),
        )
        .group_by(Speaker.channel_id)
        .subquery()
    )

    rows = session.exec(
        select(
            Channel,
            func.coalesce(video_stats.c.video_count, 0).label("video_count"),
            func.coalesce(video_stats.c.processed_count, 0).label("processed_count"),
            func.coalesce(video_stats.c.total_duration_seconds, 0).label("total_duration_seconds"),
            func.coalesce(speaker_stats.c.speaker_count, 0).label("speaker_count"),
        )
        .outerjoin(video_stats, video_stats.c.channel_id == Channel.id)
        .outerjoin(speaker_stats, speaker_stats.c.channel_id == Channel.id)
        .order_by(Channel.id.asc())
    ).all()

    return [
        ChannelOverviewRead(
            id=channel.id,
            url=channel.url,
            name=channel.name,
            icon_url=channel.icon_url,
            header_image_url=channel.header_image_url,
            last_updated=channel.last_updated,
            status=channel.status,
            video_count=int(video_count or 0),
            processed_count=int(processed_count or 0),
            total_duration_seconds=int(total_duration_seconds or 0),
            speaker_count=int(speaker_count or 0),
        )
        for channel, video_count, processed_count, total_duration_seconds, speaker_count in rows
    ]

@app.get("/channels/{channel_id}", response_model=Channel)
def read_channel(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel

@app.post("/channels/{channel_id}/refresh")
def refresh_channel(channel_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingestion_service.refresh_channel, channel_id)
    return {"status": "refresh_started"}

@app.get("/channels/{channel_id}/youtube-publish-ownership-check")
def check_channel_youtube_publish_ownership(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    try:
        check = _youtube_channel_ownership_check_for_app_channel(channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ownership check failed: {e}")
    return {
        "channel_id": channel.id,
        "channel_name": channel.name,
        "channel_url": channel.url,
        **check,
    }

@app.post("/channels/{channel_id}/add-video")
def add_video_to_channel(channel_id: int, url: str, session: Session = Depends(get_session)):
    """Manually add a video to a channel by YouTube URL.
    Useful when a new upload hasn't been picked up by the channel refresh yet."""
    import re
    from datetime import datetime
    
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Extract video ID from URL
    match = re.search(r'(?:v=|youtu\.be/|/v/)([a-zA-Z0-9_-]{11})', url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    yt_id = match.group(1)
    
    # Check if already exists
    existing = session.exec(select(Video).where(Video.youtube_id == yt_id)).first()
    if existing:
        return existing
    
    # Fetch metadata
    try:
        import yt_dlp
        ydl_opts = {'quiet': True, 'no_warnings': True}
        ydl_opts = _apply_ytdlp_auth_opts(ydl_opts)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={yt_id}", download=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video info: {e}")
    
    pub_date = None
    upload_date_str = info.get('upload_date')
    if upload_date_str:
        try:
            pub_date = datetime.strptime(upload_date_str, "%Y%m%d")
        except ValueError:
            pass
    if not pub_date and info.get('release_timestamp'):
        pub_date = datetime.fromtimestamp(info['release_timestamp'])
    
    thumbnails = info.get('thumbnails', [])
    thumbnail_url = None
    if thumbnails:
        for t in reversed(thumbnails):
            if t.get('url'):
                thumbnail_url = t['url']
                break
    
    video = Video(
        youtube_id=yt_id,
        channel_id=channel.id,
        title=info.get('title', 'Unknown Title'),
        description=info.get('description'),
        published_at=pub_date,
        duration=info.get('duration'),
        thumbnail_url=thumbnail_url,
        status="pending"
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video

@app.get("/channels/{channel_id}/stats")
def get_channel_stats(channel_id: int, session: Session = Depends(get_session)):
    """Get statistics for a channel"""
    from sqlalchemy import func
    
    video_count = session.exec(select(func.count(Video.id)).where(Video.channel_id == channel_id)).one()
    processed_count = session.exec(select(func.count(Video.id)).where(Video.channel_id == channel_id, Video.processed == True)).one()
    speaker_count = session.exec(select(func.count(Speaker.id)).where(Speaker.channel_id == channel_id)).one()
    transcript_count = session.exec(select(func.count(TranscriptSegment.id)).join(Video).where(Video.channel_id == channel_id)).one()
    
    return {
        "video_count": video_count,
        "processed_count": processed_count,
        "speaker_count": speaker_count,
        "transcript_count": transcript_count
    }


@app.get("/channels/{channel_id}/delete-preview")
def delete_channel_preview(channel_id: int, session: Session = Depends(get_session)):
    """Preview what will be deleted when a channel is removed."""
    from sqlalchemy import func

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    video_ids = [v.id for v in session.exec(select(Video).where(Video.channel_id == channel_id)).all()]
    video_count = len(video_ids)
    segment_count = 0
    job_count = 0
    clip_count = 0
    active_jobs = 0
    if video_ids:
        segment_count = session.exec(select(func.count(TranscriptSegment.id)).where(TranscriptSegment.video_id.in_(video_ids))).one()
        job_count = session.exec(select(func.count(Job.id)).where(Job.video_id.in_(video_ids))).one()
        clip_count = session.exec(select(func.count(Clip.id)).where(Clip.video_id.in_(video_ids))).one()
        active_jobs = session.exec(select(func.count(Job.id)).where(
            Job.video_id.in_(video_ids),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )).one()
    speaker_count = session.exec(select(func.count(Speaker.id)).where(Speaker.channel_id == channel_id)).one()

    return {
        "channel_name": channel.name,
        "video_count": video_count,
        "segment_count": segment_count,
        "speaker_count": speaker_count,
        "job_count": job_count,
        "clip_count": clip_count,
        "active_jobs": active_jobs,
    }


@app.delete("/channels/{channel_id}")
def delete_channel(channel_id: int, session: Session = Depends(get_session)):
    """Delete a channel and ALL associated data. Irreversible."""
    from sqlalchemy import delete as sa_delete, or_
    from sqlalchemy.exc import IntegrityError

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Gather related IDs
    videos = session.exec(select(Video).where(Video.channel_id == channel_id)).all()
    video_ids = [v.id for v in videos]
    speakers = session.exec(select(Speaker).where(Speaker.channel_id == channel_id)).all()
    speaker_ids = [s.id for s in speakers]

    # Block if active jobs
    if video_ids:
        active = session.exec(select(Job).where(
            Job.video_id.in_(video_ids),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )).first()
        if active:
            raise HTTPException(status_code=400, detail=f"Cannot delete channel with active job (job {active.id}, status: {active.status}). Cancel active jobs first.")

    # Cascade delete in FK-safe order.
    deleted = {
        "segment_revisions": 0,
        "segments": 0,
        "funny_moments": 0,
        "clip_exports": 0,
        "clips": 0,
        "jobs": 0,
        "description_revisions": 0,
        "embeddings": 0,
        "speakers": 0,
        "videos": 0,
    }

    try:
        if video_ids:
            # Delete revisions first to avoid FK failures.
            res = session.exec(sa_delete(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id.in_(video_ids)))
            deleted["segment_revisions"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(FunnyMoment).where(FunnyMoment.video_id.in_(video_ids)))
            deleted["funny_moments"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(ClipExportArtifact).where(ClipExportArtifact.video_id.in_(video_ids)))
            deleted["clip_exports"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(Clip).where(Clip.video_id.in_(video_ids)))
            deleted["clips"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(Job).where(Job.video_id.in_(video_ids)))
            deleted["jobs"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(VideoDescriptionRevision).where(VideoDescriptionRevision.video_id.in_(video_ids)))
            deleted["description_revisions"] = int(res.rowcount or 0)

            res = session.exec(sa_delete(TranscriptSegment).where(TranscriptSegment.video_id.in_(video_ids)))
            deleted["segments"] = int(res.rowcount or 0)

        if speaker_ids or video_ids:
            emb_conditions = []
            if speaker_ids:
                emb_conditions.append(SpeakerEmbedding.speaker_id.in_(speaker_ids))
            if video_ids:
                emb_conditions.append(SpeakerEmbedding.source_video_id.in_(video_ids))
            if emb_conditions:
                res = session.exec(sa_delete(SpeakerEmbedding).where(or_(*emb_conditions)))
                deleted["embeddings"] = int(res.rowcount or 0)

        if speaker_ids:
            res = session.exec(sa_delete(Speaker).where(Speaker.id.in_(speaker_ids)))
            deleted["speakers"] = int(res.rowcount or 0)

        if video_ids:
            res = session.exec(sa_delete(Video).where(Video.id.in_(video_ids)))
            deleted["videos"] = int(res.rowcount or 0)

        session.delete(channel)
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Delete failed due to dependent records: {e.orig}") from e

    # File cleanup (best-effort, don't fail if files are missing)
    try:
        safe_channel = ingestion_service.sanitize_filename(channel.name)
        audio_dir = Path(__file__).parent.parent / "data" / "audio" / safe_channel
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
    except Exception as e:
        print(f"Warning: failed to clean audio dir: {e}")

    # Clean temp files for each video
    temp_dir = Path(__file__).parent.parent / "data" / "temp"
    for vid in video_ids:
        for pattern in [f"transcript_{vid}_partial.json", f"diarization_{vid}.rttm"]:
            p = temp_dir / pattern
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Clean speaker thumbnails
    images_dir = Path(__file__).parent.parent / "data" / "images"
    for sid in speaker_ids:
        for f in images_dir.glob(f"speaker_{sid}_*"):
            try:
                f.unlink()
            except Exception:
                pass
    thumbs_dir = Path(__file__).parent.parent / "data" / "thumbnails" / "speakers"
    if thumbs_dir.exists():
        for sid in speaker_ids:
            for f in thumbs_dir.glob(f"speaker_{sid}_*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    return {"status": "deleted", "channel": channel.name, "deleted": deleted}


@app.get("/channels/{channel_id}/export")
def export_channel(
    channel_id: int,
    compact: bool = True,
    session: Session = Depends(get_session),
):
    """Export a channel archive as JSON with transcripts and speaker profiles."""
    import json as _json
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    backend_data_dir = Path(__file__).parent.parent / "data"

    def _load_speaker_thumbnail_from_path(thumbnail_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not thumbnail_path:
            return None, None
        p = thumbnail_path.strip()
        if not p.startswith("/"):
            return None, None
        if not (p.startswith("/images/") or p.startswith("/thumbnails/speakers/")):
            return None, None
        local = backend_data_dir / p.lstrip("/")
        try:
            if not local.exists() or not local.is_file():
                return None, None
            return base64.b64encode(local.read_bytes()).decode("ascii"), (local.suffix.lower() or None)
        except Exception:
            return None, None

    # Gather speakers with embeddings
    speakers = session.exec(select(Speaker).where(Speaker.channel_id == channel_id)).all()
    speakers_data = []
    for sp in speakers:
        sp_embeddings = session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.speaker_id == sp.id)).all()
        emb_data = []
        for emb in sp_embeddings:
            source_yt_id = None
            if emb.source_video_id:
                source_vid = session.get(Video, emb.source_video_id)
                if source_vid:
                    source_yt_id = source_vid.youtube_id
            emb_item = {
                "embedding_blob_b64": base64.b64encode(emb.embedding_blob).decode("ascii"),
                "source_video_youtube_id": source_yt_id,
                "sample_start_time": emb.sample_start_time,
                "sample_end_time": emb.sample_end_time,
                "sample_text": emb.sample_text,
            }
            if compact:
                emb_item.pop("sample_text", None)
            emb_data.append(emb_item)
        thumb_b64, thumb_ext = _load_speaker_thumbnail_from_path(sp.thumbnail_path)
        speakers_data.append({
            "name": sp.name,
            "embedding_blob_b64": base64.b64encode(sp.embedding_blob).decode("ascii"),
            "is_extra": sp.is_extra,
            "thumbnail_path": sp.thumbnail_path,
            "thumbnail_b64": thumb_b64,
            "thumbnail_ext": thumb_ext,
            "embeddings": emb_data,
        })

    # Gather videos with segments
    videos = session.exec(select(Video).where(Video.channel_id == channel_id)).all()
    videos_data = []
    for v in videos:
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == v.id).order_by(TranscriptSegment.start_time)
        ).all()
        seg_data = []
        for seg in segments:
            speaker_name = None
            if seg.speaker_id:
                sp = session.get(Speaker, seg.speaker_id)
                if sp:
                    speaker_name = sp.name
            seg_item = {
                "speaker_name": speaker_name,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "words": seg.words,
            }
            if compact:
                seg_item.pop("words", None)
            seg_data.append(seg_item)
        video_item = {
            "youtube_id": v.youtube_id,
            "title": v.title,
            "published_at": v.published_at.isoformat() if v.published_at else None,
            "description": v.description,
            "thumbnail_url": v.thumbnail_url,
            "duration": v.duration,
            "muted": v.muted,
            "segments": seg_data,
        }
        if compact:
            video_item.pop("description", None)
            video_item.pop("thumbnail_url", None)
        videos_data.append(video_item)

    archive = {
        "format_version": 2 if compact else 1,
        "exported_at": datetime.now().isoformat(),
        "compact": bool(compact),
        "channel": {
            "url": channel.url,
            "name": channel.name,
            "icon_url": getattr(channel, "icon_url", None),
            "header_image_url": getattr(channel, "header_image_url", None),
        },
        "speakers": speakers_data,
        "videos": videos_data,
    }

    from fastapi.responses import JSONResponse
    headers = {"Content-Disposition": f'attachment; filename="{ingestion_service.sanitize_filename(channel.name)}_archive.json"'}
    return JSONResponse(content=archive, headers=headers)


@app.post("/channels/import")
def import_channel(archive: dict, session: Session = Depends(get_session)):
    """Import a channel from an archive JSON. Restores speakers, transcripts, and video metadata."""
    if archive.get("format_version") not in {1, 2}:
        raise HTTPException(status_code=400, detail="Unsupported archive format version")

    ch_data = archive.get("channel", {})
    if not ch_data.get("url") or not ch_data.get("name"):
        raise HTTPException(status_code=400, detail="Archive missing channel url/name")

    # Check if channel already exists
    existing = session.exec(select(Channel).where(Channel.url == ch_data["url"])).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Channel '{existing.name}' already exists (id={existing.id}). Delete it first or use a different URL.")

    # Create channel
    channel = Channel(
        url=ch_data["url"],
        name=ch_data["name"],
        icon_url=ch_data.get("icon_url"),
        header_image_url=ch_data.get("header_image_url"),
        status="active",
        last_updated=datetime.now(),
    )
    session.add(channel)
    session.commit()
    session.refresh(channel)

    thumb_dir = Path(__file__).parent.parent / "data" / "thumbnails" / "speakers"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    def _restore_speaker_thumbnail(speaker_id: int, sp_data: dict) -> Optional[str]:
        thumb_b64 = sp_data.get("thumbnail_b64")
        if not thumb_b64:
            return None
        ext = (sp_data.get("thumbnail_ext") or ".jpg").strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        filename = f"speaker_{speaker_id}_import_{int(time.time() * 1000)}{ext}"
        out = thumb_dir / filename
        try:
            out.write_bytes(base64.b64decode(thumb_b64))
            return f"/thumbnails/speakers/{filename}"
        except Exception:
            return None

    # Create speakers and build name→id map
    speaker_map = {}  # name → speaker_id
    for sp_data in archive.get("speakers", []):
        blob = base64.b64decode(sp_data["embedding_blob_b64"])
        speaker = Speaker(
            channel_id=channel.id,
            name=sp_data["name"],
            embedding_blob=blob,
            thumbnail_path=None,
            is_extra=sp_data.get("is_extra", False),
        )
        session.add(speaker)
        session.commit()
        session.refresh(speaker)
        restored_thumbnail_path = _restore_speaker_thumbnail(speaker.id, sp_data)
        if restored_thumbnail_path:
            speaker.thumbnail_path = restored_thumbnail_path
            session.add(speaker)
            session.commit()
            session.refresh(speaker)
        speaker_map[sp_data["name"]] = speaker.id

        # Create speaker embeddings (deferred source_video_id linking)
        for emb_data in sp_data.get("embeddings", []):
            emb_blob = base64.b64decode(emb_data["embedding_blob_b64"])
            emb = SpeakerEmbedding(
                speaker_id=speaker.id,
                embedding_blob=emb_blob,
                source_video_id=None,  # Will link after videos are created
                sample_start_time=emb_data.get("sample_start_time"),
                sample_end_time=emb_data.get("sample_end_time"),
                sample_text=emb_data.get("sample_text"),
            )
            session.add(emb)
            # Store youtube_id for later linking
            emb._yt_id = emb_data.get("source_video_youtube_id")
        session.commit()

    # Create videos and segments
    yt_to_video_id = {}  # youtube_id → video.id
    imported = {"videos": 0, "segments": 0, "speakers": len(speaker_map)}
    for v_data in archive.get("videos", []):
        pub_at = None
        if v_data.get("published_at"):
            try:
                pub_at = datetime.fromisoformat(v_data["published_at"])
            except (ValueError, TypeError):
                pass

        video = Video(
            youtube_id=v_data["youtube_id"],
            channel_id=channel.id,
            title=v_data["title"],
            published_at=pub_at,
            description=v_data.get("description"),
            thumbnail_url=v_data.get("thumbnail_url"),
            duration=v_data.get("duration"),
            muted=v_data.get("muted", False),
            status="completed" if v_data.get("segments") else "pending",
            processed=bool(v_data.get("segments")),
        )
        session.add(video)
        session.commit()
        session.refresh(video)
        yt_to_video_id[v_data["youtube_id"]] = video.id
        imported["videos"] += 1

        for seg_data in v_data.get("segments", []):
            sp_id = speaker_map.get(seg_data.get("speaker_name"))
            segment = TranscriptSegment(
                video_id=video.id,
                speaker_id=sp_id,
                start_time=seg_data["start_time"],
                end_time=seg_data["end_time"],
                text=seg_data["text"],
                words=seg_data.get("words"),
            )
            session.add(segment)
            imported["segments"] += 1
        session.commit()

    # Link speaker embeddings to video IDs now that videos exist
    for sp_data in archive.get("speakers", []):
        sp_id = speaker_map.get(sp_data["name"])
        if not sp_id:
            continue
        embeddings = session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.speaker_id == sp_id)).all()
        emb_idx = 0
        for emb_data in sp_data.get("embeddings", []):
            yt_id = emb_data.get("source_video_youtube_id")
            if yt_id and yt_id in yt_to_video_id and emb_idx < len(embeddings):
                embeddings[emb_idx].source_video_id = yt_to_video_id[yt_id]
                session.add(embeddings[emb_idx])
            emb_idx += 1
        session.commit()

    return {
        "status": "imported",
        "channel_id": channel.id,
        "channel_name": channel.name,
        "imported": imported,
    }


# --- Videos ---

@app.get("/videos", response_model=List[Video])
def read_videos(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    from sqlalchemy import case
    query = select(Video)
    if channel_id:
        query = query.where(Video.channel_id == channel_id)
    # NULLs first (newest videos often lack dates from flat extraction), then by date desc, then by id desc
    query = query.order_by(
        case((Video.published_at.is_(None), 0), else_=1),
        Video.published_at.desc(),
        Video.id.desc(),
    )
    return session.exec(query).all()

@app.get("/videos/list", response_model=List[VideoListItemRead])
def read_videos_list(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    from sqlalchemy import case

    query = select(
        Video.id,
        Video.youtube_id,
        Video.channel_id,
        Video.title,
        Video.published_at,
        Video.description,
        Video.thumbnail_url,
        Video.duration,
        Video.processed,
        Video.muted,
        Video.status,
    )
    if channel_id:
        query = query.where(Video.channel_id == channel_id)
    query = query.order_by(
        case((Video.published_at.is_(None), 0), else_=1),
        Video.published_at.desc(),
        Video.id.desc(),
    )

    rows = session.exec(query).all()
    return [
        VideoListItemRead(
            id=row[0],
            youtube_id=row[1],
            channel_id=row[2],
            title=row[3],
            published_at=row[4],
            description=row[5],
            thumbnail_url=row[6],
            duration=row[7],
            processed=bool(row[8]),
            muted=bool(row[9]),
            status=row[10],
        )
        for row in rows
    ]

@app.get("/videos/{video_id}", response_model=Video)
def read_video(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")
    return video

def _archive_video_description_if_needed(
    session: Session,
    video: Video,
    *,
    reason: str,
    ai_model: Optional[str] = None,
    note: Optional[str] = None,
) -> Optional[VideoDescriptionRevision]:
    current_text = (video.description or "").strip()
    if not current_text:
        return None

    latest = session.exec(
        select(VideoDescriptionRevision)
        .where(VideoDescriptionRevision.video_id == video.id)
        .order_by(VideoDescriptionRevision.created_at.desc(), VideoDescriptionRevision.id.desc())
    ).first()
    if latest and (latest.description_text or "").strip() == current_text:
        return None

    existing_count = len(session.exec(
        select(VideoDescriptionRevision.id).where(VideoDescriptionRevision.video_id == video.id)
    ).all())
    source = reason
    if existing_count == 0 and reason == "before_ai_publish":
        source = "ingest_original"

    rev = VideoDescriptionRevision(
        video_id=video.id,
        description_text=current_text,
        source=source,
        ai_model=ai_model,
        note=note,
    )
    session.add(rev)
    session.flush()
    return rev


def _enqueue_unique_job(
    session: Session,
    *,
    video_id: int,
    job_type: str,
    payload: Optional[dict] = None,
) -> Job:
    payload_text = json.dumps(payload or {}, sort_keys=True) if payload is not None else None
    existing = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type == job_type,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES_CORE),
        )
    ).all()
    for j in existing:
        if (j.payload_json or None) == (payload_text or None):
            return j
    job = Job(video_id=video_id, job_type=job_type, status="queued", payload_json=payload_text)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job

@app.post("/videos/{video_id}/process")
def process_video(video_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if not job:
        job = Job(video_id=video_id, job_type="process", status="queued")
        session.add(job)
        session.commit()
    
    # Worker will pick this up automatically
    # background_tasks.add_task(ingestion_service.process_queue)
    return {"status": "processing_queued"}

@app.post("/channels/{channel_id}/process-all")
def process_all_videos(channel_id: int, session: Session = Depends(get_session)):
    """Queue all unmuted, unprocessed videos in a channel for processing"""
    videos = session.exec(
        select(Video).where(
            Video.channel_id == channel_id,
            Video.muted == False,
            Video.processed == False
        )
    ).all()

    jobs_created = []
    for video in videos:
        # Check if there's already an active job for this video
        existing = session.exec(
            select(Job).where(
                Job.video_id == video.id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES)
            )
        ).first()

        if not existing:
            job = Job(video_id=video.id, job_type="process", status="queued")
            session.add(job)
            jobs_created.append(video.id)

    session.commit()

    # Worker will pick these up automatically
    return {"queued": len(jobs_created), "video_ids": jobs_created}

@app.post("/jobs/pause-all")
def pause_all_jobs(session: Session = Depends(get_session)):
    """Pause all queued jobs. Running jobs will complete."""
    statement = select(Job).where(Job.status == "queued")
    jobs = session.exec(statement).all()
    
    count = 0
    for job in jobs:
        job.status = "paused"
        session.add(job)
        count += 1
        
    session.commit()
    return {"paused_count": count}

@app.post("/jobs/resume-all")
def resume_all_jobs(session: Session = Depends(get_session)):
    """Resume all paused jobs. Worker will pick them up automatically."""
    statement = select(Job).where(Job.status == "paused")
    jobs = session.exec(statement).all()

    count = 0
    for job in jobs:
        job.status = "queued"
        session.add(job)
        count += 1

    session.commit()
    return {"resumed_count": count}

@app.get("/videos/{video_id}/segments", response_model=List[TranscriptSegmentRead])
def read_segments(video_id: int, session: Session = Depends(get_session)):
    # Join with Speaker to get name
    results = session.exec(
        select(TranscriptSegment, Speaker.name)
        .join(Speaker, isouter=True)
        .where(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
    ).all()
    
    segments = []
    for seg, speaker_name in results:
        # Use model_dump(exclude={"speaker"}) to avoid conflict with relationship if it exists in dump
        seg_dict = seg.model_dump(exclude={"speaker"})
        read_seg = TranscriptSegmentRead(**seg_dict, speaker=speaker_name)
        segments.append(read_seg)
    return segments

@app.get("/videos/{video_id}/funny-moments", response_model=List[FunnyMomentRead])
def read_funny_moments(video_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(FunnyMoment)
        .where(FunnyMoment.video_id == video_id)
        .order_by(FunnyMoment.start_time)
    ).all()

@app.post("/videos/{video_id}/funny-moments/detect", response_model=List[FunnyMomentRead])
def detect_funny_moments(video_id: int, force: bool = False):
    try:
        return ingestion_service.detect_funny_moments(video_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment detection failed: {e}")


@app.post("/videos/{video_id}/funny-moments/detect/queue")
def queue_detect_funny_moments(video_id: int, force: bool = True, session: Session = Depends(get_session)):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    job = _enqueue_unique_job(session, video_id=video_id, job_type="funny_detect", payload={"force": bool(force)})
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@app.get("/videos/{video_id}/funny-moments/progress")
def get_funny_moments_progress(video_id: int):
    try:
        if ingestion_service is None:
            return {"video_id": video_id, "status": "idle"}
        return ingestion_service.get_funny_task_progress(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment progress unavailable: {e}")

@app.post("/videos/{video_id}/funny-moments/explain", response_model=List[FunnyMomentRead])
def explain_funny_moments(video_id: int, force: bool = False, limit: Optional[int] = None):
    try:
        return ingestion_service.explain_funny_moments(video_id, force=force, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Ollama disabled/unreachable/config issue
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funny moment explanation failed: {e}")


@app.post("/videos/{video_id}/funny-moments/explain/queue")
def queue_explain_funny_moments(
    video_id: int,
    force: bool = True,
    limit: Optional[int] = None,
    session: Session = Depends(get_session),
):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    payload = {"force": bool(force)}
    if limit is not None:
        payload["limit"] = int(limit)
    job = _enqueue_unique_job(session, video_id=video_id, job_type="funny_explain", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@app.post("/videos/{video_id}/youtube-ai/generate", response_model=Video)
def generate_youtube_ai_metadata(video_id: int, force: bool = False):
    try:
        return ingestion_service.generate_youtube_metadata_suggestion(video_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube metadata generation failed: {e}")


@app.post("/videos/{video_id}/youtube-ai/generate/queue")
def queue_generate_youtube_ai_metadata(video_id: int, force: bool = True, session: Session = Depends(get_session)):
    if not session.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    job = _enqueue_unique_job(session, video_id=video_id, job_type="youtube_metadata", payload={"force": bool(force)})
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@app.get("/videos/{video_id}/description-history", response_model=List[VideoDescriptionRevisionRead])
def get_video_description_history(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return session.exec(
        select(VideoDescriptionRevision)
        .where(VideoDescriptionRevision.video_id == video_id)
        .order_by(VideoDescriptionRevision.created_at.desc(), VideoDescriptionRevision.id.desc())
    ).all()

def _publish_video_ai_description_internal(
    session: Session,
    video: Video,
    *,
    push_to_youtube: bool,
) -> dict:
    draft = (video.youtube_ai_description_text or "").strip()
    if not draft:
        raise ValueError("No generated YouTube description draft found. Generate it first.")

    current_desc = (video.description or "").strip()
    remote_pushed = False
    try:
        if current_desc != draft:
            _archive_video_description_if_needed(
                session,
                video,
                reason="before_ai_publish",
                ai_model=video.youtube_ai_model,
                note="Archived before applying AI-generated YouTube description draft",
            )

        if push_to_youtube:
            _youtube_update_video_description_remote(video.youtube_id, draft)
            remote_pushed = True

        if current_desc != draft:
            video.description = draft
            session.add(video)

        session.commit()
        session.refresh(video)
        return {
            "video": video,
            "updated_local": current_desc != draft,
            "pushed_to_youtube": remote_pushed,
            "skipped_local_same_text": current_desc == draft,
        }
    except Exception:
        session.rollback()
        raise


@app.post("/videos/{video_id}/youtube-ai/publish-description", response_model=Video)
def publish_youtube_ai_description(video_id: int, push_to_youtube: Optional[bool] = None, session: Session = Depends(get_session)):
    """Archive current description and apply the AI-generated YouTube description draft.

    If YouTube publishing is enabled/configured, this also pushes the description to the
    actual YouTube video via `videos.update` while preserving snippet title/category/tags.
    """
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    should_push = _youtube_get_cfg()["push_enabled"] if push_to_youtube is None else bool(push_to_youtube)
    try:
        result = _publish_video_ai_description_internal(session, video, push_to_youtube=should_push)
        return result["video"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Publish failed: {e}")

@app.post("/videos/{video_id}/description-history/{revision_id}/restore", response_model=Video)
def restore_video_description_from_history(video_id: int, revision_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    revision = session.get(VideoDescriptionRevision, revision_id)
    if not revision or revision.video_id != video_id:
        raise HTTPException(status_code=404, detail="Description revision not found for this video")

    target_text = (revision.description_text or "").strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="Selected revision has an empty description")

    current_desc = (video.description or "").strip()
    if current_desc == target_text:
        return video

    _archive_video_description_if_needed(
        session,
        video,
        reason="before_restore",
        ai_model=video.youtube_ai_model,
        note=f"Archived before restoring description revision #{revision_id}",
    )
    video.description = revision.description_text
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@app.post("/channels/{channel_id}/youtube-ai/publish-descriptions")
def batch_publish_channel_youtube_descriptions(channel_id: int, req: ChannelBatchPublishRequest, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    should_push = _youtube_get_cfg()["push_enabled"] if req.push_to_youtube is None else bool(req.push_to_youtube)
    limit = None if req.limit is None else max(1, min(int(req.limit), 1000))
    ownership_check = None
    if should_push:
        ownership_check = _youtube_channel_ownership_check_for_app_channel(channel)

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id, Video.processed == True, Video.muted == False)
        .order_by(Video.published_at.asc(), Video.id.asc())
    ).all()
    if limit is not None:
        videos = videos[:limit]

    items = []
    eligible_video_ids: list[int] = []
    for v in videos:
        draft = (v.youtube_ai_description_text or "").strip()
        current_desc = (v.description or "").strip()
        has_draft = bool(draft)
        matches = bool(has_draft and current_desc == draft)
        reason = None
        if not has_draft:
            reason = "missing_ai_draft"
        elif matches and not should_push:
            reason = "already_matches_draft"
        item = {
            "video_id": v.id,
            "youtube_id": v.youtube_id,
            "title": v.title,
            "processed": bool(v.processed),
            "has_ai_draft": has_draft,
            "current_matches_draft": matches,
            "eligible": reason is None,
            "reason": reason,
            "youtube_ai_model": v.youtube_ai_model,
            "published_at": v.published_at.isoformat() if v.published_at else None,
        }
        if reason is None:
            eligible_video_ids.append(v.id)
        items.append(item)

    if req.dry_run:
        return {
            "status": "dry_run",
            "channel_id": channel_id,
            "channel_name": channel.name,
            "push_to_youtube": should_push,
            "ownership_check": ownership_check,
            "confirm_required": True,
            "counts": {
                "scanned": len(videos),
                "eligible": len(eligible_video_ids),
                "missing_ai_draft": sum(1 for i in items if i["reason"] == "missing_ai_draft"),
                "already_matches_draft": sum(1 for i in items if i["reason"] == "already_matches_draft"),
            },
            "estimated_youtube_quota_units": (len(eligible_video_ids) * 51) if should_push else 0,
            "items": items,
        }

    if not req.confirm:
        raise HTTPException(status_code=400, detail="Batch publish requires confirm=true when dry_run=false")
    if should_push:
        ownership_check = ownership_check or _youtube_channel_ownership_check_for_app_channel(channel)
        if ownership_check.get("status") != "owned":
            raise HTTPException(
                status_code=400,
                detail=f"Batch YouTube publish blocked: ownership check status is '{ownership_check.get('status')}'. "
                       "Connect the matching YouTube channel or run with push_to_youtube=false."
            )

    results = []
    success_count = 0
    error_count = 0
    for item in items:
        if not item["eligible"]:
            item["status"] = "skipped"
            results.append(item)
            continue
        try:
            video = session.get(Video, item["video_id"])
            if not video:
                raise RuntimeError("Video not found during batch publish")
            publish_result = _publish_video_ai_description_internal(session, video, push_to_youtube=should_push)
            item["status"] = "published"
            item["updated_local"] = bool(publish_result["updated_local"])
            item["pushed_to_youtube"] = bool(publish_result["pushed_to_youtube"])
            success_count += 1
        except Exception as e:
            session.rollback()
            item["status"] = "error"
            item["error"] = str(e)
            error_count += 1
        results.append(item)

    return {
        "status": "completed",
        "channel_id": channel_id,
        "channel_name": channel.name,
        "push_to_youtube": should_push,
        "ownership_check": ownership_check,
        "counts": {
            "scanned": len(videos),
            "eligible": len(eligible_video_ids),
            "published": success_count,
            "errors": error_count,
            "skipped": len(videos) - len(eligible_video_ids),
        },
        "estimated_youtube_quota_units": (len(eligible_video_ids) * 51) if should_push else 0,
        "items": results,
    }

@app.get("/search", response_model=TranscriptSearchPage)
def search_segments(
    q: str,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    sort: str = "newest",
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    from sqlalchemy import func, case

    q = (q or "").strip()
    if not q:
        return TranscriptSearchPage(items=[], total=0, limit=max(1, min(limit, 200)), offset=max(0, offset), has_more=False)

    if year is not None and (year < 2000 or year > 2100):
        raise HTTPException(status_code=400, detail="year must be between 2000 and 2100")
    if month is not None and (month < 1 or month > 12):
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")
    sort_mode = (sort or "newest").lower()
    if sort_mode == "chronological":
        sort_mode = "oldest"
    if sort_mode not in ["newest", "oldest"]:
        raise HTTPException(status_code=400, detail="sort must be 'newest' or 'oldest'")

    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)

    needs_video_join = bool(channel_id or year is not None or month is not None)

    q_like = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if IS_POSTGRES:
        text_match = TranscriptSegment.text.ilike(f"%{q_like}%", escape="\\")
    else:
        text_match = TranscriptSegment.text.contains(q)

    data_query = (
        select(
            TranscriptSegment.id,
            TranscriptSegment.video_id,
            TranscriptSegment.speaker_id,
            TranscriptSegment.matched_profile_id,
            TranscriptSegment.start_time,
            TranscriptSegment.end_time,
            TranscriptSegment.text,
            Speaker.name,
        )
        .join(Speaker, isouter=True)
        .where(text_match)
    )
    count_query = select(func.count(TranscriptSegment.id)).where(text_match)

    if video_id:
        data_query = data_query.where(TranscriptSegment.video_id == video_id)
        count_query = count_query.where(TranscriptSegment.video_id == video_id)

    if needs_video_join:
        data_query = data_query.join(Video, TranscriptSegment.video_id == Video.id)
        count_query = count_query.join(Video, TranscriptSegment.video_id == Video.id)

    if channel_id:
        data_query = data_query.where(Video.channel_id == channel_id)
        count_query = count_query.where(Video.channel_id == channel_id)

    if year is not None:
        if IS_POSTGRES:
            data_query = data_query.where(func.extract("year", Video.published_at) == year)
            count_query = count_query.where(func.extract("year", Video.published_at) == year)
        else:
            year_str = f"{year:04d}"
            data_query = data_query.where(func.strftime("%Y", Video.published_at) == year_str)
            count_query = count_query.where(func.strftime("%Y", Video.published_at) == year_str)

    if month is not None:
        if IS_POSTGRES:
            data_query = data_query.where(func.extract("month", Video.published_at) == month)
            count_query = count_query.where(func.extract("month", Video.published_at) == month)
        else:
            month_str = f"{month:02d}"
            data_query = data_query.where(func.strftime("%m", Video.published_at) == month_str)
            count_query = count_query.where(func.strftime("%m", Video.published_at) == month_str)

    # Stable ordering for pagination: newest videos first, then transcript order.
    if needs_video_join:
        nulls_last = case((Video.published_at.is_(None), 1), else_=0)
        if sort_mode == "oldest":
            data_query = data_query.order_by(
                nulls_last,
                Video.published_at.asc(),
                TranscriptSegment.video_id.asc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
        else:
            data_query = data_query.order_by(
                nulls_last,
                Video.published_at.desc(),
                TranscriptSegment.video_id.desc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
    else:
        if sort_mode == "oldest":
            data_query = data_query.order_by(
                TranscriptSegment.video_id.asc(),
                TranscriptSegment.start_time.asc(),
                TranscriptSegment.id.asc(),
            )
        else:
            data_query = data_query.order_by(TranscriptSegment.id.desc())

    total = session.exec(count_query).one()
    results = session.exec(data_query.offset(safe_offset).limit(safe_limit)).all()

    items: List[TranscriptSearchItemRead] = []
    for row in results:
        items.append(
            TranscriptSearchItemRead(
                id=int(row[0]),
                video_id=int(row[1]),
                speaker_id=row[2],
                matched_profile_id=row[3],
                start_time=float(row[4]),
                end_time=float(row[5]),
                text=row[6],
                speaker=row[7],
            )
        )

    return TranscriptSearchPage(
        items=items,
        total=int(total or 0),
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < int(total or 0),
    )

@app.patch("/segments/{segment_id}/assign-speaker")
def assign_segment_speaker(segment_id: int, body: AssignSpeakerRequest, session: Session = Depends(get_session)):
    """Assign or reassign a speaker to a transcript segment."""
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    speaker = session.get(Speaker, body.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    segment.speaker_id = body.speaker_id
    # Manual reassignment overrides diarization profile provenance.
    segment.matched_profile_id = None
    session.add(segment)
    session.commit()
    _invalidate_speaker_query_caches()
    session.refresh(segment)
    return {"id": segment.id, "speaker_id": body.speaker_id, "speaker_name": speaker.name, "matched_profile_id": None}


def _move_profile_between_speakers(
    session: Session,
    source_speaker: Speaker,
    profile: SpeakerEmbedding,
    *,
    target_speaker_id: Optional[int] = None,
    new_speaker_name: Optional[str] = None,
):
    from sqlalchemy import func

    has_target = target_speaker_id is not None
    has_new = bool((new_speaker_name or "").strip())
    if has_target == has_new:
        raise HTTPException(status_code=400, detail="Provide exactly one of target_speaker_id or new_speaker_name")

    profile_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == source_speaker.id)
    ).first() or 0
    if int(profile_count) <= 1:
        raise HTTPException(status_code=400, detail="Cannot move the last voice profile")

    target_speaker = None
    created_target = False

    if has_target:
        target_speaker = session.get(Speaker, int(target_speaker_id))
        if not target_speaker:
            raise HTTPException(status_code=404, detail="Target speaker not found")
        if target_speaker.channel_id != source_speaker.channel_id:
            raise HTTPException(status_code=400, detail="Target speaker must be in the same channel")
        if target_speaker.id == source_speaker.id:
            raise HTTPException(status_code=400, detail="Target speaker must be different from source speaker")
    else:
        new_name = (new_speaker_name or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="New speaker name is required")
        target_speaker = Speaker(
            channel_id=source_speaker.channel_id,
            name=new_name,
            embedding_blob=profile.embedding_blob,
            is_extra=False,
        )
        session.add(target_speaker)
        session.commit()
        session.refresh(target_speaker)
        created_target = True

    profile.speaker_id = target_speaker.id
    session.add(profile)

    # Keep legacy single-embedding blob fields aligned with current profiles.
    source_replacement = session.exec(
        select(SpeakerEmbedding)
        .where(SpeakerEmbedding.speaker_id == source_speaker.id, SpeakerEmbedding.id != profile.id)
        .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
    ).first()
    if source_replacement:
        source_speaker.embedding_blob = source_replacement.embedding_blob
        session.add(source_speaker)
    if not created_target:
        target_speaker.embedding_blob = profile.embedding_blob
        session.add(target_speaker)

    session.commit()

    remaining_source = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == source_speaker.id)
    ).first() or 0

    return {
        "profile_id": int(profile.id),
        "source_speaker_id": int(source_speaker.id),
        "target_speaker_id": int(target_speaker.id),
        "target_speaker_name": target_speaker.name,
        "created_target": created_target,
        "remaining_source_profiles": int(remaining_source),
    }


@app.post("/segments/{segment_id}/split-profile")
def split_segment_profile(
    segment_id: int,
    req: SplitSegmentProfileRequest,
    session: Session = Depends(get_session),
):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    if not segment.speaker_id:
        raise HTTPException(status_code=400, detail="Segment has no assigned speaker")

    source_speaker = session.get(Speaker, segment.speaker_id)
    if not source_speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")

    video = session.get(Video, segment.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Segment video not found")

    profile: Optional[SpeakerEmbedding] = None
    selection_mode = "explicit"

    if req.profile_id is not None:
        profile = session.get(SpeakerEmbedding, int(req.profile_id))
        if not profile or profile.speaker_id != source_speaker.id:
            raise HTTPException(status_code=404, detail="Voice profile not found for this speaker")
    elif getattr(segment, "matched_profile_id", None):
        candidate = session.get(SpeakerEmbedding, int(segment.matched_profile_id))
        if candidate and candidate.speaker_id == source_speaker.id:
            profile = candidate
            selection_mode = "segment_matched_profile"

    if profile is None:
        profiles = session.exec(
            select(SpeakerEmbedding)
            .where(SpeakerEmbedding.speaker_id == source_speaker.id)
            .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
        ).all()
        if not profiles:
            raise HTTPException(status_code=400, detail="Source speaker has no voice profiles")

        if len(profiles) == 1:
            profile = profiles[0]
            selection_mode = "single_profile_fallback"
        else:
            try:
                ingestion_service._load_models()
                audio_path = ingestion_service.get_audio_path(video)
                if not audio_path.exists():
                    raise HTTPException(
                        status_code=409,
                        detail="Audio file not found for this episode. Re-download or reprocess this video first."
                    )
                from pyannote.core import Segment as PyannoteSegment
                audio_input = ingestion_service._load_audio_for_pyannote(str(audio_path))
                seg_start = max(0.0, float(segment.start_time))
                seg_end = max(seg_start + 0.1, float(segment.end_time))
                seg_embedding = ingestion_service.get_speaker_embedding(audio_input, PyannoteSegment(seg_start, seg_end))
                if seg_embedding is None:
                    raise RuntimeError("Failed to extract segment embedding")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=409,
                    detail=f"Could not identify a profile for this segment automatically: {e}"
                )

            vec = np.asarray(seg_embedding, dtype=np.float32).reshape(-1)
            best_profile = None
            best_score = float("inf")
            for p in profiles:
                try:
                    p_vec = np.asarray(pickle.loads(p.embedding_blob), dtype=np.float32).reshape(-1)
                    denom = (np.linalg.norm(vec) * np.linalg.norm(p_vec))
                    if denom <= 0:
                        continue
                    cos_dist = 1.0 - float(np.dot(vec, p_vec) / denom)
                    if cos_dist < best_score:
                        best_score = cos_dist
                        best_profile = p
                except Exception:
                    continue

            if best_profile is None:
                raise HTTPException(
                    status_code=409,
                    detail="Could not identify a matching voice profile for this segment"
                )
            profile = best_profile
            selection_mode = "segment_embedding_match"

    target_name = (req.new_speaker_name or "").strip()
    if req.target_speaker_id is None and not target_name:
        base = (source_speaker.name or "Speaker").strip()
        target_name = f"{base} - split"
        existing_names = set(
            n.lower() for n in session.exec(
                select(Speaker.name).where(Speaker.channel_id == source_speaker.channel_id)
            ).all()
        )
        if target_name.lower() in existing_names:
            i = 2
            while f"{target_name} {i}".lower() in existing_names:
                i += 1
            target_name = f"{target_name} {i}"

    move_result = _move_profile_between_speakers(
        session,
        source_speaker,
        profile,
        target_speaker_id=req.target_speaker_id,
        new_speaker_name=target_name if req.target_speaker_id is None else None,
    )

    if req.reassign_segment:
        segment.speaker_id = move_result["target_speaker_id"]
        segment.matched_profile_id = profile.id
        session.add(segment)
        session.commit()

    _invalidate_speaker_query_caches()

    return {
        "status": "split_profile",
        "segment_id": int(segment.id),
        "video_id": int(segment.video_id),
        "selection_mode": selection_mode,
        **move_result,
        "segment_speaker_id": int(segment.speaker_id) if segment.speaker_id else None,
        "segment_matched_profile_id": int(segment.matched_profile_id) if segment.matched_profile_id else None,
    }

@app.patch("/segments/{segment_id}/text", response_model=TranscriptSegmentRead)
def update_segment_text(segment_id: int, body: SegmentTextUpdateRequest, session: Session = Depends(get_session)):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    requested_words: Optional[List[str]] = None
    if body.words is not None:
        requested_words = []
        for raw in body.words:
            token = " ".join(str(raw or "").replace("\n", " ").split()).strip()
            if token:
                requested_words.append(token)

    normalized_text = " ".join((body.text or "").replace("\n", " ").split()).strip()
    if requested_words is not None and len(requested_words) > 0:
        new_text = " ".join(requested_words).strip()
    else:
        new_text = normalized_text

    if not new_text or not new_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")

    old_text = (segment.text or "").strip()
    if old_text == new_text and requested_words is None:
        # Return current row flattened
        speaker_name = None
        if segment.speaker_id:
            sp = session.get(Speaker, segment.speaker_id)
            speaker_name = sp.name if sp else None
        seg_dict = segment.model_dump(exclude={"speaker"})
        return TranscriptSegmentRead(**seg_dict, speaker=speaker_name)

    # If explicit word tokens are provided, preserve existing per-word timestamps when
    # lengths match; otherwise rebuild timings uniformly over the segment window.
    if requested_words is not None and len(requested_words) > 0:
        existing_words = []
        if segment.words:
            try:
                parsed = json.loads(segment.words)
                if isinstance(parsed, list):
                    for row in parsed:
                        s = float(row.get("start"))
                        e = float(row.get("end"))
                        if e > s:
                            existing_words.append({"start": s, "end": e, "word": str(row.get("word") or "").strip()})
            except Exception:
                existing_words = []

        rebuilt_words = []
        if existing_words and len(existing_words) == len(requested_words):
            for i, token in enumerate(requested_words):
                rebuilt_words.append({
                    "start": existing_words[i]["start"],
                    "end": existing_words[i]["end"],
                    "word": token,
                })
        else:
            seg_start = float(segment.start_time)
            seg_end = max(seg_start + 0.05, float(segment.end_time))
            count = max(1, len(requested_words))
            step = (seg_end - seg_start) / count
            for i, token in enumerate(requested_words):
                s = seg_start + (i * step)
                e = seg_start + ((i + 1) * step)
                rebuilt_words.append({
                    "start": round(s, 3),
                    "end": round(max(s + 0.01, e), 3),
                    "word": token,
                })

        segment.words = json.dumps(rebuilt_words, ensure_ascii=False)

    rev = TranscriptSegmentRevision(
        segment_id=segment.id,
        video_id=segment.video_id,
        old_text=segment.text or "",
        new_text=new_text,
        source="manual_edit_words" if requested_words is not None else "manual_edit",
    )
    session.add(rev)
    segment.text = new_text
    session.add(segment)
    session.commit()
    session.refresh(segment)
    speaker_name = None
    if segment.speaker_id:
        sp = session.get(Speaker, segment.speaker_id)
        speaker_name = sp.name if sp else None
    seg_dict = segment.model_dump(exclude={"speaker"})
    return TranscriptSegmentRead(**seg_dict, speaker=speaker_name)

@app.get("/segments/{segment_id}/revisions", response_model=List[TranscriptSegmentRevisionRead])
def get_segment_revisions(segment_id: int, session: Session = Depends(get_session)):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    return session.exec(
        select(TranscriptSegmentRevision)
        .where(TranscriptSegmentRevision.segment_id == segment_id)
        .order_by(TranscriptSegmentRevision.created_at.desc(), TranscriptSegmentRevision.id.desc())
    ).all()

@app.get("/videos/{video_id}/clip")
def get_clip(video_id: int, start: float, end: float, audio_only: bool = False):
    try:
        path = ingestion_service.create_clip(video_id, start, end, audio_only=audio_only)
        media_type = "audio/mp4" if audio_only else "video/mp4"
        return FileResponse(path, filename=Path(path).name, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Saved Clips ---

def _normalize_clip_defaults(clip: Clip) -> bool:
    changed = False
    if not getattr(clip, "aspect_ratio", None):
        clip.aspect_ratio = "source"
        changed = True
    if getattr(clip, "portrait_split_enabled", None) is None:
        clip.portrait_split_enabled = False
        changed = True
    if getattr(clip, "fade_in_sec", None) is None:
        clip.fade_in_sec = 0.0
        changed = True
    if getattr(clip, "fade_out_sec", None) is None:
        clip.fade_out_sec = 0.0
        changed = True
    if getattr(clip, "burn_captions", None) is None:
        clip.burn_captions = False
        changed = True
    if getattr(clip, "caption_speaker_labels", None) is None:
        clip.caption_speaker_labels = True
        changed = True
    return changed


def _clip_to_read(clip: Clip) -> ClipRead:
    return ClipRead(
        id=int(clip.id),
        video_id=int(clip.video_id),
        start_time=float(clip.start_time),
        end_time=float(clip.end_time),
        title=str(clip.title),
        aspect_ratio=str(clip.aspect_ratio or "source"),
        crop_x=clip.crop_x,
        crop_y=clip.crop_y,
        crop_w=clip.crop_w,
        crop_h=clip.crop_h,
        portrait_split_enabled=bool(getattr(clip, "portrait_split_enabled", False)),
        portrait_top_crop_x=getattr(clip, "portrait_top_crop_x", None),
        portrait_top_crop_y=getattr(clip, "portrait_top_crop_y", None),
        portrait_top_crop_w=getattr(clip, "portrait_top_crop_w", None),
        portrait_top_crop_h=getattr(clip, "portrait_top_crop_h", None),
        portrait_bottom_crop_x=getattr(clip, "portrait_bottom_crop_x", None),
        portrait_bottom_crop_y=getattr(clip, "portrait_bottom_crop_y", None),
        portrait_bottom_crop_w=getattr(clip, "portrait_bottom_crop_w", None),
        portrait_bottom_crop_h=getattr(clip, "portrait_bottom_crop_h", None),
        script_edits_json=getattr(clip, "script_edits_json", None),
        fade_in_sec=float(getattr(clip, "fade_in_sec", 0.0) or 0.0),
        fade_out_sec=float(getattr(clip, "fade_out_sec", 0.0) or 0.0),
        burn_captions=bool(clip.burn_captions),
        caption_speaker_labels=bool(clip.caption_speaker_labels),
        created_at=clip.created_at,
    )

@app.get("/videos/{video_id}/clips", response_model=List[ClipRead])
def read_video_clips(video_id: int, session: Session = Depends(get_session)):
    clips = session.exec(select(Clip).where(Clip.video_id == video_id).order_by(Clip.start_time)).all()
    changed = False
    for c in clips:
        if _normalize_clip_defaults(c):
            session.add(c)
            changed = True
    if changed:
        session.commit()
        for c in clips:
            session.refresh(c)
    return [_clip_to_read(c) for c in clips]

@app.post("/videos/{video_id}/clips", response_model=ClipRead)
def create_clip(video_id: int, clip: ClipCreate, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    db_clip = Clip.model_validate(clip, update={"video_id": video_id})
    _normalize_clip_defaults(db_clip)
    session.add(db_clip)
    session.commit()
    session.refresh(db_clip)
    return _clip_to_read(db_clip)

@app.delete("/clips/{clip_id}")
def delete_clip(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    session.delete(clip)
    session.commit()
    return {"ok": True}

@app.patch("/clips/{clip_id}", response_model=ClipRead)
def update_clip(clip_id: int, clip_update: ClipCreate, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    clip.title = clip_update.title
    clip.start_time = clip_update.start_time
    clip.end_time = clip_update.end_time
    clip.aspect_ratio = clip_update.aspect_ratio
    clip.crop_x = clip_update.crop_x
    clip.crop_y = clip_update.crop_y
    clip.crop_w = clip_update.crop_w
    clip.crop_h = clip_update.crop_h
    clip.portrait_split_enabled = bool(clip_update.portrait_split_enabled)
    clip.portrait_top_crop_x = clip_update.portrait_top_crop_x
    clip.portrait_top_crop_y = clip_update.portrait_top_crop_y
    clip.portrait_top_crop_w = clip_update.portrait_top_crop_w
    clip.portrait_top_crop_h = clip_update.portrait_top_crop_h
    clip.portrait_bottom_crop_x = clip_update.portrait_bottom_crop_x
    clip.portrait_bottom_crop_y = clip_update.portrait_bottom_crop_y
    clip.portrait_bottom_crop_w = clip_update.portrait_bottom_crop_w
    clip.portrait_bottom_crop_h = clip_update.portrait_bottom_crop_h
    clip.script_edits_json = clip_update.script_edits_json
    clip.fade_in_sec = max(0.0, float(clip_update.fade_in_sec or 0.0))
    clip.fade_out_sec = max(0.0, float(clip_update.fade_out_sec or 0.0))
    clip.burn_captions = bool(clip_update.burn_captions)
    clip.caption_speaker_labels = bool(clip_update.caption_speaker_labels)
    
    session.add(clip)
    session.commit()
    session.refresh(clip)
    _normalize_clip_defaults(clip)
    return _clip_to_read(clip)

@app.post("/clips/{clip_id}/apply-export-preset", response_model=ClipRead)
def apply_clip_export_preset(clip_id: int, body: ClipExportPresetRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if body.aspect_ratio is not None:
        clip.aspect_ratio = body.aspect_ratio
    if body.burn_captions is not None:
        clip.burn_captions = bool(body.burn_captions)
    if body.caption_speaker_labels is not None:
        clip.caption_speaker_labels = bool(body.caption_speaker_labels)
    if body.portrait_split_enabled is not None:
        clip.portrait_split_enabled = bool(body.portrait_split_enabled)
    if body.fade_in_sec is not None:
        clip.fade_in_sec = max(0.0, float(body.fade_in_sec))
    if body.fade_out_sec is not None:
        clip.fade_out_sec = max(0.0, float(body.fade_out_sec))
    for key in (
        "crop_x", "crop_y", "crop_w", "crop_h",
        "portrait_top_crop_x", "portrait_top_crop_y", "portrait_top_crop_w", "portrait_top_crop_h",
        "portrait_bottom_crop_x", "portrait_bottom_crop_y", "portrait_bottom_crop_w", "portrait_bottom_crop_h",
    ):
        val = getattr(body, key)
        if val is not None:
            setattr(clip, key, float(val))

    session.add(clip)
    session.commit()
    session.refresh(clip)
    _normalize_clip_defaults(clip)
    return _clip_to_read(clip)

@app.get("/channels/{channel_id}/clips", response_model=List[ChannelClipRead])
def read_channel_clips(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    rows = session.exec(
        select(Clip, Video)
        .join(Video, Clip.video_id == Video.id)
        .where(Video.channel_id == channel_id)
        .order_by(Clip.created_at.desc(), Clip.id.desc())
    ).all()
    out: List[ChannelClipRead] = []
    changed = False
    for clip, video in rows:
        if _normalize_clip_defaults(clip):
            session.add(clip)
            changed = True
        out.append(ChannelClipRead(
            id=int(clip.id),
            video_id=int(clip.video_id),
            start_time=float(clip.start_time),
            end_time=float(clip.end_time),
            title=str(clip.title),
            aspect_ratio=str(clip.aspect_ratio or "source"),
            crop_x=clip.crop_x,
            crop_y=clip.crop_y,
            crop_w=clip.crop_w,
            crop_h=clip.crop_h,
            portrait_split_enabled=bool(getattr(clip, "portrait_split_enabled", False)),
            portrait_top_crop_x=getattr(clip, "portrait_top_crop_x", None),
            portrait_top_crop_y=getattr(clip, "portrait_top_crop_y", None),
            portrait_top_crop_w=getattr(clip, "portrait_top_crop_w", None),
            portrait_top_crop_h=getattr(clip, "portrait_top_crop_h", None),
            portrait_bottom_crop_x=getattr(clip, "portrait_bottom_crop_x", None),
            portrait_bottom_crop_y=getattr(clip, "portrait_bottom_crop_y", None),
            portrait_bottom_crop_w=getattr(clip, "portrait_bottom_crop_w", None),
            portrait_bottom_crop_h=getattr(clip, "portrait_bottom_crop_h", None),
            script_edits_json=getattr(clip, "script_edits_json", None),
            fade_in_sec=float(getattr(clip, "fade_in_sec", 0.0) or 0.0),
            fade_out_sec=float(getattr(clip, "fade_out_sec", 0.0) or 0.0),
            burn_captions=bool(clip.burn_captions),
            caption_speaker_labels=bool(clip.caption_speaker_labels),
            created_at=clip.created_at,
            video_title=str(video.title),
            video_youtube_id=str(video.youtube_id),
            video_published_at=video.published_at,
            video_thumbnail_url=video.thumbnail_url,
        ))
    if changed:
        session.commit()
    return out

@app.post("/clips/{clip_id}/export/mp4")
def export_clip_mp4(clip_id: int):
    try:
        path = ingestion_service.render_clip_export_mp4(clip_id)
        try:
            ingestion_service.record_clip_export_artifact(clip_id, Path(path), artifact_type="video", fmt="mp4")
        except Exception:
            pass
        return FileResponse(str(path), filename=Path(path).name, media_type="video/mp4")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clips/{clip_id}/export/mp4/queue")
def queue_export_clip_mp4(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    payload = {"clip_id": int(clip_id)}
    job = _enqueue_unique_job(session, video_id=clip.video_id, job_type="clip_export_mp4", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@app.post("/clips/{clip_id}/export/captions")
def export_clip_captions(clip_id: int, body: ClipCaptionExportRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        speaker_labels = clip.caption_speaker_labels if body.speaker_labels is None else bool(body.speaker_labels)
        path = ingestion_service.write_clip_caption_file(clip_id, fmt=body.format, speaker_labels=speaker_labels)
        try:
            ingestion_service.record_clip_export_artifact(
                clip_id,
                Path(path),
                artifact_type="captions",
                fmt=(body.format or "srt").lower(),
            )
        except Exception:
            pass
        media_type = "text/vtt" if (body.format or "").lower() == "vtt" else "application/x-subrip"
        return FileResponse(str(path), filename=Path(path).name, media_type=media_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clips/{clip_id}/exports", response_model=List[ClipExportArtifactRead])
def read_clip_export_artifacts(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    rows = session.exec(
        select(ClipExportArtifact)
        .where(ClipExportArtifact.clip_id == clip_id)
        .order_by(ClipExportArtifact.created_at.desc(), ClipExportArtifact.id.desc())
    ).all()
    return [
        ClipExportArtifactRead(
            id=int(r.id),
            clip_id=int(r.clip_id),
            video_id=int(r.video_id),
            artifact_type=str(r.artifact_type),
            format=str(r.format),
            file_path=str(r.file_path),
            file_name=str(r.file_name),
            file_size_bytes=r.file_size_bytes,
            created_at=r.created_at,
        ) for r in rows
    ]


@app.get("/videos/{video_id}/clip-exports", response_model=List[ClipExportArtifactRead])
def read_video_clip_export_artifacts(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    rows = session.exec(
        select(ClipExportArtifact)
        .where(ClipExportArtifact.video_id == video_id)
        .order_by(ClipExportArtifact.created_at.desc(), ClipExportArtifact.id.desc())
    ).all()
    return [
        ClipExportArtifactRead(
            id=int(r.id),
            clip_id=int(r.clip_id),
            video_id=int(r.video_id),
            artifact_type=str(r.artifact_type),
            format=str(r.format),
            file_path=str(r.file_path),
            file_name=str(r.file_name),
            file_size_bytes=r.file_size_bytes,
            created_at=r.created_at,
        ) for r in rows
    ]


@app.get("/clip-exports/{artifact_id}/download")
def download_clip_export_artifact(artifact_id: int, session: Session = Depends(get_session)):
    art = session.get(ClipExportArtifact, artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Clip export artifact not found")
    p = Path(art.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Archived clip file no longer exists on disk")

    fmt = (art.format or "").lower()
    media_type = "application/octet-stream"
    if fmt == "mp4":
        media_type = "video/mp4"
    elif fmt == "srt":
        media_type = "application/x-subrip"
    elif fmt == "vtt":
        media_type = "text/vtt"
    return FileResponse(str(p), filename=art.file_name or p.name, media_type=media_type)


@app.post("/clips/{clip_id}/export/captions/queue")
def queue_export_clip_captions(clip_id: int, body: ClipCaptionExportRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    fmt = (body.format or "srt").lower()
    if fmt not in {"srt", "vtt"}:
        raise HTTPException(status_code=400, detail="format must be srt or vtt")
    speaker_labels = clip.caption_speaker_labels if body.speaker_labels is None else bool(body.speaker_labels)
    payload = {"clip_id": int(clip_id), "format": fmt, "speaker_labels": bool(speaker_labels)}
    job = _enqueue_unique_job(session, video_id=clip.video_id, job_type="clip_export_captions", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}


def _upload_clip_to_youtube_internal(
    session: Session,
    clip_id: int,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy_status: str = "private",
    category_id: str = "22",
    made_for_kids: bool = False,
    tags: Optional[List[str]] = None,
) -> dict:
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
    video = session.get(Video, clip.video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Source video for clip {clip_id} not found")

    privacy = (privacy_status or "private").strip().lower()
    if privacy not in {"private", "unlisted", "public"}:
        raise HTTPException(status_code=400, detail="privacy_status must be one of: private, unlisted, public")

    safe_title = (title or clip.title or f"Clip from {video.title}").strip()
    if not safe_title:
        safe_title = f"Clip from {video.title}"
    safe_title = safe_title[:100]

    safe_description = (description or _build_default_clip_upload_description(video, clip)).strip()[:5000]
    safe_category = (category_id or "22").strip()
    safe_tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:40]  # YouTube max 500 chars across tags

    snippet = {
        "title": safe_title,
        "description": safe_description,
        "categoryId": safe_category,
    }
    if safe_tags:
        snippet["tags"] = safe_tags
    status = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": bool(made_for_kids),
    }

    try:
        export_path = ingestion_service.render_clip_export_mp4(clip_id)
        uploaded = _youtube_upload_video_resumable(export_path, snippet=snippet, status=status)
        new_video_id = str(uploaded.get("id") or "").strip()
        if not new_video_id:
            raise RuntimeError(f"YouTube upload response missing video id: {uploaded}")
        ch_info = _youtube_fetch_authenticated_channel_info()
        return {
            "clip_id": clip.id,
            "source_video_id": video.id,
            "source_video_youtube_id": video.youtube_id,
            "uploaded_video_id": new_video_id,
            "uploaded_watch_url": f"https://www.youtube.com/watch?v={new_video_id}",
            "uploaded_title": safe_title,
            "privacy_status": privacy,
            "channel_id": ch_info.get("channel_id"),
            "channel_title": ch_info.get("channel_title"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip upload failed: {e}")


@app.post("/clips/{clip_id}/youtube/upload")
def upload_clip_to_youtube(clip_id: int, body: Optional[ClipYoutubeUploadRequest] = None, session: Session = Depends(get_session)):
    req = body or ClipYoutubeUploadRequest()
    return _upload_clip_to_youtube_internal(
        session,
        clip_id,
        title=req.title,
        description=req.description,
        privacy_status=req.privacy_status,
        category_id=req.category_id,
        made_for_kids=req.made_for_kids,
        tags=req.tags,
    )


@app.post("/clips/youtube/upload-batch")
def upload_clips_to_youtube_batch(req: ClipBatchYoutubeUploadRequest, session: Session = Depends(get_session)):
    clip_ids = [int(c) for c in (req.clip_ids or []) if int(c) > 0]
    if not clip_ids:
        raise HTTPException(status_code=400, detail="clip_ids is required")

    # Ensure OAuth is valid before starting the batch.
    try:
        auth_channel = _youtube_fetch_authenticated_channel_info()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YouTube not connected/authorized: {e}")

    out = []
    success = 0
    failed = 0
    for clip_id in clip_ids:
        try:
            result = _upload_clip_to_youtube_internal(
                session,
                clip_id,
                privacy_status=req.privacy_status,
                category_id=req.category_id,
                made_for_kids=req.made_for_kids,
                tags=req.tags,
            )
            result["success"] = True
            out.append(result)
            success += 1
        except HTTPException as e:
            out.append({
                "clip_id": clip_id,
                "success": False,
                "error": e.detail,
            })
            failed += 1
        except Exception as e:
            out.append({
                "clip_id": clip_id,
                "success": False,
                "error": str(e),
            })
            failed += 1

    return {
        "auth_channel_id": auth_channel.get("channel_id"),
        "auth_channel_title": auth_channel.get("channel_title"),
        "requested": len(clip_ids),
        "uploaded": success,
        "failed": failed,
        "results": out,
    }

# --- Speakers ---

_SPEAKER_QUERY_CACHE_TTL_SECONDS = max(
    3,
    int(os.getenv("SPEAKER_QUERY_CACHE_TTL_SECONDS", "120"))
)
_speaker_list_cache_lock = threading.Lock()
_speaker_list_cache: dict[str, tuple[float, list[dict]]] = {}
_speaker_counts_cache_lock = threading.Lock()
_speaker_counts_cache: dict[str, tuple[float, dict]] = {}
_speaker_scope_cache_lock = threading.Lock()
_speaker_scope_cache: dict[str, tuple[float, list[dict]]] = {}


def _speaker_cache_fresh(ts: float) -> bool:
    return (time.time() - ts) < _SPEAKER_QUERY_CACHE_TTL_SECONDS


def _get_speaker_list_cache(key: str) -> Optional[list[dict]]:
    with _speaker_list_cache_lock:
        cached = _speaker_list_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_list_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_list_cache(key: str, value: list[dict]) -> None:
    with _speaker_list_cache_lock:
        _speaker_list_cache[key] = (time.time(), value)


def _get_speaker_counts_cache(key: str) -> Optional[dict]:
    with _speaker_counts_cache_lock:
        cached = _speaker_counts_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_counts_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_counts_cache(key: str, value: dict) -> None:
    with _speaker_counts_cache_lock:
        _speaker_counts_cache[key] = (time.time(), value)


def _get_speaker_scope_cache(key: str) -> Optional[list[dict]]:
    with _speaker_scope_cache_lock:
        cached = _speaker_scope_cache.get(key)
        if not cached:
            return None
        if not _speaker_cache_fresh(cached[0]):
            _speaker_scope_cache.pop(key, None)
            return None
        return cached[1]


def _set_speaker_scope_cache(key: str, value: list[dict]) -> None:
    with _speaker_scope_cache_lock:
        _speaker_scope_cache[key] = (time.time(), value)


def _is_unknown_speaker_name(name: Optional[str]) -> bool:
    normalized = (name or "").strip()
    if not normalized:
        return True
    if re.match(r"^unknown(\s+speaker)?$", normalized, re.IGNORECASE):
        return True
    if re.match(r"^speaker\s+\d+$", normalized, re.IGNORECASE):
        return True
    return False


def _get_speaker_scope_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
) -> list[dict]:
    """
    Build/cached aggregate speaker rows for a scope (channel or video).
    This avoids re-running the expensive transcript aggregation per page
    and per counts request.
    """
    from sqlalchemy import func

    scope_key = f"channel:{channel_id or 'all'}|video:{video_id or 'all'}"
    cached_rows = _get_speaker_scope_cache(scope_key)
    if cached_rows is not None:
        return cached_rows

    seg_duration = (TranscriptSegment.end_time - TranscriptSegment.start_time)
    agg_query = select(
        TranscriptSegment.speaker_id.label("speaker_id"),
        func.sum(seg_duration).label("total_time"),
    ).where(TranscriptSegment.speaker_id.is_not(None))

    if video_id:
        agg_query = agg_query.where(TranscriptSegment.video_id == video_id)
    if channel_id:
        # Performance critical on large channels: filter transcript rows
        # via channel video ids so DB can use video_id-oriented indexes.
        channel_video_ids = select(Video.id).where(Video.channel_id == channel_id)
        agg_query = agg_query.where(TranscriptSegment.video_id.in_(channel_video_ids))

    agg_subq = agg_query.group_by(TranscriptSegment.speaker_id).subquery()
    rows = session.exec(
        select(
            Speaker.id,
            Speaker.channel_id,
            Speaker.name,
            Speaker.thumbnail_path,
            Speaker.is_extra,
            Speaker.created_at,
            agg_subq.c.total_time,
        )
        .join(agg_subq, agg_subq.c.speaker_id == Speaker.id)
        .where(agg_subq.c.total_time > 5.0)
        .order_by(agg_subq.c.total_time.desc())
    ).all()

    out: list[dict] = []
    for speaker_id, speaker_channel_id, name, thumbnail_path, is_extra, created_at, total_time in rows:
        out.append(
            {
                "id": int(speaker_id),
                "channel_id": int(speaker_channel_id),
                "name": str(name),
                "thumbnail_path": thumbnail_path,
                "is_extra": bool(is_extra),
                "created_at": created_at,
                "total_speaking_time": round(float(total_time or 0.0), 1),
            }
        )

    _set_speaker_scope_cache(scope_key, out)
    return out


def _invalidate_speaker_query_caches() -> None:
    with _speaker_list_cache_lock:
        _speaker_list_cache.clear()
    with _speaker_counts_cache_lock:
        _speaker_counts_cache.clear()
    with _speaker_scope_cache_lock:
        _speaker_scope_cache.clear()


@app.get("/speakers", response_model=List[SpeakerRead])
def read_speakers(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: Session = Depends(get_session)
):
    from sqlalchemy import func

    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))
    scope_rows = _get_speaker_scope_rows(session=session, channel_id=channel_id, video_id=video_id)

    if safe_offset >= len(scope_rows):
        return []

    if safe_limit is None:
        page_rows = scope_rows[safe_offset:]
    else:
        page_rows = scope_rows[safe_offset:safe_offset + safe_limit]

    emb_counts = {}
    if page_rows:
        speaker_ids = [int(row["id"]) for row in page_rows]
        emb_query = select(
            SpeakerEmbedding.speaker_id,
            func.count(SpeakerEmbedding.id).label("cnt")
        ).where(
            SpeakerEmbedding.speaker_id.in_(speaker_ids)
        ).group_by(SpeakerEmbedding.speaker_id)
        for speaker_id, cnt in session.exec(emb_query).all():
            emb_counts[speaker_id] = cnt

    speakers: List[SpeakerRead] = []
    for row in page_rows:
        speakers.append(
            SpeakerRead(
                id=int(row["id"]),
                channel_id=int(row["channel_id"]),
                name=str(row["name"]),
                thumbnail_path=row.get("thumbnail_path"),
                is_extra=bool(row.get("is_extra")),
                total_speaking_time=float(row.get("total_speaking_time") or 0.0),
                embedding_count=int(emb_counts.get(int(row["id"]), 0)),
                created_at=row["created_at"],
            )
        )
    return speakers


@app.get("/speakers/stats", response_model=SpeakerCountsRead)
def read_speaker_counts(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    EXTRAS_THRESHOLD = 60.0

    cache_key = f"channel:{channel_id or 'all'}|video:{video_id or 'all'}"
    cached = _get_speaker_counts_cache(cache_key)
    if cached is not None:
        return SpeakerCountsRead(**cached)
    rows = _get_speaker_scope_rows(session=session, channel_id=channel_id, video_id=video_id)
    total = len(rows)
    unknown = 0
    identified = 0
    main = 0
    extras = 0

    for row in rows:
        total_time = float(row.get("total_speaking_time") or 0.0)
        if _is_unknown_speaker_name(row.get("name")):
            unknown += 1
            continue
        identified += 1
        if bool(row.get("is_extra")) or total_time < EXTRAS_THRESHOLD:
            extras += 1
        else:
            main += 1

    counts = SpeakerCountsRead(
        total=total,
        identified=identified,
        unknown=unknown,
        main=main,
        extras=extras,
    )
    _set_speaker_counts_cache(cache_key, counts.model_dump())
    return counts

@app.get("/speakers/{speaker_id}", response_model=SpeakerRead)
def read_speaker(speaker_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func
    
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    # Calculate total speaking time
    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == speaker_id)
    ).first()
    total_time = total_time_result or 0
    
    # Get embedding count
    emb_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0
    
    return SpeakerRead(
        id=speaker.id,
        channel_id=speaker.channel_id,
        name=speaker.name,
        thumbnail_path=speaker.thumbnail_path,
        is_extra=speaker.is_extra,
        total_speaking_time=round(total_time, 1),
        embedding_count=emb_count,
        created_at=speaker.created_at
    )

@app.get("/speakers/{speaker_id}/appearances", response_model=List[SpeakerEpisodeAppearanceRead])
def read_speaker_appearances(speaker_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func, case

    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    rows = session.exec(
        select(
            Video.id,
            Video.youtube_id,
            Video.title,
            Video.published_at,
            Video.thumbnail_url,
            func.count(TranscriptSegment.id).label("segment_count"),
            func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time).label("total_time"),
            func.min(TranscriptSegment.start_time).label("first_start"),
            func.max(TranscriptSegment.end_time).label("last_end"),
        )
        .join(TranscriptSegment, TranscriptSegment.video_id == Video.id)
        .where(TranscriptSegment.speaker_id == speaker_id)
        .group_by(Video.id)
        .order_by(
            case((Video.published_at.is_(None), 1), else_=0),
            Video.published_at.desc(),
            Video.id.desc(),
        )
    ).all()

    appearances: List[SpeakerEpisodeAppearanceRead] = []
    for row in rows:
        appearances.append(
            SpeakerEpisodeAppearanceRead(
                video_id=row[0],
                youtube_id=row[1],
                title=row[2],
                published_at=row[3],
                thumbnail_url=row[4],
                segment_count=int(row[5] or 0),
                total_speaking_time=round(float(row[6] or 0), 1),
                first_start_time=float(row[7] or 0),
                last_end_time=float(row[8] or 0),
            )
        )
    return appearances

@app.get("/speakers/{speaker_id}/profiles", response_model=List[SpeakerVoiceProfileRead])
def read_speaker_profiles(speaker_id: int, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    rows = session.exec(
        select(SpeakerEmbedding, Video)
        .join(Video, SpeakerEmbedding.source_video_id == Video.id, isouter=True)
        .where(SpeakerEmbedding.speaker_id == speaker_id)
        .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
    ).all()

    profiles: List[SpeakerVoiceProfileRead] = []
    for emb, source_video in rows:
        profiles.append(
            SpeakerVoiceProfileRead(
                id=emb.id,
                speaker_id=emb.speaker_id,
                source_video_id=emb.source_video_id,
                source_video_title=source_video.title if source_video else None,
                source_video_youtube_id=source_video.youtube_id if source_video else None,
                source_video_published_at=source_video.published_at if source_video else None,
                sample_start_time=emb.sample_start_time,
                sample_end_time=emb.sample_end_time,
                sample_text=emb.sample_text,
                created_at=emb.created_at,
            )
        )
    return profiles

@app.delete("/speakers/{speaker_id}/profiles/{profile_id}")
def delete_speaker_profile(speaker_id: int, profile_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func

    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile or profile.speaker_id != speaker_id:
        raise HTTPException(status_code=404, detail="Voice profile not found for this speaker")

    profile_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    if int(profile_count) <= 1:
        raise HTTPException(status_code=400, detail="Cannot remove the last voice profile")

    session.delete(profile)
    session.commit()
    _invalidate_speaker_query_caches()

    remaining = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    return {"status": "deleted", "profile_id": profile_id, "remaining_profiles": int(remaining)}

@app.post("/profiles/{profile_id}/reassign-segments")
def reassign_segments_for_profile(profile_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func, update

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    target_speaker = session.get(Speaker, profile.speaker_id)
    if not target_speaker:
        raise HTTPException(status_code=404, detail="Target speaker not found for this voice profile")

    total_matched = session.exec(
        select(func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(
            Video.channel_id == target_speaker.channel_id,
            TranscriptSegment.matched_profile_id == profile_id,
        )
    ).one() or 0

    to_update = session.exec(
        select(func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(
            Video.channel_id == target_speaker.channel_id,
            TranscriptSegment.matched_profile_id == profile_id,
            TranscriptSegment.speaker_id != target_speaker.id,
        )
    ).one() or 0

    if int(to_update) > 0:
        session.exec(
            update(TranscriptSegment)
            .where(
                TranscriptSegment.matched_profile_id == profile_id,
                TranscriptSegment.speaker_id != target_speaker.id,
                TranscriptSegment.video_id.in_(
                    select(Video.id).where(Video.channel_id == target_speaker.channel_id)
                ),
            )
            .values(speaker_id=target_speaker.id)
        )
        session.commit()
        _invalidate_speaker_query_caches()

    return {
        "status": "reassigned",
        "profile_id": int(profile_id),
        "target_speaker_id": int(target_speaker.id),
        "target_speaker_name": target_speaker.name,
        "channel_id": int(target_speaker.channel_id),
        "matched_segments": int(total_matched),
        "updated_segments": int(to_update),
    }


@app.post("/speakers/{speaker_id}/profiles/{profile_id}/move")
def move_speaker_profile(
    speaker_id: int,
    profile_id: int,
    req: MoveSpeakerProfileRequest,
    session: Session = Depends(get_session)
):
    source_speaker = session.get(Speaker, speaker_id)
    if not source_speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    profile = session.get(SpeakerEmbedding, profile_id)
    if not profile or profile.speaker_id != speaker_id:
        raise HTTPException(status_code=404, detail="Voice profile not found for this speaker")
    result = _move_profile_between_speakers(
        session,
        source_speaker,
        profile,
        target_speaker_id=req.target_speaker_id,
        new_speaker_name=req.new_speaker_name,
    )
    _invalidate_speaker_query_caches()
    return {
        "status": "moved",
        **result,
    }

@app.get("/speakers/{speaker_id}/samples", response_model=List[SpeakerSample])
def get_speaker_samples(speaker_id: int, count: int = 3, strategy: str = "random", session: Session = Depends(get_session)):
    """
    Get audio samples for a speaker.
    strategy: 'random' (default) or 'longest'
    """
    from sqlalchemy.sql import func
    
    # Base query joining Video to get metadata
    query = select(TranscriptSegment, Video.channel_id, Video.youtube_id)\
        .join(Video)\
        .where(
            TranscriptSegment.speaker_id == speaker_id,
            (TranscriptSegment.end_time - TranscriptSegment.start_time) > 2.0
        )
    
    if strategy == "longest":
        # Get the longest segments, up to count
        query = query.order_by((TranscriptSegment.end_time - TranscriptSegment.start_time).desc()).limit(count)
    else:
        # Get random segments
        query = query.order_by(func.random()).limit(count)
    
    results = session.exec(query).all()
    
    # Convert to response model
    samples = []
    for segment, channel_id, youtube_id in results:
        # Create a dictionary of the segment data and add the extra fields
        data = segment.model_dump()
        data["channel_id"] = channel_id
        data["youtube_id"] = youtube_id
        samples.append(SpeakerSample(**data))
        
    return samples

@app.post("/speakers/{speaker_id}/thumbnail/extract", response_model=SpeakerRead)
def extract_speaker_thumbnail(speaker_id: int, req: ExtractThumbnailRequest, session: Session = Depends(get_session)):
    # Validate existence quickly, then release the request session before the
    # long-running ffmpeg/yt-dlp extraction work to reduce SQLite lock time.
    if not session.get(Speaker, speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")

    try:
        # Use the global ingestion_service initialized in lifespan
        global ingestion_service
        if ingestion_service is None:
            from .services.ingestion import IngestionService
            ingestion_service = IngestionService()

        try:
            session.close()
        except Exception:
            pass

        # Extract and update
        thumb_path = ingestion_service.extract_frame_and_crop(req.video_id, req.timestamp, req.crop_coords)

        # SQLite can be briefly locked by the queue worker; retry a few times.
        from sqlalchemy.exc import OperationalError
        import time as _time

        last_error = None
        for attempt in range(5):
            try:
                with Session(engine) as write_session:
                    speaker = write_session.get(Speaker, speaker_id)
                    if not speaker:
                        raise HTTPException(status_code=404, detail="Speaker not found")
                    speaker.thumbnail_path = thumb_path
                    write_session.add(speaker)
                    write_session.commit()
                    _invalidate_speaker_query_caches()
                    write_session.refresh(speaker)
                    return read_speaker(speaker_id, write_session)
            except OperationalError as e:
                last_error = e
                if "database is locked" not in str(e).lower() or attempt == 4:
                    raise
                _time.sleep(0.2 * (attempt + 1))

        if last_error:
            raise last_error
        
    except Exception as e:
        # Don't print traceback to stderr as it causes issues in some environments
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.delete("/speakers/{speaker_id}/thumbnail", response_model=SpeakerRead)
def delete_speaker_thumbnail(speaker_id: int, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    old_path = speaker.thumbnail_path
    speaker.thumbnail_path = None
    session.add(speaker)
    session.commit()
    _invalidate_speaker_query_caches()
    session.refresh(speaker)

    # Best-effort cleanup of local image files we own.
    if old_path:
        try:
            path_str = str(old_path)
            local_file = None
            if path_str.startswith("/images/"):
                local_file = IMAGES_DIR / Path(path_str).name
            elif path_str.startswith("/thumbnails/"):
                local_file = THUMBNAILS_DIR / Path(path_str).relative_to("/thumbnails")
            if local_file and local_file.exists():
                local_file.unlink()
        except Exception:
            pass

    return read_speaker(speaker_id, session)

@app.post("/speakers/{speaker_id}/thumbnail", response_model=SpeakerRead)
async def upload_thumbnail(speaker_id: int, file: UploadFile = File(...), session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker: raise HTTPException(status_code=404, detail="Speaker not found")
    
    # Save file
    safe_name = f"speaker_{speaker_id}_{file.filename}"
    file_path = IMAGES_DIR / safe_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    speaker.thumbnail_path = f"/images/{safe_name}"
    session.add(speaker)
    session.commit()
    _invalidate_speaker_query_caches()
    session.refresh(speaker)
    return read_speaker(speaker_id, session)

@app.post("/speakers/{speaker_id}/thumbnail_base64", response_model=SpeakerRead)
def upload_thumbnail_base64(speaker_id: int, data: dict, session: Session = Depends(get_session)):
    # Expects {"image": "base64string..."}
    speaker = session.get(Speaker, speaker_id)
    if not speaker: raise HTTPException(status_code=404, detail="Speaker not found")
    
    b64_str = data.get("image")
    if not b64_str: raise HTTPException(status_code=400, detail="No image data")
    
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
        
    image_data = base64.b64decode(b64_str)
    filename = f"speaker_{speaker_id}_pasted.png"
    file_path = IMAGES_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(image_data)
        
    speaker.thumbnail_path = f"/images/{filename}"
    session.add(speaker)
    session.commit()
    _invalidate_speaker_query_caches()
    session.refresh(speaker)
    return read_speaker(speaker_id, session)

@app.patch("/speakers/{speaker_id}", response_model=SpeakerRead)
def update_speaker(speaker_id: int, data: dict, session: Session = Depends(get_session)):
    from sqlalchemy import func
    from sqlalchemy.exc import OperationalError
    wants_name = "name" in data
    wants_is_extra = "is_extra" in data
    new_name = None
    if wants_name:
        new_name = str(data.get("name") or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
    new_is_extra = bool(data.get("is_extra")) if wants_is_extra else None

    updated_speaker = None
    max_attempts = 24
    for attempt in range(max_attempts):
        speaker = session.get(Speaker, speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")

        if wants_name:
            speaker.name = new_name
        if wants_is_extra:
            speaker.is_extra = new_is_extra

        session.add(speaker)
        try:
            session.commit()
            session.refresh(speaker)
            updated_speaker = speaker
            break
        except OperationalError as e:
            session.rollback()
            is_locked = "database is locked" in str(e).lower()
            if not is_locked:
                raise HTTPException(status_code=500, detail=f"Failed to update speaker: {e}")
            if attempt >= (max_attempts - 1):
                raise HTTPException(status_code=503, detail="Database busy. Please retry in a moment.")
            time.sleep(min(0.2 * (attempt + 1), 1.5))

    if not updated_speaker:
        raise HTTPException(status_code=503, detail="Database busy. Please retry in a moment.")

    _invalidate_speaker_query_caches()
    
    # Calculate total speaking time for the response
    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == speaker_id)
    ).first()
    
    return SpeakerRead(
        id=updated_speaker.id,
        channel_id=updated_speaker.channel_id,
        name=updated_speaker.name,
        thumbnail_path=updated_speaker.thumbnail_path,
        is_extra=updated_speaker.is_extra,
        total_speaking_time=round(total_time_result or 0, 1),
        embedding_count=session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
        ).first() or 0,
        created_at=updated_speaker.created_at
    )

@app.post("/speakers/merge", response_model=SpeakerRead)
def merge_speakers(req: MergeRequest, session: Session = Depends(get_session)):
    """Merge multiple speakers into one target speaker.
    Reassigns all transcript segments and moves all embeddings to the target.
    Deletes the source speakers."""
    from sqlalchemy import func, update, delete
    
    target = session.get(Speaker, req.target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target speaker not found")
    
    if req.target_id in req.source_ids:
        raise HTTPException(status_code=400, detail="Target speaker cannot be in source list")
    
    # Only merge speakers in the same channel as target
    source_id_rows = session.exec(
        select(Speaker.id).where(
            Speaker.id.in_(req.source_ids),
            Speaker.channel_id == target.channel_id,
        )
    ).all()
    source_ids = [row[0] if isinstance(row, tuple) else row for row in source_id_rows]

    if not source_ids:
        raise HTTPException(status_code=400, detail="No valid source speakers found in target channel")

    try:
        # Use SQL updates/deletes to avoid ORM relationship synchronization nulling child FKs
        # (speakerembedding.speaker_id is NOT NULL).
        session.exec(
            update(TranscriptSegment)
            .where(TranscriptSegment.speaker_id.in_(source_ids))
            .values(speaker_id=req.target_id)
        )
        session.exec(
            update(SpeakerEmbedding)
            .where(SpeakerEmbedding.speaker_id.in_(source_ids))
            .values(speaker_id=req.target_id)
        )
        session.exec(
            delete(Speaker).where(Speaker.id.in_(source_ids))
        )
        session.commit()
        _invalidate_speaker_query_caches()
    except Exception:
        session.rollback()
        raise

    session.refresh(target)
    
    # Build response
    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == req.target_id)
    ).first()
    
    emb_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == req.target_id)
    ).first() or 0
    
    return SpeakerRead(
        id=target.id,
        channel_id=target.channel_id,
        name=target.name,
        thumbnail_path=target.thumbnail_path,
        is_extra=target.is_extra,
        total_speaking_time=round(total_time_result or 0, 1),
        embedding_count=emb_count,
        created_at=target.created_at
    )

@app.post("/videos/{video_id}/purge")
def purge_video(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")

    # verify no active job
    active_job = session.exec(select(Job).where(Job.video_id == video.id, Job.status.in_(PIPELINE_ACTIVE_STATUSES))).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Cannot purge video with active job {active_job.id} ({active_job.status})")

    # 1. Delete segments
    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    for s in segments:
        session.delete(s)
    funny_moments = session.exec(select(FunnyMoment).where(FunnyMoment.video_id == video_id)).all()
    for fm in funny_moments:
        session.delete(fm)
    
    # 2. Purge temp files
    ingestion_service.purge_artifacts(video.id)

    # 3. Reset video status
    try:
        if ingestion_service.get_audio_path(video).exists():
            video.status = "downloaded"
        else:
            video.status = "pending"
    except:
        video.status = "pending"
        
    video.processed = False
    session.add(video)
    session.commit()
    _invalidate_speaker_query_caches()
    return {"status": "purged", "deleted_segments": len(segments), "deleted_funny_moments": len(funny_moments)}

@app.post("/videos/{video_id}/redo-diarization")
def redo_diarization(video_id: int, session: Session = Depends(get_session)):
    """Delete segments and re-run diarization only, reusing the existing raw transcript."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Verify no active job
    active_job = session.exec(select(Job).where(
        Job.video_id == video.id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")

    # Verify raw transcript exists (so we don't have to re-transcribe)
    try:
        audio_path = ingestion_service.get_audio_path(video)
        safe_title = ingestion_service.sanitize_filename(video.title)
        raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
        if not raw_path.exists():
            raise HTTPException(status_code=400, detail="No raw transcript found. Use full redo instead.")
    except HTTPException:
        raise
    except:
        raise HTTPException(status_code=400, detail="Could not locate audio/transcript files. Use full redo instead.")

    # Backup existing transcript/funny rows so we can restore if this destructive
    # operation is interrupted (backend restart/crash) before rewrite completes.
    segments = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
    ).all()
    funny_rows = session.exec(
        select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
    ).all()

    backup_payload = {
        "video_id": int(video.id),
        "video_status": video.status,
        "video_processed": bool(video.processed),
        "saved_at": datetime.now().isoformat(),
        "segments": [
            {
                "speaker_id": s.speaker_id,
                "matched_profile_id": s.matched_profile_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "words": s.words,
            }
            for s in segments
        ],
        "funny_moments": [
            {
                "start_time": fm.start_time,
                "end_time": fm.end_time,
                "score": fm.score,
                "source": fm.source,
                "snippet": fm.snippet,
                "humor_summary": fm.humor_summary,
                "humor_confidence": fm.humor_confidence,
                "humor_model": fm.humor_model,
                "humor_explained_at": fm.humor_explained_at.isoformat() if fm.humor_explained_at else None,
                "created_at": fm.created_at.isoformat() if fm.created_at else None,
            }
            for fm in funny_rows
        ],
    }
    backup_path = ingestion_service._get_temp_redo_backup_path(video_id)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f, ensure_ascii=False)

    # Delete existing segments/funny moments after backup is written.
    for s in segments:
        session.delete(s)
    for fm in funny_rows:
        session.delete(fm)

    # Purge only the diarization temp file, keep raw transcript
    d_path = ingestion_service._get_temp_diarization_path(video_id)
    if d_path.exists():
        import os
        os.unlink(d_path)

    # Reset status and queue job
    video.status = "downloaded"
    video.processed = False
    session.add(video)

    payload = {
        "mode": "redo_diarization",
        "redo_diarization_backup_file": str(backup_path),
    }
    new_job = _enqueue_unique_job(session, video_id=video_id, job_type="process", payload=payload)
    _invalidate_speaker_query_caches()

    return {
        "status": "diarization_requeued",
        "deleted_segments": len(segments),
        "deleted_funny_moments": len(funny_rows),
        "job_id": new_job.id,
    }


@app.post("/channels/{channel_id}/redo-diarization")
def redo_channel_diarization(
    channel_id: int,
    dry_run: bool = True,
    processed_only: bool = True,
    include_muted: bool = False,
    limit: int = 0,
    session: Session = Depends(get_session),
):
    """Bulk re-queue diarization across channel videos using existing raw transcripts."""
    from sqlalchemy import func, delete

    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    query = select(Video).where(Video.channel_id == channel_id).order_by(Video.id.desc())
    videos = session.exec(query).all()
    if limit and limit > 0:
        videos = videos[:limit]

    active_statuses = ["queued", "running", "downloading", "transcribing", "diarizing", "paused"]
    result = {
        "channel_id": int(channel_id),
        "channel_name": channel.name,
        "dry_run": bool(dry_run),
        "counts": {
            "scanned": 0,
            "eligible": 0,
            "queued": 0,
            "skipped_active": 0,
            "skipped_no_raw_transcript": 0,
            "skipped_muted": 0,
            "skipped_unprocessed": 0,
            "errors": 0,
            "deleted_segments": 0,
            "deleted_funny_moments": 0,
        },
        "job_ids": [],
        "sample_skips": [],
    }

    for video in videos:
        result["counts"]["scanned"] += 1

        if not include_muted and bool(video.muted):
            result["counts"]["skipped_muted"] += 1
            continue

        if processed_only and not bool(video.processed):
            result["counts"]["skipped_unprocessed"] += 1
            continue

        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == video.id,
                Job.status.in_(active_statuses),
            )
        ).first()
        if active_job:
            result["counts"]["skipped_active"] += 1
            continue

        has_raw_transcript = False
        try:
            audio_path = ingestion_service.get_audio_path(video)
            safe_title = ingestion_service.sanitize_filename(video.title)
            raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
            has_raw_transcript = raw_path.exists()
        except Exception:
            has_raw_transcript = False

        if not has_raw_transcript:
            result["counts"]["skipped_no_raw_transcript"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": "no_raw_transcript",
                })
            continue

        result["counts"]["eligible"] += 1
        if dry_run:
            continue

        try:
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
            ).all()
            funny_rows = session.exec(
                select(FunnyMoment).where(FunnyMoment.video_id == video.id).order_by(FunnyMoment.start_time)
            ).all()
            seg_count = len(segments)
            funny_count = len(funny_rows)

            backup_payload = {
                "video_id": int(video.id),
                "video_status": video.status,
                "video_processed": bool(video.processed),
                "saved_at": datetime.now().isoformat(),
                "segments": [
                    {
                        "speaker_id": s.speaker_id,
                        "matched_profile_id": s.matched_profile_id,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "text": s.text,
                        "words": s.words,
                    }
                    for s in segments
                ],
                "funny_moments": [
                    {
                        "start_time": fm.start_time,
                        "end_time": fm.end_time,
                        "score": fm.score,
                        "source": fm.source,
                        "snippet": fm.snippet,
                        "humor_summary": fm.humor_summary,
                        "humor_confidence": fm.humor_confidence,
                        "humor_model": fm.humor_model,
                        "humor_explained_at": fm.humor_explained_at.isoformat() if fm.humor_explained_at else None,
                        "created_at": fm.created_at.isoformat() if fm.created_at else None,
                    }
                    for fm in funny_rows
                ],
            }
            backup_path = ingestion_service._get_temp_redo_backup_path(video.id)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_payload, f, ensure_ascii=False)

            if seg_count > 0:
                session.exec(delete(TranscriptSegment).where(TranscriptSegment.video_id == video.id))
            if funny_count > 0:
                session.exec(delete(FunnyMoment).where(FunnyMoment.video_id == video.id))

            d_path = ingestion_service._get_temp_diarization_path(video.id)
            if d_path.exists():
                try:
                    d_path.unlink()
                except Exception:
                    pass

            video.status = "downloaded"
            video.processed = False
            session.add(video)

            job = _enqueue_unique_job(
                session,
                video_id=video.id,
                job_type="process",
                payload={
                    "mode": "redo_diarization",
                    "redo_diarization_backup_file": str(backup_path),
                },
            )

            result["counts"]["queued"] += 1
            result["counts"]["deleted_segments"] += int(seg_count)
            result["counts"]["deleted_funny_moments"] += int(funny_count)
            result["job_ids"].append(int(job.id))
        except Exception as e:
            session.rollback()
            result["counts"]["errors"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": f"error: {e}",
                })

    if not dry_run and (result["counts"]["queued"] > 0 or result["counts"]["deleted_segments"] > 0):
        _invalidate_speaker_query_caches()

    return result


@app.post("/videos/{video_id}/redo-transcription")
def redo_transcription(video_id: int, session: Session = Depends(get_session)):
    """Purge transcript artifacts and re-run full transcription pipeline (transcribe + diarize)."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES)
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")

    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    for s in segments:
        session.delete(s)
    funny_moments = session.exec(select(FunnyMoment).where(FunnyMoment.video_id == video_id)).all()
    for fm in funny_moments:
        session.delete(fm)

    # Purge checkpoints + raw transcript so we force a new transcription pass.
    ingestion_service.purge_artifacts(video.id, delete_raw_transcript=True, delete_audio=False)

    try:
        if ingestion_service.get_audio_path(video).exists():
            video.status = "downloaded"
        else:
            video.status = "pending"
    except Exception:
        video.status = "pending"
    video.processed = False
    session.add(video)

    new_job = _enqueue_unique_job(session, video_id=video_id, job_type="process")
    _invalidate_speaker_query_caches()
    return {
        "status": "transcription_requeued",
        "deleted_segments": len(segments),
        "deleted_funny_moments": len(funny_moments),
        "job_id": new_job.id,
    }

# --- Video Controls ---

@app.patch("/videos/{video_id}/mute", response_model=Video)
def toggle_video_mute(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")
    
    video.muted = not video.muted
    session.add(video)
    session.commit()
    session.refresh(video)
    return video

# --- Jobs ---

def _job_queue_name(job_type: str) -> str:
    jt = (job_type or "").strip().lower()
    if jt in {"process", "diarize"}:
        return "pipeline"
    if jt in {"funny_detect", "funny_explain"}:
        return "funny"
    if jt == "youtube_metadata":
        return "youtube"
    if jt in {"clip_export_mp4", "clip_export_captions"}:
        return "clip"
    return "other"

@app.get("/jobs", response_model=List[JobRead])
def read_jobs(
    status: Optional[str] = None,
    channel_id: Optional[int] = None,
    job_type: Optional[str] = None,
    limit: int = 500,
    session: Session = Depends(get_session)
):
    from sqlalchemy import case
    from sqlalchemy.orm import selectinload
    query = select(Job).options(selectinload(Job.video))
    active_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    if status:
        query = query.where(Job.status == status).order_by(Job.created_at.desc())
    else:
        query = query.where(Job.status != "waiting_diarize")
        # Ensure active jobs are always included near the top even when limit truncates.
        query = query.order_by(
            case((Job.status.in_(active_like_statuses), 0), else_=1),
            Job.created_at.desc(),
        )
    if job_type:
        job_types = [jt.strip() for jt in str(job_type).split(",") if jt.strip()]
        if len(job_types) == 1:
            query = query.where(Job.job_type == job_types[0])
        elif len(job_types) > 1:
            query = query.where(Job.job_type.in_(job_types))
    if channel_id:
        query = query.join(Video).where(Video.channel_id == channel_id)
    safe_limit = max(1, min(limit, 2000))
    return session.exec(query.limit(safe_limit)).all()


@app.get("/jobs/queues/summary")
def get_job_queues_summary(session: Session = Depends(get_session)):
    jobs = session.exec(select(Job.id, Job.job_type, Job.status)).all()
    summary = {
        "pipeline": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "funny": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "youtube": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "clip": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
        "other": {"queued": 0, "running": 0, "paused": 0, "completed": 0, "failed": 0, "total": 0},
    }
    for row in jobs:
        # sqlite row can come back as tuple in this select mode
        job_type = row[1] if isinstance(row, tuple) else getattr(row, "job_type", "")
        status = row[2] if isinstance(row, tuple) else getattr(row, "status", "")
        if status == "waiting_diarize":
            continue
        q = _job_queue_name(job_type)
        bucket = summary[q]
        bucket["total"] += 1
        if status in bucket:
            bucket[status] += 1
        elif status in {"downloading", "transcribing", "diarizing"}:
            bucket["running"] += 1
    return summary

@app.get("/jobs/status")
def get_queue_status(session: Session = Depends(get_session)):
    """Get summary of job queue status"""
    running_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    running = session.exec(select(Job).where(Job.status.in_(running_like_statuses))).all()
    queued = session.exec(select(Job).where(Job.status == "queued")).all()
    paused = session.exec(select(Job).where(Job.status == "paused")).all()
    
    return {
        "running": len(running),
        "queued": len(queued),
        "paused": len(paused),
        "total_active": len(running) + len(queued) + len(paused)
    }


def _compute_pipeline_focus_counts(session: Session) -> dict:
    """Shared helper for pipeline focus count queries."""
    transcribe_active_statuses = ["running", "downloading", "transcribing"]
    diarize_active_statuses = ["running", "diarizing"]
    transcribe_active = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "process",
                Job.status.in_(transcribe_active_statuses),
            )
        ).one() or 0
    )
    transcribe_queued = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "process",
                Job.status == "queued",
            )
        ).one() or 0
    )
    diarize_active = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "diarize",
                Job.status.in_(diarize_active_statuses),
            )
        ).one() or 0
    )
    diarize_queued = int(
        session.exec(
            select(func.count(Job.id)).where(
                Job.job_type == "diarize",
                Job.status == "queued",
            )
        ).one() or 0
    )
    return {
        "transcribe_active": transcribe_active,
        "transcribe_queued": transcribe_queued,
        "diarize_active": diarize_active,
        "diarize_queued": diarize_queued,
        "auto_diarize_ready": transcribe_active == 0 and transcribe_queued == 0 and (diarize_active > 0 or diarize_queued > 0),
    }


@app.get("/jobs/pipeline/focus", response_model=PipelineFocusRead)
def get_pipeline_focus(session: Session = Depends(get_session)):
    counts = _compute_pipeline_focus_counts(session)
    return {"mode": ingestion_service.get_pipeline_focus_mode(), **counts}


@app.post("/jobs/pipeline/focus", response_model=PipelineFocusRead)
def set_pipeline_focus(payload: PipelineFocusUpdate, session: Session = Depends(get_session)):
    mode = ingestion_service.set_pipeline_focus_mode(payload.mode)
    paused_active_count = 0
    if mode == "diarize" and payload.pause_active_transcription:
        active_process_jobs = session.exec(
            select(Job).where(
                Job.job_type == "process",
                Job.status.in_(["running", "downloading", "transcribing"]),
            )
        ).all()
        for job in active_process_jobs:
            job.status = "paused"
            session.add(job)
            paused_active_count += 1
        if paused_active_count:
            session.commit()

    counts = _compute_pipeline_focus_counts(session)
    return {"mode": mode, **counts, "active_transcription_paused": paused_active_count}


@app.get("/settings/db-health")
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

@app.post("/jobs/{job_id}/pause")
def pause_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in PIPELINE_ACTIVE_STATUSES_CORE:
        raise HTTPException(status_code=400, detail="Cannot pause job in current status")
    
    job.status = "paused"
    session.add(job)
    session.commit()
    return {"status": "paused"}

@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")

    job.status = "queued"
    session.add(job)
    session.commit()

    # Worker will pick this up automatically
    return {"status": "resumed"}

@app.post("/jobs/{job_id}/move-to-top")
def move_job_to_top(job_id: int, session: Session = Depends(get_session)):
    """Move a queued/paused job to the front of the pending queue."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ["queued", "paused"]:
        raise HTTPException(status_code=400, detail="Only queued or paused jobs can be reordered")

    pending = session.exec(
        select(Job).where(
            Job.status.in_(["queued", "paused"]),
            Job.job_type == job.job_type,
            Job.id != job_id,
        ).order_by(Job.created_at.asc())
    ).all()

    oldest = pending[0].created_at if pending else datetime.now()
    # Queue worker consumes queued jobs in ascending created_at order.
    job.created_at = oldest - timedelta(microseconds=1)
    session.add(job)
    session.commit()
    session.refresh(job)
    return {"status": "moved_to_top", "job_id": job.id, "created_at": job.created_at.isoformat()}

@app.delete("/jobs/queue")
def clear_queue(session: Session = Depends(get_session)):
    """Delete all queued and paused jobs quickly and reliably."""
    from sqlalchemy import delete, func
    from sqlalchemy.exc import OperationalError

    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            clearable_statuses = ["queued", "paused", "waiting_diarize"]
            count = session.exec(
                select(func.count(Job.id)).where(Job.status.in_(clearable_statuses))
            ).one()
            deleted_count = int(count or 0)
            if deleted_count <= 0:
                return {"deleted": 0}

            session.exec(delete(Job).where(Job.status.in_(clearable_statuses)))
            session.commit()
            return {"deleted": deleted_count}
        except OperationalError as e:
            session.rollback()
            is_locked = "database is locked" in str(e).lower()
            if (not is_locked) or attempt >= (max_attempts - 1):
                raise HTTPException(status_code=503, detail="Failed to clear queue: database busy.")
            time.sleep(min(0.15 * (attempt + 1), 1.2))

    raise HTTPException(status_code=503, detail="Failed to clear queue: database busy.")

@app.delete("/jobs/history")
def clear_history(session: Session = Depends(get_session)):
    """Delete all completed and failed jobs"""
    jobs = session.exec(select(Job).where(Job.status.in_(["completed", "failed"]))).all()
    count = 0
    for job in jobs:
        session.delete(job)
        count += 1
    session.commit()
    return {"deleted": count}

@app.post("/jobs/{job_id}/resubmit")
def resubmit_job(job_id: int, session: Session = Depends(get_session)):
    """Resubmit a completed or failed job — creates a new job for the same video"""
    old_job = session.get(Job, job_id)
    if not old_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if old_job.status not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Can only resubmit completed or failed jobs")
    
    video_id = old_job.video_id
    
    # Check no active job already exists for this video
    active = session.exec(select(Job).where(
        Job.video_id == video_id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active:
        raise HTTPException(status_code=400, detail=f"Video already has an active job ({active.status})")
    
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Clear existing transcript data so re-processing starts fresh
    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    for s in segments:
        session.delete(s)
    
    # Purge temp checkpoint files
    ingestion_service.purge_artifacts(video_id)
    
    # Reset video status
    try:
        audio_path = ingestion_service.get_audio_path(video)
        video.status = "downloaded" if audio_path.exists() else "pending"
    except:
        video.status = "pending"
    video.processed = False
    session.add(video)
    
    # Create fresh job
    new_job = Job(video_id=video_id, job_type="process", status="queued")
    session.add(new_job)
    session.commit()
    session.refresh(new_job)
    
    return {"status": "resubmitted", "new_job_id": new_job.id}

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")

    # If running, we might need to kill it? 
    # For now, just remove from DB. The worker loop will handle orphaned process if robust.
    # Ideally, we flag it as cancelled.
    
    # Reset video status
    if job.video:
        video = job.video
        # Check if audio exists
        try:
            audio_path = ingestion_service.get_audio_path(video)
            if audio_path.exists():
                video.status = "downloaded"
            else:
                video.status = "pending"
        except Exception as e:
            print(f"Error checking audio path during cancel: {e}")
            video.status = "pending"
            
        video.processed = False # Ensure processed flag is reset
        session.add(video)

    payload = json.loads(job.payload_json) if job.payload_json else {}
    if job.job_type == "diarize":
        parent_job_id = int(payload.get("parent_job_id") or 0)
        if parent_job_id:
            parent_job = session.get(Job, parent_job_id)
            if parent_job:
                session.delete(parent_job)
    elif job.job_type == "process" and job.status == "waiting_diarize":
        child_job_id = int(payload.get("diarize_job_id") or 0)
        if child_job_id:
            child_job = session.get(Job, child_job_id)
            if child_job:
                session.delete(child_job)

    session.delete(job)
    session.commit()
    return {"status": "cancelled", "video_status": job.video.status if job.video else "unknown"}

@app.get("/settings", response_model=Settings)
def get_settings():
    llm_enabled = os.getenv("LLM_ENABLED")
    if llm_enabled is None:
        llm_enabled = os.getenv("OLLAMA_ENABLED", "false")
    return Settings(
        hf_token=os.getenv("HF_TOKEN") or "",
        transcription_engine=(os.getenv("TRANSCRIPTION_ENGINE") or "auto"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL") or "medium",
        transcription_compute_type=os.getenv("TRANSCRIPTION_COMPUTE_TYPE") or "int8_float16",
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
        youtube_oauth_client_id=os.getenv("YOUTUBE_OAUTH_CLIENT_ID") or "",
        youtube_oauth_client_secret=os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET") or "",
        youtube_oauth_redirect_uri=os.getenv("YOUTUBE_OAUTH_REDIRECT_URI") or "http://localhost:8000/auth/youtube/callback",
        youtube_publish_push_enabled=os.getenv("YOUTUBE_PUBLISH_PUSH_ENABLED", "false").lower() == "true",
        ytdlp_cookies_file=os.getenv("YTDLP_COOKIES_FILE") or "",
        ytdlp_cookies_from_browser=os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "",
        diarization_sensitivity=os.getenv("DIARIZATION_SENSITIVITY") or "balanced",
        speaker_match_threshold=float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.5")),
        funny_moments_max_saved=int(os.getenv("FUNNY_MOMENTS_MAX_SAVED", "25")),
        funny_moments_explain_batch_limit=int(os.getenv("FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", "12")),
        setup_wizard_completed=os.getenv("SETUP_WIZARD_COMPLETED", "false").lower() == "true",
    )

@app.post("/settings")
def update_settings(settings: Settings, session: Session = Depends(get_session)):
    """Update settings and reload models"""
    normalized_transcription_engine = (getattr(settings, "transcription_engine", "auto") or "auto").strip().lower()
    if normalized_transcription_engine not in {"auto", "whisper", "parakeet"}:
        normalized_transcription_engine = "auto"
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
    funny_moments_max_saved = max(1, min(int(getattr(settings, "funny_moments_max_saved", 25)), 200))
    funny_moments_explain_batch_limit = max(1, min(int(getattr(settings, "funny_moments_explain_batch_limit", 12)), 200))
    nvidia_nim_min_interval = max(0.0, min(float(getattr(settings, "nvidia_nim_min_request_interval_seconds", 2.5)), 30.0))
    youtube_redirect_uri = (getattr(settings, "youtube_oauth_redirect_uri", "") or "http://localhost:8000/auth/youtube/callback").strip()
    normalized_ollama_model = _normalize_ollama_model_ref(getattr(settings, "ollama_model", "") or "")

    # 1. Update .env file
    set_key(ENV_PATH, "HF_TOKEN", settings.hf_token)
    set_key(ENV_PATH, "TRANSCRIPTION_ENGINE", normalized_transcription_engine)
    set_key(ENV_PATH, "TRANSCRIPTION_MODEL", settings.transcription_model)
    set_key(ENV_PATH, "TRANSCRIPTION_COMPUTE_TYPE", settings.transcription_compute_type)
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
    set_key(ENV_PATH, "YOUTUBE_OAUTH_CLIENT_ID", settings.youtube_oauth_client_id)
    set_key(ENV_PATH, "YOUTUBE_OAUTH_CLIENT_SECRET", settings.youtube_oauth_client_secret)
    set_key(ENV_PATH, "YOUTUBE_OAUTH_REDIRECT_URI", youtube_redirect_uri)
    set_key(ENV_PATH, "YOUTUBE_PUBLISH_PUSH_ENABLED", str(bool(getattr(settings, "youtube_publish_push_enabled", False))).lower())
    set_key(ENV_PATH, "YTDLP_COOKIES_FILE", (getattr(settings, "ytdlp_cookies_file", "") or "").strip())
    set_key(ENV_PATH, "YTDLP_COOKIES_FROM_BROWSER", (getattr(settings, "ytdlp_cookies_from_browser", "") or "").strip())
    set_key(ENV_PATH, "DIARIZATION_SENSITIVITY", settings.diarization_sensitivity)
    set_key(ENV_PATH, "SPEAKER_MATCH_THRESHOLD", str(settings.speaker_match_threshold))
    set_key(ENV_PATH, "FUNNY_MOMENTS_MAX_SAVED", str(funny_moments_max_saved))
    set_key(ENV_PATH, "FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", str(funny_moments_explain_batch_limit))
    set_key(ENV_PATH, "SETUP_WIZARD_COMPLETED", str(bool(getattr(settings, "setup_wizard_completed", False))).lower())

    # 2. Update current environment
    os.environ["HF_TOKEN"] = settings.hf_token
    os.environ["TRANSCRIPTION_ENGINE"] = normalized_transcription_engine
    os.environ["TRANSCRIPTION_MODEL"] = settings.transcription_model
    os.environ["TRANSCRIPTION_COMPUTE_TYPE"] = settings.transcription_compute_type
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
    os.environ["YOUTUBE_OAUTH_CLIENT_ID"] = settings.youtube_oauth_client_id
    os.environ["YOUTUBE_OAUTH_CLIENT_SECRET"] = settings.youtube_oauth_client_secret
    os.environ["YOUTUBE_OAUTH_REDIRECT_URI"] = youtube_redirect_uri
    os.environ["YOUTUBE_PUBLISH_PUSH_ENABLED"] = str(bool(getattr(settings, "youtube_publish_push_enabled", False))).lower()
    os.environ["YTDLP_COOKIES_FILE"] = (getattr(settings, "ytdlp_cookies_file", "") or "").strip()
    os.environ["YTDLP_COOKIES_FROM_BROWSER"] = (getattr(settings, "ytdlp_cookies_from_browser", "") or "").strip()
    os.environ["DIARIZATION_SENSITIVITY"] = settings.diarization_sensitivity
    os.environ["SPEAKER_MATCH_THRESHOLD"] = str(settings.speaker_match_threshold)
    os.environ["FUNNY_MOMENTS_MAX_SAVED"] = str(funny_moments_max_saved)
    os.environ["FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT"] = str(funny_moments_explain_batch_limit)
    os.environ["SETUP_WIZARD_COMPLETED"] = str(bool(getattr(settings, "setup_wizard_completed", False))).lower()
    
    # 3. Reconfigure logging based on new verbose_logging setting
    configure_logging()
    
    # 4. Reload models in ingestion service
    if ingestion_service:
        print("Reloading models with new settings...")
        _purge_runtime_models(reason="settings_updated")
        
    return {"status": "updated"}

@app.post("/settings/test-transcription-engine")
def test_transcription_engine(request: TranscriptionEngineTestRequest):
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    engine = (request.engine or os.getenv("TRANSCRIPTION_ENGINE") or "auto").strip().lower()
    return ingestion_service.test_transcription_engine(engine)

@app.get("/youtube/oauth/status")
def youtube_oauth_status():
    cfg = _youtube_get_cfg()
    expiry = _youtube_parse_expiry(cfg["token_expiry"])
    connected = bool(cfg["refresh_token"] or cfg["access_token"])
    return {
        "configured": _youtube_oauth_is_configured(),
        "connected": connected,
        "channel_id": cfg["channel_id"] or None,
        "channel_title": cfg["channel_title"] or None,
        "redirect_uri": cfg["redirect_uri"],
        "scope": YOUTUBE_OAUTH_SCOPE,
        "token_expires_at": expiry.isoformat() if expiry else None,
        "push_enabled": cfg["push_enabled"],
    }


@app.post("/youtube/oauth/start")
def youtube_oauth_start():
    cfg = _youtube_get_cfg()
    if not _youtube_oauth_is_configured():
        raise HTTPException(status_code=400, detail="Configure YouTube OAuth client ID/secret and redirect URI in Settings first.")

    # Clear expired pending states.
    now_ts = time.time()
    for key, exp in list(youtube_oauth_pending_states.items()):
        if exp < now_ts:
            youtube_oauth_pending_states.pop(key, None)

    state = secrets.token_urlsafe(24)
    youtube_oauth_pending_states[state] = now_ts + 600  # 10 minutes
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": YOUTUBE_OAUTH_SCOPE,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    auth_url = f"{YOUTUBE_OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return {"auth_url": auth_url, "state": state}


@app.get("/auth/youtube/callback")
def youtube_oauth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    def _html_page(title: str, message: str, success: bool) -> str:
        color = "#0f766e" if success else "#b91c1c"
        icon = "Connected" if success else "Authorization Failed"
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;padding:24px;">
  <div style="max-width:640px;margin:0 auto;background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px;box-shadow:0 8px 30px rgba(15,23,42,.06)">
    <div style="font-size:18px;font-weight:700;color:{color};margin-bottom:8px">{icon}</div>
    <div style="font-size:14px;line-height:1.5;white-space:pre-wrap">{html.escape(message)}</div>
    <div style="margin-top:16px;font-size:12px;color:#64748b">You can close this window and return to Chatalogue.</div>
  </div>
  <script>setTimeout(function(){{ try {{ window.close(); }} catch(e) {{}} }}, 1200);</script>
</body></html>"""

    if error:
        return HTMLResponse(content=_html_page("YouTube OAuth Error", f"Google returned an error: {error}", False), status_code=400)
    if not code:
        return HTMLResponse(content=_html_page("YouTube OAuth Error", "Missing authorization code in callback.", False), status_code=400)
    if not state or youtube_oauth_pending_states.get(state, 0) < time.time():
        return HTMLResponse(content=_html_page("YouTube OAuth Error", "Invalid or expired OAuth state. Start the connection flow again.", False), status_code=400)
    youtube_oauth_pending_states.pop(state, None)

    try:
        token_data = _youtube_exchange_code_for_tokens(code)
        access_token = str(token_data.get("access_token") or "").strip()
        refresh_token = str(token_data.get("refresh_token") or "").strip() or (os.getenv("YOUTUBE_OAUTH_REFRESH_TOKEN") or "").strip()
        expires_in = int(token_data.get("expires_in") or 3600)
        if not access_token:
            raise RuntimeError("Google token response did not include an access token.")
        if not refresh_token:
            raise RuntimeError("Google token response did not include a refresh token. Try again with consent.")

        expiry = datetime.now() + timedelta(seconds=max(60, expires_in - 30))
        _set_env_persist("YOUTUBE_OAUTH_ACCESS_TOKEN", access_token)
        _set_env_persist("YOUTUBE_OAUTH_REFRESH_TOKEN", refresh_token)
        _set_env_persist("YOUTUBE_OAUTH_TOKEN_EXPIRY", expiry.isoformat())

        ch_info = _youtube_fetch_authenticated_channel_info()
        _set_env_persist("YOUTUBE_OAUTH_CHANNEL_ID", ch_info["channel_id"])
        _set_env_persist("YOUTUBE_OAUTH_CHANNEL_TITLE", ch_info["channel_title"])

        msg = f'Connected to YouTube channel "{ch_info["channel_title"]}" ({ch_info["channel_id"]}).'
        return HTMLResponse(content=_html_page("YouTube Connected", msg, True), status_code=200)
    except Exception as e:
        return HTMLResponse(content=_html_page("YouTube OAuth Error", str(e), False), status_code=500)


@app.post("/youtube/oauth/disconnect")
def youtube_oauth_disconnect():
    for key in [
        "YOUTUBE_OAUTH_ACCESS_TOKEN",
        "YOUTUBE_OAUTH_REFRESH_TOKEN",
        "YOUTUBE_OAUTH_TOKEN_EXPIRY",
        "YOUTUBE_OAUTH_CHANNEL_ID",
        "YOUTUBE_OAUTH_CHANNEL_TITLE",
    ]:
        _set_env_persist(key, "")
    return {"status": "disconnected"}


@app.post("/youtube/oauth/test")
def youtube_oauth_test():
    try:
        info = _youtube_fetch_authenticated_channel_info()
        return {"status": "ok", "channel_id": info["channel_id"], "channel_title": info["channel_title"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/settings/validate-token")
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

@app.post("/settings/test-ollama")
def test_ollama_connection():
    """Test Ollama connectivity, list available models, and verify the selected model works."""
    import httpx
    import time

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = _normalize_ollama_model_ref(os.getenv("OLLAMA_MODEL", "mistral"))

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
    model_found = any(_ollama_model_name_matches(m, ollama_model) for m in available_models)

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


@app.get("/settings/ollama/models")
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
            "current_model": _normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to query Ollama models: {e}",
            "models": [],
            "current_model": _normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
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
        "current_model": _normalize_ollama_model_ref((os.getenv("OLLAMA_MODEL") or "").strip()),
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

@app.post("/settings/test-nvidia-nim")
def test_nvidia_nim_connection():
    """Back-compat endpoint for NVIDIA NIM test."""
    return _test_hosted_provider("nvidia_nim")


@app.post("/settings/test-hosted-llm")
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

@app.post("/settings/ollama/pull-model")
def pull_ollama_model(req: OllamaPullRequest):
    """Pull the configured Ollama model if it is not already installed.
    Default behavior runs asynchronously to avoid long request timeouts."""
    import httpx

    ollama_url = (req.url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    ollama_model = _normalize_ollama_model_ref((req.model or os.getenv("OLLAMA_MODEL") or "mistral").strip())
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

    model_found = any(_ollama_model_name_matches(m, ollama_model) for m in available_models)
    if model_found:
        return {
            "status": "already_installed",
            "model": ollama_model,
            "available_models": available_models,
        }

    if wait_for_completion:
        _run_ollama_pull_job(ollama_url, ollama_model)
        key = _ollama_pull_job_key(ollama_url, ollama_model)
        with _ollama_pull_jobs_lock:
            job = dict(_ollama_pull_jobs.get(key) or {})
        status = str(job.get("status") or "")
        if status == "failed":
            raise HTTPException(status_code=500, detail=f"Failed to pull model '{ollama_model}': {job.get('error') or 'Unknown error'}")
        return {
            "status": "pulled" if status == "completed" else "pull_completed_unverified",
            "model": ollama_model,
            "available_models": job.get("available_models") or available_models,
            "ollama_response": job.get("ollama_response") or {},
        }

    key = _ollama_pull_job_key(ollama_url, ollama_model)
    with _ollama_pull_jobs_lock:
        existing = dict(_ollama_pull_jobs.get(key) or {})
        if existing.get("status") in {"queued", "running"}:
            return {
                "status": "already_running",
                "model": ollama_model,
                "job": existing,
            }
        _ollama_pull_jobs[key] = {
            "status": "queued",
            "started_at": time.time(),
            "updated_at": time.time(),
            "completed_at": None,
            "error": None,
            "available_models": available_models[:2000],
        }

    t = threading.Thread(
        target=_run_ollama_pull_job,
        args=(ollama_url, ollama_model),
        daemon=True,
        name=f"ollama-pull-{int(time.time())}"
    )
    t.start()
    with _ollama_pull_jobs_lock:
        job = dict(_ollama_pull_jobs.get(key) or {})
    return {
        "status": "pulling_started",
        "model": ollama_model,
        "job": job,
    }


@app.get("/settings/ollama/pull-status")
def get_ollama_pull_status(url: Optional[str] = None, model: Optional[str] = None):
    ollama_url = (url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    if not model:
        with _ollama_pull_jobs_lock:
            jobs = [
                {"key": k, **v}
                for k, v in _ollama_pull_jobs.items()
                if k.startswith(f"{ollama_url.lower()}|")
            ]
        jobs.sort(key=lambda j: float(j.get("updated_at") or j.get("started_at") or 0), reverse=True)
        return {"status": "ok", "jobs": jobs[:20]}

    normalized_model = _normalize_ollama_model_ref(model)
    key = _ollama_pull_job_key(ollama_url, normalized_model)
    with _ollama_pull_jobs_lock:
        job = dict(_ollama_pull_jobs.get(key) or {})

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


@app.get("/settings/ollama/hardware-recommendation")
def get_ollama_hardware_recommendation(objective: str = "balanced"):
    hardware = _detect_gpu_hardware()
    recommendation = _recommend_ollama_for_hardware(hardware.get("gpu_vram_gb"), objective=objective)
    return {
        "status": "ok",
        "hardware": hardware,
        "recommendation": recommendation,
    }

@app.get("/system/setup-status")
def get_setup_status():
    """Check if initial setup wizard has been completed."""
    return {
        "setup_completed": os.getenv("SETUP_WIZARD_COMPLETED", "false").lower() == "true",
        "hf_token_set": bool((os.getenv("HF_TOKEN") or "").strip()),
    }


def _repo_root_path() -> Path:
    # backend/src/main.py -> project root (../../)
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _safe_stdout(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout or "").strip()


def _safe_stderr(proc: subprocess.CompletedProcess) -> str:
    return (proc.stderr or "").strip()


def _purge_runtime_models(reason: str = "manual_restart") -> dict:
    if not ingestion_service:
        return {"ok": False, "detail": "ingestion service unavailable"}
    try:
        if hasattr(ingestion_service, "purge_loaded_models"):
            details = ingestion_service.purge_loaded_models(reason=reason)
            return {"ok": True, **(details or {})}
    except Exception as e:
        return {"ok": False, "detail": str(e)}

    # Backward-compatible fallback path.
    try:
        ingestion_service.diarization_pipeline = None
        ingestion_service.embedding_model = None
        ingestion_service.embedding_inference = None
        ingestion_service.whisper_model = None
        ingestion_service.parakeet_model = None
        ingestion_service._whisper_compute_type = None
        ingestion_service._force_float32 = False
        ingestion_service._cuda_unhealthy_reason = None
        ingestion_service._cuda_unhealthy_since = None
        ingestion_service._parakeet_dynamic_batch_cap = None
        ingestion_service.device = None
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def _get_system_version_info(check_remote: bool = True) -> dict:
    repo_root = _repo_root_path()
    info = {
        "status": "ok",
        "app_version": os.getenv("CHATALOGUE_VERSION", "").strip() or None,
        "repo_path": str(repo_root),
        "git": {
            "available": False,
            "is_repo": False,
            "branch": None,
            "head": None,
            "head_short": None,
            "remote_head": None,
            "remote_head_short": None,
            "ahead_count": 0,
            "behind_count": 0,
            "dirty": False,
            "update_available": False,
            "error": None,
            "checked_remote": bool(check_remote),
        },
    }

    # Ensure git binary is available.
    git_v = _run_git(["--version"], repo_root, timeout=5)
    if git_v.returncode != 0:
        info["git"]["error"] = "git is not available in PATH"
        return info
    info["git"]["available"] = True

    # Ensure this directory is a git repo.
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], repo_root, timeout=5)
    if inside.returncode != 0 or _safe_stdout(inside).lower() != "true":
        info["git"]["error"] = "not a git work tree"
        return info
    info["git"]["is_repo"] = True

    branch_p = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root, timeout=5)
    if branch_p.returncode == 0:
        info["git"]["branch"] = _safe_stdout(branch_p) or None

    head_p = _run_git(["rev-parse", "HEAD"], repo_root, timeout=5)
    if head_p.returncode == 0:
        info["git"]["head"] = _safe_stdout(head_p) or None

    head_short_p = _run_git(["rev-parse", "--short", "HEAD"], repo_root, timeout=5)
    if head_short_p.returncode == 0:
        info["git"]["head_short"] = _safe_stdout(head_short_p) or None

    dirty_p = _run_git(["status", "--porcelain"], repo_root, timeout=8)
    if dirty_p.returncode == 0:
        info["git"]["dirty"] = bool(_safe_stdout(dirty_p))

    if not check_remote:
        return info

    branch = info["git"]["branch"] or "main"
    fetch_p = _run_git(["fetch", "--quiet", "origin", branch], repo_root, timeout=20)
    if fetch_p.returncode != 0:
        info["git"]["error"] = _safe_stderr(fetch_p) or "git fetch failed"
        return info

    remote_ref = f"origin/{branch}"
    remote_head_p = _run_git(["rev-parse", remote_ref], repo_root, timeout=5)
    if remote_head_p.returncode == 0:
        info["git"]["remote_head"] = _safe_stdout(remote_head_p) or None

    remote_head_short_p = _run_git(["rev-parse", "--short", remote_ref], repo_root, timeout=5)
    if remote_head_short_p.returncode == 0:
        info["git"]["remote_head_short"] = _safe_stdout(remote_head_short_p) or None

    behind_p = _run_git(["rev-list", "--count", f"HEAD..{remote_ref}"], repo_root, timeout=8)
    if behind_p.returncode == 0:
        try:
            info["git"]["behind_count"] = int(_safe_stdout(behind_p) or "0")
        except Exception:
            info["git"]["behind_count"] = 0

    ahead_p = _run_git(["rev-list", "--count", f"{remote_ref}..HEAD"], repo_root, timeout=8)
    if ahead_p.returncode == 0:
        try:
            info["git"]["ahead_count"] = int(_safe_stdout(ahead_p) or "0")
        except Exception:
            info["git"]["ahead_count"] = 0

    info["git"]["update_available"] = bool(info["git"]["behind_count"] > 0)
    return info


@app.get("/system/version")
def get_system_version(check_remote: bool = True):
    try:
        return _get_system_version_info(check_remote=check_remote)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "git": {
                "update_available": False,
                "behind_count": 0,
                "ahead_count": 0,
                "error": str(e),
            },
        }

@app.post("/system/restart")
def restart_server():
    """Purge loaded models and trigger a server restart when reload mode is active."""
    import time as _time

    purge_result = _purge_runtime_models(reason="system_restart")
    main_file = Path(__file__)
    main_file.touch()
    return {
        "status": "restarting",
        "model_purge": purge_result,
        "touched": str(main_file),
        "timestamp": _time.time(),
    }


@app.post("/system/update")
def update_and_restart_server():
    """Pull latest code from origin/<branch> (fast-forward only) and trigger restart."""
    import time as _time

    info = _get_system_version_info(check_remote=True)
    git = info.get("git", {}) if isinstance(info, dict) else {}

    if not git.get("available") or not git.get("is_repo"):
        raise HTTPException(status_code=400, detail=f"Update unavailable: {git.get('error') or 'git not ready'}")

    if git.get("dirty"):
        raise HTTPException(status_code=409, detail="Local repository has uncommitted changes; refusing auto-update.")

    branch = str(git.get("branch") or "main")
    repo_root = _repo_root_path()
    old_head = str(git.get("head") or "")
    pull_p = _run_git(["pull", "--ff-only", "origin", branch], repo_root, timeout=45)
    if pull_p.returncode != 0:
        detail = _safe_stderr(pull_p) or _safe_stdout(pull_p) or "git pull failed"
        raise HTTPException(status_code=500, detail=f"Update failed: {detail[:500]}")

    # Re-read HEAD after pull.
    new_head_p = _run_git(["rev-parse", "HEAD"], repo_root, timeout=5)
    new_head = _safe_stdout(new_head_p) if new_head_p.returncode == 0 else old_head
    updated = bool(new_head and old_head and new_head != old_head)

    purge_result = _purge_runtime_models(reason="system_update")

    # Trigger existing restart mechanism.
    main_file = Path(__file__)
    main_file.touch()
    return {
        "status": "restarting",
        "updated": updated,
        "model_purge": purge_result,
        "old_head": old_head or None,
        "new_head": new_head or None,
        "branch": branch,
        "touched": str(main_file),
        "timestamp": _time.time(),
    }

@app.get("/system/worker-status")
def get_worker_status():
    """Check queue workers and process heartbeat health."""
    import time

    heartbeat_file = Path(__file__).parent.parent / "data" / "worker_heartbeat"
    worker_alive = {name: bool(t and t.is_alive()) for name, t in (worker_threads or {}).items()}
    if not heartbeat_file.exists():
        overall = "offline" if not any(worker_alive.values()) else "stalled"
        return {"status": overall, "last_heartbeat": None, "workers": worker_alive}

    try:
        last_heartbeat = float(heartbeat_file.read_text().strip())
        age = time.time() - last_heartbeat

        # If heartbeat is older than 30 seconds, process worker is stalled.
        heartbeat_status = "online" if age < 30 else "stalled"
        if not worker_alive:
            status = heartbeat_status
        elif heartbeat_status == "online" and all(worker_alive.values()):
            status = "online"
        else:
            status = "stalled"
        return {
            "status": status,
            "last_heartbeat": last_heartbeat,
            "age_seconds": age,
            "workers": worker_alive,
            "heartbeat_status": heartbeat_status,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
