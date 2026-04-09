from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Optional
from datetime import datetime, timedelta
from sqlmodel import Session, select
from sqlalchemy import func, text
from pathlib import Path
from collections import Counter, deque
import importlib.util
import shutil
import base64
import threading
import atexit
import os
import sys
import json
import pickle
import time
import html
import hashlib
import secrets
import math
import re
import logging
import subprocess
import ipaddress
import socket
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
import psutil
from dotenv import set_key, load_dotenv
from .schemas import (
    ChannelOverviewRead, ChannelBatchPublishRequest,
    VideoListItemRead,
    TranscriptSearchPage, TranscriptSearchItemRead,
    AssignSpeakerRequest, SegmentTextUpdateRequest, TranscriptQualityRead, TranscriptQualitySnapshotRead, TranscriptRunRead, TranscriptRollbackOptionRead, TranscriptRestoreResponse, TranscriptGoldWindowUpsertRequest, TranscriptGoldWindowRead, TranscriptEvaluationResultRead, TranscriptEvaluationReviewRequest, TranscriptEvaluationReviewRead, TranscriptEvaluationBatchResponse, TranscriptEvaluationSummaryRead, TranscriptDiarizationConfigBenchmarkRead, TranscriptOptimizeDryRunRequest, TranscriptOptimizeDryRunResponse, TranscriptRepairQueueRequest, TranscriptRepairQueueResponse, TranscriptRepairResultRead, TranscriptRepairBulkQueueRequest, TranscriptRepairBulkQueueResponse, TranscriptDiarizationRebuildQueueRequest, TranscriptDiarizationBenchmarkRequest, TranscriptDiarizationRebuildQueueResponse, TranscriptDiarizationRebuildBulkQueueRequest, TranscriptDiarizationRebuildBulkQueueResponse, TranscriptRetranscriptionQueueRequest, TranscriptRetranscriptionQueueResponse, TranscriptRetranscriptionBulkQueueRequest, TranscriptRetranscriptionBulkQueueResponse, TranscriptOptimizationCampaignCreateRequest, TranscriptOptimizationCampaignRead, TranscriptOptimizationCampaignItemRead, TranscriptOptimizationCampaignExecuteResponse, TranscriptOptimizationCampaignDeleteResponse,
    ClipCreate, ClipRead, ChannelClipRead, ClipCaptionExportRequest, ClipExportPresetRequest,
    ClipYoutubeUploadRequest, ClipBatchYoutubeUploadRequest,
    SpeakerRead, SpeakerCountsRead, SpeakerOverviewRead, SpeakerEpisodeAppearanceRead, SpeakerVoiceProfileRead, SpeakerMergeSuggestionRead,
    MoveSpeakerProfileRequest, SpeakerSample, ExtractThumbnailRequest, MergeRequest,
    AvatarCreateRequest, AvatarUpdateRequest, AvatarRead, AvatarSectionSummaryRead, AvatarWorkbenchSpeakerRead, AvatarWorkbenchRead,
    AvatarPersonalityDatasetExampleRead, AvatarPersonalityDatasetRead, AvatarPersonalityDatasetPageRead, AvatarPersonalityExampleStateRequest, AvatarPersonalityJudgePassRequest, AvatarPersonalityJudgeStatusRead, AvatarPersonalityJudgeFeedItemRead, AvatarPersonalityLongFormSampleRead, AvatarPersonalityLongFormConfigRead, AvatarPersonalityLongFormPageRead, AvatarPersonalityLongFormSampleStateRequest, AvatarPersonalityLongFormConfigUpdateRequest, AvatarPersonalityTrainingConfigRead, AvatarPersonalityTrainingConfigUpdateRequest, AvatarPersonalityTrainingDatasetProfileRead, AvatarPersonalityTrainingPlanRead, AvatarPersonalityTrainingPackageRead, AvatarPersonalityTrainRequest, AvatarPersonalityTrainingStatusRead, AvatarPersonalitySnapshotRead, AvatarPersonalitySnapshotSelectRequest, AvatarPersonalitySnapshotCleanupRequest, AvatarPersonalitySnapshotDeleteRequest, AvatarPersonalityTestChatRequest, AvatarPersonalityTestChatResponse, AvatarPersonalityFitCheckRequest, AvatarPersonalityFitCheckResponse, AvatarPersonalityFitCheckPromptResultRead, AvatarPersonalityBaseModelCandidateRead, AvatarPersonalityBaseModelSupportRead, AvatarPersonalityBaseModelDownloadRequest, AvatarPersonalityTrainingReadinessRead,
    JobRead, PipelineFocusRead, PipelineFocusUpdate,
    Settings, OllamaPullRequest, TranscriptionEngineTestRequest,
    YouTubeDataApiTestRequest, YouTubeDataApiTestResult,
    CleanupWorkbenchRead, CleanupWorkbenchRunCandidateRequest, CleanupWorkbenchSelectCandidateRequest,
    ClearVoiceInstallInfo, ClearVoiceTestResult,
    VoiceFixerUseCleanedRequest, VoiceFixerSettingsUpdateRequest, VoiceFixerInstallInfo, VoiceFixerTestResult,
    ReconstructionUseForPlaybackRequest, ReconstructionSettingsUpdateRequest,
    ReconstructionInstallInfo, ReconstructionTestResult, ReconstructionWorkbenchRead,
    ReconstructionSpeakerTestRequest, ReconstructionSpeakerTestResult,
    ReconstructionWorkbenchSampleCleanupRequest, ReconstructionWorkbenchSampleStateRequest,
    ReconstructionWorkbenchAddSampleRequest, ReconstructionWorkbenchSpeakerApprovalRequest,
    ReconstructionSegmentPreviewRequest, ReconstructionSegmentPreviewResult,
    WorkbenchTaskProgressRead, UploadedPlaybackSourceRequest,
    ExternalShareStartRequest, ExternalShareStatus, ExternalShareAuditEntry,
    SemanticSearchRequest, SemanticSearchPage, SemanticSearchHit, ContextChunk,
    SemanticIndexStatus, SemanticIndexRebuildResponse,
    EpisodeCloneCandidateRead, EpisodeCloneContextHitRead, EpisodeCloneEngineRead, EpisodeCloneConceptsRequest, EpisodeCloneConceptsResponse, EpisodeCloneGenerateRequest, EpisodeCloneGenerateResponse, EpisodeCloneJobRead,
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

from .db.database import create_db_and_tables, engine, get_db_metrics_snapshot, IS_POSTGRES, Channel, Video, Speaker, SpeakerEmbedding, Avatar, AvatarPersonalityProfile, AvatarAppearanceProfile, AvatarVoiceProfile, TranscriptSegment, TranscriptSegmentRead, TranscriptSegmentRevision, TranscriptSegmentRevisionRead, TranscriptRun, TranscriptQualitySnapshot, TranscriptGoldWindow, TranscriptEvaluationResult, TranscriptEvaluationReview, TranscriptOptimizationCampaign, TranscriptOptimizationCampaignItem, Job, Clip, ClipExportArtifact, ClipExportArtifactRead, FunnyMoment, FunnyMomentRead, VideoDescriptionRevision, VideoDescriptionRevisionRead, TranscriptChunkEmbedding
from .services.ingestion import IngestionService
from .services import semantic_search as sem_svc
from .services import episode_clone as clone_svc
from .services.speaker_merge import merge_speakers_in_session

ingestion_service = None
worker_threads: dict[str, threading.Thread] = {}
prefetch_thread = None
_avatar_judge_runs_lock = threading.Lock()
_avatar_judge_stop_events: dict[int, threading.Event] = {}
_avatar_judge_threads: dict[int, threading.Thread] = {}
_avatar_training_runs_lock = threading.Lock()
_avatar_training_processes: dict[int, subprocess.Popen] = {}
_avatar_chat_model_lock = threading.Lock()
_avatar_chat_models: dict[int, tuple[str, object, object]] = {}
_avatar_hf_model_downloads_lock = threading.Lock()
_avatar_hf_model_downloads: dict[str, dict[str, object]] = {}

CLEARVOICE_PACKAGE_SPEC = "clearvoice==0.1.2"
VOICEFIXER_PACKAGE_SPEC = "voicefixer==0.1.3"
RECONSTRUCTION_PACKAGE_SPEC = "qwen-tts==0.1.1"
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
MANUAL_MEDIA_DIR = Path(__file__).parent.parent / "data" / "manual_media"
MANUAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
AVATARS_DIR = Path(__file__).parent.parent / "data" / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
ENV_PATH = Path(__file__).parent.parent / ".env"
SHARE_RUNTIME_DIR = BACKEND_RUNTIME_DIR / "external_share"
SHARE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
SHARE_AUDIT_LOG_PATH = SHARE_RUNTIME_DIR / "audit.log"
SHARE_EVENT_LOG_PATH = SHARE_RUNTIME_DIR / "events.log"
SHARE_TOKEN_HEADER = "x-chatalogue-share-token"
SHARE_PASSWORD_HEADER = "x-chatalogue-share-password"
SHARE_COOKIE_TOKEN = "chatalogue_share_token"
SHARE_COOKIE_PASSWORD = "chatalogue_share_password"

external_share_lock = threading.RLock()
external_share_audit_entries: deque[dict] = deque(maxlen=300)
external_share_state: dict[str, object] = {
    "active": False,
    "mode": "off",
    "enable_tunnel": False,
    "tunnel_provider": None,
    "started_at": None,
    "expires_at": None,
    "token": None,
    "password": None,
    "ip_allowlist": [],
    "frontend_local_url": None,
    "api_local_url": None,
    "frontend_lan_url": None,
    "api_lan_url": None,
    "frontend_public_url": None,
    "api_public_url": None,
    "share_url": None,
    "processes": {},
    "cloudflared_available": False,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_service, worker_threads, prefetch_thread, backend_instance_lock
    monitor_stop_event: threading.Event | None = None

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
        try:
            ingestion_service.cleanup_orphaned_active_videos()
        except Exception as e:
            print(f"Startup orphan-video cleanup failed: {e}")
        try:
            ingestion_service.cleanup_orphaned_channel_syncs()
        except Exception as e:
            print(f"Startup orphan-channel-sync cleanup failed: {e}")

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

        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=ingestion_service.monitor_channels_loop,
            args=(monitor_stop_event,),
            daemon=True,
            name="channel-monitor-worker",
        )
        monitor_thread.start()
        print("Active channel monitor thread started.")

        yield
    finally:
        try:
            if monitor_stop_event is not None:
                monitor_stop_event.set()
        except Exception:
            pass
        _release_backend_lock()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def external_share_guard(request: Request, call_next):
    _ensure_external_share_not_expired()

    path = request.url.path or "/"
    if path.startswith("/share/public-status") or path.startswith("/share/launch/"):
        return await call_next(request)

    with external_share_lock:
        active = bool(external_share_state.get("active"))
        expected_token = str(external_share_state.get("token") or "")
        expected_password = str(external_share_state.get("password") or "")
        allowlist = list(external_share_state.get("ip_allowlist") or [])

    if not active:
        return await call_next(request)

    client_ip = _resolve_client_ip(request)
    if _is_loopback_host(client_ip) or request.method.upper() == "OPTIONS":
        return await call_next(request)

    if not _client_ip_allowed(client_ip, allowlist):
        _append_share_audit(action="request_denied", allowed=False, reason="ip_not_allowed", client_ip=client_ip, path=path)
        return _external_share_error_response(request, 403, "This IP is not permitted for the current external share session.", "share_ip_not_allowed")

    provided_token, provided_password = _get_external_share_credentials(request)
    if not expected_token or provided_token != expected_token:
        _append_share_audit(action="request_denied", allowed=False, reason="invalid_token", client_ip=client_ip, path=path)
        return _external_share_error_response(request, 401, "A valid share token is required.", "share_token_required")

    if expected_password and provided_password != expected_password:
        _append_share_audit(action="request_denied", allowed=False, reason="invalid_password", client_ip=client_ip, path=path)
        return _external_share_error_response(request, 401, "A valid share password is required.", "share_password_required")

    response = await call_next(request)
    if provided_token and request.cookies.get(SHARE_COOKIE_TOKEN) != provided_token:
        response.set_cookie(SHARE_COOKIE_TOKEN, provided_token, httponly=True, samesite="lax")
    if expected_password and provided_password and request.cookies.get(SHARE_COOKIE_PASSWORD) != provided_password:
        response.set_cookie(SHARE_COOKIE_PASSWORD, provided_password, httponly=True, samesite="lax")
    _append_share_audit(action="request_allowed", allowed=True, reason="ok", client_ip=client_ip, path=path)
    return response

# Mount static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")
app.mount("/manual-media", StaticFiles(directory=MANUAL_MEDIA_DIR), name="manual-media")


def get_session():
    with Session(engine) as session:
        yield session


def _utc_now() -> datetime:
    return datetime.utcnow()


def _append_share_event(message: str) -> None:
    timestamp = datetime.now().isoformat()
    try:
        with SHARE_EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    except Exception:
        pass


def _append_share_audit(*, action: str, allowed: bool, reason: str | None = None, client_ip: str | None = None, path: str | None = None) -> None:
    entry = {
        "at": datetime.now().isoformat(),
        "action": action,
        "allowed": bool(allowed),
        "reason": reason,
        "client_ip": client_ip,
        "path": path,
    }
    with external_share_lock:
        external_share_audit_entries.appendleft(entry)
    try:
        with SHARE_AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _snapshot_external_share_state(include_secrets: bool = False) -> dict:
    with external_share_lock:
        processes = external_share_state.get("processes") or {}
        data = {
            "active": bool(external_share_state.get("active")),
            "mode": str(external_share_state.get("mode") or "off"),
            "enable_tunnel": bool(external_share_state.get("enable_tunnel")),
            "tunnel_provider": external_share_state.get("tunnel_provider"),
            "started_at": external_share_state.get("started_at"),
            "expires_at": external_share_state.get("expires_at"),
            "frontend_local_url": external_share_state.get("frontend_local_url"),
            "api_local_url": external_share_state.get("api_local_url"),
            "frontend_lan_url": external_share_state.get("frontend_lan_url"),
            "api_lan_url": external_share_state.get("api_lan_url"),
            "frontend_public_url": external_share_state.get("frontend_public_url"),
            "api_public_url": external_share_state.get("api_public_url"),
            "share_url": external_share_state.get("share_url"),
            "token_required": True,
            "password_required": bool(external_share_state.get("password")),
            "ip_allowlist": list(external_share_state.get("ip_allowlist") or []),
            "cloudflared_available": bool(external_share_state.get("cloudflared_available")),
            "audit_log_path": str(SHARE_AUDIT_LOG_PATH),
            "audit_entries": [ExternalShareAuditEntry(**entry) for entry in list(external_share_audit_entries)],
            "process_labels": sorted(processes.keys()),
        }
        if include_secrets:
            data["token"] = external_share_state.get("token")
    return data


def _build_share_destination_url(frontend_url: str | None, api_url: str | None, token: str | None) -> str | None:
    frontend_value = str(frontend_url or "").strip()
    api_value = str(api_url or "").strip()
    token_value = str(token or "").strip()
    if not frontend_value or not api_value or not token_value:
        return None
    params = {
        "api_base": api_value,
        "share_token": token_value,
    }
    return f"{frontend_value}?{urllib.parse.urlencode(params)}"


def _build_share_launch_url(api_url: str | None, token: str | None) -> str | None:
    api_value = str(api_url or "").strip().rstrip("/")
    token_value = str(token or "").strip()
    if not api_value or not token_value:
        return None
    return f"{api_value}/share/launch/{urllib.parse.quote(token_value, safe='')}"


def _render_share_launch_page(*, destination_url: str | None, title: str, message: str, status_code: int) -> HTMLResponse:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    safe_destination = html.escape(destination_url or "")
    auto_redirect = ""
    link_html = ""
    if destination_url:
        auto_redirect = f'<meta http-equiv="refresh" content="0;url={safe_destination}">'
        link_html = f'<p><a href="{safe_destination}">Continue to shared Chatalogue</a></p>'
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  {auto_redirect}
  <style>
    body {{ font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; }}
    main {{ max-width: 560px; margin: 8vh auto; background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08); }}
    h1 {{ font-size: 20px; margin: 0 0 10px; }}
    p {{ line-height: 1.5; margin: 0 0 12px; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p>{safe_message}</p>
    {link_html}
  </main>
</body>
</html>"""
    return HTMLResponse(content=content, status_code=status_code)


def _get_cloudflared_binary() -> str | None:
    candidates = [
        os.getenv("CLOUDFLARED_BIN") or "",
        str(Path(__file__).parent.parent.parent / "bin" / "cloudflared.exe"),
        str(Path(__file__).parent.parent.parent / "bin" / "cloudflared"),
        shutil.which("cloudflared") or "",
    ]
    for candidate in candidates:
        path = (candidate or "").strip()
        if path and Path(path).exists():
            return path
    return None


def _refresh_cloudflared_availability() -> bool:
    available = bool(_get_cloudflared_binary())
    with external_share_lock:
        external_share_state["cloudflared_available"] = available
    return available


def _cloudflared_install_target() -> dict[str, str | bool]:
    platform_name = "windows" if os.name == "nt" else ("macos" if sys.platform == "darwin" else sys.platform)
    winget_path = shutil.which("winget") or ""
    brew_path = shutil.which("brew") or ""
    if platform_name == "windows":
        return {
            "platform": platform_name,
            "package_manager": "winget" if winget_path else "",
            "package_manager_available": bool(winget_path),
            "download_url": "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        }
    if platform_name == "macos":
        return {
            "platform": platform_name,
            "package_manager": "brew" if brew_path else "",
            "package_manager_available": bool(brew_path),
            "download_url": "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        }
    return {
        "platform": platform_name,
        "package_manager": "",
        "package_manager_available": False,
        "download_url": "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
    }


def _install_cloudflared_via_package_manager() -> dict[str, object]:
    target = _cloudflared_install_target()
    platform_name = str(target.get("platform") or "")
    package_manager = str(target.get("package_manager") or "")
    if platform_name == "windows" and package_manager == "winget":
        cmd = [
            "winget",
            "install",
            "--id",
            "Cloudflare.cloudflared",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ]
    elif platform_name == "macos" and package_manager == "brew":
        cmd = ["brew", "install", "cloudflared"]
    else:
        raise RuntimeError("Automatic install is only supported on Windows via winget or macOS via Homebrew.")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    installed = _refresh_cloudflared_availability()
    return {
        "platform": platform_name,
        "package_manager": package_manager,
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "installed": bool(installed),
    }


def _get_voicefixer_install_info() -> VoiceFixerInstallInfo:
    import sys
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    try:
        version = importlib_metadata.version("voicefixer")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _voicefixer_installed_via_pip():
        restart_required = True
        message = "VoiceFixer is installed in the backend environment, but the running backend needs a restart before it can use it."

    return VoiceFixerInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=VOICEFIXER_PACKAGE_SPEC,
        restart_required=restart_required,
        message=message,
    )


def _voicefixer_installed_via_pip() -> bool:
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "voicefixer"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: voicefixer" in (result.stdout or "")
    except Exception:
        return False


def _get_clearvoice_install_info() -> ClearVoiceInstallInfo:
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    runtime_ready = False
    runtime_error = None
    torch_version = None
    torchaudio_version = None
    try:
        version = importlib_metadata.version("clearvoice")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _clearvoice_installed_via_pip():
        restart_required = True
        message = "ClearVoice is installed in the backend environment, but the running backend needs a restart before it can use it."
    elif installed and not restart_required:
        runtime = _inspect_clearvoice_runtime()
        runtime_ready = bool(runtime.get("runtime_ready"))
        runtime_error = str(runtime.get("error") or "").strip() or None
        torch_version = str(runtime.get("torch_version") or "").strip() or None
        torchaudio_version = str(runtime.get("torchaudio_version") or "").strip() or None
        if not runtime_ready:
            message = str(runtime.get("detail") or "").strip() or "ClearVoice is installed, but its runtime dependencies are not healthy."

    return ClearVoiceInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=CLEARVOICE_PACKAGE_SPEC,
        restart_required=restart_required,
        runtime_ready=runtime_ready,
        runtime_error=runtime_error,
        torch_version=torch_version,
        torchaudio_version=torchaudio_version,
        message=message,
    )


def _clearvoice_installed_via_pip() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "clearvoice"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: clearvoice" in (result.stdout or "")
    except Exception:
        return False


def _normalize_torch_version(raw_version: str | None) -> str | None:
    value = str(raw_version or "").strip()
    if not value:
        return None
    return value.split("+", 1)[0].strip() or None


def _torch_wheel_variant(raw_version: str | None) -> str:
    value = str(raw_version or "").strip().lower()
    if "+" in value:
        suffix = value.split("+", 1)[1].strip()
        if suffix:
            return suffix
    return "cpu"


def _inspect_clearvoice_runtime() -> dict[str, object]:
    torch_imported = False
    torchaudio_imported = False
    clearvoice_imported = False
    class_available = False
    torch_version = None
    torchaudio_version = None
    error = None
    detail = None

    try:
        import torch  # type: ignore

        torch_imported = True
        torch_version = str(getattr(torch, "__version__", "") or "").strip() or None
    except Exception as e:
        error = f"torch import failed: {e}"
        detail = "ClearVoice is installed, but torch could not be imported in the backend environment."

    if error is None:
        try:
            import torchaudio  # type: ignore

            torchaudio_imported = True
            torchaudio_version = str(getattr(torchaudio, "__version__", "") or "").strip() or None
        except Exception as e:
            error = f"torchaudio import failed: {e}"
            detail = (
                "ClearVoice is installed, but torchaudio could not be imported. "
                "The backend environment likely has mismatched torch/torchaudio wheels. "
                "Run ClearVoice runtime repair in Settings."
            )

    try:
        from clearvoice import ClearVoice  # type: ignore

        clearvoice_imported = True
        class_available = callable(ClearVoice)
    except Exception as e:
        if error is None:
            error = f"clearvoice import failed: {e}"
            detail = "ClearVoice package import failed in the backend environment."

    runtime_ready = bool(torch_imported and torchaudio_imported and clearvoice_imported and class_available)
    if runtime_ready:
        detail = "Imported torch, torchaudio, and ClearVoice successfully. Model weights are not loaded during this self-test."

    return {
        "runtime_ready": runtime_ready,
        "torch_imported": torch_imported,
        "torchaudio_imported": torchaudio_imported,
        "clearvoice_imported": clearvoice_imported,
        "class_available": class_available,
        "torch_version": torch_version,
        "torchaudio_version": torchaudio_version,
        "error": error,
        "detail": detail,
    }


def _repair_clearvoice_runtime() -> dict[str, object]:
    try:
        import torch  # type: ignore

        torch_version_raw = str(getattr(torch, "__version__", "") or "").strip() or None
    except Exception:
        from importlib import metadata as importlib_metadata

        try:
            torch_version_raw = importlib_metadata.version("torch")
        except Exception as e:
            raise RuntimeError(f"Could not determine the installed torch version: {e}")

    normalized_version = _normalize_torch_version(torch_version_raw)
    if not normalized_version:
        raise RuntimeError("Could not determine a valid torch version for ClearVoice runtime repair.")
    variant = _torch_wheel_variant(torch_version_raw)
    index_url = f"https://download.pytorch.org/whl/{variant}" if variant.startswith("cu") else "https://download.pytorch.org/whl/cpu"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--index-url",
        index_url,
        f"torchaudio=={normalized_version}",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        detail = stderr or stdout or "unknown installer failure"
        raise RuntimeError(detail[:1200])
    return {
        "status": "repaired",
        "command": cmd,
        "index_url": index_url,
        "torch_version": torch_version_raw,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def _normalize_clearvoice_metadata() -> dict[str, object]:
    from importlib import metadata as importlib_metadata

    dist = importlib_metadata.distribution("clearvoice")
    meta_path = Path(getattr(dist, "_path", "")) / "METADATA"
    if not meta_path.exists():
        raise RuntimeError(f"Could not locate ClearVoice METADATA at {meta_path}")

    original = meta_path.read_text(encoding="utf-8")
    updated = original
    replacements = {
        "Requires-Dist: numpy<2.0,>=1.24.3": "Requires-Dist: numpy>=1.24.3",
        "Requires-Dist: soundfile==0.12.1": "Requires-Dist: soundfile>=0.12.1",
    }
    changed: list[dict[str, str]] = []
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            changed.append({"from": old, "to": new})

    if updated != original:
        meta_path.write_text(updated, encoding="utf-8")

    return {
        "metadata_path": str(meta_path),
        "updated": bool(changed),
        "changed": changed,
    }


def _get_reconstruction_install_info() -> ReconstructionInstallInfo:
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    try:
        version = importlib_metadata.version("qwen-tts")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _reconstruction_installed_via_pip():
        restart_required = True
        message = "The reconstruction runtime is installed in the backend environment, but the running backend needs a restart before it can use it."
    elif installed and not _sox_available():
        target = _sox_install_target()
        if bool(target.get("package_manager_available")):
            message = f"The reconstruction runtime is installed, but SoX is missing. The install action will add SoX via {target.get('package_manager')}."
        else:
            message = f"The reconstruction runtime is installed, but SoX is missing. Install it from {target.get('download_url')}."

    return ReconstructionInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=RECONSTRUCTION_PACKAGE_SPEC,
        restart_required=restart_required,
        message=message,
    )


def _reconstruction_installed_via_pip() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "qwen-tts"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: qwen-tts" in (result.stdout or "")
    except Exception:
        return False


def _discover_sox_binary() -> str:
    direct = shutil.which("sox") or shutil.which("sox.exe") or ""
    if direct:
        return direct
    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA") or ""
        if local_appdata:
            winget_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            try:
                matches = sorted(winget_root.glob("ChrisBagwell.SoX_*/*/sox.exe"))
            except Exception:
                matches = []
            if matches:
                return str(matches[-1])
    return ""


def _sox_available() -> bool:
    sox_bin = _discover_sox_binary()
    if not sox_bin:
        return False
    sox_dir = str(Path(sox_bin).parent)
    current_path = os.environ.get("PATH") or ""
    path_parts = current_path.split(os.pathsep) if current_path else []
    if sox_dir and sox_dir not in path_parts:
        os.environ["PATH"] = current_path + (os.pathsep if current_path else "") + sox_dir
    return True


def _sox_install_target() -> dict[str, str | bool]:
    platform_name = "windows" if os.name == "nt" else ("macos" if sys.platform == "darwin" else sys.platform)
    winget_path = shutil.which("winget") or ""
    brew_path = shutil.which("brew") or ""
    if platform_name == "windows":
        return {
            "platform": platform_name,
            "package_manager": "winget" if winget_path else "",
            "package_manager_available": bool(winget_path),
            "package_id": "ChrisBagwell.SoX",
            "download_url": "https://sourceforge.net/projects/sox/files/sox/",
        }
    if platform_name == "macos":
        return {
            "platform": platform_name,
            "package_manager": "brew" if brew_path else "",
            "package_manager_available": bool(brew_path),
            "package_id": "sox",
            "download_url": "https://formulae.brew.sh/formula/sox",
        }
    return {
        "platform": platform_name,
        "package_manager": "",
        "package_manager_available": False,
        "package_id": "",
        "download_url": "https://sourceforge.net/projects/sox/files/sox/",
    }


def _install_sox_via_package_manager() -> dict[str, object]:
    target = _sox_install_target()
    platform_name = str(target.get("platform") or "")
    package_manager = str(target.get("package_manager") or "")
    package_id = str(target.get("package_id") or "")
    if platform_name == "windows" and package_manager == "winget":
        cmd = [
            "winget",
            "install",
            "--id",
            package_id,
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ]
    elif platform_name == "macos" and package_manager == "brew":
        cmd = ["brew", "install", "sox"]
    else:
        raise RuntimeError("Automatic SoX install is only supported on Windows via winget or macOS via Homebrew.")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return {
        "platform": platform_name,
        "package_manager": package_manager,
        "package_id": package_id,
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "installed": bool(_sox_available()),
        "download_url": str(target.get("download_url") or ""),
    }


def _voicefixer_analysis_checkpoint_path() -> Path:
    return Path.home() / ".cache" / "voicefixer" / "analysis_module" / "checkpoints" / "vf.ckpt"


def _download_voicefixer_analysis_checkpoint() -> dict[str, object]:
    import torch

    url = "https://zenodo.org/record/5600188/files/vf.ckpt?download=1"
    target = _voicefixer_analysis_checkpoint_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".download")

    if tmp_path.exists():
        tmp_path.unlink()

    bytes_written = 0
    expected_bytes = None
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if curl_bin:
        head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Chatalogue/VoiceFixer-Repair"})
        with urllib.request.urlopen(head_req, timeout=60) as response:
            try:
                expected_bytes = int(response.headers.get("Content-Length") or 0) or None
            except Exception:
                expected_bytes = None

        result = subprocess.run(
            [curl_bin, "-L", "--fail", "--output", str(tmp_path), url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "curl download failed").strip()
            raise RuntimeError(detail[:1200])
        bytes_written = int(tmp_path.stat().st_size)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Chatalogue/VoiceFixer-Repair"})
        with urllib.request.urlopen(req, timeout=60) as response, tmp_path.open("wb") as out:
            try:
                expected_bytes = int(response.headers.get("Content-Length") or 0) or None
            except Exception:
                expected_bytes = None
            while True:
                chunk = response.read(1024 * 1024 * 4)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)

    if expected_bytes is not None and bytes_written != expected_bytes:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Downloaded VoiceFixer checkpoint size mismatch: got {bytes_written} bytes, expected {expected_bytes}.")

    try:
        torch.load(str(tmp_path), map_location="cpu")
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise

    target.unlink(missing_ok=True)
    tmp_path.replace(target)
    return {
        "status": "ok",
        "path": str(target),
        "bytes_written": int(bytes_written),
        "expected_bytes": expected_bytes,
    }


def _resolve_lan_host() -> str | None:
    override = (os.getenv("CHATALOGUE_SHARE_LAN_HOST") or "").strip()
    if override:
        return override
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
        sock.close()
        if host and not _is_loopback_host(host):
            return host
    except Exception:
        pass
    try:
        host = socket.gethostbyname(socket.gethostname())
        if host and not _is_loopback_host(host):
            return host
    except Exception:
        pass
    return None


def _extract_public_url_from_line(line: str) -> str | None:
    match = re.search(r"https://[a-z0-9.-]+\.trycloudflare\.com", line, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _drain_share_process_output(proc: subprocess.Popen, label: str) -> None:
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = str(raw_line or "").rstrip()
            if line:
                _append_share_event(f"[{label}] {line}")
    except Exception:
        pass


def _start_cloudflared_quick_tunnel(local_url: str, label: str) -> tuple[subprocess.Popen, str]:
    binary = _get_cloudflared_binary()
    if not binary:
        raise RuntimeError("cloudflared was not found. Install it or add CLOUDFLARED_BIN to enable public share tunnels.")

    proc = subprocess.Popen(
        [binary, "tunnel", "--url", local_url, "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    public_url = None
    deadline = time.time() + 25
    buffered: list[str] = []
    try:
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            buffered.append(line.rstrip())
            detected = _extract_public_url_from_line(line)
            if detected:
                public_url = detected
                break
    except Exception:
        pass

    if not public_url:
        try:
            proc.terminate()
        except Exception:
            pass
        raise RuntimeError(f"cloudflared failed to return a public URL for {label}. Output: {' | '.join(buffered[-6:])}")

    thread = threading.Thread(target=_drain_share_process_output, args=(proc, label), daemon=True, name=f"share-{label}-log")
    thread.start()
    return proc, public_url


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _stop_external_share_locked(reason: str = "stopped") -> None:
    processes = dict(external_share_state.get("processes") or {})
    external_share_state.update({
        "active": False,
        "mode": "off",
        "enable_tunnel": False,
        "tunnel_provider": None,
        "started_at": None,
        "expires_at": None,
        "token": None,
        "password": None,
        "ip_allowlist": [],
        "frontend_local_url": None,
        "api_local_url": None,
        "frontend_lan_url": None,
        "api_lan_url": None,
        "frontend_public_url": None,
        "api_public_url": None,
        "share_url": None,
        "processes": {},
    })
    for proc in processes.values():
        _terminate_process(proc)
    _append_share_event(f"External share stopped ({reason}).")


def _ensure_external_share_not_expired() -> None:
    with external_share_lock:
        if not external_share_state.get("active"):
            return
        expires_at = external_share_state.get("expires_at")
        if not expires_at:
            return
        try:
            expiry = datetime.fromisoformat(str(expires_at))
        except Exception:
            return
        if expiry <= _utc_now():
            _stop_external_share_locked(reason="expired")


def _parse_allowlist(value: str) -> list[str]:
    items: list[str] = []
    for raw in re.split(r"[\s,]+", str(value or "").strip()):
        entry = raw.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                ipaddress.ip_network(entry, strict=False)
            else:
                ipaddress.ip_address(entry)
        except ValueError:
            continue
        items.append(entry)
    return items


def _is_loopback_host(host: str | None) -> bool:
    value = str(host or "").strip()
    if value in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _resolve_client_ip(request: Request) -> str:
    for header in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        raw = (request.headers.get(header) or "").strip()
        if not raw:
            continue
        if header == "x-forwarded-for":
            raw = raw.split(",")[0].strip()
        if raw:
            return raw
    return getattr(request.client, "host", "") or ""


def _client_ip_allowed(client_ip: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    try:
        ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            if "/" in entry:
                if ip_obj in ipaddress.ip_network(entry, strict=False):
                    return True
            elif ip_obj == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False


def _external_share_error_response(request: Request, status_code: int, detail: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "code": code},
    )


def _get_external_share_credentials(request: Request) -> tuple[str, str]:
    token = (
        request.headers.get(SHARE_TOKEN_HEADER)
        or request.cookies.get(SHARE_COOKIE_TOKEN)
        or request.query_params.get("share_token")
        or ""
    )
    password = (
        request.headers.get(SHARE_PASSWORD_HEADER)
        or request.cookies.get(SHARE_COOKIE_PASSWORD)
        or request.query_params.get("share_password")
        or ""
    )
    return str(token).strip(), str(password).strip()


@app.get("/system/cuda-health")
def system_cuda_health():
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service not ready")
    try:
        payload = ingestion_service.get_cuda_health_status()
        if isinstance(payload, dict):
            training_memory = _collect_avatar_training_process_memory()
            component_memory = payload.get("component_memory")
            if not isinstance(component_memory, dict):
                component_memory = {}
                payload["component_memory"] = component_memory
            component_memory["avatar_training"] = {
                "loaded": bool(training_memory.get("loaded")),
                "ram_gb": float(training_memory.get("ram_gb") or 0.0),
                "vram_gb": float(training_memory.get("vram_gb") or 0.0),
            }

            memory = payload.get("memory")
            if isinstance(memory, dict):
                try:
                    memory["allocated_gb"] = round(
                        float(memory.get("allocated_gb") or 0.0) + float(training_memory.get("vram_gb") or 0.0),
                        2,
                    )
                except Exception:
                    pass
            system_memory = payload.get("system_memory")
            if isinstance(system_memory, dict):
                try:
                    system_memory["rss_gb"] = round(
                        float(system_memory.get("rss_gb") or 0.0) + float(training_memory.get("ram_gb") or 0.0),
                        2,
                    )
                except Exception:
                    pass
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect CUDA health: {e}")


@app.get("/system/cuda-restart-state")
def get_cuda_restart_state():
    state_file = BACKEND_RUNTIME_DIR / "cuda_restart_state.json"
    if not state_file.exists():
        return {"restart_timestamps": [], "permanent_cpu_mode": False}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"restart_timestamps": [], "permanent_cpu_mode": False}


@app.post("/system/cuda-restart-state/reset")
def reset_cuda_restart_state():
    state_file = BACKEND_RUNTIME_DIR / "cuda_restart_state.json"
    try:
        state_file.unlink(missing_ok=True)
    except Exception:
        pass
    return {"status": "cleared"}


def _require_local_operator(request: Request) -> None:
    client_ip = _resolve_client_ip(request)
    if not _is_loopback_host(client_ip):
        raise HTTPException(status_code=403, detail="This operation is only available from the local machine.")


@app.get("/share/public-status")
def get_public_share_status(token: Optional[str] = None):
    _ensure_external_share_not_expired()
    snapshot = _snapshot_external_share_state(include_secrets=True)
    expected_token = str(snapshot.get("token") or "")
    if not snapshot.get("active") or not token or str(token).strip() != expected_token:
        return {"active": False, "password_required": False}
    return {
        "active": True,
        "password_required": bool(snapshot.get("password_required")),
        "expires_at": snapshot.get("expires_at"),
        "share_url": snapshot.get("share_url"),
    }


@app.get("/share/launch/{token}")
def launch_external_share(token: str, request: Request):
    _ensure_external_share_not_expired()
    snapshot = _snapshot_external_share_state(include_secrets=True)
    expected_token = str(snapshot.get("token") or "").strip()
    client_ip = _resolve_client_ip(request)

    if not snapshot.get("active") or not expected_token or str(token).strip() != expected_token:
        _append_share_audit(action="share_launch_denied", allowed=False, reason="invalid_or_inactive", client_ip=client_ip, path=request.url.path)
        return _render_share_launch_page(
            destination_url=None,
            title="Share Link Unavailable",
            message="This share link is invalid or the share session has already ended.",
            status_code=404,
        )

    api_url = snapshot.get("api_public_url") or snapshot.get("api_lan_url") or snapshot.get("api_local_url")
    frontend_url = snapshot.get("frontend_public_url") or snapshot.get("frontend_lan_url") or snapshot.get("frontend_local_url")
    destination_url = _build_share_destination_url(frontend_url, api_url, expected_token)
    if not destination_url:
        _append_share_audit(action="share_launch_denied", allowed=False, reason="missing_destination", client_ip=client_ip, path=request.url.path)
        return _render_share_launch_page(
            destination_url=None,
            title="Share Link Unavailable",
            message="This share session does not currently have a valid destination URL.",
            status_code=500,
        )

    _append_share_audit(action="share_launch_allowed", allowed=True, reason="ok", client_ip=client_ip, path=request.url.path)
    return _render_share_launch_page(
        destination_url=destination_url,
        title="Opening Shared Chatalogue",
        message="Redirecting you to the shared Chatalogue session.",
        status_code=200,
    )


@app.get("/share/status", response_model=ExternalShareStatus)
def get_external_share_status(request: Request):
    _ensure_external_share_not_expired()
    _require_local_operator(request)
    _refresh_cloudflared_availability()
    snapshot = _snapshot_external_share_state()
    return ExternalShareStatus(**snapshot)


@app.get("/share/audit", response_model=List[ExternalShareAuditEntry])
def get_external_share_audit(request: Request):
    _ensure_external_share_not_expired()
    _require_local_operator(request)
    return [ExternalShareAuditEntry(**entry) for entry in list(external_share_audit_entries)]


@app.post("/share/start", response_model=ExternalShareStatus)
def start_external_share(req: ExternalShareStartRequest, request: Request):
    _require_local_operator(request)
    _ensure_external_share_not_expired()

    duration_minutes = max(5, min(int(req.duration_minutes or 60), 24 * 60))
    frontend_port = max(1, min(int(req.frontend_port or 5173), 65535))
    backend_port = max(1, min(int(req.backend_port or 8011), 65535))
    enable_tunnel = bool(req.enable_tunnel)
    allowlist = _parse_allowlist(req.ip_allowlist)
    password = str(req.password or "").strip()
    token = secrets.token_urlsafe(24)
    frontend_local_url = f"http://127.0.0.1:{frontend_port}"
    api_local_url = f"http://127.0.0.1:{backend_port}"
    lan_host = _resolve_lan_host()
    frontend_lan_url = f"http://{lan_host}:{frontend_port}" if lan_host else None
    api_lan_url = f"http://{lan_host}:{backend_port}" if lan_host else None
    if not enable_tunnel and (not frontend_lan_url or not api_lan_url):
        raise RuntimeError("Could not determine a LAN IP for this machine. Set CHATALOGUE_SHARE_LAN_HOST to the desired local network address and try again.")

    with external_share_lock:
        if external_share_state.get("active"):
            _stop_external_share_locked(reason="restarted")

    frontend_public_url = None
    api_public_url = None
    processes: dict[str, subprocess.Popen] = {}
    tunnel_provider = None
    if enable_tunnel:
        try:
            frontend_proc, frontend_public_url = _start_cloudflared_quick_tunnel(frontend_local_url, "frontend")
            processes["frontend"] = frontend_proc
            api_proc, api_public_url = _start_cloudflared_quick_tunnel(api_local_url, "api")
            processes["api"] = api_proc
            tunnel_provider = "cloudflared"
        except Exception:
            for proc in processes.values():
                _terminate_process(proc)
            raise

    mode = "public_tunnel" if enable_tunnel else "lan"
    share_url = _build_share_launch_url(
        api_public_url or api_lan_url or api_local_url,
        token,
    )

    started_at = _utc_now()
    expires_at = started_at + timedelta(minutes=duration_minutes)
    with external_share_lock:
        external_share_state.update({
            "active": True,
            "mode": mode,
            "enable_tunnel": enable_tunnel,
            "tunnel_provider": tunnel_provider,
            "started_at": started_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "token": token,
            "password": password or None,
            "ip_allowlist": allowlist,
            "frontend_local_url": frontend_local_url,
            "api_local_url": api_local_url,
            "frontend_lan_url": frontend_lan_url,
            "api_lan_url": api_lan_url,
            "frontend_public_url": frontend_public_url,
            "api_public_url": api_public_url,
            "share_url": share_url,
            "processes": processes,
        })
    _append_share_event(f"External share started. mode={mode} tunnel={enable_tunnel} expires_at={expires_at.isoformat()}")
    _append_share_audit(action="share_started", allowed=True, reason="ok", client_ip=_resolve_client_ip(request), path="/share/start")
    return ExternalShareStatus(**_snapshot_external_share_state())


@app.post("/share/stop", response_model=ExternalShareStatus)
def stop_external_share(request: Request):
    _require_local_operator(request)
    with external_share_lock:
        _stop_external_share_locked(reason="manual_stop")
    _append_share_audit(action="share_stopped", allowed=True, reason="manual_stop", client_ip=_resolve_client_ip(request), path="/share/stop")
    return ExternalShareStatus(**_snapshot_external_share_state())


@app.get("/system/cloudflared/install-info")
def get_cloudflared_install_info(request: Request):
    _require_local_operator(request)
    target = _cloudflared_install_target()
    return {
        **target,
        "installed": bool(_refresh_cloudflared_availability()),
    }


@app.post("/system/cloudflared/install")
def install_cloudflared(request: Request):
    _require_local_operator(request)
    if _refresh_cloudflared_availability():
        info = _cloudflared_install_target()
        return {
            "status": "already_installed",
            **info,
            "installed": True,
        }

    target = _cloudflared_install_target()
    if not bool(target.get("package_manager_available")):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Automatic install is unavailable on {target.get('platform')} because "
                f"{target.get('package_manager') or 'a supported package manager'} was not found. "
                f"Install cloudflared manually from {target.get('download_url')}."
            ),
        )

    try:
        result = _install_cloudflared_via_package_manager()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while installing cloudflared.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cloudflared install failed: {e}")

    if not result.get("installed"):
        detail = str(result.get("stderr") or result.get("stdout") or "unknown installer failure")
        raise HTTPException(status_code=500, detail=f"cloudflared install did not complete successfully: {detail[:700]}")

    _append_share_event(f"cloudflared installed via {result.get('package_manager')}")
    return {
        "status": "installed",
        **result,
        "download_url": target.get("download_url"),
    }


@app.get("/system/voicefixer/install-info", response_model=VoiceFixerInstallInfo)
def get_voicefixer_install_info(request: Request):
    _require_local_operator(request)
    return _get_voicefixer_install_info()


@app.post("/system/voicefixer/install", response_model=VoiceFixerInstallInfo)
def install_voicefixer(request: Request):
    import sys

    _require_local_operator(request)
    info = _get_voicefixer_install_info()
    if info.installed:
        return info

    cmd = [sys.executable, "-m", "pip", "install", VOICEFIXER_PACKAGE_SPEC]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while installing VoiceFixer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VoiceFixer install failed: {e}")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown installer failure").strip()
        raise HTTPException(status_code=500, detail=f"VoiceFixer install failed: {detail[:900]}")

    refreshed = _get_voicefixer_install_info()
    if not refreshed.installed:
        if _voicefixer_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "VoiceFixer finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return VoiceFixerInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="VoiceFixer install completed but the package is still unavailable to the backend.")
    return VoiceFixerInstallInfo(
        **refreshed.model_dump(),
        message="VoiceFixer installed successfully.",
    )


@app.post("/system/voicefixer/test", response_model=VoiceFixerTestResult)
def test_voicefixer(request: Request):
    _require_local_operator(request)
    info = _get_voicefixer_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="VoiceFixer is not installed in the backend environment yet.")

    try:
        from voicefixer import VoiceFixer  # type: ignore
        restorer = VoiceFixer()
        detail = "VoiceFixer imported and instantiated successfully."
        del restorer
        return VoiceFixerTestResult(
            status="ok",
            version=info.version,
            instantiated=True,
            detail=detail,
        )
    except Exception as e:
        error_text = str(e)
        detail = "VoiceFixer import or model initialization failed."
        if "failed finding central directory" in error_text.lower():
            detail = "VoiceFixer found a corrupted analysis checkpoint. Run the repair action in Settings, then re-run the self-test."
        return VoiceFixerTestResult(
            status="error",
            version=info.version,
            instantiated=False,
            error=error_text,
            detail=detail,
        )


@app.post("/system/voicefixer/repair")
def repair_voicefixer(request: Request):
    _require_local_operator(request)
    info = _get_voicefixer_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="VoiceFixer is not installed in the backend environment yet.")
    try:
        result = _download_voicefixer_analysis_checkpoint()
        return {
            "status": "repaired",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VoiceFixer checkpoint repair failed: {e}")


@app.get("/system/clearvoice/install-info", response_model=ClearVoiceInstallInfo)
def get_clearvoice_install_info(request: Request):
    _require_local_operator(request)
    return _get_clearvoice_install_info()


@app.post("/system/clearvoice/install", response_model=ClearVoiceInstallInfo)
def install_clearvoice(request: Request):
    _require_local_operator(request)
    info = _get_clearvoice_install_info()
    if info.installed and info.runtime_ready:
        return info

    if not info.installed:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", CLEARVOICE_PACKAGE_SPEC]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing ClearVoice.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ClearVoice install failed: {e}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown installer failure").strip()
            raise HTTPException(status_code=500, detail=f"ClearVoice install failed: {detail[:900]}")
        try:
            _normalize_clearvoice_metadata()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ClearVoice metadata normalization failed: {e}")

    refreshed = _get_clearvoice_install_info()
    if not refreshed.installed:
        if _clearvoice_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "ClearVoice finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return ClearVoiceInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="ClearVoice install completed but the package is still unavailable to the backend.")

    if not refreshed.restart_required and not refreshed.runtime_ready:
        runtime_error_text = str(refreshed.runtime_error or "").lower()
        if "torchaudio import failed" in runtime_error_text:
            try:
                _repair_clearvoice_runtime()
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Timed out while repairing the ClearVoice torchaudio runtime.")
            except Exception as e:
                payload = refreshed.model_dump()
                payload["message"] = (
                    "ClearVoice installed, but torchaudio is not usable in the backend environment. "
                    f"Run ClearVoice runtime repair in Settings. Repair error: {e}"
                )
                return ClearVoiceInstallInfo(**payload)
            refreshed = _get_clearvoice_install_info()

    return ClearVoiceInstallInfo(
        **refreshed.model_dump(),
        message=refreshed.message or ("ClearVoice installed successfully." if refreshed.runtime_ready else "ClearVoice installed, but runtime repair is still required."),
    )


@app.post("/system/clearvoice/test", response_model=ClearVoiceTestResult)
def test_clearvoice(request: Request):
    _require_local_operator(request)
    info = _get_clearvoice_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="ClearVoice is not installed in the backend environment yet.")
    runtime = _inspect_clearvoice_runtime()
    return ClearVoiceTestResult(
        status="ok" if bool(runtime.get("runtime_ready")) else "error",
        version=info.version,
        imported=bool(runtime.get("clearvoice_imported")),
        class_available=bool(runtime.get("class_available")),
        torch_imported=bool(runtime.get("torch_imported")),
        torchaudio_imported=bool(runtime.get("torchaudio_imported")),
        runtime_ready=bool(runtime.get("runtime_ready")),
        torch_version=str(runtime.get("torch_version") or "").strip() or None,
        torchaudio_version=str(runtime.get("torchaudio_version") or "").strip() or None,
        error=str(runtime.get("error") or "").strip() or None,
        detail=str(runtime.get("detail") or "").strip() or None,
    )


@app.post("/system/clearvoice/repair")
def repair_clearvoice(request: Request):
    _require_local_operator(request)
    info = _get_clearvoice_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="ClearVoice is not installed in the backend environment yet.")
    try:
        result = _repair_clearvoice_runtime()
        metadata_result = _normalize_clearvoice_metadata()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while repairing the ClearVoice runtime.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClearVoice runtime repair failed: {e}")
    runtime = _inspect_clearvoice_runtime()
    return {
        **result,
        "metadata": metadata_result,
        "runtime_ready": bool(runtime.get("runtime_ready")),
        "runtime_error": str(runtime.get("error") or "").strip() or None,
        "torch_version": str(runtime.get("torch_version") or "").strip() or None,
        "torchaudio_version": str(runtime.get("torchaudio_version") or "").strip() or None,
        "detail": str(runtime.get("detail") or "").strip() or None,
    }


@app.get("/system/reconstruction/install-info", response_model=ReconstructionInstallInfo)
def get_reconstruction_install_info(request: Request):
    _require_local_operator(request)
    return _get_reconstruction_install_info()


@app.post("/system/reconstruction/install", response_model=ReconstructionInstallInfo)
def install_reconstruction_runtime(request: Request):
    _require_local_operator(request)
    info = _get_reconstruction_install_info()
    qwen_installed = bool(info.installed)
    sox_installed = _sox_available()
    if qwen_installed and sox_installed:
        return info

    if not qwen_installed:
        cmd = [sys.executable, "-m", "pip", "install", RECONSTRUCTION_PACKAGE_SPEC]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing the reconstruction runtime.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconstruction runtime install failed: {e}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown installer failure").strip()
            raise HTTPException(status_code=500, detail=f"Reconstruction runtime install failed: {detail[:900]}")

    if not sox_installed:
        target = _sox_install_target()
        if not bool(target.get("package_manager_available")):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"SoX is required for conversation reconstruction on {target.get('platform')}, but automatic install is unavailable because "
                    f"{target.get('package_manager') or 'a supported package manager'} was not found. "
                    f"Install SoX manually from {target.get('download_url')}."
                ),
            )
        try:
            sox_result = _install_sox_via_package_manager()
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing SoX.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SoX install failed: {e}")

        if not sox_result.get("installed"):
            detail = str(sox_result.get("stderr") or sox_result.get("stdout") or "unknown installer failure")
            raise HTTPException(status_code=500, detail=f"SoX install did not complete successfully: {detail[:700]}")

    refreshed = _get_reconstruction_install_info()
    if not refreshed.installed:
        if _reconstruction_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "The Qwen TTS runtime finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return ReconstructionInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="The reconstruction runtime installed but is still unavailable to the backend.")
    if not _sox_available():
        target = _sox_install_target()
        raise HTTPException(status_code=500, detail=f"SoX still is not available on PATH after install. Install it manually from {target.get('download_url')}.")
    return ReconstructionInstallInfo(
        **refreshed.model_dump(),
        message="The reconstruction runtime and SoX installed successfully.",
    )


@app.post("/system/reconstruction/test", response_model=ReconstructionTestResult)
def test_reconstruction_runtime(request: Request):
    _require_local_operator(request)
    info = _get_reconstruction_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="The reconstruction runtime is not installed in the backend environment yet.")

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore

        detail = "Imported qwen_tts and found Qwen3TTSModel. Model weights are not loaded during this self-test."
        if not _sox_available():
            target = _sox_install_target()
            return ReconstructionTestResult(
                status="error",
                version=info.version,
                imported=True,
                model_class_available=Qwen3TTSModel is not None,
                error="SoX is not on PATH.",
                detail=f"SoX is required for stable conversation reconstruction on this machine. Install it via {target.get('package_manager') or 'a supported package manager'} or from {target.get('download_url')}.",
            )
        return ReconstructionTestResult(
            status="ok",
            version=info.version,
            imported=True,
            model_class_available=Qwen3TTSModel is not None,
            detail=detail,
        )
    except Exception as e:
        return ReconstructionTestResult(
            status="error",
            version=info.version,
            imported=False,
            model_class_available=False,
            error=str(e),
            detail="Importing qwen_tts failed.",
        )


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


def _youtube_data_api_key_request(
    path: str,
    *,
    api_key: str,
    query: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    key = str(api_key or "").strip()
    if not key:
        raise RuntimeError("YouTube Data API key is not configured.")
    params = {k: v for k, v in (query or {}).items() if v is not None}
    params["key"] = key
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{YOUTUBE_API_BASE_URL}{path}"
    if qs:
        url = f"{url}?{qs}"
    return _youtube_http_json("GET", url, timeout=timeout)


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
def create_channel(url: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    try:
        channel = ingestion_service.add_channel(url)
        managed_channel = session.get(Channel, channel.id)
        if managed_channel:
            managed_channel.status = "refreshing"
            managed_channel.sync_status_detail = "Starting channel scan..."
            managed_channel.sync_progress = 1
            managed_channel.sync_total_items = 0
            managed_channel.sync_completed_items = 0
            session.add(managed_channel)
            session.commit()
            session.refresh(managed_channel)
            channel = managed_channel
        # Auto-refresh on add
        background_tasks.add_task(ingestion_service.refresh_channel, channel.id)
        return channel
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/channels/manual", response_model=Channel)
def create_manual_channel(name: str, session: Session = Depends(get_session)):
    try:
        channel = ingestion_service.create_manual_channel(name)
        managed_channel = session.get(Channel, channel.id)
        return managed_channel or channel
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/channels/tiktok", response_model=Channel)
def create_tiktok_channel(
    name: Optional[str] = None,
    url: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session),
):
    try:
        channel = ingestion_service.create_tiktok_channel(name, url)
        managed_channel = session.get(Channel, channel.id)
        channel_obj = managed_channel or channel
        if "tiktok.com" in str(channel_obj.url or "").lower():
            channel_obj.status = "refreshing"
            channel_obj.sync_status_detail = "Starting TikTok profile scan..."
            channel_obj.sync_progress = 1
            channel_obj.sync_total_items = 0
            channel_obj.sync_completed_items = 0
            session.add(channel_obj)
            session.commit()
            session.refresh(channel_obj)
            if background_tasks is not None:
                background_tasks.add_task(ingestion_service.refresh_channel, channel_obj.id)
        return channel_obj
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/channels", response_model=List[Channel])
def read_channels(session: Session = Depends(get_session)):
    return session.exec(select(Channel)).all()

@app.get("/channels/overview", response_model=List[ChannelOverviewRead])
def read_channels_overview(session: Session = Depends(get_session)):
    from sqlalchemy import and_, case

    video_stats = (
        select(
            Video.channel_id.label("channel_id"),
            func.count(Video.id).label("video_count"),
            func.sum(case((Video.processed.is_(True), 1), else_=0)).label("processed_count"),
            func.sum(case((and_(Video.processed.is_(False), Video.muted.is_(False), Video.access_restricted.is_(False)), 1), else_=0)).label("pending_video_count"),
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
    active_job_stats = (
        select(
            Video.channel_id.label("channel_id"),
            func.count(Job.id).label("active_job_count"),
        )
        .select_from(Job)
        .join(Video, Video.id == Job.video_id)
        .where(
            Job.job_type.in_(["process", "diarize"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
        .group_by(Video.channel_id)
        .subquery()
    )

    rows = session.exec(
        select(
            Channel,
            func.coalesce(video_stats.c.video_count, 0).label("video_count"),
            func.coalesce(video_stats.c.processed_count, 0).label("processed_count"),
            func.coalesce(video_stats.c.pending_video_count, 0).label("pending_video_count"),
            func.coalesce(video_stats.c.total_duration_seconds, 0).label("total_duration_seconds"),
            func.coalesce(speaker_stats.c.speaker_count, 0).label("speaker_count"),
            func.coalesce(active_job_stats.c.active_job_count, 0).label("active_job_count"),
        )
        .outerjoin(video_stats, video_stats.c.channel_id == Channel.id)
        .outerjoin(speaker_stats, speaker_stats.c.channel_id == Channel.id)
        .outerjoin(active_job_stats, active_job_stats.c.channel_id == Channel.id)
        .order_by(Channel.id.asc())
    ).all()

    return [
        ChannelOverviewRead(
            id=channel.id,
            url=channel.url,
            name=channel.name,
            source_type=channel.source_type,
            icon_url=channel.icon_url,
            header_image_url=channel.header_image_url,
            last_updated=channel.last_updated,
            status=channel.status,
            actively_monitored=bool(getattr(channel, "actively_monitored", False)),
            sync_status_detail=channel.sync_status_detail,
            sync_progress=int(channel.sync_progress or 0),
            sync_total_items=int(channel.sync_total_items or 0),
            sync_completed_items=int(channel.sync_completed_items or 0),
            video_count=int(video_count or 0),
            processed_count=int(processed_count or 0),
            pending_video_count=int(pending_video_count or 0),
            active_job_count=int(active_job_count or 0),
            total_duration_seconds=int(total_duration_seconds or 0),
            speaker_count=int(speaker_count or 0),
        )
        for channel, video_count, processed_count, pending_video_count, total_duration_seconds, speaker_count, active_job_count in rows
    ]

@app.get("/channels/{channel_id}", response_model=Channel)
def read_channel(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel

@app.post("/channels/{channel_id}/refresh")
def refresh_channel(channel_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        raise HTTPException(status_code=409, detail="Manual channels do not support remote refresh.")
    if channel_source == "tiktok" and "tiktok.com" not in str(channel.url or "").lower():
        raise HTTPException(status_code=409, detail="This TikTok channel needs a real creator/profile URL before it can refresh.")
    channel.status = "refreshing"
    channel.sync_status_detail = "Starting channel scan..."
    channel.sync_progress = 1
    channel.sync_total_items = 0
    channel.sync_completed_items = 0
    session.add(channel)
    session.commit()
    background_tasks.add_task(ingestion_service.refresh_channel, channel_id)
    return {"status": "refresh_started"}


def _backfill_remote_channel_metadata_task(channel_ids: List[int]) -> None:
    for channel_id in channel_ids:
        with Session(engine) as session:
            channel = session.get(Channel, int(channel_id))
            if not channel:
                continue
            channel_source = (channel.source_type or "youtube").strip().lower()
            channel_name = str(channel.name or f"Channel {channel.id}")
            if channel_source == "manual":
                continue
            channel.status = "refreshing"
            channel.sync_status_detail = "Queued metadata backfill..."
            channel.sync_progress = 1
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()

        try:
            if channel_source == "tiktok":
                ingestion_service.refresh_channel(int(channel_id))
                continue

            ingestion_service._backfill_dates(
                int(channel_id),
                max_items=None,
                progress_start=5,
                progress_end=95,
                detail_prefix="Backfilling popularity metadata",
                status="refreshing",
            )
            ingestion_service._update_channel_sync_progress(
                int(channel_id),
                status="idle",
                detail="Metadata backfill complete.",
                progress=100,
            )
        except Exception as e:
            log_message = f"Bulk metadata backfill failed for channel {channel_id} ({channel_name}): {e}"
            logging.exception(log_message)
            ingestion_service._update_channel_sync_progress(
                int(channel_id),
                status="failed",
                detail=str(e)[:240] or "Metadata backfill failed.",
                progress=0,
            )


@app.post("/channels/metadata/backfill-all")
def backfill_all_channel_metadata(background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    rows = session.exec(
        select(Channel.id, Channel.source_type)
        .where(Channel.source_type.is_(None) | (Channel.source_type != "manual"))
        .order_by(Channel.id.asc())
    ).all()
    channel_ids = [int(row[0]) for row in rows if row[0] is not None]
    if not channel_ids:
        return {"status": "nothing_to_do", "channels": 0}
    background_tasks.add_task(_backfill_remote_channel_metadata_task, channel_ids)
    return {"status": "started", "channels": len(channel_ids)}

@app.patch("/channels/{channel_id}/actively-monitored", response_model=Channel)
def set_channel_actively_monitored(
    channel_id: int,
    enabled: bool,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        if enabled:
            raise HTTPException(status_code=409, detail="Manual channels do not support active monitoring.")
        channel.actively_monitored = False
        session.add(channel)
        session.commit()
        session.refresh(channel)
        return channel
    if channel_source == "tiktok" and enabled and "tiktok.com" not in str(channel.url or "").lower():
        raise HTTPException(status_code=409, detail="This TikTok channel needs a real creator/profile URL before monitoring can be enabled.")
    channel.actively_monitored = enabled
    if not enabled and channel.status == "refreshing":
        channel.status = "active"
        channel.sync_status_detail = "Monitoring disabled."
        channel.sync_progress = 0
        channel.sync_total_items = 0
        channel.sync_completed_items = 0
    session.add(channel)
    session.commit()
    session.refresh(channel)
    if enabled:
        background_tasks.add_task(ingestion_service.sync_monitored_channel, channel_id)
    return channel

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
    """Manually add a source video to a channel."""
    import re
    
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    channel_source = (channel.source_type or "youtube").strip().lower()
    if channel_source == "manual":
        raise HTTPException(status_code=409, detail="Use media upload for manual channels.")

    normalized_url = " ".join((url or "").strip().split())
    if not normalized_url:
        raise HTTPException(status_code=400, detail="A source URL is required.")

    if channel_source == "youtube":
        match = re.search(r'(?:v=|youtu\.be/|/v/)([a-zA-Z0-9_-]{11})', normalized_url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        source_id = match.group(1)
        lookup_url = f"https://www.youtube.com/watch?v={source_id}"
        existing = session.exec(select(Video).where(Video.youtube_id == source_id)).first()
        if existing:
            return existing
    elif channel_source == "tiktok":
        if "tiktok.com" not in normalized_url.lower():
            raise HTTPException(status_code=400, detail="Invalid TikTok URL")
        normalized_url = _normalize_tiktok_video_url(normalized_url)
        lookup_url = normalized_url
        existing = session.exec(select(Video).where(Video.source_url == normalized_url)).first()
        if existing:
            return existing
    else:
        raise HTTPException(status_code=409, detail="This channel type does not support remote add-video ingest yet.")

    try:
        info = _fetch_remote_video_info(lookup_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video info: {e}")

    youtube_api_meta = {}
    if channel_source == "youtube" and ingestion_service is not None:
        try:
            youtube_api_meta = ingestion_service._fetch_youtube_data_api_video_metadata_batch([source_id]).get(source_id) or {}
        except Exception as e:
            logging.warning("YouTube Data API metadata enrichment skipped for %s: %s", source_id, e)

    external_url = str(info.get("webpage_url") or normalized_url).strip() or normalized_url
    if channel_source == "tiktok":
        external_url = _normalize_tiktok_video_url(external_url)
    if channel_source == "tiktok":
        existing = session.exec(select(Video).where(Video.source_url == external_url)).first()
        if existing:
            return existing

    if channel_source == "youtube":
        external_id = str(info.get("id") or source_id).strip() or source_id
        unique_video_id = external_id
        media_source_type = "youtube"
        media_kind = None
    else:
        external_id = str(info.get("id") or "").strip()
        unique_video_id = _make_unique_external_video_id("tiktok", external_id, session)
        media_source_type = "tiktok"
        media_kind = "video"

    video = Video(
        youtube_id=unique_video_id,
        channel_id=channel.id,
        title=str(youtube_api_meta.get("title") or info.get("title") or "Unknown Title"),
        media_source_type=media_source_type,
        source_url=external_url,
        media_kind=media_kind,
        description=youtube_api_meta.get("description") or info.get("description"),
        published_at=ingestion_service._extract_published_at_from_info(youtube_api_meta or info),
        duration=youtube_api_meta.get("duration") if youtube_api_meta.get("duration") is not None else info.get("duration"),
        view_count=youtube_api_meta.get("view_count") if youtube_api_meta.get("view_count") is not None else info.get("view_count"),
        thumbnail_url=youtube_api_meta.get("thumbnail") or _extract_best_thumbnail_url(info),
        status="pending",
    )
    session.add(video)
    session.flush()
    if channel_source == "youtube":
        try:
            ingestion_service.populate_placeholder_transcript(session, video, info=info)
        except Exception as e:
            logging.warning("Placeholder transcript fetch failed for %s: %s", unique_video_id, e)
    elif channel_source == "tiktok":
        try:
            ingestion_service.populate_placeholder_transcript(session, video, info=info)
        except Exception as e:
            logging.warning("TikTok placeholder transcript fetch failed for %s: %s", unique_video_id, e)
    session.commit()
    session.refresh(video)
    return video


def _classify_manual_media_kind(filename: str, content_type: str | None) -> Optional[str]:
    suffix = Path(filename or "").suffix.lower()
    if (content_type or "").startswith("audio/"):
        return "audio"
    if (content_type or "").startswith("video/"):
        return "video"
    if suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        return "video"
    return None


@app.post("/channels/{channel_id}/upload-media", response_model=Video)
async def upload_media_to_channel(
    channel_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    if (channel.source_type or "youtube") != "manual":
        raise HTTPException(status_code=409, detail="Media uploads are only supported for manual channels.")

    original_name = (file.filename or "").strip()
    if not original_name:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    media_kind = _classify_manual_media_kind(original_name, file.content_type)
    if media_kind is None:
        raise HTTPException(status_code=400, detail="Unsupported media type. Upload an audio or video file.")

    suffix = Path(original_name).suffix.lower()
    safe_base = ingestion_service.sanitize_filename(Path(original_name).stem)
    safe_name = f"{safe_base or 'episode'}{suffix}"
    storage_rel_dir = Path(f"channel_{channel.id}")
    storage_dir = MANUAL_MEDIA_DIR / storage_rel_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    unique_prefix = secrets.token_hex(6)
    stored_path = storage_dir / f"{unique_prefix}_{safe_name}"

    try:
        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    try:
        duration_raw = ingestion_service._probe_media_duration_seconds(stored_path)
    except Exception:
        duration_raw = None
    duration = None
    if duration_raw and duration_raw > 0:
        duration = max(1, int(round(float(duration_raw))))

    resolved_title = " ".join((title or Path(original_name).stem).strip().split()) or Path(original_name).stem or "Uploaded Episode"

    youtube_id = f"upload_{secrets.token_hex(8)}"
    while session.exec(select(Video).where(Video.youtube_id == youtube_id)).first():
        youtube_id = f"upload_{secrets.token_hex(8)}"

    video = Video(
        youtube_id=youtube_id,
        channel_id=channel.id,
        title=resolved_title,
        media_source_type="upload",
        media_kind=media_kind,
        manual_media_path=(storage_rel_dir / stored_path.name).as_posix(),
        published_at=datetime.now(),
        duration=duration,
        status="pending",
        processed=False,
    )
    session.add(video)
    channel.last_updated = datetime.now()
    session.add(channel)
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
        "chunk_embeddings": 0,
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

            res = session.exec(sa_delete(TranscriptChunkEmbedding).where(TranscriptChunkEmbedding.video_id.in_(video_ids)))
            deleted["chunk_embeddings"] = int(res.rowcount or 0)

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
            view_count=v_data.get("view_count"),
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
    pipeline_job_types = ["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct", "transcript_repair"]

    query = select(
        Video.id,
        Video.youtube_id,
        Video.channel_id,
        Video.title,
        Video.media_source_type,
        Video.source_url,
        Video.media_kind,
        Video.manual_media_path,
        Video.published_at,
        Video.description,
        Video.thumbnail_url,
        Video.duration,
        Video.view_count,
        Video.processed,
        Video.muted,
        Video.access_restricted,
        Video.access_restriction_reason,
        Video.status,
        Video.transcript_source,
        Video.transcript_language,
        Video.transcript_is_placeholder,
    )
    if channel_id:
        query = query.where(Video.channel_id == channel_id)
    query = query.order_by(
        case((Video.published_at.is_(None), 0), else_=1),
        Video.published_at.desc(),
        Video.id.desc(),
    )

    rows = session.exec(query).all()
    video_ids = [int(row[0]) for row in rows if row[0] is not None]
    latest_pipeline_job_by_video: dict[int, tuple[str, str]] = {}
    if video_ids:
        job_rows = session.exec(
            select(Job.video_id, Job.job_type, Job.status, Job.created_at, Job.id)
            .where(
                Job.video_id.in_(video_ids),
                Job.job_type.in_(pipeline_job_types),
            )
            .order_by(Job.video_id, Job.created_at.desc(), Job.id.desc())
        ).all()
        for job_row in job_rows:
            video_id = int(job_row[0])
            if video_id in latest_pipeline_job_by_video:
                continue
            latest_pipeline_job_by_video[video_id] = (
                str(job_row[2] or "").strip().lower() or "queued",
                str(job_row[1] or "").strip().lower() or "process",
            )

    return [
        VideoListItemRead(
            id=row[0],
            youtube_id=row[1],
            channel_id=row[2],
            title=row[3],
            media_source_type=row[4],
            source_url=row[5],
            media_kind=row[6],
            manual_media_path=row[7],
            published_at=row[8],
            description=row[9],
            thumbnail_url=row[10],
            duration=row[11],
            view_count=row[12],
            processed=bool(row[13]),
            muted=bool(row[14]),
            access_restricted=bool(row[15]),
            access_restriction_reason=row[16],
            status=row[17],
            transcript_source=row[18],
            transcript_language=row[19],
            transcript_is_placeholder=bool(row[20]),
            last_pipeline_job_status=latest_pipeline_job_by_video.get(int(row[0]), (None, None))[0] if row[0] is not None else None,
            last_pipeline_job_type=latest_pipeline_job_by_video.get(int(row[0]), (None, None))[1] if row[0] is not None else None,
        )
        for row in rows
    ]

@app.get("/videos/{video_id}", response_model=Video)
def read_video(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video: raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.get("/videos/{video_id}/transcript-quality", response_model=TranscriptQualityRead)
def get_transcript_quality(
    video_id: int,
    persist_snapshot: bool = False,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    try:
        return ingestion_service.evaluate_transcript_quality(
            session,
            video_id,
            source="api",
            persist_snapshot=bool(persist_snapshot),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/videos/{video_id}/transcript-quality/snapshots", response_model=List[TranscriptQualitySnapshotRead])
def list_transcript_quality_snapshots(video_id: int, session: Session = Depends(get_session)):
    snapshots = session.exec(
        select(TranscriptQualitySnapshot)
        .where(TranscriptQualitySnapshot.video_id == video_id)
        .order_by(TranscriptQualitySnapshot.created_at.desc(), TranscriptQualitySnapshot.id.desc())
    ).all()
    items: list[TranscriptQualitySnapshotRead] = []
    for snapshot in snapshots:
        try:
            metrics = json.loads(snapshot.metrics_json or "{}")
        except Exception:
            metrics = {}
        try:
            reasons = json.loads(snapshot.reasons_json or "[]")
        except Exception:
            reasons = []
        items.append(
            TranscriptQualitySnapshotRead(
                id=int(snapshot.id),
                video_id=int(snapshot.video_id),
                run_id=snapshot.run_id,
                source=str(snapshot.source or "manual"),
                quality_profile=str(snapshot.quality_profile or "unknown"),
                recommended_tier=str(snapshot.recommended_tier or "none"),
                score=float(snapshot.score or 0.0),
                metrics=metrics if isinstance(metrics, dict) else {},
                reasons=reasons if isinstance(reasons, list) else [],
                created_at=snapshot.created_at,
            )
        )
    return items


@app.get("/videos/{video_id}/transcript-runs", response_model=List[TranscriptRunRead])
def list_transcript_runs(video_id: int, session: Session = Depends(get_session)):
    runs = session.exec(
        select(TranscriptRun)
        .where(TranscriptRun.video_id == video_id)
        .order_by(TranscriptRun.created_at.desc(), TranscriptRun.id.desc())
    ).all()
    items: list[TranscriptRunRead] = []
    for run in runs:
        try:
            metrics_before = json.loads(run.metrics_before_json or "{}") if run.metrics_before_json else None
        except Exception:
            metrics_before = None
        try:
            metrics_after = json.loads(run.metrics_after_json or "{}") if run.metrics_after_json else None
        except Exception:
            metrics_after = None
        try:
            artifact_refs = json.loads(run.artifact_refs_json or "{}") if run.artifact_refs_json else None
        except Exception:
            artifact_refs = None
        try:
            model_provenance = json.loads(run.model_provenance_json or "{}") if run.model_provenance_json else None
        except Exception:
            model_provenance = None
        items.append(
            TranscriptRunRead(
                id=int(run.id),
                video_id=int(run.video_id),
                input_run_id=run.input_run_id,
                mode=str(run.mode or "baseline"),
                pipeline_version=str(run.pipeline_version or "baseline-v1"),
                status=str(run.status or "completed"),
                quality_profile=run.quality_profile,
                recommended_tier=run.recommended_tier,
                started_at=run.started_at,
                completed_at=run.completed_at,
                metrics_before=metrics_before if isinstance(metrics_before, dict) else None,
                metrics_after=metrics_after if isinstance(metrics_after, dict) else None,
                artifact_refs=artifact_refs if isinstance(artifact_refs, dict) else None,
                rollback_state=run.rollback_state,
                model_provenance=model_provenance if isinstance(model_provenance, dict) else None,
                note=run.note,
                created_at=run.created_at,
            )
        )
    return items


@app.get("/videos/{video_id}/transcript-rollback-options", response_model=List[TranscriptRollbackOptionRead])
def list_transcript_rollback_options(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    return [
        TranscriptRollbackOptionRead(**item)
        for item in ingestion_service.list_transcript_rollback_options(session, video_id)
    ]


@app.post("/videos/{video_id}/transcript-runs/{run_id}/restore", response_model=TranscriptRestoreResponse)
def restore_transcript_run(
    video_id: int,
    run_id: int,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    try:
        result = ingestion_service.restore_transcript_from_run(session, video_id, run_id, source="api_restore")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptRestoreResponse(**result)


def _serialize_transcript_gold_window(window: TranscriptGoldWindow) -> TranscriptGoldWindowRead:
    try:
        entities = json.loads(window.entities_json or "[]") if window.entities_json else []
    except Exception:
        entities = []
    return TranscriptGoldWindowRead(
        id=int(window.id),
        video_id=int(window.video_id),
        label=str(window.label or "window"),
        quality_profile=window.quality_profile,
        language=window.language,
        start_time=float(window.start_time),
        end_time=float(window.end_time),
        reference_text=str(window.reference_text or ""),
        entities=entities if isinstance(entities, list) else [],
        notes=window.notes,
        active=bool(window.active),
        created_at=window.created_at,
        updated_at=window.updated_at,
    )


def _serialize_transcript_evaluation_result(result: TranscriptEvaluationResult) -> TranscriptEvaluationResultRead:
    try:
        metrics = json.loads(result.metrics_json or "{}") if result.metrics_json else {}
    except Exception:
        metrics = {}
    return TranscriptEvaluationResultRead(
        id=int(result.id),
        gold_window_id=int(result.gold_window_id),
        video_id=int(result.video_id),
        run_id=result.run_id,
        source=str(result.source or "manual"),
        candidate_text=str(result.candidate_text or ""),
        reference_text=str(result.reference_text or ""),
        wer=float(result.wer or 0.0),
        cer=float(result.cer or 0.0),
        entity_accuracy=float(result.entity_accuracy) if result.entity_accuracy is not None else None,
        matched_entity_count=int(result.matched_entity_count or 0),
        total_entity_count=int(result.total_entity_count or 0),
        segment_count=int(result.segment_count or 0),
        unknown_speaker_rate=float(result.unknown_speaker_rate or 0.0),
        punctuation_density_delta=float(result.punctuation_density_delta or 0.0),
        metrics=metrics if isinstance(metrics, dict) else {},
        created_at=result.created_at,
    )


def _serialize_transcript_evaluation_review(review: TranscriptEvaluationReview) -> TranscriptEvaluationReviewRead:
    try:
        tags = json.loads(review.tags_json or "[]") if review.tags_json else []
    except Exception:
        tags = []
    return TranscriptEvaluationReviewRead(
        id=int(review.id),
        evaluation_result_id=int(review.evaluation_result_id),
        reviewer=review.reviewer,
        verdict=str(review.verdict or "same"),
        tags=tags if isinstance(tags, list) else [],
        notes=review.notes,
        created_at=review.created_at,
    )


def _serialize_transcript_rollback_option(run: TranscriptRun) -> TranscriptRollbackOptionRead:
    rollback_state = str(run.rollback_state or "").strip() or None
    rollback_available = False
    if rollback_state:
        try:
            rollback_available = Path(rollback_state).exists()
        except Exception:
            rollback_available = False
    return TranscriptRollbackOptionRead(
        run_id=int(run.id),
        video_id=int(run.video_id),
        mode=str(run.mode or "unknown"),
        pipeline_version=str(run.pipeline_version or ""),
        note=run.note,
        created_at=run.created_at,
        rollback_available=rollback_available,
        rollback_state=rollback_state,
    )


def _serialize_transcript_campaign(campaign: TranscriptOptimizationCampaign) -> TranscriptOptimizationCampaignRead:
    try:
        tiers = json.loads(campaign.tiers_json or "[]") if campaign.tiers_json else []
    except Exception:
        tiers = []
    return TranscriptOptimizationCampaignRead(
        id=int(campaign.id),
        channel_id=campaign.channel_id,
        scope=str(campaign.scope or "global"),
        status=str(campaign.status or "draft"),
        tiers=tiers if isinstance(tiers, list) else [],
        limit=int(campaign.limit or 0),
        force_non_eligible=bool(campaign.force_non_eligible),
        queued_jobs=int(campaign.queued_jobs or 0),
        skipped_active=int(campaign.skipped_active or 0),
        skipped_no_segments=int(campaign.skipped_no_segments or 0),
        skipped_not_eligible=int(campaign.skipped_not_eligible or 0),
        skipped_other=int(campaign.skipped_other or 0),
        note=campaign.note,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
    )


def _serialize_transcript_campaign_item(item: TranscriptOptimizationCampaignItem) -> TranscriptOptimizationCampaignItemRead:
    return TranscriptOptimizationCampaignItemRead(
        id=int(item.id),
        campaign_id=int(item.campaign_id),
        video_id=int(item.video_id),
        recommended_tier=str(item.recommended_tier or "none"),
        action_tier=str(item.action_tier or "none"),
        quality_score=float(item.quality_score or 0.0),
        reason=item.reason,
        status=str(item.status or "pending"),
        job_id=item.job_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@app.get("/videos/{video_id}/transcript-gold-windows", response_model=List[TranscriptGoldWindowRead])
def list_transcript_gold_windows(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    windows = session.exec(
        select(TranscriptGoldWindow)
        .where(TranscriptGoldWindow.video_id == video_id)
        .order_by(TranscriptGoldWindow.start_time, TranscriptGoldWindow.id)
    ).all()
    return [_serialize_transcript_gold_window(item) for item in windows]


@app.post("/videos/{video_id}/transcript-gold-windows", response_model=TranscriptGoldWindowRead)
def create_transcript_gold_window(
    video_id: int,
    body: TranscriptGoldWindowUpsertRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    try:
        window = ingestion_service.upsert_transcript_gold_window(
            session,
            video_id,
            label=body.label,
            quality_profile=body.quality_profile,
            language=body.language,
            start_time=body.start_time,
            end_time=body.end_time,
            reference_text=body.reference_text,
            entities=body.entities,
            notes=body.notes,
            active=body.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_transcript_gold_window(window)


@app.put("/transcript-gold-windows/{window_id}", response_model=TranscriptGoldWindowRead)
def update_transcript_gold_window(
    window_id: int,
    body: TranscriptGoldWindowUpsertRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    existing = session.get(TranscriptGoldWindow, window_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Gold window not found")
    try:
        window = ingestion_service.upsert_transcript_gold_window(
            session,
            int(existing.video_id),
            window_id=int(window_id),
            label=body.label,
            quality_profile=body.quality_profile,
            language=body.language,
            start_time=body.start_time,
            end_time=body.end_time,
            reference_text=body.reference_text,
            entities=body.entities,
            notes=body.notes,
            active=body.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_transcript_gold_window(window)


@app.post("/videos/{video_id}/transcript-evaluation", response_model=TranscriptEvaluationBatchResponse)
def evaluate_transcript_against_gold_windows(
    video_id: int,
    run_id: Optional[int] = None,
    active_only: bool = True,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        result = ingestion_service.evaluate_transcript_gold_windows(
            session,
            video_id,
            run_id=run_id,
            source="manual_api",
            active_only=bool(active_only),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptEvaluationBatchResponse(
        video_id=int(result["video_id"]),
        run_id=result.get("run_id"),
        total_windows=int(result["total_windows"]),
        average_wer=float(result["average_wer"]),
        average_cer=float(result["average_cer"]),
        average_entity_accuracy=float(result["average_entity_accuracy"]) if result.get("average_entity_accuracy") is not None else None,
        average_unknown_speaker_rate=float(result["average_unknown_speaker_rate"]),
        items=[_serialize_transcript_evaluation_result(item) for item in result["items"]],
    )


@app.get("/videos/{video_id}/transcript-evaluation-results", response_model=List[TranscriptEvaluationResultRead])
def list_transcript_evaluation_results(video_id: int, run_id: Optional[int] = None, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    query = select(TranscriptEvaluationResult).where(TranscriptEvaluationResult.video_id == video_id)
    if run_id is not None:
        query = query.where(TranscriptEvaluationResult.run_id == run_id)
    results = session.exec(
        query.order_by(TranscriptEvaluationResult.created_at.desc(), TranscriptEvaluationResult.id.desc())
    ).all()
    return [_serialize_transcript_evaluation_result(item) for item in results]


@app.post("/transcript-evaluation-results/{result_id}/review", response_model=TranscriptEvaluationReviewRead)
def create_transcript_evaluation_review(
    result_id: int,
    body: TranscriptEvaluationReviewRequest,
    session: Session = Depends(get_session),
):
    result = session.get(TranscriptEvaluationResult, result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    review = TranscriptEvaluationReview(
        evaluation_result_id=int(result_id),
        reviewer=str(body.reviewer or "").strip() or None,
        verdict=str(body.verdict),
        tags_json=json.dumps(list(body.tags or []), ensure_ascii=False),
        notes=str(body.notes or "").strip() or None,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    return _serialize_transcript_evaluation_review(review)


@app.get("/transcript-evaluation-results/{result_id}/reviews", response_model=List[TranscriptEvaluationReviewRead])
def list_transcript_evaluation_reviews(result_id: int, session: Session = Depends(get_session)):
    result = session.get(TranscriptEvaluationResult, result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    reviews = session.exec(
        select(TranscriptEvaluationReview)
        .where(TranscriptEvaluationReview.evaluation_result_id == result_id)
        .order_by(TranscriptEvaluationReview.created_at.desc(), TranscriptEvaluationReview.id.desc())
    ).all()
    return [_serialize_transcript_evaluation_review(item) for item in reviews]


@app.get("/transcript-evaluation/summary", response_model=TranscriptEvaluationSummaryRead)
def get_transcript_evaluation_summary(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if channel_id is not None:
        channel = session.get(Channel, channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    return TranscriptEvaluationSummaryRead(**ingestion_service.summarize_transcript_evaluation(session, channel_id=channel_id))


@app.get("/transcript-evaluation/diarization-config-summary", response_model=List[TranscriptDiarizationConfigBenchmarkRead])
def get_transcript_diarization_config_summary(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if channel_id is not None:
        channel = session.get(Channel, channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    return [
        TranscriptDiarizationConfigBenchmarkRead(**item)
        for item in ingestion_service.summarize_diarization_benchmarks(session, channel_id=channel_id)
    ]


@app.post("/transcripts/optimize/dry-run", response_model=TranscriptOptimizeDryRunResponse)
def transcript_optimize_dry_run(
    request: TranscriptOptimizeDryRunRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    result = ingestion_service.transcript_optimization_dry_run(
        session,
        channel_id=request.channel_id,
        video_id=request.video_id,
        limit=request.limit,
        persist_snapshots=bool(request.persist_snapshots),
    )
    return TranscriptOptimizeDryRunResponse(**result)


@app.get("/transcript-optimization-campaigns", response_model=List[TranscriptOptimizationCampaignRead])
def list_transcript_optimization_campaigns(
    channel_id: Optional[int] = None,
    limit: int = 30,
    session: Session = Depends(get_session),
):
    query = select(TranscriptOptimizationCampaign)
    if channel_id is not None:
        query = query.where(TranscriptOptimizationCampaign.channel_id == channel_id)
    campaigns = session.exec(
        query.order_by(TranscriptOptimizationCampaign.created_at.desc(), TranscriptOptimizationCampaign.id.desc()).limit(max(1, min(int(limit), 200)))
    ).all()
    return [_serialize_transcript_campaign(item) for item in campaigns]


@app.post("/transcript-optimization-campaigns", response_model=TranscriptOptimizationCampaignRead)
def create_transcript_optimization_campaign(
    request: TranscriptOptimizationCampaignCreateRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if request.channel_id is not None:
        channel = session.get(Channel, request.channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    campaign = ingestion_service.create_transcript_optimization_campaign(
        session,
        channel_id=request.channel_id,
        limit=request.limit,
        tiers=list(request.tiers or []),
        force_non_eligible=bool(request.force_non_eligible),
        note=request.note,
    )
    return _serialize_transcript_campaign(campaign)


@app.get("/transcript-optimization-campaigns/{campaign_id}/items", response_model=List[TranscriptOptimizationCampaignItemRead])
def list_transcript_optimization_campaign_items(campaign_id: int, session: Session = Depends(get_session)):
    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    items = session.exec(
        select(TranscriptOptimizationCampaignItem)
        .where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
        .order_by(TranscriptOptimizationCampaignItem.quality_score.asc(), TranscriptOptimizationCampaignItem.id.asc())
    ).all()
    return [_serialize_transcript_campaign_item(item) for item in items]


@app.post("/transcript-optimization-campaigns/{campaign_id}/execute", response_model=TranscriptOptimizationCampaignExecuteResponse)
def execute_transcript_optimization_campaign(campaign_id: int, session: Session = Depends(get_session)):
    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    items = session.exec(
        select(TranscriptOptimizationCampaignItem)
        .where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
        .order_by(TranscriptOptimizationCampaignItem.quality_score.asc(), TranscriptOptimizationCampaignItem.id.asc())
    ).all()

    queued_jobs = 0
    skipped_active = 0
    skipped_no_segments = 0
    skipped_not_eligible = 0
    skipped_other = 0

    for item in items:
        video = session.get(Video, item.video_id)
        if not video:
            item.status = "skipped_missing_video"
            skipped_other += 1
            session.add(item)
            continue
        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == item.video_id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            item.status = "skipped_active"
            skipped_active += 1
            session.add(item)
            continue

        action_tier = str(item.action_tier or item.recommended_tier or "none")
        if action_tier not in {"low_risk_repair", "diarization_rebuild", "full_retranscription"}:
            item.status = "skipped_not_supported"
            skipped_other += 1
            session.add(item)
            continue
        if not bool(campaign.force_non_eligible) and str(item.recommended_tier or "none") != action_tier:
            item.status = "skipped_not_eligible"
            skipped_not_eligible += 1
            session.add(item)
            continue
        seg_exists = session.exec(select(TranscriptSegment.id).where(TranscriptSegment.video_id == item.video_id).limit(1)).first()
        if not seg_exists:
            item.status = "skipped_no_segments"
            skipped_no_segments += 1
            session.add(item)
            continue
        try:
            if action_tier == "low_risk_repair":
                job = _enqueue_unique_job(
                    session,
                    video_id=int(item.video_id),
                    job_type="transcript_repair",
                    payload={
                        "save_files": True,
                        "force": bool(campaign.force_non_eligible),
                        "note": str(campaign.note or "").strip() or None,
                        "queued_from": f"campaign:{campaign_id}",
                    },
                )
            elif action_tier == "diarization_rebuild":
                job, _, _, _ = _queue_diarization_rebuild_job(
                    session,
                    video=video,
                    force=bool(campaign.force_non_eligible),
                    note=campaign.note,
                    queued_from=f"campaign:{campaign_id}",
                )
            else:
                job, _, _, _ = _queue_full_retranscription_job(
                    session,
                    video=video,
                    force=bool(campaign.force_non_eligible),
                    note=campaign.note,
                    queued_from=f"campaign:{campaign_id}",
                )
            item.job_id = int(job.id)
            item.status = "queued"
            item.updated_at = datetime.now()
            session.add(item)
            queued_jobs += 1
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code in {400, 409} and "active job" in detail_text.lower():
                item.status = "skipped_active"
                skipped_active += 1
            elif "no raw transcript" in detail_text.lower() or "no segments" in detail_text.lower():
                item.status = "skipped_no_segments"
                skipped_no_segments += 1
            elif "not '" in detail_text.lower() or "not '" in detail_text:
                item.status = "skipped_not_eligible"
                skipped_not_eligible += 1
            else:
                item.status = "failed_queue"
                skipped_other += 1
            item.updated_at = datetime.now()
            session.add(item)

    campaign.status = "queued"
    campaign.queued_jobs = queued_jobs
    campaign.skipped_active = skipped_active
    campaign.skipped_no_segments = skipped_no_segments
    campaign.skipped_not_eligible = skipped_not_eligible
    campaign.skipped_other = skipped_other
    campaign.updated_at = datetime.now()
    session.add(campaign)
    session.commit()
    return TranscriptOptimizationCampaignExecuteResponse(
        campaign_id=int(campaign_id),
        status=str(campaign.status or "queued"),
        queued_jobs=queued_jobs,
        skipped_active=skipped_active,
        skipped_no_segments=skipped_no_segments,
        skipped_not_eligible=skipped_not_eligible,
        skipped_other=skipped_other,
    )


@app.delete("/transcript-optimization-campaigns/{campaign_id}", response_model=TranscriptOptimizationCampaignDeleteResponse)
def delete_transcript_optimization_campaign(campaign_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import delete as sa_delete

    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    items = session.exec(
        select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
    ).all()

    detached_job_refs = 0
    for item in items:
        if item.job_id is not None:
            item.job_id = None
            detached_job_refs += 1
            session.add(item)
    session.flush()
    session.exec(sa_delete(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id))
    session.delete(campaign)
    session.commit()

    return TranscriptOptimizationCampaignDeleteResponse(
        campaign_id=int(campaign_id),
        deleted_items=int(len(items)),
        detached_job_refs=int(detached_job_refs),
        status="deleted",
    )


@app.post("/videos/{video_id}/transcript-repair", response_model=TranscriptRepairQueueResponse)
def queue_transcript_repair(
    video_id: int,
    request: TranscriptRepairQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")

    quality = ingestion_service.evaluate_transcript_quality(session, video_id, source="queue_gate", persist_snapshot=False)
    if not request.force and str(quality.get("recommended_tier") or "") != "low_risk_repair":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'low_risk_repair'. Use force=true to queue anyway.",
        )

    job = _enqueue_unique_job(
        session,
        video_id=video_id,
        job_type="transcript_repair",
        payload={
            "save_files": bool(request.save_files),
            "force": bool(request.force),
            "note": str(request.note or "").strip() or None,
            "queued_from": "video",
        },
    )
    return TranscriptRepairQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@app.post("/channels/{channel_id}/transcript-repair/queue", response_model=TranscriptRepairBulkQueueResponse)
def queue_channel_transcript_repairs(
    channel_id: int,
    request: TranscriptRepairBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id, Video.processed == True)  # noqa: E712
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    queued_jobs: list[TranscriptRepairQueueResponse] = []
    skipped_active = 0
    skipped_no_segments = 0
    skipped_not_low_risk = 0

    for video in videos:
        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == video.id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            skipped_active += 1
            continue

        seg_exists = session.exec(
            select(TranscriptSegment.id).where(TranscriptSegment.video_id == video.id).limit(1)
        ).first()
        if not seg_exists:
            skipped_no_segments += 1
            continue

        quality = ingestion_service.evaluate_transcript_quality(session, int(video.id), source="bulk_queue_gate", persist_snapshot=False)
        if not request.force_non_eligible and str(quality.get("recommended_tier") or "") != "low_risk_repair":
            skipped_not_low_risk += 1
            continue

        job = _enqueue_unique_job(
            session,
            video_id=int(video.id),
            job_type="transcript_repair",
            payload={
                "save_files": bool(request.save_files),
                "force": bool(request.force_non_eligible),
                "note": str(request.note or "").strip() or None,
                "queued_from": "channel",
                "channel_id": int(channel_id),
            },
        )
        queued_jobs.append(
            TranscriptRepairQueueResponse(
                job_id=int(job.id),
                video_id=int(video.id),
                status="queued",
                recommended_tier=str(quality.get("recommended_tier") or "none"),
                quality_score=float(quality.get("quality_score") or 0.0),
                queued=True,
            )
        )

    return TranscriptRepairBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(queued_jobs),
        skipped_active=skipped_active,
        skipped_no_segments=skipped_no_segments,
        skipped_not_low_risk=skipped_not_low_risk,
        jobs=queued_jobs,
    )


@app.post("/videos/{video_id}/transcript-diarization-rebuild", response_model=TranscriptDiarizationRebuildQueueResponse)
def queue_transcript_diarization_rebuild(
    video_id: int,
    request: TranscriptDiarizationRebuildQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="video_rebuild",
    )
    _invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@app.post("/videos/{video_id}/transcript-diarization-benchmark", response_model=TranscriptDiarizationRebuildQueueResponse)
def queue_transcript_diarization_benchmark(
    video_id: int,
    request: TranscriptDiarizationBenchmarkRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="benchmark",
        optimization_target="diarization_benchmark",
        diarization_sensitivity_override=request.diarization_sensitivity,
        speaker_match_threshold_override=request.speaker_match_threshold,
    )
    _invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@app.post("/channels/{channel_id}/transcript-diarization-rebuild/queue", response_model=TranscriptDiarizationRebuildBulkQueueResponse)
def queue_channel_transcript_diarization_rebuilds(
    channel_id: int,
    request: TranscriptDiarizationRebuildBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    jobs: list[TranscriptDiarizationRebuildQueueResponse] = []
    skipped_active = 0
    skipped_no_raw_transcript = 0
    skipped_not_diarization_rebuild = 0
    skipped_unprocessed = 0
    skipped_muted = 0

    for video in videos:
        if bool(video.muted):
            skipped_muted += 1
            continue
        if not bool(video.processed):
            skipped_unprocessed += 1
            continue
        try:
            job, quality, _, _ = _queue_diarization_rebuild_job(
                session,
                video=video,
                force=bool(request.force_non_eligible),
                note=request.note,
                queued_from="channel_rebuild",
            )
            jobs.append(
                TranscriptDiarizationRebuildQueueResponse(
                    job_id=int(job.id),
                    video_id=int(video.id),
                    status="queued",
                    recommended_tier=str(quality.get("recommended_tier") or "none"),
                    quality_score=float(quality.get("quality_score") or 0.0),
                    queued=True,
                )
            )
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code == 400 and "active job" in detail_text.lower():
                skipped_active += 1
            elif exc.status_code == 400 and "no raw transcript" in detail_text.lower():
                skipped_no_raw_transcript += 1
            elif exc.status_code == 409 and "diarization_rebuild" in detail_text:
                skipped_not_diarization_rebuild += 1
            else:
                raise

    _invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(jobs),
        skipped_active=skipped_active,
        skipped_no_raw_transcript=skipped_no_raw_transcript,
        skipped_not_diarization_rebuild=skipped_not_diarization_rebuild,
        skipped_unprocessed=skipped_unprocessed,
        skipped_muted=skipped_muted,
        jobs=jobs,
    )


@app.post("/videos/{video_id}/transcript-retranscribe", response_model=TranscriptRetranscriptionQueueResponse)
def queue_transcript_retranscription(
    video_id: int,
    request: TranscriptRetranscriptionQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_full_retranscription_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="video_retranscription",
    )
    _invalidate_speaker_query_caches()
    return TranscriptRetranscriptionQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@app.post("/channels/{channel_id}/transcript-retranscribe/queue", response_model=TranscriptRetranscriptionBulkQueueResponse)
def queue_channel_transcript_retranscriptions(
    channel_id: int,
    request: TranscriptRetranscriptionBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    jobs: list[TranscriptRetranscriptionQueueResponse] = []
    skipped_active = 0
    skipped_not_full_retranscription = 0
    skipped_unprocessed = 0
    skipped_muted = 0

    for video in videos:
        if bool(video.muted):
            skipped_muted += 1
            continue
        if not bool(video.processed):
            skipped_unprocessed += 1
            continue
        try:
            job, quality, _, _ = _queue_full_retranscription_job(
                session,
                video=video,
                force=bool(request.force_non_eligible),
                note=request.note,
                queued_from="channel_retranscription",
            )
            jobs.append(
                TranscriptRetranscriptionQueueResponse(
                    job_id=int(job.id),
                    video_id=int(video.id),
                    status="queued",
                    recommended_tier=str(quality.get("recommended_tier") or "none"),
                    quality_score=float(quality.get("quality_score") or 0.0),
                    queued=True,
                )
            )
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code == 400 and "active job" in detail_text.lower():
                skipped_active += 1
            elif exc.status_code == 409 and "full_retranscription" in detail_text:
                skipped_not_full_retranscription += 1
            else:
                raise

    _invalidate_speaker_query_caches()
    return TranscriptRetranscriptionBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(jobs),
        skipped_active=skipped_active,
        skipped_not_full_retranscription=skipped_not_full_retranscription,
        skipped_unprocessed=skipped_unprocessed,
        skipped_muted=skipped_muted,
        jobs=jobs,
    )


@app.get("/channels/{channel_id}/episode-clone/candidates", response_model=List[EpisodeCloneCandidateRead])
def get_episode_clone_candidates(
    channel_id: int,
    limit: int = 20,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return [EpisodeCloneCandidateRead(**row) for row in clone_svc.list_clone_candidates(session, channel_id, limit=limit)]


def _get_episode_clone_engine_options() -> list[EpisodeCloneEngineRead]:
    current_provider = clone_svc._normalize_provider(os.getenv("LLM_PROVIDER") or "ollama") or "ollama"
    provider_models = {
        "ollama": (os.getenv("OLLAMA_MODEL") or "mistral").strip(),
        "nvidia_nim": (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip(),
        "openai": (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
        "anthropic": (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip(),
        "gemini": (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
        "groq": (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
        "openrouter": (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
        "xai": (os.getenv("XAI_MODEL") or "grok-2").strip(),
    }
    provider_key_env = {
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
    }
    llm_enabled = (os.getenv("LLM_ENABLED", "false").strip().lower() == "true") or (
        os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true"
    )

    def build_option(provider: str, *, is_default: bool) -> EpisodeCloneEngineRead:
        model = provider_models.get(provider) or ""
        disabled_reason = None
        available = bool(model)
        if not llm_enabled:
            available = False
            disabled_reason = "LLM is disabled in Settings."
        elif provider in provider_key_env and not (os.getenv(provider_key_env[provider]) or "").strip():
            available = False
            disabled_reason = "API key is not configured for this provider."
        elif provider == "ollama" and not model:
            available = False
            disabled_reason = "No Ollama model is configured."
        label_provider = provider.replace("_", " ").title()
        prefix = "Default" if is_default else label_provider
        return EpisodeCloneEngineRead(
            key="default" if is_default else provider,
            label=f"{prefix} ({label_provider} · {model or 'unconfigured'})",
            provider=provider,
            model=model,
            is_default=is_default,
            available=available,
            disabled_reason=disabled_reason,
        )

    options = [build_option(current_provider, is_default=True)]
    for provider in clone_svc.SUPPORTED_CLONE_PROVIDERS:
        if provider == current_provider:
            continue
        options.append(build_option(provider, is_default=False))
    return options


@app.get("/episode-clone/engines", response_model=List[EpisodeCloneEngineRead])
def list_episode_clone_engines():
    return _get_episode_clone_engine_options()


@app.post("/videos/{video_id}/episode-clone/concepts", response_model=EpisodeCloneConceptsResponse)
def detect_episode_clone_concepts(
    video_id: int,
    body: EpisodeCloneConceptsRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.channel_id:
        raise HTTPException(status_code=400, detail="Video is not attached to a channel")
    segment_exists = session.exec(
        select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id).limit(1)
    ).first()
    if segment_exists is None:
        raise HTTPException(status_code=400, detail="Source episode does not have a transcript yet")

    target_provider, target_model, target_name = ingestion_service.resolve_clone_llm_target(
        provider_override=body.provider_override,
        model_override=body.model_override,
    )
    try:
        result = clone_svc.extract_episode_clone_concepts(
            session,
            video_id=video_id,
            notes=body.notes,
            semantic_query=body.semantic_query,
            related_limit=body.related_limit,
            text_generator=lambda prompt: ingestion_service.generate_clone_text(
                prompt,
                provider_override=target_provider,
                model_override=target_model,
                temperature=0.2,
                num_predict=900,
                timeout_seconds=120,
            ),
            model_name=target_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EpisodeCloneConceptsResponse(**result)


def _build_episode_clone_job_read(job: Job) -> EpisodeCloneJobRead:
    payload: dict[str, object] = {}
    if job.payload_json:
        try:
            parsed = json.loads(job.payload_json)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

    request_payload = clone_svc.normalize_clone_request(
        style_prompt=str(payload.get("style_prompt") or ""),
        notes=payload.get("notes"),
        semantic_query=payload.get("semantic_query"),
        related_limit=int(payload.get("related_limit") or 8),
        variant_label=payload.get("variant_label"),
        provider_override=payload.get("provider_override"),
        model_override=payload.get("model_override"),
        approved_concepts=payload.get("approved_concepts"),
        excluded_references=payload.get("excluded_references"),
    )
    request_signature = str(
        payload.get("request_signature")
        or clone_svc.clone_request_signature(video_id=int(job.video_id), request_payload=request_payload)
    )

    result = None
    raw_result = payload.get("clone_result")
    if isinstance(raw_result, dict):
        try:
            result = EpisodeCloneGenerateResponse(**raw_result)
        except Exception:
            result = None

    return EpisodeCloneJobRead(
        job_id=int(job.id),
        video_id=int(job.video_id),
        status=str(job.status or ""),
        progress=int(job.progress or 0),
        status_detail=job.status_detail,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        request=EpisodeCloneGenerateRequest(**request_payload),
        request_signature=request_signature,
        result=result,
    )


@app.post("/videos/{video_id}/episode-clone/generate", response_model=EpisodeCloneJobRead)
def queue_episode_clone_generation(
    video_id: int,
    body: EpisodeCloneGenerateRequest,
    session: Session = Depends(get_session),
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.channel_id:
        raise HTTPException(status_code=400, detail="Video is not attached to a channel")
    segment_exists = session.exec(
        select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id).limit(1)
    ).first()
    if segment_exists is None:
        raise HTTPException(status_code=400, detail="Source episode does not have a transcript yet")

    request_payload = clone_svc.normalize_clone_request(
        style_prompt=body.style_prompt,
        notes=body.notes,
        semantic_query=body.semantic_query,
        related_limit=body.related_limit,
        variant_label=body.variant_label,
        provider_override=body.provider_override,
        model_override=body.model_override,
        approved_concepts=body.approved_concepts,
        excluded_references=body.excluded_references,
    )
    request_payload["request_signature"] = clone_svc.clone_request_signature(
        video_id=video_id,
        request_payload=request_payload,
    )
    job = _enqueue_unique_job(
        session,
        video_id=video_id,
        job_type="episode_clone",
        payload=request_payload,
    )
    return _build_episode_clone_job_read(job)


@app.get("/videos/{video_id}/episode-clone/jobs", response_model=List[EpisodeCloneJobRead])
def list_episode_clone_jobs(
    video_id: int,
    limit: int = 12,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    safe_limit = max(1, min(int(limit or 12), 50))
    jobs = session.exec(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.job_type == "episode_clone",
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(safe_limit)
    ).all()
    return [_build_episode_clone_job_read(job) for job in jobs]


@app.get("/jobs/{job_id}/episode-clone", response_model=EpisodeCloneJobRead)
def read_episode_clone_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(job.job_type or "").strip().lower() != "episode_clone":
        raise HTTPException(status_code=400, detail="Job is not an episode clone job")
    return _build_episode_clone_job_read(job)


@app.get("/videos/{video_id}/media")
def stream_video_media(video_id: int, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if (video.media_source_type or "youtube") == "youtube":
        raise HTTPException(status_code=409, detail="This episode is streamed from YouTube, not from local media.")

    media_path = ingestion_service.get_audio_path(video, purpose="playback")
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Local media is not available yet. Start processing or wait for download to complete.")

    media_type = mimetypes.guess_type(str(media_path))[0] or "application/octet-stream"
    return FileResponse(path=media_path, media_type=media_type, filename=media_path.name)


@app.patch("/videos/{video_id}/voicefixer/settings", response_model=Video)
def update_voicefixer_settings(
    video_id: int,
    body: VoiceFixerSettingsUpdateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot change VoiceFixer settings while this episode has an active job.")

    mode = max(0, min(2, int(body.mode)))
    mix_ratio = max(0.0, min(1.0, float(body.mix_ratio)))
    leveling_mode = str(body.leveling_mode or "off").strip().lower()
    if leveling_mode not in {"off", "gentle", "balanced", "strong"}:
        raise HTTPException(status_code=400, detail="Invalid VoiceFixer leveling mode.")
    apply_scope = str(body.apply_scope or "none").strip().lower()
    if apply_scope not in {"none", "playback", "processing", "both"}:
        raise HTTPException(status_code=400, detail="Invalid VoiceFixer apply scope.")

    settings_changed = (
        int(video.voicefixer_mode or 0) != mode
        or abs(float(video.voicefixer_mix_ratio or 1.0) - mix_ratio) > 1e-6
        or str(video.voicefixer_leveling_mode or "off").strip().lower() != leveling_mode
    )
    video.voicefixer_mode = mode
    video.voicefixer_mix_ratio = mix_ratio
    video.voicefixer_leveling_mode = leveling_mode
    video.voicefixer_apply_scope = apply_scope
    video.voicefixer_use_cleaned = apply_scope != "none"
    if settings_changed and (
        str(getattr(video, "voicefixer_cleaned_path", "") or "").strip()
        or str(getattr(video, "voicefixer_status", "") or "").strip().lower() in {"ready", "disabled"}
    ):
        _invalidate_voicefixer_output(video)
    if apply_scope == "none" and str(video.voicefixer_status or "").strip().lower() == "ready":
        video.voicefixer_status = "disabled"
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@app.post("/videos/{video_id}/reconstruct/queue", response_model=Video)
def queue_conversation_reconstruction(video_id: int, force: bool = False, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    install_info = _get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="This episode already has an active processing job. Wait for it to finish first.")

    video.reconstruction_status = "queued"
    video.reconstruction_error = None
    session.add(video)
    session.commit()
    _enqueue_unique_job(session, video_id=video_id, job_type="conversation_reconstruct", payload={"force": bool(force)})
    session.refresh(video)
    return video


@app.patch("/videos/{video_id}/reconstruction/settings", response_model=Video)
def update_reconstruction_settings(
    video_id: int,
    body: ReconstructionSettingsUpdateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot change reconstruction settings while this episode has an active job.")

    mode = "performance"
    instruction_template = str(body.instruction_template or "").strip() or None

    settings_changed = (
        str(video.reconstruction_mode or "basic").strip().lower() != mode
        or str(video.reconstruction_instruction_template or "").strip() != str(instruction_template or "")
    )
    video.reconstruction_mode = mode
    video.reconstruction_instruction_template = instruction_template
    if settings_changed and (
        str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        or str(getattr(video, "reconstruction_status", "") or "").strip().lower() == "ready"
    ):
        _invalidate_reconstruction_output(video)
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@app.get("/videos/{video_id}/reconstruction/workbench", response_model=ReconstructionWorkbenchRead)
def get_reconstruction_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    segments = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
    ).all()
    if not segments:
        raise HTTPException(status_code=409, detail="Transcript segments are required before opening the reconstruction workbench.")
    if not any(getattr(seg, "speaker_id", None) is not None for seg in segments):
        raise HTTPException(status_code=409, detail="Diarization must be completed before opening the reconstruction workbench.")

    try:
        source_path = ingestion_service.get_audio_path(video, purpose="processing")
        workbench = ingestion_service._build_reconstruction_workbench(
            video,
            segments,
            source_path,
            progress_task="load_workbench",
        )
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="load_workbench",
            status="error",
            stage="error",
            message=str(e)[:240] or "Failed to load reconstruction workbench.",
        )
        raise
    base_url = f"/videos/{video_id}/reconstruction/workbench/audio"
    workbench_dir = ingestion_service._reconstruction_workbench_dir(video)

    def _workbench_audio_url(filename: str) -> str | None:
        safe_name = str(filename or "").strip()
        if not safe_name:
            return None
        audio_path = workbench_dir / Path(safe_name).name
        version = ""
        try:
            version = str(int(audio_path.stat().st_mtime_ns))
        except Exception:
            version = str(int(time.time() * 1000))
        params = {"name": Path(safe_name).name, "v": version}
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    speakers = []
    for item in workbench["speakers"]:
        ref_name = str(item.pop("reference_audio_filename", "") or "").strip()
        test_name = str(item.pop("latest_test_audio_filename", "") or "").strip()
        samples = []
        for sample in item.pop("samples", []) or []:
            sample_audio_name = str(sample.pop("audio_filename", "") or "").strip()
            sample_cleaned_name = str(sample.pop("cleaned_audio_filename", "") or "").strip()
            sample["audio_url"] = _workbench_audio_url(sample_audio_name)
            sample["cleaned_audio_url"] = _workbench_audio_url(sample_cleaned_name)
            samples.append(sample)
        item["reference_audio_url"] = _workbench_audio_url(ref_name)
        item["latest_test_audio_url"] = _workbench_audio_url(test_name)
        item["samples"] = samples
        speakers.append(item)
    workbench["speakers"] = speakers
    return ReconstructionWorkbenchRead(**workbench)


@app.get("/videos/{video_id}/workbench/progress", response_model=WorkbenchTaskProgressRead)
def get_video_workbench_progress(video_id: int):
    try:
        if ingestion_service is None:
            return WorkbenchTaskProgressRead(video_id=int(video_id), status="idle")
        return WorkbenchTaskProgressRead(**ingestion_service.get_workbench_task_progress(video_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workbench progress unavailable: {e}")


@app.get("/videos/{video_id}/reconstruction/workbench/audio")
def stream_reconstruction_workbench_audio(video_id: int, name: str, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    safe_name = Path(str(name or "")).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Missing workbench audio filename.")
    workbench_dir = ingestion_service._reconstruction_workbench_dir(video)
    audio_path = workbench_dir / safe_name
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Workbench audio file not found.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        filename=audio_path.name,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/videos/{video_id}/reconstruction/test-speaker", response_model=ReconstructionSpeakerTestResult)
def test_reconstruction_speaker(
    video_id: int,
    body: ReconstructionSpeakerTestRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    install_info = _get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")

    try:
        result = ingestion_service.generate_reconstruction_speaker_test(
            video_id,
            speaker_id=int(body.speaker_id),
            text=body.text,
            segment_id=body.segment_id,
            performance_mode=bool(body.performance_mode),
            progress_task="speaker_test",
        )
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="speaker_test",
            status="error",
            stage="error",
            message=f"Speaker test synthesis failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Speaker test synthesis failed: {e}")

    audio_filename = str(result.pop("audio_filename"))
    try:
        audio_path = ingestion_service._reconstruction_workbench_dir(video) / Path(audio_filename).name
        version = str(int(audio_path.stat().st_mtime_ns))
    except Exception:
        version = str(int(time.time() * 1000))
    result["audio_url"] = f"/videos/{video_id}/reconstruction/workbench/audio?{urllib.parse.urlencode({'name': Path(audio_filename).name, 'v': version})}"
    return ReconstructionSpeakerTestResult(**result)


@app.post("/videos/{video_id}/reconstruction/workbench/sample-cleanup", response_model=ReconstructionWorkbenchRead)
def cleanup_reconstruction_workbench_sample(
    video_id: int,
    body: ReconstructionWorkbenchSampleCleanupRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    if not _get_voicefixer_install_info().installed:
        raise HTTPException(status_code=409, detail="VoiceFixer is not installed yet. Install and test it from Settings first.")
    try:
        ingestion_service.cleanup_reconstruction_sample(
            video_id,
            speaker_id=int(body.speaker_id),
            segment_id=int(body.segment_id),
        )
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="sample_cleanup",
            status="error",
            stage="error",
            message=f"Sample cleanup failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Sample cleanup failed: {e}")
    return get_reconstruction_workbench(video_id, session)


@app.patch("/videos/{video_id}/reconstruction/workbench/sample-state", response_model=ReconstructionWorkbenchRead)
def update_reconstruction_workbench_sample_state(
    video_id: int,
    body: ReconstructionWorkbenchSampleStateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        ingestion_service.update_reconstruction_sample_state(
            video_id,
            speaker_id=int(body.speaker_id),
            segment_id=int(body.segment_id),
            rejected=body.rejected,
            selected=body.selected,
            clear_cleaned=body.clear_cleaned,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update sample state: {e}")
    return get_reconstruction_workbench(video_id, session)


@app.post("/videos/{video_id}/reconstruction/workbench/add-sample", response_model=ReconstructionWorkbenchRead)
def add_reconstruction_workbench_sample(
    video_id: int,
    body: ReconstructionWorkbenchAddSampleRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        ingestion_service.add_reconstruction_performance_sample(video_id, speaker_id=int(body.speaker_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add a new sample: {e}")
    return get_reconstruction_workbench(video_id, session)


@app.patch("/videos/{video_id}/reconstruction/workbench/speaker-approval", response_model=ReconstructionWorkbenchRead)
def set_reconstruction_workbench_speaker_approval(
    video_id: int,
    body: ReconstructionWorkbenchSpeakerApprovalRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        ingestion_service.set_reconstruction_speaker_approval(video_id, speaker_id=int(body.speaker_id), approved=bool(body.approved))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update speaker approval: {e}")
    return get_reconstruction_workbench(video_id, session)


@app.post("/videos/{video_id}/reconstruction/preview-segment", response_model=ReconstructionSegmentPreviewResult)
def preview_reconstruction_segment(
    video_id: int,
    body: ReconstructionSegmentPreviewRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    install_info = _get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")
    try:
        result = ingestion_service.preview_reconstruction_segment(
            video_id,
            segment_id=int(body.segment_id),
            performance_mode=body.performance_mode,
        )
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="preview_segment",
            status="error",
            stage="error",
            message=f"Segment preview failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Segment preview failed: {e}")
    audio_filename = str(result.pop("audio_filename"))
    try:
        audio_path = ingestion_service._reconstruction_workbench_dir(video) / Path(audio_filename).name
        version = str(int(audio_path.stat().st_mtime_ns))
    except Exception:
        version = str(int(time.time() * 1000))
    result["audio_url"] = f"/videos/{video_id}/reconstruction/workbench/audio?{urllib.parse.urlencode({'name': Path(audio_filename).name, 'v': version})}"
    return ReconstructionSegmentPreviewResult(**result)


@app.get("/videos/{video_id}/reconstruction/audio")
def stream_reconstruction_audio(video_id: int, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
    if not rel:
        raise HTTPException(status_code=404, detail="No reconstructed audio is available for this episode yet.")
    audio_path = ingestion_service.get_manual_media_absolute_path(rel)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Reconstructed audio file is missing.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@app.patch("/videos/{video_id}/reconstruction/use-for-playback", response_model=Video)
def set_reconstruction_use_for_playback(
    video_id: int,
    body: ReconstructionUseForPlaybackRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch reconstruction playback while this episode has an active job.")

    if body.enabled:
        rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        if not rel:
            raise HTTPException(status_code=409, detail="No reconstructed audio exists yet for this episode.")
        audio_path = ingestion_service.get_manual_media_absolute_path(rel)
        if audio_path is None or not audio_path.exists():
            raise HTTPException(status_code=409, detail="The reconstructed audio file is missing.")
        video.reconstruction_use_for_playback = True
        if str(video.reconstruction_status or "").strip().lower() in {"", "disabled"}:
            video.reconstruction_status = "ready"
        video.reconstruction_error = None
    else:
        video.reconstruction_use_for_playback = False
        video.reconstruction_error = None

    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@app.patch("/videos/{video_id}/playback-source", response_model=Video)
def set_uploaded_playback_source(
    video_id: int,
    body: UploadedPlaybackSourceRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Playback source switching is only available for manually uploaded media.")

    source = str(body.source or "original").strip().lower()
    if source not in {"original", "cleaned", "reconstructed"}:
        raise HTTPException(status_code=400, detail="Invalid playback source.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch playback source while this episode has an active job.")

    apply_scope = str(getattr(video, "voicefixer_apply_scope", "") or "").strip().lower()
    processing_uses_cleaned = apply_scope in {"processing", "both"}

    if source == "cleaned":
        cleaned_path = ingestion_service.get_voicefixer_cleaned_absolute_path(video)
        if cleaned_path is None or not cleaned_path.exists():
            raise HTTPException(status_code=409, detail="No VoiceFixer-cleaned media exists yet for this episode.")
        video.reconstruction_use_for_playback = False
        video.voicefixer_apply_scope = "both" if processing_uses_cleaned else "playback"
        video.voicefixer_use_cleaned = True
        video.voicefixer_status = "ready"
        video.voicefixer_error = None
    elif source == "reconstructed":
        rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        if not rel:
            raise HTTPException(status_code=409, detail="No reconstructed audio exists yet for this episode.")
        audio_path = ingestion_service.get_manual_media_absolute_path(rel)
        if audio_path is None or not audio_path.exists():
            raise HTTPException(status_code=409, detail="The reconstructed audio file is missing.")
        video.reconstruction_use_for_playback = True
        if str(video.reconstruction_status or "").strip().lower() in {"", "disabled"}:
            video.reconstruction_status = "ready"
        video.reconstruction_error = None
    else:
        video.reconstruction_use_for_playback = False
        video.voicefixer_apply_scope = "processing" if processing_uses_cleaned else "none"
        video.voicefixer_use_cleaned = processing_uses_cleaned
        if not processing_uses_cleaned and str(video.voicefixer_status or "").strip().lower() == "ready":
            video.voicefixer_status = "disabled"
        video.voicefixer_error = None

    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@app.get("/videos/{video_id}/cleanup/workbench", response_model=CleanupWorkbenchRead)
def get_cleanup_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = ingestion_service.build_cleanup_workbench(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load the cleanup workbench: {e}")
    clearvoice_info = _get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@app.post("/videos/{video_id}/cleanup/workbench/analyze", response_model=CleanupWorkbenchRead)
def analyze_cleanup_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = ingestion_service.analyze_cleanup_workbench_audio(video_id)
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="analyze",
            status="error",
            stage="error",
            message=str(e)[:240] or "Cleanup analysis failed.",
        )
        raise HTTPException(status_code=500, detail=f"Cleanup analysis failed: {e}")
    clearvoice_info = _get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@app.post("/videos/{video_id}/cleanup/workbench/clearvoice-candidate", response_model=CleanupWorkbenchRead)
def run_cleanup_clearvoice_candidate(
    video_id: int,
    body: CleanupWorkbenchRunCandidateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    clearvoice_info = _get_clearvoice_install_info()
    if not clearvoice_info.installed:
        raise HTTPException(status_code=409, detail="ClearVoice is not installed yet. Install and test it from the cleanup workbench first.")
    if not clearvoice_info.runtime_ready:
        raise HTTPException(
            status_code=409,
            detail=clearvoice_info.message or clearvoice_info.runtime_error or "ClearVoice is installed, but its runtime is not healthy yet.",
        )
    try:
        workbench = ingestion_service.run_cleanup_clearvoice_candidate(
            video_id,
            stage=body.stage,
            model_name=body.model_name,
            source_candidate_id=body.source_candidate_id,
        )
    except Exception as e:
        ingestion_service._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="clearvoice_candidate",
            status="error",
            stage="error",
            message=str(e)[:240] or "ClearVoice candidate generation failed.",
        )
        raise HTTPException(status_code=500, detail=f"ClearVoice candidate generation failed: {e}")
    workbench["clearvoice_available"] = True
    return CleanupWorkbenchRead(**workbench)


@app.patch("/videos/{video_id}/cleanup/workbench/select-candidate", response_model=CleanupWorkbenchRead)
def select_cleanup_workbench_candidate(
    video_id: int,
    body: CleanupWorkbenchSelectCandidateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = ingestion_service.select_cleanup_workbench_candidate(video_id, candidate_id=body.candidate_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update the selected pre-cleanup candidate: {e}")
    clearvoice_info = _get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@app.get("/videos/{video_id}/cleanup/workbench/audio")
def get_cleanup_workbench_audio(video_id: int, name: str, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench audio is only available for manually uploaded media.")
    workbench_dir = ingestion_service._cleanup_workbench_dir(video)
    audio_path = workbench_dir / Path(str(name or "")).name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Cleanup workbench audio file not found.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@app.post("/videos/{video_id}/voicefixer/queue", response_model=Video)
def queue_voicefixer_cleanup(video_id: int, force: bool = False, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    install_info = _get_voicefixer_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="VoiceFixer is not installed yet. Install and test it from Settings first.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="This episode already has an active processing job. Wait for it to finish first.")

    ingestion_service._set_voicefixer_state(video_id, status="queued", error="")
    _enqueue_unique_job(session, video_id=video_id, job_type="voicefixer_cleanup", payload={"force": bool(force)})
    session.refresh(video)
    return session.get(Video, video_id)


@app.patch("/videos/{video_id}/voicefixer/use-cleaned", response_model=Video)
def set_voicefixer_use_cleaned(
    video_id: int,
    body: VoiceFixerUseCleanedRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch VoiceFixer media while this episode has an active job.")

    if body.enabled:
        cleaned_path = ingestion_service.get_voicefixer_cleaned_absolute_path(video)
        if cleaned_path is None or not cleaned_path.exists():
            raise HTTPException(status_code=409, detail="No VoiceFixer-cleaned media exists yet for this episode.")
        video.voicefixer_use_cleaned = True
        video.voicefixer_apply_scope = "both"
        video.voicefixer_status = "ready"
        video.voicefixer_error = None
    else:
        video.voicefixer_use_cleaned = False
        video.voicefixer_apply_scope = "none"
        if str(video.voicefixer_status or "").strip().lower() == "ready":
            video.voicefixer_status = "disabled"
        video.voicefixer_error = None
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def _extract_publish_datetime(info: dict) -> Optional[datetime]:
    upload_date_str = info.get("upload_date")
    if upload_date_str:
        try:
            return datetime.strptime(str(upload_date_str), "%Y%m%d")
        except ValueError:
            pass

    timestamp_value = info.get("release_timestamp") or info.get("timestamp")
    if timestamp_value:
        try:
            return datetime.fromtimestamp(float(timestamp_value))
        except Exception:
            return None
    return None


def _extract_best_thumbnail_url(info: dict) -> Optional[str]:
    thumbnails = info.get("thumbnails") or []
    for thumb in reversed(thumbnails):
        url = str(thumb.get("url") or "").strip()
        if url:
            return url
    return None


def _make_unique_external_video_id(prefix: str, raw_id: str, session: Session) -> str:
    base = f"{prefix}_{raw_id}" if raw_id else f"{prefix}_{secrets.token_hex(8)}"
    candidate = base
    while session.exec(select(Video).where(Video.youtube_id == candidate)).first():
        candidate = f"{base}_{secrets.token_hex(3)}"
    return candidate


def _normalize_tiktok_video_url(url: str) -> str:
    text = " ".join((url or "").strip().split())
    match = re.search(r"tiktok\.com/@([^/?#]+)/video/(\d+)", text, re.IGNORECASE)
    if match:
        return f"https://www.tiktok.com/@{match.group(1)}/video/{match.group(2)}"
    vm_match = re.search(r"(https?://vm\.tiktok\.com/[A-Za-z0-9]+/?|https?://vt\.tiktok\.com/[A-Za-z0-9]+/?)", text, re.IGNORECASE)
    if vm_match:
        return vm_match.group(1)
    return text


def _fetch_remote_video_info(url: str) -> dict:
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True}
    ydl_opts = _apply_ytdlp_auth_opts(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("Failed to extract remote video metadata.")
    return info

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


def _queue_diarization_rebuild_job(
    session: Session,
    *,
    video: Video,
    force: bool,
    note: Optional[str],
    queued_from: str,
    optimization_target: str = "diarization_rebuild",
    diarization_sensitivity_override: Optional[str] = None,
    speaker_match_threshold_override: Optional[float] = None,
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(select(Job).where(
        Job.video_id == video.id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before redoing diarization.")

    try:
        audio_path = ingestion_service.get_audio_path(video)
        safe_title = ingestion_service.sanitize_filename(video.title)
        raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
        if not raw_path.exists():
            raise HTTPException(status_code=400, detail="No raw transcript found. Use full redo instead.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not locate audio/transcript files. Use full redo instead.")

    quality = ingestion_service.evaluate_transcript_quality(session, int(video.id), source="diarization_queue_gate", persist_snapshot=False)
    if not force and str(quality.get("recommended_tier") or "") != "diarization_rebuild":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'diarization_rebuild'. Use force=true to queue anyway.",
        )

    segments = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)
    ).all()
    funny_rows = session.exec(
        select(FunnyMoment).where(FunnyMoment.video_id == video.id).order_by(FunnyMoment.start_time)
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
    backup_path = ingestion_service._get_temp_redo_backup_path(int(video.id))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f, ensure_ascii=False)

    d_path = ingestion_service._get_temp_diarization_path(video.id)
    if d_path.exists():
        try:
            os.unlink(d_path)
        except Exception:
            pass

    video.status = "downloaded"
    video.processed = False
    session.add(video)

    payload = {
        "mode": "redo_diarization",
        "optimization_target": str(optimization_target or "diarization_rebuild"),
        "queued_from": str(queued_from or "manual"),
        "redo_diarization_backup_file": str(backup_path),
        "quality_profile_before": quality.get("quality_profile"),
        "recommended_tier_before": quality.get("recommended_tier"),
        "quality_score_before": quality.get("quality_score"),
        "quality_metrics_before": quality.get("metrics"),
        "quality_reasons_before": quality.get("reasons"),
        "note": str(note or "").strip() or None,
        "diarization_sensitivity_override": str(diarization_sensitivity_override or "").strip() or None,
        "speaker_match_threshold_override": float(speaker_match_threshold_override) if speaker_match_threshold_override is not None else None,
        "benchmark_variant": str(optimization_target or "").strip().lower() == "diarization_benchmark",
    }

    last_process_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.job_type == "process",
            Job.status == "completed"
        ).order_by(Job.completed_at.desc())
    ).first()
    if last_process_job and last_process_job.payload_json:
        try:
            old_payload = json.loads(last_process_job.payload_json)
            for k, v in old_payload.items():
                if k.startswith("stage_transcribe") or k.startswith("parakeet_") or k.startswith("transcription_"):
                    payload[k] = v
        except Exception as e:
            log(f"Failed to inherit transcription stats for redo: {e}")

    job = _enqueue_unique_job(session, video_id=int(video.id), job_type="process", payload=payload)
    return job, quality, len(segments), len(funny_rows)


def _queue_full_retranscription_job(
    session: Session,
    *,
    video: Video,
    force: bool,
    note: Optional[str],
    queued_from: str,
):
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(select(Job).where(
        Job.video_id == video.id,
        Job.status.in_(PIPELINE_ACTIVE_STATUSES)
    )).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before redoing transcription.")

    quality = ingestion_service.evaluate_transcript_quality(session, int(video.id), source="retranscription_queue_gate", persist_snapshot=False)
    if not force and str(quality.get("recommended_tier") or "") != "full_retranscription":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'full_retranscription'. Use force=true to queue anyway.",
        )

    segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id)).all()
    funny_rows = session.exec(select(FunnyMoment).where(FunnyMoment.video_id == video.id)).all()
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
    backup_path = ingestion_service._get_temp_redo_backup_path(int(video.id))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_payload, f, ensure_ascii=False)

    ingestion_service.purge_artifacts(video.id, delete_raw_transcript=True, delete_audio=False)

    try:
        if ingestion_service.get_audio_path(video).exists():
            video.status = "downloaded"
        else:
            video.status = "pending"
    except Exception:
        video.status = "pending"
    video.transcript_source = None
    video.transcript_language = None
    video.transcript_is_placeholder = False
    video.processed = False
    session.add(video)

    payload = {
        "mode": "full_retranscription",
        "optimization_target": "full_retranscription",
        "queued_from": str(queued_from or "manual"),
        "redo_diarization_backup_file": str(backup_path),
        "force_retranscription": True,
        "quality_profile_before": quality.get("quality_profile"),
        "recommended_tier_before": quality.get("recommended_tier"),
        "quality_score_before": quality.get("quality_score"),
        "quality_metrics_before": quality.get("metrics"),
        "quality_reasons_before": quality.get("reasons"),
        "note": str(note or "").strip() or None,
    }
    job = _enqueue_unique_job(session, video_id=int(video.id), job_type="process", payload=payload)
    return job, quality, len(segments), len(funny_rows)

@app.post("/videos/{video_id}/process")
def process_video(video_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.access_restricted:
        raise HTTPException(
            status_code=409,
            detail=video.access_restriction_reason or "This video is not accessible with the current YouTube session.",
        )
    if str(getattr(video, "voicefixer_status", "") or "").lower() in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="VoiceFixer cleanup is still running for this episode. Wait for it to finish before processing.")
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
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    queued = ingestion_service._queue_channel_unprocessed_videos(session, channel_id)
    session.commit()
    return {"queued": int(queued)}

@app.post("/jobs/pause-all")
def pause_all_jobs(session: Session = Depends(get_session)):
    """Pause all queued jobs. Running jobs will complete."""
    statement = select(Job).where(Job.status == "queued")
    jobs = session.exec(statement).all()
    
    count = 0
    for job in jobs:
        job.status = "paused"
        _sync_auxiliary_video_job_state(session, job, "paused")
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
        _sync_auxiliary_video_job_state(session, job, "queued")
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
    is_youtube_video = str(getattr(video, "media_source_type", "") or "youtube").lower() == "youtube" and bool(video.youtube_id)
    should_push = (_youtube_get_cfg()["push_enabled"] and is_youtube_video) if push_to_youtube is None else bool(push_to_youtube and is_youtube_video)
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

@app.post("/search/semantic", response_model=SemanticSearchPage)
def search_semantic(body: SemanticSearchRequest):
    """Semantic or hybrid transcript search using chunk embeddings."""
    q = (body.query or "").strip()
    if not q:
        return SemanticSearchPage(items=[], total=0, limit=body.limit, offset=body.offset)

    safe_limit = max(1, min(body.limit, 200))
    safe_offset = max(0, body.offset)

    try:
        if body.mode == "hybrid":
            result = sem_svc.hybrid_search(
                query=q,
                channel_id=body.channel_id,
                video_id=body.video_id,
                speaker_id=body.speaker_id,
                year=body.year,
                month=body.month,
                limit=safe_limit,
                offset=safe_offset,
            )
        else:
            result = sem_svc.semantic_search(
                query=q,
                channel_id=body.channel_id,
                video_id=body.video_id,
                speaker_id=body.speaker_id,
                year=body.year,
                month=body.month,
                limit=safe_limit,
                offset=safe_offset,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {exc}")

    items = [SemanticSearchHit(**hit) for hit in result["items"]]
    return SemanticSearchPage(
        items=items,
        total=result["total"],
        limit=safe_limit,
        offset=safe_offset,
    )


@app.get("/semantic-index/status", response_model=SemanticIndexStatus)
def get_semantic_index_status():
    """Return current semantic indexing job progress."""
    return SemanticIndexStatus(**sem_svc.get_indexing_status())


@app.post("/videos/{video_id}/semantic-index/rebuild", response_model=SemanticIndexRebuildResponse)
def rebuild_video_semantic_index(video_id: int, session: Session = Depends(get_session)):
    """Queue a semantic index rebuild for a single video."""
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    started = sem_svc.start_index_job([video_id])
    if not started:
        return SemanticIndexRebuildResponse(
            started=False,
            message="An indexing job is already running.",
            video_ids=[video_id],
        )
    return SemanticIndexRebuildResponse(
        started=True,
        message=f"Semantic indexing started for video {video_id}.",
        video_ids=[video_id],
    )


@app.post("/channels/{channel_id}/semantic-index/rebuild", response_model=SemanticIndexRebuildResponse)
def rebuild_channel_semantic_index(channel_id: int, session: Session = Depends(get_session)):
    """Queue a semantic index rebuild for all processed videos in a channel."""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .where(Video.processed == True)
    ).all()
    video_ids = [v.id for v in videos]

    if not video_ids:
        return SemanticIndexRebuildResponse(
            started=False,
            message="No processed videos found for this channel.",
            video_ids=[],
        )

    started = sem_svc.start_index_job(video_ids)
    if not started:
        return SemanticIndexRebuildResponse(
            started=False,
            message="An indexing job is already running.",
            video_ids=video_ids,
        )
    return SemanticIndexRebuildResponse(
        started=True,
        message=f"Semantic indexing started for {len(video_ids)} video(s) in channel {channel_id}.",
        video_ids=video_ids,
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


def _speaker_scope_key(channel_id: Optional[int], video_id: Optional[int], search: Optional[str] = None) -> str:
    normalized_search = (search or "").strip().lower()
    return f"channel:{channel_id or 'all'}|video:{video_id or 'all'}|search:{normalized_search or 'all'}"


def _build_speaker_scope_totals_subquery(
    *,
    channel_id: Optional[int],
    video_id: Optional[int],
):
    from sqlalchemy import func

    seg_duration = (TranscriptSegment.end_time - TranscriptSegment.start_time)
    total_time = func.sum(seg_duration).label("total_time")
    query = (
        select(
            TranscriptSegment.speaker_id.label("speaker_id"),
            total_time,
        )
        .where(TranscriptSegment.speaker_id.is_not(None))
    )

    if video_id:
        query = query.where(TranscriptSegment.video_id == video_id)

    if channel_id:
        query = query.join(Speaker, Speaker.id == TranscriptSegment.speaker_id).where(Speaker.channel_id == channel_id)

    query = query.group_by(TranscriptSegment.speaker_id).having(total_time > 5.0)
    return query.subquery("speaker_scope_totals")


def _build_speaker_scope_list_query(
    *,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str] = None,
):
    from sqlalchemy import func

    totals = _build_speaker_scope_totals_subquery(channel_id=channel_id, video_id=video_id)
    total_time = totals.c.total_time
    query = (
        select(
            Speaker.id,
            Speaker.channel_id,
            Speaker.name,
            Speaker.thumbnail_path,
            Speaker.is_extra,
            Speaker.created_at,
            total_time,
        )
        .join(totals, totals.c.speaker_id == Speaker.id)
    )

    if channel_id:
        query = query.where(Speaker.channel_id == channel_id)

    normalized_search = (search or "").strip().lower()
    if normalized_search:
        query = query.where(func.lower(Speaker.name).like(f"%{normalized_search}%"))

    return query, total_time


def _query_speaker_page_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str],
    offset: int,
    limit: Optional[int],
) -> list[dict]:
    scope_key = _speaker_scope_key(channel_id, video_id, search)
    cache_key = f"{scope_key}|offset:{offset}|limit:{limit if limit is not None else 'all'}"
    cached_rows = _get_speaker_list_cache(cache_key)
    if cached_rows is not None:
        return cached_rows

    if limit is None:
        out = _query_full_speaker_scope_rows(
            session=session,
            channel_id=channel_id,
            video_id=video_id,
            search=search,
        )[offset:]
        _set_speaker_list_cache(cache_key, list(out))
        return list(out)

    query, total_time = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id, search=search)
    query = query.order_by(total_time.desc()).offset(max(0, offset)).limit(max(0, limit))

    rows = session.exec(query).all()
    out: list[dict] = []
    for speaker_id, speaker_channel_id, name, thumbnail_path, is_extra, created_at, total_time_value in rows:
        out.append(
            {
                "id": int(speaker_id),
                "channel_id": int(speaker_channel_id),
                "name": str(name),
                "thumbnail_path": thumbnail_path,
                "is_extra": bool(is_extra),
                "created_at": created_at,
                "total_speaking_time": round(float(total_time_value or 0.0), 1),
            }
        )
    _set_speaker_list_cache(cache_key, out)
    return out


def _query_speaker_count_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
) -> list[tuple[str, bool, float]]:
    query, _ = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id)
    rows_subquery = query.subquery("speaker_scope_count_rows")
    count_query = select(
        rows_subquery.c.name,
        rows_subquery.c.is_extra,
        rows_subquery.c.total_time,
    )
    return [
        (
            str(name or ""),
            bool(is_extra),
            float(total_time_value or 0.0),
        )
        for name, is_extra, total_time_value in session.exec(count_query).all()
    ]


def _query_speaker_count_summary(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
) -> dict[str, int]:
    cached_rows = _get_speaker_scope_cache(_speaker_scope_key(channel_id, video_id))
    if cached_rows is not None:
        return _summarize_speaker_scope_rows(cached_rows)

    from sqlalchemy import case, func, or_

    query, _ = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id)
    rows_subquery = query.subquery("speaker_scope_count_summary")
    name_col = rows_subquery.c.name
    is_extra_col = rows_subquery.c.is_extra
    total_time_col = rows_subquery.c.total_time

    if IS_POSTGRES:
        unknown_expr = or_(
            func.btrim(name_col) == "",
            func.lower(name_col).in_(["unknown", "unknown speaker"]),
            name_col.op("~*")(r"^speaker\s+\d+$"),
        )
    else:
        lowered_name = func.lower(name_col)
        unknown_expr = or_(
            func.trim(name_col) == "",
            lowered_name.in_(["unknown", "unknown speaker"]),
            lowered_name.like("speaker %"),
        )

    extras_expr = or_(is_extra_col.is_(True), total_time_col < 60.0)
    summary_query = select(
        func.count().label("total"),
        func.sum(case((unknown_expr, 1), else_=0)).label("unknown"),
        func.sum(case((~unknown_expr, 1), else_=0)).label("identified"),
        func.sum(case((~unknown_expr & extras_expr, 1), else_=0)).label("extras"),
        func.sum(case((~unknown_expr & ~extras_expr, 1), else_=0)).label("main"),
    )
    row = session.exec(summary_query).first()
    if not row:
        return {"total": 0, "identified": 0, "unknown": 0, "main": 0, "extras": 0}
    return {
        "total": int(row[0] or 0),
        "unknown": int(row[1] or 0),
        "identified": int(row[2] or 0),
        "extras": int(row[3] or 0),
        "main": int(row[4] or 0),
    }


def _query_full_speaker_scope_rows(
    *,
    session: Session,
    channel_id: Optional[int],
    video_id: Optional[int],
    search: Optional[str] = None,
) -> list[dict]:
    scope_key = _speaker_scope_key(channel_id, video_id, search)
    cached_rows = _get_speaker_scope_cache(scope_key)
    if cached_rows is not None:
        return cached_rows

    query, total_time = _build_speaker_scope_list_query(channel_id=channel_id, video_id=video_id, search=search)
    query = query.order_by(total_time.desc())

    rows = session.exec(query).all()
    out: list[dict] = []
    for speaker_id, speaker_channel_id, name, thumbnail_path, is_extra, created_at, total_time_value in rows:
        out.append(
            {
                "id": int(speaker_id),
                "channel_id": int(speaker_channel_id),
                "name": str(name),
                "thumbnail_path": thumbnail_path,
                "is_extra": bool(is_extra),
                "created_at": created_at,
                "total_speaking_time": round(float(total_time_value or 0.0), 1),
            }
        )

    _set_speaker_scope_cache(scope_key, out)
    return out


def _summarize_speaker_scope_rows(rows: list[dict]) -> dict[str, int]:
    total = 0
    unknown = 0
    identified = 0
    extras = 0
    main = 0

    for row in rows:
        total += 1
        name = str(row.get("name") or "")
        is_extra = bool(row.get("is_extra"))
        total_time = float(row.get("total_speaking_time") or 0.0)
        row_is_unknown = _is_unknown_speaker_name(name)
        if row_is_unknown:
            unknown += 1
            continue
        identified += 1
        if is_extra or total_time < 60.0:
            extras += 1
        else:
            main += 1

    return {
        "total": int(total),
        "unknown": int(unknown),
        "identified": int(identified),
        "extras": int(extras),
        "main": int(main),
    }


def _invalidate_speaker_query_caches() -> None:
    with _speaker_list_cache_lock:
        _speaker_list_cache.clear()
    with _speaker_counts_cache_lock:
        _speaker_counts_cache.clear()
    with _speaker_scope_cache_lock:
        _speaker_scope_cache.clear()


def _avatar_artifacts_dir(avatar: Avatar) -> Path:
    channel_dir = AVATARS_DIR / f"channel_{int(avatar.channel_id)}"
    speaker_dir = channel_dir / f"speaker_{int(avatar.speaker_id)}"
    avatar_dir = speaker_dir / f"avatar_{int(avatar.id)}"
    avatar_dir.mkdir(parents=True, exist_ok=True)
    return avatar_dir


def _avatar_personality_dataset_dir(avatar: Avatar) -> Path:
    path = _avatar_artifacts_dir(avatar) / "personality" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _avatar_personality_dataset_paths(avatar: Avatar) -> tuple[Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "dataset_sharegpt.jsonl",
        dataset_dir / "dataset_preview.json",
        dataset_dir / "dataset_metadata.json",
    )


def _avatar_personality_review_paths(avatar: Avatar) -> tuple[Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "dataset_review.jsonl",
        dataset_dir / "dataset_states.json",
    )


def _avatar_personality_cluster_summary_path(avatar: Avatar) -> Path:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return dataset_dir / "dataset_clusters.json"


def _avatar_personality_judge_status_path(avatar: Avatar) -> Path:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return dataset_dir / "judge_status.json"


def _avatar_personality_long_form_paths(avatar: Avatar) -> tuple[Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "long_form_samples.json",
        dataset_dir / "long_form_states.json",
        dataset_dir / "long_form_config.json",
    )


def _avatar_personality_training_paths(avatar: Avatar) -> tuple[Path, Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "training_config.json",
        dataset_dir / "training_manifest.json",
        dataset_dir / "training_train.jsonl",
        dataset_dir / "training_val.jsonl",
    )


def _avatar_personality_training_runtime_paths(avatar: Avatar) -> tuple[Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "training_status.json",
        dataset_dir / "training_stop.flag",
    )


def _default_avatar_personality_training_config() -> dict[str, object]:
    return {
        "base_model_id": "Qwen/Qwen3-8B",
        "dataset_profile": "balanced",
        "training_strength": "balanced",
        "export_strategy": "gold_balanced",
        "validation_ratio": 0.10,
        "max_examples": 2500,
        "max_long_form_examples": 80,
        "include_long_form": True,
        "training_mode": "memory_optimized",
        "snapshot_interval_steps": 0,
    }


_AVATAR_TRAINING_DATASET_PROFILES: dict[str, dict[str, object]] = {
    "focused": {
        "label": "Focused",
        "summary": "Smaller, tighter dataset for a fast first pass and lower memorization risk.",
        "conversation_target": 1000,
        "long_form_target": 32,
        "pros": [
            "Fastest prep and shortest training runs",
            "Good for a first personality smoke test",
            "Lower chance of topic memorization",
        ],
        "cons": [
            "Less coverage of the speaker's range",
            "Can feel too generic if the source data is noisy",
        ],
    },
    "balanced": {
        "label": "Balanced",
        "summary": "Recommended default with enough breadth for style and reasoning without overextending the run.",
        "conversation_target": 2500,
        "long_form_target": 80,
        "pros": [
            "Strong first-pass range for most 7B-8B personality LoRAs",
            "Usually lands in a healthy one-epoch step range",
            "Balances conversational variety with manageable runtime",
        ],
        "cons": [
            "Can still miss niche references from large channels",
        ],
        "recommended": True,
    },
    "broad": {
        "label": "Broad",
        "summary": "Wider coverage for speakers with varied topics, references, and argument patterns.",
        "conversation_target": 4000,
        "long_form_target": 120,
        "pros": [
            "Better topical coverage and richer reference patterns",
            "Useful when the speaker has many recurring argument structures",
        ],
        "cons": [
            "Longer runs and more checkpoint review",
            "Needs good curation to avoid repetitive topic drift",
        ],
    },
    "exhaustive": {
        "label": "Exhaustive",
        "summary": "Largest preset. Use only when the dataset is very clean and you want maximum coverage.",
        "conversation_target": 6000,
        "long_form_target": 160,
        "pros": [
            "Captures the broadest range of topics and metaphors",
            "Most useful when the source data has already been heavily filtered",
        ],
        "cons": [
            "Heaviest runtime and review burden",
            "Higher risk of repetition or topic memorization if the dataset is uneven",
        ],
    },
    "custom": {
        "label": "Custom",
        "summary": "Manual caps for cases where you want to tune the package size directly.",
        "conversation_target": 0,
        "long_form_target": 0,
        "pros": [
            "Full control over conversation and long-form caps",
        ],
        "cons": [
            "Easier to overshoot into long, less efficient runs",
        ],
    },
}


def _avatar_training_dataset_profile_options() -> list[AvatarPersonalityTrainingDatasetProfileRead]:
    options: list[AvatarPersonalityTrainingDatasetProfileRead] = []
    for key in ["focused", "balanced", "broad", "exhaustive", "custom"]:
        payload = dict(_AVATAR_TRAINING_DATASET_PROFILES[key])
        options.append(
            AvatarPersonalityTrainingDatasetProfileRead(
                key=key,
                label=str(payload.get("label") or key.title()),
                summary=str(payload.get("summary") or ""),
                conversation_target=int(payload.get("conversation_target") or 0),
                long_form_target=int(payload.get("long_form_target") or 0),
                pros=[str(item) for item in payload.get("pros", []) if str(item).strip()],
                cons=[str(item) for item in payload.get("cons", []) if str(item).strip()],
                recommended=bool(payload.get("recommended")),
            )
        )
    return options


def _normalize_avatar_training_dataset_profile(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _AVATAR_TRAINING_DATASET_PROFILES:
        return normalized
    return "balanced"


def _infer_avatar_training_dataset_profile(max_examples: int | None, max_long_form_examples: int | None) -> str:
    conversation_cap = max(0, int(max_examples or 0))
    long_form_cap = max(0, int(max_long_form_examples or 0))
    for key, payload in _AVATAR_TRAINING_DATASET_PROFILES.items():
        if key == "custom":
            continue
        if (
            conversation_cap == int(payload.get("conversation_target") or 0)
            and long_form_cap == int(payload.get("long_form_target") or 0)
        ):
            return key
    return "custom"


def _resolve_avatar_training_dataset_targets(
    *,
    dataset_profile: str | None,
    max_examples: int | None,
    max_long_form_examples: int | None,
) -> tuple[str, int, int]:
    profile_key = _normalize_avatar_training_dataset_profile(dataset_profile)
    if profile_key != "custom":
        preset = _AVATAR_TRAINING_DATASET_PROFILES[profile_key]
        return (
            profile_key,
            int(preset.get("conversation_target") or 0),
            int(preset.get("long_form_target") or 0),
        )
    return (
        "custom",
        max(0, int(max_examples or 0)),
        max(0, int(max_long_form_examples or 0)),
    )


def _avatar_training_config_storage_payload(payload: dict[str, object]) -> dict[str, object]:
    persisted_keys = {
        "base_model_id",
        "dataset_profile",
        "training_strength",
        "export_strategy",
        "validation_ratio",
        "max_examples",
        "max_long_form_examples",
        "include_long_form",
        "training_mode",
        "snapshot_interval_steps",
    }
    return {key: payload[key] for key in persisted_keys if key in payload}


def _estimate_avatar_validation_example_count(total_examples: int, validation_ratio: float) -> int:
    total = max(0, int(total_examples or 0))
    if total <= 1:
        return 0
    estimate = int(round(total * max(0.01, min(0.2, float(validation_ratio or 0.10)))))
    if total > 20:
        estimate = max(1, estimate)
    estimate = min(total - 1, max(0, estimate))
    return estimate


def _avatar_recommended_snapshot_interval(total_steps: int) -> int:
    steps = max(0, int(total_steps or 0))
    if steps <= 0:
        return 0
    return max(10, math.ceil(steps / 10))


def _hf_repo_cache_root() -> Path:
    custom_home = os.getenv("HF_HOME")
    if custom_home:
        return Path(custom_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(model_id: str) -> Path:
    org, repo = str(model_id or "").strip().split("/", 1)
    return _hf_repo_cache_root() / f"models--{org}--{repo}"


def _hf_model_local_snapshot(model_id: str) -> Path | None:
    repo_dir = _hf_repo_cache_dir(model_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in candidates:
        has_config = (snapshot / "config.json").exists()
        has_weights = any(snapshot.glob("*.safetensors")) or (snapshot / "model.safetensors.index.json").exists() or any(snapshot.glob("pytorch_model*.bin"))
        if has_config and has_weights:
            return snapshot
    return None


def _hf_model_is_installed(model_id: str) -> tuple[bool, str | None]:
    try:
        snapshot = _hf_model_local_snapshot(model_id)
    except Exception:
        snapshot = None
    return (snapshot is not None, str(snapshot) if snapshot else None)


def _recommended_avatar_training_models() -> list[dict[str, str]]:
    return [
        {"model_id": "Qwen/Qwen3-8B", "label": "Qwen3 8B"},
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B Instruct"},
        {"model_id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B Instruct"},
        {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct"},
        {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct"},
    ]


@lru_cache(maxsize=1)
def _detect_avatar_memory_optimized_support() -> tuple[bool, str | None]:
    if importlib.util.find_spec("bitsandbytes") is None:
        return False, "bitsandbytes is not installed in the backend training environment."
    try:
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            from transformers.utils.quantization_config import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True)
        return True, None
    except Exception as exc:
        return False, f"4-bit QLoRA support is unavailable: {exc}"


def _recommend_avatar_training_model_for_hardware() -> tuple[dict[str, object], list[dict[str, str]]]:
    hardware = _detect_gpu_hardware()
    gpu_vram_gb = hardware.get("gpu_vram_gb")
    candidates = _recommended_avatar_training_models()
    recommended = "Qwen/Qwen2.5-3B-Instruct"
    rationale = "GPU VRAM could not be detected. Defaulting to a conservative training base."
    if gpu_vram_gb is not None:
        vram = float(gpu_vram_gb)
        if vram >= 28:
            recommended = "Qwen/Qwen3-8B"
            rationale = f"Detected ~{vram:.1f} GB VRAM. Qwen3-8B is the strongest practical target for the current LoRA trainer."
        elif vram >= 18:
            recommended = "Qwen/Qwen2.5-7B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 7B is the safer high-quality target on this hardware."
        elif vram >= 10:
            recommended = "Qwen/Qwen2.5-3B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 3B is the recommended fit for stable local LoRA training."
        elif vram >= 6:
            recommended = "Qwen/Qwen2.5-1.5B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 1.5B is the largest practical fit on this hardware."
        else:
            recommended = "Qwen/Qwen2.5-0.5B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. Use a very small base for local experimentation."
    return {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_vram_gb": gpu_vram_gb,
        "recommended_model_id": recommended,
        "rationale": rationale,
    }, candidates


def _infer_avatar_model_scale_b(model_id: str | None) -> float | None:
    normalized = str(model_id or "").strip()
    if not normalized:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", normalized)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _recommend_avatar_training_launch_settings(
    *,
    model_id: str | None,
    training_mode: str,
    requested_lora_rank: int | None,
    requested_max_seq_length: int | None,
    requested_per_device_batch_size: int | None,
    requested_gradient_accumulation_steps: int | None,
) -> dict[str, object]:
    hardware = _detect_gpu_hardware()
    gpu_vram_gb_raw = hardware.get("gpu_vram_gb")
    gpu_vram_gb = float(gpu_vram_gb_raw) if gpu_vram_gb_raw is not None else None
    model_scale_b = _infer_avatar_model_scale_b(model_id)
    mode = str(training_mode or "memory_optimized").strip().lower()

    # Start with a conservative 12 GB profile so unknown hardware does not overrun
    # VRAM and spill into shared/system memory.
    recommended = {
        "lora_rank": 8,
        "max_seq_length": 768,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "cuda_memory_fraction": 0.72,
        "rationale": "Using a conservative 12 GB baseline because GPU VRAM could not be detected.",
    }

    if gpu_vram_gb is not None:
        if gpu_vram_gb >= 28:
            recommended.update(
                lora_rank=16,
                max_seq_length=1536,
                gradient_accumulation_steps=8,
                cuda_memory_fraction=0.82,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a high-end local LoRA profile with reserved CUDA headroom.",
            )
        elif gpu_vram_gb >= 18:
            recommended.update(
                lora_rank=8,
                max_seq_length=1024,
                gradient_accumulation_steps=12,
                cuda_memory_fraction=0.78,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a balanced profile that keeps headroom for activations and optimizer state.",
            )
        elif gpu_vram_gb >= 12:
            recommended.update(
                lora_rank=8,
                max_seq_length=768,
                gradient_accumulation_steps=16,
                cuda_memory_fraction=0.72,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a 12 GB-safe profile to avoid shared-memory spillover.",
            )
        elif gpu_vram_gb >= 8:
            recommended.update(
                lora_rank=4,
                max_seq_length=512,
                gradient_accumulation_steps=24,
                cuda_memory_fraction=0.68,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using an aggressive low-VRAM profile.",
            )
        else:
            recommended.update(
                lora_rank=4,
                max_seq_length=384,
                gradient_accumulation_steps=32,
                cuda_memory_fraction=0.62,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a minimal profile for experimentation only.",
            )

    if mode == "standard":
        recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
        recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 24 else 768)
        recommended["cuda_memory_fraction"] = min(float(recommended["cuda_memory_fraction"]), 0.70 if (gpu_vram_gb or 0) >= 24 else 0.62)
        recommended["gradient_accumulation_steps"] = max(int(recommended["gradient_accumulation_steps"]), 16)
        recommended["rationale"] = (
            f"{recommended['rationale']} Standard mode keeps extra VRAM headroom because full-precision optimizer state is larger."
        )

    if model_scale_b is not None:
        if model_scale_b >= 14:
            recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 512 if (gpu_vram_gb or 0) < 40 else 768)
            recommended["gradient_accumulation_steps"] = max(int(recommended["gradient_accumulation_steps"]), 16)
        elif model_scale_b >= 8:
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 32 else 768)
            if (gpu_vram_gb or 0) < 24:
                recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
        elif model_scale_b >= 7:
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 24 else 768)
        elif model_scale_b <= 3:
            if (gpu_vram_gb or 12) >= 20:
                recommended["lora_rank"] = max(int(recommended["lora_rank"]), 16)
                recommended["max_seq_length"] = max(int(recommended["max_seq_length"]), 1536)
                recommended["gradient_accumulation_steps"] = min(int(recommended["gradient_accumulation_steps"]), 8)
        elif model_scale_b <= 1.5:
            if (gpu_vram_gb or 12) >= 12:
                recommended["lora_rank"] = max(int(recommended["lora_rank"]), 16)
                recommended["max_seq_length"] = max(int(recommended["max_seq_length"]), 1024)
                recommended["gradient_accumulation_steps"] = min(int(recommended["gradient_accumulation_steps"]), 12)

    effective_lora_rank = min(
        max(4, int(requested_lora_rank or int(recommended["lora_rank"]))),
        max(4, int(recommended["lora_rank"])),
    )
    effective_max_seq_length = min(
        max(256, int(requested_max_seq_length or int(recommended["max_seq_length"]))),
        max(256, int(recommended["max_seq_length"])),
    )
    effective_per_device_batch_size = min(
        max(1, int(requested_per_device_batch_size or int(recommended["per_device_batch_size"]))),
        max(1, int(recommended["per_device_batch_size"])),
    )
    effective_gradient_accumulation_steps = max(
        max(1, int(requested_gradient_accumulation_steps or int(recommended["gradient_accumulation_steps"]))),
        max(1, int(recommended["gradient_accumulation_steps"])),
    )

    return {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_vram_gb": gpu_vram_gb,
        "model_scale_b": model_scale_b,
        "lora_rank": effective_lora_rank,
        "max_seq_length": effective_max_seq_length,
        "per_device_batch_size": effective_per_device_batch_size,
        "gradient_accumulation_steps": effective_gradient_accumulation_steps,
        "cuda_memory_fraction": float(recommended["cuda_memory_fraction"]),
        "rationale": str(recommended["rationale"]),
    }


def _estimate_avatar_available_conversation_examples(
    dataset: AvatarPersonalityDatasetRead,
    config: AvatarPersonalityTrainingConfigRead,
) -> int:
    approved_count = max(0, int(dataset.gold_example_count or 0) + int(dataset.silver_example_count or 0))
    gold_count = int(dataset.gold_example_count or 0)
    silver_count = int(dataset.silver_example_count or 0)
    strategy = str(config.export_strategy or "gold_balanced")
    if strategy == "gold_only":
        return max(0, gold_count)
    if strategy == "gold_plus_top_silver":
        silver_budget = min(max(0, silver_count), max(250, gold_count // 2))
        return max(0, gold_count + silver_budget)
    return max(0, approved_count)


def _build_avatar_personality_training_plan(
    *,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    config: AvatarPersonalityTrainingConfigRead,
    available_conversation_examples: int | None = None,
    available_long_form_examples: int | None = None,
    selected_conversation_examples: int | None = None,
    selected_long_form_examples: int | None = None,
    train_examples: int | None = None,
    validation_examples: int | None = None,
    epochs: int = 1,
) -> AvatarPersonalityTrainingPlanRead:
    dataset = _load_avatar_personality_dataset(avatar, personality)
    long_form_config = _load_avatar_personality_long_form_config(avatar)
    profile_key, conversation_target, long_form_target = _resolve_avatar_training_dataset_targets(
        dataset_profile=config.dataset_profile,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
    )
    profile_meta = _AVATAR_TRAINING_DATASET_PROFILES.get(profile_key, _AVATAR_TRAINING_DATASET_PROFILES["balanced"])
    available_conversation = max(
        0,
        int(
            available_conversation_examples
            if available_conversation_examples is not None
            else _estimate_avatar_available_conversation_examples(dataset, config)
        ),
    )
    available_long_form = max(
        0,
        int(
            available_long_form_examples
            if available_long_form_examples is not None
            else min(int(long_form_config.selected_count or 0), int(long_form_config.take_count or 0))
        ),
    )

    effective_conversation = max(
        0,
        int(
            selected_conversation_examples
            if selected_conversation_examples is not None
            else min(available_conversation, conversation_target if conversation_target > 0 else available_conversation)
        ),
    )
    effective_long_form = 0
    if bool(config.include_long_form):
        effective_long_form = max(
            0,
            int(
                selected_long_form_examples
                if selected_long_form_examples is not None
                else min(available_long_form, long_form_target if long_form_target > 0 else available_long_form)
            ),
        )

    estimated_total_examples = max(0, effective_conversation + effective_long_form)
    resolved_validation_examples = (
        max(0, int(validation_examples or 0))
        if validation_examples is not None
        else _estimate_avatar_validation_example_count(estimated_total_examples, config.validation_ratio)
    )
    resolved_train_examples = (
        max(0, int(train_examples or 0))
        if train_examples is not None
        else max(0, estimated_total_examples - resolved_validation_examples)
    )

    launch_settings = _recommend_avatar_training_launch_settings(
        model_id=str(config.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B"),
        training_mode=str(config.training_mode or "memory_optimized"),
        requested_lora_rank=None,
        requested_max_seq_length=None,
        requested_per_device_batch_size=None,
        requested_gradient_accumulation_steps=None,
    )
    effective_batch_size = max(
        1,
        int(launch_settings.get("per_device_batch_size") or 1)
        * int(launch_settings.get("gradient_accumulation_steps") or 1),
    )
    estimated_steps_per_epoch = math.ceil(resolved_train_examples / effective_batch_size) if resolved_train_examples > 0 else 0
    estimated_total_steps = estimated_steps_per_epoch * max(1, int(epochs or 1))

    if estimated_total_steps < 100:
        step_band = "light"
        headline = "Small, fast package with a lighter style imprint."
        recommendation = "Good for a smoke test, but it may underfit nuance, references, and argument structure."
    elif estimated_total_steps <= 500:
        step_band = "ideal"
        headline = "Healthy first-pass training range for personality LoRA."
        recommendation = "This is the best default zone for one-epoch training and snapshot comparison."
    elif estimated_total_steps <= 800:
        step_band = "heavy"
        headline = "Broader package with longer runs and more review overhead."
        recommendation = "Useful for diverse speakers, but watch for repeated stock phrases and over-anchoring to popular topics."
    else:
        step_band = "aggressive"
        headline = "Large package that pushes beyond the usual first-pass sweet spot."
        recommendation = "Only use this if the dataset is very clean and diverse. Prefer snapshot comparison and stop early if responses get repetitive."

    return AvatarPersonalityTrainingPlanRead(
        dataset_profile=profile_key,
        dataset_profile_label=str(profile_meta.get("label") or profile_key.title()),
        conversation_target=conversation_target,
        long_form_target=long_form_target if bool(config.include_long_form) else 0,
        available_conversation_examples=available_conversation,
        available_long_form_examples=available_long_form if bool(config.include_long_form) else 0,
        estimated_conversation_examples=effective_conversation,
        estimated_long_form_examples=effective_long_form,
        estimated_total_examples=estimated_total_examples,
        estimated_train_examples=resolved_train_examples,
        estimated_validation_examples=resolved_validation_examples,
        estimated_effective_batch_size=effective_batch_size,
        estimated_steps_per_epoch=estimated_steps_per_epoch,
        estimated_total_steps=estimated_total_steps,
        step_band=step_band,
        headline=headline,
        recommendation=recommendation,
        snapshot_interval_suggestion=_avatar_recommended_snapshot_interval(estimated_total_steps),
        pros=[str(item) for item in profile_meta.get("pros", []) if str(item).strip()],
        cons=[str(item) for item in profile_meta.get("cons", []) if str(item).strip()],
    )


def _get_avatar_hf_model_download_state(model_id: str) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    if not normalized:
        return {"status": "idle", "running": False, "message": None}
    with _avatar_hf_model_downloads_lock:
        return dict(_avatar_hf_model_downloads.get(normalized) or {})


def _set_avatar_hf_model_download_state(model_id: str, patch: dict[str, object]) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    with _avatar_hf_model_downloads_lock:
        current = dict(_avatar_hf_model_downloads.get(normalized) or {"status": "idle", "running": False, "message": None})
        current.update(patch)
        _avatar_hf_model_downloads[normalized] = current
        return dict(current)


def _start_avatar_hf_model_download(model_id: str) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    if "/" not in normalized:
        raise HTTPException(status_code=400, detail="Model id must look like org/repo")
    installed, local_path = _hf_model_is_installed(normalized)
    if installed:
        return _set_avatar_hf_model_download_state(
            normalized,
            {"status": "completed", "running": False, "message": "Model already installed locally", "local_path": local_path},
        )
    current = _get_avatar_hf_model_download_state(normalized)
    if current.get("running"):
        return current

    _set_avatar_hf_model_download_state(
        normalized,
        {"status": "running", "running": True, "message": "Starting Hugging Face model download", "local_path": None},
    )

    def _runner() -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=normalized,
                resume_download=True,
                local_files_only=False,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.tiktoken",
                    "tokenizer*",
                    "merges.txt",
                    "vocab*",
                    "*.txt",
                ],
                ignore_patterns=[
                    "*.gguf",
                    "*.onnx",
                    "*.h5",
                    "*.ot",
                    "*.msgpack",
                ],
                token=(os.getenv("HF_TOKEN") or None),
            )
            _set_avatar_hf_model_download_state(
                normalized,
                {"status": "completed", "running": False, "message": "Model download completed", "local_path": str(snapshot_path)},
            )
        except Exception as exc:
            _set_avatar_hf_model_download_state(
                normalized,
                {"status": "failed", "running": False, "message": f"Download failed: {exc}"},
            )

    threading.Thread(target=_runner, daemon=True, name=f"hf-model-download-{normalized.replace('/', '--')}").start()
    return _get_avatar_hf_model_download_state(normalized)


def _read_avatar_base_model_support(selected_model_id: str) -> AvatarPersonalityBaseModelSupportRead:
    selected = str(selected_model_id or "").strip() or "Qwen/Qwen3-8B"
    recommendation, candidates = _recommend_avatar_training_model_for_hardware()
    installed, local_path = _hf_model_is_installed(selected)
    memory_optimized_available, memory_optimized_reason = _detect_avatar_memory_optimized_support()
    download_state = _get_avatar_hf_model_download_state(selected)
    items: list[AvatarPersonalityBaseModelCandidateRead] = []
    for candidate in candidates:
        candidate_installed, _candidate_path = _hf_model_is_installed(candidate["model_id"])
        items.append(
            AvatarPersonalityBaseModelCandidateRead(
                model_id=candidate["model_id"],
                label=candidate["label"],
                recommended=(candidate["model_id"] == recommendation["recommended_model_id"]),
                installed=candidate_installed,
            )
        )
    return AvatarPersonalityBaseModelSupportRead(
        selected_model_id=selected,
        recommended_model_id=str(recommendation["recommended_model_id"]),
        installed=installed,
        local_path=local_path,
        memory_optimized_available=memory_optimized_available,
        memory_optimized_reason=memory_optimized_reason,
        downloading=bool(download_state.get("running")),
        download_status=str(download_state.get("status") or "idle"),
        download_message=(str(download_state.get("message")) if download_state.get("message") else None),
        gpu_name=(str(recommendation.get("gpu_name")) if recommendation.get("gpu_name") else None),
        gpu_vram_gb=(float(recommendation["gpu_vram_gb"]) if recommendation.get("gpu_vram_gb") is not None else None),
        rationale=(str(recommendation.get("rationale")) if recommendation.get("rationale") else None),
        candidates=items,
    )


def _default_avatar_personality_judge_status(avatar_id: int) -> dict[str, object]:
    now = datetime.now().isoformat()
    return {
        "avatar_id": int(avatar_id),
        "status": "idle",
        "active": False,
        "stop_requested": False,
        "model": None,
        "target_filter": "needs_review",
        "overwrite_existing": False,
        "max_examples": 40,
        "total_candidates": 0,
        "processed_count": 0,
        "judged_count": 0,
        "promoted_count": 0,
        "rejected_count": 0,
        "current_example_id": None,
        "current_video_title": None,
        "current_stage": None,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "error": None,
        "recent_results": [],
    }


def _load_avatar_personality_judge_status(avatar: Avatar) -> AvatarPersonalityJudgeStatusRead:
    status_path = _avatar_personality_judge_status_path(avatar)
    payload = _default_avatar_personality_judge_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityJudgeStatusRead(**payload)


def _write_avatar_personality_judge_status(avatar: Avatar, payload: dict[str, object]) -> AvatarPersonalityJudgeStatusRead:
    status_path = _avatar_personality_judge_status_path(avatar)
    existing = _default_avatar_personality_judge_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing.update(raw)
        except Exception:
            pass
    existing.update(payload)
    existing["avatar_id"] = int(avatar.id)
    existing["updated_at"] = datetime.now().isoformat()
    normalized = AvatarPersonalityJudgeStatusRead(**existing)
    status_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _default_avatar_personality_prompt(name: str) -> str:
    clean_name = _clean_avatar_dataset_text(name) or "the podcast speaker"
    return f"You are {clean_name}. Respond in their conversational style based on the approved transcript dataset."


def _clean_avatar_dataset_text(value: str | None) -> str:
    text_value = html.unescape(str(value or ""))
    text_value = re.sub(r"\s+", " ", text_value).strip()
    text_value = text_value.strip("\"' ")
    return text_value


def _avatar_word_count(text: str | None) -> int:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return 0
    return len(cleaned.split())


def _avatar_long_form_sample_id(video_id: int, start_time: float, end_time: float, segment_ids: list[int]) -> str:
    raw = f"{int(video_id)}|{round(float(start_time or 0.0), 3)}|{round(float(end_time or 0.0), 3)}|{','.join(str(int(segment_id)) for segment_id in segment_ids)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _load_avatar_personality_long_form_states(avatar: Avatar) -> dict[str, str]:
    _, states_path, _ = _avatar_personality_long_form_paths(avatar)
    if not states_path.exists():
        return {}
    try:
        raw = json.loads(states_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        state = str(value or "").strip().lower()
        if state in {"included", "rejected"}:
            out[str(key)] = state
    return out


def _write_avatar_personality_long_form_states(avatar: Avatar, state_map: dict[str, str]) -> Path:
    _, states_path, _ = _avatar_personality_long_form_paths(avatar)
    serializable = {str(key): str(value) for key, value in sorted(state_map.items()) if value in {"included", "rejected"}}
    states_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return states_path


def _load_avatar_personality_long_form_config(avatar: Avatar) -> AvatarPersonalityLongFormConfigRead:
    _, _, config_path = _avatar_personality_long_form_paths(avatar)
    payload: dict[str, object] = {"take_count": 150, "included_count": 0, "rejected_count": 0, "selected_count": 0}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityLongFormConfigRead(**payload)


def _write_avatar_personality_long_form_config(
    avatar: Avatar,
    *,
    take_count: int,
    included_count: int | None = None,
    rejected_count: int | None = None,
    selected_count: int | None = None,
) -> AvatarPersonalityLongFormConfigRead:
    _, _, config_path = _avatar_personality_long_form_paths(avatar)
    existing = _load_avatar_personality_long_form_config(avatar).model_dump()
    existing["take_count"] = max(0, int(take_count or 0))
    if included_count is not None:
        existing["included_count"] = max(0, int(included_count))
    if rejected_count is not None:
        existing["rejected_count"] = max(0, int(rejected_count))
    if selected_count is not None:
        existing["selected_count"] = max(0, int(selected_count))
    normalized = AvatarPersonalityLongFormConfigRead(**existing)
    config_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _load_avatar_personality_training_config(avatar: Avatar) -> AvatarPersonalityTrainingConfigRead:
    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    payload = _default_avatar_personality_training_config()
    raw: dict[str, object] = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    if not isinstance(raw, dict):
        raw = {}
    if "dataset_profile" in raw:
        profile_key, resolved_max_examples, resolved_max_long_form_examples = _resolve_avatar_training_dataset_targets(
            dataset_profile=raw.get("dataset_profile"),
            max_examples=payload.get("max_examples"),
            max_long_form_examples=payload.get("max_long_form_examples"),
        )
        payload["dataset_profile"] = profile_key
        if profile_key != "custom":
            payload["max_examples"] = resolved_max_examples
            payload["max_long_form_examples"] = resolved_max_long_form_examples
    else:
        payload["dataset_profile"] = _infer_avatar_training_dataset_profile(
            payload.get("max_examples"),
            payload.get("max_long_form_examples"),
        )
        payload["max_long_form_examples"] = max(0, int(payload.get("max_long_form_examples") or _default_avatar_personality_training_config()["max_long_form_examples"]))
    return AvatarPersonalityTrainingConfigRead(**payload)


def _read_avatar_personality_training_config(
    session: Session,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityTrainingConfigRead:
    config = _load_avatar_personality_training_config(avatar)
    payload = config.model_dump()
    payload["dataset_profiles"] = [option.model_dump() for option in _avatar_training_dataset_profile_options()]
    payload["training_plan"] = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingConfigRead(**payload)


def _write_avatar_personality_training_config(
    avatar: Avatar,
    *,
    base_model_id: str | None = None,
    dataset_profile: str | None = None,
    training_strength: str | None = None,
    export_strategy: str | None = None,
    validation_ratio: float | None = None,
    max_examples: int | None = None,
    max_long_form_examples: int | None = None,
    include_long_form: bool | None = None,
    training_mode: str | None = None,
    snapshot_interval_steps: int | None = None,
) -> AvatarPersonalityTrainingConfigRead:
    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    existing = _avatar_training_config_storage_payload(_load_avatar_personality_training_config(avatar).model_dump())
    if base_model_id is not None:
        existing["base_model_id"] = str(base_model_id).strip() or existing.get("base_model_id") or "Qwen/Qwen3-8B"
    if dataset_profile is not None:
        existing["dataset_profile"] = _normalize_avatar_training_dataset_profile(dataset_profile)
    if training_strength is not None:
        normalized_strength = str(training_strength or "").strip().lower()
        if normalized_strength not in {"conservative", "balanced", "strong"}:
            normalized_strength = "balanced"
        existing["training_strength"] = normalized_strength
    if export_strategy is not None:
        existing["export_strategy"] = str(export_strategy)
    if validation_ratio is not None:
        existing["validation_ratio"] = max(0.01, min(0.2, float(validation_ratio)))
    if max_examples is not None:
        existing["max_examples"] = max(0, int(max_examples))
        if dataset_profile is None and str(existing.get("dataset_profile") or "").strip().lower() != "custom":
            existing["dataset_profile"] = "custom"
    if max_long_form_examples is not None:
        existing["max_long_form_examples"] = max(0, int(max_long_form_examples))
        if dataset_profile is None and str(existing.get("dataset_profile") or "").strip().lower() != "custom":
            existing["dataset_profile"] = "custom"
    if include_long_form is not None:
        existing["include_long_form"] = bool(include_long_form)
    if training_mode is not None:
        existing["training_mode"] = str(training_mode or "memory_optimized").strip() or "memory_optimized"
    if snapshot_interval_steps is not None:
        existing["snapshot_interval_steps"] = max(0, int(snapshot_interval_steps))
    profile_key, resolved_max_examples, resolved_max_long_form_examples = _resolve_avatar_training_dataset_targets(
        dataset_profile=str(existing.get("dataset_profile") or "balanced"),
        max_examples=existing.get("max_examples"),
        max_long_form_examples=existing.get("max_long_form_examples"),
    )
    existing["dataset_profile"] = profile_key
    existing["max_examples"] = resolved_max_examples
    existing["max_long_form_examples"] = resolved_max_long_form_examples
    normalized = AvatarPersonalityTrainingConfigRead(**existing)
    config_path.write_text(
        json.dumps(_avatar_training_config_storage_payload(normalized.model_dump()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def _read_avatar_personality_training_package(avatar: Avatar) -> AvatarPersonalityTrainingPackageRead:
    config_path, manifest_path, train_path, val_path = _avatar_personality_training_paths(avatar)
    if not manifest_path.exists():
        config = _load_avatar_personality_training_config(avatar)
        return AvatarPersonalityTrainingPackageRead(
            avatar_id=int(avatar.id),
            status="not_prepared",
            dataset_profile=config.dataset_profile,
            training_strength=config.training_strength,
            export_strategy=config.export_strategy,
            validation_ratio=config.validation_ratio,
            max_examples=config.max_examples,
            max_long_form_examples=config.max_long_form_examples,
            include_long_form=config.include_long_form,
            config_path=str(config_path) if config_path.exists() else None,
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("avatar_id", int(avatar.id))
    raw.setdefault("status", "ready")
    raw.setdefault("dataset_profile", "balanced")
    raw.setdefault("training_strength", "balanced")
    raw.setdefault("max_long_form_examples", 0)
    if not raw.get("config_path") and config_path.exists():
        raw["config_path"] = str(config_path)
    if not raw.get("train_dataset_path") and train_path.exists():
        raw["train_dataset_path"] = str(train_path)
    if not raw.get("validation_dataset_path") and val_path.exists():
        raw["validation_dataset_path"] = str(val_path)
    raw["manifest_path"] = str(manifest_path)
    return AvatarPersonalityTrainingPackageRead(**raw)


def _default_avatar_personality_training_status(avatar_id: int) -> dict[str, object]:
    now = datetime.now().isoformat()
    return {
        "avatar_id": int(avatar_id),
        "status": "idle",
        "active": False,
        "stop_requested": False,
        "process_id": None,
        "base_model_id": None,
        "training_mode": "memory_optimized",
        "adapter_path": None,
        "output_dir": None,
        "current_stage": None,
        "epoch": 0.0,
        "step": 0,
        "max_steps": 0,
        "snapshot_interval_steps": 0,
        "train_examples": 0,
        "validation_examples": 0,
        "latest_loss": None,
        "message": None,
        "snapshots": [],
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "error": None,
    }


def _load_avatar_personality_training_status(avatar: Avatar) -> AvatarPersonalityTrainingStatusRead:
    status_path, _ = _avatar_personality_training_runtime_paths(avatar)
    payload = _default_avatar_personality_training_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityTrainingStatusRead(**payload)


def _write_avatar_personality_training_status(avatar: Avatar, patch: dict[str, object]) -> AvatarPersonalityTrainingStatusRead:
    status_path, _ = _avatar_personality_training_runtime_paths(avatar)
    payload = _default_avatar_personality_training_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    payload.update(patch)
    normalized = AvatarPersonalityTrainingStatusRead(**payload)
    status_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _avatar_training_output_dir(avatar: Avatar) -> Path:
    return _avatar_artifacts_dir(avatar) / "personality" / "training_runs" / "latest"


def _normalize_avatar_training_snapshots(
    snapshots: list[AvatarPersonalitySnapshotRead] | list[dict[str, object]] | None,
    *,
    selected_adapter_path: str | None = None,
) -> list[AvatarPersonalitySnapshotRead]:
    normalized: list[AvatarPersonalitySnapshotRead] = []
    selected = str(selected_adapter_path or "").strip()
    seen_paths: set[str] = set()
    for raw in snapshots or []:
        try:
            item = raw if isinstance(raw, AvatarPersonalitySnapshotRead) else AvatarPersonalitySnapshotRead(**raw)
        except Exception:
            continue
        adapter_path = str(item.adapter_path or "").strip()
        if not adapter_path or adapter_path in seen_paths:
            continue
        seen_paths.add(adapter_path)
        normalized.append(
            item.model_copy(
                update={
                    "selected": bool(selected and adapter_path == selected) or bool(item.selected and not selected),
                }
            )
        )
    normalized.sort(key=lambda item: ((item.created_at.isoformat() if item.created_at else ""), item.step, item.epoch))
    return normalized


def _avatar_resolve_training_snapshots(
    avatar: Avatar,
    *,
    selected_adapter_path: str | None = None,
) -> list[AvatarPersonalitySnapshotRead]:
    status = _load_avatar_personality_training_status(avatar)
    snapshots = _normalize_avatar_training_snapshots(
        list(status.snapshots or []),
        selected_adapter_path=selected_adapter_path or status.adapter_path,
    )
    if snapshots:
        return snapshots
    adapter_path = str(selected_adapter_path or status.adapter_path or "").strip()
    if adapter_path and Path(adapter_path).exists():
        return [
            AvatarPersonalitySnapshotRead(
                label="Final Adapter",
                kind="final",
                adapter_path=adapter_path,
                selected=True,
            )
        ]
    return []


def _avatar_update_training_snapshots(
    avatar: Avatar,
    snapshots: list[AvatarPersonalitySnapshotRead] | list[dict[str, object]],
    *,
    selected_adapter_path: str | None = None,
) -> AvatarPersonalityTrainingStatusRead:
    normalized = _normalize_avatar_training_snapshots(snapshots, selected_adapter_path=selected_adapter_path)
    return _write_avatar_personality_training_status(
        avatar,
        {
            "snapshots": [item.model_dump(mode="json") for item in normalized],
            "adapter_path": str(selected_adapter_path or "").strip() or None,
        },
    )


def _avatar_find_training_snapshot(
    avatar: Avatar,
    adapter_path: str,
) -> AvatarPersonalitySnapshotRead | None:
    selected = str(adapter_path or "").strip()
    if not selected:
        return None
    for snapshot in _avatar_resolve_training_snapshots(avatar, selected_adapter_path=selected):
        if str(snapshot.adapter_path).strip() == selected:
            return snapshot
    return None


def _avatar_training_process_is_alive(pid: int | None) -> bool:
    try:
        process_id = int(pid or 0)
    except Exception:
        return False


def _avatar_training_gpu_memory_by_pid_gb() -> dict[int, float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return {}
        out: dict[int, float] = {}
        for line in (result.stdout or "").splitlines():
            raw = str(line or "").strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
            except Exception:
                continue
            out[pid] = round(used_mb / 1024.0, 2)
        return out
    except Exception:
        return {}


def _collect_avatar_training_process_memory() -> dict[str, object]:
    gpu_by_pid = _avatar_training_gpu_memory_by_pid_gb()
    total_rss_gb = 0.0
    total_vram_gb = 0.0
    active_count = 0
    seen_pids: set[int] = set()

    for status_path in AVATARS_DIR.rglob("training_status.json"):
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        if not bool(raw.get("active")):
            continue
        try:
            pid = int(raw.get("process_id") or 0)
        except Exception:
            pid = 0
        if pid <= 0 or pid in seen_pids or not _avatar_training_process_is_alive(pid):
            continue
        seen_pids.add(pid)
        try:
            process = psutil.Process(pid)
            rss_gb = round(float(process.memory_info().rss or 0) / (1024 ** 3), 2)
        except Exception:
            rss_gb = 0.0
        total_rss_gb += max(0.0, rss_gb)
        total_vram_gb += max(0.0, float(gpu_by_pid.get(pid) or 0.0))
        active_count += 1

    return {
        "active_count": active_count,
        "ram_gb": round(total_rss_gb, 2),
        "vram_gb": round(total_vram_gb, 2),
        "loaded": active_count > 0,
    }
    if process_id <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output = str(result.stdout or "").strip()
            return bool(output) and "No tasks are running" not in output
        except Exception:
            return False
    try:
        os.kill(process_id, 0)
        return True
    except Exception:
        return False


def _reconcile_avatar_personality_training_runtime(
    avatar: Avatar,
    *,
    persist: bool = True,
) -> AvatarPersonalityTrainingStatusRead:
    status = _load_avatar_personality_training_status(avatar)
    normalized_status = str(status.status or "").strip().lower()
    if normalized_status in {"idle", "completed", "stopped", "failed"} or not status.active:
        return status

    process_alive = False
    with _avatar_training_runs_lock:
        tracked = _avatar_training_processes.get(int(avatar.id))
        if tracked is not None:
            if tracked.poll() is None:
                process_alive = True
            else:
                _avatar_training_processes.pop(int(avatar.id), None)
    if not process_alive:
        process_alive = _avatar_training_process_is_alive(status.process_id)
    if process_alive:
        return status

    adapter_exists = bool(status.adapter_path and Path(str(status.adapter_path)).exists())
    if normalized_status == "stopping" or bool(status.stop_requested):
        final_status = "stopped"
        final_message = "Recovered stale training state after the trainer process exited."
        final_error = None
    elif adapter_exists:
        final_status = "completed"
        final_message = "Recovered completed training state after the trainer process exited."
        final_error = None
    else:
        final_status = "failed"
        final_message = "Recovered stale training state after the trainer process disappeared."
        final_error = status.error or "Trainer process is no longer running."

    _, stop_path = _avatar_personality_training_runtime_paths(avatar)
    try:
        stop_path.unlink(missing_ok=True)
    except Exception:
        pass

    patch = {
        "status": final_status,
        "active": False,
        "stop_requested": False,
        "process_id": None,
        "current_stage": final_status,
        "finished_at": status.finished_at or datetime.now(),
        "updated_at": datetime.now(),
        "message": final_message,
        "error": final_error,
    }
    if persist:
        return _write_avatar_personality_training_status(avatar, patch)
    payload = status.model_dump()
    payload.update(patch)
    return AvatarPersonalityTrainingStatusRead(**payload)


def _clear_avatar_personality_training_stop_flag(avatar: Avatar) -> None:
    _, stop_path = _avatar_personality_training_runtime_paths(avatar)
    try:
        stop_path.unlink(missing_ok=True)
    except Exception:
        pass


def _sync_avatar_personality_training_completion(avatar_id: int) -> None:
    with Session(engine) as session:
        avatar = session.get(Avatar, avatar_id)
        if not avatar:
            return
        speaker = session.get(Speaker, avatar.speaker_id)
        if not speaker:
            return
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        status = _load_avatar_personality_training_status(avatar)
        normalized_status = str(status.status or "").strip().lower()
        if normalized_status == "completed" and status.adapter_path:
            personality.lora_adapter_path = status.adapter_path
            personality.status = "trained"
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()
        elif normalized_status in {"failed", "stopped"}:
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()


def _avatar_release_cached_chat_model(avatar_id: int) -> None:
    with _avatar_chat_model_lock:
        cached = _avatar_chat_models.pop(int(avatar_id), None)
    if not cached:
        return
    try:
        _cache_key, model, tokenizer = cached
        del model
        del tokenizer
    except Exception:
        pass
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _import_avatar_transformers_chat_classes():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception:
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer


def _avatar_load_chat_model(avatar_id: int, base_model_id: str, adapter_path: str, *, training_mode: str = "memory_optimized"):
    cache_key = f"{str(adapter_path).strip()}|{str(training_mode or 'memory_optimized').strip().lower()}"
    with _avatar_chat_model_lock:
        cached = _avatar_chat_models.get(int(avatar_id))
        if cached and cached[0] == cache_key:
            return cached[1], cached[2]
    _avatar_release_cached_chat_model(int(avatar_id))
    import torch
    from peft import PeftModel
    AutoModelForCausalLM, AutoTokenizer = _import_avatar_transformers_chat_classes()

    if torch.cuda.is_available():
        try:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model_kwargs: dict[str, object] = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        normalized_mode = str(training_mode or "memory_optimized").strip().lower()
        if normalized_mode == "memory_optimized" and _detect_avatar_memory_optimized_support()[0]:
            try:
                try:
                    from transformers import BitsAndBytesConfig
                except Exception:
                    from transformers.utils.quantization_config import BitsAndBytesConfig
                model_kwargs["device_map"] = {"": 0}
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            except Exception:
                model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = {"": 0}
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    with _avatar_chat_model_lock:
        _avatar_chat_models[int(avatar_id)] = (cache_key, model, tokenizer)
    return model, tokenizer


def _avatar_generate_personality_reply(
    *,
    avatar_id: int,
    base_model_id: str,
    adapter_path: str,
    training_mode: str,
    system_prompt: str,
    history: list[dict[str, str]] | None,
    message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model, tokenizer = _avatar_load_chat_model(
        int(avatar_id),
        str(base_model_id or "").strip(),
        str(adapter_path or "").strip(),
        training_mode=str(training_mode or "memory_optimized"),
    )
    import torch

    messages = [{"role": "system", "content": str(system_prompt or "").strip()}]
    for turn in history or []:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": str(message or "").strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = getattr(model, "device", None)
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = None
    if target_device is not None and str(target_device) != "meta":
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(32, min(int(max_new_tokens or 220), 512)),
            do_sample=True,
            temperature=max(0.1, min(float(temperature or 0.8), 1.5)),
            top_p=max(0.1, min(float(top_p or 0.9), 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    reply = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
    if not reply:
        raise ValueError("Generated an empty reply")
    return reply


def _avatar_is_short_backchannel(turn: dict[str, object]) -> bool:
    text = _clean_avatar_dataset_text(str(turn.get("text") or ""))
    if not text:
        return True
    if _avatar_word_count(text) > 6:
        return False
    duration = max(0.0, float(turn.get("end_time") or 0.0) - float(turn.get("start_time") or 0.0))
    if duration > 3.0:
        return False
    lowered = text.lower().strip(" .,!?:;-'\"")
    short_markers = {
        "yeah",
        "yes",
        "yep",
        "yup",
        "right",
        "ok",
        "okay",
        "mm",
        "mmm",
        "mhm",
        "uh huh",
        "huh",
        "no",
        "wow",
        "damn",
        "sure",
        "true",
        "fair",
        "exactly",
        "i know",
        "gotcha",
    }
    return lowered in short_markers or _avatar_word_count(text) <= 3


def _build_avatar_personality_long_form_samples(
    session: Session,
    avatar: Avatar,
) -> list[dict[str, object]]:
    query = (
        select(TranscriptSegment, Video.title)
        .join(Video, Video.id == TranscriptSegment.video_id)
        .where(TranscriptSegment.speaker_id == avatar.speaker_id)
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_time)
    )
    rows = session.exec(query).all()
    samples: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    max_gap_seconds = 2.5

    def flush_current():
        nonlocal current
        if not current:
            return
        text = _clean_avatar_dataset_text(" ".join(str(part) for part in current.get("parts", []) if str(part).strip()))
        duration_seconds = max(0.0, float(current.get("end_time") or 0.0) - float(current.get("start_time") or 0.0))
        segment_ids = [int(segment_id) for segment_id in current.get("segment_ids", [])]
        word_count = _avatar_word_count(text)
        if duration_seconds >= 20.0 and word_count >= 60:
            sample_id = _avatar_long_form_sample_id(int(current["video_id"]), float(current["start_time"]), float(current["end_time"]), segment_ids)
            samples.append(
                {
                    "sample_id": sample_id,
                    "video_id": int(current["video_id"]),
                    "video_title": str(current.get("video_title") or ""),
                    "start_time": float(current["start_time"]),
                    "end_time": float(current["end_time"]),
                    "duration_seconds": duration_seconds,
                    "word_count": word_count,
                    "segment_count": len(segment_ids),
                    "text": text,
                }
            )
        current = None

    for segment, video_title in rows:
        text = _clean_avatar_dataset_text(getattr(segment, "text", None))
        if not text:
            continue
        video_id = int(getattr(segment, "video_id", 0) or 0)
        start_time = float(getattr(segment, "start_time", 0.0) or 0.0)
        end_time = float(getattr(segment, "end_time", start_time) or start_time)
        segment_id = int(getattr(segment, "id", 0) or 0)
        if (
            current is None
            or int(current["video_id"]) != video_id
            or (start_time - float(current["end_time"])) > max_gap_seconds
        ):
            flush_current()
            current = {
                "video_id": video_id,
                "video_title": str(video_title or ""),
                "start_time": start_time,
                "end_time": end_time,
                "segment_ids": [segment_id],
                "parts": [text],
            }
            continue

        current["end_time"] = end_time
        cast_segment_ids = current["segment_ids"]
        cast_parts = current["parts"]
        if isinstance(cast_segment_ids, list):
            cast_segment_ids.append(segment_id)
        if isinstance(cast_parts, list):
            cast_parts.append(text)

    flush_current()
    for sample in samples:
        wc = max(1, int(sample.get("word_count") or 1))
        text = str(sample.get("text") or "")
        style_density = _avatar_style_signal_count(text) / wc
        substance_density = _avatar_substance_signal_count(text) / wc
        sample["style_density"] = round(style_density, 5)
        sample["substance_density"] = round(substance_density, 5)
    samples.sort(key=lambda row: (
        -(
            (float(row.get("style_density") or 0.0) * 0.4 + float(row.get("substance_density") or 0.0) * 0.6)
            * math.log(max(2, int(row.get("word_count") or 2)))
        ),
        -float(row.get("duration_seconds") or 0.0),
        str(row.get("video_title") or ""),
    ))
    samples_path, _, _ = _avatar_personality_long_form_paths(avatar)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples


def _read_avatar_personality_long_form_page(
    session: Session,
    avatar: Avatar,
    *,
    offset: int,
    limit: int,
    state: str = "all",
) -> AvatarPersonalityLongFormPageRead:
    normalized_state = str(state or "all").strip().lower()
    if normalized_state not in {"all", "included", "rejected"}:
        normalized_state = "all"
    samples = _build_avatar_personality_long_form_samples(session, avatar)
    state_map = _load_avatar_personality_long_form_states(avatar)
    config = _load_avatar_personality_long_form_config(avatar)

    items: list[AvatarPersonalityLongFormSampleRead] = []
    included_count = 0
    rejected_count = 0
    for row in samples:
        sample_state = str(state_map.get(str(row.get("sample_id")), "included") or "included")
        if sample_state == "rejected":
            rejected_count += 1
        else:
            included_count += 1
        if normalized_state != "all" and sample_state != normalized_state:
            continue
        items.append(
            AvatarPersonalityLongFormSampleRead(
                sample_id=str(row.get("sample_id") or ""),
                video_id=int(row.get("video_id") or 0),
                video_title=str(row.get("video_title") or ""),
                start_time=float(row.get("start_time") or 0.0),
                end_time=float(row.get("end_time") or 0.0),
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                word_count=int(row.get("word_count") or 0),
                segment_count=int(row.get("segment_count") or 0),
                text=str(row.get("text") or ""),
                style_density=float(row.get("style_density") or 0.0),
                substance_density=float(row.get("substance_density") or 0.0),
                state="rejected" if sample_state == "rejected" else "included",
            )
        )

    total = len(items)
    page_items = items[offset: offset + limit]
    selected_count = min(max(0, int(config.take_count or 0)), included_count)
    _write_avatar_personality_long_form_config(
        avatar,
        take_count=int(config.take_count or 0),
        included_count=included_count,
        rejected_count=rejected_count,
        selected_count=selected_count,
    )
    return AvatarPersonalityLongFormPageRead(
        avatar_id=int(avatar.id),
        total=total,
        included_count=included_count,
        rejected_count=rejected_count,
        selected_count=selected_count,
        take_count=int(config.take_count or 0),
        limit=int(limit),
        offset=int(offset),
        has_more=(offset + len(page_items)) < total,
        items=page_items,
    )


def _avatar_response_looks_incomplete(text: str) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return True
    if cleaned.endswith(("...", "-", ":", ";", ",")):
        return True
    if cleaned.endswith((".", "!", "?", "\"", "'")):
        return False
    trailing = cleaned.split()[-1].strip(".,!?;:'\"").lower()
    trailing_connectors = {
        "and",
        "but",
        "or",
        "so",
        "because",
        "if",
        "when",
        "while",
        "that",
        "which",
        "who",
        "where",
        "with",
        "about",
        "of",
        "to",
        "for",
        "from",
        "in",
        "on",
        "at",
        "by",
        "than",
        "then",
        "like",
    }
    if trailing in trailing_connectors:
        return True
    return _avatar_word_count(cleaned) <= 12 and cleaned[-1].islower()


def _extend_avatar_response_turn(
    turns: list[dict[str, object]],
    start_index: int,
    *,
    target_speaker_id: int,
) -> dict[str, object]:
    base_turn = turns[start_index]
    merged_text_parts = [_clean_avatar_dataset_text(str(base_turn.get("text") or ""))]
    merged_segment_ids = [int(segment_id) for segment_id in base_turn.get("source_segment_ids", [])]
    merged_end_time = float(base_turn.get("end_time") or base_turn.get("start_time") or 0.0)
    consumed_indexes = [start_index]
    continuation_gap_seconds = 16.0
    max_skipped_backchannels = 2
    max_response_segments = 5
    max_response_words = 320
    probe_index = start_index + 1

    while probe_index < len(turns) and len(consumed_indexes) < max_response_segments:
        skipped_backchannels: list[int] = []
        lookahead = probe_index
        while lookahead < len(turns):
            candidate = turns[lookahead]
            candidate_speaker_id = int(candidate.get("speaker_id") or 0)
            if candidate_speaker_id == target_speaker_id:
                candidate_text = _clean_avatar_dataset_text(str(candidate.get("text") or ""))
                gap = max(0.0, float(candidate.get("start_time") or 0.0) - merged_end_time)
                if gap > continuation_gap_seconds:
                    lookahead = len(turns)
                    break
                current_response_text = _clean_avatar_dataset_text(" ".join(merged_text_parts))
                if skipped_backchannels and not _avatar_response_looks_incomplete(current_response_text):
                    lookahead = len(turns)
                    break
                if (
                    not skipped_backchannels
                    and gap > 2.5
                    and not _avatar_response_looks_incomplete(current_response_text)
                    and not (candidate_text[:1].islower() if candidate_text else False)
                ):
                    lookahead = len(turns)
                    break
                if _avatar_word_count(current_response_text) + _avatar_word_count(candidate_text) > max_response_words:
                    lookahead = len(turns)
                    break
                merged_text_parts.append(candidate_text)
                merged_segment_ids.extend(int(segment_id) for segment_id in candidate.get("source_segment_ids", []))
                merged_end_time = float(candidate.get("end_time") or candidate.get("start_time") or merged_end_time)
                consumed_indexes.append(lookahead)
                probe_index = lookahead + 1
                break
            if len(skipped_backchannels) >= max_skipped_backchannels or not _avatar_is_short_backchannel(candidate):
                lookahead = len(turns)
                break
            skipped_backchannels.append(lookahead)
            lookahead += 1
        if lookahead >= len(turns):
            break

    return {
        "text": _clean_avatar_dataset_text(" ".join(part for part in merged_text_parts if part)),
        "end_time": merged_end_time,
        "source_segment_ids": merged_segment_ids,
        "consumed_indexes": consumed_indexes,
    }


def _avatar_normalize_curation_key(text: str | None) -> str:
    cleaned = _clean_avatar_dataset_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _avatar_has_transcript_noise(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return True
    if "http://" in cleaned or "https://" in cleaned or "www." in cleaned.lower():
        return True
    weird_chars = sum(1 for char in cleaned if not (char.isalnum() or char.isspace() or char in ".,!?':;\"()-"))
    if weird_chars > max(4, len(cleaned) * 0.08):
        return True
    repeated_words = re.findall(r"\b(\w+)\b(?:\s+\1\b){2,}", cleaned.lower())
    return bool(repeated_words)


def _avatar_is_acknowledgement_response(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text).lower().strip(" .,!?:;-'\"")
    if not cleaned:
        return True
    if _avatar_word_count(cleaned) > 8:
        return False
    acknowledgement_patterns = {
        "yeah",
        "yeah yeah",
        "yes",
        "yep",
        "yup",
        "right",
        "exactly",
        "i know",
        "sure",
        "okay",
        "ok",
        "true",
        "fair",
        "for sure",
        "totally",
        "absolutely",
        "uh huh",
        "mhm",
        "mm hmm",
        "no",
    }
    if cleaned in acknowledgement_patterns:
        return True
    if _avatar_word_count(cleaned) <= 2:
        return True
    if _avatar_word_count(cleaned) == 3:
        opinion_words = {"funny", "crazy", "wild", "insane", "amazing", "terrible", "awesome",
                         "ridiculous", "hilarious", "interesting", "wrong", "true", "fair", "weird"}
        return not any(word in cleaned.split() for word in opinion_words)
    return False


def _avatar_response_has_high_referentiality(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text).lower()
    if not cleaned:
        return False
    referential_markers = [
        "look at", "right there", "right here", "this one", "that clip",
        "that thing", "over there", "you see", "you can see", "watch this",
        "on the screen", "on screen", "in the chat", "in the comments",
        "pull that up", "scroll down", "check this out", "as you can see",
    ]
    match_count = sum(1 for marker in referential_markers if marker in cleaned)
    return match_count >= 2 or (match_count >= 1 and _avatar_word_count(cleaned) < 20)


def _avatar_context_has_question(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    lowered = cleaned.lower()
    return any(marker in lowered for marker in [" why ", " how ", " what ", " when ", " where ", " who ", "did ", "does ", "is ", "are ", "can ", "could "])


_AVATAR_BASE_STYLE_MARKERS = [
    " i think ",
    " i mean ",
    " honestly ",
    " basically ",
    " like ",
    " actually ",
    " literally ",
    " kinda ",
    " sort of ",
    " probably ",
    " maybe ",
    " because ",
    " the thing is ",
    " you know ",
    " i'm ",
    " i've ",
    " don't ",
    " can't ",
    " won't ",
]


def _avatar_build_speaker_style_weights(
    turns_by_video: dict[int, list[dict[str, object]]],
    target_speaker_id: int,
) -> dict[str, float]:
    speaker_word_count = 0
    other_word_count = 0
    speaker_marker_counts: dict[str, int] = {marker.strip(): 0 for marker in _AVATAR_BASE_STYLE_MARKERS}
    other_marker_counts: dict[str, int] = {marker.strip(): 0 for marker in _AVATAR_BASE_STYLE_MARKERS}
    for turns in turns_by_video.values():
        for turn in turns:
            text = f" {_clean_avatar_dataset_text(str(turn.get('text') or '')).lower()} "
            wc = len(text.split())
            is_target = int(turn.get("speaker_id") or 0) == target_speaker_id
            if is_target:
                speaker_word_count += wc
            else:
                other_word_count += wc
            for marker in _AVATAR_BASE_STYLE_MARKERS:
                count = text.count(marker)
                if count > 0:
                    key = marker.strip()
                    if is_target:
                        speaker_marker_counts[key] += count
                    else:
                        other_marker_counts[key] += count
    # If the comparison pool is too small, the frequency ratios are
    # unreliable and generic markers get inflated to the 3.0 cap.
    # Fall back to uniform weights when there isn't enough non-target speech.
    min_comparison_words = 500
    comparison_is_sparse = other_word_count < min_comparison_words

    weights: dict[str, float] = {}
    for marker in _AVATAR_BASE_STYLE_MARKERS:
        key = marker.strip()
        if comparison_is_sparse:
            weights[key] = 1.0
            continue
        speaker_rate = (speaker_marker_counts[key] / max(1, speaker_word_count)) * 1000
        other_rate = (other_marker_counts[key] / max(1, other_word_count)) * 1000
        if speaker_rate > 0 and other_rate > 0:
            ratio = speaker_rate / max(0.01, other_rate)
        elif speaker_rate > 0:
            ratio = 2.0
        else:
            ratio = 0.5
        weights[key] = min(3.0, max(0.2, ratio))
    return weights


_AVATAR_SUBSTANCE_MARKERS_LOGIC = [
    " therefore ", " because ", " consequently ", " the reason is ",
    " my point is ", " that's why ", " so the ", " which means ",
    " it follows ", " in other words ", " what that means is ",
    " if you think about it ", " the argument is ",
]
_AVATAR_SUBSTANCE_MARKERS_ANALOGY = [
    " it's like ", " think of it as ", " imagine ", " same way that ",
    " kind of like ", " similar to ", " just like ", " picture this ",
    " analogy ", " metaphor ", " compared to ",
]
_AVATAR_SUBSTANCE_MARKERS_EVIDENCE = [
    " studies show ", " according to ", " research ", " the data ",
    " evidence ", " historically ", " for example ", " for instance ",
    " the fact is ", " statistically ",
]
_AVATAR_SUBSTANCE_MARKERS_POSITION = [
    " i believe ", " the problem is ", " what people don't realize ",
    " the issue is ", " in my view ", " the real question ",
    " fundamentally ", " the key is ", " here's the thing ",
    " what i'm saying is ", " my position ", " i would argue ",
]
_AVATAR_ALL_SUBSTANCE_MARKERS = (
    _AVATAR_SUBSTANCE_MARKERS_LOGIC
    + _AVATAR_SUBSTANCE_MARKERS_ANALOGY
    + _AVATAR_SUBSTANCE_MARKERS_EVIDENCE
    + _AVATAR_SUBSTANCE_MARKERS_POSITION
)


_AVATAR_COMMON_SENTENCE_STARTERS = {
    "this", "that", "well", "what", "when", "where", "who", "how", "why",
    "the", "and", "but", "so", "if", "its", "they", "there", "here",
    "yeah", "yes", "no", "not", "now", "then", "also", "just", "like",
    "right", "okay", "sure", "look", "let", "see", "think", "know",
    "people", "some", "because", "every", "even", "still",
}


def _avatar_substance_signal_count(text: str | None) -> int:
    lowered = f" {_clean_avatar_dataset_text(text).lower()} "
    if not lowered.strip():
        return 0
    count = 0
    for marker in _AVATAR_ALL_SUBSTANCE_MARKERS:
        if marker in lowered:
            count += 1
    # Count mid-sentence proper nouns as reference signals, excluding
    # common words that just happen to be capitalized at sentence start.
    cleaned = _clean_avatar_dataset_text(text) or ""
    proper_noun_count = 0
    sentences = re.split(r'[.!?]+\s+', cleaned)
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:
            stripped = word.strip(".,!?;:'\"()-")
            if stripped and stripped[0].isupper() and len(stripped) >= 3 and stripped.lower() not in _AVATAR_COMMON_SENTENCE_STARTERS:
                proper_noun_count += 1
    count += min(3, proper_noun_count)
    return count


def _avatar_style_signal_count(text: str | None, speaker_style_weights: dict[str, float] | None = None) -> int:
    lowered = f" {_clean_avatar_dataset_text(text).lower()} "
    if not lowered.strip():
        return 0
    total = 0.0
    for marker in _AVATAR_BASE_STYLE_MARKERS:
        if marker in lowered:
            weight = (speaker_style_weights or {}).get(marker.strip(), 1.0)
            total += weight
    return int(round(total))


def _score_avatar_personality_example(row: dict[str, object], *, speaker_style_weights: dict[str, float] | None = None) -> dict[str, object]:
    response_text = _clean_avatar_dataset_text(str(row.get("response_text") or ""))
    context_text = _clean_avatar_dataset_text(str(row.get("context_text") or ""))
    response_word_count = _avatar_word_count(response_text)
    context_word_count = _avatar_word_count(context_text)
    context_turns = int(row.get("context_turns") or 0)
    source_segment_count = len([segment_id for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()])

    completion_score = 90
    context_score = 25
    style_score = 35
    reject_reasons: list[str] = []

    if response_word_count < 8:
        completion_score -= 40
        style_score -= 20
        reject_reasons.append("short_response")
    if _avatar_is_acknowledgement_response(response_text):
        completion_score -= 35
        style_score -= 30
        reject_reasons.append("acknowledgement_only")
    if _avatar_response_looks_incomplete(response_text):
        completion_score -= 45
        reject_reasons.append("truncated_response")
    if _avatar_has_transcript_noise(response_text):
        completion_score -= 30
        style_score -= 25
        reject_reasons.append("transcript_noise")
    if _avatar_response_has_high_referentiality(response_text):
        context_score -= 20
        style_score -= 15
        reject_reasons.append("high_referentiality")

    if context_turns <= 0 or context_word_count < 12:
        context_score -= 30
        reject_reasons.append("weak_context")
    else:
        context_score += min(35, max(0, context_word_count - 12) // 3)
        context_score += min(15, context_turns * 5)
    if _avatar_context_has_question(context_text):
        context_score += 10

    if 20 <= response_word_count <= 240:
        style_score += 20
    elif response_word_count >= 12:
        style_score += 8
    elif response_word_count > 0:
        style_score -= 10

    if source_segment_count > 1:
        completion_score += 10
        style_score += 8

    style_score += min(20, _avatar_style_signal_count(response_text, speaker_style_weights) * 5)
    if any(pronoun in f" {response_text.lower()} " for pronoun in [" i ", " i'm ", " i've ", " me ", " my "]):
        style_score += 8

    substance_score = 15
    substance_signals = _avatar_substance_signal_count(response_text)
    substance_score += min(30, substance_signals * 6)
    if response_word_count < 20:
        substance_score -= 15
    if substance_signals > 0 and _avatar_style_signal_count(response_text, speaker_style_weights) > 0:
        substance_score += 10
    substance_score = max(0, min(100, substance_score))

    completion_score = max(0, min(100, completion_score))
    context_score = max(0, min(100, context_score))
    style_score = max(0, min(100, style_score))
    quality_score = max(0, min(100, round(
        (completion_score * 0.40) + (context_score * 0.22) + (style_score * 0.28) + (substance_score * 0.10)
    )))

    hard_reject_reasons = {"acknowledgement_only", "transcript_noise", "weak_context", "truncated_response"}
    if hard_reject_reasons.intersection(reject_reasons) or quality_score < 40:
        auto_label = "reject"
    elif quality_score >= 78 and completion_score >= 70 and context_score >= 55 and style_score >= 60:
        auto_label = "gold"
    else:
        auto_label = "silver"

    row["response_word_count"] = int(response_word_count)
    row["context_word_count"] = int(context_word_count)
    row["source_segment_count"] = int(source_segment_count)
    row["quality_score"] = int(quality_score)
    row["completion_score"] = int(completion_score)
    row["context_score"] = int(context_score)
    row["style_score"] = int(style_score)
    row["substance_score"] = int(substance_score)
    row["reject_reasons"] = sorted(set(reject_reasons))
    row["auto_label"] = auto_label
    return row


def _apply_avatar_duplicate_rejects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_indexes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        dedupe_key = " || ".join(
            [
                _avatar_normalize_curation_key(str(row.get("context_text") or "")),
                _avatar_normalize_curation_key(str(row.get("response_text") or "")),
            ]
        )
        if not dedupe_key.strip(" |"):
            continue
        grouped_indexes.setdefault(dedupe_key, []).append(index)

    for duplicate_indexes in grouped_indexes.values():
        if len(duplicate_indexes) <= 1:
            continue
        group_id = int(min(int(rows[idx].get("example_id") or 0) for idx in duplicate_indexes) + 1)
        ranked = sorted(
            duplicate_indexes,
            key=lambda idx: (
                -int(rows[idx].get("quality_score") or 0),
                -int(rows[idx].get("style_score") or 0),
                int(rows[idx].get("example_id") or 0),
            ),
        )
        keep_index = ranked[0]
        rows[keep_index]["duplicate_group_id"] = group_id
        rows[keep_index]["duplicate_group_size"] = len(duplicate_indexes)
        for duplicate_index in ranked[1:]:
            row = rows[duplicate_index]
            reject_reasons = {str(reason) for reason in row.get("reject_reasons", [])}
            reject_reasons.add("duplicate_example")
            row["reject_reasons"] = sorted(reject_reasons)
            row["quality_score"] = max(0, int(row.get("quality_score") or 0) - 20)
            row["auto_label"] = "reject"
            row["duplicate_group_id"] = group_id
            row["duplicate_group_size"] = len(duplicate_indexes)
    return rows


def _avatar_context_text_for_embedding(raw_context: str | None) -> str:
    lines = []
    for line in str(raw_context or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "Conversation context:" or stripped.startswith("Podcast episode:"):
            continue
        separator_index = stripped.find(":")
        if separator_index > 0:
            stripped = stripped[separator_index + 1 :].strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def _avatar_embedding_tokens(text: str | None) -> list[str]:
    cleaned = _avatar_normalize_curation_key(text)
    if not cleaned:
        return []
    stop_words = {
        "the", "a", "an", "and", "or", "but", "so", "that", "this", "these", "those", "it", "its", "to", "of",
        "for", "on", "in", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "they", "we", "me", "my", "our", "your", "their", "them", "his", "her",
        "do", "does", "did", "not", "just", "like", "really", "very", "kind", "sort", "have", "has", "had",
    }
    return [token for token in cleaned.split() if len(token) > 1 and token not in stop_words][:96]


def _avatar_hash_embedding_feature(feature: str, *, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, "little", signed=False)
    return hashed % dimension, (1.0 if ((hashed >> 8) & 1) == 0 else -1.0)


def _avatar_build_example_embedding(row: dict[str, object], *, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    response_tokens = _avatar_embedding_tokens(str(row.get("response_text") or ""))
    context_tokens = _avatar_embedding_tokens(_avatar_context_text_for_embedding(str(row.get("context_text") or "")))

    def add_feature(feature: str, weight: float) -> None:
        index, sign = _avatar_hash_embedding_feature(feature, dimension=dimension - 8)
        vector[index] += np.float32(weight * sign)

    for token in response_tokens:
        add_feature(f"r:{token}", 1.35)
    for token in context_tokens:
        add_feature(f"c:{token}", 0.7)
    for left, right in zip(response_tokens, response_tokens[1:]):
        add_feature(f"rb:{left}_{right}", 1.6)
    for left, right in zip(context_tokens, context_tokens[1:]):
        add_feature(f"cb:{left}_{right}", 0.85)

    response_text = _clean_avatar_dataset_text(str(row.get("response_text") or "")).lower()
    style_markers = ["i think", "i mean", "honestly", "basically", "because", "the thing is", "you know", "i'm", "i've", "don't", "can't", "won't"]
    for marker in style_markers:
        if marker in response_text:
            add_feature(f"s:{marker}", 1.1)
    substance_markers = [
        "therefore", "the reason is", "my point is", "that's why", "which means",
        "it's like", "think of it as", "imagine", "same way that",
        "i believe", "the problem is", "the issue is", "here's the thing",
        "for example", "for instance", "fundamentally",
    ]
    for marker in substance_markers:
        if marker in response_text:
            add_feature(f"sub:{marker}", 1.3)

    vector[-10] = min(1.0, float(row.get("response_word_count") or 0) / 320.0)
    vector[-9] = min(1.0, float(row.get("context_word_count") or 0) / 180.0)
    vector[-8] = min(1.0, float(row.get("context_turns") or 0) / 4.0)
    vector[-7] = min(1.0, float(row.get("source_segment_count") or 0) / 5.0)
    vector[-6] = min(1.0, float(row.get("style_score") or 0) / 100.0)
    vector[-5] = min(1.0, float(row.get("substance_score") or 0) / 100.0)
    vector[-4] = 1.0 if _avatar_context_has_question(str(row.get("context_text") or "")) else 0.0
    vector[-3] = min(1.0, _avatar_substance_signal_count(str(row.get("response_text") or "")) / 5.0)
    vector[-2] = 1.0 if "truncated_response" in {str(reason) for reason in row.get("reject_reasons", [])} else 0.0
    vector[-1] = 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _avatar_assign_embedding_clusters(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not rows:
        return rows, {"embedding_model": "hashing_ngram_v1", "dimension": 128, "cluster_count": 0, "duplicate_example_count": 0, "hotspot_cluster_count": 0, "clusters": []}

    embeddings = np.vstack([_avatar_build_example_embedding(row) for row in rows]).astype(np.float32)
    
    semantic_embeddings = None
    try:
        from .services import semantic_search as sem_svc
        texts = [str(row.get("response_text") or "") for row in rows]
        semantic_embeddings = sem_svc._embed_texts(texts)
    except Exception as e:
        print(f"[_avatar_assign_embedding_clusters] Failed to generate semantic embeddings: {e}")

    rng = np.random.default_rng(42)
    lsh_a = rng.standard_normal((embeddings.shape[1], 10)).astype(np.float32)
    lsh_b = rng.standard_normal((embeddings.shape[1], 10)).astype(np.float32)
    bucket_maps: list[dict[str, list[int]]] = [{}, {}]
    projections = [lsh_a, lsh_b]
    candidate_limit = 18
    similarity_threshold = 0.89
    semantic_threshold = 0.85

    sorted_indexes = sorted(
        range(len(rows)),
        key=lambda idx: (
            -int(rows[idx].get("quality_score") or 0),
            -int(rows[idx].get("style_score") or 0),
            int(rows[idx].get("example_id") or 0),
        ),
    )

    duplicate_groups: dict[int, list[int]] = {}
    representative_group_id: dict[int, int] = {}
    next_group_id = max((int(row.get("duplicate_group_id") or 0) for row in rows), default=0) + 1
    scanned_indexes: list[int] = []

    for idx in sorted_indexes:
        vector = embeddings[idx]
        candidate_indexes: set[int] = set()
        for projection, bucket_map in zip(projections, bucket_maps):
            signature = "".join("1" if value >= 0 else "0" for value in np.matmul(vector, projection))
            candidate_indexes.update(bucket_map.get(signature, []))

        best_match_idx: int | None = None
        best_similarity = -1.0
        for candidate_idx in candidate_indexes:
            similarity = float(np.dot(vector, embeddings[candidate_idx]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = candidate_idx

        # Semantic duplication check
        semantic_best_match_idx: int | None = None
        semantic_best_similarity = -1.0
        if semantic_embeddings is not None and len(scanned_indexes) > 0:
            sem_vec = semantic_embeddings[idx]
            sem_candidates = semantic_embeddings[scanned_indexes]
            sims = np.matmul(sem_candidates, sem_vec)
            best_idx_in_scanned = int(np.argmax(sims))
            best_sim = float(sims[best_idx_in_scanned])
            if best_sim >= semantic_threshold:
                semantic_best_similarity = best_sim
                semantic_best_match_idx = scanned_indexes[best_idx_in_scanned]

        is_lsh_dup = best_match_idx is not None and best_similarity >= similarity_threshold
        is_sem_dup = semantic_best_match_idx is not None and semantic_best_similarity >= semantic_threshold

        if is_lsh_dup or is_sem_dup:
            target_match_idx = semantic_best_match_idx if is_sem_dup else best_match_idx
            if target_match_idx is None: 
                continue # Should never happen
                
            group_id = representative_group_id.get(target_match_idx)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                representative_group_id[target_match_idx] = group_id
                duplicate_groups[group_id] = [target_match_idx]
            duplicate_groups.setdefault(group_id, []).append(idx)
            rows[idx]["duplicate_group_id"] = group_id
            rows[idx]["duplicate_similarity"] = round(max(best_similarity, semantic_best_similarity), 4)
            reject_reasons = {str(reason) for reason in rows[idx].get("reject_reasons", [])}
            if is_sem_dup:
                reject_reasons.add("semantic_duplicate")
            if is_lsh_dup:
                reject_reasons.add("lsh_duplicate")
            rows[idx]["reject_reasons"] = sorted(reject_reasons)
            rows[idx]["auto_label"] = "reject"
            rows[idx]["quality_score"] = max(0, int(rows[idx].get("quality_score") or 0) - 18)
            continue

        scanned_indexes.append(idx)

        for projection, bucket_map in zip(projections, bucket_maps):
            signature = "".join("1" if value >= 0 else "0" for value in np.matmul(vector, projection))
            members = bucket_map.setdefault(signature, [])
            if len(members) < candidate_limit:
                members.append(idx)

    for group_id, member_indexes in duplicate_groups.items():
        member_count = len(member_indexes)
        for member_idx in member_indexes:
            rows[member_idx]["duplicate_group_id"] = group_id
            rows[member_idx]["duplicate_group_size"] = member_count

    active_indexes = [idx for idx, row in enumerate(rows) if str(row.get("auto_label") or "silver") != "reject"]
    cluster_count_target = max(16, min(72, int(round(np.sqrt(max(1, len(active_indexes))) * 0.8))))
    cluster_threshold = 0.58
    centroids: list[np.ndarray] = []
    centroid_sizes: list[int] = []
    cluster_members: dict[int, list[int]] = {}

    active_sorted_indexes = sorted(
        active_indexes,
        key=lambda idx: (
            -int(rows[idx].get("quality_score") or 0),
            -int(rows[idx].get("style_score") or 0),
            int(rows[idx].get("example_id") or 0),
        ),
    )

    for idx in active_sorted_indexes:
        vector = embeddings[idx]
        if not centroids:
            centroids.append(vector.copy())
            centroid_sizes.append(1)
            cluster_members[1] = [idx]
            rows[idx]["cluster_id"] = 1
            continue

        centroid_matrix = np.vstack(centroids)
        similarities = np.matmul(centroid_matrix, vector)
        best_cluster_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_cluster_index])

        if len(centroids) < cluster_count_target and best_similarity < cluster_threshold:
            cluster_id = len(centroids) + 1
            centroids.append(vector.copy())
            centroid_sizes.append(1)
            cluster_members[cluster_id] = [idx]
            rows[idx]["cluster_id"] = cluster_id
            continue

        cluster_id = best_cluster_index + 1
        cluster_members.setdefault(cluster_id, []).append(idx)
        current_size = centroid_sizes[best_cluster_index]
        updated_centroid = (centroids[best_cluster_index] * current_size) + vector
        norm = float(np.linalg.norm(updated_centroid))
        if norm > 0:
            updated_centroid /= norm
        centroids[best_cluster_index] = updated_centroid.astype(np.float32)
        centroid_sizes[best_cluster_index] = current_size + 1
        rows[idx]["cluster_id"] = cluster_id

    if centroids:
        centroid_matrix = np.vstack(centroids)
        for idx, row in enumerate(rows):
            if row.get("cluster_id") is not None:
                continue
            similarities = np.matmul(centroid_matrix, embeddings[idx])
            cluster_id = int(np.argmax(similarities)) + 1
            row["cluster_id"] = cluster_id
            cluster_members.setdefault(cluster_id, []).append(idx)

    cluster_sizes = {cluster_id: len(member_indexes) for cluster_id, member_indexes in cluster_members.items()}
    largest_cluster_size = max(cluster_sizes.values(), default=0)
    hotspot_threshold = max(18, int(np.percentile(list(cluster_sizes.values()), 85))) if cluster_sizes else 0

    for row in rows:
        cluster_id = row.get("cluster_id")
        cluster_size = cluster_sizes.get(int(cluster_id), 0) if cluster_id is not None else 0
        row["cluster_size"] = cluster_size
        if largest_cluster_size > 1 and cluster_size > 0:
            row["diversity_score"] = max(0, min(100, int(round(100 * (1 - ((cluster_size - 1) / max(1, largest_cluster_size - 1)))))))
        else:
            row["diversity_score"] = 100 if cluster_size > 0 else 0
        if cluster_size >= hotspot_threshold and hotspot_threshold > 0:
            row["cluster_hotspot"] = True

    cluster_summary = []
    for cluster_id, member_indexes in sorted(cluster_members.items(), key=lambda item: (-len(item[1]), item[0])):
        top_examples = sorted(
            member_indexes,
            key=lambda idx: (
                -int(rows[idx].get("quality_score") or 0),
                -int(rows[idx].get("style_score") or 0),
                int(rows[idx].get("example_id") or 0),
            ),
        )[:3]
        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(member_indexes)),
                "representative_example_ids": [int(rows[idx].get("example_id") or 0) for idx in top_examples],
            }
        )

    return rows, {
        "embedding_model": "hashing_ngram_v1",
        "dimension": int(embeddings.shape[1]),
        "cluster_count": int(len(cluster_members)),
        "duplicate_example_count": int(sum(1 for row in rows if _avatar_row_has_duplicate_risk(row))),
        "hotspot_cluster_count": int(sum(1 for size in cluster_sizes.values() if size >= hotspot_threshold and hotspot_threshold > 0)),
        "largest_cluster_size": int(largest_cluster_size),
        "clusters": cluster_summary[:48],
    }


def _resolve_avatar_personality_example_state(row: dict[str, object], state_map: dict[int, str]) -> tuple[str, str | None]:
    try:
        example_id = int(row.get("example_id"))
    except Exception:
        return "approved", None
    manual_state = state_map.get(example_id)
    if manual_state in {"approved", "rejected"}:
        return manual_state, manual_state
    auto_label = str(row.get("auto_label") or "silver").strip().lower()
    return ("rejected" if auto_label == "reject" else "approved"), None


def _avatar_row_has_duplicate_risk(row: dict[str, object]) -> bool:
    reasons = {str(reason) for reason in row.get("reject_reasons", [])}
    return int(row.get("duplicate_group_size") or 0) > 1 or any("duplicate" in reason for reason in reasons)


def _build_avatar_personality_training_readiness(
    *,
    approved_count: int,
    approved_gold_count: int,
    approved_duration_seconds: float,
    approved_word_count: int,
    needs_review_count: int,
    duplicate_count: int,
    hotspot_cluster_count: int,
    largest_cluster_size: int,
    total_example_count: int,
    auto_reject_count: int,
) -> AvatarPersonalityTrainingReadinessRead:
    approved_count = max(0, int(approved_count or 0))
    approved_gold_count = max(0, int(approved_gold_count or 0))
    approved_duration_seconds = max(0.0, float(approved_duration_seconds or 0.0))
    approved_word_count = max(0, int(approved_word_count or 0))
    needs_review_count = max(0, int(needs_review_count or 0))
    duplicate_count = max(0, int(duplicate_count or 0))
    hotspot_cluster_count = max(0, int(hotspot_cluster_count or 0))
    largest_cluster_size = max(0, int(largest_cluster_size or 0))
    total_example_count = max(0, int(total_example_count or 0))
    auto_reject_count = max(0, int(auto_reject_count or 0))

    approved_duration_hours = approved_duration_seconds / 3600.0
    duplicate_pressure = int(round((duplicate_count / max(1, total_example_count)) * 100))
    hotspot_pressure = int(round((largest_cluster_size / max(1, approved_count)) * 100)) if approved_count else 0
    gold_ratio = (approved_gold_count / max(1, approved_count)) if approved_count else 0.0
    reject_ratio = (auto_reject_count / max(1, total_example_count)) if total_example_count else 0.0

    coverage_score = min(100.0, (approved_count / 2000.0) * 100.0)
    gold_score = min(100.0, (approved_gold_count / 800.0) * 100.0)
    duration_score = min(100.0, (approved_duration_hours / 8.0) * 100.0)
    score = int(round((coverage_score * 0.32) + (gold_score * 0.46) + (duration_score * 0.22)))
    score -= int(max(0, duplicate_pressure - 8) * 0.8)
    score -= int(max(0, hotspot_pressure - 10) * 0.9)
    if gold_ratio < 0.35:
        score -= int(round((0.35 - gold_ratio) * 70))
    score = max(0, min(100, score))

    can_train_now = approved_count >= 800 and approved_gold_count >= 300 and approved_duration_hours >= 3.0 and approved_word_count >= 40000
    status = "insufficient"
    if approved_count >= 12000 or approved_word_count >= 600000:
        status = "oversized"
    elif approved_count >= 2500 and approved_gold_count >= 900 and approved_duration_hours >= 8.0:
        status = "strong"
    elif can_train_now:
        status = "ready"
    elif approved_count >= 300 and approved_gold_count >= 120 and approved_duration_hours >= 1.5:
        status = "borderline"

    manual_review_roi: Literal["high", "medium", "low"] = "high"
    if can_train_now:
        manual_review_roi = "medium"
    if status in {"strong", "oversized"} and needs_review_count <= max(150, int(approved_count * 0.1)):
        manual_review_roi = "low"
    elif needs_review_count > max(250, int(approved_count * 0.2)) or gold_ratio < 0.35:
        manual_review_roi = "high"

    summary = f"{approved_count:,} included examples, {approved_gold_count:,} gold, and about {approved_duration_hours:.1f}h of approved response audio."
    recommended_action = "Keep curating high-value exchanges before training."
    caution_parts: list[str] = []

    if status == "insufficient":
        recommended_action = "Do more manual approval. You do not have enough high-value personality data yet."
    elif status == "borderline":
        recommended_action = "You can run a pilot LoRA now, but more manual approval should still improve style fidelity."
    elif status == "ready":
        recommended_action = "You have enough data to train now. Further manual approval should focus on precision, not volume."
    elif status == "strong":
        recommended_action = "You are well past the minimum. Prefer pruning weak repeats over adding more volume."
    elif status == "oversized":
        recommended_action = "Train from the current set or a gold-biased subset. More raw volume is unlikely to help."

    if duplicate_pressure >= 12:
        caution_parts.append("duplicate pressure is elevated")
    if hotspot_pressure >= 14 or hotspot_cluster_count >= 8:
        caution_parts.append("one topic cluster is starting to dominate the corpus")
    if reject_ratio >= 0.22:
        caution_parts.append("the raw source pool is noisy")
    if status == "oversized":
        caution_parts.append("too much repetitive or generic data can dilute the speaker's style and slow training")

    return AvatarPersonalityTrainingReadinessRead(
        status=status,
        score=score,
        can_train_now=can_train_now,
        approved_duration_hours=round(approved_duration_hours, 2),
        approved_word_count=approved_word_count,
        recommended_action=recommended_action,
        summary=summary,
        caution=(". ".join(caution_parts).strip().rstrip(".") + ".") if caution_parts else None,
        manual_review_roi=manual_review_roi,
        duplicate_pressure=duplicate_pressure,
        hotspot_pressure=hotspot_pressure,
        gold_ratio=round(gold_ratio, 3),
        reject_ratio=round(reject_ratio, 3),
    )


def _avatar_extract_json_object(raw_text: str) -> dict[str, object]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    return data


def _avatar_normalize_llm_label(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"gold", "silver", "reject"}:
        return normalized
    return "silver"


def _avatar_resolve_local_judge_model() -> tuple[str, str]:
    import httpx

    ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    judge_override = _normalize_ollama_model_ref(os.getenv("AVATAR_JUDGE_MODEL", ""))
    requested_model = _normalize_ollama_model_ref(os.getenv("OLLAMA_MODEL", "mistral"))
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=8)
        response.raise_for_status()
        available_models = [str(model.get("name") or "") for model in response.json().get("models", []) if isinstance(model, dict)]
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot connect to local Ollama at {ollama_url}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to query local Ollama at {ollama_url}: {exc}") from exc

    if judge_override and any(_ollama_model_name_matches(name, judge_override) for name in available_models):
        return ollama_url, judge_override

    preferred_models = [
        "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
        "qwen2.5:7b",
        "mistral:latest",
        "mistral",
        "qwen3.5:27b",
    ]
    for preferred in preferred_models:
        if any(_ollama_model_name_matches(name, preferred) for name in available_models):
            return ollama_url, preferred

    if requested_model and any(_ollama_model_name_matches(name, requested_model) for name in available_models):
        return ollama_url, requested_model

    if available_models:
        return ollama_url, str(available_models[0])
    raise HTTPException(status_code=503, detail=f"No local Ollama models found at {ollama_url}")


def _avatar_run_local_judge(example: dict[str, object], *, ollama_url: str, model: str) -> dict[str, object]:
    import httpx

    prompt = (
        "You are grading a candidate training example for a personality LoRA.\n"
        "Judge whether the assistant response is a strong example of conversational personality data.\n"
        "Focus on completion, stylistic richness, conversational usefulness, and transcript cleanliness.\n"
        "Usefulness means the response reveals personality traits, speaking patterns, or emotional tendencies "
        "that would help a model replicate this person's communication style.\n"
        "Return JSON only with keys: label, confidence, completion_score, style_score, usefulness_score, reasons, rationale.\n"
        "label must be one of: gold, silver, reject.\n"
        "reasons must be a short array of snake_case strings.\n"
        "Be conservative about gold. Reject truncated, repetitive, transcript-broken, or low-value acknowledgements.\n\n"
        f"Episode: {str(example.get('video_title') or '').strip()}\n"
        f"Context:\n{str(example.get('context_text') or '').strip()}\n\n"
        f"Response:\n{str(example.get('response_text') or '').strip()}\n"
    )

    response = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 160,
            },
        },
        timeout=150,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "").strip()
    if not raw_text:
        raise ValueError("Local judge model returned an empty response")
    data = _avatar_extract_json_object(raw_text)
    completion_score = max(0, min(100, int(data.get("completion_score") or 0)))
    style_score = max(0, min(100, int(data.get("style_score") or 0)))
    usefulness_score = max(0, min(100, int(data.get("usefulness_score") or 0)))
    provided_confidence = int(data.get("confidence") or 0)
    if provided_confidence <= 0:
        provided_confidence = int(round((completion_score + style_score + usefulness_score) / 3))
    return {
        "llm_label": _avatar_normalize_llm_label(str(data.get("label") or "")),
        "llm_confidence": max(0, min(100, provided_confidence)),
        "llm_completion_score": completion_score,
        "llm_style_score": style_score,
        "llm_usefulness_score": usefulness_score,
        "llm_reasons": [str(reason).strip() for reason in data.get("reasons", []) if str(reason).strip()],
        "llm_rationale": _clean_avatar_dataset_text(str(data.get("rationale") or "")) or None,
        "llm_model": model,
        "llm_judged_at": datetime.now().isoformat(),
    }


def _avatar_personality_judge_target_match(
    row: dict[str, object],
    *,
    state_map: dict[int, str],
    normalized_target: str,
) -> tuple[bool, str | None]:
    state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
    current_label = _avatar_normalize_llm_label(str(row.get("auto_label") or "silver"))
    if normalized_target == "needs_review":
        target_match = current_label == "silver" and manual_state is None and state == "approved"
    elif normalized_target == "silver":
        target_match = current_label == "silver"
    else:
        target_match = True
    return target_match, manual_state


def _run_avatar_personality_judge_pass(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    *,
    max_examples: int,
    overwrite_existing: bool,
    target_filter: str,
) -> AvatarPersonalityDatasetRead:
    normalized_target = str(target_filter or "needs_review").strip().lower()
    if normalized_target not in {"needs_review", "silver", "all"}:
        normalized_target = "needs_review"

    ollama_url, judge_model = _avatar_resolve_local_judge_model()
    review_path, _ = _avatar_personality_review_paths(avatar)
    if not review_path.exists():
        raise HTTPException(status_code=404, detail="Dataset review artifact not found. Build the dataset first.")

    state_map = _load_avatar_personality_state_map(avatar)
    rows = list(_iter_avatar_personality_review_examples(avatar))
    judged = 0

    for row in rows:
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        current_label = _avatar_normalize_llm_label(str(row.get("auto_label") or "silver"))
        if normalized_target == "needs_review":
            target_match = current_label == "silver" and manual_state is None and state == "approved"
        elif normalized_target == "silver":
            target_match = current_label == "silver"
        else:
            target_match = True
        if not target_match:
            continue
        if not overwrite_existing and row.get("llm_label"):
            continue
        if judged >= max(1, min(int(max_examples or 40), 200)):
            break

        row["heuristic_label"] = _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))
        try:
            judge_result = _avatar_run_local_judge(row, ollama_url=ollama_url, model=judge_model)
        except Exception as exc:
            row["llm_rationale"] = f"Judge pass failed: {exc}"
            row["llm_model"] = judge_model
            row["llm_judged_at"] = datetime.now().isoformat()
            row["llm_reasons"] = ["judge_error"]
            row["llm_confidence"] = 0
            row["llm_label"] = row["heuristic_label"]
            judged += 1
            continue

        row.update(judge_result)
        if manual_state is None:
            row["auto_label"] = judge_result["llm_label"]
        judged += 1

    review_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    refreshed = _refresh_avatar_personality_dataset_exports(avatar, personality)
    return refreshed


def _start_avatar_personality_judge_pass(
    *,
    avatar_id: int,
    max_examples: int,
    overwrite_existing: bool,
    target_filter: str,
) -> AvatarPersonalityJudgeStatusRead:
    normalized_target = str(target_filter or "needs_review").strip().lower()
    if normalized_target not in {"needs_review", "silver", "all"}:
        normalized_target = "needs_review"
    normalized_max = max(1, min(int(max_examples or 40), 200))

    with Session(engine) as session:
        avatar = session.get(Avatar, avatar_id)
        if not avatar:
            raise HTTPException(status_code=404, detail="Avatar not found")
        speaker = session.get(Speaker, avatar.speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Source speaker not found")
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        review_path, _ = _avatar_personality_review_paths(avatar)
        if not review_path.exists():
            raise HTTPException(status_code=404, detail="Dataset review artifact not found. Build the dataset first.")
        state_map = _load_avatar_personality_state_map(avatar)
        rows = list(_iter_avatar_personality_review_examples(avatar))
        candidates: list[int] = []
        for idx, row in enumerate(rows):
            target_match, _ = _avatar_personality_judge_target_match(
                row,
                state_map=state_map,
                normalized_target=normalized_target,
            )
            if not target_match:
                continue
            if not overwrite_existing and row.get("llm_label"):
                continue
            candidates.append(idx)
            if len(candidates) >= normalized_max:
                break
        ollama_url, judge_model = _avatar_resolve_local_judge_model()

        with _avatar_judge_runs_lock:
            active_thread = _avatar_judge_threads.get(int(avatar_id))
            if active_thread and active_thread.is_alive():
                raise HTTPException(status_code=409, detail="A local judge pass is already running for this avatar.")
            stop_event = threading.Event()
            _avatar_judge_stop_events[int(avatar_id)] = stop_event

        started_at = datetime.now().isoformat()
        initial_status = _write_avatar_personality_judge_status(
            avatar,
            {
                "status": "running",
                "active": True,
                "stop_requested": False,
                "model": judge_model,
                "target_filter": normalized_target,
                "overwrite_existing": bool(overwrite_existing),
                "max_examples": normalized_max,
                "total_candidates": len(candidates),
                "processed_count": 0,
                "judged_count": 0,
                "promoted_count": 0,
                "rejected_count": 0,
                "current_example_id": None,
                "current_video_title": None,
                "current_stage": "queued",
                "started_at": started_at,
                "finished_at": None,
                "error": None,
                "recent_results": [],
            },
        )

    def _runner():
        try:
            with Session(engine) as inner_session:
                avatar_inner = inner_session.get(Avatar, avatar_id)
                if not avatar_inner:
                    raise RuntimeError("Avatar not found")
                speaker_inner = inner_session.get(Speaker, avatar_inner.speaker_id)
                if not speaker_inner:
                    raise RuntimeError("Source speaker not found")
                personality_inner, _, _ = _ensure_avatar_profiles(inner_session, avatar_inner, speaker_name=speaker_inner.name)
                review_path_inner, _ = _avatar_personality_review_paths(avatar_inner)
                state_map_inner = _load_avatar_personality_state_map(avatar_inner)
                rows_inner = list(_iter_avatar_personality_review_examples(avatar_inner))
                recent_results: list[dict[str, object]] = []
                judged = 0
                promoted = 0
                rejected = 0
                processed = 0

                for row_index in candidates:
                    if row_index >= len(rows_inner):
                        continue
                    row = rows_inner[row_index]
                    target_match, manual_state = _avatar_personality_judge_target_match(
                        row,
                        state_map=state_map_inner,
                        normalized_target=normalized_target,
                    )
                    if not target_match:
                        continue
                    if not overwrite_existing and row.get("llm_label"):
                        continue
                    if stop_event.is_set():
                        _write_avatar_personality_judge_status(
                            avatar_inner,
                            {
                                "status": "stopped",
                                "active": False,
                                "stop_requested": True,
                                "processed_count": processed,
                                "judged_count": judged,
                                "promoted_count": promoted,
                                "rejected_count": rejected,
                                "current_stage": "stopped",
                                "current_example_id": None,
                                "current_video_title": None,
                                "finished_at": datetime.now().isoformat(),
                                "recent_results": recent_results,
                            },
                        )
                        break

                    row["heuristic_label"] = _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))
                    _write_avatar_personality_judge_status(
                        avatar_inner,
                        {
                            "status": "stopping" if stop_event.is_set() else "running",
                            "active": True,
                            "stop_requested": stop_event.is_set(),
                            "processed_count": processed,
                            "judged_count": judged,
                            "promoted_count": promoted,
                            "rejected_count": rejected,
                            "current_example_id": int(row.get("example_id") or 0),
                            "current_video_title": str(row.get("video_title") or ""),
                            "current_stage": "judging",
                            "recent_results": recent_results,
                        },
                    )

                    try:
                        judge_result = _avatar_run_local_judge(row, ollama_url=ollama_url, model=judge_model)
                    except Exception as exc:
                        row["llm_rationale"] = f"Judge pass failed: {exc}"
                        row["llm_model"] = judge_model
                        row["llm_judged_at"] = datetime.now().isoformat()
                        row["llm_reasons"] = ["judge_error"]
                        row["llm_confidence"] = 0
                        row["llm_label"] = row["heuristic_label"]
                    else:
                        row.update(judge_result)
                        if manual_state is None:
                            row["auto_label"] = judge_result["llm_label"]

                    judged += 1
                    processed += 1
                    if row.get("llm_label") == "gold" and row.get("heuristic_label") != "gold":
                        promoted += 1
                    if row.get("llm_label") == "reject" and row.get("heuristic_label") != "reject":
                        rejected += 1
                    recent_results.insert(
                        0,
                        {
                            "example_id": int(row.get("example_id") or 0),
                            "video_title": str(row.get("video_title") or ""),
                            "llm_label": _avatar_normalize_llm_label(str(row.get("llm_label") or "silver")),
                            "llm_confidence": int(row.get("llm_confidence") or 0),
                            "llm_reasons": [str(reason) for reason in row.get("llm_reasons", [])],
                            "heuristic_label": _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver")),
                            "judged_at": row.get("llm_judged_at"),
                        },
                    )
                    recent_results = recent_results[:12]
                    _write_avatar_personality_judge_status(
                        avatar_inner,
                        {
                            "status": "stopping" if stop_event.is_set() else "running",
                            "active": True,
                            "stop_requested": stop_event.is_set(),
                            "processed_count": processed,
                            "judged_count": judged,
                            "promoted_count": promoted,
                            "rejected_count": rejected,
                            "current_example_id": int(row.get("example_id") or 0),
                            "current_video_title": str(row.get("video_title") or ""),
                            "current_stage": "writing_result",
                            "recent_results": recent_results,
                        },
                    )

                review_path_inner.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_inner), encoding="utf-8")
                refreshed = _refresh_avatar_personality_dataset_exports(avatar_inner, personality_inner)
                inner_session.add(personality_inner)
                inner_session.commit()
                final_status = "stopped" if stop_event.is_set() else "completed"
                _write_avatar_personality_judge_status(
                    avatar_inner,
                    {
                        "status": final_status,
                        "active": False,
                        "stop_requested": stop_event.is_set(),
                        "processed_count": processed,
                        "judged_count": judged,
                        "promoted_count": promoted,
                        "rejected_count": rejected,
                        "current_example_id": None,
                        "current_video_title": None,
                        "current_stage": "complete",
                        "finished_at": datetime.now().isoformat(),
                        "recent_results": recent_results,
                    },
                )
        except Exception as exc:
            with Session(engine) as error_session:
                avatar_error = error_session.get(Avatar, avatar_id)
                if avatar_error:
                    _write_avatar_personality_judge_status(
                        avatar_error,
                        {
                            "status": "failed",
                            "active": False,
                            "stop_requested": stop_event.is_set(),
                            "current_stage": "failed",
                            "error": str(exc),
                            "finished_at": datetime.now().isoformat(),
                        },
                    )
        finally:
            with _avatar_judge_runs_lock:
                _avatar_judge_stop_events.pop(int(avatar_id), None)
                _avatar_judge_threads.pop(int(avatar_id), None)

    thread = threading.Thread(target=_runner, daemon=True, name=f"avatar-judge-{avatar_id}")
    with _avatar_judge_runs_lock:
        _avatar_judge_threads[int(avatar_id)] = thread
    thread.start()
    return initial_status


def _load_avatar_personality_dataset(avatar: Avatar, personality: AvatarPersonalityProfile) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    metadata: dict[str, object] = {}
    preview_examples: list[AvatarPersonalityDatasetExampleRead] = []

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    if preview_path.exists():
        try:
            preview_rows = json.loads(preview_path.read_text(encoding="utf-8"))
            if isinstance(preview_rows, list):
                preview_examples = [
                    AvatarPersonalityDatasetExampleRead(**row)
                    for row in preview_rows
                    if isinstance(row, dict)
                ]
        except Exception:
            preview_examples = []

    generated_at = None
    raw_generated_at = metadata.get("generated_at")
    if isinstance(raw_generated_at, str):
        try:
            generated_at = datetime.fromisoformat(raw_generated_at)
        except Exception:
            generated_at = None

    raw_readiness = metadata.get("readiness")
    readiness: AvatarPersonalityTrainingReadinessRead
    if isinstance(raw_readiness, dict):
        try:
            readiness = AvatarPersonalityTrainingReadinessRead(**raw_readiness)
        except Exception:
            readiness = _build_avatar_personality_training_readiness(
                approved_count=int(metadata.get("approved_example_count") or personality.approved_example_count or 0),
                approved_gold_count=int(metadata.get("gold_example_count") or 0),
                approved_duration_seconds=float(metadata.get("approved_duration_seconds") or 0.0),
                approved_word_count=int(metadata.get("approved_word_count") or 0),
                needs_review_count=int(metadata.get("needs_review_count") or 0),
                duplicate_count=int(metadata.get("duplicate_example_count") or 0),
                hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
                largest_cluster_size=int(metadata.get("largest_cluster_size") or 0),
                total_example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
                auto_reject_count=int(metadata.get("auto_reject_count") or 0),
            )
    else:
        readiness = _build_avatar_personality_training_readiness(
            approved_count=int(metadata.get("approved_example_count") or personality.approved_example_count or 0),
            approved_gold_count=int(metadata.get("gold_example_count") or 0),
            approved_duration_seconds=float(metadata.get("approved_duration_seconds") or 0.0),
            approved_word_count=int(metadata.get("approved_word_count") or 0),
            needs_review_count=int(metadata.get("needs_review_count") or 0),
            duplicate_count=int(metadata.get("duplicate_example_count") or 0),
            hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
            largest_cluster_size=int(metadata.get("largest_cluster_size") or 0),
            total_example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
            auto_reject_count=int(metadata.get("auto_reject_count") or 0),
        )

    return AvatarPersonalityDatasetRead(
        avatar_id=int(avatar.id),
        speaker_id=int(avatar.speaker_id),
        status=str(personality.status or "draft"),
        system_prompt=personality.system_prompt,
        base_model_id=personality.base_model_id,
        dataset_path=str(dataset_path) if dataset_path.exists() else (personality.dataset_path or None),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
        gold_example_count=int(metadata.get("gold_example_count") or 0),
        silver_example_count=int(metadata.get("silver_example_count") or 0),
        auto_reject_count=int(metadata.get("auto_reject_count") or 0),
        needs_review_count=int(metadata.get("needs_review_count") or 0),
        duplicate_example_count=int(metadata.get("duplicate_example_count") or 0),
        cluster_count=int(metadata.get("cluster_count") or 0),
        hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
        llm_judged_count=int(metadata.get("llm_judged_count") or 0),
        llm_promoted_count=int(metadata.get("llm_promoted_count") or 0),
        llm_rejected_count=int(metadata.get("llm_rejected_count") or 0),
        source_turn_count=int(metadata.get("source_turn_count") or personality.source_turn_count or 0),
        discarded_turn_count=int(metadata.get("discarded_turn_count") or 0),
        readiness=readiness,
        preview_examples=preview_examples,
        generated_at=generated_at or personality.last_built_at,
    )


def _load_avatar_personality_state_map(avatar: Avatar) -> dict[int, str]:
    _, states_path = _avatar_personality_review_paths(avatar)
    if not states_path.exists():
        return {}
    try:
        raw = json.loads(states_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            example_id = int(key)
        except Exception:
            continue
        state = str(value or "").strip().lower()
        if state in {"approved", "rejected"}:
            out[example_id] = state
    return out


def _write_avatar_personality_state_map(avatar: Avatar, state_map: dict[int, str]) -> Path:
    _, states_path = _avatar_personality_review_paths(avatar)
    serializable = {str(int(key)): str(value) for key, value in sorted(state_map.items()) if value in {"approved", "rejected"}}
    states_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return states_path


def _iter_avatar_personality_review_examples(avatar: Avatar):
    review_path, _ = _avatar_personality_review_paths(avatar)
    if not review_path.exists():
        return
    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _refresh_avatar_personality_dataset_exports(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    *,
    total_example_count: int | None = None,
) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    state_map = _load_avatar_personality_state_map(avatar)
    approved_count = 0
    rejected_count = 0
    gold_count = 0
    silver_count = 0
    auto_reject_count = 0
    needs_review_count = 0
    duplicate_count = 0
    llm_judged_count = 0
    llm_promoted_count = 0
    llm_rejected_count = 0
    approved_gold_count = 0
    approved_duration_seconds = 0.0
    approved_word_count = 0
    largest_cluster_size = 0
    preview_examples: list[dict[str, object]] = []

    with dataset_path.open("w", encoding="utf-8") as export_handle:
        for row in _iter_avatar_personality_review_examples(avatar):
            try:
                example_id = int(row.get("example_id"))
            except Exception:
                continue
            auto_label = str(row.get("auto_label") or "silver").strip().lower()
            heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
            llm_label = row.get("llm_label")
            if auto_label == "gold":
                gold_count += 1
            elif auto_label == "reject":
                auto_reject_count += 1
            else:
                silver_count += 1
            if llm_label:
                llm_judged_count += 1
                normalized_llm_label = _avatar_normalize_llm_label(str(llm_label))
                if heuristic_label != "gold" and normalized_llm_label == "gold":
                    llm_promoted_count += 1
                if heuristic_label != "reject" and normalized_llm_label == "reject":
                    llm_rejected_count += 1

            state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
            if auto_label == "silver" and manual_state is None:
                needs_review_count += 1
            if _avatar_row_has_duplicate_risk(row):
                duplicate_count += 1
            if state == "rejected":
                rejected_count += 1
            else:
                approved_count += 1
                approved_duration_seconds += max(0.0, float(row.get("end_time") or 0.0) - float(row.get("start_time") or 0.0))
                approved_word_count += int(row.get("response_word_count") or 0)
                largest_cluster_size = max(largest_cluster_size, int(row.get("cluster_size") or 0))
                if auto_label == "gold":
                    approved_gold_count += 1
                messages = row.get("messages")
                if isinstance(messages, list):
                    export_handle.write(json.dumps({"messages": messages, "metadata": row.get("metadata", {})}, ensure_ascii=False) + "\n")
            if len(preview_examples) < 8:
                preview_examples.append(
                    {
                        "example_id": example_id,
                        "video_id": int(row.get("video_id") or 0),
                        "video_title": str(row.get("video_title") or ""),
                        "start_time": float(row.get("start_time") or 0.0),
                        "end_time": float(row.get("end_time") or 0.0),
                        "context_text": str(row.get("context_text") or ""),
                        "response_text": str(row.get("response_text") or ""),
                        "source_segment_ids": [int(segment_id) for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()],
                        "source_segment_count": int(row.get("source_segment_count") or 0),
                        "context_turns": int(row.get("context_turns") or 0),
                        "response_word_count": int(row.get("response_word_count") or 0),
                        "context_word_count": int(row.get("context_word_count") or 0),
                        "quality_score": int(row.get("quality_score") or 0),
                        "completion_score": int(row.get("completion_score") or 0),
                        "context_score": int(row.get("context_score") or 0),
                        "style_score": int(row.get("style_score") or 0),
                        "substance_score": int(row.get("substance_score") or 0),
                        "cluster_id": int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                        "cluster_size": int(row.get("cluster_size") or 0),
                        "diversity_score": int(row.get("diversity_score") or 0),
                        "duplicate_group_id": int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
                        "duplicate_group_size": int(row.get("duplicate_group_size") or 0),
                        "duplicate_similarity": float(row.get("duplicate_similarity") or 0.0),
                        "heuristic_label": heuristic_label,
                        "llm_label": _avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
                        "llm_confidence": int(row.get("llm_confidence") or 0),
                        "llm_completion_score": int(row.get("llm_completion_score") or 0),
                        "llm_style_score": int(row.get("llm_style_score") or 0),
                        "llm_usefulness_score": int(row.get("llm_usefulness_score") or 0),
                        "llm_rationale": (str(row.get("llm_rationale") or "").strip() or None),
                        "llm_reasons": [str(reason) for reason in row.get("llm_reasons", [])],
                        "llm_model": (str(row.get("llm_model") or "").strip() or None),
                        "llm_judged_at": row.get("llm_judged_at"),
                        "auto_label": auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
                        "manual_state": manual_state,
                        "state": state,
                        "reject_reasons": [str(reason) for reason in row.get("reject_reasons", [])],
                    }
                )

    preview_path.write_text(json.dumps(preview_examples, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata["example_count"] = int(total_example_count if total_example_count is not None else approved_count + rejected_count)
    metadata["approved_example_count"] = int(approved_count)
    metadata["rejected_example_count"] = int(rejected_count)
    metadata["gold_example_count"] = int(gold_count)
    metadata["silver_example_count"] = int(silver_count)
    metadata["auto_reject_count"] = int(auto_reject_count)
    metadata["needs_review_count"] = int(needs_review_count)
    metadata["duplicate_example_count"] = int(duplicate_count)
    metadata["llm_judged_count"] = int(llm_judged_count)
    metadata["llm_promoted_count"] = int(llm_promoted_count)
    metadata["llm_rejected_count"] = int(llm_rejected_count)
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)
    cluster_summary: dict[str, object] = {}
    if cluster_summary_path.exists():
        try:
            cluster_summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
        except Exception:
            cluster_summary = {}
    metadata["cluster_count"] = int(cluster_summary.get("cluster_count") or 0)
    metadata["hotspot_cluster_count"] = int(cluster_summary.get("hotspot_cluster_count") or 0)
    metadata["largest_cluster_size"] = int(max(largest_cluster_size, int(cluster_summary.get("largest_cluster_size") or 0)))
    metadata["approved_duration_seconds"] = round(float(approved_duration_seconds), 3)
    metadata["approved_word_count"] = int(approved_word_count)
    metadata["approved_gold_count"] = int(approved_gold_count)
    metadata["readiness"] = _build_avatar_personality_training_readiness(
        approved_count=approved_count,
        approved_gold_count=approved_gold_count,
        approved_duration_seconds=approved_duration_seconds,
        approved_word_count=approved_word_count,
        needs_review_count=needs_review_count,
        duplicate_count=duplicate_count,
        hotspot_cluster_count=int(cluster_summary.get("hotspot_cluster_count") or 0),
        largest_cluster_size=max(largest_cluster_size, int(cluster_summary.get("largest_cluster_size") or 0)),
        total_example_count=int(total_example_count if total_example_count is not None else approved_count + rejected_count),
        auto_reject_count=auto_reject_count,
    ).model_dump()
    metadata["generated_at"] = datetime.now().isoformat()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    personality.dataset_path = str(dataset_path)
    personality.dataset_example_count = int(metadata.get("example_count") or total_example_count or 0)
    personality.approved_example_count = int(approved_count)
    personality.status = "dataset_ready" if approved_count > 0 else "needs_review"
    personality.last_built_at = datetime.now()
    personality.updated_at = personality.last_built_at

    return _load_avatar_personality_dataset(avatar, personality)


def _avatar_training_effective_label(row: dict[str, object]) -> str:
    llm_label = str(row.get("llm_label") or "").strip()
    if llm_label:
        return _avatar_normalize_llm_label(llm_label)
    return _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))


def _avatar_training_priority(row: dict[str, object], *, manual_state: str | None = None) -> tuple[float, ...]:
    final_label = _avatar_training_effective_label(row)
    label_rank = 2.0 if final_label == "gold" else 1.0 if final_label == "silver" else 0.0
    manual_rank = 1.0 if manual_state == "approved" else 0.0
    quality = float(row.get("quality_score") or 0)
    substance = float(row.get("substance_score") or 0)
    style = float(row.get("style_score") or 0)
    diversity = float(row.get("diversity_score") or 0)
    word_count = float(row.get("response_word_count") or 0)
    return (manual_rank, label_rank, quality, style, substance, diversity, word_count)


def _avatar_cluster_round_robin_select(
    rows: list[dict[str, object]],
    *,
    limit: int = 0,
    manual_overrides: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    if not rows:
        return []
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            example_id = 0
        cluster_id = row.get("cluster_id")
        bucket_key = f"cluster:{int(cluster_id)}" if cluster_id is not None else f"solo:{example_id}"
        buckets.setdefault(bucket_key, []).append(row)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda item: _avatar_training_priority(
                item,
                manual_state=(manual_overrides or {}).get(int(item.get("example_id") or 0)),
            ),
            reverse=True,
        )
    ordered_keys = sorted(
        buckets.keys(),
        key=lambda key: _avatar_training_priority(
            buckets[key][0],
            manual_state=(manual_overrides or {}).get(int(buckets[key][0].get("example_id") or 0)),
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    while ordered_keys and (limit <= 0 or len(selected) < limit):
        next_keys: list[str] = []
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if limit > 0 and len(selected) >= limit:
                break
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def _avatar_select_training_conversation_examples(
    avatar: Avatar,
    config: AvatarPersonalityTrainingConfigRead,
) -> list[dict[str, object]]:
    state_map = _load_avatar_personality_state_map(avatar)
    approved_rows: list[dict[str, object]] = []
    manual_overrides: dict[int, str] = {}
    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            continue
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        if state != "approved":
            continue
        row_copy = dict(row)
        row_copy["final_label"] = _avatar_training_effective_label(row_copy)
        row_copy["manual_state"] = manual_state
        approved_rows.append(row_copy)
        if manual_state:
            manual_overrides[example_id] = manual_state

    gold_pool = [row for row in approved_rows if row.get("manual_state") == "approved" or row.get("final_label") == "gold"]
    silver_pool = [row for row in approved_rows if row not in gold_pool and row.get("final_label") == "silver"]
    full_pool = sorted(
        approved_rows,
        key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
        reverse=True,
    )

    if config.export_strategy == "full_approved":
        selected = list(full_pool)
    elif config.export_strategy == "gold_only":
        selected = sorted(
            gold_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )
    elif config.export_strategy == "gold_plus_top_silver":
        selected = sorted(
            gold_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )
        silver_budget = min(len(silver_pool), max(250, len(selected) // 2))
        silver_selected = sorted(
            silver_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )[:silver_budget]
        selected.extend(silver_selected)
    else:
        selected = _avatar_cluster_round_robin_select(gold_pool, manual_overrides=manual_overrides)
        silver_budget = min(len(silver_pool), max(150, len(selected) // 4), 1200)
        if silver_budget > 0:
            selected.extend(
                _avatar_cluster_round_robin_select(
                    silver_pool,
                    limit=silver_budget,
                    manual_overrides=manual_overrides,
                )
            )

    max_examples = max(0, int(config.max_examples or 0))
    if max_examples <= 0 and config.export_strategy != "full_approved":
        if config.export_strategy == "gold_only":
            max_examples = 4000
        elif config.export_strategy == "gold_plus_top_silver":
            max_examples = 6000
        else:
            max_examples = 5000
    if max_examples > 0 and len(selected) > max_examples:
        selected = _avatar_cluster_round_robin_select(
            selected,
            limit=max_examples,
            manual_overrides=manual_overrides,
        )
    return selected


def _avatar_estimate_example_tokens(item: dict[str, object]) -> int:
    messages = item.get("messages")
    if not isinstance(messages, list):
        return 0
    total_chars = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, dict))
    return int(total_chars / 3.5) + len(messages) * 4


def _avatar_filter_oversized_examples(
    items: list[dict[str, object]],
    *,
    max_seq_length: int,
) -> list[dict[str, object]]:
    # NOTE: This is a coarse heuristic (chars/3.5), not tokenizer-aware.
    # Chat templates (especially Qwen's) add framing tokens that can push
    # the true length above this estimate.  The 20% headroom compensates,
    # but some borderline examples may still get left-truncated during
    # training.  A tokenizer-aware pass would require loading the model
    # tokenizer at package-preparation time.
    headroom = max(64, int(max_seq_length * 0.20))
    threshold = max_seq_length - headroom
    return [item for item in items if _avatar_estimate_example_tokens(item) <= threshold]


def _avatar_cap_per_video_contribution(
    items: list[dict[str, object]],
    *,
    max_ratio: float = 0.15,
) -> list[dict[str, object]]:
    if not items or max_ratio >= 1.0:
        return items
    total = len(items)
    max_per_video = max(5, int(total * max_ratio))
    video_counts: dict[int, int] = {}
    result: list[dict[str, object]] = []
    for item in items:
        video_id = int((item.get("metadata") or {}).get("video_id") or item.get("video_id") or 0)
        current_count = video_counts.get(video_id, 0)
        if current_count < max_per_video:
            result.append(item)
            video_counts[video_id] = current_count + 1
    return result


_LONG_FORM_USER_PROMPTS = [
    "Talk about this topic in your own words.",
    "Share your thoughts on this at length.",
    "Give your take on this — speak naturally.",
    "Go into detail on this subject.",
    "Walk me through your perspective here.",
    "Tell me what you think about this.",
    "Break this down in your own style.",
    "Speak freely about this topic.",
    "What are your thoughts? Take your time.",
    "Explain this the way you normally would.",
]


def _avatar_build_long_form_training_messages(prompt: str, sample: dict[str, object]) -> list[dict[str, str]]:
    episode_title = str(sample.get("video_title") or "").strip()
    sample_id = str(sample.get("sample_id") or sample.get("video_id") or "")
    variant_index = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) % len(_LONG_FORM_USER_PROMPTS)
    user_prompt = _LONG_FORM_USER_PROMPTS[variant_index]
    if episode_title:
        user_prompt += f" Episode: {episode_title}."
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(sample.get("text") or "").strip()},
    ]


def _avatar_select_training_long_form_examples(
    session: Session,
    avatar: Avatar,
    *,
    include_long_form: bool,
    prompt: str,
    max_examples: int | None = None,
) -> list[dict[str, object]]:
    if not include_long_form:
        return []
    config = _load_avatar_personality_long_form_config(avatar)
    take_count = max(0, int(config.take_count or 0))
    if max_examples is not None:
        take_count = min(take_count, max(0, int(max_examples)))
    if take_count <= 0:
        return []
    samples = _build_avatar_personality_long_form_samples(session, avatar)
    states = _load_avatar_personality_long_form_states(avatar)
    selected_samples: list[dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        if states.get(sample_id, "included") == "rejected":
            continue
        selected_samples.append(sample)
        if len(selected_samples) >= take_count:
            break
    output: list[dict[str, object]] = []
    for sample in selected_samples:
        output.append(
            {
                "source_kind": "long_form",
                "source_id": str(sample.get("sample_id") or ""),
                "messages": _avatar_build_long_form_training_messages(prompt, sample),
                "metadata": {
                    "source_kind": "long_form",
                    "video_id": int(sample.get("video_id") or 0),
                    "video_title": str(sample.get("video_title") or ""),
                    "start_time": float(sample.get("start_time") or 0.0),
                    "end_time": float(sample.get("end_time") or 0.0),
                    "duration_seconds": float(sample.get("duration_seconds") or 0.0),
                    "word_count": int(sample.get("word_count") or 0),
                    "segment_count": int(sample.get("segment_count") or 0),
                },
            }
        )
    return output


def _avatar_split_training_examples(
    items: list[dict[str, object]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not items:
        return [], []
    safe_ratio = max(0.01, min(0.2, float(validation_ratio or 0.02)))
    train_items: list[dict[str, object]] = []
    val_items: list[dict[str, object]] = []
    for item in items:
        source_key = f"{item.get('source_kind')}|{item.get('source_id')}"
        bucket = int(hashlib.md5(source_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
        if bucket < int(safe_ratio * 10000):
            val_items.append(item)
        else:
            train_items.append(item)
    if not train_items and val_items:
        train_items.append(val_items.pop())
    if not val_items and len(train_items) > 20:
        val_items.append(train_items.pop())
    return train_items, val_items


def _prepare_avatar_personality_training_package(
    session: Session,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityTrainingPackageRead:
    config_path, manifest_path, train_path, val_path = _avatar_personality_training_paths(avatar)
    config = _load_avatar_personality_training_config(avatar)
    prompt = str(personality.system_prompt or _default_avatar_personality_prompt(avatar.name)).strip()
    base_model_id = str(config.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B").strip()

    conversation_rows = _avatar_select_training_conversation_examples(avatar, config)
    conversation_items: list[dict[str, object]] = []
    for row in conversation_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        conversation_items.append(
            {
                "source_kind": "conversation",
                "source_id": int(row.get("example_id") or 0),
                "messages": messages,
                "metadata": {
                    "source_kind": "conversation",
                    "example_id": int(row.get("example_id") or 0),
                    "video_id": int(row.get("video_id") or 0),
                    "video_title": str(row.get("video_title") or ""),
                    "start_time": float(row.get("start_time") or 0.0),
                    "end_time": float(row.get("end_time") or 0.0),
                    "quality_score": int(row.get("quality_score") or 0),
                    "style_score": int(row.get("style_score") or 0),
                    "cluster_id": int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                    "final_label": str(row.get("final_label") or "silver"),
                    "manual_state": row.get("manual_state"),
                },
            }
        )

    long_form_items = _avatar_select_training_long_form_examples(
        session,
        avatar,
        include_long_form=config.include_long_form,
        prompt=prompt,
        max_examples=config.max_long_form_examples,
    )
    all_items = conversation_items + long_form_items
    launch_settings = _recommend_avatar_training_launch_settings(
        model_id=base_model_id,
        training_mode=str(config.training_mode or "memory_optimized"),
        requested_lora_rank=None,
        requested_max_seq_length=None,
        requested_per_device_batch_size=None,
        requested_gradient_accumulation_steps=None,
    )
    effective_max_seq = int(launch_settings.get("max_seq_length") or 1024)
    all_items = _avatar_filter_oversized_examples(all_items, max_seq_length=effective_max_seq)
    all_items = _avatar_cap_per_video_contribution(all_items, max_ratio=0.15)
    train_items, val_items = _avatar_split_training_examples(all_items, validation_ratio=config.validation_ratio)
    training_plan = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        selected_conversation_examples=len(conversation_items),
        selected_long_form_examples=len(long_form_items),
        train_examples=len(train_items),
        validation_examples=len(val_items),
        epochs=1,
    )

    with train_path.open("w", encoding="utf-8") as handle:
        for item in train_items:
            handle.write(json.dumps({"messages": item["messages"], "metadata": item["metadata"]}, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as handle:
        for item in val_items:
            handle.write(json.dumps({"messages": item["messages"], "metadata": item["metadata"]}, ensure_ascii=False) + "\n")

    _write_avatar_personality_training_config(
        avatar,
        base_model_id=base_model_id,
        dataset_profile=config.dataset_profile,
        training_strength=config.training_strength,
        export_strategy=config.export_strategy,
        validation_ratio=config.validation_ratio,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
        include_long_form=config.include_long_form,
        snapshot_interval_steps=config.snapshot_interval_steps,
    )
    prepared_at = datetime.now()
    manifest = AvatarPersonalityTrainingPackageRead(
        avatar_id=int(avatar.id),
        status="ready",
        base_model_id=base_model_id,
        dataset_profile=config.dataset_profile,
        training_strength=config.training_strength,
        export_strategy=config.export_strategy,
        validation_ratio=config.validation_ratio,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
        include_long_form=config.include_long_form,
        conversation_examples_selected=len(conversation_items),
        long_form_examples_selected=len(long_form_items),
        total_examples_selected=len(all_items),
        train_examples=len(train_items),
        validation_examples=len(val_items),
        prompt=prompt,
        manifest_path=str(manifest_path),
        config_path=str(config_path),
        train_dataset_path=str(train_path),
        validation_dataset_path=str(val_path),
        command_preview=f"python backend/tools/train_avatar_personality.py --manifest \"{manifest_path}\"",
        prepared_at=prepared_at,
        training_plan=training_plan,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    personality.status = "training_ready" if len(all_items) > 0 else personality.status
    personality.updated_at = prepared_at
    return manifest


def _start_avatar_personality_training(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    body: AvatarPersonalityTrainRequest,
) -> AvatarPersonalityTrainingStatusRead:
    persisted = _reconcile_avatar_personality_training_runtime(avatar)
    if persisted.active and persisted.status in {"queued", "running", "stopping"}:
        return persisted
    with _avatar_training_runs_lock:
        existing = _avatar_training_processes.get(int(avatar.id))
        if existing and existing.poll() is None:
            return _reconcile_avatar_personality_training_runtime(avatar)

    manifest = _read_avatar_personality_training_package(avatar)
    if manifest.status != "ready" or not manifest.manifest_path:
        raise HTTPException(status_code=400, detail="Prepare the training package first.")
    manifest_path = Path(manifest.manifest_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="Training manifest is missing. Prepare the package again.")
    model_installed, model_path = _hf_model_is_installed(str(manifest.base_model_id or ""))
    if not model_installed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Base model {manifest.base_model_id} is not installed in the local Hugging Face cache. "
                "Download the selected base model before starting training."
            ),
        )

    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    status_path, stop_path = _avatar_personality_training_runtime_paths(avatar)
    config = _load_avatar_personality_training_config(avatar)
    _clear_avatar_personality_training_stop_flag(avatar)
    training_mode = str(body.training_mode or "memory_optimized").strip().lower()
    if training_mode not in {"standard", "memory_optimized"}:
        training_mode = "memory_optimized"
    if training_mode == "memory_optimized":
        memory_optimized_available, memory_optimized_reason = _detect_avatar_memory_optimized_support()
        if not memory_optimized_available:
            raise HTTPException(
                status_code=400,
                detail=memory_optimized_reason or "Memory-optimized QLoRA mode is not available in the backend environment.",
            )

    training_runs_dir = _avatar_artifacts_dir(avatar) / "personality" / "training_runs"
    training_runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = training_runs_dir / run_name
    run_suffix = 1
    while output_dir.exists():
        output_dir = training_runs_dir / f"{run_name}-{run_suffix:02d}"
        run_suffix += 1
    if bool(body.overwrite_output) and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    personality.base_model_id = str(manifest.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B")
    effective_settings = _recommend_avatar_training_launch_settings(
        model_id=personality.base_model_id,
        training_mode=training_mode,
        requested_lora_rank=body.lora_rank,
        requested_max_seq_length=body.max_seq_length,
        requested_per_device_batch_size=body.per_device_batch_size,
        requested_gradient_accumulation_steps=body.gradient_accumulation_steps,
    )
    personality.status = "training_queued"
    personality.updated_at = datetime.now()

    command = [
        str(Path(sys.executable)),
        str(Path(__file__).parent.parent / "tools" / "run_avatar_personality_training.py"),
        "--manifest",
        str(manifest_path),
        "--status-path",
        str(status_path),
        "--stop-path",
        str(stop_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(max(1, int(body.epochs or 1))),
        "--learning-rate",
        str(float(body.learning_rate or 5e-5)),
        "--lora-rank",
        str(int(effective_settings["lora_rank"])),
        "--max-seq-length",
        str(int(effective_settings["max_seq_length"])),
        "--per-device-batch-size",
        str(int(effective_settings["per_device_batch_size"])),
        "--gradient-accumulation-steps",
        str(int(effective_settings["gradient_accumulation_steps"])),
        "--warmup-ratio",
        str(max(0.0, min(0.2, float(body.warmup_ratio or 0.03)))),
        "--snapshot-interval-steps",
        str(max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps))),
        "--training-mode",
        training_mode,
        "--cuda-memory-fraction",
        str(float(effective_settings["cuda_memory_fraction"])),
    ]

    _write_avatar_personality_training_status(
        avatar,
        {
            "status": "queued",
            "active": True,
            "stop_requested": False,
            "process_id": None,
            "base_model_id": personality.base_model_id,
            "training_mode": training_mode,
            "adapter_path": None,
            "output_dir": str(output_dir),
            "current_stage": "queued",
            "epoch": 0.0,
            "step": 0,
            "max_steps": 0,
            "snapshot_interval_steps": max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps)),
            "train_examples": int(manifest.train_examples or 0),
            "validation_examples": int(manifest.validation_examples or 0),
            "latest_loss": None,
            "snapshots": [],
            "started_at": datetime.now(),
            "updated_at": datetime.now(),
            "finished_at": None,
            "error": None,
            "message": (
                f"Launching trainer subprocess with auto-tuned settings: "
                f"seq {int(effective_settings['max_seq_length'])}, "
                f"rank {int(effective_settings['lora_rank'])}, "
                f"batch {int(effective_settings['per_device_batch_size'])}, "
                f"grad {int(effective_settings['gradient_accumulation_steps'])}. "
                f"Snapshot cadence: "
                f"{max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps)) or 'auto'}. "
                f"Local model: {model_path or manifest.base_model_id}. "
                f"{str(effective_settings['rationale'])}"
            ),
        },
    )

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).parent.parent),
        creationflags=creationflags,
    )
    _write_avatar_personality_training_status(
        avatar,
        {
            "process_id": int(process.pid),
        },
    )
    with _avatar_training_runs_lock:
        _avatar_training_processes[int(avatar.id)] = process

    def _watch_training() -> None:
        return_code = process.wait()
        with _avatar_training_runs_lock:
            current = _avatar_training_processes.get(int(avatar.id))
            if current is process:
                _avatar_training_processes.pop(int(avatar.id), None)
        _sync_avatar_personality_training_completion(int(avatar.id))
        _avatar_release_cached_chat_model(int(avatar.id))
        if return_code != 0:
            try:
                status = _load_avatar_personality_training_status(avatar)
                if status.status not in {"failed", "stopped"}:
                    _write_avatar_personality_training_status(
                        avatar,
                        {
                            "status": "failed",
                            "active": False,
                            "process_id": int(process.pid),
                            "current_stage": "failed",
                            "finished_at": datetime.now(),
                            "updated_at": datetime.now(),
                            "message": f"Trainer exited with code {return_code}",
                            "error": status.error or f"Trainer exited with code {return_code}",
                        },
                    )
            except Exception:
                pass

    threading.Thread(target=_watch_training, daemon=True, name=f"avatar-train-watch-{avatar.id}").start()
    return _load_avatar_personality_training_status(avatar)


def _read_avatar_personality_dataset_page(
    avatar: Avatar,
    *,
    offset: int,
    limit: int,
    state_filter: str,
) -> AvatarPersonalityDatasetPageRead:
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(int(limit or 20), 100))
    normalized_filter = str(state_filter or "all").strip().lower()
    if normalized_filter not in {"all", "approved", "rejected", "gold", "silver", "auto_reject", "needs_review", "duplicate_risk"}:
        normalized_filter = "all"

    state_map = _load_avatar_personality_state_map(avatar)
    approved_count = 0
    rejected_count = 0
    gold_count = 0
    silver_count = 0
    auto_reject_count = 0
    needs_review_count = 0
    duplicate_count = 0
    llm_judged_count = 0
    llm_promoted_count = 0
    llm_rejected_count = 0
    matching_total = 0
    items: list[AvatarPersonalityDatasetExampleRead] = []
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)
    cluster_summary: dict[str, object] = {}
    if cluster_summary_path.exists():
        try:
            cluster_summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
        except Exception:
            cluster_summary = {}

    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            continue
        auto_label = str(row.get("auto_label") or "silver").strip().lower()
        heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
        llm_label = row.get("llm_label")
        if auto_label == "gold":
            gold_count += 1
        elif auto_label == "reject":
            auto_reject_count += 1
        else:
            silver_count += 1
        if llm_label:
            llm_judged_count += 1
            normalized_llm_label = _avatar_normalize_llm_label(str(llm_label))
            if heuristic_label != "gold" and normalized_llm_label == "gold":
                llm_promoted_count += 1
            if heuristic_label != "reject" and normalized_llm_label == "reject":
                llm_rejected_count += 1
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        if state == "rejected":
            rejected_count += 1
        else:
            approved_count += 1
        if auto_label == "silver" and manual_state is None:
            needs_review_count += 1
        is_duplicate_risk = _avatar_row_has_duplicate_risk(row)
        if is_duplicate_risk:
            duplicate_count += 1

        matches_filter = normalized_filter == "all"
        if normalized_filter == "approved":
            matches_filter = state == "approved"
        elif normalized_filter == "rejected":
            matches_filter = state == "rejected"
        elif normalized_filter == "gold":
            matches_filter = auto_label == "gold"
        elif normalized_filter == "silver":
            matches_filter = auto_label == "silver"
        elif normalized_filter == "auto_reject":
            matches_filter = auto_label == "reject"
        elif normalized_filter == "needs_review":
            matches_filter = auto_label == "silver" and manual_state is None
        elif normalized_filter == "duplicate_risk":
            matches_filter = is_duplicate_risk
        if not matches_filter:
            continue
        if matching_total >= safe_offset and len(items) < safe_limit:
            items.append(
                AvatarPersonalityDatasetExampleRead(
                    example_id=example_id,
                    video_id=int(row.get("video_id") or 0),
                    video_title=str(row.get("video_title") or ""),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    context_text=str(row.get("context_text") or ""),
                    response_text=str(row.get("response_text") or ""),
                    source_segment_ids=[int(segment_id) for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()],
                    source_segment_count=int(row.get("source_segment_count") or 0),
                    context_turns=int(row.get("context_turns") or 0),
                    response_word_count=int(row.get("response_word_count") or 0),
                    context_word_count=int(row.get("context_word_count") or 0),
                    quality_score=int(row.get("quality_score") or 0),
                    completion_score=int(row.get("completion_score") or 0),
                    context_score=int(row.get("context_score") or 0),
                    style_score=int(row.get("style_score") or 0),
                    cluster_id=int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                    cluster_size=int(row.get("cluster_size") or 0),
                    diversity_score=int(row.get("diversity_score") or 0),
                    duplicate_group_id=int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
                    duplicate_group_size=int(row.get("duplicate_group_size") or 0),
                    duplicate_similarity=float(row.get("duplicate_similarity") or 0.0),
                    heuristic_label=heuristic_label,
                    llm_label=_avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
                    llm_confidence=int(row.get("llm_confidence") or 0),
                    llm_completion_score=int(row.get("llm_completion_score") or 0),
                    llm_style_score=int(row.get("llm_style_score") or 0),
                    llm_usefulness_score=int(row.get("llm_usefulness_score") or 0),
                    llm_rationale=(str(row.get("llm_rationale") or "").strip() or None),
                    llm_reasons=[str(reason) for reason in row.get("llm_reasons", [])],
                    llm_model=(str(row.get("llm_model") or "").strip() or None),
                    llm_judged_at=row.get("llm_judged_at"),
                    auto_label=auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
                    manual_state=manual_state,
                    state=state,
                    reject_reasons=[str(reason) for reason in row.get("reject_reasons", [])],
                )
            )
        matching_total += 1

    return AvatarPersonalityDatasetPageRead(
        avatar_id=int(avatar.id),
        total=matching_total,
        approved_count=approved_count,
        rejected_count=rejected_count,
        gold_count=gold_count,
        silver_count=silver_count,
        auto_reject_count=auto_reject_count,
        needs_review_count=needs_review_count,
        duplicate_count=duplicate_count,
        cluster_count=int(cluster_summary.get("cluster_count") or 0),
        hotspot_cluster_count=int(cluster_summary.get("hotspot_cluster_count") or 0),
        llm_judged_count=llm_judged_count,
        llm_promoted_count=llm_promoted_count,
        llm_rejected_count=llm_rejected_count,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + safe_limit) < matching_total,
        state_filter=normalized_filter,
        items=items,
    )


def _row_to_example_read(row: dict, state: str, manual_state: Optional[str]) -> "AvatarPersonalityDatasetExampleRead":
    """Convert a raw review-file row dict to AvatarPersonalityDatasetExampleRead."""
    auto_label = str(row.get("auto_label") or "silver").strip().lower()
    heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
    llm_label = row.get("llm_label")
    return AvatarPersonalityDatasetExampleRead(
        example_id=int(row.get("example_id")),
        video_id=int(row.get("video_id") or 0),
        video_title=str(row.get("video_title") or ""),
        start_time=float(row.get("start_time") or 0.0),
        end_time=float(row.get("end_time") or 0.0),
        context_text=str(row.get("context_text") or ""),
        response_text=str(row.get("response_text") or ""),
        source_segment_ids=[int(s) for s in row.get("source_segment_ids", []) if str(s).isdigit() or isinstance(s, int)],
        source_segment_count=int(row.get("source_segment_count") or 0),
        context_turns=int(row.get("context_turns") or 0),
        response_word_count=int(row.get("response_word_count") or 0),
        context_word_count=int(row.get("context_word_count") or 0),
        quality_score=int(row.get("quality_score") or 0),
        completion_score=int(row.get("completion_score") or 0),
        context_score=int(row.get("context_score") or 0),
        style_score=int(row.get("style_score") or 0),
        substance_score=int(row.get("substance_score") or 0),
        cluster_id=int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
        cluster_size=int(row.get("cluster_size") or 0),
        diversity_score=int(row.get("diversity_score") or 0),
        duplicate_group_id=int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
        duplicate_group_size=int(row.get("duplicate_group_size") or 0),
        duplicate_similarity=float(row.get("duplicate_similarity") or 0.0),
        heuristic_label=heuristic_label,
        llm_label=_avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
        llm_confidence=int(row.get("llm_confidence") or 0),
        llm_completion_score=int(row.get("llm_completion_score") or 0),
        llm_style_score=int(row.get("llm_style_score") or 0),
        llm_usefulness_score=int(row.get("llm_usefulness_score") or 0),
        llm_rationale=(str(row.get("llm_rationale") or "").strip() or None),
        llm_reasons=[str(r) for r in row.get("llm_reasons", [])],
        llm_model=(str(row.get("llm_model") or "").strip() or None),
        llm_judged_at=row.get("llm_judged_at"),
        auto_label=auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
        manual_state=manual_state,
        state=state,
        reject_reasons=[str(r) for r in row.get("reject_reasons", [])],
    )


def _build_avatar_personality_dataset(
    session: Session,
    avatar: Avatar,
    speaker: Speaker,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    review_path, _ = _avatar_personality_review_paths(avatar)
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)

    target_video_ids = [
        int(video_id)
        for video_id in session.exec(
            select(TranscriptSegment.video_id)
            .where(TranscriptSegment.speaker_id == speaker.id)
            .distinct()
        ).all()
        if video_id is not None
    ]
    if not target_video_ids:
        personality.dataset_path = None
        personality.dataset_example_count = 0
        personality.approved_example_count = 0
        personality.source_turn_count = 0
        personality.status = "needs_source"
        personality.last_built_at = datetime.now()
        personality.updated_at = personality.last_built_at
        session.add(personality)
        session.commit()
        return _load_avatar_personality_dataset(avatar, personality)

    rows = session.exec(
        select(
            TranscriptSegment.id,
            TranscriptSegment.video_id,
            TranscriptSegment.speaker_id,
            TranscriptSegment.start_time,
            TranscriptSegment.end_time,
            TranscriptSegment.text,
            Video.title,
            Speaker.name,
        )
        .join(Video, TranscriptSegment.video_id == Video.id)
        .join(Speaker, TranscriptSegment.speaker_id == Speaker.id, isouter=True)
        .where(
            Video.channel_id == avatar.channel_id,
            TranscriptSegment.video_id.in_(target_video_ids),
            TranscriptSegment.speaker_id.is_not(None),
        )
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_time, TranscriptSegment.id)
    ).all()

    turns_by_video: dict[int, list[dict[str, object]]] = {}
    merge_gap_seconds = 2.0
    source_turn_count = 0
    discarded_turn_count = 0

    for segment_id, video_id, segment_speaker_id, start_time, end_time, text_value, video_title, speaker_name in rows:
        cleaned_text = _clean_avatar_dataset_text(text_value)
        if not cleaned_text:
            continue

        turn_list = turns_by_video.setdefault(int(video_id), [])
        turn_speaker_id = int(segment_speaker_id) if segment_speaker_id is not None else None
        if (
            turn_list
            and int(turn_list[-1]["speaker_id"]) == int(turn_speaker_id or -1)
            and float(start_time or 0.0) - float(turn_list[-1]["end_time"] or 0.0) <= merge_gap_seconds
        ):
            turn_list[-1]["text"] = f"{turn_list[-1]['text']} {cleaned_text}".strip()
            turn_list[-1]["end_time"] = float(end_time or start_time or 0.0)
            turn_list[-1]["source_segment_ids"].append(int(segment_id))
        else:
            turn_list.append(
                {
                    "speaker_id": int(turn_speaker_id or 0),
                    "speaker_name": _clean_avatar_dataset_text(speaker_name) or f"Speaker {turn_speaker_id}",
                    "video_id": int(video_id),
                    "video_title": _clean_avatar_dataset_text(video_title) or f"Video {video_id}",
                    "start_time": float(start_time or 0.0),
                    "end_time": float(end_time or start_time or 0.0),
                    "text": cleaned_text,
                    "source_segment_ids": [int(segment_id)],
                }
            )

    speaker_style_weights = _avatar_build_speaker_style_weights(turns_by_video, int(speaker.id))

    legacy_prompt = _default_avatar_personality_prompt(avatar.name)
    current_prompt = str(personality.system_prompt or "").strip()
    if not current_prompt or current_prompt == legacy_prompt:
        current_prompt = _default_avatar_personality_prompt(speaker.name)
        personality.system_prompt = current_prompt
    system_prompt = current_prompt

    review_rows: list[dict[str, object]] = []
    example_count = 0

    for turns in turns_by_video.values():
        consumed_turn_indexes: set[int] = set()
        for idx, turn in enumerate(turns):
            if idx in consumed_turn_indexes:
                continue
            if int(turn["speaker_id"]) != int(speaker.id):
                continue
            source_turn_count += 1
            extended_response = _extend_avatar_response_turn(turns, idx, target_speaker_id=int(speaker.id))
            response_text = _clean_avatar_dataset_text(str(extended_response.get("text") or ""))
            consumed_turn_indexes.update(int(turn_index) for turn_index in extended_response.get("consumed_indexes", [idx]) if isinstance(turn_index, int))
            if len(response_text) < 24 or len(response_text.split()) < 5:
                discarded_turn_count += 1
                continue

            raw_context = [ctx for ctx in turns[max(0, idx - 4): idx]]
            if not raw_context or not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in raw_context):
                discarded_turn_count += 1
                continue

            # Select the last 3 context turns, but guarantee at least one
            # other-speaker turn survives the trim to avoid monologue context.
            trimmed = raw_context[-3:]
            if not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in trimmed):
                other_turns = [ctx for ctx in raw_context if int(ctx["speaker_id"]) != int(speaker.id)]
                trimmed = [other_turns[-1]] + [ctx for ctx in trimmed if int(ctx["speaker_id"]) == int(speaker.id)][:2]

            context_lines = [
                f"{str(ctx.get('speaker_name') or 'Speaker').strip()}: {str(ctx.get('text') or '').strip()}"
                for ctx in trimmed
                if str(ctx.get("text") or "").strip()
            ]
            if not context_lines or not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in trimmed):
                discarded_turn_count += 1
                continue

            user_text = (
                f"Podcast episode: {str(turn.get('video_title') or '').strip()}\n"
                f"Conversation context:\n" + "\n".join(context_lines)
            ).strip()

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response_text},
                ],
                "metadata": {
                    "avatar_id": int(avatar.id),
                    "speaker_id": int(speaker.id),
                    "video_id": int(turn["video_id"]),
                    "video_title": str(turn.get("video_title") or ""),
                    "start_time": float(turn["start_time"]),
                    "end_time": float(extended_response.get("end_time") or turn["end_time"]),
                    "context_turns": len(context_lines),
                    "source_segment_ids": [int(segment_id) for segment_id in extended_response.get("source_segment_ids", [])],
                },
            }
            review_rows.append(
                _score_avatar_personality_example(
                    {
                        "example_id": int(example_count),
                        "video_id": int(turn["video_id"]),
                        "video_title": str(turn.get("video_title") or ""),
                        "start_time": float(turn["start_time"]),
                        "end_time": float(extended_response.get("end_time") or turn["end_time"]),
                        "context_text": user_text,
                        "response_text": response_text,
                        "source_segment_ids": [int(segment_id) for segment_id in extended_response.get("source_segment_ids", [])],
                        "context_turns": len(context_lines),
                        "messages": payload["messages"],
                        "metadata": payload["metadata"],
                    },
                    speaker_style_weights=speaker_style_weights,
                )
            )
            example_count += 1

    review_rows = _apply_avatar_duplicate_rejects(review_rows)
    review_rows, cluster_summary = _avatar_assign_embedding_clusters(review_rows)
    review_lines = [json.dumps(row, ensure_ascii=False) for row in review_rows]
    review_path.write_text("\n".join(review_lines), encoding="utf-8")
    cluster_summary_path.write_text(json.dumps(cluster_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_avatar_personality_state_map(avatar, {})
    dataset_path.write_text("", encoding="utf-8")
    preview_path.write_text("[]", encoding="utf-8")

    generated_at = datetime.now()
    label_counts = Counter(str(row.get("auto_label") or "silver") for row in review_rows)
    metadata = {
        "avatar_id": int(avatar.id),
        "speaker_id": int(speaker.id),
        "speaker_name": speaker.name,
        "example_count": int(example_count),
        "approved_example_count": int(example_count - int(label_counts.get("reject", 0))),
        "rejected_example_count": int(label_counts.get("reject", 0)),
        "gold_example_count": int(label_counts.get("gold", 0)),
        "silver_example_count": int(label_counts.get("silver", 0)),
        "auto_reject_count": int(label_counts.get("reject", 0)),
        "needs_review_count": int(label_counts.get("silver", 0)),
        "duplicate_example_count": int(cluster_summary.get("duplicate_example_count") or 0),
        "cluster_count": int(cluster_summary.get("cluster_count") or 0),
        "hotspot_cluster_count": int(cluster_summary.get("hotspot_cluster_count") or 0),
        "llm_judged_count": 0,
        "llm_promoted_count": 0,
        "llm_rejected_count": 0,
        "source_turn_count": int(source_turn_count),
        "discarded_turn_count": int(discarded_turn_count),
        "generated_at": generated_at.isoformat(),
        "dataset_format": "sharegpt_messages_v1",
        "base_model_id": personality.base_model_id,
        "system_prompt": system_prompt,
        "embedding_model": str(cluster_summary.get("embedding_model") or "hashing_ngram_v1"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    personality.dataset_path = str(dataset_path)
    personality.dataset_example_count = int(example_count)
    personality.approved_example_count = int(example_count - int(label_counts.get("reject", 0)))
    personality.source_turn_count = int(source_turn_count)
    personality.status = "dataset_ready" if example_count > 0 else "needs_source"
    personality.last_built_at = generated_at
    personality.updated_at = generated_at
    session.add(personality)
    session.commit()
    session.refresh(personality)

    refreshed_dataset = _refresh_avatar_personality_dataset_exports(
        avatar,
        personality,
        total_example_count=example_count,
    )
    refreshed_dataset.generated_at = generated_at
    refreshed_dataset.discarded_turn_count = int(discarded_turn_count)
    refreshed_dataset.source_turn_count = int(source_turn_count)
    return refreshed_dataset


def _serialize_avatar(avatar: Avatar) -> AvatarRead:
    return AvatarRead(
        id=int(avatar.id),
        channel_id=int(avatar.channel_id),
        speaker_id=int(avatar.speaker_id),
        name=str(avatar.name or "").strip(),
        status=str(avatar.status or "draft").strip() or "draft",
        description=(str(avatar.description).strip() if avatar.description is not None else None),
        created_at=avatar.created_at,
        updated_at=avatar.updated_at,
    )


def _ensure_avatar_profiles(
    session: Session,
    avatar: Avatar,
    *,
    speaker_name: str | None = None,
) -> tuple[AvatarPersonalityProfile, AvatarAppearanceProfile, AvatarVoiceProfile]:
    personality = session.exec(
        select(AvatarPersonalityProfile).where(AvatarPersonalityProfile.avatar_id == avatar.id)
    ).first()
    if not personality:
        personality = AvatarPersonalityProfile(
            avatar_id=int(avatar.id),
            system_prompt=_default_avatar_personality_prompt(speaker_name or avatar.name),
            base_model_id="Qwen/Qwen3-14B",
        )
        session.add(personality)

    appearance = session.exec(
        select(AvatarAppearanceProfile).where(AvatarAppearanceProfile.avatar_id == avatar.id)
    ).first()
    if not appearance:
        appearance = AvatarAppearanceProfile(avatar_id=int(avatar.id))
        session.add(appearance)

    voice = session.exec(
        select(AvatarVoiceProfile).where(AvatarVoiceProfile.avatar_id == avatar.id)
    ).first()
    if not voice:
        voice = AvatarVoiceProfile(avatar_id=int(avatar.id), provider="fish-speech")
        session.add(voice)

    session.flush()
    return personality, appearance, voice


def _build_avatar_workbench(session: Session, avatar: Avatar) -> AvatarWorkbenchRead:
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")

    personality, appearance, voice = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    legacy_prompt = _default_avatar_personality_prompt(avatar.name)
    normalized_prompt = _default_avatar_personality_prompt(speaker.name)
    if str(personality.system_prompt or "").strip() == legacy_prompt:
        personality.system_prompt = normalized_prompt

    total_speaking_time = float(
        session.exec(
            select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
            .where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0.0
    )
    embedding_count = int(
        session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker.id)
        ).first()
        or 0
    )
    appearance_count = int(
        session.exec(
            select(func.count(func.distinct(TranscriptSegment.video_id))).where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0
    )
    source_turn_count = int(
        session.exec(
            select(func.count(TranscriptSegment.id)).where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0
    )
    eligible_voice_clip_count = int(
        session.exec(
            select(func.count(TranscriptSegment.id)).where(
                TranscriptSegment.speaker_id == speaker.id,
                (TranscriptSegment.end_time - TranscriptSegment.start_time) >= 2.0,
            )
        ).first()
        or 0
    )

    personality.source_turn_count = max(int(personality.source_turn_count or 0), source_turn_count)
    if not str(personality.status or "").strip() or str(personality.status) == "draft":
        personality.status = "ready_to_curate" if source_turn_count > 0 else "needs_source"
    appearance.source_image_count = max(int(appearance.source_image_count or 0), 1 if speaker.thumbnail_path else 0)
    appearance.approved_image_count = max(int(appearance.approved_image_count or 0), 1 if appearance.primary_image_path or speaker.thumbnail_path else 0)
    if not str(appearance.status or "").strip() or str(appearance.status) == "draft":
        appearance.status = "ready_to_curate" if (appearance.source_image_count or 0) > 0 else "needs_source"
    voice.source_clip_count = max(int(voice.source_clip_count or 0), max(eligible_voice_clip_count, embedding_count))
    voice.approved_clip_count = max(int(voice.approved_clip_count or 0), embedding_count)
    if not str(voice.status or "").strip() or str(voice.status) == "draft":
        voice.status = "ready_to_curate" if (voice.source_clip_count or 0) > 0 else "needs_source"

    if not appearance.primary_image_path and speaker.thumbnail_path:
        appearance.primary_image_path = speaker.thumbnail_path

    session.add(personality)
    session.add(appearance)
    session.add(voice)
    session.commit()
    session.refresh(avatar)
    session.refresh(personality)
    session.refresh(appearance)
    session.refresh(voice)

    avatar_dir = _avatar_artifacts_dir(avatar)

    return AvatarWorkbenchRead(
        avatar=_serialize_avatar(avatar),
        speaker=AvatarWorkbenchSpeakerRead(
            id=int(speaker.id),
            channel_id=int(speaker.channel_id),
            name=speaker.name,
            thumbnail_path=speaker.thumbnail_path,
            total_speaking_time=round(total_speaking_time, 1),
            embedding_count=embedding_count,
            appearance_count=appearance_count,
        ),
        personality=AvatarSectionSummaryRead(
            status=str(personality.status or "draft"),
            source_count=source_turn_count,
            approved_count=int(personality.approved_example_count or 0),
            artifact_ready=bool(personality.dataset_path or personality.lora_adapter_path),
            summary=(
                f"Dataset built with {int(personality.dataset_example_count or 0)} training examples from {source_turn_count} source turns."
                if personality.dataset_path and int(personality.dataset_example_count or 0) > 0
                else (
                    f"{source_turn_count} source turns from diarized conversations are available for dataset curation."
                    if source_turn_count > 0
                    else "No transcript turns are currently assigned to this speaker."
                )
            ),
            artifact_path=personality.dataset_path or personality.lora_adapter_path,
            last_built_at=personality.last_built_at,
        ),
        appearance=AvatarSectionSummaryRead(
            status=str(appearance.status or "draft"),
            source_count=int(appearance.source_image_count or 0),
            approved_count=int(appearance.approved_image_count or 0),
            artifact_ready=bool(appearance.appearance_lora_path),
            summary=(
                "A primary portrait is already available."
                if appearance.primary_image_path
                else "No approved portrait is attached yet."
            ),
            artifact_path=appearance.appearance_lora_path,
            last_built_at=appearance.last_built_at,
        ),
        voice=AvatarSectionSummaryRead(
            status=str(voice.status or "draft"),
            source_count=int(voice.source_clip_count or 0),
            approved_count=int(voice.approved_clip_count or 0),
            artifact_ready=bool(voice.embedding_path),
            summary=(
                f"{embedding_count} existing diarization voice profiles are available as seed material."
                if embedding_count > 0
                else "No reusable voice profiles have been approved yet."
            ),
            artifact_path=voice.embedding_path,
            last_built_at=voice.last_built_at,
        ),
        runtime_status="not_ready",
        suggested_base_model=str(personality.base_model_id or "Qwen/Qwen3-14B"),
        artifacts_dir=str(avatar_dir),
    )


@app.get("/avatars", response_model=List[AvatarRead])
def read_avatars(
    channel_id: Optional[int] = None,
    speaker_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    query = select(Avatar)
    if channel_id is not None:
        query = query.where(Avatar.channel_id == channel_id)
    if speaker_id is not None:
        query = query.where(Avatar.speaker_id == speaker_id)
    query = query.order_by(Avatar.updated_at.desc(), Avatar.id.desc())
    avatars = session.exec(query).all()
    return [_serialize_avatar(avatar) for avatar in avatars]


@app.post("/avatars", response_model=AvatarRead)
def create_avatar(body: AvatarCreateRequest, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, body.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    avatar_name = str(body.name or "").strip() or f"{speaker.name} Avatar"
    avatar = Avatar(
        channel_id=int(speaker.channel_id),
        speaker_id=int(speaker.id),
        name=avatar_name,
        description=(str(body.description).strip() or None) if body.description is not None else None,
    )
    session.add(avatar)
    session.commit()
    session.refresh(avatar)
    _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(avatar)
    _avatar_artifacts_dir(avatar)
    return _serialize_avatar(avatar)


@app.post("/speakers/{speaker_id}/avatar", response_model=AvatarRead)
def create_or_open_speaker_avatar(speaker_id: int, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    avatar = session.exec(
        select(Avatar)
        .where(Avatar.speaker_id == speaker_id)
        .order_by(Avatar.updated_at.desc(), Avatar.id.desc())
    ).first()
    if not avatar:
        avatar = Avatar(
            channel_id=int(speaker.channel_id),
            speaker_id=int(speaker.id),
            name=f"{speaker.name} Avatar",
        )
        session.add(avatar)
        session.commit()
        session.refresh(avatar)

    _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(avatar)
    _avatar_artifacts_dir(avatar)
    return _serialize_avatar(avatar)


@app.get("/avatars/{avatar_id}", response_model=AvatarRead)
def read_avatar(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _serialize_avatar(avatar)


@app.patch("/avatars/{avatar_id}", response_model=AvatarRead)
def update_avatar(avatar_id: int, body: AvatarUpdateRequest, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    source_speaker = session.get(Speaker, avatar.speaker_id)
    personality, appearance, voice = _ensure_avatar_profiles(
        session,
        avatar,
        speaker_name=source_speaker.name if source_speaker else avatar.name,
    )

    if body.name is not None:
        trimmed_name = str(body.name).strip()
        if not trimmed_name:
            raise HTTPException(status_code=400, detail="Avatar name cannot be empty")
        avatar.name = trimmed_name
    if body.status is not None:
        avatar.status = str(body.status).strip() or "draft"
    if body.description is not None:
        avatar.description = str(body.description).strip() or None
    if body.personality_system_prompt is not None:
        personality.system_prompt = str(body.personality_system_prompt).strip() or None
    if body.personality_base_model_id is not None:
        personality.base_model_id = str(body.personality_base_model_id).strip() or None
    if body.appearance_primary_image_path is not None:
        appearance.primary_image_path = str(body.appearance_primary_image_path).strip() or None
    if body.voice_primary_reference_path is not None:
        voice.primary_reference_path = str(body.voice_primary_reference_path).strip() or None
    if body.voice_provider is not None:
        voice.provider = str(body.voice_provider).strip() or None

    now = datetime.now()
    avatar.updated_at = now
    personality.updated_at = now
    appearance.updated_at = now
    voice.updated_at = now

    session.add(avatar)
    session.add(personality)
    session.add(appearance)
    session.add(voice)
    session.commit()
    session.refresh(avatar)
    return _serialize_avatar(avatar)


@app.get("/avatars/{avatar_id}/workbench", response_model=AvatarWorkbenchRead)
def get_avatar_workbench(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _build_avatar_workbench(session, avatar)


@app.get("/avatars/{avatar_id}/personality/dataset-preview", response_model=AvatarPersonalityDatasetRead)
def get_avatar_personality_dataset_preview(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    return _load_avatar_personality_dataset(avatar, personality)


@app.post("/avatars/{avatar_id}/personality/build-dataset", response_model=AvatarPersonalityDatasetRead)
def build_avatar_personality_dataset(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(personality)
    return _build_avatar_personality_dataset(session, avatar, speaker, personality)


@app.post("/avatars/{avatar_id}/personality/run-judge-pass", response_model=AvatarPersonalityDatasetRead)
def run_avatar_personality_judge_pass(
    avatar_id: int,
    body: AvatarPersonalityJudgePassRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(personality)
    dataset = _run_avatar_personality_judge_pass(
        avatar,
        personality,
        max_examples=body.max_examples,
        overwrite_existing=body.overwrite_existing,
        target_filter=body.target_filter,
    )
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return dataset


@app.post("/avatars/{avatar_id}/personality/start-judge-pass", response_model=AvatarPersonalityJudgeStatusRead)
def start_avatar_personality_judge_pass(
    avatar_id: int,
    body: AvatarPersonalityJudgePassRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _start_avatar_personality_judge_pass(
        avatar_id=int(avatar_id),
        max_examples=int(body.max_examples or 40),
        overwrite_existing=bool(body.overwrite_existing),
        target_filter=str(body.target_filter or "needs_review"),
    )


@app.get("/avatars/{avatar_id}/personality/judge-status", response_model=AvatarPersonalityJudgeStatusRead)
def read_avatar_personality_judge_status(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _load_avatar_personality_judge_status(avatar)


@app.post("/avatars/{avatar_id}/personality/stop-judge-pass", response_model=AvatarPersonalityJudgeStatusRead)
def stop_avatar_personality_judge_pass(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    is_active = False
    with _avatar_judge_runs_lock:
        stop_event = _avatar_judge_stop_events.get(int(avatar_id))
        active_thread = _avatar_judge_threads.get(int(avatar_id))
        if stop_event and active_thread and active_thread.is_alive():
            stop_event.set()
            is_active = True
    if not is_active:
        return _load_avatar_personality_judge_status(avatar)
    return _write_avatar_personality_judge_status(
        avatar,
        {
            "status": "stopping",
            "active": True,
            "stop_requested": True,
            "current_stage": "stop_requested",
        },
    )


@app.get("/avatars/{avatar_id}/personality/long-form-samples", response_model=AvatarPersonalityLongFormPageRead)
def read_avatar_personality_long_form_samples(
    avatar_id: int,
    offset: int = 0,
    limit: int = 20,
    state: str = "all",
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _read_avatar_personality_long_form_page(
        session,
        avatar,
        offset=max(0, int(offset)),
        limit=max(1, min(int(limit), 100)),
        state=state,
    )


@app.patch("/avatars/{avatar_id}/personality/long-form-samples/{sample_id}", response_model=AvatarPersonalityLongFormSampleRead)
def update_avatar_personality_long_form_sample_state(
    avatar_id: int,
    sample_id: str,
    body: AvatarPersonalityLongFormSampleStateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    state_map = _load_avatar_personality_long_form_states(avatar)
    state_map[str(sample_id)] = str(body.state or "included")
    _write_avatar_personality_long_form_states(avatar, state_map)
    page = _read_avatar_personality_long_form_page(session, avatar, offset=0, limit=200, state="all")
    for item in page.items:
        if item.sample_id == sample_id:
            return item
    raise HTTPException(status_code=404, detail="Long-form sample not found")


@app.get("/avatars/{avatar_id}/personality/long-form-config", response_model=AvatarPersonalityLongFormConfigRead)
def read_avatar_personality_long_form_config(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _load_avatar_personality_long_form_config(avatar)


@app.patch("/avatars/{avatar_id}/personality/long-form-config", response_model=AvatarPersonalityLongFormConfigRead)
def update_avatar_personality_long_form_config(
    avatar_id: int,
    body: AvatarPersonalityLongFormConfigUpdateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    page = _read_avatar_personality_long_form_page(session, avatar, offset=0, limit=1, state="all")
    return _write_avatar_personality_long_form_config(
        avatar,
        take_count=max(0, int(body.take_count or 0)),
        included_count=page.included_count,
        rejected_count=page.rejected_count,
        selected_count=min(max(0, int(body.take_count or 0)), page.included_count),
    )


@app.get("/avatars/{avatar_id}/personality/training-config", response_model=AvatarPersonalityTrainingConfigRead)
def read_avatar_personality_training_config(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    return _read_avatar_personality_training_config(session, avatar, personality)


@app.get("/avatars/{avatar_id}/personality/base-model-support", response_model=AvatarPersonalityBaseModelSupportRead)
def read_avatar_personality_base_model_support(
    avatar_id: int,
    model_id: Optional[str] = None,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    config = _load_avatar_personality_training_config(avatar)
    selected = str(model_id or config.base_model_id or "Qwen/Qwen3-8B").strip()
    return _read_avatar_base_model_support(selected)


@app.post("/avatars/{avatar_id}/personality/base-model-download", response_model=AvatarPersonalityBaseModelSupportRead)
def download_avatar_personality_base_model(
    avatar_id: int,
    body: AvatarPersonalityBaseModelDownloadRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    config = _load_avatar_personality_training_config(avatar)
    selected = str(body.model_id or config.base_model_id or "Qwen/Qwen3-8B").strip()
    _start_avatar_hf_model_download(selected)
    return _read_avatar_base_model_support(selected)


@app.patch("/avatars/{avatar_id}/personality/training-config", response_model=AvatarPersonalityTrainingConfigRead)
def update_avatar_personality_training_config(
    avatar_id: int,
    body: AvatarPersonalityTrainingConfigUpdateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    normalized = _write_avatar_personality_training_config(
        avatar,
        base_model_id=body.base_model_id,
        dataset_profile=body.dataset_profile,
        training_strength=body.training_strength,
        export_strategy=body.export_strategy,
        validation_ratio=body.validation_ratio,
        max_examples=body.max_examples,
        max_long_form_examples=body.max_long_form_examples,
        include_long_form=body.include_long_form,
        training_mode=body.training_mode,
        snapshot_interval_steps=body.snapshot_interval_steps,
    )
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    payload = normalized.model_dump()
    payload["dataset_profiles"] = [option.model_dump() for option in _avatar_training_dataset_profile_options()]
    payload["training_plan"] = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=normalized,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingConfigRead(**payload)


@app.get("/avatars/{avatar_id}/personality/training-package", response_model=AvatarPersonalityTrainingPackageRead)
def read_avatar_personality_training_package(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    package = _read_avatar_personality_training_package(avatar)
    if package.training_plan is not None:
        return package
    config = _load_avatar_personality_training_config(avatar)
    payload = package.model_dump()
    payload["training_plan"] = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        selected_conversation_examples=package.conversation_examples_selected if package.status == "ready" else None,
        selected_long_form_examples=package.long_form_examples_selected if package.status == "ready" else None,
        train_examples=package.train_examples if package.status == "ready" else None,
        validation_examples=package.validation_examples if package.status == "ready" else None,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingPackageRead(**payload)


@app.post("/avatars/{avatar_id}/personality/prepare-training-package", response_model=AvatarPersonalityTrainingPackageRead)
def prepare_avatar_personality_training_package(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    package = _prepare_avatar_personality_training_package(session, avatar, personality)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return package


@app.get("/avatars/{avatar_id}/personality/training-status", response_model=AvatarPersonalityTrainingStatusRead)
def read_avatar_personality_training_status(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = _reconcile_avatar_personality_training_runtime(avatar)
    resolved_snapshots = _avatar_resolve_training_snapshots(avatar, selected_adapter_path=status.adapter_path)
    if resolved_snapshots:
        status = _write_avatar_personality_training_status(
            avatar,
            {
                "snapshots": [
                    item.model_dump(mode="json")
                    for item in resolved_snapshots
                ]
            },
        )
    if status.status in {"completed", "failed", "stopped"}:
        _sync_avatar_personality_training_completion(int(avatar_id))
    return status


@app.post("/avatars/{avatar_id}/personality/start-training", response_model=AvatarPersonalityTrainingStatusRead)
def start_avatar_personality_training(
    avatar_id: int,
    body: AvatarPersonalityTrainRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = _start_avatar_personality_training(avatar, personality, body)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return status


@app.post("/avatars/{avatar_id}/personality/stop-training", response_model=AvatarPersonalityTrainingStatusRead)
def stop_avatar_personality_training(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = _reconcile_avatar_personality_training_runtime(avatar)
    if not status.active or status.status in {"idle", "completed", "failed", "stopped"}:
        return status

    status_path, stop_path = _avatar_personality_training_runtime_paths(avatar)
    stop_path.write_text("stop", encoding="utf-8")
    status = _write_avatar_personality_training_status(
        avatar,
        {
            "status": "stopping",
            "active": True,
            "stop_requested": True,
            "current_stage": "stop_requested",
            "updated_at": datetime.now(),
            "message": "Stop requested. Waiting for the current training step to finish.",
        },
    )
    return status


@app.post("/avatars/{avatar_id}/personality/promote-snapshot", response_model=AvatarPersonalityTrainingStatusRead)
def promote_avatar_personality_snapshot(
    avatar_id: int,
    body: AvatarPersonalitySnapshotSelectRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = _reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before promoting a snapshot.")
    selected = str(body.adapter_path or "").strip()
    snapshot = _avatar_find_training_snapshot(avatar, selected)
    if snapshot is None or not Path(selected).exists():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    personality.lora_adapter_path = selected
    personality.status = "trained"
    personality.updated_at = datetime.now()
    session.add(personality)
    session.commit()
    _avatar_release_cached_chat_model(int(avatar.id))
    return _avatar_update_training_snapshots(
        avatar,
        _avatar_resolve_training_snapshots(avatar, selected_adapter_path=selected),
        selected_adapter_path=selected,
    )


@app.post("/avatars/{avatar_id}/personality/delete-other-snapshots", response_model=AvatarPersonalityTrainingStatusRead)
def delete_other_avatar_personality_snapshots(
    avatar_id: int,
    body: AvatarPersonalitySnapshotCleanupRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = _reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before deleting snapshots.")
    keep_path = str(body.keep_adapter_path or "").strip()
    snapshot = _avatar_find_training_snapshot(avatar, keep_path)
    if snapshot is None or not Path(keep_path).exists():
        raise HTTPException(status_code=404, detail="Snapshot to keep was not found.")
    speaker = session.get(Speaker, avatar.speaker_id)
    if speaker:
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        personality.lora_adapter_path = keep_path
        personality.status = "trained"
        personality.updated_at = datetime.now()
        session.add(personality)
        session.commit()

    for item in _avatar_resolve_training_snapshots(avatar, selected_adapter_path=keep_path):
        adapter_path = str(item.adapter_path or "").strip()
        if not adapter_path or adapter_path == keep_path:
            continue
        try:
            shutil.rmtree(adapter_path, ignore_errors=True)
        except Exception:
            pass
    _avatar_release_cached_chat_model(int(avatar.id))
    return _avatar_update_training_snapshots(avatar, [snapshot], selected_adapter_path=keep_path)


@app.post("/avatars/{avatar_id}/personality/delete-snapshot", response_model=AvatarPersonalityTrainingStatusRead)
def delete_avatar_personality_snapshot(
    avatar_id: int,
    body: AvatarPersonalitySnapshotDeleteRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = _reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before deleting snapshots.")

    delete_path = str(body.adapter_path or "").strip()
    snapshot = _avatar_find_training_snapshot(avatar, delete_path)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot was not found.")

    snapshots = _avatar_resolve_training_snapshots(avatar, selected_adapter_path=status.adapter_path)
    remaining = [item for item in snapshots if str(item.adapter_path or "").strip() != delete_path]
    if not remaining:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining snapshot.")

    next_selected_path = str(status.adapter_path or "").strip()
    if not next_selected_path or next_selected_path == delete_path or not any(str(item.adapter_path or "").strip() == next_selected_path for item in remaining):
        fallback = next((item for item in remaining if item.kind == "final"), None) or remaining[0]
        next_selected_path = str(fallback.adapter_path or "").strip()

    speaker = session.get(Speaker, avatar.speaker_id)
    if speaker:
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        if str(personality.lora_adapter_path or "").strip() == delete_path:
            personality.lora_adapter_path = next_selected_path
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()

    try:
        delete_dir = Path(delete_path)
        if delete_dir.exists():
            shutil.rmtree(delete_dir, ignore_errors=True)
    except Exception:
        pass

    _avatar_release_cached_chat_model(int(avatar.id))
    return _avatar_update_training_snapshots(avatar, remaining, selected_adapter_path=next_selected_path)


_AVATAR_FIT_CHECK_PROMPTS: list[tuple[str, str]] = [
    ("casual_checkin", "How's it going?"),
    ("self_description", "Tell me about yourself in a couple of sentences."),
    ("misunderstanding", "What is something people often misunderstand about a topic you care about?"),
    ("argument", "Give me a short argument for a position you genuinely care about."),
    ("analogy", "Explain one idea using an analogy you would naturally reach for."),
    ("disagreement", "Someone says you're overthinking it and missing the point. Respond naturally."),
]


def _avatar_fit_check_reference_examples(avatar: Avatar, *, limit: int = 3) -> list[dict[str, str]]:
    state_map = _load_avatar_personality_state_map(avatar)
    ranked: list[tuple[tuple[int, int, int, int, int], dict[str, str]]] = []
    for row in _iter_avatar_personality_review_examples(avatar):
        state, _ = _resolve_avatar_personality_example_state(row, state_map)
        if state != "approved":
            continue
        response_text = _clean_avatar_dataset_text(str(row.get("response_text") or ""))
        if not response_text:
            continue
        label = _avatar_normalize_llm_label(str(row.get("llm_label") or row.get("heuristic_label") or row.get("auto_label") or "silver"))
        ranked.append(
            (
                (
                    1 if label == "gold" else 0,
                    int(row.get("quality_score") or 0),
                    int(row.get("style_score") or 0),
                    int(row.get("substance_score") or 0),
                    int(row.get("response_word_count") or 0),
                ),
                {
                    "video_title": str(row.get("video_title") or "").strip(),
                    "response_text": response_text[:420],
                },
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[: max(0, int(limit or 0))]]


def _avatar_run_fit_check_judge(
    *,
    speaker_name: str,
    system_prompt: str,
    results: list[AvatarPersonalityFitCheckPromptResultRead],
    reference_examples: list[dict[str, str]],
) -> tuple[str, dict[str, object]]:
    import httpx

    ollama_url, judge_model = _avatar_resolve_local_judge_model()
    reference_text = "\n".join(
        f"- {item.get('video_title') or 'Reference'}: {item.get('response_text') or ''}"
        for item in reference_examples
        if str(item.get("response_text") or "").strip()
    ).strip()
    result_text = "\n\n".join(
        f"[{item.key}] Prompt: {item.prompt}\nReply: {item.reply}"
        for item in results
    )
    prompt = (
        "You are evaluating whether a personality LoRA is undertrained, balanced, or overtrained.\n"
        "The goal is not factual accuracy. The goal is whether the adapter captured the speaker's voice without collapsing into memorized transcript fragments.\n\n"
        "Definitions:\n"
        "- underfit: replies are generic, weakly persona-specific, bland, or sound like the base model instead of the speaker.\n"
        "- balanced: replies answer the prompt directly, feel specific to the speaker, stay varied, and do not loop or parrot obvious transcript fragments.\n"
        "- overfit: replies repeat stock phrases, intros/outros, transcript snippets, malformed tokens, or ignore the prompt in favor of memorized patterns.\n"
        "- unclear: mixed evidence or not enough signal.\n\n"
        "Pay close attention to prompt following, repetition or looping, transcript artifacts such as <think>, reuse of the same stock phrases across prompts, and whether arguments feel natural instead of copied.\n"
        "Return JSON only with keys: classification, confidence, summary, strengths, concerns, recommendations.\n"
        "classification must be one of: underfit, balanced, overfit, unclear.\n"
        "confidence must be 0-100.\n"
        "strengths, concerns, recommendations must each be arrays of short strings.\n\n"
        f"Speaker: {speaker_name}\n"
        f"System prompt:\n{system_prompt.strip()}\n\n"
        f"Reference excerpts from the target speaker:\n{reference_text or '(none available)'}\n\n"
        f"Prompt/reply evaluation set:\n{result_text}\n"
    )
    response = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": judge_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 320,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "").strip()
    if not raw_text:
        raise ValueError("Local fit-check judge returned an empty response")
    data = _avatar_extract_json_object(raw_text)
    classification = str(data.get("classification") or "").strip().lower()
    if classification not in {"underfit", "balanced", "overfit", "unclear"}:
        classification = "unclear"
    def _clean_string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = _clean_avatar_dataset_text(str(item))
            if text:
                cleaned.append(text)
        return cleaned
    payload: dict[str, object] = {
        "classification": classification,
        "confidence": max(0, min(100, int(data.get("confidence") or 0))),
        "summary": _clean_avatar_dataset_text(str(data.get("summary") or "")),
        "strengths": _clean_string_list(data.get("strengths")),
        "concerns": _clean_string_list(data.get("concerns")),
        "recommendations": _clean_string_list(data.get("recommendations")),
    }
    return judge_model, payload


@app.post("/avatars/{avatar_id}/personality/test-chat", response_model=AvatarPersonalityTestChatResponse)
def test_avatar_personality_chat(
    avatar_id: int,
    body: AvatarPersonalityTestChatRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = _load_avatar_personality_training_status(avatar)
    requested_adapter_path = str(body.adapter_path or "").strip()
    adapter_path = requested_adapter_path or str(status.adapter_path or personality.lora_adapter_path or "").strip()
    base_model_id = str(status.base_model_id or personality.base_model_id or _load_avatar_personality_training_config(avatar).base_model_id).strip()
    if not adapter_path or not Path(adapter_path).exists():
        raise HTTPException(status_code=400, detail="No trained adapter is available yet.")
    if not str(body.message or "").strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    snapshot = _avatar_find_training_snapshot(avatar, adapter_path)
    if requested_adapter_path and snapshot is None:
        raise HTTPException(status_code=404, detail="Selected snapshot was not found.")

    try:
        system_prompt = str(personality.system_prompt or _default_avatar_personality_prompt(speaker.name)).strip()
        reply = _avatar_generate_personality_reply(
            avatar_id=int(avatar.id),
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            training_mode=str(status.training_mode or "memory_optimized"),
            system_prompt=system_prompt,
            history=[{"role": str(turn.role), "content": str(turn.content)} for turn in body.history or []],
            message=str(body.message).strip(),
            max_new_tokens=int(body.max_new_tokens or 220),
            temperature=float(body.temperature or 0.8),
            top_p=float(body.top_p or 0.9),
        )
        return AvatarPersonalityTestChatResponse(
            avatar_id=int(avatar.id),
            reply=reply,
            model=base_model_id,
            adapter_path=adapter_path,
            snapshot_label=snapshot.label if snapshot else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate test reply: {exc}") from exc


@app.post("/avatars/{avatar_id}/personality/fit-check", response_model=AvatarPersonalityFitCheckResponse)
def run_avatar_personality_fit_check(
    avatar_id: int,
    body: AvatarPersonalityFitCheckRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = _load_avatar_personality_training_status(avatar)
    requested_adapter_path = str(body.adapter_path or "").strip()
    adapter_path = requested_adapter_path or str(status.adapter_path or personality.lora_adapter_path or "").strip()
    base_model_id = str(status.base_model_id or personality.base_model_id or _load_avatar_personality_training_config(avatar).base_model_id).strip()
    if not adapter_path or not Path(adapter_path).exists():
        raise HTTPException(status_code=400, detail="No trained adapter is available yet.")
    snapshot = _avatar_find_training_snapshot(avatar, adapter_path)
    if requested_adapter_path and snapshot is None:
        raise HTTPException(status_code=404, detail="Selected snapshot was not found.")

    system_prompt = str(personality.system_prompt or _default_avatar_personality_prompt(speaker.name)).strip()
    results: list[AvatarPersonalityFitCheckPromptResultRead] = []
    try:
        for key, prompt in _AVATAR_FIT_CHECK_PROMPTS:
            reply = _avatar_generate_personality_reply(
                avatar_id=int(avatar.id),
                base_model_id=base_model_id,
                adapter_path=adapter_path,
                training_mode=str(status.training_mode or "memory_optimized"),
                system_prompt=system_prompt,
                history=[],
                message=prompt,
                max_new_tokens=int(body.max_new_tokens or 160),
                temperature=float(body.temperature or 0.75),
                top_p=float(body.top_p or 0.9),
            )
            results.append(
                AvatarPersonalityFitCheckPromptResultRead(
                    key=key,
                    prompt=prompt,
                    reply=reply,
                )
            )
        judge_model, judge_payload = _avatar_run_fit_check_judge(
            speaker_name=str(speaker.name or avatar.name or "the speaker").strip(),
            system_prompt=system_prompt,
            results=results,
            reference_examples=_avatar_fit_check_reference_examples(avatar, limit=3),
        )
        return AvatarPersonalityFitCheckResponse(
            avatar_id=int(avatar.id),
            model=base_model_id,
            judge_model=judge_model,
            adapter_path=adapter_path,
            snapshot_label=snapshot.label if snapshot else None,
            classification=str(judge_payload.get("classification") or "unclear"),
            confidence=int(judge_payload.get("confidence") or 0),
            summary=str(judge_payload.get("summary") or ""),
            strengths=[str(item) for item in judge_payload.get("strengths", []) if str(item).strip()],
            concerns=[str(item) for item in judge_payload.get("concerns", []) if str(item).strip()],
            recommendations=[str(item) for item in judge_payload.get("recommendations", []) if str(item).strip()],
            results=results,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run personality fit check: {exc}") from exc


@app.get("/avatars/{avatar_id}/personality/examples", response_model=AvatarPersonalityDatasetPageRead)
def read_avatar_personality_examples(
    avatar_id: int,
    offset: int = 0,
    limit: int = 20,
    state: str = "all",
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _read_avatar_personality_dataset_page(
        avatar,
        offset=offset,
        limit=limit,
        state_filter=state,
    )


@app.patch("/avatars/{avatar_id}/personality/examples/{example_id}", response_model=AvatarPersonalityDatasetRead)
def update_avatar_personality_example_state(
    avatar_id: int,
    example_id: int,
    body: AvatarPersonalityExampleStateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    personality, _, _ = _ensure_avatar_profiles(session, avatar)

    found = False
    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("example_id")) == int(example_id):
                found = True
                break
        except Exception:
            continue
    if not found:
        raise HTTPException(status_code=404, detail="Dataset example not found")

    state_map = _load_avatar_personality_state_map(avatar)
    next_state = str(body.state or "").strip().lower()
    if next_state not in {"approved", "rejected", "inherit"}:
        raise HTTPException(status_code=400, detail="Invalid dataset example state")
    if next_state == "inherit":
        state_map.pop(int(example_id), None)
    else:
        state_map[int(example_id)] = next_state

    _write_avatar_personality_state_map(avatar, state_map)
    refreshed = _refresh_avatar_personality_dataset_exports(avatar, personality)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return refreshed


@app.get("/avatars/{avatar_id}/personality/duplicate-group/{group_id}", response_model=List[AvatarPersonalityDatasetExampleRead])
def get_avatar_personality_duplicate_group(
    avatar_id: int,
    group_id: int,
    session: Session = Depends(get_session),
):
    """Return all dataset examples that share the same LSH duplicate group."""
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    state_map = _load_avatar_personality_state_map(avatar)
    items: List[AvatarPersonalityDatasetExampleRead] = []
    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("duplicate_group_id") or -1) == group_id:
                state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
                items.append(_row_to_example_read(row, state, manual_state))
        except Exception:
            continue
    return items


@app.get("/avatars/{avatar_id}/personality/examples/{example_id}/find-similar", response_model=SemanticSearchPage)
def find_similar_passages_for_example(
    avatar_id: int,
    example_id: int,
    limit: int = 8,
    session: Session = Depends(get_session),
):
    """Use semantic search to find passages similar to a dataset example's response text.

    Requires the semantic index to have been built for this channel.
    Excludes the source chunk(s) that directly contain this example.
    """
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    target_row: Optional[dict] = None
    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("example_id")) == example_id:
                target_row = row
                break
        except Exception:
            continue
    if target_row is None:
        raise HTTPException(status_code=404, detail="Example not found")

    response_text = str(target_row.get("response_text") or "").strip()
    if not response_text:
        raise HTTPException(status_code=400, detail="Example has no response text")

    src_video_id = int(target_row.get("video_id") or 0)
    src_start = float(target_row.get("start_time") or 0.0)
    src_end = float(target_row.get("end_time") or 0.0)

    safe_limit = max(1, min(limit, 20))
    try:
        result = sem_svc.semantic_search(
            query=response_text,
            channel_id=avatar.channel_id,
            speaker_id=avatar.speaker_id,
            limit=safe_limit + 5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {exc}")

    # Exclude the chunk that directly overlaps with this example's own time range
    filtered = [
        hit for hit in result["items"]
        if not (hit["video_id"] == src_video_id and hit["start_time"] < src_end and hit["end_time"] > src_start)
    ][:safe_limit]

    return SemanticSearchPage(
        items=[SemanticSearchHit(**hit) for hit in filtered],
        total=len(filtered),
        limit=safe_limit,
        offset=0,
    )


@app.get("/speakers", response_model=List[SpeakerRead])
def read_speakers(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: Session = Depends(get_session)
):
    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))
    page_rows = _query_speaker_page_rows(
        session=session,
        channel_id=channel_id,
        video_id=video_id,
        search=search,
        offset=safe_offset,
        limit=safe_limit,
    )
    return _build_speaker_reads(session, page_rows)


def _build_speaker_reads(session: Session, rows: list[dict]) -> List[SpeakerRead]:
    from sqlalchemy import func

    emb_counts: dict[int, int] = {}
    if rows:
        speaker_ids = [int(row["id"]) for row in rows]
        emb_query = select(
            SpeakerEmbedding.speaker_id,
            func.count(SpeakerEmbedding.id).label("cnt")
        ).where(
            SpeakerEmbedding.speaker_id.in_(speaker_ids)
        ).group_by(SpeakerEmbedding.speaker_id)
        for speaker_id, cnt in session.exec(emb_query).all():
            emb_counts[int(speaker_id)] = int(cnt or 0)

    speakers: List[SpeakerRead] = []
    for row in rows:
        speaker_id = int(row["id"])
        speakers.append(
            SpeakerRead(
                id=speaker_id,
                channel_id=int(row["channel_id"]),
                name=str(row["name"]),
                thumbnail_path=row.get("thumbnail_path"),
                is_extra=bool(row.get("is_extra")),
                total_speaking_time=float(row.get("total_speaking_time") or 0.0),
                embedding_count=int(emb_counts.get(speaker_id, 0)),
                created_at=row["created_at"],
            )
        )
    return speakers


@app.get("/speakers/overview", response_model=SpeakerOverviewRead)
def read_speaker_overview(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: Session = Depends(get_session)
):
    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))
    full_rows = _query_full_speaker_scope_rows(
        session=session,
        channel_id=channel_id,
        video_id=video_id,
        search=search,
    )
    page_rows = full_rows[safe_offset:] if safe_limit is None else full_rows[safe_offset:safe_offset + safe_limit]
    counts = SpeakerCountsRead(**_summarize_speaker_scope_rows(full_rows))
    return SpeakerOverviewRead(
        items=_build_speaker_reads(session, page_rows),
        counts=counts,
        total=len(full_rows),
        offset=safe_offset,
        limit=safe_limit,
    )


@app.get("/speakers/stats", response_model=SpeakerCountsRead)
def read_speaker_counts(
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    cache_key = _speaker_scope_key(channel_id, video_id)
    cached = _get_speaker_counts_cache(cache_key)
    if cached is not None:
        return SpeakerCountsRead(**cached)
    summary = _query_speaker_count_summary(session=session, channel_id=channel_id, video_id=video_id)
    counts = SpeakerCountsRead(
        total=int(summary.get("total") or 0),
        identified=int(summary.get("identified") or 0),
        unknown=int(summary.get("unknown") or 0),
        main=int(summary.get("main") or 0),
        extras=int(summary.get("extras") or 0),
    )
    _set_speaker_counts_cache(cache_key, counts.model_dump())
    return counts

@app.get("/speakers/{speaker_id}", response_model=SpeakerRead)
def read_speaker(speaker_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import func

    row = session.exec(
        select(
            Speaker.id,
            Speaker.channel_id,
            Speaker.name,
            Speaker.thumbnail_path,
            Speaker.is_extra,
            Speaker.created_at,
        ).where(Speaker.id == speaker_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Speaker not found")

    total_time_result = session.exec(
        select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
        .where(TranscriptSegment.speaker_id == speaker_id)
    ).first()
    total_time = total_time_result or 0

    emb_count = session.exec(
        select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
    ).first() or 0

    return SpeakerRead(
        id=int(row[0]),
        channel_id=int(row[1]),
        name=str(row[2]),
        thumbnail_path=row[3],
        is_extra=bool(row[4]),
        total_speaking_time=round(total_time, 1),
        embedding_count=int(emb_count),
        created_at=row[5],
    )

@app.get("/speakers/{speaker_id}/appearances", response_model=List[SpeakerEpisodeAppearanceRead])
def read_speaker_appearances(
    speaker_id: int,
    offset: int = 0,
    limit: Optional[int] = None,
    response: Response = None,
    session: Session = Depends(get_session),
):
    from sqlalchemy import func, case

    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))

    base_query = (
        select(
            Video.id,
            Video.youtube_id,
            Video.title,
            Video.media_source_type,
            Video.media_kind,
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
    )
    if response is not None:
        count_rows = session.exec(
            select(func.count()).select_from(base_query.subquery("speaker_appearance_rows"))
        ).first()
        response.headers["X-Total-Count"] = str(int(count_rows or 0))
    if safe_offset:
        base_query = base_query.offset(safe_offset)
    if safe_limit is not None:
        base_query = base_query.limit(safe_limit)

    rows = session.exec(base_query).all()

    appearances: List[SpeakerEpisodeAppearanceRead] = []
    for row in rows:
        appearances.append(
            SpeakerEpisodeAppearanceRead(
                video_id=row[0],
                youtube_id=row[1],
                title=row[2],
                media_source_type=row[3] or "youtube",
                media_kind=row[4],
                published_at=row[5],
                thumbnail_url=row[6],
                segment_count=int(row[7] or 0),
                total_speaking_time=round(float(row[8] or 0), 1),
                first_start_time=float(row[9] or 0),
                last_end_time=float(row[10] or 0),
            )
        )
    return appearances

@app.get("/speakers/{speaker_id}/profiles", response_model=List[SpeakerVoiceProfileRead])
def read_speaker_profiles(
    speaker_id: int,
    offset: int = 0,
    limit: Optional[int] = None,
    response: Response = None,
    session: Session = Depends(get_session),
):
    speaker_exists = session.exec(select(Speaker.id).where(Speaker.id == speaker_id)).first()
    if not speaker_exists:
        raise HTTPException(status_code=404, detail="Speaker not found")

    safe_offset = max(0, int(offset or 0))
    safe_limit = None if limit is None else max(1, min(int(limit), 500))

    query = (
        select(
            SpeakerEmbedding.id,
            SpeakerEmbedding.speaker_id,
            SpeakerEmbedding.source_video_id,
            SpeakerEmbedding.sample_start_time,
            SpeakerEmbedding.sample_end_time,
            SpeakerEmbedding.sample_text,
            SpeakerEmbedding.created_at,
            Video.title,
            Video.youtube_id,
            Video.media_source_type,
            Video.media_kind,
            Video.published_at,
        )
        .join(Video, SpeakerEmbedding.source_video_id == Video.id, isouter=True)
        .where(SpeakerEmbedding.speaker_id == speaker_id)
        .order_by(SpeakerEmbedding.created_at.desc(), SpeakerEmbedding.id.desc())
    )
    if response is not None:
        total_profiles = session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker_id)
        ).first()
        response.headers["X-Total-Count"] = str(int(total_profiles or 0))
    if safe_offset:
        query = query.offset(safe_offset)
    if safe_limit is not None:
        query = query.limit(safe_limit)

    rows = session.exec(query).all()

    profiles: List[SpeakerVoiceProfileRead] = []
    for row in rows:
        profiles.append(
            SpeakerVoiceProfileRead(
                id=int(row[0]),
                speaker_id=int(row[1]),
                source_video_id=int(row[2]) if row[2] is not None else None,
                source_video_title=row[7],
                source_video_youtube_id=row[8],
                source_video_media_source_type=row[9],
                source_video_media_kind=row[10],
                source_video_published_at=row[11],
                sample_start_time=float(row[3]) if row[3] is not None else None,
                sample_end_time=float(row[4]) if row[4] is not None else None,
                sample_text=row[5],
                created_at=row[6],
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
    query = select(
        TranscriptSegment,
        Video.channel_id,
        Video.youtube_id,
        Video.media_source_type,
        Video.media_kind,
    )\
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
    for segment, channel_id, youtube_id, media_source_type, media_kind in results:
        # Create a dictionary of the segment data and add the extra fields
        data = segment.model_dump()
        data["channel_id"] = channel_id
        data["youtube_id"] = youtube_id
        data["media_source_type"] = media_source_type or "youtube"
        data["media_kind"] = media_kind
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
    Reassigns all speaker-owned records and moves all embeddings to the target.
    Deletes the source speakers."""
    from sqlalchemy import func
    
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
        # and to keep all speaker-owned rows consistent before deleting the sources.
        merge_speakers_in_session(session, target_id=req.target_id, source_ids=source_ids)
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


@app.get("/channels/{channel_id}/speaker-merge-suggestions", response_model=List[SpeakerMergeSuggestionRead])
def get_channel_speaker_merge_suggestions(
    channel_id: int,
    threshold: float = 0.12,
    limit: int = 100,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    threshold = max(0.0, min(float(threshold), 1.0))
    limit = max(1, min(int(limit), 500))

    rows = session.exec(
        select(SpeakerEmbedding, Speaker)
        .join(Speaker, SpeakerEmbedding.speaker_id == Speaker.id)
        .where(Speaker.channel_id == channel_id)
    ).all()
    if not rows:
        return []

    vectors_by_speaker: dict[int, list[np.ndarray]] = {}
    speaker_names: dict[int, str] = {}
    for emb, speaker in rows:
        try:
            vec = np.asarray(pickle.loads(emb.embedding_blob), dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        speaker_id = int(speaker.id)
        vectors_by_speaker.setdefault(speaker_id, []).append((vec / norm).astype(np.float32, copy=False))
        speaker_names[speaker_id] = speaker.name

    speaker_ids = sorted(vectors_by_speaker.keys())
    if len(speaker_ids) < 2:
        return []

    centroids = []
    for speaker_id in speaker_ids:
        speaker_vectors = vectors_by_speaker[speaker_id]
        centroid = np.mean(np.stack(speaker_vectors, axis=0), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            continue
        centroids.append((speaker_id, (centroid / centroid_norm).astype(np.float32, copy=False)))

    if len(centroids) < 2:
        return []

    speaker_ids = [speaker_id for speaker_id, _ in centroids]
    matrix = np.stack([vec for _, vec in centroids], axis=0)

    emb_count_rows = session.exec(
        select(SpeakerEmbedding.speaker_id, func.count(SpeakerEmbedding.id))
        .join(Speaker, SpeakerEmbedding.speaker_id == Speaker.id)
        .where(Speaker.channel_id == channel_id)
        .group_by(SpeakerEmbedding.speaker_id)
    ).all()
    segment_count_rows = session.exec(
        select(TranscriptSegment.speaker_id, func.count(TranscriptSegment.id))
        .join(Video, TranscriptSegment.video_id == Video.id)
        .where(Video.channel_id == channel_id, TranscriptSegment.speaker_id.is_not(None))
        .group_by(TranscriptSegment.speaker_id)
    ).all()
    embedding_counts = {int(speaker_id): int(count or 0) for speaker_id, count in emb_count_rows}
    segment_counts = {int(speaker_id): int(count or 0) for speaker_id, count in segment_count_rows}

    suggestions: list[SpeakerMergeSuggestionRead] = []
    distances = 1.0 - (matrix @ matrix.T)
    for i in range(len(speaker_ids)):
        for j in range(i + 1, len(speaker_ids)):
            dist = float(distances[i, j])
            if not np.isfinite(dist) or dist > threshold:
                continue

            left_id = int(speaker_ids[i])
            right_id = int(speaker_ids[j])
            left_seg_count = int(segment_counts.get(left_id, 0))
            right_seg_count = int(segment_counts.get(right_id, 0))
            left_emb_count = int(embedding_counts.get(left_id, 0))
            right_emb_count = int(embedding_counts.get(right_id, 0))

            target_id = left_id
            source_id = right_id
            if (
                right_seg_count > left_seg_count
                or (right_seg_count == left_seg_count and right_emb_count > left_emb_count)
                or (right_seg_count == left_seg_count and right_emb_count == left_emb_count and right_id < left_id)
            ):
                target_id = right_id
                source_id = left_id

            suggestions.append(
                SpeakerMergeSuggestionRead(
                    source_speaker_id=source_id,
                    source_speaker_name=speaker_names.get(source_id, f"Speaker {source_id}"),
                    target_speaker_id=target_id,
                    target_speaker_name=speaker_names.get(target_id, f"Speaker {target_id}"),
                    distance=round(dist, 6),
                    source_embedding_count=int(embedding_counts.get(source_id, 0)),
                    target_embedding_count=int(embedding_counts.get(target_id, 0)),
                    source_segment_count=int(segment_counts.get(source_id, 0)),
                    target_segment_count=int(segment_counts.get(target_id, 0)),
                )
            )

    suggestions.sort(
        key=lambda item: (
            item.distance,
            -item.target_segment_count,
            -item.target_embedding_count,
            item.source_speaker_id,
            item.target_speaker_id,
        )
    )
    return suggestions[:limit]

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
    video.transcript_source = None
    video.transcript_language = None
    video.transcript_is_placeholder = False
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
    new_job, _, segment_count, funny_count = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=True,
        note="Manual redo-diarization request",
        queued_from="redo_diarization",
    )
    _invalidate_speaker_query_caches()

    return {
        "status": "diarization_requeued",
        "deleted_segments": segment_count,
        "deleted_funny_moments": funny_count,
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

            payload = {
                "mode": "redo_diarization",
                "redo_diarization_backup_file": str(backup_path),
            }

            # Inherit transcription timestamps from the last completed pipeline job
            last_process_job = session.exec(
                select(Job).where(
                    Job.video_id == video.id,
                    Job.job_type == "process",
                    Job.status == "completed"
                ).order_by(Job.completed_at.desc())
            ).first()

            if last_process_job and last_process_job.payload_json:
                try:
                    old_payload = json.loads(last_process_job.payload_json)
                    for k, pval in old_payload.items():
                        if k.startswith("stage_transcribe") or k.startswith("parakeet_") or k.startswith("transcription_"):
                            payload[k] = pval
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to inherit transcription stats for channel redo: {e}")

            job = _enqueue_unique_job(
                session,
                video_id=video.id,
                job_type="process",
                payload=payload,
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
    new_job, _, segment_count, funny_count = _queue_full_retranscription_job(
        session,
        video=video,
        force=True,
        note="Manual redo-transcription request",
        queued_from="redo_transcription",
    )
    _invalidate_speaker_query_caches()
    return {
        "status": "transcription_requeued",
        "deleted_segments": segment_count,
        "deleted_funny_moments": funny_count,
        "job_id": new_job.id,
    }


@app.post("/videos/{video_id}/consolidate-transcript", response_model=TranscriptRepairResultRead)
def consolidate_video_transcript(video_id: int, session: Session = Depends(get_session)):
    """Post-process an existing transcript to smooth tiny speaker islands and merge same-speaker neighbors."""
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

    try:
        result = ingestion_service.repair_existing_transcript(
            session,
            video_id,
            save_files=True,
            persist_run=True,
            persist_snapshot=True,
            source="manual_sync",
            note="Manual consolidate-transcript request",
            trigger_semantic_index=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to consolidate transcript: {e}")

    _invalidate_speaker_query_caches()
    return TranscriptRepairResultRead(**result)


@app.post("/channels/{channel_id}/consolidate-transcripts")
def consolidate_channel_transcripts(
    channel_id: int,
    processed_only: bool = True,
    include_muted: bool = False,
    limit: int = 0,
    session: Session = Depends(get_session),
):
    """Bulk post-process existing transcripts for a channel without re-transcribing or re-diarizing."""
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    query = select(Video).where(Video.channel_id == channel_id).order_by(Video.id.desc())
    videos = session.exec(query).all()
    if limit and limit > 0:
        videos = videos[:limit]

    result = {
        "channel_id": int(channel_id),
        "channel_name": channel.name,
        "counts": {
            "scanned": 0,
            "eligible": 0,
            "changed": 0,
            "merged_segments": 0,
            "reassigned_islands": 0,
            "skipped_active": 0,
            "skipped_muted": 0,
            "skipped_unprocessed": 0,
            "skipped_no_segments": 0,
            "errors": 0,
        },
        "videos": [],
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
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            result["counts"]["skipped_active"] += 1
            continue

        seg_exists = session.exec(
            select(TranscriptSegment.id).where(TranscriptSegment.video_id == video.id).limit(1)
        ).first()
        if not seg_exists:
            result["counts"]["skipped_no_segments"] += 1
            continue

        result["counts"]["eligible"] += 1
        try:
            video_result = ingestion_service.repair_existing_transcript(
                session,
                int(video.id),
                save_files=True,
                persist_run=True,
                persist_snapshot=True,
                source="channel_bulk_sync",
                note=f"Bulk consolidate-transcripts for channel {channel_id}",
                trigger_semantic_index=True,
            )
            result["videos"].append(video_result)
            if video_result["changed"]:
                result["counts"]["changed"] += 1
            result["counts"]["merged_segments"] += int(video_result["merged_count"])
            result["counts"]["reassigned_islands"] += int(video_result["reassigned_islands"])
        except Exception as e:
            session.rollback()
            result["counts"]["errors"] += 1
            if len(result["sample_skips"]) < 25:
                result["sample_skips"].append({
                    "video_id": int(video.id),
                    "title": video.title,
                    "reason": f"error: {e}",
                })

    if result["counts"]["eligible"] > 0:
        _invalidate_speaker_query_caches()
    return result

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
    if jt in {"process", "diarize", "voicefixer_cleanup", "conversation_reconstruct", "transcript_repair"}:
        return "pipeline"
    if jt in {"funny_detect", "funny_explain"}:
        return "funny"
    if jt in {"youtube_metadata", "episode_clone"}:
        return "youtube"
    if jt in {"clip_export_mp4", "clip_export_captions"}:
        return "clip"
    return "other"


def _sync_auxiliary_video_job_state(session: Session, job: Job, state: str) -> None:
    video_id = int(getattr(job, "video_id", 0) or 0)
    if video_id <= 0:
        return
    video = session.get(Video, video_id)
    if not video:
        return

    jt = str(getattr(job, "job_type", "") or "").strip().lower()
    normalized = str(state or "").strip().lower()

    if jt == "voicefixer_cleanup":
        if normalized == "paused":
            video.voicefixer_status = "paused"
            video.voicefixer_error = None
        elif normalized == "queued":
            video.voicefixer_status = "queued"
            video.voicefixer_error = None
        elif normalized == "cleared":
            if str(getattr(video, "voicefixer_cleaned_path", "") or "").strip():
                apply_scope = str(getattr(video, "voicefixer_apply_scope", "none") or "none").strip().lower()
                video.voicefixer_status = "ready" if apply_scope != "none" else "disabled"
            else:
                video.voicefixer_status = None
            video.voicefixer_error = None
        session.add(video)
        return

    if jt == "conversation_reconstruct":
        if normalized == "paused":
            video.reconstruction_status = "paused"
            video.reconstruction_error = None
        elif normalized == "queued":
            video.reconstruction_status = "queued"
            video.reconstruction_error = None
        elif normalized == "cleared":
            if str(getattr(video, "reconstruction_audio_path", "") or "").strip():
                video.reconstruction_status = "ready"
            else:
                video.reconstruction_status = None
            video.reconstruction_error = None
        session.add(video)
        return


def _detach_job_from_transcript_campaign_items(session: Session, job: Job, *, deleted_from_queue: bool = False) -> None:
    job_id = int(getattr(job, "id", 0) or 0)
    if job_id <= 0:
        return
    items = session.exec(
        select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.job_id == job_id)
    ).all()
    if not items:
        return
    for item in items:
        item.job_id = None
        if deleted_from_queue:
            status_value = str(getattr(item, "status", "") or "").strip().lower()
            if status_value in {"pending", "queued", "paused", "running"}:
                item.status = "cleared"
        item.updated_at = datetime.now()
        session.add(item)


def _invalidate_voicefixer_output(video: Video) -> None:
    apply_scope = str(getattr(video, "voicefixer_apply_scope", "none") or "none").strip().lower()
    video.voicefixer_cleaned_path = None
    video.voicefixer_use_cleaned = False
    video.voicefixer_status = "disabled" if apply_scope == "none" else None
    video.voicefixer_error = None


def _invalidate_reconstruction_output(video: Video) -> None:
    video.reconstruction_audio_path = None
    video.reconstruction_use_for_playback = False
    video.reconstruction_status = None
    video.reconstruction_error = None


def _mark_job_cancelled(session: Session, job: Job, *, detail: str = "Cancelled by user.") -> None:
    if not job:
        return
    if str(getattr(job, "status", "") or "").strip().lower() == "cancelled":
        return
    job.status = "cancelled"
    job.status_detail = None
    job.error = detail
    job.completed_at = datetime.now()
    session.add(job)

@app.get("/jobs", response_model=List[JobRead])
def read_jobs(
    status: Optional[str] = None,
    channel_id: Optional[int] = None,
    video_id: Optional[int] = None,
    job_type: Optional[str] = None,
    limit: int = 500,
    sort_by: Optional[str] = None,
    sort_dir: Optional[str] = "desc",
    session: Session = Depends(get_session)
):
    from sqlalchemy import case, asc, desc
    from sqlalchemy.orm import selectinload
    
    query = select(Job).options(selectinload(Job.video))
    active_like_statuses = ["running", "downloading", "transcribing", "diarizing"]
    
    if status:
        statuses = [s.strip() for s in str(status).split(",") if s.strip()]
        if len(statuses) == 1:
            query = query.where(Job.status == statuses[0])
        elif len(statuses) > 1:
            query = query.where(Job.status.in_(statuses))
    else:
        query = query.where(Job.status != "waiting_diarize")
        
    if job_type:
        job_types = [jt.strip() for jt in str(job_type).split(",") if jt.strip()]
        if len(job_types) == 1:
            query = query.where(Job.job_type == job_types[0])
        elif len(job_types) > 1:
            query = query.where(Job.job_type.in_(job_types))
            
    if channel_id:
        query = query.join(Video, Job.video_id == Video.id).where(Video.channel_id == channel_id)
    if video_id:
        query = query.where(Job.video_id == video_id)

    # Determine sorting behavior
    s_dir = asc if sort_dir == "asc" else desc

    if sort_by == "duration":
        # we need the video join to sort by duration
        if not channel_id:
            query = query.join(Video, Job.video_id == Video.id)
        # Pin active jobs first, then sort by duration
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Video.duration)
            )
        else:
            query = query.order_by(s_dir(Video.duration))
            
    elif sort_by == "name":
        # we need the video join to sort by title
        if not channel_id:
            query = query.join(Video, Job.video_id == Video.id)
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Video.title)
            )
        else:
            query = query.order_by(s_dir(Video.title))
            
    else:
        # Default sort by created_at (Order Added)
        if not status:
            query = query.order_by(
                case((Job.status.in_(active_like_statuses), 0), else_=1),
                s_dir(Job.created_at)
            )
        else:
            query = query.order_by(s_dir(Job.created_at))

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
        "diarize_auto_start_threshold": int(os.getenv("DIARIZE_AUTO_START_THRESHOLD", "0")),
    }


@app.get("/jobs/pipeline/focus", response_model=PipelineFocusRead)
def get_pipeline_focus(session: Session = Depends(get_session)):
    counts = _compute_pipeline_focus_counts(session)
    return {
        "mode": ingestion_service.get_pipeline_focus_mode(),
        "execution_mode": ingestion_service.get_pipeline_execution_mode(),
        **counts,
    }


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
    return {
        "mode": mode,
        "execution_mode": ingestion_service.get_pipeline_execution_mode(),
        **counts,
        "active_transcription_paused": paused_active_count,
    }


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
    _sync_auxiliary_video_job_state(session, job, "paused")
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
    _sync_auxiliary_video_job_state(session, job, "queued")
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
    from sqlalchemy import func
    from sqlalchemy.exc import OperationalError

    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            clearable_statuses = ["queued", "paused", "waiting_diarize"]
            jobs = session.exec(
                select(Job).where(Job.status.in_(clearable_statuses))
            ).all()
            deleted_count = len(jobs)
            if deleted_count <= 0:
                return {"deleted": 0}

            for job in jobs:
                _detach_job_from_transcript_campaign_items(session, job, deleted_from_queue=True)
                _sync_auxiliary_video_job_state(session, job, "cleared")
                session.delete(job)
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
    """Delete completed, failed, cancelled, and waiting_diarize jobs.

    Skips waiting_diarize jobs that still have an active child diarize job
    (queued/running/diarizing) to avoid orphaning in-flight work.
    """
    active_child_statuses = {"queued", "running", "diarizing"}
    active_child_parent_ids: set[int] = set()
    active_children = session.exec(
        select(Job).where(Job.job_type == "diarize", Job.status.in_(active_child_statuses))
    ).all()
    for child in active_children:
        try:
            payload = json.loads(child.payload_json) if child.payload_json else {}
            pid = payload.get("parent_job_id")
            if pid:
                active_child_parent_ids.add(int(pid))
        except Exception:
            pass

    jobs = session.exec(
        select(Job).where(Job.status.in_(["completed", "failed", "cancelled", "waiting_diarize"]))
    ).all()
    count = 0
    for job in jobs:
        if job.status == "waiting_diarize" and job.id in active_child_parent_ids:
            continue
        _detach_job_from_transcript_campaign_items(session, job, deleted_from_queue=False)
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
    if old_job.status not in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Can only resubmit completed, failed, or cancelled jobs")
    
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
    jt = str(job.job_type or "").strip().lower()
    _sync_auxiliary_video_job_state(session, job, "cleared")

    # Reset video status only for the main transcript pipeline jobs.
    if job.video and jt in {"process", "diarize"}:
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
                _mark_job_cancelled(session, parent_job)
    elif job.job_type == "process" and job.status == "waiting_diarize":
        child_job_id = int(payload.get("diarize_job_id") or 0)
        if child_job_id:
            child_job = session.get(Job, child_job_id)
            if child_job:
                _mark_job_cancelled(session, child_job)

    _mark_job_cancelled(session, job)
    session.commit()
    return {"status": "cancelled", "video_status": job.video.status if job.video else "unknown"}

@app.get("/settings", response_model=Settings)
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

@app.post("/settings")
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
    normalized_ollama_model = _normalize_ollama_model_ref(getattr(settings, "ollama_model", "") or "")

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
    return ingestion_service.test_transcription_engine(engine, whisper_backend_override=request.whisper_backend)

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


@app.post("/youtube/data-api/test", response_model=YouTubeDataApiTestResult)
def youtube_data_api_test(body: Optional[YouTubeDataApiTestRequest] = None):
    api_key = str(getattr(body, "api_key", "") or os.getenv("YOUTUBE_DATA_API_KEY") or "").strip()
    test_video_id = "dQw4w9WgXcQ"
    try:
        data = _youtube_data_api_key_request(
            "/videos",
            api_key=api_key,
            query={
                "part": "snippet,statistics",
                "id": test_video_id,
                "maxResults": 1,
            },
            timeout=30,
        )
        items = data.get("items") or []
        if not items:
            return YouTubeDataApiTestResult(
                status="error",
                error="YouTube Data API call succeeded, but no test video metadata was returned.",
            )
        item = items[0] or {}
        snippet = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        view_count = stats.get("viewCount")
        try:
            parsed_view_count = int(view_count) if view_count is not None else None
        except Exception:
            parsed_view_count = None
        return YouTubeDataApiTestResult(
            status="ok",
            video_id=str(item.get("id") or test_video_id),
            title=str(snippet.get("title") or "").strip() or None,
            view_count=parsed_view_count,
        )
    except Exception as e:
        return YouTubeDataApiTestResult(status="error", error=str(e))

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
