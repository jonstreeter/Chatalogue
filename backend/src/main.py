from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timedelta
from sqlmodel import Session, select
from sqlalchemy import func
from pathlib import Path
from collections import deque
import shutil
import threading
import atexit
import os
import sys
import json
import time
import html
import re
import logging
import subprocess
import ipaddress
import socket
import urllib.parse
import urllib.request
import urllib.error
from dotenv import load_dotenv
from .env_utils import _set_env_persist
from .paths import (
    BACKEND_RUNTIME_DIR,
    IMAGES_DIR,
    MANUAL_MEDIA_DIR,
    THUMBNAILS_DIR,
)
from .youtube_utils import (
    YOUTUBE_API_BASE_URL,
    YOUTUBE_OAUTH_TOKEN_URL,
)
from .schemas import (
    ClearVoiceInstallInfo, VoiceFixerInstallInfo, ReconstructionInstallInfo, ExternalShareAuditEntry,
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

from .db.database import create_db_and_tables, IS_POSTGRES, Channel, Video, Speaker, TranscriptSegment, Clip
from .services.ingestion import IngestionService

ingestion_service = None
worker_threads: dict[str, threading.Thread] = {}
prefetch_thread = None

CLEARVOICE_PACKAGE_SPEC = "clearvoice==0.1.2"
VOICEFIXER_PACKAGE_SPEC = "voicefixer==0.1.3"
RECONSTRUCTION_PACKAGE_SPEC = "qwen-tts==0.1.1"
backend_instance_lock: FileLock | None = None
BACKEND_INSTANCE_LOCK_PATH = BACKEND_RUNTIME_DIR / "backend.instance.lock"
BACKEND_INSTANCE_INFO_PATH = BACKEND_RUNTIME_DIR / "backend.instance.json"

# Ensure image directory exists
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


def _require_local_operator(request: Request) -> None:
    client_ip = _resolve_client_ip(request)
    if not _is_loopback_host(client_ip):
        raise HTTPException(status_code=403, detail="This operation is only available from the local machine.")


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

    from sqlalchemy import case, or_

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


from .routers import jobs as jobs_routes  # noqa: E402

app.include_router(jobs_routes.router)

from .routers import settings as settings_routes  # noqa: E402

app.include_router(settings_routes.router)

from .routers import system as system_routes  # noqa: E402

app.include_router(system_routes.router)

from .routers import youtube_auth as youtube_auth_routes  # noqa: E402

app.include_router(youtube_auth_routes.router)

from .routers import speakers as speakers_routes  # noqa: E402

app.include_router(speakers_routes.router)

from .routers import channels as channels_routes  # noqa: E402

app.include_router(channels_routes.router)

from .routers import clips as clips_routes  # noqa: E402

app.include_router(clips_routes.router)

from .routers import share as share_routes  # noqa: E402

app.include_router(share_routes.router)

from .routers import episode_chat as episode_chat_routes  # noqa: E402

app.include_router(episode_chat_routes.router)

from .routers import transcript_ops as transcript_ops_routes  # noqa: E402

app.include_router(transcript_ops_routes.router)

from .routers import episode_clone as episode_clone_routes  # noqa: E402

app.include_router(episode_clone_routes.router)

from .routers import video_media as video_media_routes  # noqa: E402

app.include_router(video_media_routes.router)

from .routers import videos as videos_routes  # noqa: E402

app.include_router(videos_routes.router)

from .routers import search as search_routes  # noqa: E402

app.include_router(search_routes.router)

from .routers import segments as segments_routes  # noqa: E402

app.include_router(segments_routes.router)

from .routers import video_maintenance as video_maintenance_routes  # noqa: E402

app.include_router(video_maintenance_routes.router)

from .routers import avatars as avatars_routes  # noqa: E402

app.include_router(avatars_routes.router)
