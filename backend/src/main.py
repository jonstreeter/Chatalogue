from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
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
from .paths import (
    BACKEND_RUNTIME_DIR,
    IMAGES_DIR,
    MANUAL_MEDIA_DIR,
    THUMBNAILS_DIR,
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

from .db.database import create_db_and_tables
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
