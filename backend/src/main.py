from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import threading
import atexit
import os
import json
import logging
from dotenv import load_dotenv
from .paths import (
    BACKEND_RUNTIME_DIR,
    IMAGES_DIR,
    MANUAL_MEDIA_DIR,
    THUMBNAILS_DIR,
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
from .services import external_share as share

ingestion_service = None
worker_threads: dict[str, threading.Thread] = {}
prefetch_thread = None

backend_instance_lock: FileLock | None = None
BACKEND_INSTANCE_LOCK_PATH = BACKEND_RUNTIME_DIR / "backend.instance.lock"
BACKEND_INSTANCE_INFO_PATH = BACKEND_RUNTIME_DIR / "backend.instance.json"

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
    share._ensure_external_share_not_expired()

    path = request.url.path or "/"
    if path.startswith("/share/public-status") or path.startswith("/share/launch/"):
        return await call_next(request)

    with share.external_share_lock:
        active = bool(share.external_share_state.get("active"))
        expected_token = str(share.external_share_state.get("token") or "")
        expected_password = str(share.external_share_state.get("password") or "")
        allowlist = list(share.external_share_state.get("ip_allowlist") or [])

    if not active:
        return await call_next(request)

    client_ip = share._resolve_client_ip(request)
    if share._is_loopback_host(client_ip) or request.method.upper() == "OPTIONS":
        return await call_next(request)

    if not share._client_ip_allowed(client_ip, allowlist):
        share._append_share_audit(action="request_denied", allowed=False, reason="ip_not_allowed", client_ip=client_ip, path=path)
        return share._external_share_error_response(request, 403, "This IP is not permitted for the current external share session.", "share_ip_not_allowed")

    provided_token, provided_password = share._get_external_share_credentials(request)
    if not expected_token or provided_token != expected_token:
        share._append_share_audit(action="request_denied", allowed=False, reason="invalid_token", client_ip=client_ip, path=path)
        return share._external_share_error_response(request, 401, "A valid share token is required.", "share_token_required")

    if expected_password and provided_password != expected_password:
        share._append_share_audit(action="request_denied", allowed=False, reason="invalid_password", client_ip=client_ip, path=path)
        return share._external_share_error_response(request, 401, "A valid share password is required.", "share_password_required")

    response = await call_next(request)
    if provided_token and request.cookies.get(share.SHARE_COOKIE_TOKEN) != provided_token:
        response.set_cookie(share.SHARE_COOKIE_TOKEN, provided_token, httponly=True, samesite="lax")
    if expected_password and provided_password and request.cookies.get(share.SHARE_COOKIE_PASSWORD) != provided_password:
        response.set_cookie(share.SHARE_COOKIE_PASSWORD, provided_password, httponly=True, samesite="lax")
    share._append_share_audit(action="request_allowed", allowed=True, reason="ok", client_ip=client_ip, path=path)
    return response

# Mount static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")
app.mount("/manual-media", StaticFiles(directory=MANUAL_MEDIA_DIR), name="manual-media")


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
