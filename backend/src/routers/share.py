"""External share (LAN/tunnel) session endpoints."""
import secrets
import subprocess
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Request

from ..schemas import ExternalShareAuditEntry, ExternalShareStartRequest, ExternalShareStatus

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/share/public-status")
def get_public_share_status(token: Optional[str] = None):
    _main()._ensure_external_share_not_expired()
    snapshot = _main()._snapshot_external_share_state(include_secrets=True)
    expected_token = str(snapshot.get("token") or "")
    if not snapshot.get("active") or not token or str(token).strip() != expected_token:
        return {"active": False, "password_required": False}
    return {
        "active": True,
        "password_required": bool(snapshot.get("password_required")),
        "expires_at": snapshot.get("expires_at"),
        "share_url": snapshot.get("share_url"),
    }


@router.get("/share/launch/{token}")
def launch_external_share(token: str, request: Request):
    _main()._ensure_external_share_not_expired()
    snapshot = _main()._snapshot_external_share_state(include_secrets=True)
    expected_token = str(snapshot.get("token") or "").strip()
    client_ip = _main()._resolve_client_ip(request)

    if not snapshot.get("active") or not expected_token or str(token).strip() != expected_token:
        _main()._append_share_audit(action="share_launch_denied", allowed=False, reason="invalid_or_inactive", client_ip=client_ip, path=request.url.path)
        return _main()._render_share_launch_page(
            destination_url=None,
            title="Share Link Unavailable",
            message="This share link is invalid or the share session has already ended.",
            status_code=404,
        )

    api_url = snapshot.get("api_public_url") or snapshot.get("api_lan_url") or snapshot.get("api_local_url")
    frontend_url = snapshot.get("frontend_public_url") or snapshot.get("frontend_lan_url") or snapshot.get("frontend_local_url")
    destination_url = _main()._build_share_destination_url(frontend_url, api_url, expected_token)
    if not destination_url:
        _main()._append_share_audit(action="share_launch_denied", allowed=False, reason="missing_destination", client_ip=client_ip, path=request.url.path)
        return _main()._render_share_launch_page(
            destination_url=None,
            title="Share Link Unavailable",
            message="This share session does not currently have a valid destination URL.",
            status_code=500,
        )

    _main()._append_share_audit(action="share_launch_allowed", allowed=True, reason="ok", client_ip=client_ip, path=request.url.path)
    return _main()._render_share_launch_page(
        destination_url=destination_url,
        title="Opening Shared Chatalogue",
        message="Redirecting you to the shared Chatalogue session.",
        status_code=200,
    )


@router.get("/share/status", response_model=ExternalShareStatus)
def get_external_share_status(request: Request):
    _main()._ensure_external_share_not_expired()
    _main()._require_local_operator(request)
    _main()._refresh_cloudflared_availability()
    snapshot = _main()._snapshot_external_share_state()
    return ExternalShareStatus(**snapshot)


@router.get("/share/audit", response_model=List[ExternalShareAuditEntry])
def get_external_share_audit(request: Request):
    _main()._ensure_external_share_not_expired()
    _main()._require_local_operator(request)
    return [ExternalShareAuditEntry(**entry) for entry in list(_main().external_share_audit_entries)]


@router.post("/share/start", response_model=ExternalShareStatus)
def start_external_share(req: ExternalShareStartRequest, request: Request):
    _main()._require_local_operator(request)
    _main()._ensure_external_share_not_expired()

    duration_minutes = max(5, min(int(req.duration_minutes or 60), 24 * 60))
    frontend_port = max(1, min(int(req.frontend_port or 5173), 65535))
    backend_port = max(1, min(int(req.backend_port or 8011), 65535))
    enable_tunnel = bool(req.enable_tunnel)
    allowlist = _main()._parse_allowlist(req.ip_allowlist)
    password = str(req.password or "").strip()
    token = secrets.token_urlsafe(24)
    frontend_local_url = f"http://127.0.0.1:{frontend_port}"
    api_local_url = f"http://127.0.0.1:{backend_port}"
    lan_host = _main()._resolve_lan_host()
    frontend_lan_url = f"http://{lan_host}:{frontend_port}" if lan_host else None
    api_lan_url = f"http://{lan_host}:{backend_port}" if lan_host else None
    if not enable_tunnel and (not frontend_lan_url or not api_lan_url):
        raise RuntimeError("Could not determine a LAN IP for this machine. Set CHATALOGUE_SHARE_LAN_HOST to the desired local network address and try again.")

    with _main().external_share_lock:
        if _main().external_share_state.get("active"):
            _main()._stop_external_share_locked(reason="restarted")

    frontend_public_url = None
    api_public_url = None
    processes: dict[str, subprocess.Popen] = {}
    tunnel_provider = None
    if enable_tunnel:
        try:
            frontend_proc, frontend_public_url = _main()._start_cloudflared_quick_tunnel(frontend_local_url, "frontend")
            processes["frontend"] = frontend_proc
            api_proc, api_public_url = _main()._start_cloudflared_quick_tunnel(api_local_url, "api")
            processes["api"] = api_proc
            tunnel_provider = "cloudflared"
        except Exception:
            for proc in processes.values():
                _main()._terminate_process(proc)
            raise

    mode = "public_tunnel" if enable_tunnel else "lan"
    share_url = _main()._build_share_launch_url(
        api_public_url or api_lan_url or api_local_url,
        token,
    )

    started_at = _main()._utc_now()
    expires_at = started_at + timedelta(minutes=duration_minutes)
    with _main().external_share_lock:
        _main().external_share_state.update({
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
    _main()._append_share_event(f"External share started. mode={mode} tunnel={enable_tunnel} expires_at={expires_at.isoformat()}")
    _main()._append_share_audit(action="share_started", allowed=True, reason="ok", client_ip=_main()._resolve_client_ip(request), path="/share/start")
    return ExternalShareStatus(**_main()._snapshot_external_share_state())


@router.post("/share/stop", response_model=ExternalShareStatus)
def stop_external_share(request: Request):
    _main()._require_local_operator(request)
    with _main().external_share_lock:
        _main()._stop_external_share_locked(reason="manual_stop")
    _main()._append_share_audit(action="share_stopped", allowed=True, reason="manual_stop", client_ip=_main()._resolve_client_ip(request), path="/share/stop")
    return ExternalShareStatus(**_main()._snapshot_external_share_state())
