"""External share sessions: state, credential/IP guards, cloudflared tunnels, audit log.

Extracted from main.py. The FastAPI middleware in main.py and the share/system
routers all operate on this module's state via module-attribute access.
"""
import html
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..paths import BACKEND_DIR, BACKEND_RUNTIME_DIR
from ..schemas import ExternalShareAuditEntry

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
        str(BACKEND_DIR / "bin" / "cloudflared.exe"),
        str(BACKEND_DIR / "bin" / "cloudflared"),
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
