"""YouTube Data API / OAuth helpers: token refresh, metadata updates, uploads."""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..db.database import Channel, Clip, Video
from ..env_utils import _set_env_persist
from ..video_utils import _apply_ytdlp_auth_opts
from ..youtube_utils import (
    YOUTUBE_API_BASE_URL,
    YOUTUBE_OAUTH_TOKEN_URL,
)

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
