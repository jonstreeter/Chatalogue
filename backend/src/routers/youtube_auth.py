"""YouTube OAuth and Data API test endpoints."""
import html
import os
import secrets
import time
from datetime import datetime, timedelta
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..env_utils import _set_env_persist
from ..schemas import YouTubeDataApiTestRequest, YouTubeDataApiTestResult
from ..youtube_utils import (
    YOUTUBE_OAUTH_AUTH_URL,
    YOUTUBE_OAUTH_SCOPE,
)

router = APIRouter()

# Pending OAuth state tokens -> expiry timestamps (owned by this router).
youtube_oauth_pending_states: dict[str, float] = {}


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/youtube/oauth/status")
def youtube_oauth_status():
    cfg = _main()._youtube_get_cfg()
    expiry = _main()._youtube_parse_expiry(cfg["token_expiry"])
    connected = bool(cfg["refresh_token"] or cfg["access_token"])
    return {
        "configured": _main()._youtube_oauth_is_configured(),
        "connected": connected,
        "channel_id": cfg["channel_id"] or None,
        "channel_title": cfg["channel_title"] or None,
        "redirect_uri": cfg["redirect_uri"],
        "scope": YOUTUBE_OAUTH_SCOPE,
        "token_expires_at": expiry.isoformat() if expiry else None,
        "push_enabled": cfg["push_enabled"],
    }


@router.post("/youtube/oauth/start")
def youtube_oauth_start():
    cfg = _main()._youtube_get_cfg()
    if not _main()._youtube_oauth_is_configured():
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


@router.get("/auth/youtube/callback")
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
        token_data = _main()._youtube_exchange_code_for_tokens(code)
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

        ch_info = _main()._youtube_fetch_authenticated_channel_info()
        _set_env_persist("YOUTUBE_OAUTH_CHANNEL_ID", ch_info["channel_id"])
        _set_env_persist("YOUTUBE_OAUTH_CHANNEL_TITLE", ch_info["channel_title"])

        msg = f'Connected to YouTube channel "{ch_info["channel_title"]}" ({ch_info["channel_id"]}).'
        return HTMLResponse(content=_html_page("YouTube Connected", msg, True), status_code=200)
    except Exception as e:
        return HTMLResponse(content=_html_page("YouTube OAuth Error", str(e), False), status_code=500)


@router.post("/youtube/oauth/disconnect")
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


@router.post("/youtube/oauth/test")
def youtube_oauth_test():
    try:
        info = _main()._youtube_fetch_authenticated_channel_info()
        return {"status": "ok", "channel_id": info["channel_id"], "channel_title": info["channel_title"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/youtube/data-api/test", response_model=YouTubeDataApiTestResult)
def youtube_data_api_test(body: Optional[YouTubeDataApiTestRequest] = None):
    api_key = str(getattr(body, "api_key", "") or os.getenv("YOUTUBE_DATA_API_KEY") or "").strip()
    test_video_id = "dQw4w9WgXcQ"
    try:
        data = _main()._youtube_data_api_key_request(
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
