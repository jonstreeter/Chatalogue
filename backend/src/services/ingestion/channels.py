"""Channel CRUD/refresh/sync: YouTube + TikTok metadata, monitoring loop, date backfill.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import yt_dlp
import os
import time
import json
import threading
import re
import html
import urllib.request
import urllib.error
import urllib.parse
from sqlmodel import Session, select, func
from datetime import datetime

from ...db.database import Video, Channel, TranscriptSegment, Job
from ..logger import log, log_verbose
from . import runtime
from .runtime import (
    YOUTUBE_DATA_API_BASE_URL,
)


class ChannelSyncMixin:
    def _extract_channel_artwork(self, info: dict | None) -> tuple[str | None, str | None]:
        """Best-effort extraction of channel icon/banner URLs from yt-dlp metadata."""
        if not isinstance(info, dict):
            return None, None

        def _valid_url(v):
            return isinstance(v, str) and v.startswith(("http://", "https://"))

        icon_candidates: list[tuple[int, str]] = []
        header_candidates: list[tuple[int, str]] = []

        for key in ["channel_thumbnail", "channel_favicon", "uploader_avatar", "avatar", "thumbnail"]:
            v = info.get(key)
            if _valid_url(v):
                icon_candidates.append((10_000_000, v))
                break

        for key in ["banner", "channel_banner", "header_image", "artwork"]:
            v = info.get(key)
            if _valid_url(v):
                header_candidates.append((10_000_000, v))
                break

        for t in (info.get("thumbnails") or []):
            if not isinstance(t, dict):
                continue
            url = t.get("url")
            if not _valid_url(url):
                continue
            try:
                w = int(t.get("width") or 0)
                h = int(t.get("height") or 0)
            except Exception:
                w, h = 0, 0
            area = w * h if (w > 0 and h > 0) else 0
            ratio = (w / h) if (w > 0 and h > 0) else None
            tid = str(t.get("id") or "").lower()

            if "avatar" in tid or "icon" in tid:
                icon_candidates.append((area or 1, url))
            if "banner" in tid or "header" in tid or "cover" in tid:
                header_candidates.append((area or 1, url))

            if ratio is not None:
                if 0.8 <= ratio <= 1.25:
                    icon_candidates.append((area or 1, url))
                elif ratio >= 2.0:
                    header_candidates.append((area or 1, url))

        icon_url = max(icon_candidates, key=lambda x: x[0])[1] if icon_candidates else None
        header_url = max(header_candidates, key=lambda x: x[0])[1] if header_candidates else None
        return icon_url, header_url

    def _update_channel_metadata_from_ydl(self, channel: Channel, info: dict | None) -> bool:
        """Update channel display metadata from yt-dlp info. Returns True if changed."""
        if not isinstance(info, dict):
            return False

        changed = False
        for candidate in [info.get("channel"), info.get("uploader"), info.get("playlist_uploader"), info.get("playlist_title")]:
            if isinstance(candidate, str) and candidate.strip():
                name = candidate.strip()
                if channel.name != name and (not channel.name or channel.name.startswith("Unknown") or len(channel.name.strip()) < 3):
                    channel.name = name
                    changed = True
                break

        icon_url, header_url = self._extract_channel_artwork(info)
        if icon_url and getattr(channel, "icon_url", None) != icon_url:
            channel.icon_url = icon_url
            changed = True
        if header_url and getattr(channel, "header_image_url", None) != header_url:
            channel.header_image_url = header_url
            changed = True

        return changed

    def _extract_published_at_from_info(self, info: dict | None) -> datetime | None:
        """Parse best-available publish datetime from yt-dlp metadata."""
        if not isinstance(info, dict):
            return None

        direct_value = info.get("published_at")
        if isinstance(direct_value, datetime):
            return direct_value
        if direct_value not in (None, ""):
            text = str(direct_value).strip()
            if text:
                try:
                    return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass

        def _parse_yyyymmdd(value) -> datetime | None:
            text = str(value or "").strip()
            if len(text) != 8 or not text.isdigit():
                return None
            try:
                return datetime.strptime(text, "%Y%m%d")
            except ValueError:
                return None

        parsed = _parse_yyyymmdd(info.get("upload_date")) or _parse_yyyymmdd(info.get("release_date"))
        if parsed:
            return parsed

        for key in ("release_timestamp", "timestamp"):
            raw = info.get(key)
            if raw in (None, ""):
                continue
            try:
                ts = float(raw)
            except Exception:
                continue
            if ts <= 0:
                continue
            # Some extractors return milliseconds instead of seconds.
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            try:
                return datetime.fromtimestamp(ts)
            except Exception:
                continue
        return None

    def add_channel(self, url: str) -> Channel:
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'playlistend': 1 
        }
        ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="add_channel")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            channel_name = info.get('uploader') or info.get('channel') or "Unknown Channel"
            icon_url, header_image_url = self._extract_channel_artwork(info)
            
        with Session(runtime.engine) as session:
            existing = session.exec(select(Channel).where(Channel.url == url)).first()
            if existing:
                return existing
            
            channel = Channel(
                url=url,
                name=channel_name,
                icon_url=icon_url,
                header_image_url=header_image_url,
                last_updated=datetime.now(),
            )
            session.add(channel)
            session.commit()
            session.refresh(channel)
            return channel

    def create_manual_channel(self, name: str) -> Channel:
        clean_name = " ".join((name or "").strip().split())
        if not clean_name:
            raise ValueError("Channel name is required")

        safe_slug = self.sanitize_filename(clean_name).replace(" ", "_").lower()
        manual_url = f"manual://channel/{safe_slug}"

        with Session(runtime.engine) as session:
            existing = session.exec(select(Channel).where(Channel.url == manual_url)).first()
            if existing:
                return existing

            channel = Channel(
                url=manual_url,
                name=clean_name,
                source_type="manual",
                last_updated=datetime.now(),
                status="active",
            )
            session.add(channel)
            session.commit()
            session.refresh(channel)
            return channel

    def create_tiktok_channel(self, name: str | None = None, url: str | None = None) -> Channel:
        clean_name = " ".join((name or "").strip().split())
        normalized_url = " ".join((url or "").strip().split())
        if normalized_url and "tiktok.com" not in normalized_url.lower():
            raise ValueError("TikTok channel URL must point to tiktok.com")

        derived_handle = ""
        if normalized_url:
            match = re.search(r"tiktok\.com/@([^/?#]+)", normalized_url, re.IGNORECASE)
            if match:
                derived_handle = html.unescape(match.group(1)).strip()

        if not clean_name and derived_handle:
            clean_name = derived_handle if derived_handle.startswith("@") else f"@{derived_handle}"
        if not clean_name:
            raise ValueError("TikTok creator name or profile URL is required")

        safe_slug = self.sanitize_filename(clean_name).replace(" ", "_").lower()
        channel_url = normalized_url or f"tiktok://channel/{safe_slug}"

        with Session(runtime.engine) as session:
            existing = session.exec(select(Channel).where(Channel.url == channel_url)).first()
            if existing:
                return existing

            channel = Channel(
                url=channel_url,
                name=clean_name,
                source_type="tiktok",
                last_updated=datetime.now(),
                status="active",
            )
            session.add(channel)
            session.commit()
            session.refresh(channel)
            return channel

    def _update_channel_sync_progress(
        self,
        channel_id: int,
        *,
        status: str | None = None,
        detail: str | None = None,
        progress: int | None = None,
        completed_items: int | None = None,
        total_items: int | None = None,
    ) -> None:
        with Session(runtime.engine) as session:
            channel = session.get(Channel, channel_id)
            if not channel:
                return
            if status is not None:
                channel.status = status
            if detail is not None:
                channel.sync_status_detail = detail
            if progress is not None:
                channel.sync_progress = max(0, min(100, int(progress)))
            if completed_items is not None:
                channel.sync_completed_items = max(0, int(completed_items))
            if total_items is not None:
                channel.sync_total_items = max(0, int(total_items))
            session.add(channel)
            session.commit()

    def _make_tiktok_video_key(self, raw_id: str | None) -> str:
        text = str(raw_id or "").strip()
        return f"tiktok_{text}" if text else f"tiktok_{int(time.time())}"

    def _resolve_tiktok_entry_url(self, channel_url: str, info: dict | None) -> str | None:
        if not isinstance(info, dict):
            return None
        for key in ("webpage_url", "url", "original_url"):
            value = str(info.get(key) or "").strip()
            if value.startswith(("http://", "https://")):
                video_match = re.search(r"tiktok\.com/@([^/?#]+)/video/(\d+)", value, re.IGNORECASE)
                if video_match:
                    return f"https://www.tiktok.com/@{video_match.group(1)}/video/{video_match.group(2)}"
                return value
        entry_id = str(info.get("id") or "").strip()
        base = str(channel_url or "").strip().rstrip("/")
        if entry_id and "tiktok.com/@" in base:
            return f"{base}/video/{entry_id}"
        return None

    def _fetch_remote_video_info(self, url: str, *, purpose: str = "remote_video_info") -> dict | None:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return None
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "ignoreerrors": True,
        }
        ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose=purpose)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=False)
        return info if isinstance(info, dict) else None

    def _youtube_data_api_key(self) -> str:
        return str(os.getenv("YOUTUBE_DATA_API_KEY") or "").strip()

    def _youtube_data_api_request(
        self,
        path: str,
        *,
        query: dict | None = None,
        timeout: int = 30,
    ) -> dict:
        api_key = self._youtube_data_api_key()
        if not api_key:
            raise RuntimeError("YouTube Data API key is not configured.")
        params = {k: v for k, v in (query or {}).items() if v is not None}
        params["key"] = api_key
        url = f"{YOUTUBE_DATA_API_BASE_URL}{path}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"YouTube Data API request failed: HTTP {e.code} {detail[:400]}".strip())
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Invalid YouTube Data API response: {e}")

    @staticmethod
    def _parse_youtube_iso8601_duration_seconds(value: str | None) -> int | None:
        text = str(value or "").strip().upper()
        if not text:
            return None
        match = re.fullmatch(
            r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
            text,
        )
        if not match:
            return None
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        total = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return total if total > 0 else 0

    @staticmethod
    def _parse_youtube_api_published_at(value: str | None) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _best_youtube_api_thumbnail(snippet: dict | None) -> str | None:
        thumbs = (snippet or {}).get("thumbnails") or {}
        for key in ("maxres", "standard", "high", "medium", "default"):
            entry = thumbs.get(key)
            if isinstance(entry, dict):
                url = str(entry.get("url") or "").strip()
                if url:
                    return url
        return None

    def _fetch_youtube_data_api_video_metadata_batch(self, youtube_ids: list[str]) -> dict[str, dict]:
        ids = []
        seen = set()
        for raw_id in youtube_ids or []:
            youtube_id = str(raw_id or "").strip()
            if not youtube_id or youtube_id in seen:
                continue
            seen.add(youtube_id)
            ids.append(youtube_id)
        if not ids or not self._youtube_data_api_key():
            return {}

        metadata_by_id: dict[str, dict] = {}
        for start in range(0, len(ids), 50):
            chunk = ids[start:start + 50]
            data = self._youtube_data_api_request(
                "/videos",
                query={
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(chunk),
                    "maxResults": min(50, len(chunk)),
                },
                timeout=45,
            )
            items = data.get("items") or []
            for item in items:
                video_id = str(item.get("id") or "").strip()
                if not video_id:
                    continue
                snippet = item.get("snippet") or {}
                stats = item.get("statistics") or {}
                content_details = item.get("contentDetails") or {}
                view_count = None
                raw_view_count = stats.get("viewCount")
                if raw_view_count not in (None, ""):
                    try:
                        view_count = int(raw_view_count)
                    except Exception:
                        view_count = None
                metadata_by_id[video_id] = {
                    "id": video_id,
                    "title": str(snippet.get("title") or "").strip() or None,
                    "description": snippet.get("description"),
                    "published_at": self._parse_youtube_api_published_at(snippet.get("publishedAt")),
                    "duration": self._parse_youtube_iso8601_duration_seconds(content_details.get("duration")),
                    "view_count": view_count,
                    "thumbnail": self._best_youtube_api_thumbnail(snippet),
                }
        return metadata_by_id

    def _classify_tiktok_refresh_error(self, exc: Exception) -> str:
        lowered = str(exc or "").strip().lower()
        if "your ip address is blocked" in lowered or "blocked from accessing this post" in lowered:
            return "TikTok blocked metadata access from the current IP address."
        if "too many requests" in lowered or "rate limit" in lowered:
            return "TikTok rate-limited the channel refresh. Try again later."
        if "login required" in lowered or "authentication" in lowered or "sign in" in lowered:
            return "TikTok requires an authenticated session for this channel."
        return f"TikTok refresh failed: {str(exc or 'unknown error')[:220]}"

    def _best_thumbnail_url(self, info: dict | None) -> str | None:
        if not isinstance(info, dict):
            return None
        thumbnails = info.get("thumbnails") or []
        for thumb in reversed(thumbnails):
            if isinstance(thumb, dict):
                url = str(thumb.get("url") or "").strip()
                if url:
                    return url
        for key in ("thumbnail", "thumbnail_url"):
            value = str(info.get(key) or "").strip()
            if value.startswith(("http://", "https://")):
                return value
        return None

    def _derive_tiktok_channel_artwork(self, info: dict | None, current_icon: str | None = None, current_header: str | None = None) -> tuple[str | None, str | None]:
        icon_url, header_url = self._extract_channel_artwork(info)
        best_thumb = self._best_thumbnail_url(info)

        resolved_icon = icon_url or current_icon
        resolved_header = header_url or current_header

        # TikTok profile-feed extraction often omits creator avatar/banner metadata.
        # Use the best available post thumbnail as a header fallback so channel cards
        # are not blank even when profile-level artwork is unavailable.
        if not resolved_header and best_thumb:
            resolved_header = best_thumb

        return resolved_icon, resolved_header

    def _decode_tiktok_embedded_string(self, value: str | None) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        text = text.replace("\\u002F", "/").replace("\\/", "/")
        text = text.replace("\\u0026", "&").replace("\\u003D", "=").replace("\\u0025", "%")
        text = text.replace("\\u002D", "-").replace("\\u002B", "+").replace("\\u003F", "?").replace("\\u003A", ":")
        return html.unescape(text)

    def _fetch_tiktok_profile_metadata(self, profile_url: str | None) -> dict[str, str]:
        url = str(profile_url or "").strip()
        if not url or "tiktok.com" not in url.lower():
            return {}

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        patterns = {
            "icon_url": [
                r'avatarLarger\\?":\\?"([^"]+)',
                r'avatarMedium\\?":\\?"([^"]+)',
                r'avatarThumb\\?":\\?"([^"]+)',
            ],
            "display_name": [
                r'nickname\\?":\\?"([^"]+)',
            ],
            "username": [
                r'uniqueId\\?":\\?"([^"]+)',
            ],
        }

        out: dict[str, str] = {}
        for key, candidates in patterns.items():
            for pattern in candidates:
                match = re.search(pattern, raw)
                if not match:
                    continue
                decoded = self._decode_tiktok_embedded_string(match.group(1))
                if decoded:
                    out[key] = decoded
                    break
        return out

    def _refresh_tiktok_channel(self, session: Session, channel: Channel) -> int:
        profile_url = str(channel.url or "").strip()
        if "tiktok.com" not in profile_url.lower():
            channel.status = "active"
            channel.sync_status_detail = "TikTok refresh requires a real creator/profile URL."
            channel.sync_progress = 0
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()
            return 0

        try:
            profile_meta = self._fetch_tiktok_profile_metadata(profile_url)
        except Exception as e:
            profile_meta = {}
            log_verbose(f"TikTok profile metadata fallback skipped for {channel.name}: {e}")

        profile_display_name = str(profile_meta.get("display_name") or "").strip()
        profile_username = str(profile_meta.get("username") or "").strip()
        profile_icon_url = str(profile_meta.get("icon_url") or "").strip()
        if profile_display_name and channel.name != profile_display_name:
            channel.name = profile_display_name
            session.add(channel)
        elif profile_username and (
            not channel.name
            or channel.name.startswith("@")
            or len(channel.name.strip()) < 3
        ):
            normalized_username = profile_username if profile_username.startswith("@") else f"@{profile_username}"
            if channel.name != normalized_username:
                channel.name = normalized_username
                session.add(channel)
        if profile_icon_url and getattr(channel, "icon_url", None) != profile_icon_url:
            channel.icon_url = profile_icon_url
            session.add(channel)
        if session.in_transaction():
            session.commit()

        ydl_opts = {
            "extract_flat": "in_playlist",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
        }
        new_video_count = 0

        self._update_channel_sync_progress(
            channel.id,
            status="refreshing",
            detail="Scanning TikTok profile feed...",
            progress=5,
            completed_items=0,
            total_items=1,
        )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(profile_url, download=False)
        except Exception as e:
            channel.status = "active"
            channel.sync_status_detail = self._classify_tiktok_refresh_error(e)
            channel.sync_progress = 0
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()
            return 0

        if not result:
            channel.status = "active"
            channel.sync_status_detail = "TikTok profile scan returned no results."
            channel.sync_progress = 0
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()
            return 0

        if self._update_channel_metadata_from_ydl(channel, result):
            session.add(channel)

        raw_entries = result.get("entries")
        entries = list(raw_entries) if raw_entries is not None else []
        all_entries: list[dict] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            all_entries.append(entry)

        if all_entries:
            first_entry = all_entries[0]
            if isinstance(first_entry, dict):
                for candidate in [first_entry.get("channel"), first_entry.get("uploader"), first_entry.get("playlist_title")]:
                    if isinstance(candidate, str) and candidate.strip():
                        channel_name = candidate.strip()
                        if channel.name != channel_name:
                            channel.name = channel_name
                            session.add(channel)
                        break
                icon_url, header_url = self._derive_tiktok_channel_artwork(
                    first_entry,
                    current_icon=profile_icon_url or getattr(channel, "icon_url", None),
                    current_header=getattr(channel, "header_image_url", None),
                )
                changed = False
                if icon_url and getattr(channel, "icon_url", None) != icon_url:
                    channel.icon_url = icon_url
                    changed = True
                if header_url and getattr(channel, "header_image_url", None) != header_url:
                    channel.header_image_url = header_url
                    changed = True
                if changed:
                    session.add(channel)
                    session.commit()

        log(f"TikTok scan complete. {len(all_entries)} total entries discovered for {channel.name}.")
        self._update_channel_sync_progress(
            channel.id,
            status="refreshing",
            detail=f"Importing {len(all_entries)} discovered TikTok videos...",
            progress=22,
            completed_items=0,
            total_items=len(all_entries),
        )

        caption_attempts = 0
        caption_stored = 0
        for idx, info in enumerate(all_entries, start=1):
            entry_id = str(info.get("id") or "").strip()
            if not entry_id:
                continue

            video_key = self._make_tiktok_video_key(entry_id)
            source_url = self._resolve_tiktok_entry_url(profile_url, info)
            existing_video = session.exec(select(Video).where(Video.youtube_id == video_key)).first()
            if not existing_video and source_url:
                existing_video = session.exec(select(Video).where(Video.source_url == source_url)).first()
            needs_caption_hydration = False

            if not existing_video:
                video = Video(
                    youtube_id=video_key,
                    channel_id=channel.id,
                    title=str(info.get("title") or f"TikTok {entry_id}"),
                    media_source_type="tiktok",
                    source_url=source_url,
                    media_kind="video",
                    description=info.get("description"),
                    published_at=self._extract_published_at_from_info(info),
                    duration=info.get("duration"),
                    view_count=info.get("view_count"),
                    thumbnail_url=self._best_thumbnail_url(info),
                    status="pending",
                )
                session.add(video)
                session.flush()
                needs_caption_hydration = True
                new_video_count += 1
            else:
                video = existing_video
                changed = False
                title = str(info.get("title") or "").strip()
                if title and (not existing_video.title or existing_video.title.startswith("TikTok ")):
                    existing_video.title = title
                    changed = True
                if source_url and not existing_video.source_url:
                    existing_video.source_url = source_url
                    changed = True
                if not existing_video.duration and info.get("duration"):
                    existing_video.duration = info.get("duration")
                    changed = True
                if info.get("view_count") is not None and existing_video.view_count != info.get("view_count"):
                    existing_video.view_count = info.get("view_count")
                    changed = True
                thumb = self._best_thumbnail_url(info)
                if thumb and not existing_video.thumbnail_url:
                    existing_video.thumbnail_url = thumb
                    changed = True
                pub_date = self._extract_published_at_from_info(info)
                if pub_date and not existing_video.published_at:
                    existing_video.published_at = pub_date
                    changed = True
                if not existing_video.transcript_is_placeholder:
                    seg_count = session.exec(
                        select(func.count(TranscriptSegment.id)).where(TranscriptSegment.video_id == existing_video.id)
                    ).one() or 0
                    if int(seg_count or 0) == 0:
                        needs_caption_hydration = True
                if changed:
                    session.add(existing_video)

            if needs_caption_hydration and source_url:
                try:
                    full_info = self._fetch_remote_video_info(source_url, purpose="tiktok_placeholder_captions")
                    if isinstance(full_info, dict):
                        caption_attempts += 1
                        stored = self.populate_placeholder_transcript(session, video, info=full_info)
                        if stored > 0:
                            caption_stored += 1
                except Exception as e:
                    log_verbose(f"TikTok placeholder captions unavailable for {source_url}: {e}")

            if idx == 1 or idx == len(all_entries) or idx % 25 == 0:
                self._update_channel_sync_progress(
                    channel.id,
                    status="refreshing",
                    detail=f"Importing TikTok metadata {idx}/{len(all_entries)}...",
                    progress=22 + int(idx / max(1, len(all_entries)) * 74),
                    completed_items=idx,
                    total_items=len(all_entries),
                )

        channel.last_updated = datetime.now()
        channel.status = "active"
        channel.sync_status_detail = "Channel is up to date."
        channel.sync_progress = 100
        channel.sync_total_items = len(all_entries)
        channel.sync_completed_items = len(all_entries)
        session.add(channel)
        session.commit()
        log(
            f"Refresh complete. {len(all_entries)} total on TikTok, {new_video_count} new videos added, "
            f"{caption_stored}/{caption_attempts} placeholder caption tracks stored."
        )
        return new_video_count

    def refresh_channel(self, channel_id: int):
        with Session(runtime.engine) as session:
            channel = session.get(Channel, channel_id)
            if not channel:
                raise ValueError("Channel not found")
            channel_source = (channel.source_type or "youtube").strip().lower()
            if channel_source == "manual":
                channel.status = "active"
                channel.sync_status_detail = "Manual channels do not support remote refresh."
                channel.sync_progress = 0
                channel.sync_total_items = 0
                channel.sync_completed_items = 0
                session.add(channel)
                session.commit()
                return

            channel.status = "refreshing"
            channel.sync_status_detail = "Starting channel scan..."
            channel.sync_progress = 1
            channel.sync_total_items = 0
            channel.sync_completed_items = 0
            session.add(channel)
            session.commit()
            session.refresh(channel)
            log(f"Refreshing channel: {channel.name}")

            if channel_source == "tiktok":
                return self._refresh_tiktok_channel(session, channel)

            ydl_opts = {
                'extract_flat': 'in_playlist',
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
            }
            ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="refresh_channel")
            new_video_count = 0

            # Get base URL without tab suffix
            base_url = channel.url.rstrip('/')
            for suffix in ['/videos', '/streams', '/shorts', '/live', '/featured']:
                if base_url.endswith(suffix):
                    base_url = base_url[:-len(suffix)]
                    break

            # If channel artwork is missing, do a quick channel-root metadata fetch
            # to backfill icon/banner URLs during rescan.
            if not getattr(channel, "icon_url", None) or not getattr(channel, "header_image_url", None):
                try:
                    meta_opts = {
                        'extract_flat': True,
                        'quiet': True,
                        'no_warnings': True,
                        'ignoreerrors': True,
                        'playlistend': 1,
                    }
                    meta_opts = self._apply_ytdlp_auth_opts(meta_opts, purpose="refresh_channel_metadata")
                    with yt_dlp.YoutubeDL(meta_opts) as ydl:
                        channel_info = ydl.extract_info(base_url, download=False)
                    if self._update_channel_metadata_from_ydl(channel, channel_info):
                        session.add(channel)
                        log(f"  Updated channel artwork metadata for {channel.name}")
                except Exception as e:
                    log_verbose(f"  Channel artwork metadata fetch skipped: {e}")

            # Collect entries from videos, streams, and shorts tabs.
            all_entries = []
            seen_ids = set()
            tabs_to_fetch = [f"{base_url}/videos", f"{base_url}/streams", f"{base_url}/shorts"]

            self._update_channel_sync_progress(
                channel_id,
                status="refreshing",
                detail="Discovering videos from channel tabs...",
                progress=5,
                completed_items=0,
                total_items=len(tabs_to_fetch),
            )
            for tab_idx, url in enumerate(tabs_to_fetch, start=1):
                self._update_channel_sync_progress(
                    channel_id,
                    status="refreshing",
                    detail=f"Scanning YouTube tab {tab_idx}/{len(tabs_to_fetch)}...",
                    progress=5 + int((tab_idx - 1) / max(1, len(tabs_to_fetch)) * 20),
                    completed_items=tab_idx - 1,
                    total_items=len(tabs_to_fetch),
                )
                try:
                    log(f"  Fetching tab: {url}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        result = ydl.extract_info(url, download=False)
                        if not result:
                            log(f"  WARNING: yt-dlp returned None for {url}")
                            continue

                        if self._update_channel_metadata_from_ydl(channel, result):
                            session.add(channel)

                        # entries can be a generator â€” materialize it to a list
                        raw_entries = result.get('entries')
                        if raw_entries is None:
                            log(f"  WARNING: No 'entries' key in result for {url}")
                            continue

                        entries = list(raw_entries)
                        log(f"  Raw entries from yt-dlp: {len(entries)}")

                        valid_count = 0
                        for entry in entries:
                            if not entry or not entry.get('id'):
                                continue
                            yt_id = entry['id']
                            # Skip playlist/channel entries
                            if yt_id.startswith('UC') or yt_id.startswith('PL'):
                                continue
                            # Deduplicate across tabs (a stream can appear in both)
                            if yt_id in seen_ids:
                                continue
                            seen_ids.add(yt_id)
                            all_entries.append(entry)
                            valid_count += 1
                        log(f"  Found {valid_count} valid videos from {url}")
                except Exception as e:
                    log(f"  ERROR fetching {url}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                self._update_channel_sync_progress(
                    channel_id,
                    status="refreshing",
                    detail=f"Scanned YouTube tab {tab_idx}/{len(tabs_to_fetch)}.",
                    progress=5 + int(tab_idx / max(1, len(tabs_to_fetch)) * 20),
                    completed_items=tab_idx,
                    total_items=len(tabs_to_fetch),
                )

            log(f"Total unique entries collected: {len(all_entries)}")
            api_metadata_by_id: dict[str, dict] = {}
            if all_entries and self._youtube_data_api_key():
                try:
                    self._update_channel_sync_progress(
                        channel_id,
                        status="refreshing",
                        detail=f"Enriching {len(all_entries)} videos with YouTube API metadata...",
                        progress=24,
                        completed_items=0,
                        total_items=len(all_entries),
                    )
                    api_metadata_by_id = self._fetch_youtube_data_api_video_metadata_batch(
                        [str(entry.get("id") or "").strip() for entry in all_entries]
                    )
                    log(
                        f"YouTube API metadata enrichment returned {len(api_metadata_by_id)}/{len(all_entries)} "
                        f"videos for channel {channel.name}."
                    )
                except Exception as e:
                    log(f"YouTube API metadata enrichment skipped for channel {channel.name}: {e}")
            self._update_channel_sync_progress(
                channel_id,
                status="refreshing",
                detail=f"Importing {len(all_entries)} discovered videos...",
                progress=28,
                completed_items=0,
                total_items=len(all_entries),
            )

            # Process all collected entries
            for idx, info in enumerate(all_entries, start=1):
                yt_id = info.get('id')
                if not yt_id:
                    continue

                api_meta = api_metadata_by_id.get(str(yt_id))
                pub_date = self._extract_published_at_from_info(api_meta or info)
                thumbnail_url = (api_meta or {}).get("thumbnail") or self._best_thumbnail_url(info)
                title = (api_meta or {}).get("title") or info.get('title') or 'Unknown Title'
                description = (api_meta or {}).get("description") or info.get('description')
                duration = (api_meta or {}).get("duration")
                if duration is None:
                    duration = info.get('duration')
                view_count = (api_meta or {}).get("view_count")
                if view_count is None:
                    view_count = info.get('view_count')

                existing_video = session.exec(select(Video).where(Video.youtube_id == yt_id)).first()
                if not existing_video:
                    video = Video(
                        youtube_id=yt_id,
                        channel_id=channel.id,
                        title=title,
                        description=description,
                        published_at=pub_date,
                        duration=duration,
                        view_count=view_count,
                        thumbnail_url=thumbnail_url,
                        status="pending"
                    )
                    session.add(video)
                    session.flush()
                    # Surface newly discovered videos to the UI immediately
                    # instead of waiting for the full channel scan to finish.
                    session.commit()
                    new_video_count += 1
                    log(f"  + New: {info.get('title', 'Unknown')[:60]}")
                else:
                    changed = False
                    normalized_title = str(title or '').strip()
                    if normalized_title and existing_video.title != normalized_title:
                        existing_video.title = normalized_title
                        changed = True
                    if description and not existing_video.description:
                        existing_video.description = description
                        changed = True
                    if duration and not existing_video.duration:
                        existing_video.duration = duration
                        changed = True
                    if view_count is not None and existing_video.view_count != view_count:
                        existing_video.view_count = view_count
                        changed = True
                    if thumbnail_url and not existing_video.thumbnail_url:
                        existing_video.thumbnail_url = thumbnail_url
                        changed = True
                    if pub_date and not existing_video.published_at:
                        existing_video.published_at = pub_date
                        changed = True
                    if changed:
                        session.add(existing_video)
                if idx == 1 or idx == len(all_entries) or idx % 25 == 0:
                    self._update_channel_sync_progress(
                        channel_id,
                        status="refreshing",
                        detail=f"Importing video metadata {idx}/{len(all_entries)}...",
                        progress=28 + int(idx / max(1, len(all_entries)) * 42),
                        completed_items=idx,
                        total_items=len(all_entries),
                    )

            if not all_entries:
                log(f"WARNING: No video entries found for channel {channel.name}. YouTube may be blocking requests.")

            channel.last_updated = datetime.now()
            channel.status = "active"
            channel.sync_status_detail = "Initial import complete. Finishing metadata backfill..."
            channel.sync_progress = 72
            channel.sync_completed_items = len(all_entries)
            channel.sync_total_items = len(all_entries)
            session.add(channel)
            session.commit()
            log(f"Channel scan complete. {len(all_entries)} total on YouTube, {new_video_count} new videos added.")

            # Backfill missing published_at dates (extract_flat doesn't always return them).
            # First pass targets the top of the list so new channels look correct quickly.
            try:
                quick_backfill_limit = int(os.getenv("CHANNEL_DATE_BACKFILL_QUICK_ITEMS", "250"))
            except Exception:
                quick_backfill_limit = 250
            quick_backfill_limit = max(0, quick_backfill_limit)
            if quick_backfill_limit > 0:
                self._backfill_dates(
                    channel_id,
                    max_items=quick_backfill_limit,
                    progress_start=72,
                    progress_end=88,
                    detail_prefix="Backfilling publication dates and metadata",
                    status="active",
                )

            # Then continue with a larger pass so the whole channel converges.
            backfill_limit = None
            try:
                backfill_limit = int(os.getenv("CHANNEL_DATE_BACKFILL_MAX_ITEMS", "2000"))
            except Exception:
                backfill_limit = 2000
            if (
                backfill_limit <= 0
                or quick_backfill_limit <= 0
                or backfill_limit > quick_backfill_limit
            ):
                self._backfill_dates(
                    channel_id,
                    max_items=backfill_limit,
                    progress_start=88,
                    progress_end=98,
                    detail_prefix="Finishing metadata backfill",
                    status="active",
                )

            channel = session.get(Channel, channel_id)
            if channel:
                channel.status = "active"
                channel.last_updated = datetime.now()
                channel.sync_status_detail = "Channel is up to date."
                channel.sync_progress = 100
                channel.sync_completed_items = channel.sync_total_items
                session.add(channel)
                session.commit()
            log(f"Refresh complete. {len(all_entries)} total on YouTube, {new_video_count} new videos added.")
            return new_video_count

    def _queue_channel_unprocessed_videos(self, session: Session, channel_id: int) -> int:
        active_statuses = ["queued", "running", "downloading", "transcribing", "diarizing", "waiting_diarize"]
        videos = session.exec(
            select(Video).where(
                Video.channel_id == channel_id,
                Video.muted == False,
                Video.access_restricted == False,
                Video.processed == False,
            )
        ).all()

        jobs_created = 0
        for video in videos:
            existing = session.exec(
                select(Job.id).where(
                    Job.video_id == video.id,
                    Job.job_type.in_(["process", "diarize"]),
                    Job.status.in_(active_statuses),
                )
            ).first()
            if existing:
                continue
            job = Job(video_id=video.id, job_type="process", status="queued")
            session.add(job)
            video.status = "queued"
            session.add(video)
            jobs_created += 1
        return jobs_created

    def queue_channel_unprocessed_videos(self, channel_id: int) -> int:
        with Session(runtime.engine) as session:
            jobs_created = self._queue_channel_unprocessed_videos(session, channel_id)
            session.commit()
            return jobs_created

    def sync_monitored_channel(self, channel_id: int) -> dict[str, int | bool]:
        with Session(runtime.engine) as session:
            channel = session.get(Channel, channel_id)
            if not channel or not getattr(channel, "actively_monitored", False):
                return {"queued": 0, "refreshed": False}
            should_refresh = (channel.status or "").lower() != "refreshing"

        queued = self.queue_channel_unprocessed_videos(channel_id)
        refreshed = False
        if should_refresh:
            try:
                self.refresh_channel(channel_id)
                refreshed = True
            except Exception as e:
                log(f"Active monitor refresh failed for channel {channel_id}: {e}")
        queued += self.queue_channel_unprocessed_videos(channel_id)
        return {"queued": queued, "refreshed": refreshed}

    def monitor_channels_loop(self, stop_event: threading.Event):
        try:
            interval_seconds = int(os.getenv("CHANNEL_MONITOR_INTERVAL_SECONDS", "120"))
        except Exception:
            interval_seconds = 120
        interval_seconds = max(30, interval_seconds)
        log(f"Starting active channel monitor loop (interval={interval_seconds}s)...")

        while not stop_event.is_set():
            try:
                with Session(runtime.engine) as session:
                    channel_ids = session.exec(
                        select(Channel.id).where(Channel.actively_monitored == True).order_by(Channel.id.asc())
                    ).all()

                for channel_id in channel_ids:
                    if stop_event.is_set():
                        break
                    result = self.sync_monitored_channel(int(channel_id))
                    if result.get("refreshed") or result.get("queued"):
                        log(
                            f"Active monitor synced channel {channel_id}: "
                            f"refreshed={bool(result.get('refreshed'))}, queued={int(result.get('queued') or 0)}"
                        )
            except Exception as e:
                log(f"Active channel monitor loop error: {e}")

            if stop_event.wait(interval_seconds):
                break

    def _backfill_dates(
        self,
        channel_id: int,
        max_items: int | None = None,
        *,
        progress_start: int = 80,
        progress_end: int = 95,
        detail_prefix: str = "Backfilling publication dates and metadata",
        status: str | None = None,
    ):
        """Backfill missing video metadata, including view counts, newest-first."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sqlalchemy import or_

        if max_items is None:
            try:
                max_items = int(os.getenv("CHANNEL_DATE_BACKFILL_MAX_ITEMS", "2000"))
            except Exception:
                max_items = 2000
        try:
            workers = int(os.getenv("CHANNEL_DATE_BACKFILL_WORKERS", "4"))
        except Exception:
            workers = 4
        workers = max(1, min(12, workers))
        try:
            commit_batch = int(os.getenv("CHANNEL_DATE_BACKFILL_COMMIT_BATCH", "25"))
        except Exception:
            commit_batch = 25
        commit_batch = max(1, min(500, commit_batch))

        with Session(runtime.engine) as session:
            query = (
                select(Video)
                .where(
                    Video.channel_id == channel_id,
                    or_(
                        Video.published_at.is_(None),
                        Video.description.is_(None),
                        Video.duration.is_(None),
                        Video.thumbnail_url.is_(None),
                        Video.view_count.is_(None),
                    ),
                )
                .order_by(Video.id.desc())
            )
            if max_items and max_items > 0:
                query = query.limit(int(max_items))
            videos = session.exec(query).all()
            if not videos:
                return
            log(f"Backfilling dates for {len(videos)} videos (channel {channel_id}, workers={workers})...")
            self._update_channel_sync_progress(
                channel_id,
                status=status,
                detail=f"{detail_prefix} (fetching 0/{len(videos)})...",
                progress=progress_start,
                completed_items=0,
                total_items=len(videos),
            )
            channel = session.get(Channel, channel_id)
            channel_source = (getattr(channel, "source_type", None) or "youtube").strip().lower()
            ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
            ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="backfill_dates")
            targets = [
                {
                    "id": int(v.id),
                    "youtube_id": str(v.youtube_id),
                    "need_date": v.published_at is None,
                    "need_description": not bool(v.description),
                    "need_duration": not bool(v.duration),
                    "need_thumbnail": not bool(v.thumbnail_url),
                    "need_placeholder_transcript": (
                        self._placeholder_captions_enabled()
                        and not bool(v.transcript_is_placeholder)
                        and not bool(v.processed)
                    ),
                }
                for v in videos
            ]
            meta_by_id: dict[int, dict] = {}
            api_hits = 0
            if channel_source == "youtube" and self._youtube_data_api_key():
                try:
                    self._update_channel_sync_progress(
                        channel_id,
                        status=status,
                        detail=f"{detail_prefix} (YouTube API 0/{len(videos)})...",
                        progress=progress_start,
                        completed_items=0,
                        total_items=len(videos),
                    )
                    api_metadata_by_youtube_id = self._fetch_youtube_data_api_video_metadata_batch(
                        [item["youtube_id"] for item in targets]
                    )
                    for item in targets:
                        api_meta = api_metadata_by_youtube_id.get(item["youtube_id"])
                        if not api_meta:
                            continue
                        seed_meta = dict(api_meta)
                        seed_meta["placeholder_track"] = None
                        meta_by_id[item["id"]] = seed_meta
                        api_hits += 1
                    log(
                        f"YouTube API backfill metadata returned {api_hits}/{len(targets)} videos "
                        f"for channel {channel_id}."
                    )
                except Exception as e:
                    log(f"YouTube API backfill metadata fetch failed for channel {channel_id}: {e}")

            def _fetch_meta(item: dict):
                vid = item["youtube_id"]
                seed_meta = dict(meta_by_id.get(item["id"]) or {})
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(
                            f"https://www.youtube.com/watch?v={vid}",
                            download=False,
                        )
                    if not info:
                        if seed_meta:
                            return item["id"], seed_meta, None
                        return item["id"], None, "no_info"
                    seed_meta.update({
                        "upload_date": info.get("upload_date"),
                        "release_date": info.get("release_date"),
                        "release_timestamp": info.get("release_timestamp"),
                        "timestamp": info.get("timestamp"),
                        "description": info.get("description") or seed_meta.get("description"),
                        "duration": info.get("duration") if info.get("duration") is not None else seed_meta.get("duration"),
                        "view_count": info.get("view_count") if info.get("view_count") is not None else seed_meta.get("view_count"),
                        "thumbnail": info.get("thumbnail") or seed_meta.get("thumbnail"),
                        "placeholder_track": self._choose_caption_track(info) if item.get("need_placeholder_transcript") else seed_meta.get("placeholder_track"),
                    })
                    return item["id"], seed_meta, None
                except Exception as e:
                    if seed_meta:
                        return item["id"], seed_meta, None
                    return item["id"], None, str(e)

            failures = 0
            ytdlp_targets = [
                item for item in targets
                if item["id"] not in meta_by_id or item.get("need_placeholder_transcript")
            ]
            if ytdlp_targets:
                self._update_channel_sync_progress(
                    channel_id,
                    status=status,
                    detail=f"{detail_prefix} (yt-dlp fallback 0/{len(ytdlp_targets)})...",
                    progress=progress_start + int(max(1, (progress_end - progress_start) * 0.1)),
                    completed_items=0,
                    total_items=len(targets),
                )
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_target = {ex.submit(_fetch_meta, item): item for item in ytdlp_targets}
                for idx, fut in enumerate(as_completed(future_to_target), start=1):
                    item = future_to_target[fut]
                    try:
                        vid_id, meta, err = fut.result()
                    except Exception as e:
                        vid_id, meta, err = item["id"], None, str(e)
                    if meta:
                        meta_by_id[int(vid_id)] = meta
                    else:
                        failures += 1
                        log_verbose(f"  Backfill failed for {item['youtube_id']}: {err}")
                    if idx == 1 or idx == len(ytdlp_targets) or idx % 25 == 0:
                        fetch_progress = progress_start + int((idx / max(1, len(targets))) * max(1, (progress_end - progress_start) * 0.45))
                        self._update_channel_sync_progress(
                            channel_id,
                            status=status,
                            detail=f"{detail_prefix} (yt-dlp fallback {idx}/{len(ytdlp_targets)})...",
                            progress=fetch_progress,
                            completed_items=min(len(targets), api_hits + idx),
                            total_items=len(targets),
                        )
                    if idx % 200 == 0:
                        log(f"  Backfill yt-dlp fallback progress: {idx}/{len(ytdlp_targets)} videos checked...")

            filled = 0
            touched = 0
            for idx, item in enumerate(targets, start=1):
                meta = meta_by_id.get(item["id"])
                if not meta:
                    continue
                video = session.get(Video, item["id"])
                if not video:
                    continue

                pub_date = self._extract_published_at_from_info(meta)

                changed = False
                if pub_date and not video.published_at:
                    video.published_at = pub_date
                    filled += 1
                    changed = True
                if item["need_description"] and meta.get("description"):
                    video.description = meta["description"]
                    changed = True
                if item["need_duration"] and meta.get("duration"):
                    video.duration = meta["duration"]
                    changed = True
                if meta.get("view_count") is not None and video.view_count != meta.get("view_count"):
                    video.view_count = meta["view_count"]
                    changed = True
                if item["need_thumbnail"] and meta.get("thumbnail"):
                    video.thumbnail_url = meta["thumbnail"]
                    changed = True
                if item["need_placeholder_transcript"] and meta.get("placeholder_track"):
                    try:
                        if self._populate_placeholder_transcript_from_track(session, video, meta.get("placeholder_track")):
                            changed = True
                    except Exception as e:
                        log_verbose(f"  Placeholder transcript backfill skipped for {video.youtube_id}: {e}")

                if changed:
                    session.add(video)
                    touched += 1
                if touched and touched % commit_batch == 0:
                    session.commit()
                if idx == 1 or idx == len(targets) or idx % 25 == 0:
                    write_base = progress_start + int(max(1, (progress_end - progress_start) * 0.5))
                    write_span = max(1, progress_end - write_base)
                    self._update_channel_sync_progress(
                        channel_id,
                        status=status,
                        detail=f"{detail_prefix} (writing {idx}/{len(targets)})...",
                        progress=write_base + int((idx / max(1, len(targets))) * write_span),
                        completed_items=idx,
                        total_items=len(targets),
                    )
                if idx % 500 == 0:
                    log(f"  Backfill write progress: {idx}/{len(targets)} reviewed, {filled} dates filled...")
            session.commit()
            self._update_channel_sync_progress(
                channel_id,
                status=status,
                detail=f"{detail_prefix} complete.",
                progress=progress_end,
                completed_items=len(targets),
                total_items=len(targets),
            )
            log(
                f"Backfill complete. Filled dates for {filled}/{len(videos)} videos. "
                f"YouTube API hits: {api_hits}. Metadata fetch failures: {failures}."
            )
