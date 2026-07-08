"""yt-dlp auth/notice classification and placeholder captions from YouTube tracks.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import yt_dlp
import os
import json
import re
import html
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from sqlmodel import Session, select, func

from ...db.database import Video, TranscriptSegment
from ..logger import log, log_verbose


class YoutubeDownloadMixin:
    def _parse_ytdlp_cookies_from_browser(self, raw_value: str):
        text = (raw_value or "").strip()
        if not text:
            return None
        if ":" in text:
            browser, profile = text.split(":", 1)
            browser = browser.strip()
            profile = profile.strip() or None
        else:
            browser = text
            profile = None
        if not browser:
            return None
        # yt-dlp Python API expects: (browser_name, profile, keyring, container)
        return (browser, profile, None, None)

    def _apply_ytdlp_auth_opts(self, ydl_opts: dict, *, purpose: str = "") -> dict:
        opts = dict(ydl_opts or {})
        cookies_file = (os.getenv("YTDLP_COOKIES_FILE") or "").strip()
        cookies_from_browser = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip()
        tag = f" ({purpose})" if purpose else ""

        if cookies_file:
            path = Path(cookies_file).expanduser()
            if path.exists():
                opts["cookiefile"] = str(path)
            else:
                log_verbose(f"YTDLP_COOKIES_FILE not found{tag}: {path}")

        if cookies_from_browser and "cookiefile" not in opts:
            parsed = self._parse_ytdlp_cookies_from_browser(cookies_from_browser)
            if parsed:
                opts["cookiesfrombrowser"] = parsed
            else:
                log_verbose(f"Invalid YTDLP_COOKIES_FROM_BROWSER{tag}: {cookies_from_browser}")

        return opts

    def _is_ytdlp_auth_required_error(self, exc: Exception) -> bool:
        msg = str(exc or "").lower()
        checks = [
            "sign in to confirm your age",
            "use --cookies-from-browser",
            "use --cookies",
            "authentication",
            "login required",
            "age-restricted",
        ]
        return any(token in msg for token in checks)

    def _classify_ytdlp_download_notice(self, exc: Exception) -> dict | None:
        msg = str(exc or "").strip()
        lowered = msg.lower()
        if not lowered:
            return None

        if "tiktok" in lowered:
            if "your ip address is blocked" in lowered or "blocked from accessing this post" in lowered:
                return {
                    "code": "tiktok_ip_blocked",
                    "message": (
                        "TikTok blocked this request from the current IP address. "
                        "Try again later or use a different network/session."
                    ),
                    "video_status": "pending",
                    "access_restricted": False,
                }
            if "login required" in lowered or "authentication" in lowered or "sign in" in lowered:
                return {
                    "code": "tiktok_auth_required",
                    "message": (
                        "TikTok requires an authenticated session before this media can be accessed."
                    ),
                    "video_status": "pending",
                    "access_restricted": False,
                }
            if "rate limit" in lowered or "too many requests" in lowered:
                return {
                    "code": "tiktok_rate_limited",
                    "message": (
                        "TikTok temporarily rate-limited metadata access. Try again later."
                    ),
                    "video_status": "pending",
                    "access_restricted": False,
                }

        if (
            "members-only content" in lowered
            or "join this channel to get access to members-only content" in lowered
        ):
            return {
                "code": "youtube_members_only",
                "message": (
                    "This video is members-only. Chatalogue could not download it with the "
                    "current YouTube session."
                ),
                "video_status": "access_restricted",
                "access_restricted": True,
            }

        if (
            "private video" in lowered
            or "granted access to this video" in lowered
            or "this video is private" in lowered
        ):
            return {
                "code": "youtube_private_video",
                "message": (
                    "This video is private or access-restricted. Chatalogue could not "
                    "download it with the current YouTube session."
                ),
                "video_status": "access_restricted",
                "access_restricted": True,
            }

        if (
            "premieres in " in lowered
            or "premieres on " in lowered
            or "premieres at " in lowered
            or "upcoming live event" in lowered
        ):
            return {
                "code": "youtube_premiere_scheduled",
                "message": (
                    "This YouTube video is an upcoming premiere and cannot be downloaded yet. "
                    "Retry after the premiere has started or the VOD is available."
                ),
                "video_status": "pending",
                "access_restricted": False,
            }

        if self._is_ytdlp_auth_required_error(exc):
            return {
                "code": "youtube_auth_required",
                "message": (
                    "This video requires a signed-in YouTube session or browser cookies "
                    "before it can be downloaded."
                ),
                "video_status": "access_restricted",
                "access_restricted": True,
            }

        return None

    def _placeholder_captions_enabled(self) -> bool:
        raw = (os.getenv("YOUTUBE_PLACEHOLDER_CAPTIONS_ENABLED") or "true").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _preferred_placeholder_caption_languages(self) -> list[str]:
        raw = (os.getenv("YOUTUBE_PLACEHOLDER_CAPTION_LANGS") or "en,en-us,en-gb").strip()
        langs = []
        for item in raw.split(","):
            lang = item.strip()
            if lang and lang.lower() not in {x.lower() for x in langs}:
                langs.append(lang)
        return langs or ["en", "en-US", "en-GB"]

    @staticmethod
    def _normalize_caption_language(lang: str | None) -> str:
        return str(lang or "").strip().lower().replace("_", "-")

    @staticmethod
    def _caption_format_priority(ext: str | None) -> int:
        order = {
            "json3": 0,
            "srv3": 1,
            "vtt": 2,
            "ttml": 3,
            "srv2": 4,
            "srv1": 5,
            "json": 6,
            "xml": 7,
        }
        return order.get(str(ext or "").strip().lower(), 99)

    def _fetch_youtube_video_info(self, youtube_id: str, *, purpose: str = "placeholder_captions") -> dict | None:
        if not youtube_id:
            return None
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose=purpose)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=False)

    def _choose_caption_track(self, info: dict | None) -> dict | None:
        if not isinstance(info, dict):
            return None

        preferred_langs = self._preferred_placeholder_caption_languages()
        preferred_exact = {
            self._normalize_caption_language(lang): idx
            for idx, lang in enumerate(preferred_langs)
        }
        preferred_base = {
            self._normalize_caption_language(lang).split("-", 1)[0]: idx
            for idx, lang in enumerate(preferred_langs)
        }

        source_prefix = "youtube"
        webpage_url = str(info.get("webpage_url") or info.get("original_url") or "").lower()
        extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
        if "tiktok" in extractor or "tiktok.com" in webpage_url:
            source_prefix = "tiktok"

        for source_key, source_name in (
            ("subtitles", f"{source_prefix}_subtitles"),
            ("automatic_captions", f"{source_prefix}_auto_captions"),
        ):
            track_map = info.get(source_key)
            if not isinstance(track_map, dict) or not track_map:
                continue

            def lang_rank(lang: str) -> tuple[int, int, str]:
                normalized = self._normalize_caption_language(lang)
                if normalized in preferred_exact:
                    return (0, preferred_exact[normalized], normalized)
                base = normalized.split("-", 1)[0]
                if base in preferred_base:
                    return (1, preferred_base[base], normalized)
                return (2, 999, normalized)

            for language in sorted(track_map.keys(), key=lang_rank):
                candidates = track_map.get(language) or []
                if not isinstance(candidates, list):
                    continue
                best = None
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    url = str(candidate.get("url") or "").strip()
                    if not url:
                        continue
                    ext = str(candidate.get("ext") or "").strip().lower()
                    if ext == "live_chat":
                        continue
                    rank = self._caption_format_priority(ext)
                    if best is None or rank < best["rank"]:
                        best = {
                            "rank": rank,
                            "url": url,
                            "ext": ext or "vtt",
                            "language": language,
                            "source": source_name,
                        }
                if best is not None:
                    best.pop("rank", None)
                    return best
        return None

    def _choose_youtube_caption_track(self, info: dict | None) -> dict | None:
        return self._choose_caption_track(info)

    @staticmethod
    def _parse_caption_clock(text: str) -> float | None:
        raw = str(text or "").strip().replace(",", ".")
        if not raw:
            return None
        if ":" not in raw:
            try:
                return float(raw)
            except Exception:
                return None
        parts = raw.split(":")
        try:
            seconds = float(parts[-1])
            minutes = int(parts[-2]) if len(parts) >= 2 else 0
            hours = int(parts[-3]) if len(parts) >= 3 else 0
            return (hours * 3600) + (minutes * 60) + seconds
        except Exception:
            return None

    def _parse_caption_time_value(self, value, *, unit_hint: str = "seconds") -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed_clock = self._parse_caption_clock(text)
        if parsed_clock is not None and ":" in text:
            return parsed_clock
        try:
            numeric = float(text)
        except Exception:
            return None
        return numeric / 1000.0 if unit_hint == "milliseconds" else numeric

    @staticmethod
    def _clean_placeholder_caption_text(text: str | None) -> str:
        cleaned = html.unescape(str(text or ""))
        cleaned = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", " ", cleaned)
        cleaned = re.sub(r"</?[^>]+>", " ", cleaned)
        cleaned = cleaned.replace("\xa0", " ")
        lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines()]
        lines = [line for line in lines if line]
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def _parse_json3_placeholder_captions(self, payload_text: str) -> list[dict]:
        try:
            payload = json.loads(payload_text)
        except Exception:
            return []
        entries = []
        for event in payload.get("events") or []:
            if not isinstance(event, dict):
                continue
            start = self._parse_caption_time_value(event.get("tStartMs"), unit_hint="milliseconds")
            dur = self._parse_caption_time_value(event.get("dDurationMs"), unit_hint="milliseconds") or 0.0
            if start is None:
                continue
            seg_text = "".join(str(seg.get("utf8") or "") for seg in (event.get("segs") or []) if isinstance(seg, dict))
            text = self._clean_placeholder_caption_text(seg_text)
            if not text:
                continue
            entries.append({
                "start": float(start),
                "end": max(float(start) + max(float(dur), 0.01), float(start) + 0.01),
                "text": text,
            })
        return entries

    def _parse_xml_placeholder_captions(self, payload_text: str) -> list[dict]:
        try:
            root = ET.fromstring(payload_text)
        except Exception:
            return []

        entries = []
        for elem in root.iter():
            tag = str(elem.tag or "").split("}", 1)[-1].lower()
            if tag not in {"text", "p"}:
                continue
            if tag == "text":
                start = self._parse_caption_time_value(elem.attrib.get("start"), unit_hint="seconds")
                dur = self._parse_caption_time_value(elem.attrib.get("dur"), unit_hint="seconds") or 0.0
                end = None if start is None else float(start) + max(float(dur), 0.01)
            else:
                start = self._parse_caption_time_value(elem.attrib.get("t"), unit_hint="milliseconds")
                if start is None:
                    start = self._parse_caption_time_value(elem.attrib.get("begin"), unit_hint="seconds")
                dur = self._parse_caption_time_value(elem.attrib.get("d"), unit_hint="milliseconds")
                if dur is None:
                    dur = self._parse_caption_time_value(elem.attrib.get("dur"), unit_hint="seconds")
                end = self._parse_caption_time_value(elem.attrib.get("end"), unit_hint="seconds")
                if start is not None and end is None:
                    end = float(start) + max(float(dur or 0.0), 0.01)
            if start is None:
                continue
            text = self._clean_placeholder_caption_text("".join(elem.itertext()))
            if not text:
                continue
            entries.append({
                "start": float(start),
                "end": max(float(end or start), float(start) + 0.01),
                "text": text,
            })
        return entries

    def _parse_vtt_placeholder_captions(self, payload_text: str) -> list[dict]:
        lines = payload_text.splitlines()
        entries = []
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip("\ufeff").strip()
            idx += 1
            if not line:
                continue
            if line.startswith("WEBVTT"):
                continue
            if line.startswith("NOTE") or line in {"STYLE", "REGION"}:
                while idx < len(lines) and lines[idx].strip():
                    idx += 1
                continue
            if "-->" not in line:
                if idx >= len(lines):
                    continue
                timing_line = lines[idx].strip()
                if "-->" not in timing_line:
                    continue
                line = timing_line
                idx += 1
            parts = line.split("-->", 1)
            start = self._parse_caption_clock(parts[0].strip())
            end_token = parts[1].strip().split(" ", 1)[0]
            end = self._parse_caption_clock(end_token)
            if start is None or end is None:
                while idx < len(lines) and lines[idx].strip():
                    idx += 1
                continue
            text_lines = []
            while idx < len(lines) and lines[idx].strip():
                text_lines.append(lines[idx].rstrip())
                idx += 1
            text = self._clean_placeholder_caption_text("\n".join(text_lines))
            if not text:
                continue
            entries.append({
                "start": float(start),
                "end": max(float(end), float(start) + 0.01),
                "text": text,
            })
        return entries

    def _consolidate_placeholder_caption_entries(self, entries: list[dict]) -> list[dict]:
        normalized = []
        for entry in sorted(entries or [], key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0))):
            text = self._clean_placeholder_caption_text(entry.get("text"))
            if not text:
                continue
            start = float(entry.get("start") or 0.0)
            end = max(float(entry.get("end") or start), start + 0.01)
            if normalized and text == normalized[-1]["text"] and start <= normalized[-1]["end"] + 0.5:
                normalized[-1]["end"] = max(normalized[-1]["end"], end)
                continue
            normalized.append({"start": start, "end": end, "text": text})
        return normalized

    def _download_placeholder_caption_entries(self, track: dict) -> list[dict]:
        url = str((track or {}).get("url") or "").strip()
        ext = str((track or {}).get("ext") or "vtt").strip().lower()
        if not url:
            return []
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = resp.read()
        payload_text = payload.decode("utf-8-sig", errors="replace")
        if ext == "json3":
            entries = self._parse_json3_placeholder_captions(payload_text)
        elif ext in {"srv1", "srv2", "srv3", "ttml", "xml"}:
            entries = self._parse_xml_placeholder_captions(payload_text)
        else:
            entries = self._parse_vtt_placeholder_captions(payload_text)
        return self._consolidate_placeholder_caption_entries(entries)

    def _populate_placeholder_transcript_from_track(self, session: Session, video: Video, track: dict | None) -> int:
        if not video or not video.id or not video.youtube_id or not isinstance(track, dict):
            return 0

        existing_count = session.exec(
            select(func.count(TranscriptSegment.id)).where(TranscriptSegment.video_id == video.id)
        ).one()
        if int(existing_count or 0) > 0:
            return 0

        entries = self._download_placeholder_caption_entries(track)
        if not entries:
            return 0

        for entry in entries:
            session.add(
                TranscriptSegment(
                    video_id=video.id,
                    start_time=float(entry["start"]),
                    end_time=float(entry["end"]),
                    text=str(entry["text"]),
                    words=None,
                )
            )

        video.transcript_source = str(track.get("source") or "youtube_captions")
        video.transcript_language = str(track.get("language") or "").strip() or None
        video.transcript_is_placeholder = True
        session.add(video)
        log(
            f"Stored placeholder transcript for {video.youtube_id} "
            f"({video.transcript_source}, {video.transcript_language or 'unknown'}): {len(entries)} segments"
        )
        return len(entries)

    def populate_placeholder_transcript(self, session: Session, video: Video, *, info: dict | None = None) -> int:
        if not self._placeholder_captions_enabled():
            return 0
        if not video or not video.id or not video.youtube_id:
            return 0

        info = info or self._fetch_youtube_video_info(video.youtube_id)
        track = self._choose_caption_track(info)
        return self._populate_placeholder_transcript_from_track(session, video, track)
