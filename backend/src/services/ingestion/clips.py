"""Clip creation, captions, and export rendering.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import yt_dlp
import json
import subprocess
from pathlib import Path
from sqlmodel import Session, select
from datetime import datetime

from ...db.database import Video, Speaker, TranscriptSegment, Clip, ClipExportArtifact
from ..logger import log, log_verbose, is_verbose
from . import runtime
from .runtime import (
    BACKEND_DIR,
    EXPORT_DIR,
    TEMP_DIR,
)


class ClipsMixin:
    def record_clip_export_artifact(self, clip_id: int, file_path: Path, *, artifact_type: str, fmt: str) -> ClipExportArtifact | None:
        """Persist metadata for a rendered clip export artifact for later re-download."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            with Session(runtime.engine) as session:
                clip = session.get(Clip, clip_id)
                if not clip:
                    return None
                artifact = ClipExportArtifact(
                    clip_id=int(clip_id),
                    video_id=int(clip.video_id),
                    artifact_type=str(artifact_type or "video"),
                    format=str(fmt or "").lower() or "mp4",
                    file_path=str(path.resolve()),
                    file_name=path.name,
                    file_size_bytes=int(path.stat().st_size),
                    created_at=datetime.now(),
                )
                session.add(artifact)
                session.commit()
                session.refresh(artifact)
                return artifact
        except Exception as e:
            log(f"Failed to record clip export artifact for clip {clip_id}: {e}")
            return None

    def create_clip(self, video_id: int, start: float, end: float, audio_only: bool = False) -> str:
        if end <= start:
            raise ValueError("Clip end must be greater than start")
        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video: raise ValueError("Video not found")
            
            timestamp = int(datetime.now().timestamp())
            
            # Optimization: Use local audio file if we only want audio
            if audio_only:
                local_audio = self.get_audio_path(video)
                if local_audio.exists():
                    output_filename = f"clip_{video.youtube_id}_{start}_{end}_{timestamp}.m4a"
                    output_path = TEMP_DIR / output_filename
                    
                    # Determine ffmpeg
                    ffmpeg_bin = BACKEND_DIR / "bin"
                    ffmpeg_cmd = "ffmpeg"
                    if (ffmpeg_bin / "ffmpeg.exe").exists():
                        ffmpeg_cmd = str(ffmpeg_bin / "ffmpeg.exe")
                    
                    # Slice command (fast seek -ss before -i is crucial but for precise cutting we might want re-encoding or accurate seek)
                    # For simple playback, -ss before -i is fast.
                    cmd = [
                        ffmpeg_cmd, "-y",
                        "-i", str(local_audio),
                        "-ss", str(start),
                        "-to", str(end),
                        "-c", "copy", # Fast copy
                        str(output_path)
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        log_verbose(f"Created clip from local audio: {output_path}")
                        return str(output_path)
                    except Exception as e:
                        log(f"Error creating local clip: {e}. Falling back to download.")
                        # Fallthrough to yt-dlp
            
            # Fallback: Use yt-dlp to download only the requested section.
            # IMPORTANT: Use yt-dlp's API-native `download_ranges` callback. The old
            # `download_sections` dict style can be ignored by yt-dlp and fetch full videos.
            output_filename = f"clip_{video.youtube_id}_{start}_{end}_{timestamp}.mp4"
            output_path = TEMP_DIR / output_filename

            url = f"https://www.youtube.com/watch?v={video.youtube_id}"
            ffmpeg_bin = BACKEND_DIR / "bin"
            ffmpeg_loc = str(ffmpeg_bin) if (ffmpeg_bin / "ffmpeg.exe").exists() else None
            source_template = TEMP_DIR / f"clipsrc_{video.youtube_id}_{timestamp}.%(ext)s"

            def _download_source(use_range: bool) -> tuple[Path | None, bool]:
                downloaded_file = [None]

                def _track_file(d):
                    if d.get('status') == 'finished':
                        downloaded_file[0] = d.get('filename')

                ydl_opts = {
                    'outtmpl': str(source_template),
                    'quiet': True,
                    'noprogress': not is_verbose(),
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'merge_output_format': 'mp4',
                    'progress_hooks': [_track_file],
                    'retries': 5,
                    'fragment_retries': 5,
                    'socket_timeout': 30,
                    'noplaylist': True,
                }
                if ffmpeg_loc:
                    ydl_opts['ffmpeg_location'] = ffmpeg_loc
                if use_range:
                    ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(float(start), float(end))])
                    ydl_opts['force_keyframes_at_cuts'] = True
                ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="create_clip")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                src = Path(downloaded_file[0]) if downloaded_file[0] else None
                if not src or not src.exists():
                    matches = sorted(
                        TEMP_DIR.glob(f"clipsrc_{video.youtube_id}_{timestamp}*"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    src = matches[0] if matches else None
                return (src if src and src.exists() else None, use_range)

            source_path: Path | None = None
            used_range_download = False
            primary_error = None
            try:
                source_path, used_range_download = _download_source(use_range=True)
            except Exception as e:
                primary_error = e
                log(f"Range clip download failed for {video.youtube_id}: {e}. Falling back to full download + trim.")

            if not source_path:
                try:
                    source_path, used_range_download = _download_source(use_range=False)
                except Exception as e:
                    if primary_error is not None:
                        raise RuntimeError(f"Clip download failed (range + full fallback). Range error: {primary_error}; Fallback error: {e}")
                    raise

            if not source_path:
                raise RuntimeError("yt-dlp did not produce a clip file")

            # Safety check: if yt-dlp unexpectedly returned a long/full file, hard-trim
            # with ffmpeg so the final output always matches [start, end].
            def _probe_duration_seconds(path: Path):
                ffmpeg_bin = BACKEND_DIR / "bin"
                ffprobe_cmd = str(ffmpeg_bin / "ffprobe.exe") if (ffmpeg_bin / "ffprobe.exe").exists() else "ffprobe"
                try:
                    res = subprocess.run(
                        [ffprobe_cmd, "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
                        capture_output=True,
                        timeout=20,
                    )
                    if res.returncode != 0:
                        return None
                    return float((res.stdout or b"").decode(errors="replace").strip())
                except Exception:
                    return None

            requested_duration = max(0.1, float(end) - float(start))
            source_duration = _probe_duration_seconds(source_path)
            looks_like_full_video = source_duration is not None and source_duration > (requested_duration + 5.0)

            if looks_like_full_video or not used_range_download:
                ffmpeg_cmd = self._get_ffmpeg_cmd()
                trim_cmd = [
                    ffmpeg_cmd, "-y",
                    "-ss", str(float(start)),
                    "-to", str(float(end)),
                    "-i", str(source_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    str(output_path),
                ]
                proc = subprocess.run(trim_cmd, capture_output=True)
                if proc.returncode != 0:
                    stderr = proc.stderr.decode(errors="replace")[:1000]
                    raise RuntimeError(f"Failed to trim clip: {stderr}")
            else:
                if source_path != output_path:
                    try:
                        if output_path.exists():
                            output_path.unlink()
                        source_path.replace(output_path)
                    except Exception:
                        import shutil
                        shutil.copy2(source_path, output_path)

            return str(output_path)

    def _load_clip_kept_ranges(self, clip: Clip) -> list[tuple[float, float]] | None:
        """Parse and normalize text-based clip edit ranges from clip.script_edits_json.
        Returns absolute ranges on source media timeline, or None if no valid edits exist."""
        raw = getattr(clip, "script_edits_json", None)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        ranges = payload.get("kept_ranges")
        if not isinstance(ranges, list):
            return None

        out: list[tuple[float, float]] = []
        clip_start = float(clip.start_time)
        clip_end = float(clip.end_time)
        for item in ranges:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                s = max(clip_start, float(item[0]))
                e = min(clip_end, float(item[1]))
            except Exception:
                continue
            if e > s + 0.01:
                out.append((s, e))
        if not out:
            return None
        out.sort(key=lambda p: p[0])
        merged: list[tuple[float, float]] = []
        for s, e in out:
            if not merged:
                merged.append((s, e))
                continue
            ps, pe = merged[-1]
            if s <= pe + 0.05:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _probe_media_duration_seconds(self, path: Path) -> float | None:
        ffmpeg_bin = BACKEND_DIR / "bin"
        ffprobe_cmd = str(ffmpeg_bin / "ffprobe.exe") if (ffmpeg_bin / "ffprobe.exe").exists() else "ffprobe"
        try:
            res = subprocess.run(
                [ffprobe_cmd, "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
                capture_output=True,
                timeout=25,
            )
            if res.returncode != 0:
                return None
            out = (res.stdout or b"").decode(errors="replace").strip()
            val = float(out)
            if val > 0:
                return val
            return None
        except Exception:
            return None

    def _build_clip_fade_filters(self, clip: Clip, duration_sec: float | None) -> tuple[str | None, str | None]:
        fade_in = max(0.0, float(getattr(clip, "fade_in_sec", 0.0) or 0.0))
        fade_out = max(0.0, float(getattr(clip, "fade_out_sec", 0.0) or 0.0))
        if (fade_in <= 0.0 and fade_out <= 0.0) or not duration_sec or duration_sec <= 0.05:
            return None, None

        max_total = max(0.05, duration_sec * 0.95)
        if fade_in + fade_out > max_total:
            scale = max_total / max(fade_in + fade_out, 1e-6)
            fade_in *= scale
            fade_out *= scale

        v_filters: list[str] = []
        a_filters: list[str] = []
        if fade_in > 0.001:
            v_filters.append(f"fade=t=in:st=0:d={fade_in:.3f}")
            a_filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
        if fade_out > 0.001:
            out_start = max(0.0, duration_sec - fade_out)
            v_filters.append(f"fade=t=out:st={out_start:.3f}:d={fade_out:.3f}")
            a_filters.append(f"afade=t=out:st={out_start:.3f}:d={fade_out:.3f}")
        return (",".join(v_filters) if v_filters else None, ",".join(a_filters) if a_filters else None)

    def create_clip_from_ranges(self, video_id: int, ranges: list[tuple[float, float]]) -> str:
        """Create a stitched clip by concatenating multiple source ranges."""
        clean: list[tuple[float, float]] = []
        for r in ranges or []:
            if not isinstance(r, (list, tuple)) or len(r) < 2:
                continue
            try:
                s = float(r[0]); e = float(r[1])
            except Exception:
                continue
            if e > s + 0.01:
                clean.append((s, e))
        if not clean:
            raise ValueError("No valid ranges for text-edited clip export")
        clean.sort(key=lambda x: x[0])

        part_paths: list[Path] = []
        keep_path: Path | None = None
        try:
            for s, e in clean:
                part_paths.append(Path(self.create_clip(video_id, s, e, audio_only=False)))
            if len(part_paths) == 1:
                keep_path = part_paths[0]
                return str(part_paths[0])

            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError("Video not found")
                timestamp = int(datetime.now().timestamp())
                out_path = TEMP_DIR / f"clip_stitched_{video.youtube_id}_{timestamp}.mp4"

            ffmpeg_cmd = self._get_ffmpeg_cmd()
            cmd = [ffmpeg_cmd, "-y"]
            for p in part_paths:
                cmd.extend(["-i", str(p)])
            concat_inputs = "".join([f"[{idx}:v:0][{idx}:a:0]" for idx in range(len(part_paths))])
            cmd.extend([
                "-filter_complex", f"{concat_inputs}concat=n={len(part_paths)}:v=1:a=1[v][a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path),
            ])
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace")[:2000]
                raise RuntimeError(f"Failed to stitch text-edited clip: {stderr}")
            keep_path = out_path
            return str(out_path)
        finally:
            # Keep only final stitched output; cleanup intermediate parts.
            for p in part_paths:
                try:
                    if keep_path is not None and p == keep_path:
                        continue
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format: HH:MM:SS.mmm"""
        millis = int(round((seconds - int(seconds)) * 1000))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if millis >= 1000:
            millis = 0
            secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def _clip_caption_entries(self, session: Session, clip: Clip, speaker_labels: bool = True) -> list[dict]:
        segments = session.exec(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.video_id == clip.video_id,
                TranscriptSegment.end_time >= clip.start_time,
                TranscriptSegment.start_time <= clip.end_time,
            )
            .order_by(TranscriptSegment.start_time)
        ).all()
        speaker_map = {}
        if speaker_labels:
            speaker_ids = [s.speaker_id for s in segments if s.speaker_id]
            if speaker_ids:
                speakers = session.exec(select(Speaker).where(Speaker.id.in_(speaker_ids))).all()
                speaker_map = {sp.id: sp.name for sp in speakers if sp.id is not None}

        entries = []
        kept_ranges = self._load_clip_kept_ranges(clip) or [(float(clip.start_time), float(clip.end_time))]
        dst_cursor = 0.0
        mapped_ranges: list[tuple[float, float, float]] = []  # (src_start, src_end, dst_offset)
        for s, e in kept_ranges:
            if e > s + 0.01:
                mapped_ranges.append((s, e, dst_cursor))
                dst_cursor += (e - s)

        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            if speaker_labels:
                sp_name = speaker_map.get(seg.speaker_id) if seg.speaker_id else None
                if sp_name:
                    text = f"[{sp_name}] {text}"
            seg_s = float(seg.start_time)
            seg_e = float(seg.end_time)
            for src_s, src_e, dst_off in mapped_ranges:
                ov_s = max(seg_s, src_s)
                ov_e = min(seg_e, src_e)
                if ov_e <= ov_s + 0.01:
                    continue
                start = max(0.0, dst_off + (ov_s - src_s))
                end = max(start + 0.01, dst_off + (ov_e - src_s))
                entries.append({"start": start, "end": end, "text": text})
        entries.sort(key=lambda e: (float(e["start"]), float(e["end"])))
        return entries

    def write_clip_caption_file(self, clip_id: int, fmt: str = "srt", speaker_labels: bool = True) -> Path:
        fmt = (fmt or "srt").lower()
        if fmt not in {"srt", "vtt"}:
            raise ValueError("Caption format must be 'srt' or 'vtt'")
        with Session(runtime.engine) as session:
            clip = session.get(Clip, clip_id)
            if not clip:
                raise ValueError("Clip not found")
            entries = self._clip_caption_entries(session, clip, speaker_labels=speaker_labels)
            safe_title = self.sanitize_filename(clip.title or f"clip_{clip.id}")
            out_dir = EXPORT_DIR / "clips" / f"clip_{clip.id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{safe_title}.{fmt}"
            with open(out_path, "w", encoding="utf-8") as f:
                if fmt == "vtt":
                    f.write("WEBVTT\n\n")
                    for e in entries:
                        f.write(f"{self._format_vtt_timestamp(e['start'])} --> {self._format_vtt_timestamp(e['end'])}\n{e['text']}\n\n")
                else:
                    for i, e in enumerate(entries, 1):
                        f.write(f"{i}\n{self._format_timestamp(e['start'])} --> {self._format_timestamp(e['end'])}\n{e['text']}\n\n")
            return out_path

    def _normalize_clip_crop_values(
        self,
        crop_x: float | None,
        crop_y: float | None,
        crop_w: float | None,
        crop_h: float | None,
    ) -> tuple[float, float, float, float] | None:
        vals = [crop_x, crop_y, crop_w, crop_h]
        if any(v is None for v in vals):
            return None
        x, y, w, h = [float(v) for v in vals]
        x = max(0.0, min(x, 0.99))
        y = max(0.0, min(y, 0.99))
        w = max(0.01, min(w, 1.0))
        h = max(0.01, min(h, 1.0))
        if x + w > 1.0:
            w = max(0.01, 1.0 - x)
        if y + h > 1.0:
            h = max(0.01, 1.0 - y)
        return (x, y, w, h)

    def _normalize_clip_crop(self, clip: Clip) -> tuple[float, float, float, float] | None:
        return self._normalize_clip_crop_values(clip.crop_x, clip.crop_y, clip.crop_w, clip.crop_h)

    def _get_portrait_split_crops(self, clip: Clip) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
        if (getattr(clip, "aspect_ratio", "source") or "source").strip().lower() != "9:16":
            return None
        if not bool(getattr(clip, "portrait_split_enabled", False)):
            return None
        top = self._normalize_clip_crop_values(
            getattr(clip, "portrait_top_crop_x", None),
            getattr(clip, "portrait_top_crop_y", None),
            getattr(clip, "portrait_top_crop_w", None),
            getattr(clip, "portrait_top_crop_h", None),
        ) or (0.0, 0.0, 1.0, 0.5)
        bottom = self._normalize_clip_crop_values(
            getattr(clip, "portrait_bottom_crop_x", None),
            getattr(clip, "portrait_bottom_crop_y", None),
            getattr(clip, "portrait_bottom_crop_w", None),
            getattr(clip, "portrait_bottom_crop_h", None),
        ) or (0.0, 0.5, 1.0, 0.5)
        return top, bottom

    def _target_dims_for_aspect(self, aspect_ratio: str | None) -> tuple[int, int] | None:
        key = (aspect_ratio or "source").strip().lower()
        return {
            "16:9": (1280, 720),
            "9:16": (1080, 1920),
            "1:1": (1080, 1080),
            "4:5": (1080, 1350),
        }.get(key)

    def _ffmpeg_escape_subtitles_path(self, path: Path) -> str:
        p = str(path.resolve()).replace("\\", "/")
        p = p.replace(":", "\\:").replace("'", "\\'")
        return p

    def _build_clip_video_filter_chain(self, clip: Clip, subtitle_path: Path | None = None) -> str | None:
        filters = []
        crop = self._normalize_clip_crop(clip)
        if crop:
            x, y, w, h = crop
            filters.append(
                f"crop=floor(iw*{w:.6f}):floor(ih*{h:.6f}):floor(iw*{x:.6f}):floor(ih*{y:.6f})"
            )

        dims = self._target_dims_for_aspect(getattr(clip, "aspect_ratio", None))
        if dims:
            tw, th = dims
            filters.extend([
                f"scale={tw}:{th}:force_original_aspect_ratio=decrease",
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2",
                "setsar=1",
            ])

        if subtitle_path is not None:
            filters.append(f"subtitles='{self._ffmpeg_escape_subtitles_path(subtitle_path)}'")

        return ",".join(filters) if filters else None

    def render_clip_export_mp4(self, clip_id: int) -> Path:
        with Session(runtime.engine) as session:
            clip = session.get(Clip, clip_id)
            if not clip:
                raise ValueError("Clip not found")
            video = session.get(Video, clip.video_id)
            if not video:
                raise ValueError("Video not found")

            kept_ranges = self._load_clip_kept_ranges(clip)
            if kept_ranges:
                base_path = Path(self.create_clip_from_ranges(video.id, kept_ranges))
            else:
                base_path = Path(self.create_clip(video.id, clip.start_time, clip.end_time, audio_only=False))
            if not base_path.exists():
                raise RuntimeError("Failed to create base clip for export")
            base_duration = self._probe_media_duration_seconds(base_path)

            out_dir = EXPORT_DIR / "clips" / f"clip_{clip.id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_title = self.sanitize_filename(clip.title or f"clip_{clip.id}")
            out_path = out_dir / f"{safe_title}_{int(datetime.now().timestamp())}.mp4"

            subtitle_path = None
            if bool(getattr(clip, "burn_captions", False)):
                subtitle_path = self.write_clip_caption_file(
                    clip_id,
                    fmt="srt",
                    speaker_labels=bool(getattr(clip, "caption_speaker_labels", True)),
                )

            ffmpeg_cmd = self._get_ffmpeg_cmd()
            cmd = [ffmpeg_cmd, "-y", "-i", str(base_path)]
            fade_vf, fade_af = self._build_clip_fade_filters(clip, base_duration)
            split_crops = self._get_portrait_split_crops(clip)
            if split_crops:
                (tx, ty, tw, th), (bx, by, bw, bh) = split_crops
                target_w, target_h = self._target_dims_for_aspect("9:16") or (1080, 1920)
                half_h = target_h // 2
                filter_steps = [
                    f"[0:v]crop=floor(iw*{tw:.6f}):floor(ih*{th:.6f}):floor(iw*{tx:.6f}):floor(ih*{ty:.6f}),scale={target_w}:{half_h}:force_original_aspect_ratio=increase,crop={target_w}:{half_h}[top]",
                    f"[0:v]crop=floor(iw*{bw:.6f}):floor(ih*{bh:.6f}):floor(iw*{bx:.6f}):floor(ih*{by:.6f}),scale={target_w}:{half_h}:force_original_aspect_ratio=increase,crop={target_w}:{half_h}[bottom]",
                    "[top][bottom]vstack=inputs=2,setsar=1[stack]",
                ]
                map_label = "[stack]"
                if subtitle_path is not None:
                    filter_steps.append(f"[stack]subtitles='{self._ffmpeg_escape_subtitles_path(subtitle_path)}'[vout]")
                    map_label = "[vout]"
                if fade_vf:
                    next_label = "[vfaded]"
                    filter_steps.append(f"{map_label}{fade_vf}{next_label}")
                    map_label = next_label
                cmd.extend(["-filter_complex", ";".join(filter_steps), "-map", map_label, "-map", "0:a?"])
            else:
                vf = self._build_clip_video_filter_chain(clip, subtitle_path=subtitle_path)
                if fade_vf:
                    vf = f"{vf},{fade_vf}" if vf else fade_vf
                if vf:
                    cmd.extend(["-vf", vf])
            if fade_af:
                cmd.extend(["-af", fade_af])
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path),
            ])
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace")[:2000]
                raise RuntimeError(f"Clip export failed: {stderr}")
            return out_path
