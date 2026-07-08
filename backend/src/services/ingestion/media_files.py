"""Media file handling: ffmpeg/ffprobe, audio paths, download, validation, slicing.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import yt_dlp
import os
import time
import json
import subprocess
import threading
import re
import tempfile
from pathlib import Path
from typing import Literal
from sqlmodel import Session, select, func

from ...db.database import Video, Channel, Job
from ..logger import log, log_verbose, is_verbose
from .exceptions import (
    JobCancelledException,
    JobNoticeException,
    JobPausedException,
)
from . import runtime
from .runtime import (
    AUDIO_DIR,
    BACKEND_DIR,
    MANUAL_MEDIA_DIR,
    TEMP_DIR,
)


class MediaFilesMixin:
    def _get_video_download_lock(self, video_id: int) -> threading.Lock:
        with self._download_locks_guard:
            lock = self._download_locks.get(video_id)
            if lock is None:
                lock = threading.Lock()
                self._download_locks[video_id] = lock
            return lock

    def _ensure_audio_ready_for_video(self, video: Video, job_id: int = None) -> Path:
        lock = self._get_video_download_lock(int(video.id))
        with lock:
            audio_path = self.download_audio(video, job_id=job_id)
            audio_path = self._validate_and_retry_audio(video, audio_path, job_id)
        return audio_path

    def _get_ffmpeg_cmd(self):
        """Return path to ffmpeg executable."""
        ffmpeg_bin = BACKEND_DIR / "bin"
        if (ffmpeg_bin / "ffmpeg.exe").exists():
            return str(ffmpeg_bin / "ffmpeg.exe")
        return "ffmpeg"

    def _get_ffprobe_cmd(self):
        """Return path to ffprobe executable."""
        ffmpeg_bin = BACKEND_DIR / "bin"
        if (ffmpeg_bin / "ffprobe.exe").exists():
            return str(ffmpeg_bin / "ffprobe.exe")
        return "ffprobe"

    def _run_external_command(self, cmd: list[str], failure_label: str, *, timeout: int = 3600) -> None:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or failure_label).strip()
            raise RuntimeError(f"{failure_label}: {detail[:1200]}")

    def _probe_audio_file(self, audio_path: Path) -> dict:
        result = subprocess.run(
            [
                self._get_ffprobe_cmd(),
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate,channels,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "ffprobe failed").strip()
            raise RuntimeError(f"Failed to inspect audio file: {detail[:800]}")
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}
        fmt = payload.get("format") or {}
        duration = fmt.get("duration") if fmt.get("duration") is not None else stream.get("duration")
        return {
            "duration_seconds": float(duration) if duration is not None else None,
            "sample_rate": int(stream.get("sample_rate")) if stream.get("sample_rate") is not None else None,
            "channels": int(stream.get("channels")) if stream.get("channels") is not None else None,
        }

    def _analyze_audio_file(self, audio_path: Path, *, source_label: str) -> dict:
        import numpy as np
        import soundfile as sf  # type: ignore

        probe = self._probe_audio_file(audio_path)
        with tempfile.TemporaryDirectory(prefix="cleanup-analysis-") as tmp_dir:
            temp_wav = Path(tmp_dir) / "analysis.wav"
            self._run_external_command(
                [
                    self._get_ffmpeg_cmd(),
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-c:a",
                    "pcm_s16le",
                    str(temp_wav),
                ],
                "Failed to prepare audio for cleanup analysis",
                timeout=1800,
            )
            samples, _ = sf.read(str(temp_wav), dtype="float32", always_2d=False)
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        clipped_ratio = float(np.mean(np.abs(arr) >= 0.995)) if arr.size else 0.0
        return {
            "source_label": str(source_label or "Original upload"),
            "duration_seconds": probe.get("duration_seconds"),
            "sample_rate": probe.get("sample_rate"),
            "channels": probe.get("channels"),
            "peak": peak,
            "rms": rms,
            "clipped_ratio": clipped_ratio,
        }

    def _load_audio_for_pyannote(self, audio_path: str):
        """
        Load audio for pyannote using ffmpeg + soundfile.
        This bypasses torchaudio/torchcodec which have compatibility issues with dev PyTorch.
        Returns: dict with 'waveform' (torch.Tensor) and 'sample_rate' (int)
        """
        import soundfile as sf
        import torch

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate audio file isn't empty/tiny (likely corrupt download)
        file_size = audio_file.stat().st_size
        if file_size < 10_000:  # Less than 10KB is almost certainly corrupt
            raise RuntimeError(
                f"Audio file appears corrupt or empty ({file_size} bytes): {audio_path}. "
                f"This can happen with live streams still processing on YouTube. "
                f"Try deleting the audio file and re-downloading."
            )

        ffmpeg_cmd = self._get_ffmpeg_cmd()

        # Convert to mono 16kHz WAV via ffmpeg
        # Use -err_detect ignore_err to be more tolerant of minor stream issues
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [ffmpeg_cmd, '-y', '-err_detect', 'ignore_err',
                 '-i', str(audio_path), '-ar', '16000', '-ac', '1', tmp_path],
                capture_output=True,
                timeout=600,  # 10 minute timeout for very long files
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors='replace')[-800:]
                # Check for common live stream / corrupt file errors
                if 'aac' in stderr.lower() and ('error' in stderr.lower() or 'failed' in stderr.lower()):
                    raise RuntimeError(
                        f"Audio file has corrupt AAC data. This often happens with live stream "
                        f"recordings that YouTube hasn't fully processed yet. Try again later, "
                        f"or delete the audio file and re-download.\n"
                        f"FFmpeg output: {stderr[-300:]}"
                    )
                raise RuntimeError(f"FFmpeg conversion failed: {stderr}")

            # Verify the wav output is valid
            wav_size = os.path.getsize(tmp_path)
            if wav_size < 1000:
                raise RuntimeError(
                    f"FFmpeg produced an empty/tiny WAV ({wav_size} bytes) from {audio_path}. "
                    f"The source audio may be corrupt. Delete the audio file and re-download."
                )

            data, sr = sf.read(tmp_path)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sr}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _probe_audio_duration_seconds(self, audio_path: Path) -> float:
        ffprobe_cmd = self._get_ffprobe_cmd()
        try:
            result = subprocess.run(
                [
                    ffprobe_cmd,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                timeout=30,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                return float((result.stdout or "0").strip() or 0)
        except Exception:
            pass
        return 0.0

    def get_manual_media_absolute_path(self, relative_path: str | None) -> Path | None:
        rel = str(relative_path or "").strip().replace("\\", "/").lstrip("/")
        if not rel:
            return None
        candidate = (MANUAL_MEDIA_DIR / rel).resolve()
        manual_root = MANUAL_MEDIA_DIR.resolve()
        try:
            candidate.relative_to(manual_root)
        except ValueError:
            return None
        return candidate

    def get_original_manual_media_path(self, video: Video) -> Path | None:
        if (video.media_source_type or "youtube") != "upload":
            return None
        return self.get_manual_media_absolute_path(video.manual_media_path)

    def sanitize_filename(self, name: str) -> str:
        """Sanitize string to be filesystem safe"""
        # Replace invalid characters for Windows/Linux
        # Remove invalid chars: < > : " / \ | ? *
        s = re.sub(r'[<>:"/\\|?*]', '', name)
        # Strip leading/trailing spaces and dots
        s = s.strip().strip('.')
        return s or "Unknown"

    def _find_audio_file_in_dir(self, directory: Path, basenames: list[str]) -> Path | None:
        if not directory.exists():
            return None
        for base in basenames:
            for ext in ['.webm', '.m4a', '.mp4', '.opus', '.mp3', '.ogg', '.wav']:
                candidate = directory / f"{base}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def get_audio_path(self, video: Video, purpose: Literal["processing", "playback"] = "processing") -> Path:
        """Get standard audio path for a video.
        Returns the existing audio file if found (any format), otherwise
        returns the default .m4a path for new downloads."""
        if (video.media_source_type or "youtube") == "upload":
            if purpose == "playback" and bool(getattr(video, "reconstruction_use_for_playback", False)):
                rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
                if rel:
                    reconstruction_path = self.get_manual_media_absolute_path(rel)
                    if reconstruction_path is not None and reconstruction_path.exists():
                        return reconstruction_path
            cleaned_path = self.get_voicefixer_cleaned_absolute_path(video)
            apply_scope = self._voicefixer_apply_scope(video)
            use_cleaned_for_purpose = apply_scope == "both" or apply_scope == purpose
            if use_cleaned_for_purpose and cleaned_path is not None and cleaned_path.exists():
                return cleaned_path
            manual_path = self.get_original_manual_media_path(video)
            if manual_path is not None:
                return manual_path

        with Session(runtime.engine) as session:
             channel = session.get(Channel, video.channel_id)
             channel_name = channel.name if channel else "Unknown Channel"
             same_title_count = session.exec(
                 select(func.count(Video.id)).where(
                     Video.channel_id == video.channel_id,
                     Video.title == video.title,
                 )
             ).one() or 0

        safe_channel = self.sanitize_filename(channel_name)
        safe_title = self.sanitize_filename(video.title)
        safe_video_key = self.sanitize_filename((video.youtube_id or f"video_{video.id or 'unknown'}"))
        episode_slug = self.sanitize_filename(f"{safe_title}__{safe_video_key}")

        # New (collision-safe) layout: one folder per youtube_id.
        episode_dir = AUDIO_DIR / safe_channel / episode_slug
        episode_dir.mkdir(parents=True, exist_ok=True)

        basenames = [safe_title, safe_video_key, episode_slug]
        hit = self._find_audio_file_in_dir(episode_dir, basenames)
        if hit:
            return hit

        # Legacy layout (title-only folder) can collide badly for recurring stream titles
        # like "Come say hi!". Only auto-migrate when the title is unique in channel.
        legacy_dir = AUDIO_DIR / safe_channel / safe_title
        if legacy_dir.exists() and int(same_title_count) <= 1 and legacy_dir != episode_dir:
            legacy_hit = self._find_audio_file_in_dir(legacy_dir, basenames)
            if legacy_hit:
                target = episode_dir / legacy_hit.name
                try:
                    if not target.exists():
                        legacy_hit.rename(target)
                    else:
                        # Keep already migrated file, remove stale source if possible.
                        try:
                            legacy_hit.unlink(missing_ok=True)
                        except Exception:
                            pass
                    # Move transcript artifacts for this episode when available.
                    artifact_names = [
                        f"{safe_title}_transcript_raw.json",
                        f"{safe_title}.srt",
                        f"{safe_title}_speakers.srt",
                        f"{safe_title}_diarized.txt",
                    ]
                    for name in artifact_names:
                        src = legacy_dir / name
                        dst = episode_dir / name
                        if src.exists() and not dst.exists():
                            try:
                                src.rename(dst)
                            except Exception:
                                pass
                    return target
                except Exception as e:
                    log_verbose(f"Legacy audio migration skipped for video {video.id}: {e}")

        # If title is not unique in this channel, ignore legacy title-only folders to avoid
        # cross-episode transcript/audio contamination.
        if int(same_title_count) > 1 and legacy_dir.exists():
            log_verbose(
                f"Skipping legacy title-only folder for video {video.id} "
                f"({same_title_count} videos share title '{video.title}')."
            )

        # Default path for new downloads (yt-dlp will set actual extension)
        return episode_dir / f"{safe_title}.m4a"

    def download_audio(self, video: Video, job_id: int = None) -> Path:
        if (video.media_source_type or "youtube") == "upload":
            selected_path = self.get_audio_path(video)
            if not selected_path or not selected_path.exists():
                raise FileNotFoundError(f"Uploaded media file is missing for video {video.id}")
            if job_id:
                self._update_job_progress(job_id, 100)
            return selected_path

        # Determine paths
        new_output_path = self.get_audio_path(video)
        
        # Check if already exists in new location
        
        # Check if already exists in new location
        if new_output_path.exists():
            log_verbose(f"File found at new location: {new_output_path}")
            return new_output_path

        # Check for old legacy file (flat structure)
        old_output_path = AUDIO_DIR / f"{video.youtube_id}.m4a"
        if old_output_path.exists():
            log_verbose(f"Migrating legacy file from {old_output_path} to {new_output_path}")
            try:
                old_output_path.rename(new_output_path)
                return new_output_path
            except Exception as e:
                log(f"Error migrating file: {e}")
                # Fallthrough to re-download if move fails? Or just fail? 
                # Better to fail and let user know or try copy.
                pass

        # Determine ffmpeg location
        ffmpeg_bin = BACKEND_DIR / "bin"
        ffmpeg_loc = None
        if (ffmpeg_bin / "ffmpeg.exe").exists():
            ffmpeg_loc = str(ffmpeg_bin)
            log_verbose(f"Using local FFmpeg from: {ffmpeg_loc}")
        else:
            ffmpeg_loc = 'C:/Program Files/ffmpeg'
            log_verbose("Using system/fallback FFmpeg path")

        last_update = [0.0]  # Mutable for closure; throttle DB writes

        def progress_hook(d):
            if not job_id:
                return
            now = time.time()
            if now - last_update[0] < 2:
                return
            last_update[0] = now

            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes', 0)
                pct = min(int(downloaded / total * 100), 100) if total > 0 else 0

                with Session(runtime.engine) as s:
                    job = s.get(Job, job_id)
                    if job:
                        if job.status == 'paused':
                            raise JobPausedException("Job paused by user")
                        if job.status == 'cancelled':
                            raise JobCancelledException("Job cancelled by user")
                        job.progress = pct
                        s.add(job)
                        s.commit()

            elif d['status'] == 'finished':
                # Download finished, but post-processing may still occur
                log_verbose(f"Download finished: {d.get('filename', 'unknown')}")
                self._update_job_progress(job_id, 100)

        def postprocessor_hook(d):
            """Provide feedback during post-processing phases"""
            status = d.get('status')
            pp_name = d.get('postprocessor', 'Unknown')
            
            if status == 'started':
                log_verbose(f"Post-processing started: {pp_name}")
            elif status == 'finished':
                log_verbose(f"Post-processing finished: {pp_name}")

        # Prefer opus/webm over m4a â€” YouTube live stream VODs often have corrupt
        # AAC streams in m4a, while opus is consistently clean. Since we convert
        # to WAV for transcription anyway, the container format doesn't matter.

        # Store the actual downloaded file path
        downloaded_file = [None]

        def track_filename_hook(d):
            """Track the actual filename that was downloaded"""
            if d['status'] == 'finished':
                downloaded_file[0] = d.get('filename')

        ydl_opts = {
            # Prefer 'best' (mixed A/V) to avoid pacing/padding drift between YouTube's separate audio streams 
            # and the iframe player. If unavailable or too large, fallback to bestaudio.
            'format': 'best[ext=mp4]/best/bestaudio[ext=webm]/bestaudio[ext=m4a]',
            'outtmpl': str(new_output_path.with_suffix('')) + '.%(ext)s',
            'quiet': not is_verbose(),
            'verbose': is_verbose(),
            'ffmpeg_location': ffmpeg_loc,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'noprogress': not is_verbose(),
            'progress_hooks': [progress_hook, track_filename_hook],
            'postprocessor_hooks': [postprocessor_hook],
        }
        ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="download_audio")
        source_url = str(getattr(video, "source_url", "") or "").strip()
        download_url = source_url or f"https://www.youtube.com/watch?v={video.youtube_id}"
        log_verbose(f"Downloading audio for {video.youtube_id} from {download_url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([download_url])
        except Exception as e:
            notice = self._classify_ytdlp_download_notice(e)
            if notice:
                raise JobNoticeException(
                    str(notice.get("message") or "This video could not be downloaded."),
                    code=str(notice.get("code") or "notice"),
                    video_status=str(notice.get("video_status") or "pending"),
                    technical_detail=str(e),
                ) from e
            raise

        # Handle the downloaded file
        actual_file = Path(downloaded_file[0]) if downloaded_file[0] else None

        # Check for possible downloaded file paths
        if not actual_file or not actual_file.exists():
            base_path = new_output_path.with_suffix('')
            for ext in ['.webm', '.m4a', '.opus', '.mp3', '.ogg', '.wav']:
                candidate = Path(str(base_path) + ext)
                if candidate.exists():
                    actual_file = candidate
                    break

        if not actual_file or not actual_file.exists():
            raise FileNotFoundError(f"Could not find downloaded audio file for {video.youtube_id}")

        log(f"Audio downloaded as {actual_file.suffix}: {actual_file.name} ({actual_file.stat().st_size / 1024 / 1024:.1f} MB)")

        return actual_file

    def _slice_audio(self, input_path: Path, start_time: float, duration: float = None) -> Path:
        """Create a temp audio file starting from start_time, optionally capped to duration seconds."""
        start_ms = int(max(0.0, float(start_time)) * 1000)
        if duration is None:
            dur_tag = "full"
        else:
            try:
                dur_tag = str(int(max(0.0, float(duration)) * 1000))
            except Exception:
                dur_tag = "full"
        
        # Output as precise 16kHz mono WAV to avoid -c copy keyframe snapping 
        # which pulls in older audio and causes timestamp drift.
        output_path = TEMP_DIR / f"temp_slice_{start_ms}_{dur_tag}_{threading.get_ident()}_{input_path.stem}.wav"
        
        # Determine ffmpeg location (copied from download_audio logic)
        ffmpeg_bin = BACKEND_DIR / "bin"
        ffmpeg_cmd = str(ffmpeg_bin / "ffmpeg.exe") if (ffmpeg_bin / "ffmpeg.exe").exists() else "ffmpeg"

        cmd = [ffmpeg_cmd, "-y", "-ss", str(start_time), "-i", str(input_path)]
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        
        cmd.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)])
        
        log_verbose(f"Slicing audio: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            log(f"FFmpeg slice error: {e.stderr.decode()}")
            raise

    def _validate_and_retry_audio(self, video: Video, audio_path: Path, job_id: int = None) -> Path:
        """Validate downloaded audio. If corrupt, delete and retry with a fallback format.
        Returns the (possibly new) audio path."""
        ffmpeg_cmd = self._get_ffmpeg_cmd()
        try:
            probe = subprocess.run(
                [ffmpeg_cmd, '-v', 'error', '-i', str(audio_path), '-f', 'null', '-t', '10', '-'],
                capture_output=True, timeout=30,
            )
            if probe.returncode == 0:
                return audio_path  # Audio is clean

            stderr = probe.stderr.decode(errors='replace')
            log(f"Audio validation failed for {audio_path.name}: {stderr[:200]}")

            # Delete the corrupt file
            audio_path.unlink(missing_ok=True)
            log("Deleted corrupt audio file. Retrying with fallback format...")
            self._update_job_status_detail(job_id, "Re-downloading (corrupt audio detected)...")

            # Determine which format to try as fallback. Now that we default to 'best', 
            # if that is corrupt, we fall back to audio-only streams as a last resort,
            # accepting potential timing drift if it means getting the file at all.
            current_ext = audio_path.suffix.lower()
            if current_ext == '.m4a':
                fallback_format = 'bestaudio[ext=webm]/bestaudio/best'
            else:
                fallback_format = 'bestaudio[ext=m4a]/bestaudio/best'

            # Re-download with fallback format
            ffmpeg_bin = BACKEND_DIR / "bin"
            ffmpeg_loc = str(ffmpeg_bin) if (ffmpeg_bin / "ffmpeg.exe").exists() else 'C:/Program Files/ffmpeg'
            base_path = audio_path.with_suffix('')

            downloaded_file = [None]
            def track_fn(d):
                if d['status'] == 'finished':
                    downloaded_file[0] = d.get('filename')

            ydl_opts = {
                'format': fallback_format,
                'outtmpl': str(base_path) + '.%(ext)s',
                'quiet': True,
                'ffmpeg_location': ffmpeg_loc,
                'socket_timeout': 30,
                'retries': 3,
                'fragment_retries': 3,
                'progress_hooks': [track_fn],
            }
            ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="download_audio_fallback")
            import yt_dlp as _yt_dlp
            with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video.youtube_id}"])

            new_file = Path(downloaded_file[0]) if downloaded_file[0] else None
            if not new_file or not new_file.exists():
                for ext in ['.webm', '.m4a', '.opus', '.mp3', '.ogg']:
                    candidate = Path(str(base_path) + ext)
                    if candidate.exists():
                        new_file = candidate
                        break

            if not new_file or not new_file.exists():
                raise RuntimeError("Fallback download also failed â€” no audio file produced")

            # Validate the fallback download
            probe2 = subprocess.run(
                [ffmpeg_cmd, '-v', 'error', '-i', str(new_file), '-f', 'null', '-t', '10', '-'],
                capture_output=True, timeout=30,
            )
            if probe2.returncode != 0:
                new_file.unlink(missing_ok=True)
                raise RuntimeError(
                    "Both audio formats are corrupt. This video may still be processing on YouTube. "
                    "Try again later."
                )

            log(f"Fallback download OK: {new_file.name} ({new_file.stat().st_size / 1024 / 1024:.1f} MB)")
            return new_file

        except subprocess.TimeoutExpired:
            return audio_path  # Probe timed out â€” proceed with what we have
        except RuntimeError:
            raise
        except Exception as e:
            log_verbose(f"Audio validation warning (non-fatal): {e}")
            return audio_path
