"""Processing pipeline orchestration: process_video, download phase, checkpoints, transcript persistence.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import yt_dlp
import os
import time
import json
import subprocess
import gc
from pathlib import Path
from sqlmodel import Session, select

from ...db.database import Video, Channel, Speaker, TranscriptSegment, Job
from ..logger import log, log_verbose
from .exceptions import (
    JobNoticeException,
    JobPausedException,
)
from . import runtime
from .runtime import (
    AUDIO_DIR,
    BACKEND_DIR,
    DATA_DIR,
    TEMP_DIR,
)


class PipelineMixin:
    def _deserialize_transcript_words(self, raw_words, seg_start=None, seg_end=None):
        from math import isfinite

        if not raw_words:
            return None

        parsed = []
        for w in raw_words:
            try:
                ws = float(w.get("start"))
                we = float(w.get("end", ws))
                ww = str(w.get("word", "")).strip()
            except Exception:
                continue
            if not ww or not isfinite(ws):
                continue
            if not isfinite(we) or we < ws:
                we = ws
            parsed.append([ws, we, ww])

        if not parsed:
            return None

        if seg_start is not None and seg_end is not None:
            try:
                seg_start_f = float(seg_start)
                seg_end_f = float(seg_end)
            except Exception:
                seg_start_f = None
                seg_end_f = None

            if seg_start_f is not None and seg_end_f is not None and seg_end_f > seg_start_f:
                min_start = min(p[0] for p in parsed)
                max_end = max(p[1] for p in parsed)
                seg_dur = max(0.01, seg_end_f - seg_start_f)

                looks_ms_absolute = max_end > max(seg_end_f * 5, 1000)
                looks_ms_relative = min_start >= -0.5 and max_end > max(1000, seg_dur * 20)
                if looks_ms_absolute or looks_ms_relative:
                    for p in parsed:
                        p[0] /= 1000.0
                        p[1] /= 1000.0
                    min_start = min(p[0] for p in parsed)
                    max_end = max(p[1] for p in parsed)

                looks_relative = min_start >= -0.5 and max_end <= seg_dur + 1.5
                if looks_relative:
                    for p in parsed:
                        p[0] += seg_start_f
                        p[1] += seg_start_f
                    min_start = min(p[0] for p in parsed)
                    max_end = max(p[1] for p in parsed)

                if seg_start_f > 120 and max_end < seg_start_f - 5:
                    shift = seg_start_f - min_start
                    for p in parsed:
                        p[0] += shift
                        p[1] += shift

        return [
            self._build_whisper_style_word(start=ws, end=we, word=ww)
            for ws, we, ww in parsed
        ] or None

    def _word_coverage(self, items) -> float:
        if not items:
            return 1.0
        with_words = 0
        for s in items:
            try:
                if getattr(s, "words", None):
                    with_words += 1
            except Exception:
                continue
        return with_words / max(len(items), 1)

    def _load_raw_transcript_checkpoint(self, video: Video, audio_path: Path, job_id: int = None):
        safe_title = self.sanitize_filename(video.title)
        raw_transcript_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
        if not raw_transcript_path.exists():
            raise FileNotFoundError(f"Raw transcript checkpoint not found for video {video.id}")

        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for s in data.get("segments", []):
            words = self._deserialize_transcript_words(
                s.get("words"),
                seg_start=s.get("start"),
                seg_end=s.get("end"),
            )
            segments.append(
                self._build_whisper_style_segment(
                    seg_id=0,
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=words,
                )
            )

        total_duration = float(video.duration or 0)
        if total_duration <= 0 and segments:
            total_duration = float(getattr(segments[-1], "end", 0.0) or 0.0)

        engine_from_raw = str(
            data.get("transcription_engine_used")
            or data.get("engine")
            or ""
        ).strip().lower()
        if engine_from_raw not in {"parakeet", "whisper"}:
            engine_from_raw = ""

        payload_fields = {"transcription_reused_existing": True}
        if engine_from_raw:
            payload_fields["transcription_engine_used"] = engine_from_raw
        self._upsert_job_payload_fields(job_id, payload_fields)
        return segments, total_duration, engine_from_raw

    def _get_temp_transcript_path(self, video_id: int) -> Path:
        return TEMP_DIR / f"transcript_{video_id}_partial.json"

    def _get_temp_transcript_jsonl_path(self, video_id: int) -> Path:
        return TEMP_DIR / f"transcript_{video_id}_partial_segments.jsonl"

    def _get_temp_diarization_path(self, video_id: int) -> Path:
        return TEMP_DIR / f"diarization_{video_id}.rttm"

    def _reset_partial_checkpoint_state(self, video_id: int):
        with self._partial_checkpoint_guard:
            self._partial_checkpoint_counts.pop(int(video_id), None)

    def purge_artifacts(self, video_id: int, delete_raw_transcript: bool = True, delete_audio: bool = False):
        """Delete processing files. By default deletes temp checkpoints + raw transcript."""
        t_path = self._get_temp_transcript_path(video_id)
        t_jsonl_path = self._get_temp_transcript_jsonl_path(video_id)
        d_path = self._get_temp_diarization_path(video_id)

        paths_to_delete = [t_path, t_jsonl_path, d_path]

        # Also delete the raw transcript checkpoint so re-processing actually re-transcribes
        if delete_raw_transcript:
            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if video:
                    try:
                        audio_path = self.get_audio_path(video)
                        safe_title = self.sanitize_filename(video.title)
                        raw_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
                        paths_to_delete.append(raw_path)
                        if delete_audio and audio_path.exists():
                            paths_to_delete.append(audio_path)
                    except Exception:
                        pass

        for p in paths_to_delete:
            if p.exists():
                try:
                    p.unlink()
                    log_verbose(f"Purged file: {p}")
                except Exception as e:
                    log(f"Failed to purge {p}: {e}")
        self._reset_partial_checkpoint_state(video_id)

    def _save_partial_transcript(self, video_id: int, segments: list, total_duration: float):
        """Persist partial transcript incrementally (JSONL append + tiny metadata file)."""
        meta_path = self._get_temp_transcript_path(video_id)
        jsonl_path = self._get_temp_transcript_jsonl_path(video_id)
        total_segments = len(segments)

        try:
            with self._partial_checkpoint_guard:
                saved_count = self._partial_checkpoint_counts.get(int(video_id))

            if saved_count is None:
                if jsonl_path.exists():
                    line_count = 0
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                line_count += 1
                    saved_count = line_count
                else:
                    saved_count = 0

            if total_segments < saved_count:
                # Safety reset if caller restarted from an earlier state.
                saved_count = 0
                if jsonl_path.exists():
                    jsonl_path.unlink()

            if total_segments > saved_count:
                mode = "a" if saved_count > 0 else "w"
                with open(jsonl_path, mode, encoding="utf-8") as out:
                    for s in segments[saved_count:]:
                        payload = {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "words": (
                                [
                                    {
                                        "start": float(getattr(w, "start", 0.0) or 0.0),
                                        "end": float(getattr(w, "end", getattr(w, "start", 0.0)) or getattr(w, "start", 0.0)),
                                        "word": str(getattr(w, "word", "") or ""),
                                    }
                                    for w in (s.words or [])
                                ]
                                if getattr(s, "words", None)
                                else None
                            ),
                        }
                        out.write(json.dumps(payload, ensure_ascii=False))
                        out.write("\n")
                saved_count = total_segments
                with self._partial_checkpoint_guard:
                    self._partial_checkpoint_counts[int(video_id)] = saved_count

            # Metadata stays tiny and fast to rewrite.
            data = {
                "video_id": int(video_id),
                "timestamp": time.time(),
                "total_duration": float(total_duration or 0.0),
                "segment_count": int(saved_count),
                "format": "jsonl",
                "segments_path": str(jsonl_path),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            log_verbose(f"Failed to save partial transcript: {e}")

    def _load_partial_transcript(self, video_id: int):
        """Load partial segments if available"""
        meta_path = self._get_temp_transcript_path(video_id)
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        # Backward compatibility: legacy checkpoint stored full segment list inline.
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            with self._partial_checkpoint_guard:
                self._partial_checkpoint_counts[int(video_id)] = len(data.get("segments") or [])
            return data

        if not isinstance(data, dict) or (data.get("format") or "").lower() != "jsonl":
            return None

        jsonl_path_raw = data.get("segments_path")
        jsonl_path = Path(jsonl_path_raw) if jsonl_path_raw else self._get_temp_transcript_jsonl_path(video_id)
        if not jsonl_path.exists():
            return None

        segments = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        segments.append(item)
        except Exception:
            return None

        with self._partial_checkpoint_guard:
            self._partial_checkpoint_counts[int(video_id)] = len(segments)

        return {
            "video_id": int(video_id),
            "timestamp": data.get("timestamp"),
            "total_duration": data.get("total_duration"),
            "segments": segments,
        }

    def process_video(self, video_id: int, job_id: int = None):
        """Orchestrate the ingestion process via phases."""
        self._ensure_device()
        self._record_cuda_health_event("job_start", job_id=job_id, extra={"video_id": int(video_id)})
        baseline_cuda_free_b = 0
        if self.device == "cuda":
            baseline_cuda_free_b = int(self._cuda_memory_snapshot().get("free") or 0)
            self._upsert_job_payload_fields(
                job_id,
                {
                    "job_cuda_free_gb_start": round(float(baseline_cuda_free_b) / (1024 ** 3), 2) if baseline_cuda_free_b > 0 else 0.0,
                },
            )
        try:
            with Session(runtime.engine) as session:
                current_video = session.get(Video, video_id)
                if current_video and current_video.access_restricted:
                    raise JobNoticeException(
                        str(current_video.access_restriction_reason or "This video is not accessible with the current YouTube session."),
                        code="youtube_access_restricted",
                        video_status="access_restricted",
                        technical_detail=str(current_video.access_restriction_reason or "access restricted"),
                    )
            # Phase 1: Download
            video_detached, audio_path = self._process_download_phase(video_id, job_id)
            
            # Phase 2: Transcribe
            segments, duration = self._process_transcribe_phase(video_detached, audio_path, job_id)
            
            # Phase 3: Diarize
            self._process_diarize_phase(video_detached, audio_path, segments, job_id)
            payload = {}
            if job_id:
                with Session(runtime.engine) as session:
                    job = session.get(Job, job_id)
                    payload = self._load_job_payload(job.payload_json if job else None)
                self._record_transcript_optimization_completion(job_id, video_id, payload)

            # Phase 4: enqueue follow-up funny-moment analysis in its own queue so
            # transcript/diarization throughput is not blocked by LLM/acoustic tasks.
            try:
                self._enqueue_job(video_id, "funny_detect", payload={"force": True, "source": "post_process"})
            except Exception as e:
                # Keep process pipeline success even if queueing follow-up fails.
                log(f"Failed to enqueue follow-up funny_detect for video {video_id}: {e}")
            
        except JobPausedException:
            log(f"Video {video_id} paused by user")
            # Update status to pending
            with Session(runtime.engine) as session:
                v = session.get(Video, video_id)
                if v:
                    v.status = "pending"
                    session.add(v)
                    session.commit()
            raise
        except JobNoticeException as e:
            log(f"Notice processing video {video_id}: {e.notice_message}")
            with Session(runtime.engine) as session:
                v = session.get(Video, video_id)
                if v:
                    v.status = e.video_status
                    if e.video_status == "access_restricted" or e.notice_code in {"youtube_members_only", "youtube_private_video", "youtube_auth_required", "youtube_access_restricted"}:
                        v.access_restricted = True
                        v.access_restriction_reason = e.notice_message
                    session.add(v)
                    session.commit()
            if job_id:
                self._mark_job_notice(
                    job_id,
                    video_id,
                    code=e.notice_code,
                    message=e.notice_message,
                    technical_detail=e.technical_detail,
                    video_status=e.video_status,
                )
            raise RuntimeError(e.notice_message) from e
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f"Error processing video {video_id}: {e}")
            log(tb)
            
            # Fail status
            with Session(runtime.engine) as session:
                v = session.get(Video, video_id)
                if v:
                    v.status = "failed"
                    session.add(v)
                    session.commit()
            # Bubble up with traceback tail so queue job.error is actionable even
            # when worker-level traceback logging is unavailable.
            raise RuntimeError(f"{e}\n{tb[-3200:]}") from e
        finally:
            if self.device == "cuda":
                self._release_diarization_models("process_video_finally", job_id=job_id)
                self._clear_cuda_cache()
            gc.collect()
            if self.device == "cuda":
                self._maybe_recover_cuda_headroom(baseline_cuda_free_b, job_id=job_id)
            self._record_cuda_health_event("job_end", job_id=job_id, extra={"video_id": int(video_id)})

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _save_transcripts(self, session: Session, video: Video, segments: list, audio_path: Path):
        """Save SRT and Diarized Text files"""
        if not segments:
            # If we used the stream processing loop, 'segments' valid variable might be 
            # tricky if we didn't accumulate them in 'final_segments'. 
            # But the loop above saves into DB. We can re-query DB to be safe and consistent.
            # Rerunning query ensures we have the assigned speaker IDs correctly loaded.
            segments = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.start_time)).all()

        channel = session.get(Channel, video.channel_id)
        channel_name = channel.name if channel else "Unknown"
        safe_channel = self.sanitize_filename(channel_name)
        safe_title = self.sanitize_filename(video.title)
        
        # Directory logic duplicated? We can assume audio_path parent is the dir.
        # But audio_path might be the migrated one or not? 
        # Ideally rely on the passed audio_path parent.
        out_dir = audio_path.parent
        if not out_dir.exists():
            # Fallback
            out_dir = AUDIO_DIR / safe_channel / safe_title
            out_dir.mkdir(parents=True, exist_ok=True)
            
        base_name = safe_title

        # 1. Standard SRT (No Speaker Names)
        srt_path = out_dir / f"{base_name}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = self._format_timestamp(seg.start_time)
                end = self._format_timestamp(seg.end_time)
                f.write(f"{i}\n{start} --> {end}\n{seg.text}\n\n")
        log_verbose(f"Saved SRT: {srt_path}")

        # 2. Diarized Text (Readable)
        txt_path = out_dir / f"{base_name}_diarized.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript for: {video.title}\n")
            f.write(f"Channel: {channel_name}\n")
            f.write(f"Date: {video.published_at}\n\n")
            
            for seg in segments:
                speaker_name = "Unknown"
                if seg.speaker_id:
                    # We have ID, need name. Inefficient to query every time? 
                    # Prefetch map or accessing .speaker relationship if loaded
                    if seg.speaker:
                        speaker_name = seg.speaker.name
                    else:
                        # Should have been selectinload-ed or lazily loaded
                        spk = session.get(Speaker, seg.speaker_id)
                        if spk: speaker_name = spk.name
                
                time_str = f"[{self._format_timestamp(seg.start_time).replace(',','.')}]"
                f.write(f"{time_str} {speaker_name}: {seg.text}\n")
        log_verbose(f"Saved Diarized Text: {txt_path}")

        # 3. Diarized SRT (SRT with Speaker prefixes)
        spk_srt_path = out_dir / f"{base_name}_speakers.srt"
        with open(spk_srt_path, "w", encoding="utf-8") as f:
             for i, seg in enumerate(segments, 1):
                start = self._format_timestamp(seg.start_time)
                end = self._format_timestamp(seg.end_time)
                
                speaker_name = "Unknown"
                if seg.speaker:
                    speaker_name = seg.speaker.name
                elif seg.speaker_id:
                    spk = session.get(Speaker, seg.speaker_id)
                    if spk: speaker_name = spk.name

                f.write(f"{i}\n{start} --> {end}\n[{speaker_name}] {seg.text}\n\n")
        log_verbose(f"Saved Speaker SRT: {spk_srt_path}")

    def _process_download_phase(self, video_id: int, job_id: int = None):
        """Phase 1: Update status and download audio.
        Returns (video_obj, audio_path). video_obj is detached from session."""
        
        # Short-lived session for status update
        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            # Eagerly load relationship needed for path generation
            if video.channel:
                 _ = video.channel.name # Touch to load
            
            video.status = "downloading"
            session.add(video)

            if job_id:
               job = session.get(Job, job_id)
               if job:
                   job.status = "downloading"
                   session.add(job)

            session.commit()
            session.refresh(video)
            
            # Ensure relationship is loaded before detaching
            _ = video.channel

            # Detach to use outside session
            session.expunge(video)
            if video.channel:
                session.expunge(video.channel)

        self._record_job_stage_start(job_id, "download")
        
        # Download (long running, no DB lock). Serialize per-video download work so
        # the background prefetch worker and the active job cannot race each other.
        lock = self._get_video_download_lock(video_id)
        with lock:
            audio_path = self.download_audio(video, job_id=job_id)

            # Validate downloaded audio with a quick ffmpeg decode check
            audio_path = self._validate_and_retry_audio(video, audio_path, job_id)

        return video, audio_path

    def extract_frame_and_crop(self, video_id: int, timestamp: float, crop_coords: dict) -> str:
        """
        Extract a frame from video at timestamp and crop it.
        crop_coords: {x, y, w, h} (relative 0-1)
        Returns: Path string relative to static route (e.g. /thumbnails/speakers/1.jpg)
        """
        
        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            yt_id = video.youtube_id

        # 1. Get YouTube URL
        log_verbose(f"Extracting frame for {yt_id} at {timestamp}s with crop {crop_coords}")
        source_w = None
        source_h = None
        input_source = None
        media_source_type = str(getattr(video, "media_source_type", "") or "youtube").lower()
        try:
            if media_source_type in {"upload", "tiktok"}:
                local_path = self.get_audio_path(video)
                if not local_path.exists():
                    raise RuntimeError("Local media file is not available yet.")
                if str(getattr(video, "media_kind", "") or "").lower() == "audio":
                    raise RuntimeError("Cannot extract a speaker thumbnail from audio-only media.")
                input_source = str(local_path)
            else:
                # Get generic URL
                url = f"https://www.youtube.com/watch?v={yt_id}"

                # Use yt_dlp Python API to get the direct stream URL
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
                }
                ydl_opts = self._apply_ytdlp_auth_opts(ydl_opts, purpose="extract_frame")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    input_source = info.get('url')
                    source_w = info.get("width")
                    source_h = info.get("height")
                    if not input_source:
                        # If no direct url, look in formats
                        formats = info.get('formats', [])
                        for f in reversed(formats):
                            if f.get('url') and f.get('vcodec') != 'none':
                                input_source = f.get('url')
                                source_w = f.get("width") or source_w
                                source_h = f.get("height") or source_h
                                break
                    if not input_source:
                        raise RuntimeError("Could not extract video stream URL")

        except Exception as e:
            raise RuntimeError(f"Failed to get video URL: {e}")

        # 2. Extract and Crop with FFmpeg
        # ffmpeg -ss <timestamp> -i <url> -vf "crop=w:h:x:y" -vframes 1 -q:v 2 output.jpg
        # We need absolute pixel coords. We don't verify video resolution?
        # FFmpeg crop filter accepts iw and ih (input width/height)
        # crop=iw*w:ih*h:iw*x:ih*y
        
        x = float(crop_coords.get('x', 0) or 0)
        y = float(crop_coords.get('y', 0) or 0)
        w = float(crop_coords.get('w', 1) or 1)
        h = float(crop_coords.get('h', 1) or 1)

        if media_source_type not in {"upload", "tiktok"}:
            # Legacy YouTube crop overlay is rendered in a fixed 16:9 box.
            # Map those overlay-normalized coords to the real source frame.
            try:
                sw = float(source_w) if source_w else 0.0
                sh = float(source_h) if source_h else 0.0
                if sw > 0 and sh > 0:
                    overlay_aspect = 16.0 / 9.0
                    source_aspect = sw / sh
                    if source_aspect < overlay_aspect:
                        # Pillarbox: visible video occupies only the center X range.
                        visible_w = source_aspect / overlay_aspect
                        pad_x = (1.0 - visible_w) / 2.0
                        x = (x - pad_x) / visible_w
                        w = w / visible_w
                    elif source_aspect > overlay_aspect:
                        # Letterbox: visible video occupies only the center Y range.
                        visible_h = overlay_aspect / source_aspect
                        pad_y = (1.0 - visible_h) / 2.0
                        y = (y - pad_y) / visible_h
                        h = h / visible_h
            except Exception:
                # If mapping fails, continue with raw coords.
                pass

        # Add a modest safety margin so thumbnails retain context and do not look
        # unnaturally zoomed even when the face box is tight.
        try:
            pad_ratio = float(os.getenv("SPEAKER_THUMBNAIL_CROP_PADDING", "0.10") or "0.10")
        except Exception:
            pad_ratio = 0.10
        pad_ratio = max(0.0, min(0.5, pad_ratio))
        if pad_ratio > 0:
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            w = w * (1.0 + (2.0 * pad_ratio))
            h = h * (1.0 + (2.0 * pad_ratio))
            x = cx - (w / 2.0)
            y = cy - (h / 2.0)

        # Clamp and normalize crop coords defensively. The frontend should already
        # constrain the drag box, but ffmpeg crop will fail if x/y/w/h exceed the
        # frame bounds or collapse to zero.
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        if x + w > 1.0:
            w = max(0.0, 1.0 - x)
        if y + h > 1.0:
            h = max(0.0, 1.0 - y)
        min_rel = 0.01
        if w < min_rel or h < min_rel:
            raise RuntimeError("Selected crop area is too small. Draw a larger square around the face.")
        
        # Ensure target dir
        thumb_dir = DATA_DIR / "thumbnails" / "speakers"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp file name - will be renamed/moved by caller or used as is?
        # Let's generate a unique filename
        filename = f"extract_{video_id}_{int(timestamp)}_{int(time.time())}.jpg"
        output_path = thumb_dir / filename
        
        # Determine ffmpeg
        ffmpeg_bin = BACKEND_DIR / "bin"
        ffmpeg_cmd = "ffmpeg"
        if (ffmpeg_bin / "ffmpeg.exe").exists():
            ffmpeg_cmd = str(ffmpeg_bin / "ffmpeg.exe")

        if input_source and media_source_type in {"upload", "tiktok"}:
            ffprobe_cmd = "ffprobe"
            if (ffmpeg_bin / "ffprobe.exe").exists():
                ffprobe_cmd = str(ffmpeg_bin / "ffprobe.exe")
            probe_cmd = [
                ffprobe_cmd,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                str(input_source),
            ]
            try:
                probe_kwargs = {}
                if os.name == 'nt':
                    probe_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                probe = subprocess.run(
                    probe_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    **probe_kwargs,
                )
                dims = (probe.stdout or "").strip().split("x")
                if len(dims) == 2:
                    source_w = float(dims[0])
                    source_h = float(dims[1])
            except Exception:
                pass

        # Debug logging
        debug_log_path = DATA_DIR / "debug_manual.log"
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"DEBUG: ingestion.py - ffmpeg_cmd: {ffmpeg_cmd}\n")
            f.write(f"DEBUG: ingestion.py - crop_coords: {x},{y},{w},{h}\n")
            f.write(f"DEBUG: ingestion.py - input_source: {input_source}\n")
            f.write(f"DEBUG: ingestion.py - output_path: {output_path}\n")

        # FFmpeg filter: crop=w=iw*0.5:h=ih*0.5:x=iw*0.25:y=ih*0.25
        # floor() keeps ffmpeg crop dimensions integer and stable
        filter_str = f"crop=w=floor(iw*{w}):h=floor(ih*{h}):x=floor(iw*{x}):y=floor(ih*{y})"
        
        cmd = [
            ffmpeg_cmd, "-y",
            "-ss", str(timestamp),
            "-i", str(input_source),
            "-vf", filter_str,
            "-vframes", "1",
            "-q:v", "2", # High quality jpeg
            str(output_path)
        ]
        
        # print(f"RUNNING FFMPEG: {cmd}") # DEBUG
        try:
            log_verbose("Running ffmpeg frame extract...")
            
            # Use temp file for stderr to avoid memory issues and capturing crashes
            stderr_file = Path(output_path).parent / f"{Path(output_path).stem}.err"
            
            kwargs = {}
            if os.name == 'nt':
                # Prevent console window from popping up
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

            with open(stderr_file, "w") as err_log:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=err_log, **kwargs)
            
            # Clean up error log on success
            if stderr_file.exists():
                stderr_file.unlink()

            log_verbose(f"Extracted thumbnail to {output_path}")
            
            # Return relative path for API
            return f"/thumbnails/speakers/{filename}"
            
        except subprocess.CalledProcessError:
            # Read error from file
            stderr_content = "Unknown error"
            stderr_file = Path(output_path).parent / f"{Path(output_path).stem}.err"
            if stderr_file.exists():
                try:
                    stderr_content = stderr_file.read_text()
                    stderr_file.unlink() # Clean up
                except:
                    pass

            error_msg = f"FFmpeg frame extraction failed: {stderr_content}"
            log(error_msg)
            raise RuntimeError(error_msg)
        except Exception:
            raise
