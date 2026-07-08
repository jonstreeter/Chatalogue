import yt_dlp
import os
import time
import json
import subprocess
import threading
import pickle
import gc
from pathlib import Path
from typing import Literal
from sqlmodel import Session, select, func
from datetime import datetime

# NOTE: Heavy ML libraries (torch, faster_whisper, pyannote, numpy, scipy)
# are imported lazily inside _load_models() and related methods to avoid
# blocking the process at startup. Only download/queue operations run
# without them.

from ...db.database import Video, Channel, Speaker, SpeakerEmbedding, TranscriptSegment, TranscriptSegmentRevision, Job, FunnyMoment
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
    DIARIZE_JOB_TYPES,
    PROCESS_JOB_TYPES,
    TEMP_DIR,
    ensure_dirs,
)
from .cuda_memory import CudaMemoryMixin
from .transcription import TranscriptionEngineMixin
from .transcript_quality import TranscriptQualityMixin

from .job_lifecycle import JobLifecycleMixin
from .speaker_identity import SpeakerIdentityMixin
from .youtube_download import YoutubeDownloadMixin

from .channels import ChannelSyncMixin
from .media_files import MediaFilesMixin

from .audio_cleanup import AudioCleanupMixin
from .reconstruction import ReconstructionMixin

from .funny_moments import FunnyMomentsMixin
from .llm import LlmProviderMixin
from .youtube_metadata import YoutubeMetadataMixin
from .clips import ClipsMixin

class IngestionService(
    ClipsMixin,
    YoutubeMetadataMixin,
    LlmProviderMixin,
    FunnyMomentsMixin,
    ReconstructionMixin,
    AudioCleanupMixin,
    MediaFilesMixin,
    ChannelSyncMixin,
    YoutubeDownloadMixin,
    SpeakerIdentityMixin,
    JobLifecycleMixin,
    TranscriptionEngineMixin,
    CudaMemoryMixin,
    TranscriptQualityMixin,
):
    def __init__(self):
        ensure_dirs()

        # Apply safer CUDA allocator defaults before any lazy torch import. This
        # reduces allocator fragmentation on long-lived Windows workers where
        # Parakeet and diarization alternate ownership of large CUDA blocks.
        self._configure_cuda_allocator()

        # Install pkg_resources shim before ANY lazy ML import can happen.
        # ctranslate2, pyannote.audio, and NeMo all import pkg_resources on Windows.
        # The shim must be in sys.modules before the first model import regardless of
        # which transcription engine (Whisper / Parakeet) or code path is taken.
        self._ensure_ctranslate2_pkg_resources()

        # Ensure local bin (with ffmpeg DLLs) is in PATH for torchaudio/torchcodec
        # This must be done before loading models
        bin_dir = str(BACKEND_DIR / "bin")
        if bin_dir not in os.environ["PATH"]:
            log_verbose(f"Adding {bin_dir} to PATH for FFmpeg DLLs")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

        log_verbose("Ingestion Service initialized (models will load on first use)")

        # Models are loaded lazily to avoid blocking startup with heavy ML imports
        self.device = None
        self.whisper_model = None
        self.parakeet_model = None
        self.diarization_pipeline = None
        self.embedding_model = None
        self.embedding_inference = None
        self._diarization_load_error = None  # Track actual error for debugging
        self._force_float32 = False  # Set True if GPU doesn't support FP16 cuBLAS ops
        self._whisper_compute_type = None
        self._whisper_device = None
        self._whisper_backend = None
        self._whisper_model_cache_key = None
        self._gpu_total_vram_bytes = 0
        self._cuda_memory_fraction_applied = None
        self._cuda_unhealthy_reason = None
        self._cuda_unhealthy_since = None
        self._cuda_recovery_pending = False
        self._cuda_fault_count = 0
        self._cuda_consecutive_fault_count = 0
        self._cuda_degraded_reason = None
        self._cuda_soft_reset_count = 0
        self._cuda_oom_backoff_count = 0
        self._cuda_health_guard = threading.Lock()
        self._cuda_health_events = []
        self._component_memory_guard = threading.Lock()
        self._component_memory_estimates = {
            "parakeet": {"loaded": False, "ram_bytes": 0, "vram_bytes": 0},
            "whisper": {"loaded": False, "ram_bytes": 0, "vram_bytes": 0},
            "pyannote": {"loaded": False, "ram_bytes": 0, "vram_bytes": 0},
        }
        self._reconstruction_tts_model_guard = threading.Lock()
        self._reconstruction_tts_model = None
        self._reconstruction_tts_model_cache_key = None
        # Dynamic, in-process Parakeet batch cap that ratchets down after CUDA OOMs.
        # Persists for this backend process lifetime (resets on restart).
        self._parakeet_dynamic_batch_cap = None
        # Coordinate background audio prefetch and normal processing so a video is
        # never downloaded twice concurrently.
        self._download_locks_guard = threading.Lock()
        self._download_locks = {}
        self._prefetch_backoff_guard = threading.Lock()
        self._prefetch_backoff_until = {}
        self._funny_progress_lock = threading.Lock()
        self._funny_progress_by_video = {}
        self._workbench_progress_lock = threading.Lock()
        self._workbench_progress_by_video = {}
        self._speaker_match_cache_guard = threading.Lock()
        self._speaker_match_cache = {}
        self._partial_checkpoint_guard = threading.Lock()
        self._partial_checkpoint_counts = {}
        self._pipeline_focus_guard = threading.Lock()
        self._pipeline_focus_mode: Literal["transcribe", "diarize"] = "transcribe"
        # Proactive shaping for hosted NVIDIA NIM calls to reduce 429 bursts during
        # chunked/global-summary + per-moment explain runs.
        self._nvidia_nim_request_lock = threading.Lock()
        self._nvidia_nim_next_allowed_at = 0.0

        # Check if a previous auto-restart loop locked us into CPU-only mode.
        restart_state = self._read_cuda_restart_state()
        if restart_state.get("permanent_cpu_mode"):
            self._cuda_unhealthy_reason = (
                "CUDA disabled after repeated auto-restart failures. "
                "Use 'Retry GPU' in the UI or POST /system/cuda-restart-state/reset then restart."
            )
            log(f"CUDA permanently disabled due to prior restart loop. "
                f"Reason: {restart_state.get('last_restart_reason', 'unknown')}")

    def get_pipeline_focus_mode(self) -> Literal["transcribe", "diarize"]:
        with self._pipeline_focus_guard:
            return self._pipeline_focus_mode

    def get_pipeline_execution_mode(self) -> Literal["sequential", "staged"]:
        mode = (os.getenv("PIPELINE_EXECUTION_MODE") or "sequential").strip().lower()
        return "staged" if mode == "staged" else "sequential"

    def set_pipeline_focus_mode(self, mode: str) -> Literal["transcribe", "diarize"]:
        normalized: Literal["transcribe", "diarize"] = "diarize" if str(mode or "").strip().lower() == "diarize" else "transcribe"
        with self._pipeline_focus_guard:
            self._pipeline_focus_mode = normalized
        log(f"Pipeline queue focus set to {normalized}.")
        return normalized






















































































































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









































    # ── CUDA auto-restart state management ───────────────────────────────




























    def _build_pyannote_progress_hook(self, job_id: int | None):
        if not job_id:
            return None

        stage_ranges = {
            "segmentation": (0, 24),
            "speaker_counting": (24, 28),
            "embeddings": (28, 42),
            "discrete_diarization": (42, 45),
        }
        stage_labels = {
            "segmentation": "running segmentation",
            "speaker_counting": "counting active speakers",
            "embeddings": "extracting speaker embeddings",
            "discrete_diarization": "building diarization timeline",
        }
        state = {
            "last_progress": -1,
            "last_detail": None,
            "last_update_at": 0.0,
        }

        def hook(
            step_name,
            step_artifact,
            file=None,
            total: int | None = None,
            completed: int | None = None,
        ):
            name = str(step_name or "").strip().lower()
            if name not in stage_ranges:
                return

            start_pct, end_pct = stage_ranges[name]
            span = max(0, end_pct - start_pct)
            if total and total > 0 and completed is not None:
                fraction = max(0.0, min(float(completed) / float(total), 1.0))
            else:
                fraction = 1.0
            progress = int(round(start_pct + (span * fraction)))
            progress = max(0, min(45, progress))

            label = stage_labels.get(name, name.replace("_", " "))
            if total and total > 1 and completed is not None:
                detail = f"Diarizing speakers: {label} ({int(completed)}/{int(total)})..."
            else:
                detail = f"Diarizing speakers: {label}..."

            now = time.time()
            should_refresh = (
                detail != state["last_detail"]
                or progress >= int(state["last_progress"]) + 1
                or (now - float(state["last_update_at"])) >= 1.0
                or (total and completed is not None and completed >= total)
            )
            if not should_refresh:
                return

            self._update_job_status_detail(job_id, detail)
            self._update_job_progress(job_id, progress)
            state["last_detail"] = detail
            state["last_progress"] = progress
            state["last_update_at"] = now

        return hook

    def _run_diarization_with_adaptive_batch(self, audio_input, job_id: int = None):
        current_batch = self._get_pyannote_batch_size()
        min_batch = max(1, int((os.getenv("PYANNOTE_MIN_BATCH_SIZE") or "8").strip() or "8"))
        self._set_pyannote_batch_size(current_batch)
        self._upsert_job_payload_fields(
            job_id,
            {
                "pyannote_batch_size_requested": int(current_batch),
                "pyannote_batch_size_effective": int(current_batch),
            },
        )

        while True:
            try:
                hook = self._build_pyannote_progress_hook(job_id)
                return self.diarization_pipeline(audio_input, hook=hook)
            except RuntimeError as e:
                if self._is_cuda_oom(e) and self.device == "cuda" and current_batch > min_batch:
                    next_batch = max(min_batch, current_batch // 2)
                    if next_batch < current_batch:
                        log(
                            f"Pyannote CUDA OOM at batch_size={current_batch}. "
                            f"Retrying with batch_size={next_batch}."
                        )
                        self._clear_cuda_cache()
                        gc.collect()
                        current_batch = next_batch
                        self._set_pyannote_batch_size(current_batch)
                        self._upsert_job_payload_fields(
                            job_id,
                            {
                                "pyannote_batch_size_effective": int(current_batch),
                                "pyannote_oom_backoff": True,
                            },
                        )
                        self._update_job_status_detail(
                            job_id,
                            f"Diarization VRAM pressure detected. Retrying with smaller pyannote batch ({current_batch})..."
                        )
                        continue
                raise

    def _should_unload_diarization_after_job(self, job_id: int = None) -> bool:
        mode = (os.getenv("DIARIZATION_UNLOAD_AFTER_JOB") or "auto").strip().lower()
        decision = {"diarization_unload_mode": mode}
        if mode in {"1", "true", "yes", "on"}:
            decision.update({"diarization_unload_after_job": True, "diarization_unload_reason": "forced_true"})
            self._upsert_job_payload_fields(job_id, decision)
            return True
        if mode in {"0", "false", "no", "off"}:
            decision.update({"diarization_unload_after_job": False, "diarization_unload_reason": "forced_false"})
            self._upsert_job_payload_fields(job_id, decision)
            return False
        if self.device != "cuda":
            decision.update({"diarization_unload_after_job": False, "diarization_unload_reason": "non_cuda_keep_loaded"})
            self._upsert_job_payload_fields(job_id, decision)
            return False
        if self._cuda_fault_count > 0 or self._cuda_recovery_pending:
            decision.update({"diarization_unload_after_job": True, "diarization_unload_reason": "post_fault_conservative"})
            self._upsert_job_payload_fields(job_id, decision)
            return True

        if self.get_pipeline_execution_mode() == "sequential":
            coexist_ok, coexist_reason, free_gb, total_gb = self._can_keep_parakeet_and_diarization_resident(job_id=job_id)
            decision.update(
                {
                    "diarization_unload_after_job": not coexist_ok,
                    "diarization_unload_reason": "sequential_keep_loaded" if coexist_ok else f"sequential_{coexist_reason}",
                    "diarization_keep_loaded_free_gb": round(free_gb, 2),
                    "diarization_keep_loaded_total_gb": round(total_gb, 2),
                }
            )
            self._upsert_job_payload_fields(job_id, decision)
            return not coexist_ok

        focus_mode = self.get_pipeline_focus_mode()
        has_transcribe_backlog = self._has_jobs_of_types(PROCESS_JOB_TYPES, {"queued", "running", "downloading", "transcribing"})
        has_more_diarize_work = self._has_jobs_of_types(DIARIZE_JOB_TYPES, {"queued", "running", "diarizing"})

        if has_transcribe_backlog:
            decision.update({"diarization_unload_after_job": True, "diarization_unload_reason": "transcribe_backlog_present"})
            self._upsert_job_payload_fields(job_id, decision)
            return True

        # Only keep pyannote warm if the worker is explicitly focused on diarization
        # and there is more diarization work ready to drain immediately. Otherwise it
        # just sits on VRAM and steals headroom from the next Parakeet job.
        if focus_mode != "diarize":
            decision.update({"diarization_unload_after_job": True, "diarization_unload_reason": "focus_not_diarize"})
            self._upsert_job_payload_fields(job_id, decision)
            return True
        if not has_more_diarize_work:
            decision.update({"diarization_unload_after_job": True, "diarization_unload_reason": "no_diarize_backlog"})
            self._upsert_job_payload_fields(job_id, decision)
            return True

        decision.update({"diarization_unload_after_job": False, "diarization_unload_reason": "auto_keep_loaded"})
        self._upsert_job_payload_fields(job_id, decision)
        return False


























































































































































































        












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

    def _process_transcribe_phase(self, video: Video, audio_path: Path, job_id: int = None, force_non_batched: bool = False):
        """Phase 2: Transcribe audio. Checks for existing transcript to skip."""
        import os
        
        # 1. Update status
        with Session(runtime.engine) as session:
            # Re-attach or fetch fresh to update
            v = session.get(Video, video.id)
            if v:
                v.status = "transcribing"
                session.add(v)
            
            if job_id:
               j = session.get(Job, job_id)
               if j:
                   j.status = "transcribing"
                   j.progress = 0
                   session.add(j)
            session.commit()
        self._record_job_stage_start(job_id, "transcribe_phase")

        log(f"Processing: {video.title}")
        log("Transcribing...")

        # Resolve requested engine plus language-aware routing early so we can
        # persist useful metadata even when we reuse an existing raw transcript.
        route = self._resolve_transcription_route(video, audio_path, job_id)
        selected_engine = str(route.get("engine") or self._select_transcription_engine()).strip().lower()
        route_language = self._normalize_language_code(route.get("language"))
        self._upsert_job_payload_fields(
            job_id,
            {
                "transcription_engine_requested": route.get("requested_engine"),
                "transcription_engine_routed": selected_engine,
                "transcription_route_language": route_language,
                "transcription_route_language_confidence": route.get("language_confidence"),
                "transcription_route_language_source": route.get("language_source"),
                "transcription_route_language_reason": route.get("language_reason"),
                "transcription_route_multilingual": bool(route.get("multilingual_route_applied")),
                "transcription_route_whisper_model": route.get("whisper_model_override"),
                "transcription_route_operational_applied": bool(route.get("operational_route_applied")),
                "transcription_route_operational_reason": route.get("operational_route_reason"),
            },
        )
        force_retranscription = False
        if job_id:
            try:
                with Session(runtime.engine) as session:
                    job = session.get(Job, job_id)
                    payload = self._load_job_payload(job.payload_json if job else None)
                    force_retranscription = bool(payload.get("force_retranscription"))
            except Exception:
                force_retranscription = False

        # 2. Check for completed raw transcript (checkpoint)
        safe_title = self.sanitize_filename(video.title)
        raw_transcript_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
        
        existing_segments = []
        total_duration = 0

        def _deserialize_words(raw_words, seg_start=None, seg_end=None):
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
                if not ww:
                    continue
                if not isfinite(ws):
                    continue
                if not isfinite(we):
                    we = ws
                if we < ws:
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

                    # Legacy resume bug repair: words stayed near zero while segment start was offset.
                    if seg_start_f > 120 and max_end < seg_start_f - 5:
                        shift = seg_start_f - min_start
                        for p in parsed:
                            p[0] += shift
                            p[1] += shift

            out = []
            for ws, we, ww in parsed:
                out.append(self._build_whisper_style_word(start=ws, end=we, word=ww))
            return out or None

        def _word_coverage(items):
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

        min_word_coverage = float(os.getenv("TRANSCRIPTION_MIN_WORD_COVERAGE", "0.90"))
        
        if raw_transcript_path.exists() and not force_retranscription:
            try:
                log(f"Found existing raw transcript at {raw_transcript_path}. Skipping transcription (diarization only).")
                with open(raw_transcript_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Reconstruct segments
                for s in data.get("segments", []):
                     words = _deserialize_words(s.get("words"), seg_start=s.get("start"), seg_end=s.get("end"))
                     seg_obj = self._build_whisper_style_segment(
                        seg_id=0,
                        start=s["start"],
                        end=s["end"],
                        text=s["text"],
                        words=words,
                    )
                     existing_segments.append(seg_obj)
                
                # Estimate duration from last segment
                if existing_segments:
                    total_duration = existing_segments[-1].end
                if video.duration:
                    total_duration = video.duration

                coverage = _word_coverage(existing_segments)
                if coverage < min_word_coverage:
                    log(
                        f"Raw transcript checkpoint has low word coverage ({coverage:.1%}, "
                        f"target >= {min_word_coverage:.0%}). Re-transcribing for timing consistency."
                    )
                    try:
                        raw_transcript_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        self._get_temp_transcript_path(video.id).unlink(missing_ok=True)
                        self._get_temp_transcript_jsonl_path(video.id).unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._reset_partial_checkpoint_state(video.id)
                    existing_segments = []
                else:
                    engine_from_raw = str(
                        data.get("transcription_engine_used")
                        or data.get("engine")
                        or ""
                    ).strip().lower()
                    if engine_from_raw not in {"parakeet", "whisper"}:
                        engine_from_raw = selected_engine if selected_engine in {"parakeet", "whisper"} else ""
                    payload_fields = {
                        "transcription_reused_existing": True,
                        "transcription_engine_routed": selected_engine,
                    }
                    if engine_from_raw in {"parakeet", "whisper"}:
                        payload_fields["transcription_engine_used"] = engine_from_raw
                    self._upsert_job_payload_fields(job_id, payload_fields)

                    raw_language = self._normalize_language_code(data.get("transcript_language"))
                    final_language = raw_language or route_language
                    if final_language:
                        with Session(runtime.engine) as session:
                            v = session.get(Video, video.id)
                            if v:
                                v.transcript_language = final_language
                                session.add(v)
                                session.commit()

                    # Ensure progress is 100
                    self._update_job_progress(job_id, 100)
                    self._reset_partial_checkpoint_state(video.id)
                    return existing_segments, total_duration
                
            except Exception as e:
                log(f"Failed to load existing raw transcript: {e}. Will re-transcribe.")
                existing_segments = []
        elif raw_transcript_path.exists() and force_retranscription:
            log(f"Ignoring existing raw transcript at {raw_transcript_path} because force_retranscription is enabled.")

        # 3. Check for PARTIAL resumption (temp files)
        partial_data = self._load_partial_transcript(video.id)
        start_time_offset = 0.0
        transcribe_path = audio_path
        temp_slice_path = None
        
        if partial_data:
            try:
                saved_segments = partial_data.get("segments", [])
                if saved_segments:
                    last_seg = saved_segments[-1]
                    start_time_offset = last_seg["end"]
                    log_verbose(f"Resuming transcription from {start_time_offset:.2f}s")
                    
                    for s in saved_segments:
                        words = _deserialize_words(s.get("words"), seg_start=s.get("start"), seg_end=s.get("end"))
                        seg_obj = self._build_whisper_style_segment(
                            seg_id=0,
                            start=s["start"],
                            end=s["end"],
                            text=s["text"],
                            words=words,
                        )
                        existing_segments.append(seg_obj)
                    
                    temp_slice_path = self._slice_audio(audio_path, start_time_offset)
                    transcribe_path = temp_slice_path
            except Exception as e:
                log_verbose(f"Failed to resume from partial: {e}. Starting fresh.")
                existing_segments = []
                start_time_offset = 0.0
                transcribe_path = audio_path
                try:
                    self._get_temp_transcript_path(video.id).unlink(missing_ok=True)
                    self._get_temp_transcript_jsonl_path(video.id).unlink(missing_ok=True)
                except Exception:
                    pass
                self._reset_partial_checkpoint_state(video.id)

        # 4. Choose transcription engine and run transcription.
        prefer_parakeet = selected_engine == "parakeet"
        transcribe_engine_used = "whisper"
        whisper_info = None

        # Keep Parakeet transcription from oversubscribing VRAM on long-running workers:
        # always release other GPU-heavy ASR/diarization models before Parakeet transcribe.
        # This is required now that queue stages can keep models warm independently.
        if prefer_parakeet and self.device == "cuda":
            if self.whisper_model is not None:
                self._release_whisper_model("before_parakeet_transcribe", job_id=job_id)
            if any([
                self.diarization_pipeline is not None,
                self.embedding_model is not None,
                self.embedding_inference is not None,
            ]):
                self._release_diarization_models("before_parakeet_transcribe", job_id=job_id)
            self._upsert_job_payload_fields(
                job_id,
                {
                    "parakeet_released_diarization_before_transcribe": True,
                },
            )
            # Sync barrier: catch latent CUDA corruption before loading Parakeet.
            if not self._safe_cuda_sync(timeout_s=10.0):
                self._mark_cuda_unhealthy("CUDA sync failed during model transition before Parakeet", job_id=job_id)

        # Read transcription settings
        beam_size = int(os.getenv("TRANSCRIPTION_BEAM_SIZE", "1"))
        vad_filter = os.getenv("TRANSCRIPTION_VAD_FILTER", "true").lower() == "true"
        # Non-batched is more reliable for dense word-level timestamps.
        use_batched = (not force_non_batched) and (os.getenv("TRANSCRIPTION_BATCHED", "false").lower() == "true")
        segments = existing_segments

        if prefer_parakeet:
            try:
                # Mark model-load stage boundary as soon as Parakeet path is chosen so
                # download timing doesn't continue while model initialization runs.
                if self.parakeet_model is None:
                    self._record_job_stage_start(job_id, "model_load")
                self._update_job_status_detail(job_id, "Transcribing with NVIDIA Parakeet...")
                remaining_duration = float(self._probe_audio_duration_seconds(transcribe_path) or 0.0)
                force_chunked_for_long = (
                    os.getenv("PARAKEET_LONG_AUDIO_FORCE_CHUNKED", "true").strip().lower() == "true"
                )
                long_chunked = False
                long_chunk_reason = "disabled"
                if (
                    force_chunked_for_long
                    and remaining_duration > 0
                ):
                    long_chunked, long_chunk_reason = self._should_force_parakeet_long_audio_chunked(
                        remaining_duration,
                        job_id=job_id,
                    )
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "parakeet_long_audio_force_chunked_enabled": bool(force_chunked_for_long),
                        "parakeet_long_audio_force_chunked_reason": long_chunk_reason,
                    },
                )
                if long_chunked:
                    self._upsert_job_payload_fields(
                        job_id,
                        {
                            "parakeet_long_audio_chunked": True,
                            "parakeet_long_audio_seconds": int(round(remaining_duration)),
                        },
                    )
                    self._update_job_status_detail(
                        job_id,
                        f"Long audio detected. Using stable Parakeet chunk mode ({long_chunk_reason})..."
                    )
                    parakeet_segments, parakeet_duration = self._transcribe_with_parakeet_in_chunks(
                        transcribe_path, start_time_offset=start_time_offset, job_id=job_id
                    )
                else:
                    parakeet_segments, parakeet_duration = self._transcribe_with_parakeet(
                        transcribe_path, start_time_offset=start_time_offset, job_id=job_id
                    )
                transcribe_engine_used = "parakeet"
                self._upsert_job_payload_fields(job_id, {"transcription_engine_used": "parakeet"})
                est_total_duration = (
                    float(video.duration or 0)
                    or float((partial_data or {}).get("total_duration") or 0)
                    or float(parakeet_duration or 0)
                )
                last_progress_update = 0
                last_detail_update = 0.0
                for idx, segment in enumerate(parakeet_segments, start=1):
                    segments.append(segment)
                    if job_id and est_total_duration > 0:
                        transcription_pct = min(float(getattr(segment, "end", 0.0)) / est_total_duration, 1.0)
                        job_pct = int(transcription_pct * 100)
                        now = time.time()
                        if (
                            job_pct > last_progress_update + 1
                            or idx <= 3
                            or idx % 10 == 0
                            or (now - last_detail_update) >= 2.5
                        ):
                            self._update_transcription_stage_progress(
                                job_id,
                                engine="parakeet",
                                completed_seconds=float(getattr(segment, "end", 0.0)),
                                total_seconds=est_total_duration,
                                segments_completed=len(segments),
                            )
                            last_progress_update = job_pct
                            last_detail_update = now
                    if job_id and idx % 10 == 0:
                        self._save_partial_transcript(video.id, segments, est_total_duration or 0.0)
                duration_info = parakeet_duration or video.duration or 0
            except Exception as e:
                allow_whisper_fallback = (
                    os.getenv("PARAKEET_ALLOW_WHISPER_FALLBACK", "true").strip().lower() == "true"
                )
                error_text = str(e).strip()
                low_timestamp_coverage = "Parakeet returned low word timestamp coverage" in error_text
                if low_timestamp_coverage and not allow_whisper_fallback:
                    log(
                        "Parakeet returned unusable timestamp output; overriding disabled Whisper fallback "
                        "to preserve transcription progress."
                    )
                    allow_whisper_fallback = True
                if allow_whisper_fallback:
                    log(f"Parakeet unavailable/failed: {e}. Falling back to Whisper.")
                else:
                    log(f"Parakeet unavailable/failed: {e}. Whisper fallback disabled; failing job with Parakeet error.")
                if self._is_cuda_illegal_access(e):
                    self._mark_cuda_unhealthy(str(e), job_id=job_id)
                    fallback_reason = (
                        "Parakeet hit a fatal CUDA illegal-memory-access fault; "
                        "this job is quarantined to Whisper/CPU."
                    )
                else:
                    fallback_reason = str(e).strip().splitlines()[0] if str(e).strip() else "unknown error"
                    fallback_reason = fallback_reason[:220]
                    fallback_reason = f"Parakeet failed ({fallback_reason}); falling back to Whisper..."
                self._update_job_status_detail(job_id, fallback_reason)
                transcribe_engine_used = "whisper"
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "transcription_engine_used": "whisper",
                        "transcription_engine_fallback_reason": str(e)[:500],
                        "transcription_engine_fallback_detail": fallback_reason,
                        "parakeet_allow_whisper_fallback": bool(allow_whisper_fallback),
                        "parakeet_soft_fallback_override": bool(low_timestamp_coverage),
                    },
                )
                if not allow_whisper_fallback:
                    failure_reason = (
                        f"Parakeet failed and Whisper fallback is disabled. Root cause: {str(e).strip()[:500]}"
                    )
                    self._update_job_status_detail(job_id, failure_reason[:240])
                    self._upsert_job_payload_fields(
                        job_id,
                        {
                            "transcription_engine_used": "parakeet",
                            "transcription_engine_fallback_reason": None,
                            "transcription_engine_fallback_detail": None,
                            "parakeet_no_fallback_failure": failure_reason[:800],
                        },
                    )
                    raise RuntimeError(failure_reason) from e
                prefer_parakeet = False
                # Ensure failed Parakeet state does not continue occupying GPU memory
                # when Whisper fallback begins.
                self._release_parakeet_model("fallback_to_whisper", job_id=job_id)
                # Force fresh run from current checkpoint state.
                segments = existing_segments
        
        if not prefer_parakeet:
            self._upsert_job_payload_fields(job_id, {"transcription_engine_used": "whisper"})
            operational_reason = str(route.get("operational_route_reason") or "").strip()
            if bool(route.get("operational_route_applied")) and operational_reason:
                self._update_job_status_detail(job_id, operational_reason[:240])
            whisper_model_override = str(route.get("whisper_model_override") or "").strip() or None
            self._load_whisper_model(
                job_id=job_id,
                force_float32=False,
                model_size_override=whisper_model_override,
            )
            whisper_backend = self._whisper_backend or "faster_whisper"
            self._upsert_job_payload_fields(
                job_id,
                {
                    "whisper_backend_used": whisper_backend,
                },
            )
            if whisper_backend != "faster_whisper":
                use_batched = False
            transcribe_params = {
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "word_timestamps": True,
            }
            if route_language:
                transcribe_params["language"] = route_language
            if vad_filter:
                transcribe_params["vad_parameters"] = {
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200
                }

            # On some GPU architectures (e.g. Blackwell), cuBLAS FP16 kernels may fail
            # at inference time even if the model loaded successfully.
            def _run_transcribe(whisper_model, transcribe_path_value, transcribe_params_value, use_batched_value, device):
                if use_batched_value and device == "cuda" and (self._whisper_backend or "faster_whisper") == "faster_whisper":
                    try:
                        from faster_whisper import BatchedInferencePipeline
                        batched_model = BatchedInferencePipeline(model=whisper_model)
                        segments_gen, info_obj = batched_model.transcribe(
                            str(transcribe_path_value),
                            batch_size=16,
                            **transcribe_params_value
                        )
                        log_verbose("Using batched transcription pipeline")
                        return segments_gen, info_obj
                    except Exception as e:
                        if "CUBLAS" in str(e).upper():
                            raise
                        log_verbose(f"Batched transcription failed: {e}")
                return whisper_model.transcribe(str(transcribe_path_value), **transcribe_params_value)

            def _run_transcribe_with_stage_start(whisper_model, transcribe_path_value, transcribe_params_value, use_batched_value, device):
                # Start transcribe timer only when decoder inference actually starts.
                self._record_job_stage_start(job_id, "transcribe")
                return _run_transcribe(whisper_model, transcribe_path_value, transcribe_params_value, use_batched_value, device)

            whisper_runtime_device = self._whisper_device or self.device
            try:
                segments_generator, info = _run_transcribe_with_stage_start(
                    self.whisper_model,
                    transcribe_path,
                    transcribe_params,
                    use_batched,
                    whisper_runtime_device,
                )
                whisper_info = info
            except RuntimeError as e:
                if self._is_cuda_oom(e) and whisper_backend != "faster_whisper":
                    log("Transformers Whisper CUDA OOM during transcription - falling back to faster-whisper.")
                    self._update_job_status_detail(job_id, "Whisper VRAM pressure detected. Retrying with safer backend...")
                    self._upsert_job_payload_fields(
                        job_id,
                        {
                            "whisper_backend_oom_fallback_from": whisper_backend,
                            "whisper_backend_oom_fallback_to": "faster_whisper",
                            "whisper_backend_oom_fallback_reason": str(e)[:800],
                        },
                    )
                    self._release_whisper_model("transformers_whisper_oom_fallback", job_id=job_id)
                    self._clear_cuda_cache()
                    self._load_whisper_model(
                        job_id=job_id,
                        force_float32=False,
                        model_size_override=whisper_model_override,
                        backend_override="faster_whisper",
                    )
                    whisper_backend = self._whisper_backend or "faster_whisper"
                    use_batched = False
                    whisper_runtime_device = self._whisper_device or self.device
                    segments_generator, info = _run_transcribe_with_stage_start(
                        self.whisper_model,
                        transcribe_path,
                        transcribe_params,
                        use_batched,
                        whisper_runtime_device,
                    )
                    whisper_info = info
                elif "CUBLAS" in str(e).upper():
                    log("cuBLAS error during transcription - reloading model with float32...")
                    self._update_job_status_detail(job_id, "Reloading model (GPU compatibility fallback)...")
                    self._load_whisper_model(
                        job_id=job_id,
                        force_float32=True,
                        model_size_override=whisper_model_override,
                    )
                    whisper_runtime_device = self._whisper_device or self.device
                    segments_generator, info = _run_transcribe_with_stage_start(
                        self.whisper_model,
                        transcribe_path,
                        transcribe_params,
                        use_batched,
                        whisper_runtime_device,
                    )
                    whisper_info = info
                else:
                    raise

            duration_info = info.duration or video.duration or 0
            if video.duration and video.duration > 0:
                total_duration = video.duration
            elif partial_data:
                total_duration = partial_data.get("total_duration", 0)
            else:
                total_duration = duration_info
            last_progress_update = 0
            last_detail_update = 0.0
            self._update_transcription_stage_progress(
                job_id,
                engine="whisper",
                completed_seconds=start_time_offset,
                total_seconds=total_duration if total_duration and total_duration > 0 else None,
                segments_completed=len(segments),
            )
            try:
                seg_iter = iter(segments_generator)
            except RuntimeError as e:
                if "CUBLAS" in str(e).upper():
                    log("cuBLAS error starting transcription generator - reloading with float32...")
                    self._update_job_status_detail(job_id, "Reloading model (GPU compatibility fallback)...")
                    self._load_whisper_model(job_id=job_id, force_float32=True)
                    whisper_runtime_device = self._whisper_device or self.device
                    segments_generator, info = _run_transcribe_with_stage_start(
                        self.whisper_model,
                        transcribe_path,
                        transcribe_params,
                        use_batched,
                        whisper_runtime_device,
                    )
                    seg_iter = iter(segments_generator)
                else:
                    raise

            try:
                for segment in seg_iter:
                    if start_time_offset > 0:
                        shifted_words = None
                        if segment.words:
                            shifted_words = []
                            for w in segment.words:
                                try:
                                    ws = float(w.start) + start_time_offset
                                    we_raw = w.end if w.end is not None else w.start
                                    we = float(we_raw) + start_time_offset
                                except Exception:
                                    continue
                                shifted_words.append(
                                    self._build_whisper_style_word(
                                        start=ws,
                                        end=we,
                                        word=str(getattr(w, "word", "") or ""),
                                    )
                                )
                        segment = self._build_whisper_style_segment(
                            seg_id=getattr(segment, "id", 0),
                            start=segment.start + start_time_offset,
                            end=segment.end + start_time_offset,
                            text=segment.text,
                            words=shifted_words,
                        )

                    segments.append(segment)

                    if job_id and total_duration > 0:
                        transcription_pct = min(segment.end / total_duration, 1.0)
                        job_pct = int(transcription_pct * 100)
                        now = time.time()
                        if (
                            job_pct > last_progress_update + 1
                            or len(segments) <= 3
                            or len(segments) % 10 == 0
                            or (now - last_detail_update) >= 2.5
                        ):
                            self._update_transcription_stage_progress(
                                job_id,
                                engine="whisper",
                                completed_seconds=segment.end,
                                total_seconds=total_duration,
                                segments_completed=len(segments),
                            )
                            last_progress_update = job_pct
                            last_detail_update = now

                    timestamp = self._format_timestamp(segment.start)
                    log_verbose(f"[{timestamp}] {segment.text.strip()[:50]}...")

                    if len(segments) % 10 == 0:
                        self._save_partial_transcript(video.id, segments, total_duration)
            except RuntimeError as e:
                if "CUBLAS" in str(e).upper():
                    log("cuBLAS error during transcription iteration - reloading model with float32...")
                    self._update_job_status_detail(job_id, "Reloading model (GPU compatibility fallback)...")
                    if segments:
                        self._save_partial_transcript(video.id, segments, total_duration)
                    self._load_whisper_model(
                        job_id=job_id,
                        force_float32=True,
                        model_size_override=whisper_model_override,
                    )
                    log("Model reloaded with compute_type=float32. Resuming from checkpoint...")
                    return self._process_transcribe_phase(
                        video,
                        audio_path,
                        job_id,
                        force_non_batched=force_non_batched,
                    )
                raise
        else:
            duration_info = duration_info if "duration_info" in locals() else (video.duration or 0)

        if video.duration and video.duration > 0:
            total_duration = video.duration
        elif partial_data: # If resumed from partial, info.duration is checked against partial total
             total_duration = partial_data.get("total_duration", 0)
        else:
             total_duration = duration_info

        coverage = _word_coverage(segments)
        if (
            transcribe_engine_used == "whisper"
            and use_batched
            and coverage < min_word_coverage
        ):
            log(
                f"Low word timestamp coverage after batched transcription ({coverage:.1%}, "
                f"target >= {min_word_coverage:.0%}). Re-running non-batched for accuracy."
            )
            self._update_job_status_detail(
                job_id,
                f"Word timing coverage {int(round(coverage * 100))}% too low; re-running accurate pass..."
            )
            try:
                self._get_temp_transcript_path(video.id).unlink(missing_ok=True)
                self._get_temp_transcript_jsonl_path(video.id).unlink(missing_ok=True)
            except Exception:
                pass
            self._reset_partial_checkpoint_state(video.id)
            return self._process_transcribe_phase(video, audio_path, job_id, force_non_batched=True)

        # Final Cleanup & Save
        if temp_slice_path and temp_slice_path.exists():
            try:
                os.unlink(temp_slice_path)
            except: pass
            
        self._update_job_progress(job_id, 100)
        
        # Save "raw" checkpoint
        try:
            out_dir = audio_path.parent
            if out_dir.exists():
                safe_title = self.sanitize_filename(video.title)
                raw_path = out_dir / f"{safe_title}_transcript_raw.json"
                final_language = route_language or self._normalize_language_code(
                    getattr(whisper_info, "language", None) if whisper_info is not None else None
                )
                raw_data = {
                    "video_id": video.id,
                    "transcription_engine_used": transcribe_engine_used,
                    "transcription_engine_requested": route.get("requested_engine"),
                    "transcription_engine_routed": selected_engine,
                    "transcript_language": final_language,
                    "transcription_route_language_confidence": route.get("language_confidence"),
                    "transcription_route_language_source": route.get("language_source"),
                    "transcription_route_language_reason": route.get("language_reason"),
                    "transcription_route_multilingual": bool(route.get("multilingual_route_applied")),
                    "transcription_route_whisper_model": route.get("whisper_model_override"),
                    "segments": [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None
                        }
                        for s in segments
                    ]
                }
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)
                log_verbose(f"Saved raw transcript checkpoint: {raw_path}")
        except Exception as e:
            log(f"Failed to save raw transcript checkpoint: {e}")

        final_language = route_language or self._normalize_language_code(
            getattr(whisper_info, "language", None) if whisper_info is not None else None
        )
        with Session(runtime.engine) as session:
            v = session.get(Video, video.id)
            if v:
                if final_language:
                    v.transcript_language = final_language
                    session.add(v)
                    session.commit()

        if transcribe_engine_used == "parakeet" and self._should_unload_parakeet_after_transcribe(job_id=job_id):
            self._release_parakeet_model("post_transcribe_low_vram", job_id=job_id)
            
        return segments, total_duration

    def _process_diarize_phase(self, video: Video, audio_path: Path, segments: list, job_id: int = None):
        """Phase 3: Diarization and Speaker Identification"""
        from pyannote.core import Annotation, Segment
        from bisect import bisect_right
        job_payload = {}
        
        # 1. Update Status
        with Session(runtime.engine) as session:
            v = session.get(Video, video.id)
            if v: 
                v.status = "diarizing"
                session.add(v)
            if job_id:
               j = session.get(Job, job_id)
               if j:
                   j.status = "diarizing"
                   j.progress = 0
                   j.status_detail = "Diarizing speakers..."
                   session.add(j)
            session.commit()

        self._record_job_stage_start(job_id, "diarize")
        if job_id:
            try:
                with Session(runtime.engine) as session:
                    job = session.get(Job, job_id)
                    if job:
                        job_payload = self._load_job_payload(getattr(job, "payload_json", None))
            except Exception as e:
                log_verbose(f"Could not load diarization job payload for overrides: {e}")
        optimization_target = str(job_payload.get("optimization_target") or "").strip().lower()
        should_apply_text_cleanup = optimization_target not in {"diarization_rebuild", "diarization_benchmark"}
            
        log("Diarizing...")
        if self.device == "cuda" and self.whisper_model is not None:
            self._release_whisper_model("before_diarize", job_id=job_id)
        if self._should_release_parakeet_before_diarize(job_id=job_id):
            self._release_parakeet_model("before_diarize", job_id=job_id)
        # Sync barrier: catch latent CUDA corruption before loading diarization models.
        if self.device == "cuda" and not self._safe_cuda_sync(timeout_s=10.0):
            self._mark_cuda_unhealthy("CUDA sync failed before diarization", job_id=job_id)
        # Ensure the speaker-match cache starts fresh for this channel; new embeddings
        # created during this run are appended incrementally.
        self._invalidate_speaker_match_cache(video.channel_id)
        
        # 2. Run Pipeline (No session)
        diarization = None
        audio_input = None
        diarization_path = self._get_temp_diarization_path(video.id)
        
        start_diar = time.time()
        
        if diarization_path.exists():
            try:
                log_verbose(f"Loading existing diarization from {diarization_path}")
                diarization = Annotation()
                with open(diarization_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == "SPEAKER":
                            start = float(parts[3])
                            dur = float(parts[4])
                            spk = parts[7]
                            diarization[Segment(start, start + dur)] = spk
                log_verbose("Loaded diarization successfully.")
            except Exception as e:
                log_verbose(f"Failed to load RTTM: {e}. Re-running pipeline.")
                diarization = None

        if not diarization:
            # Ensure models are loaded (in case transcription was skipped)
            if self.diarization_pipeline is None:
                self._load_models(job_id)

            if self.diarization_pipeline is None:
                if self._diarization_load_error:
                    raise RuntimeError(f"Diarization pipeline failed to load: {self._diarization_load_error}")
                else:
                    raise RuntimeError("Diarization pipeline not loaded.")
            
            # Use ffmpeg+soundfile helper
            audio_input = self._load_audio_for_pyannote(str(audio_path))

            # Configure diarization sensitivity from settings
            sensitivity = str(job_payload.get("diarization_sensitivity_override") or os.getenv("DIARIZATION_SENSITIVITY", "balanced") or "balanced").strip().lower()
            if sensitivity not in {"aggressive", "balanced", "conservative"}:
                sensitivity = "balanced"
            if sensitivity == "aggressive":
                # Keep more boundaries by only filling extremely short pauses.
                self.diarization_pipeline.segmentation.min_duration_off = 0.0
            elif sensitivity == "conservative":
                # Merge across longer pauses to reduce speaker fragmentation.
                self.diarization_pipeline.segmentation.min_duration_off = 1.0
            else:
                # Balanced keeps some short-pause smoothing without over-merging.
                self.diarization_pipeline.segmentation.min_duration_off = 0.35

            try:
                diarization_output = self._run_diarization_with_adaptive_batch(audio_input, job_id=job_id)
            except RuntimeError as e:
                if "CUBLAS" in str(e).upper() and not self._force_float32:
                    import torch
                    log("cuBLAS error during diarization â€” reloading pipeline in float32...")
                    self._force_float32 = True
                    torch.set_float32_matmul_precision('high')
                    self._cast_pipeline_to_float32(self.diarization_pipeline)
                    if self.embedding_model:
                        self.embedding_model = self.embedding_model.float()
                        self.embedding_inference = None  # will be recreated
                        import warnings

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
                            )
                            from pyannote.audio import Inference as _Inf
                        self.embedding_inference = _Inf(self.embedding_model, window="whole")
                        self.embedding_inference.to(torch.device(self.device))
                    diarization_output = self._run_diarization_with_adaptive_batch(audio_input, job_id=job_id)
                else:
                    raise
            
            # Convert DiarizeOutput
            if hasattr(diarization_output, 'speaker_diarization'):
                diarization = diarization_output.speaker_diarization
            elif hasattr(diarization_output, 'to_annotation'):
                    diarization = diarization_output.to_annotation()
            else:
                diarization = diarization_output

            log_verbose(f"Diarization complete in {time.time() - start_diar:.1f}s")
            
            # Save RTTM
            try:
                with open(diarization_path, 'w') as f:
                    diarization.write_rttm(f)
                log_verbose(f"Saved diarization checkpoint to {diarization_path}")
            except Exception as e:
                log_verbose(f"Failed to save RTTM checkpoint: {e}")

        # If we resumed from a saved RTTM diarization checkpoint, we still need decoded
        # audio for speaker embedding extraction during the identification step below.
        if audio_input is None:
            audio_input = self._load_audio_for_pyannote(str(audio_path))

        # 3. Speaker ID & Final Save (Open Session)
        with Session(runtime.engine) as session:
            # Re-fetch video attached to this session
            video_attached = session.get(Video, video.id)
            if not video_attached:
                raise ValueError(f"Video {video.id} not found during diarization phase")

            # Re-processing can happen multiple times for the same video; purge previous
            # transcript rows (and edit revisions tied to them) before writing the new set.
            try:
                from sqlalchemy import delete as sa_delete
                session.exec(
                    sa_delete(TranscriptSegmentRevision).where(
                        TranscriptSegmentRevision.video_id == video_attached.id
                    )
                )
                session.exec(
                    sa_delete(TranscriptSegment).where(
                        TranscriptSegment.video_id == video_attached.id
                    )
                )
                session.exec(
                    sa_delete(FunnyMoment).where(
                        FunnyMoment.video_id == video_attached.id
                    )
                )
                session.commit()
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to clear existing transcript rows for video {video_attached.id}: {e}")

            local_speaker_map = {}
            log_verbose("Identifying speakers from diarization segments...")
            try:
                match_threshold = float(
                    job_payload.get("speaker_match_threshold_override")
                    if job_payload.get("speaker_match_threshold_override") is not None
                    else os.getenv("SPEAKER_MATCH_THRESHOLD", "0.35")
                )
            except Exception:
                match_threshold = 0.35

            def _embedding_sample_metadata(py_seg):
                """Best-effort provenance for a speaker embedding source clip."""
                try:
                    start = float(py_seg.start)
                    end = float(py_seg.end)
                except Exception:
                    return {"sample_start_time": None, "sample_end_time": None, "sample_text": None}

                overlap_texts = []
                for ws in segments or []:
                    try:
                        ws_start = float(ws.start)
                        ws_end = float(ws.end)
                    except Exception:
                        continue
                    if ws_end <= start or ws_start >= end:
                        continue
                    text = (getattr(ws, "text", "") or "").strip()
                    if text:
                        overlap_texts.append(text)

                sample_text = " ".join(overlap_texts).strip() if overlap_texts else None
                if sample_text and len(sample_text) > 320:
                    sample_text = sample_text[:317].rstrip() + "..."

                return {
                    "sample_start_time": start,
                    "sample_end_time": end,
                    "sample_text": sample_text,
                }

            def _persist_additional_speaker_profiles(
                speaker_id: int,
                sample_embeddings,
                primary_segment,
            ):
                extra_limit = max(0, int(os.getenv("SPEAKER_PROFILE_PERSIST_RAW_SAMPLES", "2")))
                if extra_limit <= 0:
                    return

                persisted = 0
                for seg, raw_embedding in sample_embeddings:
                    if persisted >= extra_limit:
                        break
                    if seg == primary_segment:
                        continue
                    extra_meta = _embedding_sample_metadata(seg)
                    extra_row = SpeakerEmbedding(
                        speaker_id=speaker_id,
                        embedding_blob=pickle.dumps(raw_embedding),
                        source_video_id=video_attached.id,
                        sample_start_time=extra_meta["sample_start_time"],
                        sample_end_time=extra_meta["sample_end_time"],
                        sample_text=extra_meta["sample_text"],
                        created_at=datetime.now()
                    )
                    session.add(extra_row)
                    session.commit()
                    session.refresh(extra_row)
                    self._append_speaker_match_cache(
                        video_attached.channel_id,
                        extra_row.id,
                        speaker_id,
                        raw_embedding,
                    )
                    persisted += 1
            
            # Pre-process speakers
            diarization_labels = list(diarization.labels())
            total_labels = len(diarization_labels) or 1
            for idx, label in enumerate(diarization_labels, start=1):
                if job_id and (idx == 1 or idx % 3 == 0 or idx == total_labels):
                    self._update_job_status_detail(job_id, f"Analyzing speakers ({idx}/{total_labels})...")
                    # Keep some visible progress movement during long diarization post-processing
                    self._update_job_progress(job_id, min(70, 45 + int((idx / total_labels) * 25)))
                timeline = diarization.label_timeline(label)
                if not timeline:
                    continue

                timeline_segments = list(timeline)
                profile_embedding, primary_segment, sample_embeddings = self._build_speaker_embedding_profile(
                    audio_input,
                    timeline_segments,
                )
                if primary_segment is None or profile_embedding is None:
                    log_verbose(f"  Warning: Speaker {label} did not produce a usable embedding profile")
                    continue

                if primary_segment.duration < 0.5:
                    log_verbose(f"  Warning: Speaker {label} primary segment is only {primary_segment.duration:.2f}s")

                log_verbose(
                    f"  Analyzing speaker {label} "
                    f"(samples: {len(sample_embeddings)}, primary segment: {primary_segment.duration:.2f}s)"
                )

                sample_meta = _embedding_sample_metadata(primary_segment)

                if profile_embedding is not None:
                    try:
                        found_speaker, matched_profile, matched_score = self.identify_speaker(
                            session, video_attached.channel_id, profile_embedding, threshold=match_threshold
                        )
                    except Exception as e:
                        log(f"Identify error: {e}")
                        found_speaker = None
                        matched_profile = None
                        matched_score = None
                    
                    if found_speaker:
                        local_speaker_map[label] = {
                            "speaker": found_speaker,
                            "matched_profile_id": matched_profile.id if matched_profile else None,
                        }
                        if matched_profile and matched_score is not None:
                            log_verbose(
                                f"    -> Matched with {found_speaker.name} via profile #{matched_profile.id} (cos={matched_score:.4f})"
                            )
                        else:
                            log_verbose(f"    -> Matched with {found_speaker.name}")
                        # Enrich: add this new embedding to improve future matching
                        new_emb_row = SpeakerEmbedding(
                            speaker_id=found_speaker.id,
                            embedding_blob=pickle.dumps(profile_embedding),
                            source_video_id=video_attached.id,
                            sample_start_time=sample_meta["sample_start_time"],
                            sample_end_time=sample_meta["sample_end_time"],
                            sample_text=sample_meta["sample_text"],
                            created_at=datetime.now()
                        )
                        session.add(new_emb_row)
                        session.commit()
                        session.refresh(new_emb_row)
                        self._append_speaker_match_cache(
                            video_attached.channel_id,
                            new_emb_row.id,
                            found_speaker.id,
                            profile_embedding,
                        )
                        _persist_additional_speaker_profiles(
                            found_speaker.id,
                            sample_embeddings,
                            primary_segment,
                        )
                    else:
                        existing_count = session.exec(
                            select(func.count(Speaker.id)).where(Speaker.channel_id == video_attached.channel_id)
                        ).one()
                        new_name = f"Speaker {existing_count + 1}"
                        
                        new_spk = Speaker(
                            channel_id=video_attached.channel_id,
                            name=new_name,
                            embedding_blob=pickle.dumps(profile_embedding),
                            created_at=datetime.now()
                        )
                        session.add(new_spk)
                        session.commit()
                        session.refresh(new_spk)
                        # Also store in the multi-embedding table
                        seed_emb = SpeakerEmbedding(
                            speaker_id=new_spk.id,
                            embedding_blob=pickle.dumps(profile_embedding),
                            source_video_id=video_attached.id,
                            sample_start_time=sample_meta["sample_start_time"],
                            sample_end_time=sample_meta["sample_end_time"],
                            sample_text=sample_meta["sample_text"],
                            created_at=datetime.now()
                        )
                        session.add(seed_emb)
                        session.commit()
                        session.refresh(seed_emb)
                        self._append_speaker_match_cache(
                            video_attached.channel_id,
                            seed_emb.id,
                            new_spk.id,
                            profile_embedding,
                        )
                        _persist_additional_speaker_profiles(
                            new_spk.id,
                            sample_embeddings,
                            primary_segment,
                        )
                        local_speaker_map[label] = {
                            "speaker": new_spk,
                            "matched_profile_id": seed_emb.id,
                        }
                        log_verbose(f"    -> Created new {new_name}")

            # Map Transcript â€” split Whisper segments at speaker boundaries
            # using word-level timestamps so each DB segment has a single speaker.
            final_segments = []
            processed_segments = 0

            # Build a flat sorted list of (start, end, label) for fast lookup
            _speaker_turns = sorted(
                (turn.start, turn.end, label)
                for label in diarization.labels()
                for turn in diarization.label_timeline(label)
            )
            _speaker_turn_starts = [t[0] for t in _speaker_turns]
            _speaker_turn_idx = 0

            def _speaker_at(t: float) -> str | None:
                """Return active speaker at t with monotonic pointer + bisect fallback."""
                nonlocal _speaker_turn_idx
                if not _speaker_turns:
                    return None

                while _speaker_turn_idx + 1 < len(_speaker_turns) and _speaker_turns[_speaker_turn_idx][1] <= t:
                    _speaker_turn_idx += 1

                start, end, label = _speaker_turns[_speaker_turn_idx]
                if start <= t < end:
                    return label

                # Rare non-monotonic/seek fallback.
                pos = bisect_right(_speaker_turn_starts, t) - 1
                if pos >= 0:
                    s2, e2, l2 = _speaker_turns[pos]
                    if s2 <= t < e2:
                        _speaker_turn_idx = pos
                        return l2
                return None

            # Conservative cleanup for tiny orphan "Unknown" word runs that appear
            # between the same speaker label due to diarization boundary jitter.
            orphan_max_words = max(0, int(os.getenv("DIARIZATION_ORPHAN_MAX_WORDS", "2")))
            orphan_max_seconds = max(0.0, float(os.getenv("DIARIZATION_ORPHAN_MAX_SECONDS", "0.65")))

            def _run_duration(run_words: list) -> float:
                if not run_words:
                    return 0.0
                try:
                    run_start = float(run_words[0].start)
                    run_end_raw = run_words[-1].end if run_words[-1].end else run_words[-1].start
                    run_end = float(run_end_raw)
                    return max(0.0, run_end - run_start)
                except Exception:
                    return 0.0

            def _smooth_word_runs(runs: list[tuple[str | None, list]]) -> list[tuple[str | None, list]]:
                if orphan_max_words <= 0 or len(runs) <= 1:
                    return runs

                editable = [[label, list(run_words)] for label, run_words in runs]

                # Bridge: A - Unknown(short) - A  =>  A - A - A
                for idx in range(1, len(editable) - 1):
                    cur_label = editable[idx][0]
                    if cur_label is not None:
                        continue
                    prev_label = editable[idx - 1][0]
                    next_label = editable[idx + 1][0]
                    run_words = editable[idx][1]
                    if (
                        prev_label
                        and next_label
                        and prev_label == next_label
                        and len(run_words) <= orphan_max_words
                        and _run_duration(run_words) <= orphan_max_seconds
                    ):
                        editable[idx][0] = prev_label

                # Optional tiny edge cleanup within the same Whisper segment.
                if len(editable) >= 2 and editable[0][0] is None and editable[1][0]:
                    run_words = editable[0][1]
                    if len(run_words) <= 1 and _run_duration(run_words) <= min(orphan_max_seconds, 0.35):
                        editable[0][0] = editable[1][0]
                if len(editable) >= 2 and editable[-1][0] is None and editable[-2][0]:
                    run_words = editable[-1][1]
                    if len(run_words) <= 1 and _run_duration(run_words) <= min(orphan_max_seconds, 0.35):
                        editable[-1][0] = editable[-2][0]

                # Re-collapse adjacent runs with the same label after smoothing.
                collapsed: list[list] = []
                for label, run_words in editable:
                    if collapsed and collapsed[-1][0] == label:
                        collapsed[-1][1].extend(run_words)
                    else:
                        collapsed.append([label, list(run_words)])

                return [(label, run_words) for label, run_words in collapsed]

            total_input_segments = len(segments) or 1
            for seg in segments:
                if job_id and (processed_segments == 0 or processed_segments % 50 == 0):
                    self._update_job_status_detail(
                        job_id,
                        f"Writing transcript segments ({processed_segments}/{total_input_segments})..."
                    )
                    self._update_job_progress(job_id, min(95, 70 + int((processed_segments / total_input_segments) * 25)))
                words = list(seg.words) if seg.words else []

                if not words:
                    # No word-level timestamps â€” fall back to whole-segment overlap
                    best_speaker_label = None
                    max_overlap = 0
                    seg_pyannote = Segment(seg.start, seg.end)
                    for label in diarization.labels():
                        overlap = diarization.label_timeline(label).crop(seg_pyannote).duration()
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker_label = label

                    db_seg = TranscriptSegment(
                        video_id=video_attached.id,
                        start_time=seg.start,
                        end_time=seg.end,
                        text=seg.text.strip(),
                        words=None
                    )
                    if best_speaker_label and best_speaker_label in local_speaker_map:
                        mapping = local_speaker_map[best_speaker_label]
                        db_seg.speaker_id = mapping["speaker"].id
                        db_seg.matched_profile_id = mapping.get("matched_profile_id")
                    final_segments.append(db_seg)
                    processed_segments += 1
                    continue

                # Split words into runs of the same speaker
                runs: list[tuple[str | None, list]] = []  # (speaker_label, [words])
                for w in words:
                    mid = (w.start + w.end) / 2 if w.end else w.start
                    label = _speaker_at(mid)
                    if runs and runs[-1][0] == label:
                        runs[-1][1].append(w)
                    else:
                        runs.append((label, [w]))
                runs = _smooth_word_runs(runs)

                for label, run_words in runs:
                    run_text = " ".join(w.word.strip() for w in run_words).strip()
                    if not run_text:
                        continue
                    run_start = run_words[0].start
                    run_end = run_words[-1].end if run_words[-1].end else run_words[-1].start
                    words_json = json.dumps([{"start": w.start, "end": w.end, "word": w.word} for w in run_words])

                    db_seg = TranscriptSegment(
                        video_id=video_attached.id,
                        start_time=run_start,
                        end_time=run_end,
                        text=run_text,
                        words=words_json
                    )
                    if label and label in local_speaker_map:
                        mapping = local_speaker_map[label]
                        db_seg.speaker_id = mapping["speaker"].id
                        db_seg.matched_profile_id = mapping.get("matched_profile_id")
                    final_segments.append(db_seg)

                processed_segments += 1

            if job_id:
                self._update_job_status_detail(
                    job_id,
                    f"Writing transcript segments ({total_input_segments}/{total_input_segments})..."
                )
                self._update_job_progress(job_id, 95)

            if job_id:
                self._update_job_status_detail(job_id, "Consolidating transcript segments...")
                self._update_job_progress(job_id, 96)
            consolidation = self._consolidate_transcript_segments(final_segments)
            final_segments = consolidation["segments"]
            if consolidation["merged_count"] > 0 or consolidation["reassigned_islands"] > 0:
                log(
                    f"Transcript consolidation for {video_attached.title}: "
                    f"{consolidation['merged_count']} merges, "
                    f"{consolidation['reassigned_islands']} reassigned short islands."
                )
            if should_apply_text_cleanup:
                if job_id:
                    self._update_job_status_detail(job_id, "Applying entity repair...")
                    self._update_job_progress(job_id, 97)
                entity_repair = self._apply_entity_repair_to_segments(session, video_attached, final_segments, persist_revisions=False)
                if entity_repair["changed"]:
                    log(
                        f"Transcript entity repair for {video_attached.title}: "
                        f"{entity_repair['replacement_count']} replacements across "
                        f"{entity_repair['segments_changed']} segments."
                    )
                if job_id:
                    self._update_job_status_detail(job_id, "Applying formatting cleanup...")
                    self._update_job_progress(job_id, 97)
                formatting_cleanup = self._apply_formatting_cleanup_to_segments(
                    session,
                    video_attached,
                    final_segments,
                    persist_revisions=False,
                )
                if formatting_cleanup["changed"]:
                    log(
                        f"Transcript formatting cleanup for {video_attached.title}: "
                        f"{formatting_cleanup['segments_changed']} segments updated."
                    )
            elif job_id:
                self._update_job_status_detail(job_id, "Skipping text cleanup for diarization rebuild...")
                self._update_job_progress(job_id, 97)
            if job_id:
                self._update_job_status_detail(job_id, "Committing transcript rows...")
                self._update_job_progress(job_id, 98)
            for db_seg in final_segments:
                session.add(db_seg)

            session.commit()
            if job_id:
                self._update_job_status_detail(job_id, "Saving transcript files...")
                self._update_job_progress(job_id, 99)
            
            # Save final files
            self._save_transcripts(session, video_attached, final_segments, audio_path)
            
            # Cleanup temp
            try:
                if diarization_path.exists(): os.unlink(diarization_path)
                transcript_temp = self._get_temp_transcript_path(video.id)
                if transcript_temp.exists(): os.unlink(transcript_temp)
                transcript_jsonl = self._get_temp_transcript_jsonl_path(video.id)
                if transcript_jsonl.exists(): os.unlink(transcript_jsonl)
            except: pass
            self._reset_partial_checkpoint_state(video.id)
            
            log(f"Processing complete for {video_attached.title}")
            video_attached.status = "completed"
            video_attached.processed = True
            video_attached.transcript_source = "local_transcription"
            video_attached.transcript_is_placeholder = False
            session.add(video_attached)
            session.commit()
        if self.device == "cuda" and self._should_unload_diarization_after_job(job_id=job_id):
            self._release_diarization_models("post_diarize", job_id=job_id)

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
