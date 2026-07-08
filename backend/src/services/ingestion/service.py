import yt_dlp
import os
import time
import json
import subprocess
import threading
import pickle
import gc
import re
import tempfile
from pathlib import Path
from typing import Literal
from sqlmodel import Session, select, func
from datetime import datetime

# NOTE: Heavy ML libraries (torch, faster_whisper, pyannote, numpy, scipy)
# are imported lazily inside _load_models() and related methods to avoid
# blocking the process at startup. Only download/queue operations run
# without them.

from ...db.database import Video, Channel, Speaker, SpeakerEmbedding, TranscriptSegment, TranscriptSegmentRevision, Clip, ClipExportArtifact, Job, FunnyMoment
from ..logger import log, log_verbose, is_verbose
from .. import episode_clone as clone_svc
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
    EXPORT_DIR,
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

class IngestionService(
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





















































































































































    def _transcript_laughter_candidates(self, segments: list[TranscriptSegment]) -> list[dict]:
        """Detect explicit laughter cues in transcript text."""

        laughter_re = re.compile(
            r"\b(?:laugh(?:ter|ing|s|ed)?|giggl(?:e|es|ing|ed)?|chuckl(?:e|es|ing|ed)?|snicker(?:s|ing|ed)?|"
            r"haha+|ha\s+ha(?:\s+ha)*|hehe+|lol)\b",
            re.IGNORECASE,
        )

        candidates: list[dict] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            matches = laughter_re.findall(text)
            if not matches:
                continue

            duration = max(0.2, float(seg.end_time - seg.start_time))
            cue_score = min(0.4, 0.12 * len(matches))
            dur_score = min(0.2, duration / 12.0)
            score = 0.65 + cue_score + dur_score

            candidates.append({
                "start_time": max(0.0, float(seg.start_time) - 0.35),
                "end_time": float(seg.end_time) + 1.1,
                "score": round(score, 4),
                "source": "transcript",
                "snippet": text[:280],
            })

        return candidates

    def _acoustic_laughter_candidates(self, audio_path: Path) -> list[dict]:
        """Lightweight acoustic laughter heuristic using bursty energy + ZCR features.

        This is intentionally CPU-only and dependency-light (ffmpeg + soundfile + numpy).
        It is not a classifier, but works well enough as a laughter candidate generator.
        """
        import os
        import numpy as np
        import soundfile as sf

        ffmpeg_cmd = self._get_ffmpeg_cmd()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # Downsample to reduce CPU/memory while keeping enough temporal structure.
            subprocess.run(
                [ffmpeg_cmd, "-y", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "8000", tmp_path],
                check=True,
                capture_output=True,
                timeout=1800,
            )

            windows: list[dict] = []
            with sf.SoundFile(tmp_path) as f:
                sr = int(f.samplerate)
                if sr <= 0:
                    return []

                frame_len = max(1, int(sr * 0.02))  # 20ms
                frames_per_window = max(1, int(round(0.5 / 0.02)))  # 0.5s windows
                carry = np.array([], dtype=np.float32)
                frame_buf_rms: list[float] = []
                frame_buf_zcr: list[float] = []
                window_index = 0

                for block in f.blocks(blocksize=sr * 30, dtype="float32", always_2d=False):
                    if block is None:
                        continue
                    data = np.asarray(block, dtype=np.float32).flatten()
                    if carry.size:
                        data = np.concatenate([carry, data])

                    usable = (data.size // frame_len) * frame_len
                    if usable <= 0:
                        carry = data
                        continue

                    chunk = data[:usable].reshape(-1, frame_len)
                    carry = data[usable:]

                    rms = np.sqrt(np.mean(chunk * chunk, axis=1) + 1e-12)
                    signs = chunk >= 0
                    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

                    frame_buf_rms.extend(rms.tolist())
                    frame_buf_zcr.extend(zcr.tolist())

                    while len(frame_buf_rms) >= frames_per_window:
                        win_rms = np.asarray(frame_buf_rms[:frames_per_window], dtype=np.float32)
                        win_zcr = np.asarray(frame_buf_zcr[:frames_per_window], dtype=np.float32)
                        del frame_buf_rms[:frames_per_window]
                        del frame_buf_zcr[:frames_per_window]

                        start_t = window_index * 0.5
                        end_t = start_t + 0.5
                        window_index += 1

                        rms_mean = float(np.mean(win_rms))
                        rms_std = float(np.std(win_rms))
                        zcr_mean = float(np.mean(win_zcr))
                        high_frac = float(np.mean(win_rms > (rms_mean + max(1e-6, rms_std * 0.4))))

                        windows.append({
                            "start_time": start_t,
                            "end_time": end_t,
                            "rms": rms_mean,
                            "rms_std": rms_std,
                            "zcr": zcr_mean,
                            "high_frac": high_frac,
                        })

                # tail window (partial)
                if frame_buf_rms:
                    win_rms = np.asarray(frame_buf_rms, dtype=np.float32)
                    win_zcr = np.asarray(frame_buf_zcr, dtype=np.float32)
                    start_t = window_index * 0.5
                    windows.append({
                        "start_time": start_t,
                        "end_time": start_t + 0.5,
                        "rms": float(np.mean(win_rms)),
                        "rms_std": float(np.std(win_rms)),
                        "zcr": float(np.mean(win_zcr)) if win_zcr.size else 0.0,
                        "high_frac": float(np.mean(win_rms > (float(np.mean(win_rms)) + max(1e-6, float(np.std(win_rms)) * 0.4)))),
                    })

            if len(windows) < 6:
                return []

            import numpy as np  # local re-import okay for type checkers/runtime consistency

            rms_vals = np.asarray([w["rms"] for w in windows], dtype=np.float32)
            cv_vals = np.asarray([w["rms_std"] / max(w["rms"], 1e-6) for w in windows], dtype=np.float32)
            zcr_vals = np.asarray([w["zcr"] for w in windows], dtype=np.float32)
            hf_vals = np.asarray([w["high_frac"] for w in windows], dtype=np.float32)

            def _norm(val: float, lo: float, hi: float) -> float:
                if hi <= lo:
                    return 0.0
                return max(0.0, min(1.5, (val - lo) / (hi - lo)))

            r75, r95 = np.percentile(rms_vals, [75, 95])
            cv60, cv95 = np.percentile(cv_vals, [60, 95])
            z40, z90 = np.percentile(zcr_vals, [40, 90])
            hf50, hf95 = np.percentile(hf_vals, [50, 95])

            raw_candidates: list[dict] = []
            for w in windows:
                rms_n = _norm(w["rms"], float(r75), float(r95))
                cv = w["rms_std"] / max(w["rms"], 1e-6)
                cv_n = _norm(cv, float(cv60), float(cv95))
                z_n = _norm(w["zcr"], float(z40), float(z90))
                hf_n = _norm(w["high_frac"], float(hf50), float(hf95))

                # Favor bursty voiced-ish noise over steady tones/noise.
                score = (0.55 * rms_n) + (0.55 * cv_n) + (0.25 * z_n) + (0.2 * hf_n)
                if score < 1.05:
                    continue
                if w["rms"] < max(0.002, float(r75) * 0.4):
                    continue

                raw_candidates.append({
                    "start_time": w["start_time"],
                    "end_time": w["end_time"],
                    "score": round(float(score), 4),
                    "source": "acoustic",
                    "snippet": None,
                })

            if not raw_candidates:
                return []

            # Merge adjacent/nearby acoustic windows into laughter events.
            raw_candidates.sort(key=lambda c: c["start_time"])
            merged: list[dict] = []
            for c in raw_candidates:
                if not merged:
                    merged.append(dict(c))
                    continue
                prev = merged[-1]
                if c["start_time"] <= prev["end_time"] + 0.8:
                    prev["end_time"] = max(prev["end_time"], c["end_time"])
                    prev["score"] = round(max(prev["score"], c["score"]) + 0.05, 4)
                    prev["source"] = "acoustic"
                else:
                    merged.append(dict(c))

            # Filter unreasonable durations / keep strongest.
            filtered = []
            for m in merged:
                dur = m["end_time"] - m["start_time"]
                if dur < 0.5 or dur > 15:
                    continue
                # Extend slightly for nicer jump/play context.
                m["start_time"] = max(0.0, m["start_time"] - 0.25)
                m["end_time"] = m["end_time"] + 0.75
                filtered.append(m)
            return filtered

        except Exception as e:
            log_verbose(f"Acoustic laughter detection skipped: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _merge_funny_candidates(self, candidates: list[dict], segments: list[TranscriptSegment]) -> list[dict]:
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: (c["start_time"], c["end_time"]))
        merged: list[dict] = []

        for c in candidates:
            cur = {
                "start_time": float(c["start_time"]),
                "end_time": float(c["end_time"]),
                "score": float(c.get("score", 0.0)),
                "source_set": set(str(c.get("source", "heuristic")).split("+")),
                "snippet": c.get("snippet"),
            }
            if not merged:
                merged.append(cur)
                continue
            prev = merged[-1]
            if cur["start_time"] <= prev["end_time"] + 1.25:
                prev["start_time"] = min(prev["start_time"], cur["start_time"])
                prev["end_time"] = max(prev["end_time"], cur["end_time"])
                prev["score"] = max(prev["score"], cur["score"]) + 0.08
                prev["source_set"].update(cur["source_set"])
                if not prev.get("snippet") and cur.get("snippet"):
                    prev["snippet"] = cur["snippet"]
            else:
                merged.append(cur)

        # Attach nearest transcript snippet for acoustic-only events and finalize score/source.
        final: list[dict] = []
        for m in merged:
            mid = (m["start_time"] + m["end_time"]) / 2.0
            if not m.get("snippet"):
                nearest = None
                nearest_dist = float("inf")
                for seg in segments:
                    seg_mid = (seg.start_time + seg.end_time) / 2.0
                    dist = abs(seg_mid - mid)
                    if dist < nearest_dist:
                        nearest = seg
                        nearest_dist = dist
                if nearest and nearest_dist <= 12:
                    m["snippet"] = (nearest.text or "").strip()[:280]

            source_parts = sorted(s for s in m["source_set"] if s)
            source = "hybrid" if len(source_parts) > 1 else (source_parts[0] if source_parts else "heuristic")
            score = round(min(2.5, float(m["score"])), 3)
            final.append({
                "start_time": round(max(0.0, m["start_time"]), 2),
                "end_time": round(max(m["start_time"] + 0.2, m["end_time"]), 2),
                "score": score,
                "source": source,
                "snippet": (m.get("snippet") or None),
            })

        # Rank by score, then keep a manageable number and restore chronological order.
        try:
            max_results = int(os.getenv("FUNNY_MOMENTS_MAX_SAVED", "25"))
        except Exception:
            max_results = 25
        max_results = max(1, min(max_results, 200))
        top = sorted(final, key=lambda x: (x["score"], x["end_time"] - x["start_time"]), reverse=True)[:max_results]
        return sorted(top, key=lambda x: x["start_time"])

    def detect_funny_moments(self, video_id: int, force: bool = False) -> list[FunnyMoment]:
        """Generate and persist candidate funny/laughter moments for a video."""
        self._set_funny_task_progress(
            video_id,
            task="detect",
            status="running",
            stage="loading",
            message="Loading transcript and existing funny moments...",
            percent=2,
        )
        try:
            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video {video_id} not found")
                _ = video.channel  # ensure relationship loaded for get_audio_path path generation

                existing = session.exec(
                    select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                ).all()
                if existing and not force:
                    self._set_funny_task_progress(
                        video_id,
                        task="detect",
                        status="completed",
                        stage="done",
                        message=f"Using {len(existing)} cached funny moments.",
                        percent=100,
                        current=len(existing),
                        total=len(existing),
                    )
                    return existing

                segments = session.exec(
                    select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
                ).all()
                if not segments:
                    raise ValueError("Transcript segments not found. Run transcription first.")

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="transcript",
                    message="Scanning transcript for laughter cues...",
                    percent=20,
                )
                transcript_candidates = self._transcript_laughter_candidates(segments)

                acoustic_candidates: list[dict] = []
                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="acoustic",
                    message="Analyzing audio for laughter bursts...",
                    percent=45,
                )
                try:
                    audio_path = self.get_audio_path(video)
                    if audio_path.exists():
                        acoustic_candidates = self._acoustic_laughter_candidates(audio_path)
                except Exception as e:
                    log_verbose(f"Funny moments audio analysis skipped for video {video_id}: {e}")

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="merge",
                    message="Merging and ranking funny moment candidates...",
                    percent=75,
                )
                combined = self._merge_funny_candidates(transcript_candidates + acoustic_candidates, segments)

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="save",
                    message="Saving funny moments...",
                    percent=90,
                )
                # Replace cached rows
                for row in existing:
                    session.delete(row)
                session.commit()

                now = datetime.now()
                rows: list[FunnyMoment] = []
                for item in combined:
                    row = FunnyMoment(
                        video_id=video_id,
                        start_time=item["start_time"],
                        end_time=item["end_time"],
                        score=item["score"],
                        source=item["source"],
                        snippet=item.get("snippet"),
                        created_at=now,
                    )
                    session.add(row)
                    rows.append(row)

                session.commit()
                for row in rows:
                    session.refresh(row)

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="completed",
                    stage="done",
                    message=f"Saved {len(rows)} funny moments.",
                    percent=100,
                    current=len(rows),
                    total=len(rows),
                )
                return rows
        except Exception as e:
            self._set_funny_task_progress(
                video_id,
                task="detect",
                status="error",
                stage="error",
                message=str(e),
                percent=100,
            )
            raise

    def _is_llm_enabled(self) -> bool:
        raw = os.getenv("LLM_ENABLED")
        if raw is None:
            raw = os.getenv("OLLAMA_ENABLED", "false")
        return str(raw).lower() == "true"

    def _is_local_ollama_provider_active(self) -> bool:
        return self._is_llm_enabled() and self._get_llm_provider() == "ollama"

    def _resolve_local_ollama_min_free_vram_gb(self, total_gb: float) -> float:
        default_gb = 6.0
        if total_gb >= 28.0:
            default_gb = 10.0
        elif total_gb >= 20.0:
            default_gb = 8.0
        raw = (os.getenv("OLLAMA_LOCAL_LLM_MIN_FREE_VRAM_GB") or "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
        return default_gb

    def _get_local_ollama_vram_guard(self) -> tuple[bool, str, float | None, float | None]:
        if not self._is_local_ollama_provider_active():
            return True, "non_local_ollama_provider", None, None

        if self._has_active_pipeline_gpu_work():
            return False, "pipeline_gpu_work_active", None, None

        self._ensure_device()
        if self.device != "cuda":
            return True, "non_cuda_worker", None, None

        snap = self._cuda_memory_snapshot()
        free_b, total_b, _, _, free_gb, _ = self._snap_unpack(snap)
        total_gb = float(total_b) / (1024 ** 3) if total_b > 0 else 0.0
        min_free_gb = self._resolve_local_ollama_min_free_vram_gb(total_gb)
        if free_gb < min_free_gb:
            return False, f"low_vram_headroom_{free_gb:.1f}gb_below_{min_free_gb:.1f}gb", free_gb, min_free_gb
        return True, "ok", free_gb, min_free_gb

    def _prepare_for_local_ollama_llm_work(self, job_id: int | None = None) -> tuple[float | None, float | None]:
        if not self._is_local_ollama_provider_active():
            return None, None
        self._ensure_device()
        if self.device != "cuda":
            return None, None

        self._release_parakeet_model("pre_local_ollama_llm", job_id=job_id)
        self._release_whisper_model("pre_local_ollama_llm", job_id=job_id)
        self._release_diarization_models("pre_local_ollama_llm", job_id=job_id)
        self._clear_cuda_cache()

        snap = self._cuda_memory_snapshot()
        free_b, total_b, _, _, free_gb, _ = self._snap_unpack(snap)
        total_gb = float(total_b) / (1024 ** 3) if total_b > 0 else 0.0
        if job_id:
            self._upsert_job_payload_fields(
                job_id,
                {
                    "local_ollama_prepared": True,
                    "local_ollama_free_gb_after_prepare": round(free_gb, 2) if free_b > 0 else 0.0,
                    "local_ollama_total_gb_after_prepare": round(total_gb, 2) if total_gb > 0 else 0.0,
                },
            )
        return free_gb, total_gb

    def _raise_if_local_ollama_llm_is_blocked(self, job_id: int | None = None):
        allowed, reason, free_gb, min_free_gb = self._get_local_ollama_vram_guard()
        if allowed:
            self._prepare_for_local_ollama_llm_work(job_id=job_id)
            return

        detail = {
            "pipeline_gpu_work_active": "Local Ollama funny-moment explanation is blocked while pipeline GPU work is active.",
            "non_local_ollama_provider": "",
            "non_cuda_worker": "",
        }.get(reason)
        if not detail:
            if free_gb is not None and min_free_gb is not None:
                detail = (
                    "Local Ollama funny-moment explanation is blocked until VRAM headroom recovers "
                    f"({free_gb:.1f} GB free, needs at least {min_free_gb:.1f} GB)."
                )
            else:
                detail = "Local Ollama funny-moment explanation is temporarily blocked."

        if job_id:
            self._upsert_job_payload_fields(
                job_id,
                {
                    "local_ollama_blocked": True,
                    "local_ollama_block_reason": reason,
                    "local_ollama_free_gb": round(float(free_gb), 2) if free_gb is not None else None,
                    "local_ollama_min_free_gb": round(float(min_free_gb), 2) if min_free_gb is not None else None,
                },
            )
        raise RuntimeError(detail)

    def _get_llm_provider(self) -> str:
        provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
        aliases = {
            "nvidia": "nvidia_nim",
            "nim": "nvidia_nim",
            "nvidia-nim": "nvidia_nim",
            "nvidia_nim": "nvidia_nim",
            "chatgpt": "openai",
            "openai": "openai",
            "claude": "anthropic",
            "anthropic": "anthropic",
            "google": "gemini",
            "google_gemini": "gemini",
            "google-gemini": "gemini",
            "gemini": "gemini",
            "groq": "groq",
            "openrouter": "openrouter",
            "xai": "xai",
            "ollama": "ollama",
        }
        return aliases.get(provider, provider)

    def _get_nvidia_nim_min_request_interval_seconds(self) -> float:
        """Minimum spacing between hosted NIM requests (process-wide)."""
        raw = (os.getenv("NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS") or "2.5").strip()
        try:
            value = float(raw)
        except Exception:
            value = 2.5
        # Keep within sane bounds.
        return max(0.0, min(value, 30.0))

    def _get_provider_default_model(self, provider: str) -> str:
        normalized = clone_svc._normalize_provider(provider) or "ollama"
        if normalized == "nvidia_nim":
            return (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip()
        if normalized == "openai":
            return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        if normalized == "anthropic":
            return (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip()
        if normalized == "gemini":
            return (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        if normalized == "groq":
            return (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        if normalized == "openrouter":
            return (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip()
        if normalized == "xai":
            return (os.getenv("XAI_MODEL") or "grok-2").strip()
        return (os.getenv("OLLAMA_MODEL") or "mistral").strip()

    def resolve_clone_llm_target(
        self,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> tuple[str, str, str]:
        provider = clone_svc._normalize_provider(provider_override) or self._get_llm_provider()
        model = str(model_override or "").strip() or self._get_provider_default_model(provider)
        return provider, model, f"{provider}:{model}"

    def _get_configured_llm_model_name(self) -> str:
        _provider, _model, target_name = self.resolve_clone_llm_target()
        return target_name

    def _extract_openai_chat_text(self, raw: str) -> str:
        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") if choices else {}
            content = (message or {}).get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").lower()
                        if item_type and ("reason" in item_type or "think" in item_type):
                            continue
                        text_part = item.get("text")
                        if isinstance(text_part, str):
                            parts.append(text_part)
                text = "\n".join(parts).strip()
            else:
                text = str(content or "").strip()
            if not text:
                text = str((message or {}).get("reasoning_content") or "").strip()
        except Exception:
            text = str(raw or "").strip()
        return self._strip_llm_reasoning_artifacts(text)

    def _openai_compatible_generate_text(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
        extra_headers: dict | None = None,
        extra_payload: dict | None = None,
    ) -> str:
        import urllib.error

        if not api_key:
            raise RuntimeError(f"{provider_name} API key is not configured in Settings.")
        if not model:
            raise RuntimeError(f"{provider_name} model is not configured in Settings.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max(32, int(num_predict)),
        }
        if extra_payload:
            payload.update(extra_payload)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        headers = {k: v for k, v in headers.items() if v is not None and str(v) != ""}

        normalized_base = base_url.rstrip("/")
        endpoint = f"{normalized_base}/chat/completions" if normalized_base.lower().endswith("/v1") else f"{normalized_base}/v1/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"{provider_name} request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach {provider_name} at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"{provider_name} request failed: {e}") from e

        text = self._extract_openai_chat_text(raw)
        if not text:
            raise RuntimeError(f"{provider_name} returned an empty response.")
        return text

    def _anthropic_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        import urllib.error

        api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        base_url = (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip()

        if not api_key:
            raise RuntimeError("Anthropic API key is not configured. Set ANTHROPIC_API_KEY in Settings.")
        if not model:
            raise RuntimeError("ANTHROPIC_MODEL is not configured.")

        payload = {
            "model": model,
            "max_tokens": max(32, int(num_predict)),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Anthropic request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Anthropic at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Anthropic request failed: {e}") from e

        try:
            data = json.loads(raw)
            parts = data.get("content") or []
            text = "\n".join(
                str(p.get("text") or "").strip()
                for p in parts
                if isinstance(p, dict) and str(p.get("type") or "").lower() == "text"
            ).strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("Anthropic returned an empty response.")
        return text

    def _gemini_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        import urllib.error
        import urllib.parse

        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        base_url = (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

        if not api_key:
            raise RuntimeError("Gemini API key is not configured. Set GEMINI_API_KEY in Settings.")
        if not model:
            raise RuntimeError("GEMINI_MODEL is not configured.")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": max(32, int(num_predict)),
            },
        }

        encoded_model = urllib.parse.quote(model, safe="")
        req = urllib.request.Request(
            f"{base_url}/v1beta/models/{encoded_model}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Gemini request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Gemini at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e

        try:
            data = json.loads(raw)
            candidates = data.get("candidates") or []
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
            text = "\n".join(
                str(p.get("text") or "").strip()
                for p in parts
                if isinstance(p, dict) and p.get("text")
            ).strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    def _nvidia_nim_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        import urllib.error

        api_key = (os.getenv("NVIDIA_NIM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA NIM API key is not configured. Set NVIDIA_NIM_API_KEY in Settings.")

        base_url = (os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip()
        if not model:
            raise RuntimeError("NVIDIA_NIM_MODEL is not configured.")

        # Kimi K2.5 supports thinking mode via chat_template_kwargs.thinking.
        thinking_mode_raw = (os.getenv("NVIDIA_NIM_THINKING_MODE") or "false").strip().lower()
        thinking_mode = thinking_mode_raw == "true"

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max(32, int(num_predict)),
        }
        if thinking_mode:
            payload["chat_template_kwargs"] = {"thinking": True}

        max_attempts = 5
        raw = ""
        last_http_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self._nvidia_nim_request_lock:
                    now = time.time()
                    wait_for = max(0.0, self._nvidia_nim_next_allowed_at - now)
                    if wait_for > 0:
                        log_verbose(f"NVIDIA NIM pacing wait: {wait_for:.2f}s")
                        time.sleep(wait_for)

                    req = urllib.request.Request(
                        f"{base_url}/v1/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "application/json",
                        },
                        method="POST",
                    )

                    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")

                    # Proactively space requests even after success to avoid bursty
                    # stage transitions (e.g., chunk loop -> moment loop).
                    self._nvidia_nim_next_allowed_at = time.time() + self._get_nvidia_nim_min_request_interval_seconds()
                    last_http_error = None
                    break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(e)

                if e.code == 429 and attempt < max_attempts:
                    retry_after = None
                    try:
                        retry_after_hdr = (e.headers or {}).get("Retry-After")
                        if retry_after_hdr:
                            retry_after = float(retry_after_hdr)
                    except Exception:
                        retry_after = None
                    backoff = retry_after if retry_after is not None else min(12.0, 1.5 * (2 ** (attempt - 1)))
                    # Push the next-allowed time forward process-wide so concurrent
                    # explain jobs/UI actions don't immediately retrigger the limit.
                    with self._nvidia_nim_request_lock:
                        self._nvidia_nim_next_allowed_at = max(
                            self._nvidia_nim_next_allowed_at,
                            time.time() + max(0.5, backoff) + self._get_nvidia_nim_min_request_interval_seconds()
                        )
                    log(f"NVIDIA NIM rate-limited (429). Retrying in {backoff:.1f}s (attempt {attempt}/{max_attempts})...")
                    time.sleep(max(0.5, backoff))
                    last_http_error = RuntimeError(f"NVIDIA NIM request failed (429): {detail[:500]}")
                    continue

                if e.code == 429:
                    raise RuntimeError(
                        "NVIDIA NIM request failed (429): Too Many Requests. "
                        "Try again in a minute, reduce Funny->Explain batch size, or disable Kimi thinking mode."
                    ) from e
                raise RuntimeError(f"NVIDIA NIM request failed ({e.code}): {detail[:500]}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Could not reach NVIDIA NIM at {base_url}: {e}") from e
            except Exception as e:
                raise RuntimeError(f"NVIDIA NIM request failed: {e}") from e

        if last_http_error is not None:
            raise last_http_error

        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") if choices else {}
            content = (message or {}).get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").lower()
                        # Some reasoning-capable responses return structured content parts
                        # that include reasoning/thinking chunks alongside the final answer.
                        if item_type and ("reason" in item_type or "think" in item_type):
                            continue
                        text_part = item.get("text")
                        if isinstance(text_part, str):
                            parts.append(text_part)
                text = "\n".join(parts).strip()
            else:
                text = str(content or "").strip()
            if not text:
                # Some thinking-mode responses may carry answer/reasoning in alternate fields.
                text = str((message or {}).get("reasoning_content") or "").strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("NVIDIA NIM returned an empty response.")
        return text

    def _ollama_generate_text(
        self,
        prompt: str,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        """Call the configured LLM provider and return raw generated text."""
        provider = clone_svc._normalize_provider(provider_override) or self._get_llm_provider()
        if provider == "nvidia_nim":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._nvidia_nim_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "openai":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="OpenAI",
                base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/"),
                api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "anthropic":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._anthropic_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "gemini":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._gemini_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "groq":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="Groq",
                base_url=(os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai").rstrip("/"),
                api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "openrouter":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="OpenRouter",
                base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api").rstrip("/"),
                api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
                extra_headers={
                    "HTTP-Referer": (os.getenv("OPENROUTER_REFERER") or "").strip(),
                    "X-Title": (os.getenv("OPENROUTER_TITLE") or "Chatalogue").strip(),
                },
            )

        if provider == "xai":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="xAI",
                base_url=(os.getenv("XAI_BASE_URL") or "https://api.x.ai").rstrip("/"),
                api_key=(os.getenv("XAI_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("XAI_MODEL") or "grok-2").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider != "ollama":
            raise RuntimeError(
                "Unsupported LLM provider "
                f"'{provider}'. Select one of: ollama, nvidia_nim, openai, anthropic, gemini, groq, openrouter, xai."
            )

        """Call Ollama and return raw generated text."""
        import urllib.error

        if not self._is_llm_enabled():
            raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")

        base_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("OLLAMA_MODEL") or "mistral").strip()
        if not model:
            raise RuntimeError("OLLAMA_MODEL is not configured.")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Reasoning-capable models (e.g., qwen3.5) may emit only `thinking`
            # and leave `response` empty when think mode is on. Force final-answer
            # generation for pipeline tasks that need parseable output.
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        text = ""
        try:
            data = json.loads(raw)
        except Exception:
            data = None
            text = raw.strip()

        if isinstance(data, dict):
            model_error = str(data.get("error") or "").strip()
            if model_error:
                raise RuntimeError(f"Ollama error: {model_error}")

            text = str(data.get("response") or "").strip()
            if not text:
                thinking = str(data.get("thinking") or "").strip()
                if thinking:
                    raise RuntimeError(
                        "Ollama returned no final response (thinking-only output). "
                        "For qwen3.5 models, disable reasoning mode (think=false)."
                    )

        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def generate_clone_text(
        self,
        prompt: str,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        return self._ollama_generate_text(
            prompt,
            provider_override=provider_override,
            model_override=model_override,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
        )

    def _strip_llm_reasoning_artifacts(self, text: str) -> str:
        """Remove common reasoning/thinking wrappers/preambles from LLM output."""

        if not text:
            return ""

        cleaned = str(text).strip()
        if not cleaned:
            return ""

        # Remove explicit thinking blocks if the provider includes them inline.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
        if "</think>" in cleaned.lower():
            cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1].strip()
        cleaned = re.sub(r"^\s*```(?:thinking|reasoning)\s*[\s\S]*?```\s*", "", cleaned, flags=re.IGNORECASE).strip()

        lowered = cleaned.lower()
        meta_start = lowered.startswith((
            "the user wants me to",
            "the user asked me to",
            "i need to analyze",
            "first, i need to",
            "let me analyze",
            "i should analyze",
        ))

        if meta_start:
            # If a likely answer cue exists later, jump to it.
            cue_patterns = [
                r"\blikely joke\b",
                r"\bthe joke likely\b",
                r"\bthis laugh is likely\b",
                r"\bthe humor is likely\b",
                r"\bsummary\s*:",
                r"\bmost likely\b",
            ]
            best_idx = None
            for pat in cue_patterns:
                m = re.search(pat, lowered)
                if m and m.start() > 40:
                    best_idx = m.start() if best_idx is None else min(best_idx, m.start())
            if best_idx is not None:
                cleaned = cleaned[best_idx:].lstrip(":- \n\t")
            else:
                # Drop leading introspection sentences and keep the first actual answer-ish sentence.
                parts = re.split(r"(?<=[.!?])\s+", cleaned)
                kept = []
                skipping = True
                for part in parts:
                    p = part.strip()
                    if not p:
                        continue
                    p_low = p.lower()
                    is_meta = (
                        p_low.startswith(("the user wants me to", "the user asked me to", "first, i need to", "i need to", "let me", "i should"))
                        or "transcript context provided" in p_low
                    )
                    if skipping and is_meta:
                        continue
                    skipping = False
                    kept.append(p)
                if kept:
                    cleaned = " ".join(kept).strip()

        # Normalize whitespace after stripping.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_ollama_summary_confidence(self, text: str, *, max_summary_chars: int = 600) -> tuple[str, str]:
        """Parse summary/confidence JSON with robust fallback handling."""

        text = self._strip_llm_reasoning_artifacts(text)
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Best-effort JSON extraction if the model wraps JSON in prose/fences.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or "").strip()
            confidence = str(parsed.get("confidence") or "low").strip().lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            if summary:
                return summary[:max_summary_chars], confidence

        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        summary_match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.IGNORECASE | re.DOTALL)
        conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        if summary_match:
            try:
                recovered_summary = json.loads(f'"{summary_match.group(1)}"')
            except Exception:
                recovered_summary = summary_match.group(1).encode("utf-8", "ignore").decode("unicode_escape", "ignore")
            recovered_summary = str(recovered_summary).strip()
            recovered_conf = (conf_match.group(1).strip().lower() if conf_match else "low")
            if recovered_conf not in {"low", "medium", "high"}:
                recovered_conf = "low"
            if recovered_summary:
                return recovered_summary[:max_summary_chars], recovered_conf

        cleaned = re.sub(r'^\s*json\s*', '', cleaned, flags=re.IGNORECASE).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned[:max_summary_chars], "low"

    def _build_transcript_context_lines(self, segments: list[TranscriptSegment], speaker_map: dict[int, str]) -> list[str]:
        lines: list[str] = []
        for s in segments:
            text = (s.text or "").replace("\n", " ").strip()
            if not text:
                continue
            speaker_name = speaker_map.get(s.speaker_id) if s.speaker_id else None
            stamp = f"{int(s.start_time // 60)}:{int(s.start_time % 60):02d}"
            who = speaker_name or "Unknown"
            lines.append(f"[{stamp}] {who}: {text}")
        return lines

    def _chunk_transcript_lines_for_llm(
        self,
        lines: list[str],
        *,
        max_chunk_chars: int = 12_000,
        max_chunk_lines: int = 140,
        max_chunks: int = 24,
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0
        for line in lines:
            line_len = len(line) + 1
            if current and (current_chars + line_len > max_chunk_chars or len(current) >= max_chunk_lines):
                chunks.append("\n".join(current))
                if len(chunks) >= max_chunks:
                    return chunks
                current = []
                current_chars = 0
            current.append(line)
            current_chars += line_len
        if current and len(chunks) < max_chunks:
            chunks.append("\n".join(current))
        return chunks

    def _ollama_generate_episode_humor_context_summary(
        self,
        transcript_lines: list[str],
        *,
        progress_video_id: int | None = None,
        stage2_total: int | None = None,
    ) -> str:
        """Generate a cached episode-wide humor context summary from the full transcript (chunked)."""
        if not transcript_lines:
            raise RuntimeError("Cannot build episode humor context summary: transcript is empty.")

        chunks = self._chunk_transcript_lines_for_llm(transcript_lines)
        if not chunks:
            raise RuntimeError("Cannot build episode humor context summary: no transcript chunks.")

        chunk_summaries: list[str] = []
        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks, start=1):
            if progress_video_id is not None:
                # Reserve ~8%-18% of the total explain progress bar for Stage 1 chunking.
                stage1_pct = 8 + int((idx - 1) / max(1, total_chunks) * 10)
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_chunks",
                    message=f"Building episode-wide humor context summary (chunk {idx}/{total_chunks})...",
                    percent=stage1_pct,
                    current=idx - 1,
                    total=total_chunks,
                )
            prompt = (
                "You are summarizing ONE chunk of a podcast transcript to support humor analysis.\n"
                "Extract comedic context only: running bits, callbacks, teasing, repeated topics, and tone.\n"
                "Do not summarize everything; focus on what could make later laughter make sense.\n\n"
                f"Chunk {idx} of {total_chunks}\n\n"
                "Return ONLY JSON with this schema:\n"
                "{\"summary\":\"chunk humor context summary\",\"confidence\":\"low|medium|high\"}\n\n"
                "Transcript chunk:\n"
                f"{chunk_text}"
            )
            text = self._ollama_generate_text(
                prompt,
                temperature=0.15,
                num_predict=220,
                timeout_seconds=120,
            )
            summary, _confidence = self._parse_ollama_summary_confidence(text, max_summary_chars=900)
            if summary:
                chunk_summaries.append(f"Chunk {idx}: {summary}")
            if progress_video_id is not None:
                stage1_pct = 8 + int(idx / max(1, total_chunks) * 10)
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_chunks",
                    message=f"Building episode-wide humor context summary (chunk {idx}/{total_chunks})...",
                    percent=stage1_pct,
                    current=idx,
                    total=total_chunks,
                )

        if not chunk_summaries:
            raise RuntimeError("Ollama did not produce usable episode chunk summaries.")

        if len(chunk_summaries) == 1:
            if progress_video_id is not None:
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_done",
                    message=f"Episode-wide humor context summary complete. Preparing to explain moments (0/{stage2_total or 0})...",
                    percent=19,
                    current=0 if stage2_total is not None else None,
                    total=stage2_total,
                )
            return chunk_summaries[0][:1600]

        merged_input = "\n".join(chunk_summaries)
        if len(merged_input) > 14_000:
            merged_input = merged_input[-14_000:]

        merge_prompt = (
            "You are combining chunk-level humor summaries from a full podcast episode.\n"
            "Create one EPISODE-WIDE humor context summary to help explain specific laughter timestamps.\n"
            "Include recurring jokes/callbacks, people being teased, repeated themes, and the comedic tone.\n"
            "Be concise and specific.\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\"summary\":\"episode-wide humor context summary\",\"confidence\":\"low|medium|high\"}\n\n"
            "Chunk summaries:\n"
            f"{merged_input}"
        )
        if progress_video_id is not None:
            self._set_funny_task_progress(
                progress_video_id,
                task="explain",
                status="running",
                stage="global_context_merge",
                message=f"Merging {len(chunk_summaries)} chunk summaries into episode-wide context...",
                percent=19,
                current=len(chunk_summaries),
                total=len(chunk_summaries),
            )
        merged_text = self._ollama_generate_text(
            merge_prompt,
            temperature=0.15,
            num_predict=320,
            timeout_seconds=150,
        )
        merged_summary, _confidence = self._parse_ollama_summary_confidence(merged_text, max_summary_chars=1600)
        if progress_video_id is not None:
            self._set_funny_task_progress(
                progress_video_id,
                task="explain",
                status="running",
                stage="global_context_done",
                message=f"Episode-wide humor context summary complete. Preparing to explain moments (0/{stage2_total or 0})...",
                percent=19,
                current=0 if stage2_total is not None else None,
                total=stage2_total,
            )
        return merged_summary or merged_input[:1600]

    def _ensure_episode_humor_context_summary(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        speaker_map: dict[int, str],
        *,
        force: bool = False,
        progress_video_id: int | None = None,
        stage2_total: int | None = None,
    ) -> str | None:
        if not force and getattr(video, "humor_context_summary", None):
            return video.humor_context_summary

        transcript_lines = self._build_transcript_context_lines(segments, speaker_map)
        if not transcript_lines:
            return None

        summary = self._ollama_generate_episode_humor_context_summary(
            transcript_lines,
            progress_video_id=progress_video_id,
            stage2_total=stage2_total,
        )
        model_name = self._get_configured_llm_model_name()
        video.humor_context_summary = summary
        video.humor_context_model = model_name
        video.humor_context_generated_at = datetime.now()
        session.add(video)
        session.commit()
        session.refresh(video)
        return summary

    def _ollama_generate_humor_summary(
        self,
        context_text: str,
        moment_start: float,
        moment_end: float,
        *,
        episode_context_summary: str | None = None,
    ) -> tuple[str, str]:
        """Ask Ollama to infer what the humor was likely about from transcript context."""
        prompt = (
            "You are analyzing a podcast transcript around a laughter moment.\n"
            "Infer what the joke/humor was LIKELY about from the transcript context.\n"
            "Use the episode-wide humor context summary to recognize callbacks and running bits, "
            "but prioritize local transcript evidence.\n"
            "Be concise and uncertain when needed.\n\n"
            f"Laughter moment timestamp: {moment_start:.1f}s to {moment_end:.1f}s\n\n"
            + (
                "Episode-wide humor context summary (may be incomplete):\n"
                f"{episode_context_summary}\n\n"
                if episode_context_summary else ""
            )
            + "Return ONLY JSON with this schema:\n"
            "{\"summary\":\"1-2 sentence explanation of the likely joke/humor\","
            "\"confidence\":\"low|medium|high\"}\n\n"
            "Transcript context:\n"
            f"{context_text}"
        )
        text = self._ollama_generate_text(prompt, temperature=0.2, num_predict=180, timeout_seconds=90)
        return self._parse_ollama_summary_confidence(text, max_summary_chars=600)

    def _seconds_to_chapter_timestamp(self, seconds: float) -> str:
        total = max(0, int(round(float(seconds))))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _parse_chapter_timestamp_to_seconds(self, value) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parts = text.split(":")
        try:
            nums = [int(p) for p in parts]
        except Exception:
            return None
        if len(nums) == 2:
            m, s = nums
            if m < 0 or s < 0 or s > 59:
                return None
            return m * 60 + s
        if len(nums) == 3:
            h, m, s = nums
            if h < 0 or m < 0 or m > 59 or s < 0 or s > 59:
                return None
            return h * 3600 + m * 60 + s
        return None

    def _parse_json_object_from_text(self, text: str) -> dict | None:
        if not text:
            return None
        cleaned = self._strip_llm_reasoning_artifacts(text)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _normalize_youtube_ai_chapters(self, chapters, *, video_duration_seconds: float | None = None) -> list[dict]:
        normalized: list[dict] = []
        duration_cap = None
        try:
            if video_duration_seconds is not None:
                duration_cap = max(1, int(float(video_duration_seconds)))
        except Exception:
            duration_cap = None

        seen_starts = set()
        if not isinstance(chapters, list):
            chapters = []

        for item in chapters:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp") or item.get("start") or item.get("time")
            start_sec = self._parse_chapter_timestamp_to_seconds(ts)
            if start_sec is None:
                # accept numeric second fields if model returns them
                for key in ("start_seconds", "start_sec", "seconds"):
                    if key in item:
                        try:
                            start_sec = max(0, int(float(item[key])))
                            break
                        except Exception:
                            start_sec = None
            if start_sec is None:
                continue
            if duration_cap is not None and start_sec >= duration_cap:
                continue
            if start_sec in seen_starts:
                continue
            seen_starts.add(start_sec)

            title = str(item.get("title") or item.get("chapter") or "").strip()
            desc = str(item.get("description") or item.get("summary") or "").strip()
            if not title:
                continue
            title = " ".join(title.split())[:140]
            desc = " ".join(desc.split())[:280]

            normalized.append({
                "start_seconds": int(start_sec),
                "timestamp": self._seconds_to_chapter_timestamp(start_sec),
                "title": title,
                "description": desc,
            })

        normalized.sort(key=lambda c: c["start_seconds"])

        if normalized and normalized[0]["start_seconds"] != 0:
            normalized.insert(0, {
                "start_seconds": 0,
                "timestamp": "0:00",
                "title": "Intro",
                "description": "",
            })
        elif not normalized:
            normalized = [{
                "start_seconds": 0,
                "timestamp": "0:00",
                "title": "Episode Start",
                "description": "",
            }]

        # Enforce increasing timestamps and prune chapters that are too dense (<15s apart).
        pruned: list[dict] = []
        for ch in normalized:
            if not pruned:
                pruned.append(ch)
                continue
            if ch["start_seconds"] <= pruned[-1]["start_seconds"]:
                continue
            if ch["start_seconds"] - pruned[-1]["start_seconds"] < 15:
                continue
            pruned.append(ch)
        return pruned[:30]

    def _build_youtube_description_text(self, summary: str, chapters: list[dict]) -> str:
        lines: list[str] = []
        summary = (summary or "").strip()
        if summary:
            lines.append(summary)
            lines.append("")
        lines.append("Chapters")
        for ch in chapters:
            stamp = str(ch.get("timestamp") or "0:00").strip()
            title = str(ch.get("title") or "").strip()
            if not title:
                continue
            lines.append(f"{stamp} {title}")
        return "\n".join(lines).strip()

    def _parse_youtube_ai_result(self, text: str, *, video_duration_seconds: float | None = None) -> tuple[str, list[dict]]:
        parsed = self._parse_json_object_from_text(text)
        summary = ""
        chapters: list[dict] = []
        if isinstance(parsed, dict):
            summary = str(
                parsed.get("video_summary")
                or parsed.get("summary")
                or parsed.get("description_summary")
                or ""
            ).strip()
            chapters = self._normalize_youtube_ai_chapters(
                parsed.get("chapters") or parsed.get("chapter_timestamps") or [],
                video_duration_seconds=video_duration_seconds,
            )
        if not summary:
            cleaned = self._strip_llm_reasoning_artifacts(text)
            summary = " ".join(cleaned.split())[:1200]
        summary = summary[:1600]
        return summary, chapters

    def generate_youtube_metadata_suggestion(self, video_id: int, force: bool = False) -> Video:
        """Generate a YouTube-style summary + chapter timestamps/descriptions from transcript."""
        if not self._is_llm_enabled():
            raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")

        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")

            if (
                not force
                and getattr(video, "youtube_ai_summary", None)
                and getattr(video, "youtube_ai_chapters_json", None)
                and getattr(video, "youtube_ai_description_text", None)
            ):
                return video

            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            if not segments:
                raise ValueError("Transcript segments not found. Run transcription first.")

            speaker_map: dict[int, str] = {}
            if video.channel_id:
                for sp in session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all():
                    if sp.id is not None:
                        speaker_map[sp.id] = sp.name

            transcript_lines = self._build_transcript_context_lines(segments, speaker_map)
            if not transcript_lines:
                raise ValueError("Transcript is empty.")

            # Allow larger transcript coverage than humor context to improve chapter generation.
            chunks = self._chunk_transcript_lines_for_llm(
                transcript_lines,
                max_chunk_chars=14_000,
                max_chunk_lines=180,
                max_chunks=36,
            )
            if not chunks:
                raise RuntimeError("Failed to prepare transcript chunks for chapter generation.")

            approx_duration = float(video.duration or (segments[-1].end_time if segments else 0) or 0)
            chunk_outputs: list[dict] = []
            total_chunks = len(chunks)
            for idx, chunk_text in enumerate(chunks, start=1):
                chunk_prompt = (
                    "You are preparing metadata for a YouTube podcast episode from a transcript chunk.\n"
                    "Identify major topics/themes and any good chapter boundaries visible in THIS chunk only.\n"
                    "Use transcript timestamps as-is. Do not invent content not present in the transcript.\n"
                    "Prefer broad thematic sections over tiny beats.\n\n"
                    f"Episode title: {video.title}\n"
                    f"Chunk {idx} of {total_chunks}\n\n"
                    "Return ONLY JSON with this schema:\n"
                    "{\"chunk_summary\":\"2-4 sentence topic summary for this chunk\","
                    "\"chapter_candidates\":[{\"timestamp\":\"MM:SS or H:MM:SS\",\"title\":\"short chapter title\",\"description\":\"one-sentence chapter description\"}]}\n\n"
                    "Transcript chunk:\n"
                    f"{chunk_text}"
                )
                raw = self._ollama_generate_text(
                    chunk_prompt,
                    temperature=0.15,
                    num_predict=500,
                    timeout_seconds=180,
                )
                parsed = self._parse_json_object_from_text(raw) or {}
                chunk_summary = str(parsed.get("chunk_summary") or parsed.get("summary") or "").strip()
                chapter_candidates = self._normalize_youtube_ai_chapters(
                    parsed.get("chapter_candidates") or [],
                    video_duration_seconds=approx_duration,
                )

                # Fallback summary if the model skipped JSON.
                if not chunk_summary:
                    fallback_summary, _ = self._parse_ollama_summary_confidence(raw, max_summary_chars=900)
                    chunk_summary = fallback_summary

                if chunk_summary or chapter_candidates:
                    chunk_outputs.append({
                        "chunk_index": idx,
                        "chunk_summary": chunk_summary[:900],
                        "chapter_candidates": chapter_candidates[:8],
                    })

            if not chunk_outputs:
                raise RuntimeError("LLM did not produce usable chunk summaries/chapters.")

            chunk_lines: list[str] = []
            flat_candidates: list[dict] = []
            for item in chunk_outputs:
                summary = (item.get("chunk_summary") or "").strip()
                if summary:
                    chunk_lines.append(f"Chunk {item['chunk_index']} summary: {summary}")
                cands = item.get("chapter_candidates") or []
                if cands:
                    for c in cands:
                        flat_candidates.append(c)
                        chunk_lines.append(
                            f"Chunk {item['chunk_index']} candidate chapter: {c['timestamp']} | {c['title']}"
                            + (f" | {c['description']}" if c.get("description") else "")
                        )

            merge_input = "\n".join(chunk_lines).strip()
            if len(merge_input) > 20_000:
                merge_input = merge_input[-20_000:]

            target_chapter_count = 8
            if approx_duration >= 3600:
                target_chapter_count = 12
            if approx_duration >= 7200:
                target_chapter_count = 16

            merge_prompt = (
                "You are generating YouTube-ready episode metadata from chunk-level transcript analyses.\n"
                "Produce:\n"
                "1) a strong YouTube-style episode summary (description intro) in 2-4 sentences\n"
                "2) chapter timestamps for major thematic sections\n"
                "3) a short one-sentence description for each chapter (for UI display)\n\n"
                "Rules:\n"
                "- Chapters must be chronological and represent major sections\n"
                "- First chapter MUST start at 0:00\n"
                "- Use timestamps only from the candidates/context; do not invent impossible times\n"
                "- Keep chapter titles concise and descriptive\n"
                "- Focus on what is actually discussed in the transcript\n"
                f"- Target about {target_chapter_count} chapters for this episode length\n\n"
                f"Episode title: {video.title}\n"
                f"Approx duration: {self._seconds_to_chapter_timestamp(approx_duration) if approx_duration else 'unknown'}\n\n"
                "Return ONLY JSON with this schema:\n"
                "{\"video_summary\":\"2-4 sentence YouTube description summary\","
                "\"chapters\":[{\"timestamp\":\"0:00\",\"title\":\"...\",\"description\":\"...\"}]}\n\n"
                "Chunk analyses and candidate chapters:\n"
                f"{merge_input}"
            )

            merged_raw = self._ollama_generate_text(
                merge_prompt,
                temperature=0.15,
                num_predict=900,
                timeout_seconds=240,
            )
            summary, chapters = self._parse_youtube_ai_result(
                merged_raw,
                video_duration_seconds=approx_duration,
            )

            if len(chapters) <= 1 and flat_candidates:
                # Fallback: use deduped candidate timestamps if merge failed to return chapters.
                chapters = self._normalize_youtube_ai_chapters(flat_candidates, video_duration_seconds=approx_duration)

            if not summary:
                # Fallback to merged chunk summaries if model refused final JSON.
                summary = " ".join(
                    [str(item.get("chunk_summary") or "").strip() for item in chunk_outputs if item.get("chunk_summary")]
                )[:1600]

            youtube_text = self._build_youtube_description_text(summary, chapters)
            model_name = self._get_configured_llm_model_name()

            video.youtube_ai_summary = summary
            video.youtube_ai_chapters_json = json.dumps(chapters, ensure_ascii=False)
            video.youtube_ai_description_text = youtube_text
            video.youtube_ai_model = model_name
            video.youtube_ai_generated_at = datetime.now()
            session.add(video)
            session.commit()
            session.refresh(video)
            return video

    def explain_funny_moments(
        self,
        video_id: int,
        force: bool = False,
        limit: int | None = None,
        *,
        job_id: int | None = None,
    ) -> list[FunnyMoment]:
        """Generate AI summaries for detected funny moments using transcript context + Ollama."""
        if limit is None:
            try:
                limit = int(os.getenv("FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", "12"))
            except Exception:
                limit = 12
        limit = max(1, min(int(limit), 200))
        self._raise_if_local_ollama_llm_is_blocked(job_id=job_id)
        self._set_funny_task_progress(
            video_id,
            task="explain",
            status="running",
            stage="loading",
            message="Loading funny moments and transcript...",
            percent=2,
        )
        try:
            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video {video_id} not found")

                moments = session.exec(
                    select(FunnyMoment)
                    .where(FunnyMoment.video_id == video_id)
                    .order_by(FunnyMoment.score.desc(), FunnyMoment.start_time)
                ).all()
                if not moments:
                    raise ValueError("No funny moments found. Run funny-moment detection first.")

                segments = session.exec(
                    select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
                ).all()
                if not segments:
                    raise ValueError("Transcript segments not found. Run transcription first.")

                speaker_map: dict[int, str] = {}
                if video.channel_id:
                    for sp in session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all():
                        if sp.id is not None:
                            speaker_map[sp.id] = sp.name

                target_moments = []
                for m in moments:
                    if force or not m.humor_summary:
                        target_moments.append(m)
                    if len(target_moments) >= limit:
                        break

                if not target_moments:
                    self._set_funny_task_progress(
                        video_id,
                        task="explain",
                        status="completed",
                        stage="done",
                        message="No moments needed explanation.",
                        percent=100,
                        current=0,
                        total=0,
                    )
                    return session.exec(
                        select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                    ).all()

                total_targets = len(target_moments)
                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="running",
                    stage="global_context",
                    message="Building episode-wide humor context summary (Stage 1)...",
                    percent=8,
                    current=0,
                    total=total_targets,
                )

                model_name = self._get_configured_llm_model_name()
                episode_context_summary = None
                try:
                    episode_context_summary = self._ensure_episode_humor_context_summary(
                        session,
                        video,
                        segments,
                        speaker_map,
                        force=force,
                        progress_video_id=video_id,
                        stage2_total=total_targets,
                    )
                except Exception as e:
                    # Keep per-moment explanations working even if the episode-wide pass
                    # fails due to timeout/context/model issues.
                    log(f"Episode humor context summary skipped for video {video_id}: {e}")

                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="running",
                    stage="moments",
                    message=f"Explaining funny moments (0/{total_targets})...",
                    percent=20,
                    current=0,
                    total=total_targets,
                )

                now = datetime.now()
                for idx, m in enumerate(target_moments, start=1):
                    ctx_start = max(0.0, m.start_time - 75.0)
                    ctx_end = m.end_time + 20.0

                    ctx_segments = [
                        s for s in segments
                        if s.end_time >= ctx_start and s.start_time <= ctx_end
                    ]
                    # Limit prompt size while preserving lead-up context.
                    if len(ctx_segments) > 40:
                        ctx_segments = ctx_segments[-40:]

                    lines = self._build_transcript_context_lines(ctx_segments, speaker_map)

                    if lines:
                        context_text = "\n".join(lines)
                        # Hard cap to keep local LLM prompts bounded.
                        if len(context_text) > 6500:
                            context_text = context_text[-6500:]

                        summary, confidence = self._ollama_generate_humor_summary(
                            context_text,
                            m.start_time,
                            m.end_time,
                            episode_context_summary=episode_context_summary,
                        )
                        m.humor_summary = summary
                        m.humor_confidence = confidence
                        m.humor_model = model_name
                        m.humor_explained_at = now
                        session.add(m)

                    percent = 20 + (idx / max(1, total_targets)) * 80
                    self._set_funny_task_progress(
                        video_id,
                        task="explain",
                        status="running",
                        stage="moments",
                        message=f"Explaining funny moments ({idx}/{total_targets})...",
                        percent=percent,
                        current=idx,
                        total=total_targets,
                    )

                session.commit()
                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="completed",
                    stage="done",
                    message=f"Explained {total_targets} funny moments.",
                    percent=100,
                    current=total_targets,
                    total=total_targets,
                )
                return session.exec(
                    select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                ).all()
        except Exception as e:
            self._set_funny_task_progress(
                video_id,
                task="explain",
                status="error",
                stage="error",
                message=str(e),
                percent=100,
            )
            raise

        












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
        import subprocess
        
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
