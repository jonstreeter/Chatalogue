import yt_dlp
import os
import time
import json
import subprocess
import threading
import pickle
import gc
import sys
import re
import html
import math
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Literal
from contextlib import contextmanager
from fastapi.encoders import jsonable_encoder
from sqlmodel import Session, select, func
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from dotenv import load_dotenv

# NOTE: Heavy ML libraries (torch, faster_whisper, pyannote, numpy, scipy)
# are imported lazily inside _load_models() and related methods to avoid
# blocking the process at startup. Only download/queue operations run
# without them.

from ..db.database import engine, Video, Channel, Speaker, SpeakerEmbedding, TranscriptSegment, TranscriptSegmentRevision, TranscriptRun, TranscriptQualitySnapshot, TranscriptGoldWindow, TranscriptEvaluationResult, TranscriptEvaluationReview, TranscriptOptimizationCampaign, TranscriptOptimizationCampaignItem, Clip, ClipExportArtifact, Job, FunnyMoment, create_db_and_tables
from .logger import log, log_verbose, is_verbose
from . import episode_clone as clone_svc

class JobPausedException(Exception):
    """Raised when a job is paused by the user during processing."""
    pass


class JobCancelledException(Exception):
    """Raised when a job is cancelled by the user during processing."""
    pass


class JobDeferredException(Exception):
    """Raised when a queued job should be deferred and retried later."""
    pass


class JobNoticeException(Exception):
    """Raised when a job should surface a user-facing notice instead of a hard error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "notice",
        video_status: str = "pending",
        technical_detail: str | None = None,
    ):
        super().__init__(message)
        self.notice_message = str(message or "Notice")
        self.notice_code = str(code or "notice")
        self.video_status = str(video_status or "pending")
        self.technical_detail = str(technical_detail or self.notice_message)


class TransformersWhisperCompatModel:
    """Compatibility adapter that presents a faster-whisper-like interface."""

    def __init__(
        self,
        *,
        service,
        pipeline_runner,
        model,
        processor,
        model_ref: str,
        runtime_device: str,
        torch_dtype_label: str,
    ):
        self._service = service
        self._pipeline_runner = pipeline_runner
        self.model = model
        self.processor = processor
        self.model_ref = model_ref
        self.runtime_device = runtime_device
        self.torch_dtype_label = torch_dtype_label

    def transcribe(self, audio_path: str, **kwargs):
        return self._service._run_transformers_whisper_transcribe(
            self._pipeline_runner,
            audio_path,
            kwargs,
        )


def _env_float(name: str, default: str) -> float:
    """Parse a float from an environment variable with a fallback default."""
    try:
        return float((os.getenv(name) or default).strip() or default)
    except (ValueError, TypeError):
        return float(default)


def _truncate_error(error: str | None, max_len: int = 4000) -> str:
    """Truncate an error message to a safe DB storage length."""
    return (error or "Unknown error")[:max_len]


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANUAL_MEDIA_DIR = DATA_DIR / "manual_media"
TEMP_DIR = DATA_DIR / "temp"
PYTHON_TEMP_DIR = TEMP_DIR / "python_runtime"
EXPORT_DIR = DATA_DIR / "exports"
HEARTBEAT_FILE = DATA_DIR / "worker_heartbeat"
RUNTIME_DIR = Path(__file__).parent.parent.parent / "runtime"
CUDA_RESTART_STATE_FILE = RUNTIME_DIR / "cuda_restart_state.json"
CUDA_MAX_AUTO_RESTARTS = int(os.getenv("CUDA_MAX_AUTO_RESTARTS", "3"))
CUDA_RESTART_WINDOW_SECONDS = int(os.getenv("CUDA_RESTART_WINDOW_SECONDS", "600"))

# Load .env from backend root
load_dotenv(Path(__file__).parent.parent.parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
YOUTUBE_DATA_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

def ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    create_db_and_tables()


def configure_python_temp_dir():
    PYTHON_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = str(PYTHON_TEMP_DIR)
    os.environ["TMP"] = temp_path
    os.environ["TEMP"] = temp_path
    os.environ["TMPDIR"] = temp_path
    tempfile.tempdir = temp_path


configure_python_temp_dir()


@contextmanager
def temporary_disabled_blackhole_proxies():
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    removed: dict[str, str] = {}
    try:
        for key in proxy_keys:
            value = str(os.environ.get(key) or "").strip()
            lowered = value.lower()
            if "127.0.0.1:9" in lowered or "localhost:9" in lowered:
                removed[key] = value
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in removed.items():
            os.environ[key] = value


PROCESS_JOB_TYPES = {"process"}
VOICEFIXER_JOB_TYPES = {"voicefixer_cleanup"}
RECONSTRUCTION_JOB_TYPES = {"conversation_reconstruct"}
DIARIZE_JOB_TYPES = {"diarize"}
FUNNY_JOB_TYPES = {"funny_detect", "funny_explain"}
YOUTUBE_JOB_TYPES = {"youtube_metadata", "episode_clone"}
CLIP_JOB_TYPES = {"clip_export_mp4", "clip_export_captions"}
TRANSCRIPT_REPAIR_JOB_TYPES = {"transcript_repair"}

class IngestionService:
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
        bin_dir = str(Path(__file__).parent.parent.parent / "bin")
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

    def _transcript_segment_assignment_key(self, seg: TranscriptSegment):
        speaker_id = getattr(seg, "speaker_id", None)
        if speaker_id is not None:
            return ("speaker", int(speaker_id))
        profile_id = getattr(seg, "matched_profile_id", None)
        if profile_id is not None:
            return ("profile", int(profile_id))
        return None

    def _transcript_segment_word_count(self, seg: TranscriptSegment) -> int:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            return 0
        return len([token for token in text.split() if token])

    def _transcript_segment_duration(self, seg: TranscriptSegment) -> float:
        try:
            return max(0.0, float(seg.end_time) - float(seg.start_time))
        except Exception:
            return 0.0

    def _transcript_segment_gap(self, left_seg: TranscriptSegment, right_seg: TranscriptSegment) -> float:
        try:
            return max(0.0, float(right_seg.start_time) - float(left_seg.end_time))
        except Exception:
            return 0.0

    def _process_memory_snapshot(self) -> dict:
        snap = {"rss": 0, "total": 0, "available": 0}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            vm = psutil.virtual_memory()
            snap["rss"] = int(getattr(process.memory_info(), "rss", 0) or 0)
            snap["total"] = int(getattr(vm, "total", 0) or 0)
            snap["available"] = int(getattr(vm, "available", 0) or 0)
        except Exception:
            pass
        return snap

    def _start_component_memory_profile(self) -> dict:
        return {
            "ram": self._process_memory_snapshot(),
            "cuda": self._cuda_memory_snapshot(),
        }

    def _finish_component_memory_profile(self, component: str, baseline: dict, *, loaded: bool = True):
        component_key = str(component or "").strip().lower()
        if component_key not in self._component_memory_estimates:
            return
        after_ram = self._process_memory_snapshot()
        after_cuda = self._cuda_memory_snapshot()
        before_ram = (baseline or {}).get("ram") or {}
        before_cuda = (baseline or {}).get("cuda") or {}

        ram_delta = max(0, int(after_ram.get("rss") or 0) - int(before_ram.get("rss") or 0))
        allocated_delta = max(0, int(after_cuda.get("allocated") or 0) - int(before_cuda.get("allocated") or 0))
        reserved_delta = max(0, int(after_cuda.get("reserved") or 0) - int(before_cuda.get("reserved") or 0))
        vram_delta = max(allocated_delta, reserved_delta)

        with self._component_memory_guard:
            slot = self._component_memory_estimates.get(component_key) or {}
            if loaded:
                slot["loaded"] = True
                slot["ram_bytes"] = max(int(slot.get("ram_bytes") or 0), ram_delta)
                slot["vram_bytes"] = max(int(slot.get("vram_bytes") or 0), vram_delta)
            else:
                slot["loaded"] = False
                slot["ram_bytes"] = 0
                slot["vram_bytes"] = 0
            self._component_memory_estimates[component_key] = slot

    def _set_component_memory_unloaded(self, component: str):
        component_key = str(component or "").strip().lower()
        if component_key not in self._component_memory_estimates:
            return
        with self._component_memory_guard:
            self._component_memory_estimates[component_key] = {
                "loaded": False,
                "ram_bytes": 0,
                "vram_bytes": 0,
            }

    def _get_component_memory_estimates(self) -> dict:
        with self._component_memory_guard:
            raw = {
                key: dict(value)
                for key, value in self._component_memory_estimates.items()
            }
        return raw

    def _parse_segment_words_json(self, raw_words: str | None) -> list[dict] | None:
        if not raw_words:
            return None
        try:
            data = json.loads(raw_words)
            return data if isinstance(data, list) else None
        except Exception:
            return None

    def _dump_segment_words_json(self, words: list[dict] | None) -> str | None:
        if not words:
            return None
        try:
            return json.dumps(words, ensure_ascii=False)
        except Exception:
            return None

    def _segment_has_strong_terminal_punctuation(self, seg: TranscriptSegment) -> bool:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            return False
        return bool(re.search(r'[.!?]["\')\]]*\s*$', text))

    def _segment_starts_like_continuation(self, seg: TranscriptSegment) -> bool:
        text = str(getattr(seg, "text", "") or "").lstrip()
        if not text:
            return False
        if text[0] in {",", ";", ":", "-", ")", "]", "}"}:
            return True

        normalized = text.lstrip("\"'([{")
        if not normalized:
            return False

        token_match = re.match(r"[A-Za-z0-9][A-Za-z0-9'\-]*", normalized)
        if not token_match:
            return False
        token = token_match.group(0).lower()
        continuation_tokens = {
            "and",
            "but",
            "so",
            "because",
            "then",
            "though",
            "although",
            "however",
            "well",
            "also",
            "plus",
            "except",
            "if",
            "when",
            "while",
            "where",
            "which",
            "that",
            "who",
            "whose",
            "whom",
            "or",
            "nor",
            "yet",
            "still",
            "anyway",
            "anyways",
            "meanwhile",
            "like",
            "because",
        }
        return token in continuation_tokens

    def _merge_transcript_segment_text(self, left_text: str, right_text: str) -> str:
        left = str(left_text or "").rstrip()
        right = str(right_text or "").lstrip()
        if not left:
            return right
        if not right:
            return left
        if left.endswith(("-", "—")):
            return f"{left}{right}"
        return f"{left} {right}".strip()

    def _merge_transcript_segment_pair(self, left_seg: TranscriptSegment, right_seg: TranscriptSegment):
        left_seg.end_time = max(float(left_seg.end_time), float(right_seg.end_time))
        left_seg.text = self._merge_transcript_segment_text(left_seg.text, right_seg.text)
        if getattr(left_seg, "speaker_id", None) is None and getattr(right_seg, "speaker_id", None) is not None:
            left_seg.speaker_id = right_seg.speaker_id
        if getattr(left_seg, "matched_profile_id", None) is None and getattr(right_seg, "matched_profile_id", None) is not None:
            left_seg.matched_profile_id = right_seg.matched_profile_id

        left_words = self._parse_segment_words_json(getattr(left_seg, "words", None))
        right_words = self._parse_segment_words_json(getattr(right_seg, "words", None))
        if left_words is not None and right_words is not None:
            left_seg.words = self._dump_segment_words_json(left_words + right_words)
        elif left_words is None and right_words is None:
            left_seg.words = None
        else:
            left_seg.words = None

    def _consolidate_transcript_segments(self, segments: list[TranscriptSegment]) -> dict:
        ordered = sorted(
            list(segments or []),
            key=lambda s: (float(getattr(s, "start_time", 0.0) or 0.0), int(getattr(s, "id", 0) or 0)),
        )
        if len(ordered) <= 1:
            return {
                "segments": ordered,
                "removed_segments": [],
                "merged_count": 0,
                "reassigned_islands": 0,
                "before_count": len(ordered),
                "after_count": len(ordered),
            }

        turn_merge_gap_seconds = max(
            0.0,
            _env_float(
                "TRANSCRIPT_TURN_MERGE_GAP_SECONDS",
                os.getenv("TRANSCRIPT_CONSOLIDATE_MERGE_GAP_SECONDS") or "1.1",
            ),
        )
        sentence_break_gap_seconds = max(
            0.0,
            _env_float(
                "TRANSCRIPT_TURN_SENTENCE_BREAK_GAP_SECONDS",
                os.getenv("TRANSCRIPT_CONSOLIDATE_SENTENCE_BREAK_GAP_SECONDS") or "0.45",
            ),
        )
        continuation_gap_seconds = max(
            0.0,
            _env_float("TRANSCRIPT_TURN_CONTINUATION_GAP_SECONDS", str(turn_merge_gap_seconds)),
        )
        noncontinuation_gap_seconds = max(
            0.0,
            _env_float("TRANSCRIPT_TURN_NONCONTINUATION_GAP_SECONDS", "0.55"),
        )
        turn_max_words = max(0, int((os.getenv("TRANSCRIPT_TURN_MAX_WORDS") or "80").strip() or "80"))
        turn_max_seconds = max(0.0, _env_float("TRANSCRIPT_TURN_MAX_SECONDS", "30"))
        island_max_words = max(
            0,
            int(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_WORDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_WORDS")
                 or "2").strip() or "2"
            ),
        )
        island_max_seconds = max(
            0.0,
            float(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_SECONDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_SECONDS")
                 or "0.65").strip() or "0.65"
            ),
        )
        island_max_gap_seconds = max(
            0.0,
            float(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_GAP_SECONDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_GAP_SECONDS")
                 or "0.35").strip() or "0.35"
            ),
        )

        reassigned_islands = 0
        for idx in range(1, len(ordered) - 1):
            prev_seg = ordered[idx - 1]
            cur_seg = ordered[idx]
            next_seg = ordered[idx + 1]

            anchor_key = self._transcript_segment_assignment_key(prev_seg)
            if not anchor_key or anchor_key != self._transcript_segment_assignment_key(next_seg):
                continue
            if self._transcript_segment_assignment_key(cur_seg) == anchor_key:
                continue
            if self._transcript_segment_word_count(cur_seg) > island_max_words:
                continue
            if self._transcript_segment_duration(cur_seg) > island_max_seconds:
                continue
            if self._transcript_segment_gap(prev_seg, cur_seg) > island_max_gap_seconds:
                continue
            if self._transcript_segment_gap(cur_seg, next_seg) > island_max_gap_seconds:
                continue

            cur_seg.speaker_id = prev_seg.speaker_id
            cur_seg.matched_profile_id = prev_seg.matched_profile_id or next_seg.matched_profile_id
            reassigned_islands += 1

        merged_count = 0
        removed_segments: list[TranscriptSegment] = []
        survivors: list[TranscriptSegment] = []
        for seg in ordered:
            if not survivors:
                survivors.append(seg)
                continue

            left_seg = survivors[-1]
            left_key = self._transcript_segment_assignment_key(left_seg)
            right_key = self._transcript_segment_assignment_key(seg)
            gap_seconds = self._transcript_segment_gap(left_seg, seg)
            combined_words = self._transcript_segment_word_count(left_seg) + self._transcript_segment_word_count(seg)
            combined_seconds = (
                self._transcript_segment_duration(left_seg)
                + gap_seconds
                + self._transcript_segment_duration(seg)
            )
            left_has_terminal_punctuation = self._segment_has_strong_terminal_punctuation(left_seg)
            right_continues_turn = self._segment_starts_like_continuation(seg)

            can_merge = (
                left_key is not None
                and left_key == right_key
                and gap_seconds <= turn_merge_gap_seconds
            )
            if can_merge and turn_max_words > 0 and combined_words > turn_max_words:
                can_merge = False
            if can_merge and turn_max_seconds > 0.0 and combined_seconds > turn_max_seconds:
                can_merge = False
            if can_merge and not right_continues_turn and gap_seconds > noncontinuation_gap_seconds:
                can_merge = False
            if can_merge and left_has_terminal_punctuation:
                if right_continues_turn:
                    if gap_seconds > continuation_gap_seconds:
                        can_merge = False
                elif gap_seconds > sentence_break_gap_seconds:
                    can_merge = False

            if can_merge:
                self._merge_transcript_segment_pair(left_seg, seg)
                removed_segments.append(seg)
                merged_count += 1
            else:
                survivors.append(seg)

        return {
            "segments": survivors,
            "removed_segments": removed_segments,
            "merged_count": merged_count,
            "reassigned_islands": reassigned_islands,
            "before_count": len(ordered),
            "after_count": len(survivors),
        }

    def consolidate_existing_transcript(self, session: Session, video_id: int, *, save_files: bool = True) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        result = self._consolidate_transcript_segments(segments)
        survivors = result["segments"]
        removed_segments = result["removed_segments"]

        for seg in survivors:
            session.add(seg)
        for seg in removed_segments:
            session.delete(seg)
        session.commit()

        if save_files:
            channel = session.get(Channel, video.channel_id)
            safe_channel = self.sanitize_filename(channel.name if channel else "Unknown")
            safe_title = self.sanitize_filename(video.title)
            out_dir = AUDIO_DIR / safe_channel / safe_title
            out_dir.mkdir(parents=True, exist_ok=True)
            synthetic_audio_path = out_dir / f"{safe_title}.m4a"
            self._save_transcripts(session, video, survivors, synthetic_audio_path)

        return {
            "video_id": int(video_id),
            "title": video.title,
            "before_count": int(result["before_count"]),
            "after_count": int(result["after_count"]),
            "merged_count": int(result["merged_count"]),
            "reassigned_islands": int(result["reassigned_islands"]),
            "changed": bool(result["before_count"] != result["after_count"] or result["reassigned_islands"] > 0),
        }

    def _serialize_transcript_segment_rows(self, segments: list[TranscriptSegment]) -> list[dict]:
        rows: list[dict] = []
        for seg in segments or []:
            rows.append(
                {
                    "id": int(seg.id) if getattr(seg, "id", None) is not None else None,
                    "video_id": int(seg.video_id),
                    "speaker_id": int(seg.speaker_id) if getattr(seg, "speaker_id", None) is not None else None,
                    "matched_profile_id": int(seg.matched_profile_id) if getattr(seg, "matched_profile_id", None) is not None else None,
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "text": str(seg.text or ""),
                    "words": getattr(seg, "words", None),
                }
            )
        return rows

    def _write_transcript_repair_backup_rows(self, video: Video, rows: list[dict]) -> Path:
        backup_dir = DATA_DIR / "transcript_repair_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_title = self.sanitize_filename(str(video.title or f"video_{video.id}"))[:80] or f"video_{video.id}"
        backup_path = backup_dir / f"{int(video.id)}_{safe_title}_{timestamp}.json"
        payload = {
            "video_id": int(video.id),
            "youtube_id": str(video.youtube_id or ""),
            "title": str(video.title or ""),
            "created_at": datetime.now().isoformat(),
            "segments": rows,
        }
        backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return backup_path

    def _write_transcript_repair_backup(self, video: Video, segments: list[TranscriptSegment]) -> Path:
        return self._write_transcript_repair_backup_rows(video, self._serialize_transcript_segment_rows(segments))

    def _entity_repair_source_stopwords(self) -> set[str]:
        return {
            "the", "and", "for", "with", "from", "that", "this", "into", "about", "your", "their", "have",
            "what", "when", "where", "which", "while", "after", "before", "episode", "interview", "podcast",
            "video", "channel", "live", "show", "part", "full", "reaction", "review", "update", "story",
            "discussion", "conversation", "talk", "news", "analysis", "breakdown", "guide", "question", "answer",
            "today", "tonight", "topic", "session", "special", "guest", "host", "official", "edition", "road",
        }

    def _entity_repair_generic_phrase_tail_words(self) -> set[str]:
        return {
            "episode", "interview", "podcast", "video", "reaction", "review", "update", "story", "stories",
            "discussion", "conversation", "talk", "analysis", "breakdown", "guide", "question", "answer",
            "session", "edition", "diagnosis", "debate", "scandal",
        }

    def _normalize_entity_phrase(self, value: str | None) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("&", " and ")
        text = re.sub(r"[^A-Za-z0-9'\- ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _tokenize_entity_phrase(self, value: str | None) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(value or ""))

    def _collect_entity_repair_candidates(self, session: Session, video: Video) -> list[dict]:
        stopwords = self._entity_repair_source_stopwords()
        candidates: dict[str, dict] = {}

        def add_candidate(raw_text: str | None, source: str, priority: float) -> None:
            text = str(raw_text or "").strip()
            if not text:
                return
            tokens = self._tokenize_entity_phrase(text)
            if not tokens:
                return
            normalized = self._normalize_entity_phrase(" ".join(tokens))
            if not normalized:
                return
            alpha_chars = sum(1 for ch in normalized if ch.isalpha())
            if alpha_chars < 4:
                return
            if len(tokens) == 1 and normalized in stopwords:
                return
            if len(tokens) == 1 and len(normalized) < 5:
                return
            if len(tokens) > 4:
                return
            existing = candidates.get(normalized)
            payload = {
                "canonical": " ".join(tokens),
                "normalized": normalized,
                "tokens": tokens,
                "token_count": len(tokens),
                "source": source,
                "priority": float(priority),
            }
            if existing is None or payload["priority"] > float(existing.get("priority") or 0.0):
                candidates[normalized] = payload

        channel = session.get(Channel, video.channel_id) if getattr(video, "channel_id", None) else None
        if channel:
            add_candidate(channel.name, "channel_name", 0.96)
            for token in self._tokenize_entity_phrase(channel.name):
                add_candidate(token, "channel_token", 0.82)

        speakers = session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all()
        for speaker in speakers:
            add_candidate(speaker.name, "speaker_name", 1.0)
            for token in self._tokenize_entity_phrase(speaker.name):
                add_candidate(token, "speaker_token", 0.92)

        title_tokens = self._tokenize_entity_phrase(getattr(video, "title", None))
        for token in title_tokens:
            normalized = self._normalize_entity_phrase(token)
            if normalized in stopwords:
                continue
            if normalized in self._entity_repair_generic_phrase_tail_words():
                continue
            add_candidate(token, "title_token", 0.76)

        # Add consecutive title bigrams/trigrams to capture named entities such as
        # guest names or branded phrases that are not yet in speaker memory.
        for size in (3, 2):
            for idx in range(0, max(0, len(title_tokens) - size + 1)):
                phrase_tokens = title_tokens[idx: idx + size]
                phrase_norm = self._normalize_entity_phrase(" ".join(phrase_tokens))
                if any(tok.lower() in stopwords for tok in phrase_tokens):
                    continue
                if self._normalize_entity_phrase(phrase_tokens[-1]) in self._entity_repair_generic_phrase_tail_words():
                    continue
                if len(phrase_norm.replace(" ", "")) < 8:
                    continue
                add_candidate(" ".join(phrase_tokens), "title_phrase", 0.84)

        return sorted(candidates.values(), key=lambda item: (-float(item["priority"]), -int(item["token_count"]), item["canonical"]))

    def _replace_word_window_with_entity(self, words_payload: list[dict] | None, start_idx: int, end_idx: int, replacement_tokens: list[str]) -> str | None:
        if not words_payload or start_idx < 0 or end_idx >= len(words_payload):
            return None
        if (end_idx - start_idx + 1) != len(replacement_tokens):
            return None
        new_words = list(words_payload)
        for offset, token in enumerate(replacement_tokens):
            current = dict(new_words[start_idx + offset] or {})
            current["word"] = token
            new_words[start_idx + offset] = current
        return self._dump_segment_words_json(new_words)

    def _repair_segment_entities(self, segment: TranscriptSegment, candidates: list[dict]) -> dict:
        original_text = str(getattr(segment, "text", "") or "").strip()
        if not original_text or not candidates:
            return {"changed": False, "applied": []}

        token_matches = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*", original_text))
        if not token_matches:
            return {"changed": False, "applied": []}
        token_texts = [match.group(0) for match in token_matches]
        normalized_tokens = [self._normalize_entity_phrase(token) for token in token_texts]
        words_payload = self._parse_segment_words_json(getattr(segment, "words", None))

        replacements: list[dict] = []
        used_indices: set[int] = set()
        max_len = min(4, len(token_texts))
        for size in range(max_len, 0, -1):
            for start_idx in range(0, len(token_texts) - size + 1):
                window_indices = set(range(start_idx, start_idx + size))
                if used_indices & window_indices:
                    continue
                current_tokens = token_texts[start_idx:start_idx + size]
                current_phrase = " ".join(current_tokens)
                current_normalized = self._normalize_entity_phrase(current_phrase)
                if not current_normalized:
                    continue
                best = None
                for candidate in candidates:
                    if int(candidate.get("token_count") or 0) != size:
                        continue
                    candidate_normalized = str(candidate.get("normalized") or "")
                    if not candidate_normalized:
                        continue
                    similarity = SequenceMatcher(None, current_normalized, candidate_normalized).ratio()
                    exact_normalized = current_normalized == candidate_normalized
                    candidate_source = str(candidate.get("source") or "")
                    required_similarity = 0.88
                    if candidate_source.startswith("title_"):
                        required_similarity = 0.84
                    if not exact_normalized and similarity < required_similarity:
                        continue
                    if abs(len(candidate_normalized) - len(current_normalized)) > 3:
                        continue
                    if current_normalized[0] != candidate_normalized[0]:
                        continue
                    canonical = str(candidate.get("canonical") or "")
                    if exact_normalized and current_phrase == canonical:
                        continue
                    score = similarity + float(candidate.get("priority") or 0.0) * 0.05
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "canonical": canonical,
                            "source": candidate_source,
                            "replacement_tokens": list(candidate.get("tokens") or []),
                            "start_idx": start_idx,
                            "end_idx": start_idx + size - 1,
                            "current_phrase": current_phrase,
                        }
                if best is not None:
                    replacements.append(best)
                    used_indices.update(range(best["start_idx"], best["end_idx"] + 1))

        if not replacements:
            return {"changed": False, "applied": []}

        updated_text = original_text
        for item in sorted(replacements, key=lambda entry: token_matches[entry["start_idx"]].start(), reverse=True):
            start_char = token_matches[item["start_idx"]].start()
            end_char = token_matches[item["end_idx"]].end()
            updated_text = f"{updated_text[:start_char]}{item['canonical']}{updated_text[end_char:]}"
            replaced_words_json = self._replace_word_window_with_entity(
                words_payload,
                item["start_idx"],
                item["end_idx"],
                item["replacement_tokens"],
            )
            if replaced_words_json is not None:
                words_payload = json.loads(replaced_words_json)

        if updated_text == original_text:
            return {"changed": False, "applied": []}

        segment.text = updated_text
        if words_payload is not None:
            segment.words = json.dumps(words_payload, ensure_ascii=False)
        return {"changed": True, "applied": replacements}

    def _apply_entity_repair_to_segments(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        *,
        persist_revisions: bool = False,
        revision_source: str = "entity_repair",
    ) -> dict:
        candidates = self._collect_entity_repair_candidates(session, video)
        if not candidates:
            return {"changed": False, "segments_changed": 0, "replacement_count": 0, "sources": []}

        changed_segments = 0
        replacements = 0
        sources: set[str] = set()
        for seg in segments or []:
            old_text = str(getattr(seg, "text", "") or "")
            result = self._repair_segment_entities(seg, candidates)
            if not result.get("changed"):
                continue
            changed_segments += 1
            replacements += len(result.get("applied") or [])
            for applied in result.get("applied") or []:
                source = str(applied.get("source") or "").strip()
                if source:
                    sources.add(source)
            if persist_revisions and getattr(seg, "id", None) is not None and seg.text != old_text:
                session.add(
                    TranscriptSegmentRevision(
                        segment_id=int(seg.id),
                        video_id=int(seg.video_id),
                        old_text=old_text,
                        new_text=str(seg.text or ""),
                        source=revision_source,
                    )
                )
        return {
            "changed": changed_segments > 0,
            "segments_changed": changed_segments,
            "replacement_count": replacements,
            "sources": sorted(sources),
        }

    def _formatting_question_words(self, language: str | None) -> set[str]:
        normalized = self._normalize_language_code(language)
        if normalized == "es":
            return {
                "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
                "por", "por que", "por qué", "quien", "quién", "cual", "cuál",
            }
        return {
            "who", "what", "when", "where", "why", "how", "which", "did", "does",
            "do", "is", "are", "can", "could", "would", "should", "will",
        }

    def _looks_like_question_segment(self, text: str, language: str | None) -> bool:
        normalized_text = self._normalize_entity_phrase(text)
        if not normalized_text:
            return False
        for phrase in sorted(self._formatting_question_words(language), key=len, reverse=True):
            if normalized_text == phrase or normalized_text.startswith(f"{phrase} "):
                return True
        return False

    def _formatting_fillers(self, language: str | None) -> set[str]:
        normalized = self._normalize_language_code(language)
        if normalized == "es":
            return {"eh", "em", "este", "pues"}
        return {"uh", "um", "ah", "er", "eh"}

    def _cleanup_segment_formatting(self, text: str, *, language: str | None = None, duration: float | None = None) -> tuple[str, list[str]]:
        original = str(text or "")
        if not original.strip():
            return original, []

        cleaned = original
        steps: list[str] = []

        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
        }
        normalized_chars = "".join(replacements.get(ch, ch) for ch in cleaned)
        if normalized_chars != cleaned:
            cleaned = normalized_chars
            steps.append("normalize_punctuation")

        whitespace_cleaned = re.sub(r"\s+", " ", cleaned).strip()
        whitespace_cleaned = re.sub(r"\s+([,.;:!?])", r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r'(["\'])\s+', r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r"\s+([)\]])", r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r"([(\[])\s+", r"\1", whitespace_cleaned)
        if whitespace_cleaned != cleaned:
            cleaned = whitespace_cleaned
            steps.append("normalize_spacing")

        punctuation_cleaned = re.sub(r"\.{4,}", "...", cleaned)
        punctuation_cleaned = re.sub(r"([!?]){2,}", r"\1", punctuation_cleaned)
        punctuation_cleaned = re.sub(r",,{2,}", ",", punctuation_cleaned)
        punctuation_cleaned = re.sub(r";{2,}", ";", punctuation_cleaned)
        if punctuation_cleaned != cleaned:
            cleaned = punctuation_cleaned
            steps.append("collapse_punctuation")

        filler_pattern = "|".join(sorted(re.escape(token) for token in self._formatting_fillers(language)))
        if filler_pattern:
            filler_cleaned = re.sub(rf"\b({filler_pattern})(?:\s+\1\b){{2,}}", r"\1", cleaned, flags=re.IGNORECASE)
            if filler_cleaned != cleaned:
                cleaned = filler_cleaned
                steps.append("collapse_fillers")

        def _capitalize_match(match: re.Match[str]) -> str:
            prefix = match.group(1)
            letter = match.group(2)
            return f"{prefix}{letter.upper()}"

        capitalized = re.sub(r"^([^A-Za-z0-9]*)([a-z])", _capitalize_match, cleaned, count=1)
        if capitalized != cleaned:
            cleaned = capitalized
            steps.append("capitalize_start")

        if cleaned and not re.search(r'[.!?]["\')\]]*$', cleaned):
            token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", cleaned))
            lower_tail = cleaned.rstrip().lower()
            if (
                token_count >= 4
                and float(duration or 0.0) >= 1.0
                and not lower_tail.endswith((",", ";", ":", "-", "--"))
            ):
                suffix = "?" if self._looks_like_question_segment(cleaned, language) else "."
                cleaned = f"{cleaned}{suffix}"
                steps.append("terminal_punctuation")

        normalized_language = self._normalize_language_code(language)
        if normalized_language == "es":
            if cleaned.endswith("?") and not cleaned.startswith("¿") and self._looks_like_question_segment(cleaned[:-1], language):
                cleaned = f"¿{cleaned}"
                steps.append("spanish_inverted_question")
            if cleaned.endswith("!") and not cleaned.startswith("¡"):
                lead = self._normalize_entity_phrase(cleaned[:-1]).split(" ", 1)[0] if cleaned[:-1].strip() else ""
                if lead in {"que", "qué", "como", "cómo", "vaya"}:
                    cleaned = f"¡{cleaned}"
                    steps.append("spanish_inverted_exclamation")

        if cleaned == original:
            return original, []
        return cleaned, steps

    def _apply_formatting_cleanup_to_segments(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        *,
        persist_revisions: bool = False,
        revision_source: str = "formatting_cleanup",
    ) -> dict:
        language = self._normalize_language_code(getattr(video, "transcript_language", None))
        changed_segments = 0
        steps_applied: dict[str, int] = {}
        for seg in segments or []:
            old_text = str(getattr(seg, "text", "") or "")
            new_text, steps = self._cleanup_segment_formatting(
                old_text,
                language=language,
                duration=float(getattr(seg, "end_time", 0.0) or 0.0) - float(getattr(seg, "start_time", 0.0) or 0.0),
            )
            if not steps or new_text == old_text:
                continue
            seg.text = new_text
            changed_segments += 1
            for step in steps:
                steps_applied[step] = int(steps_applied.get(step) or 0) + 1
            if persist_revisions and getattr(seg, "id", None) is not None:
                session.add(
                    TranscriptSegmentRevision(
                        segment_id=int(seg.id),
                        video_id=int(seg.video_id),
                        old_text=old_text,
                        new_text=new_text,
                        source=revision_source,
                    )
                )
        return {
            "changed": changed_segments > 0,
            "segments_changed": changed_segments,
            "steps": steps_applied,
        }

    def repair_existing_transcript(
        self,
        session: Session,
        video_id: int,
        *,
        save_files: bool = True,
        persist_run: bool = True,
        persist_snapshot: bool = True,
        source: str = "manual",
        note: str | None = None,
        trigger_semantic_index: bool = True,
    ) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        if not segments:
            raise ValueError("Transcript not found")
        backup_rows = self._serialize_transcript_segment_rows(segments)

        profile_before = self._detect_transcript_quality_profile(video, segments)
        metrics_before = self._compute_transcript_quality_metrics(video, segments)
        tier_before, reasons_before, score_before, _ = self._recommend_transcript_optimization(profile_before, metrics_before)

        result = self._consolidate_transcript_segments(segments)
        survivors = result["segments"]
        removed_segments = result["removed_segments"]
        entity_repair = self._apply_entity_repair_to_segments(
            session,
            video,
            survivors,
            persist_revisions=True,
            revision_source="entity_repair",
        )
        formatting_cleanup = self._apply_formatting_cleanup_to_segments(
            session,
            video,
            survivors,
            persist_revisions=True,
            revision_source="formatting_cleanup",
        )
        changed = bool(
            result["before_count"] != result["after_count"]
            or result["reassigned_islands"] > 0
            or entity_repair["changed"]
            or formatting_cleanup["changed"]
        )
        backup_path = self._write_transcript_repair_backup_rows(video, backup_rows) if changed else None

        for seg in survivors:
            session.add(seg)
        for seg in removed_segments:
            session.delete(seg)
        session.commit()

        if save_files:
            channel = session.get(Channel, video.channel_id)
            safe_channel = self.sanitize_filename(channel.name if channel else "Unknown")
            safe_title = self.sanitize_filename(video.title)
            out_dir = AUDIO_DIR / safe_channel / safe_title
            out_dir.mkdir(parents=True, exist_ok=True)
            synthetic_audio_path = out_dir / f"{safe_title}.m4a"
            self._save_transcripts(session, video, survivors, synthetic_audio_path)

        profile_after = self._detect_transcript_quality_profile(video, survivors)
        metrics_after = self._compute_transcript_quality_metrics(video, survivors)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        run = None
        snapshot = None
        if persist_run:
            artifact_refs = {
                "backup_file": str(backup_path) if backup_path else None,
                "save_files": bool(save_files),
                "changed": bool(changed),
                "merged_count": int(result["merged_count"]),
                "reassigned_islands": int(result["reassigned_islands"]),
                "entity_segments_changed": int(entity_repair["segments_changed"]),
                "entity_replacement_count": int(entity_repair["replacement_count"]),
                "entity_sources": list(entity_repair["sources"]),
                "formatting_segments_changed": int(formatting_cleanup["segments_changed"]),
                "formatting_steps": dict(formatting_cleanup["steps"]),
            }
            model_provenance = {
                "repair_strategy": "consolidate_segments_entity_repair_formatting_cleanup",
                "source": str(source or "manual"),
            }
            run = self.create_transcript_run(
                session,
                video_id,
                mode="low_risk_repair",
                pipeline_version="transcript-repair-v1",
                quality_profile=profile_after,
                recommended_tier=tier_after,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                artifact_refs=artifact_refs,
                rollback_state=str(backup_path) if backup_path else None,
                model_provenance=model_provenance,
                note=note or "Transcript low-risk repair",
            )
            if persist_snapshot:
                snapshot = TranscriptQualitySnapshot(
                    video_id=int(video_id),
                    run_id=run.id,
                    source=str(source or "manual"),
                    quality_profile=profile_after,
                    recommended_tier=tier_after,
                    score=score_after,
                    metrics_json=json.dumps(metrics_after, ensure_ascii=False),
                    reasons_json=json.dumps(reasons_after, ensure_ascii=False),
                )
                session.add(snapshot)
            session.commit()
            if run:
                session.refresh(run)
            if snapshot:
                session.refresh(snapshot)

        if changed and trigger_semantic_index:
            self._trigger_semantic_index(int(video_id))

        return {
            "video_id": int(video_id),
            "title": video.title,
            "before_count": int(result["before_count"]),
            "after_count": int(result["after_count"]),
            "merged_count": int(result["merged_count"]),
            "reassigned_islands": int(result["reassigned_islands"]),
            "entity_segments_changed": int(entity_repair["segments_changed"]),
            "entity_replacement_count": int(entity_repair["replacement_count"]),
            "entity_sources": list(entity_repair["sources"]),
            "formatting_segments_changed": int(formatting_cleanup["segments_changed"]),
            "formatting_steps": dict(formatting_cleanup["steps"]),
            "changed": bool(changed),
            "backup_file": str(backup_path) if backup_path else None,
            "run_id": int(run.id) if run and run.id is not None else None,
            "snapshot_id": int(snapshot.id) if snapshot and snapshot.id is not None else None,
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def _parse_entities_json(self, raw_entities: str | None) -> list[str]:
        if not raw_entities:
            return []
        try:
            data = json.loads(raw_entities)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        items: list[str] = []
        seen: set[str] = set()
        for value in data:
            text = str(value or "").strip()
            if not text:
                continue
            key = self._normalize_entity_phrase(text)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(text)
        return items

    def _levenshtein_distance(self, left: list[str] | str, right: list[str] | str) -> int:
        left_seq = list(left)
        right_seq = list(right)
        if not left_seq:
            return len(right_seq)
        if not right_seq:
            return len(left_seq)
        prev = list(range(len(right_seq) + 1))
        for i, left_item in enumerate(left_seq, start=1):
            cur = [i]
            for j, right_item in enumerate(right_seq, start=1):
                cost = 0 if left_item == right_item else 1
                cur.append(min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                ))
            prev = cur
        return prev[-1]

    def _normalize_eval_text(self, text: str | None) -> str:
        value = unicodedata.normalize("NFKD", str(text or ""))
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = value.lower()
        value = re.sub(r"[^a-z0-9'\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _normalize_eval_chars(self, text: str | None) -> str:
        return re.sub(r"\s+", "", self._normalize_eval_text(text))

    def _punctuation_density(self, text: str | None) -> float:
        value = str(text or "")
        if not value:
            return 0.0
        punct_count = sum(1 for ch in value if ch in ".,;:!?")
        return punct_count / max(len(value), 1)

    def _collect_transcript_window_excerpt(self, session: Session, video_id: int, start_time: float, end_time: float) -> dict:
        segments = session.exec(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.video_id == video_id,
                TranscriptSegment.end_time > float(start_time),
                TranscriptSegment.start_time < float(end_time),
            )
            .order_by(TranscriptSegment.start_time, TranscriptSegment.id)
        ).all()
        excerpt_parts: list[str] = []
        unknown_count = 0
        for seg in segments:
            text = str(getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            excerpt_parts.append(text)
            if getattr(seg, "speaker_id", None) is None:
                unknown_count += 1
        excerpt = " ".join(excerpt_parts).strip()
        return {
            "text": excerpt,
            "segments": segments,
            "segment_count": len(segments),
            "unknown_speaker_rate": round((unknown_count / max(len(segments), 1)), 4) if segments else 0.0,
        }

    def upsert_transcript_gold_window(
        self,
        session: Session,
        video_id: int,
        *,
        window_id: int | None = None,
        label: str,
        quality_profile: str | None,
        language: str | None,
        start_time: float,
        end_time: float,
        reference_text: str,
        entities: list[str] | None = None,
        notes: str | None = None,
        active: bool = True,
    ) -> TranscriptGoldWindow:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")
        if float(end_time) <= float(start_time):
            raise ValueError("end_time must be greater than start_time")

        window = session.get(TranscriptGoldWindow, window_id) if window_id else None
        if window is not None and int(window.video_id) != int(video_id):
            raise ValueError("Gold window does not belong to this video")
        if window is None:
            window = TranscriptGoldWindow(video_id=int(video_id))
        window.label = str(label or "window").strip() or "window"
        window.quality_profile = str(quality_profile or "").strip() or None
        window.language = self._normalize_language_code(language) or (self._normalize_language_code(getattr(video, "transcript_language", None)) or None)
        window.start_time = float(start_time)
        window.end_time = float(end_time)
        window.reference_text = str(reference_text or "").strip()
        window.entities_json = json.dumps(list(entities or []), ensure_ascii=False)
        window.notes = str(notes or "").strip() or None
        window.active = bool(active)
        window.updated_at = datetime.now()
        session.add(window)
        session.commit()
        session.refresh(window)
        return window

    def _score_transcript_gold_window(
        self,
        session: Session,
        gold_window: TranscriptGoldWindow,
        *,
        run_id: int | None = None,
        source: str = "manual",
        candidate_text_override: str | None = None,
    ) -> TranscriptEvaluationResult:
        reference_text = str(gold_window.reference_text or "").strip()
        excerpt = self._collect_transcript_window_excerpt(
            session,
            int(gold_window.video_id),
            float(gold_window.start_time),
            float(gold_window.end_time),
        )
        candidate_text = str(candidate_text_override if candidate_text_override is not None else excerpt["text"] or "").strip()

        ref_norm = self._normalize_eval_text(reference_text)
        cand_norm = self._normalize_eval_text(candidate_text)
        ref_words = ref_norm.split() if ref_norm else []
        cand_words = cand_norm.split() if cand_norm else []
        wer_distance = self._levenshtein_distance(ref_words, cand_words)
        wer = float(wer_distance / max(len(ref_words), 1)) if ref_words else (0.0 if not cand_words else 1.0)

        ref_chars = list(self._normalize_eval_chars(reference_text))
        cand_chars = list(self._normalize_eval_chars(candidate_text))
        cer_distance = self._levenshtein_distance(ref_chars, cand_chars)
        cer = float(cer_distance / max(len(ref_chars), 1)) if ref_chars else (0.0 if not cand_chars else 1.0)

        entities = self._parse_entities_json(gold_window.entities_json)
        matched_entities = 0
        cand_phrase = self._normalize_entity_phrase(candidate_text)
        for entity in entities:
            entity_norm = self._normalize_entity_phrase(entity)
            if entity_norm and entity_norm in cand_phrase:
                matched_entities += 1
        entity_accuracy = (matched_entities / len(entities)) if entities else None

        punctuation_delta = round(
            abs(self._punctuation_density(candidate_text) - self._punctuation_density(reference_text)),
            4,
        )

        metrics = {
            "window_seconds": round(float(gold_window.end_time) - float(gold_window.start_time), 3),
            "reference_word_count": len(ref_words),
            "candidate_word_count": len(cand_words),
            "reference_char_count": len(ref_chars),
            "candidate_char_count": len(cand_chars),
            "levenshtein_word_distance": wer_distance,
            "levenshtein_char_distance": cer_distance,
            "window_language": self._normalize_language_code(gold_window.language),
        }
        result = TranscriptEvaluationResult(
            gold_window_id=int(gold_window.id),
            video_id=int(gold_window.video_id),
            run_id=run_id,
            source=str(source or "manual"),
            candidate_text=candidate_text,
            reference_text=reference_text,
            wer=round(wer, 4),
            cer=round(cer, 4),
            entity_accuracy=round(float(entity_accuracy), 4) if entity_accuracy is not None else None,
            matched_entity_count=int(matched_entities),
            total_entity_count=len(entities),
            segment_count=int(excerpt["segment_count"]),
            unknown_speaker_rate=float(excerpt["unknown_speaker_rate"] or 0.0),
            punctuation_density_delta=punctuation_delta,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return result

    def evaluate_transcript_gold_windows(
        self,
        session: Session,
        video_id: int,
        *,
        run_id: int | None = None,
        source: str = "manual",
        active_only: bool = True,
    ) -> dict:
        windows_query = select(TranscriptGoldWindow).where(TranscriptGoldWindow.video_id == video_id)
        if active_only:
            windows_query = windows_query.where(TranscriptGoldWindow.active == True)  # noqa: E712
        windows = session.exec(
            windows_query.order_by(TranscriptGoldWindow.start_time, TranscriptGoldWindow.id)
        ).all()
        if not windows:
            raise ValueError("No gold windows defined for this video")

        results: list[TranscriptEvaluationResult] = []
        for window in windows:
            results.append(
                self._score_transcript_gold_window(
                    session,
                    window,
                    run_id=run_id,
                    source=source,
                )
            )

        avg_wer = round(sum(float(item.wer or 0.0) for item in results) / max(len(results), 1), 4)
        avg_cer = round(sum(float(item.cer or 0.0) for item in results) / max(len(results), 1), 4)
        entity_values = [float(item.entity_accuracy) for item in results if item.entity_accuracy is not None]
        avg_entity = round(sum(entity_values) / len(entity_values), 4) if entity_values else None
        avg_unknown = round(sum(float(item.unknown_speaker_rate or 0.0) for item in results) / max(len(results), 1), 4)

        return {
            "video_id": int(video_id),
            "run_id": run_id,
            "total_windows": len(results),
            "average_wer": avg_wer,
            "average_cer": avg_cer,
            "average_entity_accuracy": avg_entity,
            "average_unknown_speaker_rate": avg_unknown,
            "items": results,
        }

    def maybe_evaluate_transcript_run_against_gold_windows(
        self,
        session: Session,
        video_id: int,
        *,
        run_id: int | None = None,
        source: str,
    ) -> dict | None:
        existing = session.exec(
            select(TranscriptGoldWindow.id)
            .where(TranscriptGoldWindow.video_id == video_id, TranscriptGoldWindow.active == True)  # noqa: E712
            .limit(1)
        ).first()
        if not existing:
            return None
        return self.evaluate_transcript_gold_windows(session, video_id, run_id=run_id, source=source, active_only=True)

    def _loads_json_object(self, raw: str | None, fallback):
        if not raw:
            return fallback
        try:
            parsed = json.loads(raw)
        except Exception:
            return fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback

    def list_transcript_rollback_options(self, session: Session, video_id: int) -> list[dict]:
        runs = session.exec(
            select(TranscriptRun)
            .where(TranscriptRun.video_id == int(video_id))
            .where(TranscriptRun.rollback_state.is_not(None))
            .order_by(TranscriptRun.created_at.desc(), TranscriptRun.id.desc())
        ).all()
        options: list[dict] = []
        for run in runs:
            rollback_state = str(run.rollback_state or "").strip() or None
            rollback_available = False
            if rollback_state:
                try:
                    rollback_available = Path(rollback_state).exists()
                except Exception:
                    rollback_available = False
            options.append(
                {
                    "run_id": int(run.id),
                    "video_id": int(run.video_id),
                    "mode": str(run.mode or "unknown"),
                    "pipeline_version": str(run.pipeline_version or ""),
                    "note": run.note,
                    "created_at": run.created_at,
                    "rollback_available": rollback_available,
                    "rollback_state": rollback_state,
                }
            )
        return options

    def restore_transcript_from_run(
        self,
        session: Session,
        video_id: int,
        run_id: int,
        *,
        source: str = "api",
    ) -> dict:
        video = session.get(Video, int(video_id))
        if not video:
            raise ValueError("Video not found")
        run = session.get(TranscriptRun, int(run_id))
        if not run or int(run.video_id) != int(video_id):
            raise ValueError("Transcript run not found")
        backup_path_raw = str(run.rollback_state or "").strip()
        if not backup_path_raw:
            raise ValueError("Selected run does not have rollback data")
        backup_path = Path(backup_path_raw)
        if not backup_path.exists():
            raise ValueError("Rollback backup file is missing")

        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise ValueError(f"Rollback backup could not be read: {exc}") from exc

        current_segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time, TranscriptSegment.id)
        ).all()
        current_backup_rows = [
            {
                "speaker_id": seg.speaker_id,
                "matched_profile_id": seg.matched_profile_id,
                "start_time": float(seg.start_time or 0.0),
                "end_time": float(seg.end_time or 0.0),
                "text": str(seg.text or ""),
                "words": seg.words,
            }
            for seg in current_segments
        ]
        current_funny_rows = session.exec(
            select(FunnyMoment).where(FunnyMoment.video_id == int(video_id)).order_by(FunnyMoment.start_time, FunnyMoment.id)
        ).all()
        current_backup_payload = {
            "video_id": int(video_id),
            "video_status": video.status,
            "video_processed": bool(video.processed),
            "saved_at": datetime.now().isoformat(),
            "segments": current_backup_rows,
            "funny_moments": [
                {
                    "start_time": float(row.start_time or 0.0),
                    "end_time": float(row.end_time or 0.0),
                    "score": float(row.score or 0.0),
                    "source": str(row.source or "heuristic"),
                    "snippet": row.snippet,
                    "humor_summary": row.humor_summary,
                    "humor_confidence": row.humor_confidence,
                    "humor_model": row.humor_model,
                    "humor_explained_at": row.humor_explained_at.isoformat() if row.humor_explained_at else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in current_funny_rows
            ],
        }
        current_backup_path = self._get_temp_redo_backup_path(int(video_id))
        current_backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(current_backup_path, "w", encoding="utf-8") as f:
            json.dump(current_backup_payload, f, ensure_ascii=False)

        from sqlalchemy import delete as sa_delete

        session.exec(sa_delete(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id == int(video_id)))
        session.exec(sa_delete(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)))
        session.exec(sa_delete(FunnyMoment).where(FunnyMoment.video_id == int(video_id)))
        session.flush()

        def _parse_dt(value):
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except Exception:
                return None

        restored_segment_count = 0
        for row in payload.get("segments", []) or []:
            session.add(
                TranscriptSegment(
                    video_id=int(video_id),
                    speaker_id=row.get("speaker_id"),
                    matched_profile_id=row.get("matched_profile_id"),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    text=str(row.get("text") or ""),
                    words=row.get("words"),
                )
            )
            restored_segment_count += 1

        restored_funny_count = 0
        for row in payload.get("funny_moments", []) or []:
            session.add(
                FunnyMoment(
                    video_id=int(video_id),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    score=float(row.get("score") or 0.0),
                    source=str(row.get("source") or "heuristic"),
                    snippet=row.get("snippet"),
                    humor_summary=row.get("humor_summary"),
                    humor_confidence=row.get("humor_confidence"),
                    humor_model=row.get("humor_model"),
                    humor_explained_at=_parse_dt(row.get("humor_explained_at")),
                    created_at=_parse_dt(row.get("created_at")) or datetime.now(),
                )
            )
            restored_funny_count += 1

        session.flush()

        backup_metrics = self._loads_json_object(run.metrics_before_json, {})
        profile = self._detect_transcript_quality_profile(
            video,
            session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time)
            ).all(),
        )
        metrics = self._compute_transcript_quality_metrics(
            video,
            session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time)
            ).all(),
        )
        recommended_tier, reasons, quality_score, _ = self._recommend_transcript_optimization(profile, metrics)

        video.status = str(payload.get("video_status") or "completed")
        video.processed = bool(payload.get("video_processed")) if payload.get("video_processed") is not None else restored_segment_count > 0
        video.transcript_is_placeholder = False
        if restored_segment_count > 0 and not str(video.transcript_source or "").strip():
            video.transcript_source = "local_transcription"
        session.add(video)

        restore_run = self.create_transcript_run(
            session,
            int(video_id),
            mode="rollback_restore",
            pipeline_version="rollback-restore-v1",
            quality_profile=profile,
            recommended_tier=recommended_tier,
            metrics_before=backup_metrics if isinstance(backup_metrics, dict) else None,
            metrics_after=metrics,
            artifact_refs={
                "source_run_id": int(run.id),
                "source_backup_file": str(backup_path),
                "restore_source": str(source or "api"),
                "previous_transcript_backup": str(current_backup_path),
                "previous_funny_moment_count": len(current_funny_rows),
            },
            rollback_state=str(current_backup_path),
            model_provenance={"strategy": "rollback_restore", "source": str(source or "api")},
            note=f"Restored transcript from run {int(run.id)}",
            input_run_id=int(run.id),
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=restore_run.id,
            source="rollback_restore",
            quality_profile=profile,
            recommended_tier=recommended_tier,
            score=quality_score,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
            reasons_json=json.dumps(reasons, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(restore_run)

        try:
            self.save_transcript_files(video, session)
        except Exception as exc:
            log(f"Transcript restore save_files failed for video {video_id}: {exc}")
        try:
            self.reindex_video_semantic_embeddings(int(video_id))
        except Exception as exc:
            log(f"Transcript restore semantic reindex failed for video {video_id}: {exc}")

        return {
            "video_id": int(video_id),
            "restored_from_run_id": int(run.id),
            "restore_run_id": int(restore_run.id),
            "segment_count": restored_segment_count,
            "funny_moment_count": restored_funny_count,
            "quality_profile": profile,
            "recommended_tier": recommended_tier,
            "quality_score": quality_score,
        }

    def summarize_transcript_evaluation(self, session: Session, *, channel_id: int | None = None) -> dict:
        query = select(TranscriptEvaluationResult)
        if channel_id is not None:
            query = query.join(Video, Video.id == TranscriptEvaluationResult.video_id).where(Video.channel_id == int(channel_id))
        results = session.exec(query.order_by(TranscriptEvaluationResult.created_at.desc(), TranscriptEvaluationResult.id.desc())).all()

        windows_query = select(TranscriptGoldWindow)
        if channel_id is not None:
            windows_query = windows_query.join(Video, Video.id == TranscriptGoldWindow.video_id).where(Video.channel_id == int(channel_id))
        windows = session.exec(windows_query).all()

        result_ids = [int(item.id) for item in results if item.id is not None]
        reviews: list[TranscriptEvaluationReview] = []
        if result_ids:
            reviews = session.exec(
                select(TranscriptEvaluationReview).where(TranscriptEvaluationReview.evaluation_result_id.in_(result_ids))
            ).all()

        verdict_counts: dict[str, int] = {}
        for review in reviews:
            key = str(review.verdict or "same")
            verdict_counts[key] = verdict_counts.get(key, 0) + 1

        def _avg(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        entity_values = [float(item.entity_accuracy) for item in results if item.entity_accuracy is not None]
        return {
            "scope": "channel" if channel_id is not None else "global",
            "channel_id": int(channel_id) if channel_id is not None else None,
            "total_gold_windows": len(windows),
            "total_results": len(results),
            "total_reviewed_results": len({int(review.evaluation_result_id) for review in reviews}),
            "average_wer": _avg([float(item.wer or 0.0) for item in results]),
            "average_cer": _avg([float(item.cer or 0.0) for item in results]),
            "average_entity_accuracy": _avg(entity_values),
            "average_unknown_speaker_rate": _avg([float(item.unknown_speaker_rate or 0.0) for item in results]),
            "verdict_counts": verdict_counts,
            "latest_result_at": results[0].created_at if results else None,
        }

    def summarize_diarization_benchmarks(self, session: Session, *, channel_id: int | None = None) -> list[dict]:
        query = (
            select(TranscriptRun, TranscriptEvaluationResult)
            .join(TranscriptEvaluationResult, TranscriptEvaluationResult.run_id == TranscriptRun.id)
            .where(TranscriptRun.mode == "diarization_rebuild")
        )
        if channel_id is not None:
            query = query.join(Video, Video.id == TranscriptRun.video_id).where(Video.channel_id == int(channel_id))
        rows = session.exec(query).all()

        grouped: dict[str, dict] = {}
        for run, result in rows:
            provenance = self._loads_json_object(run.model_provenance_json, {})
            sensitivity = str(provenance.get("diarization_sensitivity") or "balanced")
            threshold_raw = provenance.get("speaker_match_threshold")
            try:
                threshold_value = round(float(threshold_raw), 2)
            except Exception:
                threshold_value = None
            label = f"{sensitivity} / {threshold_value if threshold_value is not None else 'default'}"
            bucket = grouped.setdefault(
                label,
                {
                    "label": label,
                    "run_ids": set(),
                    "wers": [],
                    "cers": [],
                    "unknown_rates": [],
                    "latest_run_id": None,
                    "latest_created_at": None,
                    "diarization_sensitivity": sensitivity,
                    "speaker_match_threshold": threshold_value,
                },
            )
            bucket["run_ids"].add(int(run.id))
            bucket["wers"].append(float(result.wer or 0.0))
            bucket["cers"].append(float(result.cer or 0.0))
            bucket["unknown_rates"].append(float(result.unknown_speaker_rate or 0.0))
            if bucket["latest_created_at"] is None or (run.created_at and run.created_at > bucket["latest_created_at"]):
                bucket["latest_created_at"] = run.created_at
                bucket["latest_run_id"] = int(run.id)

        summaries: list[dict] = []
        for bucket in grouped.values():
            summaries.append(
                {
                    "label": bucket["label"],
                    "run_count": len(bucket["run_ids"]),
                    "average_wer": round(sum(bucket["wers"]) / len(bucket["wers"]), 4) if bucket["wers"] else None,
                    "average_cer": round(sum(bucket["cers"]) / len(bucket["cers"]), 4) if bucket["cers"] else None,
                    "average_unknown_speaker_rate": round(sum(bucket["unknown_rates"]) / len(bucket["unknown_rates"]), 4) if bucket["unknown_rates"] else None,
                    "latest_run_id": bucket["latest_run_id"],
                    "latest_created_at": bucket["latest_created_at"],
                    "diarization_sensitivity": bucket["diarization_sensitivity"],
                    "speaker_match_threshold": bucket["speaker_match_threshold"],
                }
            )
        summaries.sort(key=lambda item: (item.get("average_wer") is None, item.get("average_wer") or 9999.0, item.get("average_cer") or 9999.0))
        return summaries

    def create_transcript_optimization_campaign(
        self,
        session: Session,
        *,
        channel_id: int | None = None,
        limit: int = 100,
        tiers: list[str] | None = None,
        force_non_eligible: bool = False,
        note: str | None = None,
    ) -> TranscriptOptimizationCampaign:
        requested_tiers = [str(item or "").strip() for item in (tiers or []) if str(item or "").strip()]
        allowed_tiers = {"low_risk_repair", "diarization_rebuild", "full_retranscription", "manual_review"}
        filtered_tiers = [item for item in requested_tiers if item in allowed_tiers]
        dry_run = self.transcript_optimization_dry_run(
            session,
            channel_id=channel_id,
            limit=limit,
            persist_snapshots=False,
        )
        campaign = TranscriptOptimizationCampaign(
            channel_id=int(channel_id) if channel_id is not None else None,
            scope="channel" if channel_id is not None else "global",
            status="draft",
            tiers_json=json.dumps(filtered_tiers, ensure_ascii=False),
            limit=max(1, int(limit)),
            force_non_eligible=bool(force_non_eligible),
            note=str(note or "").strip() or None,
            updated_at=datetime.now(),
        )
        session.add(campaign)
        session.flush()

        for item in dry_run.get("items", []):
            metrics = item.get("metrics") or {}
            if int(metrics.get("total_segments") or 0) <= 0:
                continue
            recommended_tier = str(item.get("recommended_tier") or "none")
            if filtered_tiers and recommended_tier not in filtered_tiers:
                continue
            reason_list = item.get("reasons") or []
            session.add(
                TranscriptOptimizationCampaignItem(
                    campaign_id=int(campaign.id),
                    video_id=int(item["video_id"]),
                    recommended_tier=recommended_tier,
                    action_tier=recommended_tier,
                    quality_score=float(item.get("quality_score") or 0.0),
                    reason=str(reason_list[0]) if reason_list else None,
                    status="pending",
                )
            )
        session.commit()
        session.refresh(campaign)
        return campaign

    def _detect_transcript_quality_profile(self, video: Video, segments: list[TranscriptSegment]) -> str:
        language = str(getattr(video, "transcript_language", "") or "").strip().lower()
        title = str(getattr(video, "title", "") or "").strip().lower()
        description = str(getattr(video, "description", "") or "").strip().lower()
        haystack = f"{language} {title} {description}"
        multilingual_markers = ("spanish", "espanol", "español", "bilingual", "portuguese", "french")
        if language.startswith("es") or any(marker in haystack for marker in multilingual_markers):
            return "multilingual_or_non_english"
        if len(segments) >= 120:
            return "english_longform"
        return "english_general"

    def _compute_transcript_quality_metrics(self, video: Video, segments: list[TranscriptSegment]) -> dict:
        ordered = sorted(
            list(segments or []),
            key=lambda s: (float(getattr(s, "start_time", 0.0) or 0.0), int(getattr(s, "id", 0) or 0)),
        )
        total_segments = len(ordered)
        if total_segments <= 0:
            return {
                "total_segments": 0,
                "unknown_speaker_segments": 0,
                "unknown_speaker_rate": 0.0,
                "micro_segment_count": 0,
                "micro_segment_rate": 0.0,
                "tiny_unknown_count": 0,
                "tiny_unknown_rate": 0.0,
                "same_speaker_interruptions": 0,
                "same_speaker_interruption_rate": 0.0,
                "punctuated_segment_rate": 0.0,
                "avg_segment_seconds": 0.0,
                "avg_words_per_segment": 0.0,
                "word_timed_segment_rate": 0.0,
                "distinct_assigned_speakers": 0,
                "assigned_speaker_segments": 0,
                "language": getattr(video, "transcript_language", None),
            }

        durations = [self._transcript_segment_duration(seg) for seg in ordered]
        word_counts = [self._transcript_segment_word_count(seg) for seg in ordered]
        unknown_segments = [
            seg for seg in ordered
            if getattr(seg, "speaker_id", None) is None and getattr(seg, "matched_profile_id", None) is None
        ]
        micro_segments = [seg for seg, duration in zip(ordered, durations) if duration > 0.0 and duration < 1.5]
        tiny_unknown_segments = [
            seg for seg in unknown_segments
            if self._transcript_segment_duration(seg) > 0.0 and self._transcript_segment_duration(seg) < 1.5
        ]
        punctuated_segments = [seg for seg in ordered if self._segment_has_strong_terminal_punctuation(seg)]
        word_timed_segments = [
            seg for seg in ordered
            if isinstance(self._parse_segment_words_json(getattr(seg, "words", None)), list)
        ]
        assigned_speaker_ids = {
            int(seg.speaker_id)
            for seg in ordered
            if getattr(seg, "speaker_id", None) is not None
        }

        same_speaker_interruptions = 0
        for idx in range(1, len(ordered) - 1):
            prev_seg = ordered[idx - 1]
            cur_seg = ordered[idx]
            next_seg = ordered[idx + 1]
            prev_key = self._transcript_segment_assignment_key(prev_seg)
            cur_key = self._transcript_segment_assignment_key(cur_seg)
            next_key = self._transcript_segment_assignment_key(next_seg)
            if not prev_key or prev_key != next_key or cur_key == prev_key:
                continue
            if self._transcript_segment_duration(cur_seg) > 2.0:
                continue
            if self._transcript_segment_word_count(cur_seg) > 8:
                continue
            same_speaker_interruptions += 1

        return {
            "total_segments": total_segments,
            "unknown_speaker_segments": len(unknown_segments),
            "unknown_speaker_rate": round(len(unknown_segments) / total_segments, 4),
            "micro_segment_count": len(micro_segments),
            "micro_segment_rate": round(len(micro_segments) / total_segments, 4),
            "tiny_unknown_count": len(tiny_unknown_segments),
            "tiny_unknown_rate": round(len(tiny_unknown_segments) / total_segments, 4),
            "same_speaker_interruptions": same_speaker_interruptions,
            "same_speaker_interruption_rate": round(same_speaker_interruptions / total_segments, 4),
            "punctuated_segment_rate": round(len(punctuated_segments) / total_segments, 4),
            "avg_segment_seconds": round(sum(durations) / total_segments, 3),
            "avg_words_per_segment": round(sum(word_counts) / total_segments, 3),
            "word_timed_segment_rate": round(len(word_timed_segments) / total_segments, 4),
            "distinct_assigned_speakers": len(assigned_speaker_ids),
            "assigned_speaker_segments": total_segments - len(unknown_segments),
            "language": getattr(video, "transcript_language", None),
        }

    def _recommend_transcript_optimization(self, profile: str, metrics: dict) -> tuple[str, list[str], float, bool]:
        total_segments = max(1, int(metrics.get("total_segments") or 0))
        unknown_rate = float(metrics.get("unknown_speaker_rate") or 0.0)
        micro_rate = float(metrics.get("micro_segment_rate") or 0.0)
        tiny_unknown_rate = float(metrics.get("tiny_unknown_rate") or 0.0)
        interruption_rate = float(metrics.get("same_speaker_interruption_rate") or 0.0)
        punctuated_rate = float(metrics.get("punctuated_segment_rate") or 0.0)
        word_timed_rate = float(metrics.get("word_timed_segment_rate") or 0.0)

        reasons: list[str] = []
        tier = "none"

        if total_segments < 5:
            reasons.append("Transcript is too small to evaluate reliably.")
            score = 100.0
            return tier, reasons, score, False

        if profile == "multilingual_or_non_english":
            if punctuated_rate < 0.9 or unknown_rate > 0.08 or micro_rate > 0.18:
                tier = "full_retranscription"
                reasons.append("Language profile suggests multilingual or non-English handling should be re-routed.")

        if tier == "none" and word_timed_rate >= 0.4 and unknown_rate >= 0.12:
            tier = "diarization_rebuild"
            reasons.append("High unknown-speaker rate with usable word timing suggests diarization can be rebuilt.")

        if tier == "none" and (interruption_rate >= 0.04 or tiny_unknown_rate >= 0.025 or micro_rate >= 0.22):
            tier = "low_risk_repair"
            reasons.append("Short interruptions and micro-segmentation indicate consolidation repair should help.")

        if tier == "none" and punctuated_rate < 0.75:
            tier = "manual_review"
            reasons.append("Formatting quality is weak without a clear automatic repair path.")

        score = 100.0
        score -= unknown_rate * 180.0
        score -= micro_rate * 70.0
        score -= interruption_rate * 120.0
        score -= max(0.0, 0.82 - punctuated_rate) * 55.0
        if profile == "multilingual_or_non_english":
            score -= 6.0
        score = max(0.0, min(100.0, round(score, 2)))

        if not reasons and tier == "none":
            reasons.append("Transcript quality is within the current automatic optimization thresholds.")
        return tier, reasons, score, tier != "none"

    def create_transcript_run(
        self,
        session: Session,
        video_id: int,
        *,
        mode: str,
        pipeline_version: str,
        status: str = "completed",
        quality_profile: str | None = None,
        recommended_tier: str | None = None,
        metrics_before: dict | None = None,
        metrics_after: dict | None = None,
        artifact_refs: dict | None = None,
        rollback_state: str | None = None,
        model_provenance: dict | None = None,
        note: str | None = None,
        input_run_id: int | None = None,
    ) -> TranscriptRun:
        run = TranscriptRun(
            video_id=int(video_id),
            input_run_id=input_run_id,
            mode=str(mode or "baseline"),
            pipeline_version=str(pipeline_version or "baseline-v1"),
            status=str(status or "completed"),
            quality_profile=quality_profile,
            recommended_tier=recommended_tier,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            metrics_before_json=json.dumps(metrics_before or {}, ensure_ascii=False),
            metrics_after_json=json.dumps(metrics_after or {}, ensure_ascii=False) if metrics_after is not None else None,
            artifact_refs_json=json.dumps(artifact_refs or {}, ensure_ascii=False) if artifact_refs is not None else None,
            rollback_state=rollback_state,
            model_provenance_json=json.dumps(model_provenance or {}, ensure_ascii=False) if model_provenance is not None else None,
            note=note,
        )
        session.add(run)
        session.flush()
        return run

    def evaluate_transcript_quality(
        self,
        session: Session,
        video_id: int,
        *,
        source: str = "manual",
        persist_snapshot: bool = False,
    ) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile = self._detect_transcript_quality_profile(video, segments)
        metrics = self._compute_transcript_quality_metrics(video, segments)
        recommended_tier, reasons, score, eligible = self._recommend_transcript_optimization(profile, metrics)

        snapshot = None
        if persist_snapshot:
            run = self.create_transcript_run(
                session,
                video_id,
                mode="quality_assessment",
                pipeline_version="quality-evaluator-v1",
                quality_profile=profile,
                recommended_tier=recommended_tier,
                metrics_before=metrics,
                note=f"Quality assessment snapshot via {source}",
            )
            snapshot = TranscriptQualitySnapshot(
                video_id=int(video_id),
                run_id=run.id,
                source=str(source or "manual"),
                quality_profile=profile,
                recommended_tier=recommended_tier,
                score=score,
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                reasons_json=json.dumps(reasons, ensure_ascii=False),
            )
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
        result = {
            "video_id": int(video_id),
            "title": str(video.title or ""),
            "channel_id": video.channel_id,
            "quality_profile": profile,
            "recommended_tier": recommended_tier,
            "quality_score": score,
            "eligible_for_optimization": bool(eligible),
            "language": getattr(video, "transcript_language", None),
            "metrics": metrics,
            "reasons": reasons,
            "created_snapshot_id": int(snapshot.id) if snapshot and snapshot.id is not None else None,
            "snapshot_created_at": snapshot.created_at if snapshot else None,
        }
        return result

    def transcript_optimization_dry_run(
        self,
        session: Session,
        *,
        channel_id: int | None = None,
        video_id: int | None = None,
        limit: int = 50,
        persist_snapshots: bool = False,
    ) -> dict:
        query = select(Video).where(Video.processed == True)  # noqa: E712
        if video_id is not None:
            query = query.where(Video.id == int(video_id))
        elif channel_id is not None:
            query = query.where(Video.channel_id == int(channel_id))
        query = query.order_by(Video.published_at.desc(), Video.id.desc()).limit(max(1, int(limit)))

        items: list[dict] = []
        for video in session.exec(query).all():
            item = self.evaluate_transcript_quality(
                session,
                int(video.id),
                source="dry_run",
                persist_snapshot=bool(persist_snapshots),
            )
            if int((item.get("metrics") or {}).get("total_segments") or 0) <= 0:
                continue
            items.append(item)

        eligible_count = len([item for item in items if item.get("eligible_for_optimization")])
        return {
            "total_scanned": len(items),
            "total_eligible": eligible_count,
            "items": items,
        }

    def create_diarization_rebuild_run(
        self,
        session: Session,
        video_id: int,
        *,
        payload: dict | None = None,
        note: str | None = None,
    ) -> dict | None:
        payload = dict(payload or {})
        optimization_target = str(payload.get("optimization_target") or "").strip().lower()
        mode = str(payload.get("mode") or "").strip().lower()
        if optimization_target != "diarization_rebuild" and mode != "redo_diarization":
            return None

        video = session.get(Video, video_id)
        if not video:
            return None
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile_after = self._detect_transcript_quality_profile(video, segments)
        metrics_after = self._compute_transcript_quality_metrics(video, segments)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        metrics_before = payload.get("quality_metrics_before")
        if not isinstance(metrics_before, dict):
            metrics_before = {}
        profile_before = str(payload.get("quality_profile_before") or "unknown")
        tier_before = str(payload.get("recommended_tier_before") or "none")
        score_before = float(payload.get("quality_score_before") or 0.0)
        reasons_before = payload.get("quality_reasons_before")
        if not isinstance(reasons_before, list):
            reasons_before = []

        artifact_refs = {
            "redo_backup_file": payload.get("redo_diarization_backup_file"),
            "optimization_target": optimization_target or "redo_diarization",
            "queued_from": payload.get("queued_from"),
            "raw_transcript_reused": True,
        }
        diarization_sensitivity = str(
            payload.get("diarization_sensitivity_override")
            or os.getenv("DIARIZATION_SENSITIVITY", "balanced")
            or "balanced"
        )
        try:
            speaker_match_threshold = float(
                payload.get("speaker_match_threshold_override")
                if payload.get("speaker_match_threshold_override") is not None
                else os.getenv("SPEAKER_MATCH_THRESHOLD", "0.35")
            )
        except Exception:
            speaker_match_threshold = 0.35
        model_provenance = {
            "strategy": "redo_diarization",
            "diarization_sensitivity": diarization_sensitivity,
            "speaker_match_threshold": speaker_match_threshold,
            "benchmark_variant": bool(payload.get("benchmark_variant")),
            "source": str(payload.get("queued_from") or "manual"),
        }
        run = self.create_transcript_run(
            session,
            video_id,
            mode="diarization_rebuild",
            pipeline_version="diarization-benchmark-v1" if bool(payload.get("benchmark_variant")) else "diarization-rebuild-v1",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            artifact_refs=artifact_refs,
            rollback_state=str(payload.get("redo_diarization_backup_file") or "") or None,
            model_provenance=model_provenance,
            note=note or str(payload.get("note") or "").strip() or "Transcript diarization rebuild",
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=run.id,
            source="queued_job",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            score=score_after,
            metrics_json=json.dumps(metrics_after, ensure_ascii=False),
            reasons_json=json.dumps(reasons_after, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(run)
        session.refresh(snapshot)
        return {
            "run_id": int(run.id),
            "snapshot_id": int(snapshot.id),
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def create_full_retranscription_run(
        self,
        session: Session,
        video_id: int,
        *,
        payload: dict | None = None,
        note: str | None = None,
    ) -> dict | None:
        payload = dict(payload or {})
        optimization_target = str(payload.get("optimization_target") or "").strip().lower()
        mode = str(payload.get("mode") or "").strip().lower()
        if optimization_target != "full_retranscription" and mode != "full_retranscription":
            return None

        video = session.get(Video, video_id)
        if not video:
            return None
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile_after = self._detect_transcript_quality_profile(video, segments)
        metrics_after = self._compute_transcript_quality_metrics(video, segments)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        metrics_before = payload.get("quality_metrics_before")
        if not isinstance(metrics_before, dict):
            metrics_before = {}
        profile_before = str(payload.get("quality_profile_before") or "unknown")
        tier_before = str(payload.get("recommended_tier_before") or "none")
        score_before = float(payload.get("quality_score_before") or 0.0)
        reasons_before = payload.get("quality_reasons_before")
        if not isinstance(reasons_before, list):
            reasons_before = []

        artifact_refs = {
            "redo_backup_file": payload.get("redo_diarization_backup_file"),
            "optimization_target": optimization_target or "full_retranscription",
            "queued_from": payload.get("queued_from"),
            "raw_transcript_reused": False,
            "force_retranscription": bool(payload.get("force_retranscription")),
        }
        model_provenance = {
            "strategy": "full_retranscription",
            "transcription_engine_requested": payload.get("transcription_engine_requested"),
            "transcription_engine_routed": payload.get("transcription_engine_routed"),
            "transcription_engine_used": payload.get("transcription_engine_used"),
            "transcription_route_language": payload.get("transcription_route_language"),
            "transcription_route_language_source": payload.get("transcription_route_language_source"),
            "transcription_route_multilingual": payload.get("transcription_route_multilingual"),
            "transcription_route_whisper_model": payload.get("transcription_route_whisper_model"),
            "source": str(payload.get("queued_from") or "manual"),
        }
        run = self.create_transcript_run(
            session,
            video_id,
            mode="full_retranscription",
            pipeline_version="full-retranscription-v1",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            artifact_refs=artifact_refs,
            rollback_state=str(payload.get("redo_diarization_backup_file") or "") or None,
            model_provenance=model_provenance,
            note=note or str(payload.get("note") or "").strip() or "Transcript full retranscription",
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=run.id,
            source="queued_job",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            score=score_after,
            metrics_json=json.dumps(metrics_after, ensure_ascii=False),
            reasons_json=json.dumps(reasons_after, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(run)
        session.refresh(snapshot)
        return {
            "run_id": int(run.id),
            "snapshot_id": int(snapshot.id),
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def _record_transcript_optimization_completion(self, job_id: int, video_id: int, payload: dict | None):
        payload = dict(payload or {})
        result = None
        evaluation = None
        with Session(engine) as session:
            result = (
                self.create_diarization_rebuild_run(session, video_id, payload=payload)
                or self.create_full_retranscription_run(session, video_id, payload=payload)
            )
            if result and result.get("run_id"):
                try:
                    evaluation = self.maybe_evaluate_transcript_run_against_gold_windows(
                        session,
                        video_id,
                        run_id=int(result["run_id"]),
                        source="optimization_completion",
                    )
                except Exception as e:
                    log(f"Transcript evaluation-on-completion failed for video {video_id}: {e}")
        if not result:
            return
        payload_fields = {
            "optimization_run_id": result.get("run_id"),
            "optimization_snapshot_id": result.get("snapshot_id"),
            "optimization_recommended_tier_after": result.get("recommended_tier_after"),
            "optimization_quality_score_after": result.get("quality_score_after"),
            "rebuild_run_id": result.get("run_id"),
            "rebuild_snapshot_id": result.get("snapshot_id"),
            "rebuild_recommended_tier_after": result.get("recommended_tier_after"),
            "rebuild_quality_score_after": result.get("quality_score_after"),
        }
        if evaluation:
            payload_fields.update(
                {
                    "optimization_evaluation_window_count": evaluation.get("total_windows"),
                    "optimization_evaluation_average_wer": evaluation.get("average_wer"),
                    "optimization_evaluation_average_cer": evaluation.get("average_cer"),
                    "optimization_evaluation_average_entity_accuracy": evaluation.get("average_entity_accuracy"),
                }
            )
        self._upsert_job_payload_fields(job_id, payload_fields)

    def _configure_cuda_allocator(self):
        if sys.platform != "win32":
            return
        raw = (os.getenv("PYTORCH_CUDA_ALLOC_CONF") or "").strip()
        if raw:
            return
        parts = [
            f"max_split_size_mb:{int((os.getenv('CUDA_ALLOC_MAX_SPLIT_MB') or '128').strip() or '128')}",
            f"garbage_collection_threshold:{float((os.getenv('CUDA_ALLOC_GC_THRESHOLD') or '0.8').strip() or '0.8')}",
            "expandable_segments:True",
        ]
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
        log_verbose(f"Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    def _get_temp_redo_backup_path(self, video_id: int, token: str | None = None) -> Path:
        token = token or str(int(time.time() * 1000))
        return TEMP_DIR / f"redo_diarization_backup_{int(video_id)}_{token}.json"

    def _get_redo_backup_path_from_payload(self, payload_json: str | None) -> Path | None:
        payload = self._load_job_payload(payload_json)
        mode = (payload.get("mode") or "").strip().lower()
        if mode not in {"redo_diarization", "full_retranscription"}:
            return None
        backup_file = (payload.get("redo_diarization_backup_file") or "").strip()
        if not backup_file:
            return None
        try:
            return Path(backup_file)
        except Exception:
            return None

    def _cleanup_redo_backup_for_job(self, payload_json: str | None):
        backup_path = self._get_redo_backup_path_from_payload(payload_json)
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass

    def _restore_redo_backup_if_needed(self, session: Session, job: Job, reason: str = "") -> bool:
        """Restore transcript/funny rows from redo-diarization backup if the job failed before rewrite."""
        backup_path = self._get_redo_backup_path_from_payload(job.payload_json)
        if not backup_path or not backup_path.exists():
            return False

        video = session.get(Video, job.video_id)
        if not video:
            return False

        existing_segments = session.exec(
            select(TranscriptSegment.id).where(TranscriptSegment.video_id == video.id).limit(1)
        ).first()
        if existing_segments:
            # New transcript exists; stale backup can be removed.
            self._cleanup_redo_backup_for_job(job.payload_json)
            return False

        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            log(f"Could not read redo backup {backup_path}: {e}")
            return False

        restored_segments = 0
        restored_funny = 0

        def _parse_dt(v):
            if not v:
                return None
            try:
                return datetime.fromisoformat(str(v))
            except Exception:
                return None

        for row in payload.get("segments", []) or []:
            try:
                session.add(TranscriptSegment(
                    video_id=video.id,
                    speaker_id=row.get("speaker_id"),
                    matched_profile_id=row.get("matched_profile_id"),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    text=str(row.get("text") or ""),
                    words=row.get("words"),
                ))
                restored_segments += 1
            except Exception:
                continue

        for row in payload.get("funny_moments", []) or []:
            try:
                session.add(FunnyMoment(
                    video_id=video.id,
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    score=float(row.get("score") or 0.0),
                    source=str(row.get("source") or "heuristic"),
                    snippet=row.get("snippet"),
                    humor_summary=row.get("humor_summary"),
                    humor_confidence=row.get("humor_confidence"),
                    humor_model=row.get("humor_model"),
                    humor_explained_at=_parse_dt(row.get("humor_explained_at")),
                    created_at=_parse_dt(row.get("created_at")) or datetime.now(),
                ))
                restored_funny += 1
            except Exception:
                continue

        video.status = payload.get("video_status") or ("completed" if restored_segments > 0 else "downloaded")
        video.processed = bool(payload.get("video_processed")) if payload.get("video_processed") is not None else (restored_segments > 0)
        session.add(video)

        suffix = f" ({reason})" if reason else ""
        if restored_segments > 0:
            job.error = ((job.error or "").strip() + f" | Transcript restored from redo backup{suffix}.").strip(" |")
            session.add(job)

        session.commit()
        self._cleanup_redo_backup_for_job(job.payload_json)
        log(f"Restored redo backup for video {video.id}: {restored_segments} segments, {restored_funny} funny moments{suffix}.")
        return restored_segments > 0

    def _get_video_download_lock(self, video_id: int) -> threading.Lock:
        with self._download_locks_guard:
            lock = self._download_locks.get(video_id)
            if lock is None:
                lock = threading.Lock()
                self._download_locks[video_id] = lock
            return lock

    def _invalidate_speaker_match_cache(self, channel_id: int | None):
        if not channel_id:
            return
        with self._speaker_match_cache_guard:
            self._speaker_match_cache.pop(int(channel_id), None)

    def _append_speaker_match_cache(
        self,
        channel_id: int | None,
        profile_id: int | None,
        speaker_id: int | None,
        embedding,
    ):
        """Append a newly-created embedding to the in-memory channel cache."""
        if not channel_id or not profile_id or not speaker_id or embedding is None:
            return
        import numpy as np

        channel_key = int(channel_id)
        with self._speaker_match_cache_guard:
            cache = self._speaker_match_cache.get(channel_key)
            if not cache:
                return
            try:
                vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            except Exception:
                return
            if vec.size == 0 or int(vec.size) != int(cache.get("dim", -1)):
                return
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                return
            vec = (vec / norm).astype(np.float32, copy=False)
            matrix = cache.get("matrix")
            profile_ids = cache.get("profile_ids")
            speaker_ids = cache.get("speaker_ids")
            if profile_ids is None or speaker_ids is None:
                return
            try:
                if matrix is None or int(cache.get("dim") or 0) <= 0 or profile_ids.size == 0:
                    cache["dim"] = int(vec.size)
                    cache["matrix"] = vec[None, :]
                    cache["profile_ids"] = np.asarray([int(profile_id)], dtype=np.int64)
                    cache["speaker_ids"] = np.asarray([int(speaker_id)], dtype=np.int64)
                    cache["count"] = 1
                    return
                cache["matrix"] = np.vstack([matrix, vec[None, :]])
                cache["profile_ids"] = np.append(profile_ids, int(profile_id))
                cache["speaker_ids"] = np.append(speaker_ids, int(speaker_id))
                cache["count"] = int(cache["profile_ids"].shape[0])
            except Exception:
                # If append fails for any reason, force a safe rebuild on next use.
                self._speaker_match_cache.pop(channel_key, None)

    def _get_speaker_match_cache(self, session: Session, channel_id: int):
        import numpy as np

        channel_key = int(channel_id)
        with self._speaker_match_cache_guard:
            cached = self._speaker_match_cache.get(channel_key)
        if cached:
            return cached

        rows = session.exec(
            select(SpeakerEmbedding)
            .join(Speaker)
            .where(Speaker.channel_id == channel_key)
        ).all()
        if not rows:
            cache = {"dim": 0, "matrix": None, "profile_ids": np.array([], dtype=np.int64), "speaker_ids": np.array([], dtype=np.int64), "count": 0}
            with self._speaker_match_cache_guard:
                self._speaker_match_cache[channel_key] = cache
            return cache

        parsed_rows = []
        for row in rows:
            vec = self._normalize_embedding_vector(getattr(row, "embedding", None))
            if vec is None:
                continue
            parsed_rows.append((int(row.id), int(row.speaker_id), vec))

        vectors = []
        profile_ids = []
        speaker_ids = []
        expected_dim = 0
        if parsed_rows:
            dim_counts = {}
            for _, _, vec in parsed_rows:
                dim = int(vec.size)
                dim_counts[dim] = int(dim_counts.get(dim, 0)) + 1
            expected_dim = max(dim_counts.items(), key=lambda item: (item[1], item[0]))[0]
            for profile_id, speaker_id, vec in parsed_rows:
                if int(vec.size) != int(expected_dim):
                    continue
                vectors.append(vec)
                profile_ids.append(int(profile_id))
                speaker_ids.append(int(speaker_id))

        if not vectors:
            cache = {"dim": 0, "matrix": None, "profile_ids": np.array([], dtype=np.int64), "speaker_ids": np.array([], dtype=np.int64), "count": 0}
        else:
            cache = {
                "dim": int(expected_dim or 0),
                "matrix": np.stack(vectors, axis=0),
                "profile_ids": np.asarray(profile_ids, dtype=np.int64),
                "speaker_ids": np.asarray(speaker_ids, dtype=np.int64),
                "count": len(profile_ids),
            }

        with self._speaker_match_cache_guard:
            self._speaker_match_cache[channel_key] = cache
        return cache

    def _normalize_embedding_vector(self, embedding):
        import numpy as np

        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim > 1:
            try:
                arr = np.mean(arr, axis=0, dtype=np.float32)
            except Exception:
                arr = arr.reshape(-1)
        vec = np.asarray(arr, dtype=np.float32).reshape(-1)
        if vec.size < 16:
            return None
        if not np.all(np.isfinite(vec)):
            return None
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return (vec / norm).astype(np.float32, copy=False)

    def _select_speaker_embedding_segments(self, timeline):
        max_samples = max(1, int(os.getenv("SPEAKER_PROFILE_SAMPLE_COUNT", "4")))
        min_sample_seconds = max(0.5, _env_float("SPEAKER_PROFILE_MIN_SAMPLE_SECONDS", "2.0"))
        fallback_min_seconds = max(0.5, _env_float("SPEAKER_PROFILE_FALLBACK_MIN_SECONDS", "0.75"))

        candidates = []
        for seg in timeline or []:
            try:
                duration = float(seg.duration)
            except Exception:
                duration = 0.0
            if duration <= 0.0:
                continue
            candidates.append(seg)

        if not candidates:
            return []

        candidates.sort(key=lambda seg: (-float(seg.duration), float(seg.start)))
        selected = [seg for seg in candidates if float(seg.duration) >= min_sample_seconds]
        if not selected:
            selected = [seg for seg in candidates if float(seg.duration) >= fallback_min_seconds]
        if not selected:
            selected = candidates

        return selected[:max_samples]

    def _build_speaker_embedding_profile(self, audio_source, timeline):
        import numpy as np

        selected_segments = self._select_speaker_embedding_segments(timeline)
        sample_embeddings = []
        for seg in selected_segments:
            emb = self.get_speaker_embedding(audio_source, seg)
            normalized = self._normalize_embedding_vector(emb)
            if normalized is None:
                continue
            sample_embeddings.append((seg, normalized))

        if not sample_embeddings:
            return None, None, []

        dim_counts = {}
        for _, vec in sample_embeddings:
            dim_counts[int(vec.size)] = int(dim_counts.get(int(vec.size), 0)) + 1
        target_dim = max(dim_counts.items(), key=lambda item: (item[1], item[0]))[0]
        sample_embeddings = [(seg, vec) for seg, vec in sample_embeddings if int(vec.size) == int(target_dim)]
        if not sample_embeddings:
            return None, None, []

        centroid = np.mean(np.stack([vec for _, vec in sample_embeddings], axis=0), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            return None, None, []
        centroid = (centroid / centroid_norm).astype(np.float32, copy=False)
        primary_segment = max((seg for seg, _ in sample_embeddings), key=lambda seg: float(seg.duration))
        return centroid, primary_segment, sample_embeddings

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

    def _set_prefetch_backoff(self, video_id: int, seconds: float):
        until = time.time() + max(1.0, float(seconds))
        with self._prefetch_backoff_guard:
            self._prefetch_backoff_until[int(video_id)] = until

    def _is_prefetch_backoff_active(self, video_id: int) -> bool:
        now = time.time()
        with self._prefetch_backoff_guard:
            until = self._prefetch_backoff_until.get(int(video_id))
            if not until:
                return False
            if now >= until:
                self._prefetch_backoff_until.pop(int(video_id), None)
                return False
            return True

    def _clear_prefetch_backoff(self, video_id: int):
        with self._prefetch_backoff_guard:
            self._prefetch_backoff_until.pop(int(video_id), None)

    def _update_job_progress(self, job_id: int, progress: int):
        """Helper to update job progress in DB"""
        if not job_id:
            return

        from sqlalchemy.exc import OperationalError

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    if job.status == 'paused':
                        raise JobPausedException("Job paused by user")
                    if job.status == 'cancelled':
                        raise JobCancelledException("Job cancelled by user")
                    job.progress = progress
                    session.add(job)
                    session.commit()
                    return
            except JobPausedException:
                raise
            except JobCancelledException:
                raise
            except OperationalError as e:
                if "database is locked" not in str(e).lower():
                    log(f"Failed to update progress: {e}")
                    return
                if attempt >= (max_attempts - 1):
                    log_verbose(f"Skipped progress update for job {job_id} due to DB lock contention.")
                    return
                time.sleep(min(0.05 * (attempt + 1), 0.3))
            except Exception as e:
                log(f"Failed to update progress: {e}")
                return

    def _update_job_status_detail(self, job_id: int, detail: str | None):
        """Helper to update job status_detail for fine-grained progress feedback"""
        if not job_id:
            return
        from sqlalchemy.exc import OperationalError

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    if job.status == 'paused':
                        raise JobPausedException("Job paused by user")
                    if job.status == 'cancelled':
                        raise JobCancelledException("Job cancelled by user")
                    job.status_detail = detail
                    session.add(job)
                    session.commit()
                    return
            except JobPausedException:
                raise
            except JobCancelledException:
                raise
            except OperationalError as e:
                if "database is locked" not in str(e).lower():
                    log(f"Failed to update status_detail: {e}")
                    return
                if attempt >= (max_attempts - 1):
                    log_verbose(f"Skipped status_detail update for job {job_id} due to DB lock contention.")
                    return
                time.sleep(min(0.05 * (attempt + 1), 0.3))
            except Exception as e:
                log(f"Failed to update status_detail: {e}")
                return

    def _upsert_job_payload_fields(self, job_id: int, fields: dict):
        """Merge fields into job.payload_json with lock-tolerant retries."""
        if not job_id or not isinstance(fields, dict) or not fields:
            return
        from sqlalchemy.exc import OperationalError

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    if job.status == 'paused':
                        raise JobPausedException("Job paused by user")
                    if job.status == 'cancelled':
                        raise JobCancelledException("Job cancelled by user")
                    payload = self._load_job_payload(job.payload_json)
                    payload.update({k: v for k, v in fields.items() if v is not None})
                    job.payload_json = json.dumps(payload, sort_keys=True)
                    session.add(job)
                    session.commit()
                    return
            except (JobPausedException, JobCancelledException):
                raise
            except OperationalError as e:
                if "database is locked" not in str(e).lower():
                    log(f"Failed to update job payload_json: {e}")
                    return
                if attempt >= (max_attempts - 1):
                    log_verbose(f"Skipped payload_json update for job {job_id} due to DB lock contention.")
                    return
                time.sleep(min(0.05 * (attempt + 1), 0.3))
            except Exception as e:
                log(f"Failed to update job payload_json: {e}")
                return

    def _record_job_stage_start(self, job_id: int, stage: str):
        """Record per-stage start timestamps for pipeline timing UI."""
        if not job_id:
            return
        stage_key = (stage or "").strip().lower()
        if stage_key not in {"download", "model_load", "transcribe", "transcribe_phase", "diarize", "funny"}:
            return
        now_iso = datetime.now().isoformat()
        payload = {}
        try:
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job and job.payload_json:
                    payload = self._load_job_payload(job.payload_json)
        except Exception:
            payload = {}

        fields = {"stage_last": stage_key, "stage_last_started_at": now_iso}

        stage_field = f"stage_{stage_key}_started_at"
        if not payload.get(stage_field):
            fields[stage_field] = now_iso

        if stage_key == "download" and not payload.get("pipeline_started_at"):
            fields["pipeline_started_at"] = now_iso
        self._upsert_job_payload_fields(job_id, fields)

    def _strip_transient_job_payload_fields(self, payload: dict | None) -> dict:
        source = payload if isinstance(payload, dict) else {}
        cleaned: dict = {}
        transient_prefixes = ("stage_", "parakeet_", "whisper_")
        preserved_exact = set()
        transient_exact = {
            "pipeline_started_at",
            "transcription_engine_requested",
            "transcription_engine_used",
            "transcription_engine_fallback_reason",
            "transcription_engine_fallback_detail",
            "transcription_reused_existing",
            "parakeet_no_fallback_failure",
            "result_kind",
            "result_notice_code",
            "result_notice_message",
            "result_notice_source",
            "result_notice_retryable",
            "result_notice_detail",
        }
        for key, value in source.items():
            if key in preserved_exact:
                cleaned[key] = value
                continue
            if key in transient_exact:
                continue
            if any(key.startswith(prefix) for prefix in transient_prefixes):
                continue
            cleaned[key] = value
        return cleaned

    def _set_funny_task_progress(
        self,
        video_id: int,
        *,
        task: str,
        status: str = "running",
        stage: str | None = None,
        message: str | None = None,
        percent: int | float | None = None,
        current: int | None = None,
        total: int | None = None,
    ):
        try:
            pct = None if percent is None else max(0, min(100, int(round(float(percent)))))
        except Exception:
            pct = None
        payload = {
            "video_id": int(video_id),
            "task": task,
            "status": status,  # running | completed | error
            "stage": stage,
            "message": message,
            "percent": pct,
            "current": None if current is None else int(current),
            "total": None if total is None else int(total),
            "updated_at": time.time(),
        }
        with self._funny_progress_lock:
            self._funny_progress_by_video[int(video_id)] = payload

    def _set_workbench_task_progress(
        self,
        video_id: int,
        *,
        area: str,
        task: str,
        status: str = "running",
        stage: str | None = None,
        message: str | None = None,
        percent: int | float | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        try:
            pct = None if percent is None else max(0, min(100, int(round(float(percent)))))
        except Exception:
            pct = None
        payload = {
            "video_id": int(video_id),
            "area": str(area or "").strip().lower() or None,
            "task": str(task or "").strip().lower() or None,
            "status": str(status or "running").strip().lower() or "running",
            "stage": stage,
            "message": message,
            "percent": pct,
            "current": None if current is None else int(current),
            "total": None if total is None else int(total),
            "updated_at": time.time(),
        }
        with self._workbench_progress_lock:
            self._workbench_progress_by_video[int(video_id)] = payload

    def get_workbench_task_progress(self, video_id: int) -> dict:
        vid = int(video_id)
        with self._workbench_progress_lock:
            progress = self._workbench_progress_by_video.get(vid)
            if not progress:
                return {"video_id": vid, "status": "idle"}

            updated_at = float(progress.get("updated_at") or 0)
            age = time.time() - updated_at if updated_at else 999999
            if progress.get("status") in {"completed", "error"} and age > 20:
                self._workbench_progress_by_video.pop(vid, None)
                return {"video_id": vid, "status": "idle"}

            return dict(progress)

    def get_funny_task_progress(self, video_id: int) -> dict:
        """Return latest funny-moment task progress for a video (best-effort)."""
        vid = int(video_id)
        with self._funny_progress_lock:
            progress = self._funny_progress_by_video.get(vid)
            if not progress:
                return {"video_id": vid, "status": "idle"}

            # Expire old completed/error states so stale progress does not linger forever.
            updated_at = float(progress.get("updated_at") or 0)
            age = time.time() - updated_at if updated_at else 999999
            if progress.get("status") in {"completed", "error"} and age > 120:
                self._funny_progress_by_video.pop(vid, None)
                return {"video_id": vid, "status": "idle"}

            return dict(progress)

    def _claim_next_queued_job(self, allowed_job_types: set[str]):
        """Atomically claim the next queued job for a given queue."""
        from sqlalchemy import update as sa_update

        with Session(engine) as session:
            candidates = session.exec(
                select(Job.id, Job.video_id, Job.job_type, Job.payload_json)
                .where(Job.status == "queued", Job.job_type.in_(list(allowed_job_types)))
                .order_by(Job.created_at.asc(), Job.id.asc())
                .limit(12)
            ).all()
            if not candidates:
                return None

            for job_id, video_id, job_type, payload_json in candidates:
                payload = self._strip_transient_job_payload_fields(self._load_job_payload(payload_json))
                normalized_payload_json = json.dumps(payload, sort_keys=True) if payload else None
                claim_time = datetime.now()
                result = session.exec(
                    sa_update(Job)
                    .where(Job.id == job_id, Job.status == "queued")
                    .values(
                        status="running",
                        started_at=claim_time,
                        completed_at=None,
                        error=None,
                        progress=0,
                        status_detail=None,
                        payload_json=normalized_payload_json,
                    )
                )
                if int(getattr(result, "rowcount", 0) or 0) <= 0:
                    session.rollback()
                    continue
                session.commit()
                return {
                    "id": int(job_id),
                    "video_id": int(video_id),
                    "job_type": str(job_type),
                    "payload_json": normalized_payload_json,
                }
            return None

    def _mark_job_success(self, job_id: int):
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return
            if str(job.status or "").strip().lower() == "cancelled":
                return
            job.status = "completed"
            job.completed_at = datetime.now()
            job.progress = 100
            job.status_detail = None
            session.add(job)
            session.commit()

    def _trigger_semantic_index(self, video_id: int) -> None:
        """Fire-and-forget semantic index build for a single video after transcription."""
        try:
            from .semantic_search import start_index_job
            start_index_job([video_id])
            log(f"[semantic] Queued semantic indexing for video {video_id}.")
        except Exception as exc:
            log(f"[semantic] Failed to queue semantic index for video {video_id}: {exc}")

    def _mark_job_failure(self, job_id: int, error: str):
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return
            if str(job.status or "").strip().lower() == "cancelled":
                return
            job.status = "failed"
            job.error = _truncate_error(error)
            job.completed_at = datetime.now()
            job.status_detail = None
            session.add(job)
            session.commit()

    def _infer_recoverable_video_status(self, session: Session, video: Video) -> str:
        """Infer the safest stable video status from persisted artifacts."""
        if bool(video.processed):
            return "completed"

        has_segments = session.exec(
            select(TranscriptSegment.id)
            .where(TranscriptSegment.video_id == video.id)
            .limit(1)
        ).first() is not None
        if has_segments:
            return "transcribed"

        try:
            audio_path = self.get_audio_path(video)
        except Exception:
            audio_path = None

        raw_transcript_path = None
        if audio_path is not None:
            try:
                safe_title = self.sanitize_filename(video.title)
                raw_transcript_path = audio_path.parent / f"{safe_title}_transcript_raw.json"
            except Exception:
                raw_transcript_path = None

        if raw_transcript_path is not None and raw_transcript_path.exists():
            return "transcribed"
        if audio_path is not None and audio_path.exists():
            return "downloaded"
        return "pending"

    def _recover_inactive_video_status(self, video_id: int) -> str | None:
        """Restore a video from an active-looking state when no active job exists."""
        active_job_statuses = ["queued", "running", "downloading", "transcribing", "diarizing", "waiting_diarize"]
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                return None

            has_active_job = session.exec(
                select(Job.id)
                .where(
                    Job.video_id == video_id,
                    Job.status.in_(active_job_statuses),
                )
                .limit(1)
            ).first() is not None
            if has_active_job:
                return str(video.status or "")

            prev_status = str(video.status or "")
            recovered_status = self._infer_recoverable_video_status(session, video)
            if recovered_status != prev_status:
                video.status = recovered_status
                session.add(video)
                session.commit()
                log(
                    f"Recovered orphaned video status for video {video_id}: "
                    f"{prev_status or 'unknown'} -> {recovered_status}"
                )
            return recovered_status

    def _mark_job_notice(
        self,
        job_id: int | None,
        video_id: int | None,
        *,
        code: str,
        message: str,
        technical_detail: str | None = None,
        video_status: str = "pending",
    ):
        restricted_codes = {"youtube_members_only", "youtube_private_video", "youtube_auth_required"}
        with Session(engine) as session:
            if job_id:
                job = session.get(Job, job_id)
                if job:
                    payload = self._load_job_payload(job.payload_json)
                    payload.update(
                        {
                            "result_kind": "notice",
                            "result_notice_code": str(code or "notice"),
                            "result_notice_message": str(message or "Notice"),
                            "result_notice_source": "download",
                            "result_notice_retryable": bool(code not in restricted_codes),
                        }
                    )
                    if technical_detail:
                        payload["result_notice_detail"] = str(technical_detail)[:1000]
                    job.payload_json = json.dumps(payload, sort_keys=True)
                    job.status = "failed"
                    job.error = _truncate_error(message)
                    job.completed_at = datetime.now()
                    job.status_detail = None
                    session.add(job)

            if video_id:
                video = session.get(Video, video_id)
                if video:
                    video.status = str(video_status or "pending")
                    if code in restricted_codes:
                        video.access_restricted = True
                        video.access_restriction_reason = str(message or "Access restricted")
                    session.add(video)

            session.commit()

    def _load_job_payload(self, payload_json: str | None) -> dict:
        if not payload_json:
            return {}
        try:
            obj = json.loads(payload_json)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _has_jobs_of_types(self, job_types: set[str], statuses: set[str]) -> bool:
        if not job_types or not statuses:
            return False
        with Session(engine) as session:
            row = session.exec(
                select(Job.id)
                .where(Job.job_type.in_(list(job_types)), Job.status.in_(list(statuses)))
                .limit(1)
            ).first()
            return row is not None

    def _set_oldest_queued_job_status_detail(self, job_type: str, detail: str | None):
        with Session(engine) as session:
            job = session.exec(
                select(Job)
                .where(Job.status == "queued", Job.job_type == job_type)
                .order_by(Job.created_at.asc(), Job.id.asc())
            ).first()
            if not job:
                return
            normalized = detail or None
            if (job.status_detail or None) == normalized:
                return
            job.status_detail = normalized
            session.add(job)
            session.commit()

    def _has_active_pipeline_gpu_work(self) -> bool:
        active_statuses = {"running", "downloading", "transcribing", "diarizing"}
        return self._has_jobs_of_types(PROCESS_JOB_TYPES | DIARIZE_JOB_TYPES, active_statuses)

    def _get_detached_video(self, video_id: int) -> Video:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise RuntimeError(f"Video {video_id} not found")
            _ = video.channel
            session.expunge(video)
            if video.channel:
                session.expunge(video.channel)
            return video

    def _ensure_audio_ready_for_video(self, video: Video, job_id: int = None) -> Path:
        lock = self._get_video_download_lock(int(video.id))
        with lock:
            audio_path = self.download_audio(video, job_id=job_id)
            audio_path = self._validate_and_retry_audio(video, audio_path, job_id)
        return audio_path

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

    def _queue_diarize_followup(self, video_id: int, parent_job_id: int):
        with Session(engine) as session:
            parent = session.get(Job, parent_job_id)
            if not parent:
                raise RuntimeError(f"Parent process job {parent_job_id} not found")
            payload = self._load_job_payload(parent.payload_json)
            payload["parent_job_id"] = int(parent_job_id)
            payload["pipeline_stage"] = "diarize"
            child = self._enqueue_job(video_id, "diarize", payload=payload)
            return int(child.id)

    def _mark_process_job_waiting_for_diarize(self, job_id: int, diarize_job_id: int):
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return
            video = session.get(Video, job.video_id)
            now = datetime.now()
            payload = self._load_job_payload(job.payload_json)
            transcribe_start_ms = None
            try:
                transcribe_started_at = payload.get("stage_transcribe_started_at") or payload.get("stage_transcribing_started_at")
                if not transcribe_started_at and job.started_at:
                    transcribe_started_at = job.started_at.isoformat()
                if transcribe_started_at:
                    transcribe_start_ms = datetime.fromisoformat(str(transcribe_started_at))
            except Exception:
                transcribe_start_ms = None
            payload.update(
                {
                    "pipeline_stage": "waiting_diarize",
                    "diarize_job_id": int(diarize_job_id),
                    "stage_transcribe_completed_at": now.isoformat(),
                }
            )
            if transcribe_start_ms is not None:
                try:
                    payload["stage_transcribe_seconds"] = max(0.0, (now - transcribe_start_ms).total_seconds())
                except Exception:
                    pass
            job.status = "waiting_diarize"
            job.progress = max(int(job.progress or 0), 55)
            job.status_detail = f"Queued for diarization (job {diarize_job_id})"
            if not payload.get("stage_transcribe_completed_at"):
                payload["stage_transcribe_completed_at"] = datetime.now().isoformat()
            job.payload_json = json.dumps(payload, sort_keys=True)
            session.add(job)
            if video:
                video.status = "transcribed"
                session.add(video)
            session.commit()

    def _finalize_process_job_from_child(self, parent_job_id: int, child_job_id: int, status: str, error: str = None):
        """Update a parent process job based on its child job's outcome."""
        with Session(engine) as session:
            parent = session.get(Job, parent_job_id)
            child = session.get(Job, child_job_id)
            if not parent:
                return
            payload = self._load_job_payload(parent.payload_json)
            child_payload = self._load_job_payload(child.payload_json if child else None)
            for key, value in child_payload.items():
                if key.startswith("stage_") or key.startswith("parakeet_") or key.startswith("transcription_"):
                    payload[key] = value
            payload["pipeline_stage"] = status
            payload["diarize_job_id"] = int(child_job_id)
            parent.payload_json = json.dumps(payload, sort_keys=True)
            parent.status = status
            if status == "completed":
                parent.progress = 100
            if error:
                parent.error = _truncate_error(error)
            parent.status_detail = None
            parent.completed_at = datetime.now()
            session.add(parent)
            session.commit()

    def _enqueue_job(self, video_id: int, job_type: str, payload: dict | None = None):
        """Add a queued job if one of the same type+payload isn't already active/queued."""
        payload_text = json.dumps(payload or {}, sort_keys=True) if payload is not None else None
        with Session(engine) as session:
            existing = session.exec(
                select(Job).where(
                    Job.video_id == video_id,
                    Job.job_type == job_type,
                    Job.status.in_(["queued", "running", "downloading", "transcribing", "diarizing"]),
                )
            ).all()
            for j in existing:
                if (j.payload_json or None) == (payload_text or None):
                    return j
            job = Job(video_id=video_id, job_type=job_type, status="queued", payload_json=payload_text)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def _run_worker_loop(self, worker_name: str, allowed_job_types: set[str], handler):
        """Generic queue worker loop for a queue partition."""
        log(f"Starting {worker_name} queue worker...")
        sleep_empty = 2.0
        while True:
            try:
                claimed = self._claim_next_queued_job(allowed_job_types)
                if not claimed:
                    time.sleep(sleep_empty)
                    continue

                job_id = claimed["id"]
                video_id = claimed["video_id"]
                job_type = claimed["job_type"]
                payload = self._load_job_payload(claimed.get("payload_json"))
                try:
                    handler(job_id, video_id, job_type, payload)
                    self._mark_job_success(job_id)
                except JobPausedException:
                    log(f"Job {job_id} paused by user")
                except JobCancelledException:
                    log(f"Job {job_id} cancelled by user")
                except Exception as e:
                    log(f"{worker_name} job {job_id} failed: {e}")
                    if is_verbose():
                        import traceback
                        traceback.print_exc()
                    self._mark_job_failure(job_id, str(e))
            except Exception as e:
                log(f"{worker_name} worker loop error: {e}")
                if is_verbose():
                    import traceback
                    traceback.print_exc()
                time.sleep(3)

    def _handle_funny_job(self, job_id: int, video_id: int, job_type: str, payload: dict):
        if job_type == "funny_detect":
            force = bool(payload.get("force", True))
            self._update_job_status_detail(job_id, "Detecting funny moments...")
            self._update_job_progress(job_id, 5)
            self.detect_funny_moments(video_id, force=force)
            self._update_job_progress(job_id, 100)
            self._update_job_status_detail(job_id, None)
            return
        if job_type == "funny_explain":
            force = bool(payload.get("force", True))
            limit = payload.get("limit")
            if limit is not None:
                try:
                    limit = int(limit)
                except Exception:
                    limit = None
            self._update_job_status_detail(job_id, "Generating funny-moment explanations...")
            self._update_job_progress(job_id, 5)
            self.explain_funny_moments(video_id, force=force, limit=limit, job_id=job_id)
            self._update_job_progress(job_id, 100)
            self._update_job_status_detail(job_id, None)
            return
        raise ValueError(f"Unsupported funny queue job_type '{job_type}'")

    def _handle_youtube_job(self, job_id: int, video_id: int, job_type: str, payload: dict):
        if job_type == "youtube_metadata":
            force = bool(payload.get("force", True))
            self._update_job_status_detail(job_id, "Generating YouTube summary + chapters...")
            self._update_job_progress(job_id, 5)
            self.generate_youtube_metadata_suggestion(video_id, force=force)
            self._update_job_progress(job_id, 100)
            self._update_job_status_detail(job_id, None)
            return
        if job_type == "episode_clone":
            request_payload = clone_svc.normalize_clone_request(
                style_prompt=str(payload.get("style_prompt") or ""),
                notes=payload.get("notes"),
                semantic_query=payload.get("semantic_query"),
                related_limit=int(payload.get("related_limit") or 8),
                variant_label=payload.get("variant_label"),
                provider_override=payload.get("provider_override"),
                model_override=payload.get("model_override"),
                approved_concepts=payload.get("approved_concepts"),
                excluded_references=payload.get("excluded_references"),
            )
            request_signature = clone_svc.clone_request_signature(video_id=video_id, request_payload=request_payload)
            self._upsert_job_payload_fields(
                job_id,
                {
                    **request_payload,
                    "request_signature": request_signature,
                },
            )
            self._update_job_status_detail(job_id, "Gathering source and channel context...")
            self._update_job_progress(job_id, 15)
            target_provider, target_model, model_name = self.resolve_clone_llm_target(
                provider_override=str(request_payload.get("provider_override") or ""),
                model_override=str(request_payload.get("model_override") or ""),
            )
            with Session(engine) as session:
                result = clone_svc.generate_episode_clone(
                    session,
                    video_id=video_id,
                    style_prompt=str(request_payload["style_prompt"]),
                    notes=request_payload.get("notes"),
                    semantic_query=request_payload.get("semantic_query"),
                    related_limit=int(request_payload.get("related_limit") or 8),
                    approved_concepts=list(request_payload.get("approved_concepts") or []),
                    excluded_references=list(request_payload.get("excluded_references") or []),
                    text_generator=lambda prompt: self.generate_clone_text(
                        self._mark_clone_job_generation_started(job_id, prompt),
                        provider_override=target_provider,
                        model_override=target_model,
                        temperature=0.35,
                        num_predict=1800,
                        timeout_seconds=180,
                    ),
                    model_name=model_name,
                )
            self._update_job_status_detail(job_id, "Saving clone result...")
            self._update_job_progress(job_id, 95)
            self._upsert_job_payload_fields(
                job_id,
                {
                    "request_signature": request_signature,
                    "clone_result": jsonable_encoder(result),
                    "clone_result_ready": True,
                    "clone_generated_at": datetime.now().isoformat(),
                },
            )
            self._update_job_progress(job_id, 100)
            self._update_job_status_detail(job_id, None)
            return
        raise ValueError(f"Unsupported youtube queue job_type '{job_type}'")

    def _mark_clone_job_generation_started(self, job_id: int, prompt: str) -> str:
        self._update_job_status_detail(job_id, "Generating clone script...")
        self._update_job_progress(job_id, 65)
        return prompt

    def _handle_clip_job(self, job_id: int, video_id: int, job_type: str, payload: dict):
        clip_id = payload.get("clip_id")
        if not clip_id:
            raise ValueError("Clip queue job missing clip_id payload")
        clip_id = int(clip_id)
        if job_type == "clip_export_mp4":
            self._update_job_status_detail(job_id, f"Rendering MP4 for clip {clip_id}...")
            self._update_job_progress(job_id, 5)
            out = self.render_clip_export_mp4(clip_id)
            self.record_clip_export_artifact(clip_id, out, artifact_type="video", fmt="mp4")
            self._update_job_status_detail(job_id, f"Rendered: {out.name}")
            self._update_job_progress(job_id, 100)
            return
        if job_type == "clip_export_captions":
            fmt = str(payload.get("format") or "srt").lower()
            speaker_labels = bool(payload.get("speaker_labels", True))
            self._update_job_status_detail(job_id, f"Rendering {fmt.upper()} captions for clip {clip_id}...")
            self._update_job_progress(job_id, 5)
            out = self.write_clip_caption_file(clip_id, fmt=fmt, speaker_labels=speaker_labels)
            self.record_clip_export_artifact(clip_id, out, artifact_type="captions", fmt=fmt)
            self._update_job_status_detail(job_id, f"Rendered: {out.name}")
            self._update_job_progress(job_id, 100)
            return
        raise ValueError(f"Unsupported clip queue job_type '{job_type}'")

    def _handle_transcript_repair_job(self, job_id: int, video_id: int, payload: dict):
        self._update_job_status_detail(job_id, "Repairing transcript segmentation...")
        self._update_job_progress(job_id, 5)
        with Session(engine) as session:
            result = self.repair_existing_transcript(
                session,
                video_id,
                save_files=bool(payload.get("save_files", True)),
                persist_run=True,
                persist_snapshot=True,
                source="queued_job",
                note=str(payload.get("note") or "").strip() or f"Queued transcript repair job {job_id}",
                trigger_semantic_index=True,
            )
        self._upsert_job_payload_fields(
            job_id,
            {
                "repair_run_id": result.get("run_id"),
                "repair_snapshot_id": result.get("snapshot_id"),
                "repair_changed": bool(result.get("changed")),
                "repair_backup_file": result.get("backup_file"),
                "repair_recommended_tier_after": result.get("recommended_tier_after"),
                "repair_quality_score_after": result.get("quality_score_after"),
            },
        )
        self._update_job_progress(job_id, 100)
        self._update_job_status_detail(job_id, None)

    def _handle_voicefixer_job(self, job_id: int, video_id: int, payload: dict):
        force = bool(payload.get("force", False))
        self._run_voicefixer_cleanup(video_id, job_id=job_id, force=force)
        self._update_job_progress(job_id, 100)
        self._update_job_status_detail(job_id, None)

    def _handle_reconstruction_job(self, job_id: int, video_id: int, payload: dict):
        force = bool(payload.get("force", False))
        self._run_conversation_reconstruction(video_id, job_id=job_id, force=force)
        self._update_job_progress(job_id, 100)
        self._update_job_status_detail(job_id, None)

    def _run_voicefixer_cleanup(self, video_id: int, *, job_id: int | None = None, force: bool = False) -> Path:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("VoiceFixer is only available for uploaded manual media.")
            original_path, input_wav, restored_wav, final_path = self._voicefixer_paths_for_manual_video(video)
            voicefixer_source_path = self._get_cleanup_selected_candidate_path(video) or original_path
            blended_wav = input_wav.parent / f"{input_wav.stem.replace('.input', '')}.voicefixer.blended.wav"
            leveled_wav = input_wav.parent / f"{input_wav.stem.replace('.input', '')}.voicefixer.leveled.wav"
            cleaned_rel_path = final_path.relative_to(MANUAL_MEDIA_DIR).as_posix()
            current_cleaned_exists = final_path.exists()
            mode = max(0, min(2, int(getattr(video, "voicefixer_mode", 0) or 0)))
            mix_ratio = max(0.0, min(1.0, float(getattr(video, "voicefixer_mix_ratio", 1.0) or 1.0)))
            leveling_mode = str(getattr(video, "voicefixer_leveling_mode", "off") or "off").strip().lower()
            use_cleaned = self._voicefixer_apply_scope(video) != "none"

        if current_cleaned_exists and not force:
            self._set_voicefixer_state(
                video_id,
                status="ready",
                use_cleaned=use_cleaned,
                cleaned_path=cleaned_rel_path,
                error="",
            )
            return final_path

        self._set_voicefixer_state(video_id, status="processing", error="")
        ffmpeg_cmd = self._get_ffmpeg_cmd()

        for path in (input_wav, restored_wav, blended_wav, leveled_wav, final_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        try:
            if job_id:
                self._update_job_progress(job_id, 10)
                self._update_job_status_detail(job_id, "Preparing media for VoiceFixer...")

            self._run_external_command(
                [
                    ffmpeg_cmd,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(voicefixer_source_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-c:a",
                    "pcm_s16le",
                    str(input_wav),
                ],
                "Failed to prepare audio for VoiceFixer",
            )

            if job_id:
                self._update_job_progress(job_id, 45)
                self._update_job_status_detail(job_id, "Running VoiceFixer restoration...")

            from voicefixer import VoiceFixer  # type: ignore
            try:
                self._ensure_device()
            except Exception:
                pass

            use_cuda = False
            try:
                import torch  # type: ignore
                use_cuda = bool(torch.cuda.is_available() and str(self.device or "") == "cuda" and not self._cuda_recovery_pending)
            except Exception:
                use_cuda = False

            restorer = VoiceFixer()
            restorer.restore(input=str(input_wav), output=str(restored_wav), cuda=use_cuda, mode=mode)

            current_audio_path = restored_wav

            if mix_ratio < 0.999:
                if job_id:
                    self._update_job_progress(job_id, 62)
                    self._update_job_status_detail(job_id, "Blending restored and original audio...")
                restored_weight = max(0.0, min(1.0, mix_ratio))
                original_weight = max(0.0, 1.0 - restored_weight)
                self._run_external_command(
                    [
                        ffmpeg_cmd,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(restored_wav),
                        "-i",
                        str(input_wav),
                        "-filter_complex",
                        f"amix=inputs=2:weights='{restored_weight:.4f} {original_weight:.4f}':normalize=0",
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        str(blended_wav),
                    ],
                    "Failed to blend VoiceFixer-restored and original audio",
                )
                current_audio_path = blended_wav

            leveling_filter = self._voicefixer_leveling_filter(leveling_mode)
            if leveling_filter:
                if job_id:
                    self._update_job_progress(job_id, 72)
                    self._update_job_status_detail(job_id, "Applying voice leveling...")
                self._run_external_command(
                    [
                        ffmpeg_cmd,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(current_audio_path),
                        "-af",
                        leveling_filter,
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        str(leveled_wav),
                    ],
                    "Failed to apply VoiceFixer leveling",
                )
                current_audio_path = leveled_wav

            if final_path.suffix.lower() == ".mp4":
                if job_id:
                    self._update_job_progress(job_id, 80)
                    self._update_job_status_detail(job_id, "Merging restored audio back into video...")
                self._run_external_command(
                    [
                        ffmpeg_cmd,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(original_path),
                        "-i",
                        str(current_audio_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-shortest",
                        str(final_path),
                    ],
                    "Failed to merge VoiceFixer audio into video",
                )
            else:
                current_audio_path.replace(final_path)

            self._set_voicefixer_state(
                video_id,
                status="ready",
                use_cleaned=use_cleaned,
                cleaned_path=cleaned_rel_path,
                error="",
            )
            return final_path
        except JobPausedException:
            self._set_voicefixer_state(video_id, status="paused", error="")
            raise
        except JobCancelledException:
            raise
        except Exception as e:
            self._set_voicefixer_state(video_id, status="failed", error=str(e))
            raise
        finally:
            for temp_path in (input_wav, restored_wav, blended_wav, leveled_wav):
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _run_conversation_reconstruction(self, video_id: int, *, job_id: int | None = None, force: bool = False) -> Path:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Conversation reconstruction is currently available for uploaded manual media only.")
            output_wav, sidecar_json = self._reconstruction_output_paths(video)
            rel_output = output_wav.relative_to(MANUAL_MEDIA_DIR).as_posix()
            transcript_segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            if not transcript_segments:
                raise ValueError("Transcript segments are required before running conversation reconstruction.")
            if not any(getattr(seg, "speaker_id", None) is not None for seg in transcript_segments):
                raise ValueError("Diarization must be completed before running conversation reconstruction.")
            source_path = self.get_audio_path(video, purpose="processing")
            performance_source_path = self.get_original_manual_media_path(video) or source_path
            reconstruction_mode = self._get_reconstruction_mode(video)
            performance_mode = reconstruction_mode == "performance"
            instruction_template = self._get_reconstruction_instruction_template(video)
            current_exists = output_wav.exists()

        if current_exists and not force:
            self._set_reconstruction_state(video_id, status="ready", audio_path=rel_output, error="")
            return output_wav

        model_name = (os.getenv("RECONSTRUCTION_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-Base").strip()
        self._set_reconstruction_state(video_id, status="processing", error="", model=model_name)
        work_dir = output_wav.parent / f".reconstruction_{video_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if job_id:
                self._update_job_progress(job_id, 5)
                self._update_job_status_detail(job_id, "Extracting speaker references...")
            self._check_job_not_paused(job_id)
            grouped_segments: dict[int, list[TranscriptSegment]] = {}
            for seg in transcript_segments:
                seg_speaker_id = getattr(seg, "speaker_id", None)
                if seg_speaker_id is not None:
                    grouped_segments.setdefault(int(seg_speaker_id), []).append(seg)
            state = self._sync_reconstruction_workbench_state(video, grouped_segments)
            references = self._extract_reconstruction_references(
                source_path,
                transcript_segments,
                work_dir,
                video=video,
                state=state,
            )
            if not references:
                raise ValueError("Could not extract any speaker reference clips from the diarized transcript.")

            if job_id:
                self._update_job_progress(job_id, 15)
                self._update_job_status_detail(job_id, "Loading reconstruction TTS model...")

            import soundfile as sf  # type: ignore

            tts_model, use_cuda, _ = self._get_reconstruction_tts_model(model_name)

            xvector_only_mode = (os.getenv("RECONSTRUCTION_XVECTOR_ONLY") or "true").strip().lower() in {"1", "true", "yes", "on"}
            clean_prompts_by_speaker: dict[int, object] = {}
            for speaker_id, ref in references.items():
                prompt_items = tts_model.create_voice_clone_prompt(
                    ref_audio=str(ref["path"]),
                    ref_text=None if xvector_only_mode else str(ref["text"]),
                    x_vector_only_mode=xvector_only_mode,
                )
                if not prompt_items:
                    continue
                clean_prompts_by_speaker[speaker_id] = prompt_items[0]

            sample_rate = 44100
            placements: list[dict] = []
            total_segments = max(1, len(transcript_segments))
            sidecar_rows: list[dict] = []
            use_cuda_batch = bool(use_cuda)
            batch_size = self._reconstruction_batch_size(use_cuda_batch)
            batch_token_budget = self._reconstruction_batch_token_budget(use_cuda_batch)
            batch_duration_budget = self._reconstruction_batch_duration_budget(use_cuda_batch)

            def append_original(seg, *, synthesized: bool = False):
                seg_text = str(getattr(seg, "text", "") or "").strip()
                start_time = float(seg.start_time)
                end_time = float(seg.end_time)
                target_duration = max(0.05, end_time - start_time)
                wav, wav_sr = self._load_original_audio_segment(source_path, start_time, end_time)
                fitted = self._fit_waveform_duration(wav, wav_sr, target_duration)
                placements.append({
                    "start_time": start_time,
                    "wav": fitted,
                })
                sidecar_rows.append({
                    "segment_id": int(getattr(seg, "id", 0) or 0),
                    "speaker_id": int(getattr(seg, "speaker_id", None)) if getattr(seg, "speaker_id", None) is not None else None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": seg_text,
                    "source": "tts" if synthesized else "original",
                })

            batch_items: list[dict] = []
            batch_cost = 0
            batch_duration = 0.0

            def flush_batch(end_index: int) -> None:
                nonlocal batch_items, batch_cost, batch_duration, batch_token_budget, batch_duration_budget
                if not batch_items:
                    return
                self._check_job_not_paused(job_id)
                batch_token_budget = self._reconstruction_batch_token_budget(use_cuda_batch)
                batch_duration_budget = self._reconstruction_batch_duration_budget(use_cuda_batch)
                if job_id:
                    first_idx = int(batch_items[0]["index"]) + 1
                    last_idx = int(batch_items[-1]["index"]) + 1
                    progress = 15 + int((min(end_index, total_segments) / total_segments) * 75)
                    self._update_job_progress(job_id, progress)
                    self._update_job_status_detail(job_id, f"Reconstructing segments {first_idx}-{last_idx}/{total_segments}...")
                try:
                    batch_prompts, batch_texts = self._prepare_reconstruction_batch_prompts(
                        tts_model,
                        batch_items,
                        clean_prompts_by_speaker=clean_prompts_by_speaker,
                        performance_source_path=performance_source_path,
                        performance_mode=performance_mode,
                        work_dir=work_dir,
                    )
                    wavs, batch_sr = self._synthesize_reconstruction_batch(
                        tts_model,
                        batch_prompts,
                        batch_texts,
                        [item["target_duration"] for item in batch_items],
                        model_name,
                    )
                    if len(wavs) != len(batch_items):
                        raise RuntimeError(f"Qwen3-TTS returned {len(wavs)} outputs for {len(batch_items)} requested segments.")
                    for item, wav in zip(batch_items, wavs):
                        validation_error = self._validate_reconstruction_waveform(wav, int(batch_sr), float(item["target_duration"]))
                        if validation_error and performance_mode:
                            fitted, _, _ = self._synthesize_validated_reconstruction_segment(
                                tts_model,
                                clean_prompts_by_speaker[int(item["speaker_id"])],
                                str(item["text"]),
                                model_name=model_name,
                                target_seconds=float(item["target_duration"]),
                            )
                        elif validation_error:
                            raise RuntimeError(validation_error)
                        else:
                            fitted = self._fit_waveform_duration(wav, batch_sr, float(item["target_duration"]))
                        placements.append({
                            "start_time": float(item["start_time"]),
                            "wav": fitted,
                        })
                        sidecar_rows.append({
                            "segment_id": int(item["segment_id"]),
                            "speaker_id": int(item["speaker_id"]) if item["speaker_id"] is not None else None,
                            "start_time": float(item["start_time"]),
                            "end_time": float(item["end_time"]),
                            "text": str(item["text"]),
                            "source": "tts",
                        })
                except Exception:
                    for item in batch_items:
                        try:
                            perf_prompt = item["prompt"]
                            perf_text = str(item["generation_text"])
                            if performance_mode:
                                prompt_list, text_list = self._prepare_reconstruction_batch_prompts(
                                    tts_model,
                                    [item],
                                    clean_prompts_by_speaker=clean_prompts_by_speaker,
                                    performance_source_path=performance_source_path,
                                    performance_mode=True,
                                    work_dir=work_dir,
                                )
                                perf_prompt = prompt_list[0]
                                perf_text = text_list[0]
                            fitted, _, _ = self._synthesize_validated_reconstruction_segment(
                                tts_model,
                                perf_prompt,
                                perf_text,
                                model_name=model_name,
                                target_seconds=float(item["target_duration"]),
                                fallback_prompt=clean_prompts_by_speaker[int(item["speaker_id"])] if performance_mode else None,
                                fallback_text=str(item["text"]),
                            )
                            placements.append({
                                "start_time": float(item["start_time"]),
                                "wav": fitted,
                            })
                            sidecar_rows.append({
                                "segment_id": int(item["segment_id"]),
                                "speaker_id": int(item["speaker_id"]) if item["speaker_id"] is not None else None,
                                "start_time": float(item["start_time"]),
                                "end_time": float(item["end_time"]),
                                "text": str(item["text"]),
                                "source": "tts",
                            })
                        except Exception:
                            append_original(item["segment_obj"], synthesized=False)
                finally:
                    batch_items = []
                    batch_cost = 0
                    batch_duration = 0.0
                    self._release_reconstruction_cuda_cache()

            def synthesize_isolated_item(item: dict, end_index: int) -> None:
                self._check_job_not_paused(job_id)
                if job_id:
                    progress = 15 + int((min(end_index, total_segments) / total_segments) * 75)
                    self._update_job_progress(job_id, progress)
                    self._update_job_status_detail(
                        job_id,
                        f"Reconstructing long segment {int(item['index']) + 1}/{total_segments}...",
                    )
                try:
                    prompt_to_use = item["prompt"]
                    text_to_use = str(item["generation_text"])
                    if performance_mode:
                        prompt_list, text_list = self._prepare_reconstruction_batch_prompts(
                            tts_model,
                            [item],
                            clean_prompts_by_speaker=clean_prompts_by_speaker,
                            performance_source_path=performance_source_path,
                            performance_mode=True,
                            work_dir=work_dir,
                        )
                        prompt_to_use = prompt_list[0]
                        text_to_use = text_list[0]
                    fitted, _, _ = self._synthesize_validated_reconstruction_segment(
                        tts_model,
                        prompt_to_use,
                        text_to_use,
                        model_name=model_name,
                        target_seconds=float(item["target_duration"]),
                        fallback_prompt=clean_prompts_by_speaker[int(item["speaker_id"])] if performance_mode else None,
                        fallback_text=str(item["text"]),
                    )
                    placements.append({
                        "start_time": float(item["start_time"]),
                        "wav": fitted,
                    })
                    sidecar_rows.append({
                        "segment_id": int(item["segment_id"]),
                        "speaker_id": int(item["speaker_id"]) if item["speaker_id"] is not None else None,
                        "start_time": float(item["start_time"]),
                        "end_time": float(item["end_time"]),
                        "text": str(item["text"]),
                        "source": "tts",
                    })
                except Exception:
                    append_original(item["segment_obj"], synthesized=False)
                finally:
                    self._release_reconstruction_cuda_cache()

            for idx, seg in enumerate(transcript_segments):
                self._check_job_not_paused(job_id)
                seg_text = str(getattr(seg, "text", "") or "").strip()
                start_time = float(seg.start_time)
                end_time = float(seg.end_time)
                target_duration = max(0.05, end_time - start_time)
                speaker_id = getattr(seg, "speaker_id", None)
                can_use_tts = (
                    speaker_id is not None
                    and int(speaker_id) in clean_prompts_by_speaker
                    and self._should_reconstruct_with_tts(seg_text, target_duration)
                )
                if can_use_tts:
                    item_cost = self._estimate_reconstruction_item_cost(model_name, target_duration, seg_text)
                    item = {
                        "index": idx,
                        "segment_obj": seg,
                        "segment_id": int(getattr(seg, "id", 0) or 0),
                        "speaker_id": int(speaker_id),
                        "start_time": start_time,
                        "end_time": end_time,
                        "target_duration": target_duration,
                        "text": seg_text,
                        "generation_text": self._build_reconstruction_generation_text(
                            seg_text,
                            performance_mode=performance_mode,
                            instruction_template=instruction_template,
                        ),
                        "prompt": clean_prompts_by_speaker[int(speaker_id)],
                    }
                    if self._should_force_original_reconstruction_segment(seg_text, target_duration):
                        flush_batch(idx)
                        if job_id:
                            progress = 15 + int(((idx + 1) / total_segments) * 75)
                            self._update_job_progress(job_id, progress)
                            self._update_job_status_detail(job_id, f"Using original audio for long segment {idx + 1}/{total_segments}...")
                        append_original(seg, synthesized=False)
                    elif self._should_isolate_reconstruction_segment(seg_text, target_duration):
                        flush_batch(idx)
                        synthesize_isolated_item(item, idx + 1)
                    else:
                        if batch_items and (
                            (batch_cost + item_cost) > batch_token_budget
                            or (batch_duration + target_duration) > batch_duration_budget
                        ):
                            flush_batch(idx)
                        batch_items.append(item)
                        batch_cost += item_cost
                        batch_duration += float(target_duration)
                    if batch_items and (
                        len(batch_items) >= batch_size
                        or batch_cost >= batch_token_budget
                        or batch_duration >= batch_duration_budget
                    ):
                        flush_batch(idx + 1)
                else:
                    flush_batch(idx)
                    if job_id:
                        progress = 15 + int(((idx + 1) / total_segments) * 75)
                        self._update_job_progress(job_id, progress)
                        self._update_job_status_detail(job_id, f"Copying original segment {idx + 1}/{total_segments}...")
                    append_original(seg, synthesized=False)

            flush_batch(total_segments)

            total_duration = max(float(transcript_segments[-1].end_time), max((row["end_time"] for row in sidecar_rows), default=0.0))
            assembled = self._assemble_reconstructed_audio(sample_rate, total_duration, placements)

            if job_id:
                self._update_job_progress(job_id, 94)
                self._update_job_status_detail(job_id, "Writing reconstructed studio mix...")

            sf.write(str(output_wav), assembled, sample_rate)
            sidecar_json.write_text(json.dumps(sidecar_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            self._set_reconstruction_state(video_id, status="ready", audio_path=rel_output, error="", model=model_name)
            return output_wav
        except JobPausedException:
            self._set_reconstruction_state(video_id, status="paused", error="", model=model_name)
            raise
        except JobCancelledException:
            raise
        except Exception as e:
            self._set_reconstruction_state(video_id, status="failed", error=str(e), model=model_name)
            raise
        finally:
            try:
                for child in work_dir.glob("*"):
                    child.unlink(missing_ok=True)
                work_dir.rmdir()
            except Exception:
                pass

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

    def _get_ffmpeg_cmd(self):
        """Return path to ffmpeg executable."""
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
        if (ffmpeg_bin / "ffmpeg.exe").exists():
            return str(ffmpeg_bin / "ffmpeg.exe")
        return "ffmpeg"

    def _get_ffprobe_cmd(self):
        """Return path to ffprobe executable."""
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
        import tempfile

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
        import tempfile
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

    def _cast_pipeline_to_float32(self, pipeline):
        """Cast pyannote pipeline sub-models to float32 for GPU compatibility.
        
        The pyannote Pipeline/SpeakerDiarization class does NOT inherit from
        torch.nn.Module and has no .float() method. Instead we cast the
        individual sub-models (_segmentation, _embedding) which ARE nn.Modules.
        """
        import torch
        for attr in ('_segmentation', '_embedding'):
            model = getattr(pipeline, attr, None)
            if model is not None and hasattr(model, 'float'):
                model.float()
                log_verbose(f"  Cast pipeline.{attr} to float32")
        torch.set_float32_matmul_precision('high')

    def _detect_nvidia_gpu_name(self) -> str | None:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode != 0:
                return None
            names = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
            return names[0] if names else None
        except Exception:
            return None

    def _ensure_device(self):
        import torch
        if self.device is None:
            if self._cuda_unhealthy_reason:
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log(f"Using device: {self.device}")
            if self.device == "cpu" and not self._cuda_unhealthy_reason:
                gpu_name = self._detect_nvidia_gpu_name()
                if gpu_name:
                    try:
                        import importlib.metadata as importlib_metadata
                        torch_version = importlib_metadata.version("torch")
                    except Exception:
                        torch_version = "unknown"
                    try:
                        import importlib.metadata as importlib_metadata
                        torchaudio_version = importlib_metadata.version("torchaudio")
                    except Exception:
                        torchaudio_version = "unknown"
                    log(
                        f"NVIDIA GPU detected ({gpu_name}) but torch CUDA is unavailable. "
                        f"Installed torch={torch_version}, torchaudio={torchaudio_version}, "
                        f"torch.version.cuda={getattr(torch.version, 'cuda', None)}. "
                        "The backend is running transcription on CPU, which will be much slower."
                    )
            if self.device == "cuda":
                try:
                    log_verbose(f"  GPU: {torch.cuda.get_device_name(0)}")
                    props = torch.cuda.get_device_properties(0)
                    self._gpu_total_vram_bytes = int(getattr(props, "total_memory", 0) or 0)
                    if self._gpu_total_vram_bytes > 0:
                        log_verbose(f"  VRAM total: {self._gpu_total_vram_bytes / (1024 ** 3):.1f} GB")
                except Exception as e:
                    msg = str(e or "")
                    if "cuda allocator config" in msg.lower() or "unrecognized key" in msg.lower():
                        bad_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                        log(
                            "CUDA initialization failed due to invalid allocator config. "
                            f"Cleared PYTORCH_CUDA_ALLOC_CONF={bad_conf!r} and falling back to CPU for this process."
                        )
                        self.device = "cpu"
                    self._gpu_total_vram_bytes = 0

    def _cuda_memory_snapshot(self) -> dict:
        """Best-effort snapshot of CUDA memory (bytes)."""
        snap = {"free": 0, "total": 0, "allocated": 0, "reserved": 0}
        if self.device != "cuda":
            return snap
        try:
            import torch
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                snap["free"] = int(free_b or 0)
                snap["total"] = int(total_b or 0)
            except Exception:
                snap["free"] = 0
                snap["total"] = int(self._gpu_total_vram_bytes or 0)
            snap["allocated"] = int(torch.cuda.memory_allocated(0) or 0)
            snap["reserved"] = int(torch.cuda.memory_reserved(0) or 0)
        except Exception:
            pass
        return snap

    def _record_cuda_health_event(self, label: str, job_id: int = None, extra: dict | None = None):
        snap = self._cuda_memory_snapshot()
        entry = {
            "ts": datetime.now().isoformat(),
            "label": str(label or "unknown"),
            "job_id": int(job_id) if job_id else None,
            "device": self.device or "unknown",
            **self._snap_to_gb_dict(snap),
            "parakeet_dynamic_batch_cap": (
                int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else None
            ),
            "cuda_unhealthy": bool(self._cuda_unhealthy_reason),
            "cuda_unhealthy_reason": self._cuda_unhealthy_reason,
            "cuda_degraded_reason": self._cuda_degraded_reason,
            "cuda_fault_count_this_worker": int(self._cuda_fault_count),
            "cuda_soft_reset_count": int(self._cuda_soft_reset_count),
            "cuda_oom_backoff_count": int(self._cuda_oom_backoff_count),
        }
        if isinstance(extra, dict) and extra:
            entry.update(extra)
        with self._cuda_health_guard:
            self._cuda_health_events.append(entry)
            if len(self._cuda_health_events) > 60:
                self._cuda_health_events = self._cuda_health_events[-60:]
        return entry

    def _recent_cuda_health_events(self, limit: int = 12) -> list[dict]:
        with self._cuda_health_guard:
            items = list(self._cuda_health_events[-max(1, int(limit)):])
        return items

    def _evaluate_cuda_degradation(self, label: str = "probe", job_id: int = None) -> str | None:
        if self.device != "cuda" or self._cuda_recovery_pending:
            return None

        snap = self._cuda_memory_snapshot()
        free_b, total_b, allocated_b, reserved_b, free_gb, free_ratio = self._snap_unpack(snap)
        if total_b <= 0:
            return None

        reserved_ratio = float(reserved_b) / float(total_b) if reserved_b > 0 else 0.0
        allocated_gb = float(allocated_b) / (1024 ** 3) if allocated_b > 0 else 0.0
        reserved_gb = float(reserved_b) / (1024 ** 3) if reserved_b > 0 else 0.0

        recent = [
            e for e in self._recent_cuda_health_events(limit=10)
            if str(e.get("device") or "") == "cuda" and float(e.get("total_gb") or 0) > 0
        ]
        recent_peak_free = max([float(e.get("free_gb") or 0.0) for e in recent], default=free_gb)

        free_drop_threshold_gb = _env_float("CUDA_DEGRADE_FREE_DROP_GB", "3.0")
        min_free_ratio = _env_float("CUDA_DEGRADE_MIN_FREE_RATIO", "0.28")
        reserved_ratio_threshold = _env_float("CUDA_DEGRADE_RESERVED_RATIO", "0.50")
        low_headroom_gb = _env_float("CUDA_DEGRADE_LOW_HEADROOM_GB", "8.0")

        reasons: list[str] = []
        dynamic_cap = int(self._parakeet_dynamic_batch_cap or 0)
        if recent_peak_free - free_gb >= free_drop_threshold_gb and free_ratio <= min_free_ratio:
            reasons.append(
                f"free_vram_drop={recent_peak_free - free_gb:.1f}GB (now {free_gb:.1f}GB, free_ratio={free_ratio:.2f})"
            )
        if reserved_ratio >= reserved_ratio_threshold and reserved_b > allocated_b + (1024 ** 3):
            reasons.append(
                f"reserved_ratio={reserved_ratio:.2f} with reserved {reserved_gb:.1f}GB > allocated {allocated_gb:.1f}GB + 1GB"
            )
        if dynamic_cap <= 1 and free_gb <= low_headroom_gb:
            reasons.append(
                f"parakeet_cap={dynamic_cap or 1} under low_headroom={free_gb:.1f}GB"
            )
        if self._cuda_oom_backoff_count >= 2 and free_ratio <= 0.35:
            reasons.append(
                f"repeated_oom_backoff={self._cuda_oom_backoff_count} with free_ratio={free_ratio:.2f}"
            )

        if not reasons:
            self._cuda_degraded_reason = None
            return None

        reason = "; ".join(reasons)[:500]
        self._cuda_degraded_reason = reason
        self._record_cuda_health_event(
            f"{label}_degraded",
            job_id=job_id,
            extra={
                "degradation_reason": reason,
                "degradation_free_ratio": round(free_ratio, 3),
                "degradation_reserved_ratio": round(reserved_ratio, 3),
            },
        )
        self._upsert_job_payload_fields(
            job_id,
            {
                "cuda_degraded": True,
                "cuda_degraded_reason": reason,
                "cuda_free_gb_before_reset": round(free_gb, 2),
                "cuda_reserved_gb_before_reset": round(reserved_gb, 2),
                "cuda_allocated_gb_before_reset": round(allocated_gb, 2),
            },
        )
        return reason

    def _soft_reset_cuda_if_degraded(self, label: str = "pre_parakeet", job_id: int = None) -> bool:
        reason = self._evaluate_cuda_degradation(label=label, job_id=job_id)
        if not reason:
            return False

        log(
            "CUDA degradation detected before Parakeet work. "
            f"Performing a worker soft reset to recover headroom ({reason})."
        )
        self._cuda_soft_reset_count += 1
        self.purge_loaded_models(reason=f"cuda_soft_reset:{label}")
        self._ensure_device()
        if self.device == "cuda":
            self._apply_cuda_memory_fraction_limit()
        self._record_cuda_health_event(
            f"{label}_soft_reset",
            job_id=job_id,
            extra={"soft_reset_reason": reason},
        )
        self._upsert_job_payload_fields(
            job_id,
            {
                "cuda_soft_reset_applied": True,
                "cuda_soft_reset_reason": reason,
                "cuda_soft_reset_count": int(self._cuda_soft_reset_count),
            },
        )
        self._cuda_degraded_reason = None
        return True

    def get_cuda_health_status(self) -> dict:
        self._ensure_device()
        snap = self._cuda_memory_snapshot()
        process_mem = self._process_memory_snapshot()
        restart_state = self._read_cuda_restart_state()
        component_estimates = self._get_component_memory_estimates()
        def _to_gb(value: int) -> float:
            return round(float(value or 0) / (1024 ** 3), 2) if value else 0.0
        return {
            "device": self.device,
            "cuda_unhealthy": bool(self._cuda_unhealthy_reason),
            "cuda_unhealthy_reason": self._cuda_unhealthy_reason,
            "cuda_degraded_reason": self._cuda_degraded_reason,
            "cuda_recovery_pending": bool(self._cuda_recovery_pending),
            "cuda_fault_count_this_worker": int(self._cuda_fault_count),
            "cuda_soft_reset_count": int(self._cuda_soft_reset_count),
            "cuda_oom_backoff_count": int(self._cuda_oom_backoff_count),
            "parakeet_dynamic_batch_cap": (
                int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else None
            ),
            "memory": self._snap_to_gb_dict(snap),
            "system_memory": {
                "rss_gb": _to_gb(int(process_mem.get("rss") or 0)),
                "total_gb": _to_gb(int(process_mem.get("total") or 0)),
                "available_gb": _to_gb(int(process_mem.get("available") or 0)),
            },
            "component_memory": {
                key: {
                    "loaded": (
                        self.parakeet_model is not None if key == "parakeet"
                        else self.whisper_model is not None if key == "whisper"
                        else any([
                            self.diarization_pipeline is not None,
                            self.embedding_model is not None,
                            self.embedding_inference is not None,
                        ]) if key == "pyannote"
                        else bool(item.get("loaded"))
                    ),
                    "ram_gb": _to_gb(int(item.get("ram_bytes") or 0)),
                    "vram_gb": _to_gb(int(item.get("vram_bytes") or 0)),
                }
                for key, item in component_estimates.items()
            },
            "recent_events": self._recent_cuda_health_events(limit=12),
            "auto_restart_count": len(restart_state.get("restart_timestamps", [])),
            "auto_restart_limit": CUDA_MAX_AUTO_RESTARTS,
            "permanent_cpu_mode": bool(restart_state.get("permanent_cpu_mode")),
            "last_restart_reason": restart_state.get("last_restart_reason"),
        }

    def _format_gb(self, value_bytes: int) -> str:
        try:
            return f"{float(value_bytes) / (1024 ** 3):.1f}GB"
        except Exception:
            return "unknown"

    def _snap_to_gb_dict(self, snap: dict) -> dict:
        """Convert a raw CUDA memory snapshot to a dict with GB values."""
        def _gb(key: str) -> float:
            v = snap.get(key) or 0
            return round(float(v) / (1024 ** 3), 2) if v else 0.0
        return {"free_gb": _gb("free"), "total_gb": _gb("total"), "allocated_gb": _gb("allocated"), "reserved_gb": _gb("reserved")}

    def _snap_unpack(self, snap: dict) -> tuple:
        """Unpack a raw CUDA memory snapshot into (free_b, total_b, allocated_b, reserved_b, free_gb, free_ratio)."""
        free_b = int(snap.get("free") or 0)
        total_b = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
        allocated_b = int(snap.get("allocated") or 0)
        reserved_b = int(snap.get("reserved") or 0)
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
        free_ratio = (float(free_b) / float(total_b)) if total_b > 0 and free_b > 0 else 0.0
        return free_b, total_b, allocated_b, reserved_b, free_gb, free_ratio

    def _is_cuda_oom(self, error: Exception) -> bool:
        msg = str(error or "").lower()
        return (
            ("cuda" in msg and "out of memory" in msg)
            or ("cuda oom" in msg)
            or ("oom even at batch_size" in msg)
            or ("cudnn_status_alloc_failed" in msg)
            or ("cuda error: out of memory" in msg)
        )

    def _is_cuda_illegal_access(self, error: Exception) -> bool:
        msg = str(error or "").lower()
        return (
            ("illegal memory access" in msg)
            or ("cudaerrorillegaladdress" in msg)
            or ("device-side assert triggered" in msg)
        )

    def _mark_cuda_unhealthy(self, reason: str, job_id: int = None):
        """Quarantine only the current job after a fatal CUDA runtime fault.

        Certain CUDA faults (e.g. illegal memory access) can poison the context until
        process restart. We isolate the current job onto CPU fallback, then attempt
        an explicit GPU recovery before the next queued job is claimed.
        """
        self._cuda_unhealthy_reason = (reason or "unknown").strip()[:400]
        self._cuda_unhealthy_since = datetime.now()
        self._cuda_recovery_pending = True
        self._cuda_fault_count += 1
        self._cuda_consecutive_fault_count += 1
        self._parakeet_dynamic_batch_cap = 1
        self._record_cuda_health_event(
            "cuda_fault_quarantine",
            job_id=job_id,
            extra={"fault_reason": self._cuda_unhealthy_reason},
        )

        # Drop GPU-bound models and force this job onto CPU fallback.
        self._release_parakeet_model("cuda_fault_job_quarantine")
        self._release_whisper_model("cuda_fault_job_quarantine")
        self._release_diarization_models("cuda_fault_job_quarantine")
        self.device = "cpu"
        self._gpu_total_vram_bytes = 0
        self._cuda_memory_fraction_applied = None

        self._upsert_job_payload_fields(
            job_id,
            {
                "cuda_unhealthy": True,
                "cuda_unhealthy_reason": self._cuda_unhealthy_reason,
                "cuda_unhealthy_since": self._cuda_unhealthy_since.isoformat() if self._cuda_unhealthy_since else None,
                "cuda_job_quarantined": True,
                "cuda_recovery_pending": True,
                "cuda_fault_count_this_worker": int(self._cuda_fault_count),
                "parakeet_dynamic_batch_cap": 1,
                "parakeet_dynamic_batch_cap_source": "illegal_access_quarantine",
            },
        )
        log(
            f"Parakeet abandoned for job {job_id or 'unknown'} due to fatal CUDA runtime error: "
            f"{self._cuda_unhealthy_reason}. Quarantining this job to Whisper/CPU and scheduling "
            f"GPU recovery before the next job."
        )

    def _recover_cuda_after_fault_if_needed(self):
        """Try to restore GPU execution after a job-local CUDA fault quarantine."""
        if not self._cuda_recovery_pending:
            return True

        previous_reason = self._cuda_unhealthy_reason or "unknown CUDA fault"
        log(
            "Attempting CUDA recovery after job-local Parakeet fault. "
            f"Previous reason: {previous_reason}"
        )

        try:
            self._release_parakeet_model("post_fault_recovery")
            self._release_whisper_model("post_fault_recovery")
            self._release_diarization_models("post_fault_recovery")
            gc.collect()

            # Thorough GPU state cleanup: synchronize pending ops and release
            # all cached/IPC memory before re-probing the device.
            # Use timeout-protected sync — bare synchronize() can hang on a
            # poisoned CUDA context.
            if not self._safe_cuda_sync(timeout_s=15.0):
                raise RuntimeError("CUDA synchronize hung or raised illegal access during recovery")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

            self.device = None
            self._gpu_total_vram_bytes = 0
            self._cuda_memory_fraction_applied = None
            self._cuda_unhealthy_reason = None
            self._cuda_unhealthy_since = None
            self._ensure_device()

            if self.device != "cuda":
                self._cuda_recovery_pending = False
                log("CUDA recovery result: GPU not available after reset probe. Continuing on CPU.")
                return False

            self._apply_cuda_memory_fraction_limit()

            # Probe test: allocate a small tensor on GPU with timeout protection.
            # If this hangs, the CUDA context is still corrupted.
            import torch
            probe_ok = [False]
            def _probe():
                try:
                    p = torch.empty((1,), device="cuda")
                    del p
                    probe_ok[0] = True
                except Exception:
                    pass
            pt = threading.Thread(target=_probe, daemon=True)
            pt.start()
            pt.join(timeout=10.0)
            if not probe_ok[0]:
                raise RuntimeError("CUDA probe tensor allocation failed or hung during recovery")
            self._clear_cuda_cache()
            self._cuda_recovery_pending = False

            # Reset fault counters so the worker returns to normal (non-degraded)
            # Parakeet behavior. Without this, stale counters force aggressive
            # chunked mode + per-chunk model recycling that paradoxically makes
            # subsequent faults more likely.
            self._cuda_fault_count = 0
            self._cuda_oom_backoff_count = 0
            self._cuda_degraded_reason = None
            self._parakeet_dynamic_batch_cap = None

            self._record_cuda_health_event("cuda_recovery_succeeded")
            log("CUDA recovery succeeded. Reset fault counters — Parakeet GPU execution re-enabled for subsequent jobs.")
            return True
        except Exception as e:
            self._cuda_unhealthy_reason = f"{previous_reason} | recovery failed: {str(e)[:220]}"
            self._cuda_unhealthy_since = datetime.now()
            self._cuda_recovery_pending = True
            self.device = "cpu"
            self._gpu_total_vram_bytes = 0
            self._cuda_memory_fraction_applied = None
            self._record_cuda_health_event(
                "cuda_recovery_failed",
                extra={"recovery_error": str(e)[:220]},
            )
            if self._can_auto_restart():
                log("CUDA recovery failed — triggering automatic process restart.")
                self._trigger_auto_restart(f"CUDA recovery failed: {str(e)[:200]}")
            else:
                log(
                    "CUDA recovery failed after a fatal Parakeet fault. "
                    f"Keeping this worker on CPU until manual backend restart. Reason: {e}"
                )
            return False

    def _safe_cuda_sync(self, timeout_s: float = 10.0) -> bool:
        """Run torch.cuda.synchronize() with a timeout guard.

        Returns True if sync completed normally, False if it hung or raised
        an illegal-access error (indicating a poisoned CUDA context).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return True
        except Exception:
            return True

        result = [False]
        error = [None]

        def _sync():
            try:
                torch.cuda.synchronize()
                result[0] = True
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_sync, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            log(f"torch.cuda.synchronize() hung for {timeout_s}s — CUDA context likely corrupted")
            return False
        if error[0] is not None:
            if self._is_cuda_illegal_access(error[0]):
                log(f"CUDA sync raised illegal access: {error[0]}")
                return False
            log(f"CUDA sync raised non-fatal error: {error[0]}")
        return result[0]

    # ── CUDA auto-restart state management ───────────────────────────────

    def _read_cuda_restart_state(self) -> dict:
        """Read restart tracking state from disk (survives process restarts)."""
        try:
            if CUDA_RESTART_STATE_FILE.exists():
                data = json.loads(CUDA_RESTART_STATE_FILE.read_text(encoding="utf-8"))
                cutoff = time.time() - CUDA_RESTART_WINDOW_SECONDS
                data["restart_timestamps"] = [
                    ts for ts in (data.get("restart_timestamps") or [])
                    if ts > cutoff
                ]
                return data
        except Exception:
            pass
        return {"restart_timestamps": [], "permanent_cpu_mode": False}

    def _write_cuda_restart_state(self, state: dict):
        try:
            CUDA_RESTART_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CUDA_RESTART_STATE_FILE.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log(f"Failed to write CUDA restart state: {e}")

    def _can_auto_restart(self) -> bool:
        state = self._read_cuda_restart_state()
        if state.get("permanent_cpu_mode"):
            return False
        return len(state.get("restart_timestamps", [])) < CUDA_MAX_AUTO_RESTARTS

    def _trigger_auto_restart(self, reason: str):
        """Trigger process restart via uvicorn --reload file touch."""
        state = self._read_cuda_restart_state()
        state["restart_timestamps"] = state.get("restart_timestamps", []) + [time.time()]
        state["last_restart_reason"] = reason

        if len(state["restart_timestamps"]) >= CUDA_MAX_AUTO_RESTARTS:
            state["permanent_cpu_mode"] = True
            state["permanent_cpu_since"] = datetime.now().isoformat()
            self._write_cuda_restart_state(state)
            log(
                f"CUDA auto-restart limit reached ({CUDA_MAX_AUTO_RESTARTS} restarts in "
                f"{CUDA_RESTART_WINDOW_SECONDS}s). Staying on CPU until manual restart."
            )
            self._record_cuda_health_event("cuda_restart_limit_reached", extra={"reason": reason})
            return

        self._write_cuda_restart_state(state)
        restart_count = len(state["restart_timestamps"])
        self._record_cuda_health_event(
            "cuda_auto_restart_triggered",
            extra={"reason": reason, "restart_count": restart_count},
        )
        log(f"Triggering automatic process restart ({restart_count}/{CUDA_MAX_AUTO_RESTARTS}) "
            f"due to unrecoverable CUDA fault: {reason}")

        # Signal restart: touch main.py for --reload mode, then exit the process.
        # In non-reload mode (run_windows.bat), the exit code tells the wrapper to respawn.
        main_py = Path(__file__).parent.parent / "main.py"
        try:
            main_py.touch()
        except Exception:
            pass
        # Write a marker file so the wrapper script (run_windows.bat) knows to restart
        restart_marker = RUNTIME_DIR / "cuda_restart_requested"
        try:
            restart_marker.write_text(reason[:200], encoding="utf-8")
        except Exception:
            pass
        log("Exiting process for CUDA restart (exit code 75)...")
        # Give logs a moment to flush
        time.sleep(1)
        os._exit(75)

    def _clear_cuda_cache(self):
        if self.device != "cuda":
            return
        try:
            import torch
            try:
                # Avoid freeing/recycling CUDA allocations while kernels are still
                # in flight (can trigger hard native aborts on Windows).
                torch.cuda.synchronize()
            except Exception:
                pass
            # NeMo's batch decoding (parakeet-tdt) uses CUDA graphs. Calling
            # empty_cache() while graphs hold tensor references causes
            # cudaErrorIllegalAddress on the next inference run.
            # See: https://github.com/NVIDIA-NeMo/NeMo/issues/14727
            self._disable_nemo_cuda_graphs()
            gc.collect()
            torch.cuda.empty_cache()
            self._enable_nemo_cuda_graphs()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass
        except Exception:
            pass

    def _disable_nemo_cuda_graphs(self):
        """Disable CUDA graphs on the loaded Parakeet model before clearing cache."""
        try:
            if self.parakeet_model is not None and hasattr(self.parakeet_model, 'disable_cuda_graphs'):
                self.parakeet_model.disable_cuda_graphs()
        except Exception:
            pass

    def _enable_nemo_cuda_graphs(self):
        """Re-enable CUDA graphs on the loaded Parakeet model after clearing cache."""
        try:
            if self.parakeet_model is not None and hasattr(self.parakeet_model, 'enable_cuda_graphs'):
                self.parakeet_model.enable_cuda_graphs()
        except Exception:
            pass

    def _log_cuda_memory(self, label: str, job_id: int = None):
        if self.device != "cuda":
            return
        snap = self._cuda_memory_snapshot()
        free_b, total_b, allocated_b, reserved_b, _, _ = self._snap_unpack(snap)
        log_verbose(
            f"{label}: free {self._format_gb(free_b)} / total {self._format_gb(total_b)} "
            f"(alloc {self._format_gb(allocated_b)}, resv {self._format_gb(reserved_b)})"
        )
        if job_id:
            gb = self._snap_to_gb_dict(snap)
            self._upsert_job_payload_fields(
                job_id,
                {f"{label}_cuda_{k}": v for k, v in gb.items()},
            )

    def _move_module_to_cpu(self, module):
        if module is None:
            return
        try:
            if self._module_has_meta_tensors(module):
                log_verbose("Skipping CPU move for meta-backed module during release.")
                return
        except Exception:
            pass
        try:
            if hasattr(module, "to"):
                module.to("cpu")
                return
        except Exception:
            pass
        try:
            if hasattr(module, "cpu"):
                module.cpu()
        except Exception:
            pass

    def _module_has_meta_tensors(self, module) -> bool:
        if module is None:
            return False
        try:
            for param in module.parameters():
                if bool(getattr(param, "is_meta", False)):
                    return True
        except Exception:
            pass
        try:
            for buffer in module.buffers():
                if bool(getattr(buffer, "is_meta", False)):
                    return True
        except Exception:
            pass
        return False

    def _apply_cuda_memory_fraction_limit(self):
        """Cap process CUDA memory to avoid spilling into shared memory on WDDM."""
        if self.device != "cuda":
            return
        raw = (os.getenv("PARAKEET_MAX_GPU_MEMORY_FRACTION") or "").strip()
        if raw:
            try:
                fraction = float(raw)
            except Exception:
                fraction = 0.85
        else:
            total = int(self._gpu_total_vram_bytes or self._cuda_memory_snapshot().get("total") or 0)
            total_gb = float(total) / (1024 ** 3) if total > 0 else 0.0
            if total_gb >= 28.0:
                fraction = 0.92
            elif total_gb >= 20.0:
                fraction = 0.88
            else:
                fraction = 0.85
        fraction = max(0.50, min(fraction, 0.98))
        if self._cuda_memory_fraction_applied is not None and abs(self._cuda_memory_fraction_applied - fraction) < 1e-6:
            return
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            self._cuda_memory_fraction_applied = fraction
            log_verbose(f"Applied CUDA per-process memory fraction limit: {fraction:.2f}")
        except Exception as e:
            log_verbose(f"Could not apply CUDA memory fraction limit: {e}")

    def _release_parakeet_model(self, reason: str = "", job_id: int = None):
        if self.parakeet_model is None:
            return
        model = self.parakeet_model
        self._log_cuda_memory("pre_release_parakeet", job_id=job_id)
        self.parakeet_model = None
        self._move_module_to_cpu(model)
        del model
        self._clear_cuda_cache()
        self._log_cuda_memory("post_release_parakeet", job_id=job_id)
        self._set_component_memory_unloaded("parakeet")
        if reason:
            log(f"Released Parakeet model from GPU memory ({reason}).")
        else:
            log("Released Parakeet model from GPU memory.")

    def _release_whisper_model(self, reason: str = "", job_id: int = None):
        if self.whisper_model is None:
            return
        model = self.whisper_model
        self._log_cuda_memory("pre_release_whisper", job_id=job_id)
        self.whisper_model = None
        self._whisper_compute_type = None
        self._whisper_device = None
        self._whisper_backend = None
        self._whisper_model_cache_key = None
        try:
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                self._move_module_to_cpu(inner_model)
        except Exception:
            pass
        del model
        self._clear_cuda_cache()
        self._log_cuda_memory("post_release_whisper", job_id=job_id)
        self._set_component_memory_unloaded("whisper")
        if reason:
            log(f"Released Whisper model from GPU memory ({reason}).")
        else:
            log("Released Whisper model from GPU memory.")

    def _release_diarization_models(self, reason: str = "", job_id: int = None):
        """Release pyannote models/inference objects to recover GPU memory."""
        had_models = any([
            self.diarization_pipeline is not None,
            self.embedding_model is not None,
            self.embedding_inference is not None,
        ])
        pipeline = self.diarization_pipeline
        embedding_model = self.embedding_model
        embedding_inference = self.embedding_inference
        if had_models:
            self._log_cuda_memory("pre_release_diarization", job_id=job_id)
        self.diarization_pipeline = None
        self.embedding_model = None
        self.embedding_inference = None
        if had_models:
            self._move_module_to_cpu(pipeline)
            self._move_module_to_cpu(embedding_model)
            try:
                if hasattr(embedding_inference, "to"):
                    embedding_inference.to("cpu")
            except Exception:
                pass
            del pipeline
            del embedding_model
            del embedding_inference
            self._clear_cuda_cache()
            self._log_cuda_memory("post_release_diarization", job_id=job_id)
            self._set_component_memory_unloaded("pyannote")
            if reason:
                log(f"Released diarization/embedding models from GPU memory ({reason}).")
            else:
                log("Released diarization/embedding models from GPU memory.")

    def purge_loaded_models(self, reason: str = "manual"):
        """Best-effort runtime purge of loaded ML models and CUDA fault state.

        This is used by restart/reload controls so a user can recover GPU memory
        and allow Parakeet retries without needing a full machine reboot.
        """
        had_whisper = self.whisper_model is not None
        had_parakeet = self.parakeet_model is not None
        had_diar = any([
            self.diarization_pipeline is not None,
            self.embedding_model is not None,
            self.embedding_inference is not None,
        ])

        self._release_parakeet_model(reason)
        self._release_whisper_model(reason)
        self._release_diarization_models(reason)

        # Reset runtime state so next job re-detects device and can attempt
        # Parakeet again after a previous CUDA unhealthy fallback.
        self._force_float32 = False
        self._whisper_compute_type = None
        self._whisper_device = None
        self._parakeet_dynamic_batch_cap = None
        self._cuda_unhealthy_reason = None
        self._cuda_unhealthy_since = None
        self._cuda_recovery_pending = False
        self._cuda_fault_count = 0
        self._cuda_degraded_reason = None
        self._cuda_oom_backoff_count = 0
        self.device = None
        self._gpu_total_vram_bytes = 0
        self._cuda_memory_fraction_applied = None
        gc.collect()
        self._record_cuda_health_event(f"purge_loaded_models:{reason}")

        return {
            "purged_whisper": had_whisper,
            "purged_parakeet": had_parakeet,
            "purged_diarization": had_diar,
        }

    def _maybe_recover_cuda_headroom(self, baseline_free_b: int, job_id: int = None):
        if self.device != "cuda":
            return
        snap = self._cuda_memory_snapshot()
        free_b = int(snap.get("free") or 0)
        total_b = int(snap.get("total") or 0)
        reserved_b = int(snap.get("reserved") or 0)
        allocated_b = int(snap.get("allocated") or 0)
        free_drop_b = max(0, int(baseline_free_b or 0) - free_b)
        free_drop_gb = float(free_drop_b) / (1024 ** 3) if free_drop_b > 0 else 0.0
        reserved_gb = float(reserved_b) / (1024 ** 3) if reserved_b > 0 else 0.0
        allocated_gb = float(allocated_b) / (1024 ** 3) if allocated_b > 0 else 0.0
        self._upsert_job_payload_fields(
            job_id,
            {
                "job_cuda_free_gb_end": round(float(free_b) / (1024 ** 3), 2) if free_b > 0 else 0.0,
                "job_cuda_free_drop_gb": round(free_drop_gb, 2),
                "job_cuda_reserved_gb_end": round(reserved_gb, 2),
                "job_cuda_allocated_gb_end": round(allocated_gb, 2),
            },
        )

        recover = False
        reason_bits = []
        if free_drop_gb >= _env_float("CUDA_HEADROOM_RECOVERY_DROP_GB", "1.5"):
            recover = True
            reason_bits.append(f"free_drop={free_drop_gb:.1f}GB")
        if reserved_b > 0 and total_b > 0:
            reserved_ratio = float(reserved_b) / float(total_b)
            if reserved_ratio >= _env_float("CUDA_HEADROOM_RECOVERY_RESERVED_RATIO", "0.45"):
                recover = True
                reason_bits.append(f"reserved_ratio={reserved_ratio:.2f}")
        if reserved_b > allocated_b + (1024 ** 3):
            recover = True
            reason_bits.append("reserved_gt_allocated+1GB")

        if not recover:
            return

        reason = ",".join(reason_bits) if reason_bits else "post_job_headroom_recovery"
        log(
            "CUDA headroom did not rebound after job cleanup; purging loaded models "
            f"to recover allocator state ({reason})."
        )
        self._record_cuda_health_event(
            "post_job_headroom_recovery",
            job_id=job_id,
            extra={"recovery_reason": reason},
        )
        self.purge_loaded_models(reason=f"post_job_headroom_recovery:{reason}")

    def _resolve_parakeet_batch_size(self, requested_batch_size: int) -> int:
        requested = max(1, min(int(requested_batch_size or 1), 64))
        dynamic_cap = self._parakeet_dynamic_batch_cap
        if dynamic_cap is not None:
            try:
                requested = min(requested, max(1, int(dynamic_cap)))
            except Exception:
                pass
        if self.device != "cuda":
            return requested
        hard_cap = max(1, min(int(os.getenv("PARAKEET_BATCH_HARD_MAX", "4")), 64))
        auto_enabled = os.getenv("PARAKEET_BATCH_AUTO", "true").strip().lower() == "true"
        if not auto_enabled:
            return min(requested, hard_cap)

        total = int(self._gpu_total_vram_bytes or 0)
        if total <= 0:
            snap = self._cuda_memory_snapshot()
            total = int(snap.get("total") or 0)
        total_gb = (total / (1024 ** 3)) if total > 0 else 0.0

        if total_gb <= 0:
            return requested
        if total_gb <= 6:
            cap = 1
        elif total_gb <= 8:
            cap = 2
        elif total_gb <= 10:
            cap = 3
        elif total_gb <= 12:
            cap = 4
        elif total_gb <= 16:
            cap = 6
        elif total_gb <= 24:
            cap = 8
        elif total_gb <= 32:
            cap = 12
        else:
            cap = 16

        snap = self._cuda_memory_snapshot()
        free_b = int(snap.get("free") or 0)
        if total > 0 and free_b > 0:
            free_ratio = float(free_b) / float(total)
            if free_ratio < 0.20:
                cap = min(cap, 2)
            elif free_ratio < 0.30:
                cap = min(cap, 4)
            elif free_ratio < 0.40:
                cap = min(cap, 6)

        return max(1, min(requested, cap, hard_cap))

    def _record_parakeet_oom_batch_cap(self, next_batch: int, job_id: int = None):
        """Persist a lower Parakeet batch cap for subsequent jobs in this process."""
        try:
            new_cap = max(1, min(int(next_batch), 64))
        except Exception:
            return
        self._cuda_oom_backoff_count += 1
        prev = self._parakeet_dynamic_batch_cap
        if prev is None or new_cap < int(prev):
            self._parakeet_dynamic_batch_cap = new_cap
            self._upsert_job_payload_fields(
                job_id,
                {
                    "parakeet_dynamic_batch_cap": int(new_cap),
                    "parakeet_dynamic_batch_cap_source": "oom_backoff",
                    "cuda_oom_backoff_count": int(self._cuda_oom_backoff_count),
                },
            )
            self._record_cuda_health_event(
                "parakeet_oom_backoff",
                job_id=job_id,
                extra={"next_batch": int(new_cap)},
            )
            if prev is None:
                log(f"Persisting Parakeet batch cap at {new_cap} after CUDA OOM.")
            else:
                log(f"Lowering persisted Parakeet batch cap from {int(prev)} to {new_cap} after CUDA OOM.")

    def _resolve_parakeet_keep_loaded_thresholds(self, total_gb: float) -> tuple[float, float]:
        """Return (min_free_gb, min_free_ratio) thresholds for keeping Parakeet loaded."""
        if total_gb >= 36:
            default_gb, default_ratio = 6.0, 0.12
        elif total_gb >= 28:
            default_gb, default_ratio = 5.0, 0.14
        elif total_gb >= 20:
            default_gb, default_ratio = 4.0, 0.16
        else:
            # Sub-20GB cards are usually better off unloading between episodes.
            default_gb, default_ratio = 999.0, 1.0

        raw_gb = (os.getenv("PARAKEET_KEEP_LOADED_MIN_FREE_GB") or "").strip()
        raw_ratio = (os.getenv("PARAKEET_KEEP_LOADED_MIN_FREE_RATIO") or "").strip()

        min_free_gb = default_gb
        min_free_ratio = default_ratio
        try:
            if raw_gb:
                min_free_gb = max(0.0, float(raw_gb))
        except Exception:
            pass
        try:
            if raw_ratio:
                min_free_ratio = max(0.0, min(1.0, float(raw_ratio)))
        except Exception:
            pass

        return min_free_gb, min_free_ratio

    def _resolve_parakeet_pyannote_coexist_thresholds(self, total_gb: float) -> tuple[float, float]:
        """Return (min_free_gb, min_free_ratio) thresholds for keeping Parakeet + pyannote resident together."""
        if total_gb >= 36:
            default_gb, default_ratio = 12.0, 0.28
        elif total_gb >= 28:
            default_gb, default_ratio = 10.0, 0.30
        elif total_gb >= 20:
            default_gb, default_ratio = 8.0, 0.34
        else:
            default_gb, default_ratio = 999.0, 1.0

        raw_gb = (os.getenv("PARAKEET_PYANNOTE_COEXIST_MIN_FREE_GB") or "").strip()
        raw_ratio = (os.getenv("PARAKEET_PYANNOTE_COEXIST_MIN_FREE_RATIO") or "").strip()

        min_free_gb = default_gb
        min_free_ratio = default_ratio
        try:
            if raw_gb:
                min_free_gb = max(0.0, float(raw_gb))
        except Exception:
            pass
        try:
            if raw_ratio:
                min_free_ratio = max(0.0, min(1.0, float(raw_ratio)))
        except Exception:
            pass

        return min_free_gb, min_free_ratio

    def _can_keep_parakeet_and_diarization_resident(self, job_id: int = None) -> tuple[bool, str, float, float]:
        if self.device != "cuda":
            return False, "non_cuda", 0.0, 0.0
        if self._cuda_fault_count > 0 or self._cuda_recovery_pending:
            return False, "post_fault_conservative", 0.0, 0.0
        if (self._cuda_degraded_reason or "").strip():
            return False, "cuda_degraded", 0.0, 0.0
        if self._cuda_oom_backoff_count > 0:
            return False, "oom_backoff_present", 0.0, 0.0
        if self._parakeet_dynamic_batch_cap is not None and int(self._parakeet_dynamic_batch_cap) <= 1:
            return False, "parakeet_batch_cap_low", 0.0, 0.0

        snap = self._cuda_memory_snapshot()
        total = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
        free_b = int(snap.get("free") or 0)
        total_gb = float(total) / (1024 ** 3) if total > 0 else 0.0
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
        free_ratio = (free_b / total) if total > 0 else 0.0
        min_free_gb, min_free_ratio = self._resolve_parakeet_pyannote_coexist_thresholds(total_gb)

        if free_gb >= min_free_gb and free_ratio >= min_free_ratio:
            return True, "high_headroom_coexist", free_gb, total_gb
        return False, "insufficient_headroom", free_gb, total_gb

    def _should_unload_parakeet_after_transcribe(self, job_id: int = None) -> bool:
        mode = (os.getenv("PARAKEET_UNLOAD_AFTER_TRANSCRIBE") or "auto").strip().lower()
        decision = {"parakeet_unload_mode": mode}
        if self._cuda_fault_count > 0 or self._cuda_recovery_pending:
            decision.update(
                {
                    "parakeet_unload_after_transcribe": True,
                    "parakeet_unload_reason": "post_fault_conservative",
                    "parakeet_cuda_fault_count": int(self._cuda_fault_count),
                    "parakeet_cuda_recovery_pending": bool(self._cuda_recovery_pending),
                }
            )
            self._upsert_job_payload_fields(job_id, decision)
            return True
        if mode in {"1", "true", "yes", "on"}:
            decision.update({"parakeet_unload_after_transcribe": True, "parakeet_unload_reason": "forced_true"})
            self._upsert_job_payload_fields(job_id, decision)
            return True
        if mode in {"0", "false", "no", "off"}:
            decision.update({"parakeet_unload_after_transcribe": False, "parakeet_unload_reason": "forced_false"})
            self._upsert_job_payload_fields(job_id, decision)
            return False

        # auto: keep model loaded when there is enough free VRAM headroom,
        # unload only under memory pressure.
        if self.device != "cuda":
            decision.update({"parakeet_unload_after_transcribe": True, "parakeet_unload_reason": "non_cuda"})
            self._upsert_job_payload_fields(job_id, decision)
            return True

        total = int(self._gpu_total_vram_bytes or 0)
        if total <= 0:
            total = int(self._cuda_memory_snapshot().get("total") or 0)
        if total <= 0:
            decision.update({"parakeet_unload_after_transcribe": True, "parakeet_unload_reason": "unknown_vram"})
            self._upsert_job_payload_fields(job_id, decision)
            return True

        total_gb = float(total) / (1024 ** 3)
        snap = self._cuda_memory_snapshot()
        free_b = int(snap.get("free") or 0)
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
        free_ratio = (float(free_b) / float(total)) if total > 0 and free_b > 0 else 0.0

        # Conservative auto behavior on smaller cards to avoid spill to shared memory.
        if total_gb < 20.0:
            decision.update(
                {
                    "parakeet_unload_after_transcribe": True,
                    "parakeet_unload_reason": "auto_small_gpu",
                    "parakeet_cuda_free_gb_end": round(free_gb, 2),
                    "parakeet_cuda_total_gb": round(total_gb, 2),
                    "parakeet_cuda_free_ratio_end": round(free_ratio, 3),
                }
            )
            self._upsert_job_payload_fields(job_id, decision)
            return True

        keep_min_free_gb, keep_min_free_ratio = self._resolve_parakeet_keep_loaded_thresholds(total_gb)
        keep_loaded = free_gb >= keep_min_free_gb and free_ratio >= keep_min_free_ratio
        decision.update(
            {
                "parakeet_unload_after_transcribe": (not keep_loaded),
                "parakeet_unload_reason": "auto_keep_loaded" if keep_loaded else "auto_low_headroom",
                "parakeet_keep_loaded_min_free_gb": round(keep_min_free_gb, 2),
                "parakeet_keep_loaded_min_free_ratio": round(keep_min_free_ratio, 3),
                "parakeet_cuda_free_gb_end": round(free_gb, 2),
                "parakeet_cuda_total_gb": round(total_gb, 2),
                "parakeet_cuda_free_ratio_end": round(free_ratio, 3),
            }
        )
        self._upsert_job_payload_fields(job_id, decision)

        if keep_loaded:
            log_verbose(
                f"Keeping Parakeet loaded (free {free_gb:.1f}GB/{total_gb:.1f}GB, "
                f"ratio {free_ratio:.2f}, thresholds {keep_min_free_gb:.1f}GB/{keep_min_free_ratio:.2f})."
            )
            return False
        log_verbose(
            f"Unloading Parakeet after transcribe (free {free_gb:.1f}GB/{total_gb:.1f}GB, "
            f"ratio {free_ratio:.2f}, thresholds {keep_min_free_gb:.1f}GB/{keep_min_free_ratio:.2f})."
        )
        return True

    def _should_release_parakeet_before_diarize(self, job_id: int = None) -> bool:
        """Diarization should not compete with a retained Parakeet model for VRAM."""
        if self.parakeet_model is None:
            return False
        if self.device != "cuda":
            return True

        snap = self._cuda_memory_snapshot()
        total = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
        free_b = int(snap.get("free") or 0)
        total_gb = float(total) / (1024 ** 3) if total > 0 else 0.0
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0

        coexist_ok, coexist_reason, _, _ = self._can_keep_parakeet_and_diarization_resident(job_id=job_id)
        if coexist_ok:
            reason = coexist_reason
            release = False
        else:
            reason = coexist_reason
            release = True

        self._upsert_job_payload_fields(
            job_id,
            {
                "parakeet_release_before_diarize": bool(release),
                "parakeet_release_before_diarize_reason": reason,
                "parakeet_release_before_diarize_free_gb": round(free_gb, 2),
                "parakeet_release_before_diarize_total_gb": round(total_gb, 2),
            },
        )
        return release

    def _get_pyannote_batch_size(self) -> int:
        return max(1, int((os.getenv("PYANNOTE_BATCH_SIZE") or "64").strip() or "64"))

    def _set_pyannote_batch_size(self, batch_size: int):
        batch = max(1, int(batch_size))
        if self.diarization_pipeline is not None:
            try:
                self.diarization_pipeline.segmentation_batch_size = batch
            except Exception:
                pass
            try:
                self.diarization_pipeline.embedding_batch_size = batch
            except Exception:
                pass

    def _format_progress_clock(self, seconds: float | int | None) -> str:
        try:
            total = max(0, int(float(seconds or 0)))
        except Exception:
            total = 0
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        if hours > 0:
            return f"{hours}:{minutes:02}:{secs:02}"
        return f"{minutes}:{secs:02}"

    def _update_transcription_stage_progress(
        self,
        job_id: int | None,
        *,
        engine: str,
        completed_seconds: float | None = None,
        total_seconds: float | None = None,
        segments_completed: int | None = None,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        extra_label: str | None = None,
    ) -> None:
        if not job_id:
            return

        engine_label = "Parakeet" if str(engine or "").strip().lower() == "parakeet" else "Whisper"
        detail_parts: list[str] = []
        payload_fields: dict[str, object] = {
            "stage_transcribe_engine": engine_label.lower(),
        }

        completed_pct = None
        if total_seconds is not None:
            try:
                total_val = max(0.0, float(total_seconds))
            except Exception:
                total_val = 0.0
            if total_val > 0:
                payload_fields["stage_transcribe_progress_total_seconds"] = round(total_val, 3)
                if completed_seconds is not None:
                    try:
                        completed_val = max(0.0, min(float(completed_seconds), total_val))
                    except Exception:
                        completed_val = 0.0
                    payload_fields["stage_transcribe_progress_seconds"] = round(completed_val, 3)
                    detail_parts.append(f"{self._format_progress_clock(completed_val)}/{self._format_progress_clock(total_val)}")
                    completed_pct = int(round((completed_val / total_val) * 100))
        elif completed_seconds is not None:
            try:
                completed_val = max(0.0, float(completed_seconds))
                payload_fields["stage_transcribe_progress_seconds"] = round(completed_val, 3)
            except Exception:
                pass

        if chunk_index is not None:
            payload_fields["stage_transcribe_chunk_index"] = int(chunk_index)
            if chunk_total is not None and int(chunk_total) > 0:
                payload_fields["stage_transcribe_chunk_total"] = int(chunk_total)
                detail_parts.append(f"chunk {int(chunk_index)}/{int(chunk_total)}")
            else:
                detail_parts.append(f"chunk {int(chunk_index)}")

        if segments_completed is not None and int(segments_completed) >= 0:
            payload_fields["stage_transcribe_segments_completed"] = int(segments_completed)
            detail_parts.append(f"{int(segments_completed)} segments")

        if extra_label:
            detail_parts.append(str(extra_label).strip())

        detail = f"Transcribing with {engine_label}"
        if detail_parts:
            detail += f" ({', '.join(part for part in detail_parts if part)})"
        detail += "..."

        self._upsert_job_payload_fields(job_id, payload_fields)
        self._update_job_status_detail(job_id, detail)
        if completed_pct is not None:
            self._update_job_progress(job_id, max(0, min(100, int(completed_pct))))

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

    def _ensure_ctranslate2_pkg_resources(self):
        """Install a pkg_resources shim when setuptools is absent or broken.

        ctranslate2, pyannote.audio, and NeMo all import pkg_resources on Windows.
        This is called in __init__ so the shim is in sys.modules before any lazy
        ML import fires â€” regardless of which transcription engine or code path runs.
        """
        import importlib
        import sys
        import types
        import warnings

        if sys.platform != "win32":
            return

        # If a real, functional pkg_resources is present, do nothing.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                import pkg_resources as _pkg_resources  # type: ignore
            if hasattr(_pkg_resources, "resource_filename"):
                return
        except Exception:
            pass

        # If our shim is already installed, do nothing.
        existing = sys.modules.get("pkg_resources")
        if existing is not None and hasattr(existing, "resource_filename"):
            return

        # Build a shim that covers the pkg_resources API surface used by
        # ctranslate2 (resource_filename), pyannote.audio and NeMo
        # (get_distribution, require, iter_entry_points, working_set).
        shim = types.ModuleType("pkg_resources")

        # --- resource_filename ---
        def resource_filename(module_name: str, resource_name: str = "") -> str:
            try:
                module = importlib.import_module(module_name)
                module_file = getattr(module, "__file__", "") or ""
                base_dir = Path(module_file).parent
                return str(base_dir / (resource_name or ""))
            except Exception:
                return resource_name or ""

        # --- get_distribution / require ---
        class _FakeDist:
            def __init__(self, project_name: str, version: str = "0.0.0"):
                self.project_name = project_name
                self.key = project_name.lower().replace("-", "_")
                self.version = version
                self.location = ""
                self.extras = []

            def __str__(self) -> str:
                return f"{self.project_name}=={self.version}"

            def requires(self, _extras=()):
                return []

        def get_distribution(name: str) -> _FakeDist:
            try:
                from importlib.metadata import distribution as _dist
                d = _dist(name)
                return _FakeDist(name, d.version)
            except Exception:
                return _FakeDist(name, "0.0.0")

        def require(_requirements):
            return []

        # --- iter_entry_points ---
        def iter_entry_points(_group, _name=None):
            return iter([])

        # --- working_set (iterable of installed dists) ---
        class _WorkingSet:
            def __iter__(self):
                return iter([])

            def __getitem__(self, key: str) -> _FakeDist:
                return get_distribution(key)

            def by_key(self):
                return {}

        # --- parse_version ---
        def parse_version(v: str):
            try:
                from packaging.version import Version
                return Version(str(v))
            except Exception:
                return str(v)

        # --- DistributionNotFound / VersionConflict ---
        class DistributionNotFound(Exception):
            pass

        class VersionConflict(Exception):
            pass

        shim.resource_filename = resource_filename
        shim.get_distribution = get_distribution
        shim.require = require
        shim.iter_entry_points = iter_entry_points
        shim.working_set = _WorkingSet()
        shim.parse_version = parse_version
        shim.DistributionNotFound = DistributionNotFound
        shim.VersionConflict = VersionConflict

        sys.modules["pkg_resources"] = shim
        log("Applied runtime pkg_resources shim (ctranslate2 / pyannote / NeMo compatibility).")

    def _normalize_whisper_backend(self, value: str | None) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        if raw in {"", "default", "faster", "faster_whisper"}:
            return "faster_whisper"
        if raw in {"transformers", "hf_transformers", "transformers_whisper", "insanely_fast_whisper"}:
            return "insanely_fast_whisper"
        return "faster_whisper"

    def _resolve_whisper_backend(self, requested_backend: str | None = None) -> dict:
        import importlib.util

        requested = self._normalize_whisper_backend(requested_backend or os.getenv("WHISPER_BACKEND"))
        if requested == "faster_whisper":
            return {
                "requested": requested,
                "resolved": "faster_whisper",
                "fallback_used": False,
                "fallback_reason": None,
                "available": True,
            }

        has_insanely_fast = importlib.util.find_spec("insanely_fast_whisper") is not None
        has_transformers = importlib.util.find_spec("transformers") is not None

        if has_insanely_fast and has_transformers:
            return {
                "requested": requested,
                "resolved": "insanely_fast_whisper",
                "fallback_used": False,
                "fallback_reason": None,
                "available": True,
            }

        if has_transformers:
            return {
                "requested": requested,
                "resolved": "transformers_compat",
                "fallback_used": False,
                "fallback_reason": None,
                "available": True,
            }

        return {
            "requested": requested,
            "resolved": "faster_whisper",
            "fallback_used": True,
            "fallback_reason": (
                "Requested insanely_fast_whisper-compatible backend, but transformers is not installed. "
                "Falling back to faster_whisper."
            ),
            "available": False,
        }

    def _resolve_transformers_whisper_model_ref(self, model_size: str) -> str:
        normalized = str(model_size or "small").strip() or "small"
        if "/" in normalized:
            return normalized
        aliases = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
            "large-v3-turbo": "openai/whisper-large-v3-turbo",
            "turbo": "openai/whisper-large-v3-turbo",
        }
        return aliases.get(normalized.lower(), normalized)

    def _resolve_transformers_whisper_dtype(self, requested_compute_type: str, device: str) -> tuple[object, str]:
        import torch

        compute = str(requested_compute_type or "").strip().lower()
        if device != "cuda":
            return torch.float32, "float32"
        if compute == "float32":
            return torch.float32, "float32"
        return torch.float16, "float16"

    def _join_transcribed_words(self, words: list) -> str:
        text = " ".join(str(getattr(word, "word", "") or "").strip() for word in words if str(getattr(word, "word", "") or "").strip())
        return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()

    def _resolve_whisper_language_hint(self, language_code: str | None) -> str | None:
        normalized = self._normalize_language_code(language_code)
        if not normalized:
            return None
        return {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "pl": "polish",
            "uk": "ukrainian",
            "ru": "russian",
        }.get(normalized, normalized)

    def _build_transformers_whisper_segments(self, words: list, full_text: str, total_duration: float) -> list:
        if not words:
            if not str(full_text or "").strip():
                return []
            end_time = float(total_duration or 0.0)
            if end_time <= 0.0:
                end_time = max(1.0, float(len(str(full_text).split())) * 0.35)
            return [
                self._build_whisper_style_segment(
                    seg_id=0,
                    start=0.0,
                    end=end_time,
                    text=str(full_text).strip(),
                    words=None,
                )
            ]

        segments = []
        current_words = []
        max_words = 24
        max_seconds = 14.0
        punctuation_marks = {".", "!", "?", ";", ":"}

        for index, word in enumerate(words):
            current_words.append(word)
            is_last = index == len(words) - 1
            current_duration = float(current_words[-1].end) - float(current_words[0].start)
            current_text = str(getattr(current_words[-1], "word", "") or "").strip()
            next_gap = 0.0
            if not is_last:
                next_gap = max(0.0, float(words[index + 1].start) - float(current_words[-1].end))
            should_split = False
            if is_last:
                should_split = True
            elif len(current_words) >= max_words or current_duration >= max_seconds:
                should_split = True
            elif next_gap >= 0.85:
                should_split = True
            elif current_text and current_text[-1] in punctuation_marks and len(current_words) >= 6:
                should_split = True

            if not should_split:
                continue

            segments.append(
                self._build_whisper_style_segment(
                    seg_id=len(segments),
                    start=float(current_words[0].start),
                    end=float(current_words[-1].end),
                    text=self._join_transcribed_words(current_words),
                    words=list(current_words),
                )
            )
            current_words = []

        return segments

    def _run_transformers_whisper_transcribe(self, pipeline_runner, audio_path: str, kwargs: dict):
        from types import SimpleNamespace

        word_timestamps = bool(kwargs.get("word_timestamps", True))
        beam_size = max(1, int(kwargs.get("beam_size", 1) or 1))
        batch_size = 8 if os.getenv("TRANSCRIPTION_BATCHED", "true").strip().lower() == "true" else 1
        total_duration = float(self._probe_audio_duration_seconds(Path(audio_path)) or 0.0)

        generate_kwargs = {"task": "transcribe", "num_beams": beam_size}
        language_hint = self._resolve_whisper_language_hint(kwargs.get("language"))
        if language_hint:
            generate_kwargs["language"] = language_hint
        result = pipeline_runner(
            str(audio_path),
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps="word" if word_timestamps else True,
            generate_kwargs=generate_kwargs,
        )

        chunk_items = result.get("chunks") if isinstance(result, dict) else None
        words = []
        if word_timestamps and isinstance(chunk_items, list):
            for chunk in chunk_items:
                text_value = str(chunk.get("text") or "").strip()
                timestamp = chunk.get("timestamp") or chunk.get("timestamps") or ()
                if not text_value:
                    continue
                if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
                    continue
                start_value = timestamp[0]
                end_value = timestamp[1]
                if start_value is None:
                    continue
                try:
                    ws = float(start_value)
                    we = float(end_value if end_value is not None else start_value)
                except Exception:
                    continue
                if we < ws:
                    we = ws
                words.append(self._build_whisper_style_word(start=ws, end=we, word=text_value))

        text_value = ""
        if isinstance(result, dict):
            text_value = str(result.get("text") or "").strip()
        segments = self._build_transformers_whisper_segments(words, text_value, total_duration)
        info = SimpleNamespace(
            duration=total_duration,
            language=None,
            language_probability=0.0,
        )
        return iter(segments), info

    def _load_transformers_whisper_model(
        self,
        *,
        model_size: str,
        requested_compute_type: str,
        desired_cache_key: str,
        job_id: int | None,
        memory_profile,
        backend_label: str = "transformers_compat",
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_ref = self._resolve_transformers_whisper_model_ref(model_size)
        attempted_cuda = self.device == "cuda"
        dtype, dtype_label = self._resolve_transformers_whisper_dtype(requested_compute_type, self.device)
        self._release_whisper_model("reload_transformers_whisper", job_id=job_id)

        def _load_for_device(runtime_device: str, runtime_dtype, runtime_dtype_label: str):
            with temporary_disabled_blackhole_proxies():
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_ref,
                    torch_dtype=runtime_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                processor = AutoProcessor.from_pretrained(model_ref)
            if runtime_device == "cuda":
                model.to("cuda")
                device_arg = 0
            else:
                device_arg = -1
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=runtime_dtype,
                device=device_arg,
            )
            self.whisper_model = TransformersWhisperCompatModel(
                service=self,
                pipeline_runner=pipe,
                model=model,
                processor=processor,
                model_ref=model_ref,
                runtime_device=runtime_device,
                torch_dtype_label=runtime_dtype_label,
            )
            self._whisper_compute_type = runtime_dtype_label
            self._whisper_device = runtime_device
            self._whisper_backend = backend_label
            self._whisper_model_cache_key = desired_cache_key
            self._upsert_job_payload_fields(
                job_id,
                {
                    "whisper_compute_type": runtime_dtype_label,
                    "whisper_runtime_device": runtime_device,
                    "whisper_fallback_to_cpu": runtime_device == "cpu",
                    "stage_model_load_completed_at": datetime.now().isoformat(),
                },
            )
            self._finish_component_memory_profile("whisper", memory_profile, loaded=True)
            log(f"Whisper {backend_label} backend loaded ({model_ref}, dtype={runtime_dtype_label}, device={runtime_device})")
            if runtime_device == "cuda" and runtime_dtype_label != "float16":
                self._force_float32 = True

        try:
            _load_for_device(self.device, dtype, dtype_label)
            return
        except Exception as e:
            if self._is_cuda_illegal_access(e):
                self._mark_cuda_unhealthy(str(e), job_id=job_id)
            log(f"Failed to load transformers Whisper backend on {self.device}: {e}")
            if not attempted_cuda:
                raise

        try:
            _load_for_device("cpu", torch.float32, "float32")
            self._update_job_status_detail(job_id, "Transformers Whisper CUDA load failed; using CPU fallback.")
            return
        except Exception as e:
            log(f"Failed to load transformers Whisper backend on CPU fallback: {e}")
            raise RuntimeError(
                f"Could not load transformers-compatible Whisper backend ({model_ref}) on CUDA or CPU. "
                f"Last error: {e}"
            ) from e

    def _load_whisper_model(
        self,
        job_id: int = None,
        force_float32: bool = False,
        model_size_override: str | None = None,
        backend_override: str | None = None,
    ):
        self._ensure_ctranslate2_pkg_resources()

        self._ensure_device()
        model_size = str(model_size_override or os.getenv("TRANSCRIPTION_MODEL", "tiny")).strip() or "tiny"
        backend_info = self._resolve_whisper_backend(backend_override)
        requested_backend = backend_info["requested"]
        resolved_backend = backend_info["resolved"]
        if force_float32:
            requested_compute_type = "float32"
        else:
            requested_compute_type = os.getenv("TRANSCRIPTION_COMPUTE_TYPE", "").strip()

        if self.device != "cuda" and requested_compute_type not in {"", "int8", "float32"}:
            log(
                f"Ignoring unsupported Whisper compute_type={requested_compute_type!r} on device={self.device}; "
                "using CPU-safe candidates instead."
            )
            requested_compute_type = ""

        desired_cache_key = f"{resolved_backend}::{model_size}"

        # Keep an already-loaded compatible model.
        if (
            self.whisper_model is not None
            and self._whisper_compute_type
            and self._whisper_device
            and self._whisper_backend == resolved_backend
            and self._whisper_model_cache_key == desired_cache_key
        ):
            if not force_float32 or self._whisper_compute_type == "float32":
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "whisper_backend_requested": requested_backend,
                        "whisper_backend_resolved": resolved_backend,
                        "whisper_backend_fallback_used": bool(backend_info.get("fallback_used")),
                        "whisper_backend_fallback_reason": backend_info.get("fallback_reason"),
                    },
                )
                return

        self._record_job_stage_start(job_id, "model_load")
        log(f"Loading Whisper model ({model_size}, backend={resolved_backend})...")
        self._update_job_status_detail(job_id, f"Loading Whisper model ({model_size}, {resolved_backend})...")
        memory_profile = self._start_component_memory_profile()
        self._upsert_job_payload_fields(
            job_id,
            {
                "whisper_backend_requested": requested_backend,
                "whisper_backend_resolved": resolved_backend,
                "whisper_backend_fallback_used": bool(backend_info.get("fallback_used")),
                "whisper_backend_fallback_reason": backend_info.get("fallback_reason"),
            },
        )

        if resolved_backend in {"transformers_compat", "insanely_fast_whisper"}:
            try:
                self._load_transformers_whisper_model(
                    model_size=model_size,
                    requested_compute_type=requested_compute_type,
                    desired_cache_key=desired_cache_key,
                    job_id=job_id,
                    memory_profile=memory_profile,
                    backend_label=resolved_backend,
                )
                return
            except Exception as e:
                fallback_reason = (
                    f"{resolved_backend} backend failed to load ({type(e).__name__}: {e}). "
                    "Falling back to faster_whisper."
                )
                log(fallback_reason)
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "whisper_backend_requested": requested_backend,
                        "whisper_backend_resolved": "faster_whisper",
                        "whisper_backend_fallback_used": True,
                        "whisper_backend_fallback_reason": fallback_reason[:1000],
                    },
                )
                resolved_backend = "faster_whisper"
                desired_cache_key = f"{resolved_backend}::{model_size}"

        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model as resolve_whisper_model

        self._release_whisper_model("reload_faster_whisper", job_id=job_id)
        attempted_cuda = (self.device == "cuda")
        whisper_model_ref = model_size
        whisper_local_only = False

        try:
            with temporary_disabled_blackhole_proxies():
                resolved_model_path = resolve_whisper_model(model_size, local_files_only=True)
            if resolved_model_path:
                whisper_model_ref = resolved_model_path
                whisper_local_only = True
                log(f"Using cached Whisper model from {resolved_model_path}")
        except Exception as e:
            log_verbose(f"Whisper local cache probe missed for {model_size}: {e}")

        if requested_compute_type:
            try:
                with temporary_disabled_blackhole_proxies():
                    self.whisper_model = WhisperModel(
                        whisper_model_ref,
                        device=self.device,
                        compute_type=requested_compute_type,
                        local_files_only=whisper_local_only,
                    )
                self._whisper_compute_type = requested_compute_type
                self._whisper_device = self.device
                self._whisper_backend = "faster_whisper"
                self._whisper_model_cache_key = desired_cache_key
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "whisper_compute_type": requested_compute_type,
                        "whisper_runtime_device": self.device,
                        "whisper_fallback_to_cpu": False,
                        "stage_model_load_completed_at": datetime.now().isoformat(),
                    },
                )
                self._finish_component_memory_profile("whisper", memory_profile, loaded=True)
                log(f"Whisper loaded with compute_type={requested_compute_type}")
                if requested_compute_type != "float16" and self.device == "cuda":
                    self._force_float32 = True
                    log("Non-float16 compute type detected - pyannote models will use float32")
                return
            except Exception as e:
                if self._is_cuda_illegal_access(e):
                    self._mark_cuda_unhealthy(str(e), job_id=job_id)
                if force_float32 and not attempted_cuda:
                    raise
                log(f"Failed to load Whisper with compute_type={requested_compute_type}: {e}")
                log("Falling back to auto-detected compute type...")

        candidates = ["float16", "int8_float16", "float32"] if self.device == "cuda" else ["int8", "float32"]
        for ct in candidates:
            try:
                with temporary_disabled_blackhole_proxies():
                    self.whisper_model = WhisperModel(
                        whisper_model_ref,
                        device=self.device,
                        compute_type=ct,
                        local_files_only=whisper_local_only,
                    )
                self._whisper_compute_type = ct
                self._whisper_device = self.device
                self._whisper_backend = "faster_whisper"
                self._whisper_model_cache_key = desired_cache_key
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "whisper_compute_type": ct,
                        "whisper_runtime_device": self.device,
                        "whisper_fallback_to_cpu": False,
                        "stage_model_load_completed_at": datetime.now().isoformat(),
                    },
                )
                self._finish_component_memory_profile("whisper", memory_profile, loaded=True)
                log(f"Whisper loaded with compute_type={ct}")
                if ct != "float16" and self.device == "cuda":
                    self._force_float32 = True
                    log("GPU FP16 cuBLAS unsupported - pyannote models will use float32")
                return
            except Exception as e:
                if self._is_cuda_illegal_access(e):
                    self._mark_cuda_unhealthy(str(e), job_id=job_id)
                    break
                log(f"compute_type={ct} not supported ({type(e).__name__}), trying next...")

        if attempted_cuda:
            cpu_candidates = ["int8", "float32"]
            for ct in cpu_candidates:
                try:
                    with temporary_disabled_blackhole_proxies():
                        self.whisper_model = WhisperModel(
                            whisper_model_ref,
                            device="cpu",
                            compute_type=ct,
                            local_files_only=whisper_local_only,
                        )
                    self._whisper_compute_type = ct
                    self._whisper_device = "cpu"
                    self._whisper_backend = "faster_whisper"
                    self._whisper_model_cache_key = desired_cache_key
                    self._upsert_job_payload_fields(
                        job_id,
                        {
                            "whisper_compute_type": ct,
                            "whisper_runtime_device": "cpu",
                            "whisper_fallback_to_cpu": True,
                            "stage_model_load_completed_at": datetime.now().isoformat(),
                        },
                    )
                    self._finish_component_memory_profile("whisper", memory_profile, loaded=True)
                    self._update_job_status_detail(job_id, "Whisper CUDA load failed; using CPU fallback.")
                    log(f"Whisper loaded with compute_type={ct} on CPU fallback")
                    return
                except Exception as e:
                    log(f"CPU fallback compute_type={ct} not supported ({type(e).__name__}), trying next...")

        device_label = self.device
        tried = list(candidates)
        if attempted_cuda and "int8" not in tried:
            tried.extend(["cpu:int8", "cpu:float32"])
        raise RuntimeError(
            f"Could not load Whisper model ({model_size}) on device={device_label}. "
            f"Tried compute types: {tried}."
        )

    def _parakeet_dependencies_available(self) -> bool:
        try:
            import nemo.collections.asr  # noqa: F401
            return True
        except Exception:
            return False

    def _load_parakeet_model(self, job_id: int = None):
        self._ensure_device()
        if self.device == "cuda":
            self._record_cuda_health_event("pre_parakeet_load", job_id=job_id)
            self._soft_reset_cuda_if_degraded(label="pre_parakeet_load", job_id=job_id)
            self._record_cuda_health_event("post_parakeet_reset_check", job_id=job_id)
        if self._cuda_unhealthy_reason:
            raise RuntimeError(
                f"Parakeet disabled for this worker until restart due to prior CUDA fault: {self._cuda_unhealthy_reason}"
            )
        if self.parakeet_model is not None:
            self._upsert_job_payload_fields(job_id, {"parakeet_model_cached": True})
            return

        payload = {}
        if job_id:
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if job and job.payload_json:
                        payload = self._load_job_payload(job.payload_json)
            except Exception:
                payload = {}

        transcribe_started_at = payload.get("stage_transcribe_started_at")
        reload_during_transcribe = bool(transcribe_started_at)
        max_reload_during_transcribe = max(
            1,
            int((os.getenv("PARAKEET_MAX_RELOADS_DURING_TRANSCRIBE") or "2").strip() or "2"),
        )
        existing_reload_count = int(payload.get("parakeet_model_reload_count") or 0)
        observed_soft_resets = int(payload.get("cuda_soft_reset_count") or self._cuda_soft_reset_count or 0)
        if reload_during_transcribe and existing_reload_count >= max_reload_during_transcribe:
            raise RuntimeError(
                "Parakeet reload-thrash detected during transcription "
                f"(reloads={existing_reload_count}, soft_resets={observed_soft_resets}). "
                "Falling back to Whisper to avoid prolonged GPU stalls."
            )

        if not self._parakeet_dependencies_available():
            raise RuntimeError(
                "Parakeet dependencies are not installed. Install backend/requirements-parakeet.txt in the app venv."
            )

        import torch
        from nemo.collections.asr.models import ASRModel

        parakeet_model = (os.getenv("PARAKEET_MODEL") or "nvidia/parakeet-tdt-0.6b-v2").strip()
        if not reload_during_transcribe:
            self._record_job_stage_start(job_id, "model_load")
        load_started = time.time()
        chunk_index = payload.get("stage_transcribe_chunk_index")
        chunk_total = payload.get("stage_transcribe_chunk_total")
        segments_completed = payload.get("stage_transcribe_segments_completed")
        progress_completed_seconds = payload.get("stage_transcribe_progress_seconds")
        progress_total_seconds = payload.get("stage_transcribe_progress_total_seconds")
        if reload_during_transcribe:
            reload_parts = []
            try:
                if progress_total_seconds is not None:
                    reload_parts.append(
                        f"{self._format_progress_clock(progress_completed_seconds)}/"
                        f"{self._format_progress_clock(progress_total_seconds)}"
                    )
            except Exception:
                pass
            try:
                if chunk_index is not None and chunk_total is not None and int(chunk_total) > 0:
                    reload_parts.append(f"chunk {int(chunk_index)}/{int(chunk_total)}")
                elif chunk_index is not None:
                    reload_parts.append(f"chunk {int(chunk_index)}")
            except Exception:
                pass
            reload_detail = "Reloading Parakeet during transcription"
            if reload_parts:
                reload_detail += f" ({', '.join(reload_parts)})"
            reload_detail += "..."
            self._update_job_status_detail(job_id, reload_detail)
            self._upsert_job_payload_fields(
                job_id,
                {
                    "parakeet_model_reload_during_transcribe": True,
                    "parakeet_model_reload_count": int(payload.get("parakeet_model_reload_count") or 0) + 1,
                },
            )
        else:
            self._update_job_status_detail(job_id, f"Restoring Parakeet checkpoint from disk ({parakeet_model})...")
        log(f"Loading Parakeet model ({parakeet_model})...")
        memory_profile = self._start_component_memory_profile()
        self._apply_cuda_memory_fraction_limit()
        # Restore onto CPU first. Letting NeMo deserialize directly to CUDA can
        # spike VRAM during checkpoint restore and fail before the model is even usable.
        self.parakeet_model = ASRModel.from_pretrained(
            model_name=parakeet_model,
            map_location=torch.device("cpu"),
        )

        if self.device == "cuda":
            self._update_job_status_detail(job_id, "Moving Parakeet model to GPU...")
        else:
            self._update_job_status_detail(job_id, "Initializing Parakeet model on CPU...")
        if self.device == "cuda":
            self.parakeet_model = self.parakeet_model.to(torch.device("cuda"))
        else:
            self.parakeet_model = self.parakeet_model.to(torch.device("cpu"))

        self._update_job_status_detail(job_id, "Initializing Parakeet decoder...")
        self.parakeet_model.eval()
        load_seconds = max(0.0, time.time() - load_started)
        self._upsert_job_payload_fields(
            job_id,
            {
                "parakeet_model_cached": False,
                "stage_model_load_seconds": round(load_seconds, 2),
                "stage_model_load_completed_at": datetime.now().isoformat(),
            },
        )
        self._finish_component_memory_profile("parakeet", memory_profile, loaded=True)
        if reload_during_transcribe:
            self._update_transcription_stage_progress(
                job_id,
                engine="parakeet",
                completed_seconds=progress_completed_seconds,
                total_seconds=progress_total_seconds,
                segments_completed=segments_completed,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                extra_label="resuming",
            )
        log("Parakeet model loaded.")

    def _set_parakeet_decoding_profile(self, profile: str = "optimized", job_id: int = None):
        """Apply a bounded set of decode-time profiles for Parakeet retries."""
        if self.parakeet_model is None:
            return

        from omegaconf import open_dict

        name = str(profile or "optimized").strip().lower()
        if name == "optimized":
            target_strategy = "greedy_batch"
            target_preserve_alignments = False
            target_use_cuda_graph_decoder = True
        elif name == "safe_no_graph":
            target_strategy = "greedy_batch"
            target_preserve_alignments = True
            target_use_cuda_graph_decoder = False
        elif name == "safe_greedy":
            target_strategy = "greedy"
            target_preserve_alignments = True
            target_use_cuda_graph_decoder = False
        else:
            raise ValueError(f"Unknown Parakeet decoding profile: {profile}")

        cfg = self.parakeet_model.cfg.decoding
        changed = False
        with open_dict(cfg):
            if cfg.get("compute_timestamps", None) is not True:
                cfg.compute_timestamps = True
                changed = True
            if bool(cfg.get("preserve_alignments", False)) != target_preserve_alignments:
                cfg.preserve_alignments = target_preserve_alignments
                changed = True
            if str(cfg.get("strategy") or "greedy_batch") != target_strategy:
                cfg.strategy = target_strategy
                changed = True
            if bool(cfg.greedy.get("use_cuda_graph_decoder", True)) != target_use_cuda_graph_decoder:
                cfg.greedy.use_cuda_graph_decoder = target_use_cuda_graph_decoder
                changed = True

        if changed:
            self.parakeet_model.change_decoding_strategy(cfg, verbose=False)

        self._upsert_job_payload_fields(
            job_id,
            {
                "parakeet_decode_profile": name,
                "parakeet_decode_strategy": target_strategy,
                "parakeet_preserve_alignments": bool(target_preserve_alignments),
                "parakeet_use_cuda_graph_decoder": bool(target_use_cuda_graph_decoder),
            },
        )
        log_verbose(
            "Parakeet decoding profile "
            f"{name}: strategy={target_strategy}, "
            f"preserve_alignments={target_preserve_alignments}, "
            f"use_cuda_graph_decoder={target_use_cuda_graph_decoder}"
        )

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

    def _convert_audio_for_parakeet(self, audio_path: Path) -> tuple[Path, bool]:
        if audio_path.suffix.lower() == ".wav":
            return audio_path, False

        out_path = TEMP_DIR / f"parakeet_{int(time.time() * 1000)}_{audio_path.stem}.wav"
        ffmpeg_cmd = self._get_ffmpeg_cmd()
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        if result.returncode != 0 or not out_path.exists():
            stderr = (result.stderr or b"").decode(errors="replace")[-500:]
            raise RuntimeError(f"Failed to convert audio for Parakeet: {stderr}")
        return out_path, True

    def _load_waveform_for_parakeet(self, wav_path: Path):
        """Load mono waveform as contiguous float32 numpy array.

        Using in-memory waveform input avoids NeMo's temp manifest-file path, which
        can intermittently fail on Windows with file-lock errors (WinError 32).
        """
        import numpy as np
        import soundfile as sf

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if samples is None:
            raise RuntimeError(f"Parakeet audio load returned no samples: {wav_path}")
        if getattr(samples, "ndim", 1) > 1:
            # Downmix to mono if needed.
            samples = np.mean(samples, axis=1, dtype=np.float32)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError(f"Parakeet audio load returned empty waveform: {wav_path}")
        if int(sample_rate or 0) != 16000:
            raise RuntimeError(
                f"Parakeet expects 16kHz audio; got {sample_rate}Hz from {wav_path}. "
                "Convert to 16kHz before transcription."
            )
        return samples

    def _build_whisper_style_segment(self, seg_id: int, start: float, end: float, text: str, words: list):
        # Keep an object shape compatible with faster-whisper Segment without
        # requiring ctranslate2/pkg_resources imports in non-Whisper paths.
        from types import SimpleNamespace
        return SimpleNamespace(
            id=int(seg_id),
            seek=0,
            start=float(start),
            end=float(max(end, start)),
            text=str(text or "").strip(),
            tokens=[],
            temperature=0.0,
            avg_logprob=0.0,
            compression_ratio=0.0,
            no_speech_prob=0.0,
            words=words or None,
        )

    def _build_whisper_style_word(self, start: float, end: float, word: str):
        from types import SimpleNamespace
        return SimpleNamespace(
            start=float(start),
            end=float(max(end, start)),
            word=str(word or ""),
            probability=1.0,
        )

    def _extract_parakeet_timestamp_items(self, payload, *, _depth: int = 0) -> tuple[list[dict], list[dict]]:
        if payload is None or _depth > 3:
            return [], []

        raw_words: list[dict] = []
        raw_segments: list[dict] = []

        if isinstance(payload, dict):
            raw_words = list(payload.get("word") or payload.get("words") or [])
            raw_segments = list(payload.get("segment") or payload.get("segments") or [])
            if raw_words or raw_segments:
                return raw_words, raw_segments

            # Some NeMo builds return a timestamp container whose actual timing
            # payload lives under a nested `timestep` key, while the top-level
            # `word` / `segment` entries are empty placeholders.
            nested = (
                payload.get("timestep")
                or payload.get("timestamp")
                or payload.get("timestamps")
            )
            if nested is not None and nested is not payload:
                return self._extract_parakeet_timestamp_items(nested, _depth=_depth + 1)
            return [], []

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and ("word" in payload[0] or "text" in payload[0]):
                return list(payload), []
            if payload and isinstance(payload[0], dict):
                return [], list(payload)
            if payload and isinstance(payload[0], (list, tuple, dict)):
                words, segments = self._extract_parakeet_timestamp_items(payload[0], _depth=_depth + 1)
                if words or segments:
                    return words, segments

        if isinstance(payload, tuple) and payload:
            return self._extract_parakeet_timestamp_items(list(payload), _depth=_depth + 1)

        return [], []

    def _extract_parakeet_transcript_items(self, hypothesis) -> tuple[str, list[dict], list[dict]]:
        text = ""
        raw_words = []
        raw_segments = []

        if isinstance(hypothesis, str):
            return hypothesis, raw_words, raw_segments

        if isinstance(hypothesis, dict):
            text = str(hypothesis.get("text") or hypothesis.get("pred_text") or "")
            ts = (
                hypothesis.get("timestep")
                or hypothesis.get("timestamp")
                or hypothesis.get("timestamps")
                or {}
            )
        else:
            text = str(getattr(hypothesis, "text", "") or getattr(hypothesis, "pred_text", "") or "")
            ts = (
                getattr(hypothesis, "timestep", None)
                or getattr(hypothesis, "timestamp", None)
                or getattr(hypothesis, "timestamps", None)
                or {}
            )

        raw_words, raw_segments = self._extract_parakeet_timestamp_items(ts)
        return text, raw_words, raw_segments

    def _normalize_parakeet_hypothesis(self, result):
        """Normalize NeMo transcription outputs down to a single hypothesis-like object.

        NeMo's `transcribe()` can return list/tuple shapes that vary by model and kwargs.
        For a single input audio item, we want the first hypothesis payload regardless of
        whether it arrives as a flat hypothesis, `[hyp]`, `([hyp],)`, or similar nesting.
        """
        current = result
        visited = 0
        while isinstance(current, (list, tuple)) and len(current):
            first = current[0]
            if isinstance(first, (list, tuple)) and len(first):
                current = first
            else:
                current = first
            visited += 1
            if visited >= 8:
                break
        return current

    def _describe_parakeet_timestamp_payload(self, hypothesis) -> str:
        if hypothesis is None:
            return "none"
        if isinstance(hypothesis, dict):
            ts = (
                hypothesis.get("timestep")
                or hypothesis.get("timestamp")
                or hypothesis.get("timestamps")
            )
        else:
            ts = (
                getattr(hypothesis, "timestep", None)
                or getattr(hypothesis, "timestamp", None)
                or getattr(hypothesis, "timestamps", None)
            )
        if isinstance(ts, dict):
            return f"dict(keys={sorted(str(k) for k in ts.keys())})"
        if isinstance(ts, list):
            return f"list(len={len(ts)})"
        if ts is None:
            return "none"
        return type(ts).__name__

    def _parakeet_oom_chunk_retry_enabled(self) -> bool:
        return (os.getenv("PARAKEET_OOM_CHUNK_RETRY", "true").strip().lower() == "true")

    def _resolve_parakeet_initial_chunk_seconds(self, total_duration_seconds: float | None = None) -> int:
        """Choose a safer initial Parakeet chunk size for long episodes.

        The default env chunk can be too aggressive on long-form videos, which increases
        the chance of repeated OOM/backoff churn and eventual CUDA instability.
        """
        try:
            base_chunk = int(os.getenv("PARAKEET_OOM_CHUNK_SECONDS", "600"))
        except Exception:
            base_chunk = 600
        base_chunk = max(120, min(base_chunk, 3600))

        duration = float(total_duration_seconds or 0.0)
        if duration <= 0:
            return base_chunk

        # High-VRAM GPUs can handle much larger chunks without OOM risk.
        if self.device == "cuda":
            snap = self._cuda_memory_snapshot()
            free_b = int(snap.get("free") or 0)
            total_b = int(snap.get("total") or 0)
            free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
            free_ratio = (float(free_b) / float(total_b)) if total_b > 0 and free_b > 0 else 0.0

            if free_gb >= 20.0 and free_ratio >= 0.60:
                # Plenty of headroom — use large chunks to minimize overhead.
                target = base_chunk  # default 600s
            elif duration >= 7200:
                target = 150
            elif duration >= 5400:
                target = 180
            elif duration >= 3600:
                target = 240
            elif duration >= 1800:
                target = 300
            else:
                target = base_chunk

            # Under tighter free-memory headroom, bias lower.
            if free_ratio < 0.22:
                target = min(target, 120)
            elif free_ratio < 0.30:
                target = min(target, 150)
            elif free_ratio < 0.40:
                target = min(target, 180)
        else:
            # Non-CUDA: conservative defaults for long content.
            if duration >= 7200:
                target = 150
            elif duration >= 5400:
                target = 180
            elif duration >= 3600:
                target = 240
            elif duration >= 1800:
                target = 300
            else:
                target = base_chunk

        return max(120, min(base_chunk, int(target)))

    def _resolve_parakeet_oom_chunk_settings(self, total_duration_seconds: float | None = None) -> tuple[int, int, float]:
        """Return (chunk_seconds, min_chunk_seconds, overlap_seconds) for OOM chunk fallback."""
        try:
            chunk_seconds = int(self._resolve_parakeet_initial_chunk_seconds(total_duration_seconds))
        except Exception:
            chunk_seconds = 600
        try:
            min_chunk_seconds = int(os.getenv("PARAKEET_OOM_MIN_CHUNK_SECONDS", "30"))
        except Exception:
            min_chunk_seconds = 30
        try:
            overlap_seconds = float(os.getenv("PARAKEET_OOM_CHUNK_OVERLAP_SECONDS", "0.35"))
        except Exception:
            overlap_seconds = 0.35

        if self.device == "cuda":
            snap = self._cuda_memory_snapshot()
            free_b = int(snap.get("free") or 0)
            total_b = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
            free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
            free_ratio = (float(free_b) / float(total_b)) if total_b > 0 and free_b > 0 else 0.0
            # None means unrestricted — treat as high cap, not zero
            dynamic_cap = int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else 999

            if dynamic_cap <= 1 or free_gb < 8.0 or free_ratio < 0.25:
                chunk_seconds = min(chunk_seconds, 120)
            elif dynamic_cap <= 2 or free_gb < 12.0 or free_ratio < 0.35:
                chunk_seconds = min(chunk_seconds, 180)
            elif free_gb < 16.0 or free_ratio < 0.45:
                chunk_seconds = min(chunk_seconds, 240)

        chunk_seconds = max(120, min(chunk_seconds, 3600))
        min_chunk_seconds = max(30, min(min_chunk_seconds, chunk_seconds))
        overlap_seconds = max(0.0, min(overlap_seconds, 2.0))
        return chunk_seconds, min_chunk_seconds, overlap_seconds

    def _resolve_parakeet_chunk_recycle_every(self, total_duration_seconds: float) -> int:
        """Return chunk cadence for Parakeet model recycle during long chunked runs.

        Recycling periodically reduces cumulative CUDA fragmentation/state drift on
        long episodes without forcing unloads between every episode.
        """
        raw = (os.getenv("PARAKEET_CHUNK_MODEL_RECYCLE_EVERY") or "").strip()
        if raw:
            try:
                return max(0, min(int(raw), 200))
            except Exception:
                pass
        if self.device != "cuda":
            return 0
        snap = self._cuda_memory_snapshot()
        free_b = int(snap.get("free") or 0)
        total_b = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
        free_ratio = (float(free_b) / float(total_b)) if total_b > 0 and free_b > 0 else 0.0
        # None means unrestricted — treat as high cap, not zero
        dynamic_cap = int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else 999
        # On healthy high-headroom runs, keep the model resident across chunks. That
        # avoids the repeated "Loading Parakeet model..." loop that looks hung and
        # adds large overhead without improving stability.
        if (
            free_gb >= 20.0
            and free_ratio >= 0.60
            and self._cuda_fault_count <= 0
            and not bool(self._cuda_degraded_reason)
            and not bool(self._cuda_unhealthy_reason)
        ):
            return 0
        # Only recycle every chunk when the worker has already shown memory pressure
        # or prior CUDA instability. On healthy high-headroom runs, per-chunk reloads
        # destroy throughput and can look like the model-load stage is hung.
        if dynamic_cap <= 1 and (
            self._cuda_oom_backoff_count > 0
            or self._cuda_fault_count > 0
            or bool(self._cuda_degraded_reason)
            or bool(self._cuda_unhealthy_reason)
        ):
            return 1
        if dynamic_cap == 2 and (
            self._cuda_oom_backoff_count > 0
            or self._cuda_fault_count > 0
            or bool(self._cuda_degraded_reason)
            or bool(self._cuda_unhealthy_reason)
        ):
            return 2
        if total_duration_seconds >= 7200:
            return 6
        if total_duration_seconds >= 5400:
            return 8
        if total_duration_seconds >= 3600:
            return 10
        if total_duration_seconds >= 1800:
            return 14
        return 0

    def _resolve_parakeet_chunk_reload_floor_gb(self, total_gb: float) -> float:
        """Minimum free VRAM required before attempting a mid-job Parakeet reload."""
        if total_gb >= 36:
            default_gb = 8.0
        elif total_gb >= 28:
            default_gb = 6.0
        elif total_gb >= 20:
            default_gb = 5.0
        else:
            default_gb = 4.0
        raw = (os.getenv("PARAKEET_CHUNK_RELOAD_MIN_FREE_GB") or "").strip()
        try:
            if raw:
                return max(0.0, float(raw))
        except Exception:
            pass
        return default_gb

    def _should_disable_parakeet_chunk_recycle(self, job_id: int = None) -> tuple[bool, str, float, float]:
        """Return whether chunk-mode model recycling should be disabled for the rest of the job."""
        if self.device != "cuda":
            return False, "non_cuda", 0.0, 0.0

        snap = self._cuda_memory_snapshot()
        total_b = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
        free_b = int(snap.get("free") or 0)
        total_gb = float(total_b) / (1024 ** 3) if total_b > 0 else 0.0
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
        floor_gb = self._resolve_parakeet_chunk_reload_floor_gb(total_gb)
        max_soft_resets = max(1, int((os.getenv("PARAKEET_CHUNK_RECYCLE_MAX_SOFT_RESETS") or "2").strip() or "2"))
        dynamic_cap = int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else 999

        if free_gb < floor_gb:
            return True, f"low_reload_headroom_{free_gb:.1f}gb_below_{floor_gb:.1f}gb", free_gb, floor_gb
        if self._cuda_soft_reset_count >= max_soft_resets and dynamic_cap <= 1:
            return True, f"soft_reset_limit_{self._cuda_soft_reset_count}_cap_{dynamic_cap}", free_gb, floor_gb
        return False, "ok", free_gb, floor_gb

    def _should_force_parakeet_long_audio_chunked(self, duration_seconds: float, job_id: int = None) -> tuple[bool, str]:
        try:
            # NVIDIA's Parakeet TDT v2 model card documents efficient single-pass
            # transcription up to roughly 24 minutes; above that, chunking is the
            # stable path on long-form episodes.
            threshold_seconds = _env_float("PARAKEET_LONG_AUDIO_SECONDS", "1440")
        except Exception:
            threshold_seconds = 1440.0
        if duration_seconds < threshold_seconds:
            return False, "below_threshold"

        # Above 2x threshold: always chunk.
        if duration_seconds >= threshold_seconds * 2.0:
            return True, "duration_very_long"

        if self.device != "cuda":
            return True, "non_cuda_long_audio"

        # Predictively chunk long-form audio. Direct whole-audio inference on ~30min+
        # inputs is much more likely to trip avoidable OOM paths even on large GPUs.
        # Chunking is fast enough if we keep the model resident across chunks.
        if self._cuda_fault_count > 0 or bool(self._cuda_degraded_reason) or bool(self._cuda_unhealthy_reason):
            return True, "worker_fault_history"

        snap = self._cuda_memory_snapshot()
        free_b = int(snap.get("free") or 0)
        free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0

        # High-VRAM GPUs with clean fault history can handle whole-audio inference
        # for the 30-60min range without chunking overhead.
        if free_gb >= 20.0:
            return False, f"high_vram_whole_audio_{free_gb:.1f}gb"

        return True, f"predictive_long_audio_chunking_{free_gb:.1f}gb"

    def _transcribe_with_parakeet_in_chunks(self, audio_path: Path, start_time_offset: float = 0.0, job_id: int = None):
        normalized_audio_path, cleanup_normalized = self._convert_audio_for_parakeet(audio_path)
        total_duration = float(self._probe_audio_duration_seconds(normalized_audio_path) or 0.0)
        if total_duration <= 0:
            if cleanup_normalized:
                try:
                    if normalized_audio_path.exists():
                        normalized_audio_path.unlink()
                except Exception:
                    pass
            raise RuntimeError("Could not determine audio duration for Parakeet chunk fallback.")

        chunk_seconds, min_chunk_seconds, overlap_seconds = self._resolve_parakeet_oom_chunk_settings(total_duration)
        recycle_every = self._resolve_parakeet_chunk_recycle_every(total_duration)
        log(
            f"Retrying Parakeet in chunked mode "
            f"(chunk={chunk_seconds}s, min={min_chunk_seconds}s, overlap={overlap_seconds:.2f}s)."
        )
        self._upsert_job_payload_fields(
            job_id,
            {
                "parakeet_chunk_fallback_used": True,
                "parakeet_chunk_seconds_initial": int(chunk_seconds),
                "parakeet_chunk_min_seconds": int(min_chunk_seconds),
                "parakeet_chunk_overlap_seconds": float(overlap_seconds),
                "parakeet_chunk_recycle_every": int(recycle_every),
                "parakeet_chunk_source_normalized": cleanup_normalized,
            },
        )

        segments = []
        chunk_start = 0.0
        chunk_index = 0
        chunk_recycle_disabled = False

        while chunk_start < total_duration - 0.01:
            chunk_index += 1
            est_total_chunks = max(1, int(math.ceil(total_duration / max(chunk_seconds, 1.0))))

            if recycle_every > 0 and chunk_index > 1 and ((chunk_index - 1) % recycle_every == 0):
                disable_recycle, disable_reason, free_gb, floor_gb = self._should_disable_parakeet_chunk_recycle(job_id=job_id)
                if disable_recycle:
                    recycle_every = 0
                    chunk_recycle_disabled = True
                    self._upsert_job_payload_fields(
                        job_id,
                        {
                            "parakeet_chunk_recycle_disabled": True,
                            "parakeet_chunk_recycle_disabled_reason": disable_reason,
                            "parakeet_chunk_reload_free_gb": round(free_gb, 2),
                            "parakeet_chunk_reload_floor_gb": round(floor_gb, 2),
                            "parakeet_chunk_recycle_every_effective": 0,
                        },
                    )
                    log(
                        "Disabling Parakeet chunk model recycle for the remainder of this job "
                        f"({disable_reason}). Continuing with the resident model."
                    )
                else:
                    if job_id:
                        est_chunks = max(1, int(total_duration // max(1, chunk_seconds)) + 1)
                        self._update_job_status_detail(
                            job_id,
                            f"Refreshing Parakeet model for stability (chunk {chunk_index}/{est_chunks})..."
                        )
                    self._release_parakeet_model("chunk_recycle")
                    self._clear_cuda_cache()
                    gc.collect()

            remaining = max(0.0, total_duration - chunk_start)
            attempt_chunk_seconds = min(chunk_seconds, remaining)

            while True:
                chunk_duration_with_overlap = min(remaining, attempt_chunk_seconds + overlap_seconds)
                if job_id:
                    self._update_transcription_stage_progress(
                        job_id,
                        engine="parakeet",
                        completed_seconds=start_time_offset + chunk_start,
                        total_seconds=start_time_offset + total_duration,
                        chunk_index=chunk_index,
                        chunk_total=est_total_chunks,
                        extra_label=f"window {int(attempt_chunk_seconds)}s",
                    )

                chunk_path = self._slice_audio(normalized_audio_path, chunk_start, chunk_duration_with_overlap)
                try:
                    chunk_segments, _ = self._transcribe_with_parakeet(
                        chunk_path,
                        start_time_offset=start_time_offset + chunk_start,
                        job_id=job_id,
                        allow_oom_chunk_retry=False,
                        forced_batch_size=1,
                        progress_completed_seconds=start_time_offset + chunk_start,
                        progress_total_seconds=start_time_offset + total_duration,
                        progress_chunk_index=chunk_index,
                        progress_chunk_total=est_total_chunks,
                    )
                    if segments and chunk_segments:
                        prev_end = float(getattr(segments[-1], "end", 0.0))
                        chunk_segments = [
                            s for s in chunk_segments
                            if float(getattr(s, "end", 0.0)) > (prev_end + 0.01)
                        ]
                    segments.extend(chunk_segments)
                    if job_id:
                        self._update_transcription_stage_progress(
                            job_id,
                            engine="parakeet",
                            completed_seconds=start_time_offset + chunk_start + attempt_chunk_seconds,
                            total_seconds=start_time_offset + total_duration,
                            segments_completed=len(segments),
                            chunk_index=chunk_index,
                            chunk_total=est_total_chunks,
                        )
                    chunk_seconds = min(chunk_seconds, attempt_chunk_seconds)
                    # Keep GPU memory pressure stable over long chunk loops.
                    self._clear_cuda_cache()
                    gc.collect()
                    if chunk_recycle_disabled:
                        self._upsert_job_payload_fields(
                            job_id,
                            {
                                "parakeet_chunk_recycle_disabled": True,
                                "parakeet_chunk_recycle_every_effective": 0,
                            },
                        )
                    # Sync barrier: catch latent corruption between chunks before
                    # the next chunk triggers a harder-to-diagnose fault.
                    if self.device == "cuda" and not self._safe_cuda_sync(timeout_s=10.0):
                        raise RuntimeError("CUDA sync failed between Parakeet chunks — GPU context may be corrupted")
                    break
                except Exception as e:
                    if self._is_cuda_oom(e) and attempt_chunk_seconds > min_chunk_seconds:
                        next_chunk = max(min_chunk_seconds, int(attempt_chunk_seconds // 2))
                        if next_chunk < attempt_chunk_seconds:
                            log(
                                f"Parakeet chunk OOM at {int(attempt_chunk_seconds)}s window. "
                                f"Retrying with {int(next_chunk)}s."
                            )
                            self._clear_cuda_cache()
                            attempt_chunk_seconds = next_chunk
                            continue
                    raise
                finally:
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                    except Exception:
                        pass

            chunk_start += attempt_chunk_seconds

        if cleanup_normalized:
            try:
                if normalized_audio_path.exists():
                    normalized_audio_path.unlink()
            except Exception:
                pass

        return segments, (start_time_offset + total_duration)

    def _transcribe_with_parakeet(
        self,
        audio_path: Path,
        start_time_offset: float = 0.0,
        job_id: int = None,
        allow_oom_chunk_retry: bool = True,
        forced_batch_size: int = None,
        progress_completed_seconds: float | None = None,
        progress_total_seconds: float | None = None,
        progress_chunk_index: int | None = None,
        progress_chunk_total: int | None = None,
    ):
        self._load_parakeet_model(job_id)
        parakeet_input, cleanup_input = self._convert_audio_for_parakeet(audio_path)
        waveform = None
        transcribe_input = None
        result = None
        hypothesis = None
        raw_words = None
        raw_segments = None
        try:
            input_duration = float(self._probe_audio_duration_seconds(parakeet_input) or 0.0)
            parakeet_input_mode = "path"
            transcribe_input = [str(parakeet_input)]
            try:
                waveform = self._load_waveform_for_parakeet(parakeet_input)
                transcribe_input = [waveform]
                parakeet_input_mode = "tensor"
            except Exception as e:
                # Keep path-mode as a fallback path if waveform loading fails.
                log_verbose(f"Parakeet waveform load failed; using path input fallback: {e}")

            if forced_batch_size is not None:
                parakeet_batch_size_requested = max(1, min(int(forced_batch_size), 64))
                parakeet_batch_size = parakeet_batch_size_requested
            else:
                parakeet_batch_size_requested = max(1, min(int(os.getenv("PARAKEET_BATCH_SIZE", "16")), 64))
                parakeet_batch_size = self._resolve_parakeet_batch_size(parakeet_batch_size_requested)
                if len(transcribe_input) == 1 and parakeet_batch_size > 1:
                    parakeet_batch_size = 1
            mem_snap = self._cuda_memory_snapshot()
            self._upsert_job_payload_fields(
                job_id,
                {
                    "parakeet_batch_size_requested": int(parakeet_batch_size_requested),
                    "parakeet_batch_size_effective": int(parakeet_batch_size),
                    "parakeet_dynamic_batch_cap": (
                        int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else None
                    ),
                    "parakeet_batch_auto": os.getenv("PARAKEET_BATCH_AUTO", "true").strip().lower() == "true",
                    "parakeet_input_mode": parakeet_input_mode,
                },
            )
            if self.device == "cuda":
                mem_line = (
                    f"CUDA mem free {self._format_gb(int(mem_snap.get('free') or 0))} / "
                    f"total {self._format_gb(int(mem_snap.get('total') or 0))} "
                    f"(alloc {self._format_gb(int(mem_snap.get('allocated') or 0))}, "
                    f"resv {self._format_gb(int(mem_snap.get('reserved') or 0))})"
                )
                log_verbose(mem_line)
                self._upsert_job_payload_fields(
                    job_id,
                    {
                        "parakeet_cuda_free_gb_start": round(float(mem_snap.get("free") or 0) / (1024 ** 3), 2),
                        "parakeet_cuda_total_gb": round(float(mem_snap.get("total") or 0) / (1024 ** 3), 2),
                    },
                )
            elif job_id:
                self._update_transcription_stage_progress(
                    job_id,
                    engine="parakeet",
                    completed_seconds=(
                        progress_completed_seconds
                        if progress_completed_seconds is not None
                        else start_time_offset
                    ),
                    total_seconds=(
                        progress_total_seconds
                        if progress_total_seconds is not None
                        else ((start_time_offset + input_duration) if input_duration > 0 else None)
                    ),
                    chunk_index=progress_chunk_index,
                    chunk_total=progress_chunk_total,
                    extra_label=(
                        f"batch {parakeet_batch_size}, auto from {parakeet_batch_size_requested}"
                        if parakeet_batch_size != parakeet_batch_size_requested
                        else f"batch {parakeet_batch_size}"
                    ),
                )
            if self.device == "cuda" and job_id:
                self._update_transcription_stage_progress(
                    job_id,
                    engine="parakeet",
                    completed_seconds=(
                        progress_completed_seconds
                        if progress_completed_seconds is not None
                        else start_time_offset
                    ),
                    total_seconds=(
                        progress_total_seconds
                        if progress_total_seconds is not None
                        else ((start_time_offset + input_duration) if input_duration > 0 else None)
                    ),
                    chunk_index=progress_chunk_index,
                    chunk_total=progress_chunk_total,
                    extra_label=(
                        f"batch {parakeet_batch_size}, auto from {parakeet_batch_size_requested}"
                        if parakeet_batch_size != parakeet_batch_size_requested
                        else f"batch {parakeet_batch_size}"
                    ),
                )
            call_variants = [
                {"timestamps": True, "return_hypotheses": True},
                {"timestamps": True},
                {"return_hypotheses": True},
                {},
            ]
            retry_profiles = ["optimized", "safe_no_graph"]
            if self.device == "cuda":
                retry_profiles.append("safe_greedy")
            last_error = None
            transcribe_started = False
            profile_attempt_count = 0
            for profile_index, retry_profile in enumerate(retry_profiles):
                profile_attempt_count += 1
                self._set_parakeet_decoding_profile(retry_profile, job_id=job_id)
                result = None
                hypothesis = None
                raw_words = None
                raw_segments = None
                current_batch = parakeet_batch_size
                oom_hit = False

                while result is None:
                    for kwargs in call_variants:
                        try:
                            if not transcribe_started:
                                # Start transcribe timer only when model loading/prep is done and
                                # we're about to execute actual decoder inference.
                                self._record_job_stage_start(job_id, "transcribe")
                                transcribe_started = True
                            import torch
                            with torch.inference_mode():
                                transcribe_kwargs = dict(kwargs)
                                # Explicitly pin to 0 workers on Windows to reduce file
                                # handle contention in transcribe dataloaders.
                                transcribe_kwargs["num_workers"] = 0
                                if parakeet_input_mode == "tensor":
                                    # Tensor input path bypasses temporary manifest files.
                                    transcribe_kwargs["use_lhotse"] = False
                                result = self.parakeet_model.transcribe(
                                    transcribe_input,
                                    batch_size=current_batch,
                                    **transcribe_kwargs,
                                )
                            break
                        except TypeError as e:
                            last_error = e
                            continue
                        except Exception as e:
                            last_error = e
                            if self._is_cuda_oom(e):
                                oom_hit = True
                                break
                            if "unexpected keyword" in str(e).lower():
                                continue
                            break

                    if result is not None:
                        break
                    if oom_hit and self.device == "cuda" and current_batch > 1:
                        next_batch = max(1, current_batch // 2)
                        self._record_parakeet_oom_batch_cap(next_batch, job_id=job_id)
                        log(
                            f"Parakeet CUDA OOM at batch_size={current_batch}. "
                            f"Retrying with batch_size={next_batch}."
                        )
                        if job_id:
                            self._update_job_status_detail(
                                job_id,
                                f"Parakeet VRAM pressure detected. Retrying with smaller batch ({next_batch})..."
                            )
                        self._clear_cuda_cache()
                        current_batch = next_batch
                        continue
                    if oom_hit and current_batch <= 1:
                        if allow_oom_chunk_retry and self._parakeet_oom_chunk_retry_enabled():
                            self._clear_cuda_cache()
                            if job_id:
                                self._update_job_status_detail(
                                    job_id,
                                    "Parakeet hit VRAM limit. Retrying in adaptive chunked mode..."
                                )
                            return self._transcribe_with_parakeet_in_chunks(
                                audio_path, start_time_offset=start_time_offset, job_id=job_id
                            )
                        raise RuntimeError(
                            "Parakeet failed due to CUDA OOM even at batch_size=1. "
                            "Use Whisper or enable lower-VRAM settings."
                        )
                    break

                if result is None:
                    if (
                        profile_index + 1 < len(retry_profiles)
                        and last_error is not None
                        and not self._is_cuda_illegal_access(last_error)
                        and not self._is_cuda_oom(last_error)
                    ):
                        log(
                            f"Parakeet decode failed under profile {retry_profile}: {last_error}. "
                            f"Retrying with {retry_profiles[profile_index + 1]}."
                        )
                        self._upsert_job_payload_fields(
                            job_id,
                            {
                                "parakeet_profile_retry_from": retry_profile,
                                "parakeet_profile_retry_to": retry_profiles[profile_index + 1],
                                "parakeet_profile_retry_reason": str(last_error)[:300],
                            },
                        )
                        continue
                    raise RuntimeError(f"Parakeet transcription failed: {last_error}")

                hypothesis = self._normalize_parakeet_hypothesis(result)
                transcript_text, raw_words, raw_segments = self._extract_parakeet_transcript_items(hypothesis)

                words = []
                for item in raw_words:
                    if not isinstance(item, dict):
                        continue
                    ws = item.get("start", item.get("t0"))
                    we = item.get("end", item.get("t1", ws))
                    ww = item.get("word", item.get("text", ""))
                    if ws is None or ww is None:
                        continue
                    try:
                        ws_f = float(ws) + start_time_offset
                        we_f = float(we) + start_time_offset
                    except Exception:
                        continue
                    if not str(ww).strip():
                        continue
                    words.append(self._build_whisper_style_word(ws_f, we_f, str(ww).strip()))

                segments = []
                seg_id = 0
                for item in raw_segments:
                    if not isinstance(item, dict):
                        continue
                    seg_start = item.get("start", item.get("t0"))
                    seg_end = item.get("end", item.get("t1", seg_start))
                    seg_text = item.get("text", item.get("segment", ""))
                    if seg_start is None:
                        continue
                    try:
                        seg_start_f = float(seg_start) + start_time_offset
                        seg_end_f = float(seg_end) + start_time_offset
                    except Exception:
                        continue
                    seg_words = [
                        w for w in words
                        if float(getattr(w, "start", 0.0)) >= seg_start_f - 0.01 and float(getattr(w, "end", 0.0)) <= seg_end_f + 0.01
                    ]
                    segments.append(
                        self._build_whisper_style_segment(
                            seg_id,
                            seg_start_f,
                            seg_end_f,
                            str(seg_text or "").strip(),
                            seg_words,
                        )
                    )
                    seg_id += 1

                if not segments and words:
                    # Chunk into sentence-like segments if explicit segment timestamps are missing.
                    chunk = []
                    for w in words:
                        chunk.append(w)
                        token = (getattr(w, "word", "") or "").strip()
                        if token.endswith((".", "!", "?")) or len(chunk) >= 30:
                            seg_text = " ".join((getattr(x, "word", "") or "").strip() for x in chunk).strip()
                            segments.append(
                                self._build_whisper_style_segment(
                                    seg_id,
                                    float(getattr(chunk[0], "start", 0.0)),
                                    float(getattr(chunk[-1], "end", getattr(chunk[-1], "start", 0.0))),
                                    seg_text,
                                    list(chunk),
                                )
                            )
                            seg_id += 1
                            chunk = []
                    if chunk:
                        seg_text = " ".join((getattr(x, "word", "") or "").strip() for x in chunk).strip()
                        segments.append(
                            self._build_whisper_style_segment(
                                seg_id,
                                float(getattr(chunk[0], "start", 0.0)),
                                float(getattr(chunk[-1], "end", getattr(chunk[-1], "start", 0.0))),
                                seg_text,
                                list(chunk),
                            )
                        )

                if not segments and transcript_text.strip():
                    segments = [
                        self._build_whisper_style_segment(
                            0,
                            start_time_offset,
                            start_time_offset + max(0.1, input_duration),
                            transcript_text.strip(),
                            words or None,
                        )
                    ]

                require_word_timestamps = os.getenv("PARAKEET_REQUIRE_WORD_TIMESTAMPS", "true").lower() == "true"
                allow_whisper_fallback = (
                    os.getenv("PARAKEET_ALLOW_WHISPER_FALLBACK", "true").strip().lower() == "true"
                )
                word_coverage = 0.0
                if segments:
                    with_words = sum(1 for s in segments if getattr(s, "words", None))
                    word_coverage = with_words / max(len(segments), 1)
                if require_word_timestamps and word_coverage < 0.9:
                    payload = {
                        "parakeet_word_coverage": round(word_coverage, 4),
                        "parakeet_hypothesis_type": type(hypothesis).__name__ if hypothesis is not None else "NoneType",
                        "parakeet_result_type": type(result).__name__ if result is not None else "NoneType",
                        "parakeet_timestamp_payload": self._describe_parakeet_timestamp_payload(hypothesis),
                        "parakeet_raw_word_count": len(raw_words or []),
                        "parakeet_raw_segment_count": len(raw_segments or []),
                        "parakeet_segment_count": len(segments or []),
                        "parakeet_decode_profile_attempts": int(profile_attempt_count),
                    }
                    self._upsert_job_payload_fields(job_id, payload)
                    log(
                        "Parakeet low timestamp coverage: "
                        f"coverage={word_coverage:.1%}, hypothesis={payload['parakeet_hypothesis_type']}, "
                        f"result={payload['parakeet_result_type']}, timestamp_payload={payload['parakeet_timestamp_payload']}, "
                        f"raw_words={payload['parakeet_raw_word_count']}, raw_segments={payload['parakeet_raw_segment_count']}, "
                        f"segments={payload['parakeet_segment_count']}"
                    )
                    if profile_index + 1 < len(retry_profiles):
                        next_profile = retry_profiles[profile_index + 1]
                        log(
                            f"Parakeet low timestamp coverage under profile {retry_profile}. "
                            f"Retrying with {next_profile}."
                        )
                        self._upsert_job_payload_fields(
                            job_id,
                            {
                                "parakeet_profile_retry_from": retry_profile,
                                "parakeet_profile_retry_to": next_profile,
                                "parakeet_profile_retry_reason": (
                                    f"low_word_coverage_{word_coverage:.3f}"
                                ),
                            },
                        )
                        continue
                    if not allow_whisper_fallback and segments:
                        payload.update(
                            {
                                "parakeet_word_timestamps_degraded": True,
                                "parakeet_word_timestamps_degraded_reason": (
                                    f"low_word_coverage_{word_coverage:.3f}"
                                ),
                            }
                        )
                        self._upsert_job_payload_fields(job_id, payload)
                        log(
                            "Parakeet word timestamps unavailable and Whisper fallback disabled. "
                            f"Continuing with segment-level timing only (coverage={word_coverage:.1%})."
                        )
                        self._update_job_status_detail(
                            job_id,
                            "Parakeet returned segment timing without reliable word timestamps. Continuing."
                        )
                        for segment in segments:
                            segment.words = None
                        duration_guess = float(getattr(segments[-1], "end", 0.0)) if segments else 0.0
                        return segments, duration_guess
                    raise RuntimeError(
                        f"Parakeet returned low word timestamp coverage ({word_coverage:.1%}); falling back to Whisper."
                    )

                duration_guess = 0.0
                if segments:
                    duration_guess = float(getattr(segments[-1], "end", 0.0))
                return segments, duration_guess
        finally:
            try:
                del waveform
            except Exception:
                pass
            try:
                del transcribe_input
            except Exception:
                pass
            try:
                del result
            except Exception:
                pass
            try:
                del hypothesis
            except Exception:
                pass
            try:
                del raw_words
            except Exception:
                pass
            try:
                del raw_segments
            except Exception:
                pass
            gc.collect()
            self._clear_cuda_cache()
            if cleanup_input and parakeet_input.exists():
                try:
                    parakeet_input.unlink()
                except Exception:
                    pass

    def _select_transcription_engine(self):
        self._ensure_device()
        pref = (os.getenv("TRANSCRIPTION_ENGINE") or "auto").strip().lower()
        if pref not in {"auto", "whisper", "parakeet"}:
            pref = "auto"
        if self._cuda_unhealthy_reason:
            return "whisper"
        if pref == "whisper":
            return "whisper"
        if pref == "parakeet":
            return "parakeet"
        if self.device == "cuda" and self._parakeet_dependencies_available():
            return "parakeet"
        return "whisper"

    def _transcription_queue_pressure(self, current_job_id: int | None = None) -> int:
        try:
            with Session(engine) as session:
                statement = select(func.count()).select_from(Job).where(
                    Job.job_type == "process",
                    Job.status.in_(("queued", "downloading", "transcribing", "diarizing", "processing")),
                )
                if current_job_id:
                    statement = statement.where(Job.id != int(current_job_id))
                count = session.exec(statement).one()
                return max(0, int(count or 0))
        except Exception:
            return 0

    def _should_reroute_parakeet_for_queue_throughput(
        self,
        video: Video,
        audio_path: Path | None,
        *,
        requested_engine: str,
        job_id: int | None = None,
    ) -> tuple[bool, str | None]:
        if requested_engine != "parakeet":
            return False, None
        configured_preference = (os.getenv("TRANSCRIPTION_ENGINE") or "auto").strip().lower()
        if configured_preference == "parakeet":
            return False, None
        if (os.getenv("PIPELINE_EXECUTION_MODE") or "parallel").strip().lower() != "sequential":
            return False, None
        if os.getenv("PARAKEET_QUEUE_THROUGHPUT_GUARD", "true").strip().lower() != "true":
            return False, None
        if self.parakeet_model is not None:
            return False, None

        try:
            backlog_threshold = max(
                1,
                int((os.getenv("PARAKEET_QUEUE_BACKLOG_WHISPER_THRESHOLD") or "1").strip() or "1"),
            )
        except Exception:
            backlog_threshold = 1
        backlog = self._transcription_queue_pressure(current_job_id=job_id)
        if backlog < backlog_threshold:
            return False, None

        try:
            duration_threshold = max(
                300.0,
                float((os.getenv("PARAKEET_QUEUE_LONG_AUDIO_WHISPER_THRESHOLD_SECONDS") or "1200").strip() or "1200"),
            )
        except Exception:
            duration_threshold = 1200.0

        duration_seconds = 0.0
        try:
            duration_seconds = float(getattr(video, "duration", 0) or 0.0)
        except Exception:
            duration_seconds = 0.0
        if duration_seconds <= 0.0 and audio_path is not None:
            try:
                duration_seconds = float(self._probe_audio_duration_seconds(audio_path) or 0.0)
            except Exception:
                duration_seconds = 0.0
        if duration_seconds < duration_threshold:
            return False, None

        return (
            True,
            "Sequential queue throughput guard rerouted this job to Whisper "
            f"(backlog={backlog}, duration={int(round(duration_seconds))}s, "
            f"threshold={int(round(duration_threshold))}s, parakeet_cached=false).",
        )

    def _normalize_language_code(self, value: str | None) -> str | None:
        raw = str(value or "").strip().lower()
        if not raw:
            return None
        raw = raw.replace("_", "-")
        aliases = {
            "english": "en",
            "spanish": "es",
            "espanol": "es",
            "español": "es",
            "portuguese": "pt",
            "portugues": "pt",
            "português": "pt",
            "french": "fr",
            "francais": "fr",
            "français": "fr",
            "multilingual": "multilingual",
            "bilingual": "multilingual",
            "code-switch": "multilingual",
            "code_switched": "multilingual",
            "mixed": "multilingual",
        }
        if raw in aliases:
            return aliases[raw]
        if len(raw) > 2 and "-" in raw:
            raw = raw.split("-", 1)[0]
        if len(raw) == 2 and raw.isalpha():
            return raw
        return None

    def _language_routing_enabled(self) -> bool:
        return str(os.getenv("MULTILINGUAL_ROUTING_ENABLED", "true")).strip().lower() == "true"

    def _multilingual_whisper_model(self) -> str:
        return str(os.getenv("MULTILINGUAL_WHISPER_MODEL") or "large-v3").strip() or "large-v3"

    def _language_detection_sample_seconds(self) -> int:
        try:
            return max(15, min(int(os.getenv("LANGUAGE_DETECTION_SAMPLE_SECONDS", "45")), 180))
        except Exception:
            return 45

    def _language_detection_confidence_threshold(self) -> float:
        try:
            return max(0.30, min(float(os.getenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.65")), 0.99))
        except Exception:
            return 0.65

    def _infer_language_from_text_hints(self, video: Video) -> dict:
        title = str(getattr(video, "title", "") or "")
        description = str(getattr(video, "description", "") or "")
        haystack = f"{title}\n{description}".lower()
        normalized_existing = self._normalize_language_code(getattr(video, "transcript_language", None))
        if normalized_existing:
            return {
                "language": normalized_existing,
                "confidence": 0.98,
                "source": "video_metadata_existing",
                "reason": f"Existing transcript language '{normalized_existing}' is already stored on the video.",
            }

        bilingual_markers = (
            "bilingual",
            "multilingual",
            "code-switch",
            "code switch",
            "english + spanish",
            "spanish + english",
            "english and spanish",
            "spanish and english",
        )
        if any(marker in haystack for marker in bilingual_markers):
            return {
                "language": "multilingual",
                "confidence": 0.93,
                "source": "video_metadata_text",
                "reason": "Title or description explicitly signals bilingual or multilingual content.",
            }

        language_markers = {
            "es": (" en español", "español", "espanol", "spanish", "latino", "castellano"),
            "pt": ("portuguese", "português", "portugues", "brasileiro"),
            "fr": ("french", "français", "francais"),
        }
        for code, markers in language_markers.items():
            if any(marker in haystack for marker in markers):
                return {
                    "language": code,
                    "confidence": 0.89,
                    "source": "video_metadata_text",
                    "reason": f"Title or description contains clear {code} language markers.",
                }

        return {
            "language": None,
            "confidence": 0.0,
            "source": "video_metadata_text",
            "reason": "No strong non-English or multilingual text markers were found.",
        }

    def _probe_language_with_whisper(
        self,
        audio_path: Path,
        *,
        job_id: int | None = None,
        model_override: str | None = None,
        sample_seconds: int | None = None,
    ) -> dict:
        sample_path = None
        try:
            probe_seconds = max(15, min(int(sample_seconds or self._language_detection_sample_seconds()), 180))
            sample_path = self._slice_audio(audio_path, 0.0, probe_seconds)
            self._load_whisper_model(
                job_id=job_id,
                force_float32=False,
                model_size_override=model_override,
                backend_override="faster_whisper",
            )
            transcribe_params = {
                "beam_size": 1,
                "vad_filter": False,
                "word_timestamps": False,
                "condition_on_previous_text": False,
                "temperature": 0.0,
            }
            whisper_runtime_device = self._whisper_device or self.device
            segments_generator, info = self.whisper_model.transcribe(str(sample_path), **transcribe_params)
            segment_iter = iter(segments_generator)
            try:
                for _ in range(2):
                    next(segment_iter)
            except StopIteration:
                pass
            language = self._normalize_language_code(getattr(info, "language", None))
            confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
            return {
                "language": language,
                "confidence": confidence,
                "source": "audio_probe",
                "reason": f"Whisper language probe on the opening {probe_seconds}s reported {language or 'unknown'} ({confidence:.2f}) on {whisper_runtime_device}.",
                "model": str(model_override or os.getenv("TRANSCRIPTION_MODEL", "tiny")).strip() or "tiny",
            }
        except Exception as e:
            return {
                "language": None,
                "confidence": 0.0,
                "source": "audio_probe",
                "reason": f"Audio language probe failed: {e}",
                "error": str(e),
                "model": str(model_override or os.getenv("TRANSCRIPTION_MODEL", "tiny")).strip() or "tiny",
            }
        finally:
            if sample_path and Path(sample_path).exists():
                try:
                    Path(sample_path).unlink()
                except Exception:
                    pass

    def _resolve_transcription_route(self, video: Video, audio_path: Path | None, job_id: int | None = None) -> dict:
        requested_engine = self._select_transcription_engine()
        route = {
            "requested_engine": requested_engine,
            "engine": requested_engine,
            "language": None,
            "language_confidence": 0.0,
            "language_source": None,
            "language_reason": None,
            "whisper_model_override": None,
            "multilingual_route_applied": False,
            "operational_route_applied": False,
            "operational_route_reason": None,
        }
        text_hint = self._infer_language_from_text_hints(video)
        detected_language = self._normalize_language_code(text_hint.get("language"))
        detected_confidence = float(text_hint.get("confidence") or 0.0)
        detected_source = text_hint.get("source")
        detected_reason = text_hint.get("reason")
        threshold = self._language_detection_confidence_threshold()

        should_probe_audio = (
            audio_path is not None
            and self._language_routing_enabled()
            and (not detected_language or detected_confidence < threshold)
        )
        if should_probe_audio:
            probe_model = self._multilingual_whisper_model()
            audio_probe = self._probe_language_with_whisper(
                audio_path,
                job_id=job_id,
                model_override=probe_model,
                sample_seconds=self._language_detection_sample_seconds(),
            )
            probe_language = self._normalize_language_code(audio_probe.get("language"))
            probe_confidence = float(audio_probe.get("confidence") or 0.0)
            if probe_language and probe_confidence >= max(0.45, threshold - 0.10):
                if detected_language and detected_language != probe_language and detected_confidence >= threshold:
                    detected_language = "multilingual"
                    detected_confidence = max(detected_confidence, probe_confidence)
                    detected_source = "metadata_plus_audio_probe"
                    detected_reason = (
                        f"{text_hint.get('reason')} Audio probe disagreed with {probe_language} ({probe_confidence:.2f}), "
                        "so the episode is treated as multilingual."
                    )
                else:
                    detected_language = probe_language
                    detected_confidence = probe_confidence
                    detected_source = audio_probe.get("source")
                    detected_reason = audio_probe.get("reason")
            elif not detected_language and audio_probe.get("reason"):
                detected_source = audio_probe.get("source")
                detected_reason = audio_probe.get("reason")

        route["language"] = detected_language
        route["language_confidence"] = detected_confidence
        route["language_source"] = detected_source
        route["language_reason"] = detected_reason

        non_english = detected_language not in {None, "", "en"}
        if self._language_routing_enabled() and non_english:
            route["engine"] = "whisper"
            route["whisper_model_override"] = self._multilingual_whisper_model()
            route["multilingual_route_applied"] = True
            return route

        reroute_for_queue, reroute_reason = self._should_reroute_parakeet_for_queue_throughput(
            video,
            audio_path,
            requested_engine=requested_engine,
            job_id=job_id,
        )
        if reroute_for_queue:
            route["engine"] = "whisper"
            route["operational_route_applied"] = True
            route["operational_route_reason"] = reroute_reason
        return route

    def test_transcription_engine(self, requested_engine: str = "auto", whisper_backend_override: str | None = None) -> dict:
        self._ensure_device()
        req = (requested_engine or "auto").strip().lower()
        if req not in {"auto", "whisper", "parakeet"}:
            req = "auto"
        requested_whisper_backend = self._normalize_whisper_backend(whisper_backend_override or os.getenv("WHISPER_BACKEND"))

        response = {
            "status": "ok",
            "requested_engine": req,
            "resolved_engine": None,
            "device": self.device,
            "whisper_runtime_device": None,
            "whisper_backend_requested": requested_whisper_backend,
            "whisper_backend_resolved": None,
            "whisper_backend_available": None,
            "cuda_unhealthy": bool(self._cuda_unhealthy_reason),
            "cuda_unhealthy_reason": self._cuda_unhealthy_reason,
            "parakeet_dependencies_available": self._parakeet_dependencies_available(),
            "whisper_model": os.getenv("TRANSCRIPTION_MODEL", "medium"),
            "whisper_compute_type": os.getenv("TRANSCRIPTION_COMPUTE_TYPE", "").strip() or None,
            "parakeet_model": (os.getenv("PARAKEET_MODEL") or "nvidia/parakeet-tdt-0.6b-v2").strip(),
            "parakeet_batch_size_requested": max(1, min(int(os.getenv("PARAKEET_BATCH_SIZE", "16")), 64)),
            "parakeet_batch_auto": os.getenv("PARAKEET_BATCH_AUTO", "true").strip().lower() == "true",
            "parakeet_batch_hard_max": max(1, min(int(os.getenv("PARAKEET_BATCH_HARD_MAX", "4")), 64)),
            "parakeet_dynamic_batch_cap": (
                int(self._parakeet_dynamic_batch_cap) if self._parakeet_dynamic_batch_cap is not None else None
            ),
            "parakeet_max_gpu_memory_fraction": None,
            "parakeet_unload_after_transcribe": os.getenv("PARAKEET_UNLOAD_AFTER_TRANSCRIBE", "auto"),
            "parakeet_release_other_models_before_transcribe": os.getenv("PARAKEET_RELEASE_OTHER_MODELS_BEFORE_TRANSCRIBE", "false").strip().lower() == "true",
            "parakeet_keep_loaded_min_free_gb": None,
            "parakeet_keep_loaded_min_free_ratio": None,
            "fallback_used": False,
            "error": None,
        }
        total_vram_for_thresholds = float(self._gpu_total_vram_bytes or 0) / (1024 ** 3) if self._gpu_total_vram_bytes else 0.0
        if total_vram_for_thresholds <= 0 and self.device == "cuda":
            snap = self._cuda_memory_snapshot()
            total_vram_for_thresholds = float(snap.get("total") or 0) / (1024 ** 3)
        raw_fraction = (os.getenv("PARAKEET_MAX_GPU_MEMORY_FRACTION") or "").strip()
        try:
            if raw_fraction:
                response["parakeet_max_gpu_memory_fraction"] = max(0.50, min(float(raw_fraction), 0.98))
            elif total_vram_for_thresholds >= 28.0:
                response["parakeet_max_gpu_memory_fraction"] = 0.92
            elif total_vram_for_thresholds >= 20.0:
                response["parakeet_max_gpu_memory_fraction"] = 0.88
            else:
                response["parakeet_max_gpu_memory_fraction"] = 0.85
        except Exception:
            response["parakeet_max_gpu_memory_fraction"] = 0.85
        keep_gb, keep_ratio = self._resolve_parakeet_keep_loaded_thresholds(total_vram_for_thresholds)
        response["parakeet_keep_loaded_min_free_gb"] = round(float(keep_gb), 2)
        response["parakeet_keep_loaded_min_free_ratio"] = round(float(keep_ratio), 3)

        def _check_whisper():
            self._load_whisper_model(job_id=None, force_float32=False, backend_override=requested_whisper_backend)
            response["resolved_engine"] = "whisper"
            response["whisper_compute_type"] = self._whisper_compute_type or response["whisper_compute_type"]
            response["whisper_runtime_device"] = self._whisper_device or self.device
            response["whisper_backend_resolved"] = self._whisper_backend or "faster_whisper"
            backend_info = self._resolve_whisper_backend(requested_whisper_backend)
            response["whisper_backend_available"] = bool(backend_info.get("available"))
            if backend_info.get("fallback_used"):
                response["fallback_used"] = True
                response["whisper_backend_fallback_reason"] = backend_info.get("fallback_reason")
            elif (
                requested_whisper_backend == "insanely_fast_whisper"
                and response["whisper_backend_resolved"] not in {"insanely_fast_whisper", "transformers_compat"}
            ):
                response["fallback_used"] = True
                response["whisper_backend_fallback_reason"] = (
                    "Requested insanely_fast_whisper-compatible backend could not be used at runtime. "
                    "Falling back to faster_whisper."
                )

        def _check_parakeet():
            if self.device == "cuda" and response.get("parakeet_release_other_models_before_transcribe"):
                self._release_diarization_models("test_before_parakeet")
            self._load_parakeet_model(job_id=None)
            response["resolved_engine"] = "parakeet"
            response["parakeet_effective_batch_size"] = self._resolve_parakeet_batch_size(
                int(response.get("parakeet_batch_size_requested") or 16)
            )
            if self.device == "cuda":
                response["cuda_memory"] = self._snap_to_gb_dict(self._cuda_memory_snapshot())

        try:
            if req == "whisper":
                _check_whisper()
                return response
            if req == "parakeet":
                _check_parakeet()
                return response

            # auto
            resolved = self._select_transcription_engine()
            if resolved == "parakeet":
                try:
                    _check_parakeet()
                except Exception:
                    response["fallback_used"] = True
                    _check_whisper()
            else:
                _check_whisper()
            return response
        except Exception as e:
            response["status"] = "error"
            response["error"] = str(e)
            # Keep resolved_engine best-effort for UI clarity.
            if response.get("resolved_engine") is None:
                response["resolved_engine"] = "parakeet" if req == "parakeet" else ("whisper" if req == "whisper" else self._select_transcription_engine())
            return response

    def _load_models(self, job_id: int = None, load_transcription_model: bool = False):
        import torch
        import warnings
        from huggingface_hub import snapshot_download

        with warnings.catch_warnings():
            # This app feeds pyannote preloaded audio tensors, so its optional
            # torchcodec-backed file decoder is not required at runtime.
            warnings.filterwarnings(
                "ignore",
                message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
            )
            from pyannote.audio import Pipeline, Inference, Model

        self._ensure_device()

        if load_transcription_model:
            self._load_whisper_model(job_id=job_id, force_float32=False)

        # If FP16 cuBLAS isn't safe, tell PyTorch to use float32 for matmul
        if self._force_float32 and self.device == "cuda":
            torch.set_float32_matmul_precision('high')

        pyannote_profile = None
        if not self.diarization_pipeline or not self.embedding_model or not self.embedding_inference:
            pyannote_profile = self._start_component_memory_profile()

        diarization_model_ref = "pyannote/speaker-diarization-3.1"
        embedding_model_ref = "pyannote/embedding"
        try:
            with temporary_disabled_blackhole_proxies():
                resolved_diarization = snapshot_download(diarization_model_ref, local_files_only=True)
            if resolved_diarization:
                diarization_model_ref = resolved_diarization
                log(f"Using cached diarization pipeline from {resolved_diarization}")
        except Exception as e:
            log_verbose(f"Diarization cache probe missed for {diarization_model_ref}: {e}")

        try:
            with temporary_disabled_blackhole_proxies():
                resolved_embedding = snapshot_download(embedding_model_ref, local_files_only=True)
            if resolved_embedding:
                embedding_model_ref = resolved_embedding
                log(f"Using cached speaker embedding model from {resolved_embedding}")
        except Exception as e:
            log_verbose(f"Embedding cache probe missed for {embedding_model_ref}: {e}")

        if not self.diarization_pipeline:
           log("Loading Diarization pipeline...")
           self._update_job_status_detail(job_id, "Loading diarization pipeline...")
           try:
               with temporary_disabled_blackhole_proxies():
                   self.diarization_pipeline = Pipeline.from_pretrained(
                       diarization_model_ref,
                       token=os.getenv("HF_TOKEN")
                   )
               if self.diarization_pipeline:
                   # By default, Pyannote uses a batch_size of 32 which massively underutilizes
                   # modern GPUs and results in very low VRAM allocation and slower processing.
                   # Scaling this to 128+ saturates the GPU computation for huge performance gains.
                   batch_size = self._get_pyannote_batch_size()
                   self._set_pyannote_batch_size(batch_size)
                   log_verbose(f"Pyannote properly scaled for GPU - Batch Size: {batch_size}")

                   if self._force_float32:
                       # Cast all sub-models to float32 to avoid cuBLAS FP16 errors
                       self._cast_pipeline_to_float32(self.diarization_pipeline)
                       log_verbose("Diarization pipeline cast to float32 for GPU compatibility")
                   self.diarization_pipeline.to(torch.device(self.device))
           except Exception as e:
               log(f"Failed to load Diarization pipeline: {e}")
               # Store the actual error so we can show it to users at diarization time
               self._diarization_load_error = str(e)
               # Continue - we'll show the actual error when diarization is attempted

        if not self.embedding_model:
            log("Loading Embedding model...")
            self._update_job_status_detail(job_id, "Loading speaker embedding model...")
            try:
                import warnings
                with warnings.catch_warnings():
                    # Suppress known harmless warnings from lightning checkpoint migration
                    # and pyannote task-dependent loss function notice
                    warnings.filterwarnings("ignore", message=".*Redirecting import of pytorch_lightning.*")
                    warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*callback states.*")
                    warnings.filterwarnings("ignore", message=".*task-dependent loss function.*")
                    warnings.filterwarnings("ignore", message=".*Found keys that are not in the model state dict.*")
                    with temporary_disabled_blackhole_proxies():
                        self.embedding_model = Model.from_pretrained(
                            embedding_model_ref,
                            token=os.getenv("HF_TOKEN")
                        )
                if self.embedding_model:
                    if self._force_float32:
                        self.embedding_model = self.embedding_model.float()
                        log_verbose("Embedding model cast to float32 for GPU compatibility")
                    self.embedding_inference = Inference(self.embedding_model, window="whole")
                    self.embedding_inference.to(torch.device(self.device))
            except Exception as e:
                log(f"Failed to load Embedding model: {e}")

        if (
            pyannote_profile is not None
            and self.diarization_pipeline is not None
            and self.embedding_model is not None
            and self.embedding_inference is not None
        ):
            self._finish_component_memory_profile("pyannote", pyannote_profile, loaded=True)

        # Clear status detail after models are loaded
        self._update_job_status_detail(job_id, None)

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
            
        with Session(engine) as session:
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

        with Session(engine) as session:
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

        with Session(engine) as session:
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

    def get_voicefixer_cleaned_absolute_path(self, video: Video) -> Path | None:
        if (video.media_source_type or "youtube") != "upload":
            return None
        return self.get_manual_media_absolute_path(getattr(video, "voicefixer_cleaned_path", None))

    def _voicefixer_apply_scope(self, video: Video) -> str:
        scope = str(getattr(video, "voicefixer_apply_scope", "") or "").strip().lower()
        if scope in {"none", "playback", "processing", "both"}:
            return scope
        return "both" if bool(getattr(video, "voicefixer_use_cleaned", False)) else "none"

    def _voicefixer_leveling_filter(self, mode: str) -> str | None:
        preset = str(mode or "off").strip().lower()
        if preset == "gentle":
            return "dynaudnorm=f=150:g=7:p=0.9:m=8"
        if preset == "balanced":
            return "dynaudnorm=f=200:g=11:p=0.92:m=12,acompressor=threshold=0.18:ratio=2.2:attack=15:release=220:makeup=1"
        if preset == "strong":
            return "dynaudnorm=f=250:g=15:p=0.95:m=16,acompressor=threshold=0.12:ratio=3.0:attack=10:release=180:makeup=2"
        return None

    def _voicefixer_paths_for_manual_video(self, video: Video) -> tuple[Path, Path, Path, Path]:
        original_path = self.get_original_manual_media_path(video)
        if original_path is None:
            raise FileNotFoundError(f"Original uploaded media is missing for video {video.id}")

        stem = original_path.stem
        working_dir = original_path.parent
        input_wav = working_dir / f"{stem}.voicefixer.input.wav"
        restored_wav = working_dir / f"{stem}.voicefixer.restored.wav"
        final_ext = ".mp4" if str(getattr(video, "media_kind", "") or "").lower() == "video" else ".wav"
        final_path = working_dir / f"{stem}.voicefixer.cleaned{final_ext}"
        return original_path, input_wav, restored_wav, final_path

    def _cleanup_workbench_dir(self, video: Video) -> Path:
        original_path = self.get_original_manual_media_path(video)
        if original_path is None:
            raise FileNotFoundError(f"Original uploaded media is missing for video {video.id}")
        workbench_dir = original_path.parent / ".cleanup_workbench"
        workbench_dir.mkdir(parents=True, exist_ok=True)
        return workbench_dir

    def _cleanup_workbench_state_path(self, video: Video) -> Path:
        return self._cleanup_workbench_dir(video) / "state.json"

    def _load_cleanup_workbench_state(self, video: Video) -> dict:
        path = self._cleanup_workbench_state_path(video)
        default_state = {"analysis": None, "selected_candidate_id": None, "candidates": {}}
        if not path.exists():
            return dict(default_state)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("analysis", None)
                data.setdefault("selected_candidate_id", None)
                data.setdefault("candidates", {})
                return data
        except Exception:
            pass
        return dict(default_state)

    def _save_cleanup_workbench_state(self, video: Video, state: dict) -> None:
        path = self._cleanup_workbench_state_path(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cleanup_candidate_stage_config(self, stage: str) -> dict:
        stage_key = str(stage or "").strip().lower()
        if stage_key == "enhancement":
            return {
                "stage": "enhancement",
                "task": "speech_enhancement",
                "models": {
                    "FRCRN_SE_16K": "Focused on rough 16k speech cleanup",
                    "MossFormerGAN_SE_16K": "Alternative 16k enhancement with stronger denoising",
                    "MossFormer2_SE_48K": "Full-band 48k enhancement for higher fidelity material",
                },
            }
        raise ValueError("Unsupported ClearVoice stage.")

    def _cleanup_candidate_filename(self, stage: str, model_name: str, source_candidate_id: str | None = None) -> str:
        safe_stage = re.sub(r"[^a-z0-9]+", "_", str(stage or "").strip().lower()).strip("_") or "stage"
        safe_model = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name or "").strip()).strip("_") or "model"
        if source_candidate_id:
            safe_source = re.sub(r"[^A-Za-z0-9]+", "_", str(source_candidate_id or "").strip()).strip("_")
            return f"{safe_stage}.{safe_model}.from_{safe_source}.wav"
        return f"{safe_stage}.{safe_model}.wav"

    def _cleanup_candidate_path(self, video: Video, stage: str, model_name: str, source_candidate_id: str | None = None) -> Path:
        return self._cleanup_workbench_dir(video) / self._cleanup_candidate_filename(stage, model_name, source_candidate_id)

    def _cleanup_candidate_audio_url(self, video: Video, filename: str) -> str | None:
        safe_name = str(filename or "").strip()
        if not safe_name:
            return None
        audio_path = self._cleanup_workbench_dir(video) / Path(safe_name).name
        if not audio_path.exists():
            return None
        version = ""
        try:
            version = str(int(audio_path.stat().st_mtime_ns))
        except Exception:
            version = str(int(time.time() * 1000))
        params = {"name": Path(safe_name).name, "v": version}
        return f"/videos/{int(video.id)}/cleanup/workbench/audio?{urllib.parse.urlencode(params)}"

    def _get_cleanup_selected_candidate_path(self, video: Video) -> Path | None:
        state = self._load_cleanup_workbench_state(video)
        selected_candidate_id = str(state.get("selected_candidate_id") or "").strip()
        if not selected_candidate_id:
            return None
        candidates = state.get("candidates", {})
        if not isinstance(candidates, dict):
            return None
        candidate = candidates.get(selected_candidate_id)
        if not isinstance(candidate, dict):
            return None
        filename = str(candidate.get("audio_filename") or "").strip()
        if not filename:
            return None
        candidate_path = self._cleanup_workbench_dir(video) / Path(filename).name
        if not candidate_path.exists():
            return None
        return candidate_path

    def build_cleanup_workbench(self, video_id: int) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Cleanup workbench is only available for uploaded manual media.")
            state = self._load_cleanup_workbench_state(video)
            candidates_state = state.get("candidates", {}) if isinstance(state.get("candidates"), dict) else {}
            selected_candidate_id = str(state.get("selected_candidate_id") or "").strip() or None
            candidates: list[dict] = []
            stale_keys: list[str] = []
            for candidate_id, item in sorted(candidates_state.items(), key=lambda row: str(row[0])):
                if not isinstance(item, dict):
                    stale_keys.append(str(candidate_id))
                    continue
                filename = str(item.get("audio_filename") or "").strip()
                if not filename:
                    stale_keys.append(str(candidate_id))
                    continue
                audio_path = self._cleanup_workbench_dir(video) / Path(filename).name
                if not audio_path.exists():
                    stale_keys.append(str(candidate_id))
                    continue
                candidates.append({
                    "candidate_id": str(candidate_id),
                    "stage": str(item.get("stage") or "enhancement"),
                    "task": str(item.get("task") or "speech_enhancement"),
                    "model_name": str(item.get("model_name") or ""),
                    "source_candidate_id": str(item.get("source_candidate_id") or "").strip() or None,
                    "source_label": str(item.get("source_label") or "Original upload"),
                    "selected_for_processing": str(candidate_id) == selected_candidate_id,
                    "audio_url": self._cleanup_candidate_audio_url(video, filename),
                    "sample_rate": int(item.get("sample_rate")) if item.get("sample_rate") is not None else None,
                    "duration_seconds": float(item.get("duration_seconds")) if item.get("duration_seconds") is not None else None,
                    "peak": float(item.get("peak")) if item.get("peak") is not None else None,
                    "rms": float(item.get("rms")) if item.get("rms") is not None else None,
                })
            if stale_keys:
                for key in stale_keys:
                    candidates_state.pop(key, None)
                if selected_candidate_id and selected_candidate_id in stale_keys:
                    state["selected_candidate_id"] = None
                    selected_candidate_id = None
                state["candidates"] = candidates_state
                self._save_cleanup_workbench_state(video, state)

            selected_source_label = "Original upload"
            if selected_candidate_id:
                selected_row = next((row for row in candidates if row["candidate_id"] == selected_candidate_id), None)
                if selected_row:
                    selected_source_label = f"{selected_row['model_name']} candidate"

            return {
                "selected_candidate_id": selected_candidate_id,
                "selected_source_label": selected_source_label,
                "analysis": state.get("analysis"),
                "candidates": candidates,
            }

    def analyze_cleanup_workbench_audio(self, video_id: int) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Cleanup workbench is only available for uploaded manual media.")
            source_path = self.get_original_manual_media_path(video)
            if source_path is None or not source_path.exists():
                raise FileNotFoundError("Original uploaded media is missing.")
            state = self._load_cleanup_workbench_state(video)

        self._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="analyze",
            status="running",
            stage="analyze",
            message="Inspecting the original uploaded audio for pre-cleanup workbench setup...",
            percent=16,
        )
        analysis = self._analyze_audio_file(source_path, source_label="Original upload")
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            state = self._load_cleanup_workbench_state(video)
            state["analysis"] = analysis
            self._save_cleanup_workbench_state(video, state)
        self._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="analyze",
            status="completed",
            stage="complete",
            message="Cleanup analysis is ready.",
            percent=100,
        )
        return self.build_cleanup_workbench(video_id)

    def run_cleanup_clearvoice_candidate(
        self,
        video_id: int,
        *,
        stage: str,
        model_name: str,
        source_candidate_id: str | None = None,
    ) -> dict:
        stage_config = self._cleanup_candidate_stage_config(stage)
        model_name = str(model_name or "").strip()
        if model_name not in stage_config["models"]:
            raise ValueError("Unsupported ClearVoice model for this stage.")

        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Cleanup workbench is only available for uploaded manual media.")
            state = self._load_cleanup_workbench_state(video)
            source_path = self.get_original_manual_media_path(video)
            source_label = "Original upload"
            if source_path is None or not source_path.exists():
                raise FileNotFoundError("Original uploaded media is missing.")
            if source_candidate_id:
                candidates_state = state.get("candidates", {}) if isinstance(state.get("candidates"), dict) else {}
                candidate = candidates_state.get(str(source_candidate_id))
                if not isinstance(candidate, dict):
                    raise ValueError("Selected ClearVoice source candidate was not found.")
                filename = str(candidate.get("audio_filename") or "").strip()
                candidate_path = self._cleanup_workbench_dir(video) / Path(filename).name
                if not candidate_path.exists():
                    raise FileNotFoundError("Selected ClearVoice source candidate audio file is missing.")
                source_path = candidate_path
                source_label = f"{str(candidate.get('model_name') or 'candidate')} candidate"
            output_path = self._cleanup_candidate_path(video, stage_config["stage"], model_name, source_candidate_id)

        self._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="clearvoice_candidate",
            status="running",
            stage="load_model",
            message=f"Loading ClearVoice {model_name} for pre-cleanup enhancement...",
            percent=18,
        )
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(f"ClearVoice runtime could not import torch: {e}")
        try:
            import torchaudio  # type: ignore
        except Exception as e:
            torch_version = str(getattr(torch, "__version__", "") or "").strip() or "unknown"
            raise RuntimeError(
                "ClearVoice runtime could not import torchaudio. "
                f"Installed torch version: {torch_version}. "
                "This usually means the backend environment has mismatched torch/torchaudio wheels. "
                f"Repair the ClearVoice runtime from Settings, then try again. Raw error: {e}"
            )
        try:
            from clearvoice import ClearVoice  # type: ignore
        except Exception as e:
            raise RuntimeError(f"ClearVoice is not available: {e}")

        processor = ClearVoice(task=stage_config["task"], model_names=[model_name])
        self._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="clearvoice_candidate",
            status="running",
            stage="render",
            message=f"Generating {model_name} candidate audio...",
            percent=62,
        )
        result = processor(input_path=str(source_path), online_write=False)
        processor.write(result, output_path=str(output_path))
        stats = self._analyze_audio_file(output_path, source_label=f"{model_name} candidate")

        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            state = self._load_cleanup_workbench_state(video)
            candidates_state = state.setdefault("candidates", {})
            candidate_id = f"{stage_config['stage']}::{model_name}" if not source_candidate_id else f"{stage_config['stage']}::{model_name}::{source_candidate_id}"
            candidates_state[candidate_id] = {
                "stage": stage_config["stage"],
                "task": stage_config["task"],
                "model_name": model_name,
                "source_candidate_id": str(source_candidate_id or "").strip() or None,
                "source_label": source_label,
                "audio_filename": output_path.name,
                "sample_rate": stats.get("sample_rate"),
                "duration_seconds": stats.get("duration_seconds"),
                "peak": stats.get("peak"),
                "rms": stats.get("rms"),
            }
            self._save_cleanup_workbench_state(video, state)
        self._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="clearvoice_candidate",
            status="completed",
            stage="complete",
            message=f"{model_name} candidate is ready to preview.",
            percent=100,
        )
        return self.build_cleanup_workbench(video_id)

    def select_cleanup_workbench_candidate(self, video_id: int, *, candidate_id: str | None = None) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Cleanup workbench is only available for uploaded manual media.")
            state = self._load_cleanup_workbench_state(video)
            candidates_state = state.get("candidates", {}) if isinstance(state.get("candidates"), dict) else {}
            resolved_id = str(candidate_id or "").strip() or None
            if resolved_id is not None and resolved_id not in candidates_state:
                raise ValueError("Selected ClearVoice candidate was not found.")
            if resolved_id is not None:
                filename = str((candidates_state.get(resolved_id) or {}).get("audio_filename") or "").strip()
                candidate_path = self._cleanup_workbench_dir(video) / Path(filename).name
                if not candidate_path.exists():
                    raise FileNotFoundError("Selected ClearVoice candidate audio file is missing.")
            state["selected_candidate_id"] = resolved_id
            self._save_cleanup_workbench_state(video, state)
        return self.build_cleanup_workbench(video_id)

    def _set_voicefixer_state(
        self,
        video_id: int,
        *,
        status: str | None = None,
        use_cleaned: bool | None = None,
        cleaned_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                return
            if status is not None:
                video.voicefixer_status = status
            if use_cleaned is not None:
                video.voicefixer_use_cleaned = bool(use_cleaned)
            if cleaned_path is not None:
                video.voicefixer_cleaned_path = cleaned_path
            if error is not None:
                video.voicefixer_error = error
            session.add(video)
            session.commit()

    def _set_reconstruction_state(
        self,
        video_id: int,
        *,
        status: str | None = None,
        audio_path: str | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> None:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                return
            if status is not None:
                video.reconstruction_status = status
            if audio_path is not None:
                video.reconstruction_audio_path = audio_path
            if error is not None:
                video.reconstruction_error = error
            if model is not None:
                video.reconstruction_model = model
            session.add(video)
            session.commit()

    def _check_job_not_paused(self, job_id: int | None) -> None:
        if not job_id:
            return
        with Session(engine) as session:
            job = session.get(Job, int(job_id))
            if not job:
                return
            if str(job.status or "").strip().lower() == "paused":
                raise JobPausedException("Job paused by user")
            if str(job.status or "").strip().lower() == "cancelled":
                raise JobCancelledException("Job cancelled by user")

    def _reconstruction_output_paths(self, video: Video) -> tuple[Path, Path]:
        original_path = self.get_original_manual_media_path(video)
        if original_path is None:
            raise FileNotFoundError(f"Original uploaded media is missing for video {video.id}")
        working_dir = original_path.parent
        stem = original_path.stem
        wav_path = working_dir / f"{stem}.reconstructed.wav"
        json_path = working_dir / f"{stem}.reconstructed.segments.json"
        return wav_path, json_path

    def _reconstruction_workbench_dir(self, video: Video) -> Path:
        original_path = self.get_original_manual_media_path(video)
        if original_path is None:
            raise FileNotFoundError(f"Original uploaded media is missing for video {video.id}")
        workbench_dir = original_path.parent / ".reconstruction_workbench"
        workbench_dir.mkdir(parents=True, exist_ok=True)
        return workbench_dir

    def _reconstruction_workbench_state_path(self, video: Video) -> Path:
        return self._reconstruction_workbench_dir(video) / "state.json"

    def _load_reconstruction_workbench_state(self, video: Video) -> dict:
        path = self._reconstruction_workbench_state_path(video)
        if not path.exists():
            return {"speakers": {}, "preview": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("speakers", {})
                data.setdefault("preview", {})
                return data
        except Exception:
            pass
        return {"speakers": {}, "preview": {}}

    def _save_reconstruction_workbench_state(self, video: Video, state: dict) -> None:
        path = self._reconstruction_workbench_state_path(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _reconstruction_sample_clip_path(self, work_dir: Path, speaker_id: int, segment_id: int) -> Path:
        return work_dir / f"speaker_{int(speaker_id)}_sample_{int(segment_id)}.wav"

    def _reconstruction_sample_cleaned_clip_path(self, work_dir: Path, speaker_id: int, segment_id: int) -> Path:
        return work_dir / f"speaker_{int(speaker_id)}_sample_{int(segment_id)}.cleaned.wav"

    def _reconstruction_speaker_test_path(self, work_dir: Path, speaker_id: int) -> Path:
        return work_dir / f"speaker_{int(speaker_id)}.test.wav"

    def _reconstruction_preview_path(self, work_dir: Path, segment_id: int) -> Path:
        return work_dir / f"preview_segment_{int(segment_id)}.wav"

    def _ensure_reconstruction_sample_clip(
        self,
        source_path: Path,
        work_dir: Path,
        speaker_id: int,
        seg: TranscriptSegment,
    ) -> Path:
        clip_path = self._reconstruction_sample_clip_path(work_dir, int(speaker_id), int(getattr(seg, "id", 0) or 0))
        if not clip_path.exists():
            self._write_audio_clip(source_path, clip_path, float(seg.start_time), float(seg.end_time))
        return clip_path

    def _reconstruction_sample_state(self, state: dict, speaker_id: int, segment_id: int) -> dict:
        speakers = state.setdefault("speakers", {})
        speaker_state = speakers.setdefault(str(int(speaker_id)), {})
        samples = speaker_state.setdefault("samples", {})
        return samples.setdefault(str(int(segment_id)), {})

    def _sync_reconstruction_workbench_state(
        self,
        video: Video,
        grouped_segments: dict[int, list[TranscriptSegment]],
    ) -> dict:
        state = self._load_reconstruction_workbench_state(video)
        speakers_state = state.setdefault("speakers", {})
        valid_speaker_keys = {str(int(speaker_id)) for speaker_id in grouped_segments.keys()}
        for stale_key in list(speakers_state.keys()):
            if stale_key not in valid_speaker_keys:
                speakers_state.pop(stale_key, None)

        for speaker_id, speaker_segments in grouped_segments.items():
            speaker_key = str(int(speaker_id))
            speaker_state = speakers_state.setdefault(speaker_key, {})
            sample_state = speaker_state.setdefault("samples", {})
            ranked = sorted(
                speaker_segments,
                key=lambda s: (-(float(s.end_time) - float(s.start_time)), float(s.start_time)),
            )
            all_ids = [int(getattr(seg, "id", 0) or 0) for seg in ranked if int(getattr(seg, "id", 0) or 0) > 0]
            valid_ids = {str(seg_id) for seg_id in all_ids}
            for stale_segment_key in list(sample_state.keys()):
                if stale_segment_key not in valid_ids:
                    sample_state.pop(stale_segment_key, None)

            active_ids = [int(seg_id) for seg_id in speaker_state.get("active_sample_segment_ids", []) if int(seg_id) in all_ids]
            if not active_ids:
                active_ids = []
            for seg in ranked:
                seg_id = int(getattr(seg, "id", 0) or 0)
                if seg_id <= 0:
                    continue
                seg_state = sample_state.setdefault(str(seg_id), {})
                if seg_id in active_ids:
                    continue
                if len(active_ids) >= 3:
                    break
                if bool(seg_state.get("rejected")):
                    continue
                active_ids.append(seg_id)
            speaker_state["active_sample_segment_ids"] = active_ids[:8]

            selected_id = speaker_state.get("selected_sample_segment_id")
            try:
                selected_id = int(selected_id) if selected_id is not None else None
            except Exception:
                selected_id = None
            if selected_id not in active_ids:
                selected_id = next((seg_id for seg_id in active_ids if not bool(sample_state.get(str(seg_id), {}).get("rejected"))), active_ids[0] if active_ids else None)
            speaker_state["selected_sample_segment_id"] = selected_id
            speaker_state["approved"] = bool(speaker_state.get("approved", False))
            speaker_state.setdefault("latest_test_audio_filename", None)
            speaker_state.setdefault("latest_test_text", None)
            speaker_state.setdefault("latest_test_mode", None)

        self._save_reconstruction_workbench_state(video, state)
        return state

    def _resolve_reconstruction_reference_for_speaker(
        self,
        video: Video,
        speaker_id: int,
        speaker_segments: list[TranscriptSegment],
        source_path: Path,
        work_dir: Path,
        state: dict,
    ) -> dict | None:
        speaker_state = state.get("speakers", {}).get(str(int(speaker_id)), {})
        sample_state = speaker_state.get("samples", {}) if isinstance(speaker_state, dict) else {}
        by_id = {int(getattr(seg, "id", 0) or 0): seg for seg in speaker_segments}
        preferred_segment_id = speaker_state.get("selected_sample_segment_id")
        try:
            preferred_segment_id = int(preferred_segment_id) if preferred_segment_id is not None else None
        except Exception:
            preferred_segment_id = None
        if preferred_segment_id and preferred_segment_id in by_id:
            seg = by_id[preferred_segment_id]
            seg_state = sample_state.get(str(preferred_segment_id), {})
            cleaned_name = str(seg_state.get("cleaned_audio_filename") or "").strip() if isinstance(seg_state, dict) else ""
            if cleaned_name:
                cleaned_path = work_dir / cleaned_name
                if cleaned_path.exists():
                    return {
                        "path": cleaned_path,
                        "text": str(getattr(seg, "text", "") or "").strip(),
                        "start": float(seg.start_time),
                        "end": float(seg.end_time),
                        "source": "selected_cleaned_sample",
                    }
            try:
                clip_path = self._ensure_reconstruction_sample_clip(source_path, work_dir, int(speaker_id), seg)
                return {
                    "path": clip_path,
                    "text": str(getattr(seg, "text", "") or "").strip(),
                    "start": float(seg.start_time),
                    "end": float(seg.end_time),
                    "source": "selected_sample",
                }
            except Exception:
                pass
        return None

    def _reconstruction_default_instruction_template(self) -> str:
        return (
            "Speak with the exact same intonation, emotion, rhythm, breathing, pauses, and emphasis as "
            "the reference audio. Maintain the original speaking style and prosody precisely, but deliver "
            "clearly at normal volume without background noise."
        )

    def _get_reconstruction_mode(self, video: Video) -> str:
        return "performance"

    def _get_reconstruction_instruction_template(self, video: Video) -> str:
        value = str(getattr(video, "reconstruction_instruction_template", "") or "").strip()
        return value or self._reconstruction_default_instruction_template()

    def _reference_duration_score(self, start_time: float, end_time: float, text: str) -> float:
        duration = max(0.0, float(end_time) - float(start_time))
        if duration <= 0:
            return 0.0
        words = len([token for token in str(text or "").split() if token])
        return duration + min(words / 50.0, 1.5)

    def _fit_waveform_duration_with_ffmpeg(self, wav, sample_rate: int, stretch_rate: float):
        import numpy as np
        import soundfile as sf  # type: ignore
        import tempfile

        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr.astype(np.float32)

        def _build_atempo_chain(rate: float) -> str:
            rate = max(0.05, float(rate))
            factors: list[float] = []
            while rate < 0.5:
                factors.append(0.5)
                rate /= 0.5
            while rate > 2.0:
                factors.append(2.0)
                rate /= 2.0
            factors.append(rate)
            return ",".join(f"atempo={max(0.5, min(2.0, factor)):.6f}" for factor in factors)

        with tempfile.TemporaryDirectory(prefix="reconstruction-fit-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_wav = tmp_path / "input.wav"
            output_wav = tmp_path / "output.wav"
            sf.write(str(input_wav), arr, int(sample_rate))
            self._run_external_command(
                [
                    self._get_ffmpeg_cmd(),
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(input_wav),
                    "-af",
                    _build_atempo_chain(stretch_rate),
                    "-ac",
                    "1",
                    "-ar",
                    str(int(sample_rate)),
                    "-c:a",
                    "pcm_s16le",
                    str(output_wav),
                ],
                "Failed to time-fit reconstruction audio with ffmpeg",
                timeout=1800,
            )
            stretched, stretched_sr = sf.read(str(output_wav), dtype="float32", always_2d=False)
            result = np.asarray(stretched, dtype=np.float32).reshape(-1)
            if int(stretched_sr or 0) != int(sample_rate):
                raise RuntimeError(f"ffmpeg time-fit changed sample rate from {sample_rate} to {stretched_sr}")
            return result.astype(np.float32)

    def _fit_waveform_duration(self, wav, sample_rate: int, target_seconds: float):
        import numpy as np

        target_seconds = max(0.01, float(target_seconds))
        target_samples = max(1, int(round(target_seconds * sample_rate)))
        if wav is None:
            return np.zeros(target_samples, dtype=np.float32)
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros(target_samples, dtype=np.float32)
        current_seconds = arr.size / float(sample_rate)
        if abs(current_seconds - target_seconds) <= 0.02:
            if arr.size == target_samples:
                return arr.astype(np.float32)
            if arr.size > target_samples:
                return arr[:target_samples].astype(np.float32)
            out = np.zeros(target_samples, dtype=np.float32)
            out[: arr.size] = arr
            return out

        stretch_rate = current_seconds / target_seconds if target_seconds > 0 else 1.0
        try:
            import librosa  # type: ignore
            stretched = librosa.effects.time_stretch(arr.astype(np.float32), rate=max(0.5, min(2.0, stretch_rate)))
        except Exception:
            try:
                stretched = self._fit_waveform_duration_with_ffmpeg(arr, int(sample_rate), stretch_rate)
            except Exception:
                if arr.size > target_samples:
                    return arr[:target_samples].astype(np.float32)
                out = np.zeros(target_samples, dtype=np.float32)
                out[: arr.size] = arr
                return out
        if stretched.size > target_samples:
            return stretched[:target_samples].astype(np.float32)
        if stretched.size < target_samples:
            out = np.zeros(target_samples, dtype=np.float32)
            out[: stretched.size] = stretched
            return out
        return stretched.astype(np.float32)

    def _validate_reconstruction_waveform(self, wav, sample_rate: int, target_seconds: float) -> str | None:
        import numpy as np

        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size == 0 or sample_rate <= 0:
            return "no audio samples were generated"

        current_seconds = arr.size / float(sample_rate)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        target_seconds = max(0.05, float(target_seconds))
        # Qwen can legitimately overshoot the requested duration by a wide margin
        # and still produce usable audio that can be time-fit back to the target.
        # Only reject duration when it is wildly beyond what post-fit handling can
        # reasonably salvage.
        max_allowed_seconds = max(12.0, target_seconds + 8.0, target_seconds * 5.0)

        if current_seconds > max_allowed_seconds:
            return f"generated duration {current_seconds:.2f}s is far above target {target_seconds:.2f}s"
        # Quiet renders can still have a non-trivial sample peak while sounding
        # effectively blank overall. Reject those so tests/previews behave like
        # the real reconstruction path instead of surfacing unusable audio.
        if peak < 0.080 and rms < 0.0030:
            return f"generated waveform energy is too low (peak={peak:.4f}, rms={rms:.4f})"
        return None

    def _synthesize_validated_reconstruction_segment(
        self,
        tts_model,
        prompt,
        text: str,
        *,
        model_name: str,
        target_seconds: float,
        fallback_prompt=None,
        fallback_text: str | None = None,
    ):
        def _generate_and_validate(current_prompt, current_text: str):
            wav, sample_rate = self._synthesize_reconstruction_segment(
                tts_model,
                current_prompt,
                current_text,
                model_name=model_name,
                target_seconds=target_seconds,
            )
            validation_error = self._validate_reconstruction_waveform(wav, int(sample_rate), target_seconds)
            if validation_error:
                return None, int(sample_rate), validation_error

            fitted = self._fit_waveform_duration(wav, int(sample_rate), target_seconds)
            fitted_error = self._validate_reconstruction_waveform(fitted, int(sample_rate), target_seconds)
            if fitted_error:
                return None, int(sample_rate), f"{fitted_error} after duration fitting"

            return fitted, int(sample_rate), None

        fitted_wav, sample_rate, validation_error = _generate_and_validate(prompt, text)
        used_fallback = False

        if validation_error and fallback_prompt is not None:
            fitted_wav, sample_rate, validation_error = _generate_and_validate(
                fallback_prompt,
                str(fallback_text if fallback_text is not None else text),
            )
            used_fallback = True

        if validation_error or fitted_wav is None:
            raise RuntimeError(validation_error or "reconstruction synthesis did not return usable audio")

        return fitted_wav, int(sample_rate), used_fallback

    def _write_audio_clip(self, source_path: Path, output_path: Path, start_time: float, end_time: float) -> None:
        ffmpeg_cmd = self._get_ffmpeg_cmd()
        duration = max(0.05, float(end_time) - float(start_time))
        result = subprocess.run(
            [
                ffmpeg_cmd,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, float(start_time)):.3f}",
                "-i",
                str(source_path),
                "-t",
                f"{duration:.3f}",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "ffmpeg clip extraction failed").strip()
            raise RuntimeError(detail[:1000])

    def _extract_reconstruction_references(
        self,
        source_path: Path,
        segments: list[TranscriptSegment],
        work_dir: Path,
        *,
        video: Video | None = None,
        state: dict | None = None,
        progress_task: str | None = None,
    ) -> dict[int, dict]:
        references: dict[int, dict] = {}
        groups: dict[int, list[TranscriptSegment]] = {}
        for seg in segments:
            speaker_id = getattr(seg, "speaker_id", None)
            if speaker_id is None:
                continue
            if not str(getattr(seg, "text", "") or "").strip():
                continue
            groups.setdefault(int(speaker_id), []).append(seg)

        state = state or {"speakers": {}}
        total_groups = max(1, len(groups))
        for idx, (speaker_id, speaker_segments) in enumerate(groups.items(), start=1):
            if video is not None and progress_task:
                self._set_workbench_task_progress(
                    int(video.id),
                    area="reconstruction",
                    task=progress_task,
                    status="running",
                    stage="references",
                    message=f"Preparing speaker references ({idx}/{total_groups})...",
                    percent=32 + int((idx / total_groups) * 28),
                    current=idx,
                    total=total_groups,
                )
            if video is not None:
                preferred = self._resolve_reconstruction_reference_for_speaker(
                    video,
                    int(speaker_id),
                    speaker_segments,
                    source_path,
                    work_dir,
                    state,
                )
                if preferred is not None:
                    references[int(speaker_id)] = preferred
                    continue
            best = None
            best_score = -1.0
            for seg in speaker_segments:
                duration = max(0.0, float(seg.end_time) - float(seg.start_time))
                if duration < 2.5:
                    continue
                score = self._reference_duration_score(seg.start_time, seg.end_time, seg.text)
                if score > best_score:
                    best_score = score
                    best = seg
            if best is None and speaker_segments:
                best = max(speaker_segments, key=lambda s: self._reference_duration_score(s.start_time, s.end_time, s.text))
            if best is None:
                continue
            clip_path = work_dir / f"speaker_{speaker_id}_ref.wav"
            ref_start = float(best.start_time)
            ref_end = min(float(best.end_time), ref_start + 30.0)
            self._write_audio_clip(source_path, clip_path, ref_start, ref_end)
            references[speaker_id] = {
                "path": clip_path,
                "text": str(best.text or "").strip(),
                "start": ref_start,
                "end": ref_end,
                "source": "auto_reference",
            }
        return references

    def _build_reconstruction_workbench(self, video: Video, segments: list[TranscriptSegment], source_path: Path, *, progress_task: str | None = None) -> dict:
        if progress_task:
            self._set_workbench_task_progress(
                int(video.id),
                area="reconstruction",
                task=progress_task,
                status="running",
                stage="speakers",
                message="Loading speaker roster for the workbench...",
                percent=8,
            )
        with Session(engine) as session:
            speaker_name_map = {
                int(row.id): str(row.name or f"Speaker {row.id}")
                for row in session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all()
            }

        workbench_dir = self._reconstruction_workbench_dir(video)
        grouped: dict[int, list[TranscriptSegment]] = {}
        for seg in segments:
            speaker_id = getattr(seg, "speaker_id", None)
            if speaker_id is None:
                continue
            grouped.setdefault(int(speaker_id), []).append(seg)
        if progress_task:
            self._set_workbench_task_progress(
                int(video.id),
                area="reconstruction",
                task=progress_task,
                status="running",
                stage="state",
                message="Syncing workbench state from diarized segments...",
                percent=18,
                current=len(grouped),
                total=len(grouped),
            )
        state = self._sync_reconstruction_workbench_state(video, grouped)
        references = self._extract_reconstruction_references(
            source_path,
            segments,
            workbench_dir,
            video=video,
            state=state,
            progress_task=progress_task,
        )

        speakers: list[dict] = []
        all_speakers_approved = True
        total_speakers = max(1, len(grouped))
        for idx, (speaker_id, speaker_segments) in enumerate(sorted(grouped.items(), key=lambda item: item[0]), start=1):
            if progress_task:
                self._set_workbench_task_progress(
                    int(video.id),
                    area="reconstruction",
                    task=progress_task,
                    status="running",
                    stage="assemble",
                    message=f"Preparing speaker cards ({idx}/{total_speakers})...",
                    percent=62 + int((idx / total_speakers) * 30),
                    current=idx,
                    total=total_speakers,
                )
            ref = references.get(int(speaker_id))
            ranked_segments = sorted(
                speaker_segments,
                key=lambda s: (-(float(s.end_time) - float(s.start_time)), float(s.start_time)),
            )
            speaker_state = state.get("speakers", {}).get(str(int(speaker_id)), {})
            sample_state = speaker_state.get("samples", {}) if isinstance(speaker_state, dict) else {}
            active_ids = [int(seg_id) for seg_id in speaker_state.get("active_sample_segment_ids", [])] if isinstance(speaker_state, dict) else []
            selected_sample_segment_id = speaker_state.get("selected_sample_segment_id") if isinstance(speaker_state, dict) else None
            approved = bool(speaker_state.get("approved", False)) if isinstance(speaker_state, dict) else False
            all_speakers_approved = all_speakers_approved and approved
            sample_rows = []
            remaining_candidate_exists = False
            for seg in ranked_segments:
                seg_id = int(getattr(seg, "id", 0) or 0)
                if seg_id <= 0:
                    continue
                seg_state = sample_state.get(str(seg_id), {}) if isinstance(sample_state, dict) else {}
                if seg_id not in active_ids:
                    if not bool(seg_state.get("rejected", False)):
                        remaining_candidate_exists = True
                    continue
                clip_path = self._ensure_reconstruction_sample_clip(source_path, workbench_dir, int(speaker_id), seg)
                cleaned_name = str(seg_state.get("cleaned_audio_filename") or "").strip() if isinstance(seg_state, dict) else ""
                sample_rows.append({
                    "segment_id": seg_id,
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "duration": max(0.0, float(seg.end_time) - float(seg.start_time)),
                    "text": str(getattr(seg, "text", "") or "").strip(),
                    "audio_filename": clip_path.name,
                    "cleaned_audio_filename": cleaned_name or None,
                    "rejected": bool(seg_state.get("rejected", False)) if isinstance(seg_state, dict) else False,
                    "selected": seg_id == selected_sample_segment_id,
                })

            test_audio = self._reconstruction_speaker_test_path(workbench_dir, int(speaker_id))
            speakers.append({
                "speaker_id": int(speaker_id),
                "speaker_name": speaker_name_map.get(int(speaker_id), f"Speaker {speaker_id}"),
                "segment_count": len(speaker_segments),
                "approved": approved,
                "selected_sample_segment_id": int(selected_sample_segment_id) if selected_sample_segment_id is not None else None,
                "reference_text": str(ref.get("text") or "").strip() if ref else None,
                "reference_start_time": float(ref.get("start")) if ref and ref.get("start") is not None else None,
                "reference_end_time": float(ref.get("end")) if ref and ref.get("end") is not None else None,
                "reference_audio_filename": ref["path"].name if ref and ref.get("path") else None,
                "samples": sample_rows,
                "latest_test_audio_filename": test_audio.name if test_audio.exists() else None,
                "latest_test_text": str(speaker_state.get("latest_test_text") or "").strip() if isinstance(speaker_state, dict) else None,
                "latest_test_mode": str(speaker_state.get("latest_test_mode") or "").strip() if isinstance(speaker_state, dict) else None,
                "can_add_sample": bool(remaining_candidate_exists),
            })

        if progress_task:
            self._set_workbench_task_progress(
                int(video.id),
                area="reconstruction",
                task=progress_task,
                status="completed",
                stage="complete",
                message="Reconstruction workbench is ready.",
                percent=100,
                current=len(speakers),
                total=len(speakers),
            )
        return {
            "mode": self._get_reconstruction_mode(video),
            "instruction_template": self._get_reconstruction_instruction_template(video),
            "performance_supported": True,
            "speaker_count": len(speakers),
            "all_speakers_approved": bool(speakers) and all_speakers_approved,
            "speakers": speakers,
        }

    def _build_performance_reconstruction_prompt(self, tts_model, clean_prompt, performance_audio_path: Path, ref_text: str):
        prompt_items = tts_model.create_voice_clone_prompt(
            ref_audio=str(performance_audio_path),
            ref_text=str(ref_text or "").strip(),
            x_vector_only_mode=False,
        )
        if not prompt_items:
            raise RuntimeError("Qwen3-TTS did not return a performance prompt item.")
        performance_prompt = prompt_items[0]
        prompt_cls = type(performance_prompt)
        return prompt_cls(
            ref_code=performance_prompt.ref_code,
            ref_spk_embedding=clean_prompt.ref_spk_embedding,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=performance_prompt.ref_text,
        )

    def _build_reconstruction_generation_text(self, text: str, *, performance_mode: bool, instruction_template: str) -> str:
        base_text = str(text or "").strip()
        if not performance_mode:
            return base_text
        instruction = str(instruction_template or "").strip()
        if not instruction:
            instruction = self._reconstruction_default_instruction_template()
        return f"<instruction>{instruction}</instruction>\n{base_text}"

    def _prepare_reconstruction_batch_prompts(
        self,
        tts_model,
        batch_items: list[dict],
        *,
        clean_prompts_by_speaker: dict[int, object],
        performance_source_path: Path | None,
        performance_mode: bool,
        work_dir: Path,
    ) -> tuple[list[object], list[str]]:
        prompts: list[object] = []
        texts: list[str] = []
        for item in batch_items:
            texts.append(str(item["generation_text"]))
            prompts.append(clean_prompts_by_speaker[int(item["speaker_id"])])

        if not performance_mode or performance_source_path is None:
            return prompts, texts

        for idx, item in enumerate(batch_items):
            performance_clip = work_dir / f"segment_{int(item['segment_id'])}_perf.wav"
            try:
                self._write_audio_clip(
                    performance_source_path,
                    performance_clip,
                    float(item["start_time"]),
                    float(item["end_time"]),
                )
                prompts[idx] = self._build_performance_reconstruction_prompt(
                    tts_model,
                    clean_prompts_by_speaker[int(item["speaker_id"])],
                    performance_clip,
                    str(item["text"]),
                )
            except Exception:
                prompts[idx] = clean_prompts_by_speaker[int(item["speaker_id"])]
            finally:
                try:
                    performance_clip.unlink(missing_ok=True)
                except Exception:
                    pass
        return prompts, texts

    def generate_reconstruction_speaker_test(
        self,
        video_id: int,
        *,
        speaker_id: int,
        text: str | None = None,
        segment_id: int | None = None,
        performance_mode: bool = False,
        progress_task: str = "speaker_test",
    ) -> dict:
        import soundfile as sf  # type: ignore

        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="prepare",
            message="Preparing speaker reference and test text...",
            percent=6,
        )

        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Conversation reconstruction is currently available for uploaded manual media only.")
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            if not segments:
                raise ValueError("Transcript segments are required before testing reconstruction voices.")
            source_path = self.get_audio_path(video, purpose="processing")
            performance_source_path = self.get_original_manual_media_path(video) or source_path
            workbench = self._build_reconstruction_workbench(video, segments, source_path)
            state = self._load_reconstruction_workbench_state(video)

        speaker_entry = next((row for row in workbench["speakers"] if int(row["speaker_id"]) == int(speaker_id)), None)
        if not speaker_entry:
            raise ValueError("Speaker reference could not be prepared for this speaker.")

        selected_segment = None
        if segment_id is not None:
            selected_segment = next((seg for seg in segments if int(getattr(seg, "id", 0) or 0) == int(segment_id)), None)
        if selected_segment is None:
            selected_segment = next((seg for seg in segments if int(getattr(seg, "speaker_id", 0) or 0) == int(speaker_id)), None)
        if selected_segment is None:
            raise ValueError("No segment could be found for this speaker.")

        target_text = str(text or getattr(selected_segment, "text", "") or "").strip()
        if not target_text:
            raise ValueError("A non-empty test line is required.")

        workbench_dir = self._reconstruction_workbench_dir(video)
        ref_filename = str(speaker_entry.get("reference_audio_filename") or "").strip()
        if not ref_filename:
            raise ValueError("Speaker reference audio is missing.")
        ref_path = workbench_dir / ref_filename
        if not ref_path.exists():
            raise ValueError("Speaker reference audio file is missing.")

        model_name = (os.getenv("RECONSTRUCTION_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-Base").strip()
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="model",
            message="Loading the reconstruction TTS model into memory. The first test after backend start can take a while.",
            percent=26,
        )
        tts_model, _, reused_cached_model = self._get_reconstruction_tts_model(model_name)
        if reused_cached_model:
            self._set_workbench_task_progress(
                int(video_id),
                area="reconstruction",
                task=progress_task,
                status="running",
                stage="model",
                message="Reusing the loaded reconstruction TTS model...",
                percent=32,
            )

        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="reference_prompt",
            message="Encoding the clean speaker reference...",
            percent=40,
        )
        clean_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=str(ref_path),
            ref_text=None,
            x_vector_only_mode=True,
        )[0]
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="prompt",
            message="Building speaker prompt from the reference audio...",
            percent=52,
        )

        use_performance = bool(performance_mode)
        if use_performance:
            performance_clip = workbench_dir / f"speaker_{int(speaker_id)}.performance.test.ref.wav"
            try:
                self._set_workbench_task_progress(
                    int(video_id),
                    area="reconstruction",
                    task=progress_task,
                    status="running",
                    stage="performance_prompt",
                    message="Encoding the performance reference clip...",
                    percent=60,
                )
                self._write_audio_clip(
                    performance_source_path,
                    performance_clip,
                    float(selected_segment.start_time),
                    float(selected_segment.end_time),
                )
                prompt = self._build_performance_reconstruction_prompt(
                    tts_model,
                    clean_prompt,
                    performance_clip,
                    str(getattr(selected_segment, "text", "") or "").strip(),
                )
            except Exception:
                prompt = clean_prompt
                use_performance = False
            finally:
                try:
                    performance_clip.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            prompt = clean_prompt

        source_segment_seconds = max(0.35, float(selected_segment.end_time) - float(selected_segment.start_time))
        target_seconds = source_segment_seconds if use_performance else max(1.2, min(8.0, source_segment_seconds))
        generation_text = self._build_reconstruction_generation_text(
            target_text,
            performance_mode=use_performance,
            instruction_template=self._get_reconstruction_instruction_template(video),
        )
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="synthesis",
            message="Synthesizing the voice test clip...",
            percent=74,
        )
        wav, sample_rate, used_fallback = self._synthesize_validated_reconstruction_segment(
            tts_model,
            prompt,
            generation_text,
            model_name=model_name,
            target_seconds=target_seconds,
            fallback_prompt=clean_prompt if use_performance else None,
            fallback_text=target_text,
        )
        if used_fallback:
            use_performance = False
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="validate",
            message="Validating the generated test audio...",
            percent=90,
        )
        output_path = workbench_dir / f"speaker_{int(speaker_id)}.test.wav"
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="running",
            stage="write",
            message="Saving the generated test audio to the workbench...",
            percent=96,
        )
        sf.write(str(output_path), wav, int(sample_rate))
        self._release_reconstruction_cuda_cache()
        speaker_state = state.setdefault("speakers", {}).setdefault(str(int(speaker_id)), {})
        speaker_state["latest_test_audio_filename"] = output_path.name
        speaker_state["latest_test_text"] = target_text
        speaker_state["latest_test_mode"] = "performance" if use_performance else "basic"
        self._save_reconstruction_workbench_state(video, state)
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task=progress_task,
            status="completed",
            stage="complete",
            message="Voice test is ready.",
            percent=100,
        )
        return {
            "speaker_id": int(speaker_id),
            "mode": "performance" if use_performance else "basic",
            "segment_id": int(getattr(selected_segment, "id", 0) or 0),
            "text": target_text,
            "audio_filename": output_path.name,
            "detail": (
                "Performance reference used from the original segment."
                if use_performance
                else "Performance prompt was unstable, so the test fell back to the clean reference clip."
                if used_fallback
                else "Speaker timbre test generated from the clean reference clip."
            ),
        }

    def update_reconstruction_sample_state(
        self,
        video_id: int,
        *,
        speaker_id: int,
        segment_id: int,
        rejected: bool | None = None,
        selected: bool | None = None,
        clear_cleaned: bool | None = None,
    ) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            grouped: dict[int, list[TranscriptSegment]] = {}
            for seg in segments:
                seg_speaker = getattr(seg, "speaker_id", None)
                if seg_speaker is not None:
                    grouped.setdefault(int(seg_speaker), []).append(seg)
            if int(speaker_id) not in grouped:
                raise ValueError("Speaker not found in reconstruction workbench.")
            self._sync_reconstruction_workbench_state(video, grouped)
            state = self._load_reconstruction_workbench_state(video)
            speaker_state = state.setdefault("speakers", {}).setdefault(str(int(speaker_id)), {})
            active_ids = [int(seg_id) for seg_id in speaker_state.get("active_sample_segment_ids", [])]
            if int(segment_id) not in active_ids and selected:
                active_ids.append(int(segment_id))
            sample_state = self._reconstruction_sample_state(state, int(speaker_id), int(segment_id))
            if rejected is not None:
                sample_state["rejected"] = bool(rejected)
                if bool(rejected):
                    active_ids = [seg_id for seg_id in active_ids if seg_id != int(segment_id)]
                    if int(speaker_state.get("selected_sample_segment_id") or 0) == int(segment_id):
                        speaker_state["selected_sample_segment_id"] = next((seg_id for seg_id in active_ids if not bool(self._reconstruction_sample_state(state, int(speaker_id), seg_id).get("rejected"))), None)
                        speaker_state["approved"] = False
            if selected:
                sample_state["rejected"] = False
                speaker_state["selected_sample_segment_id"] = int(segment_id)
                if int(segment_id) not in active_ids:
                    active_ids.append(int(segment_id))
                speaker_state["approved"] = False
            if clear_cleaned:
                cleaned_name = str(sample_state.get("cleaned_audio_filename") or "").strip()
                if cleaned_name:
                    cleaned_path = self._reconstruction_workbench_dir(video) / Path(cleaned_name).name
                    try:
                        cleaned_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                sample_state.pop("cleaned_audio_filename", None)
                speaker_state["approved"] = False
            speaker_state["active_sample_segment_ids"] = active_ids[:8]
            self._save_reconstruction_workbench_state(video, state)
            return state

    def add_reconstruction_performance_sample(self, video_id: int, *, speaker_id: int) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            grouped: dict[int, list[TranscriptSegment]] = {}
            for seg in segments:
                seg_speaker = getattr(seg, "speaker_id", None)
                if seg_speaker is not None:
                    grouped.setdefault(int(seg_speaker), []).append(seg)
            speaker_segments = grouped.get(int(speaker_id))
            if not speaker_segments:
                raise ValueError("Speaker not found in reconstruction workbench.")
            state = self._sync_reconstruction_workbench_state(video, grouped)
            speaker_state = state.setdefault("speakers", {}).setdefault(str(int(speaker_id)), {})
            active_ids = [int(seg_id) for seg_id in speaker_state.get("active_sample_segment_ids", [])]
            ranked = sorted(
                speaker_segments,
                key=lambda s: (-(float(s.end_time) - float(s.start_time)), float(s.start_time)),
            )
            for seg in ranked:
                seg_id = int(getattr(seg, "id", 0) or 0)
                if seg_id <= 0 or seg_id in active_ids:
                    continue
                seg_state = self._reconstruction_sample_state(state, int(speaker_id), seg_id)
                if bool(seg_state.get("rejected")):
                    continue
                active_ids.append(seg_id)
                speaker_state["active_sample_segment_ids"] = active_ids[:8]
                self._save_reconstruction_workbench_state(video, state)
                return state
            raise ValueError("No additional performance sample candidates are available for this speaker.")

    def set_reconstruction_speaker_approval(self, video_id: int, *, speaker_id: int, approved: bool) -> dict:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            state = self._load_reconstruction_workbench_state(video)
            speaker_state = state.setdefault("speakers", {}).setdefault(str(int(speaker_id)), {})
            speaker_state["approved"] = bool(approved)
            self._save_reconstruction_workbench_state(video, state)
            return state

    def cleanup_reconstruction_sample(self, video_id: int, *, speaker_id: int, segment_id: int) -> dict:
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task="sample_cleanup",
            status="running",
            stage="prepare",
            message="Preparing the selected performance sample for cleanup...",
            percent=8,
        )
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            if (video.media_source_type or "youtube") != "upload":
                raise ValueError("Sample cleanup is only available for uploaded manual media.")
            source_path = self.get_audio_path(video, purpose="processing")
            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            target_segment = next((seg for seg in segments if int(getattr(seg, "id", 0) or 0) == int(segment_id) and int(getattr(seg, "speaker_id", 0) or 0) == int(speaker_id)), None)
            if target_segment is None:
                raise ValueError("Performance sample not found.")
            work_dir = self._reconstruction_workbench_dir(video)
            input_clip = self._ensure_reconstruction_sample_clip(source_path, work_dir, int(speaker_id), target_segment)
            cleaned_clip = self._reconstruction_sample_cleaned_clip_path(work_dir, int(speaker_id), int(segment_id))
            restored_wav = work_dir / f"speaker_{int(speaker_id)}_sample_{int(segment_id)}.voicefixer.restored.wav"
            blended_wav = work_dir / f"speaker_{int(speaker_id)}_sample_{int(segment_id)}.voicefixer.blended.wav"
            leveled_wav = work_dir / f"speaker_{int(speaker_id)}_sample_{int(segment_id)}.voicefixer.leveled.wav"
            mode = max(0, min(2, int(getattr(video, "voicefixer_mode", 0) or 0)))
            mix_ratio = max(0.0, min(1.0, float(getattr(video, "voicefixer_mix_ratio", 1.0) or 1.0)))
            leveling_mode = str(getattr(video, "voicefixer_leveling_mode", "off") or "off").strip().lower()

        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task="sample_cleanup",
            status="running",
            stage="load_model",
            message="Loading VoiceFixer for sample cleanup...",
            percent=22,
        )
        try:
            from voicefixer import VoiceFixer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"VoiceFixer is not available: {e}")

        try:
            restorer = VoiceFixer()
            use_cuda = str(self.device or "") == "cuda"
            self._set_workbench_task_progress(
                int(video_id),
                area="reconstruction",
                task="sample_cleanup",
                status="running",
                stage="restore",
                message="Restoring the selected sample audio...",
                percent=48,
            )
            restorer.restore(
                input=str(input_clip),
                output=str(restored_wav),
                cuda=use_cuda,
                mode=mode,
            )
            current_path = restored_wav
            if mix_ratio < 0.999:
                restored_weight = max(0.0, min(1.0, mix_ratio))
                original_weight = max(0.0, 1.0 - restored_weight)
                self._set_workbench_task_progress(
                    int(video_id),
                    area="reconstruction",
                    task="sample_cleanup",
                    status="running",
                    stage="blend",
                    message="Blending the cleaned sample with the original audio...",
                    percent=68,
                )
                self._run_external_command(
                    [
                        self._get_ffmpeg_cmd(),
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(restored_wav),
                        "-i",
                        str(input_clip),
                        "-filter_complex",
                        f"amix=inputs=2:weights='{restored_weight:.4f} {original_weight:.4f}':normalize=0",
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        str(blended_wav),
                    ],
                    "Failed to blend cleaned performance sample",
                )
                current_path = blended_wav
            leveling_filter = self._voicefixer_leveling_filter(leveling_mode)
            if leveling_filter:
                self._set_workbench_task_progress(
                    int(video_id),
                    area="reconstruction",
                    task="sample_cleanup",
                    status="running",
                    stage="level",
                    message="Applying voice leveling to the cleaned sample...",
                    percent=82,
                )
                self._run_external_command(
                    [
                        self._get_ffmpeg_cmd(),
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(current_path),
                        "-af",
                        leveling_filter,
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        str(leveled_wav),
                    ],
                    "Failed to level cleaned performance sample",
                )
                current_path = leveled_wav
            self._set_workbench_task_progress(
                int(video_id),
                area="reconstruction",
                task="sample_cleanup",
                status="running",
                stage="save",
                message="Saving the cleaned sample back into the workbench...",
                percent=94,
            )
            shutil.copyfile(current_path, cleaned_clip)

            with Session(engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError("Video not found")
                state = self._load_reconstruction_workbench_state(video)
                sample_state = self._reconstruction_sample_state(state, int(speaker_id), int(segment_id))
                sample_state["cleaned_audio_filename"] = cleaned_clip.name
                sample_state["rejected"] = False
                self._save_reconstruction_workbench_state(video, state)
                self._set_workbench_task_progress(
                    int(video_id),
                    area="reconstruction",
                    task="sample_cleanup",
                    status="completed",
                    stage="complete",
                    message="Sample cleanup finished.",
                    percent=100,
                )
                return state
        finally:
            for temp_path in (restored_wav, blended_wav, leveled_wav):
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def preview_reconstruction_segment(
        self,
        video_id: int,
        *,
        segment_id: int,
        performance_mode: bool | None = None,
    ) -> dict:
        import soundfile as sf  # type: ignore

        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            target_segment = session.get(TranscriptSegment, int(segment_id))
            if not target_segment or int(getattr(target_segment, "video_id", 0) or 0) != int(video_id):
                raise ValueError("Transcript segment not found.")
        speaker_id = getattr(target_segment, "speaker_id", None)
        if speaker_id is None:
            raise ValueError("The selected segment does not have an assigned speaker.")
        resolved_performance = self._get_reconstruction_mode(video) == "performance" if performance_mode is None else bool(performance_mode)
        result = self.generate_reconstruction_speaker_test(
            video_id,
            speaker_id=int(speaker_id),
            text=str(getattr(target_segment, "text", "") or "").strip(),
            segment_id=int(segment_id),
            performance_mode=resolved_performance,
            progress_task="preview_segment",
        )
        work_dir = self._reconstruction_workbench_dir(video)
        source_test_path = work_dir / str(result["audio_filename"])
        preview_path = self._reconstruction_preview_path(work_dir, int(segment_id))
        if source_test_path.exists():
            self._set_workbench_task_progress(
                int(video_id),
                area="reconstruction",
                task="preview_segment",
                status="running",
                stage="finalize",
                message="Finalizing the segment preview audio...",
                percent=97,
            )
            shutil.copyfile(source_test_path, preview_path)
        self._set_workbench_task_progress(
            int(video_id),
            area="reconstruction",
            task="preview_segment",
            status="completed",
            stage="complete",
            message="Segment preview is ready.",
            percent=100,
        )
        return {
            "segment_id": int(segment_id),
            "speaker_id": int(speaker_id),
            "mode": str(result.get("mode") or ("performance" if resolved_performance else "basic")),
            "text": str(result.get("text") or ""),
            "audio_filename": preview_path.name,
            "detail": str(result.get("detail") or ""),
        }

    def _load_original_audio_segment(self, source_path: Path, start_time: float, end_time: float):
        import soundfile as sf  # type: ignore

        temp_path = source_path.parent / f"fallback_{int(start_time * 1000)}_{int(end_time * 1000)}.wav"
        try:
            self._write_audio_clip(source_path, temp_path, start_time, end_time)
            wav, sr = sf.read(str(temp_path), dtype="float32")
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(axis=1)
            return wav, int(sr)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _reconstruction_token_rate_hz(self, model_name: str) -> int:
        name = str(model_name or "").lower()
        if "25hz" in name:
            return 25
        return 12

    def _reconstruction_sampling_enabled(self) -> bool:
        do_sample_env = (os.getenv("RECONSTRUCTION_DO_SAMPLE") or "false").strip().lower()
        return do_sample_env in {"1", "true", "yes", "on"}

    def _reconstruction_batch_size(self, use_cuda: bool) -> int:
        default_size = "6" if use_cuda else "2"
        try:
            return max(1, min(12, int((os.getenv("RECONSTRUCTION_BATCH_SIZE") or default_size).strip() or default_size)))
        except Exception:
            return 6 if use_cuda else 2

    def _reconstruction_batch_token_budget(self, use_cuda: bool) -> int:
        default_budget = "900" if use_cuda else "320"
        try:
            budget = int((os.getenv("RECONSTRUCTION_BATCH_TOKEN_BUDGET") or default_budget).strip() or default_budget)
        except Exception:
            budget = 900 if use_cuda else 320
        budget = max(96, min(4096, budget))
        if not use_cuda:
            return budget
        try:
            import torch  # type: ignore

            if str(self.device or "") == "cuda" and torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = float(free_bytes) / (1024.0 * 1024.0)
                if free_mb < 6000:
                    budget = min(budget, 220)
                elif free_mb < 9000:
                    budget = min(budget, 360)
                elif free_mb < 12000:
                    budget = min(budget, 520)
        except Exception:
            pass
        return budget

    def _reconstruction_batch_duration_budget(self, use_cuda: bool) -> float:
        default_budget = "9.0" if use_cuda else "3.5"
        try:
            budget = float((os.getenv("RECONSTRUCTION_BATCH_DURATION_BUDGET_SECONDS") or default_budget).strip() or default_budget)
        except Exception:
            budget = 9.0 if use_cuda else 3.5
        return max(0.5, min(60.0, budget))

    def _estimate_reconstruction_item_cost(self, model_name: str, target_seconds: float, text: str) -> int:
        base = self._estimate_reconstruction_max_new_tokens(model_name, target_seconds, text)
        words = len([token for token in str(text or "").split() if token])
        return int(base + max(0, words * 2))

    def _release_reconstruction_cuda_cache(self) -> None:
        if str(self.device or "") != "cuda":
            return
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return
            gc.collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _should_reconstruct_with_tts(self, text: str, target_seconds: float) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        words = len([token for token in text.split() if token])
        if target_seconds <= 0.22:
            return False
        if words <= 1 and target_seconds <= 0.45:
            return False
        if words <= 2 and target_seconds <= 0.35:
            return False
        return True

    def _should_force_original_reconstruction_segment(self, text: str, target_seconds: float) -> bool:
        text = str(text or "").strip()
        words = len([token for token in text.split() if token])
        chars = len(text)
        try:
            max_seconds = float((os.getenv("RECONSTRUCTION_FALLBACK_SEGMENT_SECONDS") or "14").strip() or "14")
        except Exception:
            max_seconds = 14.0
        try:
            max_words = int((os.getenv("RECONSTRUCTION_FALLBACK_SEGMENT_WORDS") or "72").strip() or "72")
        except Exception:
            max_words = 72
        try:
            max_chars = int((os.getenv("RECONSTRUCTION_FALLBACK_SEGMENT_CHARS") or "440").strip() or "440")
        except Exception:
            max_chars = 440
        return float(target_seconds) >= max_seconds or words >= max_words or chars >= max_chars

    def _should_isolate_reconstruction_segment(self, text: str, target_seconds: float) -> bool:
        text = str(text or "").strip()
        words = len([token for token in text.split() if token])
        chars = len(text)
        try:
            isolate_seconds = float((os.getenv("RECONSTRUCTION_ISOLATE_SEGMENT_SECONDS") or "7.5").strip() or "7.5")
        except Exception:
            isolate_seconds = 7.5
        try:
            isolate_words = int((os.getenv("RECONSTRUCTION_ISOLATE_SEGMENT_WORDS") or "34").strip() or "34")
        except Exception:
            isolate_words = 34
        try:
            isolate_chars = int((os.getenv("RECONSTRUCTION_ISOLATE_SEGMENT_CHARS") or "220").strip() or "220")
        except Exception:
            isolate_chars = 220
        return float(target_seconds) >= isolate_seconds or words >= isolate_words or chars >= isolate_chars

    def _configure_reconstruction_model_runtime(self, tts_model) -> None:
        sampling_enabled = self._reconstruction_sampling_enabled()

        try:
            defaults = dict(getattr(tts_model, "generate_defaults", {}) or {})
            defaults["do_sample"] = sampling_enabled
            defaults["subtalker_dosample"] = sampling_enabled
            if sampling_enabled:
                defaults.setdefault("top_k", 24)
                defaults.setdefault("top_p", 0.92)
                defaults.setdefault("temperature", 0.7)
                defaults.setdefault("subtalker_top_k", 24)
                defaults.setdefault("subtalker_top_p", 0.92)
                defaults.setdefault("subtalker_temperature", 0.7)
            else:
                defaults["top_k"] = None
                defaults["top_p"] = None
                defaults["temperature"] = None
                defaults["subtalker_top_k"] = None
                defaults["subtalker_top_p"] = None
                defaults["subtalker_temperature"] = None
            tts_model.generate_defaults = defaults
        except Exception:
            pass

        try:
            generation_config = getattr(getattr(tts_model, "model", None), "generation_config", None)
            if generation_config is not None:
                generation_config.do_sample = sampling_enabled
                if not sampling_enabled:
                    for attr in ("top_k", "top_p", "temperature"):
                        if hasattr(generation_config, attr):
                            setattr(generation_config, attr, None)
        except Exception:
            pass

    def _get_reconstruction_tts_model(self, model_name: str):
        import torch  # type: ignore
        from qwen_tts import Qwen3TTSModel  # type: ignore

        self._ensure_device()
        use_cuda = str(self.device or "") == "cuda" and torch.cuda.is_available() and not self._cuda_recovery_pending
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_label = "cuda:0" if use_cuda else "cpu"
        cache_key = (str(model_name or "").strip(), device_label, str(dtype))

        with self._reconstruction_tts_model_guard:
            if self._reconstruction_tts_model is not None and self._reconstruction_tts_model_cache_key == cache_key:
                self._configure_reconstruction_model_runtime(self._reconstruction_tts_model)
                return self._reconstruction_tts_model, use_cuda, True

            self._reconstruction_tts_model = None
            self._reconstruction_tts_model_cache_key = None
            gc.collect()

            kwargs = {
                "device_map": device_label,
                "dtype": dtype,
            }
            if use_cuda:
                kwargs["attn_implementation"] = "flash_attention_2"
            try:
                tts_model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            except Exception:
                kwargs.pop("attn_implementation", None)
                tts_model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            self._configure_reconstruction_model_runtime(tts_model)
            self._reconstruction_tts_model = tts_model
            self._reconstruction_tts_model_cache_key = cache_key
            return tts_model, use_cuda, False

    def _estimate_reconstruction_max_new_tokens(self, model_name: str, target_seconds: float, text: str) -> int:
        rate_hz = self._reconstruction_token_rate_hz(model_name)
        words = len([token for token in str(text or "").split() if token])
        chars = len(str(text or "").strip())
        duration_budget = int(math.ceil(max(0.20, float(target_seconds)) * rate_hz * 2.25))
        text_budget = int(math.ceil(words * 5.0 + max(0, chars - (words * 4)) * 0.15))
        token_budget = max(24, duration_budget + text_budget + 12)
        return max(24, min(512, token_budget))

    def _reconstruction_generation_kwargs(self, model_name: str, target_seconds: float, text: str) -> dict:
        do_sample = self._reconstruction_sampling_enabled()
        kwargs = {
            "max_new_tokens": self._estimate_reconstruction_max_new_tokens(model_name, target_seconds, text),
            "non_streaming_mode": True,
            "do_sample": do_sample,
            "repetition_penalty": 1.02,
        }
        if do_sample:
            kwargs.update({
                "top_k": 24,
                "top_p": 0.92,
                "temperature": 0.7,
                "subtalker_dosample": True,
                "subtalker_top_k": 24,
                "subtalker_top_p": 0.92,
                "subtalker_temperature": 0.7,
            })
        else:
            kwargs.update({
                "subtalker_dosample": False,
            })
        return kwargs

    def _synthesize_reconstruction_batch(self, tts_model, prompts: list, texts: list[str], target_durations: list[float], model_name: str):
        if not texts:
            return [], 44100
        kwargs = self._reconstruction_generation_kwargs(
            model_name,
            max(target_durations or [0.25]),
            " ".join(texts[:4]),
        )
        kwargs["max_new_tokens"] = max(
            self._estimate_reconstruction_max_new_tokens(model_name, duration, text)
            for duration, text in zip(target_durations, texts)
        )
        wavs, sample_rate = tts_model.generate_voice_clone(
            text=texts,
            language=["Auto"] * len(texts),
            voice_clone_prompt=prompts,
            **kwargs,
        )
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio for batch synthesis.")
        return wavs, int(sample_rate)

    def _synthesize_reconstruction_segment(self, tts_model, prompt, text: str, *, model_name: str, target_seconds: float):
        wavs, sample_rate = self._synthesize_reconstruction_batch(
            tts_model,
            [prompt],
            [text],
            [target_seconds],
            model_name,
        )
        return wavs[0], int(sample_rate)

    def _assemble_reconstructed_audio(self, sample_rate: int, total_duration: float, placements: list[dict]):
        import numpy as np

        total_samples = max(1, int(round(max(0.01, total_duration) * sample_rate)))
        canvas = np.zeros(total_samples, dtype=np.float32)
        fade_samples = max(1, int(sample_rate * 0.05))

        for item in placements:
            wav = np.asarray(item["wav"], dtype=np.float32).reshape(-1)
            start_idx = max(0, int(round(float(item["start_time"]) * sample_rate)))
            end_idx = min(total_samples, start_idx + wav.size)
            if end_idx <= start_idx:
                continue
            wav = wav[: end_idx - start_idx]
            overlap = max(0, min(fade_samples, canvas[start_idx:end_idx].size, wav.size))
            if overlap > 0:
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                fade_out = 1.0 - fade_in
                canvas[start_idx:start_idx + overlap] = (
                    canvas[start_idx:start_idx + overlap] * fade_out + wav[:overlap] * fade_in
                )
                canvas[start_idx + overlap:end_idx] = wav[overlap:]
            else:
                canvas[start_idx:end_idx] = wav

        peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
        if peak > 0.98:
            canvas = canvas / peak * 0.97
        return canvas

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
        with Session(engine) as session:
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
        with Session(engine) as session:
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
        with Session(engine) as session:
            jobs_created = self._queue_channel_unprocessed_videos(session, channel_id)
            session.commit()
            return jobs_created

    def sync_monitored_channel(self, channel_id: int) -> dict[str, int | bool]:
        with Session(engine) as session:
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
                with Session(engine) as session:
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

        with Session(engine) as session:
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

    def sanitize_filename(self, name: str) -> str:
        """Sanitize string to be filesystem safe"""
        # Replace invalid characters for Windows/Linux
        import re
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

        with Session(engine) as session:
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

    def _transcript_laughter_candidates(self, segments: list[TranscriptSegment]) -> list[dict]:
        """Detect explicit laughter cues in transcript text."""
        import re

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
        import tempfile
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
            with Session(engine) as session:
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
        import urllib.request
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
        import urllib.request
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
        import urllib.request
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
        import urllib.request
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
        import urllib.request
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
        import re

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
        import re

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

        with Session(engine) as session:
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
            with Session(engine) as session:
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
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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

                with Session(engine) as s:
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
        
    def get_speaker_embedding(self, audio_source, segment):
        # Extract embedding for a specific segment
        # We need to crop the audio or use the inference with cropping
        # Pyannote Inference with window="whole" expects a file path or waveform.
        # But we want a segment. 
        # Efficient way: Load audio once, crop in memory.
        # For simplicity/speed prototype: rely on Inference to handle cropping if passed Excerpt?
        # Inference usually takes (file, segment)
        try:
            # Reuse a preloaded pyannote audio object when provided. Falling back to a path
            # here is very expensive on long episodes because it decodes the full file.
            if isinstance(audio_source, (str, Path)):
                audio_input = self._load_audio_for_pyannote(str(audio_source))
            else:
                audio_input = audio_source
            embedding = self.embedding_inference.crop(audio_input, segment)
            # embedding is (1, dimension) or similar. We want mean if it returns multiple frames?
            # crop usually returns one embedding for the window if window="whole" but we are passing a segment...
            # Actually, pyannote.audio `Inference` with `window="whole"` on a `crop` returns the embedding for that crop.
            return embedding
        except Exception as e:
            log_verbose(f"Embedding error for segment {segment}: {e}")
            return None

    def identify_speaker(self, session: Session, channel_id: int, embedding, threshold: float = 0.5):
        """Find best speaker/profile match by cosine distance using cached normalized vectors."""
        import numpy as np

        if not channel_id or embedding is None:
            return None, None, float("inf")

        try:
            q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except Exception:
            return None, None, float("inf")
        if q.size == 0 or not np.all(np.isfinite(q)):
            return None, None, float("inf")
        q_norm = float(np.linalg.norm(q))
        if not np.isfinite(q_norm) or q_norm <= 1e-12:
            return None, None, float("inf")
        q = q / q_norm

        cache = self._get_speaker_match_cache(session, int(channel_id))
        matrix = cache.get("matrix")
        profile_ids = cache.get("profile_ids")
        speaker_ids = cache.get("speaker_ids")
        dim = int(cache.get("dim") or 0)
        if matrix is None or profile_ids is None or speaker_ids is None or dim <= 0 or int(q.size) != dim:
            return None, None, float("inf")

        # matrix rows are normalized => cosine distance = 1 - dot(row, q)
        sims = matrix @ q
        if sims.size == 0:
            return None, None, float("inf")
        finite_mask = np.isfinite(sims)
        if not np.any(finite_mask):
            return None, None, float("inf")
        if not np.all(finite_mask):
            sims = sims.copy()
            sims[~finite_mask] = -np.inf
        best_idx = int(np.argmax(sims))
        best_dist = float(1.0 - float(sims[best_idx]))
        best_profile_id = int(profile_ids[best_idx])
        best_speaker_id = int(speaker_ids[best_idx])

        if best_dist < threshold:
            return session.get(Speaker, best_speaker_id), session.get(SpeakerEmbedding, best_profile_id), best_dist
        return None, None, best_dist

    def _heartbeat_loop(self):
        """Background thread to update heartbeat while worker is processing"""
        log_verbose(f"Heartbeat thread started. Writing to: {HEARTBEAT_FILE}")
        while True:
            try:
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(str(time.time()))
            except Exception as e:
                log_verbose(f"Heartbeat write error: {e}")
            time.sleep(10)

    def cleanup_orphaned_active_jobs(self) -> int:
        """Requeue jobs left in active states after a crash/restart.

        This runs at app startup before queue workers begin polling. Any orphaned
        active jobs are moved back to `queued` and prioritized to the front so work
        resumes automatically instead of being left failed/orphaned.
        """
        orphan_statuses = ["running", "downloading", "transcribing", "diarizing"]
        requeued = 0
        per_type_front_offsets: dict[str, int] = {}
        with Session(engine) as session:
            jobs = session.exec(select(Job).where(Job.status.in_(orphan_statuses))).all()
            now = datetime.now()
            for job in jobs:
                prev_status = job.status or "running"

                # Move to front of the same queue by assigning a created_at just
                # older than the current oldest queued/paused item of that type.
                oldest_same_type = session.exec(
                    select(Job)
                    .where(
                        Job.job_type == job.job_type,
                        Job.status.in_(["queued", "paused"]),
                        Job.id != job.id,
                    )
                    .order_by(Job.created_at.asc(), Job.id.asc())
                ).first()
                offset = per_type_front_offsets.get(job.job_type, 0) + 1
                per_type_front_offsets[job.job_type] = offset
                front_anchor = oldest_same_type.created_at if oldest_same_type and oldest_same_type.created_at else now

                job.status = "queued"
                job.progress = 0
                job.status_detail = None
                job.started_at = None
                job.completed_at = None
                job.error = None
                job.created_at = front_anchor - timedelta(microseconds=offset)

                payload = self._load_job_payload(job.payload_json)
                payload.update(
                    {
                        "recovered_from_orphan": True,
                        "recovered_from_orphan_at": now.isoformat(),
                        "recovered_from_orphan_prev_status": prev_status,
                    }
                )
                job.payload_json = json.dumps(payload, sort_keys=True)
                session.add(job)
                requeued += 1

                video = session.get(Video, job.video_id)
                if not video:
                    continue
                if video.status in orphan_statuses or job.job_type == "process":
                    video.status = "queued"
                    session.add(video)

            session.commit()

        if requeued:
            log(f"Startup recovery: requeued {requeued} orphaned active job(s) to front of queue.")
        return requeued

    def cleanup_orphaned_channel_syncs(self) -> int:
        """Clear channel sync states left behind by an interrupted backend run.

        Channel refresh/backfill work is not resumable across process restarts.
        Any channel still marked `refreshing` when the backend starts again is
        therefore stale UI state, not an active sync.
        """
        cleaned = 0
        with Session(engine) as session:
            channels = session.exec(select(Channel).where(Channel.status == "refreshing")).all()
            for channel in channels:
                channel.status = "active"
                channel.sync_status_detail = "Previous sync was interrupted. Refresh to resume metadata backfill."
                channel.sync_progress = 0
                channel.sync_total_items = 0
                channel.sync_completed_items = 0
                session.add(channel)
                cleaned += 1
            session.commit()

        if cleaned:
            log(f"Startup recovery: cleared {cleaned} orphaned channel sync state(s).")
        return cleaned

    def cleanup_orphaned_active_videos(self) -> int:
        """Restore videos left in active-looking states without any active job."""
        active_video_statuses = ["queued", "downloading", "transcribing", "diarizing"]
        active_job_statuses = ["queued", "running", "downloading", "transcribing", "diarizing", "waiting_diarize"]
        cleaned = 0
        with Session(engine) as session:
            videos = session.exec(
                select(Video).where(Video.status.in_(active_video_statuses))
            ).all()
            for video in videos:
                has_active_job = session.exec(
                    select(Job.id)
                    .where(
                        Job.video_id == video.id,
                        Job.status.in_(active_job_statuses),
                    )
                    .limit(1)
                ).first() is not None
                if has_active_job:
                    continue

                recovered_status = self._infer_recoverable_video_status(session, video)
                if recovered_status == str(video.status or ""):
                    continue

                prev_status = str(video.status or "")
                video.status = recovered_status
                session.add(video)
                cleaned += 1
                log(
                    f"Startup recovery: restored video {video.id} "
                    f"from {prev_status or 'unknown'} to {recovered_status}."
                )
            session.commit()

        if cleaned:
            log(f"Startup recovery: restored {cleaned} orphaned active video state(s).")
        return cleaned

    def process_queue(self):
        """Process transcription-stage pipeline jobs."""
        log("Starting process queue worker...")

        # Crash recovery is handled once at app startup by cleanup_orphaned_active_jobs().
        # Avoid resetting running jobs here because other queue workers may already be active.

        # Start heartbeat thread
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        while True:
            try:
                if self._cuda_recovery_pending:
                    recovered = self._recover_cuda_after_fault_if_needed()

                    # Circuit breaker: if the same CUDA fault keeps recurring after
                    # each successful recovery, stop wasting ~5 min per job on
                    # futile Parakeet model loads and stick with Whisper.
                    max_consecutive = int(os.getenv("PARAKEET_MAX_CONSECUTIVE_CUDA_FAULTS", "3"))
                    if (
                        recovered
                        and self._cuda_consecutive_fault_count >= max_consecutive
                    ):
                        self._cuda_unhealthy_reason = (
                            f"Parakeet permanently disabled after "
                            f"{self._cuda_consecutive_fault_count} consecutive CUDA faults. "
                            f"Restart backend to retry."
                        )
                        self._cuda_recovery_pending = False
                        if self._can_auto_restart():
                            log(f"Circuit breaker tripped after {self._cuda_consecutive_fault_count} "
                                f"consecutive CUDA faults — triggering automatic restart.")
                            self._trigger_auto_restart(
                                f"Circuit breaker: {self._cuda_consecutive_fault_count} consecutive CUDA faults"
                            )
                        else:
                            log(
                                f"Parakeet permanently disabled for this worker after "
                                f"{self._cuda_consecutive_fault_count} consecutive CUDA faults. "
                                f"Restart the backend to retry Parakeet."
                            )

                execution_mode = self.get_pipeline_execution_mode()
                focus_mode = self.get_pipeline_focus_mode()

                if self._has_jobs_of_types(DIARIZE_JOB_TYPES, {"running", "diarizing"}):
                    time.sleep(1)
                    continue

                if execution_mode == "sequential" and self._has_jobs_of_types(DIARIZE_JOB_TYPES, {"queued"}):
                    time.sleep(1)
                    continue

                if execution_mode == "staged":
                    # Auto-switch to diarize if threshold is met
                    auto_threshold = int(os.getenv("DIARIZE_AUTO_START_THRESHOLD", "0"))
                    if auto_threshold > 0 and focus_mode == "transcribe":
                        with Session(engine) as session:
                            diarize_count = session.exec(
                                select(func.count(Job.id)).where(
                                    Job.job_type.in_(DIARIZE_JOB_TYPES),
                                    Job.status.in_(["queued", "running", "diarizing"])
                                )
                            ).one() or 0
                        if diarize_count >= auto_threshold:
                            log(f"Auto-switching pipeline focus to diarize (queue {diarize_count} >= threshold {auto_threshold})")
                            self.set_pipeline_focus_mode("diarize")
                            focus_mode = "diarize"

                    if (
                        focus_mode == "diarize"
                        and self._has_jobs_of_types(DIARIZE_JOB_TYPES, {"queued", "running", "diarizing"})
                    ):
                        time.sleep(1)
                        continue

                claimed = self._claim_next_queued_job(PROCESS_JOB_TYPES | VOICEFIXER_JOB_TYPES | RECONSTRUCTION_JOB_TYPES | TRANSCRIPT_REPAIR_JOB_TYPES)
                if not claimed:
                    time.sleep(2)
                    continue
                job_id = claimed["id"]
                video_id = claimed["video_id"]
                job_type = str(claimed.get("job_type") or "")
                payload = self._load_job_payload(claimed.get("payload_json"))
                log_verbose(f"Processing transcription-stage job {job_id} for video {video_id}")
                baseline_cuda_free_b = 0

                # 2. Process
                try:
                    if job_type == "voicefixer_cleanup":
                        self._handle_voicefixer_job(job_id, video_id, payload)
                        self._mark_job_success(job_id)
                        continue
                    if job_type == "conversation_reconstruct":
                        self._handle_reconstruction_job(job_id, video_id, payload)
                        self._mark_job_success(job_id)
                        continue
                    if job_type == "transcript_repair":
                        self._handle_transcript_repair_job(job_id, video_id, payload)
                        self._mark_job_success(job_id)
                        continue

                    self._ensure_device()
                    if self.device == "cuda":
                        baseline_cuda_free_b = int(self._cuda_memory_snapshot().get("free") or 0)
                    video_detached, audio_path = self._process_download_phase(video_id, job_id)
                    segments, _ = self._process_transcribe_phase(video_detached, audio_path, job_id)
                    if execution_mode == "sequential":
                        self._process_diarize_phase(video_detached, audio_path, segments, job_id)
                        self._record_transcript_optimization_completion(job_id, video_id, payload)
                        self._mark_job_success(job_id)
                        with Session(engine) as session:
                            job = session.get(Job, job_id)
                            if job:
                                self._cleanup_redo_backup_for_job(job.payload_json)
                        log(f"Pipeline complete for {video_detached.title}.")
                        self._trigger_semantic_index(video_id)
                    else:
                        diarize_job_id = self._queue_diarize_followup(video_id, job_id)
                        self._mark_process_job_waiting_for_diarize(job_id, diarize_job_id)
                        log(f"Transcription complete for {video_detached.title}; queued diarization job {diarize_job_id}.")
                    # Job succeeded without a CUDA fault — reset the consecutive
                    # fault counter so the circuit breaker doesn't carry over
                    # from a previous (now-recovered) fault sequence.
                    self._cuda_consecutive_fault_count = 0

                except JobPausedException:
                    log(f"Job {job_id} paused by user")
                    # Job status already set to "paused" by the API endpoint;
                    # video status reset to "pending" by the pipeline phase handlers.

                except JobNoticeException as e:
                    log(f"Notice processing job {job_id}: {e.notice_message}")
                    self._mark_job_notice(
                        job_id,
                        video_id,
                        code=e.notice_code,
                        message=e.notice_message,
                        technical_detail=e.technical_detail,
                        video_status=e.video_status,
                    )
                    # If this was a destructive redo-diarization run, restore backup transcript rows.
                    try:
                        with Session(engine) as session:
                            job = session.get(Job, job_id)
                            if job:
                                self._restore_redo_backup_if_needed(session, job, reason="job notice")
                    except Exception as re:
                        log(f"Redo-backup restore failed after job notice {job_id}: {re}")

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    log(f"Error processing job {job_id}: {e}")
                    log(tb)

                    # 4. Mark failure
                    with Session(engine) as session:
                        job = session.get(Job, job_id)
                        if job:
                            job.status = "failed"
                            # Persist traceback fragment to make UI/API failures diagnosable.
                            job.error = f"{e}\n{tb[-3500:]}"
                            session.add(job)
                            session.commit()
                    self._recover_inactive_video_status(video_id)
                    # If this was a destructive redo-diarization run, restore backup transcript rows.
                    try:
                        with Session(engine) as session:
                            job = session.get(Job, job_id)
                            if job:
                                self._restore_redo_backup_if_needed(session, job, reason="job failure")
                    except Exception as re:
                        log(f"Redo-backup restore failed after job error {job_id}: {re}")
                finally:
                    if self.device == "cuda":
                        self._clear_cuda_cache()
                        if execution_mode == "sequential":
                            self._maybe_recover_cuda_headroom(baseline_cuda_free_b, job_id=job_id)
                    gc.collect()
            except Exception as e:
                log(f"Queue worker loop error: {e}")
                import traceback
                log(traceback.format_exc())
                log_verbose("Retrying in 5 seconds...")
                time.sleep(5)
                continue

    def process_diarize_queue(self):
        """Process diarization-stage pipeline jobs after transcription jobs drain."""
        log("Starting diarize queue worker...")
        while True:
            try:
                if self._cuda_recovery_pending:
                    self._recover_cuda_after_fault_if_needed()

                execution_mode = self.get_pipeline_execution_mode()
                focus_mode = self.get_pipeline_focus_mode()
                if execution_mode == "sequential" and not self._has_jobs_of_types(DIARIZE_JOB_TYPES, {"queued", "running", "diarizing"}):
                    time.sleep(2)
                    continue
                if execution_mode != "sequential" and (
                    focus_mode != "diarize"
                    and self._has_jobs_of_types(PROCESS_JOB_TYPES, {"queued", "running", "downloading", "transcribing"})
                ):
                    time.sleep(1)
                    continue

                claimed = self._claim_next_queued_job(DIARIZE_JOB_TYPES)
                if not claimed:
                    # Auto-switch back to transcribe if queue is empty and threshold is enabled
                    if focus_mode == "diarize":
                        auto_threshold = int(os.getenv("DIARIZE_AUTO_START_THRESHOLD", "0"))
                        if auto_threshold > 0:
                            if self._has_jobs_of_types(PROCESS_JOB_TYPES, {"queued", "running", "downloading", "transcribing"}):
                                log("Auto-switching pipeline focus back to transcribe (diarization queue empty)")
                                self.set_pipeline_focus_mode("transcribe")
                    time.sleep(2)
                    continue

                job_id = claimed["id"]
                video_id = claimed["video_id"]
                payload = self._load_job_payload(claimed.get("payload_json"))
                parent_job_id = int(payload.get("parent_job_id") or 0)
                self._ensure_device()
                baseline_cuda_free_b = 0
                if self.device == "cuda":
                    baseline_cuda_free_b = int(self._cuda_memory_snapshot().get("free") or 0)

                try:
                    video = self._get_detached_video(video_id)
                    audio_path = self._ensure_audio_ready_for_video(video, job_id=None)
                    segments, _, _ = self._load_raw_transcript_checkpoint(video, audio_path, job_id=job_id)
                    self._process_diarize_phase(video, audio_path, segments, job_id)
                    self._record_transcript_optimization_completion(job_id, video_id, payload)
                    if parent_job_id:
                        self._finalize_process_job_from_child(parent_job_id, job_id, "completed")
                        with Session(engine) as session:
                            parent = session.get(Job, parent_job_id)
                            if parent:
                                self._cleanup_redo_backup_for_job(parent.payload_json)
                    self._mark_job_success(job_id)
                    log(f"Diarization complete for {video.title}.")
                    self._trigger_semantic_index(video_id)
                except JobPausedException:
                    log(f"Diarize job {job_id} paused by user")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    log(f"Error diarizing job {job_id}: {e}")
                    log(tb)
                    self._mark_job_failure(job_id, f"{e}\n{tb[-3500:]}")
                    if parent_job_id:
                        self._finalize_process_job_from_child(parent_job_id, job_id, "failed", error=f"{e}\n{tb[-3500:]}")
                    self._recover_inactive_video_status(video_id)
                finally:
                    if self.device == "cuda":
                        self._clear_cuda_cache()
                    gc.collect()
                    if self.device == "cuda":
                        self._maybe_recover_cuda_headroom(baseline_cuda_free_b, job_id=job_id)
            except Exception as e:
                log(f"diarize worker loop error: {e}")
                if is_verbose():
                    import traceback
                    traceback.print_exc()
                time.sleep(3)

    def process_funny_queue(self):
        """Process funny-moment analysis/explanation jobs."""
        log("Starting funny queue worker...")
        sleep_empty = 2.0
        while True:
            try:
                claimed = self._claim_next_queued_job({"funny_detect"})
                if not claimed:
                    allowed, reason, free_gb, min_free_gb = self._get_local_ollama_vram_guard()
                    if not allowed and self._has_jobs_of_types({"funny_explain"}, {"queued"}):
                        if reason == "pipeline_gpu_work_active":
                            detail = "Waiting for pipeline GPU work to finish before local Ollama funny explanation."
                        elif free_gb is not None and min_free_gb is not None:
                            detail = (
                                "Waiting for VRAM headroom before local Ollama funny explanation "
                                f"({free_gb:.1f}/{min_free_gb:.1f} GB free)."
                            )
                        else:
                            detail = "Waiting for local Ollama funny explanation resources."
                        self._set_oldest_queued_job_status_detail("funny_explain", detail)
                        time.sleep(sleep_empty)
                        continue
                    claimed = self._claim_next_queued_job({"funny_explain"})

                if not claimed:
                    time.sleep(sleep_empty)
                    continue

                job_id = claimed["id"]
                video_id = claimed["video_id"]
                job_type = claimed["job_type"]
                payload = self._load_job_payload(claimed.get("payload_json"))
                try:
                    self._handle_funny_job(job_id, video_id, job_type, payload)
                    self._mark_job_success(job_id)
                except JobPausedException:
                    log(f"Job {job_id} paused by user")
                except Exception as e:
                    log(f"funny job {job_id} failed: {e}")
                    if is_verbose():
                        import traceback
                        traceback.print_exc()
                    self._mark_job_failure(job_id, str(e))
            except Exception as e:
                log(f"funny worker loop error: {e}")
                if is_verbose():
                    import traceback
                    traceback.print_exc()
                time.sleep(3)

    def process_youtube_queue(self):
        """Process YouTube summary/chapter generation jobs."""
        self._run_worker_loop("youtube", YOUTUBE_JOB_TYPES, self._handle_youtube_job)

    def process_clip_queue(self):
        """Process clip rendering/export jobs."""
        self._run_worker_loop("clip", CLIP_JOB_TYPES, self._handle_clip_job)

    def prefetch_queue_audio(self):
        """Best-effort background prefetch of audio for queued jobs.

        This runs alongside the main processing worker so queued items can finish
        the network download stage before they reach the front of the queue.
        """
        log("Starting audio prefetch worker...")
        while True:
            try:
                candidate = None

                with Session(engine) as session:
                    queued_jobs = session.exec(
                        select(Job)
                        .where(Job.status == "queued", Job.job_type == "process")
                        .order_by(Job.created_at)
                        .limit(30)
                    ).all()

                    for job in queued_jobs:
                        video = session.get(Video, job.video_id)
                        if not video or video.muted or video.access_restricted:
                            continue

                        # Ensure relation is loaded before using path generation after detach.
                        if video.channel:
                            _ = video.channel.name

                        if self._is_prefetch_backoff_active(video.id):
                            continue

                        try:
                            audio_path = self.get_audio_path(video)
                        except Exception:
                            continue

                        if audio_path.exists():
                            continue

                        session.expunge(video)
                        if video.channel:
                            session.expunge(video.channel)
                        candidate = (job.id, video)
                        break

                if not candidate:
                    time.sleep(4)
                    continue

                queued_job_id, video = candidate
                lock = self._get_video_download_lock(video.id)
                if not lock.acquire(blocking=False):
                    # Another thread (likely the main worker) is already downloading this video.
                    time.sleep(1)
                    continue

                try:
                    # Re-check under lock in case another worker finished the file first.
                    existing = self.get_audio_path(video)
                    if existing.exists():
                        time.sleep(0.25)
                        continue

                    log(f"Prefetching audio for queued job {queued_job_id} (video {video.id})...")
                    downloaded = self.download_audio(video, job_id=None)
                    self._validate_and_retry_audio(video, downloaded, job_id=None)
                    # Mark video as downloaded so the UI can show that this queued item
                    # is pre-fetched and ready for transcription when it reaches the front.
                    with Session(engine) as session:
                        v = session.get(Video, video.id)
                        if v and v.status in ["pending", "failed", "downloaded"] and not v.processed:
                            v.status = "downloaded"
                            session.add(v)
                            session.commit()
                    self._clear_prefetch_backoff(video.id)
                    log_verbose(f"Prefetch complete for video {video.id}")
                except Exception as e:
                    notice = e if isinstance(e, JobNoticeException) else None
                    if notice is None:
                        classified = self._classify_ytdlp_download_notice(e)
                        if classified:
                            notice = JobNoticeException(
                                str(classified.get("message") or "This video could not be downloaded."),
                                code=str(classified.get("code") or "notice"),
                                video_status=str(classified.get("video_status") or "pending"),
                                technical_detail=str(e),
                            )

                    if notice is not None:
                        self._set_prefetch_backoff(video.id, 600)
                        log(f"Audio prefetch notice for queued video {video.id}: {notice.notice_message}")
                    else:
                        self._set_prefetch_backoff(video.id, 60)
                        log(f"Audio prefetch failed for queued video {video.id}: {e}")
                finally:
                    lock.release()

                # Small delay so the prefetch worker does not starve DB polling.
                time.sleep(0.5)

            except Exception as e:
                log(f"Audio prefetch worker error: {e}")
                if is_verbose():
                    import traceback
                    traceback.print_exc()
                time.sleep(5)

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
            with Session(engine) as session:
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
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
            with Session(engine) as session:
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
                with Session(engine) as session:
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
            with Session(engine) as session:
                v = session.get(Video, video_id)
                if v:
                    v.status = "pending"
                    session.add(v)
                    session.commit()
            raise
        except JobNoticeException as e:
            log(f"Notice processing video {video_id}: {e.notice_message}")
            with Session(engine) as session:
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
            with Session(engine) as session:
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
            with Session(engine) as session:
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
        with Session(engine) as session:
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
                    ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
            ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
                ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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

            with Session(engine) as session:
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
        with Session(engine) as session:
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
        with Session(engine) as session:
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
            log(f"Deleted corrupt audio file. Retrying with fallback format...")
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
            ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
                    f"Both audio formats are corrupt. This video may still be processing on YouTube. "
                    f"Try again later."
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

    def _process_download_phase(self, video_id: int, job_id: int = None):
        """Phase 1: Update status and download audio.
        Returns (video_obj, audio_path). video_obj is detached from session."""
        
        # Short-lived session for status update
        with Session(engine) as session:
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
        with Session(engine) as session:
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
                with Session(engine) as session:
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
                        with Session(engine) as session:
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
                if "CUBLAS" in str(e).upper():
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
        with Session(engine) as session:
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
        with Session(engine) as session:
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
                with Session(engine) as session:
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
        with Session(engine) as session:
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
            orphan_max_gap_seconds = max(0.0, float(os.getenv("DIARIZATION_ORPHAN_MAX_GAP_SECONDS", "0.35")))

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
        
        with Session(engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError("Video not found")
            speaker_id = None # Passed how? The caller will use this to update speaker.
            # Wait, this method just returns the path. The caller handles DB.
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
        ffmpeg_bin = Path(__file__).parent.parent.parent / "bin"
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
            log_verbose(f"Running ffmpeg frame extract...")
            
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
            
        except subprocess.CalledProcessError as e:
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
        except Exception as e:
            raise
