import os
import threading
from typing import Literal

# NOTE: Heavy ML libraries (torch, faster_whisper, pyannote, numpy, scipy)
# are imported lazily inside _load_models() and related methods to avoid
# blocking the process at startup. Only download/queue operations run
# without them.

from ..logger import log, log_verbose
from .runtime import (
    BACKEND_DIR,
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

from .pipeline import PipelineMixin
from .pipeline_transcribe import TranscribePhaseMixin
from .pipeline_diarize import DiarizePhaseMixin

class IngestionService(
    DiarizePhaseMixin,
    TranscribePhaseMixin,
    PipelineMixin,
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

































































































































































    # ── CUDA auto-restart state management ───────────────────────────────
























































































































































































































        











































