"""Transcription engines: Whisper backends, Parakeet, language routing, engine selection, model loading.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import subprocess
import gc
import sys
import re
import math
from pathlib import Path
from sqlmodel import Session, select, func
from datetime import datetime

from ...db.database import Video, Job
from ..logger import log, log_verbose
from . import runtime
from .runtime import (
    TEMP_DIR,
    _env_float,
    temporary_disabled_blackhole_proxies,
)


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



class TranscriptionEngineMixin:
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

    def _ensure_ctranslate2_pkg_resources(self):
        """Install a pkg_resources shim when setuptools is absent or broken.

        ctranslate2, pyannote.audio, and NeMo all import pkg_resources on Windows.
        This is called in __init__ so the shim is in sys.modules before any lazy
        ML import fires â€” regardless of which transcription engine or code path runs.
        """
        import importlib
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
        total_duration = float(self._probe_audio_duration_seconds(Path(audio_path)) or 0.0)

        def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
            try:
                value = int((os.getenv(name) or str(default)).strip() or default)
            except Exception:
                value = default
            return max(min_value, min(value, max_value))

        def _resolve_initial_batch_size() -> int:
            if os.getenv("TRANSCRIPTION_BATCHED", "true").strip().lower() != "true":
                return 1
            requested = _env_int("TRANSFORMERS_WHISPER_BATCH_SIZE", 4, 1, 16)
            if self.device != "cuda":
                return requested
            snap = self._cuda_memory_snapshot()
            free_b = int(snap.get("free") or 0)
            total_b = int(snap.get("total") or self._gpu_total_vram_bytes or 0)
            free_gb = float(free_b) / (1024 ** 3) if free_b > 0 else 0.0
            free_ratio = (float(free_b) / float(total_b)) if total_b > 0 and free_b > 0 else 0.0
            cap = 8
            if free_gb < 8.0 or free_ratio < 0.25:
                cap = 1
            elif free_gb < 12.0 or free_ratio < 0.35:
                cap = 2
            elif free_gb < 18.0 or free_ratio < 0.50:
                cap = 4
            return max(1, min(requested, cap))

        initial_batch_size = _resolve_initial_batch_size()
        initial_chunk_length = _env_int("TRANSFORMERS_WHISPER_CHUNK_LENGTH_S", 30, 5, 30)

        generate_kwargs = {"task": "transcribe", "num_beams": beam_size}
        language_hint = self._resolve_whisper_language_hint(kwargs.get("language"))
        if language_hint:
            generate_kwargs["language"] = language_hint
        attempts = []
        seen = set()
        batch_candidates = [initial_batch_size]
        if initial_batch_size > 4:
            batch_candidates.append(4)
        if initial_batch_size > 2:
            batch_candidates.append(2)
        if initial_batch_size > 1:
            batch_candidates.append(1)
        chunk_candidates = [initial_chunk_length]
        if initial_chunk_length > 20:
            chunk_candidates.append(20)
        if initial_chunk_length > 15:
            chunk_candidates.append(15)
        if initial_chunk_length > 10:
            chunk_candidates.append(10)
        for batch_candidate in batch_candidates:
            for chunk_candidate in chunk_candidates:
                key = (int(batch_candidate), int(chunk_candidate))
                if key not in seen:
                    seen.add(key)
                    attempts.append(key)

        result = None
        last_oom = None
        for attempt_index, (batch_size, chunk_length_s) in enumerate(attempts, start=1):
            if self.device == "cuda":
                self._log_cuda_memory(f"pre_transformers_whisper_attempt_{attempt_index}", job_id=kwargs.get("job_id"))
            try:
                log_verbose(
                    "Transformers Whisper transcription attempt "
                    f"{attempt_index}/{len(attempts)} "
                    f"(batch_size={batch_size}, chunk_length_s={chunk_length_s}, beams={beam_size})"
                )
                result = pipeline_runner(
                    str(audio_path),
                    chunk_length_s=chunk_length_s,
                    batch_size=batch_size,
                    return_timestamps="word" if word_timestamps else True,
                    generate_kwargs=generate_kwargs,
                )
                if self.device == "cuda":
                    self._log_cuda_memory(f"post_transformers_whisper_attempt_{attempt_index}", job_id=kwargs.get("job_id"))
                if attempt_index > 1:
                    self._upsert_job_payload_fields(
                        kwargs.get("job_id"),
                        {
                            "transformers_whisper_oom_recovered": True,
                            "transformers_whisper_recovery_attempt": int(attempt_index),
                            "transformers_whisper_batch_size_effective": int(batch_size),
                            "transformers_whisper_chunk_length_s_effective": int(chunk_length_s),
                        },
                    )
                break
            except Exception as e:
                if not self._is_cuda_oom(e):
                    raise
                last_oom = e
                self._upsert_job_payload_fields(
                    kwargs.get("job_id"),
                    {
                        "transformers_whisper_cuda_oom": True,
                        "transformers_whisper_oom_attempt": int(attempt_index),
                        "transformers_whisper_oom_batch_size": int(batch_size),
                        "transformers_whisper_oom_chunk_length_s": int(chunk_length_s),
                        "transformers_whisper_oom_error": str(e)[:800],
                    },
                )
                log(
                    "Transformers Whisper CUDA OOM "
                    f"(attempt {attempt_index}/{len(attempts)}, batch_size={batch_size}, "
                    f"chunk_length_s={chunk_length_s}). Retrying with safer settings."
                )
                self._clear_cuda_cache()

        if result is None:
            if last_oom is not None:
                raise RuntimeError(
                    "Transformers Whisper failed due to CUDA OOM after retrying reduced "
                    "batch/chunk settings. Falling back to a safer Whisper backend or CPU is required."
                ) from last_oom
            raise RuntimeError("Transformers Whisper returned no transcription result.")

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
                with Session(runtime.engine) as session:
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
            with Session(runtime.engine) as session:
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
