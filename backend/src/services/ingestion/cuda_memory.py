"""CUDA/GPU health, memory accounting, model residency and release decisions.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import json
import subprocess
import threading
import gc
import sys
from datetime import datetime

from ..logger import log, log_verbose
from .runtime import (
    BACKEND_DIR,
    CUDA_MAX_AUTO_RESTARTS,
    CUDA_RESTART_STATE_FILE,
    CUDA_RESTART_WINDOW_SECONDS,
    RUNTIME_DIR,
    _env_float,
)


class CudaMemoryMixin:
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
        main_py = BACKEND_DIR / "src" / "main.py"
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
