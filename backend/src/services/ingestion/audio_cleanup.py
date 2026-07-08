"""VoiceFixer cleanup runs and the audio cleanup workbench.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import time
import json
import re
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from sqlmodel import Session

from ...db.database import Video
from .exceptions import (
    JobCancelledException,
    JobPausedException,
)
from . import runtime
from .runtime import (
    MANUAL_MEDIA_DIR,
)


class AudioCleanupMixin:
    def _run_voicefixer_cleanup(self, video_id: int, *, job_id: int | None = None, force: bool = False) -> Path:
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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

        with Session(runtime.engine) as session:
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
            pass  # type: ignore
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

        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
