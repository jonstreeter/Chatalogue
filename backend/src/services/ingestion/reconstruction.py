"""Conversation reconstruction: TTS synthesis, workbench, assembly.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import json
import subprocess
import gc
import math
import shutil
import tempfile
from pathlib import Path
from sqlmodel import Session, select

from ...db.database import Video, Speaker, TranscriptSegment
from .exceptions import (
    JobCancelledException,
    JobPausedException,
)
from . import runtime
from .runtime import (
    MANUAL_MEDIA_DIR,
)


class ReconstructionMixin:
    def _run_conversation_reconstruction(self, video_id: int, *, job_id: int | None = None, force: bool = False) -> Path:
        with Session(runtime.engine) as session:
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

    def _set_reconstruction_state(
        self,
        video_id: int,
        *,
        status: str | None = None,
        audio_path: str | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> None:
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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

        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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

            with Session(runtime.engine) as session:
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

        with Session(runtime.engine) as session:
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
