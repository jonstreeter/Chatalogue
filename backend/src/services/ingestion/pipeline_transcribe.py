"""Transcription phase of the processing pipeline.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import json
from pathlib import Path
from sqlmodel import Session

from ...db.database import Video, Job
from ..logger import log, log_verbose
from . import runtime


class TranscribePhaseMixin:
    def _process_transcribe_phase(self, video: Video, audio_path: Path, job_id: int = None, force_non_batched: bool = False):
        """Phase 2: Transcribe audio. Checks for existing transcript to skip."""
        
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
