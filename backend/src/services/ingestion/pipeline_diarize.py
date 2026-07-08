"""Diarization phase of the processing pipeline.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import json
import pickle
import gc
from pathlib import Path
from sqlmodel import Session, select, func
from datetime import datetime

from ...db.database import Video, Speaker, SpeakerEmbedding, TranscriptSegment, TranscriptSegmentRevision, Job, FunnyMoment
from ..logger import log, log_verbose
from . import runtime
from .runtime import (
    DIARIZE_JOB_TYPES,
    PROCESS_JOB_TYPES,
)


class DiarizePhaseMixin:
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
