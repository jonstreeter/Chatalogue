"""Job queue lifecycle: claim/progress/stage bookkeeping, worker loops, dispatch handlers, orphan cleanup.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import json
import threading
import gc
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from sqlmodel import Session, select, func
from datetime import datetime, timedelta

from ...db.database import Video, Channel, TranscriptSegment, Job, FunnyMoment
from ..logger import log, log_verbose, is_verbose
from .. import episode_clone as clone_svc
from .exceptions import (
    JobCancelledException,
    JobNoticeException,
    JobPausedException,
)
from . import runtime
from .runtime import (
    CLIP_JOB_TYPES,
    DIARIZE_JOB_TYPES,
    HEARTBEAT_FILE,
    PROCESS_JOB_TYPES,
    RECONSTRUCTION_JOB_TYPES,
    TEMP_DIR,
    TRANSCRIPT_REPAIR_JOB_TYPES,
    VOICEFIXER_JOB_TYPES,
    YOUTUBE_JOB_TYPES,
    _truncate_error,
)


class JobLifecycleMixin:
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
                with Session(runtime.engine) as session:
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
                with Session(runtime.engine) as session:
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
                with Session(runtime.engine) as session:
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
            with Session(runtime.engine) as session:
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
        if stage_key == "diarize":
            transcribe_started_at = payload.get("stage_transcribe_started_at") or payload.get("stage_transcribing_started_at")
            transcribe_completed_at = payload.get("stage_transcribe_completed_at")
            if transcribe_started_at and not transcribe_completed_at:
                fields["stage_transcribe_completed_at"] = now_iso
                try:
                    start_dt = datetime.fromisoformat(str(transcribe_started_at))
                    now_dt = datetime.fromisoformat(now_iso)
                    fields["stage_transcribe_seconds"] = max(0.0, (now_dt - start_dt).total_seconds())
                except Exception:
                    pass
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

        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
            from ..semantic_search import start_index_job
            start_index_job([video_id])
            log(f"[semantic] Queued semantic indexing for video {video_id}.")
        except Exception as exc:
            log(f"[semantic] Failed to queue semantic index for video {video_id}: {exc}")

    def _mark_job_failure(self, job_id: int, error: str):
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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

    @staticmethod
    def _load_job_payload(payload_json: str | None) -> dict:
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
        with Session(runtime.engine) as session:
            row = session.exec(
                select(Job.id)
                .where(Job.job_type.in_(list(job_types)), Job.status.in_(list(statuses)))
                .limit(1)
            ).first()
            return row is not None

    def _set_oldest_queued_job_status_detail(self, job_type: str, detail: str | None):
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise RuntimeError(f"Video {video_id} not found")
            _ = video.channel
            session.expunge(video)
            if video.channel:
                session.expunge(video.channel)
            return video

    def _queue_diarize_followup(self, video_id: int, parent_job_id: int):
        with Session(runtime.engine) as session:
            parent = session.get(Job, parent_job_id)
            if not parent:
                raise RuntimeError(f"Parent process job {parent_job_id} not found")
            payload = self._load_job_payload(parent.payload_json)
            payload["parent_job_id"] = int(parent_job_id)
            payload["pipeline_stage"] = "diarize"
            child = self._enqueue_job(video_id, "diarize", payload=payload)
            return int(child.id)

    def _mark_process_job_waiting_for_diarize(self, job_id: int, diarize_job_id: int):
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
            with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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

    def _check_job_not_paused(self, job_id: int | None) -> None:
        if not job_id:
            return
        with Session(runtime.engine) as session:
            job = session.get(Job, int(job_id))
            if not job:
                return
            if str(job.status or "").strip().lower() == "paused":
                raise JobPausedException("Job paused by user")
            if str(job.status or "").strip().lower() == "cancelled":
                raise JobCancelledException("Job cancelled by user")

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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
        with Session(runtime.engine) as session:
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
                        with Session(runtime.engine) as session:
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
                        with Session(runtime.engine) as session:
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
                        with Session(runtime.engine) as session:
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
                    with Session(runtime.engine) as session:
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
                        with Session(runtime.engine) as session:
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
                        with Session(runtime.engine) as session:
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

                with Session(runtime.engine) as session:
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
                    with Session(runtime.engine) as session:
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
