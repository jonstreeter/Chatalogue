"""Video media, VoiceFixer, ClearVoice cleanup, and reconstruction workbench endpoints."""
import mimetypes
import time
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..db.database import Job, TranscriptSegment, Video
from ..deps import get_ingestion_service, get_session
from ..job_utils import PIPELINE_ACTIVE_STATUSES
from ..video_utils import _enqueue_unique_job
from ..schemas import (
    CleanupWorkbenchRead,
    CleanupWorkbenchRunCandidateRequest,
    CleanupWorkbenchSelectCandidateRequest,
    ReconstructionSegmentPreviewRequest,
    ReconstructionSegmentPreviewResult,
    ReconstructionSettingsUpdateRequest,
    ReconstructionSpeakerTestRequest,
    ReconstructionSpeakerTestResult,
    ReconstructionUseForPlaybackRequest,
    ReconstructionWorkbenchAddSampleRequest,
    ReconstructionWorkbenchRead,
    ReconstructionWorkbenchSampleCleanupRequest,
    ReconstructionWorkbenchSampleStateRequest,
    ReconstructionWorkbenchSpeakerApprovalRequest,
    UploadedPlaybackSourceRequest,
    VoiceFixerSettingsUpdateRequest,
    WorkbenchTaskProgressRead,
    VoiceFixerUseCleanedRequest,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


def _invalidate_voicefixer_output(video: Video) -> None:
    apply_scope = str(getattr(video, "voicefixer_apply_scope", "none") or "none").strip().lower()
    video.voicefixer_cleaned_path = None
    video.voicefixer_use_cleaned = False
    video.voicefixer_status = "disabled" if apply_scope == "none" else None
    video.voicefixer_error = None


def _invalidate_reconstruction_output(video: Video) -> None:
    video.reconstruction_audio_path = None
    video.reconstruction_use_for_playback = False
    video.reconstruction_status = None
    video.reconstruction_error = None


@router.get("/videos/{video_id}/media")
def stream_video_media(video_id: int, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if (video.media_source_type or "youtube") == "youtube":
        raise HTTPException(status_code=409, detail="This episode is streamed from YouTube, not from local media.")

    media_path = get_ingestion_service().get_audio_path(video, purpose="playback")
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Local media is not available yet. Start processing or wait for download to complete.")

    media_type = mimetypes.guess_type(str(media_path))[0] or "application/octet-stream"
    return FileResponse(path=media_path, media_type=media_type, filename=media_path.name)


@router.patch("/videos/{video_id}/voicefixer/settings", response_model=Video)
def update_voicefixer_settings(
    video_id: int,
    body: VoiceFixerSettingsUpdateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot change VoiceFixer settings while this episode has an active job.")

    mode = max(0, min(2, int(body.mode)))
    mix_ratio = max(0.0, min(1.0, float(body.mix_ratio)))
    leveling_mode = str(body.leveling_mode or "off").strip().lower()
    if leveling_mode not in {"off", "gentle", "balanced", "strong"}:
        raise HTTPException(status_code=400, detail="Invalid VoiceFixer leveling mode.")
    apply_scope = str(body.apply_scope or "none").strip().lower()
    if apply_scope not in {"none", "playback", "processing", "both"}:
        raise HTTPException(status_code=400, detail="Invalid VoiceFixer apply scope.")

    settings_changed = (
        int(video.voicefixer_mode or 0) != mode
        or abs(float(video.voicefixer_mix_ratio or 1.0) - mix_ratio) > 1e-6
        or str(video.voicefixer_leveling_mode or "off").strip().lower() != leveling_mode
    )
    video.voicefixer_mode = mode
    video.voicefixer_mix_ratio = mix_ratio
    video.voicefixer_leveling_mode = leveling_mode
    video.voicefixer_apply_scope = apply_scope
    video.voicefixer_use_cleaned = apply_scope != "none"
    if settings_changed and (
        str(getattr(video, "voicefixer_cleaned_path", "") or "").strip()
        or str(getattr(video, "voicefixer_status", "") or "").strip().lower() in {"ready", "disabled"}
    ):
        _invalidate_voicefixer_output(video)
    if apply_scope == "none" and str(video.voicefixer_status or "").strip().lower() == "ready":
        video.voicefixer_status = "disabled"
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@router.post("/videos/{video_id}/reconstruct/queue", response_model=Video)
def queue_conversation_reconstruction(video_id: int, force: bool = False, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    install_info = _main()._get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="This episode already has an active processing job. Wait for it to finish first.")

    video.reconstruction_status = "queued"
    video.reconstruction_error = None
    session.add(video)
    session.commit()
    _enqueue_unique_job(session, video_id=video_id, job_type="conversation_reconstruct", payload={"force": bool(force)})
    session.refresh(video)
    return video


@router.patch("/videos/{video_id}/reconstruction/settings", response_model=Video)
def update_reconstruction_settings(
    video_id: int,
    body: ReconstructionSettingsUpdateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot change reconstruction settings while this episode has an active job.")

    mode = "performance"
    instruction_template = str(body.instruction_template or "").strip() or None

    settings_changed = (
        str(video.reconstruction_mode or "basic").strip().lower() != mode
        or str(video.reconstruction_instruction_template or "").strip() != str(instruction_template or "")
    )
    video.reconstruction_mode = mode
    video.reconstruction_instruction_template = instruction_template
    if settings_changed and (
        str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        or str(getattr(video, "reconstruction_status", "") or "").strip().lower() == "ready"
    ):
        _invalidate_reconstruction_output(video)
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@router.get("/videos/{video_id}/reconstruction/workbench", response_model=ReconstructionWorkbenchRead)
def get_reconstruction_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    segments = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
    ).all()
    if not segments:
        raise HTTPException(status_code=409, detail="Transcript segments are required before opening the reconstruction workbench.")
    if not any(getattr(seg, "speaker_id", None) is not None for seg in segments):
        raise HTTPException(status_code=409, detail="Diarization must be completed before opening the reconstruction workbench.")

    try:
        source_path = get_ingestion_service().get_audio_path(video, purpose="processing")
        workbench = get_ingestion_service()._build_reconstruction_workbench(
            video,
            segments,
            source_path,
            progress_task="load_workbench",
        )
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="load_workbench",
            status="error",
            stage="error",
            message=str(e)[:240] or "Failed to load reconstruction workbench.",
        )
        raise
    base_url = f"/videos/{video_id}/reconstruction/workbench/audio"
    workbench_dir = get_ingestion_service()._reconstruction_workbench_dir(video)

    def _workbench_audio_url(filename: str) -> str | None:
        safe_name = str(filename or "").strip()
        if not safe_name:
            return None
        audio_path = workbench_dir / Path(safe_name).name
        version = ""
        try:
            version = str(int(audio_path.stat().st_mtime_ns))
        except Exception:
            version = str(int(time.time() * 1000))
        params = {"name": Path(safe_name).name, "v": version}
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    speakers = []
    for item in workbench["speakers"]:
        ref_name = str(item.pop("reference_audio_filename", "") or "").strip()
        test_name = str(item.pop("latest_test_audio_filename", "") or "").strip()
        samples = []
        for sample in item.pop("samples", []) or []:
            sample_audio_name = str(sample.pop("audio_filename", "") or "").strip()
            sample_cleaned_name = str(sample.pop("cleaned_audio_filename", "") or "").strip()
            sample["audio_url"] = _workbench_audio_url(sample_audio_name)
            sample["cleaned_audio_url"] = _workbench_audio_url(sample_cleaned_name)
            samples.append(sample)
        item["reference_audio_url"] = _workbench_audio_url(ref_name)
        item["latest_test_audio_url"] = _workbench_audio_url(test_name)
        item["samples"] = samples
        speakers.append(item)
    workbench["speakers"] = speakers
    return ReconstructionWorkbenchRead(**workbench)


@router.get("/videos/{video_id}/workbench/progress", response_model=WorkbenchTaskProgressRead)
def get_video_workbench_progress(video_id: int):
    try:
        if get_ingestion_service() is None:
            return WorkbenchTaskProgressRead(video_id=int(video_id), status="idle")
        return WorkbenchTaskProgressRead(**get_ingestion_service().get_workbench_task_progress(video_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workbench progress unavailable: {e}")


@router.get("/videos/{video_id}/reconstruction/workbench/audio")
def stream_reconstruction_workbench_audio(video_id: int, name: str, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    safe_name = Path(str(name or "")).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Missing workbench audio filename.")
    workbench_dir = get_ingestion_service()._reconstruction_workbench_dir(video)
    audio_path = workbench_dir / safe_name
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Workbench audio file not found.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        filename=audio_path.name,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.post("/videos/{video_id}/reconstruction/test-speaker", response_model=ReconstructionSpeakerTestResult)
def test_reconstruction_speaker(
    video_id: int,
    body: ReconstructionSpeakerTestRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")

    install_info = _main()._get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")

    try:
        result = get_ingestion_service().generate_reconstruction_speaker_test(
            video_id,
            speaker_id=int(body.speaker_id),
            text=body.text,
            segment_id=body.segment_id,
            performance_mode=bool(body.performance_mode),
            progress_task="speaker_test",
        )
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="speaker_test",
            status="error",
            stage="error",
            message=f"Speaker test synthesis failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Speaker test synthesis failed: {e}")

    audio_filename = str(result.pop("audio_filename"))
    try:
        audio_path = get_ingestion_service()._reconstruction_workbench_dir(video) / Path(audio_filename).name
        version = str(int(audio_path.stat().st_mtime_ns))
    except Exception:
        version = str(int(time.time() * 1000))
    result["audio_url"] = f"/videos/{video_id}/reconstruction/workbench/audio?{urllib.parse.urlencode({'name': Path(audio_filename).name, 'v': version})}"
    return ReconstructionSpeakerTestResult(**result)


@router.post("/videos/{video_id}/reconstruction/workbench/sample-cleanup", response_model=ReconstructionWorkbenchRead)
def cleanup_reconstruction_workbench_sample(
    video_id: int,
    body: ReconstructionWorkbenchSampleCleanupRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    if not _main()._get_voicefixer_install_info().installed:
        raise HTTPException(status_code=409, detail="VoiceFixer is not installed yet. Install and test it from Settings first.")
    try:
        get_ingestion_service().cleanup_reconstruction_sample(
            video_id,
            speaker_id=int(body.speaker_id),
            segment_id=int(body.segment_id),
        )
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="sample_cleanup",
            status="error",
            stage="error",
            message=f"Sample cleanup failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Sample cleanup failed: {e}")
    return get_reconstruction_workbench(video_id, session)


@router.patch("/videos/{video_id}/reconstruction/workbench/sample-state", response_model=ReconstructionWorkbenchRead)
def update_reconstruction_workbench_sample_state(
    video_id: int,
    body: ReconstructionWorkbenchSampleStateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        get_ingestion_service().update_reconstruction_sample_state(
            video_id,
            speaker_id=int(body.speaker_id),
            segment_id=int(body.segment_id),
            rejected=body.rejected,
            selected=body.selected,
            clear_cleaned=body.clear_cleaned,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update sample state: {e}")
    return get_reconstruction_workbench(video_id, session)


@router.post("/videos/{video_id}/reconstruction/workbench/add-sample", response_model=ReconstructionWorkbenchRead)
def add_reconstruction_workbench_sample(
    video_id: int,
    body: ReconstructionWorkbenchAddSampleRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        get_ingestion_service().add_reconstruction_performance_sample(video_id, speaker_id=int(body.speaker_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add a new sample: {e}")
    return get_reconstruction_workbench(video_id, session)


@router.patch("/videos/{video_id}/reconstruction/workbench/speaker-approval", response_model=ReconstructionWorkbenchRead)
def set_reconstruction_workbench_speaker_approval(
    video_id: int,
    body: ReconstructionWorkbenchSpeakerApprovalRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    try:
        get_ingestion_service().set_reconstruction_speaker_approval(video_id, speaker_id=int(body.speaker_id), approved=bool(body.approved))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update speaker approval: {e}")
    return get_reconstruction_workbench(video_id, session)


@router.post("/videos/{video_id}/reconstruction/preview-segment", response_model=ReconstructionSegmentPreviewResult)
def preview_reconstruction_segment(
    video_id: int,
    body: ReconstructionSegmentPreviewRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is currently available for manually uploaded media only.")
    install_info = _main()._get_reconstruction_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="The reconstruction runtime is not installed yet. Install and test it from Settings first.")
    try:
        result = get_ingestion_service().preview_reconstruction_segment(
            video_id,
            segment_id=int(body.segment_id),
            performance_mode=body.performance_mode,
        )
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            video_id,
            area="reconstruction",
            task="preview_segment",
            status="error",
            stage="error",
            message=f"Segment preview failed: {str(e)[:220]}",
        )
        raise HTTPException(status_code=500, detail=f"Segment preview failed: {e}")
    audio_filename = str(result.pop("audio_filename"))
    try:
        audio_path = get_ingestion_service()._reconstruction_workbench_dir(video) / Path(audio_filename).name
        version = str(int(audio_path.stat().st_mtime_ns))
    except Exception:
        version = str(int(time.time() * 1000))
    result["audio_url"] = f"/videos/{video_id}/reconstruction/workbench/audio?{urllib.parse.urlencode({'name': Path(audio_filename).name, 'v': version})}"
    return ReconstructionSegmentPreviewResult(**result)


@router.get("/videos/{video_id}/reconstruction/audio")
def stream_reconstruction_audio(video_id: int, session: Session = Depends(get_session)):
    import mimetypes

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
    if not rel:
        raise HTTPException(status_code=404, detail="No reconstructed audio is available for this episode yet.")
    audio_path = get_ingestion_service().get_manual_media_absolute_path(rel)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Reconstructed audio file is missing.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@router.patch("/videos/{video_id}/reconstruction/use-for-playback", response_model=Video)
def set_reconstruction_use_for_playback(
    video_id: int,
    body: ReconstructionUseForPlaybackRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Conversation reconstruction is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch reconstruction playback while this episode has an active job.")

    if body.enabled:
        rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        if not rel:
            raise HTTPException(status_code=409, detail="No reconstructed audio exists yet for this episode.")
        audio_path = get_ingestion_service().get_manual_media_absolute_path(rel)
        if audio_path is None or not audio_path.exists():
            raise HTTPException(status_code=409, detail="The reconstructed audio file is missing.")
        video.reconstruction_use_for_playback = True
        if str(video.reconstruction_status or "").strip().lower() in {"", "disabled"}:
            video.reconstruction_status = "ready"
        video.reconstruction_error = None
    else:
        video.reconstruction_use_for_playback = False
        video.reconstruction_error = None

    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@router.patch("/videos/{video_id}/playback-source", response_model=Video)
def set_uploaded_playback_source(
    video_id: int,
    body: UploadedPlaybackSourceRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Playback source switching is only available for manually uploaded media.")

    source = str(body.source or "original").strip().lower()
    if source not in {"original", "cleaned", "reconstructed"}:
        raise HTTPException(status_code=400, detail="Invalid playback source.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup", "conversation_reconstruct"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch playback source while this episode has an active job.")

    apply_scope = str(getattr(video, "voicefixer_apply_scope", "") or "").strip().lower()
    processing_uses_cleaned = apply_scope in {"processing", "both"}

    if source == "cleaned":
        cleaned_path = get_ingestion_service().get_voicefixer_cleaned_absolute_path(video)
        if cleaned_path is None or not cleaned_path.exists():
            raise HTTPException(status_code=409, detail="No VoiceFixer-cleaned media exists yet for this episode.")
        video.reconstruction_use_for_playback = False
        video.voicefixer_apply_scope = "both" if processing_uses_cleaned else "playback"
        video.voicefixer_use_cleaned = True
        video.voicefixer_status = "ready"
        video.voicefixer_error = None
    elif source == "reconstructed":
        rel = str(getattr(video, "reconstruction_audio_path", "") or "").strip()
        if not rel:
            raise HTTPException(status_code=409, detail="No reconstructed audio exists yet for this episode.")
        audio_path = get_ingestion_service().get_manual_media_absolute_path(rel)
        if audio_path is None or not audio_path.exists():
            raise HTTPException(status_code=409, detail="The reconstructed audio file is missing.")
        video.reconstruction_use_for_playback = True
        if str(video.reconstruction_status or "").strip().lower() in {"", "disabled"}:
            video.reconstruction_status = "ready"
        video.reconstruction_error = None
    else:
        video.reconstruction_use_for_playback = False
        video.voicefixer_apply_scope = "processing" if processing_uses_cleaned else "none"
        video.voicefixer_use_cleaned = processing_uses_cleaned
        if not processing_uses_cleaned and str(video.voicefixer_status or "").strip().lower() == "ready":
            video.voicefixer_status = "disabled"
        video.voicefixer_error = None

    session.add(video)
    session.commit()
    session.refresh(video)
    return video


@router.get("/videos/{video_id}/cleanup/workbench", response_model=CleanupWorkbenchRead)
def get_cleanup_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = get_ingestion_service().build_cleanup_workbench(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load the cleanup workbench: {e}")
    clearvoice_info = _main()._get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@router.post("/videos/{video_id}/cleanup/workbench/analyze", response_model=CleanupWorkbenchRead)
def analyze_cleanup_workbench(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = get_ingestion_service().analyze_cleanup_workbench_audio(video_id)
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="analyze",
            status="error",
            stage="error",
            message=str(e)[:240] or "Cleanup analysis failed.",
        )
        raise HTTPException(status_code=500, detail=f"Cleanup analysis failed: {e}")
    clearvoice_info = _main()._get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@router.post("/videos/{video_id}/cleanup/workbench/clearvoice-candidate", response_model=CleanupWorkbenchRead)
def run_cleanup_clearvoice_candidate(
    video_id: int,
    body: CleanupWorkbenchRunCandidateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    clearvoice_info = _main()._get_clearvoice_install_info()
    if not clearvoice_info.installed:
        raise HTTPException(status_code=409, detail="ClearVoice is not installed yet. Install and test it from the cleanup workbench first.")
    if not clearvoice_info.runtime_ready:
        raise HTTPException(
            status_code=409,
            detail=clearvoice_info.message or clearvoice_info.runtime_error or "ClearVoice is installed, but its runtime is not healthy yet.",
        )
    try:
        workbench = get_ingestion_service().run_cleanup_clearvoice_candidate(
            video_id,
            stage=body.stage,
            model_name=body.model_name,
            source_candidate_id=body.source_candidate_id,
        )
    except Exception as e:
        get_ingestion_service()._set_workbench_task_progress(
            int(video_id),
            area="cleanup",
            task="clearvoice_candidate",
            status="error",
            stage="error",
            message=str(e)[:240] or "ClearVoice candidate generation failed.",
        )
        raise HTTPException(status_code=500, detail=f"ClearVoice candidate generation failed: {e}")
    workbench["clearvoice_available"] = True
    return CleanupWorkbenchRead(**workbench)


@router.patch("/videos/{video_id}/cleanup/workbench/select-candidate", response_model=CleanupWorkbenchRead)
def select_cleanup_workbench_candidate(
    video_id: int,
    body: CleanupWorkbenchSelectCandidateRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench is only available for manually uploaded media.")
    try:
        workbench = get_ingestion_service().select_cleanup_workbench_candidate(video_id, candidate_id=body.candidate_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update the selected pre-cleanup candidate: {e}")
    clearvoice_info = _main()._get_clearvoice_install_info()
    workbench["clearvoice_available"] = bool(clearvoice_info.installed and clearvoice_info.runtime_ready)
    return CleanupWorkbenchRead(**workbench)


@router.get("/videos/{video_id}/cleanup/workbench/audio")
def get_cleanup_workbench_audio(video_id: int, name: str, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="Cleanup workbench audio is only available for manually uploaded media.")
    workbench_dir = get_ingestion_service()._cleanup_workbench_dir(video)
    audio_path = workbench_dir / Path(str(name or "")).name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Cleanup workbench audio file not found.")
    media_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@router.post("/videos/{video_id}/voicefixer/queue", response_model=Video)
def queue_voicefixer_cleanup(video_id: int, force: bool = False, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    install_info = _main()._get_voicefixer_install_info()
    if not install_info.installed:
        raise HTTPException(status_code=409, detail="VoiceFixer is not installed yet. Install and test it from Settings first.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="This episode already has an active processing job. Wait for it to finish first.")

    get_ingestion_service()._set_voicefixer_state(video_id, status="queued", error="")
    _enqueue_unique_job(session, video_id=video_id, job_type="voicefixer_cleanup", payload={"force": bool(force)})
    session.refresh(video)
    return session.get(Video, video_id)


@router.patch("/videos/{video_id}/voicefixer/use-cleaned", response_model=Video)
def set_voicefixer_use_cleaned(
    video_id: int,
    body: VoiceFixerUseCleanedRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(getattr(video, "media_source_type", "") or "").lower() != "upload":
        raise HTTPException(status_code=409, detail="VoiceFixer is only available for manually uploaded media.")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.job_type.in_(["process", "diarize", "voicefixer_cleanup"]),
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail="Cannot switch VoiceFixer media while this episode has an active job.")

    if body.enabled:
        cleaned_path = get_ingestion_service().get_voicefixer_cleaned_absolute_path(video)
        if cleaned_path is None or not cleaned_path.exists():
            raise HTTPException(status_code=409, detail="No VoiceFixer-cleaned media exists yet for this episode.")
        video.voicefixer_use_cleaned = True
        video.voicefixer_apply_scope = "both"
        video.voicefixer_status = "ready"
        video.voicefixer_error = None
    else:
        video.voicefixer_use_cleaned = False
        video.voicefixer_apply_scope = "none"
        if str(video.voicefixer_status or "").strip().lower() == "ready":
            video.voicefixer_status = "disabled"
        video.voicefixer_error = None
    session.add(video)
    session.commit()
    session.refresh(video)
    return video
