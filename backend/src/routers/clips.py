"""Clip CRUD, export, and YouTube upload endpoints."""
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    Clip,
    ClipExportArtifact,
    ClipExportArtifactRead,
    Video,
)
from ..deps import get_ingestion_service, get_session
from ..video_utils import (
    _enqueue_unique_job,
)
from ..schemas import (
    ChannelClipRead,
    ClipBatchYoutubeUploadRequest,
    ClipCaptionExportRequest,
    ClipCreate,
    ClipExportPresetRequest,
    ClipRead,
    ClipYoutubeUploadRequest,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


def _normalize_clip_defaults(clip: Clip) -> bool:
    changed = False
    if not getattr(clip, "aspect_ratio", None):
        clip.aspect_ratio = "source"
        changed = True
    if getattr(clip, "portrait_split_enabled", None) is None:
        clip.portrait_split_enabled = False
        changed = True
    if getattr(clip, "fade_in_sec", None) is None:
        clip.fade_in_sec = 0.0
        changed = True
    if getattr(clip, "fade_out_sec", None) is None:
        clip.fade_out_sec = 0.0
        changed = True
    if getattr(clip, "burn_captions", None) is None:
        clip.burn_captions = False
        changed = True
    if getattr(clip, "caption_speaker_labels", None) is None:
        clip.caption_speaker_labels = True
        changed = True
    return changed


def _clip_to_read(clip: Clip) -> ClipRead:
    return ClipRead(
        id=int(clip.id),
        video_id=int(clip.video_id),
        start_time=float(clip.start_time),
        end_time=float(clip.end_time),
        title=str(clip.title),
        aspect_ratio=str(clip.aspect_ratio or "source"),
        crop_x=clip.crop_x,
        crop_y=clip.crop_y,
        crop_w=clip.crop_w,
        crop_h=clip.crop_h,
        portrait_split_enabled=bool(getattr(clip, "portrait_split_enabled", False)),
        portrait_top_crop_x=getattr(clip, "portrait_top_crop_x", None),
        portrait_top_crop_y=getattr(clip, "portrait_top_crop_y", None),
        portrait_top_crop_w=getattr(clip, "portrait_top_crop_w", None),
        portrait_top_crop_h=getattr(clip, "portrait_top_crop_h", None),
        portrait_bottom_crop_x=getattr(clip, "portrait_bottom_crop_x", None),
        portrait_bottom_crop_y=getattr(clip, "portrait_bottom_crop_y", None),
        portrait_bottom_crop_w=getattr(clip, "portrait_bottom_crop_w", None),
        portrait_bottom_crop_h=getattr(clip, "portrait_bottom_crop_h", None),
        script_edits_json=getattr(clip, "script_edits_json", None),
        fade_in_sec=float(getattr(clip, "fade_in_sec", 0.0) or 0.0),
        fade_out_sec=float(getattr(clip, "fade_out_sec", 0.0) or 0.0),
        burn_captions=bool(clip.burn_captions),
        caption_speaker_labels=bool(clip.caption_speaker_labels),
        created_at=clip.created_at,
    )

@router.get("/videos/{video_id}/clips", response_model=List[ClipRead])
def read_video_clips(video_id: int, session: Session = Depends(get_session)):
    clips = session.exec(select(Clip).where(Clip.video_id == video_id).order_by(Clip.start_time)).all()
    changed = False
    for c in clips:
        if _normalize_clip_defaults(c):
            session.add(c)
            changed = True
    if changed:
        session.commit()
        for c in clips:
            session.refresh(c)
    return [_clip_to_read(c) for c in clips]

@router.post("/videos/{video_id}/clips", response_model=ClipRead)
def create_clip(video_id: int, clip: ClipCreate, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    db_clip = Clip.model_validate(clip, update={"video_id": video_id})
    _normalize_clip_defaults(db_clip)
    session.add(db_clip)
    session.commit()
    session.refresh(db_clip)
    return _clip_to_read(db_clip)

@router.delete("/clips/{clip_id}")
def delete_clip(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    session.delete(clip)
    session.commit()
    return {"ok": True}

@router.patch("/clips/{clip_id}", response_model=ClipRead)
def update_clip(clip_id: int, clip_update: ClipCreate, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    clip.title = clip_update.title
    clip.start_time = clip_update.start_time
    clip.end_time = clip_update.end_time
    clip.aspect_ratio = clip_update.aspect_ratio
    clip.crop_x = clip_update.crop_x
    clip.crop_y = clip_update.crop_y
    clip.crop_w = clip_update.crop_w
    clip.crop_h = clip_update.crop_h
    clip.portrait_split_enabled = bool(clip_update.portrait_split_enabled)
    clip.portrait_top_crop_x = clip_update.portrait_top_crop_x
    clip.portrait_top_crop_y = clip_update.portrait_top_crop_y
    clip.portrait_top_crop_w = clip_update.portrait_top_crop_w
    clip.portrait_top_crop_h = clip_update.portrait_top_crop_h
    clip.portrait_bottom_crop_x = clip_update.portrait_bottom_crop_x
    clip.portrait_bottom_crop_y = clip_update.portrait_bottom_crop_y
    clip.portrait_bottom_crop_w = clip_update.portrait_bottom_crop_w
    clip.portrait_bottom_crop_h = clip_update.portrait_bottom_crop_h
    clip.script_edits_json = clip_update.script_edits_json
    clip.fade_in_sec = max(0.0, float(clip_update.fade_in_sec or 0.0))
    clip.fade_out_sec = max(0.0, float(clip_update.fade_out_sec or 0.0))
    clip.burn_captions = bool(clip_update.burn_captions)
    clip.caption_speaker_labels = bool(clip_update.caption_speaker_labels)
    
    session.add(clip)
    session.commit()
    session.refresh(clip)
    _normalize_clip_defaults(clip)
    return _clip_to_read(clip)

@router.post("/clips/{clip_id}/apply-export-preset", response_model=ClipRead)
def apply_clip_export_preset(clip_id: int, body: ClipExportPresetRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if body.aspect_ratio is not None:
        clip.aspect_ratio = body.aspect_ratio
    if body.burn_captions is not None:
        clip.burn_captions = bool(body.burn_captions)
    if body.caption_speaker_labels is not None:
        clip.caption_speaker_labels = bool(body.caption_speaker_labels)
    if body.portrait_split_enabled is not None:
        clip.portrait_split_enabled = bool(body.portrait_split_enabled)
    if body.fade_in_sec is not None:
        clip.fade_in_sec = max(0.0, float(body.fade_in_sec))
    if body.fade_out_sec is not None:
        clip.fade_out_sec = max(0.0, float(body.fade_out_sec))
    for key in (
        "crop_x", "crop_y", "crop_w", "crop_h",
        "portrait_top_crop_x", "portrait_top_crop_y", "portrait_top_crop_w", "portrait_top_crop_h",
        "portrait_bottom_crop_x", "portrait_bottom_crop_y", "portrait_bottom_crop_w", "portrait_bottom_crop_h",
    ):
        val = getattr(body, key)
        if val is not None:
            setattr(clip, key, float(val))

    session.add(clip)
    session.commit()
    session.refresh(clip)
    _normalize_clip_defaults(clip)
    return _clip_to_read(clip)

@router.get("/channels/{channel_id}/clips", response_model=List[ChannelClipRead])
def read_channel_clips(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    rows = session.exec(
        select(Clip, Video)
        .join(Video, Clip.video_id == Video.id)
        .where(Video.channel_id == channel_id)
        .order_by(Clip.created_at.desc(), Clip.id.desc())
    ).all()
    out: List[ChannelClipRead] = []
    changed = False
    for clip, video in rows:
        if _normalize_clip_defaults(clip):
            session.add(clip)
            changed = True
        out.append(ChannelClipRead(
            id=int(clip.id),
            video_id=int(clip.video_id),
            start_time=float(clip.start_time),
            end_time=float(clip.end_time),
            title=str(clip.title),
            aspect_ratio=str(clip.aspect_ratio or "source"),
            crop_x=clip.crop_x,
            crop_y=clip.crop_y,
            crop_w=clip.crop_w,
            crop_h=clip.crop_h,
            portrait_split_enabled=bool(getattr(clip, "portrait_split_enabled", False)),
            portrait_top_crop_x=getattr(clip, "portrait_top_crop_x", None),
            portrait_top_crop_y=getattr(clip, "portrait_top_crop_y", None),
            portrait_top_crop_w=getattr(clip, "portrait_top_crop_w", None),
            portrait_top_crop_h=getattr(clip, "portrait_top_crop_h", None),
            portrait_bottom_crop_x=getattr(clip, "portrait_bottom_crop_x", None),
            portrait_bottom_crop_y=getattr(clip, "portrait_bottom_crop_y", None),
            portrait_bottom_crop_w=getattr(clip, "portrait_bottom_crop_w", None),
            portrait_bottom_crop_h=getattr(clip, "portrait_bottom_crop_h", None),
            script_edits_json=getattr(clip, "script_edits_json", None),
            fade_in_sec=float(getattr(clip, "fade_in_sec", 0.0) or 0.0),
            fade_out_sec=float(getattr(clip, "fade_out_sec", 0.0) or 0.0),
            burn_captions=bool(clip.burn_captions),
            caption_speaker_labels=bool(clip.caption_speaker_labels),
            created_at=clip.created_at,
            video_title=str(video.title),
            video_youtube_id=str(video.youtube_id),
            video_published_at=video.published_at,
            video_thumbnail_url=video.thumbnail_url,
        ))
    if changed:
        session.commit()
    return out

@router.post("/clips/{clip_id}/export/mp4")
def export_clip_mp4(clip_id: int):
    try:
        path = get_ingestion_service().render_clip_export_mp4(clip_id)
        try:
            get_ingestion_service().record_clip_export_artifact(clip_id, Path(path), artifact_type="video", fmt="mp4")
        except Exception:
            pass
        return FileResponse(str(path), filename=Path(path).name, media_type="video/mp4")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clips/{clip_id}/export/mp4/queue")
def queue_export_clip_mp4(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    payload = {"clip_id": int(clip_id)}
    job = _enqueue_unique_job(session, video_id=clip.video_id, job_type="clip_export_mp4", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}

@router.post("/clips/{clip_id}/export/captions")
def export_clip_captions(clip_id: int, body: ClipCaptionExportRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        speaker_labels = clip.caption_speaker_labels if body.speaker_labels is None else bool(body.speaker_labels)
        path = get_ingestion_service().write_clip_caption_file(clip_id, fmt=body.format, speaker_labels=speaker_labels)
        try:
            get_ingestion_service().record_clip_export_artifact(
                clip_id,
                Path(path),
                artifact_type="captions",
                fmt=(body.format or "srt").lower(),
            )
        except Exception:
            pass
        media_type = "text/vtt" if (body.format or "").lower() == "vtt" else "application/x-subrip"
        return FileResponse(str(path), filename=Path(path).name, media_type=media_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clips/{clip_id}/exports", response_model=List[ClipExportArtifactRead])
def read_clip_export_artifacts(clip_id: int, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    rows = session.exec(
        select(ClipExportArtifact)
        .where(ClipExportArtifact.clip_id == clip_id)
        .order_by(ClipExportArtifact.created_at.desc(), ClipExportArtifact.id.desc())
    ).all()
    return [
        ClipExportArtifactRead(
            id=int(r.id),
            clip_id=int(r.clip_id),
            video_id=int(r.video_id),
            artifact_type=str(r.artifact_type),
            format=str(r.format),
            file_path=str(r.file_path),
            file_name=str(r.file_name),
            file_size_bytes=r.file_size_bytes,
            created_at=r.created_at,
        ) for r in rows
    ]


@router.get("/videos/{video_id}/clip-exports", response_model=List[ClipExportArtifactRead])
def read_video_clip_export_artifacts(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    rows = session.exec(
        select(ClipExportArtifact)
        .where(ClipExportArtifact.video_id == video_id)
        .order_by(ClipExportArtifact.created_at.desc(), ClipExportArtifact.id.desc())
    ).all()
    return [
        ClipExportArtifactRead(
            id=int(r.id),
            clip_id=int(r.clip_id),
            video_id=int(r.video_id),
            artifact_type=str(r.artifact_type),
            format=str(r.format),
            file_path=str(r.file_path),
            file_name=str(r.file_name),
            file_size_bytes=r.file_size_bytes,
            created_at=r.created_at,
        ) for r in rows
    ]


@router.get("/clip-exports/{artifact_id}/download")
def download_clip_export_artifact(artifact_id: int, session: Session = Depends(get_session)):
    art = session.get(ClipExportArtifact, artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Clip export artifact not found")
    p = Path(art.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Archived clip file no longer exists on disk")

    fmt = (art.format or "").lower()
    media_type = "application/octet-stream"
    if fmt == "mp4":
        media_type = "video/mp4"
    elif fmt == "srt":
        media_type = "application/x-subrip"
    elif fmt == "vtt":
        media_type = "text/vtt"
    return FileResponse(str(p), filename=art.file_name or p.name, media_type=media_type)


@router.post("/clips/{clip_id}/export/captions/queue")
def queue_export_clip_captions(clip_id: int, body: ClipCaptionExportRequest, session: Session = Depends(get_session)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    fmt = (body.format or "srt").lower()
    if fmt not in {"srt", "vtt"}:
        raise HTTPException(status_code=400, detail="format must be srt or vtt")
    speaker_labels = clip.caption_speaker_labels if body.speaker_labels is None else bool(body.speaker_labels)
    payload = {"clip_id": int(clip_id), "format": fmt, "speaker_labels": bool(speaker_labels)}
    job = _enqueue_unique_job(session, video_id=clip.video_id, job_type="clip_export_captions", payload=payload)
    return {"job_id": job.id, "video_id": job.video_id, "job_type": job.job_type, "status": job.status}


def _upload_clip_to_youtube_internal(
    session: Session,
    clip_id: int,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy_status: str = "private",
    category_id: str = "22",
    made_for_kids: bool = False,
    tags: Optional[List[str]] = None,
) -> dict:
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
    video = session.get(Video, clip.video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Source video for clip {clip_id} not found")

    privacy = (privacy_status or "private").strip().lower()
    if privacy not in {"private", "unlisted", "public"}:
        raise HTTPException(status_code=400, detail="privacy_status must be one of: private, unlisted, public")

    safe_title = (title or clip.title or f"Clip from {video.title}").strip()
    if not safe_title:
        safe_title = f"Clip from {video.title}"
    safe_title = safe_title[:100]

    safe_description = (description or _main()._build_default_clip_upload_description(video, clip)).strip()[:5000]
    safe_category = (category_id or "22").strip()
    safe_tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:40]  # YouTube max 500 chars across tags

    snippet = {
        "title": safe_title,
        "description": safe_description,
        "categoryId": safe_category,
    }
    if safe_tags:
        snippet["tags"] = safe_tags
    status = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": bool(made_for_kids),
    }

    try:
        export_path = get_ingestion_service().render_clip_export_mp4(clip_id)
        uploaded = _main()._youtube_upload_video_resumable(export_path, snippet=snippet, status=status)
        new_video_id = str(uploaded.get("id") or "").strip()
        if not new_video_id:
            raise RuntimeError(f"YouTube upload response missing video id: {uploaded}")
        ch_info = _main()._youtube_fetch_authenticated_channel_info()
        return {
            "clip_id": clip.id,
            "source_video_id": video.id,
            "source_video_youtube_id": video.youtube_id,
            "uploaded_video_id": new_video_id,
            "uploaded_watch_url": f"https://www.youtube.com/watch?v={new_video_id}",
            "uploaded_title": safe_title,
            "privacy_status": privacy,
            "channel_id": ch_info.get("channel_id"),
            "channel_title": ch_info.get("channel_title"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip upload failed: {e}")


@router.post("/clips/{clip_id}/youtube/upload")
def upload_clip_to_youtube(clip_id: int, body: Optional[ClipYoutubeUploadRequest] = None, session: Session = Depends(get_session)):
    req = body or ClipYoutubeUploadRequest()
    return _upload_clip_to_youtube_internal(
        session,
        clip_id,
        title=req.title,
        description=req.description,
        privacy_status=req.privacy_status,
        category_id=req.category_id,
        made_for_kids=req.made_for_kids,
        tags=req.tags,
    )


@router.post("/clips/youtube/upload-batch")
def upload_clips_to_youtube_batch(req: ClipBatchYoutubeUploadRequest, session: Session = Depends(get_session)):
    clip_ids = [int(c) for c in (req.clip_ids or []) if int(c) > 0]
    if not clip_ids:
        raise HTTPException(status_code=400, detail="clip_ids is required")

    # Ensure OAuth is valid before starting the batch.
    try:
        auth_channel = _main()._youtube_fetch_authenticated_channel_info()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YouTube not connected/authorized: {e}")

    out = []
    success = 0
    failed = 0
    for clip_id in clip_ids:
        try:
            result = _upload_clip_to_youtube_internal(
                session,
                clip_id,
                privacy_status=req.privacy_status,
                category_id=req.category_id,
                made_for_kids=req.made_for_kids,
                tags=req.tags,
            )
            result["success"] = True
            out.append(result)
            success += 1
        except HTTPException as e:
            out.append({
                "clip_id": clip_id,
                "success": False,
                "error": e.detail,
            })
            failed += 1
        except Exception as e:
            out.append({
                "clip_id": clip_id,
                "success": False,
                "error": str(e),
            })
            failed += 1

    return {
        "auth_channel_id": auth_channel.get("channel_id"),
        "auth_channel_title": auth_channel.get("channel_title"),
        "requested": len(clip_ids),
        "uploaded": success,
        "failed": failed,
        "results": out,
    }
