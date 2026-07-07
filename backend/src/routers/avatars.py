"""Avatar and personality-training endpoints: workbench, datasets, judge passes,
long-form samples, training lifecycle, snapshots, test chat, fit checks."""
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db.database import (
    Avatar,
    Speaker,
)
from ..deps import get_session
from ..services import avatar_personality as avatar_svc
from ..services import semantic_search as sem_svc
from ..schemas import (
    AvatarCreateRequest,
    AvatarPersonalityBaseModelDownloadRequest,
    AvatarPersonalityBaseModelSupportRead,
    AvatarPersonalityDatasetExampleRead,
    AvatarPersonalityDatasetPageRead,
    AvatarPersonalityDatasetRead,
    AvatarPersonalityExampleStateRequest,
    AvatarPersonalityFitCheckRequest,
    AvatarPersonalityFitCheckPromptResultRead,
    AvatarPersonalityFitCheckResponse,
    AvatarPersonalityJudgePassRequest,
    AvatarPersonalityJudgeStatusRead,
    AvatarPersonalityLongFormConfigRead,
    AvatarPersonalityLongFormConfigUpdateRequest,
    AvatarPersonalityLongFormPageRead,
    AvatarPersonalityLongFormSampleRead,
    AvatarPersonalityLongFormSampleStateRequest,
    AvatarPersonalitySnapshotCleanupRequest,
    AvatarPersonalitySnapshotDeleteRequest,
    AvatarPersonalitySnapshotSelectRequest,
    AvatarPersonalityTestChatRequest,
    AvatarPersonalityTestChatResponse,
    AvatarPersonalityTrainingConfigRead,
    AvatarPersonalityTrainingConfigUpdateRequest,
    AvatarPersonalityTrainingPackageRead,
    AvatarPersonalityTrainingStatusRead,
    AvatarPersonalityTrainRequest,
    AvatarRead,
    AvatarUpdateRequest,
    AvatarWorkbenchRead,
    SemanticSearchHit,
    SemanticSearchPage,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/avatars", response_model=List[AvatarRead])
def read_avatars(
    channel_id: Optional[int] = None,
    speaker_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    query = select(Avatar)
    if channel_id is not None:
        query = query.where(Avatar.channel_id == channel_id)
    if speaker_id is not None:
        query = query.where(Avatar.speaker_id == speaker_id)
    query = query.order_by(Avatar.updated_at.desc(), Avatar.id.desc())
    avatars = session.exec(query).all()
    return [avatar_svc._serialize_avatar(avatar) for avatar in avatars]


@router.post("/avatars", response_model=AvatarRead)
def create_avatar(body: AvatarCreateRequest, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, body.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    avatar_name = str(body.name or "").strip() or f"{speaker.name} Avatar"
    avatar = Avatar(
        channel_id=int(speaker.channel_id),
        speaker_id=int(speaker.id),
        name=avatar_name,
        description=(str(body.description).strip() or None) if body.description is not None else None,
    )
    session.add(avatar)
    session.commit()
    session.refresh(avatar)
    avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(avatar)
    avatar_svc._avatar_artifacts_dir(avatar)
    return avatar_svc._serialize_avatar(avatar)


@router.post("/speakers/{speaker_id}/avatar", response_model=AvatarRead)
def create_or_open_speaker_avatar(speaker_id: int, session: Session = Depends(get_session)):
    speaker = session.get(Speaker, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    avatar = session.exec(
        select(Avatar)
        .where(Avatar.speaker_id == speaker_id)
        .order_by(Avatar.updated_at.desc(), Avatar.id.desc())
    ).first()
    if not avatar:
        avatar = Avatar(
            channel_id=int(speaker.channel_id),
            speaker_id=int(speaker.id),
            name=f"{speaker.name} Avatar",
        )
        session.add(avatar)
        session.commit()
        session.refresh(avatar)

    avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(avatar)
    avatar_svc._avatar_artifacts_dir(avatar)
    return avatar_svc._serialize_avatar(avatar)


@router.get("/avatars/{avatar_id}", response_model=AvatarRead)
def read_avatar(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._serialize_avatar(avatar)


@router.patch("/avatars/{avatar_id}", response_model=AvatarRead)
def update_avatar(avatar_id: int, body: AvatarUpdateRequest, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    source_speaker = session.get(Speaker, avatar.speaker_id)
    personality, appearance, voice = avatar_svc._ensure_avatar_profiles(
        session,
        avatar,
        speaker_name=source_speaker.name if source_speaker else avatar.name,
    )

    if body.name is not None:
        trimmed_name = str(body.name).strip()
        if not trimmed_name:
            raise HTTPException(status_code=400, detail="Avatar name cannot be empty")
        avatar.name = trimmed_name
    if body.status is not None:
        avatar.status = str(body.status).strip() or "draft"
    if body.description is not None:
        avatar.description = str(body.description).strip() or None
    if body.personality_system_prompt is not None:
        personality.system_prompt = str(body.personality_system_prompt).strip() or None
    if body.personality_base_model_id is not None:
        personality.base_model_id = str(body.personality_base_model_id).strip() or None
    if body.appearance_primary_image_path is not None:
        appearance.primary_image_path = str(body.appearance_primary_image_path).strip() or None
    if body.voice_primary_reference_path is not None:
        voice.primary_reference_path = str(body.voice_primary_reference_path).strip() or None
    if body.voice_provider is not None:
        voice.provider = str(body.voice_provider).strip() or None

    now = datetime.now()
    avatar.updated_at = now
    personality.updated_at = now
    appearance.updated_at = now
    voice.updated_at = now

    session.add(avatar)
    session.add(personality)
    session.add(appearance)
    session.add(voice)
    session.commit()
    session.refresh(avatar)
    return avatar_svc._serialize_avatar(avatar)


@router.get("/avatars/{avatar_id}/workbench", response_model=AvatarWorkbenchRead)
def get_avatar_workbench(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._build_avatar_workbench(session, avatar)


@router.get("/avatars/{avatar_id}/personality/dataset-preview", response_model=AvatarPersonalityDatasetRead)
def get_avatar_personality_dataset_preview(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    return avatar_svc._load_avatar_personality_dataset(avatar, personality)


@router.post("/avatars/{avatar_id}/personality/build-dataset", response_model=AvatarPersonalityDatasetRead)
def build_avatar_personality_dataset(avatar_id: int, session: Session = Depends(get_session)):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(personality)
    return avatar_svc._build_avatar_personality_dataset(session, avatar, speaker, personality)


@router.post("/avatars/{avatar_id}/personality/run-judge-pass", response_model=AvatarPersonalityDatasetRead)
def run_avatar_personality_judge_pass(
    avatar_id: int,
    body: AvatarPersonalityJudgePassRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    session.commit()
    session.refresh(personality)
    dataset = avatar_svc._run_avatar_personality_judge_pass(
        avatar,
        personality,
        max_examples=body.max_examples,
        overwrite_existing=body.overwrite_existing,
        target_filter=body.target_filter,
    )
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return dataset


@router.post("/avatars/{avatar_id}/personality/start-judge-pass", response_model=AvatarPersonalityJudgeStatusRead)
def start_avatar_personality_judge_pass(
    avatar_id: int,
    body: AvatarPersonalityJudgePassRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._start_avatar_personality_judge_pass(
        avatar_id=int(avatar_id),
        max_examples=int(body.max_examples or 40),
        overwrite_existing=bool(body.overwrite_existing),
        target_filter=str(body.target_filter or "needs_review"),
    )


@router.get("/avatars/{avatar_id}/personality/judge-status", response_model=AvatarPersonalityJudgeStatusRead)
def read_avatar_personality_judge_status(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._load_avatar_personality_judge_status(avatar)


@router.post("/avatars/{avatar_id}/personality/stop-judge-pass", response_model=AvatarPersonalityJudgeStatusRead)
def stop_avatar_personality_judge_pass(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    is_active = False
    with avatar_svc._avatar_judge_runs_lock:
        stop_event = avatar_svc._avatar_judge_stop_events.get(int(avatar_id))
        active_thread = avatar_svc._avatar_judge_threads.get(int(avatar_id))
        if stop_event and active_thread and active_thread.is_alive():
            stop_event.set()
            is_active = True
    if not is_active:
        return avatar_svc._load_avatar_personality_judge_status(avatar)
    return avatar_svc._write_avatar_personality_judge_status(
        avatar,
        {
            "status": "stopping",
            "active": True,
            "stop_requested": True,
            "current_stage": "stop_requested",
        },
    )


@router.get("/avatars/{avatar_id}/personality/long-form-samples", response_model=AvatarPersonalityLongFormPageRead)
def read_avatar_personality_long_form_samples(
    avatar_id: int,
    offset: int = 0,
    limit: int = 20,
    state: str = "all",
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._read_avatar_personality_long_form_page(
        session,
        avatar,
        offset=max(0, int(offset)),
        limit=max(1, min(int(limit), 100)),
        state=state,
    )


@router.patch("/avatars/{avatar_id}/personality/long-form-samples/{sample_id}", response_model=AvatarPersonalityLongFormSampleRead)
def update_avatar_personality_long_form_sample_state(
    avatar_id: int,
    sample_id: str,
    body: AvatarPersonalityLongFormSampleStateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    state_map = avatar_svc._load_avatar_personality_long_form_states(avatar)
    state_map[str(sample_id)] = str(body.state or "included")
    avatar_svc._write_avatar_personality_long_form_states(avatar, state_map)
    page = avatar_svc._read_avatar_personality_long_form_page(session, avatar, offset=0, limit=200, state="all")
    for item in page.items:
        if item.sample_id == sample_id:
            return item
    raise HTTPException(status_code=404, detail="Long-form sample not found")


@router.get("/avatars/{avatar_id}/personality/long-form-config", response_model=AvatarPersonalityLongFormConfigRead)
def read_avatar_personality_long_form_config(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._load_avatar_personality_long_form_config(avatar)


@router.patch("/avatars/{avatar_id}/personality/long-form-config", response_model=AvatarPersonalityLongFormConfigRead)
def update_avatar_personality_long_form_config(
    avatar_id: int,
    body: AvatarPersonalityLongFormConfigUpdateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    page = avatar_svc._read_avatar_personality_long_form_page(session, avatar, offset=0, limit=1, state="all")
    return avatar_svc._write_avatar_personality_long_form_config(
        avatar,
        take_count=max(0, int(body.take_count or 0)),
        included_count=page.included_count,
        rejected_count=page.rejected_count,
        selected_count=min(max(0, int(body.take_count or 0)), page.included_count),
    )


@router.get("/avatars/{avatar_id}/personality/training-config", response_model=AvatarPersonalityTrainingConfigRead)
def read_avatar_personality_training_config(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    return avatar_svc._read_avatar_personality_training_config(session, avatar, personality)


@router.get("/avatars/{avatar_id}/personality/base-model-support", response_model=AvatarPersonalityBaseModelSupportRead)
def read_avatar_personality_base_model_support(
    avatar_id: int,
    model_id: Optional[str] = None,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    config = avatar_svc._load_avatar_personality_training_config(avatar)
    selected = str(model_id or config.base_model_id or "Qwen/Qwen3-8B").strip()
    return avatar_svc._read_avatar_base_model_support(selected)


@router.post("/avatars/{avatar_id}/personality/base-model-download", response_model=AvatarPersonalityBaseModelSupportRead)
def download_avatar_personality_base_model(
    avatar_id: int,
    body: AvatarPersonalityBaseModelDownloadRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    config = avatar_svc._load_avatar_personality_training_config(avatar)
    selected = str(body.model_id or config.base_model_id or "Qwen/Qwen3-8B").strip()
    avatar_svc._start_avatar_hf_model_download(selected)
    return avatar_svc._read_avatar_base_model_support(selected)


@router.patch("/avatars/{avatar_id}/personality/training-config", response_model=AvatarPersonalityTrainingConfigRead)
def update_avatar_personality_training_config(
    avatar_id: int,
    body: AvatarPersonalityTrainingConfigUpdateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    normalized = avatar_svc._write_avatar_personality_training_config(
        avatar,
        base_model_id=body.base_model_id,
        dataset_profile=body.dataset_profile,
        training_strength=body.training_strength,
        export_strategy=body.export_strategy,
        validation_ratio=body.validation_ratio,
        max_examples=body.max_examples,
        max_long_form_examples=body.max_long_form_examples,
        include_long_form=body.include_long_form,
        training_mode=body.training_mode,
        snapshot_interval_steps=body.snapshot_interval_steps,
    )
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    payload = normalized.model_dump()
    payload["dataset_profiles"] = [option.model_dump() for option in avatar_svc._avatar_training_dataset_profile_options()]
    payload["training_plan"] = avatar_svc._build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=normalized,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingConfigRead(**payload)


@router.get("/avatars/{avatar_id}/personality/training-package", response_model=AvatarPersonalityTrainingPackageRead)
def read_avatar_personality_training_package(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    package = avatar_svc._read_avatar_personality_training_package(avatar)
    if package.training_plan is not None:
        return package
    config = avatar_svc._load_avatar_personality_training_config(avatar)
    payload = package.model_dump()
    payload["training_plan"] = avatar_svc._build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        selected_conversation_examples=package.conversation_examples_selected if package.status == "ready" else None,
        selected_long_form_examples=package.long_form_examples_selected if package.status == "ready" else None,
        train_examples=package.train_examples if package.status == "ready" else None,
        validation_examples=package.validation_examples if package.status == "ready" else None,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingPackageRead(**payload)


@router.post("/avatars/{avatar_id}/personality/prepare-training-package", response_model=AvatarPersonalityTrainingPackageRead)
def prepare_avatar_personality_training_package(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    package = avatar_svc._prepare_avatar_personality_training_package(session, avatar, personality)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return package


@router.get("/avatars/{avatar_id}/personality/training-status", response_model=AvatarPersonalityTrainingStatusRead)
def read_avatar_personality_training_status(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = avatar_svc._reconcile_avatar_personality_training_runtime(avatar)
    resolved_snapshots = avatar_svc._avatar_resolve_training_snapshots(avatar, selected_adapter_path=status.adapter_path)
    if resolved_snapshots:
        status = avatar_svc._write_avatar_personality_training_status(
            avatar,
            {
                "snapshots": [
                    item.model_dump(mode="json")
                    for item in resolved_snapshots
                ]
            },
        )
    if status.status in {"completed", "failed", "stopped"}:
        avatar_svc._sync_avatar_personality_training_completion(int(avatar_id))
    return status


@router.post("/avatars/{avatar_id}/personality/start-training", response_model=AvatarPersonalityTrainingStatusRead)
def start_avatar_personality_training(
    avatar_id: int,
    body: AvatarPersonalityTrainRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = avatar_svc._start_avatar_personality_training(avatar, personality, body)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return status


@router.post("/avatars/{avatar_id}/personality/stop-training", response_model=AvatarPersonalityTrainingStatusRead)
def stop_avatar_personality_training(
    avatar_id: int,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = avatar_svc._reconcile_avatar_personality_training_runtime(avatar)
    if not status.active or status.status in {"idle", "completed", "failed", "stopped"}:
        return status

    status_path, stop_path = avatar_svc._avatar_personality_training_runtime_paths(avatar)
    stop_path.write_text("stop", encoding="utf-8")
    status = avatar_svc._write_avatar_personality_training_status(
        avatar,
        {
            "status": "stopping",
            "active": True,
            "stop_requested": True,
            "current_stage": "stop_requested",
            "updated_at": datetime.now(),
            "message": "Stop requested. Waiting for the current training step to finish.",
        },
    )
    return status


@router.post("/avatars/{avatar_id}/personality/promote-snapshot", response_model=AvatarPersonalityTrainingStatusRead)
def promote_avatar_personality_snapshot(
    avatar_id: int,
    body: AvatarPersonalitySnapshotSelectRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = avatar_svc._reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before promoting a snapshot.")
    selected = str(body.adapter_path or "").strip()
    snapshot = avatar_svc._avatar_find_training_snapshot(avatar, selected)
    if snapshot is None or not Path(selected).exists():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    personality.lora_adapter_path = selected
    personality.status = "trained"
    personality.updated_at = datetime.now()
    session.add(personality)
    session.commit()
    avatar_svc._avatar_release_cached_chat_model(int(avatar.id))
    return avatar_svc._avatar_update_training_snapshots(
        avatar,
        avatar_svc._avatar_resolve_training_snapshots(avatar, selected_adapter_path=selected),
        selected_adapter_path=selected,
    )


@router.post("/avatars/{avatar_id}/personality/delete-other-snapshots", response_model=AvatarPersonalityTrainingStatusRead)
def delete_other_avatar_personality_snapshots(
    avatar_id: int,
    body: AvatarPersonalitySnapshotCleanupRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = avatar_svc._reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before deleting snapshots.")
    keep_path = str(body.keep_adapter_path or "").strip()
    snapshot = avatar_svc._avatar_find_training_snapshot(avatar, keep_path)
    if snapshot is None or not Path(keep_path).exists():
        raise HTTPException(status_code=404, detail="Snapshot to keep was not found.")
    speaker = session.get(Speaker, avatar.speaker_id)
    if speaker:
        personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        personality.lora_adapter_path = keep_path
        personality.status = "trained"
        personality.updated_at = datetime.now()
        session.add(personality)
        session.commit()

    for item in avatar_svc._avatar_resolve_training_snapshots(avatar, selected_adapter_path=keep_path):
        adapter_path = str(item.adapter_path or "").strip()
        if not adapter_path or adapter_path == keep_path:
            continue
        try:
            shutil.rmtree(adapter_path, ignore_errors=True)
        except Exception:
            pass
    avatar_svc._avatar_release_cached_chat_model(int(avatar.id))
    return avatar_svc._avatar_update_training_snapshots(avatar, [snapshot], selected_adapter_path=keep_path)


@router.post("/avatars/{avatar_id}/personality/delete-snapshot", response_model=AvatarPersonalityTrainingStatusRead)
def delete_avatar_personality_snapshot(
    avatar_id: int,
    body: AvatarPersonalitySnapshotDeleteRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    status = avatar_svc._reconcile_avatar_personality_training_runtime(avatar)
    if status.active:
        raise HTTPException(status_code=400, detail="Stop training before deleting snapshots.")

    delete_path = str(body.adapter_path or "").strip()
    snapshot = avatar_svc._avatar_find_training_snapshot(avatar, delete_path)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot was not found.")

    snapshots = avatar_svc._avatar_resolve_training_snapshots(avatar, selected_adapter_path=status.adapter_path)
    remaining = [item for item in snapshots if str(item.adapter_path or "").strip() != delete_path]
    if not remaining:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining snapshot.")

    next_selected_path = str(status.adapter_path or "").strip()
    if not next_selected_path or next_selected_path == delete_path or not any(str(item.adapter_path or "").strip() == next_selected_path for item in remaining):
        fallback = next((item for item in remaining if item.kind == "final"), None) or remaining[0]
        next_selected_path = str(fallback.adapter_path or "").strip()

    speaker = session.get(Speaker, avatar.speaker_id)
    if speaker:
        personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        if str(personality.lora_adapter_path or "").strip() == delete_path:
            personality.lora_adapter_path = next_selected_path
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()

    try:
        delete_dir = Path(delete_path)
        if delete_dir.exists():
            shutil.rmtree(delete_dir, ignore_errors=True)
    except Exception:
        pass

    avatar_svc._avatar_release_cached_chat_model(int(avatar.id))
    return avatar_svc._avatar_update_training_snapshots(avatar, remaining, selected_adapter_path=next_selected_path)


_AVATAR_FIT_CHECK_PROMPTS: list[tuple[str, str]] = [
    ("casual_checkin", "How's it going?"),
    ("self_description", "Tell me about yourself in a couple of sentences."),
    ("misunderstanding", "What is something people often misunderstand about a topic you care about?"),
    ("argument", "Give me a short argument for a position you genuinely care about."),
    ("analogy", "Explain one idea using an analogy you would naturally reach for."),
    ("disagreement", "Someone says you're overthinking it and missing the point. Respond naturally."),
]


def _avatar_fit_check_reference_examples(avatar: Avatar, *, limit: int = 3) -> list[dict[str, str]]:
    state_map = avatar_svc._load_avatar_personality_state_map(avatar)
    ranked: list[tuple[tuple[int, int, int, int, int], dict[str, str]]] = []
    for row in avatar_svc._iter_avatar_personality_review_examples(avatar):
        state, _ = avatar_svc._resolve_avatar_personality_example_state(row, state_map)
        if state != "approved":
            continue
        response_text = avatar_svc._clean_avatar_dataset_text(str(row.get("response_text") or ""))
        if not response_text:
            continue
        label = avatar_svc._avatar_normalize_llm_label(str(row.get("llm_label") or row.get("heuristic_label") or row.get("auto_label") or "silver"))
        ranked.append(
            (
                (
                    1 if label == "gold" else 0,
                    int(row.get("quality_score") or 0),
                    int(row.get("style_score") or 0),
                    int(row.get("substance_score") or 0),
                    int(row.get("response_word_count") or 0),
                ),
                {
                    "video_title": str(row.get("video_title") or "").strip(),
                    "response_text": response_text[:420],
                },
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[: max(0, int(limit or 0))]]


def _avatar_run_fit_check_judge(
    *,
    speaker_name: str,
    system_prompt: str,
    results: list[AvatarPersonalityFitCheckPromptResultRead],
    reference_examples: list[dict[str, str]],
) -> tuple[str, dict[str, object]]:
    import httpx

    ollama_url, judge_model = avatar_svc._avatar_resolve_local_judge_model()
    reference_text = "\n".join(
        f"- {item.get('video_title') or 'Reference'}: {item.get('response_text') or ''}"
        for item in reference_examples
        if str(item.get("response_text") or "").strip()
    ).strip()
    result_text = "\n\n".join(
        f"[{item.key}] Prompt: {item.prompt}\nReply: {item.reply}"
        for item in results
    )
    prompt = (
        "You are evaluating whether a personality LoRA is undertrained, balanced, or overtrained.\n"
        "The goal is not factual accuracy. The goal is whether the adapter captured the speaker's voice without collapsing into memorized transcript fragments.\n\n"
        "Definitions:\n"
        "- underfit: replies are generic, weakly persona-specific, bland, or sound like the base model instead of the speaker.\n"
        "- balanced: replies answer the prompt directly, feel specific to the speaker, stay varied, and do not loop or parrot obvious transcript fragments.\n"
        "- overfit: replies repeat stock phrases, intros/outros, transcript snippets, malformed tokens, or ignore the prompt in favor of memorized patterns.\n"
        "- unclear: mixed evidence or not enough signal.\n\n"
        "Pay close attention to prompt following, repetition or looping, transcript artifacts such as <think>, reuse of the same stock phrases across prompts, and whether arguments feel natural instead of copied.\n"
        "Return JSON only with keys: classification, confidence, summary, strengths, concerns, recommendations.\n"
        "classification must be one of: underfit, balanced, overfit, unclear.\n"
        "confidence must be 0-100.\n"
        "strengths, concerns, recommendations must each be arrays of short strings.\n\n"
        f"Speaker: {speaker_name}\n"
        f"System prompt:\n{system_prompt.strip()}\n\n"
        f"Reference excerpts from the target speaker:\n{reference_text or '(none available)'}\n\n"
        f"Prompt/reply evaluation set:\n{result_text}\n"
    )
    response = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": judge_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 320,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "").strip()
    if not raw_text:
        raise ValueError("Local fit-check judge returned an empty response")
    data = avatar_svc._avatar_extract_json_object(raw_text)
    classification = str(data.get("classification") or "").strip().lower()
    if classification not in {"underfit", "balanced", "overfit", "unclear"}:
        classification = "unclear"
    def _clean_string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = avatar_svc._clean_avatar_dataset_text(str(item))
            if text:
                cleaned.append(text)
        return cleaned
    payload: dict[str, object] = {
        "classification": classification,
        "confidence": max(0, min(100, int(data.get("confidence") or 0))),
        "summary": avatar_svc._clean_avatar_dataset_text(str(data.get("summary") or "")),
        "strengths": _clean_string_list(data.get("strengths")),
        "concerns": _clean_string_list(data.get("concerns")),
        "recommendations": _clean_string_list(data.get("recommendations")),
    }
    return judge_model, payload


@router.post("/avatars/{avatar_id}/personality/test-chat", response_model=AvatarPersonalityTestChatResponse)
def test_avatar_personality_chat(
    avatar_id: int,
    body: AvatarPersonalityTestChatRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = avatar_svc._load_avatar_personality_training_status(avatar)
    requested_adapter_path = str(body.adapter_path or "").strip()
    adapter_path = requested_adapter_path or str(status.adapter_path or personality.lora_adapter_path or "").strip()
    base_model_id = str(status.base_model_id or personality.base_model_id or avatar_svc._load_avatar_personality_training_config(avatar).base_model_id).strip()
    if not adapter_path or not Path(adapter_path).exists():
        raise HTTPException(status_code=400, detail="No trained adapter is available yet.")
    if not str(body.message or "").strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    snapshot = avatar_svc._avatar_find_training_snapshot(avatar, adapter_path)
    if requested_adapter_path and snapshot is None:
        raise HTTPException(status_code=404, detail="Selected snapshot was not found.")

    try:
        system_prompt = str(personality.system_prompt or avatar_svc._default_avatar_personality_prompt(speaker.name)).strip()
        reply = avatar_svc._avatar_generate_personality_reply(
            avatar_id=int(avatar.id),
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            training_mode=str(status.training_mode or "memory_optimized"),
            system_prompt=system_prompt,
            history=[{"role": str(turn.role), "content": str(turn.content)} for turn in body.history or []],
            message=str(body.message).strip(),
            max_new_tokens=int(body.max_new_tokens or 220),
            temperature=float(body.temperature or 0.8),
            top_p=float(body.top_p or 0.9),
        )
        return AvatarPersonalityTestChatResponse(
            avatar_id=int(avatar.id),
            reply=reply,
            model=base_model_id,
            adapter_path=adapter_path,
            snapshot_label=snapshot.label if snapshot else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate test reply: {exc}") from exc


@router.post("/avatars/{avatar_id}/personality/fit-check", response_model=AvatarPersonalityFitCheckResponse)
def run_avatar_personality_fit_check(
    avatar_id: int,
    body: AvatarPersonalityFitCheckRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    status = avatar_svc._load_avatar_personality_training_status(avatar)
    requested_adapter_path = str(body.adapter_path or "").strip()
    adapter_path = requested_adapter_path or str(status.adapter_path or personality.lora_adapter_path or "").strip()
    base_model_id = str(status.base_model_id or personality.base_model_id or avatar_svc._load_avatar_personality_training_config(avatar).base_model_id).strip()
    if not adapter_path or not Path(adapter_path).exists():
        raise HTTPException(status_code=400, detail="No trained adapter is available yet.")
    snapshot = avatar_svc._avatar_find_training_snapshot(avatar, adapter_path)
    if requested_adapter_path and snapshot is None:
        raise HTTPException(status_code=404, detail="Selected snapshot was not found.")

    system_prompt = str(personality.system_prompt or avatar_svc._default_avatar_personality_prompt(speaker.name)).strip()
    results: list[AvatarPersonalityFitCheckPromptResultRead] = []
    try:
        for key, prompt in _AVATAR_FIT_CHECK_PROMPTS:
            reply = avatar_svc._avatar_generate_personality_reply(
                avatar_id=int(avatar.id),
                base_model_id=base_model_id,
                adapter_path=adapter_path,
                training_mode=str(status.training_mode or "memory_optimized"),
                system_prompt=system_prompt,
                history=[],
                message=prompt,
                max_new_tokens=int(body.max_new_tokens or 160),
                temperature=float(body.temperature or 0.75),
                top_p=float(body.top_p or 0.9),
            )
            results.append(
                AvatarPersonalityFitCheckPromptResultRead(
                    key=key,
                    prompt=prompt,
                    reply=reply,
                )
            )
        judge_model, judge_payload = _avatar_run_fit_check_judge(
            speaker_name=str(speaker.name or avatar.name or "the speaker").strip(),
            system_prompt=system_prompt,
            results=results,
            reference_examples=_avatar_fit_check_reference_examples(avatar, limit=3),
        )
        return AvatarPersonalityFitCheckResponse(
            avatar_id=int(avatar.id),
            model=base_model_id,
            judge_model=judge_model,
            adapter_path=adapter_path,
            snapshot_label=snapshot.label if snapshot else None,
            classification=str(judge_payload.get("classification") or "unclear"),
            confidence=int(judge_payload.get("confidence") or 0),
            summary=str(judge_payload.get("summary") or ""),
            strengths=[str(item) for item in judge_payload.get("strengths", []) if str(item).strip()],
            concerns=[str(item) for item in judge_payload.get("concerns", []) if str(item).strip()],
            recommendations=[str(item) for item in judge_payload.get("recommendations", []) if str(item).strip()],
            results=results,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run personality fit check: {exc}") from exc


@router.get("/avatars/{avatar_id}/personality/examples", response_model=AvatarPersonalityDatasetPageRead)
def read_avatar_personality_examples(
    avatar_id: int,
    offset: int = 0,
    limit: int = 20,
    state: str = "all",
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return avatar_svc._read_avatar_personality_dataset_page(
        avatar,
        offset=offset,
        limit=limit,
        state_filter=state,
    )


@router.patch("/avatars/{avatar_id}/personality/examples/{example_id}", response_model=AvatarPersonalityDatasetRead)
def update_avatar_personality_example_state(
    avatar_id: int,
    example_id: int,
    body: AvatarPersonalityExampleStateRequest,
    session: Session = Depends(get_session),
):
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    personality, _, _ = avatar_svc._ensure_avatar_profiles(session, avatar)

    found = False
    for row in avatar_svc._iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("example_id")) == int(example_id):
                found = True
                break
        except Exception:
            continue
    if not found:
        raise HTTPException(status_code=404, detail="Dataset example not found")

    state_map = avatar_svc._load_avatar_personality_state_map(avatar)
    next_state = str(body.state or "").strip().lower()
    if next_state not in {"approved", "rejected", "inherit"}:
        raise HTTPException(status_code=400, detail="Invalid dataset example state")
    if next_state == "inherit":
        state_map.pop(int(example_id), None)
    else:
        state_map[int(example_id)] = next_state

    avatar_svc._write_avatar_personality_state_map(avatar, state_map)
    refreshed = avatar_svc._refresh_avatar_personality_dataset_exports(avatar, personality)
    session.add(personality)
    session.commit()
    session.refresh(personality)
    return refreshed


@router.get("/avatars/{avatar_id}/personality/duplicate-group/{group_id}", response_model=List[AvatarPersonalityDatasetExampleRead])
def get_avatar_personality_duplicate_group(
    avatar_id: int,
    group_id: int,
    session: Session = Depends(get_session),
):
    """Return all dataset examples that share the same LSH duplicate group."""
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    state_map = avatar_svc._load_avatar_personality_state_map(avatar)
    items: List[AvatarPersonalityDatasetExampleRead] = []
    for row in avatar_svc._iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("duplicate_group_id") or -1) == group_id:
                state, manual_state = avatar_svc._resolve_avatar_personality_example_state(row, state_map)
                items.append(avatar_svc._row_to_example_read(row, state, manual_state))
        except Exception:
            continue
    return items


@router.get("/avatars/{avatar_id}/personality/examples/{example_id}/find-similar", response_model=SemanticSearchPage)
def find_similar_passages_for_example(
    avatar_id: int,
    example_id: int,
    limit: int = 8,
    session: Session = Depends(get_session),
):
    """Use semantic search to find passages similar to a dataset example's response text.

    Requires the semantic index to have been built for this channel.
    Excludes the source chunk(s) that directly contain this example.
    """
    avatar = session.get(Avatar, avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    target_row: Optional[dict] = None
    for row in avatar_svc._iter_avatar_personality_review_examples(avatar):
        try:
            if int(row.get("example_id")) == example_id:
                target_row = row
                break
        except Exception:
            continue
    if target_row is None:
        raise HTTPException(status_code=404, detail="Example not found")

    response_text = str(target_row.get("response_text") or "").strip()
    if not response_text:
        raise HTTPException(status_code=400, detail="Example has no response text")

    src_video_id = int(target_row.get("video_id") or 0)
    src_start = float(target_row.get("start_time") or 0.0)
    src_end = float(target_row.get("end_time") or 0.0)

    safe_limit = max(1, min(limit, 20))
    try:
        result = sem_svc.semantic_search(
            query=response_text,
            channel_id=avatar.channel_id,
            speaker_id=avatar.speaker_id,
            limit=safe_limit + 5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {exc}")

    # Exclude the chunk that directly overlaps with this example's own time range
    filtered = [
        hit for hit in result["items"]
        if not (hit["video_id"] == src_video_id and hit["start_time"] < src_end and hit["end_time"] > src_start)
    ][:safe_limit]

    return SemanticSearchPage(
        items=[SemanticSearchHit(**hit) for hit in filtered],
        total=len(filtered),
        limit=safe_limit,
        offset=0,
    )
