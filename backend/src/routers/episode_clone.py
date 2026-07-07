"""AI episode cloning endpoints: candidates, engines, concepts, generation."""
import json
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db.database import Channel, Job, TranscriptSegment, Video
from ..deps import get_ingestion_service, get_session
from ..video_utils import (
    _enqueue_unique_job,
)
from ..services import episode_clone as clone_svc
from ..schemas import (
    EpisodeCloneCandidateRead,
    EpisodeCloneConceptsResponse,
    EpisodeCloneConceptsRequest,
    EpisodeCloneEngineRead,
    EpisodeCloneGenerateResponse,
    EpisodeCloneGenerateRequest,
    EpisodeCloneJobRead,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/channels/{channel_id}/episode-clone/candidates", response_model=List[EpisodeCloneCandidateRead])
def get_episode_clone_candidates(
    channel_id: int,
    limit: int = 20,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return [EpisodeCloneCandidateRead(**row) for row in clone_svc.list_clone_candidates(session, channel_id, limit=limit)]


def _get_episode_clone_engine_options() -> list[EpisodeCloneEngineRead]:
    current_provider = clone_svc._normalize_provider(os.getenv("LLM_PROVIDER") or "ollama") or "ollama"
    provider_models = {
        "ollama": (os.getenv("OLLAMA_MODEL") or "mistral").strip(),
        "nvidia_nim": (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip(),
        "openai": (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
        "anthropic": (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip(),
        "gemini": (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
        "groq": (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
        "openrouter": (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
        "xai": (os.getenv("XAI_MODEL") or "grok-2").strip(),
    }
    provider_key_env = {
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
    }
    llm_enabled = (os.getenv("LLM_ENABLED", "false").strip().lower() == "true") or (
        os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true"
    )

    def build_option(provider: str, *, is_default: bool) -> EpisodeCloneEngineRead:
        model = provider_models.get(provider) or ""
        disabled_reason = None
        available = bool(model)
        if not llm_enabled:
            available = False
            disabled_reason = "LLM is disabled in Settings."
        elif provider in provider_key_env and not (os.getenv(provider_key_env[provider]) or "").strip():
            available = False
            disabled_reason = "API key is not configured for this provider."
        elif provider == "ollama" and not model:
            available = False
            disabled_reason = "No Ollama model is configured."
        label_provider = provider.replace("_", " ").title()
        prefix = "Default" if is_default else label_provider
        return EpisodeCloneEngineRead(
            key="default" if is_default else provider,
            label=f"{prefix} ({label_provider} · {model or 'unconfigured'})",
            provider=provider,
            model=model,
            is_default=is_default,
            available=available,
            disabled_reason=disabled_reason,
        )

    options = [build_option(current_provider, is_default=True)]
    for provider in clone_svc.SUPPORTED_CLONE_PROVIDERS:
        if provider == current_provider:
            continue
        options.append(build_option(provider, is_default=False))
    return options


@router.get("/episode-clone/engines", response_model=List[EpisodeCloneEngineRead])
def list_episode_clone_engines():
    return _get_episode_clone_engine_options()


@router.post("/videos/{video_id}/episode-clone/concepts", response_model=EpisodeCloneConceptsResponse)
def detect_episode_clone_concepts(
    video_id: int,
    body: EpisodeCloneConceptsRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.channel_id:
        raise HTTPException(status_code=400, detail="Video is not attached to a channel")
    segment_exists = session.exec(
        select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id).limit(1)
    ).first()
    if segment_exists is None:
        raise HTTPException(status_code=400, detail="Source episode does not have a transcript yet")

    target_provider, target_model, target_name = get_ingestion_service().resolve_clone_llm_target(
        provider_override=body.provider_override,
        model_override=body.model_override,
    )
    try:
        result = clone_svc.extract_episode_clone_concepts(
            session,
            video_id=video_id,
            notes=body.notes,
            semantic_query=body.semantic_query,
            related_limit=body.related_limit,
            text_generator=lambda prompt: get_ingestion_service().generate_clone_text(
                prompt,
                provider_override=target_provider,
                model_override=target_model,
                temperature=0.2,
                num_predict=900,
                timeout_seconds=120,
            ),
            model_name=target_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EpisodeCloneConceptsResponse(**result)


def _build_episode_clone_job_read(job: Job) -> EpisodeCloneJobRead:
    payload: dict[str, object] = {}
    if job.payload_json:
        try:
            parsed = json.loads(job.payload_json)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

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
    request_signature = str(
        payload.get("request_signature")
        or clone_svc.clone_request_signature(video_id=int(job.video_id), request_payload=request_payload)
    )

    result = None
    raw_result = payload.get("clone_result")
    if isinstance(raw_result, dict):
        try:
            result = EpisodeCloneGenerateResponse(**raw_result)
        except Exception:
            result = None

    return EpisodeCloneJobRead(
        job_id=int(job.id),
        video_id=int(job.video_id),
        status=str(job.status or ""),
        progress=int(job.progress or 0),
        status_detail=job.status_detail,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        request=EpisodeCloneGenerateRequest(**request_payload),
        request_signature=request_signature,
        result=result,
    )


@router.post("/videos/{video_id}/episode-clone/generate", response_model=EpisodeCloneJobRead)
def queue_episode_clone_generation(
    video_id: int,
    body: EpisodeCloneGenerateRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")

    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.channel_id:
        raise HTTPException(status_code=400, detail="Video is not attached to a channel")
    segment_exists = session.exec(
        select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id).limit(1)
    ).first()
    if segment_exists is None:
        raise HTTPException(status_code=400, detail="Source episode does not have a transcript yet")

    request_payload = clone_svc.normalize_clone_request(
        style_prompt=body.style_prompt,
        notes=body.notes,
        semantic_query=body.semantic_query,
        related_limit=body.related_limit,
        variant_label=body.variant_label,
        provider_override=body.provider_override,
        model_override=body.model_override,
        approved_concepts=body.approved_concepts,
        excluded_references=body.excluded_references,
    )
    request_payload["request_signature"] = clone_svc.clone_request_signature(
        video_id=video_id,
        request_payload=request_payload,
    )
    job = _enqueue_unique_job(
        session,
        video_id=video_id,
        job_type="episode_clone",
        payload=request_payload,
    )
    return _build_episode_clone_job_read(job)


@router.get("/videos/{video_id}/episode-clone/jobs", response_model=List[EpisodeCloneJobRead])
def list_episode_clone_jobs(
    video_id: int,
    limit: int = 12,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    safe_limit = max(1, min(int(limit or 12), 50))
    jobs = session.exec(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.job_type == "episode_clone",
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(safe_limit)
    ).all()
    return [_build_episode_clone_job_read(job) for job in jobs]


@router.get("/jobs/{job_id}/episode-clone", response_model=EpisodeCloneJobRead)
def read_episode_clone_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(job.job_type or "").strip().lower() != "episode_clone":
        raise HTTPException(status_code=400, detail="Job is not an episode clone job")
    return _build_episode_clone_job_read(job)
