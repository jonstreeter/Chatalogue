"""Episode AI chat thread and message endpoints."""
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    TranscriptSegment,
    EpisodeChatMessage,
    EpisodeChatMessageContext,
    EpisodeChatThread,
    Video,
)
from ..deps import get_ingestion_service, get_session
from ..services import episode_chat as chat_svc
from ..schemas import (
    EpisodeChatCitationRead,
    EpisodeChatMessageContextRead,
    EpisodeChatMessageCreateRequest,
    EpisodeChatChannelItemRead,
    EpisodeChatMessageRead,
    EpisodeChatSendResponse,
    EpisodeChatThreadCreateRequest,
    EpisodeChatThreadDetailRead,
    EpisodeChatThreadRead,
    EpisodeChatThreadUpdateRequest,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


def _ensure_episode_chat_video_ready(session: Session, video_id: int) -> Video:
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    segment_exists = session.exec(
        select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id).limit(1)
    ).first()
    if segment_exists is None:
        raise HTTPException(status_code=400, detail="Episode does not have a transcript yet.")
    return video


def _build_episode_chat_citation_read(raw: object) -> EpisodeChatCitationRead:
    source = raw if isinstance(raw, dict) else {}
    return EpisodeChatCitationRead(
        chunk_id=int(source["chunk_id"]) if source.get("chunk_id") is not None else None,
        video_id=int(source["video_id"]) if source.get("video_id") is not None else None,
        video_title=(str(source.get("video_title") or "").strip() or None),
        citation_scope=("related" if str(source.get("citation_scope") or "").strip().lower() == "related" else "episode"),
        segment_ids=[int(v) for v in (source.get("segment_ids") or []) if str(v).isdigit()],
        score=(round(float(source.get("score") or 0.0), 4) if source.get("score") is not None else None),
        speaker_name=(str(source.get("speaker_name") or "").strip() or None),
        start_time=float(source.get("start_time") or 0.0),
        end_time=float(source.get("end_time") or 0.0),
        support_text=str(source.get("support_text") or "").strip(),
    )


def _load_episode_chat_context_map(session: Session, message_ids: list[int]) -> dict[int, EpisodeChatMessageContext]:
    if not message_ids:
        return {}
    rows = session.exec(
        select(EpisodeChatMessageContext).where(EpisodeChatMessageContext.message_id.in_(message_ids))
    ).all()
    return {int(row.message_id): row for row in rows}


def _build_episode_chat_message_read(
    message: EpisodeChatMessage,
    *,
    context: EpisodeChatMessageContext | None = None,
) -> EpisodeChatMessageRead:
    parsed_context = None
    if context:
        try:
            citations_raw = json.loads(context.citations_json or "[]")
        except Exception:
            citations_raw = []
        try:
            related_citations_raw = json.loads(context.related_citations_json or "[]")
        except Exception:
            related_citations_raw = []
        try:
            retrieved_chunk_ids = [int(v) for v in json.loads(context.retrieved_chunk_ids_json or "[]") if str(v).isdigit()]
        except Exception:
            retrieved_chunk_ids = []
        try:
            retrieved_segment_ids = [int(v) for v in json.loads(context.retrieved_segment_ids_json or "[]") if str(v).isdigit()]
        except Exception:
            retrieved_segment_ids = []
        try:
            related_video_ids = [int(v) for v in json.loads(context.related_video_ids_json or "[]") if str(v).isdigit()]
        except Exception:
            related_video_ids = []
        parsed_context = EpisodeChatMessageContextRead(
            scope_mode=("episode_related" if str(context.scope_mode or "").strip().lower() == "episode_related" else "episode"),
            retrieval_mode=str(context.retrieval_mode or "episode"),
            semantic_query=context.semantic_query,
            prompt_version=str(context.prompt_version or "episode-chat-v2"),
            citations=[
                _build_episode_chat_citation_read(item)
                for item in (citations_raw if isinstance(citations_raw, list) else [])
                if isinstance(item, dict)
            ],
            related_citations=[
                _build_episode_chat_citation_read(item)
                for item in (related_citations_raw if isinstance(related_citations_raw, list) else [])
                if isinstance(item, dict)
            ],
            related_video_ids=related_video_ids,
            used_related_context=bool(context.used_related_context),
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_segment_ids=retrieved_segment_ids,
            token_estimate=int(context.token_estimate or 0),
            latency_ms=(int(context.latency_ms) if context.latency_ms is not None else None),
        )
    return EpisodeChatMessageRead(
        id=int(message.id),
        thread_id=int(message.thread_id),
        role="assistant" if str(message.role or "").strip().lower() == "assistant" else "user",
        status=str(message.status or "completed"),
        content=str(message.content or ""),
        provider=(str(message.provider or "").strip() or None),
        model=(str(message.model or "").strip() or None),
        parent_message_id=(int(message.parent_message_id) if message.parent_message_id is not None else None),
        error=message.error,
        created_at=message.created_at,
        completed_at=message.completed_at,
        context=parsed_context,
    )


def _build_episode_chat_thread_read(
    session: Session,
    thread: EpisodeChatThread,
    *,
    message_count: int | None = None,
) -> EpisodeChatThreadRead:
    count = message_count
    if count is None:
        count = int(
            session.exec(
                select(func.count(EpisodeChatMessage.id)).where(EpisodeChatMessage.thread_id == thread.id)
            ).one()
            or 0
        )
    return EpisodeChatThreadRead(
        id=int(thread.id),
        video_id=int(thread.video_id),
        channel_id=(int(thread.channel_id) if thread.channel_id is not None else None),
        title=str(thread.title or "New Chat"),
        status=str(thread.status or "active"),
        scope_mode=("episode_related" if str(thread.scope_mode or "").strip().lower() == "episode_related" else "episode"),
        provider=(str(thread.provider or "").strip() or None),
        model=(str(thread.model or "").strip() or None),
        system_prompt=thread.system_prompt,
        message_count=int(count or 0),
        last_message_at=thread.last_message_at,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


def _build_episode_chat_thread_detail_read(session: Session, thread: EpisodeChatThread) -> EpisodeChatThreadDetailRead:
    messages = session.exec(
        select(EpisodeChatMessage)
        .where(EpisodeChatMessage.thread_id == thread.id)
        .order_by(EpisodeChatMessage.created_at.asc(), EpisodeChatMessage.id.asc())
    ).all()
    context_map = _load_episode_chat_context_map(session, [int(msg.id) for msg in messages if msg.id is not None])
    base = _build_episode_chat_thread_read(session, thread, message_count=len(messages))
    return EpisodeChatThreadDetailRead(
        **base.model_dump(),
        messages=[
            _build_episode_chat_message_read(msg, context=context_map.get(int(msg.id)))
            for msg in messages
        ],
    )


@router.post("/videos/{video_id}/episode-chat/threads", response_model=EpisodeChatThreadRead)
def create_episode_chat_thread(
    video_id: int,
    body: EpisodeChatThreadCreateRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")
    video = _ensure_episode_chat_video_ready(session, video_id)
    request_payload = chat_svc.normalize_thread_request(
        title=body.title,
        provider_override=body.provider_override,
        model_override=body.model_override,
        system_prompt=body.system_prompt,
        scope_mode=body.scope_mode,
    )
    provider, model, _target_name = get_ingestion_service().resolve_clone_llm_target(
        provider_override=str(request_payload.get("provider_override") or "") or None,
        model_override=str(request_payload.get("model_override") or "") or None,
    )
    now = datetime.now()
    thread = EpisodeChatThread(
        video_id=int(video.id),
        channel_id=(int(video.channel_id) if video.channel_id is not None else None),
        title=str(request_payload.get("title") or "New Chat"),
        status="active",
        scope_mode=str(request_payload.get("scope_mode") or "episode"),
        provider=provider,
        model=model,
        system_prompt=(str(request_payload.get("system_prompt") or "").strip() or None),
        created_at=now,
        updated_at=now,
    )
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return _build_episode_chat_thread_read(session, thread, message_count=0)


@router.get("/videos/{video_id}/episode-chat/threads", response_model=List[EpisodeChatThreadRead])
def list_episode_chat_threads(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    threads = session.exec(
        select(EpisodeChatThread)
        .where(EpisodeChatThread.video_id == video_id)
        .order_by(
            func.coalesce(EpisodeChatThread.last_message_at, EpisodeChatThread.updated_at).desc(),
            EpisodeChatThread.id.desc(),
        )
    ).all()
    thread_ids = [int(thread.id) for thread in threads if thread.id is not None]
    counts = {
        int(thread_id): int(count or 0)
        for thread_id, count in session.exec(
            select(EpisodeChatMessage.thread_id, func.count(EpisodeChatMessage.id))
            .where(EpisodeChatMessage.thread_id.in_(thread_ids or [-1]))
            .group_by(EpisodeChatMessage.thread_id)
        ).all()
    }
    return [
        _build_episode_chat_thread_read(session, thread, message_count=counts.get(int(thread.id), 0))
        for thread in threads
    ]


@router.get("/channels/{channel_id}/episode-chat", response_model=List[EpisodeChatChannelItemRead])
def list_channel_episode_chat_threads(channel_id: int, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    threads = session.exec(
        select(EpisodeChatThread)
        .where(EpisodeChatThread.channel_id == channel_id)
        .order_by(
            func.coalesce(EpisodeChatThread.last_message_at, EpisodeChatThread.updated_at).desc(),
            EpisodeChatThread.id.desc(),
        )
    ).all()
    if not threads:
        return []

    thread_ids = [int(thread.id) for thread in threads if thread.id is not None]
    thread_message_counts = {
        int(thread_id): int(count or 0)
        for thread_id, count in session.exec(
            select(EpisodeChatMessage.thread_id, func.count(EpisodeChatMessage.id))
            .where(EpisodeChatMessage.thread_id.in_(thread_ids or [-1]))
            .group_by(EpisodeChatMessage.thread_id)
        ).all()
    }

    video_ids = sorted({int(thread.video_id) for thread in threads if thread.video_id is not None})
    videos = session.exec(
        select(Video).where(Video.id.in_(video_ids or [-1]))
    ).all()
    video_map = {int(video.id): video for video in videos if video.id is not None}

    grouped: dict[int, dict[str, object]] = {}
    for thread in threads:
        video_id = int(thread.video_id)
        video = video_map.get(video_id)
        if not video:
            continue
        summary = grouped.get(video_id)
        if summary is None:
            summary = {
                "video_id": video_id,
                "channel_id": int(channel_id),
                "video_title": str(video.title or f"Video {video_id}"),
                "video_thumbnail_url": str(video.thumbnail_url or "").strip() or None,
                "video_published_at": video.published_at,
                "thread_count": 0,
                "message_count": 0,
                "latest_thread_id": int(thread.id) if thread.id is not None else None,
                "latest_thread_title": str(thread.title or "").strip() or "New Chat",
                "latest_scope_mode": ("episode_related" if str(thread.scope_mode or "").strip().lower() == "episode_related" else "episode"),
                "provider": str(thread.provider or "").strip() or None,
                "model": str(thread.model or "").strip() or None,
                "last_message_at": thread.last_message_at or thread.updated_at,
            }
            grouped[video_id] = summary
        summary["thread_count"] = int(summary["thread_count"] or 0) + 1
        summary["message_count"] = int(summary["message_count"] or 0) + thread_message_counts.get(int(thread.id), 0)

        current_last = summary.get("last_message_at")
        candidate_last = thread.last_message_at or thread.updated_at
        if current_last is None or (candidate_last is not None and candidate_last > current_last):
            summary["latest_thread_id"] = int(thread.id) if thread.id is not None else None
            summary["latest_thread_title"] = str(thread.title or "").strip() or "New Chat"
            summary["latest_scope_mode"] = ("episode_related" if str(thread.scope_mode or "").strip().lower() == "episode_related" else "episode")
            summary["provider"] = str(thread.provider or "").strip() or None
            summary["model"] = str(thread.model or "").strip() or None
            summary["last_message_at"] = candidate_last

    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            item.get("last_message_at") or datetime.min,
            item.get("video_id") or 0,
        ),
        reverse=True,
    )
    return [EpisodeChatChannelItemRead(**item) for item in ordered]


@router.get("/episode-chat/threads/{thread_id}", response_model=EpisodeChatThreadDetailRead)
def read_episode_chat_thread(thread_id: int, session: Session = Depends(get_session)):
    thread = session.get(EpisodeChatThread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Episode chat thread not found")
    return _build_episode_chat_thread_detail_read(session, thread)


@router.patch("/episode-chat/threads/{thread_id}", response_model=EpisodeChatThreadRead)
def update_episode_chat_thread(
    thread_id: int,
    body: EpisodeChatThreadUpdateRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")
    thread = session.get(EpisodeChatThread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Episode chat thread not found")
    current_provider = str(thread.provider or "").strip() or None
    current_model = str(thread.model or "").strip() or None
    request_payload = chat_svc.normalize_thread_request(
        title=body.title if body.title is not None else thread.title,
        provider_override=body.provider_override if body.provider_override is not None else current_provider,
        model_override=body.model_override if body.model_override is not None else current_model,
        system_prompt=body.system_prompt if body.system_prompt is not None else thread.system_prompt,
        scope_mode=body.scope_mode if body.scope_mode is not None else thread.scope_mode,
    )
    provider, model, _target_name = get_ingestion_service().resolve_clone_llm_target(
        provider_override=str(request_payload.get("provider_override") or "") or None,
        model_override=str(request_payload.get("model_override") or "") or None,
    )
    if body.status is not None:
        thread.status = str(body.status)
    thread.title = str(request_payload.get("title") or thread.title)
    thread.scope_mode = str(request_payload.get("scope_mode") or thread.scope_mode or "episode")
    thread.provider = provider
    thread.model = model
    thread.system_prompt = str(request_payload.get("system_prompt") or "").strip() or None
    thread.updated_at = datetime.now()
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return _build_episode_chat_thread_read(session, thread)


@router.delete("/episode-chat/threads/{thread_id}")
def delete_episode_chat_thread(thread_id: int, session: Session = Depends(get_session)):
    thread = session.get(EpisodeChatThread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Episode chat thread not found")
    messages = session.exec(
        select(EpisodeChatMessage).where(EpisodeChatMessage.thread_id == thread_id)
    ).all()
    message_ids = [int(message.id) for message in messages if message.id is not None]
    if message_ids:
        contexts = session.exec(
            select(EpisodeChatMessageContext).where(EpisodeChatMessageContext.message_id.in_(message_ids))
        ).all()
        for context in contexts:
            session.delete(context)
    for message in messages:
        session.delete(message)
    session.delete(thread)
    session.commit()
    return {"status": "deleted", "thread_id": int(thread_id)}


@router.get("/episode-chat/threads/{thread_id}/messages", response_model=List[EpisodeChatMessageRead])
def list_episode_chat_messages(thread_id: int, session: Session = Depends(get_session)):
    thread = session.get(EpisodeChatThread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Episode chat thread not found")
    messages = session.exec(
        select(EpisodeChatMessage)
        .where(EpisodeChatMessage.thread_id == thread_id)
        .order_by(EpisodeChatMessage.created_at.asc(), EpisodeChatMessage.id.asc())
    ).all()
    context_map = _load_episode_chat_context_map(session, [int(msg.id) for msg in messages if msg.id is not None])
    return [
        _build_episode_chat_message_read(msg, context=context_map.get(int(msg.id)))
        for msg in messages
    ]


@router.post("/episode-chat/threads/{thread_id}/messages", response_model=EpisodeChatSendResponse)
def send_episode_chat_message(
    thread_id: int,
    body: EpisodeChatMessageCreateRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Backend services are still starting up")
    thread = session.get(EpisodeChatThread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Episode chat thread not found")
    video = _ensure_episode_chat_video_ready(session, int(thread.video_id))

    request_payload = chat_svc.normalize_message_request(
        message=body.message,
        provider_override=body.provider_override,
        model_override=body.model_override,
        max_context_chunks=body.max_context_chunks,
    )
    message_text = str(request_payload.get("message") or "").strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    target_provider, target_model, _target_name = get_ingestion_service().resolve_clone_llm_target(
        provider_override=str(request_payload.get("provider_override") or "") or (str(thread.provider or "").strip() or None),
        model_override=str(request_payload.get("model_override") or "") or (str(thread.model or "").strip() or None),
    )
    now = datetime.now()
    prior_messages = session.exec(
        select(EpisodeChatMessage)
        .where(EpisodeChatMessage.thread_id == thread_id)
        .order_by(EpisodeChatMessage.created_at.asc(), EpisodeChatMessage.id.asc())
    ).all()

    user_message = EpisodeChatMessage(
        thread_id=int(thread.id),
        role="user",
        status="completed",
        content=message_text,
        created_at=now,
        completed_at=now,
    )
    session.add(user_message)
    session.flush()

    assistant_message = EpisodeChatMessage(
        thread_id=int(thread.id),
        role="assistant",
        status="running",
        content="",
        provider=target_provider,
        model=target_model,
        parent_message_id=int(user_message.id),
        created_at=now,
    )
    session.add(assistant_message)
    session.flush()

    thread.last_message_at = now
    thread.updated_at = now
    thread.provider = target_provider
    thread.model = target_model
    if str(thread.title or "").strip().lower() == "new chat":
        thread.title = chat_svc.build_thread_title(thread.title, fallback_from_message=message_text)
    session.add(thread)
    session.commit()
    session.refresh(user_message)
    session.refresh(assistant_message)
    session.refresh(thread)

    try:
        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=prior_messages + [user_message],
            latest_message=message_text,
            scope_mode=str(thread.scope_mode or "episode"),
            max_context_chunks=int(request_payload.get("max_context_chunks") or 10),
            text_generator=lambda prompt: get_ingestion_service().generate_clone_text(
                prompt,
                provider_override=target_provider,
                model_override=target_model,
                temperature=0.2,
                num_predict=700,
                timeout_seconds=120,
            ),
            system_prompt=thread.system_prompt,
        )
        assistant_message.status = "completed"
        assistant_message.content = str(answer.get("answer") or "").strip()
        assistant_message.provider = target_provider
        assistant_message.model = target_model
        assistant_message.completed_at = datetime.now()
        session.add(assistant_message)
        session.flush()
        session.add(
            EpisodeChatMessageContext(
                message_id=int(assistant_message.id),
                video_id=int(video.id),
                scope_mode=str(answer.get("scope_mode") or thread.scope_mode or "episode"),
                retrieval_mode=str(answer.get("retrieval_mode") or "episode"),
                semantic_query=str(answer.get("semantic_query") or "").strip() or None,
                recent_history_json=json.dumps(answer.get("recent_history") or [], ensure_ascii=False),
                citations_json=json.dumps(answer.get("citations") or [], ensure_ascii=False),
                related_citations_json=json.dumps(answer.get("related_citations") or [], ensure_ascii=False),
                retrieved_chunk_ids_json=json.dumps(answer.get("retrieved_chunk_ids") or []),
                retrieved_segment_ids_json=json.dumps(answer.get("retrieved_segment_ids") or []),
                related_video_ids_json=json.dumps(answer.get("related_video_ids") or []),
                used_related_context=bool(answer.get("used_related_context")),
                prompt_version=str(answer.get("prompt_version") or "episode-chat-v2"),
                token_estimate=int(answer.get("token_estimate") or 0),
                latency_ms=(int(answer.get("latency_ms")) if answer.get("latency_ms") is not None else None),
            )
        )
    except Exception as exc:
        assistant_message.status = "failed"
        assistant_message.error = str(exc)[:500]
        assistant_message.content = ""
        assistant_message.provider = target_provider
        assistant_message.model = target_model
        assistant_message.completed_at = datetime.now()
        session.add(assistant_message)
    thread.last_message_at = datetime.now()
    thread.updated_at = datetime.now()
    session.add(thread)
    session.commit()
    session.refresh(thread)
    session.refresh(user_message)
    session.refresh(assistant_message)
    context_map = _load_episode_chat_context_map(
        session,
        [int(assistant_message.id)] if assistant_message.id is not None else [],
    )
    return EpisodeChatSendResponse(
        thread=_build_episode_chat_thread_read(session, thread),
        user_message=_build_episode_chat_message_read(user_message),
        assistant_message=_build_episode_chat_message_read(
            assistant_message,
            context=context_map.get(int(assistant_message.id)) if assistant_message.id is not None else None,
        ),
    )


@router.get("/episode-chat/messages/{message_id}", response_model=EpisodeChatMessageRead)
def read_episode_chat_message(message_id: int, session: Session = Depends(get_session)):
    message = session.get(EpisodeChatMessage, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Episode chat message not found")
    context = session.exec(
        select(EpisodeChatMessageContext).where(EpisodeChatMessageContext.message_id == message_id)
    ).first()
    return _build_episode_chat_message_read(message, context=context)

