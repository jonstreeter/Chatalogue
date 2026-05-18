from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable, Optional, TypeVar

from sqlmodel import Session, select

from ..db.database import EpisodeChatMessage, TranscriptChunkEmbedding, TranscriptSegment, Video
from . import semantic_search as sem_svc
from .episode_clone import _normalize_provider

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about a single episode using only the provided transcript evidence. "
    "Be concise, factual, and grounded. If the transcript does not support a claim, say that clearly. "
    "Do not imply outside knowledge. Distinguish direct evidence from inference."
)

T = TypeVar("T")


def _normalize_scope_mode(value: object | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "episode_related":
        return "episode_related"
    return "episode"


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clean_answer_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _clip_text(value: object, limit: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _clip_answer_text(value: object, limit: int) -> str:
    cleaned = _clean_answer_text(value)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _format_time_label(seconds: object) -> str:
    total_seconds = max(0, int(float(seconds or 0.0)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def build_thread_title(title: object | None, *, fallback_from_message: object | None = None) -> str:
    requested = _clip_text(title, 160)
    if requested:
        return requested
    fallback = _clip_text(fallback_from_message, 72)
    return fallback or "New Chat"


def normalize_thread_request(
    *,
    title: object | None = None,
    provider_override: object | None = None,
    model_override: object | None = None,
    system_prompt: object | None = None,
    scope_mode: object | None = None,
    fallback_from_message: object | None = None,
) -> dict[str, object]:
    return {
        "title": build_thread_title(title, fallback_from_message=fallback_from_message),
        "provider_override": _normalize_provider(provider_override),
        "model_override": _clip_text(model_override, 160) or None,
        "system_prompt": _clip_text(system_prompt, 2000) or None,
        "scope_mode": _normalize_scope_mode(scope_mode),
    }


def normalize_message_request(
    *,
    message: object,
    provider_override: object | None = None,
    model_override: object | None = None,
    max_context_chunks: int = 10,
) -> dict[str, object]:
    return {
        "message": _clip_text(message, 4000),
        "provider_override": _normalize_provider(provider_override),
        "model_override": _clip_text(model_override, 160) or None,
        "max_context_chunks": max(2, min(int(max_context_chunks or 10), 10)),
    }


def _serialize_recent_history(messages: list[EpisodeChatMessage], *, limit: int = 6) -> list[dict[str, str]]:
    recent = messages[-max(1, limit) :]
    return [
        {
            "role": "assistant" if str(msg.role or "").strip().lower() == "assistant" else "user",
            "content": _clip_text(msg.content, 1200),
        }
        for msg in recent
        if _clean_text(msg.content)
    ]


def build_semantic_query(messages: list[EpisodeChatMessage], latest_message: str) -> str:
    recent_user_turns = [
        _clean_text(msg.content)
        for msg in messages
        if str(msg.role or "").strip().lower() == "user" and _clean_text(msg.content)
    ][-3:]
    parts = recent_user_turns + ([_clean_text(latest_message)] if _clean_text(latest_message) else [])
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return _clip_text(" ".join(deduped), 1200)


def _segment_citation(seg: TranscriptSegment, score: float) -> dict[str, object]:
    return {
        "chunk_id": None,
        "video_id": int(seg.video_id),
        "video_title": None,
        "citation_scope": "episode",
        "segment_ids": [int(seg.id)] if seg.id is not None else [],
        "score": round(float(score), 4),
        "speaker_name": str(getattr(getattr(seg, "speaker", None), "name", "") or "") or None,
        "start_time": float(seg.start_time),
        "end_time": float(seg.end_time),
        "support_text": _clip_text(seg.text, 420),
    }


def _chunk_citation(chunk: TranscriptChunkEmbedding, score: float) -> dict[str, object]:
    segment_ids: list[int] = []
    try:
        loaded = json.loads(chunk.segment_ids_json or "[]")
        if isinstance(loaded, list):
            segment_ids = [int(v) for v in loaded if str(v).isdigit()]
    except Exception:
        segment_ids = []
    return {
        "chunk_id": int(chunk.id) if chunk.id is not None else None,
        "video_id": int(chunk.video_id),
        "video_title": None,
        "citation_scope": "episode",
        "segment_ids": segment_ids,
        "score": round(float(score), 4),
        "speaker_name": None,
        "start_time": float(chunk.start_time),
        "end_time": float(chunk.end_time),
        "support_text": _clip_text(chunk.chunk_text, 420),
    }


def _sample_evenly(items: list[T], target_count: int) -> list[T]:
    if target_count <= 0 or not items:
        return []
    if len(items) <= target_count:
        return list(items)
    if target_count == 1:
        return [items[0]]
    indexes: list[int] = []
    last_index = len(items) - 1
    for idx in range(target_count):
        pick = round((idx * last_index) / (target_count - 1))
        if pick not in indexes:
            indexes.append(pick)
    return [items[idx] for idx in indexes]


def _looks_like_episode_wide_question(question: str) -> bool:
    normalized = _clean_text(question).lower()
    broad_tokens = [
        "top ",
        "top ten",
        "core arguments",
        "main arguments",
        "main points",
        "key points",
        "major points",
        "main themes",
        "core themes",
        "key themes",
        "takeaways",
        "biggest takeaways",
        "overview",
        "outline",
        "summarize",
        "summary",
        "what is this episode about",
        "what are the arguments",
        "what are the themes",
        "overall",
        "throughout the episode",
        "in this episode",
    ]
    return any(token in normalized for token in broad_tokens)


def _build_episode_context_map(
    session: Session,
    *,
    video: Video,
    broad_scope: bool,
) -> str:
    section_target = 10 if broad_scope else 6
    lines: list[str] = []

    summary = _clip_text(video.youtube_ai_summary or video.description or "", 1000)
    if summary:
        lines.append(f"Summary seed: {summary}")

    chunks = session.exec(
        select(TranscriptChunkEmbedding)
        .where(TranscriptChunkEmbedding.video_id == int(video.id))
        .order_by(TranscriptChunkEmbedding.start_time.asc(), TranscriptChunkEmbedding.id.asc())
    ).all()
    sampled_chunks = _sample_evenly(chunks, section_target)
    if sampled_chunks:
        lines.append("Timeline map:")
        for idx, chunk in enumerate(sampled_chunks, start=1):
            text = _clip_text(chunk.chunk_text, 260)
            if not text:
                continue
            lines.append(
                f"- [{idx}] {_format_time_label(chunk.start_time)}-{_format_time_label(chunk.end_time)}: {text}"
            )
        return "\n".join(lines).strip()

    segments = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.video_id == int(video.id))
        .order_by(TranscriptSegment.start_time.asc(), TranscriptSegment.id.asc())
    ).all()
    sampled_segments = _sample_evenly(segments, section_target)
    if sampled_segments:
        lines.append("Timeline map:")
        for idx, seg in enumerate(sampled_segments, start=1):
            text = _clip_text(seg.text, 220)
            if not text:
                continue
            lines.append(
                f"- [{idx}] {_format_time_label(seg.start_time)}-{_format_time_label(seg.end_time)}: {text}"
            )
    return "\n".join(lines).strip()


def _fallback_segment_retrieval(
    session: Session,
    *,
    video_id: int,
    query: str,
    limit: int,
) -> list[dict[str, object]]:
    segments = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
    ).all()
    if not segments:
        return []

    terms = [term for term in re.findall(r"[a-z0-9]{3,}", query.lower())]
    if not terms:
        selected = segments[:limit]
        return [_segment_citation(seg, 0.0) for seg in selected]

    scored: list[tuple[int, TranscriptSegment]] = []
    for seg in segments:
        text = str(seg.text or "").lower()
        score = sum(text.count(term) for term in terms)
        if score > 0:
            scored.append((score, seg))
    scored.sort(key=lambda item: (-item[0], float(item[1].start_time)))
    selected = [seg for _, seg in scored[:limit]]
    return [_segment_citation(seg, score=1.0) for seg in selected]


def retrieve_episode_chat_citations(
    session: Session,
    *,
    video_id: int,
    query: str,
    limit: int,
    broad_scope: bool = False,
) -> list[dict[str, object]]:
    effective_limit = max(2, min(limit, 10))
    search_limit = max(12, effective_limit * 4)
    results = sem_svc.hybrid_search(query=query, video_id=video_id, limit=search_limit)
    items = results.get("items") if isinstance(results, dict) else None
    semantic_citations: list[dict[str, object]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            semantic_citations.append(
                {
                    "chunk_id": int(item["id"]) if item.get("id") is not None else None,
                    "video_id": int(item.get("video_id") or video_id),
                    "video_title": str(item.get("video_title") or "").strip() or None,
                    "citation_scope": "episode",
                    "segment_ids": [int(v) for v in (item.get("segment_ids") or []) if str(v).isdigit()],
                    "score": round(float(item.get("score") or 0.0), 4),
                    "speaker_name": _clean_text(item.get("speaker_name")) or None,
                    "start_time": float(item.get("start_time") or 0.0),
                    "end_time": float(item.get("end_time") or 0.0),
                    "support_text": _clip_text(item.get("chunk_text"), 420),
                }
            )
    if not broad_scope:
        if semantic_citations:
            return semantic_citations[:effective_limit]
        return _fallback_segment_retrieval(session, video_id=video_id, query=query, limit=effective_limit)

    citations: list[dict[str, object]] = []
    seen_chunk_ids: set[int] = set()
    seen_segment_ids: set[int] = set()

    def _add_citation(item: dict[str, object]) -> None:
        chunk_id = item.get("chunk_id")
        if chunk_id is not None:
            try:
                parsed_chunk_id = int(chunk_id)
            except Exception:
                parsed_chunk_id = None
            if parsed_chunk_id is not None and parsed_chunk_id in seen_chunk_ids:
                return
            if parsed_chunk_id is not None:
                seen_chunk_ids.add(parsed_chunk_id)
        seg_ids = [int(v) for v in (item.get("segment_ids") or []) if str(v).isdigit()]
        if seg_ids and all(seg_id in seen_segment_ids for seg_id in seg_ids):
            return
        seen_segment_ids.update(seg_ids)
        citations.append(item)

    semantic_target = max(4, min(effective_limit // 2 + 1, effective_limit))
    for item in semantic_citations[:semantic_target]:
        _add_citation(item)
        if len(citations) >= effective_limit:
            break

    if len(citations) < effective_limit:
        chunks = session.exec(
            select(TranscriptChunkEmbedding)
            .where(TranscriptChunkEmbedding.video_id == video_id)
            .order_by(TranscriptChunkEmbedding.start_time.asc(), TranscriptChunkEmbedding.id.asc())
        ).all()
        for chunk in _sample_evenly(chunks, effective_limit):
            _add_citation(_chunk_citation(chunk, score=0.0))
            if len(citations) >= effective_limit:
                break

    if not citations:
        citations = _fallback_segment_retrieval(session, video_id=video_id, query=query, limit=effective_limit)

    citations.sort(key=lambda item: (float(item.get("start_time") or 0.0), float(item.get("end_time") or 0.0)))
    return citations[:effective_limit]


def retrieve_related_episode_chat_citations(
    session: Session,
    *,
    video: Video,
    query: str,
    limit: int,
    max_videos: int = 3,
    max_per_video: int = 2,
) -> list[dict[str, object]]:
    if video.channel_id is None or limit <= 0:
        return []
    result = sem_svc.hybrid_search(
        query=query,
        channel_id=int(video.channel_id),
        limit=max(18, min(limit * 6, 48)),
    )
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return []

    citations: list[dict[str, object]] = []
    seen_chunk_ids: set[int] = set()
    per_video_counts: dict[int, int] = {}
    selected_video_ids: list[int] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_video_id = int(item.get("video_id") or 0)
        if item_video_id <= 0 or item_video_id == int(video.id):
            continue
        chunk_id = int(item.get("id") or 0)
        if chunk_id <= 0 or chunk_id in seen_chunk_ids:
            continue
        if per_video_counts.get(item_video_id, 0) >= max_per_video:
            continue
        if item_video_id not in per_video_counts and len(selected_video_ids) >= max_videos:
            continue

        seen_chunk_ids.add(chunk_id)
        per_video_counts[item_video_id] = per_video_counts.get(item_video_id, 0) + 1
        if item_video_id not in selected_video_ids:
            selected_video_ids.append(item_video_id)
        citations.append(
            {
                "chunk_id": chunk_id,
                "video_id": item_video_id,
                "video_title": str(item.get("video_title") or "").strip() or None,
                "citation_scope": "related",
                "segment_ids": [int(v) for v in (item.get("segment_ids") or []) if str(v).isdigit()],
                "score": round(float(item.get("score") or 0.0), 4),
                "speaker_name": _clean_text(item.get("speaker_name")) or None,
                "start_time": float(item.get("start_time") or 0.0),
                "end_time": float(item.get("end_time") or 0.0),
                "support_text": _clip_text(item.get("chunk_text"), 420),
            }
        )
        if len(citations) >= limit:
            break

    return citations[:limit]


def _format_history_lines(history: list[dict[str, str]]) -> str:
    if not history:
        return "No prior conversation."
    return "\n".join(f"{entry['role'].title()}: {entry['content']}" for entry in history if entry.get("content"))


def _format_citation_lines(citations: list[dict[str, object]]) -> str:
    if not citations:
        return "No transcript evidence was found."
    lines: list[str] = []
    for idx, citation in enumerate(citations, start=1):
        speaker = _clean_text(citation.get("speaker_name")) or "Unknown speaker"
        video_title = _clean_text(citation.get("video_title"))
        start_time = float(citation.get("start_time") or 0.0)
        end_time = float(citation.get("end_time") or 0.0)
        support_text = _clean_text(citation.get("support_text")) or "(empty excerpt)"
        prefix = f"{video_title} | " if video_title else ""
        lines.append(
            f"[{idx}] {prefix}{speaker} {start_time:.2f}-{end_time:.2f}: {support_text}"
        )
    return "\n".join(lines)


def build_episode_chat_prompt(
    *,
    video: Video,
    latest_message: str,
    history: list[dict[str, str]],
    citations: list[dict[str, object]],
    related_citations: list[dict[str, object]],
    episode_context_map: str,
    scope_mode: str = "episode",
    broad_scope: bool = False,
    system_prompt: str | None = None,
) -> str:
    episode_title = _clip_text(video.title, 200)
    episode_description = _clip_text(video.description, 800)
    evidence_block = _format_citation_lines(citations)
    related_evidence_block = _format_citation_lines(related_citations)
    history_block = _format_history_lines(history)
    instructions = _clean_text(system_prompt) or DEFAULT_SYSTEM_PROMPT
    return (
        f"{instructions}\n\n"
        "Return only a JSON object with this shape:\n"
        '{'
        '"answer": "string", '
        '"citation_indexes": [1,2], '
        '"related_citation_indexes": [1,2], '
        '"grounding_note": "optional string"'
        "}\n\n"
        "Rules:\n"
        "- Base the answer only on the provided transcript evidence.\n"
        "- If the evidence is insufficient, say so in the answer.\n"
        "- `citation_indexes` must reference only the numbered evidence items you actually used.\n"
        "- `related_citation_indexes` must reference only the numbered related-evidence items you actually used.\n"
        "- Keep the answer focused on this episode.\n\n"
        "- Use light markdown when it helps readability: bullet lists, numbered lists, and short emphasis are allowed.\n"
        "- Preserve line breaks when the user asks for a list, outline, or step-by-step structure.\n\n"
        "- For whole-episode synthesis questions, use the episode-wide context map to cover the full arc of the episode, then support the answer with the transcript evidence excerpts.\n"
        "- Do not claim the episode lacks enough evidence just because the focused citations are partial if the episode-wide context map provides broader coverage.\n\n"
        "- The current episode is primary.\n"
        "- Related episodes may only provide supporting context, comparison, or recurring-theme evidence.\n"
        "- Never imply that a related-episode quote was said in the current episode.\n"
        "- If related-episode context materially informs the answer, mention that clearly.\n\n"
        f"Episode title: {episode_title}\n"
        f"Episode description: {episode_description or 'None'}\n\n"
        f"Thread scope mode: {scope_mode}\n"
        f"Whole-episode synthesis mode: {'on' if broad_scope else 'off'}\n\n"
        f"Episode-wide context map:\n{episode_context_map or 'No episode-wide context map available.'}\n\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"User question:\n{_clip_text(latest_message, 2000)}\n\n"
        f"Current episode evidence:\n{evidence_block}\n\n"
        f"Related episode evidence:\n{related_evidence_block if scope_mode == 'episode_related' else 'Not enabled for this thread.'}\n"
    )


def _extract_json_object(raw_text: str) -> dict[str, object]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    return data


def _detect_requested_answer_format(question: str) -> str:
    normalized = _clean_text(question).lower()
    if any(token in normalized for token in ["bullet point", "bullet-point", "bulleted", "bullet list", "bullet", "list of"]):
        return "bullets"
    if any(token in normalized for token in ["numbered list", "numbered", "step-by-step", "step by step", "steps"]):
        return "numbered"
    return "plain"


def _looks_like_list(text: str) -> bool:
    normalized = _clean_answer_text(text)
    return bool(re.search(r"(^|\n)\s*(?:[-*]|\d+\.)\s+\S+", normalized))


def _strip_inline_citation_markers(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"\[(?:\d{1,2})(?:\s*,\s*\d{1,2})*\]", " ", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _split_answer_items(text: str) -> list[str]:
    normalized = _clean_answer_text(_strip_inline_citation_markers(text))
    if not normalized:
        return []
    if ":" in normalized and "\n" not in normalized:
        prefix, suffix = normalized.split(":", 1)
        if len(prefix.split()) <= 8 and suffix.strip():
            normalized = suffix.strip()
    pieces = [
        part.strip(" -\t")
        for part in re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"']|\*))", normalized)
        if part.strip()
    ]
    return [piece for piece in pieces if piece]


def _apply_requested_answer_format(answer_text: str, latest_message: str) -> str:
    requested_format = _detect_requested_answer_format(latest_message)
    normalized = _clean_answer_text(_strip_inline_citation_markers(answer_text))
    if requested_format == "plain" or not normalized or _looks_like_list(normalized):
        return normalized

    items = _split_answer_items(normalized)
    if len(items) < 2:
        return normalized

    if requested_format == "numbered":
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))
    return "\n".join(f"- {item}" for item in items)


def answer_episode_chat_question(
    session: Session,
    *,
    video: Video,
    prior_messages: list[EpisodeChatMessage],
    latest_message: str,
    scope_mode: str,
    max_context_chunks: int,
    text_generator: Callable[[str], str],
    system_prompt: str | None = None,
) -> dict[str, object]:
    recent_history = _serialize_recent_history(prior_messages, limit=6)
    semantic_query = build_semantic_query(prior_messages, latest_message)
    broad_scope = _looks_like_episode_wide_question(latest_message)
    normalized_scope_mode = _normalize_scope_mode(scope_mode)
    effective_context_chunks = max_context_chunks
    if broad_scope:
        effective_context_chunks = max(effective_context_chunks, 10)
    citations = retrieve_episode_chat_citations(
        session,
        video_id=int(video.id),
        query=semantic_query or _clean_text(latest_message),
        limit=effective_context_chunks,
        broad_scope=broad_scope,
    )
    related_citations: list[dict[str, object]] = []
    if normalized_scope_mode == "episode_related":
        related_citations = retrieve_related_episode_chat_citations(
            session,
            video=video,
            query=semantic_query or _clean_text(latest_message),
            limit=6,
        )
    episode_context_map = _build_episode_context_map(
        session,
        video=video,
        broad_scope=broad_scope,
    )
    prompt = build_episode_chat_prompt(
        video=video,
        latest_message=latest_message,
        history=recent_history,
        citations=citations,
        related_citations=related_citations,
        episode_context_map=episode_context_map,
        scope_mode=normalized_scope_mode,
        broad_scope=broad_scope,
        system_prompt=system_prompt,
    )
    started_at = datetime.now()
    raw_response = text_generator(prompt)
    latency_ms = max(0, int((datetime.now() - started_at).total_seconds() * 1000))

    answer_text = _clean_answer_text(raw_response)
    selected_indexes: list[int] = []
    selected_related_indexes: list[int] = []
    grounding_note = None
    try:
        data = _extract_json_object(raw_response)
        answer_text = _clip_answer_text(data.get("answer"), 8000) or answer_text
        grounding_note = _clip_text(data.get("grounding_note"), 400) or None
        raw_indexes = data.get("citation_indexes") or []
        if isinstance(raw_indexes, list):
            for value in raw_indexes:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if 1 <= idx <= len(citations) and idx not in selected_indexes:
                    selected_indexes.append(idx)
        raw_related_indexes = data.get("related_citation_indexes") or []
        if isinstance(raw_related_indexes, list):
            for value in raw_related_indexes:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if 1 <= idx <= len(related_citations) and idx not in selected_related_indexes:
                    selected_related_indexes.append(idx)
    except Exception:
        selected_indexes = []
        selected_related_indexes = []

    answer_text = _apply_requested_answer_format(answer_text, latest_message)

    if not selected_indexes and citations:
        selected_indexes = list(range(1, min(len(citations), 2) + 1))

    selected_citations = [citations[idx - 1] for idx in selected_indexes if 1 <= idx <= len(citations)]
    selected_related_citations = [
        related_citations[idx - 1]
        for idx in selected_related_indexes
        if 1 <= idx <= len(related_citations)
    ]
    retrieved_chunk_ids = [int(item["chunk_id"]) for item in citations if item.get("chunk_id") is not None]
    retrieved_chunk_ids.extend(int(item["chunk_id"]) for item in related_citations if item.get("chunk_id") is not None)
    retrieved_segment_ids = sorted(
        {
            int(seg_id)
            for item in citations + related_citations
            for seg_id in (item.get("segment_ids") or [])
            if str(seg_id).isdigit()
        }
    )
    if grounding_note:
        answer_text = f"{answer_text}\n\nGrounding note: {grounding_note}"

    related_video_ids = sorted(
        {
            int(item.get("video_id"))
            for item in related_citations
            if str(item.get("video_id") or "").isdigit() and int(item.get("video_id") or 0) > 0
        }
    )
    retrieval_mode = normalized_scope_mode
    if broad_scope and normalized_scope_mode == "episode_related":
        retrieval_mode = "episode_related_synthesis"
    elif broad_scope:
        retrieval_mode = "episode_synthesis"

    return {
        "answer": answer_text,
        "scope_mode": normalized_scope_mode,
        "retrieval_mode": retrieval_mode,
        "semantic_query": semantic_query,
        "citations": selected_citations,
        "related_citations": selected_related_citations,
        "related_video_ids": related_video_ids,
        "used_related_context": bool(selected_related_citations),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_segment_ids": retrieved_segment_ids,
        "recent_history": recent_history,
        "prompt_version": "episode-chat-v2",
        "token_estimate": int((len(prompt) + len(answer_text)) / 4),
        "latency_ms": latency_ms,
    }
