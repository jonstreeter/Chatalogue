from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from typing import Callable, Optional

from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import Channel, TranscriptChunkEmbedding, TranscriptSegment, Video

SUPPORTED_CLONE_PROVIDERS = (
    "ollama",
    "nvidia_nim",
    "openai",
    "anthropic",
    "gemini",
    "groq",
    "openrouter",
    "xai",
)

_SOURCE_CONTAMINATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("first-person callback", re.compile(r"\b(?:i|we)\s+(?:mentioned|said|covered|talked about|noted|shared|explained)\b", re.IGNORECASE)),
    ("episode callback", re.compile(r"\b(?:earlier|before|previously|last time|in another episode|in the last episode)\b", re.IGNORECASE)),
    ("Q&A callback", re.compile(r"\bq\s*&\s*a\b|\bqanda\b|\bqa session\b", re.IGNORECASE)),
    ("channel self-reference", re.compile(r"\b(?:this|my|our)\s+channel\b", re.IGNORECASE)),
    ("audience self-reference", re.compile(r"\b(?:my|our)\s+(?:audience|subscribers|viewers|community)\b", re.IGNORECASE)),
]


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clip_text(value: str, limit: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _normalize_provider(provider: object) -> str | None:
    value = str(provider or "").strip().lower()
    if not value:
        return None
    aliases = {
        "nvidia": "nvidia_nim",
        "nim": "nvidia_nim",
        "nvidia-nim": "nvidia_nim",
        "chatgpt": "openai",
        "claude": "anthropic",
        "google": "gemini",
        "google-gemini": "gemini",
        "google_gemini": "gemini",
    }
    normalized = aliases.get(value, value)
    return normalized if normalized in SUPPORTED_CLONE_PROVIDERS else None


def _normalize_text_list(values: object, *, limit: int, item_limit: int) -> list[str]:
    if values is None:
        return []
    raw_items = values if isinstance(values, list) else [values]
    cleaned_items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = _clip_text(_clean_text(item), item_limit)
        if not cleaned:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned_items.append(cleaned)
        if len(cleaned_items) >= limit:
            break
    return cleaned_items


def normalize_clone_request(
    *,
    style_prompt: str,
    notes: Optional[str] = None,
    semantic_query: Optional[str] = None,
    related_limit: int = 8,
    variant_label: Optional[str] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    approved_concepts: Optional[list[str]] = None,
    excluded_references: Optional[list[str]] = None,
) -> dict[str, object]:
    return {
        "style_prompt": str(style_prompt or "").strip(),
        "notes": _clean_text(notes) or None,
        "semantic_query": _clean_text(semantic_query) or None,
        "related_limit": max(1, min(int(related_limit or 8), 12)),
        "variant_label": _clip_text(str(variant_label or "").strip(), 120) or None,
        "provider_override": _normalize_provider(provider_override),
        "model_override": _clip_text(str(model_override or "").strip(), 160) or None,
        "approved_concepts": _normalize_text_list(approved_concepts, limit=18, item_limit=220),
        "excluded_references": _normalize_text_list(excluded_references, limit=24, item_limit=180),
    }


def clone_request_signature(*, video_id: int, request_payload: dict[str, object]) -> str:
    canonical = {
        "video_id": int(video_id),
        "style_prompt": str(request_payload.get("style_prompt") or "").strip(),
        "notes": _clean_text(request_payload.get("notes")) or "",
        "semantic_query": _clean_text(request_payload.get("semantic_query")) or "",
        "related_limit": max(1, min(int(request_payload.get("related_limit") or 8), 12)),
        "variant_label": _clean_text(request_payload.get("variant_label")) or "",
        "provider_override": _normalize_provider(request_payload.get("provider_override")) or "",
        "model_override": _clean_text(request_payload.get("model_override")) or "",
        "approved_concepts": _normalize_text_list(request_payload.get("approved_concepts"), limit=18, item_limit=220),
        "excluded_references": _normalize_text_list(request_payload.get("excluded_references"), limit=24, item_limit=180),
    }
    encoded = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _normalize_overlap_text(value: object) -> str:
    cleaned = _clean_text(value).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize_overlap_text(value: object) -> list[str]:
    normalized = _normalize_overlap_text(value)
    return [token for token in normalized.split(" ") if token]


def _sentence_fingerprints(value: object, *, min_words: int = 8) -> set[str]:
    fingerprints: set[str] = set()
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", str(value or "")):
        words = _tokenize_overlap_text(chunk)
        if len(words) >= min_words:
            fingerprints.add(" ".join(words))
    return fingerprints


def _ngram_fingerprints(words: list[str], *, size: int = 5) -> set[str]:
    if len(words) < size:
        return set()
    return {" ".join(words[idx : idx + size]) for idx in range(0, len(words) - size + 1)}


def _evaluate_originality_overlap(script: str, references: list[tuple[str, str]]) -> dict[str, object]:
    script_words = _tokenize_overlap_text(script)
    if len(script_words) < 12:
        return {
            "top_reference": None,
            "max_ngram_overlap": 0.0,
            "copied_sentence_count": 0,
            "copied_sentence_reference": None,
        }

    script_ngrams = _ngram_fingerprints(script_words, size=5)
    script_sentences = _sentence_fingerprints(script)
    top_reference: str | None = None
    copied_sentence_reference: str | None = None
    max_ngram_overlap = 0.0
    copied_sentence_count = 0

    for label, text in references:
        ref_words = _tokenize_overlap_text(text)
        if len(ref_words) < 5:
            continue
        ref_ngrams = _ngram_fingerprints(ref_words, size=5)
        if script_ngrams and ref_ngrams:
            overlap_ratio = len(script_ngrams & ref_ngrams) / max(1, min(len(script_ngrams), len(ref_ngrams)))
            if overlap_ratio > max_ngram_overlap:
                max_ngram_overlap = overlap_ratio
                top_reference = label

        shared_sentences = len(script_sentences & _sentence_fingerprints(text))
        if shared_sentences > copied_sentence_count:
            copied_sentence_count = shared_sentences
            copied_sentence_reference = label

    return {
        "top_reference": top_reference,
        "max_ngram_overlap": round(max_ngram_overlap, 4),
        "copied_sentence_count": copied_sentence_count,
        "copied_sentence_reference": copied_sentence_reference,
    }


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


def _video_age_days(video: Video) -> Optional[float]:
    if not video or not video.published_at:
        return None
    delta = datetime.now() - video.published_at
    return max(1.0, round(delta.total_seconds() / 86400.0, 2))


def _views_per_day(video: Video) -> Optional[float]:
    if video.view_count is None:
        return None
    age_days = _video_age_days(video)
    if not age_days:
        return None
    return round(float(video.view_count) / max(1.0, age_days), 2)


def _load_channel_segment_counts(session: Session, channel_id: int) -> dict[int, int]:
    rows = session.exec(
        select(TranscriptSegment.video_id, func.count(TranscriptSegment.id))
        .join(Video, Video.id == TranscriptSegment.video_id)
        .where(Video.channel_id == channel_id)
        .group_by(TranscriptSegment.video_id)
    ).all()
    return {int(video_id): int(count or 0) for video_id, count in rows}


def _candidate_payload(
    video: Video,
    *,
    transcript_segment_count: int = 0,
    semantic_hit_count: int = 0,
    semantic_score: Optional[float] = None,
) -> dict[str, object]:
    age_days = _video_age_days(video)
    views_per_day = _views_per_day(video)
    return {
        "video_id": int(video.id),
        "title": str(video.title or f"Video {video.id}"),
        "published_at": video.published_at,
        "duration": int(video.duration) if video.duration is not None else None,
        "view_count": int(video.view_count) if video.view_count is not None else None,
        "age_days": age_days,
        "views_per_day": views_per_day,
        "transcript_segment_count": int(transcript_segment_count or 0),
        "semantic_hit_count": int(semantic_hit_count or 0),
        "semantic_score": round(float(semantic_score), 4) if semantic_score is not None else None,
    }


def list_clone_candidates(session: Session, channel_id: int, *, limit: int = 20) -> list[dict[str, object]]:
    safe_limit = max(1, min(int(limit or 20), 100))
    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .order_by(Video.published_at.desc(), Video.id.desc())
    ).all()
    segment_counts = _load_channel_segment_counts(session, channel_id)

    candidates: list[dict[str, object]] = []
    for video in videos:
        seg_count = int(segment_counts.get(int(video.id), 0))
        if seg_count <= 0:
            continue
        candidates.append(_candidate_payload(video, transcript_segment_count=seg_count))

    candidates.sort(
        key=lambda item: (
            item.get("views_per_day") is not None,
            float(item.get("views_per_day") or -1.0),
            int(item.get("view_count") or -1),
            item.get("published_at") or datetime.min,
            int(item.get("video_id") or 0),
        ),
        reverse=True,
    )
    return candidates[:safe_limit]


def _load_source_excerpt(session: Session, video_id: int, *, max_chars: int = 4200) -> str:
    parts: list[str] = []

    chunks = session.exec(
        select(TranscriptChunkEmbedding)
        .where(TranscriptChunkEmbedding.video_id == video_id)
        .order_by(TranscriptChunkEmbedding.start_time.asc(), TranscriptChunkEmbedding.id.asc())
    ).all()
    for chunk in chunks:
        text = _clean_text(chunk.chunk_text)
        if not text:
            continue
        parts.append(text)
        if len(" ".join(parts)) >= max_chars:
            break

    if not parts:
        segments = session.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.video_id == video_id)
            .order_by(TranscriptSegment.start_time.asc(), TranscriptSegment.id.asc())
        ).all()
        for segment in segments:
            text = _clean_text(segment.text)
            if not text:
                continue
            parts.append(text)
            if len(" ".join(parts)) >= max_chars:
                break

    return _clip_text(" ".join(parts), max_chars)


def _build_source_brief(session: Session, video: Video) -> str:
    brief_parts: list[str] = []
    if video.youtube_ai_summary:
        brief_parts.append(_clip_text(video.youtube_ai_summary, 1200))
    elif video.description:
        brief_parts.append(_clip_text(video.description, 900))

    excerpt = _load_source_excerpt(session, int(video.id), max_chars=3600)
    if excerpt:
        brief_parts.append(f"Transcript excerpt: {excerpt}")

    return "\n\n".join(part for part in brief_parts if part).strip()


def _derive_semantic_query(video: Video, source_brief: str) -> str:
    parts = [
        _clean_text(video.title),
        _clip_text(video.youtube_ai_summary or "", 280),
        _clip_text(video.description or "", 220),
        _clip_text(source_brief, 420),
    ]
    return _clip_text(" | ".join(part for part in parts if part), 900)


def _default_excluded_references(
    *,
    channel_name: str,
    source_video: Video,
    related_videos: list[dict[str, object]],
) -> list[str]:
    values = [
        channel_name,
        source_video.title,
        "Q&A session",
        "this channel",
        "last episode",
        "previous episode",
    ]
    values.extend(str(item.get("title") or "") for item in related_videos[:6])
    return _normalize_text_list(values, limit=24, item_limit=180)


def _collect_related_context(
    session: Session,
    *,
    video: Video,
    semantic_query: str,
    related_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    from . import semantic_search as sem_svc

    warnings: list[str] = []
    indexed_rows = session.exec(
        select(func.count(TranscriptChunkEmbedding.id))
        .where(TranscriptChunkEmbedding.channel_id == video.channel_id)
    ).first()
    if not int(indexed_rows or 0):
        warnings.append("Semantic index is not available for this channel yet. Rebuild it for richer cross-episode context.")
        return [], [], warnings

    result = sem_svc.hybrid_search(
        query=semantic_query,
        channel_id=int(video.channel_id),
        limit=max(24, int(related_limit or 8) * 6),
        offset=0,
    )

    filtered_hits: list[dict[str, object]] = []
    seen_chunk_ids: set[int] = set()
    for item in result.get("items") or []:
        item_video_id = int(item.get("video_id") or 0)
        chunk_id = int(item.get("id") or 0)
        if item_video_id == int(video.id) or chunk_id <= 0 or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        filtered_hits.append(
            {
                "chunk_id": chunk_id,
                "score": round(float(item.get("score") or 0.0), 4),
                "video_id": item_video_id,
                "video_title": item.get("video_title"),
                "speaker_name": item.get("speaker_name"),
                "start_time": float(item.get("start_time") or 0.0),
                "end_time": float(item.get("end_time") or 0.0),
                "chunk_text": _clip_text(str(item.get("chunk_text") or ""), 600),
            }
        )
        if len(filtered_hits) >= max(10, int(related_limit or 8) * 2):
            break

    if not filtered_hits:
        warnings.append("Semantic search did not find strong cross-episode matches for this source episode.")
        return [], [], warnings

    by_video: dict[int, dict[str, object]] = {}
    video_ids = sorted({int(hit["video_id"]) for hit in filtered_hits})
    videos = session.exec(select(Video).where(Video.id.in_(video_ids))).all() if video_ids else []
    video_map = {int(item.id): item for item in videos}
    segment_counts = _load_channel_segment_counts(session, int(video.channel_id))

    for hit in filtered_hits:
        vid = int(hit["video_id"])
        bucket = by_video.setdefault(
            vid,
            {
                "semantic_hit_count": 0,
                "semantic_score": 0.0,
            },
        )
        bucket["semantic_hit_count"] = int(bucket["semantic_hit_count"]) + 1
        bucket["semantic_score"] = max(float(bucket["semantic_score"]), float(hit["score"]))

    related_videos: list[dict[str, object]] = []
    for vid, bucket in by_video.items():
        video_row = video_map.get(vid)
        if not video_row:
            continue
        related_videos.append(
            _candidate_payload(
                video_row,
                transcript_segment_count=int(segment_counts.get(vid, 0)),
                semantic_hit_count=int(bucket["semantic_hit_count"]),
                semantic_score=float(bucket["semantic_score"]),
            )
        )

    related_videos.sort(
        key=lambda item: (
            float(item.get("semantic_score") or 0.0),
            int(item.get("semantic_hit_count") or 0),
            float(item.get("views_per_day") or -1.0),
            int(item.get("view_count") or -1),
        ),
        reverse=True,
    )
    return related_videos[: max(1, min(int(related_limit or 8), 12))], filtered_hits, warnings


def _build_concept_prompt(
    *,
    channel_name: str,
    source_video: Video,
    source_metrics: dict[str, object],
    source_brief: str,
    notes: str,
    semantic_query: str,
    related_videos: list[dict[str, object]],
    context_hits: list[dict[str, object]],
) -> str:
    related_lines: list[str] = []
    for item in related_videos:
        related_lines.append(
            "- "
            f"{item.get('title')}"
            f" | views={item.get('view_count') if item.get('view_count') is not None else 'unknown'}"
            f" | views/day={item.get('views_per_day') if item.get('views_per_day') is not None else 'unknown'}"
            f" | semantic_score={item.get('semantic_score') if item.get('semantic_score') is not None else 'n/a'}"
        )

    context_lines: list[str] = []
    for hit in context_hits[:10]:
        fallback_title = str(hit.get("video_title") or f"Video {hit.get('video_id')}")
        context_lines.append(
            "- "
            f"{fallback_title}"
            f" @ {float(hit.get('start_time') or 0.0):.1f}-{float(hit.get('end_time') or 0.0):.1f}s"
            f": {hit.get('chunk_text') or ''}"
        )

    return (
        "You are extracting PORTABLE episode concepts from a source episode plus semantically related episodes.\n"
        "The goal is to isolate reusable ideas while stripping away source-specific framing.\n\n"
        "Rules:\n"
        "- Concepts must be reusable by a completely different creator.\n"
        "- Do not mention the source channel, source episode title, other episode titles, or channel-internal callbacks.\n"
        "- Do not keep first-person references like 'I said before' or 'on this channel'.\n"
        "- Ignore tangents, housekeeping, sponsorships, greetings, audience management, and intra-channel references unless central to the topic.\n"
        "- Prefer generalized framing, claims, tensions, and narrative angles.\n\n"
        "Return JSON only with keys:\n"
        "concepts, excluded_references, warnings\n"
        "- concepts: array of 4-10 short strings describing major reusable concepts\n"
        "- excluded_references: array of source-specific references or phrases that must NOT appear in the final clone\n"
        "- warnings: array of short strings\n\n"
        f"Channel: {channel_name}\n"
        f"Source episode title: {source_video.title}\n"
        f"Source episode views: {source_metrics.get('view_count') if source_metrics.get('view_count') is not None else 'unknown'}\n"
        f"Source episode views/day: {source_metrics.get('views_per_day') if source_metrics.get('views_per_day') is not None else 'unknown'}\n"
        f"Semantic retrieval query: {semantic_query}\n\n"
        f"Additional user notes for concept filtering:\n{notes or '(none)'}\n\n"
        f"Source episode brief:\n{source_brief or '(no source brief available)'}\n\n"
        f"Related channel episodes:\n{chr(10).join(related_lines) if related_lines else '(no strong related episodes found)'}\n\n"
        f"Related semantic context hits:\n{chr(10).join(context_lines) if context_lines else '(none)'}\n"
    )


def _build_clone_prompt(
    *,
    channel_name: str,
    source_video: Video,
    source_metrics: dict[str, object],
    source_brief: str,
    style_prompt: str,
    notes: str,
    semantic_query: str,
    approved_concepts: list[str],
    excluded_references: list[str],
    related_videos: list[dict[str, object]],
    context_hits: list[dict[str, object]],
) -> str:
    concept_lines = "\n".join(f"- {item}" for item in approved_concepts) if approved_concepts else "- (none supplied)"
    excluded_lines = "\n".join(f"- {item}" for item in excluded_references) if excluded_references else "- (none supplied)"
    related_lines = "\n".join(
        "- "
        f"{item.get('title')} | views={item.get('view_count') if item.get('view_count') is not None else 'unknown'}"
        f" | views/day={item.get('views_per_day') if item.get('views_per_day') is not None else 'unknown'}"
        f" | semantic_score={item.get('semantic_score') if item.get('semantic_score') is not None else 'n/a'}"
        for item in related_videos[:8]
    ) or "- (none supplied)"
    context_lines_list: list[str] = []
    for hit in context_hits[:8]:
        fallback_title = str(hit.get("video_title") or f"Video {hit.get('video_id')}")
        context_lines_list.append(
            "- "
            f"{fallback_title}"
            f" @ {float(hit.get('start_time') or 0.0):.1f}-{float(hit.get('end_time') or 0.0):.1f}s"
            f": {hit.get('chunk_text') or ''}"
        )
    context_lines = "\n".join(context_lines_list) or "- (none supplied)"

    return (
        "You are drafting a NEW original episode script from a sanitized concept brief plus reference evidence.\n"
        "The job is to create independent content, not to sound like the source creator.\n\n"
        "Rules:\n"
        "- Treat the approved concepts below as the canonical brief.\n"
        "- Use the reference evidence only to add specificity, nuance, and supporting detail when it clearly reinforces an approved concept.\n"
        "- If the evidence includes tangents, callbacks, or creator-specific framing, discard them.\n"
        "- Do not mention the source creator, source channel, source episode, other episodes, Q&A sessions, or prior conversations.\n"
        "- Do not write first-person callbacks such as 'I said before', 'as we mentioned', or 'on this channel'.\n"
        "- Do not copy distinctive phrasing or structure from any source material.\n"
        "- If a concept implies channel-specific history, rewrite it into a general claim or leave it out.\n"
        "- Never cite or refer to the source evidence directly in the final script.\n"
        "- Follow the user's requested style and format exactly.\n"
        "- Keep the result ready to record.\n\n"
        "Return JSON only with keys:\n"
        "suggested_title, opening_hook, angle_summary, originality_notes, script\n"
        "originality_notes must be an array of short strings.\n\n"
        f"Source channel: {channel_name}\n"
        f"Source episode title: {source_video.title}\n"
        f"Source episode views: {source_metrics.get('view_count') if source_metrics.get('view_count') is not None else 'unknown'}\n"
        f"Source episode views/day: {source_metrics.get('views_per_day') if source_metrics.get('views_per_day') is not None else 'unknown'}\n"
        f"Semantic retrieval query: {semantic_query}\n\n"
        f"Approved concepts:\n{concept_lines}\n\n"
        f"Forbidden references / phrases:\n{excluded_lines}\n\n"
        f"Source episode evidence brief:\n{source_brief or '- (none supplied)'}\n\n"
        f"Related episode evidence:\n{related_lines}\n\n"
        f"Related semantic transcript evidence:\n{context_lines}\n\n"
        f"Requested target style:\n{style_prompt.strip()}\n\n"
        f"Additional user notes:\n{notes or '(none)'}\n"
    )


def _normalize_clone_output(raw_text: str) -> tuple[dict[str, object], bool]:
    cleaned_raw = str(raw_text or "").strip()
    try:
        data = _extract_json_object(cleaned_raw)
    except Exception:
        return ({
            "suggested_title": "",
            "opening_hook": "",
            "angle_summary": "",
            "originality_notes": [],
            "script": "",
        }, False)

    notes = data.get("originality_notes")
    if not isinstance(notes, list):
        notes = []

    return ({
        "suggested_title": _clean_text(data.get("suggested_title")),
        "opening_hook": _clean_text(data.get("opening_hook")),
        "angle_summary": _clean_text(data.get("angle_summary")),
        "originality_notes": [_clean_text(item) for item in notes if _clean_text(item)],
        "script": str(data.get("script") or "").strip(),
    }, True)


def _normalize_concept_output(raw_text: str) -> tuple[dict[str, object], bool]:
    cleaned_raw = str(raw_text or "").strip()
    try:
        data = _extract_json_object(cleaned_raw)
    except Exception:
        return ({
            "concepts": [],
            "excluded_references": [],
            "warnings": ["The model did not return structured concept JSON."],
        }, False)

    return ({
        "concepts": _normalize_text_list(data.get("concepts"), limit=10, item_limit=220),
        "excluded_references": _normalize_text_list(data.get("excluded_references"), limit=24, item_limit=180),
        "warnings": _normalize_text_list(data.get("warnings"), limit=8, item_limit=180),
    }, True)


def _evaluate_source_contamination(
    script: str,
    *,
    channel_name: str,
    source_title: str,
    excluded_references: list[str],
) -> dict[str, object]:
    findings: list[str] = []
    normalized_script = _normalize_overlap_text(script)
    literal_matches: list[str] = []

    for ref in _normalize_text_list(
        [channel_name, source_title, *excluded_references],
        limit=32,
        item_limit=180,
    ):
        normalized_ref = _normalize_overlap_text(ref)
        if len(normalized_ref) < 5:
            continue
        if normalized_ref in normalized_script:
            literal_matches.append(ref)

    for label, pattern in _SOURCE_CONTAMINATION_PATTERNS:
        if pattern.search(script):
            findings.append(label)

    if literal_matches:
        findings.append("source-specific reference")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in findings:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return {
        "findings": deduped,
        "literal_matches": literal_matches[:8],
        "should_warn": bool(deduped),
        "should_suppress": bool(literal_matches) or len(deduped) >= 2,
    }


def extract_episode_clone_concepts(
    session: Session,
    *,
    video_id: int,
    notes: Optional[str] = None,
    semantic_query: Optional[str] = None,
    related_limit: int = 8,
    text_generator: Callable[[str], str],
    model_name: str,
) -> dict[str, object]:
    video = session.get(Video, video_id)
    if not video:
        raise ValueError("Video not found")
    if not video.channel_id:
        raise ValueError("Video is not attached to a channel")

    channel = session.get(Channel, int(video.channel_id))
    if not channel:
        raise ValueError("Channel not found")

    segment_counts = _load_channel_segment_counts(session, int(video.channel_id))
    source_seg_count = int(segment_counts.get(int(video.id), 0))
    if source_seg_count <= 0:
        raise ValueError("Source episode does not have a transcript yet")

    source_metrics = _candidate_payload(video, transcript_segment_count=source_seg_count)
    source_brief = _build_source_brief(session, video)
    effective_query = _clean_text(semantic_query) or _derive_semantic_query(video, source_brief)
    related_videos, context_hits, warnings = _collect_related_context(
        session,
        video=video,
        semantic_query=effective_query,
        related_limit=related_limit,
    )

    prompt = _build_concept_prompt(
        channel_name=str(channel.name or f"Channel {channel.id}"),
        source_video=video,
        source_metrics=source_metrics,
        source_brief=source_brief,
        notes=str(notes or "").strip(),
        semantic_query=effective_query,
        related_videos=related_videos,
        context_hits=context_hits,
    )
    payload, parsed_ok = _normalize_concept_output(text_generator(prompt))
    concepts = _normalize_text_list(payload.get("concepts"), limit=10, item_limit=220)
    excluded_references = _normalize_text_list(
        [
            *list(payload.get("excluded_references") or []),
            *_default_excluded_references(
                channel_name=str(channel.name or f"Channel {channel.id}"),
                source_video=video,
                related_videos=related_videos,
            ),
        ],
        limit=24,
        item_limit=180,
    )
    concept_warnings = _normalize_text_list(payload.get("warnings"), limit=8, item_limit=180)

    if not parsed_ok:
        warnings.append("Theme detection returned unstructured output. Review extracted concepts carefully.")
    if not concepts:
        warnings.append("No clear reusable concepts were extracted from this episode.")

    return {
        "video_id": int(video.id),
        "channel_id": int(channel.id),
        "source_title": str(video.title or f"Video {video.id}"),
        "source_metrics": source_metrics,
        "semantic_query": effective_query,
        "source_brief": source_brief,
        "concepts": concepts,
        "excluded_references": excluded_references,
        "related_videos": related_videos,
        "warnings": _normalize_text_list([*warnings, *concept_warnings], limit=12, item_limit=180),
        "model": str(model_name or "").strip(),
    }


def generate_episode_clone(
    session: Session,
    *,
    video_id: int,
    style_prompt: str,
    notes: Optional[str] = None,
    semantic_query: Optional[str] = None,
    related_limit: int = 8,
    approved_concepts: Optional[list[str]] = None,
    excluded_references: Optional[list[str]] = None,
    text_generator: Callable[[str], str],
    model_name: str,
) -> dict[str, object]:
    style_prompt = str(style_prompt or "").strip()
    if not style_prompt:
        raise ValueError("style_prompt is required")

    video = session.get(Video, video_id)
    if not video:
        raise ValueError("Video not found")
    if not video.channel_id:
        raise ValueError("Video is not attached to a channel")

    channel = session.get(Channel, int(video.channel_id))
    if not channel:
        raise ValueError("Channel not found")

    segment_counts = _load_channel_segment_counts(session, int(video.channel_id))
    source_seg_count = int(segment_counts.get(int(video.id), 0))
    if source_seg_count <= 0:
        raise ValueError("Source episode does not have a transcript yet")

    source_metrics = _candidate_payload(video, transcript_segment_count=source_seg_count)
    source_brief = _build_source_brief(session, video)
    effective_query = _clean_text(semantic_query) or _derive_semantic_query(video, source_brief)
    related_videos, context_hits, warnings = _collect_related_context(
        session,
        video=video,
        semantic_query=effective_query,
        related_limit=related_limit,
    )
    approved_concepts = _normalize_text_list(approved_concepts, limit=18, item_limit=220)
    excluded_references = _normalize_text_list(excluded_references, limit=24, item_limit=180)
    if not approved_concepts:
        concept_payload = extract_episode_clone_concepts(
            session,
            video_id=video_id,
            notes=notes,
            semantic_query=effective_query,
            related_limit=related_limit,
            text_generator=text_generator,
            model_name=model_name,
        )
        approved_concepts = _normalize_text_list(concept_payload.get("concepts"), limit=18, item_limit=220)
        excluded_references = _normalize_text_list(
            [
                *excluded_references,
                *list(concept_payload.get("excluded_references") or []),
            ],
            limit=24,
            item_limit=180,
        )
        warnings.extend(_normalize_text_list(concept_payload.get("warnings"), limit=8, item_limit=180))

    excluded_references = _normalize_text_list(
        [
            *excluded_references,
            *_default_excluded_references(
                channel_name=str(channel.name or f"Channel {channel.id}"),
                source_video=video,
                related_videos=related_videos,
            ),
        ],
        limit=24,
        item_limit=180,
    )

    prompt = _build_clone_prompt(
        channel_name=str(channel.name or f"Channel {channel.id}"),
        source_video=video,
        source_metrics=source_metrics,
        source_brief=source_brief,
        style_prompt=style_prompt,
        notes=str(notes or "").strip(),
        semantic_query=effective_query,
        approved_concepts=approved_concepts,
        excluded_references=excluded_references,
        related_videos=related_videos,
        context_hits=context_hits,
    )
    llm_output = text_generator(prompt)
    payload, parsed_ok = _normalize_clone_output(llm_output)
    script = str(payload.get("script") or "").strip()
    if not parsed_ok:
        warnings.append("The model returned unstructured clone output, so the draft was not accepted.")
    if not script:
        warnings.append("The model returned no structured clone script.")
    else:
        overlap = _evaluate_originality_overlap(
            script,
            [("source brief", source_brief)]
            + [
                (f"related context: {str(hit.get('video_title') or hit.get('video_id') or 'unknown')}", str(hit.get("chunk_text") or ""))
                for hit in context_hits[:10]
            ],
        )
        max_ngram_overlap = float(overlap.get("max_ngram_overlap") or 0.0)
        copied_sentence_count = int(overlap.get("copied_sentence_count") or 0)
        top_reference = str(overlap.get("top_reference") or overlap.get("copied_sentence_reference") or "source context")

        if max_ngram_overlap >= 0.18 or copied_sentence_count >= 1:
            warnings.append(
                "Clone draft may reuse too much source phrasing. Review originality before using it."
            )
            payload["originality_notes"] = list(payload.get("originality_notes") or []) + [
                f"Overlap review flagged phrasing similarity against {top_reference}."
            ]
        if max_ngram_overlap >= 0.3 or copied_sentence_count >= 2:
            warnings.append(
                "Clone draft was suppressed because it overlaps too closely with source/context wording."
            )
            payload["originality_notes"] = list(payload.get("originality_notes") or []) + [
                f"Suppressed for excessive wording overlap against {top_reference}."
            ]
            payload["suggested_title"] = ""
            payload["opening_hook"] = ""
            payload["angle_summary"] = ""
            payload["script"] = ""
            script = ""

    if script:
        contamination = _evaluate_source_contamination(
            script,
            channel_name=str(channel.name or f"Channel {channel.id}"),
            source_title=str(video.title or f"Video {video.id}"),
            excluded_references=excluded_references,
        )
        if contamination["should_warn"]:
            literal_matches = list(contamination.get("literal_matches") or [])
            findings = list(contamination.get("findings") or [])
            payload["originality_notes"] = list(payload.get("originality_notes") or []) + [
                f"Source contamination review flagged: {', '.join(findings[:3])}."
            ]
            if literal_matches:
                payload["originality_notes"].append(
                    f"Matched forbidden references: {', '.join(str(item) for item in literal_matches[:4])}."
                )
            warnings.append("Clone draft may still contain source-specific references. Review before using it.")
        if contamination["should_suppress"]:
            warnings.append("Clone draft was suppressed because it still contains source-specific references.")
            payload["suggested_title"] = ""
            payload["opening_hook"] = ""
            payload["angle_summary"] = ""
            payload["script"] = ""

    return {
        "video_id": int(video.id),
        "channel_id": int(channel.id),
        "source_title": str(video.title or f"Video {video.id}"),
        "source_metrics": source_metrics,
        "style_prompt": style_prompt,
        "notes": str(notes or "").strip() or None,
        "semantic_query": effective_query,
        "source_brief": source_brief,
        "approved_concepts": approved_concepts,
        "excluded_references": excluded_references,
        "related_videos": related_videos,
        "context_hits": context_hits[:10],
        "suggested_title": str(payload.get("suggested_title") or ""),
        "opening_hook": str(payload.get("opening_hook") or ""),
        "angle_summary": str(payload.get("angle_summary") or ""),
        "originality_notes": list(payload.get("originality_notes") or []),
        "script": str(payload.get("script") or "").strip(),
        "warnings": warnings,
        "model": str(model_name or "").strip(),
    }
