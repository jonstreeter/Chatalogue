"""Transcript quality: consolidation, repair, gold windows, evaluation, optimization campaigns, runs, rollback.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import json
import re
import unicodedata
from pathlib import Path
from sqlmodel import Session, select
from difflib import SequenceMatcher
from datetime import datetime

from ...db.database import Video, Channel, Speaker, TranscriptSegment, TranscriptSegmentRevision, TranscriptRun, TranscriptQualitySnapshot, TranscriptGoldWindow, TranscriptEvaluationResult, TranscriptEvaluationReview, TranscriptOptimizationCampaign, TranscriptOptimizationCampaignItem, FunnyMoment
from ..logger import log
from . import runtime
from .runtime import (
    AUDIO_DIR,
    DATA_DIR,
    _env_float,
)


class TranscriptQualityMixin:
    def _transcript_segment_assignment_key(self, seg: TranscriptSegment):
        speaker_id = getattr(seg, "speaker_id", None)
        if speaker_id is not None:
            return ("speaker", int(speaker_id))
        profile_id = getattr(seg, "matched_profile_id", None)
        if profile_id is not None:
            return ("profile", int(profile_id))
        return None

    def _transcript_segment_word_count(self, seg: TranscriptSegment) -> int:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            return 0
        return len([token for token in text.split() if token])

    def _transcript_segment_duration(self, seg: TranscriptSegment) -> float:
        try:
            return max(0.0, float(seg.end_time) - float(seg.start_time))
        except Exception:
            return 0.0

    def _transcript_segment_gap(self, left_seg: TranscriptSegment, right_seg: TranscriptSegment) -> float:
        try:
            return max(0.0, float(right_seg.start_time) - float(left_seg.end_time))
        except Exception:
            return 0.0

    def _parse_segment_words_json(self, raw_words: str | None) -> list[dict] | None:
        if not raw_words:
            return None
        try:
            data = json.loads(raw_words)
            return data if isinstance(data, list) else None
        except Exception:
            return None

    def _dump_segment_words_json(self, words: list[dict] | None) -> str | None:
        if not words:
            return None
        try:
            return json.dumps(words, ensure_ascii=False)
        except Exception:
            return None

    def _segment_has_strong_terminal_punctuation(self, seg: TranscriptSegment) -> bool:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            return False
        return bool(re.search(r'[.!?]["\')\]]*\s*$', text))

    def _segment_starts_like_continuation(self, seg: TranscriptSegment) -> bool:
        text = str(getattr(seg, "text", "") or "").lstrip()
        if not text:
            return False
        if text[0] in {",", ";", ":", "-", ")", "]", "}"}:
            return True

        normalized = text.lstrip("\"'([{")
        if not normalized:
            return False

        token_match = re.match(r"[A-Za-z0-9][A-Za-z0-9'\-]*", normalized)
        if not token_match:
            return False
        token = token_match.group(0).lower()
        continuation_tokens = {
            "and",
            "but",
            "so",
            "because",
            "then",
            "though",
            "although",
            "however",
            "well",
            "also",
            "plus",
            "except",
            "if",
            "when",
            "while",
            "where",
            "which",
            "that",
            "who",
            "whose",
            "whom",
            "or",
            "nor",
            "yet",
            "still",
            "anyway",
            "anyways",
            "meanwhile",
            "like",
            "because",
        }
        return token in continuation_tokens

    def _merge_transcript_segment_text(self, left_text: str, right_text: str) -> str:
        left = str(left_text or "").rstrip()
        right = str(right_text or "").lstrip()
        if not left:
            return right
        if not right:
            return left
        if left.endswith(("-", "—")):
            return f"{left}{right}"
        return f"{left} {right}".strip()

    def _merge_transcript_segment_pair(self, left_seg: TranscriptSegment, right_seg: TranscriptSegment):
        left_seg.end_time = max(float(left_seg.end_time), float(right_seg.end_time))
        left_seg.text = self._merge_transcript_segment_text(left_seg.text, right_seg.text)
        if getattr(left_seg, "speaker_id", None) is None and getattr(right_seg, "speaker_id", None) is not None:
            left_seg.speaker_id = right_seg.speaker_id
        if getattr(left_seg, "matched_profile_id", None) is None and getattr(right_seg, "matched_profile_id", None) is not None:
            left_seg.matched_profile_id = right_seg.matched_profile_id

        left_words = self._parse_segment_words_json(getattr(left_seg, "words", None))
        right_words = self._parse_segment_words_json(getattr(right_seg, "words", None))
        if left_words is not None and right_words is not None:
            left_seg.words = self._dump_segment_words_json(left_words + right_words)
        elif left_words is None and right_words is None:
            left_seg.words = None
        else:
            left_seg.words = None

    def _consolidate_transcript_segments(self, segments: list[TranscriptSegment]) -> dict:
        ordered = sorted(
            list(segments or []),
            key=lambda s: (float(getattr(s, "start_time", 0.0) or 0.0), int(getattr(s, "id", 0) or 0)),
        )
        if len(ordered) <= 1:
            return {
                "segments": ordered,
                "removed_segments": [],
                "merged_count": 0,
                "reassigned_islands": 0,
                "before_count": len(ordered),
                "after_count": len(ordered),
            }

        turn_merge_gap_seconds = max(
            0.0,
            _env_float(
                "TRANSCRIPT_TURN_MERGE_GAP_SECONDS",
                os.getenv("TRANSCRIPT_CONSOLIDATE_MERGE_GAP_SECONDS") or "1.1",
            ),
        )
        sentence_break_gap_seconds = max(
            0.0,
            _env_float(
                "TRANSCRIPT_TURN_SENTENCE_BREAK_GAP_SECONDS",
                os.getenv("TRANSCRIPT_CONSOLIDATE_SENTENCE_BREAK_GAP_SECONDS") or "0.45",
            ),
        )
        continuation_gap_seconds = max(
            0.0,
            _env_float("TRANSCRIPT_TURN_CONTINUATION_GAP_SECONDS", str(turn_merge_gap_seconds)),
        )
        noncontinuation_gap_seconds = max(
            0.0,
            _env_float("TRANSCRIPT_TURN_NONCONTINUATION_GAP_SECONDS", "0.55"),
        )
        turn_max_words = max(0, int((os.getenv("TRANSCRIPT_TURN_MAX_WORDS") or "80").strip() or "80"))
        turn_max_seconds = max(0.0, _env_float("TRANSCRIPT_TURN_MAX_SECONDS", "30"))
        island_max_words = max(
            0,
            int(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_WORDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_WORDS")
                 or "2").strip() or "2"
            ),
        )
        island_max_seconds = max(
            0.0,
            float(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_SECONDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_SECONDS")
                 or "0.65").strip() or "0.65"
            ),
        )
        island_max_gap_seconds = max(
            0.0,
            float(
                (os.getenv("TRANSCRIPT_CONSOLIDATE_ISLAND_MAX_GAP_SECONDS")
                 or os.getenv("DIARIZATION_ORPHAN_MAX_GAP_SECONDS")
                 or "0.35").strip() or "0.35"
            ),
        )

        reassigned_islands = 0
        for idx in range(1, len(ordered) - 1):
            prev_seg = ordered[idx - 1]
            cur_seg = ordered[idx]
            next_seg = ordered[idx + 1]

            anchor_key = self._transcript_segment_assignment_key(prev_seg)
            if not anchor_key or anchor_key != self._transcript_segment_assignment_key(next_seg):
                continue
            if self._transcript_segment_assignment_key(cur_seg) == anchor_key:
                continue
            if self._transcript_segment_word_count(cur_seg) > island_max_words:
                continue
            if self._transcript_segment_duration(cur_seg) > island_max_seconds:
                continue
            if self._transcript_segment_gap(prev_seg, cur_seg) > island_max_gap_seconds:
                continue
            if self._transcript_segment_gap(cur_seg, next_seg) > island_max_gap_seconds:
                continue

            cur_seg.speaker_id = prev_seg.speaker_id
            cur_seg.matched_profile_id = prev_seg.matched_profile_id or next_seg.matched_profile_id
            reassigned_islands += 1

        merged_count = 0
        removed_segments: list[TranscriptSegment] = []
        survivors: list[TranscriptSegment] = []
        for seg in ordered:
            if not survivors:
                survivors.append(seg)
                continue

            left_seg = survivors[-1]
            left_key = self._transcript_segment_assignment_key(left_seg)
            right_key = self._transcript_segment_assignment_key(seg)
            gap_seconds = self._transcript_segment_gap(left_seg, seg)
            combined_words = self._transcript_segment_word_count(left_seg) + self._transcript_segment_word_count(seg)
            combined_seconds = (
                self._transcript_segment_duration(left_seg)
                + gap_seconds
                + self._transcript_segment_duration(seg)
            )
            left_has_terminal_punctuation = self._segment_has_strong_terminal_punctuation(left_seg)
            right_continues_turn = self._segment_starts_like_continuation(seg)

            can_merge = (
                left_key is not None
                and left_key == right_key
                and gap_seconds <= turn_merge_gap_seconds
            )
            if can_merge and turn_max_words > 0 and combined_words > turn_max_words:
                can_merge = False
            if can_merge and turn_max_seconds > 0.0 and combined_seconds > turn_max_seconds:
                can_merge = False
            if can_merge and not right_continues_turn and gap_seconds > noncontinuation_gap_seconds:
                can_merge = False
            if can_merge and left_has_terminal_punctuation:
                if right_continues_turn:
                    if gap_seconds > continuation_gap_seconds:
                        can_merge = False
                elif gap_seconds > sentence_break_gap_seconds:
                    can_merge = False

            if can_merge:
                self._merge_transcript_segment_pair(left_seg, seg)
                removed_segments.append(seg)
                merged_count += 1
            else:
                survivors.append(seg)

        return {
            "segments": survivors,
            "removed_segments": removed_segments,
            "merged_count": merged_count,
            "reassigned_islands": reassigned_islands,
            "before_count": len(ordered),
            "after_count": len(survivors),
        }

    def consolidate_existing_transcript(self, session: Session, video_id: int, *, save_files: bool = True) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        result = self._consolidate_transcript_segments(segments)
        survivors = result["segments"]
        removed_segments = result["removed_segments"]

        for seg in survivors:
            session.add(seg)
        for seg in removed_segments:
            session.delete(seg)
        session.commit()

        if save_files:
            channel = session.get(Channel, video.channel_id)
            safe_channel = self.sanitize_filename(channel.name if channel else "Unknown")
            safe_title = self.sanitize_filename(video.title)
            out_dir = AUDIO_DIR / safe_channel / safe_title
            out_dir.mkdir(parents=True, exist_ok=True)
            synthetic_audio_path = out_dir / f"{safe_title}.m4a"
            self._save_transcripts(session, video, survivors, synthetic_audio_path)

        return {
            "video_id": int(video_id),
            "title": video.title,
            "before_count": int(result["before_count"]),
            "after_count": int(result["after_count"]),
            "merged_count": int(result["merged_count"]),
            "reassigned_islands": int(result["reassigned_islands"]),
            "changed": bool(result["before_count"] != result["after_count"] or result["reassigned_islands"] > 0),
        }

    def _serialize_transcript_segment_rows(self, segments: list[TranscriptSegment]) -> list[dict]:
        rows: list[dict] = []
        for seg in segments or []:
            rows.append(
                {
                    "id": int(seg.id) if getattr(seg, "id", None) is not None else None,
                    "video_id": int(seg.video_id),
                    "speaker_id": int(seg.speaker_id) if getattr(seg, "speaker_id", None) is not None else None,
                    "matched_profile_id": int(seg.matched_profile_id) if getattr(seg, "matched_profile_id", None) is not None else None,
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "text": str(seg.text or ""),
                    "words": getattr(seg, "words", None),
                }
            )
        return rows

    def _write_transcript_repair_backup_rows(self, video: Video, rows: list[dict]) -> Path:
        backup_dir = DATA_DIR / "transcript_repair_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_title = self.sanitize_filename(str(video.title or f"video_{video.id}"))[:80] or f"video_{video.id}"
        backup_path = backup_dir / f"{int(video.id)}_{safe_title}_{timestamp}.json"
        payload = {
            "video_id": int(video.id),
            "youtube_id": str(video.youtube_id or ""),
            "title": str(video.title or ""),
            "created_at": datetime.now().isoformat(),
            "segments": rows,
        }
        backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return backup_path

    def _write_transcript_repair_backup(self, video: Video, segments: list[TranscriptSegment]) -> Path:
        return self._write_transcript_repair_backup_rows(video, self._serialize_transcript_segment_rows(segments))

    def _entity_repair_source_stopwords(self) -> set[str]:
        return {
            "the", "and", "for", "with", "from", "that", "this", "into", "about", "your", "their", "have",
            "what", "when", "where", "which", "while", "after", "before", "episode", "interview", "podcast",
            "video", "channel", "live", "show", "part", "full", "reaction", "review", "update", "story",
            "discussion", "conversation", "talk", "news", "analysis", "breakdown", "guide", "question", "answer",
            "today", "tonight", "topic", "session", "special", "guest", "host", "official", "edition", "road",
        }

    def _entity_repair_generic_phrase_tail_words(self) -> set[str]:
        return {
            "episode", "interview", "podcast", "video", "reaction", "review", "update", "story", "stories",
            "discussion", "conversation", "talk", "analysis", "breakdown", "guide", "question", "answer",
            "session", "edition", "diagnosis", "debate", "scandal",
        }

    def _normalize_entity_phrase(self, value: str | None) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("&", " and ")
        text = re.sub(r"[^A-Za-z0-9'\- ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _tokenize_entity_phrase(self, value: str | None) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(value or ""))

    def _collect_entity_repair_candidates(self, session: Session, video: Video) -> list[dict]:
        stopwords = self._entity_repair_source_stopwords()
        candidates: dict[str, dict] = {}

        def add_candidate(raw_text: str | None, source: str, priority: float) -> None:
            text = str(raw_text or "").strip()
            if not text:
                return
            tokens = self._tokenize_entity_phrase(text)
            if not tokens:
                return
            normalized = self._normalize_entity_phrase(" ".join(tokens))
            if not normalized:
                return
            alpha_chars = sum(1 for ch in normalized if ch.isalpha())
            if alpha_chars < 4:
                return
            if len(tokens) == 1 and normalized in stopwords:
                return
            if len(tokens) == 1 and len(normalized) < 5:
                return
            if len(tokens) > 4:
                return
            existing = candidates.get(normalized)
            payload = {
                "canonical": " ".join(tokens),
                "normalized": normalized,
                "tokens": tokens,
                "token_count": len(tokens),
                "source": source,
                "priority": float(priority),
            }
            if existing is None or payload["priority"] > float(existing.get("priority") or 0.0):
                candidates[normalized] = payload

        channel = session.get(Channel, video.channel_id) if getattr(video, "channel_id", None) else None
        if channel:
            add_candidate(channel.name, "channel_name", 0.96)
            for token in self._tokenize_entity_phrase(channel.name):
                add_candidate(token, "channel_token", 0.82)

        speakers = session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all()
        for speaker in speakers:
            add_candidate(speaker.name, "speaker_name", 1.0)
            for token in self._tokenize_entity_phrase(speaker.name):
                add_candidate(token, "speaker_token", 0.92)

        title_tokens = self._tokenize_entity_phrase(getattr(video, "title", None))
        for token in title_tokens:
            normalized = self._normalize_entity_phrase(token)
            if normalized in stopwords:
                continue
            if normalized in self._entity_repair_generic_phrase_tail_words():
                continue
            add_candidate(token, "title_token", 0.76)

        # Add consecutive title bigrams/trigrams to capture named entities such as
        # guest names or branded phrases that are not yet in speaker memory.
        for size in (3, 2):
            for idx in range(0, max(0, len(title_tokens) - size + 1)):
                phrase_tokens = title_tokens[idx: idx + size]
                phrase_norm = self._normalize_entity_phrase(" ".join(phrase_tokens))
                if any(tok.lower() in stopwords for tok in phrase_tokens):
                    continue
                if self._normalize_entity_phrase(phrase_tokens[-1]) in self._entity_repair_generic_phrase_tail_words():
                    continue
                if len(phrase_norm.replace(" ", "")) < 8:
                    continue
                add_candidate(" ".join(phrase_tokens), "title_phrase", 0.84)

        return sorted(candidates.values(), key=lambda item: (-float(item["priority"]), -int(item["token_count"]), item["canonical"]))

    def _replace_word_window_with_entity(self, words_payload: list[dict] | None, start_idx: int, end_idx: int, replacement_tokens: list[str]) -> str | None:
        if not words_payload or start_idx < 0 or end_idx >= len(words_payload):
            return None
        if (end_idx - start_idx + 1) != len(replacement_tokens):
            return None
        new_words = list(words_payload)
        for offset, token in enumerate(replacement_tokens):
            current = dict(new_words[start_idx + offset] or {})
            current["word"] = token
            new_words[start_idx + offset] = current
        return self._dump_segment_words_json(new_words)

    def _repair_segment_entities(self, segment: TranscriptSegment, candidates: list[dict]) -> dict:
        original_text = str(getattr(segment, "text", "") or "").strip()
        if not original_text or not candidates:
            return {"changed": False, "applied": []}

        token_matches = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*", original_text))
        if not token_matches:
            return {"changed": False, "applied": []}
        token_texts = [match.group(0) for match in token_matches]
        words_payload = self._parse_segment_words_json(getattr(segment, "words", None))

        replacements: list[dict] = []
        used_indices: set[int] = set()
        max_len = min(4, len(token_texts))
        for size in range(max_len, 0, -1):
            for start_idx in range(0, len(token_texts) - size + 1):
                window_indices = set(range(start_idx, start_idx + size))
                if used_indices & window_indices:
                    continue
                current_tokens = token_texts[start_idx:start_idx + size]
                current_phrase = " ".join(current_tokens)
                current_normalized = self._normalize_entity_phrase(current_phrase)
                if not current_normalized:
                    continue
                best = None
                for candidate in candidates:
                    if int(candidate.get("token_count") or 0) != size:
                        continue
                    candidate_normalized = str(candidate.get("normalized") or "")
                    if not candidate_normalized:
                        continue
                    similarity = SequenceMatcher(None, current_normalized, candidate_normalized).ratio()
                    exact_normalized = current_normalized == candidate_normalized
                    candidate_source = str(candidate.get("source") or "")
                    required_similarity = 0.88
                    if candidate_source.startswith("title_"):
                        required_similarity = 0.84
                    if not exact_normalized and similarity < required_similarity:
                        continue
                    if abs(len(candidate_normalized) - len(current_normalized)) > 3:
                        continue
                    if current_normalized[0] != candidate_normalized[0]:
                        continue
                    canonical = str(candidate.get("canonical") or "")
                    if exact_normalized and current_phrase == canonical:
                        continue
                    score = similarity + float(candidate.get("priority") or 0.0) * 0.05
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "canonical": canonical,
                            "source": candidate_source,
                            "replacement_tokens": list(candidate.get("tokens") or []),
                            "start_idx": start_idx,
                            "end_idx": start_idx + size - 1,
                            "current_phrase": current_phrase,
                        }
                if best is not None:
                    replacements.append(best)
                    used_indices.update(range(best["start_idx"], best["end_idx"] + 1))

        if not replacements:
            return {"changed": False, "applied": []}

        updated_text = original_text
        for item in sorted(replacements, key=lambda entry: token_matches[entry["start_idx"]].start(), reverse=True):
            start_char = token_matches[item["start_idx"]].start()
            end_char = token_matches[item["end_idx"]].end()
            updated_text = f"{updated_text[:start_char]}{item['canonical']}{updated_text[end_char:]}"
            replaced_words_json = self._replace_word_window_with_entity(
                words_payload,
                item["start_idx"],
                item["end_idx"],
                item["replacement_tokens"],
            )
            if replaced_words_json is not None:
                words_payload = json.loads(replaced_words_json)

        if updated_text == original_text:
            return {"changed": False, "applied": []}

        segment.text = updated_text
        if words_payload is not None:
            segment.words = json.dumps(words_payload, ensure_ascii=False)
        return {"changed": True, "applied": replacements}

    def _apply_entity_repair_to_segments(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        *,
        persist_revisions: bool = False,
        revision_source: str = "entity_repair",
    ) -> dict:
        candidates = self._collect_entity_repair_candidates(session, video)
        if not candidates:
            return {"changed": False, "segments_changed": 0, "replacement_count": 0, "sources": []}

        changed_segments = 0
        replacements = 0
        sources: set[str] = set()
        for seg in segments or []:
            old_text = str(getattr(seg, "text", "") or "")
            result = self._repair_segment_entities(seg, candidates)
            if not result.get("changed"):
                continue
            changed_segments += 1
            replacements += len(result.get("applied") or [])
            for applied in result.get("applied") or []:
                source = str(applied.get("source") or "").strip()
                if source:
                    sources.add(source)
            if persist_revisions and getattr(seg, "id", None) is not None and seg.text != old_text:
                session.add(
                    TranscriptSegmentRevision(
                        segment_id=int(seg.id),
                        video_id=int(seg.video_id),
                        old_text=old_text,
                        new_text=str(seg.text or ""),
                        source=revision_source,
                    )
                )
        return {
            "changed": changed_segments > 0,
            "segments_changed": changed_segments,
            "replacement_count": replacements,
            "sources": sorted(sources),
        }

    def _formatting_question_words(self, language: str | None) -> set[str]:
        normalized = self._normalize_language_code(language)
        if normalized == "es":
            return {
                "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
                "por", "por que", "por qué", "quien", "quién", "cual", "cuál",
            }
        return {
            "who", "what", "when", "where", "why", "how", "which", "did", "does",
            "do", "is", "are", "can", "could", "would", "should", "will",
        }

    def _looks_like_question_segment(self, text: str, language: str | None) -> bool:
        normalized_text = self._normalize_entity_phrase(text)
        if not normalized_text:
            return False
        for phrase in sorted(self._formatting_question_words(language), key=len, reverse=True):
            if normalized_text == phrase or normalized_text.startswith(f"{phrase} "):
                return True
        return False

    def _formatting_fillers(self, language: str | None) -> set[str]:
        normalized = self._normalize_language_code(language)
        if normalized == "es":
            return {"eh", "em", "este", "pues"}
        return {"uh", "um", "ah", "er", "eh"}

    def _cleanup_segment_formatting(self, text: str, *, language: str | None = None, duration: float | None = None) -> tuple[str, list[str]]:
        original = str(text or "")
        if not original.strip():
            return original, []

        cleaned = original
        steps: list[str] = []

        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
        }
        normalized_chars = "".join(replacements.get(ch, ch) for ch in cleaned)
        if normalized_chars != cleaned:
            cleaned = normalized_chars
            steps.append("normalize_punctuation")

        whitespace_cleaned = re.sub(r"\s+", " ", cleaned).strip()
        whitespace_cleaned = re.sub(r"\s+([,.;:!?])", r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r'(["\'])\s+', r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r"\s+([)\]])", r"\1", whitespace_cleaned)
        whitespace_cleaned = re.sub(r"([(\[])\s+", r"\1", whitespace_cleaned)
        if whitespace_cleaned != cleaned:
            cleaned = whitespace_cleaned
            steps.append("normalize_spacing")

        punctuation_cleaned = re.sub(r"\.{4,}", "...", cleaned)
        punctuation_cleaned = re.sub(r"([!?]){2,}", r"\1", punctuation_cleaned)
        punctuation_cleaned = re.sub(r",,{2,}", ",", punctuation_cleaned)
        punctuation_cleaned = re.sub(r";{2,}", ";", punctuation_cleaned)
        if punctuation_cleaned != cleaned:
            cleaned = punctuation_cleaned
            steps.append("collapse_punctuation")

        filler_pattern = "|".join(sorted(re.escape(token) for token in self._formatting_fillers(language)))
        if filler_pattern:
            filler_cleaned = re.sub(rf"\b({filler_pattern})(?:\s+\1\b){{2,}}", r"\1", cleaned, flags=re.IGNORECASE)
            if filler_cleaned != cleaned:
                cleaned = filler_cleaned
                steps.append("collapse_fillers")

        def _capitalize_match(match: re.Match[str]) -> str:
            prefix = match.group(1)
            letter = match.group(2)
            return f"{prefix}{letter.upper()}"

        capitalized = re.sub(r"^([^A-Za-z0-9]*)([a-z])", _capitalize_match, cleaned, count=1)
        if capitalized != cleaned:
            cleaned = capitalized
            steps.append("capitalize_start")

        if cleaned and not re.search(r'[.!?]["\')\]]*$', cleaned):
            token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", cleaned))
            lower_tail = cleaned.rstrip().lower()
            if (
                token_count >= 4
                and float(duration or 0.0) >= 1.0
                and not lower_tail.endswith((",", ";", ":", "-", "--"))
            ):
                suffix = "?" if self._looks_like_question_segment(cleaned, language) else "."
                cleaned = f"{cleaned}{suffix}"
                steps.append("terminal_punctuation")

        normalized_language = self._normalize_language_code(language)
        if normalized_language == "es":
            if cleaned.endswith("?") and not cleaned.startswith("¿") and self._looks_like_question_segment(cleaned[:-1], language):
                cleaned = f"¿{cleaned}"
                steps.append("spanish_inverted_question")
            if cleaned.endswith("!") and not cleaned.startswith("¡"):
                lead = self._normalize_entity_phrase(cleaned[:-1]).split(" ", 1)[0] if cleaned[:-1].strip() else ""
                if lead in {"que", "qué", "como", "cómo", "vaya"}:
                    cleaned = f"¡{cleaned}"
                    steps.append("spanish_inverted_exclamation")

        if cleaned == original:
            return original, []
        return cleaned, steps

    def _apply_formatting_cleanup_to_segments(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        *,
        persist_revisions: bool = False,
        revision_source: str = "formatting_cleanup",
    ) -> dict:
        language = self._normalize_language_code(getattr(video, "transcript_language", None))
        changed_segments = 0
        steps_applied: dict[str, int] = {}
        for seg in segments or []:
            old_text = str(getattr(seg, "text", "") or "")
            new_text, steps = self._cleanup_segment_formatting(
                old_text,
                language=language,
                duration=float(getattr(seg, "end_time", 0.0) or 0.0) - float(getattr(seg, "start_time", 0.0) or 0.0),
            )
            if not steps or new_text == old_text:
                continue
            seg.text = new_text
            changed_segments += 1
            for step in steps:
                steps_applied[step] = int(steps_applied.get(step) or 0) + 1
            if persist_revisions and getattr(seg, "id", None) is not None:
                session.add(
                    TranscriptSegmentRevision(
                        segment_id=int(seg.id),
                        video_id=int(seg.video_id),
                        old_text=old_text,
                        new_text=new_text,
                        source=revision_source,
                    )
                )
        return {
            "changed": changed_segments > 0,
            "segments_changed": changed_segments,
            "steps": steps_applied,
        }

    def repair_existing_transcript(
        self,
        session: Session,
        video_id: int,
        *,
        save_files: bool = True,
        persist_run: bool = True,
        persist_snapshot: bool = True,
        source: str = "manual",
        note: str | None = None,
        trigger_semantic_index: bool = True,
    ) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        if not segments:
            raise ValueError("Transcript not found")
        backup_rows = self._serialize_transcript_segment_rows(segments)

        profile_before = self._detect_transcript_quality_profile(video, segments)
        metrics_before = self._compute_transcript_quality_metrics(video, segments)
        tier_before, reasons_before, score_before, _ = self._recommend_transcript_optimization(profile_before, metrics_before)

        result = self._consolidate_transcript_segments(segments)
        survivors = result["segments"]
        removed_segments = result["removed_segments"]
        entity_repair = self._apply_entity_repair_to_segments(
            session,
            video,
            survivors,
            persist_revisions=True,
            revision_source="entity_repair",
        )
        formatting_cleanup = self._apply_formatting_cleanup_to_segments(
            session,
            video,
            survivors,
            persist_revisions=True,
            revision_source="formatting_cleanup",
        )
        changed = bool(
            result["before_count"] != result["after_count"]
            or result["reassigned_islands"] > 0
            or entity_repair["changed"]
            or formatting_cleanup["changed"]
        )
        backup_path = self._write_transcript_repair_backup_rows(video, backup_rows) if changed else None

        for seg in survivors:
            session.add(seg)
        for seg in removed_segments:
            session.delete(seg)
        session.commit()

        if save_files:
            channel = session.get(Channel, video.channel_id)
            safe_channel = self.sanitize_filename(channel.name if channel else "Unknown")
            safe_title = self.sanitize_filename(video.title)
            out_dir = AUDIO_DIR / safe_channel / safe_title
            out_dir.mkdir(parents=True, exist_ok=True)
            synthetic_audio_path = out_dir / f"{safe_title}.m4a"
            self._save_transcripts(session, video, survivors, synthetic_audio_path)

        profile_after = self._detect_transcript_quality_profile(video, survivors)
        metrics_after = self._compute_transcript_quality_metrics(video, survivors)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        run = None
        snapshot = None
        if persist_run:
            artifact_refs = {
                "backup_file": str(backup_path) if backup_path else None,
                "save_files": bool(save_files),
                "changed": bool(changed),
                "merged_count": int(result["merged_count"]),
                "reassigned_islands": int(result["reassigned_islands"]),
                "entity_segments_changed": int(entity_repair["segments_changed"]),
                "entity_replacement_count": int(entity_repair["replacement_count"]),
                "entity_sources": list(entity_repair["sources"]),
                "formatting_segments_changed": int(formatting_cleanup["segments_changed"]),
                "formatting_steps": dict(formatting_cleanup["steps"]),
            }
            model_provenance = {
                "repair_strategy": "consolidate_segments_entity_repair_formatting_cleanup",
                "source": str(source or "manual"),
            }
            run = self.create_transcript_run(
                session,
                video_id,
                mode="low_risk_repair",
                pipeline_version="transcript-repair-v1",
                quality_profile=profile_after,
                recommended_tier=tier_after,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                artifact_refs=artifact_refs,
                rollback_state=str(backup_path) if backup_path else None,
                model_provenance=model_provenance,
                note=note or "Transcript low-risk repair",
            )
            if persist_snapshot:
                snapshot = TranscriptQualitySnapshot(
                    video_id=int(video_id),
                    run_id=run.id,
                    source=str(source or "manual"),
                    quality_profile=profile_after,
                    recommended_tier=tier_after,
                    score=score_after,
                    metrics_json=json.dumps(metrics_after, ensure_ascii=False),
                    reasons_json=json.dumps(reasons_after, ensure_ascii=False),
                )
                session.add(snapshot)
            session.commit()
            if run:
                session.refresh(run)
            if snapshot:
                session.refresh(snapshot)

        if changed and trigger_semantic_index:
            self._trigger_semantic_index(int(video_id))

        return {
            "video_id": int(video_id),
            "title": video.title,
            "before_count": int(result["before_count"]),
            "after_count": int(result["after_count"]),
            "merged_count": int(result["merged_count"]),
            "reassigned_islands": int(result["reassigned_islands"]),
            "entity_segments_changed": int(entity_repair["segments_changed"]),
            "entity_replacement_count": int(entity_repair["replacement_count"]),
            "entity_sources": list(entity_repair["sources"]),
            "formatting_segments_changed": int(formatting_cleanup["segments_changed"]),
            "formatting_steps": dict(formatting_cleanup["steps"]),
            "changed": bool(changed),
            "backup_file": str(backup_path) if backup_path else None,
            "run_id": int(run.id) if run and run.id is not None else None,
            "snapshot_id": int(snapshot.id) if snapshot and snapshot.id is not None else None,
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def _parse_entities_json(self, raw_entities: str | None) -> list[str]:
        if not raw_entities:
            return []
        try:
            data = json.loads(raw_entities)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        items: list[str] = []
        seen: set[str] = set()
        for value in data:
            text = str(value or "").strip()
            if not text:
                continue
            key = self._normalize_entity_phrase(text)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(text)
        return items

    def _levenshtein_distance(self, left: list[str] | str, right: list[str] | str) -> int:
        left_seq = list(left)
        right_seq = list(right)
        if not left_seq:
            return len(right_seq)
        if not right_seq:
            return len(left_seq)
        prev = list(range(len(right_seq) + 1))
        for i, left_item in enumerate(left_seq, start=1):
            cur = [i]
            for j, right_item in enumerate(right_seq, start=1):
                cost = 0 if left_item == right_item else 1
                cur.append(min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                ))
            prev = cur
        return prev[-1]

    def _normalize_eval_text(self, text: str | None) -> str:
        value = unicodedata.normalize("NFKD", str(text or ""))
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = value.lower()
        value = re.sub(r"[^a-z0-9'\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _normalize_eval_chars(self, text: str | None) -> str:
        return re.sub(r"\s+", "", self._normalize_eval_text(text))

    def _punctuation_density(self, text: str | None) -> float:
        value = str(text or "")
        if not value:
            return 0.0
        punct_count = sum(1 for ch in value if ch in ".,;:!?")
        return punct_count / max(len(value), 1)

    def _collect_transcript_window_excerpt(self, session: Session, video_id: int, start_time: float, end_time: float) -> dict:
        segments = session.exec(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.video_id == video_id,
                TranscriptSegment.end_time > float(start_time),
                TranscriptSegment.start_time < float(end_time),
            )
            .order_by(TranscriptSegment.start_time, TranscriptSegment.id)
        ).all()
        excerpt_parts: list[str] = []
        unknown_count = 0
        for seg in segments:
            text = str(getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            excerpt_parts.append(text)
            if getattr(seg, "speaker_id", None) is None:
                unknown_count += 1
        excerpt = " ".join(excerpt_parts).strip()
        return {
            "text": excerpt,
            "segments": segments,
            "segment_count": len(segments),
            "unknown_speaker_rate": round((unknown_count / max(len(segments), 1)), 4) if segments else 0.0,
        }

    def upsert_transcript_gold_window(
        self,
        session: Session,
        video_id: int,
        *,
        window_id: int | None = None,
        label: str,
        quality_profile: str | None,
        language: str | None,
        start_time: float,
        end_time: float,
        reference_text: str,
        entities: list[str] | None = None,
        notes: str | None = None,
        active: bool = True,
    ) -> TranscriptGoldWindow:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")
        if float(end_time) <= float(start_time):
            raise ValueError("end_time must be greater than start_time")

        window = session.get(TranscriptGoldWindow, window_id) if window_id else None
        if window is not None and int(window.video_id) != int(video_id):
            raise ValueError("Gold window does not belong to this video")
        if window is None:
            window = TranscriptGoldWindow(video_id=int(video_id))
        window.label = str(label or "window").strip() or "window"
        window.quality_profile = str(quality_profile or "").strip() or None
        window.language = self._normalize_language_code(language) or (self._normalize_language_code(getattr(video, "transcript_language", None)) or None)
        window.start_time = float(start_time)
        window.end_time = float(end_time)
        window.reference_text = str(reference_text or "").strip()
        window.entities_json = json.dumps(list(entities or []), ensure_ascii=False)
        window.notes = str(notes or "").strip() or None
        window.active = bool(active)
        window.updated_at = datetime.now()
        session.add(window)
        session.commit()
        session.refresh(window)
        return window

    def _score_transcript_gold_window(
        self,
        session: Session,
        gold_window: TranscriptGoldWindow,
        *,
        run_id: int | None = None,
        source: str = "manual",
        candidate_text_override: str | None = None,
    ) -> TranscriptEvaluationResult:
        reference_text = str(gold_window.reference_text or "").strip()
        excerpt = self._collect_transcript_window_excerpt(
            session,
            int(gold_window.video_id),
            float(gold_window.start_time),
            float(gold_window.end_time),
        )
        candidate_text = str(candidate_text_override if candidate_text_override is not None else excerpt["text"] or "").strip()

        ref_norm = self._normalize_eval_text(reference_text)
        cand_norm = self._normalize_eval_text(candidate_text)
        ref_words = ref_norm.split() if ref_norm else []
        cand_words = cand_norm.split() if cand_norm else []
        wer_distance = self._levenshtein_distance(ref_words, cand_words)
        wer = float(wer_distance / max(len(ref_words), 1)) if ref_words else (0.0 if not cand_words else 1.0)

        ref_chars = list(self._normalize_eval_chars(reference_text))
        cand_chars = list(self._normalize_eval_chars(candidate_text))
        cer_distance = self._levenshtein_distance(ref_chars, cand_chars)
        cer = float(cer_distance / max(len(ref_chars), 1)) if ref_chars else (0.0 if not cand_chars else 1.0)

        entities = self._parse_entities_json(gold_window.entities_json)
        matched_entities = 0
        cand_phrase = self._normalize_entity_phrase(candidate_text)
        for entity in entities:
            entity_norm = self._normalize_entity_phrase(entity)
            if entity_norm and entity_norm in cand_phrase:
                matched_entities += 1
        entity_accuracy = (matched_entities / len(entities)) if entities else None

        punctuation_delta = round(
            abs(self._punctuation_density(candidate_text) - self._punctuation_density(reference_text)),
            4,
        )

        metrics = {
            "window_seconds": round(float(gold_window.end_time) - float(gold_window.start_time), 3),
            "reference_word_count": len(ref_words),
            "candidate_word_count": len(cand_words),
            "reference_char_count": len(ref_chars),
            "candidate_char_count": len(cand_chars),
            "levenshtein_word_distance": wer_distance,
            "levenshtein_char_distance": cer_distance,
            "window_language": self._normalize_language_code(gold_window.language),
        }
        result = TranscriptEvaluationResult(
            gold_window_id=int(gold_window.id),
            video_id=int(gold_window.video_id),
            run_id=run_id,
            source=str(source or "manual"),
            candidate_text=candidate_text,
            reference_text=reference_text,
            wer=round(wer, 4),
            cer=round(cer, 4),
            entity_accuracy=round(float(entity_accuracy), 4) if entity_accuracy is not None else None,
            matched_entity_count=int(matched_entities),
            total_entity_count=len(entities),
            segment_count=int(excerpt["segment_count"]),
            unknown_speaker_rate=float(excerpt["unknown_speaker_rate"] or 0.0),
            punctuation_density_delta=punctuation_delta,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return result

    def evaluate_transcript_gold_windows(
        self,
        session: Session,
        video_id: int,
        *,
        run_id: int | None = None,
        source: str = "manual",
        active_only: bool = True,
    ) -> dict:
        windows_query = select(TranscriptGoldWindow).where(TranscriptGoldWindow.video_id == video_id)
        if active_only:
            windows_query = windows_query.where(TranscriptGoldWindow.active == True)  # noqa: E712
        windows = session.exec(
            windows_query.order_by(TranscriptGoldWindow.start_time, TranscriptGoldWindow.id)
        ).all()
        if not windows:
            raise ValueError("No gold windows defined for this video")

        results: list[TranscriptEvaluationResult] = []
        for window in windows:
            results.append(
                self._score_transcript_gold_window(
                    session,
                    window,
                    run_id=run_id,
                    source=source,
                )
            )

        avg_wer = round(sum(float(item.wer or 0.0) for item in results) / max(len(results), 1), 4)
        avg_cer = round(sum(float(item.cer or 0.0) for item in results) / max(len(results), 1), 4)
        entity_values = [float(item.entity_accuracy) for item in results if item.entity_accuracy is not None]
        avg_entity = round(sum(entity_values) / len(entity_values), 4) if entity_values else None
        avg_unknown = round(sum(float(item.unknown_speaker_rate or 0.0) for item in results) / max(len(results), 1), 4)

        return {
            "video_id": int(video_id),
            "run_id": run_id,
            "total_windows": len(results),
            "average_wer": avg_wer,
            "average_cer": avg_cer,
            "average_entity_accuracy": avg_entity,
            "average_unknown_speaker_rate": avg_unknown,
            "items": results,
        }

    def maybe_evaluate_transcript_run_against_gold_windows(
        self,
        session: Session,
        video_id: int,
        *,
        run_id: int | None = None,
        source: str,
    ) -> dict | None:
        existing = session.exec(
            select(TranscriptGoldWindow.id)
            .where(TranscriptGoldWindow.video_id == video_id, TranscriptGoldWindow.active == True)  # noqa: E712
            .limit(1)
        ).first()
        if not existing:
            return None
        return self.evaluate_transcript_gold_windows(session, video_id, run_id=run_id, source=source, active_only=True)

    def _loads_json_object(self, raw: str | None, fallback):
        if not raw:
            return fallback
        try:
            parsed = json.loads(raw)
        except Exception:
            return fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback

    def list_transcript_rollback_options(self, session: Session, video_id: int) -> list[dict]:
        runs = session.exec(
            select(TranscriptRun)
            .where(TranscriptRun.video_id == int(video_id))
            .where(TranscriptRun.rollback_state.is_not(None))
            .order_by(TranscriptRun.created_at.desc(), TranscriptRun.id.desc())
        ).all()
        options: list[dict] = []
        for run in runs:
            rollback_state = str(run.rollback_state or "").strip() or None
            rollback_available = False
            if rollback_state:
                try:
                    rollback_available = Path(rollback_state).exists()
                except Exception:
                    rollback_available = False
            options.append(
                {
                    "run_id": int(run.id),
                    "video_id": int(run.video_id),
                    "mode": str(run.mode or "unknown"),
                    "pipeline_version": str(run.pipeline_version or ""),
                    "note": run.note,
                    "created_at": run.created_at,
                    "rollback_available": rollback_available,
                    "rollback_state": rollback_state,
                }
            )
        return options

    def restore_transcript_from_run(
        self,
        session: Session,
        video_id: int,
        run_id: int,
        *,
        source: str = "api",
    ) -> dict:
        video = session.get(Video, int(video_id))
        if not video:
            raise ValueError("Video not found")
        run = session.get(TranscriptRun, int(run_id))
        if not run or int(run.video_id) != int(video_id):
            raise ValueError("Transcript run not found")
        backup_path_raw = str(run.rollback_state or "").strip()
        if not backup_path_raw:
            raise ValueError("Selected run does not have rollback data")
        backup_path = Path(backup_path_raw)
        if not backup_path.exists():
            raise ValueError("Rollback backup file is missing")

        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise ValueError(f"Rollback backup could not be read: {exc}") from exc

        current_segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time, TranscriptSegment.id)
        ).all()
        current_backup_rows = [
            {
                "speaker_id": seg.speaker_id,
                "matched_profile_id": seg.matched_profile_id,
                "start_time": float(seg.start_time or 0.0),
                "end_time": float(seg.end_time or 0.0),
                "text": str(seg.text or ""),
                "words": seg.words,
            }
            for seg in current_segments
        ]
        current_funny_rows = session.exec(
            select(FunnyMoment).where(FunnyMoment.video_id == int(video_id)).order_by(FunnyMoment.start_time, FunnyMoment.id)
        ).all()
        current_backup_payload = {
            "video_id": int(video_id),
            "video_status": video.status,
            "video_processed": bool(video.processed),
            "saved_at": datetime.now().isoformat(),
            "segments": current_backup_rows,
            "funny_moments": [
                {
                    "start_time": float(row.start_time or 0.0),
                    "end_time": float(row.end_time or 0.0),
                    "score": float(row.score or 0.0),
                    "source": str(row.source or "heuristic"),
                    "snippet": row.snippet,
                    "humor_summary": row.humor_summary,
                    "humor_confidence": row.humor_confidence,
                    "humor_model": row.humor_model,
                    "humor_explained_at": row.humor_explained_at.isoformat() if row.humor_explained_at else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in current_funny_rows
            ],
        }
        current_backup_path = self._get_temp_redo_backup_path(int(video_id))
        current_backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(current_backup_path, "w", encoding="utf-8") as f:
            json.dump(current_backup_payload, f, ensure_ascii=False)

        from sqlalchemy import delete as sa_delete

        session.exec(sa_delete(TranscriptSegmentRevision).where(TranscriptSegmentRevision.video_id == int(video_id)))
        session.exec(sa_delete(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)))
        session.exec(sa_delete(FunnyMoment).where(FunnyMoment.video_id == int(video_id)))
        session.flush()

        def _parse_dt(value):
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except Exception:
                return None

        restored_segment_count = 0
        for row in payload.get("segments", []) or []:
            session.add(
                TranscriptSegment(
                    video_id=int(video_id),
                    speaker_id=row.get("speaker_id"),
                    matched_profile_id=row.get("matched_profile_id"),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    text=str(row.get("text") or ""),
                    words=row.get("words"),
                )
            )
            restored_segment_count += 1

        restored_funny_count = 0
        for row in payload.get("funny_moments", []) or []:
            session.add(
                FunnyMoment(
                    video_id=int(video_id),
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
                )
            )
            restored_funny_count += 1

        session.flush()

        backup_metrics = self._loads_json_object(run.metrics_before_json, {})
        profile = self._detect_transcript_quality_profile(
            video,
            session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time)
            ).all(),
        )
        metrics = self._compute_transcript_quality_metrics(
            video,
            session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == int(video_id)).order_by(TranscriptSegment.start_time)
            ).all(),
        )
        recommended_tier, reasons, quality_score, _ = self._recommend_transcript_optimization(profile, metrics)

        video.status = str(payload.get("video_status") or "completed")
        video.processed = bool(payload.get("video_processed")) if payload.get("video_processed") is not None else restored_segment_count > 0
        video.transcript_is_placeholder = False
        if restored_segment_count > 0 and not str(video.transcript_source or "").strip():
            video.transcript_source = "local_transcription"
        session.add(video)

        restore_run = self.create_transcript_run(
            session,
            int(video_id),
            mode="rollback_restore",
            pipeline_version="rollback-restore-v1",
            quality_profile=profile,
            recommended_tier=recommended_tier,
            metrics_before=backup_metrics if isinstance(backup_metrics, dict) else None,
            metrics_after=metrics,
            artifact_refs={
                "source_run_id": int(run.id),
                "source_backup_file": str(backup_path),
                "restore_source": str(source or "api"),
                "previous_transcript_backup": str(current_backup_path),
                "previous_funny_moment_count": len(current_funny_rows),
            },
            rollback_state=str(current_backup_path),
            model_provenance={"strategy": "rollback_restore", "source": str(source or "api")},
            note=f"Restored transcript from run {int(run.id)}",
            input_run_id=int(run.id),
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=restore_run.id,
            source="rollback_restore",
            quality_profile=profile,
            recommended_tier=recommended_tier,
            score=quality_score,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
            reasons_json=json.dumps(reasons, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(restore_run)

        try:
            self.save_transcript_files(video, session)
        except Exception as exc:
            log(f"Transcript restore save_files failed for video {video_id}: {exc}")
        try:
            self.reindex_video_semantic_embeddings(int(video_id))
        except Exception as exc:
            log(f"Transcript restore semantic reindex failed for video {video_id}: {exc}")

        return {
            "video_id": int(video_id),
            "restored_from_run_id": int(run.id),
            "restore_run_id": int(restore_run.id),
            "segment_count": restored_segment_count,
            "funny_moment_count": restored_funny_count,
            "quality_profile": profile,
            "recommended_tier": recommended_tier,
            "quality_score": quality_score,
        }

    def summarize_transcript_evaluation(self, session: Session, *, channel_id: int | None = None) -> dict:
        query = select(TranscriptEvaluationResult)
        if channel_id is not None:
            query = query.join(Video, Video.id == TranscriptEvaluationResult.video_id).where(Video.channel_id == int(channel_id))
        results = session.exec(query.order_by(TranscriptEvaluationResult.created_at.desc(), TranscriptEvaluationResult.id.desc())).all()

        windows_query = select(TranscriptGoldWindow)
        if channel_id is not None:
            windows_query = windows_query.join(Video, Video.id == TranscriptGoldWindow.video_id).where(Video.channel_id == int(channel_id))
        windows = session.exec(windows_query).all()

        result_ids = [int(item.id) for item in results if item.id is not None]
        reviews: list[TranscriptEvaluationReview] = []
        if result_ids:
            reviews = session.exec(
                select(TranscriptEvaluationReview).where(TranscriptEvaluationReview.evaluation_result_id.in_(result_ids))
            ).all()

        verdict_counts: dict[str, int] = {}
        for review in reviews:
            key = str(review.verdict or "same")
            verdict_counts[key] = verdict_counts.get(key, 0) + 1

        def _avg(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        entity_values = [float(item.entity_accuracy) for item in results if item.entity_accuracy is not None]
        return {
            "scope": "channel" if channel_id is not None else "global",
            "channel_id": int(channel_id) if channel_id is not None else None,
            "total_gold_windows": len(windows),
            "total_results": len(results),
            "total_reviewed_results": len({int(review.evaluation_result_id) for review in reviews}),
            "average_wer": _avg([float(item.wer or 0.0) for item in results]),
            "average_cer": _avg([float(item.cer or 0.0) for item in results]),
            "average_entity_accuracy": _avg(entity_values),
            "average_unknown_speaker_rate": _avg([float(item.unknown_speaker_rate or 0.0) for item in results]),
            "verdict_counts": verdict_counts,
            "latest_result_at": results[0].created_at if results else None,
        }

    def summarize_diarization_benchmarks(self, session: Session, *, channel_id: int | None = None) -> list[dict]:
        query = (
            select(TranscriptRun, TranscriptEvaluationResult)
            .join(TranscriptEvaluationResult, TranscriptEvaluationResult.run_id == TranscriptRun.id)
            .where(TranscriptRun.mode == "diarization_rebuild")
        )
        if channel_id is not None:
            query = query.join(Video, Video.id == TranscriptRun.video_id).where(Video.channel_id == int(channel_id))
        rows = session.exec(query).all()

        grouped: dict[str, dict] = {}
        for run, result in rows:
            provenance = self._loads_json_object(run.model_provenance_json, {})
            sensitivity = str(provenance.get("diarization_sensitivity") or "balanced")
            threshold_raw = provenance.get("speaker_match_threshold")
            try:
                threshold_value = round(float(threshold_raw), 2)
            except Exception:
                threshold_value = None
            label = f"{sensitivity} / {threshold_value if threshold_value is not None else 'default'}"
            bucket = grouped.setdefault(
                label,
                {
                    "label": label,
                    "run_ids": set(),
                    "wers": [],
                    "cers": [],
                    "unknown_rates": [],
                    "latest_run_id": None,
                    "latest_created_at": None,
                    "diarization_sensitivity": sensitivity,
                    "speaker_match_threshold": threshold_value,
                },
            )
            bucket["run_ids"].add(int(run.id))
            bucket["wers"].append(float(result.wer or 0.0))
            bucket["cers"].append(float(result.cer or 0.0))
            bucket["unknown_rates"].append(float(result.unknown_speaker_rate or 0.0))
            if bucket["latest_created_at"] is None or (run.created_at and run.created_at > bucket["latest_created_at"]):
                bucket["latest_created_at"] = run.created_at
                bucket["latest_run_id"] = int(run.id)

        summaries: list[dict] = []
        for bucket in grouped.values():
            summaries.append(
                {
                    "label": bucket["label"],
                    "run_count": len(bucket["run_ids"]),
                    "average_wer": round(sum(bucket["wers"]) / len(bucket["wers"]), 4) if bucket["wers"] else None,
                    "average_cer": round(sum(bucket["cers"]) / len(bucket["cers"]), 4) if bucket["cers"] else None,
                    "average_unknown_speaker_rate": round(sum(bucket["unknown_rates"]) / len(bucket["unknown_rates"]), 4) if bucket["unknown_rates"] else None,
                    "latest_run_id": bucket["latest_run_id"],
                    "latest_created_at": bucket["latest_created_at"],
                    "diarization_sensitivity": bucket["diarization_sensitivity"],
                    "speaker_match_threshold": bucket["speaker_match_threshold"],
                }
            )
        summaries.sort(key=lambda item: (item.get("average_wer") is None, item.get("average_wer") or 9999.0, item.get("average_cer") or 9999.0))
        return summaries

    def create_transcript_optimization_campaign(
        self,
        session: Session,
        *,
        channel_id: int | None = None,
        limit: int = 100,
        tiers: list[str] | None = None,
        force_non_eligible: bool = False,
        note: str | None = None,
    ) -> TranscriptOptimizationCampaign:
        requested_tiers = [str(item or "").strip() for item in (tiers or []) if str(item or "").strip()]
        allowed_tiers = {"low_risk_repair", "diarization_rebuild", "full_retranscription", "manual_review"}
        filtered_tiers = [item for item in requested_tiers if item in allowed_tiers]
        dry_run = self.transcript_optimization_dry_run(
            session,
            channel_id=channel_id,
            limit=limit,
            persist_snapshots=False,
        )
        campaign = TranscriptOptimizationCampaign(
            channel_id=int(channel_id) if channel_id is not None else None,
            scope="channel" if channel_id is not None else "global",
            status="draft",
            tiers_json=json.dumps(filtered_tiers, ensure_ascii=False),
            limit=max(1, int(limit)),
            force_non_eligible=bool(force_non_eligible),
            note=str(note or "").strip() or None,
            updated_at=datetime.now(),
        )
        session.add(campaign)
        session.flush()

        for item in dry_run.get("items", []):
            metrics = item.get("metrics") or {}
            if int(metrics.get("total_segments") or 0) <= 0:
                continue
            recommended_tier = str(item.get("recommended_tier") or "none")
            if filtered_tiers and recommended_tier not in filtered_tiers:
                continue
            reason_list = item.get("reasons") or []
            session.add(
                TranscriptOptimizationCampaignItem(
                    campaign_id=int(campaign.id),
                    video_id=int(item["video_id"]),
                    recommended_tier=recommended_tier,
                    action_tier=recommended_tier,
                    quality_score=float(item.get("quality_score") or 0.0),
                    reason=str(reason_list[0]) if reason_list else None,
                    status="pending",
                )
            )
        session.commit()
        session.refresh(campaign)
        return campaign

    def _detect_transcript_quality_profile(self, video: Video, segments: list[TranscriptSegment]) -> str:
        language = str(getattr(video, "transcript_language", "") or "").strip().lower()
        title = str(getattr(video, "title", "") or "").strip().lower()
        description = str(getattr(video, "description", "") or "").strip().lower()
        haystack = f"{language} {title} {description}"
        multilingual_markers = ("spanish", "espanol", "español", "bilingual", "portuguese", "french")
        if language.startswith("es") or any(marker in haystack for marker in multilingual_markers):
            return "multilingual_or_non_english"
        if len(segments) >= 120:
            return "english_longform"
        return "english_general"

    def _compute_transcript_quality_metrics(self, video: Video, segments: list[TranscriptSegment]) -> dict:
        ordered = sorted(
            list(segments or []),
            key=lambda s: (float(getattr(s, "start_time", 0.0) or 0.0), int(getattr(s, "id", 0) or 0)),
        )
        total_segments = len(ordered)
        if total_segments <= 0:
            return {
                "total_segments": 0,
                "unknown_speaker_segments": 0,
                "unknown_speaker_rate": 0.0,
                "micro_segment_count": 0,
                "micro_segment_rate": 0.0,
                "tiny_unknown_count": 0,
                "tiny_unknown_rate": 0.0,
                "same_speaker_interruptions": 0,
                "same_speaker_interruption_rate": 0.0,
                "punctuated_segment_rate": 0.0,
                "avg_segment_seconds": 0.0,
                "avg_words_per_segment": 0.0,
                "word_timed_segment_rate": 0.0,
                "distinct_assigned_speakers": 0,
                "assigned_speaker_segments": 0,
                "language": getattr(video, "transcript_language", None),
            }

        durations = [self._transcript_segment_duration(seg) for seg in ordered]
        word_counts = [self._transcript_segment_word_count(seg) for seg in ordered]
        unknown_segments = [
            seg for seg in ordered
            if getattr(seg, "speaker_id", None) is None and getattr(seg, "matched_profile_id", None) is None
        ]
        micro_segments = [seg for seg, duration in zip(ordered, durations) if duration > 0.0 and duration < 1.5]
        tiny_unknown_segments = [
            seg for seg in unknown_segments
            if self._transcript_segment_duration(seg) > 0.0 and self._transcript_segment_duration(seg) < 1.5
        ]
        punctuated_segments = [seg for seg in ordered if self._segment_has_strong_terminal_punctuation(seg)]
        word_timed_segments = [
            seg for seg in ordered
            if isinstance(self._parse_segment_words_json(getattr(seg, "words", None)), list)
        ]
        assigned_speaker_ids = {
            int(seg.speaker_id)
            for seg in ordered
            if getattr(seg, "speaker_id", None) is not None
        }

        same_speaker_interruptions = 0
        for idx in range(1, len(ordered) - 1):
            prev_seg = ordered[idx - 1]
            cur_seg = ordered[idx]
            next_seg = ordered[idx + 1]
            prev_key = self._transcript_segment_assignment_key(prev_seg)
            cur_key = self._transcript_segment_assignment_key(cur_seg)
            next_key = self._transcript_segment_assignment_key(next_seg)
            if not prev_key or prev_key != next_key or cur_key == prev_key:
                continue
            if self._transcript_segment_duration(cur_seg) > 2.0:
                continue
            if self._transcript_segment_word_count(cur_seg) > 8:
                continue
            same_speaker_interruptions += 1

        return {
            "total_segments": total_segments,
            "unknown_speaker_segments": len(unknown_segments),
            "unknown_speaker_rate": round(len(unknown_segments) / total_segments, 4),
            "micro_segment_count": len(micro_segments),
            "micro_segment_rate": round(len(micro_segments) / total_segments, 4),
            "tiny_unknown_count": len(tiny_unknown_segments),
            "tiny_unknown_rate": round(len(tiny_unknown_segments) / total_segments, 4),
            "same_speaker_interruptions": same_speaker_interruptions,
            "same_speaker_interruption_rate": round(same_speaker_interruptions / total_segments, 4),
            "punctuated_segment_rate": round(len(punctuated_segments) / total_segments, 4),
            "avg_segment_seconds": round(sum(durations) / total_segments, 3),
            "avg_words_per_segment": round(sum(word_counts) / total_segments, 3),
            "word_timed_segment_rate": round(len(word_timed_segments) / total_segments, 4),
            "distinct_assigned_speakers": len(assigned_speaker_ids),
            "assigned_speaker_segments": total_segments - len(unknown_segments),
            "language": getattr(video, "transcript_language", None),
        }

    def _recommend_transcript_optimization(self, profile: str, metrics: dict) -> tuple[str, list[str], float, bool]:
        total_segments = max(1, int(metrics.get("total_segments") or 0))
        unknown_rate = float(metrics.get("unknown_speaker_rate") or 0.0)
        micro_rate = float(metrics.get("micro_segment_rate") or 0.0)
        tiny_unknown_rate = float(metrics.get("tiny_unknown_rate") or 0.0)
        interruption_rate = float(metrics.get("same_speaker_interruption_rate") or 0.0)
        punctuated_rate = float(metrics.get("punctuated_segment_rate") or 0.0)
        word_timed_rate = float(metrics.get("word_timed_segment_rate") or 0.0)

        reasons: list[str] = []
        tier = "none"

        if total_segments < 5:
            reasons.append("Transcript is too small to evaluate reliably.")
            score = 100.0
            return tier, reasons, score, False

        if profile == "multilingual_or_non_english":
            if punctuated_rate < 0.9 or unknown_rate > 0.08 or micro_rate > 0.18:
                tier = "full_retranscription"
                reasons.append("Language profile suggests multilingual or non-English handling should be re-routed.")

        if tier == "none" and word_timed_rate >= 0.4 and unknown_rate >= 0.12:
            tier = "diarization_rebuild"
            reasons.append("High unknown-speaker rate with usable word timing suggests diarization can be rebuilt.")

        if tier == "none" and (interruption_rate >= 0.04 or tiny_unknown_rate >= 0.025 or micro_rate >= 0.22):
            tier = "low_risk_repair"
            reasons.append("Short interruptions and micro-segmentation indicate consolidation repair should help.")

        if tier == "none" and punctuated_rate < 0.75:
            tier = "manual_review"
            reasons.append("Formatting quality is weak without a clear automatic repair path.")

        score = 100.0
        score -= unknown_rate * 180.0
        score -= micro_rate * 70.0
        score -= interruption_rate * 120.0
        score -= max(0.0, 0.82 - punctuated_rate) * 55.0
        if profile == "multilingual_or_non_english":
            score -= 6.0
        score = max(0.0, min(100.0, round(score, 2)))

        if not reasons and tier == "none":
            reasons.append("Transcript quality is within the current automatic optimization thresholds.")
        return tier, reasons, score, tier != "none"

    def create_transcript_run(
        self,
        session: Session,
        video_id: int,
        *,
        mode: str,
        pipeline_version: str,
        status: str = "completed",
        quality_profile: str | None = None,
        recommended_tier: str | None = None,
        metrics_before: dict | None = None,
        metrics_after: dict | None = None,
        artifact_refs: dict | None = None,
        rollback_state: str | None = None,
        model_provenance: dict | None = None,
        note: str | None = None,
        input_run_id: int | None = None,
    ) -> TranscriptRun:
        run = TranscriptRun(
            video_id=int(video_id),
            input_run_id=input_run_id,
            mode=str(mode or "baseline"),
            pipeline_version=str(pipeline_version or "baseline-v1"),
            status=str(status or "completed"),
            quality_profile=quality_profile,
            recommended_tier=recommended_tier,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            metrics_before_json=json.dumps(metrics_before or {}, ensure_ascii=False),
            metrics_after_json=json.dumps(metrics_after or {}, ensure_ascii=False) if metrics_after is not None else None,
            artifact_refs_json=json.dumps(artifact_refs or {}, ensure_ascii=False) if artifact_refs is not None else None,
            rollback_state=rollback_state,
            model_provenance_json=json.dumps(model_provenance or {}, ensure_ascii=False) if model_provenance is not None else None,
            note=note,
        )
        session.add(run)
        session.flush()
        return run

    def evaluate_transcript_quality(
        self,
        session: Session,
        video_id: int,
        *,
        source: str = "manual",
        persist_snapshot: bool = False,
    ) -> dict:
        video = session.get(Video, video_id)
        if not video:
            raise ValueError("Video not found")

        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile = self._detect_transcript_quality_profile(video, segments)
        metrics = self._compute_transcript_quality_metrics(video, segments)
        recommended_tier, reasons, score, eligible = self._recommend_transcript_optimization(profile, metrics)

        snapshot = None
        if persist_snapshot:
            run = self.create_transcript_run(
                session,
                video_id,
                mode="quality_assessment",
                pipeline_version="quality-evaluator-v1",
                quality_profile=profile,
                recommended_tier=recommended_tier,
                metrics_before=metrics,
                note=f"Quality assessment snapshot via {source}",
            )
            snapshot = TranscriptQualitySnapshot(
                video_id=int(video_id),
                run_id=run.id,
                source=str(source or "manual"),
                quality_profile=profile,
                recommended_tier=recommended_tier,
                score=score,
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                reasons_json=json.dumps(reasons, ensure_ascii=False),
            )
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
        result = {
            "video_id": int(video_id),
            "title": str(video.title or ""),
            "channel_id": video.channel_id,
            "quality_profile": profile,
            "recommended_tier": recommended_tier,
            "quality_score": score,
            "eligible_for_optimization": bool(eligible),
            "language": getattr(video, "transcript_language", None),
            "metrics": metrics,
            "reasons": reasons,
            "created_snapshot_id": int(snapshot.id) if snapshot and snapshot.id is not None else None,
            "snapshot_created_at": snapshot.created_at if snapshot else None,
        }
        return result

    def transcript_optimization_dry_run(
        self,
        session: Session,
        *,
        channel_id: int | None = None,
        video_id: int | None = None,
        limit: int = 50,
        persist_snapshots: bool = False,
    ) -> dict:
        query = select(Video).where(Video.processed == True)  # noqa: E712
        if video_id is not None:
            query = query.where(Video.id == int(video_id))
        elif channel_id is not None:
            query = query.where(Video.channel_id == int(channel_id))
        query = query.order_by(Video.published_at.desc(), Video.id.desc()).limit(max(1, int(limit)))

        items: list[dict] = []
        for video in session.exec(query).all():
            item = self.evaluate_transcript_quality(
                session,
                int(video.id),
                source="dry_run",
                persist_snapshot=bool(persist_snapshots),
            )
            if int((item.get("metrics") or {}).get("total_segments") or 0) <= 0:
                continue
            items.append(item)

        eligible_count = len([item for item in items if item.get("eligible_for_optimization")])
        return {
            "total_scanned": len(items),
            "total_eligible": eligible_count,
            "items": items,
        }

    def create_diarization_rebuild_run(
        self,
        session: Session,
        video_id: int,
        *,
        payload: dict | None = None,
        note: str | None = None,
    ) -> dict | None:
        payload = dict(payload or {})
        optimization_target = str(payload.get("optimization_target") or "").strip().lower()
        mode = str(payload.get("mode") or "").strip().lower()
        if optimization_target != "diarization_rebuild" and mode != "redo_diarization":
            return None

        video = session.get(Video, video_id)
        if not video:
            return None
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile_after = self._detect_transcript_quality_profile(video, segments)
        metrics_after = self._compute_transcript_quality_metrics(video, segments)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        metrics_before = payload.get("quality_metrics_before")
        if not isinstance(metrics_before, dict):
            metrics_before = {}
        profile_before = str(payload.get("quality_profile_before") or "unknown")
        tier_before = str(payload.get("recommended_tier_before") or "none")
        score_before = float(payload.get("quality_score_before") or 0.0)
        reasons_before = payload.get("quality_reasons_before")
        if not isinstance(reasons_before, list):
            reasons_before = []

        artifact_refs = {
            "redo_backup_file": payload.get("redo_diarization_backup_file"),
            "optimization_target": optimization_target or "redo_diarization",
            "queued_from": payload.get("queued_from"),
            "raw_transcript_reused": True,
        }
        diarization_sensitivity = str(
            payload.get("diarization_sensitivity_override")
            or os.getenv("DIARIZATION_SENSITIVITY", "balanced")
            or "balanced"
        )
        try:
            speaker_match_threshold = float(
                payload.get("speaker_match_threshold_override")
                if payload.get("speaker_match_threshold_override") is not None
                else os.getenv("SPEAKER_MATCH_THRESHOLD", "0.35")
            )
        except Exception:
            speaker_match_threshold = 0.35
        model_provenance = {
            "strategy": "redo_diarization",
            "diarization_sensitivity": diarization_sensitivity,
            "speaker_match_threshold": speaker_match_threshold,
            "benchmark_variant": bool(payload.get("benchmark_variant")),
            "source": str(payload.get("queued_from") or "manual"),
        }
        run = self.create_transcript_run(
            session,
            video_id,
            mode="diarization_rebuild",
            pipeline_version="diarization-benchmark-v1" if bool(payload.get("benchmark_variant")) else "diarization-rebuild-v1",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            artifact_refs=artifact_refs,
            rollback_state=str(payload.get("redo_diarization_backup_file") or "") or None,
            model_provenance=model_provenance,
            note=note or str(payload.get("note") or "").strip() or "Transcript diarization rebuild",
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=run.id,
            source="queued_job",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            score=score_after,
            metrics_json=json.dumps(metrics_after, ensure_ascii=False),
            reasons_json=json.dumps(reasons_after, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(run)
        session.refresh(snapshot)
        return {
            "run_id": int(run.id),
            "snapshot_id": int(snapshot.id),
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def create_full_retranscription_run(
        self,
        session: Session,
        video_id: int,
        *,
        payload: dict | None = None,
        note: str | None = None,
    ) -> dict | None:
        payload = dict(payload or {})
        optimization_target = str(payload.get("optimization_target") or "").strip().lower()
        mode = str(payload.get("mode") or "").strip().lower()
        if optimization_target != "full_retranscription" and mode != "full_retranscription":
            return None

        video = session.get(Video, video_id)
        if not video:
            return None
        segments = session.exec(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
        ).all()
        profile_after = self._detect_transcript_quality_profile(video, segments)
        metrics_after = self._compute_transcript_quality_metrics(video, segments)
        tier_after, reasons_after, score_after, _ = self._recommend_transcript_optimization(profile_after, metrics_after)

        metrics_before = payload.get("quality_metrics_before")
        if not isinstance(metrics_before, dict):
            metrics_before = {}
        profile_before = str(payload.get("quality_profile_before") or "unknown")
        tier_before = str(payload.get("recommended_tier_before") or "none")
        score_before = float(payload.get("quality_score_before") or 0.0)
        reasons_before = payload.get("quality_reasons_before")
        if not isinstance(reasons_before, list):
            reasons_before = []

        artifact_refs = {
            "redo_backup_file": payload.get("redo_diarization_backup_file"),
            "optimization_target": optimization_target or "full_retranscription",
            "queued_from": payload.get("queued_from"),
            "raw_transcript_reused": False,
            "force_retranscription": bool(payload.get("force_retranscription")),
        }
        model_provenance = {
            "strategy": "full_retranscription",
            "transcription_engine_requested": payload.get("transcription_engine_requested"),
            "transcription_engine_routed": payload.get("transcription_engine_routed"),
            "transcription_engine_used": payload.get("transcription_engine_used"),
            "transcription_route_language": payload.get("transcription_route_language"),
            "transcription_route_language_source": payload.get("transcription_route_language_source"),
            "transcription_route_multilingual": payload.get("transcription_route_multilingual"),
            "transcription_route_whisper_model": payload.get("transcription_route_whisper_model"),
            "source": str(payload.get("queued_from") or "manual"),
        }
        run = self.create_transcript_run(
            session,
            video_id,
            mode="full_retranscription",
            pipeline_version="full-retranscription-v1",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            artifact_refs=artifact_refs,
            rollback_state=str(payload.get("redo_diarization_backup_file") or "") or None,
            model_provenance=model_provenance,
            note=note or str(payload.get("note") or "").strip() or "Transcript full retranscription",
        )
        snapshot = TranscriptQualitySnapshot(
            video_id=int(video_id),
            run_id=run.id,
            source="queued_job",
            quality_profile=profile_after,
            recommended_tier=tier_after,
            score=score_after,
            metrics_json=json.dumps(metrics_after, ensure_ascii=False),
            reasons_json=json.dumps(reasons_after, ensure_ascii=False),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(run)
        session.refresh(snapshot)
        return {
            "run_id": int(run.id),
            "snapshot_id": int(snapshot.id),
            "quality_profile_before": profile_before,
            "quality_profile_after": profile_after,
            "recommended_tier_before": tier_before,
            "recommended_tier_after": tier_after,
            "quality_score_before": score_before,
            "quality_score_after": score_after,
            "reasons_before": reasons_before,
            "reasons_after": reasons_after,
        }

    def _record_transcript_optimization_completion(self, job_id: int, video_id: int, payload: dict | None):
        payload = dict(payload or {})
        result = None
        evaluation = None
        with Session(runtime.engine) as session:
            result = (
                self.create_diarization_rebuild_run(session, video_id, payload=payload)
                or self.create_full_retranscription_run(session, video_id, payload=payload)
            )
            if result and result.get("run_id"):
                try:
                    evaluation = self.maybe_evaluate_transcript_run_against_gold_windows(
                        session,
                        video_id,
                        run_id=int(result["run_id"]),
                        source="optimization_completion",
                    )
                except Exception as e:
                    log(f"Transcript evaluation-on-completion failed for video {video_id}: {e}")
        if not result:
            return
        payload_fields = {
            "optimization_run_id": result.get("run_id"),
            "optimization_snapshot_id": result.get("snapshot_id"),
            "optimization_recommended_tier_after": result.get("recommended_tier_after"),
            "optimization_quality_score_after": result.get("quality_score_after"),
            "rebuild_run_id": result.get("run_id"),
            "rebuild_snapshot_id": result.get("snapshot_id"),
            "rebuild_recommended_tier_after": result.get("recommended_tier_after"),
            "rebuild_quality_score_after": result.get("quality_score_after"),
        }
        if evaluation:
            payload_fields.update(
                {
                    "optimization_evaluation_window_count": evaluation.get("total_windows"),
                    "optimization_evaluation_average_wer": evaluation.get("average_wer"),
                    "optimization_evaluation_average_cer": evaluation.get("average_cer"),
                    "optimization_evaluation_average_entity_accuracy": evaluation.get("average_entity_accuracy"),
                }
            )
        self._upsert_job_payload_fields(job_id, payload_fields)
