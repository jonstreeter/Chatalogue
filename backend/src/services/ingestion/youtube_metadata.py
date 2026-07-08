"""YouTube AI metadata: humor summaries, chapters, description suggestions.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import json
from sqlmodel import Session, select
from datetime import datetime

from ...db.database import Video, Speaker, TranscriptSegment
from . import runtime


class YoutubeMetadataMixin:
    def _build_transcript_context_lines(self, segments: list[TranscriptSegment], speaker_map: dict[int, str]) -> list[str]:
        lines: list[str] = []
        for s in segments:
            text = (s.text or "").replace("\n", " ").strip()
            if not text:
                continue
            speaker_name = speaker_map.get(s.speaker_id) if s.speaker_id else None
            stamp = f"{int(s.start_time // 60)}:{int(s.start_time % 60):02d}"
            who = speaker_name or "Unknown"
            lines.append(f"[{stamp}] {who}: {text}")
        return lines

    def _chunk_transcript_lines_for_llm(
        self,
        lines: list[str],
        *,
        max_chunk_chars: int = 12_000,
        max_chunk_lines: int = 140,
        max_chunks: int = 24,
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0
        for line in lines:
            line_len = len(line) + 1
            if current and (current_chars + line_len > max_chunk_chars or len(current) >= max_chunk_lines):
                chunks.append("\n".join(current))
                if len(chunks) >= max_chunks:
                    return chunks
                current = []
                current_chars = 0
            current.append(line)
            current_chars += line_len
        if current and len(chunks) < max_chunks:
            chunks.append("\n".join(current))
        return chunks

    def _ollama_generate_episode_humor_context_summary(
        self,
        transcript_lines: list[str],
        *,
        progress_video_id: int | None = None,
        stage2_total: int | None = None,
    ) -> str:
        """Generate a cached episode-wide humor context summary from the full transcript (chunked)."""
        if not transcript_lines:
            raise RuntimeError("Cannot build episode humor context summary: transcript is empty.")

        chunks = self._chunk_transcript_lines_for_llm(transcript_lines)
        if not chunks:
            raise RuntimeError("Cannot build episode humor context summary: no transcript chunks.")

        chunk_summaries: list[str] = []
        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks, start=1):
            if progress_video_id is not None:
                # Reserve ~8%-18% of the total explain progress bar for Stage 1 chunking.
                stage1_pct = 8 + int((idx - 1) / max(1, total_chunks) * 10)
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_chunks",
                    message=f"Building episode-wide humor context summary (chunk {idx}/{total_chunks})...",
                    percent=stage1_pct,
                    current=idx - 1,
                    total=total_chunks,
                )
            prompt = (
                "You are summarizing ONE chunk of a podcast transcript to support humor analysis.\n"
                "Extract comedic context only: running bits, callbacks, teasing, repeated topics, and tone.\n"
                "Do not summarize everything; focus on what could make later laughter make sense.\n\n"
                f"Chunk {idx} of {total_chunks}\n\n"
                "Return ONLY JSON with this schema:\n"
                "{\"summary\":\"chunk humor context summary\",\"confidence\":\"low|medium|high\"}\n\n"
                "Transcript chunk:\n"
                f"{chunk_text}"
            )
            text = self._ollama_generate_text(
                prompt,
                temperature=0.15,
                num_predict=220,
                timeout_seconds=120,
            )
            summary, _confidence = self._parse_ollama_summary_confidence(text, max_summary_chars=900)
            if summary:
                chunk_summaries.append(f"Chunk {idx}: {summary}")
            if progress_video_id is not None:
                stage1_pct = 8 + int(idx / max(1, total_chunks) * 10)
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_chunks",
                    message=f"Building episode-wide humor context summary (chunk {idx}/{total_chunks})...",
                    percent=stage1_pct,
                    current=idx,
                    total=total_chunks,
                )

        if not chunk_summaries:
            raise RuntimeError("Ollama did not produce usable episode chunk summaries.")

        if len(chunk_summaries) == 1:
            if progress_video_id is not None:
                self._set_funny_task_progress(
                    progress_video_id,
                    task="explain",
                    status="running",
                    stage="global_context_done",
                    message=f"Episode-wide humor context summary complete. Preparing to explain moments (0/{stage2_total or 0})...",
                    percent=19,
                    current=0 if stage2_total is not None else None,
                    total=stage2_total,
                )
            return chunk_summaries[0][:1600]

        merged_input = "\n".join(chunk_summaries)
        if len(merged_input) > 14_000:
            merged_input = merged_input[-14_000:]

        merge_prompt = (
            "You are combining chunk-level humor summaries from a full podcast episode.\n"
            "Create one EPISODE-WIDE humor context summary to help explain specific laughter timestamps.\n"
            "Include recurring jokes/callbacks, people being teased, repeated themes, and the comedic tone.\n"
            "Be concise and specific.\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\"summary\":\"episode-wide humor context summary\",\"confidence\":\"low|medium|high\"}\n\n"
            "Chunk summaries:\n"
            f"{merged_input}"
        )
        if progress_video_id is not None:
            self._set_funny_task_progress(
                progress_video_id,
                task="explain",
                status="running",
                stage="global_context_merge",
                message=f"Merging {len(chunk_summaries)} chunk summaries into episode-wide context...",
                percent=19,
                current=len(chunk_summaries),
                total=len(chunk_summaries),
            )
        merged_text = self._ollama_generate_text(
            merge_prompt,
            temperature=0.15,
            num_predict=320,
            timeout_seconds=150,
        )
        merged_summary, _confidence = self._parse_ollama_summary_confidence(merged_text, max_summary_chars=1600)
        if progress_video_id is not None:
            self._set_funny_task_progress(
                progress_video_id,
                task="explain",
                status="running",
                stage="global_context_done",
                message=f"Episode-wide humor context summary complete. Preparing to explain moments (0/{stage2_total or 0})...",
                percent=19,
                current=0 if stage2_total is not None else None,
                total=stage2_total,
            )
        return merged_summary or merged_input[:1600]

    def _ensure_episode_humor_context_summary(
        self,
        session: Session,
        video: Video,
        segments: list[TranscriptSegment],
        speaker_map: dict[int, str],
        *,
        force: bool = False,
        progress_video_id: int | None = None,
        stage2_total: int | None = None,
    ) -> str | None:
        if not force and getattr(video, "humor_context_summary", None):
            return video.humor_context_summary

        transcript_lines = self._build_transcript_context_lines(segments, speaker_map)
        if not transcript_lines:
            return None

        summary = self._ollama_generate_episode_humor_context_summary(
            transcript_lines,
            progress_video_id=progress_video_id,
            stage2_total=stage2_total,
        )
        model_name = self._get_configured_llm_model_name()
        video.humor_context_summary = summary
        video.humor_context_model = model_name
        video.humor_context_generated_at = datetime.now()
        session.add(video)
        session.commit()
        session.refresh(video)
        return summary

    def _ollama_generate_humor_summary(
        self,
        context_text: str,
        moment_start: float,
        moment_end: float,
        *,
        episode_context_summary: str | None = None,
    ) -> tuple[str, str]:
        """Ask Ollama to infer what the humor was likely about from transcript context."""
        prompt = (
            "You are analyzing a podcast transcript around a laughter moment.\n"
            "Infer what the joke/humor was LIKELY about from the transcript context.\n"
            "Use the episode-wide humor context summary to recognize callbacks and running bits, "
            "but prioritize local transcript evidence.\n"
            "Be concise and uncertain when needed.\n\n"
            f"Laughter moment timestamp: {moment_start:.1f}s to {moment_end:.1f}s\n\n"
            + (
                "Episode-wide humor context summary (may be incomplete):\n"
                f"{episode_context_summary}\n\n"
                if episode_context_summary else ""
            )
            + "Return ONLY JSON with this schema:\n"
            "{\"summary\":\"1-2 sentence explanation of the likely joke/humor\","
            "\"confidence\":\"low|medium|high\"}\n\n"
            "Transcript context:\n"
            f"{context_text}"
        )
        text = self._ollama_generate_text(prompt, temperature=0.2, num_predict=180, timeout_seconds=90)
        return self._parse_ollama_summary_confidence(text, max_summary_chars=600)

    def _seconds_to_chapter_timestamp(self, seconds: float) -> str:
        total = max(0, int(round(float(seconds))))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _parse_chapter_timestamp_to_seconds(self, value) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parts = text.split(":")
        try:
            nums = [int(p) for p in parts]
        except Exception:
            return None
        if len(nums) == 2:
            m, s = nums
            if m < 0 or s < 0 or s > 59:
                return None
            return m * 60 + s
        if len(nums) == 3:
            h, m, s = nums
            if h < 0 or m < 0 or m > 59 or s < 0 or s > 59:
                return None
            return h * 3600 + m * 60 + s
        return None

    def _parse_json_object_from_text(self, text: str) -> dict | None:
        if not text:
            return None
        cleaned = self._strip_llm_reasoning_artifacts(text)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _normalize_youtube_ai_chapters(self, chapters, *, video_duration_seconds: float | None = None) -> list[dict]:
        normalized: list[dict] = []
        duration_cap = None
        try:
            if video_duration_seconds is not None:
                duration_cap = max(1, int(float(video_duration_seconds)))
        except Exception:
            duration_cap = None

        seen_starts = set()
        if not isinstance(chapters, list):
            chapters = []

        for item in chapters:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp") or item.get("start") or item.get("time")
            start_sec = self._parse_chapter_timestamp_to_seconds(ts)
            if start_sec is None:
                # accept numeric second fields if model returns them
                for key in ("start_seconds", "start_sec", "seconds"):
                    if key in item:
                        try:
                            start_sec = max(0, int(float(item[key])))
                            break
                        except Exception:
                            start_sec = None
            if start_sec is None:
                continue
            if duration_cap is not None and start_sec >= duration_cap:
                continue
            if start_sec in seen_starts:
                continue
            seen_starts.add(start_sec)

            title = str(item.get("title") or item.get("chapter") or "").strip()
            desc = str(item.get("description") or item.get("summary") or "").strip()
            if not title:
                continue
            title = " ".join(title.split())[:140]
            desc = " ".join(desc.split())[:280]

            normalized.append({
                "start_seconds": int(start_sec),
                "timestamp": self._seconds_to_chapter_timestamp(start_sec),
                "title": title,
                "description": desc,
            })

        normalized.sort(key=lambda c: c["start_seconds"])

        if normalized and normalized[0]["start_seconds"] != 0:
            normalized.insert(0, {
                "start_seconds": 0,
                "timestamp": "0:00",
                "title": "Intro",
                "description": "",
            })
        elif not normalized:
            normalized = [{
                "start_seconds": 0,
                "timestamp": "0:00",
                "title": "Episode Start",
                "description": "",
            }]

        # Enforce increasing timestamps and prune chapters that are too dense (<15s apart).
        pruned: list[dict] = []
        for ch in normalized:
            if not pruned:
                pruned.append(ch)
                continue
            if ch["start_seconds"] <= pruned[-1]["start_seconds"]:
                continue
            if ch["start_seconds"] - pruned[-1]["start_seconds"] < 15:
                continue
            pruned.append(ch)
        return pruned[:30]

    def _build_youtube_description_text(self, summary: str, chapters: list[dict]) -> str:
        lines: list[str] = []
        summary = (summary or "").strip()
        if summary:
            lines.append(summary)
            lines.append("")
        lines.append("Chapters")
        for ch in chapters:
            stamp = str(ch.get("timestamp") or "0:00").strip()
            title = str(ch.get("title") or "").strip()
            if not title:
                continue
            lines.append(f"{stamp} {title}")
        return "\n".join(lines).strip()

    def _parse_youtube_ai_result(self, text: str, *, video_duration_seconds: float | None = None) -> tuple[str, list[dict]]:
        parsed = self._parse_json_object_from_text(text)
        summary = ""
        chapters: list[dict] = []
        if isinstance(parsed, dict):
            summary = str(
                parsed.get("video_summary")
                or parsed.get("summary")
                or parsed.get("description_summary")
                or ""
            ).strip()
            chapters = self._normalize_youtube_ai_chapters(
                parsed.get("chapters") or parsed.get("chapter_timestamps") or [],
                video_duration_seconds=video_duration_seconds,
            )
        if not summary:
            cleaned = self._strip_llm_reasoning_artifacts(text)
            summary = " ".join(cleaned.split())[:1200]
        summary = summary[:1600]
        return summary, chapters

    def generate_youtube_metadata_suggestion(self, video_id: int, force: bool = False) -> Video:
        """Generate a YouTube-style summary + chapter timestamps/descriptions from transcript."""
        if not self._is_llm_enabled():
            raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")

        with Session(runtime.engine) as session:
            video = session.get(Video, video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")

            if (
                not force
                and getattr(video, "youtube_ai_summary", None)
                and getattr(video, "youtube_ai_chapters_json", None)
                and getattr(video, "youtube_ai_description_text", None)
            ):
                return video

            segments = session.exec(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
            ).all()
            if not segments:
                raise ValueError("Transcript segments not found. Run transcription first.")

            speaker_map: dict[int, str] = {}
            if video.channel_id:
                for sp in session.exec(select(Speaker).where(Speaker.channel_id == video.channel_id)).all():
                    if sp.id is not None:
                        speaker_map[sp.id] = sp.name

            transcript_lines = self._build_transcript_context_lines(segments, speaker_map)
            if not transcript_lines:
                raise ValueError("Transcript is empty.")

            # Allow larger transcript coverage than humor context to improve chapter generation.
            chunks = self._chunk_transcript_lines_for_llm(
                transcript_lines,
                max_chunk_chars=14_000,
                max_chunk_lines=180,
                max_chunks=36,
            )
            if not chunks:
                raise RuntimeError("Failed to prepare transcript chunks for chapter generation.")

            approx_duration = float(video.duration or (segments[-1].end_time if segments else 0) or 0)
            chunk_outputs: list[dict] = []
            total_chunks = len(chunks)
            for idx, chunk_text in enumerate(chunks, start=1):
                chunk_prompt = (
                    "You are preparing metadata for a YouTube podcast episode from a transcript chunk.\n"
                    "Identify major topics/themes and any good chapter boundaries visible in THIS chunk only.\n"
                    "Use transcript timestamps as-is. Do not invent content not present in the transcript.\n"
                    "Prefer broad thematic sections over tiny beats.\n\n"
                    f"Episode title: {video.title}\n"
                    f"Chunk {idx} of {total_chunks}\n\n"
                    "Return ONLY JSON with this schema:\n"
                    "{\"chunk_summary\":\"2-4 sentence topic summary for this chunk\","
                    "\"chapter_candidates\":[{\"timestamp\":\"MM:SS or H:MM:SS\",\"title\":\"short chapter title\",\"description\":\"one-sentence chapter description\"}]}\n\n"
                    "Transcript chunk:\n"
                    f"{chunk_text}"
                )
                raw = self._ollama_generate_text(
                    chunk_prompt,
                    temperature=0.15,
                    num_predict=500,
                    timeout_seconds=180,
                )
                parsed = self._parse_json_object_from_text(raw) or {}
                chunk_summary = str(parsed.get("chunk_summary") or parsed.get("summary") or "").strip()
                chapter_candidates = self._normalize_youtube_ai_chapters(
                    parsed.get("chapter_candidates") or [],
                    video_duration_seconds=approx_duration,
                )

                # Fallback summary if the model skipped JSON.
                if not chunk_summary:
                    fallback_summary, _ = self._parse_ollama_summary_confidence(raw, max_summary_chars=900)
                    chunk_summary = fallback_summary

                if chunk_summary or chapter_candidates:
                    chunk_outputs.append({
                        "chunk_index": idx,
                        "chunk_summary": chunk_summary[:900],
                        "chapter_candidates": chapter_candidates[:8],
                    })

            if not chunk_outputs:
                raise RuntimeError("LLM did not produce usable chunk summaries/chapters.")

            chunk_lines: list[str] = []
            flat_candidates: list[dict] = []
            for item in chunk_outputs:
                summary = (item.get("chunk_summary") or "").strip()
                if summary:
                    chunk_lines.append(f"Chunk {item['chunk_index']} summary: {summary}")
                cands = item.get("chapter_candidates") or []
                if cands:
                    for c in cands:
                        flat_candidates.append(c)
                        chunk_lines.append(
                            f"Chunk {item['chunk_index']} candidate chapter: {c['timestamp']} | {c['title']}"
                            + (f" | {c['description']}" if c.get("description") else "")
                        )

            merge_input = "\n".join(chunk_lines).strip()
            if len(merge_input) > 20_000:
                merge_input = merge_input[-20_000:]

            target_chapter_count = 8
            if approx_duration >= 3600:
                target_chapter_count = 12
            if approx_duration >= 7200:
                target_chapter_count = 16

            merge_prompt = (
                "You are generating YouTube-ready episode metadata from chunk-level transcript analyses.\n"
                "Produce:\n"
                "1) a strong YouTube-style episode summary (description intro) in 2-4 sentences\n"
                "2) chapter timestamps for major thematic sections\n"
                "3) a short one-sentence description for each chapter (for UI display)\n\n"
                "Rules:\n"
                "- Chapters must be chronological and represent major sections\n"
                "- First chapter MUST start at 0:00\n"
                "- Use timestamps only from the candidates/context; do not invent impossible times\n"
                "- Keep chapter titles concise and descriptive\n"
                "- Focus on what is actually discussed in the transcript\n"
                f"- Target about {target_chapter_count} chapters for this episode length\n\n"
                f"Episode title: {video.title}\n"
                f"Approx duration: {self._seconds_to_chapter_timestamp(approx_duration) if approx_duration else 'unknown'}\n\n"
                "Return ONLY JSON with this schema:\n"
                "{\"video_summary\":\"2-4 sentence YouTube description summary\","
                "\"chapters\":[{\"timestamp\":\"0:00\",\"title\":\"...\",\"description\":\"...\"}]}\n\n"
                "Chunk analyses and candidate chapters:\n"
                f"{merge_input}"
            )

            merged_raw = self._ollama_generate_text(
                merge_prompt,
                temperature=0.15,
                num_predict=900,
                timeout_seconds=240,
            )
            summary, chapters = self._parse_youtube_ai_result(
                merged_raw,
                video_duration_seconds=approx_duration,
            )

            if len(chapters) <= 1 and flat_candidates:
                # Fallback: use deduped candidate timestamps if merge failed to return chapters.
                chapters = self._normalize_youtube_ai_chapters(flat_candidates, video_duration_seconds=approx_duration)

            if not summary:
                # Fallback to merged chunk summaries if model refused final JSON.
                summary = " ".join(
                    [str(item.get("chunk_summary") or "").strip() for item in chunk_outputs if item.get("chunk_summary")]
                )[:1600]

            youtube_text = self._build_youtube_description_text(summary, chapters)
            model_name = self._get_configured_llm_model_name()

            video.youtube_ai_summary = summary
            video.youtube_ai_chapters_json = json.dumps(chapters, ensure_ascii=False)
            video.youtube_ai_description_text = youtube_text
            video.youtube_ai_model = model_name
            video.youtube_ai_generated_at = datetime.now()
            session.add(video)
            session.commit()
            session.refresh(video)
            return video
