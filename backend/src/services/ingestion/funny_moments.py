"""Funny-moment detection and explanation.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import subprocess
import re
import tempfile
from pathlib import Path
from sqlmodel import Session, select
from datetime import datetime

from ...db.database import Video, Speaker, TranscriptSegment, FunnyMoment
from ..logger import log, log_verbose
from . import runtime


class FunnyMomentsMixin:
    def _transcript_laughter_candidates(self, segments: list[TranscriptSegment]) -> list[dict]:
        """Detect explicit laughter cues in transcript text."""

        laughter_re = re.compile(
            r"\b(?:laugh(?:ter|ing|s|ed)?|giggl(?:e|es|ing|ed)?|chuckl(?:e|es|ing|ed)?|snicker(?:s|ing|ed)?|"
            r"haha+|ha\s+ha(?:\s+ha)*|hehe+|lol)\b",
            re.IGNORECASE,
        )

        candidates: list[dict] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            matches = laughter_re.findall(text)
            if not matches:
                continue

            duration = max(0.2, float(seg.end_time - seg.start_time))
            cue_score = min(0.4, 0.12 * len(matches))
            dur_score = min(0.2, duration / 12.0)
            score = 0.65 + cue_score + dur_score

            candidates.append({
                "start_time": max(0.0, float(seg.start_time) - 0.35),
                "end_time": float(seg.end_time) + 1.1,
                "score": round(score, 4),
                "source": "transcript",
                "snippet": text[:280],
            })

        return candidates

    def _acoustic_laughter_candidates(self, audio_path: Path) -> list[dict]:
        """Lightweight acoustic laughter heuristic using bursty energy + ZCR features.

        This is intentionally CPU-only and dependency-light (ffmpeg + soundfile + numpy).
        It is not a classifier, but works well enough as a laughter candidate generator.
        """
        import os
        import numpy as np
        import soundfile as sf

        ffmpeg_cmd = self._get_ffmpeg_cmd()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # Downsample to reduce CPU/memory while keeping enough temporal structure.
            subprocess.run(
                [ffmpeg_cmd, "-y", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "8000", tmp_path],
                check=True,
                capture_output=True,
                timeout=1800,
            )

            windows: list[dict] = []
            with sf.SoundFile(tmp_path) as f:
                sr = int(f.samplerate)
                if sr <= 0:
                    return []

                frame_len = max(1, int(sr * 0.02))  # 20ms
                frames_per_window = max(1, int(round(0.5 / 0.02)))  # 0.5s windows
                carry = np.array([], dtype=np.float32)
                frame_buf_rms: list[float] = []
                frame_buf_zcr: list[float] = []
                window_index = 0

                for block in f.blocks(blocksize=sr * 30, dtype="float32", always_2d=False):
                    if block is None:
                        continue
                    data = np.asarray(block, dtype=np.float32).flatten()
                    if carry.size:
                        data = np.concatenate([carry, data])

                    usable = (data.size // frame_len) * frame_len
                    if usable <= 0:
                        carry = data
                        continue

                    chunk = data[:usable].reshape(-1, frame_len)
                    carry = data[usable:]

                    rms = np.sqrt(np.mean(chunk * chunk, axis=1) + 1e-12)
                    signs = chunk >= 0
                    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

                    frame_buf_rms.extend(rms.tolist())
                    frame_buf_zcr.extend(zcr.tolist())

                    while len(frame_buf_rms) >= frames_per_window:
                        win_rms = np.asarray(frame_buf_rms[:frames_per_window], dtype=np.float32)
                        win_zcr = np.asarray(frame_buf_zcr[:frames_per_window], dtype=np.float32)
                        del frame_buf_rms[:frames_per_window]
                        del frame_buf_zcr[:frames_per_window]

                        start_t = window_index * 0.5
                        end_t = start_t + 0.5
                        window_index += 1

                        rms_mean = float(np.mean(win_rms))
                        rms_std = float(np.std(win_rms))
                        zcr_mean = float(np.mean(win_zcr))
                        high_frac = float(np.mean(win_rms > (rms_mean + max(1e-6, rms_std * 0.4))))

                        windows.append({
                            "start_time": start_t,
                            "end_time": end_t,
                            "rms": rms_mean,
                            "rms_std": rms_std,
                            "zcr": zcr_mean,
                            "high_frac": high_frac,
                        })

                # tail window (partial)
                if frame_buf_rms:
                    win_rms = np.asarray(frame_buf_rms, dtype=np.float32)
                    win_zcr = np.asarray(frame_buf_zcr, dtype=np.float32)
                    start_t = window_index * 0.5
                    windows.append({
                        "start_time": start_t,
                        "end_time": start_t + 0.5,
                        "rms": float(np.mean(win_rms)),
                        "rms_std": float(np.std(win_rms)),
                        "zcr": float(np.mean(win_zcr)) if win_zcr.size else 0.0,
                        "high_frac": float(np.mean(win_rms > (float(np.mean(win_rms)) + max(1e-6, float(np.std(win_rms)) * 0.4)))),
                    })

            if len(windows) < 6:
                return []

            import numpy as np  # local re-import okay for type checkers/runtime consistency

            rms_vals = np.asarray([w["rms"] for w in windows], dtype=np.float32)
            cv_vals = np.asarray([w["rms_std"] / max(w["rms"], 1e-6) for w in windows], dtype=np.float32)
            zcr_vals = np.asarray([w["zcr"] for w in windows], dtype=np.float32)
            hf_vals = np.asarray([w["high_frac"] for w in windows], dtype=np.float32)

            def _norm(val: float, lo: float, hi: float) -> float:
                if hi <= lo:
                    return 0.0
                return max(0.0, min(1.5, (val - lo) / (hi - lo)))

            r75, r95 = np.percentile(rms_vals, [75, 95])
            cv60, cv95 = np.percentile(cv_vals, [60, 95])
            z40, z90 = np.percentile(zcr_vals, [40, 90])
            hf50, hf95 = np.percentile(hf_vals, [50, 95])

            raw_candidates: list[dict] = []
            for w in windows:
                rms_n = _norm(w["rms"], float(r75), float(r95))
                cv = w["rms_std"] / max(w["rms"], 1e-6)
                cv_n = _norm(cv, float(cv60), float(cv95))
                z_n = _norm(w["zcr"], float(z40), float(z90))
                hf_n = _norm(w["high_frac"], float(hf50), float(hf95))

                # Favor bursty voiced-ish noise over steady tones/noise.
                score = (0.55 * rms_n) + (0.55 * cv_n) + (0.25 * z_n) + (0.2 * hf_n)
                if score < 1.05:
                    continue
                if w["rms"] < max(0.002, float(r75) * 0.4):
                    continue

                raw_candidates.append({
                    "start_time": w["start_time"],
                    "end_time": w["end_time"],
                    "score": round(float(score), 4),
                    "source": "acoustic",
                    "snippet": None,
                })

            if not raw_candidates:
                return []

            # Merge adjacent/nearby acoustic windows into laughter events.
            raw_candidates.sort(key=lambda c: c["start_time"])
            merged: list[dict] = []
            for c in raw_candidates:
                if not merged:
                    merged.append(dict(c))
                    continue
                prev = merged[-1]
                if c["start_time"] <= prev["end_time"] + 0.8:
                    prev["end_time"] = max(prev["end_time"], c["end_time"])
                    prev["score"] = round(max(prev["score"], c["score"]) + 0.05, 4)
                    prev["source"] = "acoustic"
                else:
                    merged.append(dict(c))

            # Filter unreasonable durations / keep strongest.
            filtered = []
            for m in merged:
                dur = m["end_time"] - m["start_time"]
                if dur < 0.5 or dur > 15:
                    continue
                # Extend slightly for nicer jump/play context.
                m["start_time"] = max(0.0, m["start_time"] - 0.25)
                m["end_time"] = m["end_time"] + 0.75
                filtered.append(m)
            return filtered

        except Exception as e:
            log_verbose(f"Acoustic laughter detection skipped: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _merge_funny_candidates(self, candidates: list[dict], segments: list[TranscriptSegment]) -> list[dict]:
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: (c["start_time"], c["end_time"]))
        merged: list[dict] = []

        for c in candidates:
            cur = {
                "start_time": float(c["start_time"]),
                "end_time": float(c["end_time"]),
                "score": float(c.get("score", 0.0)),
                "source_set": set(str(c.get("source", "heuristic")).split("+")),
                "snippet": c.get("snippet"),
            }
            if not merged:
                merged.append(cur)
                continue
            prev = merged[-1]
            if cur["start_time"] <= prev["end_time"] + 1.25:
                prev["start_time"] = min(prev["start_time"], cur["start_time"])
                prev["end_time"] = max(prev["end_time"], cur["end_time"])
                prev["score"] = max(prev["score"], cur["score"]) + 0.08
                prev["source_set"].update(cur["source_set"])
                if not prev.get("snippet") and cur.get("snippet"):
                    prev["snippet"] = cur["snippet"]
            else:
                merged.append(cur)

        # Attach nearest transcript snippet for acoustic-only events and finalize score/source.
        final: list[dict] = []
        for m in merged:
            mid = (m["start_time"] + m["end_time"]) / 2.0
            if not m.get("snippet"):
                nearest = None
                nearest_dist = float("inf")
                for seg in segments:
                    seg_mid = (seg.start_time + seg.end_time) / 2.0
                    dist = abs(seg_mid - mid)
                    if dist < nearest_dist:
                        nearest = seg
                        nearest_dist = dist
                if nearest and nearest_dist <= 12:
                    m["snippet"] = (nearest.text or "").strip()[:280]

            source_parts = sorted(s for s in m["source_set"] if s)
            source = "hybrid" if len(source_parts) > 1 else (source_parts[0] if source_parts else "heuristic")
            score = round(min(2.5, float(m["score"])), 3)
            final.append({
                "start_time": round(max(0.0, m["start_time"]), 2),
                "end_time": round(max(m["start_time"] + 0.2, m["end_time"]), 2),
                "score": score,
                "source": source,
                "snippet": (m.get("snippet") or None),
            })

        # Rank by score, then keep a manageable number and restore chronological order.
        try:
            max_results = int(os.getenv("FUNNY_MOMENTS_MAX_SAVED", "25"))
        except Exception:
            max_results = 25
        max_results = max(1, min(max_results, 200))
        top = sorted(final, key=lambda x: (x["score"], x["end_time"] - x["start_time"]), reverse=True)[:max_results]
        return sorted(top, key=lambda x: x["start_time"])

    def detect_funny_moments(self, video_id: int, force: bool = False) -> list[FunnyMoment]:
        """Generate and persist candidate funny/laughter moments for a video."""
        self._set_funny_task_progress(
            video_id,
            task="detect",
            status="running",
            stage="loading",
            message="Loading transcript and existing funny moments...",
            percent=2,
        )
        try:
            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video {video_id} not found")
                _ = video.channel  # ensure relationship loaded for get_audio_path path generation

                existing = session.exec(
                    select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                ).all()
                if existing and not force:
                    self._set_funny_task_progress(
                        video_id,
                        task="detect",
                        status="completed",
                        stage="done",
                        message=f"Using {len(existing)} cached funny moments.",
                        percent=100,
                        current=len(existing),
                        total=len(existing),
                    )
                    return existing

                segments = session.exec(
                    select(TranscriptSegment).where(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time)
                ).all()
                if not segments:
                    raise ValueError("Transcript segments not found. Run transcription first.")

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="transcript",
                    message="Scanning transcript for laughter cues...",
                    percent=20,
                )
                transcript_candidates = self._transcript_laughter_candidates(segments)

                acoustic_candidates: list[dict] = []
                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="acoustic",
                    message="Analyzing audio for laughter bursts...",
                    percent=45,
                )
                try:
                    audio_path = self.get_audio_path(video)
                    if audio_path.exists():
                        acoustic_candidates = self._acoustic_laughter_candidates(audio_path)
                except Exception as e:
                    log_verbose(f"Funny moments audio analysis skipped for video {video_id}: {e}")

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="merge",
                    message="Merging and ranking funny moment candidates...",
                    percent=75,
                )
                combined = self._merge_funny_candidates(transcript_candidates + acoustic_candidates, segments)

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="running",
                    stage="save",
                    message="Saving funny moments...",
                    percent=90,
                )
                # Replace cached rows
                for row in existing:
                    session.delete(row)
                session.commit()

                now = datetime.now()
                rows: list[FunnyMoment] = []
                for item in combined:
                    row = FunnyMoment(
                        video_id=video_id,
                        start_time=item["start_time"],
                        end_time=item["end_time"],
                        score=item["score"],
                        source=item["source"],
                        snippet=item.get("snippet"),
                        created_at=now,
                    )
                    session.add(row)
                    rows.append(row)

                session.commit()
                for row in rows:
                    session.refresh(row)

                self._set_funny_task_progress(
                    video_id,
                    task="detect",
                    status="completed",
                    stage="done",
                    message=f"Saved {len(rows)} funny moments.",
                    percent=100,
                    current=len(rows),
                    total=len(rows),
                )
                return rows
        except Exception as e:
            self._set_funny_task_progress(
                video_id,
                task="detect",
                status="error",
                stage="error",
                message=str(e),
                percent=100,
            )
            raise

    def explain_funny_moments(
        self,
        video_id: int,
        force: bool = False,
        limit: int | None = None,
        *,
        job_id: int | None = None,
    ) -> list[FunnyMoment]:
        """Generate AI summaries for detected funny moments using transcript context + Ollama."""
        if limit is None:
            try:
                limit = int(os.getenv("FUNNY_MOMENTS_EXPLAIN_BATCH_LIMIT", "12"))
            except Exception:
                limit = 12
        limit = max(1, min(int(limit), 200))
        self._raise_if_local_ollama_llm_is_blocked(job_id=job_id)
        self._set_funny_task_progress(
            video_id,
            task="explain",
            status="running",
            stage="loading",
            message="Loading funny moments and transcript...",
            percent=2,
        )
        try:
            with Session(runtime.engine) as session:
                video = session.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video {video_id} not found")

                moments = session.exec(
                    select(FunnyMoment)
                    .where(FunnyMoment.video_id == video_id)
                    .order_by(FunnyMoment.score.desc(), FunnyMoment.start_time)
                ).all()
                if not moments:
                    raise ValueError("No funny moments found. Run funny-moment detection first.")

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

                target_moments = []
                for m in moments:
                    if force or not m.humor_summary:
                        target_moments.append(m)
                    if len(target_moments) >= limit:
                        break

                if not target_moments:
                    self._set_funny_task_progress(
                        video_id,
                        task="explain",
                        status="completed",
                        stage="done",
                        message="No moments needed explanation.",
                        percent=100,
                        current=0,
                        total=0,
                    )
                    return session.exec(
                        select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                    ).all()

                total_targets = len(target_moments)
                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="running",
                    stage="global_context",
                    message="Building episode-wide humor context summary (Stage 1)...",
                    percent=8,
                    current=0,
                    total=total_targets,
                )

                model_name = self._get_configured_llm_model_name()
                episode_context_summary = None
                try:
                    episode_context_summary = self._ensure_episode_humor_context_summary(
                        session,
                        video,
                        segments,
                        speaker_map,
                        force=force,
                        progress_video_id=video_id,
                        stage2_total=total_targets,
                    )
                except Exception as e:
                    # Keep per-moment explanations working even if the episode-wide pass
                    # fails due to timeout/context/model issues.
                    log(f"Episode humor context summary skipped for video {video_id}: {e}")

                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="running",
                    stage="moments",
                    message=f"Explaining funny moments (0/{total_targets})...",
                    percent=20,
                    current=0,
                    total=total_targets,
                )

                now = datetime.now()
                for idx, m in enumerate(target_moments, start=1):
                    ctx_start = max(0.0, m.start_time - 75.0)
                    ctx_end = m.end_time + 20.0

                    ctx_segments = [
                        s for s in segments
                        if s.end_time >= ctx_start and s.start_time <= ctx_end
                    ]
                    # Limit prompt size while preserving lead-up context.
                    if len(ctx_segments) > 40:
                        ctx_segments = ctx_segments[-40:]

                    lines = self._build_transcript_context_lines(ctx_segments, speaker_map)

                    if lines:
                        context_text = "\n".join(lines)
                        # Hard cap to keep local LLM prompts bounded.
                        if len(context_text) > 6500:
                            context_text = context_text[-6500:]

                        summary, confidence = self._ollama_generate_humor_summary(
                            context_text,
                            m.start_time,
                            m.end_time,
                            episode_context_summary=episode_context_summary,
                        )
                        m.humor_summary = summary
                        m.humor_confidence = confidence
                        m.humor_model = model_name
                        m.humor_explained_at = now
                        session.add(m)

                    percent = 20 + (idx / max(1, total_targets)) * 80
                    self._set_funny_task_progress(
                        video_id,
                        task="explain",
                        status="running",
                        stage="moments",
                        message=f"Explaining funny moments ({idx}/{total_targets})...",
                        percent=percent,
                        current=idx,
                        total=total_targets,
                    )

                session.commit()
                self._set_funny_task_progress(
                    video_id,
                    task="explain",
                    status="completed",
                    stage="done",
                    message=f"Explained {total_targets} funny moments.",
                    percent=100,
                    current=total_targets,
                    total=total_targets,
                )
                return session.exec(
                    select(FunnyMoment).where(FunnyMoment.video_id == video_id).order_by(FunnyMoment.start_time)
                ).all()
        except Exception as e:
            self._set_funny_task_progress(
                video_id,
                task="explain",
                status="error",
                stage="error",
                message=str(e),
                percent=100,
            )
            raise
