"""Speaker embedding profiles, match cache, and identification.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
from pathlib import Path
from sqlmodel import Session, select

from ...db.database import Speaker, SpeakerEmbedding
from ..logger import log_verbose
from .runtime import (
    _env_float,
)


class SpeakerIdentityMixin:
    def _invalidate_speaker_match_cache(self, channel_id: int | None):
        if not channel_id:
            return
        with self._speaker_match_cache_guard:
            self._speaker_match_cache.pop(int(channel_id), None)

    def _append_speaker_match_cache(
        self,
        channel_id: int | None,
        profile_id: int | None,
        speaker_id: int | None,
        embedding,
    ):
        """Append a newly-created embedding to the in-memory channel cache."""
        if not channel_id or not profile_id or not speaker_id or embedding is None:
            return
        import numpy as np

        channel_key = int(channel_id)
        with self._speaker_match_cache_guard:
            cache = self._speaker_match_cache.get(channel_key)
            if not cache:
                return
            try:
                vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            except Exception:
                return
            if vec.size == 0 or int(vec.size) != int(cache.get("dim", -1)):
                return
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                return
            vec = (vec / norm).astype(np.float32, copy=False)
            matrix = cache.get("matrix")
            profile_ids = cache.get("profile_ids")
            speaker_ids = cache.get("speaker_ids")
            if profile_ids is None or speaker_ids is None:
                return
            try:
                if matrix is None or int(cache.get("dim") or 0) <= 0 or profile_ids.size == 0:
                    cache["dim"] = int(vec.size)
                    cache["matrix"] = vec[None, :]
                    cache["profile_ids"] = np.asarray([int(profile_id)], dtype=np.int64)
                    cache["speaker_ids"] = np.asarray([int(speaker_id)], dtype=np.int64)
                    cache["count"] = 1
                    return
                cache["matrix"] = np.vstack([matrix, vec[None, :]])
                cache["profile_ids"] = np.append(profile_ids, int(profile_id))
                cache["speaker_ids"] = np.append(speaker_ids, int(speaker_id))
                cache["count"] = int(cache["profile_ids"].shape[0])
            except Exception:
                # If append fails for any reason, force a safe rebuild on next use.
                self._speaker_match_cache.pop(channel_key, None)

    def _get_speaker_match_cache(self, session: Session, channel_id: int):
        import numpy as np

        channel_key = int(channel_id)
        with self._speaker_match_cache_guard:
            cached = self._speaker_match_cache.get(channel_key)
        if cached:
            return cached

        rows = session.exec(
            select(SpeakerEmbedding)
            .join(Speaker)
            .where(Speaker.channel_id == channel_key)
        ).all()
        if not rows:
            cache = {"dim": 0, "matrix": None, "profile_ids": np.array([], dtype=np.int64), "speaker_ids": np.array([], dtype=np.int64), "count": 0}
            with self._speaker_match_cache_guard:
                self._speaker_match_cache[channel_key] = cache
            return cache

        parsed_rows = []
        for row in rows:
            vec = self._normalize_embedding_vector(getattr(row, "embedding", None))
            if vec is None:
                continue
            parsed_rows.append((int(row.id), int(row.speaker_id), vec))

        vectors = []
        profile_ids = []
        speaker_ids = []
        expected_dim = 0
        if parsed_rows:
            dim_counts = {}
            for _, _, vec in parsed_rows:
                dim = int(vec.size)
                dim_counts[dim] = int(dim_counts.get(dim, 0)) + 1
            expected_dim = max(dim_counts.items(), key=lambda item: (item[1], item[0]))[0]
            for profile_id, speaker_id, vec in parsed_rows:
                if int(vec.size) != int(expected_dim):
                    continue
                vectors.append(vec)
                profile_ids.append(int(profile_id))
                speaker_ids.append(int(speaker_id))

        if not vectors:
            cache = {"dim": 0, "matrix": None, "profile_ids": np.array([], dtype=np.int64), "speaker_ids": np.array([], dtype=np.int64), "count": 0}
        else:
            cache = {
                "dim": int(expected_dim or 0),
                "matrix": np.stack(vectors, axis=0),
                "profile_ids": np.asarray(profile_ids, dtype=np.int64),
                "speaker_ids": np.asarray(speaker_ids, dtype=np.int64),
                "count": len(profile_ids),
            }

        with self._speaker_match_cache_guard:
            self._speaker_match_cache[channel_key] = cache
        return cache

    def _normalize_embedding_vector(self, embedding):
        import numpy as np

        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim > 1:
            try:
                arr = np.mean(arr, axis=0, dtype=np.float32)
            except Exception:
                arr = arr.reshape(-1)
        vec = np.asarray(arr, dtype=np.float32).reshape(-1)
        if vec.size < 16:
            return None
        if not np.all(np.isfinite(vec)):
            return None
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return (vec / norm).astype(np.float32, copy=False)

    def _select_speaker_embedding_segments(self, timeline):
        max_samples = max(1, int(os.getenv("SPEAKER_PROFILE_SAMPLE_COUNT", "4")))
        min_sample_seconds = max(0.5, _env_float("SPEAKER_PROFILE_MIN_SAMPLE_SECONDS", "2.0"))
        fallback_min_seconds = max(0.5, _env_float("SPEAKER_PROFILE_FALLBACK_MIN_SECONDS", "0.75"))

        candidates = []
        for seg in timeline or []:
            try:
                duration = float(seg.duration)
            except Exception:
                duration = 0.0
            if duration <= 0.0:
                continue
            candidates.append(seg)

        if not candidates:
            return []

        candidates.sort(key=lambda seg: (-float(seg.duration), float(seg.start)))
        selected = [seg for seg in candidates if float(seg.duration) >= min_sample_seconds]
        if not selected:
            selected = [seg for seg in candidates if float(seg.duration) >= fallback_min_seconds]
        if not selected:
            selected = candidates

        return selected[:max_samples]

    def _build_speaker_embedding_profile(self, audio_source, timeline):
        import numpy as np

        selected_segments = self._select_speaker_embedding_segments(timeline)
        sample_embeddings = []
        for seg in selected_segments:
            emb = self.get_speaker_embedding(audio_source, seg)
            normalized = self._normalize_embedding_vector(emb)
            if normalized is None:
                continue
            sample_embeddings.append((seg, normalized))

        if not sample_embeddings:
            return None, None, []

        dim_counts = {}
        for _, vec in sample_embeddings:
            dim_counts[int(vec.size)] = int(dim_counts.get(int(vec.size), 0)) + 1
        target_dim = max(dim_counts.items(), key=lambda item: (item[1], item[0]))[0]
        sample_embeddings = [(seg, vec) for seg, vec in sample_embeddings if int(vec.size) == int(target_dim)]
        if not sample_embeddings:
            return None, None, []

        centroid = np.mean(np.stack([vec for _, vec in sample_embeddings], axis=0), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            return None, None, []
        centroid = (centroid / centroid_norm).astype(np.float32, copy=False)
        primary_segment = max((seg for seg, _ in sample_embeddings), key=lambda seg: float(seg.duration))
        return centroid, primary_segment, sample_embeddings

    def get_speaker_embedding(self, audio_source, segment):
        # Extract embedding for a specific segment
        # We need to crop the audio or use the inference with cropping
        # Pyannote Inference with window="whole" expects a file path or waveform.
        # But we want a segment. 
        # Efficient way: Load audio once, crop in memory.
        # For simplicity/speed prototype: rely on Inference to handle cropping if passed Excerpt?
        # Inference usually takes (file, segment)
        try:
            # Reuse a preloaded pyannote audio object when provided. Falling back to a path
            # here is very expensive on long episodes because it decodes the full file.
            if isinstance(audio_source, (str, Path)):
                audio_input = self._load_audio_for_pyannote(str(audio_source))
            else:
                audio_input = audio_source
            embedding = self.embedding_inference.crop(audio_input, segment)
            # embedding is (1, dimension) or similar. We want mean if it returns multiple frames?
            # crop usually returns one embedding for the window if window="whole" but we are passing a segment...
            # Actually, pyannote.audio `Inference` with `window="whole"` on a `crop` returns the embedding for that crop.
            return embedding
        except Exception as e:
            log_verbose(f"Embedding error for segment {segment}: {e}")
            return None

    def identify_speaker(self, session: Session, channel_id: int, embedding, threshold: float = 0.5):
        """Find best speaker/profile match by cosine distance using cached normalized vectors."""
        import numpy as np

        if not channel_id or embedding is None:
            return None, None, float("inf")

        try:
            q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except Exception:
            return None, None, float("inf")
        if q.size == 0 or not np.all(np.isfinite(q)):
            return None, None, float("inf")
        q_norm = float(np.linalg.norm(q))
        if not np.isfinite(q_norm) or q_norm <= 1e-12:
            return None, None, float("inf")
        q = q / q_norm

        cache = self._get_speaker_match_cache(session, int(channel_id))
        matrix = cache.get("matrix")
        profile_ids = cache.get("profile_ids")
        speaker_ids = cache.get("speaker_ids")
        dim = int(cache.get("dim") or 0)
        if matrix is None or profile_ids is None or speaker_ids is None or dim <= 0 or int(q.size) != dim:
            return None, None, float("inf")

        # matrix rows are normalized => cosine distance = 1 - dot(row, q)
        sims = matrix @ q
        if sims.size == 0:
            return None, None, float("inf")
        finite_mask = np.isfinite(sims)
        if not np.any(finite_mask):
            return None, None, float("inf")
        if not np.all(finite_mask):
            sims = sims.copy()
            sims[~finite_mask] = -np.inf
        best_idx = int(np.argmax(sims))
        best_dist = float(1.0 - float(sims[best_idx]))
        best_profile_id = int(profile_ids[best_idx])
        best_speaker_id = int(speaker_ids[best_idx])

        if best_dist < threshold:
            return session.get(Speaker, best_speaker_id), session.get(SpeakerEmbedding, best_profile_id), best_dist
        return None, None, best_dist
