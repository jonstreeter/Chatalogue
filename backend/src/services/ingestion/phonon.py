"""Bounded app-owned Phonon adapter and shared word-clock validation."""
from __future__ import annotations

import io
import math
import wave
from .phonon_runtime import MAX_INPUT_BYTES

MAX_BODY = MAX_INPUT_BYTES
CHUNK_SECONDS = 600
OVERLAP_SECONDS = 0.35
SAMPLE_RATE = 16000


class PhononError(RuntimeError):
    """Safe to display: never includes runtime bodies, credentials or URLs."""


class ManagedPhononClient:
    """Job-scoped adapter; the service-owned worker outlives each client/job."""

    def __init__(self, manager, timeout=None):
        self.manager = manager
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass  # Explicit model purge, GPU phase handoff and app shutdown own lifetime.

    def transcribe(self, audio: bytes, duration: float):
        from .phonon_runtime import PhononRuntimeError
        try:
            data = self.manager.transcribe(audio, duration, self.timeout)
            words = validate_result(data, duration)
            return words, {**data.get("runtime", {}), "segments": data.get("segments") or []}
        except PhononRuntimeError as error:
            raise PhononError(str(error)) from None


def wav_bytes(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return out.getvalue()


def validate_result(data: dict, duration: float) -> list[dict]:
    """Validate runtime word and segment clocks before accepting any output."""
    try:
        if not isinstance(data, dict) or data.get("truncated"):
            raise ValueError
        reported = float(data.get("duration", duration))
        # Runtime duration is rounded to milliseconds.
        if not math.isfinite(reported) or abs(reported - duration) > 0.001:
            raise ValueError
        words = data.get("words") or []
        segments = data.get("segments") or []
        if not isinstance(words, list) or not isinstance(segments, list):
            raise ValueError
        for items in (words, segments):
            previous = 0.0
            previous_end = 0.0
            for item in items:
                start, end = float(item["start"]), float(item["end"])
                if not (math.isfinite(start) and math.isfinite(end)
                        and 0 <= start <= end <= duration and start >= previous and end >= previous_end):
                    raise ValueError
                previous = start
                previous_end = end
        if not words and (str(data.get("text") or "").strip()
                          or any(str(s.get("text") or "").strip() for s in segments)):
            raise ValueError
        result = []
        for word in words:
            text = str(word.get("word", word.get("text", ""))).strip()
            if not text:
                raise ValueError
            result.append({"start": float(word["start"]), "end": float(word["end"]), "word": text})
        return result
    except (ValueError, TypeError, KeyError, AttributeError):
        raise PhononError("Phonon returned invalid or missing word timestamps, duration, or truncated output") from None
