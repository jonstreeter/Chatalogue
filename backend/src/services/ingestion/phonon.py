"""Bounded, authenticated transport to an operator-managed local Phonon runtime."""
from __future__ import annotations

import io
import ipaddress
import math
import os
from pathlib import Path
import wave
from urllib.parse import urlsplit

import httpx

# Verified CUDA 1.0.8 server limits; PCM chunks use the existing app window.
MAX_BODY = 32 * 1024 * 1024
CHUNK_SECONDS = 600
OVERLAP_SECONDS = 0.35
SAMPLE_RATE = 16000


class PhononError(RuntimeError):
    """Safe to display: never includes runtime bodies, credentials or URLs."""


def validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        local = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
        if (not local or parsed.scheme not in {"http", "https"} or parsed.username
                or parsed.password or parsed.query or parsed.fragment or not parsed.port
                and parsed.netloc.endswith(":")):
            raise ValueError
    except ValueError:
        raise ValueError("Phonon URL must be loopback HTTP(S), without credentials, query or fragment") from None
    return value.rstrip("/")


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


class PhononClient:
    def __init__(self, url="http://127.0.0.1:8010", api_key="", timeout=None, *, transport=None):
        self.url = validate_url(url)
        if timeout is not None and (not math.isfinite(float(timeout)) or float(timeout) <= 0):
            raise ValueError("Phonon inference deadline must be a positive finite number")
        self.timeout = httpx.Timeout(None if timeout is None else float(timeout), connect=5.0)
        self.client = httpx.Client(
            trust_env=False, follow_redirects=False, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.client.close()

    def _send(self, request):
        try:
            response = self.client.send(request)
        except httpx.TimeoutException:
            raise PhononError("Phonon request timed out; check runtime or inference deadline") from None
        except httpx.HTTPError:
            raise PhononError("Phonon runtime is unreachable; start the local service and check its URL") from None
        if response.status_code != 200:
            reason = {401: "authentication failed", 403: "authentication refused", 413: "upload exceeds runtime limit",
                      429: "runtime busy", 503: "runtime loading, busy, or unavailable", 500: "runtime decode failed (check GPU memory)"}.get(
                          response.status_code, "unexpected runtime response (redirects are refused)")
            raise PhononError(f"Phonon {reason} (HTTP {response.status_code})")
        try:
            return response.json()
        except ValueError:
            raise PhononError("Phonon returned invalid JSON") from None

    def transcribe(self, audio: Path | bytes, duration: float):
        if not math.isfinite(duration) or not 0 < duration <= 7200:
            raise PhononError("Phonon input must be nonempty and at most two hours")
        if isinstance(audio, Path):
            if audio.stat().st_size >= MAX_BODY:
                raise PhononError("Phonon upload exceeds runtime limit")
            blob = audio.read_bytes()
        else:
            blob = audio
        if len(blob) >= MAX_BODY:
            raise PhononError("Phonon upload exceeds runtime limit")
        request = self.client.build_request(
            "POST", self.url + "/v1/audio/transcriptions",
            data={"model": "phonon-2", "response_format": "verbose_json", "timestamp_granularities[]": "word"},
            files={"file": ("chunk.wav", blob, "audio/wav")}, timeout=self.timeout,
        )
        length = int(request.headers.get("content-length", "0"))
        if not 0 < length <= MAX_BODY or "transfer-encoding" in request.headers:
            raise PhononError("Phonon multipart upload exceeds runtime limit or lacks Content-Length")
        data = self._send(request)
        words = validate_result(data, duration)
        return words, {**(data.get("x_phonon") or data.get("x_fermion") or {}), "segments": data.get("segments") or []}

    def readiness(self):
        data = self._send(self.client.build_request("GET", self.url + "/health"))
        if not isinstance(data, dict) or data.get("status") not in {"ok", "ready"}:
            raise PhononError("Phonon runtime is loading or not ready")
        if str(data.get("model", "")).lower() not in {"phonon-2", "fermionresearch/phonon-2"}:
            raise PhononError("Phonon runtime serves the wrong model; phonon-2 is required")
        # CUDA health is unauthenticated. A one-second silent decode proves auth.
        self.transcribe(wav_bytes(bytes(SAMPLE_RATE * 2)), 1.0)
        return {key: data[key] for key in ("model", "version", "gpu", "device", "path", "dtype", "cuda_graph_steps",
                                          "warm_buckets", "uptime_seconds", "stats") if key in data}


def test_connection(url=None, api_key=None):
    """Probe unsaved values without importing host ASR models or persisting settings."""
    result = {"requested_engine": "phonon", "resolved_engine": "phonon"}
    try:
        with PhononClient(
            url if url is not None else os.getenv("PHONON_URL", "http://127.0.0.1:8010"),
            api_key if api_key is not None else os.getenv("PHONON_API_KEY", ""),
        ) as client:
            details = client.readiness()
        return {**result, "status": "ok", **details, "device": details.get("device") or details.get("gpu")}
    except (ValueError, PhononError) as error:
        return {**result, "status": "error", "error": str(error)}
