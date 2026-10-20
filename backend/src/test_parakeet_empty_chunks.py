import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from src.services.ingestion.transcription import TranscriptionEngineMixin


@pytest.mark.parametrize(
    "speech_result,text,chunk_index,accepted",
    [
        ([], "", 24, True),
        ([{"start": 0, "end": 100}], "", 24, False),
        (RuntimeError("VAD unavailable"), "", 24, False),
        ([], "recognized speech without timestamps", 24, False),
        ([], "", None, False),
    ],
)
def test_empty_chunk_requires_independent_no_speech_confirmation(
    monkeypatch, speech_result, text, chunk_index, accepted
):
    monkeypatch.setenv("PARAKEET_REQUIRE_WORD_TIMESTAMPS", "true")
    monkeypatch.setenv("PARAKEET_ALLOW_WHISPER_FALLBACK", "true")
    vad_calls = []

    def detect(audio):
        vad_calls.append(audio)
        if isinstance(speech_result, Exception):
            raise speech_result
        return speech_result

    vad = ModuleType("faster_whisper.vad")
    vad.get_speech_timestamps = detect
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)

    service = TranscriptionEngineMixin()
    service.device = "cpu"
    service._parakeet_dynamic_batch_cap = None
    waveform = object()
    hypothesis = SimpleNamespace(
        text=text, timestamp={"char": [], "word": [], "segment": [], "timestep": []}
    )
    service.parakeet_model = SimpleNamespace(transcribe=lambda *a, **kw: [hypothesis])
    service._load_parakeet_model = lambda *a: None
    service._convert_audio_for_parakeet = lambda path: (path, False)
    service._probe_audio_duration_seconds = lambda path: 5.726
    service._load_waveform_for_parakeet = lambda path: waveform
    service._cuda_memory_snapshot = lambda: {}
    service._upsert_job_payload_fields = lambda *a: None
    service._set_parakeet_decoding_profile = lambda *a, **kw: None
    service._record_job_stage_start = lambda *a: None
    service._clear_cuda_cache = lambda: None

    def transcribe():
        return service._transcribe_with_parakeet(
            Path("tail.wav"), start_time_offset=13800,
            forced_batch_size=1, progress_chunk_index=chunk_index,
        )

    if accepted:
        segments, duration = transcribe()
        assert segments == []
        assert duration == pytest.approx(13805.726)
        assert vad_calls == [waveform]
    else:
        with pytest.raises(RuntimeError, match="low word timestamp coverage"):
            transcribe()
