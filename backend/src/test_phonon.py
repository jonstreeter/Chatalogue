"""Managed Phonon adapter, settings and checkpoint regression tests."""
import httpx
import pytest
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from src.services.ingestion.phonon import PhononError, validate_result, wav_bytes


@pytest.mark.parametrize("data", [
    {"text": "speech"}, {"duration": 2}, {"truncated": True},
    {"words": [{"word": "bad", "start": float("nan"), "end": .5}]},
    {"words": [{"word": "bad", "start": .5, "end": 2}]},
    {"words": [{"word": "bad", "start": .5, "end": .1}]},
])
def test_invalid_responses(data):
    with pytest.raises(PhononError):
        validate_result(data, 1)


def test_silence_is_valid():
    assert validate_result({"text": "", "words": [], "segments": [], "duration": 1}, 1) == []


@pytest.fixture
def isolated_service(tmp_path, monkeypatch):
    from sqlmodel import SQLModel, create_engine
    from sqlalchemy.pool import StaticPool
    from src.services.ingestion import IngestionService, runtime
    from src.services.ingestion import pipeline

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(runtime, "engine", engine)
    monkeypatch.setattr(runtime, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(runtime, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(runtime, "PHONON_RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "phonon")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "false")
    service = IngestionService()
    yield service
    service.shutdown_phonon()
    engine.dispose()


def test_checkpoint_silence_and_interrupted_tail(isolated_service):
    service = isolated_service
    service._save_phonon_partial(1, [], 1000., 600.)
    with service._get_temp_transcript_jsonl_path(1).open("ab") as stream:
        stream.write(b'{"interrupted":')
    loaded = service._load_phonon_partial(1)
    assert loaded["completed_until_seconds"] == 600
    assert loaded["segments"] == []
    assert service._get_temp_transcript_jsonl_path(1).read_bytes() == b""


def test_managed_adapter_reuses_runtime_and_ignores_legacy_url(monkeypatch):
    from src.services.ingestion.phonon import ManagedPhononClient
    calls = []

    class Manager:
        def transcribe(self, audio, duration, timeout):
            calls.append((audio, duration, timeout))
            return {"duration": duration, "text": "hello", "words": [{"text": "hello", "start": 0, "end": .5}],
                    "runtime": {"device": "cuda", "version": "native"}}

    monkeypatch.setenv("PHONON_URL", "https://must-never-connect.invalid")
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("managed inference must not open HTTP"))
    manager = Manager()
    for _ in range(2):
        with ManagedPhononClient(manager, timeout=10) as client:
            words, details = client.transcribe(b"owned audio", 1.)
            assert words[0]["word"] == "hello"
            assert details["device"] == "cuda"
    assert len(calls) == 2 and calls[0][2] == 10


def test_purge_releases_worker_without_host_device(isolated_service):
    from types import SimpleNamespace
    service = isolated_service
    calls = []
    service._phonon_runtime = SimpleNamespace(status=lambda: {"model_loaded": True},
        unload=lambda **_: calls.append("unload"), shutdown=lambda: None)
    assert service.device is None
    result = service.purge_loaded_models()
    assert result["purged_phonon"] is True and calls == ["unload"]


def test_busy_worker_purge_preserves_other_models(isolated_service):
    from types import SimpleNamespace
    from src.services.ingestion.phonon_runtime import PhononRuntimeError
    service = isolated_service
    whisper = service.whisper_model = object()
    parakeet = service.parakeet_model = object()

    def busy(**_):
        raise PhononRuntimeError("Phonon is busy")

    service._phonon_runtime = SimpleNamespace(status=lambda: {"model_loaded": True}, unload=busy, shutdown=lambda: None)
    with pytest.raises(PhononRuntimeError, match="busy"):
        service.purge_loaded_models()
    assert service.whisper_model is whisper and service.parakeet_model is parakeet


def test_cpu_only_setup_does_not_block_competing_engine(isolated_service):
    from types import SimpleNamespace
    from src.services.ingestion.phonon_runtime import PhononBusyError
    service = isolated_service

    def busy(**_):
        raise PhononBusyError("setup busy")

    service._phonon_runtime = SimpleNamespace(status=lambda: {"state": "installing", "model_loaded": False,
        "process_running": False}, unload=busy, shutdown=lambda: None)
    assert service._release_phonon_model("before_whisper") is False
    with pytest.raises(PhononBusyError):
        service.purge_loaded_models()


@pytest.mark.parametrize("retained,host_reserved,expected_purge", [(True, 0, False), (False, 0, True), (True, 16, True)])
def test_end_of_job_recovery_distinguishes_owned_worker_memory(isolated_service, monkeypatch, retained, host_reserved, expected_purge):
    from types import SimpleNamespace
    service = isolated_service
    monkeypatch.setenv("CUDA_HEADROOM_RECOVERY_DROP_GB", "1.5")
    monkeypatch.setenv("CUDA_HEADROOM_RECOVERY_RESERVED_RATIO", "0.45")
    service.device = "cuda"
    service._phonon_runtime = SimpleNamespace(status=lambda: {"model_loaded": retained}, shutdown=lambda: None)
    gib = 1024 ** 3
    monkeypatch.setattr(service, "_cuda_memory_snapshot", lambda: {"free": 22 * gib, "total": 32 * gib,
        "reserved": host_reserved * gib, "allocated": host_reserved * gib})
    monkeypatch.setattr(service, "_upsert_job_payload_fields", lambda *_: None)
    monkeypatch.setattr(service, "_record_cuda_health_event", lambda *_, **__: None)
    purged = []
    monkeypatch.setattr(service, "purge_loaded_models", lambda **_: purged.append(True))
    service._maybe_recover_cuda_headroom(24 * gib)
    assert bool(purged) == expected_purge


@pytest.mark.parametrize("method,reason", [("_load_whisper_model", "before_whisper"), ("_load_parakeet_model", "before_parakeet")])
def test_competing_gpu_loader_releases_phonon_first(isolated_service, monkeypatch, method, reason):
    service = isolated_service
    monkeypatch.setattr(service, "_ensure_device", lambda: setattr(service, "device", "cuda"))
    monkeypatch.setattr(service, "_ensure_ctranslate2_pkg_resources", lambda: None)
    calls = []

    def released(label, job_id=None):
        calls.append(label)
        raise RuntimeError("verified handoff; stop before host model load")

    monkeypatch.setattr(service, "_release_phonon_model", released)
    with pytest.raises(RuntimeError, match="verified handoff"):
        getattr(service, method)()
    assert calls == [reason]


def test_actual_diarization_phase_hands_off_owned_worker(isolated_service, tmp_path):
    from types import SimpleNamespace
    from src.db.database import Video
    service = isolated_service
    calls = []

    def unload(**_):
        calls.append("worker unload reached before diarization model load")
        raise RuntimeError("handoff boundary reached")

    service._phonon_runtime = SimpleNamespace(status=lambda: {"model_loaded": True}, unload=unload, shutdown=lambda: None)
    with pytest.raises(RuntimeError, match="handoff boundary"):
        service._process_diarize_phase(Video(id=77, youtube_id="isolated-diarize", title="handoff"), tmp_path / "audio.wav", [])
    assert len(calls) == 1


def test_checkpoint_atomic_failure_recovers_prior_commit(isolated_service, monkeypatch):
    service = isolated_service
    first = {"start": 0., "end": 1., "text": "hi", "words": [{"word": "hi", "start": 0., "end": 1.}]}
    second = {"start": 600., "end": 601., "text": "next", "words": [{"word": "next", "start": 600., "end": 601.}]}
    service._save_phonon_partial(1, [first], 1200., 600.)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk failure")))
        with pytest.raises(OSError):
            service._save_phonon_partial(1, [first, second], 1200., 1200.)
    loaded = service._load_phonon_partial(1)
    assert loaded["completed_until_seconds"] == 600.
    assert loaded["segments"] == [first]
    service._save_phonon_partial(1, [first, second], 1200., 1200.)
    assert len(service._load_phonon_partial(1)["segments"]) == 2


def test_checkpoint_rejects_missing_committed_records_and_unknown_engine(isolated_service):
    import json
    service = isolated_service
    path = service._get_temp_transcript_path(1)
    path.write_text(json.dumps({"engine": "whisper"}))
    with pytest.raises(PhononError, match="another or unknown"):
        service._load_phonon_partial(1)
    assert path.exists()
    path.write_text(json.dumps({"engine": "phonon", "segment_count": 1, "completed_until_seconds": 1, "total_duration": 2}))
    service._get_temp_transcript_jsonl_path(1).write_text("")
    with pytest.raises(PhononError, match="incomplete or invalid"):
        service._load_phonon_partial(1)


def test_orphan_jsonl_without_engine_metadata_is_preserved(isolated_service):
    records = isolated_service._get_temp_transcript_jsonl_path(1)
    records.write_bytes(b'unknown engine output')
    with pytest.raises(PhononError, match="unknown engine provenance"):
        isolated_service._load_phonon_partial(1)
    assert records.read_bytes() == b'unknown engine output'


def test_first_chunk_metadata_failure_keeps_owned_zero_boundary(isolated_service, monkeypatch):
    service = isolated_service
    original = os.replace
    calls = []
    def interrupted(source, target):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("interrupted first chunk commit")
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises(OSError):
            service._save_phonon_partial(1, [], 10., 10.)
    assert service._load_phonon_partial(1)["completed_until_seconds"] == 0.
    service._save_phonon_partial(1, [], 10., 10.)
    assert service._load_phonon_partial(1)["completed_until_seconds"] == 10.


def test_actual_phase_failure_resume_and_absolute_seam(isolated_service, tmp_path, monkeypatch):
    import json
    from src.services.ingestion import phonon
    from src.db.database import Video

    service = isolated_service
    audio = tmp_path / "source.wav"
    audio.write_bytes(wav_bytes(bytes(32000 * 1201)))
    calls = []
    failing = [True]

    def transcribe(client, blob, duration):
        calls.append(duration)
        if len(calls) == 2 and failing[0]:
            raise PhononError("runtime unavailable")
        return [{"word": "hello", "start": 0.1, "end": .2},
                {"word": "seam", "start": duration - .3, "end": duration - .1}], {}

    monkeypatch.setattr(phonon.ManagedPhononClient, "transcribe", transcribe)
    video = Video(id=1, youtube_id="isolated-phonon", title="test", duration=1201)
    with pytest.raises(PhononError, match="unavailable"):
        service._process_transcribe_phase(video, audio)
    assert service._load_phonon_partial(1)["completed_until_seconds"] == pytest.approx(600.175)
    failing[0] = False
    segments, duration = service._process_transcribe_phase(video, audio)
    words = [w for segment in segments for w in segment.words]
    assert duration == 1201
    assert calls[0] == pytest.approx(600.35)
    assert len(calls) == 4
    assert len([w for w in words if w.word == "hello" and w.start < 1]) == 1
    assert all(0 <= w.start <= w.end <= duration for w in words)
    assert [w.start for w in words] == sorted(w.start for w in words)
    raw = json.loads((tmp_path / "test_transcript_raw.json").read_text())
    assert raw["transcription_engine_used"] == "phonon"
    before = len(calls)
    service._process_transcribe_phase(video, audio)
    assert len(calls) == before  # completed checkpoint reused


def test_phonon_selection_ignores_host_cuda_health(isolated_service):
    isolated_service._cuda_unhealthy_reason = "host CUDA fault"
    assert isolated_service._select_transcription_engine() == "phonon"


@pytest.mark.parametrize("state", ["paused", "cancelled"])
def test_pause_cancel_after_response_preserves_previous_boundary(isolated_service, tmp_path, monkeypatch, state):
    from src.db.database import Video
    from src.services.ingestion import phonon
    from src.services.ingestion.exceptions import JobPausedException, JobCancelledException
    audio = tmp_path / "interrupt.wav"
    audio.write_bytes(wav_bytes(bytes(32000)))
    exception = JobPausedException if state == "paused" else JobCancelledException
    checks = []
    def check(job):
        checks.append(job)
        if len(checks) == 3:
            raise exception(state)
    monkeypatch.setattr(isolated_service, "_check_job_not_paused", check)
    monkeypatch.setattr(phonon.ManagedPhononClient, "transcribe", lambda *args: ([], {}))
    with pytest.raises(exception):
        isolated_service._process_transcribe_phase(Video(id=8, youtube_id="interrupt", title="interrupt"), audio)
    assert not isolated_service._get_temp_transcript_path(8).exists()
    assert not (tmp_path / "interrupt_transcript_raw.json").exists()


def test_runtime_segment_boundaries_and_force_retranscription(isolated_service, tmp_path, monkeypatch):
    from sqlmodel import Session
    from src.db.database import Video, Job
    from src.services.ingestion import phonon, runtime
    audio = tmp_path / "segments.wav"
    audio.write_bytes(wav_bytes(bytes(32000 * 2)))
    calls = []
    def transcribe(*args):
        calls.append(1)
        return [{"word": "one", "start": .1, "end": .2}, {"word": "two", "start": 1.1, "end": 1.2}], {"segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}]}
    monkeypatch.setattr(phonon.ManagedPhononClient, "transcribe", transcribe)
    video = Video(id=3, youtube_id="segments", title="segments", duration=2)
    segments, _ = isolated_service._process_transcribe_phase(video, audio)
    assert [segment.text for segment in segments] == ["one", "two"]
    with Session(runtime.engine) as session:
        session.add(Job(id=3, video_id=3, job_type="process", status="queued", payload_json='{"force_retranscription": true}'))
        session.commit()
    isolated_service._process_transcribe_phase(video, audio, job_id=3)
    assert len(calls) == 2
    with Session(runtime.engine) as session:
        payload = session.get(Job, 3).payload_json
        assert 'phonon' in payload
    cleaned = isolated_service._strip_transient_job_payload_fields({"phonon_chunk_count": 3, "keep": "value"})
    assert cleaned == {"keep": "value"}


def test_phonon_still_routes_spanish_to_whisper(isolated_service, monkeypatch):
    from src.db.database import Video
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "true")
    video = Video(youtube_id="spanish", title="Entrevista en Español", description="Conversación completa")
    route = isolated_service._resolve_transcription_route(video, None)
    assert route["requested_engine"] == "phonon"
    assert route["engine"] == "whisper" and route["multilingual_route_applied"]


def test_multihour_silence_uses_bounded_requests_and_completed_offsets(isolated_service, tmp_path, monkeypatch):
    import wave
    import json
    from src.db.database import Video
    from src.services.ingestion import phonon
    service = isolated_service
    calls = []
    # Sparse PCM source avoids constructing a multi-hour byte array in memory.
    audio = tmp_path / "multi.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.setnframes(7201 * 16000)
        writer.writeframesraw(b"\0\0")
        writer._file.seek(44 + 7201 * 32000 - 1)
        writer._file.write(b"\0")
        writer._datawritten = 7201 * 32000
    # Exercise the actual phase, while copying the normalized sparse source.
    import subprocess
    import shutil
    from types import SimpleNamespace

    def normalize(cmd, **kwargs):
        shutil.copyfile(audio, cmd[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", normalize)
    def silent(client, blob, duration):
        calls.append((len(blob), duration))
        return [], {}
    monkeypatch.setattr(phonon.ManagedPhononClient, "transcribe", silent)
    result, duration = service._process_transcribe_phase(Video(id=9, youtube_id="long", title="silent", duration=7201), audio)
    assert result == [] and duration == 7201
    assert len(calls) == 13
    assert all(size < phonon.MAX_BODY and seconds <= 600.35 for size, seconds in calls)
    assert service._load_phonon_partial(9)["completed_until_seconds"] == 7201
    assert json.loads((tmp_path / "silent_transcript_raw.json").read_text())["transcription_engine_used"] == "phonon"


def test_settings_deprecated_connection_fields_are_noops(isolated_service, tmp_path, monkeypatch):
    from src.routers import settings
    from src.schemas import Settings, TranscriptionEngineTestRequest
    from types import SimpleNamespace
    path = tmp_path / "scratch.env"
    path.write_text("PHONON_API_KEY='saved-key'\nPHONON_URL='http://localhost:8010'\n")
    monkeypatch.setattr(settings, "ENV_PATH", path)
    monkeypatch.setattr(settings, "get_ingestion_service", lambda: None)
    monkeypatch.setattr(settings, "_main", lambda: SimpleNamespace(configure_logging=lambda: None))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("PHONON_API_KEY", "saved-key")
    monkeypatch.setenv("PHONON_URL", "http://localhost:8010")
    settings.update_settings(Settings(hf_token="", transcription_engine="phonon", phonon_url="https://ignored.invalid",
        phonon_api_key="ignored-key", phonon_request_timeout_seconds=90))
    saved = settings.get_settings()
    assert saved.transcription_engine == "phonon" and saved.model_dump()["phonon_api_key"] == ""
    assert saved.phonon_request_timeout_seconds == 90
    assert os.environ["PHONON_API_KEY"] == "saved-key"
    assert "saved-key" in path.read_text() and "ignored-key" not in path.read_text()
    settings.update_settings(Settings(hf_token=""))
    before = path.read_bytes()
    state = {"state": "installed", "detail": "unloaded", "model_loaded": False, "process_running": False, "runtime": {}}

    def request_load(before_load=None):
        state.update(state="loading", detail="loading")

    manager = SimpleNamespace(status=lambda: dict(state), request_load=request_load,
        unload=lambda: state.update(state="installed", model_loaded=False, process_running=False), shutdown=lambda: None)
    monkeypatch.setattr(isolated_service, "_get_phonon_runtime", lambda: manager)
    monkeypatch.setattr(settings, "get_ingestion_service", lambda: isolated_service)
    result = settings.test_transcription_engine(TranscriptionEngineTestRequest(engine="phonon", phonon_url="https://ignored.invalid", phonon_api_key="unsaved-secret"))
    assert result["state"] == "loading" and "unsaved-secret" not in str(result)
    assert os.environ["TRANSCRIPTION_ENGINE"] == "auto"  # Testing unsaved selection does not activate it.
    assert path.read_bytes() == before
    result = settings.test_transcription_engine(TranscriptionEngineTestRequest(engine="phonon", phonon_action="unload"))
    assert result["state"] == "installed" and not result["model_loaded"]


def test_settings_model_test_refuses_active_pipeline(isolated_service, monkeypatch):
    monkeypatch.setattr(isolated_service, "_has_active_pipeline_gpu_work", lambda: True)
    result = isolated_service.test_phonon_engine("load")
    assert result["status"] == "error" and "active" in result["error"]
    assert isolated_service._phonon_runtime.status()["process_running"] is False


@pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan")])
def test_settings_invalid_deadline(deadline):
    from src.schemas import Settings
    with pytest.raises(ValueError):
        Settings(hf_token="", phonon_request_timeout_seconds=deadline)
