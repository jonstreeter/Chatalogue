"""Optional local-runtime transport and checkpoint regression tests."""
import httpx
import pytest
import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from src.services.ingestion.phonon import PhononClient, PhononError, validate_url, validate_result, wav_bytes


@pytest.mark.parametrize("url", ["https://example.com", "http://127.0.0.1@evil.test", "http://127.0.0.1/?key=secret", "file:///x", "http://0.0.0.0", "http://localhost/#secret"])
def test_remote_urls_refused(url):
    with pytest.raises(ValueError):
        validate_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8010", "http://localhost:8010", "http://[::1]:8010"])
def test_loopback_urls(url):
    assert validate_url(url) == url


def test_auth_bounded_multipart_and_reuse():
    seen = []

    def server(req):
        seen.append(req)
        assert req.headers["authorization"] == "Bearer secret"
        assert int(req.headers["content-length"]) == len(req.read()) < 32 * 1024 * 1024
        assert b'phonon-2' in req.content and b'timestamp_granularities[]' in req.content
        return httpx.Response(200, json={"duration": 1, "text": "hi", "words": [{"text": "hi", "start": 0, "end": .5}]})

    with PhononClient(api_key="secret", transport=httpx.MockTransport(server)) as client:
        for _ in range(2):
            words, _ = client.transcribe(wav_bytes(bytes(32000)), 1)
            assert words == [{"word": "hi", "start": 0., "end": .5}]
    assert len(seen) == 2


@pytest.mark.parametrize("status", [302, 401, 403, 413, 429, 500, 503])
def test_http_errors_never_leak_body_or_key(status):
    with PhononClient(api_key="secret", transport=httpx.MockTransport(lambda r: httpx.Response(status, text="secret"))) as client:
        with pytest.raises(PhononError) as error:
            client.transcribe(b"wav", 1)
    assert "secret" not in str(error.value)


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


def test_upload_checks_encoded_length_before_sending(monkeypatch):
    from src.services.ingestion import phonon
    sent = []
    monkeypatch.setattr(phonon, "MAX_BODY", 100)
    with PhononClient(transport=httpx.MockTransport(lambda request: sent.append(request))) as client:
        with pytest.raises(PhononError, match="multipart"):
            client.transcribe(b"small", 1.)
    assert not sent


def test_timeout_unreachable_and_invalid_json_are_safe():
    for response in (httpx.ConnectError("secret"), httpx.ReadTimeout("secret"), httpx.Response(200, text="not-json secret")):
        def handler(request):
            if isinstance(response, Exception):
                raise response
            return response
        with PhononClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(PhononError) as error:
                client.transcribe(b"wav", 1.)
        assert "secret" not in str(error.value)


def test_readiness_proves_auth_even_when_health_public():
    calls = []

    def server(req):
        calls.append(req.method)
        return httpx.Response(200, json={"model": "phonon-2", "status": "ok"}) if req.method == "GET" else httpx.Response(401)

    with PhononClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(PhononError, match="authentication"):
            client.readiness()
    assert calls == ["GET", "POST"]


@pytest.mark.parametrize("model,status,expected", [("phonon-1", "ok", "wrong model"), ("phonon-2", "loading", "not ready")])
def test_readiness_rejects_wrong_model_and_loading(model, status, expected):
    with PhononClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": status, "model": model}))) as client:
        with pytest.raises(PhononError, match=expected):
            client.readiness()


def test_cpu_runtime_model_identity_and_vendor_metadata():
    def server(request):
        return httpx.Response(200, json={"status": "ok", "model": "FermionResearch/Phonon-2", "version": "0.2.11", "device": "cpu"}) if request.method == "GET" else httpx.Response(200, json={"duration": 1, "text": "", "words": [], "x_fermion": {"backend": "cpu"}})
    with PhononClient(transport=httpx.MockTransport(server)) as client:
        assert client.readiness()["device"] == "cpu"


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
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "phonon")
    monkeypatch.setenv("MULTILINGUAL_ROUTING_ENABLED", "false")
    service = IngestionService()
    yield service
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

    monkeypatch.setattr(phonon.PhononClient, "transcribe", transcribe)
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
    monkeypatch.setattr(phonon.PhononClient, "transcribe", lambda *args: ([], {}))
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
    monkeypatch.setattr(phonon.PhononClient, "transcribe", transcribe)
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
    monkeypatch.setattr(phonon.PhononClient, "transcribe", silent)
    result, duration = service._process_transcribe_phase(Video(id=9, youtube_id="long", title="silent", duration=7201), audio)
    assert result == [] and duration == 7201
    assert len(calls) == 13
    assert all(size < phonon.MAX_BODY and seconds <= 600.35 for size, seconds in calls)
    assert service._load_phonon_partial(9)["completed_until_seconds"] == 7201
    assert json.loads((tmp_path / "silent_transcript_raw.json").read_text())["transcription_engine_used"] == "phonon"


def test_settings_unsaved_probe_and_validation(tmp_path, monkeypatch):
    from src.routers import settings
    from src.schemas import Settings, TranscriptionEngineTestRequest
    path = tmp_path / "scratch.env"
    path.write_text("")
    monkeypatch.setattr(settings, "ENV_PATH", path)
    monkeypatch.setattr(settings, "get_ingestion_service", lambda: None)
    from types import SimpleNamespace
    monkeypatch.setattr(settings, "_main", lambda: SimpleNamespace(configure_logging=lambda: None))
    # Isolate every process-local setting update from the rest of the suite.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in ("PHONON_URL", "PHONON_API_KEY", "PHONON_REQUEST_TIMEOUT_SECONDS"):
        monkeypatch.setenv(key, "")
    settings.update_settings(Settings(hf_token="", transcription_engine="phonon", phonon_url="http://localhost:8010",
        phonon_api_key="test-only-key", phonon_request_timeout_seconds=90))
    saved = settings.get_settings()
    assert saved.transcription_engine == "phonon" and saved.phonon_api_key == "test-only-key"
    assert saved.phonon_request_timeout_seconds == 90
    settings.update_settings(Settings(hf_token=""))
    assert settings.get_settings().phonon_api_key == "test-only-key"
    before = path.read_bytes()
    result = settings.test_transcription_engine(TranscriptionEngineTestRequest(engine="phonon", phonon_url="https://example.com", phonon_api_key="unsaved-secret"))
    assert result["status"] == "error" and "unsaved-secret" not in str(result)
    assert path.read_bytes() == before
    with pytest.raises(Exception, match="loopback"):
        settings.update_settings(Settings(hf_token="", phonon_url="https://example.com"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan")])
def test_settings_invalid_deadline(deadline):
    from src.schemas import Settings
    with pytest.raises(ValueError):
        Settings(hf_token="", phonon_request_timeout_seconds=deadline)
