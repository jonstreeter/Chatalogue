"""Owned installer recovery tests, using scratch files and no production DB."""
import hashlib
import json
import os
from pathlib import Path
import threading
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from src.services.ingestion import moss_runtime as runtime


@pytest.fixture
def installer(tmp_path, monkeypatch):
    body = b"test-model"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(runtime, "PINNED_FILES", {"model/test.bin": (len(body), digest)})
    monkeypatch.setattr(runtime.MossInstallation, "_check_platform", staticmethod(lambda: None))
    install = runtime.MossInstallation(tmp_path / "moss")
    cache = install.root / "downloads"
    cache.mkdir(parents=True)
    (cache / digest).write_bytes(body)

    def run(command):
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[3]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"private-interpreter")

    monkeypatch.setattr(install, "_run", run)
    monkeypatch.setattr(install, "_prepare_wheels", lambda python, lock, stage: (stage, lock))
    return install


def test_install_publication_and_offline_reuse(installer, monkeypatch):
    assert installer.status()["state"] == "absent"
    installed = installer.ensure()
    assert installed.python.is_file()
    assert installed.model.joinpath("test.bin").read_bytes() == b"test-model"
    assert installer.status()["state"] == "installed"
    manifest = json.loads((installed.directory / "installed.json").read_text())
    assert manifest["requirements_sha256"] == installer.version
    monkeypatch.setattr(installer, "_download", lambda _: pytest.fail("valid offline installation must not download"))
    monkeypatch.setattr(installer, "_run", lambda _: pytest.fail("valid offline installation must not run pip"))
    assert installer.ensure() == installed


def test_interrupted_install_never_publishes_and_retry_works(installer, monkeypatch):
    original = installer._run
    monkeypatch.setattr(installer, "_run", lambda _: (_ for _ in ()).throw(OSError("private details")))
    with pytest.raises(runtime.MossRuntimeError, match="disk space"):
        installer.ensure()
    assert installer.status()["state"] == "error"
    assert not installer.destination.exists()
    assert not list((installer.root / "installs").glob(".staging-*"))
    monkeypatch.setattr(installer, "_run", original)
    assert installer.ensure().python.is_file()


def test_failed_new_version_preserves_previous_install(installer, monkeypatch):
    installed = installer.ensure()
    monkeypatch.setattr(installer, "version", "new-version")
    monkeypatch.setattr(installer, "destination", installed.directory.parent / "new-version")
    monkeypatch.setattr(installer, "_run", lambda _: (_ for _ in ()).throw(OSError()))
    with pytest.raises(runtime.MossRuntimeError):
        installer.ensure()
    assert installed.python.is_file()
    assert installed.directory.joinpath("installed.json").is_file()
    assert not installer.destination.exists()


def test_concurrent_setup_uses_one_install(installer, monkeypatch):
    original = installer._run
    calls = []
    barrier = threading.Barrier(2)

    def run(command):
        calls.append(command)
        original(command)

    def setup():
        barrier.wait()
        return installer.ensure()

    monkeypatch.setattr(installer, "_run", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: setup(), range(2)))
    assert first == second
    assert sum(command[1:3] == ["-m", "venv"] for command in calls) == 1


class DownloadResponse:
    def __init__(self, body, status=200, headers=None):
        import io
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, count):
        return self.body.read(count)


@pytest.mark.parametrize("range_supported", [True, False])
def test_download_resume_and_range_ignored_restart(installer, monkeypatch, range_supported):
    body = b"test-model"
    digest = hashlib.sha256(body).hexdigest()
    cache = installer.root / "downloads"
    (cache / digest).unlink()
    (cache / (digest + ".part")).write_bytes(body[:4])

    def open_url(request, **kwargs):
        assert request.headers["Range"] == "bytes=4-"
        if range_supported:
            return DownloadResponse(body[4:], 206, {"Content-Range": f"bytes 4-{len(body)-1}/{len(body)}"})
        return DownloadResponse(body)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", open_url)
    assert installer._download("model/test.bin").read_bytes() == body
    assert not (cache / (digest + ".part")).exists()


def test_checksum_failure_is_not_published(installer, monkeypatch):
    size, digest = runtime.PINNED_FILES["model/test.bin"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: DownloadResponse(b"x" * size))
    with pytest.raises(runtime.MossRuntimeError, match="checksum"):
        installer._download("model/test.bin")
    assert not (cache / digest).exists()
    assert not (cache / (digest + ".part")).exists()


def test_interrupted_transfer_preserves_resumable_prefix(installer, monkeypatch):
    _, digest = runtime.PINNED_FILES["model/test.bin"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()

    class Interrupted(DownloadResponse):
        def read(self, count):
            if self.body.tell():
                raise OSError("network detail must not reach UI")
            return self.body.read(4)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: Interrupted(b"test-model"))
    with pytest.raises(runtime.MossRuntimeError, match="interrupted or offline"):
        installer._download("model/test.bin")
    assert (cache / (digest + ".part")).read_bytes() == b"test"
    assert not (cache / digest).exists()


def test_corrupted_installed_model_is_not_reused(installer):
    installed = installer.ensure()
    installed.model.joinpath("test.bin").write_bytes(b"bad-model!")
    assert installer._installed(verify=True) is None


def test_interrupted_manifest_write_never_marks_installed(installer, monkeypatch):
    def fail(*_):
        raise OSError("interrupted manifest write")

    monkeypatch.setattr(runtime.json, "dump", fail)
    with pytest.raises(runtime.MossRuntimeError):
        installer.ensure()
    assert installer._installed() is None
    assert not installer.destination.exists()


def test_native_import_failure_is_rejected_before_publication(installer, monkeypatch):
    original = installer._run

    def run(command):
        if command[1] == "-c":
            raise runtime.MossRuntimeError("Moss runtime installation failed; retry setup")
        original(command)

    monkeypatch.setattr(installer, "_run", run)
    with pytest.raises(runtime.MossRuntimeError, match="installation failed"):
        installer.ensure()
    assert installer._installed() is None
    assert not installer.destination.exists()


def test_invalid_resume_range_is_not_appended(installer, monkeypatch):
    _, digest = runtime.PINNED_FILES["model/test.bin"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()
    partial = cache / (digest + ".part")
    partial.write_bytes(b"test")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: DownloadResponse(b"-model", 206, {"Content-Range": "wrong"}))
    with pytest.raises(runtime.MossRuntimeError, match="resume range"):
        installer._download("model/test.bin")
    assert partial.read_bytes() == b"test"


def test_requirement_snapshot_survives_source_update(installer, monkeypatch):
    original = installer._run
    supplied = []
    replacement = installer.root / "updated-requirements.txt"
    replacement.write_text("a-different-requirement==1\n")
    monkeypatch.setattr(runtime, "REQUIREMENTS", replacement)

    def run(command):
        if "-r" in command:
            supplied.append(Path(command[-1]).read_bytes())
        original(command)

    monkeypatch.setattr(installer, "_run", run)
    installer.ensure()
    assert supplied == [installer._requirements]
    assert supplied[0] == installer._requirements


def test_unsupported_platform_fails_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    installer = runtime.MossInstallation(tmp_path / "not-created")
    with pytest.raises(runtime.MossRuntimeError, match="Windows"):
        installer.ensure()
    assert not installer.root.exists()


FAKE_WORKER = 'import json,sys,os,threading\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[-1])\nfrom moss import parse_generation\nloaded=False\ncount=0\nfor line in sys.stdin:\n    m=json.loads(line)\n    op=m["operation"]\n    mode=Path(__file__).with_name("behavior").read_text()\n    result=None\n    if op=="load": loaded=True\n    if op in {"transcribe","probe"}:\n        if mode=="crash": os._exit(3)\n        if mode=="block": threading.Event().wait()\n        if mode=="invalid":\n            print("invalid-json",flush=True)\n            continue\n        count+=1\n        c={"complete":True,"reason":"eos","eos_token_id":151645,"terminal_token_id":151645,"input_tokens":20,"output_tokens":5,"max_new_tokens":131052,"context_limit":131072}\n        result={"raw_text":"hallucinated silence", "completion":c} if op=="probe" else parse_generation("[0][S01]hi[0.5]",m["sample_count"],m["sample_rate"],c)\n        if mode=="invalid-timings": result["sample_count"]=1\n    r={"pid":os.getpid(),"model_loaded":loaded,"model":"MOSS-Transcribe-Diarize","device":"cuda","dtype":"float32" if mode=="wrong-optimization" else "bfloat16","attention":"sdpa","model_revision":"704aa4a9c304e8520be88901e0d1960158ef5b15","source_revision":"e3bdd93f8ad58418e04659818efd5fd87a0bc486","request_count":count}\n    print("fake diagnostic",file=sys.stderr,flush=True)\n    reply=json.dumps({"protocol":1,"id":"wrong" if mode=="wrong-id" else m["id"],"runtime":r,"result":result})\n    print(reply,flush=True)\n    if mode=="duplicate": print(reply,flush=True)\n    if op=="shutdown": break\n'


@pytest.fixture
def owned_runtime(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Native Moss process ownership is supported on Windows only")
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER.replace("sys.argv[-1]", repr(str(Path(runtime.__file__).parent))))
    behavior = worker.with_name("behavior")
    behavior.write_text("")
    monkeypatch.setattr(runtime, "WORKER", worker)
    manager = runtime.MossRuntime(tmp_path / "cache", tmp_path / "audio")
    manager.installation.root.mkdir()
    installed = runtime.InstalledMoss(tmp_path, Path(sys.executable), tmp_path, tmp_path)

    def ensure():
        manager.installation._set_state("installed", "test runtime installed")
        return installed

    monkeypatch.setattr(manager.installation, "ensure", ensure)
    yield manager, behavior
    manager.shutdown()


def _silence():
    import io, wave
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(32000))
    return stream.getvalue()


def test_owned_worker_reuses_actual_child_and_unloads(owned_runtime):
    manager, _ = owned_runtime
    loaded = manager.load()
    pid = loaded["runtime"]["pid"]
    if sys.prefix != sys.base_prefix:
        assert pid != manager._owned.process.pid  # The real venv descendant.
    assert loaded["state"] == "ready"
    for count in (2, 3):
        result = manager.transcribe(_silence())
        assert result["runtime"]["pid"] == pid
        assert result["runtime"]["request_count"] == count
    assert "fake diagnostic" in (manager.installation.root / "worker.log").read_text()
    unloaded = manager.unload()
    assert not unloaded["model_loaded"] and not unloaded["process_running"]
    assert unloaded["runtime"] == {}
    assert not list(manager.temp_root.iterdir())


@pytest.mark.parametrize("mode,reason", [("invalid", "invalid worker data"), ("wrong-id", "invalid worker response"), ("crash", "worker exited"), ("invalid-timings", "Invalid or incomplete MOSS")])
def test_failed_worker_can_be_explicitly_reloaded(owned_runtime, mode, reason):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text(mode)
    with pytest.raises(runtime.MossRuntimeError, match=reason) as caught:
        manager.transcribe(_silence())
    assert not isinstance(caught.value, runtime.MossResourceError)
    assert manager.status()["state"] == "error"
    assert not manager.status()["process_running"]
    assert not list(manager.temp_root.iterdir())
    behavior.write_text("")
    assert manager.load()["state"] == "ready"


def test_configured_deadline_stops_only_owned_worker(owned_runtime):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text("block")
    with pytest.raises(runtime.MossRuntimeError, match="configured deadline"):
        manager.transcribe(_silence(), timeout=0.01)
    assert not manager.status()["process_running"]
    assert not list(manager.temp_root.iterdir())


def test_busy_unload_rejected_and_shutdown_stops_inflight(owned_runtime, monkeypatch):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text("block")
    busy = threading.Event()
    publish = manager._publish

    def observe(state, detail):
        publish(state, detail)
        if state == "busy":
            busy.set()

    monkeypatch.setattr(manager, "_publish", observe)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(manager.transcribe, _silence())
        busy.wait()
        with pytest.raises(runtime.MossRuntimeError, match="busy"):
            manager.unload()
        manager.shutdown()
        with pytest.raises(runtime.MossRuntimeError):
            future.result()
    assert not manager.status()["process_running"]
    with pytest.raises(runtime.MossRuntimeError, match="stopping"):
        manager.load()


def test_background_load_does_not_spawn_duplicate_models(owned_runtime, monkeypatch):
    manager, _ = owned_runtime
    calls = []
    spawn = manager._spawn

    def observed(installed):
        calls.append(installed)
        spawn(installed)

    monkeypatch.setattr(manager, "_spawn", observed)
    manager.request_load()
    manager.request_load()
    manager._background.join()
    assert manager.status()["state"] == "ready"
    assert len(calls) == 1


def test_explicit_test_runs_inference_on_existing_model(owned_runtime):
    manager, _ = owned_runtime
    before = manager.load()["runtime"]
    tested = manager.load(test_existing=True)["runtime"]
    assert tested["pid"] == before["pid"]
    assert tested["request_count"] == before["request_count"] + 1


def test_busy_preload_check_preserves_healthy_worker(owned_runtime):
    manager, _ = owned_runtime
    before = manager.load()["runtime"]["pid"]

    def busy():
        raise runtime.MossBusyError("pipeline active")

    with pytest.raises(runtime.MossBusyError):
        manager.load(test_existing=True, before_load=busy)
    assert manager.status()["model_loaded"] and manager.status()["process_running"]
    assert manager.status()["runtime"]["pid"] == before
    assert manager.load(test_existing=True)["state"] == "ready"


def test_internal_handoff_waits_for_operation_instead_of_busy_error(owned_runtime):
    manager, _ = owned_runtime
    manager.load()
    manager._operation.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(manager.unload, wait=True)
        manager._operation.release()
        assert future.result()["process_running"] is False


@pytest.mark.parametrize("mode", ["wrong-optimization", "invalid"])
def test_readiness_never_reports_ready_for_failed_validation(owned_runtime, mode):
    manager, behavior = owned_runtime
    behavior.write_text(mode)
    with pytest.raises(runtime.MossRuntimeError):
        manager.load()
    assert manager.status()["state"] == "error"
    assert not manager.status()["model_loaded"]
    assert not manager.status()["process_running"]


def test_worker_validates_owned_wav_before_reading(tmp_path):
    from src.services.ingestion.moss_worker import read_pcm
    temp = tmp_path / "owned"
    temp.mkdir()
    path = temp / "audio.wav"
    path.write_bytes(_silence())
    assert read_pcm(path, 16000, 16000, temp) == bytes(32000)
    with pytest.raises(ValueError):
        read_pcm(path, 16000, 16000, tmp_path / "other")
    with pytest.raises(ValueError):
        read_pcm(path, 32000, 16000, temp)
    path.write_bytes(_silence()[:-100])
    with pytest.raises(ValueError):
        read_pcm(path, 16000, 16000, temp)


def test_status_never_imports_cuda_or_runs_setup(tmp_path, monkeypatch):
    install = runtime.MossInstallation(tmp_path)
    monkeypatch.setattr(install, "_run", lambda *_: pytest.fail("status must be read-only"))
    monkeypatch.setattr(install, "_check_platform", lambda: pytest.fail("status must not initialize CUDA"))
    assert install.status()["state"] == "absent"


@pytest.mark.parametrize("prompt,ids", [(131072, [151645]), (131071, [151645]),
                                       (100, [1, 2]), (100, [151645, 1, 151645])])
def test_incomplete_or_exhausted_generation_fails(prompt, ids):
    from src.services.ingestion.moss_worker import completion_for
    with pytest.raises(ValueError):
        completion_for(ids, prompt, 131072)


def test_output_budget_uses_remaining_context():
    from src.services.ingestion.moss_worker import completion_for
    result = completion_for([1, 151645], 200, 131072)
    assert result["max_new_tokens"] == 130872
    assert result["output_tokens"] == 2


@pytest.mark.parametrize("ids,prompt", [([1], 131072), ([1, 2], 131070)])
def test_context_exhaustion_is_typed(ids, prompt):
    from src.services.ingestion.moss_worker import completion_for, ContextBudgetError
    with pytest.raises(ContextBudgetError):
        completion_for(ids, prompt, 131072)


@pytest.mark.parametrize("ids", [[], [1], [151645, 1]])
def test_incomplete_eos_is_not_resource_exhaustion(ids):
    from src.services.ingestion.moss_worker import completion_for, ContextBudgetError
    with pytest.raises(ValueError) as caught:
        completion_for(ids, 200, 131072)
    assert not isinstance(caught.value, ContextBudgetError)


@pytest.mark.parametrize("ids", [[151645, 1], [1, 2, 3]])
def test_malformed_output_does_not_hide_behind_budget(ids):
    from src.services.ingestion.moss_worker import completion_for, ContextBudgetError
    with pytest.raises(ValueError) as caught:
        completion_for(ids, 131070, 131072)
    assert not isinstance(caught.value, ContextBudgetError)


@pytest.mark.parametrize("is_oom", [True, False])
def test_decoder_classifies_actual_oom_class_only(monkeypatch, is_oom):
    import torch
    from src.services.ingestion.moss_worker import Decoder, CudaMemoryError
    decoder = Decoder.__new__(Decoder)
    error = (torch.cuda.OutOfMemoryError("private") if is_oom
             else RuntimeError("CUDA out of memory context exhausted"))
    def fail(_):
        raise error
    monkeypatch.setattr(decoder, "_generate", fail)
    with pytest.raises(CudaMemoryError if is_oom else RuntimeError) as caught:
        decoder.generate(b"")
    assert isinstance(caught.value, CudaMemoryError) == is_oom


@pytest.mark.parametrize("code,kind", [("cuda_oom", runtime.MossCudaOOMError),
                                      ("context_exhausted", runtime.MossContextBudgetError),
                                      ("decode_failed", runtime.MossRuntimeError),
                                      ("load_failed", runtime.MossRuntimeError)])
def test_typed_worker_reply_preserves_cleanup(owned_runtime, monkeypatch, code, kind):
    manager, _ = owned_runtime
    manager.load()
    original = manager._owned.process.stdin
    class ReplyInput:
        def write(self, line):
            message = json.loads(line)
            manager._replies.put({"protocol": 1, "id": message["id"], "error": code})
        def flush(self):
            pass
        def close(self):
            original.close()
    manager._owned.process.stdin = ReplyInput()
    try:
        with pytest.raises(kind) as caught:
            manager.transcribe(_silence())
        assert type(caught.value) is kind
        assert not manager.status()["process_running"]
        assert not list(manager.temp_root.iterdir())
    finally:
        original.close()


def test_worker_resource_codes_are_private_to_transcription(tmp_path):
    from types import SimpleNamespace
    import queue
    manager = runtime.MossRuntime(tmp_path, tmp_path / "audio")
    manager._replies = queue.Queue()
    class Input:
        def write(self, line):
            m = json.loads(line)
            manager._replies.put({"protocol": 1, "id": m["id"], "error": "cuda_oom"})
        def flush(self):
            pass
    manager._owned = SimpleNamespace(process=SimpleNamespace(stdin=Input()))
    with pytest.raises(runtime.MossRuntimeError) as caught:
        manager._exchange("load")
    assert not isinstance(caught.value, runtime.MossResourceError)


@pytest.mark.parametrize("failure,code", [("oom", "cuda_oom"),
                                         ("context", "context_exhausted"),
                                         ("malformed", "decode_failed"),
                                         ("diagnostic", "decode_failed")])
def test_worker_emits_only_typed_resource_codes(monkeypatch, tmp_path, failure, code):
    import io
    import torch
    from src.services.ingestion import moss, moss_worker as worker
    decoder = worker.Decoder.__new__(worker.Decoder)
    decoder.metadata = {}
    def generate(_):
        if failure == "oom":
            raise torch.cuda.OutOfMemoryError("private model details")
        if failure == "context":
            worker.completion_for([1, 2], 131070, 131072)
        if failure == "diagnostic":
            raise RuntimeError("CUDA out of memory context exhausted")
        return "malformed output", worker.completion_for([1, 151645], 10, 131072)
    monkeypatch.setattr(decoder, "_generate", generate)
    monkeypatch.setattr(worker, "Decoder", lambda *args: decoder)
    monkeypatch.setattr(worker, "read_pcm", lambda *args: b"\0\0")
    monkeypatch.setitem(sys.modules, "moss", moss)
    messages = [{"protocol": 1, "id": "load", "operation": "load"},
                {"protocol": 1, "id": "decode", "operation": "transcribe",
                 "audio": "owned.wav", "sample_count": 1, "sample_rate": 16000}]
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, messages))))
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["worker", "--model-dir", str(tmp_path),
                                     "--source-dir", str(tmp_path), "--temp-root", str(tmp_path)])
    worker.main()
    reply = json.loads(output.getvalue().splitlines()[-1])
    assert reply == {"protocol": 1, "id": "decode", "error": code}
    assert "private" not in output.getvalue()


@pytest.mark.parametrize("bad", [b"", b"bad", None])
def test_manager_rejects_bad_audio_before_loading(tmp_path, monkeypatch, bad):
    manager = runtime.MossRuntime(tmp_path, tmp_path / "audio")
    monkeypatch.setattr(manager, "_load_locked", lambda: pytest.fail("invalid input must fail before load"))
    with pytest.raises(runtime.MossRuntimeError, match="PCM16"):
        manager.transcribe(bad)


def test_pcm_duration_is_sample_exact_and_bounded(tmp_path):
    import io
    import wave
    from src.services.ingestion.moss_worker import validate_wav, read_pcm
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 3)
    data = stream.getvalue()
    assert validate_wav(data)[1:] == (3, 16000)
    path = tmp_path / "tiny.wav"
    path.write_bytes(data)
    assert read_pcm(path, 3, 16000, tmp_path) == bytes(6)
    with pytest.raises(ValueError):
        read_pcm(path.parent / ".." / path.parent.name / path.name, 3, 16000, tmp_path)
    with pytest.raises(ValueError):
        validate_wav(data + b"trailing")
    import struct
    forged = bytearray(data)
    struct.pack_into("<I", forged, 40, (5400 * 16000 + 1) * 2)
    with pytest.raises(ValueError):
        validate_wav(bytes(forged))


def test_setup_cancellation_never_spawns_after_shutdown(owned_runtime, monkeypatch):
    manager, _ = owned_runtime
    entered, finish = threading.Event(), threading.Event()
    ensure = manager.installation.ensure
    def setup():
        entered.set()
        finish.wait()
        return ensure()
    monkeypatch.setattr(manager.installation, "ensure", setup)
    monkeypatch.setattr(manager, "_spawn", lambda *_: pytest.fail("shutdown must prevent spawn"))
    manager.request_load()
    assert entered.wait(5)
    with pytest.raises(runtime.MossRuntimeError, match="busy"):
        manager.unload()
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopped = pool.submit(manager.shutdown)
        assert manager._closing.wait(5)
        assert not stopped.done()
        finish.set()
        stopped.result(timeout=5)
    assert not manager.status()["process_running"]
    assert not manager._background.is_alive()


def test_duplicate_reply_is_rejected(owned_runtime):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text("duplicate")
    # First response can arrive before the duplicate. The next exchange must
    # reject queued extra data, rather than pairing it with a new request.
    try:
        manager.transcribe(_silence())
        with pytest.raises(runtime.MossRuntimeError, match="duplicate|invalid worker response"):
            manager.transcribe(_silence())
    except runtime.MossRuntimeError:
        assert manager.status()["state"] == "error"


def test_abandoned_staging_is_recovered(installer):
    abandoned = installer.root / "installs/.staging-interrupted"
    abandoned.mkdir(parents=True)
    (abandoned / "partial").write_bytes(b"unfinished")
    assert installer.ensure().python.is_file()
    assert not abandoned.exists()


def test_cancelled_install_preserves_previous_version(installer, monkeypatch):
    previous = installer.ensure()
    monkeypatch.setattr(installer, "destination", installer.destination.with_name("replacement"))
    installer.cancel()
    with pytest.raises(runtime.MossRuntimeError, match="stopping"):
        installer.ensure()
    assert previous.python.is_file()
    assert not installer.destination.exists()


def test_shutdown_can_cancel_wait_for_other_setup_process(installer):
    from filelock import FileLock
    with FileLock(str(installer.root / "install.lock")):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(installer.ensure)
            installer.cancel()
            with pytest.raises(runtime.MossRuntimeError, match="stopping"):
                future.result(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows owned setup process")
def test_setup_process_environment_is_private_and_actual_exit_verified(tmp_path, monkeypatch):
    install = runtime.MossInstallation(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "contaminated")
    monkeypatch.setenv("PYTHONHOME", "contaminated")
    monkeypatch.setenv("VIRTUAL_ENV", "backend")
    result = tmp_path / "environment.json"
    code = "import os,json; from pathlib import Path; Path(" + repr(str(result)) + ").write_text(json.dumps(dict(os.environ)))"
    install._run([sys.executable, "-c", code])
    env = json.loads(result.read_text())
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert install._setup_owned is None


@pytest.mark.skipif(os.name != "nt", reason="Windows owned setup process")
def test_cancel_interrupts_only_owned_setup_interpreter(tmp_path):
    import time
    install = runtime.MossInstallation(tmp_path)
    entered = tmp_path / "entered"
    code = "import time; from pathlib import Path; Path(" + repr(str(entered)) + ").write_text('ready'); time.sleep(600)"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(install._run, [sys.executable, "-c", code])
        deadline = time.monotonic() + 10
        while not entered.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert entered.exists()
        owned = install._setup_owned
        assert owned.tracked
        install.cancel()
        with pytest.raises(runtime.MossRuntimeError, match="stopping"):
            future.result(timeout=10)
    assert owned.process.poll() is not None
    assert not owned.tracked
    assert install._setup_owned is None


def test_separate_setup_processes_publish_once(tmp_path):
    import subprocess
    root = tmp_path / "isolated"
    root.mkdir()
    helper = tmp_path / "setup.py"
    helper.write_text('''import importlib.util,sys,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location("isolated_moss",sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
root=Path(sys.argv[2]);body=b"test-model";digest=hashlib.sha256(body).hexdigest()
m.PINNED_FILES={"model/test.bin":(len(body),digest)}
m.MossInstallation._check_platform=staticmethod(lambda:None)
i=m.MossInstallation(root)
i._prepare_wheels=lambda python,lock,stage:(stage,lock)
def run(command):
    if command[1:3]==["-m","venv"]:
        with (root/"created").open("a") as stream: stream.write("created\\n")
        python=Path(command[3])/"Scripts/python.exe"
        python.parent.mkdir(parents=True);python.write_bytes(b"fake")
i._run=run
i.ensure()
''')
    digest = hashlib.sha256(b"test-model").hexdigest()
    (root / "downloads").mkdir()
    (root / "downloads" / digest).write_bytes(b"test-model")
    command = [sys.executable, str(helper), runtime.__file__, str(root)]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
    assert (root / "created").read_text().splitlines() == ["created"]
    assert len(list((root / "installs").glob("*/installed.json"))) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership regression")
def test_abrupt_parent_exit_kills_actual_venv_interpreter(tmp_path):
    import ctypes
    import subprocess
    from ctypes import wintypes as w

    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER.replace("sys.argv[-1]", repr(str(Path(runtime.__file__).parent))))
    worker.with_name("behavior").write_text("")
    parent = tmp_path / "parent.py"
    parent.write_text('''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
os.environ["DB_PROVIDER"]="sqlite"
os.environ["DATABASE_URL"]="sqlite://"
from src.services.ingestion import moss_runtime as m
root=Path(__file__).parent
m.WORKER=root/"fake_worker.py"
manager=m.MossRuntime(root/"cache",root/"audio")
manager.installation.root.mkdir()
ready=m.InstalledMoss(root,Path(sys.executable),root,root)
manager.installation.ensure=lambda:ready
print(manager.load()["runtime"]["pid"],flush=True)
sys.stdin.read(1)
os._exit(0)
''')
    process = subprocess.Popen([sys.executable, str(parent), str(Path(__file__).parent.parent)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
    kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = None
    try:
        line = process.stdout.readline()
        assert line.strip().isdigit(), process.stderr.read() if process.poll() is not None else line
        pid = int(line)
        handle = kernel.OpenProcess(0x100000, False, pid)
        assert handle
        process.stdin.write("x")
        process.stdin.flush()
        process.wait()
        assert kernel.WaitForSingleObject(handle, 0xFFFFFFFF) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        if handle:
            kernel.CloseHandle(handle)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


@pytest.fixture
def app_moss(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.services.ingestion.service import IngestionService
    from src.services.ingestion import runtime as app_runtime

    service = IngestionService.__new__(IngestionService)
    service._moss_runtime = None
    service._moss_runtime_guard = threading.Lock()
    calls = []
    state = {"state": "absent", "model_loaded": False, "process_running": False}
    def load(before_load):
        before_load()
        calls.append("load")
        state.update(state="ready", model_loaded=True, process_running=True)
    def unload(**kwargs):
        calls.append("unload")
        state.update(state="absent", model_loaded=False, process_running=False)
    manager = SimpleNamespace(status=lambda: dict(state), request_load=load,
                              unload=unload, shutdown=lambda: calls.append("shutdown"))
    def factory(root, temp):
        assert root == tmp_path / "runtime"
        assert temp == tmp_path / "temp" / "moss-ipc"
        calls.append("manager")
        return manager
    monkeypatch.setattr(runtime, "MossRuntime", factory)
    monkeypatch.setattr(app_runtime, "MOSS_RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(app_runtime, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(service, "_has_active_pipeline_gpu_work", lambda: False)
    for name in ("phonon", "whisper", "parakeet", "diarization"):
        monkeypatch.setattr(service, "_release_" + name + ("_models" if name == "diarization" else "_model"),
                            lambda *a, component=name, **kw: calls.append(component))
    return service, manager, state, calls


def test_app_manager_is_lazy_once_status_read_only_and_load_order(app_moss, monkeypatch):
    service, manager, state, calls = app_moss
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "auto")
    assert calls == []
    assert service.test_moss_engine("status")["status"] == "idle"
    assert calls == ["manager"]
    response = service.test_moss_engine("load")
    assert calls == ["manager", "phonon", "whisper", "parakeet", "diarization", "load"]
    assert response["status"] == "ok"
    assert "generation completed" in response["detail"]
    assert os.environ["TRANSCRIPTION_ENGINE"] == "auto"
    service.test_moss_engine("unload")
    assert not state["process_running"]
    service.shutdown_moss()
    assert calls[-1] == "shutdown"


@pytest.mark.parametrize("action", ["load", "unload"])
def test_app_busy_controls_preserve_worker(app_moss, monkeypatch, action):
    service, manager, state, calls = app_moss
    monkeypatch.setattr(service, "_has_active_pipeline_gpu_work", lambda: True)
    assert service.test_moss_engine(action)["status"] == "error"
    assert calls == ["manager"]
    with pytest.raises(runtime.MossBusyError, match="Pipeline work"):
        service.purge_loaded_models()
    assert calls == ["manager"]


def test_app_error_and_release_wait_for_owned_exit(app_moss):
    service, manager, state, calls = app_moss
    state.update(state="error", detail="Actionable failure")
    assert service.test_moss_engine("status")["error"] == "Actionable failure"
    state.update(state="ready", model_loaded=True, process_running=True)
    def unload(*, wait):
        assert wait
        state["process_running"] = False
    manager.unload = unload
    assert service._release_moss_model("before_voice")
    assert not state["process_running"]


def test_explicit_moss_route_never_probes_or_falls_back(app_moss, monkeypatch):
    service, _, _, _ = app_moss
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "moss")
    monkeypatch.setattr(service, "_ensure_device", lambda: pytest.fail("explicit engine must not initialize host CUDA"))
    monkeypatch.setattr(service, "_infer_language_from_text_hints", lambda _: pytest.fail("Moss does not need Whisper routing"))
    route = service._resolve_transcription_route(object(), Path("spanish.wav"))
    assert route["engine"] == route["requested_engine"] == "moss"
    assert not route["multilingual_route_applied"]


def test_settings_moss_actions_and_save_are_isolated(app_moss, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.routers import settings
    from src.schemas import Settings, TranscriptionEngineTestRequest
    service, _, _, calls = app_moss
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "auto")
    monkeypatch.setattr(settings, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(settings, "get_ingestion_service", lambda: service)
    monkeypatch.setattr(settings, "_main", lambda: SimpleNamespace(configure_logging=lambda: None))
    purges = []
    monkeypatch.setattr(settings, "_purge_runtime_models", lambda **kw: purges.append(kw))
    request = TranscriptionEngineTestRequest(engine="moss", moss_action="status")
    assert settings.test_transcription_engine(request)["status"] == "idle"
    assert not (tmp_path / ".env").exists()
    assert os.environ["TRANSCRIPTION_ENGINE"] == "auto"
    assert calls == ["manager"]
    settings.update_settings(Settings(hf_token="", transcription_engine="moss"), session=None)
    assert "TRANSCRIPTION_ENGINE='moss'" in (tmp_path / ".env").read_text()
    assert settings.get_settings().transcription_engine == "moss"
    assert purges == [{"reason": "settings_updated"}]



def test_voice_loading_stops_moss_before_ml_initialization(app_moss, monkeypatch):
    service, _, _, _ = app_moss
    class ExitNotConfirmed(RuntimeError):
        pass
    def refuse(*args, **kwargs):
        raise ExitNotConfirmed()
    monkeypatch.setattr(service, "_release_moss_model", refuse)
    with pytest.raises(ExitNotConfirmed):
        service._load_models()


def test_app_shutdown_moss_runs_after_phonon_failure(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from src import main
    calls = []
    def phonon():
        calls.append("phonon")
        raise RuntimeError("simulated shutdown failure")
    fake = SimpleNamespace(shutdown_phonon=phonon, shutdown_moss=lambda: calls.append("moss"))
    for name in ("cleanup_orphaned_active_jobs", "cleanup_orphaned_active_videos",
                 "cleanup_orphaned_channel_syncs", "process_queue", "process_diarize_queue",
                 "process_funny_queue", "process_youtube_queue", "process_clip_queue",
                 "prefetch_queue_audio", "monitor_channels_loop"):
        setattr(fake, name, lambda *args: None)
    monkeypatch.setattr(main, "IngestionService", lambda: fake)
    monkeypatch.setattr(main, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(main, "BACKEND_INSTANCE_LOCK_PATH", tmp_path / "app.lock")
    monkeypatch.setattr(main, "BACKEND_INSTANCE_INFO_PATH", tmp_path / "app.json")
    monkeypatch.setattr(main.atexit, "register", lambda fn: None)
    monkeypatch.setattr(main, "ingestion_service", None)
    monkeypatch.setattr(main, "worker_threads", {})
    monkeypatch.setattr(main, "prefetch_thread", None)
    monkeypatch.setattr(main, "backend_instance_lock", None)
    async def run():
        async with main.lifespan(main.app):
            pass
    asyncio.run(run())
    assert calls == ["phonon", "moss"]
    assert not (tmp_path / "app.json").exists()



def test_retained_moss_is_not_evicted_for_its_own_vram(app_moss, monkeypatch):
    service, manager, state, _ = app_moss
    service._moss_runtime = manager
    state.update(state="ready", model_loaded=True, process_running=True)
    service.device = "cuda"
    monkeypatch.setattr(service, "_cuda_memory_snapshot", lambda: {
        "free": 4 * 1024**3, "total": 32 * 1024**3, "reserved": 0, "allocated": 0})
    monkeypatch.setattr(service, "_upsert_job_payload_fields", lambda *a: None)
    monkeypatch.setattr(service, "purge_loaded_models", lambda **kw: pytest.fail("resident child is intentional"))
    service._maybe_recover_cuda_headroom(8 * 1024**3)
