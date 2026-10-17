"""Owned installer recovery tests, using scratch files and no production DB."""
import hashlib
import json
import os
from pathlib import Path
import threading
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite://")

from src.services.ingestion import phonon_runtime as runtime


@pytest.fixture
def installer(tmp_path, monkeypatch):
    body = b"test-model"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(runtime, "MODEL_FILES", {"model.fermion": (len(body), digest)})
    monkeypatch.setattr(runtime.PhononInstallation, "_check_platform", staticmethod(lambda: None))
    install = runtime.PhononInstallation(tmp_path / "phonon")
    cache = install.root / "downloads"
    cache.mkdir(parents=True)
    (cache / digest).write_bytes(body)

    def run(command):
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[3]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"private-interpreter")

    monkeypatch.setattr(install, "_run", run)
    return install


def test_install_publication_and_offline_reuse(installer, monkeypatch):
    assert installer.status()["state"] == "absent"
    installed = installer.ensure()
    assert installed.python.is_file()
    assert installed.model.joinpath("model.fermion").read_bytes() == b"test-model"
    assert installer.status()["state"] == "installed"
    manifest = json.loads((installed.directory / "installed.json").read_text())
    assert manifest["requirements_sha256"] == installer.version
    monkeypatch.setattr(installer, "_download", lambda _: pytest.fail("valid offline installation must not download"))
    monkeypatch.setattr(installer, "_run", lambda _: pytest.fail("valid offline installation must not run pip"))
    assert installer.ensure() == installed


def test_interrupted_install_never_publishes_and_retry_works(installer, monkeypatch):
    original = installer._run
    monkeypatch.setattr(installer, "_run", lambda _: (_ for _ in ()).throw(OSError("private details")))
    with pytest.raises(runtime.PhononRuntimeError, match="disk space"):
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
    with pytest.raises(runtime.PhononRuntimeError):
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

    def open_url(request):
        assert request.headers["Range"] == "bytes=4-"
        if range_supported:
            return DownloadResponse(body[4:], 206, {"Content-Range": f"bytes 4-{len(body)-1}/{len(body)}"})
        return DownloadResponse(body)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", open_url)
    assert installer._download("model.fermion").read_bytes() == body
    assert not (cache / (digest + ".part")).exists()


def test_checksum_failure_is_not_published(installer, monkeypatch):
    size, digest = runtime.MODEL_FILES["model.fermion"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda _: DownloadResponse(b"x" * size))
    with pytest.raises(runtime.PhononRuntimeError, match="checksum"):
        installer._download("model.fermion")
    assert not (cache / digest).exists()
    assert not (cache / (digest + ".part")).exists()


def test_interrupted_transfer_preserves_resumable_prefix(installer, monkeypatch):
    _, digest = runtime.MODEL_FILES["model.fermion"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()

    class Interrupted(DownloadResponse):
        def read(self, count):
            if self.body.tell():
                raise OSError("network detail must not reach UI")
            return self.body.read(4)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda _: Interrupted(b"test-model"))
    with pytest.raises(runtime.PhononRuntimeError, match="interrupted or offline"):
        installer._download("model.fermion")
    assert (cache / (digest + ".part")).read_bytes() == b"test"
    assert not (cache / digest).exists()


def test_corrupted_installed_model_is_not_reused(installer):
    installed = installer.ensure()
    installed.model.joinpath("model.fermion").write_bytes(b"bad-model!")
    assert installer._installed(verify=True) is None


def test_interrupted_manifest_write_never_marks_installed(installer, monkeypatch):
    def fail(*_):
        raise OSError("interrupted manifest write")

    monkeypatch.setattr(runtime.json, "dump", fail)
    with pytest.raises(runtime.PhononRuntimeError):
        installer.ensure()
    assert installer._installed() is None
    assert not installer.destination.exists()


def test_native_import_failure_is_rejected_before_publication(installer, monkeypatch):
    original = installer._run

    def run(command):
        if command[1] == "-c":
            raise runtime.PhononRuntimeError("Phonon runtime installation failed; retry setup")
        original(command)

    monkeypatch.setattr(installer, "_run", run)
    with pytest.raises(runtime.PhononRuntimeError, match="installation failed"):
        installer.ensure()
    assert installer._installed() is None
    assert not installer.destination.exists()


def test_invalid_resume_range_is_not_appended(installer, monkeypatch):
    _, digest = runtime.MODEL_FILES["model.fermion"]
    cache = installer.root / "downloads"
    (cache / digest).unlink()
    partial = cache / (digest + ".part")
    partial.write_bytes(b"test")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda _: DownloadResponse(b"-model", 206, {"Content-Range": "wrong"}))
    with pytest.raises(runtime.PhononRuntimeError, match="resume range"):
        installer._download("model.fermion")
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
    assert hashlib.sha256(supplied[0]).hexdigest() == installer.version


def test_unsupported_platform_fails_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    installer = runtime.PhononInstallation(tmp_path / "not-created")
    with pytest.raises(runtime.PhononRuntimeError, match="Windows"):
        installer.ensure()
    assert not installer.root.exists()


def test_private_commands_do_not_inherit_python_environment(tmp_path, monkeypatch):
    install = runtime.PhononInstallation(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "outside-environment")
    monkeypatch.setenv("PYTHONHOME", "outside-home")
    monkeypatch.setenv("VIRTUAL_ENV", "backend-environment")
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    install._run(["private-python", "-m", "pip", "check"])
    command, kwargs = seen[0]
    assert isinstance(command, list)
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & kwargs["env"].keys()
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert not kwargs.get("shell")


FAKE_WORKER = '''import json,sys,os,threading
from pathlib import Path
loaded=False
count=0
for line in sys.stdin:
    m=json.loads(line)
    op=m["operation"]
    mode=Path(__file__).with_name("behavior").read_text()
    result=None
    if op=="load": loaded=True
    if op=="transcribe":
        if mode=="crash": os._exit(3)
        if mode=="block": threading.Event().wait()
        if mode=="invalid":
            print("invalid-json",flush=True)
            continue
        count+=1
        result={"duration":m["duration"],"text":"hi","segments":[{"start":0,"end":0.5,"text":"hi"}],"words":[{"start":0,"end":0.5,"text":"hi"}]}
    if result and mode=="invalid-timings": result["words"][0]["end"]=2
    r={"pid":os.getpid(),"model_loaded":loaded,"model":"phonon-2","device":"cuda","dtype":"float32" if mode=="wrong-optimization" else "bfloat16","cuda_graph_steps":16,"warm_buckets":6,"request_count":count}
    print("fake diagnostic",file=sys.stderr,flush=True)
    print(json.dumps({"protocol":1,"id":"wrong" if mode=="wrong-id" else m["id"],"runtime":r,"result":result}),flush=True)
    if op=="shutdown": break
'''


@pytest.fixture
def owned_runtime(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Native Phonon process ownership is supported on Windows only")
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    behavior = worker.with_name("behavior")
    behavior.write_text("")
    monkeypatch.setattr(runtime, "WORKER", worker)
    manager = runtime.PhononRuntime(tmp_path / "cache", tmp_path / "audio")
    manager.installation.root.mkdir()
    installed = runtime.InstalledPhonon(tmp_path, Path(sys.executable), tmp_path)

    def ensure():
        manager.installation._set_state("installed", "test runtime installed")
        return installed

    monkeypatch.setattr(manager.installation, "ensure", ensure)
    yield manager, behavior
    manager.shutdown()


def _silence():
    from src.services.ingestion.phonon import wav_bytes
    return wav_bytes(bytes(32000))


def test_owned_worker_reuses_actual_child_and_unloads(owned_runtime):
    manager, _ = owned_runtime
    loaded = manager.load()
    pid = loaded["runtime"]["pid"]
    assert loaded["state"] == "ready"
    for count in (2, 3):
        result = manager.transcribe(_silence(), 1.0)
        assert result["runtime"]["pid"] == pid
        assert result["runtime"]["request_count"] == count
    assert "fake diagnostic" in (manager.installation.root / "worker.log").read_text()
    unloaded = manager.unload()
    assert not unloaded["model_loaded"] and not unloaded["process_running"]
    assert unloaded["runtime"] == {}
    assert not list(manager.temp_root.iterdir())


@pytest.mark.parametrize("mode,reason", [("invalid", "invalid worker data"), ("wrong-id", "invalid worker response"), ("crash", "worker exited"), ("invalid-timings", "invalid or missing word timestamps")])
def test_failed_worker_can_be_explicitly_reloaded(owned_runtime, mode, reason):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text(mode)
    with pytest.raises(runtime.PhononRuntimeError, match=reason):
        manager.transcribe(_silence(), 1.0)
    assert manager.status()["state"] == "error"
    assert not manager.status()["process_running"]
    assert not list(manager.temp_root.iterdir())
    behavior.write_text("")
    assert manager.load()["state"] == "ready"


def test_configured_deadline_stops_only_owned_worker(owned_runtime):
    manager, behavior = owned_runtime
    manager.load()
    behavior.write_text("block")
    with pytest.raises(runtime.PhononRuntimeError, match="configured deadline"):
        manager.transcribe(_silence(), 1.0, timeout=0.01)
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
        future = pool.submit(manager.transcribe, _silence(), 1.0)
        busy.wait()
        with pytest.raises(runtime.PhononRuntimeError, match="busy"):
            manager.unload()
        manager.shutdown()
        with pytest.raises(runtime.PhononRuntimeError):
            future.result()
    assert not manager.status()["process_running"]
    with pytest.raises(runtime.PhononRuntimeError, match="stopping"):
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


@pytest.mark.parametrize("mode", ["wrong-optimization", "invalid-timings"])
def test_readiness_never_reports_ready_for_failed_validation(owned_runtime, mode):
    manager, behavior = owned_runtime
    behavior.write_text(mode)
    with pytest.raises(runtime.PhononRuntimeError):
        manager.load()
    assert manager.status()["state"] == "error"
    assert not manager.status()["model_loaded"]
    assert not manager.status()["process_running"]


def test_incomplete_source_manifest_rejected_before_spawn(owned_runtime, tmp_path, monkeypatch):
    manager, _ = owned_runtime
    source = tmp_path / "bad-source"
    source.mkdir()
    (source / "source-manifest.json").write_text("{}")
    monkeypatch.setattr(runtime, "UPSTREAM", source)
    with pytest.raises(runtime.PhononRuntimeError, match="manifest is incomplete"):
        manager.load()
    assert not manager.status()["process_running"]


def test_worker_validates_owned_wav_before_reading(tmp_path):
    from src.services.ingestion.phonon_worker import read_pcm
    temp = tmp_path / "owned"
    temp.mkdir()
    path = temp / "audio.wav"
    path.write_bytes(_silence())
    assert read_pcm(path, 1.0, temp, runtime.MAX_INPUT_BYTES) == bytes(32000)
    with pytest.raises(ValueError, match="invalid input"):
        read_pcm(path, 1.0, tmp_path / "another-root", runtime.MAX_INPUT_BYTES)
    with pytest.raises(ValueError, match="invalid WAV"):
        read_pcm(path, 2.0, temp, runtime.MAX_INPUT_BYTES)
    path.write_bytes(_silence()[:-100])
    with pytest.raises(ValueError, match="invalid WAV"):
        read_pcm(path, 1.0, temp, runtime.MAX_INPUT_BYTES)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership regression")
def test_abrupt_parent_exit_kills_actual_venv_interpreter(tmp_path):
    import ctypes
    import subprocess
    from ctypes import wintypes as w

    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    worker.with_name("behavior").write_text("")
    parent = tmp_path / "parent.py"
    parent.write_text('''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
os.environ["DB_PROVIDER"]="sqlite"
os.environ["DATABASE_URL"]="sqlite://"
from src.services.ingestion import phonon_runtime as m
root=Path(__file__).parent
m.WORKER=root/"fake_worker.py"
manager=m.PhononRuntime(root/"cache",root/"audio")
manager.installation.root.mkdir()
ready=m.InstalledPhonon(root,Path(sys.executable),root)
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
