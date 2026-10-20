"""App-owned optional Moss installation. No application/DB imports here."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import io
import json
import logging
import math
import re
import os
from pathlib import Path
import platform
import queue
import shutil
import sys
import threading
import tempfile
import urllib.error
import urllib.request
import uuid
import wave

from filelock import FileLock, Timeout


MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
SOURCE_REVISION = "e3bdd93f8ad58418e04659818efd5fd87a0bc486"
PINNED_FILES = {'model/.gitattributes': (226, '97e0134c7dd6994fe5aa1ada1210b1cea898c5f455b22ab8fc17c86a819dc3e9'),
 'model/added_tokens.json': (707,
                             'c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680'),
 'model/chat_template.jinja': (4762,
                               '8641466a16b184ebaf7c4903391e607cfd532ab937e81a64120628ab79d827f4'),
 'model/config.json': (2335, '2b2b7a6e61334152bdd7ecf8a4da3073b4940a097e193d1d2b22093e77535234'),
 'model/configuration_moss_transcribe_diarize.py': (2663,
                                                    'b4d12b0f4609af69b61c2fe3aa5fbaf476af22278369e4540745bc47d1d37892'),
 'model/generation_config.json': (107,
                                  'e53a4b3ce4f944230cf1ca8fed0c42f4ff0d8c1443eaf98b5315d987334dd9e4'),
 'model/merges.txt': (1671853, '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'),
 'model/model-00000-of-00001.safetensors': (1817113576,
                                            '9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4'),
 'model/model.safetensors.index.json': (65401,
                                        '0345ac5d8f360abe4e9adadb5fecd38e7730052b75f64bb58182818b1544cc36'),
 'model/Model_Architecture.png': (59860,
                                  '41d733ca1f4d7cc0dd11e59ffc1c4fb54559572ca920fa536f33c4650d729579'),
 'model/modeling_moss_transcribe_diarize.py': (15101,
                                               'a01da90fe1f7cb88942b8c56f443e7b4ecd307ed4b4c356ac08d7503dd7422c1'),
 'model/preprocessor_config.json': (315,
                                    'ba2e601484abc80f4cded977f9a4fd4a53175b7d35c2f2511f0cfc3a32ad2499'),
 'model/processing_moss_transcribe_diarize.py': (10975,
                                                 '6f228d22d9379e2f6a6830b18ce7336b22da8267547e96e65545d871d7f48766'),
 'model/processor_config.json': (292,
                                 'a978c2dd54a65b576c3dae4b654fe9bcbac1184c6db2df0afb2c90fcdc872ae7'),
 'model/README.md': (13985, '0d87aefa6a5cbe31b4d93ebe7942fe2480722923675623e4504ce4148f695bfb'),
 'model/special_tokens_map.json': (613,
                                   '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd'),
 'model/tokenizer.json': (11423222,
                          'bcf03774334462d6e34b5005cb11120a62275f146ee2953e68731ecdbce84fbb'),
 'model/tokenizer_config.json': (503,
                                 '61d04c96104177240688396655ae3f7cf38ce2ea036db867a1d2b6883e27c3d5'),
 'model/vocab.json': (2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
 # Pin immutable download bytes, not a Windows checkout's CRLF conversion.
 'source/LICENSE': (11357, 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
 'source/README.md': (19546, 'abcfe4aca1ae5d373dfe0f3ac8a8d8c7e6abf8e362ce139961775f2141d543eb'),
 'source/pyproject.toml': (1091,
                           '181e406a953d1b6b1a18971dada4427732597fa592252fe8e63080169192d72b'),
 'source/moss_transcribe_diarize/inference_utils.py': (8784,
                                                       '1d97700b83ed95438be2e3a59529b3726d7bf7f7cdd488ad2097180bad15b86b')}
REQUIREMENTS = Path(__file__).with_name("moss_worker_requirements.txt")
WORKER = Path(__file__).with_name("moss_worker.py")
MAX_INPUT_BYTES = 5400 * 32000 + 44


class MossRuntimeError(RuntimeError):
    """Actionable app error; raw subprocess/download errors stay out of the UI."""


class MossBusyError(MossRuntimeError):
    """A model-control request must wait without stopping a healthy worker."""


class MossResourceError(MossRuntimeError):
    """Only this failure family permits splitting an uncommitted request."""


class MossCudaOOMError(MossResourceError):
    """The decoder raised a typed CUDA out-of-memory failure."""


class MossContextBudgetError(MossResourceError):
    """The decoder exhausted its pinned output context budget."""


@dataclass(frozen=True)
class InstalledMoss:
    directory: Path
    python: Path
    model: Path
    source: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class MossInstallation:
    """One versioned environment/cache; failed upgrades preserve valid versions."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._state_lock = threading.Lock()
        self._state = {"state": "absent", "detail": "Moss has not been set up"}
        self._requirements = REQUIREMENTS.read_bytes()
        self._cancelled = threading.Event()
        self._setup_guard = threading.Lock()
        self._setup_owned = None
        identity = self._requirements + json.dumps(PINNED_FILES, sort_keys=True).encode() + SOURCE_REVISION.encode() + MODEL_REVISION.encode()
        self.version = hashlib.sha256(identity).hexdigest()
        self.destination = self.root / "installs" / self.version

    def status(self) -> dict:
        with self._state_lock:
            state = dict(self._state)
        if state["state"] == "absent" and self._installed() is not None:
            return {"state": "installed", "detail": "Moss is installed; model is unloaded"}
        return state

    def _set_state(self, state: str, detail: str, **fields):
        with self._state_lock:
            self._state = {"state": state, "detail": detail, **fields,
                           **({"error": detail} if state == "error" else {})}

    @staticmethod
    def _check_platform():
        if (platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}
                or sys.version_info[:2] != (3, 11)):
            raise MossRuntimeError("Integrated Moss currently requires Windows x64 and Python 3.11; select another engine on this installation")

    def _installed(self, *, verify=False) -> InstalledMoss | None:
        destination = self.destination
        try:
            manifest = json.loads((destination / "installed.json").read_text(encoding="utf-8"))
            if manifest != {"requirements_sha256": self.version, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
                            "model_files": {name: digest for name, (_, digest) in PINNED_FILES.items()}}:
                return None
            python = destination / "venv" / "Scripts" / "python.exe"
            if not python.is_file():
                return None
            for name, (size, digest) in PINNED_FILES.items():
                path = destination / name
                if path.stat().st_size != size or (verify and _sha256(path) != digest):
                    return None
            return InstalledMoss(destination, python, destination / "model", destination / "source")
        except (OSError, ValueError, TypeError):
            return None

    def _download(self, name: str) -> Path:
        """Resume only an acknowledged HTTP byte range, then check pinned bytes."""
        size, digest = PINNED_FILES[name]
        cache = self.root / "downloads"
        cache.mkdir(parents=True, exist_ok=True)
        complete = cache / digest
        partial = cache / (digest + ".part")
        if complete.is_file() and complete.stat().st_size == size and _sha256(complete) == digest:
            return complete
        if partial.exists() and partial.stat().st_size > size:
            partial.unlink()
        offset = partial.stat().st_size if partial.exists() else 0
        if offset < size:
            request = urllib.request.Request(
                (f"https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/{MODEL_REVISION}/{name[6:]}"
                 if name.startswith("model/") else f"https://raw.githubusercontent.com/OpenMOSS/MOSS-Transcribe-Diarize/{SOURCE_REVISION}/{name[7:]}"),
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status == 206:
                        expected = f"bytes {offset}-{size - 1}/{size}"
                        if response.headers.get("Content-Range") != expected:
                            raise MossRuntimeError("Moss download returned an invalid resume range; retry setup")
                    elif response.status == 200:
                        offset = 0  # Server ignored Range: restart instead of appending.
                    else:
                        raise MossRuntimeError("Moss model download failed; retry setup when online")
                    with partial.open("ab" if offset else "wb") as stream:
                        received = offset
                        while block := response.read(io.DEFAULT_BUFFER_SIZE):
                            self._check_cancelled()
                            received += len(block)
                            if received > size:
                                raise MossRuntimeError("Moss download exceeds its pinned size; retry setup")
                            stream.write(block)
                            self._set_state("installing", f"Downloading {name}", downloaded_bytes=received, total_bytes=size)
                        stream.flush()
                        os.fsync(stream.fileno())
            except (OSError, urllib.error.URLError):
                raise MossRuntimeError("Moss model download interrupted or offline; reconnect and retry setup") from None
        if not partial.is_file() or partial.stat().st_size != size or _sha256(partial) != digest:
            partial.unlink(missing_ok=True)
            raise MossRuntimeError("Moss model checksum failed; retry setup")
        os.replace(partial, complete)
        return complete

    def _check_cancelled(self):
        if self._cancelled.is_set():
            raise MossRuntimeError("MOSS setup is stopping; retry after app restart")

    def cancel(self):
        self._cancelled.set()
        with self._setup_guard:
            if self._setup_owned:
                self._setup_owned.close()
                self._setup_owned = None

    @contextmanager
    def _installation_lock(self):
        lock = FileLock(str(self.root / "install.lock"))
        while True:
            self._check_cancelled()
            try:
                lock.acquire(timeout=0.1)
                break
            except Timeout:
                continue
        try:
            yield
        finally:
            lock.release()

    def _run(self, command: list[str]):
        from .phonon_process import OwnedPhononProcess
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1")
        # Handshake from the actual interpreter before executing pip/venv/import
        # checks; a Windows venv executable may itself only be a launcher.
        bootstrap = (
            "import os,runpy,sys; print('MOSS_SETUP_PID='+str(os.getpid()),flush=True); "
            "assert sys.stdin.readline().strip()=='continue'; "
            "mode,value=sys.argv[1:3]; sys.argv=sys.argv[2:]; "
            "runpy.run_module(value,run_name='__main__',alter_sys=True) if mode=='-m' else exec(value)"
        )
        with (self.root / "setup.log").open("ab") as log:
            with self._setup_guard:
                self._check_cancelled()
                owned = self._setup_owned = OwnedPhononProcess(
                    [command[0], "-u", "-c", bootstrap, *command[1:]], env=env, stderr=log)
            try:
                identity = owned.process.stdout.readline().strip()
                if not identity.startswith("MOSS_SETUP_PID=") or not identity[15:].isdigit():
                    raise MossRuntimeError("MOSS setup interpreter identity failed")
                with self._setup_guard:
                    self._check_cancelled()
                    owned.track_worker(int(identity[15:]))
                    owned.process.stdin.write("continue\n")
                    owned.process.stdin.flush()
                for line in owned.process.stdout:
                    log.write(line.encode("utf-8"))
                    log.flush()
                code = owned.process.wait()
            except (OSError, ValueError):
                self._check_cancelled()
                raise
            finally:
                with self._setup_guard:
                    owned.close()
                    if self._setup_owned is owned:
                        self._setup_owned = None
            self._check_cancelled()
            if code:
                raise MossRuntimeError("MOSS runtime installation failed; retry setup (details in the local MOSS setup log)")

    def _prepare_wheels(self, python, lock, stage):
        # Dedicated hash-verified wheelhouse is sufficient to rebuild offline.
        wheels = self.root / "wheels" / self.version
        wheels.mkdir(parents=True, exist_ok=True)
        text = self._requirements.decode()
        hashes = set(re.findall(r"(?:sha256:|sha256=)([0-9a-f]{64})", text))
        required = [line for line in text.splitlines() if line and not line.startswith("#")]
        valid = []
        for path in wheels.glob("*.whl"):
            if _sha256(path) in hashes:
                valid.append(path)
            else:
                path.unlink()
        if len(valid) != len(required):
            self._run([str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "download",
                       "--index-url", "https://pypi.org/simple", "--require-hashes", "--only-binary=:all:",
                       "--dest", str(wheels), "-r", str(lock)])
        # Direct URL requirements would bypass --no-index even with a wheelhouse.
        local = re.sub(r"(torch(?:audio)?) @ [^\n]+-([0-9.]+)%2Bcu128-cp311-cp311-win_amd64.whl#sha256=([0-9a-f]+)",
                       r"\1==\2+cu128 --hash=sha256:\3", text)
        local_lock = stage / "offline-requirements.txt"
        local_lock.write_text(local, encoding="utf-8")
        return wheels, local_lock

    def ensure(self) -> InstalledMoss:
        self._check_cancelled()
        self._check_platform()
        self.root.mkdir(parents=True, exist_ok=True)
        self._set_state("installing", "Preparing Moss; another setup request may be finishing")
        try:
            with self._installation_lock():
                self._check_cancelled()
                for abandoned in (self.root / "installs").glob(".staging-*"):
                    if abandoned.resolve().parent == (self.root / "installs").resolve():
                        shutil.rmtree(abandoned)
                existing = self._installed(verify=True)
                if existing:
                    self._set_state("installed", "Moss is installed; model is unloaded")
                    return existing
                stage = self.root / "installs" / (".staging-" + uuid.uuid4().hex)
                stage.mkdir(parents=True)
                try:
                    for name in PINNED_FILES:
                        self._check_cancelled()
                        source = self._download(name)
                        target = stage / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                    self._set_state("installing", "Installing the private CUDA runtime; this can take several minutes")
                    # An app update must not change the lock beneath an active setup.
                    lock = stage / "requirements.txt"
                    lock.write_bytes(self._requirements)
                    self._run([sys._base_executable, "-m", "venv", str(stage / "venv")])
                    python = stage / "venv" / "Scripts" / "python.exe"
                    wheels, local_lock = self._prepare_wheels(python, lock, stage)
                    self._run([str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
                               "--no-index", "--find-links", str(wheels), "--require-hashes", "--only-binary=:all:",
                               "--no-input", "-r", str(local_lock)])
                    self._run([str(python), "-m", "pip", "--isolated", "check"])
                    # Metadata alone misses optional native DLL failures (e.g. SoundFile on Windows).
                    self._run([str(python), "-c", "from transformers import AutoModelForCausalLM, AutoProcessor; import soundfile; import torch; assert torch.cuda.is_available(); assert torch.__version__ == '2.10.0+cu128'"])
                    self._check_cancelled()
                    manifest = {"requirements_sha256": self.version, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
                                "model_files": {name: digest for name, (_, digest) in PINNED_FILES.items()}}
                    with (stage / "installed.json").open("w", encoding="utf-8") as stream:
                        json.dump(manifest, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if self.destination.exists():
                        # Preserve a damaged previous version for diagnosis; never overwrite a valid install.
                        self.destination.rename(self.destination.with_name(".invalid-" + uuid.uuid4().hex))
                    stage.rename(self.destination)
                finally:
                    if stage.exists() and stage.resolve().parent == (self.root / "installs").resolve():
                        shutil.rmtree(stage)
                installed = self._installed(verify=True)
                if installed is None:
                    raise MossRuntimeError("Moss installation is incomplete; retry setup")
                self._set_state("installed", "Moss is installed; model is unloaded")
                return installed
        except Exception as error:
            safe = str(error) if isinstance(error, MossRuntimeError) else "Moss installation failed; check local disk space and retry setup"
            self._set_state("error", safe)
            raise MossRuntimeError(safe) from None


class MossRuntime:
    """One app-owned model, serialized inference and observable lazy setup."""

    def __init__(self, root: Path, temp_root: Path):
        self.installation = MossInstallation(root)
        self.temp_root = temp_root.resolve()
        self._operation = threading.Lock()
        self._dispose_guard = threading.Lock()
        self._closing = threading.Event()
        self._state_guard = threading.Lock()
        self._state = {"state": "absent", "detail": "Moss is unloaded"}
        self._runtime = {}
        self._owned = None
        self._reader = None
        self._replies = None
        self._log = None
        self._loaded = False
        self._background = None
        self._protocol_error = threading.Event()

    def _publish(self, state, detail):
        with self._state_guard:
            self._state = {"state": state, "detail": detail,
                           **({"error": detail} if state == "error" else {})}

    def status(self):
        with self._state_guard:
            state = dict(self._state)
            runtime = dict(self._runtime)
        process = self._owned.process if self._owned else None
        alive = process is not None and process.poll() is None
        if self._protocol_error.is_set() and alive:
            state = {"state": "error", "detail": "MOSS returned invalid or duplicate worker data"}
        if state["state"] == "absent":
            state = self.installation.status()
        elif state["state"] == "installing":
            setup = self.installation.status()
            if setup["state"] in {"installing", "error"}:
                state = setup
        elif state["state"] in {"ready", "busy"} and not alive:
            state = {"state": "error", "detail": "Moss worker exited; retry Load/Test"}
        return {**state, "model_loaded": self._loaded and alive and not self._protocol_error.is_set(),
                "runtime": runtime, "process_running": alive}

    def _dispose(self):
        with self._dispose_guard:
            self._dispose_locked()

    def _dispose_locked(self):
        owned = self._owned
        if owned:
            owned.close()  # Verifies actual interpreter exit, including venv descendants.
            self._owned = None
        if self._reader:
            self._reader.join()
            self._reader = None
        if self._log:
            self._log.close()
            self._log = None
        self._loaded = False
        self._runtime = {}

    def _exchange(self, operation, *, timeout=None, **values):
        request_id = uuid.uuid4().hex
        message = {"protocol": 1, "id": request_id, "operation": operation, **values}
        try:
            if self._protocol_error.is_set() or not self._replies.empty():
                raise MossRuntimeError("MOSS returned duplicate or unsolicited worker data")
            self._owned.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self._owned.process.stdin.flush()
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            raise MossRuntimeError("MOSS inference reached its configured deadline; retry the complete input") from None
        except (OSError, ValueError):
            raise MossRuntimeError("Moss worker connection failed; retry Load/Test") from None
        if isinstance(reply, Exception):
            raise reply
        if (not isinstance(reply, dict) or type(reply.get("protocol")) is not int
                or reply.get("protocol") != 1 or reply.get("id") != request_id):
            raise MossRuntimeError("Moss returned an invalid worker response; retry the complete input")
        if reply.get("error"):
            if operation == "transcribe":
                if reply["error"] == "cuda_oom":
                    raise MossCudaOOMError("MOSS CUDA memory exhausted for this request")
                if reply["error"] == "context_exhausted":
                    raise MossContextBudgetError("MOSS output context exhausted for this request")
            if reply["error"] == "invalid_input":
                raise MossRuntimeError("Moss input must be a valid bounded 16 kHz mono PCM16 WAV")
            reason = "could not load its model" if reply["error"] == "load_failed" else "could not decode the input"
            raise MossRuntimeError(f"Moss {reason}; check GPU memory and the local worker log, then retry")
        if not isinstance(reply.get("runtime"), dict):
            raise MossRuntimeError("Moss returned invalid runtime metadata")
        with self._state_guard:
            self._runtime = dict(reply["runtime"])
        return reply

    def _spawn(self, installed):
        from .phonon_process import OwnedPhononProcess
        self.temp_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(installed.directory / "hf"), HF_MODULES_CACHE=str(installed.directory / "modules"))
        with self._dispose_guard:
            if self._closing.is_set():
                raise MossRuntimeError("Moss runtime is stopping")
            self._log = (self.installation.root / "worker.log").open("ab")
            self._owned = OwnedPhononProcess(
                [str(installed.python), "-u", str(WORKER), "--model-dir", str(installed.model),
                 "--source-dir", str(installed.source), "--temp-root", str(self.temp_root)],
                env=env, stderr=self._log,
            )
            replies = self._replies = queue.Queue()
            self._protocol_error.clear()
            stream = self._owned.process.stdout

            def receive():
                seen = set()
                try:
                    for line in stream:
                        try:
                            value = json.loads(line)
                            if not isinstance(value, dict) or not isinstance(value.get("id"), str) or value["id"] in seen:
                                raise ValueError("duplicate or invalid reply")
                            seen.add(value["id"])
                            replies.put(value)
                        except (ValueError, TypeError):
                            self._protocol_error.set()
                            replies.put(MossRuntimeError("Moss emitted invalid worker data"))
                            break
                except (OSError, ValueError, UnicodeError):
                    pass
                finally:
                    replies.put(MossRuntimeError("Moss worker exited; retry the complete input"))

            self._reader = threading.Thread(target=receive, name="moss-output", daemon=True)
            self._reader.start()
        identity = self._exchange("status")["runtime"]
        pid = identity.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise MossRuntimeError("Moss worker returned an invalid process identity")
        self._owned.track_worker(pid)  # Membership checked before the first GPU load.

    def _load_locked(self, before_load=None):
        if self._closing.is_set():
            raise MossRuntimeError("Moss runtime is stopping")
        if self._loaded and self._owned and self._owned.process.poll() is None:
            return False
        self._dispose()
        self._publish("installing", "Preparing Moss")
        installed = self.installation.ensure()
        if self._closing.is_set():
            raise MossRuntimeError("MOSS runtime is stopping")
        self._publish("loading", "Loading MOSS CUDA model")
        if before_load:
            before_load()
        self._spawn(installed)
        runtime = self._exchange("load")["runtime"]
        if (runtime.get("model") != "MOSS-Transcribe-Diarize" or runtime.get("device") != "cuda"
                or runtime.get("dtype") != "bfloat16" or runtime.get("attention") != "sdpa"
                or runtime.get("model_revision") != MODEL_REVISION
                or runtime.get("source_revision") != SOURCE_REVISION or not runtime.get("model_loaded")):
            raise MossRuntimeError("MOSS could not verify its pinned CUDA model; check the local worker log")
        self._loaded = True
        self._readiness_locked()
        return True

    def _readiness_locked(self):
        reply = self._exchange("probe")
        from .moss import parse_generation
        try:
            # Validate generation metadata independently; a silence hallucination
            # is probe evidence only and never a transcription result.
            parse_generation("", 16000, 16000, reply["result"]["completion"])
            if not isinstance(reply["result"]["raw_text"], str):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise MossRuntimeError("MOSS readiness generation was incomplete") from None
        with self._state_guard:
            self._runtime["readiness_probe"] = reply["result"]
        self._publish("ready", "Moss is loaded and ready")

    def _failure(self, error):
        logging.getLogger(__name__).error("Moss runtime operation failed", exc_info=(type(error), error, error.__traceback__))
        try:
            self._dispose()
        except Exception:
            safe = "Moss could not confirm its worker stopped; check the local worker log"
            self._publish("error", safe)
            return MossRuntimeError(safe)
        safe = str(error) if isinstance(error, MossRuntimeError) else "Moss worker failed; check the local worker log and retry Load/Test"
        self._publish("error", safe)
        return type(error)(safe) if isinstance(error, MossResourceError) else MossRuntimeError(safe)

    def load(self, *, test_existing=False, before_load=None):
        with self._operation:
            try:
                created = self._load_locked(before_load=before_load)
                if test_existing and not created:
                    if before_load:
                        before_load()
                    self._publish("busy", "Testing Moss readiness")
                    self._readiness_locked()
                return self.status()
            except MossBusyError as error:
                self._publish("error", str(error))
                raise
            except Exception as error:
                raise self._failure(error) from None

    def request_load(self, before_load=None):
        with self._state_guard:
            if self._closing.is_set():
                raise MossRuntimeError("Moss runtime is stopping")
            if self._background and self._background.is_alive():
                return

            def run():
                try:
                    self.load(test_existing=True, before_load=before_load)
                except MossRuntimeError:
                    pass  # Observable error state; the next explicit action may retry.

            self._background = threading.Thread(target=run, name="moss-load", daemon=True)
            self._state = {"state": "loading", "detail": "Preparing Moss"}
            self._background.start()

    def _transcribe_locked(self, audio, count, rate, timeout=None):
        path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="moss-ipc-", suffix=".wav", dir=self.temp_root, delete=False) as stream:
                path = Path(stream.name)
                stream.write(audio)
            reply = self._exchange("transcribe", audio=str(path), sample_count=count, sample_rate=rate, timeout=timeout)
            if not isinstance(reply.get("result"), dict):
                raise MossRuntimeError("Moss returned invalid transcription data")
            from .moss import MossValidationError, validate_worker_result
            try:
                validate_worker_result(reply["result"], count, rate)
            except MossValidationError as error:
                raise MossRuntimeError(str(error)) from None
            return {"result": reply["result"], "runtime": reply["runtime"]}
        finally:
            if path:
                path.unlink(missing_ok=True)

    def transcribe(self, audio: bytes, timeout=None):
        from .moss_worker import validate_wav
        try:
            _, count, rate = validate_wav(audio)
        except (ValueError, EOFError, wave.Error):
            raise MossRuntimeError("MOSS requires a complete PCM16 mono 16 kHz WAV of at most 90 minutes; select another engine for longer audio") from None
        if timeout is not None and (not math.isfinite(float(timeout)) or float(timeout) <= 0):
            raise MossRuntimeError("Moss inference deadline must be positive and finite")
        with self._operation:
            try:
                self._load_locked()
                self._publish("busy", "Moss is transcribing")
                result = self._transcribe_locked(audio, count, rate, timeout)
                self._publish("ready", "Moss is loaded and ready")
                return result
            except Exception as error:
                raise self._failure(error) from None

    def unload(self, *, wait=False):
        if wait and self._background and self._background is not threading.current_thread():
            self._background.join()
        if not wait and self._background and self._background.is_alive():
            raise MossRuntimeError("Moss is busy loading; unload after the current operation completes")
        if not self._operation.acquire(blocking=wait):
            raise MossRuntimeError("Moss is busy loading or decoding; unload after the current operation completes")
        try:
            self._publish("unloading", "Unloading Moss")
            if self._owned and self._owned.process.poll() is None:
                self._exchange("shutdown")
            self._dispose()
            self._publish("absent", "Moss is unloaded; cached model and runtime are retained")
            return self.status()
        except Exception as error:
            raise self._failure(error) from None
        finally:
            self._operation.release()

    def shutdown(self):
        # App shutdown may terminate its own in-flight worker. No other process is touched.
        self._closing.set()
        self.installation.cancel()
        self._dispose()
        # A setup thread may still be returning from a download. It cannot spawn
        # after closing is set; wait before claiming that shutdown completed.
        with self._operation:
            self._dispose()
        if self._background and self._background is not threading.current_thread():
            self._background.join()
        self._publish("absent", "MOSS is stopped")
