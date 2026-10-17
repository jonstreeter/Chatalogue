"""App-owned optional Phonon installation. No application/DB imports here."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import threading
import tempfile
import urllib.error
import urllib.request
import uuid
import wave

from filelock import FileLock


MODEL_REVISION = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
MODEL_FILES = {
    "model.fermion": (177438361, "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"),
    "config.json": (277493, "d0daad3b2a182893844f4abdc11e4f5b7083f7d42c8ad7e8203f71559785a31b"),
    "packed_manifest.json": (821, "3c1874501a1efb6ef569eab082aa68d91b93ec6e4d2fb00dbef7851edf2ae510"),
}
REQUIREMENTS = Path(__file__).with_name("phonon_worker_requirements.txt")
WORKER = Path(__file__).with_name("phonon_worker.py")
UPSTREAM = Path(__file__).with_name("phonon_upstream")
MAX_INPUT_BYTES = 32 * 1024 * 1024  # Retained verified encoded-input bound; no HTTP transport.


class PhononRuntimeError(RuntimeError):
    """Actionable app error; raw subprocess/download errors stay out of the UI."""


class PhononBusyError(PhononRuntimeError):
    """A model-control request must wait without stopping a healthy worker."""


@dataclass(frozen=True)
class InstalledPhonon:
    directory: Path
    python: Path
    model: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class PhononInstallation:
    """One versioned environment/cache; failed upgrades preserve valid versions."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._state_lock = threading.Lock()
        self._state = {"state": "absent", "detail": "Phonon has not been set up"}
        self._requirements = REQUIREMENTS.read_bytes()
        self.version = hashlib.sha256(self._requirements).hexdigest()
        self.destination = self.root / "installs" / self.version

    def status(self) -> dict:
        with self._state_lock:
            state = dict(self._state)
        if state["state"] == "absent" and self._installed() is not None:
            return {"state": "installed", "detail": "Phonon is installed; model is unloaded"}
        return state

    def _set_state(self, state: str, detail: str, **fields):
        with self._state_lock:
            self._state = {"state": state, "detail": detail, **fields}

    @staticmethod
    def _check_platform():
        if (platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}
                or sys.version_info[:2] != (3, 11)):
            raise PhononRuntimeError("Integrated Phonon currently requires Windows x64 and Python 3.11; select another engine on this installation")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise PhononRuntimeError("Integrated Phonon requires an NVIDIA CUDA GPU supporting BF16; select another engine on this hardware")

    def _installed(self, *, verify=False) -> InstalledPhonon | None:
        destination = self.destination
        try:
            manifest = json.loads((destination / "installed.json").read_text(encoding="utf-8"))
            if manifest != {"requirements_sha256": self.version, "model_revision": MODEL_REVISION,
                            "model_files": {name: digest for name, (_, digest) in MODEL_FILES.items()}}:
                return None
            python = destination / "venv" / "Scripts" / "python.exe"
            if not python.is_file():
                return None
            for name, (size, digest) in MODEL_FILES.items():
                path = destination / "model" / name
                if path.stat().st_size != size or (verify and _sha256(path) != digest):
                    return None
            return InstalledPhonon(destination, python, destination / "model")
        except (OSError, ValueError, TypeError):
            return None

    def _download(self, name: str) -> Path:
        """Resume only an acknowledged HTTP byte range, then check pinned bytes."""
        size, digest = MODEL_FILES[name]
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
                f"https://huggingface.co/FermionResearch/Phonon-2/resolve/{MODEL_REVISION}/{name}",
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            try:
                with urllib.request.urlopen(request) as response:
                    if response.status == 206:
                        expected = f"bytes {offset}-{size - 1}/{size}"
                        if response.headers.get("Content-Range") != expected:
                            raise PhononRuntimeError("Phonon download returned an invalid resume range; retry setup")
                    elif response.status == 200:
                        offset = 0  # Server ignored Range: restart instead of appending.
                    else:
                        raise PhononRuntimeError("Phonon model download failed; retry setup when online")
                    with partial.open("ab" if offset else "wb") as stream:
                        received = offset
                        while block := response.read(io.DEFAULT_BUFFER_SIZE):
                            received += len(block)
                            if received > size:
                                raise PhononRuntimeError("Phonon download exceeds its pinned size; retry setup")
                            stream.write(block)
                            self._set_state("installing", f"Downloading {name}", downloaded_bytes=received, total_bytes=size)
                        stream.flush()
                        os.fsync(stream.fileno())
            except (OSError, urllib.error.URLError):
                raise PhononRuntimeError("Phonon model download interrupted or offline; reconnect and retry setup") from None
        if not partial.is_file() or partial.stat().st_size != size or _sha256(partial) != digest:
            partial.unlink(missing_ok=True)
            raise PhononRuntimeError("Phonon model checksum failed; retry setup")
        os.replace(partial, complete)
        return complete

    def _run(self, command: list[str]):
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        with (self.root / "setup.log").open("ab") as log:
            result = subprocess.run(command, stdout=log, stderr=log, env=env,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise PhononRuntimeError("Phonon runtime installation failed; retry setup (details in the local Phonon setup log)")

    def ensure(self) -> InstalledPhonon:
        self._check_platform()
        self.root.mkdir(parents=True, exist_ok=True)
        self._set_state("installing", "Preparing Phonon; another setup request may be finishing")
        try:
            with FileLock(str(self.root / "install.lock")):
                existing = self._installed(verify=True)
                if existing:
                    self._set_state("installed", "Phonon is installed; model is unloaded")
                    return existing
                stage = self.root / "installs" / (".staging-" + uuid.uuid4().hex)
                stage.mkdir(parents=True)
                try:
                    model = stage / "model"
                    model.mkdir()
                    for name in MODEL_FILES:
                        source = self._download(name)
                        try:
                            os.link(source, model / name)
                        except OSError:
                            shutil.copyfile(source, model / name)
                    self._set_state("installing", "Installing the private CUDA runtime; this can take several minutes")
                    # An app update must not change the lock beneath an active setup.
                    lock = stage / "requirements.txt"
                    lock.write_bytes(self._requirements)
                    self._run([sys._base_executable, "-m", "venv", str(stage / "venv")])
                    python = stage / "venv" / "Scripts" / "python.exe"
                    self._run([str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
                               "--index-url", "https://pypi.org/simple", "--require-hashes", "--only-binary=:all:",
                               "--no-input", "-r", str(lock)])
                    self._run([str(python), "-m", "pip", "--isolated", "check"])
                    # Metadata alone misses optional native DLL failures (e.g. SoundFile on Windows).
                    self._run([str(python), "-c", "from transformers import ParakeetForTDT; import soundfile; import torch; assert torch.cuda.is_available()"])
                    manifest = {"requirements_sha256": self.version, "model_revision": MODEL_REVISION,
                                "model_files": {name: digest for name, (_, digest) in MODEL_FILES.items()}}
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
                    raise PhononRuntimeError("Phonon installation is incomplete; retry setup")
                self._set_state("installed", "Phonon is installed; model is unloaded")
                return installed
        except Exception as error:
            safe = str(error) if isinstance(error, PhononRuntimeError) else "Phonon installation failed; check local disk space and retry setup"
            self._set_state("error", safe)
            raise PhononRuntimeError(safe) from None


class PhononRuntime:
    """One app-owned model, serialized inference and observable lazy setup."""

    def __init__(self, root: Path, temp_root: Path):
        self.installation = PhononInstallation(root)
        self.temp_root = temp_root.resolve()
        self._operation = threading.Lock()
        self._dispose_guard = threading.Lock()
        self._closing = threading.Event()
        self._state_guard = threading.Lock()
        self._state = {"state": "absent", "detail": "Phonon is unloaded"}
        self._runtime = {}
        self._owned = None
        self._reader = None
        self._replies = None
        self._log = None
        self._loaded = False
        self._background = None

    def _publish(self, state, detail):
        with self._state_guard:
            self._state = {"state": state, "detail": detail}

    def status(self):
        with self._state_guard:
            state = dict(self._state)
            runtime = dict(self._runtime)
        process = self._owned.process if self._owned else None
        alive = process is not None and process.poll() is None
        if state["state"] == "absent":
            state = self.installation.status()
        elif state["state"] == "installing":
            setup = self.installation.status()
            if setup["state"] in {"installing", "error"}:
                state = setup
        elif state["state"] in {"ready", "busy"} and not alive:
            state = {"state": "error", "detail": "Phonon worker exited; retry Load/Test"}
        return {**state, "model_loaded": self._loaded and alive,
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
            self._owned.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self._owned.process.stdin.flush()
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            raise PhononRuntimeError("Phonon inference reached its configured deadline; completed chunks are preserved") from None
        except (OSError, ValueError):
            raise PhononRuntimeError("Phonon worker connection failed; retry Load/Test") from None
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, dict) or reply.get("protocol") != 1 or reply.get("id") != request_id:
            raise PhononRuntimeError("Phonon returned an invalid worker response; completed chunks are preserved")
        if reply.get("error"):
            if reply["error"] == "invalid_input":
                raise PhononRuntimeError("Phonon input must be a valid bounded 16 kHz mono PCM16 WAV")
            reason = "could not load its model" if reply["error"] == "load_failed" else "could not decode the input"
            raise PhononRuntimeError(f"Phonon {reason}; check GPU memory and the local worker log, then retry")
        if not isinstance(reply.get("runtime"), dict):
            raise PhononRuntimeError("Phonon returned invalid runtime metadata")
        with self._state_guard:
            self._runtime = dict(reply["runtime"])
        return reply

    def _spawn(self, installed):
        from .phonon_process import OwnedPhononProcess
        source = UPSTREAM
        manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
        if set(manifest) != {"phonon2_cuda_engine.py", "fermion_container.py", "hotwords.py", "cues.py", "_live.py"}:
            raise PhononRuntimeError("Phonon packaged source manifest is incomplete; update the app")
        for name, entry in manifest.items():
            if Path(name).name != name or _sha256(source / name) != entry["sha256"]:
                raise PhononRuntimeError("Phonon packaged inference source is damaged; update the app before retrying")
        self.temp_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PHONON2_CUDA_DTYPE="bfloat16", PHONON2_CUDA_K="16",
                   PHONON2_CUDA_PAD_S="5", PHONON2_CUDA_WARMUP="all")
        with self._dispose_guard:
            if self._closing.is_set():
                raise PhononRuntimeError("Phonon runtime is stopping")
            self._log = (self.installation.root / "worker.log").open("ab")
            self._owned = OwnedPhononProcess(
                [str(installed.python), "-u", str(WORKER), "--model-dir", str(installed.model),
                 "--temp-root", str(self.temp_root), "--max-bytes", str(MAX_INPUT_BYTES)],
                env=env, stderr=self._log,
            )
            replies = self._replies = queue.Queue()
            stream = self._owned.process.stdout

            def receive():
                try:
                    for line in stream:
                        try:
                            replies.put(json.loads(line))
                        except (ValueError, TypeError):
                            replies.put(PhononRuntimeError("Phonon emitted invalid worker data"))
                            break
                except (OSError, ValueError, UnicodeError):
                    pass
                finally:
                    replies.put(PhononRuntimeError("Phonon worker exited; completed chunks are preserved"))

            self._reader = threading.Thread(target=receive, name="phonon-output", daemon=True)
            self._reader.start()
        identity = self._exchange("status")["runtime"]
        pid = identity.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise PhononRuntimeError("Phonon worker returned an invalid process identity")
        self._owned.track_worker(pid)  # Membership checked before the first GPU load.

    def _load_locked(self, before_load=None):
        if self._closing.is_set():
            raise PhononRuntimeError("Phonon runtime is stopping")
        if self._loaded and self._owned and self._owned.process.poll() is None:
            return False
        self._dispose()
        self._publish("installing", "Preparing Phonon")
        installed = self.installation.ensure()
        self._publish("loading", "Loading Phonon and warming its CUDA graphs")
        if before_load:
            before_load()
        self._spawn(installed)
        runtime = self._exchange("load")["runtime"]
        if (runtime.get("model") != "phonon-2" or runtime.get("device") != "cuda"
                or runtime.get("dtype") != "bfloat16" or runtime.get("cuda_graph_steps") != 16
                or runtime.get("warm_buckets") != 6 or not runtime.get("model_loaded")):
            raise PhononRuntimeError("Phonon could not verify its optimized CUDA model; check the local worker log")
        self._loaded = True
        self._readiness_locked()
        return True

    def _readiness_locked(self):
        silence = io.BytesIO()
        with wave.open(silence, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(32000))
        self._transcribe_locked(silence.getvalue(), 1.0)
        self._publish("ready", "Phonon is loaded and ready")

    def _failure(self, error):
        logging.getLogger(__name__).error("Phonon runtime operation failed", exc_info=(type(error), error, error.__traceback__))
        try:
            self._dispose()
        except Exception:
            safe = "Phonon could not confirm its worker stopped; check the local worker log"
            self._publish("error", safe)
            return PhononRuntimeError(safe)
        safe = str(error) if isinstance(error, PhononRuntimeError) else "Phonon worker failed; check the local worker log and retry Load/Test"
        self._publish("error", safe)
        return PhononRuntimeError(safe)

    def load(self, *, test_existing=False, before_load=None):
        with self._operation:
            try:
                created = self._load_locked(before_load=before_load)
                if test_existing and not created:
                    if before_load:
                        before_load()
                    self._publish("busy", "Testing Phonon readiness")
                    self._readiness_locked()
                return self.status()
            except PhononBusyError as error:
                self._publish("error", str(error))
                raise
            except Exception as error:
                raise self._failure(error) from None

    def request_load(self, before_load=None):
        with self._state_guard:
            if self._closing.is_set():
                raise PhononRuntimeError("Phonon runtime is stopping")
            if self._background and self._background.is_alive():
                return

            def run():
                try:
                    self.load(test_existing=True, before_load=before_load)
                except PhononRuntimeError:
                    pass  # Observable error state; the next explicit action may retry.

            self._background = threading.Thread(target=run, name="phonon-load", daemon=True)
            self._state = {"state": "loading", "detail": "Preparing Phonon"}
            self._background.start()

    def _transcribe_locked(self, audio, duration, timeout=None):
        path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="phonon-ipc-", suffix=".wav", dir=self.temp_root, delete=False) as stream:
                path = Path(stream.name)
                stream.write(audio)
            reply = self._exchange("transcribe", audio=str(path), duration=duration, timeout=timeout)
            if not isinstance(reply.get("result"), dict):
                raise PhononRuntimeError("Phonon returned invalid transcription data")
            from .phonon import PhononError, validate_result
            try:
                validate_result(reply["result"], duration)
            except PhononError as error:
                raise PhononRuntimeError(str(error)) from None
            return {**reply["result"], "runtime": reply["runtime"]}
        finally:
            if path:
                path.unlink(missing_ok=True)

    def transcribe(self, audio: bytes, duration: float, timeout=None):
        if (not isinstance(audio, bytes) or not 0 < len(audio) < MAX_INPUT_BYTES
                or not math.isfinite(duration) or not 0 < duration <= 7200):
            raise PhononRuntimeError("Phonon requires a bounded, nonempty WAV input of at most two hours")
        if timeout is not None and (not math.isfinite(float(timeout)) or float(timeout) <= 0):
            raise PhononRuntimeError("Phonon inference deadline must be positive and finite")
        with self._operation:
            try:
                self._load_locked()
                self._publish("busy", "Phonon is transcribing")
                result = self._transcribe_locked(audio, duration, timeout)
                self._publish("ready", "Phonon is loaded and ready")
                return result
            except Exception as error:
                raise self._failure(error) from None

    def unload(self, *, wait=False):
        if not wait and self._background and self._background.is_alive():
            raise PhononRuntimeError("Phonon is busy loading; unload after the current operation completes")
        if not self._operation.acquire(blocking=wait):
            raise PhononRuntimeError("Phonon is busy loading or decoding; unload after the current operation completes")
        try:
            self._publish("unloading", "Unloading Phonon")
            if self._owned and self._owned.process.poll() is None:
                self._exchange("shutdown")
            self._dispose()
            self._publish("absent", "Phonon is unloaded; cached model and runtime are retained")
            return self.status()
        except Exception as error:
            raise self._failure(error) from None
        finally:
            self._operation.release()

    def shutdown(self):
        # App shutdown may terminate its own in-flight worker. No other process is touched.
        self._closing.set()
        self._dispose()
        self._publish("absent", "Phonon is stopped")
