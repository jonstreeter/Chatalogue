"""App-owned optional Phonon installation. No application/DB imports here."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid

from filelock import FileLock


MODEL_REVISION = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
MODEL_FILES = {
    "model.fermion": (177438361, "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"),
    "config.json": (277493, "d0daad3b2a182893844f4abdc11e4f5b7083f7d42c8ad7e8203f71559785a31b"),
    "packed_manifest.json": (821, "3c1874501a1efb6ef569eab082aa68d91b93ec6e4d2fb00dbef7851edf2ae510"),
}
REQUIREMENTS = Path(__file__).with_name("phonon_worker_requirements.txt")


class PhononRuntimeError(RuntimeError):
    """Actionable app error; raw subprocess/download errors stay out of the UI."""


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
