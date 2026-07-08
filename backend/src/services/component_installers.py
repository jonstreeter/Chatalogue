"""Optional component installers: VoiceFixer, ClearVoice, reconstruction TTS, SoX.

Install/inspect/repair helpers extracted from main.py; used by the system and
video_media routers.
"""
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from ..schemas import ClearVoiceInstallInfo, ReconstructionInstallInfo, VoiceFixerInstallInfo

CLEARVOICE_PACKAGE_SPEC = "clearvoice==0.1.2"
VOICEFIXER_PACKAGE_SPEC = "voicefixer==0.1.3"
RECONSTRUCTION_PACKAGE_SPEC = "qwen-tts==0.1.1"


def _get_voicefixer_install_info() -> VoiceFixerInstallInfo:
    import sys
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    try:
        version = importlib_metadata.version("voicefixer")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _voicefixer_installed_via_pip():
        restart_required = True
        message = "VoiceFixer is installed in the backend environment, but the running backend needs a restart before it can use it."

    return VoiceFixerInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=VOICEFIXER_PACKAGE_SPEC,
        restart_required=restart_required,
        message=message,
    )


def _voicefixer_installed_via_pip() -> bool:
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "voicefixer"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: voicefixer" in (result.stdout or "")
    except Exception:
        return False


def _get_clearvoice_install_info() -> ClearVoiceInstallInfo:
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    runtime_ready = False
    runtime_error = None
    torch_version = None
    torchaudio_version = None
    try:
        version = importlib_metadata.version("clearvoice")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _clearvoice_installed_via_pip():
        restart_required = True
        message = "ClearVoice is installed in the backend environment, but the running backend needs a restart before it can use it."
    elif installed and not restart_required:
        runtime = _inspect_clearvoice_runtime()
        runtime_ready = bool(runtime.get("runtime_ready"))
        runtime_error = str(runtime.get("error") or "").strip() or None
        torch_version = str(runtime.get("torch_version") or "").strip() or None
        torchaudio_version = str(runtime.get("torchaudio_version") or "").strip() or None
        if not runtime_ready:
            message = str(runtime.get("detail") or "").strip() or "ClearVoice is installed, but its runtime dependencies are not healthy."

    return ClearVoiceInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=CLEARVOICE_PACKAGE_SPEC,
        restart_required=restart_required,
        runtime_ready=runtime_ready,
        runtime_error=runtime_error,
        torch_version=torch_version,
        torchaudio_version=torchaudio_version,
        message=message,
    )


def _clearvoice_installed_via_pip() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "clearvoice"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: clearvoice" in (result.stdout or "")
    except Exception:
        return False


def _normalize_torch_version(raw_version: str | None) -> str | None:
    value = str(raw_version or "").strip()
    if not value:
        return None
    return value.split("+", 1)[0].strip() or None


def _torch_wheel_variant(raw_version: str | None) -> str:
    value = str(raw_version or "").strip().lower()
    if "+" in value:
        suffix = value.split("+", 1)[1].strip()
        if suffix:
            return suffix
    return "cpu"


def _inspect_clearvoice_runtime() -> dict[str, object]:
    torch_imported = False
    torchaudio_imported = False
    clearvoice_imported = False
    class_available = False
    torch_version = None
    torchaudio_version = None
    error = None
    detail = None

    try:
        import torch  # type: ignore

        torch_imported = True
        torch_version = str(getattr(torch, "__version__", "") or "").strip() or None
    except Exception as e:
        error = f"torch import failed: {e}"
        detail = "ClearVoice is installed, but torch could not be imported in the backend environment."

    if error is None:
        try:
            import torchaudio  # type: ignore

            torchaudio_imported = True
            torchaudio_version = str(getattr(torchaudio, "__version__", "") or "").strip() or None
        except Exception as e:
            error = f"torchaudio import failed: {e}"
            detail = (
                "ClearVoice is installed, but torchaudio could not be imported. "
                "The backend environment likely has mismatched torch/torchaudio wheels. "
                "Run ClearVoice runtime repair in Settings."
            )

    try:
        from clearvoice import ClearVoice  # type: ignore

        clearvoice_imported = True
        class_available = callable(ClearVoice)
    except Exception as e:
        if error is None:
            error = f"clearvoice import failed: {e}"
            detail = "ClearVoice package import failed in the backend environment."

    runtime_ready = bool(torch_imported and torchaudio_imported and clearvoice_imported and class_available)
    if runtime_ready:
        detail = "Imported torch, torchaudio, and ClearVoice successfully. Model weights are not loaded during this self-test."

    return {
        "runtime_ready": runtime_ready,
        "torch_imported": torch_imported,
        "torchaudio_imported": torchaudio_imported,
        "clearvoice_imported": clearvoice_imported,
        "class_available": class_available,
        "torch_version": torch_version,
        "torchaudio_version": torchaudio_version,
        "error": error,
        "detail": detail,
    }


def _repair_clearvoice_runtime() -> dict[str, object]:
    try:
        import torch  # type: ignore

        torch_version_raw = str(getattr(torch, "__version__", "") or "").strip() or None
    except Exception:
        from importlib import metadata as importlib_metadata

        try:
            torch_version_raw = importlib_metadata.version("torch")
        except Exception as e:
            raise RuntimeError(f"Could not determine the installed torch version: {e}")

    normalized_version = _normalize_torch_version(torch_version_raw)
    if not normalized_version:
        raise RuntimeError("Could not determine a valid torch version for ClearVoice runtime repair.")
    variant = _torch_wheel_variant(torch_version_raw)
    index_url = f"https://download.pytorch.org/whl/{variant}" if variant.startswith("cu") else "https://download.pytorch.org/whl/cpu"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--index-url",
        index_url,
        f"torchaudio=={normalized_version}",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        detail = stderr or stdout or "unknown installer failure"
        raise RuntimeError(detail[:1200])
    return {
        "status": "repaired",
        "command": cmd,
        "index_url": index_url,
        "torch_version": torch_version_raw,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def _normalize_clearvoice_metadata() -> dict[str, object]:
    from importlib import metadata as importlib_metadata

    dist = importlib_metadata.distribution("clearvoice")
    meta_path = Path(getattr(dist, "_path", "")) / "METADATA"
    if not meta_path.exists():
        raise RuntimeError(f"Could not locate ClearVoice METADATA at {meta_path}")

    original = meta_path.read_text(encoding="utf-8")
    updated = original
    replacements = {
        "Requires-Dist: numpy<2.0,>=1.24.3": "Requires-Dist: numpy>=1.24.3",
        "Requires-Dist: soundfile==0.12.1": "Requires-Dist: soundfile>=0.12.1",
    }
    changed: list[dict[str, str]] = []
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            changed.append({"from": old, "to": new})

    if updated != original:
        meta_path.write_text(updated, encoding="utf-8")

    return {
        "metadata_path": str(meta_path),
        "updated": bool(changed),
        "changed": changed,
    }


def _get_reconstruction_install_info() -> ReconstructionInstallInfo:
    from importlib import metadata as importlib_metadata

    installed = False
    version = None
    restart_required = False
    message = None
    try:
        version = importlib_metadata.version("qwen-tts")
        installed = True
    except importlib_metadata.PackageNotFoundError:
        installed = False
    except Exception:
        installed = False

    if not installed and _reconstruction_installed_via_pip():
        restart_required = True
        message = "The reconstruction runtime is installed in the backend environment, but the running backend needs a restart before it can use it."
    elif installed and not _sox_available():
        target = _sox_install_target()
        if bool(target.get("package_manager_available")):
            message = f"The reconstruction runtime is installed, but SoX is missing. The install action will add SoX via {target.get('package_manager')}."
        else:
            message = f"The reconstruction runtime is installed, but SoX is missing. Install it from {target.get('download_url')}."

    return ReconstructionInstallInfo(
        installed=installed,
        version=version,
        python_executable=sys.executable,
        package_spec=RECONSTRUCTION_PACKAGE_SPEC,
        restart_required=restart_required,
        message=message,
    )


def _reconstruction_installed_via_pip() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "qwen-tts"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and "Name: qwen-tts" in (result.stdout or "")
    except Exception:
        return False


def _discover_sox_binary() -> str:
    direct = shutil.which("sox") or shutil.which("sox.exe") or ""
    if direct:
        return direct
    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA") or ""
        if local_appdata:
            winget_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            try:
                matches = sorted(winget_root.glob("ChrisBagwell.SoX_*/*/sox.exe"))
            except Exception:
                matches = []
            if matches:
                return str(matches[-1])
    return ""


def _sox_available() -> bool:
    sox_bin = _discover_sox_binary()
    if not sox_bin:
        return False
    sox_dir = str(Path(sox_bin).parent)
    current_path = os.environ.get("PATH") or ""
    path_parts = current_path.split(os.pathsep) if current_path else []
    if sox_dir and sox_dir not in path_parts:
        os.environ["PATH"] = current_path + (os.pathsep if current_path else "") + sox_dir
    return True


def _sox_install_target() -> dict[str, str | bool]:
    platform_name = "windows" if os.name == "nt" else ("macos" if sys.platform == "darwin" else sys.platform)
    winget_path = shutil.which("winget") or ""
    brew_path = shutil.which("brew") or ""
    if platform_name == "windows":
        return {
            "platform": platform_name,
            "package_manager": "winget" if winget_path else "",
            "package_manager_available": bool(winget_path),
            "package_id": "ChrisBagwell.SoX",
            "download_url": "https://sourceforge.net/projects/sox/files/sox/",
        }
    if platform_name == "macos":
        return {
            "platform": platform_name,
            "package_manager": "brew" if brew_path else "",
            "package_manager_available": bool(brew_path),
            "package_id": "sox",
            "download_url": "https://formulae.brew.sh/formula/sox",
        }
    return {
        "platform": platform_name,
        "package_manager": "",
        "package_manager_available": False,
        "package_id": "",
        "download_url": "https://sourceforge.net/projects/sox/files/sox/",
    }


def _install_sox_via_package_manager() -> dict[str, object]:
    target = _sox_install_target()
    platform_name = str(target.get("platform") or "")
    package_manager = str(target.get("package_manager") or "")
    package_id = str(target.get("package_id") or "")
    if platform_name == "windows" and package_manager == "winget":
        cmd = [
            "winget",
            "install",
            "--id",
            package_id,
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ]
    elif platform_name == "macos" and package_manager == "brew":
        cmd = ["brew", "install", "sox"]
    else:
        raise RuntimeError("Automatic SoX install is only supported on Windows via winget or macOS via Homebrew.")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return {
        "platform": platform_name,
        "package_manager": package_manager,
        "package_id": package_id,
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "installed": bool(_sox_available()),
        "download_url": str(target.get("download_url") or ""),
    }


def _voicefixer_analysis_checkpoint_path() -> Path:
    return Path.home() / ".cache" / "voicefixer" / "analysis_module" / "checkpoints" / "vf.ckpt"


def _download_voicefixer_analysis_checkpoint() -> dict[str, object]:
    import torch

    url = "https://zenodo.org/record/5600188/files/vf.ckpt?download=1"
    target = _voicefixer_analysis_checkpoint_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".download")

    if tmp_path.exists():
        tmp_path.unlink()

    bytes_written = 0
    expected_bytes = None
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if curl_bin:
        head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Chatalogue/VoiceFixer-Repair"})
        with urllib.request.urlopen(head_req, timeout=60) as response:
            try:
                expected_bytes = int(response.headers.get("Content-Length") or 0) or None
            except Exception:
                expected_bytes = None

        result = subprocess.run(
            [curl_bin, "-L", "--fail", "--output", str(tmp_path), url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "curl download failed").strip()
            raise RuntimeError(detail[:1200])
        bytes_written = int(tmp_path.stat().st_size)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Chatalogue/VoiceFixer-Repair"})
        with urllib.request.urlopen(req, timeout=60) as response, tmp_path.open("wb") as out:
            try:
                expected_bytes = int(response.headers.get("Content-Length") or 0) or None
            except Exception:
                expected_bytes = None
            while True:
                chunk = response.read(1024 * 1024 * 4)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)

    if expected_bytes is not None and bytes_written != expected_bytes:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Downloaded VoiceFixer checkpoint size mismatch: got {bytes_written} bytes, expected {expected_bytes}.")

    try:
        torch.load(str(tmp_path), map_location="cpu")
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise

    target.unlink(missing_ok=True)
    tmp_path.replace(target)
    return {
        "status": "ok",
        "path": str(target),
        "bytes_written": int(bytes_written),
        "expected_bytes": expected_bytes,
    }
