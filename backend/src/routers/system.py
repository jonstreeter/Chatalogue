"""System runtime endpoints: CUDA health, component installers, version, restart, workers."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..deps import get_ingestion_service
from ..schemas import (
    ClearVoiceInstallInfo,
    ClearVoiceTestResult,
    ReconstructionInstallInfo,
    ReconstructionTestResult,
    VoiceFixerInstallInfo,
    VoiceFixerTestResult,
)

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/system/cuda-health")
def system_cuda_health():
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service not ready")
    try:
        payload = get_ingestion_service().get_cuda_health_status()
        if isinstance(payload, dict):
            training_memory = _main()._collect_avatar_training_process_memory()
            component_memory = payload.get("component_memory")
            if not isinstance(component_memory, dict):
                component_memory = {}
                payload["component_memory"] = component_memory
            component_memory["avatar_training"] = {
                "loaded": bool(training_memory.get("loaded")),
                "ram_gb": float(training_memory.get("ram_gb") or 0.0),
                "vram_gb": float(training_memory.get("vram_gb") or 0.0),
            }

            memory = payload.get("memory")
            if isinstance(memory, dict):
                try:
                    memory["allocated_gb"] = round(
                        float(memory.get("allocated_gb") or 0.0) + float(training_memory.get("vram_gb") or 0.0),
                        2,
                    )
                except Exception:
                    pass
            system_memory = payload.get("system_memory")
            if isinstance(system_memory, dict):
                try:
                    system_memory["rss_gb"] = round(
                        float(system_memory.get("rss_gb") or 0.0) + float(training_memory.get("ram_gb") or 0.0),
                        2,
                    )
                except Exception:
                    pass
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect CUDA health: {e}")


@router.get("/system/cuda-restart-state")
def get_cuda_restart_state():
    state_file = _main().BACKEND_RUNTIME_DIR / "cuda_restart_state.json"
    if not state_file.exists():
        return {"restart_timestamps": [], "permanent_cpu_mode": False}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"restart_timestamps": [], "permanent_cpu_mode": False}


@router.post("/system/cuda-restart-state/reset")
def reset_cuda_restart_state():
    state_file = _main().BACKEND_RUNTIME_DIR / "cuda_restart_state.json"
    try:
        state_file.unlink(missing_ok=True)
    except Exception:
        pass
    return {"status": "cleared"}


@router.get("/system/cloudflared/install-info")
def get_cloudflared_install_info(request: Request):
    _main()._require_local_operator(request)
    target = _main()._cloudflared_install_target()
    return {
        **target,
        "installed": bool(_main()._refresh_cloudflared_availability()),
    }


@router.post("/system/cloudflared/install")
def install_cloudflared(request: Request):
    _main()._require_local_operator(request)
    if _main()._refresh_cloudflared_availability():
        info = _main()._cloudflared_install_target()
        return {
            "status": "already_installed",
            **info,
            "installed": True,
        }

    target = _main()._cloudflared_install_target()
    if not bool(target.get("package_manager_available")):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Automatic install is unavailable on {target.get('platform')} because "
                f"{target.get('package_manager') or 'a supported package manager'} was not found. "
                f"Install cloudflared manually from {target.get('download_url')}."
            ),
        )

    try:
        result = _main()._install_cloudflared_via_package_manager()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while installing cloudflared.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cloudflared install failed: {e}")

    if not result.get("installed"):
        detail = str(result.get("stderr") or result.get("stdout") or "unknown installer failure")
        raise HTTPException(status_code=500, detail=f"cloudflared install did not complete successfully: {detail[:700]}")

    _main()._append_share_event(f"cloudflared installed via {result.get('package_manager')}")
    return {
        "status": "installed",
        **result,
        "download_url": target.get("download_url"),
    }


@router.get("/system/voicefixer/install-info", response_model=VoiceFixerInstallInfo)
def get_voicefixer_install_info(request: Request):
    _main()._require_local_operator(request)
    return _main()._get_voicefixer_install_info()


@router.post("/system/voicefixer/install", response_model=VoiceFixerInstallInfo)
def install_voicefixer(request: Request):
    import sys

    _main()._require_local_operator(request)
    info = _main()._get_voicefixer_install_info()
    if info.installed:
        return info

    cmd = [sys.executable, "-m", "pip", "install", _main().VOICEFIXER_PACKAGE_SPEC]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while installing VoiceFixer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VoiceFixer install failed: {e}")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown installer failure").strip()
        raise HTTPException(status_code=500, detail=f"VoiceFixer install failed: {detail[:900]}")

    refreshed = _main()._get_voicefixer_install_info()
    if not refreshed.installed:
        if _main()._voicefixer_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "VoiceFixer finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return VoiceFixerInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="VoiceFixer install completed but the package is still unavailable to the backend.")
    return VoiceFixerInstallInfo(
        **refreshed.model_dump(),
        message="VoiceFixer installed successfully.",
    )


@router.post("/system/voicefixer/test", response_model=VoiceFixerTestResult)
def test_voicefixer(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_voicefixer_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="VoiceFixer is not installed in the backend environment yet.")

    try:
        from voicefixer import VoiceFixer  # type: ignore
        restorer = VoiceFixer()
        detail = "VoiceFixer imported and instantiated successfully."
        del restorer
        return VoiceFixerTestResult(
            status="ok",
            version=info.version,
            instantiated=True,
            detail=detail,
        )
    except Exception as e:
        error_text = str(e)
        detail = "VoiceFixer import or model initialization failed."
        if "failed finding central directory" in error_text.lower():
            detail = "VoiceFixer found a corrupted analysis checkpoint. Run the repair action in Settings, then re-run the self-test."
        return VoiceFixerTestResult(
            status="error",
            version=info.version,
            instantiated=False,
            error=error_text,
            detail=detail,
        )


@router.post("/system/voicefixer/repair")
def repair_voicefixer(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_voicefixer_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="VoiceFixer is not installed in the backend environment yet.")
    try:
        result = _main()._download_voicefixer_analysis_checkpoint()
        return {
            "status": "repaired",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VoiceFixer checkpoint repair failed: {e}")


@router.get("/system/clearvoice/install-info", response_model=ClearVoiceInstallInfo)
def get_clearvoice_install_info(request: Request):
    _main()._require_local_operator(request)
    return _main()._get_clearvoice_install_info()


@router.post("/system/clearvoice/install", response_model=ClearVoiceInstallInfo)
def install_clearvoice(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_clearvoice_install_info()
    if info.installed and info.runtime_ready:
        return info

    if not info.installed:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", _main().CLEARVOICE_PACKAGE_SPEC]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing ClearVoice.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ClearVoice install failed: {e}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown installer failure").strip()
            raise HTTPException(status_code=500, detail=f"ClearVoice install failed: {detail[:900]}")
        try:
            _main()._normalize_clearvoice_metadata()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ClearVoice metadata normalization failed: {e}")

    refreshed = _main()._get_clearvoice_install_info()
    if not refreshed.installed:
        if _main()._clearvoice_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "ClearVoice finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return ClearVoiceInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="ClearVoice install completed but the package is still unavailable to the backend.")

    if not refreshed.restart_required and not refreshed.runtime_ready:
        runtime_error_text = str(refreshed.runtime_error or "").lower()
        if "torchaudio import failed" in runtime_error_text:
            try:
                _main()._repair_clearvoice_runtime()
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Timed out while repairing the ClearVoice torchaudio runtime.")
            except Exception as e:
                payload = refreshed.model_dump()
                payload["message"] = (
                    "ClearVoice installed, but torchaudio is not usable in the backend environment. "
                    f"Run ClearVoice runtime repair in Settings. Repair error: {e}"
                )
                return ClearVoiceInstallInfo(**payload)
            refreshed = _main()._get_clearvoice_install_info()

    return ClearVoiceInstallInfo(
        **refreshed.model_dump(),
        message=refreshed.message or ("ClearVoice installed successfully." if refreshed.runtime_ready else "ClearVoice installed, but runtime repair is still required."),
    )


@router.post("/system/clearvoice/test", response_model=ClearVoiceTestResult)
def test_clearvoice(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_clearvoice_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="ClearVoice is not installed in the backend environment yet.")
    runtime = _main()._inspect_clearvoice_runtime()
    return ClearVoiceTestResult(
        status="ok" if bool(runtime.get("runtime_ready")) else "error",
        version=info.version,
        imported=bool(runtime.get("clearvoice_imported")),
        class_available=bool(runtime.get("class_available")),
        torch_imported=bool(runtime.get("torch_imported")),
        torchaudio_imported=bool(runtime.get("torchaudio_imported")),
        runtime_ready=bool(runtime.get("runtime_ready")),
        torch_version=str(runtime.get("torch_version") or "").strip() or None,
        torchaudio_version=str(runtime.get("torchaudio_version") or "").strip() or None,
        error=str(runtime.get("error") or "").strip() or None,
        detail=str(runtime.get("detail") or "").strip() or None,
    )


@router.post("/system/clearvoice/repair")
def repair_clearvoice(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_clearvoice_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="ClearVoice is not installed in the backend environment yet.")
    try:
        result = _main()._repair_clearvoice_runtime()
        metadata_result = _main()._normalize_clearvoice_metadata()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while repairing the ClearVoice runtime.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClearVoice runtime repair failed: {e}")
    runtime = _main()._inspect_clearvoice_runtime()
    return {
        **result,
        "metadata": metadata_result,
        "runtime_ready": bool(runtime.get("runtime_ready")),
        "runtime_error": str(runtime.get("error") or "").strip() or None,
        "torch_version": str(runtime.get("torch_version") or "").strip() or None,
        "torchaudio_version": str(runtime.get("torchaudio_version") or "").strip() or None,
        "detail": str(runtime.get("detail") or "").strip() or None,
    }


@router.get("/system/reconstruction/install-info", response_model=ReconstructionInstallInfo)
def get_reconstruction_install_info(request: Request):
    _main()._require_local_operator(request)
    return _main()._get_reconstruction_install_info()


@router.post("/system/reconstruction/install", response_model=ReconstructionInstallInfo)
def install_reconstruction_runtime(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_reconstruction_install_info()
    qwen_installed = bool(info.installed)
    sox_installed = _main()._sox_available()
    if qwen_installed and sox_installed:
        return info

    if not qwen_installed:
        cmd = [sys.executable, "-m", "pip", "install", _main().RECONSTRUCTION_PACKAGE_SPEC]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing the reconstruction runtime.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconstruction runtime install failed: {e}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown installer failure").strip()
            raise HTTPException(status_code=500, detail=f"Reconstruction runtime install failed: {detail[:900]}")

    if not sox_installed:
        target = _main()._sox_install_target()
        if not bool(target.get("package_manager_available")):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"SoX is required for conversation reconstruction on {target.get('platform')}, but automatic install is unavailable because "
                    f"{target.get('package_manager') or 'a supported package manager'} was not found. "
                    f"Install SoX manually from {target.get('download_url')}."
                ),
            )
        try:
            sox_result = _main()._install_sox_via_package_manager()
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Timed out while installing SoX.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SoX install failed: {e}")

        if not sox_result.get("installed"):
            detail = str(sox_result.get("stderr") or sox_result.get("stdout") or "unknown installer failure")
            raise HTTPException(status_code=500, detail=f"SoX install did not complete successfully: {detail[:700]}")

    refreshed = _main()._get_reconstruction_install_info()
    if not refreshed.installed:
        if _main()._reconstruction_installed_via_pip():
            payload = refreshed.model_dump()
            payload.update({
                "installed": False,
                "restart_required": True,
                "message": "The Qwen TTS runtime finished installing, but the running backend cannot see it yet. Restart the backend to finish activation.",
            })
            return ReconstructionInstallInfo(**payload)
        raise HTTPException(status_code=500, detail="The reconstruction runtime installed but is still unavailable to the backend.")
    if not _main()._sox_available():
        target = _main()._sox_install_target()
        raise HTTPException(status_code=500, detail=f"SoX still is not available on PATH after install. Install it manually from {target.get('download_url')}.")
    return ReconstructionInstallInfo(
        **refreshed.model_dump(),
        message="The reconstruction runtime and SoX installed successfully.",
    )


@router.post("/system/reconstruction/test", response_model=ReconstructionTestResult)
def test_reconstruction_runtime(request: Request):
    _main()._require_local_operator(request)
    info = _main()._get_reconstruction_install_info()
    if not info.installed:
        raise HTTPException(status_code=400, detail="The reconstruction runtime is not installed in the backend environment yet.")

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore

        detail = "Imported qwen_tts and found Qwen3TTSModel. Model weights are not loaded during this self-test."
        if not _main()._sox_available():
            target = _main()._sox_install_target()
            return ReconstructionTestResult(
                status="error",
                version=info.version,
                imported=True,
                model_class_available=Qwen3TTSModel is not None,
                error="SoX is not on PATH.",
                detail=f"SoX is required for stable conversation reconstruction on this machine. Install it via {target.get('package_manager') or 'a supported package manager'} or from {target.get('download_url')}.",
            )
        return ReconstructionTestResult(
            status="ok",
            version=info.version,
            imported=True,
            model_class_available=Qwen3TTSModel is not None,
            detail=detail,
        )
    except Exception as e:
        return ReconstructionTestResult(
            status="error",
            version=info.version,
            imported=False,
            model_class_available=False,
            error=str(e),
            detail="Importing qwen_tts failed.",
        )




@router.get("/system/setup-status")
def get_setup_status():
    """Check if initial setup wizard has been completed."""
    return {
        "setup_completed": os.getenv("SETUP_WIZARD_COMPLETED", "false").lower() == "true",
        "hf_token_set": bool((os.getenv("HF_TOKEN") or "").strip()),
    }


def _repo_root_path() -> Path:
    # backend/src/main.py -> project root (../../)
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _safe_stdout(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout or "").strip()


def _safe_stderr(proc: subprocess.CompletedProcess) -> str:
    return (proc.stderr or "").strip()


def _purge_runtime_models(reason: str = "manual_restart") -> dict:
    if not get_ingestion_service():
        return {"ok": False, "detail": "ingestion service unavailable"}
    try:
        if hasattr(get_ingestion_service(), "purge_loaded_models"):
            details = get_ingestion_service().purge_loaded_models(reason=reason)
            return {"ok": True, **(details or {})}
    except Exception as e:
        return {"ok": False, "detail": str(e)}

    # Backward-compatible fallback path.
    try:
        get_ingestion_service().diarization_pipeline = None
        get_ingestion_service().embedding_model = None
        get_ingestion_service().embedding_inference = None
        get_ingestion_service().whisper_model = None
        get_ingestion_service().parakeet_model = None
        get_ingestion_service()._whisper_compute_type = None
        get_ingestion_service()._force_float32 = False
        get_ingestion_service()._cuda_unhealthy_reason = None
        get_ingestion_service()._cuda_unhealthy_since = None
        get_ingestion_service()._parakeet_dynamic_batch_cap = None
        get_ingestion_service().device = None
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def _get_system_version_info(check_remote: bool = True) -> dict:
    repo_root = _repo_root_path()
    info = {
        "status": "ok",
        "app_version": os.getenv("CHATALOGUE_VERSION", "").strip() or None,
        "repo_path": str(repo_root),
        "git": {
            "available": False,
            "is_repo": False,
            "branch": None,
            "head": None,
            "head_short": None,
            "remote_head": None,
            "remote_head_short": None,
            "ahead_count": 0,
            "behind_count": 0,
            "dirty": False,
            "update_available": False,
            "error": None,
            "checked_remote": bool(check_remote),
        },
    }

    # Ensure git binary is available.
    git_v = _run_git(["--version"], repo_root, timeout=5)
    if git_v.returncode != 0:
        info["git"]["error"] = "git is not available in PATH"
        return info
    info["git"]["available"] = True

    # Ensure this directory is a git repo.
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], repo_root, timeout=5)
    if inside.returncode != 0 or _safe_stdout(inside).lower() != "true":
        info["git"]["error"] = "not a git work tree"
        return info
    info["git"]["is_repo"] = True

    branch_p = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root, timeout=5)
    if branch_p.returncode == 0:
        info["git"]["branch"] = _safe_stdout(branch_p) or None

    head_p = _run_git(["rev-parse", "HEAD"], repo_root, timeout=5)
    if head_p.returncode == 0:
        info["git"]["head"] = _safe_stdout(head_p) or None

    head_short_p = _run_git(["rev-parse", "--short", "HEAD"], repo_root, timeout=5)
    if head_short_p.returncode == 0:
        info["git"]["head_short"] = _safe_stdout(head_short_p) or None

    dirty_p = _run_git(["status", "--porcelain"], repo_root, timeout=8)
    if dirty_p.returncode == 0:
        info["git"]["dirty"] = bool(_safe_stdout(dirty_p))

    if not check_remote:
        return info

    branch = info["git"]["branch"] or "main"
    fetch_p = _run_git(["fetch", "--quiet", "origin", branch], repo_root, timeout=20)
    if fetch_p.returncode != 0:
        info["git"]["error"] = _safe_stderr(fetch_p) or "git fetch failed"
        return info

    remote_ref = f"origin/{branch}"
    remote_head_p = _run_git(["rev-parse", remote_ref], repo_root, timeout=5)
    if remote_head_p.returncode == 0:
        info["git"]["remote_head"] = _safe_stdout(remote_head_p) or None

    remote_head_short_p = _run_git(["rev-parse", "--short", remote_ref], repo_root, timeout=5)
    if remote_head_short_p.returncode == 0:
        info["git"]["remote_head_short"] = _safe_stdout(remote_head_short_p) or None

    behind_p = _run_git(["rev-list", "--count", f"HEAD..{remote_ref}"], repo_root, timeout=8)
    if behind_p.returncode == 0:
        try:
            info["git"]["behind_count"] = int(_safe_stdout(behind_p) or "0")
        except Exception:
            info["git"]["behind_count"] = 0

    ahead_p = _run_git(["rev-list", "--count", f"{remote_ref}..HEAD"], repo_root, timeout=8)
    if ahead_p.returncode == 0:
        try:
            info["git"]["ahead_count"] = int(_safe_stdout(ahead_p) or "0")
        except Exception:
            info["git"]["ahead_count"] = 0

    info["git"]["update_available"] = bool(info["git"]["behind_count"] > 0)
    return info


@router.get("/system/version")
def get_system_version(check_remote: bool = True):
    try:
        return _get_system_version_info(check_remote=check_remote)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "git": {
                "update_available": False,
                "behind_count": 0,
                "ahead_count": 0,
                "error": str(e),
            },
        }

@router.post("/system/restart")
def restart_server():
    """Purge loaded models and trigger a server restart when reload mode is active."""
    import time as _time

    purge_result = _purge_runtime_models(reason="system_restart")
    main_file = Path(__file__)
    main_file.touch()
    return {
        "status": "restarting",
        "model_purge": purge_result,
        "touched": str(main_file),
        "timestamp": _time.time(),
    }


@router.post("/system/update")
def update_and_restart_server():
    """Pull latest code from origin/<branch> (fast-forward only) and trigger restart."""
    import time as _time

    info = _get_system_version_info(check_remote=True)
    git = info.get("git", {}) if isinstance(info, dict) else {}

    if not git.get("available") or not git.get("is_repo"):
        raise HTTPException(status_code=400, detail=f"Update unavailable: {git.get('error') or 'git not ready'}")

    if git.get("dirty"):
        raise HTTPException(status_code=409, detail="Local repository has uncommitted changes; refusing auto-update.")

    branch = str(git.get("branch") or "main")
    repo_root = _repo_root_path()
    old_head = str(git.get("head") or "")
    pull_p = _run_git(["pull", "--ff-only", "origin", branch], repo_root, timeout=45)
    if pull_p.returncode != 0:
        detail = _safe_stderr(pull_p) or _safe_stdout(pull_p) or "git pull failed"
        raise HTTPException(status_code=500, detail=f"Update failed: {detail[:500]}")

    # Re-read HEAD after pull.
    new_head_p = _run_git(["rev-parse", "HEAD"], repo_root, timeout=5)
    new_head = _safe_stdout(new_head_p) if new_head_p.returncode == 0 else old_head
    updated = bool(new_head and old_head and new_head != old_head)

    purge_result = _purge_runtime_models(reason="system_update")

    # Trigger existing restart mechanism.
    main_file = Path(__file__)
    main_file.touch()
    return {
        "status": "restarting",
        "updated": updated,
        "model_purge": purge_result,
        "old_head": old_head or None,
        "new_head": new_head or None,
        "branch": branch,
        "touched": str(main_file),
        "timestamp": _time.time(),
    }

@router.get("/system/worker-status")
def get_worker_status():
    """Check queue workers and process heartbeat health."""

    heartbeat_file = Path(__file__).parent.parent / "data" / "worker_heartbeat"
    worker_alive = {name: bool(t and t.is_alive()) for name, t in (_main().worker_threads or {}).items()}
    if not heartbeat_file.exists():
        overall = "offline" if not any(worker_alive.values()) else "stalled"
        return {"status": overall, "last_heartbeat": None, "workers": worker_alive}

    try:
        last_heartbeat = float(heartbeat_file.read_text().strip())
        age = time.time() - last_heartbeat

        # If heartbeat is older than 30 seconds, process worker is stalled.
        heartbeat_status = "online" if age < 30 else "stalled"
        if not worker_alive:
            status = heartbeat_status
        elif heartbeat_status == "online" and all(worker_alive.values()):
            status = "online"
        else:
            status = "stalled"
        return {
            "status": status,
            "last_heartbeat": last_heartbeat,
            "age_seconds": age,
            "workers": worker_alive,
            "heartbeat_status": heartbeat_status,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
