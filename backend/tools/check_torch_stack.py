from __future__ import annotations

import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys


def _get_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except Exception:
        return None


def _detect_nvidia_gpu() -> dict:
    result = {
        "present": False,
        "count": 0,
        "names": [],
    }
    exe = shutil.which("nvidia-smi")
    if not exe:
        return result

    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return result

    if proc.returncode != 0:
        return result

    names = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    result["present"] = bool(names)
    result["count"] = len(names)
    result["names"] = names
    return result


def main() -> int:
    gpu = _detect_nvidia_gpu()
    summary = {
        "nvidia_gpu_present": gpu["present"],
        "nvidia_gpu_count": gpu["count"],
        "nvidia_gpu_names": gpu["names"],
        "torch_version": _get_version("torch"),
        "torchaudio_version": _get_version("torchaudio"),
    }

    try:
        import torch
    except Exception as exc:
        summary["torch_import_error"] = str(exc)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        if gpu["present"]:
            print(
                "Torch could not be imported even though an NVIDIA GPU is present. "
                "The backend would fall back to CPU.",
                file=sys.stderr,
            )
            return 1
        return 0

    summary["torch_cuda_available"] = bool(torch.cuda.is_available())
    summary["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    summary["torch_cuda_device_count"] = int(torch.cuda.device_count() or 0)

    torch_version = str(summary["torch_version"] or "")
    torchaudio_version = str(summary["torchaudio_version"] or "")
    cpu_only_wheel = (
        torch_version.endswith("+cpu")
        or torchaudio_version.endswith("+cpu")
        or not summary["torch_cuda_version"]
    )

    print(json.dumps(summary, indent=2))

    if gpu["present"] and (cpu_only_wheel or not summary["torch_cuda_available"]):
        print(
            "NVIDIA GPU detected, but the backend virtualenv has a CPU-only torch stack. "
            "Transcription would run on CPU and be dramatically slower.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
