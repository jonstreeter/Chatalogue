from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def hf_repo_cache_root() -> Path:
    custom_home = str(os.getenv("HF_HOME") or "").strip()
    if custom_home:
        custom_path = Path(custom_home).expanduser()
        return custom_path / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_is_installed(model_id: str) -> tuple[bool, Path | None]:
    normalized = str(model_id or "").strip()
    if "/" not in normalized:
        return False, None
    org, repo = normalized.split("/", 1)
    repo_dir = hf_repo_cache_root() / f"models--{org}--{repo}" / "snapshots"
    if not repo_dir.exists():
        return False, None
    for snapshot in sorted(repo_dir.iterdir(), reverse=True):
        if not snapshot.is_dir():
            continue
        has_config = (snapshot / "config.json").exists()
        has_weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
        if has_config and has_weights:
            return True, snapshot
    return False, None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"Invalid JSONL row at {path}:{line_number}: expected object")
            rows.append(item)
    return rows


def validate_rows(rows: list[dict[str, Any]], *, dataset_name: str) -> dict[str, int]:
    assistant_rows = 0
    total_messages = 0
    total_assistant_words = 0
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{dataset_name} row {index} is missing messages")
        has_assistant = False
        for message in messages:
            if not isinstance(message, dict):
                raise SystemExit(f"{dataset_name} row {index} contains a non-object message")
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            if role not in {"system", "user", "assistant"}:
                raise SystemExit(f"{dataset_name} row {index} has unsupported role: {role or '<empty>'}")
            if not content:
                raise SystemExit(f"{dataset_name} row {index} has an empty {role or 'unknown'} message")
            if role == "assistant":
                has_assistant = True
                total_assistant_words += len(content.split())
        if not has_assistant:
            raise SystemExit(f"{dataset_name} row {index} has no assistant message to supervise")
        assistant_rows += 1
        total_messages += len(messages)
    return {
        "rows": len(rows),
        "assistant_rows": assistant_rows,
        "total_messages": total_messages,
        "assistant_words": total_assistant_words,
    }


def inspect_runtime() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": __import__("sys").executable,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "bf16_supported": False,
        "bitsandbytes_available": False,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                result["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            try:
                props = torch.cuda.get_device_properties(0)
                result["gpu_vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 1)
            except Exception:
                pass
            try:
                result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                pass
    except Exception as exc:
        result["torch_error"] = str(exc)
    try:
        import bitsandbytes  # noqa: F401

        result["bitsandbytes_available"] = True
    except Exception:
        result["bitsandbytes_available"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight-check a prepared avatar personality training package.")
    parser.add_argument("--manifest", required=True, help="Path to training_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("Manifest root must be a JSON object")

    train_path = Path(str(manifest.get("train_dataset_path") or "")).expanduser().resolve()
    val_path = Path(str(manifest.get("validation_dataset_path") or "")).expanduser().resolve()
    missing = [str(path) for path in (train_path, val_path) if not path.exists()]
    if missing:
        raise SystemExit("Training package is incomplete. Missing: " + ", ".join(missing))

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    if not train_rows:
        raise SystemExit("Train dataset is empty")

    train_stats = validate_rows(train_rows, dataset_name="train")
    val_stats = validate_rows(val_rows, dataset_name="validation") if val_rows else {
        "rows": 0,
        "assistant_rows": 0,
        "total_messages": 0,
        "assistant_words": 0,
    }

    base_model_id = str(manifest.get("base_model_id") or "").strip()
    model_installed, model_path = hf_model_is_installed(base_model_id)
    runtime = inspect_runtime()

    print("Avatar personality training package preflight passed.")
    print(f"Manifest: {manifest_path}")
    print(f"Base model: {base_model_id or 'unknown'}")
    print(f"Base model installed locally: {'yes' if model_installed else 'no'}")
    if model_path:
        print(f"Local model snapshot: {model_path}")
    print(f"Export strategy: {manifest.get('export_strategy')}")
    print(f"Conversation examples: {manifest.get('conversation_examples_selected')}")
    print(f"Long-form examples: {manifest.get('long_form_examples_selected')}")
    print(f"Train rows: {train_stats['rows']} | assistant words: {train_stats['assistant_words']}")
    print(f"Validation rows: {val_stats['rows']} | assistant words: {val_stats['assistant_words']}")
    print(f"Python: {runtime.get('python')}")
    if runtime.get("torch_version"):
        print(f"Torch: {runtime.get('torch_version')}")
    if runtime.get("cuda_available"):
        print(
            "CUDA: available"
            f" | GPU: {runtime.get('gpu_name') or 'unknown'}"
            f" | VRAM: {runtime.get('gpu_vram_gb') or '?'} GB"
            f" | bf16: {'yes' if runtime.get('bf16_supported') else 'no'}"
        )
    elif runtime.get("torch_error"):
        print(f"CUDA/Torch check failed: {runtime.get('torch_error')}")
    else:
        print("CUDA: not available")
    print(f"bitsandbytes available: {'yes' if runtime.get('bitsandbytes_available') else 'no'}")
    print("Recommended smoke test: use the web UI with memory-optimized mode, 1 epoch, and backend auto-tuned settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
