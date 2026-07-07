"""Avatar personality service: dataset building, judge passes, long-form
sampling, and training orchestration. Extracted from main.py."""
import hashlib
import html
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import psutil
from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from ..db.database import (
    Avatar,
    AvatarAppearanceProfile,
    AvatarPersonalityProfile,
    AvatarVoiceProfile,
    SpeakerEmbedding,
    Speaker,
    TranscriptSegment,
    Video,
    engine,
)
from ..paths import AVATARS_DIR
from ..schemas import (
    AvatarPersonalityBaseModelCandidateRead,
    AvatarPersonalityBaseModelSupportRead,
    AvatarPersonalityDatasetExampleRead,
    AvatarPersonalityDatasetPageRead,
    AvatarPersonalityDatasetRead,
    AvatarPersonalityJudgeStatusRead,
    AvatarPersonalityLongFormConfigRead,
    AvatarPersonalityLongFormPageRead,
    AvatarPersonalityLongFormSampleRead,
    AvatarPersonalitySnapshotRead,
    AvatarPersonalityTrainRequest,
    AvatarPersonalityTrainingConfigRead,
    AvatarPersonalityTrainingDatasetProfileRead,
    AvatarPersonalityTrainingPackageRead,
    AvatarPersonalityTrainingPlanRead,
    AvatarPersonalityTrainingReadinessRead,
    AvatarPersonalityTrainingStatusRead,
    AvatarRead,
    AvatarSectionSummaryRead,
    AvatarWorkbenchRead,
    AvatarWorkbenchSpeakerRead,
)


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


_avatar_judge_runs_lock = threading.Lock()
_avatar_judge_stop_events: dict[int, threading.Event] = {}
_avatar_judge_threads: dict[int, threading.Thread] = {}
_avatar_training_runs_lock = threading.Lock()
_avatar_training_processes: dict[int, subprocess.Popen] = {}
_avatar_chat_model_lock = threading.Lock()
_avatar_chat_models: dict[int, tuple[str, object, object]] = {}
_avatar_hf_model_downloads_lock = threading.Lock()
_avatar_hf_model_downloads: dict[str, dict[str, object]] = {}


def _avatar_artifacts_dir(avatar: Avatar) -> Path:
    channel_dir = AVATARS_DIR / f"channel_{int(avatar.channel_id)}"
    speaker_dir = channel_dir / f"speaker_{int(avatar.speaker_id)}"
    avatar_dir = speaker_dir / f"avatar_{int(avatar.id)}"
    avatar_dir.mkdir(parents=True, exist_ok=True)
    return avatar_dir


def _avatar_personality_dataset_dir(avatar: Avatar) -> Path:
    path = _avatar_artifacts_dir(avatar) / "personality" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _avatar_personality_dataset_paths(avatar: Avatar) -> tuple[Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "dataset_sharegpt.jsonl",
        dataset_dir / "dataset_preview.json",
        dataset_dir / "dataset_metadata.json",
    )


def _avatar_personality_review_paths(avatar: Avatar) -> tuple[Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "dataset_review.jsonl",
        dataset_dir / "dataset_states.json",
    )


def _avatar_personality_cluster_summary_path(avatar: Avatar) -> Path:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return dataset_dir / "dataset_clusters.json"


def _avatar_personality_judge_status_path(avatar: Avatar) -> Path:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return dataset_dir / "judge_status.json"


def _avatar_personality_long_form_paths(avatar: Avatar) -> tuple[Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "long_form_samples.json",
        dataset_dir / "long_form_states.json",
        dataset_dir / "long_form_config.json",
    )


def _avatar_personality_training_paths(avatar: Avatar) -> tuple[Path, Path, Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "training_config.json",
        dataset_dir / "training_manifest.json",
        dataset_dir / "training_train.jsonl",
        dataset_dir / "training_val.jsonl",
    )


def _avatar_personality_training_runtime_paths(avatar: Avatar) -> tuple[Path, Path]:
    dataset_dir = _avatar_personality_dataset_dir(avatar)
    return (
        dataset_dir / "training_status.json",
        dataset_dir / "training_stop.flag",
    )


def _default_avatar_personality_training_config() -> dict[str, object]:
    return {
        "base_model_id": "Qwen/Qwen3-8B",
        "dataset_profile": "balanced",
        "training_strength": "balanced",
        "export_strategy": "gold_balanced",
        "validation_ratio": 0.10,
        "max_examples": 2500,
        "max_long_form_examples": 80,
        "include_long_form": True,
        "training_mode": "memory_optimized",
        "snapshot_interval_steps": 0,
    }


_AVATAR_TRAINING_DATASET_PROFILES: dict[str, dict[str, object]] = {
    "focused": {
        "label": "Focused",
        "summary": "Smaller, tighter dataset for a fast first pass and lower memorization risk.",
        "conversation_target": 1000,
        "long_form_target": 32,
        "pros": [
            "Fastest prep and shortest training runs",
            "Good for a first personality smoke test",
            "Lower chance of topic memorization",
        ],
        "cons": [
            "Less coverage of the speaker's range",
            "Can feel too generic if the source data is noisy",
        ],
    },
    "balanced": {
        "label": "Balanced",
        "summary": "Recommended default with enough breadth for style and reasoning without overextending the run.",
        "conversation_target": 2500,
        "long_form_target": 80,
        "pros": [
            "Strong first-pass range for most 7B-8B personality LoRAs",
            "Usually lands in a healthy one-epoch step range",
            "Balances conversational variety with manageable runtime",
        ],
        "cons": [
            "Can still miss niche references from large channels",
        ],
        "recommended": True,
    },
    "broad": {
        "label": "Broad",
        "summary": "Wider coverage for speakers with varied topics, references, and argument patterns.",
        "conversation_target": 4000,
        "long_form_target": 120,
        "pros": [
            "Better topical coverage and richer reference patterns",
            "Useful when the speaker has many recurring argument structures",
        ],
        "cons": [
            "Longer runs and more checkpoint review",
            "Needs good curation to avoid repetitive topic drift",
        ],
    },
    "exhaustive": {
        "label": "Exhaustive",
        "summary": "Largest preset. Use only when the dataset is very clean and you want maximum coverage.",
        "conversation_target": 6000,
        "long_form_target": 160,
        "pros": [
            "Captures the broadest range of topics and metaphors",
            "Most useful when the source data has already been heavily filtered",
        ],
        "cons": [
            "Heaviest runtime and review burden",
            "Higher risk of repetition or topic memorization if the dataset is uneven",
        ],
    },
    "custom": {
        "label": "Custom",
        "summary": "Manual caps for cases where you want to tune the package size directly.",
        "conversation_target": 0,
        "long_form_target": 0,
        "pros": [
            "Full control over conversation and long-form caps",
        ],
        "cons": [
            "Easier to overshoot into long, less efficient runs",
        ],
    },
}


def _avatar_training_dataset_profile_options() -> list[AvatarPersonalityTrainingDatasetProfileRead]:
    options: list[AvatarPersonalityTrainingDatasetProfileRead] = []
    for key in ["focused", "balanced", "broad", "exhaustive", "custom"]:
        payload = dict(_AVATAR_TRAINING_DATASET_PROFILES[key])
        options.append(
            AvatarPersonalityTrainingDatasetProfileRead(
                key=key,
                label=str(payload.get("label") or key.title()),
                summary=str(payload.get("summary") or ""),
                conversation_target=int(payload.get("conversation_target") or 0),
                long_form_target=int(payload.get("long_form_target") or 0),
                pros=[str(item) for item in payload.get("pros", []) if str(item).strip()],
                cons=[str(item) for item in payload.get("cons", []) if str(item).strip()],
                recommended=bool(payload.get("recommended")),
            )
        )
    return options


def _normalize_avatar_training_dataset_profile(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _AVATAR_TRAINING_DATASET_PROFILES:
        return normalized
    return "balanced"


def _infer_avatar_training_dataset_profile(max_examples: int | None, max_long_form_examples: int | None) -> str:
    conversation_cap = max(0, int(max_examples or 0))
    long_form_cap = max(0, int(max_long_form_examples or 0))
    for key, payload in _AVATAR_TRAINING_DATASET_PROFILES.items():
        if key == "custom":
            continue
        if (
            conversation_cap == int(payload.get("conversation_target") or 0)
            and long_form_cap == int(payload.get("long_form_target") or 0)
        ):
            return key
    return "custom"


def _resolve_avatar_training_dataset_targets(
    *,
    dataset_profile: str | None,
    max_examples: int | None,
    max_long_form_examples: int | None,
) -> tuple[str, int, int]:
    profile_key = _normalize_avatar_training_dataset_profile(dataset_profile)
    if profile_key != "custom":
        preset = _AVATAR_TRAINING_DATASET_PROFILES[profile_key]
        return (
            profile_key,
            int(preset.get("conversation_target") or 0),
            int(preset.get("long_form_target") or 0),
        )
    return (
        "custom",
        max(0, int(max_examples or 0)),
        max(0, int(max_long_form_examples or 0)),
    )


def _avatar_training_config_storage_payload(payload: dict[str, object]) -> dict[str, object]:
    persisted_keys = {
        "base_model_id",
        "dataset_profile",
        "training_strength",
        "export_strategy",
        "validation_ratio",
        "max_examples",
        "max_long_form_examples",
        "include_long_form",
        "training_mode",
        "snapshot_interval_steps",
    }
    return {key: payload[key] for key in persisted_keys if key in payload}


def _estimate_avatar_validation_example_count(total_examples: int, validation_ratio: float) -> int:
    total = max(0, int(total_examples or 0))
    if total <= 1:
        return 0
    estimate = int(round(total * max(0.01, min(0.2, float(validation_ratio or 0.10)))))
    if total > 20:
        estimate = max(1, estimate)
    estimate = min(total - 1, max(0, estimate))
    return estimate


def _avatar_recommended_snapshot_interval(total_steps: int) -> int:
    steps = max(0, int(total_steps or 0))
    if steps <= 0:
        return 0
    return max(10, math.ceil(steps / 10))


def _hf_repo_cache_root() -> Path:
    custom_home = os.getenv("HF_HOME")
    if custom_home:
        return Path(custom_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(model_id: str) -> Path:
    org, repo = str(model_id or "").strip().split("/", 1)
    return _hf_repo_cache_root() / f"models--{org}--{repo}"


def _hf_model_local_snapshot(model_id: str) -> Path | None:
    repo_dir = _hf_repo_cache_dir(model_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in candidates:
        has_config = (snapshot / "config.json").exists()
        has_weights = any(snapshot.glob("*.safetensors")) or (snapshot / "model.safetensors.index.json").exists() or any(snapshot.glob("pytorch_model*.bin"))
        if has_config and has_weights:
            return snapshot
    return None


def _hf_model_is_installed(model_id: str) -> tuple[bool, str | None]:
    try:
        snapshot = _hf_model_local_snapshot(model_id)
    except Exception:
        snapshot = None
    return (snapshot is not None, str(snapshot) if snapshot else None)


def _recommended_avatar_training_models() -> list[dict[str, str]]:
    return [
        {"model_id": "Qwen/Qwen3-8B", "label": "Qwen3 8B"},
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B Instruct"},
        {"model_id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B Instruct"},
        {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct"},
        {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct"},
    ]


@lru_cache(maxsize=1)
def _detect_avatar_memory_optimized_support() -> tuple[bool, str | None]:
    if importlib.util.find_spec("bitsandbytes") is None:
        return False, "bitsandbytes is not installed in the backend training environment."
    try:
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            from transformers.utils.quantization_config import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True)
        return True, None
    except Exception as exc:
        return False, f"4-bit QLoRA support is unavailable: {exc}"


def _recommend_avatar_training_model_for_hardware() -> tuple[dict[str, object], list[dict[str, str]]]:
    hardware = _main()._detect_gpu_hardware()
    gpu_vram_gb = hardware.get("gpu_vram_gb")
    candidates = _recommended_avatar_training_models()
    recommended = "Qwen/Qwen2.5-3B-Instruct"
    rationale = "GPU VRAM could not be detected. Defaulting to a conservative training base."
    if gpu_vram_gb is not None:
        vram = float(gpu_vram_gb)
        if vram >= 28:
            recommended = "Qwen/Qwen3-8B"
            rationale = f"Detected ~{vram:.1f} GB VRAM. Qwen3-8B is the strongest practical target for the current LoRA trainer."
        elif vram >= 18:
            recommended = "Qwen/Qwen2.5-7B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 7B is the safer high-quality target on this hardware."
        elif vram >= 10:
            recommended = "Qwen/Qwen2.5-3B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 3B is the recommended fit for stable local LoRA training."
        elif vram >= 6:
            recommended = "Qwen/Qwen2.5-1.5B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. 1.5B is the largest practical fit on this hardware."
        else:
            recommended = "Qwen/Qwen2.5-0.5B-Instruct"
            rationale = f"Detected ~{vram:.1f} GB VRAM. Use a very small base for local experimentation."
    return {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_vram_gb": gpu_vram_gb,
        "recommended_model_id": recommended,
        "rationale": rationale,
    }, candidates


def _infer_avatar_model_scale_b(model_id: str | None) -> float | None:
    normalized = str(model_id or "").strip()
    if not normalized:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", normalized)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _recommend_avatar_training_launch_settings(
    *,
    model_id: str | None,
    training_mode: str,
    requested_lora_rank: int | None,
    requested_max_seq_length: int | None,
    requested_per_device_batch_size: int | None,
    requested_gradient_accumulation_steps: int | None,
) -> dict[str, object]:
    hardware = _main()._detect_gpu_hardware()
    gpu_vram_gb_raw = hardware.get("gpu_vram_gb")
    gpu_vram_gb = float(gpu_vram_gb_raw) if gpu_vram_gb_raw is not None else None
    model_scale_b = _infer_avatar_model_scale_b(model_id)
    mode = str(training_mode or "memory_optimized").strip().lower()

    # Start with a conservative 12 GB profile so unknown hardware does not overrun
    # VRAM and spill into shared/system memory.
    recommended = {
        "lora_rank": 8,
        "max_seq_length": 768,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "cuda_memory_fraction": 0.72,
        "rationale": "Using a conservative 12 GB baseline because GPU VRAM could not be detected.",
    }

    if gpu_vram_gb is not None:
        if gpu_vram_gb >= 28:
            recommended.update(
                lora_rank=16,
                max_seq_length=1536,
                gradient_accumulation_steps=8,
                cuda_memory_fraction=0.82,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a high-end local LoRA profile with reserved CUDA headroom.",
            )
        elif gpu_vram_gb >= 18:
            recommended.update(
                lora_rank=8,
                max_seq_length=1024,
                gradient_accumulation_steps=12,
                cuda_memory_fraction=0.78,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a balanced profile that keeps headroom for activations and optimizer state.",
            )
        elif gpu_vram_gb >= 12:
            recommended.update(
                lora_rank=8,
                max_seq_length=768,
                gradient_accumulation_steps=16,
                cuda_memory_fraction=0.72,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a 12 GB-safe profile to avoid shared-memory spillover.",
            )
        elif gpu_vram_gb >= 8:
            recommended.update(
                lora_rank=4,
                max_seq_length=512,
                gradient_accumulation_steps=24,
                cuda_memory_fraction=0.68,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using an aggressive low-VRAM profile.",
            )
        else:
            recommended.update(
                lora_rank=4,
                max_seq_length=384,
                gradient_accumulation_steps=32,
                cuda_memory_fraction=0.62,
                rationale=f"Detected ~{gpu_vram_gb:.1f} GB VRAM. Using a minimal profile for experimentation only.",
            )

    if mode == "standard":
        recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
        recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 24 else 768)
        recommended["cuda_memory_fraction"] = min(float(recommended["cuda_memory_fraction"]), 0.70 if (gpu_vram_gb or 0) >= 24 else 0.62)
        recommended["gradient_accumulation_steps"] = max(int(recommended["gradient_accumulation_steps"]), 16)
        recommended["rationale"] = (
            f"{recommended['rationale']} Standard mode keeps extra VRAM headroom because full-precision optimizer state is larger."
        )

    if model_scale_b is not None:
        if model_scale_b >= 14:
            recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 512 if (gpu_vram_gb or 0) < 40 else 768)
            recommended["gradient_accumulation_steps"] = max(int(recommended["gradient_accumulation_steps"]), 16)
        elif model_scale_b >= 8:
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 32 else 768)
            if (gpu_vram_gb or 0) < 24:
                recommended["lora_rank"] = min(int(recommended["lora_rank"]), 8)
        elif model_scale_b >= 7:
            recommended["max_seq_length"] = min(int(recommended["max_seq_length"]), 1024 if (gpu_vram_gb or 0) >= 24 else 768)
        elif model_scale_b <= 3:
            if (gpu_vram_gb or 12) >= 20:
                recommended["lora_rank"] = max(int(recommended["lora_rank"]), 16)
                recommended["max_seq_length"] = max(int(recommended["max_seq_length"]), 1536)
                recommended["gradient_accumulation_steps"] = min(int(recommended["gradient_accumulation_steps"]), 8)
        elif model_scale_b <= 1.5:
            if (gpu_vram_gb or 12) >= 12:
                recommended["lora_rank"] = max(int(recommended["lora_rank"]), 16)
                recommended["max_seq_length"] = max(int(recommended["max_seq_length"]), 1024)
                recommended["gradient_accumulation_steps"] = min(int(recommended["gradient_accumulation_steps"]), 12)

    effective_lora_rank = min(
        max(4, int(requested_lora_rank or int(recommended["lora_rank"]))),
        max(4, int(recommended["lora_rank"])),
    )
    effective_max_seq_length = min(
        max(256, int(requested_max_seq_length or int(recommended["max_seq_length"]))),
        max(256, int(recommended["max_seq_length"])),
    )
    effective_per_device_batch_size = min(
        max(1, int(requested_per_device_batch_size or int(recommended["per_device_batch_size"]))),
        max(1, int(recommended["per_device_batch_size"])),
    )
    effective_gradient_accumulation_steps = max(
        max(1, int(requested_gradient_accumulation_steps or int(recommended["gradient_accumulation_steps"]))),
        max(1, int(recommended["gradient_accumulation_steps"])),
    )

    return {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_vram_gb": gpu_vram_gb,
        "model_scale_b": model_scale_b,
        "lora_rank": effective_lora_rank,
        "max_seq_length": effective_max_seq_length,
        "per_device_batch_size": effective_per_device_batch_size,
        "gradient_accumulation_steps": effective_gradient_accumulation_steps,
        "cuda_memory_fraction": float(recommended["cuda_memory_fraction"]),
        "rationale": str(recommended["rationale"]),
    }


def _estimate_avatar_available_conversation_examples(
    dataset: AvatarPersonalityDatasetRead,
    config: AvatarPersonalityTrainingConfigRead,
) -> int:
    approved_count = max(0, int(dataset.gold_example_count or 0) + int(dataset.silver_example_count or 0))
    gold_count = int(dataset.gold_example_count or 0)
    silver_count = int(dataset.silver_example_count or 0)
    strategy = str(config.export_strategy or "gold_balanced")
    if strategy == "gold_only":
        return max(0, gold_count)
    if strategy == "gold_plus_top_silver":
        silver_budget = min(max(0, silver_count), max(250, gold_count // 2))
        return max(0, gold_count + silver_budget)
    return max(0, approved_count)


def _build_avatar_personality_training_plan(
    *,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    config: AvatarPersonalityTrainingConfigRead,
    available_conversation_examples: int | None = None,
    available_long_form_examples: int | None = None,
    selected_conversation_examples: int | None = None,
    selected_long_form_examples: int | None = None,
    train_examples: int | None = None,
    validation_examples: int | None = None,
    epochs: int = 1,
) -> AvatarPersonalityTrainingPlanRead:
    dataset = _load_avatar_personality_dataset(avatar, personality)
    long_form_config = _load_avatar_personality_long_form_config(avatar)
    profile_key, conversation_target, long_form_target = _resolve_avatar_training_dataset_targets(
        dataset_profile=config.dataset_profile,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
    )
    profile_meta = _AVATAR_TRAINING_DATASET_PROFILES.get(profile_key, _AVATAR_TRAINING_DATASET_PROFILES["balanced"])
    available_conversation = max(
        0,
        int(
            available_conversation_examples
            if available_conversation_examples is not None
            else _estimate_avatar_available_conversation_examples(dataset, config)
        ),
    )
    available_long_form = max(
        0,
        int(
            available_long_form_examples
            if available_long_form_examples is not None
            else min(int(long_form_config.selected_count or 0), int(long_form_config.take_count or 0))
        ),
    )

    effective_conversation = max(
        0,
        int(
            selected_conversation_examples
            if selected_conversation_examples is not None
            else min(available_conversation, conversation_target if conversation_target > 0 else available_conversation)
        ),
    )
    effective_long_form = 0
    if bool(config.include_long_form):
        effective_long_form = max(
            0,
            int(
                selected_long_form_examples
                if selected_long_form_examples is not None
                else min(available_long_form, long_form_target if long_form_target > 0 else available_long_form)
            ),
        )

    estimated_total_examples = max(0, effective_conversation + effective_long_form)
    resolved_validation_examples = (
        max(0, int(validation_examples or 0))
        if validation_examples is not None
        else _estimate_avatar_validation_example_count(estimated_total_examples, config.validation_ratio)
    )
    resolved_train_examples = (
        max(0, int(train_examples or 0))
        if train_examples is not None
        else max(0, estimated_total_examples - resolved_validation_examples)
    )

    launch_settings = _recommend_avatar_training_launch_settings(
        model_id=str(config.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B"),
        training_mode=str(config.training_mode or "memory_optimized"),
        requested_lora_rank=None,
        requested_max_seq_length=None,
        requested_per_device_batch_size=None,
        requested_gradient_accumulation_steps=None,
    )
    effective_batch_size = max(
        1,
        int(launch_settings.get("per_device_batch_size") or 1)
        * int(launch_settings.get("gradient_accumulation_steps") or 1),
    )
    estimated_steps_per_epoch = math.ceil(resolved_train_examples / effective_batch_size) if resolved_train_examples > 0 else 0
    estimated_total_steps = estimated_steps_per_epoch * max(1, int(epochs or 1))

    if estimated_total_steps < 100:
        step_band = "light"
        headline = "Small, fast package with a lighter style imprint."
        recommendation = "Good for a smoke test, but it may underfit nuance, references, and argument structure."
    elif estimated_total_steps <= 500:
        step_band = "ideal"
        headline = "Healthy first-pass training range for personality LoRA."
        recommendation = "This is the best default zone for one-epoch training and snapshot comparison."
    elif estimated_total_steps <= 800:
        step_band = "heavy"
        headline = "Broader package with longer runs and more review overhead."
        recommendation = "Useful for diverse speakers, but watch for repeated stock phrases and over-anchoring to popular topics."
    else:
        step_band = "aggressive"
        headline = "Large package that pushes beyond the usual first-pass sweet spot."
        recommendation = "Only use this if the dataset is very clean and diverse. Prefer snapshot comparison and stop early if responses get repetitive."

    return AvatarPersonalityTrainingPlanRead(
        dataset_profile=profile_key,
        dataset_profile_label=str(profile_meta.get("label") or profile_key.title()),
        conversation_target=conversation_target,
        long_form_target=long_form_target if bool(config.include_long_form) else 0,
        available_conversation_examples=available_conversation,
        available_long_form_examples=available_long_form if bool(config.include_long_form) else 0,
        estimated_conversation_examples=effective_conversation,
        estimated_long_form_examples=effective_long_form,
        estimated_total_examples=estimated_total_examples,
        estimated_train_examples=resolved_train_examples,
        estimated_validation_examples=resolved_validation_examples,
        estimated_effective_batch_size=effective_batch_size,
        estimated_steps_per_epoch=estimated_steps_per_epoch,
        estimated_total_steps=estimated_total_steps,
        step_band=step_band,
        headline=headline,
        recommendation=recommendation,
        snapshot_interval_suggestion=_avatar_recommended_snapshot_interval(estimated_total_steps),
        pros=[str(item) for item in profile_meta.get("pros", []) if str(item).strip()],
        cons=[str(item) for item in profile_meta.get("cons", []) if str(item).strip()],
    )


def _get_avatar_hf_model_download_state(model_id: str) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    if not normalized:
        return {"status": "idle", "running": False, "message": None}
    with _avatar_hf_model_downloads_lock:
        return dict(_avatar_hf_model_downloads.get(normalized) or {})


def _set_avatar_hf_model_download_state(model_id: str, patch: dict[str, object]) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    with _avatar_hf_model_downloads_lock:
        current = dict(_avatar_hf_model_downloads.get(normalized) or {"status": "idle", "running": False, "message": None})
        current.update(patch)
        _avatar_hf_model_downloads[normalized] = current
        return dict(current)


def _start_avatar_hf_model_download(model_id: str) -> dict[str, object]:
    normalized = str(model_id or "").strip()
    if "/" not in normalized:
        raise HTTPException(status_code=400, detail="Model id must look like org/repo")
    installed, local_path = _hf_model_is_installed(normalized)
    if installed:
        return _set_avatar_hf_model_download_state(
            normalized,
            {"status": "completed", "running": False, "message": "Model already installed locally", "local_path": local_path},
        )
    current = _get_avatar_hf_model_download_state(normalized)
    if current.get("running"):
        return current

    _set_avatar_hf_model_download_state(
        normalized,
        {"status": "running", "running": True, "message": "Starting Hugging Face model download", "local_path": None},
    )

    def _runner() -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=normalized,
                resume_download=True,
                local_files_only=False,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.tiktoken",
                    "tokenizer*",
                    "merges.txt",
                    "vocab*",
                    "*.txt",
                ],
                ignore_patterns=[
                    "*.gguf",
                    "*.onnx",
                    "*.h5",
                    "*.ot",
                    "*.msgpack",
                ],
                token=(os.getenv("HF_TOKEN") or None),
            )
            _set_avatar_hf_model_download_state(
                normalized,
                {"status": "completed", "running": False, "message": "Model download completed", "local_path": str(snapshot_path)},
            )
        except Exception as exc:
            _set_avatar_hf_model_download_state(
                normalized,
                {"status": "failed", "running": False, "message": f"Download failed: {exc}"},
            )

    threading.Thread(target=_runner, daemon=True, name=f"hf-model-download-{normalized.replace('/', '--')}").start()
    return _get_avatar_hf_model_download_state(normalized)


def _read_avatar_base_model_support(selected_model_id: str) -> AvatarPersonalityBaseModelSupportRead:
    selected = str(selected_model_id or "").strip() or "Qwen/Qwen3-8B"
    recommendation, candidates = _recommend_avatar_training_model_for_hardware()
    installed, local_path = _hf_model_is_installed(selected)
    memory_optimized_available, memory_optimized_reason = _detect_avatar_memory_optimized_support()
    download_state = _get_avatar_hf_model_download_state(selected)
    items: list[AvatarPersonalityBaseModelCandidateRead] = []
    for candidate in candidates:
        candidate_installed, _candidate_path = _hf_model_is_installed(candidate["model_id"])
        items.append(
            AvatarPersonalityBaseModelCandidateRead(
                model_id=candidate["model_id"],
                label=candidate["label"],
                recommended=(candidate["model_id"] == recommendation["recommended_model_id"]),
                installed=candidate_installed,
            )
        )
    return AvatarPersonalityBaseModelSupportRead(
        selected_model_id=selected,
        recommended_model_id=str(recommendation["recommended_model_id"]),
        installed=installed,
        local_path=local_path,
        memory_optimized_available=memory_optimized_available,
        memory_optimized_reason=memory_optimized_reason,
        downloading=bool(download_state.get("running")),
        download_status=str(download_state.get("status") or "idle"),
        download_message=(str(download_state.get("message")) if download_state.get("message") else None),
        gpu_name=(str(recommendation.get("gpu_name")) if recommendation.get("gpu_name") else None),
        gpu_vram_gb=(float(recommendation["gpu_vram_gb"]) if recommendation.get("gpu_vram_gb") is not None else None),
        rationale=(str(recommendation.get("rationale")) if recommendation.get("rationale") else None),
        candidates=items,
    )


def _default_avatar_personality_judge_status(avatar_id: int) -> dict[str, object]:
    now = datetime.now().isoformat()
    return {
        "avatar_id": int(avatar_id),
        "status": "idle",
        "active": False,
        "stop_requested": False,
        "model": None,
        "target_filter": "needs_review",
        "overwrite_existing": False,
        "max_examples": 40,
        "total_candidates": 0,
        "processed_count": 0,
        "judged_count": 0,
        "promoted_count": 0,
        "rejected_count": 0,
        "current_example_id": None,
        "current_video_title": None,
        "current_stage": None,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "error": None,
        "recent_results": [],
    }


def _load_avatar_personality_judge_status(avatar: Avatar) -> AvatarPersonalityJudgeStatusRead:
    status_path = _avatar_personality_judge_status_path(avatar)
    payload = _default_avatar_personality_judge_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityJudgeStatusRead(**payload)


def _write_avatar_personality_judge_status(avatar: Avatar, payload: dict[str, object]) -> AvatarPersonalityJudgeStatusRead:
    status_path = _avatar_personality_judge_status_path(avatar)
    existing = _default_avatar_personality_judge_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing.update(raw)
        except Exception:
            pass
    existing.update(payload)
    existing["avatar_id"] = int(avatar.id)
    existing["updated_at"] = datetime.now().isoformat()
    normalized = AvatarPersonalityJudgeStatusRead(**existing)
    status_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _default_avatar_personality_prompt(name: str) -> str:
    clean_name = _clean_avatar_dataset_text(name) or "the podcast speaker"
    return f"You are {clean_name}. Respond in their conversational style based on the approved transcript dataset."


def _clean_avatar_dataset_text(value: str | None) -> str:
    text_value = html.unescape(str(value or ""))
    text_value = re.sub(r"\s+", " ", text_value).strip()
    text_value = text_value.strip("\"' ")
    return text_value


def _avatar_word_count(text: str | None) -> int:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return 0
    return len(cleaned.split())


def _avatar_long_form_sample_id(video_id: int, start_time: float, end_time: float, segment_ids: list[int]) -> str:
    raw = f"{int(video_id)}|{round(float(start_time or 0.0), 3)}|{round(float(end_time or 0.0), 3)}|{','.join(str(int(segment_id)) for segment_id in segment_ids)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _load_avatar_personality_long_form_states(avatar: Avatar) -> dict[str, str]:
    _, states_path, _ = _avatar_personality_long_form_paths(avatar)
    if not states_path.exists():
        return {}
    try:
        raw = json.loads(states_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        state = str(value or "").strip().lower()
        if state in {"included", "rejected"}:
            out[str(key)] = state
    return out


def _write_avatar_personality_long_form_states(avatar: Avatar, state_map: dict[str, str]) -> Path:
    _, states_path, _ = _avatar_personality_long_form_paths(avatar)
    serializable = {str(key): str(value) for key, value in sorted(state_map.items()) if value in {"included", "rejected"}}
    states_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return states_path


def _load_avatar_personality_long_form_config(avatar: Avatar) -> AvatarPersonalityLongFormConfigRead:
    _, _, config_path = _avatar_personality_long_form_paths(avatar)
    payload: dict[str, object] = {"take_count": 150, "included_count": 0, "rejected_count": 0, "selected_count": 0}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityLongFormConfigRead(**payload)


def _write_avatar_personality_long_form_config(
    avatar: Avatar,
    *,
    take_count: int,
    included_count: int | None = None,
    rejected_count: int | None = None,
    selected_count: int | None = None,
) -> AvatarPersonalityLongFormConfigRead:
    _, _, config_path = _avatar_personality_long_form_paths(avatar)
    existing = _load_avatar_personality_long_form_config(avatar).model_dump()
    existing["take_count"] = max(0, int(take_count or 0))
    if included_count is not None:
        existing["included_count"] = max(0, int(included_count))
    if rejected_count is not None:
        existing["rejected_count"] = max(0, int(rejected_count))
    if selected_count is not None:
        existing["selected_count"] = max(0, int(selected_count))
    normalized = AvatarPersonalityLongFormConfigRead(**existing)
    config_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _load_avatar_personality_training_config(avatar: Avatar) -> AvatarPersonalityTrainingConfigRead:
    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    payload = _default_avatar_personality_training_config()
    raw: dict[str, object] = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    if not isinstance(raw, dict):
        raw = {}
    if "dataset_profile" in raw:
        profile_key, resolved_max_examples, resolved_max_long_form_examples = _resolve_avatar_training_dataset_targets(
            dataset_profile=raw.get("dataset_profile"),
            max_examples=payload.get("max_examples"),
            max_long_form_examples=payload.get("max_long_form_examples"),
        )
        payload["dataset_profile"] = profile_key
        if profile_key != "custom":
            payload["max_examples"] = resolved_max_examples
            payload["max_long_form_examples"] = resolved_max_long_form_examples
    else:
        payload["dataset_profile"] = _infer_avatar_training_dataset_profile(
            payload.get("max_examples"),
            payload.get("max_long_form_examples"),
        )
        payload["max_long_form_examples"] = max(0, int(payload.get("max_long_form_examples") or _default_avatar_personality_training_config()["max_long_form_examples"]))
    return AvatarPersonalityTrainingConfigRead(**payload)


def _read_avatar_personality_training_config(
    session: Session,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityTrainingConfigRead:
    config = _load_avatar_personality_training_config(avatar)
    payload = config.model_dump()
    payload["dataset_profiles"] = [option.model_dump() for option in _avatar_training_dataset_profile_options()]
    payload["training_plan"] = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        epochs=1,
    ).model_dump()
    return AvatarPersonalityTrainingConfigRead(**payload)


def _write_avatar_personality_training_config(
    avatar: Avatar,
    *,
    base_model_id: str | None = None,
    dataset_profile: str | None = None,
    training_strength: str | None = None,
    export_strategy: str | None = None,
    validation_ratio: float | None = None,
    max_examples: int | None = None,
    max_long_form_examples: int | None = None,
    include_long_form: bool | None = None,
    training_mode: str | None = None,
    snapshot_interval_steps: int | None = None,
) -> AvatarPersonalityTrainingConfigRead:
    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    existing = _avatar_training_config_storage_payload(_load_avatar_personality_training_config(avatar).model_dump())
    if base_model_id is not None:
        existing["base_model_id"] = str(base_model_id).strip() or existing.get("base_model_id") or "Qwen/Qwen3-8B"
    if dataset_profile is not None:
        existing["dataset_profile"] = _normalize_avatar_training_dataset_profile(dataset_profile)
    if training_strength is not None:
        normalized_strength = str(training_strength or "").strip().lower()
        if normalized_strength not in {"conservative", "balanced", "strong"}:
            normalized_strength = "balanced"
        existing["training_strength"] = normalized_strength
    if export_strategy is not None:
        existing["export_strategy"] = str(export_strategy)
    if validation_ratio is not None:
        existing["validation_ratio"] = max(0.01, min(0.2, float(validation_ratio)))
    if max_examples is not None:
        existing["max_examples"] = max(0, int(max_examples))
        if dataset_profile is None and str(existing.get("dataset_profile") or "").strip().lower() != "custom":
            existing["dataset_profile"] = "custom"
    if max_long_form_examples is not None:
        existing["max_long_form_examples"] = max(0, int(max_long_form_examples))
        if dataset_profile is None and str(existing.get("dataset_profile") or "").strip().lower() != "custom":
            existing["dataset_profile"] = "custom"
    if include_long_form is not None:
        existing["include_long_form"] = bool(include_long_form)
    if training_mode is not None:
        existing["training_mode"] = str(training_mode or "memory_optimized").strip() or "memory_optimized"
    if snapshot_interval_steps is not None:
        existing["snapshot_interval_steps"] = max(0, int(snapshot_interval_steps))
    profile_key, resolved_max_examples, resolved_max_long_form_examples = _resolve_avatar_training_dataset_targets(
        dataset_profile=str(existing.get("dataset_profile") or "balanced"),
        max_examples=existing.get("max_examples"),
        max_long_form_examples=existing.get("max_long_form_examples"),
    )
    existing["dataset_profile"] = profile_key
    existing["max_examples"] = resolved_max_examples
    existing["max_long_form_examples"] = resolved_max_long_form_examples
    normalized = AvatarPersonalityTrainingConfigRead(**existing)
    config_path.write_text(
        json.dumps(_avatar_training_config_storage_payload(normalized.model_dump()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def _read_avatar_personality_training_package(avatar: Avatar) -> AvatarPersonalityTrainingPackageRead:
    config_path, manifest_path, train_path, val_path = _avatar_personality_training_paths(avatar)
    if not manifest_path.exists():
        config = _load_avatar_personality_training_config(avatar)
        return AvatarPersonalityTrainingPackageRead(
            avatar_id=int(avatar.id),
            status="not_prepared",
            dataset_profile=config.dataset_profile,
            training_strength=config.training_strength,
            export_strategy=config.export_strategy,
            validation_ratio=config.validation_ratio,
            max_examples=config.max_examples,
            max_long_form_examples=config.max_long_form_examples,
            include_long_form=config.include_long_form,
            config_path=str(config_path) if config_path.exists() else None,
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("avatar_id", int(avatar.id))
    raw.setdefault("status", "ready")
    raw.setdefault("dataset_profile", "balanced")
    raw.setdefault("training_strength", "balanced")
    raw.setdefault("max_long_form_examples", 0)
    if not raw.get("config_path") and config_path.exists():
        raw["config_path"] = str(config_path)
    if not raw.get("train_dataset_path") and train_path.exists():
        raw["train_dataset_path"] = str(train_path)
    if not raw.get("validation_dataset_path") and val_path.exists():
        raw["validation_dataset_path"] = str(val_path)
    raw["manifest_path"] = str(manifest_path)
    return AvatarPersonalityTrainingPackageRead(**raw)


def _default_avatar_personality_training_status(avatar_id: int) -> dict[str, object]:
    now = datetime.now().isoformat()
    return {
        "avatar_id": int(avatar_id),
        "status": "idle",
        "active": False,
        "stop_requested": False,
        "process_id": None,
        "base_model_id": None,
        "training_mode": "memory_optimized",
        "adapter_path": None,
        "output_dir": None,
        "current_stage": None,
        "epoch": 0.0,
        "step": 0,
        "max_steps": 0,
        "snapshot_interval_steps": 0,
        "train_examples": 0,
        "validation_examples": 0,
        "latest_loss": None,
        "message": None,
        "snapshots": [],
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "error": None,
    }


def _load_avatar_personality_training_status(avatar: Avatar) -> AvatarPersonalityTrainingStatusRead:
    status_path, _ = _avatar_personality_training_runtime_paths(avatar)
    payload = _default_avatar_personality_training_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    return AvatarPersonalityTrainingStatusRead(**payload)


def _write_avatar_personality_training_status(avatar: Avatar, patch: dict[str, object]) -> AvatarPersonalityTrainingStatusRead:
    status_path, _ = _avatar_personality_training_runtime_paths(avatar)
    payload = _default_avatar_personality_training_status(int(avatar.id))
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    payload.update(patch)
    normalized = AvatarPersonalityTrainingStatusRead(**payload)
    status_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    return normalized


def _avatar_training_output_dir(avatar: Avatar) -> Path:
    return _avatar_artifacts_dir(avatar) / "personality" / "training_runs" / "latest"


def _normalize_avatar_training_snapshots(
    snapshots: list[AvatarPersonalitySnapshotRead] | list[dict[str, object]] | None,
    *,
    selected_adapter_path: str | None = None,
) -> list[AvatarPersonalitySnapshotRead]:
    normalized: list[AvatarPersonalitySnapshotRead] = []
    selected = str(selected_adapter_path or "").strip()
    seen_paths: set[str] = set()
    for raw in snapshots or []:
        try:
            item = raw if isinstance(raw, AvatarPersonalitySnapshotRead) else AvatarPersonalitySnapshotRead(**raw)
        except Exception:
            continue
        adapter_path = str(item.adapter_path or "").strip()
        if not adapter_path or adapter_path in seen_paths:
            continue
        seen_paths.add(adapter_path)
        normalized.append(
            item.model_copy(
                update={
                    "selected": bool(selected and adapter_path == selected) or bool(item.selected and not selected),
                }
            )
        )
    normalized.sort(key=lambda item: ((item.created_at.isoformat() if item.created_at else ""), item.step, item.epoch))
    return normalized


def _avatar_resolve_training_snapshots(
    avatar: Avatar,
    *,
    selected_adapter_path: str | None = None,
) -> list[AvatarPersonalitySnapshotRead]:
    status = _load_avatar_personality_training_status(avatar)
    snapshots = _normalize_avatar_training_snapshots(
        list(status.snapshots or []),
        selected_adapter_path=selected_adapter_path or status.adapter_path,
    )
    if snapshots:
        return snapshots
    adapter_path = str(selected_adapter_path or status.adapter_path or "").strip()
    if adapter_path and Path(adapter_path).exists():
        return [
            AvatarPersonalitySnapshotRead(
                label="Final Adapter",
                kind="final",
                adapter_path=adapter_path,
                selected=True,
            )
        ]
    return []


def _avatar_update_training_snapshots(
    avatar: Avatar,
    snapshots: list[AvatarPersonalitySnapshotRead] | list[dict[str, object]],
    *,
    selected_adapter_path: str | None = None,
) -> AvatarPersonalityTrainingStatusRead:
    normalized = _normalize_avatar_training_snapshots(snapshots, selected_adapter_path=selected_adapter_path)
    return _write_avatar_personality_training_status(
        avatar,
        {
            "snapshots": [item.model_dump(mode="json") for item in normalized],
            "adapter_path": str(selected_adapter_path or "").strip() or None,
        },
    )


def _avatar_find_training_snapshot(
    avatar: Avatar,
    adapter_path: str,
) -> AvatarPersonalitySnapshotRead | None:
    selected = str(adapter_path or "").strip()
    if not selected:
        return None
    for snapshot in _avatar_resolve_training_snapshots(avatar, selected_adapter_path=selected):
        if str(snapshot.adapter_path).strip() == selected:
            return snapshot
    return None


def _avatar_training_process_is_alive(pid: int | None) -> bool:
    try:
        process_id = int(pid or 0)
    except Exception:
        return False
    if process_id <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output = str(result.stdout or "").strip()
            return bool(output) and "No tasks are running" not in output
        except Exception:
            return False
    try:
        os.kill(process_id, 0)
        return True
    except Exception:
        return False


def _avatar_training_gpu_memory_by_pid_gb() -> dict[int, float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return {}
        out: dict[int, float] = {}
        for line in (result.stdout or "").splitlines():
            raw = str(line or "").strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
            except Exception:
                continue
            out[pid] = round(used_mb / 1024.0, 2)
        return out
    except Exception:
        return {}


def _collect_avatar_training_process_memory() -> dict[str, object]:
    gpu_by_pid = _avatar_training_gpu_memory_by_pid_gb()
    total_rss_gb = 0.0
    total_vram_gb = 0.0
    active_count = 0
    seen_pids: set[int] = set()

    for status_path in AVATARS_DIR.rglob("training_status.json"):
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        if not bool(raw.get("active")):
            continue
        try:
            pid = int(raw.get("process_id") or 0)
        except Exception:
            pid = 0
        if pid <= 0 or pid in seen_pids or not _avatar_training_process_is_alive(pid):
            continue
        seen_pids.add(pid)
        try:
            process = psutil.Process(pid)
            rss_gb = round(float(process.memory_info().rss or 0) / (1024 ** 3), 2)
        except Exception:
            rss_gb = 0.0
        total_rss_gb += max(0.0, rss_gb)
        total_vram_gb += max(0.0, float(gpu_by_pid.get(pid) or 0.0))
        active_count += 1

    return {
        "active_count": active_count,
        "ram_gb": round(total_rss_gb, 2),
        "vram_gb": round(total_vram_gb, 2),
        "loaded": active_count > 0,
    }


def _reconcile_avatar_personality_training_runtime(
    avatar: Avatar,
    *,
    persist: bool = True,
) -> AvatarPersonalityTrainingStatusRead:
    status = _load_avatar_personality_training_status(avatar)
    normalized_status = str(status.status or "").strip().lower()
    if normalized_status in {"idle", "completed", "stopped", "failed"} or not status.active:
        return status

    process_alive = False
    with _avatar_training_runs_lock:
        tracked = _avatar_training_processes.get(int(avatar.id))
        if tracked is not None:
            if tracked.poll() is None:
                process_alive = True
            else:
                _avatar_training_processes.pop(int(avatar.id), None)
    if not process_alive:
        process_alive = _avatar_training_process_is_alive(status.process_id)
    if process_alive:
        return status

    adapter_exists = bool(status.adapter_path and Path(str(status.adapter_path)).exists())
    if normalized_status == "stopping" or bool(status.stop_requested):
        final_status = "stopped"
        final_message = "Recovered stale training state after the trainer process exited."
        final_error = None
    elif adapter_exists:
        final_status = "completed"
        final_message = "Recovered completed training state after the trainer process exited."
        final_error = None
    else:
        final_status = "failed"
        final_message = "Recovered stale training state after the trainer process disappeared."
        final_error = status.error or "Trainer process is no longer running."

    _, stop_path = _avatar_personality_training_runtime_paths(avatar)
    try:
        stop_path.unlink(missing_ok=True)
    except Exception:
        pass

    patch = {
        "status": final_status,
        "active": False,
        "stop_requested": False,
        "process_id": None,
        "current_stage": final_status,
        "finished_at": status.finished_at or datetime.now(),
        "updated_at": datetime.now(),
        "message": final_message,
        "error": final_error,
    }
    if persist:
        return _write_avatar_personality_training_status(avatar, patch)
    payload = status.model_dump()
    payload.update(patch)
    return AvatarPersonalityTrainingStatusRead(**payload)


def _clear_avatar_personality_training_stop_flag(avatar: Avatar) -> None:
    _, stop_path = _avatar_personality_training_runtime_paths(avatar)
    try:
        stop_path.unlink(missing_ok=True)
    except Exception:
        pass


def _sync_avatar_personality_training_completion(avatar_id: int) -> None:
    with Session(engine) as session:
        avatar = session.get(Avatar, avatar_id)
        if not avatar:
            return
        speaker = session.get(Speaker, avatar.speaker_id)
        if not speaker:
            return
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        status = _load_avatar_personality_training_status(avatar)
        normalized_status = str(status.status or "").strip().lower()
        if normalized_status == "completed" and status.adapter_path:
            personality.lora_adapter_path = status.adapter_path
            personality.status = "trained"
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()
        elif normalized_status in {"failed", "stopped"}:
            personality.updated_at = datetime.now()
            session.add(personality)
            session.commit()


def _avatar_release_cached_chat_model(avatar_id: int) -> None:
    with _avatar_chat_model_lock:
        cached = _avatar_chat_models.pop(int(avatar_id), None)
    if not cached:
        return
    try:
        _cache_key, model, tokenizer = cached
        del model
        del tokenizer
    except Exception:
        pass
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _import_avatar_transformers_chat_classes():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception:
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer


def _avatar_load_chat_model(avatar_id: int, base_model_id: str, adapter_path: str, *, training_mode: str = "memory_optimized"):
    cache_key = f"{str(adapter_path).strip()}|{str(training_mode or 'memory_optimized').strip().lower()}"
    with _avatar_chat_model_lock:
        cached = _avatar_chat_models.get(int(avatar_id))
        if cached and cached[0] == cache_key:
            return cached[1], cached[2]
    _avatar_release_cached_chat_model(int(avatar_id))
    import torch
    from peft import PeftModel
    AutoModelForCausalLM, AutoTokenizer = _import_avatar_transformers_chat_classes()

    if torch.cuda.is_available():
        try:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model_kwargs: dict[str, object] = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        normalized_mode = str(training_mode or "memory_optimized").strip().lower()
        if normalized_mode == "memory_optimized" and _detect_avatar_memory_optimized_support()[0]:
            try:
                try:
                    from transformers import BitsAndBytesConfig
                except Exception:
                    from transformers.utils.quantization_config import BitsAndBytesConfig
                model_kwargs["device_map"] = {"": 0}
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            except Exception:
                model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = {"": 0}
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    with _avatar_chat_model_lock:
        _avatar_chat_models[int(avatar_id)] = (cache_key, model, tokenizer)
    return model, tokenizer


def _avatar_generate_personality_reply(
    *,
    avatar_id: int,
    base_model_id: str,
    adapter_path: str,
    training_mode: str,
    system_prompt: str,
    history: list[dict[str, str]] | None,
    message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model, tokenizer = _avatar_load_chat_model(
        int(avatar_id),
        str(base_model_id or "").strip(),
        str(adapter_path or "").strip(),
        training_mode=str(training_mode or "memory_optimized"),
    )
    import torch

    messages = [{"role": "system", "content": str(system_prompt or "").strip()}]
    for turn in history or []:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": str(message or "").strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = getattr(model, "device", None)
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = None
    if target_device is not None and str(target_device) != "meta":
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(32, min(int(max_new_tokens or 220), 512)),
            do_sample=True,
            temperature=max(0.1, min(float(temperature or 0.8), 1.5)),
            top_p=max(0.1, min(float(top_p or 0.9), 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    reply = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
    if not reply:
        raise ValueError("Generated an empty reply")
    return reply


def _avatar_is_short_backchannel(turn: dict[str, object]) -> bool:
    text = _clean_avatar_dataset_text(str(turn.get("text") or ""))
    if not text:
        return True
    if _avatar_word_count(text) > 6:
        return False
    duration = max(0.0, float(turn.get("end_time") or 0.0) - float(turn.get("start_time") or 0.0))
    if duration > 3.0:
        return False
    lowered = text.lower().strip(" .,!?:;-'\"")
    short_markers = {
        "yeah",
        "yes",
        "yep",
        "yup",
        "right",
        "ok",
        "okay",
        "mm",
        "mmm",
        "mhm",
        "uh huh",
        "huh",
        "no",
        "wow",
        "damn",
        "sure",
        "true",
        "fair",
        "exactly",
        "i know",
        "gotcha",
    }
    return lowered in short_markers or _avatar_word_count(text) <= 3


def _build_avatar_personality_long_form_samples(
    session: Session,
    avatar: Avatar,
) -> list[dict[str, object]]:
    query = (
        select(TranscriptSegment, Video.title)
        .join(Video, Video.id == TranscriptSegment.video_id)
        .where(TranscriptSegment.speaker_id == avatar.speaker_id)
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_time)
    )
    rows = session.exec(query).all()
    samples: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    max_gap_seconds = 2.5

    def flush_current():
        nonlocal current
        if not current:
            return
        text = _clean_avatar_dataset_text(" ".join(str(part) for part in current.get("parts", []) if str(part).strip()))
        duration_seconds = max(0.0, float(current.get("end_time") or 0.0) - float(current.get("start_time") or 0.0))
        segment_ids = [int(segment_id) for segment_id in current.get("segment_ids", [])]
        word_count = _avatar_word_count(text)
        if duration_seconds >= 20.0 and word_count >= 60:
            sample_id = _avatar_long_form_sample_id(int(current["video_id"]), float(current["start_time"]), float(current["end_time"]), segment_ids)
            samples.append(
                {
                    "sample_id": sample_id,
                    "video_id": int(current["video_id"]),
                    "video_title": str(current.get("video_title") or ""),
                    "start_time": float(current["start_time"]),
                    "end_time": float(current["end_time"]),
                    "duration_seconds": duration_seconds,
                    "word_count": word_count,
                    "segment_count": len(segment_ids),
                    "text": text,
                }
            )
        current = None

    for segment, video_title in rows:
        text = _clean_avatar_dataset_text(getattr(segment, "text", None))
        if not text:
            continue
        video_id = int(getattr(segment, "video_id", 0) or 0)
        start_time = float(getattr(segment, "start_time", 0.0) or 0.0)
        end_time = float(getattr(segment, "end_time", start_time) or start_time)
        segment_id = int(getattr(segment, "id", 0) or 0)
        if (
            current is None
            or int(current["video_id"]) != video_id
            or (start_time - float(current["end_time"])) > max_gap_seconds
        ):
            flush_current()
            current = {
                "video_id": video_id,
                "video_title": str(video_title or ""),
                "start_time": start_time,
                "end_time": end_time,
                "segment_ids": [segment_id],
                "parts": [text],
            }
            continue

        current["end_time"] = end_time
        cast_segment_ids = current["segment_ids"]
        cast_parts = current["parts"]
        if isinstance(cast_segment_ids, list):
            cast_segment_ids.append(segment_id)
        if isinstance(cast_parts, list):
            cast_parts.append(text)

    flush_current()
    for sample in samples:
        wc = max(1, int(sample.get("word_count") or 1))
        text = str(sample.get("text") or "")
        style_density = _avatar_style_signal_count(text) / wc
        substance_density = _avatar_substance_signal_count(text) / wc
        sample["style_density"] = round(style_density, 5)
        sample["substance_density"] = round(substance_density, 5)
    samples.sort(key=lambda row: (
        -(
            (float(row.get("style_density") or 0.0) * 0.4 + float(row.get("substance_density") or 0.0) * 0.6)
            * math.log(max(2, int(row.get("word_count") or 2)))
        ),
        -float(row.get("duration_seconds") or 0.0),
        str(row.get("video_title") or ""),
    ))
    samples_path, _, _ = _avatar_personality_long_form_paths(avatar)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples


def _read_avatar_personality_long_form_page(
    session: Session,
    avatar: Avatar,
    *,
    offset: int,
    limit: int,
    state: str = "all",
) -> AvatarPersonalityLongFormPageRead:
    normalized_state = str(state or "all").strip().lower()
    if normalized_state not in {"all", "included", "rejected"}:
        normalized_state = "all"
    samples = _build_avatar_personality_long_form_samples(session, avatar)
    state_map = _load_avatar_personality_long_form_states(avatar)
    config = _load_avatar_personality_long_form_config(avatar)

    items: list[AvatarPersonalityLongFormSampleRead] = []
    included_count = 0
    rejected_count = 0
    for row in samples:
        sample_state = str(state_map.get(str(row.get("sample_id")), "included") or "included")
        if sample_state == "rejected":
            rejected_count += 1
        else:
            included_count += 1
        if normalized_state != "all" and sample_state != normalized_state:
            continue
        items.append(
            AvatarPersonalityLongFormSampleRead(
                sample_id=str(row.get("sample_id") or ""),
                video_id=int(row.get("video_id") or 0),
                video_title=str(row.get("video_title") or ""),
                start_time=float(row.get("start_time") or 0.0),
                end_time=float(row.get("end_time") or 0.0),
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                word_count=int(row.get("word_count") or 0),
                segment_count=int(row.get("segment_count") or 0),
                text=str(row.get("text") or ""),
                style_density=float(row.get("style_density") or 0.0),
                substance_density=float(row.get("substance_density") or 0.0),
                state="rejected" if sample_state == "rejected" else "included",
            )
        )

    total = len(items)
    page_items = items[offset: offset + limit]
    selected_count = min(max(0, int(config.take_count or 0)), included_count)
    _write_avatar_personality_long_form_config(
        avatar,
        take_count=int(config.take_count or 0),
        included_count=included_count,
        rejected_count=rejected_count,
        selected_count=selected_count,
    )
    return AvatarPersonalityLongFormPageRead(
        avatar_id=int(avatar.id),
        total=total,
        included_count=included_count,
        rejected_count=rejected_count,
        selected_count=selected_count,
        take_count=int(config.take_count or 0),
        limit=int(limit),
        offset=int(offset),
        has_more=(offset + len(page_items)) < total,
        items=page_items,
    )


def _avatar_response_looks_incomplete(text: str) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return True
    if cleaned.endswith(("...", "-", ":", ";", ",")):
        return True
    if cleaned.endswith((".", "!", "?", "\"", "'")):
        return False
    trailing = cleaned.split()[-1].strip(".,!?;:'\"").lower()
    trailing_connectors = {
        "and",
        "but",
        "or",
        "so",
        "because",
        "if",
        "when",
        "while",
        "that",
        "which",
        "who",
        "where",
        "with",
        "about",
        "of",
        "to",
        "for",
        "from",
        "in",
        "on",
        "at",
        "by",
        "than",
        "then",
        "like",
    }
    if trailing in trailing_connectors:
        return True
    return _avatar_word_count(cleaned) <= 12 and cleaned[-1].islower()


def _extend_avatar_response_turn(
    turns: list[dict[str, object]],
    start_index: int,
    *,
    target_speaker_id: int,
) -> dict[str, object]:
    base_turn = turns[start_index]
    merged_text_parts = [_clean_avatar_dataset_text(str(base_turn.get("text") or ""))]
    merged_segment_ids = [int(segment_id) for segment_id in base_turn.get("source_segment_ids", [])]
    merged_end_time = float(base_turn.get("end_time") or base_turn.get("start_time") or 0.0)
    consumed_indexes = [start_index]
    continuation_gap_seconds = 16.0
    max_skipped_backchannels = 2
    max_response_segments = 5
    max_response_words = 320
    probe_index = start_index + 1

    while probe_index < len(turns) and len(consumed_indexes) < max_response_segments:
        skipped_backchannels: list[int] = []
        lookahead = probe_index
        while lookahead < len(turns):
            candidate = turns[lookahead]
            candidate_speaker_id = int(candidate.get("speaker_id") or 0)
            if candidate_speaker_id == target_speaker_id:
                candidate_text = _clean_avatar_dataset_text(str(candidate.get("text") or ""))
                gap = max(0.0, float(candidate.get("start_time") or 0.0) - merged_end_time)
                if gap > continuation_gap_seconds:
                    lookahead = len(turns)
                    break
                current_response_text = _clean_avatar_dataset_text(" ".join(merged_text_parts))
                if skipped_backchannels and not _avatar_response_looks_incomplete(current_response_text):
                    lookahead = len(turns)
                    break
                if (
                    not skipped_backchannels
                    and gap > 2.5
                    and not _avatar_response_looks_incomplete(current_response_text)
                    and not (candidate_text[:1].islower() if candidate_text else False)
                ):
                    lookahead = len(turns)
                    break
                if _avatar_word_count(current_response_text) + _avatar_word_count(candidate_text) > max_response_words:
                    lookahead = len(turns)
                    break
                merged_text_parts.append(candidate_text)
                merged_segment_ids.extend(int(segment_id) for segment_id in candidate.get("source_segment_ids", []))
                merged_end_time = float(candidate.get("end_time") or candidate.get("start_time") or merged_end_time)
                consumed_indexes.append(lookahead)
                probe_index = lookahead + 1
                break
            if len(skipped_backchannels) >= max_skipped_backchannels or not _avatar_is_short_backchannel(candidate):
                lookahead = len(turns)
                break
            skipped_backchannels.append(lookahead)
            lookahead += 1
        if lookahead >= len(turns):
            break

    return {
        "text": _clean_avatar_dataset_text(" ".join(part for part in merged_text_parts if part)),
        "end_time": merged_end_time,
        "source_segment_ids": merged_segment_ids,
        "consumed_indexes": consumed_indexes,
    }


def _avatar_normalize_curation_key(text: str | None) -> str:
    cleaned = _clean_avatar_dataset_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _avatar_has_transcript_noise(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return True
    if "http://" in cleaned or "https://" in cleaned or "www." in cleaned.lower():
        return True
    weird_chars = sum(1 for char in cleaned if not (char.isalnum() or char.isspace() or char in ".,!?':;\"()-"))
    if weird_chars > max(4, len(cleaned) * 0.08):
        return True
    repeated_words = re.findall(r"\b(\w+)\b(?:\s+\1\b){2,}", cleaned.lower())
    return bool(repeated_words)


def _avatar_is_acknowledgement_response(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text).lower().strip(" .,!?:;-'\"")
    if not cleaned:
        return True
    if _avatar_word_count(cleaned) > 8:
        return False
    acknowledgement_patterns = {
        "yeah",
        "yeah yeah",
        "yes",
        "yep",
        "yup",
        "right",
        "exactly",
        "i know",
        "sure",
        "okay",
        "ok",
        "true",
        "fair",
        "for sure",
        "totally",
        "absolutely",
        "uh huh",
        "mhm",
        "mm hmm",
        "no",
    }
    if cleaned in acknowledgement_patterns:
        return True
    if _avatar_word_count(cleaned) <= 2:
        return True
    if _avatar_word_count(cleaned) == 3:
        opinion_words = {"funny", "crazy", "wild", "insane", "amazing", "terrible", "awesome",
                         "ridiculous", "hilarious", "interesting", "wrong", "true", "fair", "weird"}
        return not any(word in cleaned.split() for word in opinion_words)
    return False


def _avatar_response_has_high_referentiality(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text).lower()
    if not cleaned:
        return False
    referential_markers = [
        "look at", "right there", "right here", "this one", "that clip",
        "that thing", "over there", "you see", "you can see", "watch this",
        "on the screen", "on screen", "in the chat", "in the comments",
        "pull that up", "scroll down", "check this out", "as you can see",
    ]
    match_count = sum(1 for marker in referential_markers if marker in cleaned)
    return match_count >= 2 or (match_count >= 1 and _avatar_word_count(cleaned) < 20)


def _avatar_context_has_question(text: str | None) -> bool:
    cleaned = _clean_avatar_dataset_text(text)
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    lowered = cleaned.lower()
    return any(marker in lowered for marker in [" why ", " how ", " what ", " when ", " where ", " who ", "did ", "does ", "is ", "are ", "can ", "could "])


_AVATAR_BASE_STYLE_MARKERS = [
    " i think ",
    " i mean ",
    " honestly ",
    " basically ",
    " like ",
    " actually ",
    " literally ",
    " kinda ",
    " sort of ",
    " probably ",
    " maybe ",
    " because ",
    " the thing is ",
    " you know ",
    " i'm ",
    " i've ",
    " don't ",
    " can't ",
    " won't ",
]


def _avatar_build_speaker_style_weights(
    turns_by_video: dict[int, list[dict[str, object]]],
    target_speaker_id: int,
) -> dict[str, float]:
    speaker_word_count = 0
    other_word_count = 0
    speaker_marker_counts: dict[str, int] = {marker.strip(): 0 for marker in _AVATAR_BASE_STYLE_MARKERS}
    other_marker_counts: dict[str, int] = {marker.strip(): 0 for marker in _AVATAR_BASE_STYLE_MARKERS}
    for turns in turns_by_video.values():
        for turn in turns:
            text = f" {_clean_avatar_dataset_text(str(turn.get('text') or '')).lower()} "
            wc = len(text.split())
            is_target = int(turn.get("speaker_id") or 0) == target_speaker_id
            if is_target:
                speaker_word_count += wc
            else:
                other_word_count += wc
            for marker in _AVATAR_BASE_STYLE_MARKERS:
                count = text.count(marker)
                if count > 0:
                    key = marker.strip()
                    if is_target:
                        speaker_marker_counts[key] += count
                    else:
                        other_marker_counts[key] += count
    # If the comparison pool is too small, the frequency ratios are
    # unreliable and generic markers get inflated to the 3.0 cap.
    # Fall back to uniform weights when there isn't enough non-target speech.
    min_comparison_words = 500
    comparison_is_sparse = other_word_count < min_comparison_words

    weights: dict[str, float] = {}
    for marker in _AVATAR_BASE_STYLE_MARKERS:
        key = marker.strip()
        if comparison_is_sparse:
            weights[key] = 1.0
            continue
        speaker_rate = (speaker_marker_counts[key] / max(1, speaker_word_count)) * 1000
        other_rate = (other_marker_counts[key] / max(1, other_word_count)) * 1000
        if speaker_rate > 0 and other_rate > 0:
            ratio = speaker_rate / max(0.01, other_rate)
        elif speaker_rate > 0:
            ratio = 2.0
        else:
            ratio = 0.5
        weights[key] = min(3.0, max(0.2, ratio))
    return weights


_AVATAR_SUBSTANCE_MARKERS_LOGIC = [
    " therefore ", " because ", " consequently ", " the reason is ",
    " my point is ", " that's why ", " so the ", " which means ",
    " it follows ", " in other words ", " what that means is ",
    " if you think about it ", " the argument is ",
]
_AVATAR_SUBSTANCE_MARKERS_ANALOGY = [
    " it's like ", " think of it as ", " imagine ", " same way that ",
    " kind of like ", " similar to ", " just like ", " picture this ",
    " analogy ", " metaphor ", " compared to ",
]
_AVATAR_SUBSTANCE_MARKERS_EVIDENCE = [
    " studies show ", " according to ", " research ", " the data ",
    " evidence ", " historically ", " for example ", " for instance ",
    " the fact is ", " statistically ",
]
_AVATAR_SUBSTANCE_MARKERS_POSITION = [
    " i believe ", " the problem is ", " what people don't realize ",
    " the issue is ", " in my view ", " the real question ",
    " fundamentally ", " the key is ", " here's the thing ",
    " what i'm saying is ", " my position ", " i would argue ",
]
_AVATAR_ALL_SUBSTANCE_MARKERS = (
    _AVATAR_SUBSTANCE_MARKERS_LOGIC
    + _AVATAR_SUBSTANCE_MARKERS_ANALOGY
    + _AVATAR_SUBSTANCE_MARKERS_EVIDENCE
    + _AVATAR_SUBSTANCE_MARKERS_POSITION
)


_AVATAR_COMMON_SENTENCE_STARTERS = {
    "this", "that", "well", "what", "when", "where", "who", "how", "why",
    "the", "and", "but", "so", "if", "its", "they", "there", "here",
    "yeah", "yes", "no", "not", "now", "then", "also", "just", "like",
    "right", "okay", "sure", "look", "let", "see", "think", "know",
    "people", "some", "because", "every", "even", "still",
}


def _avatar_substance_signal_count(text: str | None) -> int:
    lowered = f" {_clean_avatar_dataset_text(text).lower()} "
    if not lowered.strip():
        return 0
    count = 0
    for marker in _AVATAR_ALL_SUBSTANCE_MARKERS:
        if marker in lowered:
            count += 1
    # Count mid-sentence proper nouns as reference signals, excluding
    # common words that just happen to be capitalized at sentence start.
    cleaned = _clean_avatar_dataset_text(text) or ""
    proper_noun_count = 0
    sentences = re.split(r'[.!?]+\s+', cleaned)
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:
            stripped = word.strip(".,!?;:'\"()-")
            if stripped and stripped[0].isupper() and len(stripped) >= 3 and stripped.lower() not in _AVATAR_COMMON_SENTENCE_STARTERS:
                proper_noun_count += 1
    count += min(3, proper_noun_count)
    return count


def _avatar_style_signal_count(text: str | None, speaker_style_weights: dict[str, float] | None = None) -> int:
    lowered = f" {_clean_avatar_dataset_text(text).lower()} "
    if not lowered.strip():
        return 0
    total = 0.0
    for marker in _AVATAR_BASE_STYLE_MARKERS:
        if marker in lowered:
            weight = (speaker_style_weights or {}).get(marker.strip(), 1.0)
            total += weight
    return int(round(total))


def _score_avatar_personality_example(row: dict[str, object], *, speaker_style_weights: dict[str, float] | None = None) -> dict[str, object]:
    response_text = _clean_avatar_dataset_text(str(row.get("response_text") or ""))
    context_text = _clean_avatar_dataset_text(str(row.get("context_text") or ""))
    response_word_count = _avatar_word_count(response_text)
    context_word_count = _avatar_word_count(context_text)
    context_turns = int(row.get("context_turns") or 0)
    source_segment_count = len([segment_id for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()])

    completion_score = 90
    context_score = 25
    style_score = 35
    reject_reasons: list[str] = []

    if response_word_count < 8:
        completion_score -= 40
        style_score -= 20
        reject_reasons.append("short_response")
    if _avatar_is_acknowledgement_response(response_text):
        completion_score -= 35
        style_score -= 30
        reject_reasons.append("acknowledgement_only")
    if _avatar_response_looks_incomplete(response_text):
        completion_score -= 45
        reject_reasons.append("truncated_response")
    if _avatar_has_transcript_noise(response_text):
        completion_score -= 30
        style_score -= 25
        reject_reasons.append("transcript_noise")
    if _avatar_response_has_high_referentiality(response_text):
        context_score -= 20
        style_score -= 15
        reject_reasons.append("high_referentiality")

    if context_turns <= 0 or context_word_count < 12:
        context_score -= 30
        reject_reasons.append("weak_context")
    else:
        context_score += min(35, max(0, context_word_count - 12) // 3)
        context_score += min(15, context_turns * 5)
    if _avatar_context_has_question(context_text):
        context_score += 10

    if 20 <= response_word_count <= 240:
        style_score += 20
    elif response_word_count >= 12:
        style_score += 8
    elif response_word_count > 0:
        style_score -= 10

    if source_segment_count > 1:
        completion_score += 10
        style_score += 8

    style_score += min(20, _avatar_style_signal_count(response_text, speaker_style_weights) * 5)
    if any(pronoun in f" {response_text.lower()} " for pronoun in [" i ", " i'm ", " i've ", " me ", " my "]):
        style_score += 8

    substance_score = 15
    substance_signals = _avatar_substance_signal_count(response_text)
    substance_score += min(30, substance_signals * 6)
    if response_word_count < 20:
        substance_score -= 15
    if substance_signals > 0 and _avatar_style_signal_count(response_text, speaker_style_weights) > 0:
        substance_score += 10
    substance_score = max(0, min(100, substance_score))

    completion_score = max(0, min(100, completion_score))
    context_score = max(0, min(100, context_score))
    style_score = max(0, min(100, style_score))
    quality_score = max(0, min(100, round(
        (completion_score * 0.40) + (context_score * 0.22) + (style_score * 0.28) + (substance_score * 0.10)
    )))

    hard_reject_reasons = {"acknowledgement_only", "transcript_noise", "weak_context", "truncated_response"}
    if hard_reject_reasons.intersection(reject_reasons) or quality_score < 40:
        auto_label = "reject"
    elif quality_score >= 78 and completion_score >= 70 and context_score >= 55 and style_score >= 60:
        auto_label = "gold"
    else:
        auto_label = "silver"

    row["response_word_count"] = int(response_word_count)
    row["context_word_count"] = int(context_word_count)
    row["source_segment_count"] = int(source_segment_count)
    row["quality_score"] = int(quality_score)
    row["completion_score"] = int(completion_score)
    row["context_score"] = int(context_score)
    row["style_score"] = int(style_score)
    row["substance_score"] = int(substance_score)
    row["reject_reasons"] = sorted(set(reject_reasons))
    row["auto_label"] = auto_label
    return row


def _apply_avatar_duplicate_rejects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_indexes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        dedupe_key = " || ".join(
            [
                _avatar_normalize_curation_key(str(row.get("context_text") or "")),
                _avatar_normalize_curation_key(str(row.get("response_text") or "")),
            ]
        )
        if not dedupe_key.strip(" |"):
            continue
        grouped_indexes.setdefault(dedupe_key, []).append(index)

    for duplicate_indexes in grouped_indexes.values():
        if len(duplicate_indexes) <= 1:
            continue
        group_id = int(min(int(rows[idx].get("example_id") or 0) for idx in duplicate_indexes) + 1)
        ranked = sorted(
            duplicate_indexes,
            key=lambda idx: (
                -int(rows[idx].get("quality_score") or 0),
                -int(rows[idx].get("style_score") or 0),
                int(rows[idx].get("example_id") or 0),
            ),
        )
        keep_index = ranked[0]
        rows[keep_index]["duplicate_group_id"] = group_id
        rows[keep_index]["duplicate_group_size"] = len(duplicate_indexes)
        for duplicate_index in ranked[1:]:
            row = rows[duplicate_index]
            reject_reasons = {str(reason) for reason in row.get("reject_reasons", [])}
            reject_reasons.add("duplicate_example")
            row["reject_reasons"] = sorted(reject_reasons)
            row["quality_score"] = max(0, int(row.get("quality_score") or 0) - 20)
            row["auto_label"] = "reject"
            row["duplicate_group_id"] = group_id
            row["duplicate_group_size"] = len(duplicate_indexes)
    return rows


def _avatar_context_text_for_embedding(raw_context: str | None) -> str:
    lines = []
    for line in str(raw_context or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "Conversation context:" or stripped.startswith("Podcast episode:"):
            continue
        separator_index = stripped.find(":")
        if separator_index > 0:
            stripped = stripped[separator_index + 1 :].strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def _avatar_embedding_tokens(text: str | None) -> list[str]:
    cleaned = _avatar_normalize_curation_key(text)
    if not cleaned:
        return []
    stop_words = {
        "the", "a", "an", "and", "or", "but", "so", "that", "this", "these", "those", "it", "its", "to", "of",
        "for", "on", "in", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "they", "we", "me", "my", "our", "your", "their", "them", "his", "her",
        "do", "does", "did", "not", "just", "like", "really", "very", "kind", "sort", "have", "has", "had",
    }
    return [token for token in cleaned.split() if len(token) > 1 and token not in stop_words][:96]


def _avatar_hash_embedding_feature(feature: str, *, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, "little", signed=False)
    return hashed % dimension, (1.0 if ((hashed >> 8) & 1) == 0 else -1.0)


def _avatar_build_example_embedding(row: dict[str, object], *, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    response_tokens = _avatar_embedding_tokens(str(row.get("response_text") or ""))
    context_tokens = _avatar_embedding_tokens(_avatar_context_text_for_embedding(str(row.get("context_text") or "")))

    def add_feature(feature: str, weight: float) -> None:
        index, sign = _avatar_hash_embedding_feature(feature, dimension=dimension - 8)
        vector[index] += np.float32(weight * sign)

    for token in response_tokens:
        add_feature(f"r:{token}", 1.35)
    for token in context_tokens:
        add_feature(f"c:{token}", 0.7)
    for left, right in zip(response_tokens, response_tokens[1:]):
        add_feature(f"rb:{left}_{right}", 1.6)
    for left, right in zip(context_tokens, context_tokens[1:]):
        add_feature(f"cb:{left}_{right}", 0.85)

    response_text = _clean_avatar_dataset_text(str(row.get("response_text") or "")).lower()
    style_markers = ["i think", "i mean", "honestly", "basically", "because", "the thing is", "you know", "i'm", "i've", "don't", "can't", "won't"]
    for marker in style_markers:
        if marker in response_text:
            add_feature(f"s:{marker}", 1.1)
    substance_markers = [
        "therefore", "the reason is", "my point is", "that's why", "which means",
        "it's like", "think of it as", "imagine", "same way that",
        "i believe", "the problem is", "the issue is", "here's the thing",
        "for example", "for instance", "fundamentally",
    ]
    for marker in substance_markers:
        if marker in response_text:
            add_feature(f"sub:{marker}", 1.3)

    vector[-10] = min(1.0, float(row.get("response_word_count") or 0) / 320.0)
    vector[-9] = min(1.0, float(row.get("context_word_count") or 0) / 180.0)
    vector[-8] = min(1.0, float(row.get("context_turns") or 0) / 4.0)
    vector[-7] = min(1.0, float(row.get("source_segment_count") or 0) / 5.0)
    vector[-6] = min(1.0, float(row.get("style_score") or 0) / 100.0)
    vector[-5] = min(1.0, float(row.get("substance_score") or 0) / 100.0)
    vector[-4] = 1.0 if _avatar_context_has_question(str(row.get("context_text") or "")) else 0.0
    vector[-3] = min(1.0, _avatar_substance_signal_count(str(row.get("response_text") or "")) / 5.0)
    vector[-2] = 1.0 if "truncated_response" in {str(reason) for reason in row.get("reject_reasons", [])} else 0.0
    vector[-1] = 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _avatar_assign_embedding_clusters(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not rows:
        return rows, {"embedding_model": "hashing_ngram_v1", "dimension": 128, "cluster_count": 0, "duplicate_example_count": 0, "hotspot_cluster_count": 0, "clusters": []}

    embeddings = np.vstack([_avatar_build_example_embedding(row) for row in rows]).astype(np.float32)
    
    semantic_embeddings = None
    try:
        from .services import semantic_search as sem_svc
        texts = [str(row.get("response_text") or "") for row in rows]
        semantic_embeddings = sem_svc._embed_texts(texts)
    except Exception as e:
        print(f"[_avatar_assign_embedding_clusters] Failed to generate semantic embeddings: {e}")

    rng = np.random.default_rng(42)
    lsh_a = rng.standard_normal((embeddings.shape[1], 10)).astype(np.float32)
    lsh_b = rng.standard_normal((embeddings.shape[1], 10)).astype(np.float32)
    bucket_maps: list[dict[str, list[int]]] = [{}, {}]
    projections = [lsh_a, lsh_b]
    candidate_limit = 18
    similarity_threshold = 0.89
    semantic_threshold = 0.85

    sorted_indexes = sorted(
        range(len(rows)),
        key=lambda idx: (
            -int(rows[idx].get("quality_score") or 0),
            -int(rows[idx].get("style_score") or 0),
            int(rows[idx].get("example_id") or 0),
        ),
    )

    duplicate_groups: dict[int, list[int]] = {}
    representative_group_id: dict[int, int] = {}
    next_group_id = max((int(row.get("duplicate_group_id") or 0) for row in rows), default=0) + 1
    scanned_indexes: list[int] = []

    for idx in sorted_indexes:
        vector = embeddings[idx]
        candidate_indexes: set[int] = set()
        for projection, bucket_map in zip(projections, bucket_maps):
            signature = "".join("1" if value >= 0 else "0" for value in np.matmul(vector, projection))
            candidate_indexes.update(bucket_map.get(signature, []))

        best_match_idx: int | None = None
        best_similarity = -1.0
        for candidate_idx in candidate_indexes:
            similarity = float(np.dot(vector, embeddings[candidate_idx]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = candidate_idx

        # Semantic duplication check
        semantic_best_match_idx: int | None = None
        semantic_best_similarity = -1.0
        if semantic_embeddings is not None and len(scanned_indexes) > 0:
            sem_vec = semantic_embeddings[idx]
            sem_candidates = semantic_embeddings[scanned_indexes]
            sims = np.matmul(sem_candidates, sem_vec)
            best_idx_in_scanned = int(np.argmax(sims))
            best_sim = float(sims[best_idx_in_scanned])
            if best_sim >= semantic_threshold:
                semantic_best_similarity = best_sim
                semantic_best_match_idx = scanned_indexes[best_idx_in_scanned]

        is_lsh_dup = best_match_idx is not None and best_similarity >= similarity_threshold
        is_sem_dup = semantic_best_match_idx is not None and semantic_best_similarity >= semantic_threshold

        if is_lsh_dup or is_sem_dup:
            target_match_idx = semantic_best_match_idx if is_sem_dup else best_match_idx
            if target_match_idx is None: 
                continue # Should never happen
                
            group_id = representative_group_id.get(target_match_idx)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                representative_group_id[target_match_idx] = group_id
                duplicate_groups[group_id] = [target_match_idx]
            duplicate_groups.setdefault(group_id, []).append(idx)
            rows[idx]["duplicate_group_id"] = group_id
            rows[idx]["duplicate_similarity"] = round(max(best_similarity, semantic_best_similarity), 4)
            reject_reasons = {str(reason) for reason in rows[idx].get("reject_reasons", [])}
            if is_sem_dup:
                reject_reasons.add("semantic_duplicate")
            if is_lsh_dup:
                reject_reasons.add("lsh_duplicate")
            rows[idx]["reject_reasons"] = sorted(reject_reasons)
            rows[idx]["auto_label"] = "reject"
            rows[idx]["quality_score"] = max(0, int(rows[idx].get("quality_score") or 0) - 18)
            continue

        scanned_indexes.append(idx)

        for projection, bucket_map in zip(projections, bucket_maps):
            signature = "".join("1" if value >= 0 else "0" for value in np.matmul(vector, projection))
            members = bucket_map.setdefault(signature, [])
            if len(members) < candidate_limit:
                members.append(idx)

    for group_id, member_indexes in duplicate_groups.items():
        member_count = len(member_indexes)
        for member_idx in member_indexes:
            rows[member_idx]["duplicate_group_id"] = group_id
            rows[member_idx]["duplicate_group_size"] = member_count

    active_indexes = [idx for idx, row in enumerate(rows) if str(row.get("auto_label") or "silver") != "reject"]
    cluster_count_target = max(16, min(72, int(round(np.sqrt(max(1, len(active_indexes))) * 0.8))))
    cluster_threshold = 0.58
    centroids: list[np.ndarray] = []
    centroid_sizes: list[int] = []
    cluster_members: dict[int, list[int]] = {}

    active_sorted_indexes = sorted(
        active_indexes,
        key=lambda idx: (
            -int(rows[idx].get("quality_score") or 0),
            -int(rows[idx].get("style_score") or 0),
            int(rows[idx].get("example_id") or 0),
        ),
    )

    for idx in active_sorted_indexes:
        vector = embeddings[idx]
        if not centroids:
            centroids.append(vector.copy())
            centroid_sizes.append(1)
            cluster_members[1] = [idx]
            rows[idx]["cluster_id"] = 1
            continue

        centroid_matrix = np.vstack(centroids)
        similarities = np.matmul(centroid_matrix, vector)
        best_cluster_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_cluster_index])

        if len(centroids) < cluster_count_target and best_similarity < cluster_threshold:
            cluster_id = len(centroids) + 1
            centroids.append(vector.copy())
            centroid_sizes.append(1)
            cluster_members[cluster_id] = [idx]
            rows[idx]["cluster_id"] = cluster_id
            continue

        cluster_id = best_cluster_index + 1
        cluster_members.setdefault(cluster_id, []).append(idx)
        current_size = centroid_sizes[best_cluster_index]
        updated_centroid = (centroids[best_cluster_index] * current_size) + vector
        norm = float(np.linalg.norm(updated_centroid))
        if norm > 0:
            updated_centroid /= norm
        centroids[best_cluster_index] = updated_centroid.astype(np.float32)
        centroid_sizes[best_cluster_index] = current_size + 1
        rows[idx]["cluster_id"] = cluster_id

    if centroids:
        centroid_matrix = np.vstack(centroids)
        for idx, row in enumerate(rows):
            if row.get("cluster_id") is not None:
                continue
            similarities = np.matmul(centroid_matrix, embeddings[idx])
            cluster_id = int(np.argmax(similarities)) + 1
            row["cluster_id"] = cluster_id
            cluster_members.setdefault(cluster_id, []).append(idx)

    cluster_sizes = {cluster_id: len(member_indexes) for cluster_id, member_indexes in cluster_members.items()}
    largest_cluster_size = max(cluster_sizes.values(), default=0)
    hotspot_threshold = max(18, int(np.percentile(list(cluster_sizes.values()), 85))) if cluster_sizes else 0

    for row in rows:
        cluster_id = row.get("cluster_id")
        cluster_size = cluster_sizes.get(int(cluster_id), 0) if cluster_id is not None else 0
        row["cluster_size"] = cluster_size
        if largest_cluster_size > 1 and cluster_size > 0:
            row["diversity_score"] = max(0, min(100, int(round(100 * (1 - ((cluster_size - 1) / max(1, largest_cluster_size - 1)))))))
        else:
            row["diversity_score"] = 100 if cluster_size > 0 else 0
        if cluster_size >= hotspot_threshold and hotspot_threshold > 0:
            row["cluster_hotspot"] = True

    cluster_summary = []
    for cluster_id, member_indexes in sorted(cluster_members.items(), key=lambda item: (-len(item[1]), item[0])):
        top_examples = sorted(
            member_indexes,
            key=lambda idx: (
                -int(rows[idx].get("quality_score") or 0),
                -int(rows[idx].get("style_score") or 0),
                int(rows[idx].get("example_id") or 0),
            ),
        )[:3]
        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(member_indexes)),
                "representative_example_ids": [int(rows[idx].get("example_id") or 0) for idx in top_examples],
            }
        )

    return rows, {
        "embedding_model": "hashing_ngram_v1",
        "dimension": int(embeddings.shape[1]),
        "cluster_count": int(len(cluster_members)),
        "duplicate_example_count": int(sum(1 for row in rows if _avatar_row_has_duplicate_risk(row))),
        "hotspot_cluster_count": int(sum(1 for size in cluster_sizes.values() if size >= hotspot_threshold and hotspot_threshold > 0)),
        "largest_cluster_size": int(largest_cluster_size),
        "clusters": cluster_summary[:48],
    }


def _resolve_avatar_personality_example_state(row: dict[str, object], state_map: dict[int, str]) -> tuple[str, str | None]:
    try:
        example_id = int(row.get("example_id"))
    except Exception:
        return "approved", None
    manual_state = state_map.get(example_id)
    if manual_state in {"approved", "rejected"}:
        return manual_state, manual_state
    auto_label = str(row.get("auto_label") or "silver").strip().lower()
    return ("rejected" if auto_label == "reject" else "approved"), None


def _avatar_row_has_duplicate_risk(row: dict[str, object]) -> bool:
    reasons = {str(reason) for reason in row.get("reject_reasons", [])}
    return int(row.get("duplicate_group_size") or 0) > 1 or any("duplicate" in reason for reason in reasons)


def _build_avatar_personality_training_readiness(
    *,
    approved_count: int,
    approved_gold_count: int,
    approved_duration_seconds: float,
    approved_word_count: int,
    needs_review_count: int,
    duplicate_count: int,
    hotspot_cluster_count: int,
    largest_cluster_size: int,
    total_example_count: int,
    auto_reject_count: int,
) -> AvatarPersonalityTrainingReadinessRead:
    approved_count = max(0, int(approved_count or 0))
    approved_gold_count = max(0, int(approved_gold_count or 0))
    approved_duration_seconds = max(0.0, float(approved_duration_seconds or 0.0))
    approved_word_count = max(0, int(approved_word_count or 0))
    needs_review_count = max(0, int(needs_review_count or 0))
    duplicate_count = max(0, int(duplicate_count or 0))
    hotspot_cluster_count = max(0, int(hotspot_cluster_count or 0))
    largest_cluster_size = max(0, int(largest_cluster_size or 0))
    total_example_count = max(0, int(total_example_count or 0))
    auto_reject_count = max(0, int(auto_reject_count or 0))

    approved_duration_hours = approved_duration_seconds / 3600.0
    duplicate_pressure = int(round((duplicate_count / max(1, total_example_count)) * 100))
    hotspot_pressure = int(round((largest_cluster_size / max(1, approved_count)) * 100)) if approved_count else 0
    gold_ratio = (approved_gold_count / max(1, approved_count)) if approved_count else 0.0
    reject_ratio = (auto_reject_count / max(1, total_example_count)) if total_example_count else 0.0

    coverage_score = min(100.0, (approved_count / 2000.0) * 100.0)
    gold_score = min(100.0, (approved_gold_count / 800.0) * 100.0)
    duration_score = min(100.0, (approved_duration_hours / 8.0) * 100.0)
    score = int(round((coverage_score * 0.32) + (gold_score * 0.46) + (duration_score * 0.22)))
    score -= int(max(0, duplicate_pressure - 8) * 0.8)
    score -= int(max(0, hotspot_pressure - 10) * 0.9)
    if gold_ratio < 0.35:
        score -= int(round((0.35 - gold_ratio) * 70))
    score = max(0, min(100, score))

    can_train_now = approved_count >= 800 and approved_gold_count >= 300 and approved_duration_hours >= 3.0 and approved_word_count >= 40000
    status = "insufficient"
    if approved_count >= 12000 or approved_word_count >= 600000:
        status = "oversized"
    elif approved_count >= 2500 and approved_gold_count >= 900 and approved_duration_hours >= 8.0:
        status = "strong"
    elif can_train_now:
        status = "ready"
    elif approved_count >= 300 and approved_gold_count >= 120 and approved_duration_hours >= 1.5:
        status = "borderline"

    manual_review_roi: Literal["high", "medium", "low"] = "high"
    if can_train_now:
        manual_review_roi = "medium"
    if status in {"strong", "oversized"} and needs_review_count <= max(150, int(approved_count * 0.1)):
        manual_review_roi = "low"
    elif needs_review_count > max(250, int(approved_count * 0.2)) or gold_ratio < 0.35:
        manual_review_roi = "high"

    summary = f"{approved_count:,} included examples, {approved_gold_count:,} gold, and about {approved_duration_hours:.1f}h of approved response audio."
    recommended_action = "Keep curating high-value exchanges before training."
    caution_parts: list[str] = []

    if status == "insufficient":
        recommended_action = "Do more manual approval. You do not have enough high-value personality data yet."
    elif status == "borderline":
        recommended_action = "You can run a pilot LoRA now, but more manual approval should still improve style fidelity."
    elif status == "ready":
        recommended_action = "You have enough data to train now. Further manual approval should focus on precision, not volume."
    elif status == "strong":
        recommended_action = "You are well past the minimum. Prefer pruning weak repeats over adding more volume."
    elif status == "oversized":
        recommended_action = "Train from the current set or a gold-biased subset. More raw volume is unlikely to help."

    if duplicate_pressure >= 12:
        caution_parts.append("duplicate pressure is elevated")
    if hotspot_pressure >= 14 or hotspot_cluster_count >= 8:
        caution_parts.append("one topic cluster is starting to dominate the corpus")
    if reject_ratio >= 0.22:
        caution_parts.append("the raw source pool is noisy")
    if status == "oversized":
        caution_parts.append("too much repetitive or generic data can dilute the speaker's style and slow training")

    return AvatarPersonalityTrainingReadinessRead(
        status=status,
        score=score,
        can_train_now=can_train_now,
        approved_duration_hours=round(approved_duration_hours, 2),
        approved_word_count=approved_word_count,
        recommended_action=recommended_action,
        summary=summary,
        caution=(". ".join(caution_parts).strip().rstrip(".") + ".") if caution_parts else None,
        manual_review_roi=manual_review_roi,
        duplicate_pressure=duplicate_pressure,
        hotspot_pressure=hotspot_pressure,
        gold_ratio=round(gold_ratio, 3),
        reject_ratio=round(reject_ratio, 3),
    )


def _avatar_extract_json_object(raw_text: str) -> dict[str, object]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    return data


def _avatar_normalize_llm_label(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"gold", "silver", "reject"}:
        return normalized
    return "silver"


def _avatar_resolve_local_judge_model() -> tuple[str, str]:
    import httpx

    ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    judge_override = _main()._normalize_ollama_model_ref(os.getenv("AVATAR_JUDGE_MODEL", ""))
    requested_model = _main()._normalize_ollama_model_ref(os.getenv("OLLAMA_MODEL", "mistral"))
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=8)
        response.raise_for_status()
        available_models = [str(model.get("name") or "") for model in response.json().get("models", []) if isinstance(model, dict)]
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot connect to local Ollama at {ollama_url}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to query local Ollama at {ollama_url}: {exc}") from exc

    if judge_override and any(_main()._ollama_model_name_matches(name, judge_override) for name in available_models):
        return ollama_url, judge_override

    preferred_models = [
        "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
        "qwen2.5:7b",
        "mistral:latest",
        "mistral",
        "qwen3.5:27b",
    ]
    for preferred in preferred_models:
        if any(_main()._ollama_model_name_matches(name, preferred) for name in available_models):
            return ollama_url, preferred

    if requested_model and any(_main()._ollama_model_name_matches(name, requested_model) for name in available_models):
        return ollama_url, requested_model

    if available_models:
        return ollama_url, str(available_models[0])
    raise HTTPException(status_code=503, detail=f"No local Ollama models found at {ollama_url}")


def _avatar_run_local_judge(example: dict[str, object], *, ollama_url: str, model: str) -> dict[str, object]:
    import httpx

    prompt = (
        "You are grading a candidate training example for a personality LoRA.\n"
        "Judge whether the assistant response is a strong example of conversational personality data.\n"
        "Focus on completion, stylistic richness, conversational usefulness, and transcript cleanliness.\n"
        "Usefulness means the response reveals personality traits, speaking patterns, or emotional tendencies "
        "that would help a model replicate this person's communication style.\n"
        "Return JSON only with keys: label, confidence, completion_score, style_score, usefulness_score, reasons, rationale.\n"
        "label must be one of: gold, silver, reject.\n"
        "reasons must be a short array of snake_case strings.\n"
        "Be conservative about gold. Reject truncated, repetitive, transcript-broken, or low-value acknowledgements.\n\n"
        f"Episode: {str(example.get('video_title') or '').strip()}\n"
        f"Context:\n{str(example.get('context_text') or '').strip()}\n\n"
        f"Response:\n{str(example.get('response_text') or '').strip()}\n"
    )

    response = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 160,
            },
        },
        timeout=150,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "").strip()
    if not raw_text:
        raise ValueError("Local judge model returned an empty response")
    data = _avatar_extract_json_object(raw_text)
    completion_score = max(0, min(100, int(data.get("completion_score") or 0)))
    style_score = max(0, min(100, int(data.get("style_score") or 0)))
    usefulness_score = max(0, min(100, int(data.get("usefulness_score") or 0)))
    provided_confidence = int(data.get("confidence") or 0)
    if provided_confidence <= 0:
        provided_confidence = int(round((completion_score + style_score + usefulness_score) / 3))
    return {
        "llm_label": _avatar_normalize_llm_label(str(data.get("label") or "")),
        "llm_confidence": max(0, min(100, provided_confidence)),
        "llm_completion_score": completion_score,
        "llm_style_score": style_score,
        "llm_usefulness_score": usefulness_score,
        "llm_reasons": [str(reason).strip() for reason in data.get("reasons", []) if str(reason).strip()],
        "llm_rationale": _clean_avatar_dataset_text(str(data.get("rationale") or "")) or None,
        "llm_model": model,
        "llm_judged_at": datetime.now().isoformat(),
    }


def _avatar_personality_judge_target_match(
    row: dict[str, object],
    *,
    state_map: dict[int, str],
    normalized_target: str,
) -> tuple[bool, str | None]:
    state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
    current_label = _avatar_normalize_llm_label(str(row.get("auto_label") or "silver"))
    if normalized_target == "needs_review":
        target_match = current_label == "silver" and manual_state is None and state == "approved"
    elif normalized_target == "silver":
        target_match = current_label == "silver"
    else:
        target_match = True
    return target_match, manual_state


def _run_avatar_personality_judge_pass(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    *,
    max_examples: int,
    overwrite_existing: bool,
    target_filter: str,
) -> AvatarPersonalityDatasetRead:
    normalized_target = str(target_filter or "needs_review").strip().lower()
    if normalized_target not in {"needs_review", "silver", "all"}:
        normalized_target = "needs_review"

    ollama_url, judge_model = _avatar_resolve_local_judge_model()
    review_path, _ = _avatar_personality_review_paths(avatar)
    if not review_path.exists():
        raise HTTPException(status_code=404, detail="Dataset review artifact not found. Build the dataset first.")

    state_map = _load_avatar_personality_state_map(avatar)
    rows = list(_iter_avatar_personality_review_examples(avatar))
    judged = 0

    for row in rows:
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        current_label = _avatar_normalize_llm_label(str(row.get("auto_label") or "silver"))
        if normalized_target == "needs_review":
            target_match = current_label == "silver" and manual_state is None and state == "approved"
        elif normalized_target == "silver":
            target_match = current_label == "silver"
        else:
            target_match = True
        if not target_match:
            continue
        if not overwrite_existing and row.get("llm_label"):
            continue
        if judged >= max(1, min(int(max_examples or 40), 200)):
            break

        row["heuristic_label"] = _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))
        try:
            judge_result = _avatar_run_local_judge(row, ollama_url=ollama_url, model=judge_model)
        except Exception as exc:
            row["llm_rationale"] = f"Judge pass failed: {exc}"
            row["llm_model"] = judge_model
            row["llm_judged_at"] = datetime.now().isoformat()
            row["llm_reasons"] = ["judge_error"]
            row["llm_confidence"] = 0
            row["llm_label"] = row["heuristic_label"]
            judged += 1
            continue

        row.update(judge_result)
        if manual_state is None:
            row["auto_label"] = judge_result["llm_label"]
        judged += 1

    review_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    refreshed = _refresh_avatar_personality_dataset_exports(avatar, personality)
    return refreshed


def _start_avatar_personality_judge_pass(
    *,
    avatar_id: int,
    max_examples: int,
    overwrite_existing: bool,
    target_filter: str,
) -> AvatarPersonalityJudgeStatusRead:
    normalized_target = str(target_filter or "needs_review").strip().lower()
    if normalized_target not in {"needs_review", "silver", "all"}:
        normalized_target = "needs_review"
    normalized_max = max(1, min(int(max_examples or 40), 200))

    with Session(engine) as session:
        avatar = session.get(Avatar, avatar_id)
        if not avatar:
            raise HTTPException(status_code=404, detail="Avatar not found")
        speaker = session.get(Speaker, avatar.speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Source speaker not found")
        personality, _, _ = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
        review_path, _ = _avatar_personality_review_paths(avatar)
        if not review_path.exists():
            raise HTTPException(status_code=404, detail="Dataset review artifact not found. Build the dataset first.")
        state_map = _load_avatar_personality_state_map(avatar)
        rows = list(_iter_avatar_personality_review_examples(avatar))
        candidates: list[int] = []
        for idx, row in enumerate(rows):
            target_match, _ = _avatar_personality_judge_target_match(
                row,
                state_map=state_map,
                normalized_target=normalized_target,
            )
            if not target_match:
                continue
            if not overwrite_existing and row.get("llm_label"):
                continue
            candidates.append(idx)
            if len(candidates) >= normalized_max:
                break
        ollama_url, judge_model = _avatar_resolve_local_judge_model()

        with _avatar_judge_runs_lock:
            active_thread = _avatar_judge_threads.get(int(avatar_id))
            if active_thread and active_thread.is_alive():
                raise HTTPException(status_code=409, detail="A local judge pass is already running for this avatar.")
            stop_event = threading.Event()
            _avatar_judge_stop_events[int(avatar_id)] = stop_event

        started_at = datetime.now().isoformat()
        initial_status = _write_avatar_personality_judge_status(
            avatar,
            {
                "status": "running",
                "active": True,
                "stop_requested": False,
                "model": judge_model,
                "target_filter": normalized_target,
                "overwrite_existing": bool(overwrite_existing),
                "max_examples": normalized_max,
                "total_candidates": len(candidates),
                "processed_count": 0,
                "judged_count": 0,
                "promoted_count": 0,
                "rejected_count": 0,
                "current_example_id": None,
                "current_video_title": None,
                "current_stage": "queued",
                "started_at": started_at,
                "finished_at": None,
                "error": None,
                "recent_results": [],
            },
        )

    def _runner():
        try:
            with Session(engine) as inner_session:
                avatar_inner = inner_session.get(Avatar, avatar_id)
                if not avatar_inner:
                    raise RuntimeError("Avatar not found")
                speaker_inner = inner_session.get(Speaker, avatar_inner.speaker_id)
                if not speaker_inner:
                    raise RuntimeError("Source speaker not found")
                personality_inner, _, _ = _ensure_avatar_profiles(inner_session, avatar_inner, speaker_name=speaker_inner.name)
                review_path_inner, _ = _avatar_personality_review_paths(avatar_inner)
                state_map_inner = _load_avatar_personality_state_map(avatar_inner)
                rows_inner = list(_iter_avatar_personality_review_examples(avatar_inner))
                recent_results: list[dict[str, object]] = []
                judged = 0
                promoted = 0
                rejected = 0
                processed = 0

                for row_index in candidates:
                    if row_index >= len(rows_inner):
                        continue
                    row = rows_inner[row_index]
                    target_match, manual_state = _avatar_personality_judge_target_match(
                        row,
                        state_map=state_map_inner,
                        normalized_target=normalized_target,
                    )
                    if not target_match:
                        continue
                    if not overwrite_existing and row.get("llm_label"):
                        continue
                    if stop_event.is_set():
                        _write_avatar_personality_judge_status(
                            avatar_inner,
                            {
                                "status": "stopped",
                                "active": False,
                                "stop_requested": True,
                                "processed_count": processed,
                                "judged_count": judged,
                                "promoted_count": promoted,
                                "rejected_count": rejected,
                                "current_stage": "stopped",
                                "current_example_id": None,
                                "current_video_title": None,
                                "finished_at": datetime.now().isoformat(),
                                "recent_results": recent_results,
                            },
                        )
                        break

                    row["heuristic_label"] = _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))
                    _write_avatar_personality_judge_status(
                        avatar_inner,
                        {
                            "status": "stopping" if stop_event.is_set() else "running",
                            "active": True,
                            "stop_requested": stop_event.is_set(),
                            "processed_count": processed,
                            "judged_count": judged,
                            "promoted_count": promoted,
                            "rejected_count": rejected,
                            "current_example_id": int(row.get("example_id") or 0),
                            "current_video_title": str(row.get("video_title") or ""),
                            "current_stage": "judging",
                            "recent_results": recent_results,
                        },
                    )

                    try:
                        judge_result = _avatar_run_local_judge(row, ollama_url=ollama_url, model=judge_model)
                    except Exception as exc:
                        row["llm_rationale"] = f"Judge pass failed: {exc}"
                        row["llm_model"] = judge_model
                        row["llm_judged_at"] = datetime.now().isoformat()
                        row["llm_reasons"] = ["judge_error"]
                        row["llm_confidence"] = 0
                        row["llm_label"] = row["heuristic_label"]
                    else:
                        row.update(judge_result)
                        if manual_state is None:
                            row["auto_label"] = judge_result["llm_label"]

                    judged += 1
                    processed += 1
                    if row.get("llm_label") == "gold" and row.get("heuristic_label") != "gold":
                        promoted += 1
                    if row.get("llm_label") == "reject" and row.get("heuristic_label") != "reject":
                        rejected += 1
                    recent_results.insert(
                        0,
                        {
                            "example_id": int(row.get("example_id") or 0),
                            "video_title": str(row.get("video_title") or ""),
                            "llm_label": _avatar_normalize_llm_label(str(row.get("llm_label") or "silver")),
                            "llm_confidence": int(row.get("llm_confidence") or 0),
                            "llm_reasons": [str(reason) for reason in row.get("llm_reasons", [])],
                            "heuristic_label": _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver")),
                            "judged_at": row.get("llm_judged_at"),
                        },
                    )
                    recent_results = recent_results[:12]
                    _write_avatar_personality_judge_status(
                        avatar_inner,
                        {
                            "status": "stopping" if stop_event.is_set() else "running",
                            "active": True,
                            "stop_requested": stop_event.is_set(),
                            "processed_count": processed,
                            "judged_count": judged,
                            "promoted_count": promoted,
                            "rejected_count": rejected,
                            "current_example_id": int(row.get("example_id") or 0),
                            "current_video_title": str(row.get("video_title") or ""),
                            "current_stage": "writing_result",
                            "recent_results": recent_results,
                        },
                    )

                review_path_inner.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_inner), encoding="utf-8")
                _refresh_avatar_personality_dataset_exports(avatar_inner, personality_inner)
                inner_session.add(personality_inner)
                inner_session.commit()
                final_status = "stopped" if stop_event.is_set() else "completed"
                _write_avatar_personality_judge_status(
                    avatar_inner,
                    {
                        "status": final_status,
                        "active": False,
                        "stop_requested": stop_event.is_set(),
                        "processed_count": processed,
                        "judged_count": judged,
                        "promoted_count": promoted,
                        "rejected_count": rejected,
                        "current_example_id": None,
                        "current_video_title": None,
                        "current_stage": "complete",
                        "finished_at": datetime.now().isoformat(),
                        "recent_results": recent_results,
                    },
                )
        except Exception as exc:
            with Session(engine) as error_session:
                avatar_error = error_session.get(Avatar, avatar_id)
                if avatar_error:
                    _write_avatar_personality_judge_status(
                        avatar_error,
                        {
                            "status": "failed",
                            "active": False,
                            "stop_requested": stop_event.is_set(),
                            "current_stage": "failed",
                            "error": str(exc),
                            "finished_at": datetime.now().isoformat(),
                        },
                    )
        finally:
            with _avatar_judge_runs_lock:
                _avatar_judge_stop_events.pop(int(avatar_id), None)
                _avatar_judge_threads.pop(int(avatar_id), None)

    thread = threading.Thread(target=_runner, daemon=True, name=f"avatar-judge-{avatar_id}")
    with _avatar_judge_runs_lock:
        _avatar_judge_threads[int(avatar_id)] = thread
    thread.start()
    return initial_status


def _load_avatar_personality_dataset(avatar: Avatar, personality: AvatarPersonalityProfile) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    metadata: dict[str, object] = {}
    preview_examples: list[AvatarPersonalityDatasetExampleRead] = []

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    if preview_path.exists():
        try:
            preview_rows = json.loads(preview_path.read_text(encoding="utf-8"))
            if isinstance(preview_rows, list):
                preview_examples = [
                    AvatarPersonalityDatasetExampleRead(**row)
                    for row in preview_rows
                    if isinstance(row, dict)
                ]
        except Exception:
            preview_examples = []

    generated_at = None
    raw_generated_at = metadata.get("generated_at")
    if isinstance(raw_generated_at, str):
        try:
            generated_at = datetime.fromisoformat(raw_generated_at)
        except Exception:
            generated_at = None

    raw_readiness = metadata.get("readiness")
    readiness: AvatarPersonalityTrainingReadinessRead
    if isinstance(raw_readiness, dict):
        try:
            readiness = AvatarPersonalityTrainingReadinessRead(**raw_readiness)
        except Exception:
            readiness = _build_avatar_personality_training_readiness(
                approved_count=int(metadata.get("approved_example_count") or personality.approved_example_count or 0),
                approved_gold_count=int(metadata.get("gold_example_count") or 0),
                approved_duration_seconds=float(metadata.get("approved_duration_seconds") or 0.0),
                approved_word_count=int(metadata.get("approved_word_count") or 0),
                needs_review_count=int(metadata.get("needs_review_count") or 0),
                duplicate_count=int(metadata.get("duplicate_example_count") or 0),
                hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
                largest_cluster_size=int(metadata.get("largest_cluster_size") or 0),
                total_example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
                auto_reject_count=int(metadata.get("auto_reject_count") or 0),
            )
    else:
        readiness = _build_avatar_personality_training_readiness(
            approved_count=int(metadata.get("approved_example_count") or personality.approved_example_count or 0),
            approved_gold_count=int(metadata.get("gold_example_count") or 0),
            approved_duration_seconds=float(metadata.get("approved_duration_seconds") or 0.0),
            approved_word_count=int(metadata.get("approved_word_count") or 0),
            needs_review_count=int(metadata.get("needs_review_count") or 0),
            duplicate_count=int(metadata.get("duplicate_example_count") or 0),
            hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
            largest_cluster_size=int(metadata.get("largest_cluster_size") or 0),
            total_example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
            auto_reject_count=int(metadata.get("auto_reject_count") or 0),
        )

    return AvatarPersonalityDatasetRead(
        avatar_id=int(avatar.id),
        speaker_id=int(avatar.speaker_id),
        status=str(personality.status or "draft"),
        system_prompt=personality.system_prompt,
        base_model_id=personality.base_model_id,
        dataset_path=str(dataset_path) if dataset_path.exists() else (personality.dataset_path or None),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        example_count=int(metadata.get("example_count") or personality.dataset_example_count or 0),
        gold_example_count=int(metadata.get("gold_example_count") or 0),
        silver_example_count=int(metadata.get("silver_example_count") or 0),
        auto_reject_count=int(metadata.get("auto_reject_count") or 0),
        needs_review_count=int(metadata.get("needs_review_count") or 0),
        duplicate_example_count=int(metadata.get("duplicate_example_count") or 0),
        cluster_count=int(metadata.get("cluster_count") or 0),
        hotspot_cluster_count=int(metadata.get("hotspot_cluster_count") or 0),
        llm_judged_count=int(metadata.get("llm_judged_count") or 0),
        llm_promoted_count=int(metadata.get("llm_promoted_count") or 0),
        llm_rejected_count=int(metadata.get("llm_rejected_count") or 0),
        source_turn_count=int(metadata.get("source_turn_count") or personality.source_turn_count or 0),
        discarded_turn_count=int(metadata.get("discarded_turn_count") or 0),
        readiness=readiness,
        preview_examples=preview_examples,
        generated_at=generated_at or personality.last_built_at,
    )


def _load_avatar_personality_state_map(avatar: Avatar) -> dict[int, str]:
    _, states_path = _avatar_personality_review_paths(avatar)
    if not states_path.exists():
        return {}
    try:
        raw = json.loads(states_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            example_id = int(key)
        except Exception:
            continue
        state = str(value or "").strip().lower()
        if state in {"approved", "rejected"}:
            out[example_id] = state
    return out


def _write_avatar_personality_state_map(avatar: Avatar, state_map: dict[int, str]) -> Path:
    _, states_path = _avatar_personality_review_paths(avatar)
    serializable = {str(int(key)): str(value) for key, value in sorted(state_map.items()) if value in {"approved", "rejected"}}
    states_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return states_path


def _iter_avatar_personality_review_examples(avatar: Avatar):
    review_path, _ = _avatar_personality_review_paths(avatar)
    if not review_path.exists():
        return
    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _refresh_avatar_personality_dataset_exports(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    *,
    total_example_count: int | None = None,
) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    state_map = _load_avatar_personality_state_map(avatar)
    approved_count = 0
    rejected_count = 0
    gold_count = 0
    silver_count = 0
    auto_reject_count = 0
    needs_review_count = 0
    duplicate_count = 0
    llm_judged_count = 0
    llm_promoted_count = 0
    llm_rejected_count = 0
    approved_gold_count = 0
    approved_duration_seconds = 0.0
    approved_word_count = 0
    largest_cluster_size = 0
    preview_examples: list[dict[str, object]] = []

    with dataset_path.open("w", encoding="utf-8") as export_handle:
        for row in _iter_avatar_personality_review_examples(avatar):
            try:
                example_id = int(row.get("example_id"))
            except Exception:
                continue
            auto_label = str(row.get("auto_label") or "silver").strip().lower()
            heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
            llm_label = row.get("llm_label")
            if auto_label == "gold":
                gold_count += 1
            elif auto_label == "reject":
                auto_reject_count += 1
            else:
                silver_count += 1
            if llm_label:
                llm_judged_count += 1
                normalized_llm_label = _avatar_normalize_llm_label(str(llm_label))
                if heuristic_label != "gold" and normalized_llm_label == "gold":
                    llm_promoted_count += 1
                if heuristic_label != "reject" and normalized_llm_label == "reject":
                    llm_rejected_count += 1

            state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
            if auto_label == "silver" and manual_state is None:
                needs_review_count += 1
            if _avatar_row_has_duplicate_risk(row):
                duplicate_count += 1
            if state == "rejected":
                rejected_count += 1
            else:
                approved_count += 1
                approved_duration_seconds += max(0.0, float(row.get("end_time") or 0.0) - float(row.get("start_time") or 0.0))
                approved_word_count += int(row.get("response_word_count") or 0)
                largest_cluster_size = max(largest_cluster_size, int(row.get("cluster_size") or 0))
                if auto_label == "gold":
                    approved_gold_count += 1
                messages = row.get("messages")
                if isinstance(messages, list):
                    export_handle.write(json.dumps({"messages": messages, "metadata": row.get("metadata", {})}, ensure_ascii=False) + "\n")
            if len(preview_examples) < 8:
                preview_examples.append(
                    {
                        "example_id": example_id,
                        "video_id": int(row.get("video_id") or 0),
                        "video_title": str(row.get("video_title") or ""),
                        "start_time": float(row.get("start_time") or 0.0),
                        "end_time": float(row.get("end_time") or 0.0),
                        "context_text": str(row.get("context_text") or ""),
                        "response_text": str(row.get("response_text") or ""),
                        "source_segment_ids": [int(segment_id) for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()],
                        "source_segment_count": int(row.get("source_segment_count") or 0),
                        "context_turns": int(row.get("context_turns") or 0),
                        "response_word_count": int(row.get("response_word_count") or 0),
                        "context_word_count": int(row.get("context_word_count") or 0),
                        "quality_score": int(row.get("quality_score") or 0),
                        "completion_score": int(row.get("completion_score") or 0),
                        "context_score": int(row.get("context_score") or 0),
                        "style_score": int(row.get("style_score") or 0),
                        "substance_score": int(row.get("substance_score") or 0),
                        "cluster_id": int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                        "cluster_size": int(row.get("cluster_size") or 0),
                        "diversity_score": int(row.get("diversity_score") or 0),
                        "duplicate_group_id": int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
                        "duplicate_group_size": int(row.get("duplicate_group_size") or 0),
                        "duplicate_similarity": float(row.get("duplicate_similarity") or 0.0),
                        "heuristic_label": heuristic_label,
                        "llm_label": _avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
                        "llm_confidence": int(row.get("llm_confidence") or 0),
                        "llm_completion_score": int(row.get("llm_completion_score") or 0),
                        "llm_style_score": int(row.get("llm_style_score") or 0),
                        "llm_usefulness_score": int(row.get("llm_usefulness_score") or 0),
                        "llm_rationale": (str(row.get("llm_rationale") or "").strip() or None),
                        "llm_reasons": [str(reason) for reason in row.get("llm_reasons", [])],
                        "llm_model": (str(row.get("llm_model") or "").strip() or None),
                        "llm_judged_at": row.get("llm_judged_at"),
                        "auto_label": auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
                        "manual_state": manual_state,
                        "state": state,
                        "reject_reasons": [str(reason) for reason in row.get("reject_reasons", [])],
                    }
                )

    preview_path.write_text(json.dumps(preview_examples, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata["example_count"] = int(total_example_count if total_example_count is not None else approved_count + rejected_count)
    metadata["approved_example_count"] = int(approved_count)
    metadata["rejected_example_count"] = int(rejected_count)
    metadata["gold_example_count"] = int(gold_count)
    metadata["silver_example_count"] = int(silver_count)
    metadata["auto_reject_count"] = int(auto_reject_count)
    metadata["needs_review_count"] = int(needs_review_count)
    metadata["duplicate_example_count"] = int(duplicate_count)
    metadata["llm_judged_count"] = int(llm_judged_count)
    metadata["llm_promoted_count"] = int(llm_promoted_count)
    metadata["llm_rejected_count"] = int(llm_rejected_count)
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)
    cluster_summary: dict[str, object] = {}
    if cluster_summary_path.exists():
        try:
            cluster_summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
        except Exception:
            cluster_summary = {}
    metadata["cluster_count"] = int(cluster_summary.get("cluster_count") or 0)
    metadata["hotspot_cluster_count"] = int(cluster_summary.get("hotspot_cluster_count") or 0)
    metadata["largest_cluster_size"] = int(max(largest_cluster_size, int(cluster_summary.get("largest_cluster_size") or 0)))
    metadata["approved_duration_seconds"] = round(float(approved_duration_seconds), 3)
    metadata["approved_word_count"] = int(approved_word_count)
    metadata["approved_gold_count"] = int(approved_gold_count)
    metadata["readiness"] = _build_avatar_personality_training_readiness(
        approved_count=approved_count,
        approved_gold_count=approved_gold_count,
        approved_duration_seconds=approved_duration_seconds,
        approved_word_count=approved_word_count,
        needs_review_count=needs_review_count,
        duplicate_count=duplicate_count,
        hotspot_cluster_count=int(cluster_summary.get("hotspot_cluster_count") or 0),
        largest_cluster_size=max(largest_cluster_size, int(cluster_summary.get("largest_cluster_size") or 0)),
        total_example_count=int(total_example_count if total_example_count is not None else approved_count + rejected_count),
        auto_reject_count=auto_reject_count,
    ).model_dump()
    metadata["generated_at"] = datetime.now().isoformat()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    personality.dataset_path = str(dataset_path)
    personality.dataset_example_count = int(metadata.get("example_count") or total_example_count or 0)
    personality.approved_example_count = int(approved_count)
    personality.status = "dataset_ready" if approved_count > 0 else "needs_review"
    personality.last_built_at = datetime.now()
    personality.updated_at = personality.last_built_at

    return _load_avatar_personality_dataset(avatar, personality)


def _avatar_training_effective_label(row: dict[str, object]) -> str:
    llm_label = str(row.get("llm_label") or "").strip()
    if llm_label:
        return _avatar_normalize_llm_label(llm_label)
    return _avatar_normalize_llm_label(str(row.get("heuristic_label") or row.get("auto_label") or "silver"))


def _avatar_training_priority(row: dict[str, object], *, manual_state: str | None = None) -> tuple[float, ...]:
    final_label = _avatar_training_effective_label(row)
    label_rank = 2.0 if final_label == "gold" else 1.0 if final_label == "silver" else 0.0
    manual_rank = 1.0 if manual_state == "approved" else 0.0
    quality = float(row.get("quality_score") or 0)
    substance = float(row.get("substance_score") or 0)
    style = float(row.get("style_score") or 0)
    diversity = float(row.get("diversity_score") or 0)
    word_count = float(row.get("response_word_count") or 0)
    return (manual_rank, label_rank, quality, style, substance, diversity, word_count)


def _avatar_cluster_round_robin_select(
    rows: list[dict[str, object]],
    *,
    limit: int = 0,
    manual_overrides: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    if not rows:
        return []
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            example_id = 0
        cluster_id = row.get("cluster_id")
        bucket_key = f"cluster:{int(cluster_id)}" if cluster_id is not None else f"solo:{example_id}"
        buckets.setdefault(bucket_key, []).append(row)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda item: _avatar_training_priority(
                item,
                manual_state=(manual_overrides or {}).get(int(item.get("example_id") or 0)),
            ),
            reverse=True,
        )
    ordered_keys = sorted(
        buckets.keys(),
        key=lambda key: _avatar_training_priority(
            buckets[key][0],
            manual_state=(manual_overrides or {}).get(int(buckets[key][0].get("example_id") or 0)),
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    while ordered_keys and (limit <= 0 or len(selected) < limit):
        next_keys: list[str] = []
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if limit > 0 and len(selected) >= limit:
                break
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def _avatar_select_training_conversation_examples(
    avatar: Avatar,
    config: AvatarPersonalityTrainingConfigRead,
) -> list[dict[str, object]]:
    state_map = _load_avatar_personality_state_map(avatar)
    approved_rows: list[dict[str, object]] = []
    manual_overrides: dict[int, str] = {}
    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            continue
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        if state != "approved":
            continue
        row_copy = dict(row)
        row_copy["final_label"] = _avatar_training_effective_label(row_copy)
        row_copy["manual_state"] = manual_state
        approved_rows.append(row_copy)
        if manual_state:
            manual_overrides[example_id] = manual_state

    gold_pool = [row for row in approved_rows if row.get("manual_state") == "approved" or row.get("final_label") == "gold"]
    silver_pool = [row for row in approved_rows if row not in gold_pool and row.get("final_label") == "silver"]
    full_pool = sorted(
        approved_rows,
        key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
        reverse=True,
    )

    if config.export_strategy == "full_approved":
        selected = list(full_pool)
    elif config.export_strategy == "gold_only":
        selected = sorted(
            gold_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )
    elif config.export_strategy == "gold_plus_top_silver":
        selected = sorted(
            gold_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )
        silver_budget = min(len(silver_pool), max(250, len(selected) // 2))
        silver_selected = sorted(
            silver_pool,
            key=lambda row: _avatar_training_priority(row, manual_state=row.get("manual_state")),
            reverse=True,
        )[:silver_budget]
        selected.extend(silver_selected)
    else:
        selected = _avatar_cluster_round_robin_select(gold_pool, manual_overrides=manual_overrides)
        silver_budget = min(len(silver_pool), max(150, len(selected) // 4), 1200)
        if silver_budget > 0:
            selected.extend(
                _avatar_cluster_round_robin_select(
                    silver_pool,
                    limit=silver_budget,
                    manual_overrides=manual_overrides,
                )
            )

    max_examples = max(0, int(config.max_examples or 0))
    if max_examples <= 0 and config.export_strategy != "full_approved":
        if config.export_strategy == "gold_only":
            max_examples = 4000
        elif config.export_strategy == "gold_plus_top_silver":
            max_examples = 6000
        else:
            max_examples = 5000
    if max_examples > 0 and len(selected) > max_examples:
        selected = _avatar_cluster_round_robin_select(
            selected,
            limit=max_examples,
            manual_overrides=manual_overrides,
        )
    return selected


def _avatar_estimate_example_tokens(item: dict[str, object]) -> int:
    messages = item.get("messages")
    if not isinstance(messages, list):
        return 0
    total_chars = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, dict))
    return int(total_chars / 3.5) + len(messages) * 4


def _avatar_filter_oversized_examples(
    items: list[dict[str, object]],
    *,
    max_seq_length: int,
) -> list[dict[str, object]]:
    # NOTE: This is a coarse heuristic (chars/3.5), not tokenizer-aware.
    # Chat templates (especially Qwen's) add framing tokens that can push
    # the true length above this estimate.  The 20% headroom compensates,
    # but some borderline examples may still get left-truncated during
    # training.  A tokenizer-aware pass would require loading the model
    # tokenizer at package-preparation time.
    headroom = max(64, int(max_seq_length * 0.20))
    threshold = max_seq_length - headroom
    return [item for item in items if _avatar_estimate_example_tokens(item) <= threshold]


def _avatar_cap_per_video_contribution(
    items: list[dict[str, object]],
    *,
    max_ratio: float = 0.15,
) -> list[dict[str, object]]:
    if not items or max_ratio >= 1.0:
        return items
    total = len(items)
    max_per_video = max(5, int(total * max_ratio))
    video_counts: dict[int, int] = {}
    result: list[dict[str, object]] = []
    for item in items:
        video_id = int((item.get("metadata") or {}).get("video_id") or item.get("video_id") or 0)
        current_count = video_counts.get(video_id, 0)
        if current_count < max_per_video:
            result.append(item)
            video_counts[video_id] = current_count + 1
    return result


_LONG_FORM_USER_PROMPTS = [
    "Talk about this topic in your own words.",
    "Share your thoughts on this at length.",
    "Give your take on this — speak naturally.",
    "Go into detail on this subject.",
    "Walk me through your perspective here.",
    "Tell me what you think about this.",
    "Break this down in your own style.",
    "Speak freely about this topic.",
    "What are your thoughts? Take your time.",
    "Explain this the way you normally would.",
]


def _avatar_build_long_form_training_messages(prompt: str, sample: dict[str, object]) -> list[dict[str, str]]:
    episode_title = str(sample.get("video_title") or "").strip()
    sample_id = str(sample.get("sample_id") or sample.get("video_id") or "")
    variant_index = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) % len(_LONG_FORM_USER_PROMPTS)
    user_prompt = _LONG_FORM_USER_PROMPTS[variant_index]
    if episode_title:
        user_prompt += f" Episode: {episode_title}."
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(sample.get("text") or "").strip()},
    ]


def _avatar_select_training_long_form_examples(
    session: Session,
    avatar: Avatar,
    *,
    include_long_form: bool,
    prompt: str,
    max_examples: int | None = None,
) -> list[dict[str, object]]:
    if not include_long_form:
        return []
    config = _load_avatar_personality_long_form_config(avatar)
    take_count = max(0, int(config.take_count or 0))
    if max_examples is not None:
        take_count = min(take_count, max(0, int(max_examples)))
    if take_count <= 0:
        return []
    samples = _build_avatar_personality_long_form_samples(session, avatar)
    states = _load_avatar_personality_long_form_states(avatar)
    selected_samples: list[dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        if states.get(sample_id, "included") == "rejected":
            continue
        selected_samples.append(sample)
        if len(selected_samples) >= take_count:
            break
    output: list[dict[str, object]] = []
    for sample in selected_samples:
        output.append(
            {
                "source_kind": "long_form",
                "source_id": str(sample.get("sample_id") or ""),
                "messages": _avatar_build_long_form_training_messages(prompt, sample),
                "metadata": {
                    "source_kind": "long_form",
                    "video_id": int(sample.get("video_id") or 0),
                    "video_title": str(sample.get("video_title") or ""),
                    "start_time": float(sample.get("start_time") or 0.0),
                    "end_time": float(sample.get("end_time") or 0.0),
                    "duration_seconds": float(sample.get("duration_seconds") or 0.0),
                    "word_count": int(sample.get("word_count") or 0),
                    "segment_count": int(sample.get("segment_count") or 0),
                },
            }
        )
    return output


def _avatar_split_training_examples(
    items: list[dict[str, object]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not items:
        return [], []
    safe_ratio = max(0.01, min(0.2, float(validation_ratio or 0.02)))
    train_items: list[dict[str, object]] = []
    val_items: list[dict[str, object]] = []
    for item in items:
        source_key = f"{item.get('source_kind')}|{item.get('source_id')}"
        bucket = int(hashlib.md5(source_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
        if bucket < int(safe_ratio * 10000):
            val_items.append(item)
        else:
            train_items.append(item)
    if not train_items and val_items:
        train_items.append(val_items.pop())
    if not val_items and len(train_items) > 20:
        val_items.append(train_items.pop())
    return train_items, val_items


def _prepare_avatar_personality_training_package(
    session: Session,
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityTrainingPackageRead:
    config_path, manifest_path, train_path, val_path = _avatar_personality_training_paths(avatar)
    config = _load_avatar_personality_training_config(avatar)
    prompt = str(personality.system_prompt or _default_avatar_personality_prompt(avatar.name)).strip()
    base_model_id = str(config.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B").strip()

    conversation_rows = _avatar_select_training_conversation_examples(avatar, config)
    conversation_items: list[dict[str, object]] = []
    for row in conversation_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        conversation_items.append(
            {
                "source_kind": "conversation",
                "source_id": int(row.get("example_id") or 0),
                "messages": messages,
                "metadata": {
                    "source_kind": "conversation",
                    "example_id": int(row.get("example_id") or 0),
                    "video_id": int(row.get("video_id") or 0),
                    "video_title": str(row.get("video_title") or ""),
                    "start_time": float(row.get("start_time") or 0.0),
                    "end_time": float(row.get("end_time") or 0.0),
                    "quality_score": int(row.get("quality_score") or 0),
                    "style_score": int(row.get("style_score") or 0),
                    "cluster_id": int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                    "final_label": str(row.get("final_label") or "silver"),
                    "manual_state": row.get("manual_state"),
                },
            }
        )

    long_form_items = _avatar_select_training_long_form_examples(
        session,
        avatar,
        include_long_form=config.include_long_form,
        prompt=prompt,
        max_examples=config.max_long_form_examples,
    )
    all_items = conversation_items + long_form_items
    launch_settings = _recommend_avatar_training_launch_settings(
        model_id=base_model_id,
        training_mode=str(config.training_mode or "memory_optimized"),
        requested_lora_rank=None,
        requested_max_seq_length=None,
        requested_per_device_batch_size=None,
        requested_gradient_accumulation_steps=None,
    )
    effective_max_seq = int(launch_settings.get("max_seq_length") or 1024)
    all_items = _avatar_filter_oversized_examples(all_items, max_seq_length=effective_max_seq)
    all_items = _avatar_cap_per_video_contribution(all_items, max_ratio=0.15)
    train_items, val_items = _avatar_split_training_examples(all_items, validation_ratio=config.validation_ratio)
    training_plan = _build_avatar_personality_training_plan(
        avatar=avatar,
        personality=personality,
        config=config,
        selected_conversation_examples=len(conversation_items),
        selected_long_form_examples=len(long_form_items),
        train_examples=len(train_items),
        validation_examples=len(val_items),
        epochs=1,
    )

    with train_path.open("w", encoding="utf-8") as handle:
        for item in train_items:
            handle.write(json.dumps({"messages": item["messages"], "metadata": item["metadata"]}, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as handle:
        for item in val_items:
            handle.write(json.dumps({"messages": item["messages"], "metadata": item["metadata"]}, ensure_ascii=False) + "\n")

    _write_avatar_personality_training_config(
        avatar,
        base_model_id=base_model_id,
        dataset_profile=config.dataset_profile,
        training_strength=config.training_strength,
        export_strategy=config.export_strategy,
        validation_ratio=config.validation_ratio,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
        include_long_form=config.include_long_form,
        snapshot_interval_steps=config.snapshot_interval_steps,
    )
    prepared_at = datetime.now()
    manifest = AvatarPersonalityTrainingPackageRead(
        avatar_id=int(avatar.id),
        status="ready",
        base_model_id=base_model_id,
        dataset_profile=config.dataset_profile,
        training_strength=config.training_strength,
        export_strategy=config.export_strategy,
        validation_ratio=config.validation_ratio,
        max_examples=config.max_examples,
        max_long_form_examples=config.max_long_form_examples,
        include_long_form=config.include_long_form,
        conversation_examples_selected=len(conversation_items),
        long_form_examples_selected=len(long_form_items),
        total_examples_selected=len(all_items),
        train_examples=len(train_items),
        validation_examples=len(val_items),
        prompt=prompt,
        manifest_path=str(manifest_path),
        config_path=str(config_path),
        train_dataset_path=str(train_path),
        validation_dataset_path=str(val_path),
        command_preview=f"python backend/tools/train_avatar_personality.py --manifest \"{manifest_path}\"",
        prepared_at=prepared_at,
        training_plan=training_plan,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    personality.status = "training_ready" if len(all_items) > 0 else personality.status
    personality.updated_at = prepared_at
    return manifest


def _start_avatar_personality_training(
    avatar: Avatar,
    personality: AvatarPersonalityProfile,
    body: AvatarPersonalityTrainRequest,
) -> AvatarPersonalityTrainingStatusRead:
    persisted = _reconcile_avatar_personality_training_runtime(avatar)
    if persisted.active and persisted.status in {"queued", "running", "stopping"}:
        return persisted
    with _avatar_training_runs_lock:
        existing = _avatar_training_processes.get(int(avatar.id))
        if existing and existing.poll() is None:
            return _reconcile_avatar_personality_training_runtime(avatar)

    manifest = _read_avatar_personality_training_package(avatar)
    if manifest.status != "ready" or not manifest.manifest_path:
        raise HTTPException(status_code=400, detail="Prepare the training package first.")
    manifest_path = Path(manifest.manifest_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="Training manifest is missing. Prepare the package again.")
    model_installed, model_path = _hf_model_is_installed(str(manifest.base_model_id or ""))
    if not model_installed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Base model {manifest.base_model_id} is not installed in the local Hugging Face cache. "
                "Download the selected base model before starting training."
            ),
        )

    config_path, _, _, _ = _avatar_personality_training_paths(avatar)
    status_path, stop_path = _avatar_personality_training_runtime_paths(avatar)
    config = _load_avatar_personality_training_config(avatar)
    _clear_avatar_personality_training_stop_flag(avatar)
    training_mode = str(body.training_mode or "memory_optimized").strip().lower()
    if training_mode not in {"standard", "memory_optimized"}:
        training_mode = "memory_optimized"
    if training_mode == "memory_optimized":
        memory_optimized_available, memory_optimized_reason = _detect_avatar_memory_optimized_support()
        if not memory_optimized_available:
            raise HTTPException(
                status_code=400,
                detail=memory_optimized_reason or "Memory-optimized QLoRA mode is not available in the backend environment.",
            )

    training_runs_dir = _avatar_artifacts_dir(avatar) / "personality" / "training_runs"
    training_runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = training_runs_dir / run_name
    run_suffix = 1
    while output_dir.exists():
        output_dir = training_runs_dir / f"{run_name}-{run_suffix:02d}"
        run_suffix += 1
    if bool(body.overwrite_output) and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    personality.base_model_id = str(manifest.base_model_id or personality.base_model_id or "Qwen/Qwen3-8B")
    effective_settings = _recommend_avatar_training_launch_settings(
        model_id=personality.base_model_id,
        training_mode=training_mode,
        requested_lora_rank=body.lora_rank,
        requested_max_seq_length=body.max_seq_length,
        requested_per_device_batch_size=body.per_device_batch_size,
        requested_gradient_accumulation_steps=body.gradient_accumulation_steps,
    )
    personality.status = "training_queued"
    personality.updated_at = datetime.now()

    command = [
        str(Path(sys.executable)),
        str(Path(__file__).parent.parent / "tools" / "run_avatar_personality_training.py"),
        "--manifest",
        str(manifest_path),
        "--status-path",
        str(status_path),
        "--stop-path",
        str(stop_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(max(1, int(body.epochs or 1))),
        "--learning-rate",
        str(float(body.learning_rate or 5e-5)),
        "--lora-rank",
        str(int(effective_settings["lora_rank"])),
        "--max-seq-length",
        str(int(effective_settings["max_seq_length"])),
        "--per-device-batch-size",
        str(int(effective_settings["per_device_batch_size"])),
        "--gradient-accumulation-steps",
        str(int(effective_settings["gradient_accumulation_steps"])),
        "--warmup-ratio",
        str(max(0.0, min(0.2, float(body.warmup_ratio or 0.03)))),
        "--snapshot-interval-steps",
        str(max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps))),
        "--training-mode",
        training_mode,
        "--cuda-memory-fraction",
        str(float(effective_settings["cuda_memory_fraction"])),
    ]

    _write_avatar_personality_training_status(
        avatar,
        {
            "status": "queued",
            "active": True,
            "stop_requested": False,
            "process_id": None,
            "base_model_id": personality.base_model_id,
            "training_mode": training_mode,
            "adapter_path": None,
            "output_dir": str(output_dir),
            "current_stage": "queued",
            "epoch": 0.0,
            "step": 0,
            "max_steps": 0,
            "snapshot_interval_steps": max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps)),
            "train_examples": int(manifest.train_examples or 0),
            "validation_examples": int(manifest.validation_examples or 0),
            "latest_loss": None,
            "snapshots": [],
            "started_at": datetime.now(),
            "updated_at": datetime.now(),
            "finished_at": None,
            "error": None,
            "message": (
                f"Launching trainer subprocess with auto-tuned settings: "
                f"seq {int(effective_settings['max_seq_length'])}, "
                f"rank {int(effective_settings['lora_rank'])}, "
                f"batch {int(effective_settings['per_device_batch_size'])}, "
                f"grad {int(effective_settings['gradient_accumulation_steps'])}. "
                f"Snapshot cadence: "
                f"{max(0, int(body.snapshot_interval_steps if body.snapshot_interval_steps is not None else config.snapshot_interval_steps)) or 'auto'}. "
                f"Local model: {model_path or manifest.base_model_id}. "
                f"{str(effective_settings['rationale'])}"
            ),
        },
    )

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).parent.parent),
        creationflags=creationflags,
    )
    _write_avatar_personality_training_status(
        avatar,
        {
            "process_id": int(process.pid),
        },
    )
    with _avatar_training_runs_lock:
        _avatar_training_processes[int(avatar.id)] = process

    def _watch_training() -> None:
        return_code = process.wait()
        with _avatar_training_runs_lock:
            current = _avatar_training_processes.get(int(avatar.id))
            if current is process:
                _avatar_training_processes.pop(int(avatar.id), None)
        _sync_avatar_personality_training_completion(int(avatar.id))
        _avatar_release_cached_chat_model(int(avatar.id))
        if return_code != 0:
            try:
                status = _load_avatar_personality_training_status(avatar)
                if status.status not in {"failed", "stopped"}:
                    _write_avatar_personality_training_status(
                        avatar,
                        {
                            "status": "failed",
                            "active": False,
                            "process_id": int(process.pid),
                            "current_stage": "failed",
                            "finished_at": datetime.now(),
                            "updated_at": datetime.now(),
                            "message": f"Trainer exited with code {return_code}",
                            "error": status.error or f"Trainer exited with code {return_code}",
                        },
                    )
            except Exception:
                pass

    threading.Thread(target=_watch_training, daemon=True, name=f"avatar-train-watch-{avatar.id}").start()
    return _load_avatar_personality_training_status(avatar)


def _read_avatar_personality_dataset_page(
    avatar: Avatar,
    *,
    offset: int,
    limit: int,
    state_filter: str,
) -> AvatarPersonalityDatasetPageRead:
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(int(limit or 20), 100))
    normalized_filter = str(state_filter or "all").strip().lower()
    if normalized_filter not in {"all", "approved", "rejected", "gold", "silver", "auto_reject", "needs_review", "duplicate_risk"}:
        normalized_filter = "all"

    state_map = _load_avatar_personality_state_map(avatar)
    approved_count = 0
    rejected_count = 0
    gold_count = 0
    silver_count = 0
    auto_reject_count = 0
    needs_review_count = 0
    duplicate_count = 0
    llm_judged_count = 0
    llm_promoted_count = 0
    llm_rejected_count = 0
    matching_total = 0
    items: list[AvatarPersonalityDatasetExampleRead] = []
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)
    cluster_summary: dict[str, object] = {}
    if cluster_summary_path.exists():
        try:
            cluster_summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
        except Exception:
            cluster_summary = {}

    for row in _iter_avatar_personality_review_examples(avatar):
        try:
            example_id = int(row.get("example_id"))
        except Exception:
            continue
        auto_label = str(row.get("auto_label") or "silver").strip().lower()
        heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
        llm_label = row.get("llm_label")
        if auto_label == "gold":
            gold_count += 1
        elif auto_label == "reject":
            auto_reject_count += 1
        else:
            silver_count += 1
        if llm_label:
            llm_judged_count += 1
            normalized_llm_label = _avatar_normalize_llm_label(str(llm_label))
            if heuristic_label != "gold" and normalized_llm_label == "gold":
                llm_promoted_count += 1
            if heuristic_label != "reject" and normalized_llm_label == "reject":
                llm_rejected_count += 1
        state, manual_state = _resolve_avatar_personality_example_state(row, state_map)
        if state == "rejected":
            rejected_count += 1
        else:
            approved_count += 1
        if auto_label == "silver" and manual_state is None:
            needs_review_count += 1
        is_duplicate_risk = _avatar_row_has_duplicate_risk(row)
        if is_duplicate_risk:
            duplicate_count += 1

        matches_filter = normalized_filter == "all"
        if normalized_filter == "approved":
            matches_filter = state == "approved"
        elif normalized_filter == "rejected":
            matches_filter = state == "rejected"
        elif normalized_filter == "gold":
            matches_filter = auto_label == "gold"
        elif normalized_filter == "silver":
            matches_filter = auto_label == "silver"
        elif normalized_filter == "auto_reject":
            matches_filter = auto_label == "reject"
        elif normalized_filter == "needs_review":
            matches_filter = auto_label == "silver" and manual_state is None
        elif normalized_filter == "duplicate_risk":
            matches_filter = is_duplicate_risk
        if not matches_filter:
            continue
        if matching_total >= safe_offset and len(items) < safe_limit:
            items.append(
                AvatarPersonalityDatasetExampleRead(
                    example_id=example_id,
                    video_id=int(row.get("video_id") or 0),
                    video_title=str(row.get("video_title") or ""),
                    start_time=float(row.get("start_time") or 0.0),
                    end_time=float(row.get("end_time") or 0.0),
                    context_text=str(row.get("context_text") or ""),
                    response_text=str(row.get("response_text") or ""),
                    source_segment_ids=[int(segment_id) for segment_id in row.get("source_segment_ids", []) if isinstance(segment_id, int) or str(segment_id).isdigit()],
                    source_segment_count=int(row.get("source_segment_count") or 0),
                    context_turns=int(row.get("context_turns") or 0),
                    response_word_count=int(row.get("response_word_count") or 0),
                    context_word_count=int(row.get("context_word_count") or 0),
                    quality_score=int(row.get("quality_score") or 0),
                    completion_score=int(row.get("completion_score") or 0),
                    context_score=int(row.get("context_score") or 0),
                    style_score=int(row.get("style_score") or 0),
                    cluster_id=int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
                    cluster_size=int(row.get("cluster_size") or 0),
                    diversity_score=int(row.get("diversity_score") or 0),
                    duplicate_group_id=int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
                    duplicate_group_size=int(row.get("duplicate_group_size") or 0),
                    duplicate_similarity=float(row.get("duplicate_similarity") or 0.0),
                    heuristic_label=heuristic_label,
                    llm_label=_avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
                    llm_confidence=int(row.get("llm_confidence") or 0),
                    llm_completion_score=int(row.get("llm_completion_score") or 0),
                    llm_style_score=int(row.get("llm_style_score") or 0),
                    llm_usefulness_score=int(row.get("llm_usefulness_score") or 0),
                    llm_rationale=(str(row.get("llm_rationale") or "").strip() or None),
                    llm_reasons=[str(reason) for reason in row.get("llm_reasons", [])],
                    llm_model=(str(row.get("llm_model") or "").strip() or None),
                    llm_judged_at=row.get("llm_judged_at"),
                    auto_label=auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
                    manual_state=manual_state,
                    state=state,
                    reject_reasons=[str(reason) for reason in row.get("reject_reasons", [])],
                )
            )
        matching_total += 1

    return AvatarPersonalityDatasetPageRead(
        avatar_id=int(avatar.id),
        total=matching_total,
        approved_count=approved_count,
        rejected_count=rejected_count,
        gold_count=gold_count,
        silver_count=silver_count,
        auto_reject_count=auto_reject_count,
        needs_review_count=needs_review_count,
        duplicate_count=duplicate_count,
        cluster_count=int(cluster_summary.get("cluster_count") or 0),
        hotspot_cluster_count=int(cluster_summary.get("hotspot_cluster_count") or 0),
        llm_judged_count=llm_judged_count,
        llm_promoted_count=llm_promoted_count,
        llm_rejected_count=llm_rejected_count,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + safe_limit) < matching_total,
        state_filter=normalized_filter,
        items=items,
    )


def _row_to_example_read(row: dict, state: str, manual_state: Optional[str]) -> "AvatarPersonalityDatasetExampleRead":
    """Convert a raw review-file row dict to AvatarPersonalityDatasetExampleRead."""
    auto_label = str(row.get("auto_label") or "silver").strip().lower()
    heuristic_label = _avatar_normalize_llm_label(str(row.get("heuristic_label") or auto_label))
    llm_label = row.get("llm_label")
    return AvatarPersonalityDatasetExampleRead(
        example_id=int(row.get("example_id")),
        video_id=int(row.get("video_id") or 0),
        video_title=str(row.get("video_title") or ""),
        start_time=float(row.get("start_time") or 0.0),
        end_time=float(row.get("end_time") or 0.0),
        context_text=str(row.get("context_text") or ""),
        response_text=str(row.get("response_text") or ""),
        source_segment_ids=[int(s) for s in row.get("source_segment_ids", []) if str(s).isdigit() or isinstance(s, int)],
        source_segment_count=int(row.get("source_segment_count") or 0),
        context_turns=int(row.get("context_turns") or 0),
        response_word_count=int(row.get("response_word_count") or 0),
        context_word_count=int(row.get("context_word_count") or 0),
        quality_score=int(row.get("quality_score") or 0),
        completion_score=int(row.get("completion_score") or 0),
        context_score=int(row.get("context_score") or 0),
        style_score=int(row.get("style_score") or 0),
        substance_score=int(row.get("substance_score") or 0),
        cluster_id=int(row.get("cluster_id")) if row.get("cluster_id") is not None else None,
        cluster_size=int(row.get("cluster_size") or 0),
        diversity_score=int(row.get("diversity_score") or 0),
        duplicate_group_id=int(row.get("duplicate_group_id")) if row.get("duplicate_group_id") is not None else None,
        duplicate_group_size=int(row.get("duplicate_group_size") or 0),
        duplicate_similarity=float(row.get("duplicate_similarity") or 0.0),
        heuristic_label=heuristic_label,
        llm_label=_avatar_normalize_llm_label(str(llm_label)) if llm_label else None,
        llm_confidence=int(row.get("llm_confidence") or 0),
        llm_completion_score=int(row.get("llm_completion_score") or 0),
        llm_style_score=int(row.get("llm_style_score") or 0),
        llm_usefulness_score=int(row.get("llm_usefulness_score") or 0),
        llm_rationale=(str(row.get("llm_rationale") or "").strip() or None),
        llm_reasons=[str(r) for r in row.get("llm_reasons", [])],
        llm_model=(str(row.get("llm_model") or "").strip() or None),
        llm_judged_at=row.get("llm_judged_at"),
        auto_label=auto_label if auto_label in {"gold", "silver", "reject"} else "silver",
        manual_state=manual_state,
        state=state,
        reject_reasons=[str(r) for r in row.get("reject_reasons", [])],
    )


def _build_avatar_personality_dataset(
    session: Session,
    avatar: Avatar,
    speaker: Speaker,
    personality: AvatarPersonalityProfile,
) -> AvatarPersonalityDatasetRead:
    dataset_path, preview_path, metadata_path = _avatar_personality_dataset_paths(avatar)
    review_path, _ = _avatar_personality_review_paths(avatar)
    cluster_summary_path = _avatar_personality_cluster_summary_path(avatar)

    target_video_ids = [
        int(video_id)
        for video_id in session.exec(
            select(TranscriptSegment.video_id)
            .where(TranscriptSegment.speaker_id == speaker.id)
            .distinct()
        ).all()
        if video_id is not None
    ]
    if not target_video_ids:
        personality.dataset_path = None
        personality.dataset_example_count = 0
        personality.approved_example_count = 0
        personality.source_turn_count = 0
        personality.status = "needs_source"
        personality.last_built_at = datetime.now()
        personality.updated_at = personality.last_built_at
        session.add(personality)
        session.commit()
        return _load_avatar_personality_dataset(avatar, personality)

    rows = session.exec(
        select(
            TranscriptSegment.id,
            TranscriptSegment.video_id,
            TranscriptSegment.speaker_id,
            TranscriptSegment.start_time,
            TranscriptSegment.end_time,
            TranscriptSegment.text,
            Video.title,
            Speaker.name,
        )
        .join(Video, TranscriptSegment.video_id == Video.id)
        .join(Speaker, TranscriptSegment.speaker_id == Speaker.id, isouter=True)
        .where(
            Video.channel_id == avatar.channel_id,
            TranscriptSegment.video_id.in_(target_video_ids),
            TranscriptSegment.speaker_id.is_not(None),
        )
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_time, TranscriptSegment.id)
    ).all()

    turns_by_video: dict[int, list[dict[str, object]]] = {}
    merge_gap_seconds = 2.0
    source_turn_count = 0
    discarded_turn_count = 0

    for segment_id, video_id, segment_speaker_id, start_time, end_time, text_value, video_title, speaker_name in rows:
        cleaned_text = _clean_avatar_dataset_text(text_value)
        if not cleaned_text:
            continue

        turn_list = turns_by_video.setdefault(int(video_id), [])
        turn_speaker_id = int(segment_speaker_id) if segment_speaker_id is not None else None
        if (
            turn_list
            and int(turn_list[-1]["speaker_id"]) == int(turn_speaker_id or -1)
            and float(start_time or 0.0) - float(turn_list[-1]["end_time"] or 0.0) <= merge_gap_seconds
        ):
            turn_list[-1]["text"] = f"{turn_list[-1]['text']} {cleaned_text}".strip()
            turn_list[-1]["end_time"] = float(end_time or start_time or 0.0)
            turn_list[-1]["source_segment_ids"].append(int(segment_id))
        else:
            turn_list.append(
                {
                    "speaker_id": int(turn_speaker_id or 0),
                    "speaker_name": _clean_avatar_dataset_text(speaker_name) or f"Speaker {turn_speaker_id}",
                    "video_id": int(video_id),
                    "video_title": _clean_avatar_dataset_text(video_title) or f"Video {video_id}",
                    "start_time": float(start_time or 0.0),
                    "end_time": float(end_time or start_time or 0.0),
                    "text": cleaned_text,
                    "source_segment_ids": [int(segment_id)],
                }
            )

    speaker_style_weights = _avatar_build_speaker_style_weights(turns_by_video, int(speaker.id))

    legacy_prompt = _default_avatar_personality_prompt(avatar.name)
    current_prompt = str(personality.system_prompt or "").strip()
    if not current_prompt or current_prompt == legacy_prompt:
        current_prompt = _default_avatar_personality_prompt(speaker.name)
        personality.system_prompt = current_prompt
    system_prompt = current_prompt

    review_rows: list[dict[str, object]] = []
    example_count = 0

    for turns in turns_by_video.values():
        consumed_turn_indexes: set[int] = set()
        for idx, turn in enumerate(turns):
            if idx in consumed_turn_indexes:
                continue
            if int(turn["speaker_id"]) != int(speaker.id):
                continue
            source_turn_count += 1
            extended_response = _extend_avatar_response_turn(turns, idx, target_speaker_id=int(speaker.id))
            response_text = _clean_avatar_dataset_text(str(extended_response.get("text") or ""))
            consumed_turn_indexes.update(int(turn_index) for turn_index in extended_response.get("consumed_indexes", [idx]) if isinstance(turn_index, int))
            if len(response_text) < 24 or len(response_text.split()) < 5:
                discarded_turn_count += 1
                continue

            raw_context = [ctx for ctx in turns[max(0, idx - 4): idx]]
            if not raw_context or not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in raw_context):
                discarded_turn_count += 1
                continue

            # Select the last 3 context turns, but guarantee at least one
            # other-speaker turn survives the trim to avoid monologue context.
            trimmed = raw_context[-3:]
            if not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in trimmed):
                other_turns = [ctx for ctx in raw_context if int(ctx["speaker_id"]) != int(speaker.id)]
                trimmed = [other_turns[-1]] + [ctx for ctx in trimmed if int(ctx["speaker_id"]) == int(speaker.id)][:2]

            context_lines = [
                f"{str(ctx.get('speaker_name') or 'Speaker').strip()}: {str(ctx.get('text') or '').strip()}"
                for ctx in trimmed
                if str(ctx.get("text") or "").strip()
            ]
            if not context_lines or not any(int(ctx["speaker_id"]) != int(speaker.id) for ctx in trimmed):
                discarded_turn_count += 1
                continue

            user_text = (
                f"Podcast episode: {str(turn.get('video_title') or '').strip()}\n"
                f"Conversation context:\n" + "\n".join(context_lines)
            ).strip()

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response_text},
                ],
                "metadata": {
                    "avatar_id": int(avatar.id),
                    "speaker_id": int(speaker.id),
                    "video_id": int(turn["video_id"]),
                    "video_title": str(turn.get("video_title") or ""),
                    "start_time": float(turn["start_time"]),
                    "end_time": float(extended_response.get("end_time") or turn["end_time"]),
                    "context_turns": len(context_lines),
                    "source_segment_ids": [int(segment_id) for segment_id in extended_response.get("source_segment_ids", [])],
                },
            }
            review_rows.append(
                _score_avatar_personality_example(
                    {
                        "example_id": int(example_count),
                        "video_id": int(turn["video_id"]),
                        "video_title": str(turn.get("video_title") or ""),
                        "start_time": float(turn["start_time"]),
                        "end_time": float(extended_response.get("end_time") or turn["end_time"]),
                        "context_text": user_text,
                        "response_text": response_text,
                        "source_segment_ids": [int(segment_id) for segment_id in extended_response.get("source_segment_ids", [])],
                        "context_turns": len(context_lines),
                        "messages": payload["messages"],
                        "metadata": payload["metadata"],
                    },
                    speaker_style_weights=speaker_style_weights,
                )
            )
            example_count += 1

    review_rows = _apply_avatar_duplicate_rejects(review_rows)
    review_rows, cluster_summary = _avatar_assign_embedding_clusters(review_rows)
    review_lines = [json.dumps(row, ensure_ascii=False) for row in review_rows]
    review_path.write_text("\n".join(review_lines), encoding="utf-8")
    cluster_summary_path.write_text(json.dumps(cluster_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_avatar_personality_state_map(avatar, {})
    dataset_path.write_text("", encoding="utf-8")
    preview_path.write_text("[]", encoding="utf-8")

    generated_at = datetime.now()
    label_counts = Counter(str(row.get("auto_label") or "silver") for row in review_rows)
    metadata = {
        "avatar_id": int(avatar.id),
        "speaker_id": int(speaker.id),
        "speaker_name": speaker.name,
        "example_count": int(example_count),
        "approved_example_count": int(example_count - int(label_counts.get("reject", 0))),
        "rejected_example_count": int(label_counts.get("reject", 0)),
        "gold_example_count": int(label_counts.get("gold", 0)),
        "silver_example_count": int(label_counts.get("silver", 0)),
        "auto_reject_count": int(label_counts.get("reject", 0)),
        "needs_review_count": int(label_counts.get("silver", 0)),
        "duplicate_example_count": int(cluster_summary.get("duplicate_example_count") or 0),
        "cluster_count": int(cluster_summary.get("cluster_count") or 0),
        "hotspot_cluster_count": int(cluster_summary.get("hotspot_cluster_count") or 0),
        "llm_judged_count": 0,
        "llm_promoted_count": 0,
        "llm_rejected_count": 0,
        "source_turn_count": int(source_turn_count),
        "discarded_turn_count": int(discarded_turn_count),
        "generated_at": generated_at.isoformat(),
        "dataset_format": "sharegpt_messages_v1",
        "base_model_id": personality.base_model_id,
        "system_prompt": system_prompt,
        "embedding_model": str(cluster_summary.get("embedding_model") or "hashing_ngram_v1"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    personality.dataset_path = str(dataset_path)
    personality.dataset_example_count = int(example_count)
    personality.approved_example_count = int(example_count - int(label_counts.get("reject", 0)))
    personality.source_turn_count = int(source_turn_count)
    personality.status = "dataset_ready" if example_count > 0 else "needs_source"
    personality.last_built_at = generated_at
    personality.updated_at = generated_at
    session.add(personality)
    session.commit()
    session.refresh(personality)

    refreshed_dataset = _refresh_avatar_personality_dataset_exports(
        avatar,
        personality,
        total_example_count=example_count,
    )
    refreshed_dataset.generated_at = generated_at
    refreshed_dataset.discarded_turn_count = int(discarded_turn_count)
    refreshed_dataset.source_turn_count = int(source_turn_count)
    return refreshed_dataset


def _serialize_avatar(avatar: Avatar) -> AvatarRead:
    return AvatarRead(
        id=int(avatar.id),
        channel_id=int(avatar.channel_id),
        speaker_id=int(avatar.speaker_id),
        name=str(avatar.name or "").strip(),
        status=str(avatar.status or "draft").strip() or "draft",
        description=(str(avatar.description).strip() if avatar.description is not None else None),
        created_at=avatar.created_at,
        updated_at=avatar.updated_at,
    )


def _ensure_avatar_profiles(
    session: Session,
    avatar: Avatar,
    *,
    speaker_name: str | None = None,
) -> tuple[AvatarPersonalityProfile, AvatarAppearanceProfile, AvatarVoiceProfile]:
    personality = session.exec(
        select(AvatarPersonalityProfile).where(AvatarPersonalityProfile.avatar_id == avatar.id)
    ).first()
    if not personality:
        personality = AvatarPersonalityProfile(
            avatar_id=int(avatar.id),
            system_prompt=_default_avatar_personality_prompt(speaker_name or avatar.name),
            base_model_id="Qwen/Qwen3-14B",
        )
        session.add(personality)

    appearance = session.exec(
        select(AvatarAppearanceProfile).where(AvatarAppearanceProfile.avatar_id == avatar.id)
    ).first()
    if not appearance:
        appearance = AvatarAppearanceProfile(avatar_id=int(avatar.id))
        session.add(appearance)

    voice = session.exec(
        select(AvatarVoiceProfile).where(AvatarVoiceProfile.avatar_id == avatar.id)
    ).first()
    if not voice:
        voice = AvatarVoiceProfile(avatar_id=int(avatar.id), provider="fish-speech")
        session.add(voice)

    session.flush()
    return personality, appearance, voice


def _build_avatar_workbench(session: Session, avatar: Avatar) -> AvatarWorkbenchRead:
    speaker = session.get(Speaker, avatar.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Source speaker not found for this avatar")

    personality, appearance, voice = _ensure_avatar_profiles(session, avatar, speaker_name=speaker.name)
    legacy_prompt = _default_avatar_personality_prompt(avatar.name)
    normalized_prompt = _default_avatar_personality_prompt(speaker.name)
    if str(personality.system_prompt or "").strip() == legacy_prompt:
        personality.system_prompt = normalized_prompt

    total_speaking_time = float(
        session.exec(
            select(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time))
            .where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0.0
    )
    embedding_count = int(
        session.exec(
            select(func.count(SpeakerEmbedding.id)).where(SpeakerEmbedding.speaker_id == speaker.id)
        ).first()
        or 0
    )
    appearance_count = int(
        session.exec(
            select(func.count(func.distinct(TranscriptSegment.video_id))).where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0
    )
    source_turn_count = int(
        session.exec(
            select(func.count(TranscriptSegment.id)).where(TranscriptSegment.speaker_id == speaker.id)
        ).first()
        or 0
    )
    eligible_voice_clip_count = int(
        session.exec(
            select(func.count(TranscriptSegment.id)).where(
                TranscriptSegment.speaker_id == speaker.id,
                (TranscriptSegment.end_time - TranscriptSegment.start_time) >= 2.0,
            )
        ).first()
        or 0
    )

    personality.source_turn_count = max(int(personality.source_turn_count or 0), source_turn_count)
    if not str(personality.status or "").strip() or str(personality.status) == "draft":
        personality.status = "ready_to_curate" if source_turn_count > 0 else "needs_source"
    appearance.source_image_count = max(int(appearance.source_image_count or 0), 1 if speaker.thumbnail_path else 0)
    appearance.approved_image_count = max(int(appearance.approved_image_count or 0), 1 if appearance.primary_image_path or speaker.thumbnail_path else 0)
    if not str(appearance.status or "").strip() or str(appearance.status) == "draft":
        appearance.status = "ready_to_curate" if (appearance.source_image_count or 0) > 0 else "needs_source"
    voice.source_clip_count = max(int(voice.source_clip_count or 0), max(eligible_voice_clip_count, embedding_count))
    voice.approved_clip_count = max(int(voice.approved_clip_count or 0), embedding_count)
    if not str(voice.status or "").strip() or str(voice.status) == "draft":
        voice.status = "ready_to_curate" if (voice.source_clip_count or 0) > 0 else "needs_source"

    if not appearance.primary_image_path and speaker.thumbnail_path:
        appearance.primary_image_path = speaker.thumbnail_path

    session.add(personality)
    session.add(appearance)
    session.add(voice)
    session.commit()
    session.refresh(avatar)
    session.refresh(personality)
    session.refresh(appearance)
    session.refresh(voice)

    avatar_dir = _avatar_artifacts_dir(avatar)

    return AvatarWorkbenchRead(
        avatar=_serialize_avatar(avatar),
        speaker=AvatarWorkbenchSpeakerRead(
            id=int(speaker.id),
            channel_id=int(speaker.channel_id),
            name=speaker.name,
            thumbnail_path=speaker.thumbnail_path,
            total_speaking_time=round(total_speaking_time, 1),
            embedding_count=embedding_count,
            appearance_count=appearance_count,
        ),
        personality=AvatarSectionSummaryRead(
            status=str(personality.status or "draft"),
            source_count=source_turn_count,
            approved_count=int(personality.approved_example_count or 0),
            artifact_ready=bool(personality.dataset_path or personality.lora_adapter_path),
            summary=(
                f"Dataset built with {int(personality.dataset_example_count or 0)} training examples from {source_turn_count} source turns."
                if personality.dataset_path and int(personality.dataset_example_count or 0) > 0
                else (
                    f"{source_turn_count} source turns from diarized conversations are available for dataset curation."
                    if source_turn_count > 0
                    else "No transcript turns are currently assigned to this speaker."
                )
            ),
            artifact_path=personality.dataset_path or personality.lora_adapter_path,
            last_built_at=personality.last_built_at,
        ),
        appearance=AvatarSectionSummaryRead(
            status=str(appearance.status or "draft"),
            source_count=int(appearance.source_image_count or 0),
            approved_count=int(appearance.approved_image_count or 0),
            artifact_ready=bool(appearance.appearance_lora_path),
            summary=(
                "A primary portrait is already available."
                if appearance.primary_image_path
                else "No approved portrait is attached yet."
            ),
            artifact_path=appearance.appearance_lora_path,
            last_built_at=appearance.last_built_at,
        ),
        voice=AvatarSectionSummaryRead(
            status=str(voice.status or "draft"),
            source_count=int(voice.source_clip_count or 0),
            approved_count=int(voice.approved_clip_count or 0),
            artifact_ready=bool(voice.embedding_path),
            summary=(
                f"{embedding_count} existing diarization voice profiles are available as seed material."
                if embedding_count > 0
                else "No reusable voice profiles have been approved yet."
            ),
            artifact_path=voice.embedding_path,
            last_built_at=voice.last_built_at,
        ),
        runtime_status="not_ready",
        suggested_base_model=str(personality.base_model_id or "Qwen/Qwen3-14B"),
        artifacts_dir=str(avatar_dir),
    )
