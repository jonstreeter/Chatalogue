from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now().isoformat()


def write_status(path: Path, patch: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "avatar_id": 0,
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
        "updated_at": iso_now(),
        "finished_at": None,
        "error": None,
    }
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload.update(raw)
        except Exception:
            pass
    payload.update(patch)
    payload["updated_at"] = iso_now()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_prompt(tokenizer, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def tokenize_messages(tokenizer, messages: list[dict[str, Any]], *, max_length: int) -> dict[str, list[int]] | None:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded_full = tokenizer(full_text, add_special_tokens=False)
            input_ids = list(encoded_full.get("input_ids") or [])
            if input_ids:
                labels = [-100] * len(input_ids)
                prefix_messages: list[dict[str, Any]] = []
                prefix_token_count = 0
                for message in messages:
                    prefix_messages.append(message)
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    prefix_ids = list(
                        (tokenizer(prefix_text, add_special_tokens=False).get("input_ids") or [])
                    )
                    role = str(message.get("role") or "").strip().lower()
                    if role == "assistant":
                        end = min(len(prefix_ids), len(input_ids))
                        for index in range(prefix_token_count, end):
                            labels[index] = input_ids[index]
                    prefix_token_count = len(prefix_ids)

                if max_length > 0 and len(input_ids) > max_length:
                    input_ids = input_ids[-max_length:]
                    labels = labels[-max_length:]
                if any(label != -100 for label in labels):
                    return {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": labels,
                    }
        except Exception:
            pass

    prompt = build_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, truncation=True, max_length=max_length)
    input_ids = list(encoded.get("input_ids") or [])
    attention_mask = list(encoded.get("attention_mask") or [])
    if not input_ids:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),
    }


def choose_target_modules(model) -> list[str]:
    preferred = {"q_proj", "v_proj", "o_proj", "gate_proj"}
    found: set[str] = set()
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "linear" not in cls_name:
            continue
        leaf = name.split(".")[-1]
        if leaf in preferred:
            found.add(leaf)
    if found:
        return sorted(found)
    fallback: set[str] = set()
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "linear" not in cls_name:
            continue
        leaf = name.split(".")[-1]
        if leaf != "lm_head":
            fallback.add(leaf)
    return sorted(fallback)[:16] or ["q_proj", "v_proj"]


def resolve_training_precision(torch) -> tuple[object, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    bf16_supported = False
    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False
    if bf16_supported:
        return torch.bfloat16, True, False
    return torch.float16, False, True


def configure_cuda_allocator() -> None:
    raw = str(os.getenv("PYTORCH_CUDA_ALLOC_CONF") or "").strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    existing = {part.split(":", 1)[0].strip().lower() for part in parts if ":" in part}
    defaults = (
        ("max_split_size_mb", str(int((os.getenv("CUDA_ALLOC_MAX_SPLIT_MB") or "128").strip() or "128"))),
        ("garbage_collection_threshold", str(float((os.getenv("CUDA_ALLOC_GC_THRESHOLD") or "0.8").strip() or "0.8"))),
        ("expandable_segments", "True"),
    )
    for key, value in defaults:
        if key.lower() not in existing:
            parts.append(f"{key}:{value}")
    if parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc or "").strip().lower()
    return any(
        token in message
        for token in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudaerrormemoryallocation",
            "out of memory",
            "cublas_status_alloc_failed",
            "hip out of memory",
        )
    )


def next_sequence_length(current: int) -> int | None:
    current = max(256, int(current or 256))
    for candidate in (1536, 1024, 768, 512, 384, 256):
        if current > candidate:
            return candidate
    return None


def next_lora_rank(current: int) -> int | None:
    current = max(4, int(current or 4))
    for candidate in (8, 4):
        if current > candidate:
            return candidate
    return None


def release_cuda_memory(torch) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def oom_retry_plan(settings: dict[str, Any], *, has_validation: bool) -> tuple[dict[str, Any] | None, str | None]:
    batch_size = max(1, int(settings.get("per_device_batch_size") or 1))
    if batch_size > 1:
        next_batch_size = max(1, batch_size // 2)
        updated = dict(settings)
        updated["per_device_batch_size"] = next_batch_size
        return updated, f"reducing per-device batch size from {batch_size} to {next_batch_size}"

    if bool(settings.get("run_evaluation", True)) and has_validation:
        updated = dict(settings)
        updated["run_evaluation"] = False
        return updated, "disabling validation passes"

    max_seq_length = max(256, int(settings.get("max_seq_length") or 256))
    next_max_seq_length = next_sequence_length(max_seq_length)
    if next_max_seq_length is not None:
        updated = dict(settings)
        updated["max_seq_length"] = next_max_seq_length
        return updated, f"reducing max sequence length from {max_seq_length} to {next_max_seq_length}"

    lora_rank = max(4, int(settings.get("lora_rank") or 4))
    next_rank = next_lora_rank(lora_rank)
    if next_rank is not None:
        updated = dict(settings)
        updated["lora_rank"] = next_rank
        return updated, f"reducing LoRA rank from {lora_rank} to {next_rank}"

    return None, None


def resolve_snapshot_interval_steps(requested_steps: int, total_steps: int) -> int:
    total = max(0, int(total_steps or 0))
    requested = max(0, int(requested_steps or 0))
    if total <= 0:
        return requested

    if total <= 150:
        auto_interval = 25
    elif total <= 300:
        auto_interval = 50
    elif total <= 600:
        auto_interval = 100
    elif total <= 1000:
        auto_interval = 200
    else:
        auto_interval = 250

    if requested > 0:
        if requested >= total:
            return min(auto_interval, max(1, total))
        return requested

    return min(auto_interval, max(1, total))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--stop-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--snapshot-interval-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--training-mode", choices=["standard", "memory_optimized"], default="memory_optimized")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    status_path = Path(args.status_path).resolve()
    stop_path = Path(args.stop_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        write_status(status_path, {"status": "failed", "active": False, "error": f"Failed to read manifest: {exc}", "finished_at": iso_now()})
        return 1

    base_model_id = str(manifest.get("base_model_id") or "").strip()
    train_path = Path(str(manifest.get("train_dataset_path") or "")).resolve()
    val_path = Path(str(manifest.get("validation_dataset_path") or "")).resolve()
    avatar_id = int(manifest.get("avatar_id") or 0)
    if not base_model_id or not train_path.exists():
        write_status(status_path, {"avatar_id": avatar_id, "status": "failed", "active": False, "error": "Training manifest is incomplete.", "finished_at": iso_now()})
        return 1

    write_status(
        status_path,
        {
            "avatar_id": avatar_id,
            "status": "running",
            "active": True,
            "stop_requested": stop_path.exists(),
            "process_id": int(os.getpid()),
            "base_model_id": base_model_id,
            "training_mode": str(args.training_mode or "memory_optimized"),
            "output_dir": str(output_dir),
            "current_stage": "loading_model",
            "snapshot_interval_steps": max(0, int(args.snapshot_interval_steps or 0)),
            "train_examples": int(manifest.get("train_examples") or 0),
            "validation_examples": int(manifest.get("validation_examples") or 0),
            "started_at": iso_now(),
            "message": "Loading tokenizer and base model",
            "error": None,
        },
    )

    configure_cuda_allocator()

    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    training_mode = str(args.training_mode or "memory_optimized")
    snapshot_state: dict[str, Any] = {
        "interval_steps": max(0, int(args.snapshot_interval_steps or 0)),
        "latest_train_loss": None,
        "latest_eval_loss": None,
        "snapshots": [],
        "saved_keys": set(),
    }
    if torch.cuda.is_available() and float(args.cuda_memory_fraction or 0.0) > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(
                max(0.4, min(float(args.cuda_memory_fraction), 0.95)),
                device=0,
            )
        except Exception:
            pass

    class JsonlChatDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
            self.items: list[dict[str, Any]] = []
            for row in rows:
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    continue
                tokenized = tokenize_messages(tokenizer, messages, max_length=max_length)
                if not tokenized or not tokenized.get("input_ids"):
                    continue
                self.items.append(tokenized)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.items[index]

    class SimpleCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            max_len = max(len(item["input_ids"]) for item in features)
            pad_id = self.tokenizer.pad_token_id
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for item in features:
                pad = max_len - len(item["input_ids"])
                batch["input_ids"].append(item["input_ids"] + ([pad_id] * pad))
                batch["attention_mask"].append(item["attention_mask"] + ([0] * pad))
                batch["labels"].append(item["labels"] + ([-100] * pad))
            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    def save_snapshot(
        *,
        model,
        tokenizer,
        kind: str,
        step: int,
        epoch: float,
        label: str | None = None,
        selected: bool = False,
    ) -> dict[str, Any] | None:
        snapshot_kind = "final" if str(kind) == "final" else ("epoch" if str(kind) == "epoch" else "step")
        snapshot_step = max(0, int(step or 0))
        snapshot_epoch = round(float(epoch or 0.0), 3)
        snapshot_key = f"{snapshot_kind}:{snapshot_step}:{snapshot_epoch:.3f}"
        if snapshot_key in snapshot_state["saved_keys"]:
            return None
        if snapshot_kind == "final":
            snapshot_dir = output_dir / "adapter"
            snapshot_label = label or "Final Adapter"
        elif snapshot_kind == "epoch":
            epoch_index = max(1, int(round(snapshot_epoch or 0.0)))
            snapshot_dir = output_dir / "snapshots" / f"epoch-{epoch_index:02d}-step-{snapshot_step:06d}"
            snapshot_label = label or f"Epoch {epoch_index}"
        else:
            snapshot_dir = output_dir / "snapshots" / f"step-{snapshot_step:06d}"
            snapshot_label = label or f"Step {snapshot_step}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(snapshot_dir))
        tokenizer.save_pretrained(str(snapshot_dir))
        entry = {
            "label": snapshot_label,
            "kind": snapshot_kind,
            "adapter_path": str(snapshot_dir),
            "step": snapshot_step,
            "epoch": snapshot_epoch,
            "train_loss": (
                float(snapshot_state["latest_train_loss"])
                if snapshot_state["latest_train_loss"] is not None
                else None
            ),
            "eval_loss": (
                float(snapshot_state["latest_eval_loss"])
                if snapshot_state["latest_eval_loss"] is not None
                else None
            ),
            "created_at": iso_now(),
            "selected": bool(selected),
        }
        (snapshot_dir / "snapshot_info.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        snapshot_state["saved_keys"].add(snapshot_key)
        snapshots = [
            item
            for item in snapshot_state["snapshots"]
            if str(item.get("adapter_path") or "").strip() != str(snapshot_dir)
        ]
        if selected:
            for item in snapshots:
                item["selected"] = False
        snapshots.append(entry)
        snapshot_state["snapshots"] = snapshots
        patch: dict[str, Any] = {"snapshots": snapshot_state["snapshots"]}
        if selected:
            patch["adapter_path"] = str(snapshot_dir)
        write_status(status_path, patch)
        return entry

    class StatusCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            write_status(
                status_path,
                {
                    "current_stage": "training",
                    "max_steps": int(state.max_steps or 0),
                    "snapshot_interval_steps": int(snapshot_state["interval_steps"] or 0),
                    "message": "Training started",
                },
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            latest_loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            epoch = float(logs.get("epoch") or state.epoch or 0.0)
            if latest_loss is not None:
                snapshot_state["latest_train_loss"] = float(latest_loss)
            if eval_loss is not None:
                snapshot_state["latest_eval_loss"] = float(eval_loss)
            write_status(
                status_path,
                {
                    "status": "running",
                    "active": True,
                    "process_id": int(os.getpid()),
                    "training_mode": training_mode,
                    "epoch": round(epoch, 3),
                    "step": int(state.global_step or 0),
                    "max_steps": int(state.max_steps or 0),
                    "latest_loss": float(latest_loss) if latest_loss is not None else None,
                    "snapshot_interval_steps": int(snapshot_state["interval_steps"] or 0),
                    "message": f"epoch {epoch:.2f} step {int(state.global_step or 0)}",
                    "stop_requested": stop_path.exists(),
                },
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                snapshot_state["latest_eval_loss"] = float(eval_loss)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if stop_path.exists():
                write_status(status_path, {"status": "stopping", "active": True, "stop_requested": True, "message": "Stop requested; waiting for trainer to halt"})
                control.should_training_stop = True
                return control
            interval_steps = max(0, int(snapshot_state["interval_steps"] or 0))
            current_step = int(state.global_step or 0)
            if interval_steps > 0 and current_step > 0 and current_step % interval_steps == 0:
                callback_model = kwargs.get("model") or model
                callback_tokenizer = kwargs.get("tokenizer") or tokenizer
                if callback_model is not None and callback_tokenizer is not None:
                    save_snapshot(
                        model=callback_model,
                        tokenizer=callback_tokenizer,
                        kind="step",
                        step=current_step,
                        epoch=float(state.epoch or 0.0),
                    )
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            callback_model = kwargs.get("model") or model
            callback_tokenizer = kwargs.get("tokenizer") or tokenizer
            if callback_model is not None and callback_tokenizer is not None:
                save_snapshot(
                    model=callback_model,
                    tokenizer=callback_tokenizer,
                    kind="epoch",
                    step=int(state.global_step or 0),
                    epoch=float(state.epoch or 0.0),
                )
            return control

    effective_settings: dict[str, Any] = {
        "lora_rank": max(4, int(args.lora_rank or 16)),
        "max_seq_length": max(256, int(args.max_seq_length or 1024)),
        "per_device_batch_size": max(1, int(args.per_device_batch_size or 1)),
        "gradient_accumulation_steps": max(1, int(args.gradient_accumulation_steps or 4)),
        "run_evaluation": True,
    }

    try:
        train_rows = read_jsonl(train_path)
        val_rows = read_jsonl(val_path)
        if not train_rows:
            raise ValueError("Prepared training dataset is empty")

        attempt = 1
        while True:
            tokenizer = None
            model = None
            train_dataset = None
            val_dataset = None
            trainer = None
            attempt_stage = "loading_model"
            try:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                write_status(
                    status_path,
                    {
                        "status": "running",
                        "active": True,
                        "process_id": int(os.getpid()),
                        "training_mode": training_mode,
                        "current_stage": "loading_model",
                        "message": (
                            f"Loading tokenizer and base model "
                            f"(attempt {attempt}, seq {effective_settings['max_seq_length']}, "
                            f"rank {effective_settings['lora_rank']}, batch {effective_settings['per_device_batch_size']})"
                        ),
                        "error": None,
                    },
                )

                torch_dtype, use_bf16, use_fp16 = resolve_training_precision(torch)
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
                if training_mode == "memory_optimized":
                    if not torch.cuda.is_available():
                        raise RuntimeError("Memory-optimized QLoRA mode requires CUDA.")
                    try:
                        try:
                            from transformers import BitsAndBytesConfig
                        except Exception:
                            from transformers.utils.quantization_config import BitsAndBytesConfig
                    except Exception as exc:
                        raise RuntimeError(f"BitsAndBytesConfig is unavailable: {exc}") from exc
                    # Keep the quantized model fully resident on the primary GPU.
                    # If it does not fit inside the reserved VRAM budget, fail fast
                    # and let the backend choose a smaller training profile/model
                    # instead of spilling into shared/system memory.
                    model_kwargs["device_map"] = {"": 0}
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
                model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
                if training_mode == "standard" and torch.cuda.is_available():
                    model = model.to("cuda")
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
                if training_mode == "memory_optimized":
                    model = prepare_model_for_kbit_training(model)

                target_modules = choose_target_modules(model)
                lora_rank = max(4, int(effective_settings["lora_rank"]))
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    lora_dropout=0.10,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                )
                model = get_peft_model(model, lora_config)

                attempt_stage = "loading_dataset"
                write_status(
                    status_path,
                    {
                        "current_stage": "loading_dataset",
                        "message": f"Tokenizing training package (attempt {attempt})",
                        "error": None,
                    },
                )
                max_seq_length = max(256, int(effective_settings["max_seq_length"]))
                train_dataset = JsonlChatDataset(train_rows, tokenizer, max_length=max_seq_length)
                val_dataset = JsonlChatDataset(val_rows, tokenizer, max_length=max_seq_length)
                if len(train_dataset) == 0:
                    raise ValueError("Prepared training dataset is empty")

                per_device_batch_size = max(1, int(effective_settings["per_device_batch_size"]))
                gradient_accumulation_steps = max(1, int(effective_settings["gradient_accumulation_steps"]))
                optimizer_steps_per_epoch = max(
                    1,
                    math.ceil(len(train_dataset) / max(1, per_device_batch_size * gradient_accumulation_steps)),
                )
                total_steps = optimizer_steps_per_epoch * max(1, int(args.epochs))
                snapshot_state["interval_steps"] = resolve_snapshot_interval_steps(
                    int(args.snapshot_interval_steps or 0),
                    int(total_steps),
                )
                attempt_stage = "training"
                write_status(
                    status_path,
                    {
                        "current_stage": "training",
                        "max_steps": int(total_steps),
                        "snapshot_interval_steps": int(snapshot_state["interval_steps"] or 0),
                        "message": (
                            f"Launching Trainer (attempt {attempt}, "
                            f"snapshots every {int(snapshot_state['interval_steps'] or 0) or 'epoch'} steps)"
                        ),
                        "error": None,
                    },
                )

                run_evaluation = bool(effective_settings.get("run_evaluation", True)) and len(val_dataset) > 0
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    overwrite_output_dir=True,
                    num_train_epochs=max(1, int(args.epochs)),
                    learning_rate=float(args.learning_rate),
                    per_device_train_batch_size=per_device_batch_size,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_ratio=max(0.0, min(0.2, float(args.warmup_ratio))),
                    weight_decay=0.05,
                    logging_steps=1,
                    logging_first_step=True,
                    save_strategy="epoch",
                    eval_strategy="epoch" if run_evaluation else "no",
                    save_total_limit=1,
                    load_best_model_at_end=run_evaluation,
                    metric_for_best_model="eval_loss" if run_evaluation else None,
                    greater_is_better=False if run_evaluation else None,
                    bf16=use_bf16,
                    fp16=use_fp16,
                    optim="paged_adamw_8bit" if training_mode == "memory_optimized" else "adamw_torch",
                    report_to=[],
                    remove_unused_columns=False,
                    dataloader_pin_memory=torch.cuda.is_available(),
                    seed=42,
                    data_seed=42,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset if run_evaluation else None,
                    data_collator=SimpleCollator(tokenizer),
                    callbacks=[StatusCallback()] + ([EarlyStoppingCallback(early_stopping_patience=3)] if run_evaluation else []),
                )
                trainer.train()
                final_snapshot = save_snapshot(
                    model=model,
                    tokenizer=tokenizer,
                    kind="final",
                    step=int(getattr(trainer.state, "global_step", 0) or 0),
                    epoch=float(getattr(trainer.state, "epoch", 0.0) or 0.0),
                    selected=True,
                )
                adapter_dir = Path(str((final_snapshot or {}).get("adapter_path") or (output_dir / "adapter")))

                final_status = "stopped" if stop_path.exists() else "completed"
                write_status(
                    status_path,
                    {
                        "status": final_status,
                        "active": False,
                        "stop_requested": stop_path.exists(),
                        "process_id": int(os.getpid()),
                        "training_mode": training_mode,
                        "adapter_path": str(adapter_dir),
                        "current_stage": "completed",
                        "snapshot_interval_steps": int(snapshot_state["interval_steps"] or 0),
                        "snapshots": snapshot_state["snapshots"],
                        "finished_at": iso_now(),
                        "message": "Training finished" if final_status == "completed" else "Training stopped",
                        "error": None,
                    },
                )
                return 0
            except Exception as exc:
                if stop_path.exists():
                    raise
                if not is_cuda_oom(exc):
                    raise

                next_settings, retry_reason = oom_retry_plan(effective_settings, has_validation=bool(val_rows))
                trainer = None
                train_dataset = None
                val_dataset = None
                model = None
                tokenizer = None
                snapshot_state["snapshots"] = []
                snapshot_state["saved_keys"] = set()
                snapshot_state["latest_train_loss"] = None
                snapshot_state["latest_eval_loss"] = None
                try:
                    shutil.rmtree(output_dir / "snapshots", ignore_errors=True)
                except Exception:
                    pass
                try:
                    shutil.rmtree(output_dir / "adapter", ignore_errors=True)
                except Exception:
                    pass
                release_cuda_memory(torch)
                if next_settings is None or retry_reason is None:
                    guidance = (
                        "The selected base model or current training settings exceed available VRAM. "
                        "Try a smaller base model, keep memory-optimized mode enabled, or reduce max sequence length."
                    )
                    raise RuntimeError(
                        f"CUDA out of memory during {attempt_stage}. {guidance} Original error: {exc}"
                    ) from exc

                effective_settings = next_settings
                attempt += 1
                write_status(
                    status_path,
                    {
                        "status": "running",
                        "active": True,
                        "process_id": int(os.getpid()),
                        "training_mode": training_mode,
                        "current_stage": "retrying",
                        "snapshots": [],
                        "message": f"CUDA OOM during {attempt_stage}; retrying with {retry_reason}.",
                        "error": None,
                    },
                )
                continue
            finally:
                trainer = None
                train_dataset = None
                val_dataset = None
                model = None
                tokenizer = None
                release_cuda_memory(torch)
    except Exception as exc:
        write_status(
            status_path,
            {
                "status": "failed",
                "active": False,
                "process_id": int(os.getpid()),
                "training_mode": str(args.training_mode or "memory_optimized"),
                "current_stage": "failed",
                "finished_at": iso_now(),
                "error": str(exc),
                "message": "Training failed",
            },
        )
        return 1
    finally:
        try:
            stop_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
