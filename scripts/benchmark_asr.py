"""Isolated ASR benchmark; application mode uses only an in-memory database.

Run each engine on the same WAV/FLAC, in separate processes. Output includes
raw word timestamps for review. Model downloads must finish before comparing
load times. App mode calls the project's transcription phase, including routing,
chunking and fallback. It excludes queue waiting, diarization and other jobs.

Examples (PowerShell, from repo root):
  backend/.venv/Scripts/python scripts/benchmark_asr.py app clip.wav --output backend/data/asr-benchmark/app.json
  python scripts/benchmark_asr.py phonon clip.wav --output backend/data/asr-benchmark/phonon.json --command .venv-phonon/Scripts/fermion.exe

For CUDA Phonon, use --command docker run --rm --gpus all -v
<absolute-audio-directory>:/audio -v phonon-cache:/home/phonon/.cache
ghcr.io/fermionresearch/phonon-cuda:1.0.8, with --phonon-cli cuda and
--engine-audio /audio/clip.wav before --command.
Do not compare CPU and CUDA runs as if they used the same hardware.
"""

import argparse
import json
from pathlib import Path
import subprocess
import os
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack


def benchmark_application(audio, duration, setup_only=False, parakeet_checkpoint=None, app_engine=None, phonon_env=None, phonon_runtime=None, phonon_handoff=False):
    """Run the real transcription phase; isolate DB before any app import."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))
    from dotenv import load_dotenv

    load_dotenv(root / "backend/.env")
    if phonon_env:
        load_dotenv(phonon_env, override=True)
    if app_engine:
        os.environ["TRANSCRIPTION_ENGINE"] = app_engine
    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["DATABASE_URL"] = "sqlite://"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from src.db import database
    from src.services.ingestion import IngestionService, runtime
    from sqlmodel import SQLModel

    if str(database.engine.url) != "sqlite://" or runtime.engine is not database.engine:
        raise RuntimeError("Database isolation failed")
    SQLModel.metadata.create_all(database.engine)
    settings = {key: os.getenv(key) for key in (
        "TRANSCRIPTION_ENGINE", "PARAKEET_MODEL", "PARAKEET_BATCH_SIZE",
        "PARAKEET_UNLOAD_AFTER_TRANSCRIBE", "PIPELINE_EXECUTION_MODE",
        "WHISPER_BACKEND", "TRANSCRIPTION_MODEL", "TRANSCRIPTION_COMPUTE_TYPE",
        "TRANSCRIPTION_BATCHED", "TRANSCRIPTION_BEAM_SIZE", "MULTILINGUAL_ROUTING_ENABLED")}
    with tempfile.TemporaryDirectory(prefix="app-", dir=root / "backend/data/asr-benchmark") as scratch, ExitStack() as cleanup:
        scratch = Path(scratch)
        names = ("DATA_DIR", "TEMP_DIR", "PYTHON_TEMP_DIR", "RUNTIME_DIR",
                 "AUDIO_DIR", "MANUAL_MEDIA_DIR", "EXPORT_DIR", "PHONON_RUNTIME_DIR")
        # Mixins import directory constants by value; redirect every loaded copy.
        for name, module in list(sys.modules.items()):
            if name.startswith("src.services.ingestion") and module:
                for key in names:
                    if hasattr(module, key):
                        target = scratch / key.lower()
                        target.mkdir(exist_ok=True)
                        setattr(module, key, target)
                for key in ("HEARTBEAT_FILE", "CUDA_RESTART_STATE_FILE"):
                    if hasattr(module, key):
                        setattr(module, key, scratch / key.lower())
        if phonon_runtime:
            runtime.PHONON_RUNTIME_DIR = phonon_runtime.resolve(strict=True)
        runtime.configure_python_temp_dir()
        service = IngestionService()
        cleanup.callback(service.shutdown_phonon)
        if parakeet_checkpoint:
            # Resolve the already-downloaded artifact only. All app loading,
            # device placement, decoding, chunking and fallback methods stay intact.
            service._ensure_ctranslate2_pkg_resources()
            from nemo.collections.asr.models import ASRModel

            checkpoint = parakeet_checkpoint.resolve(strict=True)
            original_resolver = ASRModel._get_hf_hub_pretrained_model_info

            def local_resolver(cls, model_name, refresh_cache=False):
                if model_name == (settings["PARAKEET_MODEL"] or "nvidia/parakeet-tdt-0.6b-v2"):
                    return cls, str(checkpoint)
                return original_resolver(model_name, refresh_cache=refresh_cache)

            ASRModel._get_hf_hub_pretrained_model_info = classmethod(local_resolver)
        provenance = {}
        # No job exists in this harness; retain the normal job metadata in memory.
        service._upsert_job_payload_fields = lambda job_id, fields: provenance.update(fields)
        result = {"settings": settings, "database": "in-memory SQLite",
                  "scope": "Actual _process_transcribe_phase; synthetic video metadata; no production jobs",
                  "offline": True, "runs": [],
                  "local_parakeet_checkpoint": str(parakeet_checkpoint) if parakeet_checkpoint else None,
                  "comparison_note": "Offline cache misses can force fallback. This is not proof of a live app's engine.",
                  "vram_note": "PyTorch allocation excludes CTranslate2; zero does not mean zero GPU memory."}
        if setup_only:
            result["setup_verified"] = True
            return result, []
        if app_engine == "phonon":
            if not phonon_runtime:
                raise ValueError("Phonon benchmark requires explicit --phonon-runtime; production runtime is never selected implicitly")
            from src.services.ingestion.phonon_runtime import PhononInstallation

            if PhononInstallation(phonon_runtime)._installed(verify=True) is None:
                raise ValueError("Offline benchmark requires an already verified Phonon installation")
            begin = time.perf_counter()
            result["runtime_before"] = service._get_phonon_runtime().load()["runtime"]
            result["runtime_startup_seconds"] = time.perf_counter() - begin
        import torch

        words = []
        labels = ("cold", "warm", "after_gpu_handoff") if phonon_handoff else ("cold", "warm")
        if phonon_handoff and app_engine != "phonon":
            raise ValueError("GPU handoff benchmark requires --app-engine phonon")
        for index, label in enumerate(labels):
            if label == "after_gpu_handoff":
                begin = time.perf_counter()
                service._release_phonon_model("benchmark competing GPU phase boundary")
                result["handoff_unload_seconds"] = time.perf_counter() - begin
                result["handoff_note"] = "Actual app release boundary and subsequent transcription reload; competing model inference is excluded"
            provenance.clear()
            run_dir = scratch / label
            run_dir.mkdir()
            run_audio = run_dir / audio.name
            shutil.copyfile(audio, run_audio)
            video = database.Video(id=-(index + 1), channel_id=-1,
                                   youtube_id="asr-benchmark", title="ASR benchmark",
                                   duration=int(duration))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            begin = time.perf_counter()
            segments, _ = service._process_transcribe_phase(video, run_audio)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            checkpoint = next(run_dir.glob("*_transcript_raw.json"))
            raw = json.loads(checkpoint.read_text(encoding="utf-8"))
            words = [dict(start=w.start, end=w.end, text=w.word)
                     for seg in segments for w in (seg.words or [])]
            result["runs"].append({"label": label, "phase_wall_seconds": elapsed,
                                   "provenance": dict(provenance),
                                   "warm_note": "warm means same service process; routing may reload models",
                                   "raw": raw, "timestamp_review": timestamp_review(words, duration),
                                   "peak_pytorch_bytes": torch.cuda.max_memory_allocated()
                                   if torch.cuda.is_available() else None})
            if app_engine == "phonon":
                result["runs"][-1]["runtime"] = service._get_phonon_runtime().status()["runtime"]
        if app_engine == "phonon":
            result["runtime_after"] = service._get_phonon_runtime().status()["runtime"]
        return result, words


def benchmark_native_phonon(audio, source_dir, model_dir):
    """Run with a private interpreter; never import the application or its DB."""
    import importlib.metadata as metadata
    import soundfile as sf
    import torch

    sys.path.insert(0, str(source_dir.resolve()))
    from phonon2_cuda_engine import Transcriber

    samples, rate = sf.read(str(audio), dtype="float32")
    if rate != 16000 or samples.ndim != 1:
        raise ValueError("Native probe requires normalized 16 kHz mono audio")
    started = time.perf_counter()
    model = Transcriber("phonon-2", model_dir.resolve())
    report = {"pid": os.getpid(), "load_seconds": time.perf_counter() - started,
              "runtime": model.describe(), "runs": [],
              "packages": {name: metadata.version(name) for name in
                           ("torch", "transformers", "numpy", "huggingface-hub", "soundfile")}}
    for label in ("first", "repeat"):
        started = time.perf_counter()
        text, segments, words = model.transcribe_long_timed(samples)
        torch.cuda.synchronize()
        report["runs"].append({"label": label, "seconds": time.perf_counter() - started,
                               "text": text, "segments": segments, "words": words,
                               "timestamp_review": timestamp_review(words, len(samples) / rate)})
    report["allocated_bytes"] = torch.cuda.memory_allocated()
    report["reserved_bytes"] = torch.cuda.memory_reserved()
    return report, words


def timestamp_review(words, duration):
    """Structural checks only: accurate alignment requires listening to audio."""
    issues = []
    previous = 0.0
    for index, word in enumerate(words):
        try:
            start, end = float(word["start"]), float(word["end"])
            if not (0 <= start <= end <= duration):
                issues.append(f"word {index}: outside audio bounds or reversed")
            if start < previous:
                issues.append(f"word {index}: start precedes previous start")
            previous = start
        except (KeyError, TypeError, ValueError):
            issues.append(f"word {index}: missing or invalid timing")
    return {"word_count": len(words), "issues": issues,
            "alignment_accuracy": "not measured; review against audio"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=["app", "phonon", "phonon-native"])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine-audio", help="Audio path inside a container")
    parser.add_argument("--native-source", type=Path, help="Pinned extracted upstream CUDA source directory")
    parser.add_argument("--native-model", type=Path, help="Verified Phonon-2 model directory")
    parser.add_argument("--phonon-cli", choices=["fermion", "cuda"], default="fermion")
    parser.add_argument("--setup-only", action="store_true", help="Validate app isolation without loading models")
    parser.add_argument("--parakeet-checkpoint", type=Path, help="Local checkpoint for app's configured Parakeet model")
    parser.add_argument("--app-engine", choices=["phonon", "parakeet", "whisper", "auto"], help="Process-local engine override; never saved")
    parser.add_argument("--phonon-env", type=Path, help="Process-local benchmark environment overrides; legacy connection values are ignored")
    parser.add_argument("--phonon-runtime", type=Path, help="Explicit existing app-owned runtime cache for isolated Phonon app benchmarks")
    parser.add_argument("--phonon-handoff", action="store_true", help="Add a third app phase after releasing Phonon at the competing-GPU boundary")
    parser.add_argument("--phonon-repetitions", type=int, default=1, help="Inputs in one Phonon process (CUDA CLI only)")
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="Phonon executable or docker command; must be last")
    args = parser.parse_args()
    if args.output.resolve() == args.audio.resolve():
        parser.error("output must not overwrite input audio")
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve prior results")
    if args.engine == "phonon" and not args.command:
        parser.error("phonon requires --command pointing to its isolated runtime")
    if args.engine == "phonon-native" and not (args.native_source and args.native_model):
        parser.error("phonon-native requires --native-source and --native-model in a private environment")
    if args.phonon_repetitions < 1 or (args.phonon_repetitions != 1 and args.phonon_cli != "cuda"):
        parser.error("repetitions must be positive; multiple inputs require CUDA CLI")
    import soundfile as sf

    audio_info = sf.info(str(args.audio))
    duration = audio_info.duration
    report = {"engine": args.engine, "audio": str(args.audio.resolve()),
              "audio_seconds": duration, "sample_rate": audio_info.samplerate,
              "channels": audio_info.channels}
    begin = time.perf_counter()
    if args.engine == "app":
        report["application"], words = benchmark_application(
            args.audio, duration, args.setup_only, args.parakeet_checkpoint, args.app_engine, args.phonon_env, args.phonon_runtime, args.phonon_handoff)
        report["process_wall_seconds"] = time.perf_counter() - begin
    elif args.engine == "phonon-native":
        report["native"], words = benchmark_native_phonon(args.audio, args.native_source, args.native_model)
        report["process_wall_seconds"] = time.perf_counter() - begin
    elif args.engine == "phonon":
        model_args = ["--model", "phonon-2"] if args.phonon_cli == "cuda" else ["phonon-2"]
        command = args.command + ["transcribe"] + model_args + [
            args.engine_audio or str(args.audio.resolve())] * args.phonon_repetitions + ["--json"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        report["process_wall_seconds"] = time.perf_counter() - begin
        outputs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        raw = outputs[-1]
        report.update(command=command, raw=raw, runtime_stderr=result.stderr,
                      runs=outputs,
                      timing_note="Process wall includes startup/loading; retain runtime timings in raw output.",
                      peak_vram_bytes=None,
                      vram_note="Not measured by this wrapper; use GPU telemetry for CUDA container.")
        words = raw.get("words") or []
    report["timestamp_review"] = timestamp_review(words, duration)
    report["accuracy_note"] = "WER unmeasured: requires a human-corrected reference transcript."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved {args.engine} benchmark to {args.output.resolve()}")


if __name__ == "__main__":
    main()
