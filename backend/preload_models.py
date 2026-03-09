import argparse
import os
import sys
import time

# Prevent huggingface_hub from blocking on interactive login prompts
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_NO_ADVISORY_WARNINGS", "1")

from src.services.ingestion import IngestionService


def log(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preload local ASR models into cache.")
    parser.add_argument("--engine", choices=["auto", "whisper", "parakeet"], default=os.getenv("TRANSCRIPTION_ENGINE", "auto"))
    parser.add_argument("--whisper-model", default=os.getenv("TRANSCRIPTION_MODEL", "medium"))
    parser.add_argument("--parakeet-model", default=os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"))
    args = parser.parse_args()

    os.environ["TRANSCRIPTION_ENGINE"] = args.engine
    os.environ["TRANSCRIPTION_MODEL"] = args.whisper_model
    os.environ["PARAKEET_MODEL"] = args.parakeet_model

    log("Initializing IngestionService...")
    svc = IngestionService()
    svc._ensure_device()

    log(f"Device: {svc.device}")

    if args.engine in {"auto", "whisper"}:
        log(f"Preloading Whisper model: {args.whisper_model} (this may download ~1 GB on first run)...")
        t0 = time.time()
        svc._load_whisper_model(job_id=None)
        log(f"Whisper model ready. ({time.time() - t0:.1f}s)")

    if args.engine in {"auto", "parakeet"}:
        try:
            log(f"Preloading Parakeet model: {args.parakeet_model} (this may download ~2 GB on first run)...")
            t0 = time.time()
            svc._load_parakeet_model(job_id=None)
            log(f"Parakeet model ready. ({time.time() - t0:.1f}s)")
        except Exception as e:
            log(f"Parakeet preload skipped: {e}")
            if args.engine == "parakeet":
                raise

    log("Done.")


if __name__ == "__main__":
    main()
