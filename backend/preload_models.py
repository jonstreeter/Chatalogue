import argparse
import os

from src.services.ingestion import IngestionService


def main():
    parser = argparse.ArgumentParser(description="Preload local ASR models into cache.")
    parser.add_argument("--engine", choices=["auto", "whisper", "parakeet"], default=os.getenv("TRANSCRIPTION_ENGINE", "auto"))
    parser.add_argument("--whisper-model", default=os.getenv("TRANSCRIPTION_MODEL", "medium"))
    parser.add_argument("--parakeet-model", default=os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"))
    args = parser.parse_args()

    os.environ["TRANSCRIPTION_ENGINE"] = args.engine
    os.environ["TRANSCRIPTION_MODEL"] = args.whisper_model
    os.environ["PARAKEET_MODEL"] = args.parakeet_model

    svc = IngestionService()
    svc._ensure_device()

    print(f"Device: {svc.device}")

    if args.engine in {"auto", "whisper"}:
        print(f"Preloading Whisper model: {args.whisper_model}")
        svc._load_whisper_model(job_id=None)
        print("Whisper model ready.")

    if args.engine in {"auto", "parakeet"}:
        try:
            print(f"Preloading Parakeet model: {args.parakeet_model}")
            svc._load_parakeet_model(job_id=None)
            print("Parakeet model ready.")
        except Exception as e:
            print(f"Parakeet preload skipped: {e}")
            if args.engine == "parakeet":
                raise

    print("Done.")


if __name__ == "__main__":
    main()
