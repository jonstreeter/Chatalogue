"""Private native worker. Never imports the application, service or database."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import wave


def read_pcm(path, duration, temp_root, max_bytes):
    path = Path(path).resolve()
    if not path.is_relative_to(temp_root.resolve()) or not math.isfinite(duration) or not 0 < duration <= 7200:
        raise ValueError("invalid input")
    size = path.stat().st_size
    if not 0 < size < max_bytes:
        raise ValueError("invalid input size")
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                or audio.getframerate() != 16000 or frames / 16000 != duration or frames * 2 > size):
            raise ValueError("invalid WAV")
        pcm = audio.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError("truncated WAV")
    return pcm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args()
    temp_root = args.temp_root.resolve()
    output = sys.stdout
    model = None
    requests = 0
    started = time.monotonic()

    def status():
        return {"pid": os.getpid(), "model_loaded": model is not None,
                "request_count": requests, "uptime_seconds": time.monotonic() - started,
                **(model.describe() if model is not None else {}),
                "version": "cuda-1.0.8-source/native", "device": "cuda" if model is not None else None}

    for line in sys.stdin:
        request_id = None
        operation = None
        error_code = "request_failed"
        try:
            message = json.loads(line)
            request_id = message["id"]
            operation = message["operation"]
            if message["protocol"] != 1 or not isinstance(request_id, str):
                raise ValueError("invalid protocol")
            result = None
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "load":
                    error_code = "load_failed"
                    if model is None:
                        sys.path.insert(0, str(Path(__file__).with_name("phonon_upstream")))
                        from phonon2_cuda_engine import Transcriber
                        model = Transcriber("phonon-2", args.model_dir)
                elif operation == "transcribe":
                    if model is None:
                        raise ValueError("model not loaded")
                    duration = float(message["duration"])
                    error_code = "invalid_input"
                    pcm = read_pcm(message["audio"], duration, temp_root, args.max_bytes)
                    error_code = "decode_failed"
                    import numpy as np
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                    text, segments, words = model.transcribe_long_timed(samples)
                    result = {"text": text, "segments": segments, "words": words, "duration": duration}
                    requests += 1
                elif operation not in {"status", "shutdown"}:
                    raise ValueError("unknown operation")
            reply = {"protocol": 1, "id": request_id, "runtime": status(), "result": result}
            encoded = json.dumps(reply, allow_nan=False)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            encoded = json.dumps({"protocol": 1, "id": request_id,
                                  "error": error_code})
        output.write(encoded + "\n")
        output.flush()
        if operation == "shutdown":
            break


if __name__ == "__main__":
    main()
