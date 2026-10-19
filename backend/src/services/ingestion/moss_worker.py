"""Standalone private MOSS interpreter. No application, service or DB imports."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import wave


def validate_wav(data):
    """Validate actual normalized samples, never rounded caller duration."""
    if (type(data) is not bytes or len(data) < 44 or len(data) > 5400 * 32000 + 44
            or data[:4] != b"RIFF" or data[8:12] != b"WAVE"
            or struct.unpack_from("<I", data, 4)[0] + 8 != len(data)
            or data[12:16] != b"fmt " or struct.unpack_from("<I", data, 16)[0] != 16
            or struct.unpack_from("<HHIIHH", data, 20) != (1, 1, 16000, 32000, 2, 16)
            or data[36:40] != b"data" or struct.unpack_from("<I", data, 40)[0] != len(data) - 44):
        raise ValueError("MOSS requires a complete PCM16 mono 16 kHz WAV of at most 90 minutes")
    with wave.open(io.BytesIO(data), "rb") as audio:
        count, rate = audio.getnframes(), audio.getframerate()
        if (audio.getnchannels() != 1 or audio.getsampwidth() != 2 or rate != 16000
                or audio.getcomptype() != "NONE" or not 0 < count <= 5400 * rate):
            raise ValueError("invalid normalized MOSS WAV")
        pcm = audio.readframes(count + 1)
    if len(pcm) != count * 2:
        raise ValueError("truncated MOSS WAV")
    return pcm, count, rate


def read_pcm(path, sample_count, sample_rate, temp_root):
    supplied = Path(path)
    resolved = supplied.resolve(strict=True)
    if (not supplied.is_absolute() or ".." in supplied.parts or supplied.is_symlink()
            or not resolved.is_relative_to(temp_root.resolve())
            or type(sample_count) is not int or type(sample_rate) is not int):
        raise ValueError("invalid owned MOSS input")
    if resolved.stat().st_size > 5400 * 32000 + 44:
        raise ValueError("MOSS input exceeds 90 minutes")
    pcm, count, rate = validate_wav(resolved.read_bytes())
    if (count, rate) != (sample_count, sample_rate):
        raise ValueError("MOSS sample dimensions do not match")
    return pcm


def completion_for(ids, prompt, context):
    budget = context - prompt
    if (context != 131072 or budget <= 0 or not ids or len(ids) >= budget
            or ids[-1] != 151645 or 151645 in ids[:-1]):
        raise ValueError("MOSS generation incomplete or context exhausted")
    return {"complete": True, "reason": "eos", "eos_token_id": 151645,
            "terminal_token_id": ids[-1], "input_tokens": prompt,
            "output_tokens": len(ids), "max_new_tokens": budget, "context_limit": context}


class Decoder:
    def __init__(self, model_dir, source_dir):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("MOSS requires NVIDIA CUDA BF16")
        spec = importlib.util.spec_from_file_location("moss_pinned_helper", source_dir / "moss_transcribe_diarize/inference_utils.py")
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        started = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        assert self.model.config.text_config.max_position_embeddings == 131072
        assert self.model.generation_config.eos_token_id == 151645
        assert self.model.config._attn_implementation == "sdpa"
        assert next(self.model.parameters()).device.type == "cuda"
        assert next(self.model.parameters()).dtype == torch.bfloat16
        torch.cuda.synchronize()
        self.metadata = {"model": "MOSS-Transcribe-Diarize", "model_revision": "704aa4a9c304e8520be88901e0d1960158ef5b15",
                         "source_revision": "e3bdd93f8ad58418e04659818efd5fd87a0bc486",
                         "device": "cuda", "dtype": "bfloat16", "attention": "sdpa",
                         "torch": torch.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(), "load_seconds": time.perf_counter() - started}

    def generate(self, pcm):
        import numpy as np
        import torch
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        # Pinned prepare_inputs semantics, using the already validated PCM rather
        # than reopening a path through a general media/URL decoder.
        messages = self.helper.build_transcription_messages("owned.wav")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        context = self.model.config.text_config.max_position_embeddings
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.processor(text=text, audio=[samples], max_length=context,
                                    audio_kwargs={"device": "cuda:0"}, return_tensors="pt").to("cuda:0")
        prompt = inputs["input_ids"].shape[1]
        budget = context - prompt
        if budget <= 0:
            raise ValueError("MOSS has no remaining output context; select another engine")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        torch.cuda.synchronize()
        ids = output[0, prompt:].tolist()
        completion = completion_for(ids, prompt, context)
        raw = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        self.metadata.update(decode_seconds=time.perf_counter() - started,
                             allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             generated_ids=ids,
                             raw_special_text=self.processor.tokenizer.decode(ids, skip_special_tokens=False))
        return raw, completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    args = parser.parse_args()
    from moss import parse_generation
    output, model, seen, requests = sys.stdout, None, set(), 0
    for line in sys.stdin:
        request_id, operation, error_code = None, None, "request_failed"
        try:
            message = json.loads(line)
            request_id, operation = message["id"], message["operation"]
            if (type(message.get("protocol")) is not int or message["protocol"] != 1
                    or type(request_id) is not str or not request_id or request_id in seen):
                raise ValueError("invalid or duplicate request")
            seen.add(request_id)
            result = None
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "load":
                    error_code = "load_failed"
                    if model is None:
                        model = Decoder(args.model_dir, args.source_dir)
                elif operation in {"transcribe", "probe"}:
                    if model is None:
                        raise ValueError("model not loaded")
                    if operation == "probe":
                        # Silence can hallucinate words/times. Readiness proves a
                        # real complete generation, never invents a transcript.
                        pcm = bytes(32000)
                    else:
                        error_code = "invalid_input"
                        pcm = read_pcm(message["audio"], message["sample_count"], message["sample_rate"], args.temp_root)
                    error_code = "decode_failed"
                    raw, completion = model.generate(pcm)
                    if operation == "transcribe":
                        result = parse_generation(raw, message["sample_count"], message["sample_rate"], completion)
                    else:
                        result = {"completion": completion, "raw_text": raw}
                    requests += 1
                elif operation not in {"status", "shutdown"}:
                    raise ValueError("unknown operation")
            runtime = {"pid": os.getpid(), "model_loaded": model is not None, "request_count": requests,
                       **(model.metadata if model else {})}
            encoded = json.dumps({"protocol": 1, "id": request_id, "runtime": runtime, "result": result}, allow_nan=False)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            encoded = json.dumps({"protocol": 1, "id": request_id, "error": error_code})
        output.write(encoded + "\n")
        output.flush()
        if operation == "shutdown":
            break


if __name__ == "__main__":
    main()
