"""Ollama model management and local GPU hardware detection helpers."""
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional


def _detect_gpu_hardware() -> dict:
    gpu_name = None
    gpu_vram_gb = None
    gpu_vendor = None
    gpu_count = 0
    detection_method = None

    # 1) torch CUDA (most reliable for active compute device)
    try:
        import torch  # lazy import so backend still runs without torch
        if torch.cuda.is_available():
            gpu_count = int(torch.cuda.device_count() or 0)
            best_mem = -1
            best_name = None
            for idx in range(gpu_count):
                props = torch.cuda.get_device_properties(idx)
                mem = int(getattr(props, "total_memory", 0) or 0)
                name = str(torch.cuda.get_device_name(idx) or f"CUDA GPU {idx}")
                if mem > best_mem:
                    best_mem = mem
                    best_name = name
            if best_name and best_mem > 0:
                gpu_name = best_name
                gpu_vram_gb = round(best_mem / (1024 ** 3), 1)
                detection_method = "torch_cuda"
    except Exception:
        pass

    # 2) nvidia-smi fallback
    if not gpu_name:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode == 0:
                lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
                best_mem = -1
                best_name = None
                for ln in lines:
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    try:
                        mem_mb = float(parts[1])
                    except Exception:
                        continue
                    if mem_mb > best_mem:
                        best_mem = mem_mb
                        best_name = name
                if best_name and best_mem > 0:
                    gpu_name = best_name
                    gpu_vram_gb = round(best_mem / 1024.0, 1)
                    gpu_count = len(lines)
                    detection_method = "nvidia_smi"
        except Exception:
            pass

    # 3) Windows video controller fallback
    if not gpu_name and os.name == "nt":
        try:
            ps = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM | ConvertTo-Json -Compress"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                data = json.loads(proc.stdout)
                cards = data if isinstance(data, list) else [data]
                cards = [c for c in cards if isinstance(c, dict)]
                # Prefer likely discrete GPU rows over software adapters
                filtered = [
                    c for c in cards
                    if "microsoft basic" not in str(c.get("Name", "")).lower()
                ] or cards
                best = None
                best_mem = -1
                for card in filtered:
                    try:
                        mem_bytes = float(card.get("AdapterRAM") or 0)
                    except Exception:
                        mem_bytes = 0
                    if mem_bytes > best_mem:
                        best_mem = mem_bytes
                        best = card
                if best:
                    name = str(best.get("Name") or "").strip()
                    if name:
                        gpu_name = name
                        gpu_count = len(filtered)
                        if best_mem > 0:
                            gpu_vram_gb = round(best_mem / (1024 ** 3), 1)
                        detection_method = "win32_video_controller"
        except Exception:
            pass

    if gpu_name:
        low = gpu_name.lower()
        if "nvidia" in low:
            gpu_vendor = "nvidia"
        elif "amd" in low or "radeon" in low:
            gpu_vendor = "amd"
        elif "intel" in low:
            gpu_vendor = "intel"
        else:
            gpu_vendor = "unknown"
    else:
        gpu_vendor = "cpu_only"

    return {
        "gpu_name": gpu_name,
        "gpu_vendor": gpu_vendor,
        "gpu_vram_gb": gpu_vram_gb,
        "gpu_count": gpu_count,
        "detection_method": detection_method or "none",
    }


def _build_ollama_quant_tag(base_model: str, tier: str) -> str:
    base = (base_model or "").strip()
    if not base:
        return ""
    # Keep medium/default as canonical base tag.
    if tier == "medium":
        return base
    if tier == "lite":
        return f"{base}-q4_K_M"
    if tier == "q8":
        return f"{base}-q8_0"
    return base


def _normalize_ollama_model_ref(model_ref: str) -> str:
    """
    Normalize model refs for Ollama pull/generate.
    Supports:
    - direct Ollama tags (unchanged), e.g. qwen3.5:35b-a3b
    - HF refs, e.g. hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M
    - HF URLs, e.g. https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
    """
    raw = (model_ref or "").strip()
    if not raw:
        return ""

    lower_raw = raw.lower()
    if lower_raw.startswith("hf.co/"):
        return raw

    m = re.match(r"^https?://huggingface\.co/([^/\s]+)/([^/\s?#:]+)", raw, flags=re.IGNORECASE)
    if not m:
        return raw

    owner = m.group(1).strip()
    repo = m.group(2).strip()
    normalized = f"hf.co/{owner}/{repo}"

    quant_match = re.search(r"(?::|[?&](?:quant|gguf|q)=)([A-Za-z0-9_]+)", raw, flags=re.IGNORECASE)
    if quant_match:
        quant = quant_match.group(1).strip()
        if quant:
            normalized = f"{normalized}:{quant}"
    return normalized


def _ollama_model_name_matches(local_model_name: str, requested_model_ref: str) -> bool:
    local = (local_model_name or "").strip().lower()
    req = (requested_model_ref or "").strip().lower()
    if not local or not req:
        return False

    if local == req or local.startswith(f"{req}:"):
        return True

    def _sig(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    local_base = re.sub(r":latest$", "", local)
    req_base = re.sub(r":latest$", "", req)
    if local_base == req_base:
        return True

    local_sig = _sig(local_base)
    req_sig = _sig(req_base)
    if local_sig and req_sig and (local_sig == req_sig or local_sig.startswith(req_sig) or req_sig.startswith(local_sig)):
        return True

    # HF/Unsloth refs may be downloaded under normalized Ollama names
    # (e.g. qwen3.5:35b-a3b-q4_k_m or unsloth/Qwen...).
    if req_base.startswith("hf.co/"):
        no_hf = req_base[len("hf.co/"):]
        if ":" in no_hf:
            repo_path, req_quant = no_hf.rsplit(":", 1)
        else:
            repo_path, req_quant = no_hf, ""
        repo_tail = (repo_path.split("/")[-1] if repo_path else "").strip().lower()
        repo_tail_no_gguf = re.sub(r"-gguf$", "", repo_tail)
        req_quant_norm = _sig(req_quant)

        tail_sig = _sig(repo_tail_no_gguf)
        if tail_sig and tail_sig in local_sig:
            if not req_quant_norm or req_quant_norm in local_sig:
                return True

    return False


def _ollama_pull_job_key(ollama_url: str, model_ref: str) -> str:
    return f"{(ollama_url or '').rstrip('/').lower()}|{(model_ref or '').strip().lower()}"


def _set_ollama_pull_job(key: str, patch: dict) -> None:
    with _ollama_pull_jobs_lock:
        job = dict(_ollama_pull_jobs.get(key) or {})
        job.update(patch)
        _ollama_pull_jobs[key] = job


def _run_ollama_pull_job(ollama_url: str, model_ref: str) -> None:
    import httpx

    key = _ollama_pull_job_key(ollama_url, model_ref)
    started_ts = time.time()
    _set_ollama_pull_job(key, {
        "status": "running",
        "started_at": started_ts,
        "updated_at": started_ts,
        "completed_at": None,
        "error": None,
        "ollama_response": None,
        "pull_event_status": "starting",
        "pull_completed": None,
        "pull_total": None,
        "pull_percent": None,
    })

    try:
        pull_data: dict = {}
        with httpx.stream(
            "POST",
            f"{ollama_url.rstrip('/')}/api/pull",
            json={"name": model_ref, "stream": True},
            timeout=7200,
        ) as pull_resp:
            pull_resp.raise_for_status()
            for raw_line in pull_resp.iter_lines():
                line = (raw_line or "").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    err_text = str(event.get("error") or "").strip() or "Ollama pull failed"
                    now_ts = time.time()
                    _set_ollama_pull_job(key, {
                        "status": "failed",
                        "updated_at": now_ts,
                        "completed_at": now_ts,
                        "error": err_text[:1200],
                        "pull_event_status": "failed",
                    })
                    return
                pull_data = event
                status_text = str(event.get("status") or "").strip()
                completed = event.get("completed")
                total = event.get("total")
                percent = None
                try:
                    if completed is not None and total is not None:
                        c = float(completed)
                        t = float(total)
                        if t > 0:
                            percent = round(max(0.0, min(100.0, (c / t) * 100.0)), 1)
                except Exception:
                    percent = None
                _set_ollama_pull_job(key, {
                    "status": "running",
                    "updated_at": time.time(),
                    "pull_event_status": status_text or "downloading",
                    "pull_completed": completed if isinstance(completed, (int, float)) else None,
                    "pull_total": total if isinstance(total, (int, float)) else None,
                    "pull_percent": percent,
                })

        available_models: list[str] = []
        try:
            tags_resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
            tags_resp.raise_for_status()
            available_models = [m.get("name", "") for m in (tags_resp.json().get("models") or [])]
        except Exception:
            available_models = []

        model_found_after = any(_ollama_model_name_matches(m, model_ref) for m in available_models)
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "completed" if model_found_after else "completed_unverified",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": None,
            "ollama_response": pull_data,
            "available_models": available_models[:2000],
            "pull_event_status": "completed",
            "pull_percent": 100.0 if model_found_after else pull_data.get("pull_percent"),
        })
    except httpx.HTTPStatusError as e:
        now_ts = time.time()
        detail = ""
        try:
            body = (e.response.text or "").strip()
            if body:
                detail = f" | response: {body[:800]}"
        except Exception:
            detail = ""
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": f"HTTP {getattr(e.response, 'status_code', 'error')} from Ollama /api/pull{detail}"[:1200],
            "pull_event_status": "failed",
        })
    except httpx.ConnectError:
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": f"Cannot connect to Ollama at {ollama_url}",
            "pull_event_status": "failed",
        })
    except Exception as e:
        now_ts = time.time()
        _set_ollama_pull_job(key, {
            "status": "failed",
            "updated_at": now_ts,
            "completed_at": now_ts,
            "error": str(e)[:1200],
            "pull_event_status": "failed",
        })


OLLAMA_KNOWN_TAG_SIZES_GB = {
    # Sourced from Ollama library tag pages (2026-02-28).
    "qwen2.5:3b": 1.9,
    "qwen2.5:7b": 4.7,
    "qwen2.5:14b": 9.0,
    "qwen3.5:27b": 17.0,
    "qwen3.5:27b-q4_k_m": 17.0,
    "qwen3.5:35b-a3b": 24.0,
    "qwen3.5:35b-a3b-q4_k_m": 24.0,
}
_ollama_size_cache_lock = threading.Lock()
_ollama_size_cache: dict[str, tuple[float, float]] = {}
_OLLAMA_SIZE_CACHE_TTL_SECONDS = 12 * 60 * 60
_ollama_pull_jobs_lock = threading.Lock()
_ollama_pull_jobs: dict[str, dict] = {}


def _get_ollama_exact_size_gb_for_tag(model_tag: str) -> Optional[float]:
    tag = (model_tag or "").strip().lower()
    if not tag:
        return None

    if tag in OLLAMA_KNOWN_TAG_SIZES_GB:
        return float(OLLAMA_KNOWN_TAG_SIZES_GB[tag])

    now_ts = time.time()
    with _ollama_size_cache_lock:
        cached = _ollama_size_cache.get(tag)
        if cached and (now_ts - cached[1]) < _OLLAMA_SIZE_CACHE_TTL_SECONDS:
            return float(cached[0]) if cached[0] >= 0 else None

    try:
        safe_tag = urllib.parse.quote(tag, safe=":-_.")
        url = f"https://ollama.com/library/{safe_tag}"
        html_text = urllib.request.urlopen(url, timeout=6).read().decode("utf-8", errors="ignore")
        m = re.search(r"·\s*(\d+(?:\.\d+)?)GB\s*·", html_text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)GB", html_text, flags=re.IGNORECASE)
        if m:
            size_gb = round(float(m.group(1)), 1)
            with _ollama_size_cache_lock:
                _ollama_size_cache[tag] = (size_gb, time.time())
            return size_gb
    except Exception:
        pass

    with _ollama_size_cache_lock:
        _ollama_size_cache[tag] = (-1.0, time.time())
    return None


def _ollama_quant_info(tier: str) -> tuple[str, float]:
    t = (tier or "").strip().lower()
    if t == "lite":
        return "Q4_K_M", 4.5
    if t == "q8":
        return "Q8_0", 8.5
    # Ollama default tag (no explicit quant suffix) varies by model release.
    return "Default (varies by model, typically Q5/Q6 class)", 5.5


def _estimate_ollama_model_size_gb(base_model: str, tier: str) -> tuple[Optional[float], str]:
    """
    Heuristic estimate of on-disk GGUF model size for the selected tag.
    This is informational only; actual size depends on exact upstream build/tag.
    """
    base = (base_model or "").strip().lower()
    if not base:
        return None, "unknown"

    # Prefer exact Ollama library tag size when available.
    tag = _build_ollama_quant_tag(base, tier)
    exact = _get_ollama_exact_size_gb_for_tag(tag)
    if exact is None and tier in {"medium", "lite"}:
        # Many models expose default q4-ish tags without explicit suffix.
        exact = _get_ollama_exact_size_gb_for_tag(base)
    if exact is not None:
        return round(float(exact), 1), "ollama_exact"

    # Extract first "<num>b" token from model tag (e.g. qwen2.5:14b, qwen3.5:35b-a3b)
    m = re.search(r"(\d+(?:\.\d+)?)b", base)
    if not m:
        return None, "unknown"
    try:
        params_b = float(m.group(1))
    except Exception:
        return None, "unknown"

    # Approximate effective bits-per-weight for displayed tiers.
    _label, bits = _ollama_quant_info(tier)

    # Convert params+quant to rough GB with format/index overhead factor.
    estimated_gb = params_b * (bits / 8.0) * 1.15
    return round(estimated_gb, 1), "estimated"


def _recommend_ollama_for_hardware(gpu_vram_gb: Optional[float], objective: str = "balanced") -> dict:
    """Recommend a single Ollama model tag based on VRAM and user tradeoff objective."""
    vram = float(gpu_vram_gb) if gpu_vram_gb is not None else None
    normalized = (objective or "balanced").strip().lower()
    if normalized not in {"speed", "balanced", "capability"}:
        normalized = "balanced"

    # Default fallback when GPU VRAM cannot be reliably detected.
    if vram is None:
        if normalized == "speed":
            base, tier = "qwen2.5:3b", "lite"
        elif normalized == "capability":
            base, tier = "qwen2.5:14b", "medium"
        else:
            base, tier = "qwen2.5:7b", "medium"
        reason = (
            "GPU VRAM could not be detected. "
            f'Using "{normalized}" objective fallback.'
        )
    else:
        # Speed-first: prioritize lower latency while staying useful.
        if normalized == "speed":
            if vram >= 36:
                base, tier = "qwen3.5:27b", "lite"
            elif vram >= 24:
                base, tier = "qwen2.5:14b", "lite"
            elif vram >= 12:
                base, tier = "qwen2.5:7b", "lite"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "lite"
            elif vram >= 4:
                base, tier = "qwen2.5:3b", "medium"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Speed objective favors smaller/faster quantized tags."
        # Capability-first: maximize output quality within practical VRAM targets.
        elif normalized == "capability":
            if vram >= 48:
                base, tier = "qwen3.5:35b-a3b", "q8"
            elif vram >= 30:
                base, tier = "qwen3.5:35b-a3b", "medium"
            elif vram >= 20:
                base, tier = "qwen3.5:27b", "medium"
            elif vram >= 12:
                base, tier = "qwen2.5:14b", "q8"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "q8"
            elif vram >= 4:
                base, tier = "qwen2.5:7b", "lite"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Capability objective favors stronger models and higher-quality quants."
        # Balanced: default compromise of latency and output quality.
        else:
            if vram >= 32:
                base, tier = "qwen3.5:35b-a3b", "medium"
            elif vram >= 20:
                base, tier = "qwen3.5:27b", "medium"
            elif vram >= 12:
                base, tier = "qwen2.5:14b", "medium"
            elif vram >= 8:
                base, tier = "qwen2.5:7b", "q8"
            elif vram >= 4:
                base, tier = "qwen2.5:7b", "lite"
            else:
                base, tier = "qwen2.5:3b", "lite"
            reason = f"Detected ~{vram:.1f} GB VRAM. Balanced objective targets speed/quality stability."

    model_tag = _build_ollama_quant_tag(base, tier)
    estimated_size_gb, size_source = _estimate_ollama_model_size_gb(base, tier)

    # If capability suggested q8 but tag size cannot be resolved from Ollama,
    # prefer a known-available medium tag for this model family.
    if tier == "q8" and size_source != "ollama_exact":
        tier = "medium"
        model_tag = _build_ollama_quant_tag(base, tier)
        estimated_size_gb, size_source = _estimate_ollama_model_size_gb(base, tier)

    quant_level, quant_bits_estimate = _ollama_quant_info(tier)

    # Explain why capability may still land on medium/default quant.
    if normalized == "capability" and tier != "q8" and vram is not None:
        q8_est, _q8_source = _estimate_ollama_model_size_gb(base, "q8")
        if q8_est is not None and q8_est > (vram * 0.9):
            reason = (
                f"{reason} "
                f"Q8 for {base} is estimated around {q8_est:.1f} GB, so this recommendation keeps a lower quant for fit/stability."
            )

    return {
        "objective": normalized,
        "base_model": base,
        "tier": tier,
        "model_tag": model_tag,
        "estimated_size_gb": estimated_size_gb,
        "size_source": size_source,
        "quant_level": quant_level,
        "quant_bits_estimate": round(float(quant_bits_estimate), 1),
        "fallback_tag": base,
        "reason": reason,
    }
