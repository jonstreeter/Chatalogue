"""LLM provider plumbing: provider selection, VRAM guard, text generation across providers.

Part of IngestionService (mixin). Split from the former single-module
ingestion.py; see docs/ingestion-service-split.md.
"""
import os
import time
import json
import re
import urllib.request
import urllib.error

from ..logger import log, log_verbose
from .. import episode_clone as clone_svc


class LlmProviderMixin:
    def _is_llm_enabled(self) -> bool:
        raw = os.getenv("LLM_ENABLED")
        if raw is None:
            raw = os.getenv("OLLAMA_ENABLED", "false")
        return str(raw).lower() == "true"

    def _is_local_ollama_provider_active(self) -> bool:
        return self._is_llm_enabled() and self._get_llm_provider() == "ollama"

    def _resolve_local_ollama_min_free_vram_gb(self, total_gb: float) -> float:
        default_gb = 6.0
        if total_gb >= 28.0:
            default_gb = 10.0
        elif total_gb >= 20.0:
            default_gb = 8.0
        raw = (os.getenv("OLLAMA_LOCAL_LLM_MIN_FREE_VRAM_GB") or "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except Exception:
                pass
        return default_gb

    def _get_local_ollama_vram_guard(self) -> tuple[bool, str, float | None, float | None]:
        if not self._is_local_ollama_provider_active():
            return True, "non_local_ollama_provider", None, None

        if self._has_active_pipeline_gpu_work():
            return False, "pipeline_gpu_work_active", None, None

        self._ensure_device()
        if self.device != "cuda":
            return True, "non_cuda_worker", None, None

        snap = self._cuda_memory_snapshot()
        free_b, total_b, _, _, free_gb, _ = self._snap_unpack(snap)
        total_gb = float(total_b) / (1024 ** 3) if total_b > 0 else 0.0
        min_free_gb = self._resolve_local_ollama_min_free_vram_gb(total_gb)
        if free_gb < min_free_gb:
            return False, f"low_vram_headroom_{free_gb:.1f}gb_below_{min_free_gb:.1f}gb", free_gb, min_free_gb
        return True, "ok", free_gb, min_free_gb

    def _prepare_for_local_ollama_llm_work(self, job_id: int | None = None) -> tuple[float | None, float | None]:
        if not self._is_local_ollama_provider_active():
            return None, None
        self._ensure_device()
        if self.device != "cuda":
            return None, None

        self._release_parakeet_model("pre_local_ollama_llm", job_id=job_id)
        self._release_whisper_model("pre_local_ollama_llm", job_id=job_id)
        self._release_diarization_models("pre_local_ollama_llm", job_id=job_id)
        self._clear_cuda_cache()

        snap = self._cuda_memory_snapshot()
        free_b, total_b, _, _, free_gb, _ = self._snap_unpack(snap)
        total_gb = float(total_b) / (1024 ** 3) if total_b > 0 else 0.0
        if job_id:
            self._upsert_job_payload_fields(
                job_id,
                {
                    "local_ollama_prepared": True,
                    "local_ollama_free_gb_after_prepare": round(free_gb, 2) if free_b > 0 else 0.0,
                    "local_ollama_total_gb_after_prepare": round(total_gb, 2) if total_gb > 0 else 0.0,
                },
            )
        return free_gb, total_gb

    def _raise_if_local_ollama_llm_is_blocked(self, job_id: int | None = None):
        allowed, reason, free_gb, min_free_gb = self._get_local_ollama_vram_guard()
        if allowed:
            self._prepare_for_local_ollama_llm_work(job_id=job_id)
            return

        detail = {
            "pipeline_gpu_work_active": "Local Ollama funny-moment explanation is blocked while pipeline GPU work is active.",
            "non_local_ollama_provider": "",
            "non_cuda_worker": "",
        }.get(reason)
        if not detail:
            if free_gb is not None and min_free_gb is not None:
                detail = (
                    "Local Ollama funny-moment explanation is blocked until VRAM headroom recovers "
                    f"({free_gb:.1f} GB free, needs at least {min_free_gb:.1f} GB)."
                )
            else:
                detail = "Local Ollama funny-moment explanation is temporarily blocked."

        if job_id:
            self._upsert_job_payload_fields(
                job_id,
                {
                    "local_ollama_blocked": True,
                    "local_ollama_block_reason": reason,
                    "local_ollama_free_gb": round(float(free_gb), 2) if free_gb is not None else None,
                    "local_ollama_min_free_gb": round(float(min_free_gb), 2) if min_free_gb is not None else None,
                },
            )
        raise RuntimeError(detail)

    def _get_llm_provider(self) -> str:
        provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
        aliases = {
            "nvidia": "nvidia_nim",
            "nim": "nvidia_nim",
            "nvidia-nim": "nvidia_nim",
            "nvidia_nim": "nvidia_nim",
            "chatgpt": "openai",
            "openai": "openai",
            "claude": "anthropic",
            "anthropic": "anthropic",
            "google": "gemini",
            "google_gemini": "gemini",
            "google-gemini": "gemini",
            "gemini": "gemini",
            "groq": "groq",
            "openrouter": "openrouter",
            "xai": "xai",
            "ollama": "ollama",
        }
        return aliases.get(provider, provider)

    def _get_nvidia_nim_min_request_interval_seconds(self) -> float:
        """Minimum spacing between hosted NIM requests (process-wide)."""
        raw = (os.getenv("NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS") or "2.5").strip()
        try:
            value = float(raw)
        except Exception:
            value = 2.5
        # Keep within sane bounds.
        return max(0.0, min(value, 30.0))

    def _get_provider_default_model(self, provider: str) -> str:
        normalized = clone_svc._normalize_provider(provider) or "ollama"
        if normalized == "nvidia_nim":
            return (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip()
        if normalized == "openai":
            return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        if normalized == "anthropic":
            return (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip()
        if normalized == "gemini":
            return (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        if normalized == "groq":
            return (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        if normalized == "openrouter":
            return (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip()
        if normalized == "xai":
            return (os.getenv("XAI_MODEL") or "grok-2").strip()
        return (os.getenv("OLLAMA_MODEL") or "mistral").strip()

    def resolve_clone_llm_target(
        self,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> tuple[str, str, str]:
        provider = clone_svc._normalize_provider(provider_override) or self._get_llm_provider()
        model = str(model_override or "").strip() or self._get_provider_default_model(provider)
        return provider, model, f"{provider}:{model}"

    def _get_configured_llm_model_name(self) -> str:
        _provider, _model, target_name = self.resolve_clone_llm_target()
        return target_name

    def _extract_openai_chat_text(self, raw: str) -> str:
        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") if choices else {}
            content = (message or {}).get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").lower()
                        if item_type and ("reason" in item_type or "think" in item_type):
                            continue
                        text_part = item.get("text")
                        if isinstance(text_part, str):
                            parts.append(text_part)
                text = "\n".join(parts).strip()
            else:
                text = str(content or "").strip()
            if not text:
                text = str((message or {}).get("reasoning_content") or "").strip()
        except Exception:
            text = str(raw or "").strip()
        return self._strip_llm_reasoning_artifacts(text)

    def _openai_compatible_generate_text(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
        extra_headers: dict | None = None,
        extra_payload: dict | None = None,
    ) -> str:

        if not api_key:
            raise RuntimeError(f"{provider_name} API key is not configured in Settings.")
        if not model:
            raise RuntimeError(f"{provider_name} model is not configured in Settings.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max(32, int(num_predict)),
        }
        if extra_payload:
            payload.update(extra_payload)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        headers = {k: v for k, v in headers.items() if v is not None and str(v) != ""}

        normalized_base = base_url.rstrip("/")
        endpoint = f"{normalized_base}/chat/completions" if normalized_base.lower().endswith("/v1") else f"{normalized_base}/v1/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"{provider_name} request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach {provider_name} at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"{provider_name} request failed: {e}") from e

        text = self._extract_openai_chat_text(raw)
        if not text:
            raise RuntimeError(f"{provider_name} returned an empty response.")
        return text

    def _anthropic_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:

        api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        base_url = (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest").strip()

        if not api_key:
            raise RuntimeError("Anthropic API key is not configured. Set ANTHROPIC_API_KEY in Settings.")
        if not model:
            raise RuntimeError("ANTHROPIC_MODEL is not configured.")

        payload = {
            "model": model,
            "max_tokens": max(32, int(num_predict)),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Anthropic request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Anthropic at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Anthropic request failed: {e}") from e

        try:
            data = json.loads(raw)
            parts = data.get("content") or []
            text = "\n".join(
                str(p.get("text") or "").strip()
                for p in parts
                if isinstance(p, dict) and str(p.get("type") or "").lower() == "text"
            ).strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("Anthropic returned an empty response.")
        return text

    def _gemini_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        import urllib.parse

        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        base_url = (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

        if not api_key:
            raise RuntimeError("Gemini API key is not configured. Set GEMINI_API_KEY in Settings.")
        if not model:
            raise RuntimeError("GEMINI_MODEL is not configured.")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": max(32, int(num_predict)),
            },
        }

        encoded_model = urllib.parse.quote(model, safe="")
        req = urllib.request.Request(
            f"{base_url}/v1beta/models/{encoded_model}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Gemini request failed ({e.code}): {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Gemini at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e

        try:
            data = json.loads(raw)
            candidates = data.get("candidates") or []
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
            text = "\n".join(
                str(p.get("text") or "").strip()
                for p in parts
                if isinstance(p, dict) and p.get("text")
            ).strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    def _nvidia_nim_generate_text(
        self,
        prompt: str,
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:

        api_key = (os.getenv("NVIDIA_NIM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA NIM API key is not configured. Set NVIDIA_NIM_API_KEY in Settings.")

        base_url = (os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("NVIDIA_NIM_MODEL") or "moonshotai/kimi-k2.5").strip()
        if not model:
            raise RuntimeError("NVIDIA_NIM_MODEL is not configured.")

        # Kimi K2.5 supports thinking mode via chat_template_kwargs.thinking.
        thinking_mode_raw = (os.getenv("NVIDIA_NIM_THINKING_MODE") or "false").strip().lower()
        thinking_mode = thinking_mode_raw == "true"

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max(32, int(num_predict)),
        }
        if thinking_mode:
            payload["chat_template_kwargs"] = {"thinking": True}

        max_attempts = 5
        raw = ""
        last_http_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self._nvidia_nim_request_lock:
                    now = time.time()
                    wait_for = max(0.0, self._nvidia_nim_next_allowed_at - now)
                    if wait_for > 0:
                        log_verbose(f"NVIDIA NIM pacing wait: {wait_for:.2f}s")
                        time.sleep(wait_for)

                    req = urllib.request.Request(
                        f"{base_url}/v1/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "application/json",
                        },
                        method="POST",
                    )

                    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")

                    # Proactively space requests even after success to avoid bursty
                    # stage transitions (e.g., chunk loop -> moment loop).
                    self._nvidia_nim_next_allowed_at = time.time() + self._get_nvidia_nim_min_request_interval_seconds()
                    last_http_error = None
                    break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(e)

                if e.code == 429 and attempt < max_attempts:
                    retry_after = None
                    try:
                        retry_after_hdr = (e.headers or {}).get("Retry-After")
                        if retry_after_hdr:
                            retry_after = float(retry_after_hdr)
                    except Exception:
                        retry_after = None
                    backoff = retry_after if retry_after is not None else min(12.0, 1.5 * (2 ** (attempt - 1)))
                    # Push the next-allowed time forward process-wide so concurrent
                    # explain jobs/UI actions don't immediately retrigger the limit.
                    with self._nvidia_nim_request_lock:
                        self._nvidia_nim_next_allowed_at = max(
                            self._nvidia_nim_next_allowed_at,
                            time.time() + max(0.5, backoff) + self._get_nvidia_nim_min_request_interval_seconds()
                        )
                    log(f"NVIDIA NIM rate-limited (429). Retrying in {backoff:.1f}s (attempt {attempt}/{max_attempts})...")
                    time.sleep(max(0.5, backoff))
                    last_http_error = RuntimeError(f"NVIDIA NIM request failed (429): {detail[:500]}")
                    continue

                if e.code == 429:
                    raise RuntimeError(
                        "NVIDIA NIM request failed (429): Too Many Requests. "
                        "Try again in a minute, reduce Funny->Explain batch size, or disable Kimi thinking mode."
                    ) from e
                raise RuntimeError(f"NVIDIA NIM request failed ({e.code}): {detail[:500]}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Could not reach NVIDIA NIM at {base_url}: {e}") from e
            except Exception as e:
                raise RuntimeError(f"NVIDIA NIM request failed: {e}") from e

        if last_http_error is not None:
            raise last_http_error

        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") if choices else {}
            content = (message or {}).get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").lower()
                        # Some reasoning-capable responses return structured content parts
                        # that include reasoning/thinking chunks alongside the final answer.
                        if item_type and ("reason" in item_type or "think" in item_type):
                            continue
                        text_part = item.get("text")
                        if isinstance(text_part, str):
                            parts.append(text_part)
                text = "\n".join(parts).strip()
            else:
                text = str(content or "").strip()
            if not text:
                # Some thinking-mode responses may carry answer/reasoning in alternate fields.
                text = str((message or {}).get("reasoning_content") or "").strip()
        except Exception:
            text = raw.strip()

        text = self._strip_llm_reasoning_artifacts(text)
        if not text:
            raise RuntimeError("NVIDIA NIM returned an empty response.")
        return text

    def _ollama_generate_text(
        self,
        prompt: str,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        """Call the configured LLM provider and return raw generated text."""
        provider = clone_svc._normalize_provider(provider_override) or self._get_llm_provider()
        if provider == "nvidia_nim":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._nvidia_nim_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "openai":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="OpenAI",
                base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/"),
                api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "anthropic":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._anthropic_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "gemini":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._gemini_generate_text(
                prompt,
                model_override=model_override,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "groq":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="Groq",
                base_url=(os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai").rstrip("/"),
                api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider == "openrouter":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="OpenRouter",
                base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api").rstrip("/"),
                api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
                extra_headers={
                    "HTTP-Referer": (os.getenv("OPENROUTER_REFERER") or "").strip(),
                    "X-Title": (os.getenv("OPENROUTER_TITLE") or "Chatalogue").strip(),
                },
            )

        if provider == "xai":
            if not self._is_llm_enabled():
                raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")
            return self._openai_compatible_generate_text(
                provider_name="xAI",
                base_url=(os.getenv("XAI_BASE_URL") or "https://api.x.ai").rstrip("/"),
                api_key=(os.getenv("XAI_API_KEY") or "").strip(),
                model=str(model_override or "").strip() or (os.getenv("XAI_MODEL") or "grok-2").strip(),
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
            )

        if provider != "ollama":
            raise RuntimeError(
                "Unsupported LLM provider "
                f"'{provider}'. Select one of: ollama, nvidia_nim, openai, anthropic, gemini, groq, openrouter, xai."
            )

        """Call Ollama and return raw generated text."""

        if not self._is_llm_enabled():
            raise RuntimeError("LLM summaries are disabled. Enable LLM in Settings first.")

        base_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        model = str(model_override or "").strip() or (os.getenv("OLLAMA_MODEL") or "mistral").strip()
        if not model:
            raise RuntimeError("OLLAMA_MODEL is not configured.")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Reasoning-capable models (e.g., qwen3.5) may emit only `thinking`
            # and leave `response` empty when think mode is on. Force final-answer
            # generation for pipeline tasks that need parseable output.
            "think": False,
            "chat_template_kwargs": {"thinking": False},
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {base_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        text = ""
        try:
            data = json.loads(raw)
        except Exception:
            data = None
            text = raw.strip()

        if isinstance(data, dict):
            model_error = str(data.get("error") or "").strip()
            if model_error:
                raise RuntimeError(f"Ollama error: {model_error}")

            text = str(data.get("response") or "").strip()
            if not text:
                thinking = str(data.get("thinking") or "").strip()
                if thinking:
                    raise RuntimeError(
                        "Ollama returned no final response (thinking-only output). "
                        "For qwen3.5 models, disable reasoning mode (think=false)."
                    )

        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def generate_clone_text(
        self,
        prompt: str,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 180,
        timeout_seconds: int = 90,
    ) -> str:
        return self._ollama_generate_text(
            prompt,
            provider_override=provider_override,
            model_override=model_override,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
        )

    def _strip_llm_reasoning_artifacts(self, text: str) -> str:
        """Remove common reasoning/thinking wrappers/preambles from LLM output."""

        if not text:
            return ""

        cleaned = str(text).strip()
        if not cleaned:
            return ""

        # Remove explicit thinking blocks if the provider includes them inline.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
        if "</think>" in cleaned.lower():
            cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1].strip()
        cleaned = re.sub(r"^\s*```(?:thinking|reasoning)\s*[\s\S]*?```\s*", "", cleaned, flags=re.IGNORECASE).strip()

        lowered = cleaned.lower()
        meta_start = lowered.startswith((
            "the user wants me to",
            "the user asked me to",
            "i need to analyze",
            "first, i need to",
            "let me analyze",
            "i should analyze",
        ))

        if meta_start:
            # If a likely answer cue exists later, jump to it.
            cue_patterns = [
                r"\blikely joke\b",
                r"\bthe joke likely\b",
                r"\bthis laugh is likely\b",
                r"\bthe humor is likely\b",
                r"\bsummary\s*:",
                r"\bmost likely\b",
            ]
            best_idx = None
            for pat in cue_patterns:
                m = re.search(pat, lowered)
                if m and m.start() > 40:
                    best_idx = m.start() if best_idx is None else min(best_idx, m.start())
            if best_idx is not None:
                cleaned = cleaned[best_idx:].lstrip(":- \n\t")
            else:
                # Drop leading introspection sentences and keep the first actual answer-ish sentence.
                parts = re.split(r"(?<=[.!?])\s+", cleaned)
                kept = []
                skipping = True
                for part in parts:
                    p = part.strip()
                    if not p:
                        continue
                    p_low = p.lower()
                    is_meta = (
                        p_low.startswith(("the user wants me to", "the user asked me to", "first, i need to", "i need to", "let me", "i should"))
                        or "transcript context provided" in p_low
                    )
                    if skipping and is_meta:
                        continue
                    skipping = False
                    kept.append(p)
                if kept:
                    cleaned = " ".join(kept).strip()

        # Normalize whitespace after stripping.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_ollama_summary_confidence(self, text: str, *, max_summary_chars: int = 600) -> tuple[str, str]:
        """Parse summary/confidence JSON with robust fallback handling."""

        text = self._strip_llm_reasoning_artifacts(text)
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Best-effort JSON extraction if the model wraps JSON in prose/fences.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or "").strip()
            confidence = str(parsed.get("confidence") or "low").strip().lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            if summary:
                return summary[:max_summary_chars], confidence

        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        summary_match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.IGNORECASE | re.DOTALL)
        conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        if summary_match:
            try:
                recovered_summary = json.loads(f'"{summary_match.group(1)}"')
            except Exception:
                recovered_summary = summary_match.group(1).encode("utf-8", "ignore").decode("unicode_escape", "ignore")
            recovered_summary = str(recovered_summary).strip()
            recovered_conf = (conf_match.group(1).strip().lower() if conf_match else "low")
            if recovered_conf not in {"low", "medium", "high"}:
                recovered_conf = "low"
            if recovered_summary:
                return recovered_summary[:max_summary_chars], recovered_conf

        cleaned = re.sub(r'^\s*json\s*', '', cleaned, flags=re.IGNORECASE).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned[:max_summary_chars], "low"
