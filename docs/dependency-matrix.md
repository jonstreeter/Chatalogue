# Dependency Matrix

Last reviewed: March 18, 2026

## Backend

Use the pinned files in `backend/` instead of inline installer package lists:

- `requirements-windows-cu128.txt`: Windows RTX 50xx torch nightly (`2.11.0.dev20260201+cu128`)
- `requirements-macos.txt`: stable torch fallback for macOS and Windows fallback (`2.10.0`)
- `requirements.txt`: shared backend pins
- `requirements-parakeet.txt`: optional Parakeet / NeMo pin (`nemo_toolkit[asr]==2.7.0`)

Key decisions:

- `ctranslate2` moved from `<4.6` to `4.7.1`, which stays inside `faster-whisper 1.2.1`'s supported `<5` range.
- `pyannote.audio` moved to `4.0.4`.
- `lightning` stays on `2.4.0` because that matches the current NeMo 2.7.0 compatibility line better than newer Lightning releases.
- `huggingface-hub` stays on `0.36.2`. The repo was previously floating this to newer major versions without validation.
- `setuptools` is pinned to `80.10.2` while Windows still relies on `pkg_resources` compatibility shims in the transcription stack.

## Frontend

The frontend dependency tree was audited, but I did not force major tooling jumps here.

- Keep the current `React 19` / `Vite 7` / `ESLint 9` line for now.
- Patch/minor updates are available for several packages, but they are outside the Parakeet stabilization path and should be validated separately.
