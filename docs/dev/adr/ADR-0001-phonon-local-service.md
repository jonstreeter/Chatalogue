# ADR-0001: Own optional Phonon through a persistent private worker

**Date:** 2026-10-15
**Status:** Accepted
**Deciders:** Jon (approved 2026-10-15); Codex (proposal)

---

## Accepted replacement — 2026-10-16

The owner explicitly requested full app integration and authorized revision of
spec/plan. This accepted amendment defines an app-owned private native worker using
stdin/stdout, app-managed installation/cache and real load/unload/shutdown. No
URL/key UI, Docker prerequisite or operator-managed server. Preserve upstream
inference and strict checkpoints. The revised spec governs the proposal details.
The original decision below is historical; the app-owned implementation replaces it;
the owner separately approved this replacement on 2026-10-16 (`d`). This
amendment is now the decision and the original is historical only.

Alternatives: embedding into the backend conflicts with its pinned ML stack;
app-managed Docker preserves the tested Linux path but retains the prerequisite
the owner rejected; an isolated native worker is recommended subject to a hard
Windows CUDA feasibility gate (passed on 2026-10-16). If that gate fails, stop before downstream product
work and seek an owner decision. Do not silently fall back to CPU or Docker.

Consequences: app owns installation, progress, concurrent lifecycle and parent-exit
cleanup. Warm reuse applies while Phonon owns the GPU; sequential diarization may
require reload, staged transcription can reuse across episodes. No native speed
or platform-support claim precedes measurement. Rollback is selecting an existing
engine; preserve model caches, checkpoints and deprecated settings without using
external connection values. No DB migration or automatic media processing.

## Context (original accepted decision)

The user requested an optional faster engine for very large channel backlogs.
Measured Phonon CUDA startup is roughly 33–35 seconds per process, while warm
120-second requests finish in roughly 1.3 seconds. The verified server provides
word timing through the same local API across CUDA/CPU/MLX runtimes. The desktop
backend already carries a carefully pinned NeMo/PyTorch stack and a separate
local Ollama service precedent. CUDA 1.0.8 imposes a 32 MiB HTTP body limit and
two-hour per-input limit; these are verified runtime constraints, not proposed
application episode limits.

## Decision

Connect an explicitly selected Phonon engine to an operator-managed persistent
loopback server. Reuse one loaded model across requests/jobs and keep upstream
optimized BF16/CUDA graph defaults. Bound and checkpoint request chunks using
existing pipeline objects and storage. Keep production app defaults and existing
engines unchanged. Detailed contract: `docs/dev/specs/2026-10-15-phonon-engine-design.md`.

## Alternatives considered

1. **CLI/container per episode** — simple transport, but measured startup would
   repeatedly consume the backlog throughput benefit and complicate warm behavior.
2. **Embed Phonon Python/CUDA internals in the backend** — direct calls and shared
   lifecycle, but changes a pinned ML environment, lacks a validated native Windows
   GPU deployment path and creates custom runtime maintenance without proven gain.
3. **Persistent local service (chosen)** — existing optimized upstream runtime,
   amortized startup and separate dependencies; requires local setup and honest
   independent GPU lifecycle handling.

## Consequences

- Batch backlogs reuse model weights and warmed graphs across episode boundaries.
- Phonon remains optional, with one URL/key and testable readiness.
- Long recordings need request chunking and compatible checkpoint/offset handling.
- Host model unload/purge does not unload sidecar VRAM. Operators control server
  startup/shutdown, and Docker remains an optional runtime prerequisite.
- No automatic server manager, custom kernel fork, remote audio service or new
  database layout is introduced.

## What would change our mind

A supported native Windows GPU runtime that fits the pinned backend and improves
measured integrated backlog performance, or a persistent server boundary that
prevents reliable word timing, cancellation/resume or required GPU coexistence.

## Rollback

Select the previously configured engine and stop the optional Phonon service.
Existing transcript/checkpoint formats remain readable; no migration rollback.
