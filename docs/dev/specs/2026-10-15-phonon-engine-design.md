# Optional Phonon Engine — Design Spec

**Date:** 2026-10-15
**Status:** Approved for implementation
**Owner:** Jon / Codex
**Plan:** `docs/dev/plans/2026-10-15-phonon-engine.md`
**Scope:** Optional Phonon-2 with app-owned installation and model lifecycle; preserve existing transcription, routing, diarization and checkpoint behavior.

## Motivation

The original operator-managed HTTP integration was approved on 2026-10-15 and
implemented in 76976c0. On 2026-10-16 the owner rejected the connection-oriented
experience and explicitly requested revision of this spec and plan for full
integration. Original approval remains historical; the owner approved this revision on 2026-10-16 by replying `d` to its approval Handhold.
The installed UI remains the original implementation until the revised plan is built.

Selecting Phonon should work like selecting another local model: Chatalogue owns
setup, download, loading, reuse and unloading. Users must not operate a server,
enter a URL/API key, install Docker, or diagnose a separate service.

## Design tenets

1. Keep explicit `phonon` selection, existing defaults and language routing.
   English-only Phonon still routes multilingual audio to Whisper when enabled.
2. Use an app-owned child worker in a private Python environment, avoiding changes
   to the backend's pinned NeMo/Transformers stack. No listening network service.
3. Reuse weights and warmed graphs across chunks and consecutive transcription
   jobs. Release at competing GPU phases, explicit unload and app shutdown.
   Sequential transcribe/diarize may require reload between episodes; disclose and
   measure this cost. Staged transcription can retain the worker across episodes.
4. Preserve upstream BF16, CUDA graphs K16, duration buckets, GPU mel and internal
   window behavior. Do not fork kernels or advertise batch-128 throughput.
5. Reuse implemented chunking, strict checkpoints and provenance; replace transport
   and lifecycle rather than rewriting the proven transcription pipeline.

## Feasibility boundary

Upstream documents NVIDIA CUDA through Docker; a supported native Windows CUDA
package has not been established. The backend's Transformers 4.57.1 lacks the
ParakeetForTDT class used by the upstream CUDA implementation. Isolated native
worker feasibility was therefore the first implementation gate. Task 7 passed on
2026-10-16: native Windows inference retained BF16/K16/six buckets and valid
repeated word timings. See the research report for package differences, cold-load
costs and process-exit evidence; full app integration remains unimplemented.

Task 7 must pin the source/dependencies from verified image
`ghcr.io/fermionresearch/phonon-cuda:1.0.8` (digest
`sha256:c37e7b0ae99f6a74a2618ecae3fe9cbbc342c1069e02464eadba9c42d19802d5`),
exercise native Windows CUDA with cached weights and actual audio, and demonstrate
BF16/graph reuse and process-exit VRAM release. Record exact package versions,
source hashes, upstream license/model notices and reproducible commands. Reuse
unmodified upstream inference code where possible; package it with attribution.
If this cannot run without a kernel fork or incompatible driver/backend changes,
stop and report evidence. Do not silently substitute CPU or reintroduce Docker.
That outcome requires an owner decision before downstream implementation.

The initial acceptance platform is this Windows NVIDIA installation. Existing
engines remain available on other platforms. Phonon reports unsupported hardware
clearly there; native CPU/MLX expansion is outside this revision. Do not remove
existing engines or claim untested cross-platform support.

## Data flow

```text
Settings: Phonon-2 -> setup/load/test through existing engine-test operation
  -> app-owned runtime manager -> private worker environment and model cache
  -> dedicated stdin/stdout messages + bounded temporary WAV paths
  -> upstream optimized inference -> validated words/segments
  -> existing absolute offsets, overlap ownership and committed checkpoints
  -> existing diarization and queue provenance
```

## Interface contracts

### Settings and API

- Keep `transcription_engine=phonon` additive; no auto/default changes or DB migration.
- Replace connection controls with model availability, setup/download progress,
  Load/Test and existing unload behavior. Setup/test must not save engine selection
  or enqueue existing recordings. Selecting an installed engine loads lazily on use.
- Existing `/settings/test-engine` request remains compatible; for Phonon it ensures
  setup and load, then runs the one-second silence readiness test and reports actual
  runtime/model/device. Add optional state/progress fields to its existing response;
  use existing progress facilities where available rather than inventing a job queue.
  Inspect route timeout behavior before implementation and return observable setup
  state if setup outlives the request. A reconnect must not start duplicate installs.
- `phonon_action` on the existing engine-test request is `status`, `load` or
  `unload` (default `load`). Status is read-only. Load/test returns observable
  asynchronous state; explicit unload refuses busy work. Internal GPU handoff
  waits for the current operation. CPU-only installation does not block other
  engines; before GPU loading, recheck active pipeline work and release idle host
  models. A refused test preserves an already healthy owned worker.
- Old `phonon_url` and `phonon_api_key` inputs remain accepted as deprecated no-ops
  during compatibility transition. Do not connect to them or erase stored values.
  Responses never reveal saved keys; new UI does not read or display these fields.
  Retain optional positive inference deadline only if used by the worker, under
  advanced model controls. Unset preserves current chunk-boundary cancellation.
- Errors distinguish unsupported hardware, missing/offline download, installation
  failure, loading, inference failure and worker exit, with actionable app guidance.
  Never report ready before a real inference succeeds. No secrets or raw worker
  output in UI errors.

### Installation and ownership

- Place environment, weights and version manifest below app-owned runtime/cache
  directories; implementation resolves their exact paths using existing runtime.py
  conventions. Never import application/database modules inside the worker.
- Install only on explicit setup/test or first actual Phonon use. No app-startup
  download or global engine activation. Show progress and reuse valid cached files.
- Pin the complete tested environment and immutable upstream/model revisions.
  Validate downloads against pinned hashes; stage then atomically publish a complete
  manifest. Interrupted installs are recoverable and never count as installed.
  Serialize concurrent setup using an installation lock; old valid installations
  survive a failed replacement. Offline cached operation requires no network.
- Invoke the private interpreter directly with argument arrays, never shell-built
  commands. No arbitrary executable/path from settings. Parent owns only its worker;
  never kill unrelated Python, Docker, GPU or app processes. No administrator access.

### Worker and lifecycle

- Dedicated stdin/stdout newline-delimited JSON protocol with request ID, protocol
  version and operations load, transcribe, status and shutdown. Capture diagnostics
  on stderr separately. Requests are serialized; one decode at a time.
- Transcribe accepts only manager-created temporary WAVs under the app temp root,
  validated duration and encoded size. Responses use existing validated segment/
  word schema plus actual runtime version/device/optimization metadata. Treat
  malformed output, mismatched IDs and unexpected exit as failures.
- Keep existing 600-second PCM16 mono 16 kHz chunks and 0.35-second overlap.
  Preserve the verified 32 MiB encoded-input and two-hour per-input bounds when
  replacing HTTP, without claiming these limits belong to the native API. Validate
  message/output size against the pinned worker's actual response contract during
  task 7; do not invent an arbitrary protocol cap.
- One manager per ingestion service; lifecycle transitions absent/installing/
  installed/loading/ready/busy/unloading/error are observable. One ownership lock
  coordinates load, decode, purge and shutdown. Purge rejects busy work visibly;
  it never kills an in-flight request while pretending it succeeded. Shutdown can
  terminate its own worker; only completed chunks survive interruption.
- Before diarization or another competing GPU model phase, wait for current decode,
  stop the owned Phonon process and verify exit before loading the next model.
  Integrate the same rule with existing model purge and engine switching. Do not
  infer worker device from the host service's lazily initialized `self.device`.
- An intentionally resident Phonon child is not a host allocator leak. End-of-job
  global free-memory drop alone must not evict it; existing host reserved/allocated
  recovery checks and thresholds remain unchanged.
- Normal shutdown closes IPC and waits for exit. Unexpected parent exit also ends
  the worker (Windows Job Object kill-on-close or equivalent proven ownership).
  No orphan model process or false unloaded status. Failed shutdown is visible;
  use process ownership rather than broad process-name kills.
- Inference cancellation remains at chunk boundaries; explicit configured deadline
  may terminate the owned worker and fail the current chunk. No hidden fallback or
  unbounded automatic retry; next explicit retry may recreate a failed worker.

### Checkpoints and provenance

Keep the implemented format: `engine=phonon`, committed `segment_count` and
`completed_until_seconds`, initial zero-count ownership, fsynced JSONL append then
atomic metadata replacement. Silence advances the completion boundary. Resume
truncates only uncommitted tails; rejects missing committed data or unknown/other
engine partials without deletion. Preserve segment groups for diarization, finite
absolute word times and overlap midpoint ownership. Completed transcript reuse,
force retranscription, retry metadata cleanup and routed/used engine remain intact.
The runtime transport change does not invalidate existing valid Phonon partials.
Temporary audio is removed after use; no production settings/media rewrite in tests.

## Acceptance and measurement

Browser: choose Phonon, observe setup/load/readiness and failure recovery, save and
reload, unload via existing model controls; no connection fields or manual-server
instructions. Existing engine options, language routing and queue badges work.
Backend: isolated SQLite set before imports, temporary files/env, real app phases,
worker crash/setup interruption/concurrent setup/busy purge/parent exit coverage,
strict checkpoint regressions, no production work. Verify no listening service.
Benchmark actual app phases on cached 120s and 2880s audio: cold setup/load separately,
warm repeated jobs in one process, sequential GPU handoff and staged reuse; inspect
BF16/graph state, process identity, VRAM release, word counts and timestamps.
Original HTTP app baseline: 9.813/7.305s short, 46.257/62.129s long; not native-worker
measurements or accuracy certification. Report all measured costs, no invented
speed target. Existing cached evidence stays in the handoff/research report.

## Adversarial review

2026-10-16, inline architect, planner, simplification-expert and code-reviewer;
no subagents. Spec-only followed by paired spec/plan review.

- BLOCKER architect: native Windows CUDA unproven. Resolved in plan ordering:
  task 7 is an evidence gate before product implementation, with explicit stop
  condition; design makes no native performance/support claim before it passes.
- HIGH planner: sequential diarization conflicts with warm reuse. Resolved:
  release at GPU phase boundaries, preserve staged reuse and benchmark both.
- HIGH code-reviewer: old settings, partials and worker exit can lose state.
  Resolved: deprecated no-op settings, unchanged strict commits, owned process
  supervision, parent-exit test and busy-purge exclusion.
- HIGH simplification: hiding a managed Docker service still leaves a prerequisite
  the owner rejected. Removed network runtime, Docker install and dual external mode.
- HIGH planner: interrupted setup and racing readiness requests. Resolved:
  serialized staging/atomic manifest, observable progress and offline cache tests.

No necessary planning finding remains open. Final paired artifact review passes;
revised owner approval and the task-7 native feasibility gate passed on 2026-10-16.
Remember to follow the MSW deletion rule for all claims — no exceptions.

## Out of scope

Remote inference, Docker management, changing backend ML pins, custom kernels,
batch serving, CPU/MLX expansion, scheduler rewrite, new DB schema, recognition
accuracy claims, defaults/routing changes or automatic production reprocessing.

## Open questions

Owner approval of this revision passed on 2026-10-16. Native Windows
CUDA inference feasibility passed task 7 on 2026-10-16; integrated lifecycle and
app performance verification remain in the numbered implementation tasks.

Sources: [upstream CUDA deployment](https://github.com/fermionresearch/phonon/blob/main/docs/cuda.md),
[installation](https://github.com/fermionresearch/phonon/blob/main/docs/install.md),
[CUDA engine](https://github.com/fermionresearch/phonon/blob/main/docker-phonon2/phonon2_cuda_engine.py).
Exact shipped pins must come from verified image 1.0.8, not mutable main.
