# Optional Phonon Engine — Design Spec

**Date:** 2026-10-15
**Status:** Approved for implementation
**Owner:** Jon / Codex
**Plan:** `docs/dev/plans/2026-10-15-phonon-engine.md`
**Scope:** Add Phonon-2 as an explicitly selected English transcription option through a persistent local runtime. Preserve completed transcript, diarization, identity, queue, and database formats; extend engine-valued settings/provenance and add backward-compatible Phonon partial-checkpoint metadata.

---

## Motivation

The owner requested an optional Phonon engine for channels with hundreds of hours
of recordings, including verified performance improvements. The local comparison
in `docs/research/2026-10-15-transcription-performance.md` found approximately
1.6x better long-audio decoder timing than the configured Parakeet v2 path, with
different instrumentation boundaries. Recognition accuracy remains unmeasured;
the owner explicitly chose an optional engine rather than waiting for a quality
comparison. Keep existing engine selection and defaults.

Launching the CLI/container per episode costs roughly 33–35 seconds in measured
startup. A persistent authenticated local server was verified with two 120-second
requests: 1.512 s and 1.267 s including HTTP upload/response, with 384 word timestamps
each. No repeated model loading occurred. This is the primary integration optimization.

## Design tenets

1. Explicit `phonon` selection only. `auto`, `parakeet`, and `whisper` retain
   existing semantics. Phonon remains English-only; normal language routing still
   sends known non-English/multilingual audio to Whisper when enabled.
2. Reuse a separately running local server across chunks and jobs. Never spawn
   a model process/container per episode, embed another ML stack in the backend
   venv, or fork upstream CUDA kernels.
3. Preserve upstream optimizations already observed in CUDA 1.0.8: dense BF16,
   warmed CUDA graphs, 16-step graph decoding, duration buckets, GPU mel processing,
   and energy-gated internal windows. Do not apply Phonon-1-only packed-kernel
   switches to Phonon-2 or claim unsupported batch API throughput.
4. Normalize source audio once to 16 kHz mono PCM16 using existing FFmpeg helpers.
   Bound upload memory and checkpoint progress; do not load whole multi-hour files
   into RAM. Reuse the HTTP client within a transcription job and serialize
   requests to one server (the tested runtime has one GPU decode lock).
5. Preserve valid words and timestamps. Never invent equal-spaced word timings
   for a response that lacks them. Validate finite, ordered, in-range intervals
   and convert to the existing Whisper-style segment/word objects.
6. Scope rollback to selecting the previous engine and stopping the optional
   runtime. No DB migration, destructive reprocessing, automatic engine switch,
   installer-wide Docker dependency, or production `.env` changes during build.

## Data flow

```text
Settings.tsx: explicit phonon + local URL/key
  -> schemas.Settings / POST /settings -> backend .env and in-process settings
  -> _select_transcription_engine / existing _resolve_transcription_route
  -> _process_transcribe_phase -> phonon.py adapter
  -> one persistent local /v1/audio/transcriptions server
  -> validated word/segment timestamps + absolute chunk offsets
  -> existing partial/raw checkpoint helpers -> unchanged diarization phase
  -> job provenance / JobQueue.tsx displays actual phonon or routed whisper
```

## Interface contracts

### Client → API

- Existing `Settings.transcription_engine` accepts and round-trips `phonon`.
  Existing values/defaults are unchanged; audit normalization whitelists in API,
  service, checkpoint loading, test endpoint, and frontend.
- New settings: `phonon_url` / `PHONON_URL`, default
  `http://127.0.0.1:8010`; `phonon_api_key` / `PHONON_API_KEY`, default empty;
  `phonon_request_timeout_seconds` / `PHONON_REQUEST_TIMEOUT_SECONDS`, optional
  positive number, default unset. The local runtime has no imposed decode deadline;
  unset means wait for inference. Health/connect checks use httpx's documented
  default connect timeout; inference deadline is owner-configurable, not invented.
- URL accepts loopback HTTP(S) only (`localhost`, loopback IP literals), with
  no userinfo/query/fragment. Do not follow redirects or use environment proxies.
  This release does not upload audio to remote inference providers.
- Existing engine-test request gains optional Phonon URL/key overrides, so testing
  unsaved settings never writes `.env`. For Phonon it checks health/model identity,
  authentication and readiness, without importing/loading host NeMo or starting
  a runtime. Return actual runtime model/version/device/path when supplied;
  distinguish unreachable, unauthorized, loading, and wrong-model errors.
- UI: optional engine entry, essential local URL and masked key controls,
  advanced optional inference deadline, Test connection, and setup instructions.
  Settings save/reload round-trips these fields. No CUDA batch-size/beam controls
  falsely advertised as Phonon tuning knobs.

### Service → runtime / worker

- Pin the verified setup recipe to CUDA image `1.0.8` (digest in research report),
  currently shipping runtime 0.2.11 optimizations. Docs also cover an operator-run
  CPU/MLX `fermion serve phonon-2` using the same API. App never silently substitutes
  native Windows CPU inference for the advertised NVIDIA path.
- Use authenticated multipart POST `/v1/audio/transcriptions` with
  `model=phonon-2`, `response_format=verbose_json`,
  `timestamp_granularities[]=word`. Support runtime word text as `word` or `text`.
  Consume `segments` and `words`; ignore nonessential vendor extensions.
- Authority for upload/file limits is installed
  `/opt/phonon/entrypoint.py`: `MAX_BODY=32*1024*1024`,
  `MAX_LONG_SECONDS=2*3600`, `MAX_SECONDS=30`. External chunks default to the
  existing app's 600-second long-audio window: PCM16 mono 16 kHz is 19,200,000
  bytes plus WAV/multipart headers, below 32 MiB and two hours. Validate the
  actual encoded Content-Length before sending. Shorten a chunk if required;
  never stream an unknown-length multipart body (runtime refuses chunked HTTP).
- Reuse the existing 0.35-second long-audio overlap convention. Keep timestamps
  absolute; trim ownership at overlap midpoints to prevent duplicate boundary
  words. Add a boundary test for a word crossing the seam and for resumed offsets.
- Completed external chunks are committed via existing partial checkpoint helpers
  before advancing progress. Check pause/cancel before requests and after responses;
  never swallow the existing pause/cancel exceptions. An in-flight synchronous
  request finishes, fails, or reaches an owner-configured deadline before control
  returns, matching the current chunk-boundary cancellation model.
- Runtime offline/401/403/413/429/503/OOM/invalid JSON/missing word timings must
  produce a specific, non-secret error and preserve prior completed chunks.
  Do not silently use a different engine after partial Phonon output. Retry via
  the existing job retry/resume mechanism; do not create an unbounded hidden retry
  loop or mix engines without provenance. Existing Parakeet fallback behavior
  remains unchanged. Known non-English routing to Whisper is not a runtime failure.
- Store actual `transcription_engine_requested`, `transcription_engine_routed`,
  `transcription_engine_used` using existing keys, with `phonon` as an additive value.
  Optional `phonon_*` details include runtime version/device, chunk count and
  request/decode timing when provided. Strip transient `phonon_*` fields on retry
  alongside existing engine prefixes. Do not log API keys or authorization headers.
- Release host ASR models when entering Phonon where existing switching logic
  already releases competing models. Runtime GPU ownership is independent: do not
  pretend host model-purge endpoints unload its weights. Existing scheduler handles
  jobs; no additional GPU workers or speculative request concurrency are added.

### Persistence / filesystem

- Existing `.env` settings storage; no table/schema changes.
- Existing raw transcript format, partial JSONL segment format and checkpoint
  naming remain. Phonon partial metadata additionally records `engine=phonon`
  and finite `completed_until_seconds`, with existing `segment_count` as the
  committed segment boundary. Last spoken-word end is not the completion boundary:
  this distinction preserves completed silence and overlap ownership on resume.
- For Phonon, create the JSONL even for an entirely silent chunk. Append validated
  segments, then atomically replace metadata only after the append succeeds. A
  failed checkpoint write is an error, not a logged-and-ignored success. The
  Phonon load path reads only the committed segment count and removes any
  uncommitted suffix before appending on retry. If committed metadata claims
  missing segments, reject that checkpoint visibly rather than skipping audio.
  Existing engine callers keep their current helper behavior.
- Phonon resumes use the explicit completed boundary exactly once. Reject an
  existing partial checkpoint from an unknown/other engine without deleting it;
  the operator may resume with its prior engine or use the existing deliberate
  full-retranscription workflow. Preserve engine provenance when reusing completed
  transcripts. Old checkpoints remain readable by their existing engine paths.
- Input normalization/chunk files remain under existing temporary directories and
  are removed after use. Existing production audio and completed transcripts are
  not rewritten merely by selecting or testing the engine.
- Runtime model cache remains owned by its separately installed service. Setup
  downloads weights once into `phonon-cache`, bound to loopback with a key for
  Docker's internal non-loopback bind. Never commit generated keys.

## Edge cases & failure modes

- Silence: accept empty text with empty word/segment lists; advance by completed
  chunk duration so retry does not repeatedly decode silence. Nonempty text without
  usable word timing is rejected; no false-success episode completion.
- Truncated/invalid timings: reject non-finite/out-of-range intervals, mismatched
  duration or explicit truncation; keep the last valid checkpoint. Preserve short
  final chunks and offset clipping.
- Resume: combine existing segments only once, start at the persisted completion
  boundary, offset new words once, and avoid completed-raw reuse during a requested
  force-retranscription. Do not re-run language/model setup per external chunk.
- Busy/dead server: actionable job error, partial state retained; settings test
  identifies prerequisites before queueing a large backlog. A configured deadline
  bounds inference if the operator needs that behavior.
- Device sharing: persistent GPU service consumes VRAM during diarization/cloning.
  Document existing staged/sequential behavior and how to stop the optional service.
  Do not change the scheduler or automatically weaken diarization accuracy.
- Save/test operations must not change current engine or erase keys accidentally.
  Old clients that omit new settings receive safe defaults; changing engines never
  processes existing recordings by itself.

## Adversarial review

2026-10-15, inline architect, planner, simplification, and code-reviewer lenses;
no subagents. Spec-only review and final spec+plan review both completed.

- **BLOCKER, architect:** per-job CLI would negate backlog performance. Resolved:
  persistent operator-run local service; no per-episode process/container startup.
- **BLOCKER, code reviewer:** whole episode uploads exceed 32 MiB and two-hour
  runtime bounds. Resolved: measured transport contract, bounded PCM chunks,
  Content-Length verification, and explicit multi-hour/resume tests.
- **HIGH, planner:** hardcoded engine unions lose provenance/checkpoint recognition.
  Resolved: plan task 3 audits all reader/normalizer/display/retry locations.
- **BLOCKER, final paired review:** existing partial resume starts at the last
  spoken segment and cannot represent completed silence; ignored checkpoint write
  errors and uncommitted JSONL tails can lose the true boundary. Resolved: Phonon
  adds committed offset/engine metadata, strict append plus atomic metadata commit,
  committed-count recovery, empty JSONL support and explicit silence/failure tests.
- **HIGH, code reviewer:** failure after partial output can duplicate text if
  fallback restarts the full file. Resolved: no hidden engine fallback for partial
  Phonon failures; preserve checkpoint and retry the remainder.
- **HIGH, architect:** Docker service rejects non-loopback binds without a key.
  Resolved: verified authenticated loopback-published setup and explicit key setting.
- **HIGH, planner:** runtime's model memory outlives host purge actions. Resolved:
  explicit independent lifecycle and device-sharing documentation, no false unload.
- **MEDIUM, simplification:** custom batching, kernel flags, engine process manager,
  hotword UI and new queue are unnecessary. Removed from this feature. Upstream
  server is serial; published batch-128 speed is not an available adapter setting.
- **MEDIUM, code reviewer:** unset timeout permits waiting for stalled inference.
  Explicitly documented; optional positive deadline, health test and retained
  checkpoints provide operator control without inventing a timeout policy.

Final paired review: no necessary design finding remains open. Artifact review
gate passed. On 2026-10-15 the owner explicitly replied "d approve" and requested
a prompt to implement in a new session. The separate owner plan-approval gate has
passed. No implementation code has been written for this feature yet.

## Out of scope

Changing defaults/auto selection, claiming improved recognition accuracy, updating
diarization models, modifying storage-of-record/auth/tenancy, remote cloud audio
uploads, automatically installing Docker, runtime process orchestration, custom
CUDA kernels, batch-128 serving, hotword tuning UI, or changing language routing.

## Open questions

No blocking technical questions. Owner plan approval was granted on 2026-10-15.

Sources: [runtime/server](https://github.com/fermionresearch/phonon/blob/main/docs/server.md),
[CUDA deployment](https://github.com/fermionresearch/phonon/blob/main/docs/cuda.md),
[release optimizations](https://www.fermionresearch.com/changelog/), and installed
CUDA 1.0.8 help/source/health plus `backend/data/asr-benchmark/phonon-http-probe.json`.
