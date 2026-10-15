# Optional Phonon Engine — Implementation Plan

**Date:** 2026-10-15
**Status:** In progress
**Spec:** `docs/dev/specs/2026-10-15-phonon-engine-design.md`

## Summary

Add an opt-in Phonon-2 transcription engine through an authenticated persistent
local server. Keep upstream optimized inference warm across a channel backlog,
bound uploads, preserve word timings and resumable progress, and make actual
engine/runtime availability visible. Preserve existing auto, Whisper, Parakeet,
diarization and database behavior. The owner explicitly approved this reviewed
plan on 2026-10-15 ("d approve") and requested implementation in a new session.
The approval gate is satisfied; implementation has not started.

## Affected modules

- `backend/src/services/ingestion/phonon.py` — local client, limits, response validation.
- `backend/src/services/ingestion/service.py` — adapter lifetime without worker startup.
- `backend/src/services/ingestion/transcription.py` — explicit engine selection,
  English routing and health-test dispatch.
- `backend/src/services/ingestion/pipeline_transcribe.py` — chunk iteration,
  pause/resume/progress, compatible word objects and failure preservation.
- `backend/src/services/ingestion/pipeline.py` — completed checkpoint engine reader
  and strict Phonon partial commits/recovery with completed-offset metadata.
- `backend/src/services/ingestion/job_lifecycle.py` — retry transient metadata handling.
- `backend/src/schemas.py`, `backend/src/routers/settings.py` — settings/test contracts.
- `backend/requirements-common.txt` — explicitly pin existing httpx 0.28.1 client.
- `frontend/src/pages/Settings.tsx` — option, URL, masked key, deadline and availability.
- `frontend/src/pages/JobQueue.tsx` — actual engine badge and fallback/routing display.
- `backend/src/test_phonon.py`, `backend/src/test_transcription_routing.py` — boundary,
  transport and cross-engine regressions.
- `frontend/e2e/phonon-settings.spec.ts` — settings save/reload/test and queue display;
  resolve the actual existing e2e layout before creation.
- `docs/phonon-engine.md` — pinned runtime setup, platform choices and operation.
- `scripts/benchmark_asr.py` — integrated Phonon benchmarking through app mode.
- `docs/dev/contracts.json` — register only new adapter path with approved authority
  after owner approval; no existing registry or overlapping authority was found.

## Task tracker (source of truth for progress)

| # | Task | Files | Size | Status |
| --- | --- | --- | --- | --- |
| 1 | After owner approval, record authority and implement local HTTP adapter with URL/auth validation, health/model identity and bounded requests; verify with fake transport | `docs/dev/contracts.json`, `backend/src/services/ingestion/phonon.py`, `backend/requirements-common.txt`, `backend/src/test_phonon.py` | M | ✅ done |
| 2 | Add settings persistence and unsaved-settings engine-test overrides; test missing/new fields and secret-safe errors without production .env changes | `backend/src/schemas.py`, `backend/src/routers/settings.py`, `backend/src/test_phonon.py` | M | ✅ done |
| 3 | Wire explicit selection and actual phase/chunk/progress/resume into existing transcription flow; audit every engine reader/normalizer and keep old-engine behavior | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline.py`, `backend/src/services/ingestion/job_lifecycle.py`, `backend/src/test_transcription_routing.py`, `backend/src/test_phonon.py` | L | ✅ done |
| 4 | Expose optional engine/settings and truthful queue provenance; browser-verify save/reload, health errors and existing engine options | `frontend/src/pages/Settings.tsx`, `frontend/src/pages/JobQueue.tsx`, `frontend/e2e/phonon-settings.spec.ts` | M | ✅ done |
| 5 | Document and exercise persistent CUDA setup and CPU/MLX equivalent; benchmark integrated app on short/long audio and multiple requests without reloading | `docs/phonon-engine.md`, `scripts/benchmark_asr.py`, `docs/research/2026-10-15-transcription-performance.md` | M | ✅ done |
| 6 | Run relevant tests/static checks, review final diff with code-reviewer lens, update tracker/handoff and report deployment steps without activating engine or pushing | touched modules and docs | M | ▶ in progress |

## Test plan

- **Unit:** loopback URL validation, redirect/proxy refusal, auth header not logged;
  finite/ordered/duration-checked word/segment data; silence vs missing word timing;
  real Content-Length bounded below installed 32 MiB limit; no unknown-length
  transfer; overlapping seams, short final chunk, exact absolute resume offsets;
  pause/cancel exceptions, errors after completed chunks and retry metadata cleanup;
  entirely silent chunks, persisted completion offset versus last-word end, atomic
  metadata failure, uncommitted JSONL-tail recovery and unknown-engine partials.
- **Integration:** actual isolated app phase with fake server, in-memory DB, scratch
  .env and files; explicit Phonon actual metadata/checkpoint values; existing
  multilingual override, old auto/Whisper/Parakeet behavior, completed checkpoint
  reuse and force-retranscription. Failed chunk must not mark episode done or lose
  prior completed work. Test settings reads/writes without exposing probe key.
- **Manual / browser:** run mocked settings/job responses; choose Phonon, test unsaved
  URL/key, save and reload, verify selected engine/fields/badge; show unavailable
  runtime and authentication errors. Test normal existing engine selection too.
- **Real runtime:** persistent CUDA 1.0.8 with loopback publication and generated
  uncommitted key. Normalize once, send same short/long recordings through the full
  Phonon app phase, verify timestamps and completed chunks. Repeat jobs and confirm
  health uptime/request counters increase without model reload. Exercise a source
  exceeding one upload and two hours by chunking; use bounded mocked transport for
  byte-boundary/very-long cases to avoid unnecessary repeated GPU work.
- **Checks:** focused pytest then broader backend suite as justified; Ruff on
  changed Python, frontend typecheck/lint and relevant browser tests, SDD validation
  and contract-impact check. Do not commit with an unresolved failing required gate.
  Runtime integration is not a speech accuracy certification.

## Risks

- Word timing corruption/resume duplication — code-reviewer; explicit conversion,
  seam and failure/retry tests before long live runs.
- Runtime deadline intentionally unset by default — architect/planner; configurable
  positive deadline and chunk-boundary cancellation semantics stated in spec.
- Shared GPU memory and runtime startup — architect; independent lifecycle and
  actual health/device information, no per-episode launch or host-unload claims.
- HTTP limits and long recordings — planner; installed runtime limits verified,
  PCM chunk request length tested and measured before queue integration.
- Performance scope creep — simplification; use runtime defaults verified by
  health, no custom CUDA forks or unsupported concurrent/batch serving.
- Runtime dependency availability — planner; optional operator-managed setup, existing
  app installs unaffected. Missing runtime fails visibly with resume state retained.

## Open questions

No blocking technical questions. Owner plan approval is recorded. No promise that
the currently measured decoder ratio equals the final integrated speedup.

## Adversarial review

2026-10-15: architect, planner, simplification-expert and code-reviewer lenses
applied inline; no subagents. Spec-only and final paired review complete. All
necessary BLOCKER/HIGH findings were resolved in the spec and sequenced here:
persistent runtime, byte/file limits, engine whitelist audit, authenticated
Docker setup, strict committed-offset checkpoint preservation/recovery, no mixed-engine silent fallback, and
independent GPU lifecycle. No necessary design finding remains open. Artifact
review gate passed; the separate owner approval gate passed on 2026-10-15.

## Verification log

- 2026-10-15: owner explicitly approved the plan and asked for a new-session
  implementation prompt using super looper and SDD Handhold. Implementation remains
  unstarted. No matching super looper skill was found in repo or installed skill
  roots; the prompt must state a transparent autonomous-loop fallback if unavailable.
- 2026-10-15: contract preflight across settings/schema/service/checkpoint/UI and
  new doc paths → no mapped authority; no existing registry found.
- 2026-10-15: inspected installed CUDA 1.0.8 CLI/source → 32 MiB HTTP body cap,
  two-hour per-input cap, authenticated non-loopback bind, serial GPU lock.
- 2026-10-15: persistent authenticated loopback API feasibility → PASS: health
  confirms BF16, CUDA graph steps 16, warm buckets 6, Phonon-2; two 120-second
  HTTP requests completed in 1.512/1.267 seconds with 384 word timestamps each.
  Only feasibility tooling ran; no implementation or production settings changed.
- 2026-10-15: upstream runtime changelog/source/deployment docs reviewed. CUDA
  1.0.8 is the current documented image; no verified extra Phonon-2 batch/packed
  performance toggle was found. Keep known optimized defaults.
- 2026-10-15: targeted spec/plan header, reciprocal-link, archive-location and
  handoff checks using the repository validator helpers → PASS; `git diff --check`
  → PASS. Full repository validation is not claimed (prior generated-tree traversal
  stalled); no implementation, commit or production configuration change occurred.

- 2026-10-15 implementation attempt: registered the adapter authority in docs/dev/contracts.json. Required mapped preflight stalled in _repository_files(), which traverses the entire checkout before filtering. No adapter code written, no gate bypassed, no full validation claimed. Owner decision required for a scoped process-tooling repair outside the approved engine design. Tracker remains 0/6.

- 2026-10-15: owner authorized validator patch and continuation. Discovery now walks only intended artifact directories while retaining tracked paths and original non-Git fallback. Normal-temp regression suite: 46 passed; real preflight and full SDD validation PASS. Adapter transport suite: 25 passed, Ruff PASS. Task 1 complete; settings/pipeline work in progress.

- 2026-10-15: tasks 2–4 verified: settings scratch env round trip/old-client key preservation and unsaved probe; actual isolated app phase failure/resume, silent and >two-hour sources, atomic failure/tail recovery, segment boundaries, pause/cancel, force retranscription, multilingual routing and retry metadata. Chromium settings and queue tests: 2 passed with every API mocked. Existing frontend engine choices preserved. Inline code review resolved segment grouping for diarization and unknown orphan provenance; no subagents.

- 2026-10-15: task 5 complete: pinned persistent authenticated CUDA setup exercised;
  docs include separate CPU/MLX installation and same API (native performance not
  measured). Final integrated short phases 9.813/7.305 s, long 46.257/62.129 s;
  384/8709 words, 41/1045 upstream-preserving segments, no structural timestamp
  issues. Same runtime uptime and counters increase across jobs; BF16/graphs K16,
  six warm buckets retained. No consistent large integrated speedup established.
- 2026-10-15: full backend suite after review fixes: 86 passed, one upstream
  deprecation warning. Ruff entire repository PASS; frontend typecheck PASS;
  full SDD validation and real mapped preflight PASS; Chromium settings/queue
  checks PASS. Final additions to readiness tests are being checked separately.
- 2026-10-15: task 6 blocked by required frontend lint gate: 205 errors and 43
  warnings across repository. Baseline HEAD copies of Settings/JobQueue have
  identical 15/11 errors and 4/1 warnings respectively to working copies. New
  browser test lint is clean. Per workflows/verify.md, no commit while blocked;
  no bypass or unrelated lint repair. Existing live-data E2E suite not run because
  it triggers channel scans/deletions against an installed app; only fully mocked
  new tests run. Owner decision needed for scope/policy on pre-existing lint gate.
- Test isolation correction: the first transport test collection imported app
  defaults before setting SQLite; PostgreSQL bootstrap removed a stale PID file
  and failed to start. No media processing or production settings writes occurred.
  All subsequent test/benchmark processes explicitly used isolated SQLite.

- Final focused readiness/boundary suite: 47 passed (includes CPU API shape, wrong model and loading); live settings probe returned ready, wrong-key 401 and offline errors as expected. Queue/settings screenshots reviewed. Final native CPU/MLX runtime was not run on this host. Commit/commit-impact check pending lint-gate owner decision; no passing claim for an unrun check.

- 2026-10-15: owner explicitly authorized a separate frontend lint cleanup. Treat it as contained maintenance with no API/storage contract changes, tracked here as the prerequisite for task 6; no new spec required. Replaced untyped catch access with shared guarded error extraction, typed existing data/player/event shapes, removed dead preview UI and unused reads, fixed conditional hook order. ESLint rules/config unchanged. Four isolated regression checks passed, including draft open/close, reconstruction visibility and existing Phonon browser flows. Commit cleanup separately from Phonon.
