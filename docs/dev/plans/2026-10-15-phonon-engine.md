# Optional Phonon Engine — Implementation Plan

**Date:** 2026-10-15
**Status:** In progress
**Spec:** `docs/dev/specs/2026-10-15-phonon-engine-design.md`

## Summary

Revision approved 2026-10-16: replace the completed operator-managed HTTP
integration with app-owned installation and an isolated persistent native worker.
The original approval and completed tasks 1–6 remain historical. The owner
explicitly authorized revising the spec and plan; revised implementation approval
passed on 2026-10-16 (`d`). Tasks 7 onward implement this approved revision.
Task 7 native Windows CUDA feasibility passed on 2026-10-16; product work can proceed.

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
| 6 | Run relevant tests/static checks, review final diff with code-reviewer lens, update tracker/handoff and report deployment steps without activating engine or pushing | touched modules and docs | M | ✅ done |

## Revision modules and tracker continuation

The affected-module list above records original tasks 1–6. For this revision:

| # | Task | Files | Size | Status |
| --- | --- | --- | --- | --- |
| 7 | Prove native Windows CUDA in a disposable private environment using exact 1.0.8 inference source and cached weights; record pins/hashes, real short repeated inference, BF16/graphs and exit VRAM evidence. Stop if this gate fails; do not build a Docker/CPU substitute | `scripts/benchmark_asr.py`, `docs/research/2026-10-15-transcription-performance.md` (research artifacts under ignored `backend/data/asr-benchmark/`) | M | ✅ done |
| 8 | Implement isolated optional installer/cache with pinned manifest, progress, lock, staged atomic publication and interrupted/offline recovery; verify scratch installs and concurrent requests | `backend/src/services/ingestion/phonon_runtime.py` (new), `backend/src/services/ingestion/phonon_worker_requirements.txt` (new), `backend/src/services/ingestion/runtime.py`, `backend/src/test_phonon_runtime.py` (new) | M | ✅ done |
| 9 | Package attributed upstream inference worker and serialized versioned IPC; implement owned process load/reuse/shutdown/parent-exit behavior; verify real requests, crash recovery, stderr separation and VRAM release | `backend/src/services/ingestion/phonon_worker.py` (new), `backend/src/services/ingestion/phonon_process.py` (new), `backend/src/services/ingestion/phonon_runtime.py`, `backend/src/services/ingestion/phonon.py`, `backend/src/test_phonon_runtime.py`, `backend/src/services/ingestion/phonon_upstream/` (pinned source and notices, only if redistribution needed) | M | ✅ done |
| 10 | Wire worker into existing app phase and GPU lifecycle, replacing HTTP calls; retain checkpoint/routing semantics; verify repeated isolated app jobs, busy purge, engine switch and diarization handoff | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline_diarize.py`, `backend/src/services/ingestion/cuda_memory.py`, `backend/src/services/ingestion/audio_cleanup.py`, `backend/src/services/ingestion/reconstruction.py`, `backend/src/services/ingestion/llm.py`, `scripts/benchmark_asr.py`, `backend/src/routers/system.py`, `backend/src/main.py`, `backend/src/test_phonon.py`, `backend/src/test_phonon_runtime.py`, `backend/src/test_transcription_routing.py` | L | ✅ done |
| 11 | Replace connection settings with app-owned setup/load/readiness state; preserve old-client no-op compatibility and sanitized responses; browser-verify progress/failure/save/reload/unload and existing engines | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/phonon_runtime.py`, `backend/src/services/ingestion/cuda_memory.py`, `backend/src/services/ingestion/phonon.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/test_phonon_runtime.py`, `backend/src/schemas.py`, `backend/src/routers/settings.py`, `frontend/src/pages/Settings.tsx`, `frontend/e2e/phonon-settings.spec.ts`, `backend/src/test_phonon.py` | M | ✅ done |
| 12 | Document integrated operation and run short/long actual app benchmarks including cold load, warm staged reuse and sequential GPU handoff; update runtime identity/provenance evidence | `docs/phonon-engine.md`, `scripts/benchmark_asr.py`, `docs/research/2026-10-15-transcription-performance.md`, `frontend/src/pages/JobQueue.tsx` (only if metadata requires display adjustment) | M | ☐ todo |
| 13 | Full applicable verification and inline code review; resolve necessary findings, update tracker/handoff and local installation using existing update workflow without changing engine settings; shipping remains separately authorized | spec/plan/handoff, changed modules and tests, `docs/dev/contracts.json` (extend authority mapping only to durable runtime paths after revised approval) | M | ☐ todo |

Each task uses implement → verify → review → resolve necessary findings → tracker
and Handhold; commit only coherent verified slices through existing gates.
Task 7 passed with evidence in the research report and verification log.
Before tasks 8–13 run preflight and read updated authorities. Keep the backend
ML environment unchanged. Resolve exact private-environment and model paths via
runtime.py; record source/dependency hashes from the tested image before installer
code. No arbitrary native-worker output cap or setup timeout is invented.

### Revision test plan

- Installer unit/integration: scratch cache/env, hash mismatch, offline ready cache,
  interrupted write/download, failed replacement retains old version, concurrent
  readiness causes one setup. Inspect existing endpoint timeout/progress patterns
  before wiring long setup. Never count a partial manifest as installed.
- Worker: real load and silent/speech inference, unchanged PID across warm jobs,
  malformed/mismatched protocol replies and process crash, owned shutdown and parent
  death, actual GPU memory release. Verify no TCP listener or external service.
- Pipeline: rerun strict silence/overlap/absolute offsets/resume/interrupted writes,
  old Phonon partial reuse, existing language routing and actual engine provenance.
  Set SQLite environment before imports and use temporary .env/files only.
- Lifecycle: simultaneous load/decode/purge, visible busy result, sequential
  diarization releases worker before loading competing model; staged jobs reuse.
- Browser: Phonon selection offers setup/load/test and status, zero URL/key fields;
  setup progress/retry/offline/error, save/reload, actual queue badge and existing
  engine choices. All writes isolated; no installed backlog or production settings.
- Real measurement: 120s/2880s cached inputs through actual app phase, cold setup/load
  separated from warm jobs, staged and sequential GPU handoff; inspect optimization
  state and timings. Compare original HTTP evidence without promising a speedup.
- Required gates: focused then applicable full backend checks, Ruff, frontend lint
  and TypeScript, isolated browser flows, full SDD validate and contract-impact.
  Do not claim existing live-data destructive E2E suite passed or bypass any gate.

### Revision risks and review

2026-10-16 paired review, inline architect/planner/simplification/code-reviewer,
no subagents. Necessary findings resolved in the revised spec and sequenced above:
native feasibility first, GPU phase ownership, backward-compatible partials and
settings, supervised exit, atomic installer and truthful UI. Artifact review gate
passes. Task 7 now verifies native inference; a subsequent feasibility failure
requires an owner decision. Original tests below prove the HTTP implementation only. Adding CPU/MLX,
Docker management, a new queue or kernel tuning fails the MSW necessity test for
this revision and is excluded. Revised owner approval passed on 2026-10-16.

## Test plan (historical HTTP implementation)

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

## Risks (historical HTTP implementation)

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

Revision: owner approval and task 7 native feasibility passed on 2026-10-16. The following
paragraph records the original design only.


No blocking technical questions. Owner plan approval is recorded. No promise that
the currently measured decoder ratio equals the final integrated speedup.

## Adversarial review (historical approval)

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

- 2026-10-15 final verification: owner-authorized separate lint cleanup committed as
  88e2517 after all hooks passed; no lint rules were weakened. Frontend lint has
  zero errors and 43 existing non-blocking warnings. TypeScript build PASS;
  four isolated error/hook/settings/queue checks PASS. Stable full backend suite:
  89 passed, one upstream warning. One earlier run overlapped a pre-commit stash
  and imported baseline settings; discarded that invalid run and reran with the
  checkout stable. Ruff, full SDD validation and diff whitespace checks PASS.
- Inline code-reviewer lens applied to cleanup and feature; no subagents. Required
  findings resolved and regression-checked; no necessary code finding remains.
  Commit hooks also enforce required whitespace normalization. No production
  settings activation, media reprocessing, paid action, push or deployment.
- Tasks 1–6 complete. Source remains in active SDD folders until authorized shipping
  and archival. Contract-impact validation follows the feature commit; no passing
  claim is made before that check runs. Existing live-data E2E tests remain outside
  this verification environment; four isolated checks cover the changed flows.

- Final local checkpoint: 76976c0 Phonon feature, 88e2517 separate lint cleanup,
  942ab4b authority/process repair. All commit hooks passed with no bypass.
  Actual contract-impact CLI from 9482348 through the feature commit PASS using
  its spec/ADR acknowledgement; full SDD validation PASS. Tracker is 6/6 complete;
  active artifact status is retained until authorized push/deploy/archive. Only
  nine unrelated initial dirty paths remain. Temporary verification servers stopped.

- 2026-10-16: owner chose revise spec/plan for full integration after seeing the
  operator-managed connection UI. Revised artifacts reviewed together inline;
  no implementation changes or native feasibility runs performed in this revision.
- 2026-10-16 revised planning verification: full `python scripts/check_sdd_docs.py validate` PASS; `git diff --check` PASS. Tracker reports 6/13 complete. Four scoped planning/handoff files changed; nine unrelated existing paths preserved. Product tests and native benchmarks were not rerun because this revision changes documentation only.

- 2026-10-16: owner replied `d` to approve the revised integration plan. Proceed to planning checkpoint and task 7; no repeat approval required.

- 2026-10-16 task 7 PASS: native Windows CUDA unmodified upstream source; BF16/K16/six buckets, repeated real 120s inference with valid timings, actual child exit checked. Reproducible native benchmark added and exercised; Ruff PASS. Details, cold-load costs, package differences and WDDM limits in research report. No product runtime or production settings changed.

- Task 8 in progress: versioned private installer/cache, hash-pinned wheel closure, atomic publication, download resume, installation lock and recovery implemented. Fifteen focused installer tests PASS; Ruff PASS. First real scratch installation passed pip check but failed model import because the generic SoundFile wheel omitted its Windows DLL. Corrected to the official pinned Windows wheel; added mandatory native-import gate before publication. Fresh corrected installation is running, so task 8 remains in progress.

- 2026-10-16 task 8 PASS: corrected clean standalone installation, full pinned/hash-verified wheel closure, pip check and pre-publication model/DLL import PASS. Post-publication interpreter imports ParakeetForTDT/SoundFile, sees CUDA, has user site disabled and no backend package path; Torch 2.11.0+cu128 / Transformers 5.17.0 / NumPy 2.4.6. Fifteen installer tests, Ruff, full SDD validation and whitespace checks PASS. Inline review resolved the missing Windows DLL and immutable-lock race; no necessary finding remains. Backend environment/defaults/settings remain unchanged.

- 2026-10-16 task 9 PASS: owned stdin/stdout worker, immutable source verification/notices, serialized inference, fresh explicit readiness, failure/deadline cleanup, busy-unload refusal and parent-exit ownership. Twenty-nine installer/worker tests PASS, including actual Windows venv child death on abrupt parent exit. Real standalone GPU worker reused PID 42932 across readiness and two speech requests (55.729s startup; 4.745/4.691s requests), valid word clocks, BF16/K16/six buckets; actual interpreter exit verified on unload and temp audio removed. Inline review resolved suspended-launcher ownership race, shutdown concurrency and response validation. Ruff and SDD checks PASS; full app phase wiring remains task 10.

- 2026-10-16 task 10 PASS: actual transcription now uses one service-owned manager, not URL/key inference. Whisper/Parakeet/diarization, VoiceFixer, reconstruction TTS and local Ollama handoff release the worker; existing purge delegates to the new lifecycle and main shutdown stops its owned child. Fixed false end-of-job eviction: retained child VRAM is not a host allocator leak; host allocator checks/thresholds and other-engine behavior remain. Full backend suite 126 PASS (one upstream warning); subsequent isolated diarization-handoff/recovery tests 4 PASS. Ruff and SDD PASS. Real isolated app phases: startup 44.599s, 120s phases 5.780/6.824s, same PID 18868, Phonon provenance and 383 words/no structural timing issues each. Language routing disabled only in that benchmark process; production settings untouched. Settings connection UI/test compatibility remains for task 11.

- 2026-10-16 task 11 PASS: no URL/key UI or HTTP inference/probe remains. Model status is read-only; asynchronous setup/load/test/unload uses the app-owned manager. Legacy connection inputs are accepted but ignored, saved secrets retained and never returned. GPU work is rechecked after setup; CPU-only installation does not block other engines; internal handoff waits, explicit purge reports busy. Focused backend 69 PASS plus 3 new concurrency cases PASS. TypeScript PASS; frontend lint zero errors/43 unchanged warnings; Ruff PASS. Chromium settings/queue 2 PASS with all API requests intercepted (stdout reporter used after unrelated HTML-report folder EPERM). Live Playwright CLI against an isolated backend verified real RTX5090/BFLOAT16 readiness, unload and engine save/reload using SQLite memory and a temporary .env. Screenshot output/playwright/phonon-ready.png; evidence backend/data/asr-benchmark/phonon-ui-evidence.json confirms model stopped and legacy secret retained. No production settings, media or workers used.
