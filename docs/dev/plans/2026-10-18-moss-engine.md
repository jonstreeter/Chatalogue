# Optional MOSS Engine — Implementation Plan

**Date:** 2026-10-18
**Status:** Blocked
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`

## Summary

Add an app-managed optional MOSS engine using Phonon's private-worker pattern.
Use native MOSS speaker turns and existing named voice-profile matching across
recordings. Initial behavior is segment-timed, single-input up to the upstream
90-minute range, with full-input inference retry. No product implementation starts
before owner plan approval. No product UI integration proceeds before the native
runtime feasibility gate passes.

## Affected modules

- `backend/src/services/ingestion/moss.py` — strict native segment validation.
- `backend/src/services/ingestion/moss_runtime.py`, `moss_worker.py`,
  `moss_worker_requirements.txt` — isolated install, inference and lifecycle.
- `backend/src/services/ingestion/phonon_process.py` — inspect process supervision;
  extract only reusable ownership machinery if needed, preserving Phonon contract.
- `backend/src/services/ingestion/service.py`, `runtime.py`, `cuda_memory.py`,
  `backend/src/main.py` — owned manager, GPU handoff, purge, shutdown.
- `backend/src/services/ingestion/transcription.py`, `pipeline_transcribe.py`,
  `pipeline.py` — selection/routing, complete native checkpoint and retry.
- `backend/src/services/ingestion/pipeline_diarize.py`, `speaker_identity.py` —
  native speaker mapping, existing matching and overlap-safe sample selection.
- `backend/src/schemas.py`, `backend/src/routers/settings.py` — additive engine
  and model controls using current endpoints.
- `frontend/src/pages/Settings.tsx`, `components/SetupWizard.tsx`,
  `pages/JobQueue.tsx` — selection, capability notice, setup status and badge.
- `backend/src/test_moss.py`, `backend/src/test_moss_runtime.py`,
  `backend/src/test_moss_pipeline.py`, `frontend/e2e/moss.spec.ts` — scoped checks.
- `docs/dev/contracts.json`, `docs/dev/adr/ADR-0002-moss-managed-engine.md`,
  `docs/research/2026-10-18-moss-runtime.md` — accepted authority and measured proof.

## Task tracker (source of truth for progress)

| # | Task | Files | Size | Status |
| --- | --- | --- | --- | --- |
| 1 | Prove native Windows CUDA load/decode with pinned MOSS source/model in an isolated environment; record dependency pins, parse completeness, overlapping example, cold/warm timing, VRAM and process exit. Stop on failure before task 2. | `docs/research/2026-10-18-moss-runtime.md`, ignored scratch runtime | M | ⛔ blocked: strict source-bound timestamp failure |
| 2 | Record approved architecture/registry mapping; implement overlap-aware parser/result validation and complete-generation checks with malformed/nested-clock tests. | `docs/dev/adr/ADR-0002-moss-managed-engine.md`, `docs/dev/contracts.json`, `backend/src/services/ingestion/moss.py`, `backend/src/test_moss.py` | M | ☐ todo |
| 3 | Add isolated managed installer/worker using task-1 pins, process ownership, observable setup and offline cache; test concurrent setup, interruption, busy unload, worker death and parent-exit cleanup. | `backend/src/services/ingestion/moss_runtime.py`, `backend/src/services/ingestion/moss_worker.py`, `backend/src/services/ingestion/moss_worker_requirements.txt`, `backend/src/services/ingestion/phonon_process.py`, `backend/src/test_moss_runtime.py` | L | ☐ todo |
| 4 | Wire manager/GPU lifecycle, settings engine-test actions and explicit multilingual MOSS route; verify no default change or production job creation. | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/runtime.py`, `backend/src/services/ingestion/cuda_memory.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/main.py`, `backend/src/schemas.py`, `backend/src/routers/settings.py`, `backend/src/test_moss_runtime.py` | M | ☐ todo |
| 5 | Add full-input transcription and atomic native-label checkpoint, duration/context rejection, provenance and retry semantics. Test old cache reuse, mixed-engine partial rejection, failure/cancellation and forced reprocessing. | `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline.py`, `backend/src/test_moss_pipeline.py` | M | ☐ todo |
| 6 | Reuse identity/profile matching with explicit native speaker attribution and overlap-safe samples; retain standard save/cleanup. Test two recordings matching saved names, nested overlapping speakers, no clean sample and missing embedding model. | `backend/src/services/ingestion/pipeline_diarize.py`, `backend/src/services/ingestion/speaker_identity.py`, `backend/src/test_moss_pipeline.py` | L | ☐ todo |
| 7 | Add engine option, setup/status/unload controls, segment/90-minute notice and queue badge; browser-verify save/reload, failure recovery and segment playback without word clocks. | `frontend/src/pages/Settings.tsx`, `frontend/src/components/SetupWizard.tsx`, `frontend/src/pages/JobQueue.tsx`, `frontend/e2e/moss.spec.ts` | M | ☐ todo |
| 8 | Run isolated real app-phase short/long audio, identity across recordings, sequential GPU release/staged reuse; run workflow checks, record evidence and prepare local closeout. | `docs/research/2026-10-18-moss-runtime.md`, this plan and spec | M | ☐ todo |

## Test plan

- **Unit:** strict complete parser, native labels, finite source-bounded clocks,
  overlapping/nested intervals, invalid/truncated output and context exhaustion.
- **Integration:** in-memory SQLite and temporary paths set before imports;
  production .env/DB/jobs untouched. Test settings/route controls, checkpoints,
  profile matching, missing voice model, mixed audio enrollment avoidance,
  cancellation, process cleanup and purge/engine switching. Existing Phonon,
  Parakeet, Whisper, cached-transcript and diarization tests remain green.
- **Manual / browser:** use test backend/mocked API, select MOSS in Settings and
  SetupWizard, observe setup/load/error/unload, save/reload selection, queue badge,
  timed-segment playback/search/export. Verify both speakers survive overlap.
- **Real runtime:** cached research audio with isolated store; compare cold load,
  warm inference, full identification/persistence time and VRAM. Use saved named
  profiles from one test recording in another. No accuracy claim without corrected
  reference; no invented speed threshold. Full episode input is not a throughput
  promise derived from the upstream H100 numbers.
- **Final checks:** `npx tsc -b`, `npm run lint`, `npm run test:e2e` from frontend;
  `ruff check .` from root; backend `.venv` `python -m pytest src -q`;
  `python scripts/check_sdd_docs.py validate`, `git diff --check`.

## Risks

- Native dependency/runtime failure — architect; task 1 hard gate before product.
- Cross-talk attribution loss or contaminated identity — architect/code-reviewer;
  task 6 native labels and clean samples, real cross-recording verification.
- Segment timing and full-input restart differ from Phonon — planner; explicit
  spec/UI notice and approval, no invented word alignment or hidden fallback.
- Worker process leaks and overlapping GPU loads — code-reviewer; lifecycle tests
  and owned-process exit verification before competing model load.
- Overgeneralization/copying the entire Phonon stack — simplification-expert;
  reuse process ownership without widening engine architecture unnecessarily.

## Open questions

Owner approved this reviewed plan and its explicit initial capability tradeoffs
on 2026-10-18. The required Super Looper workflow was found in the user's
`jonstreeter/dev-process-kit` repository; that earlier blocker is resolved.
Task 1 proved native Windows CUDA/SDPA BF16 load and complete ordinary-fixture
decodes, but the synthetic overlap fixture emitted 9.01s for a 9.00s source.
Strict validation rejected it; the exact diagnostic rerun reproduced it.
Task 1 remains blocked and tasks 2–8 are unstarted. Owner decision is required
before further attempts or any contract change; no clamping/tolerance, fallback
or product implementation is authorized by this result. See the research report
and active handoff. Do not ask for original plan approval again.

## Adversarial review

2026-10-18: all four reviewer lenses applied inline to spec and plan; no subagents.
Spec findings and resolutions are recorded in its Adversarial review section.
Ordering gates runtime first, then parser/runtime, API, checkpoint, identity, UI
and real integration. Every tracker task has specific verification evidence.
Final planning review passed; owner approval received and task 1 executed.
Runtime acceptance remains blocked as recorded below. No tests or
runtime feasibility are claimed complete by the planning review.
Remember to follow the MSW deletion rule for all claims — no exceptions.

## Verification log

- 2026-10-18 resumed task 1: reused existing setup processes; isolated Torch
  2.10.0+cu128/Transformers 5.6.0 CUDA imports passed. Immutable source/model and
  complete dependency freeze recorded in `docs/research/2026-10-18-moss-runtime.md`.
  Two fresh ordinary-audio processes completed eight strict decodes; measured
  cold loads, warm repeats, memory and exit 0/VRAM release. Synthetic 9s overlap
  generated native overlapping turns but ended at 9.01s: strict clock assertion
  failed; one diagnostic capture-only rerun failed identically (exit 1). No
  workaround or task 2 started. Gate FAIL; owner decision needed. SDD validation
  and whitespace verification are recorded in the active handoff.

- 2026-10-18 continuation: user directed discovery to jonstreeter/dev-process-kit.
  Loaded docs/dev/workflows/super-looper.md from its master branch through GitHub.
  Local SDD implementation exists but omits this workflow. Using the actual source
  workflow with one fresh worker per task; no parallel workers or product code
  in the conductor. Earlier missing-workflow blocker resolved; task 1 starts now.

- 2026-10-18 execution attempt: owner approval received. Rechecked 12 dirty paths,
  main at 534a7c6c and no stashes. Read requested MOSS documents, Phonon ADR and
  archived spec. Documentation-path preflight returned no mapped contract.
  Superlooper was absent from available skills, local skill/plugin searches and
  plugin search results. Stopped as requested before runtime setup or inference;
  task 1 remains untested and all eight tasks remain incomplete. No product or
  unrelated files changed. See active handoff for the prerequisite blocker.

- 2026-10-18: repository status and contract preflight read; Phonon ADR and shipped
  spec reviewed. Nine pre-existing docs/tooling paths remain untouched.
- 2026-10-18: spec-only and paired planning review performed inline; design findings
  resolved in spec/tasks. No application changes or model installations performed.
