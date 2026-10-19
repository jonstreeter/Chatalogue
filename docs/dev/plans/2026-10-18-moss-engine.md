# Optional MOSS Engine — Implementation Plan

**Date:** 2026-10-18
**Status:** In progress
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
| 1 | Prove native Windows CUDA load/decode with pinned MOSS source/model in an isolated environment; record dependency pins, parse completeness, overlapping example, cold/warm timing, VRAM and process exit. Reuse existing proof and rerun retained overlap fixture under spec's moss-eof-v1 rule, retaining raw clocks and correction evidence. Stop on failure before task 2. | `docs/research/2026-10-18-moss-runtime.md`, ignored scratch runtime | M | ✅ done: separate moss-eof-v1 proof PASS; retained runtime evidence complete |
| 2 | Record approved architecture/registry mapping; implement overlap-aware parser/result validation and complete-generation checks with malformed/nested-clock tests, exact moss-eof-v1 boundary/rejection tests and backend validation of raw correction evidence. | `docs/dev/adr/ADR-0002-moss-managed-engine.md`, `docs/dev/contracts.json`, `backend/src/services/ingestion/moss.py`, `backend/src/test_moss.py` | M | ✅ done: strict pure parser/backend evidence boundary; 63 tests PASS |
| 3 | Add isolated managed installer/worker using task-1 pins, process ownership, observable setup and offline cache; test concurrent setup, interruption, busy unload, worker death and parent-exit cleanup. | `backend/src/services/ingestion/moss_runtime.py`, `backend/src/services/ingestion/moss_worker.py`, `backend/src/services/ingestion/moss_worker_requirements.txt`, `backend/src/services/ingestion/phonon_process.py`, `backend/src/test_moss_runtime.py` | L | ☐ todo |
| 4 | Wire manager/GPU lifecycle, settings engine-test actions and explicit multilingual MOSS route; verify no default change or production job creation. | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/runtime.py`, `backend/src/services/ingestion/cuda_memory.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/main.py`, `backend/src/schemas.py`, `backend/src/routers/settings.py`, `backend/src/test_moss_runtime.py` | M | ☐ todo |
| 5 | Add full-input transcription and atomic native-label checkpoint, duration/context rejection, provenance and retry semantics. Preserve raw clock tokens, source sample count/rate and moss-eof-v1 corrections across identity retries without cumulative normalization. Test old cache reuse, mixed-engine partial rejection, failure/cancellation and forced reprocessing. | `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline.py`, `backend/src/test_moss_pipeline.py` | M | ☐ todo |
| 6 | Reuse identity/profile matching with explicit native speaker attribution and overlap-safe samples; retain standard save/cleanup. Test two recordings matching saved names, nested overlapping speakers, no clean sample and missing embedding model. | `backend/src/services/ingestion/pipeline_diarize.py`, `backend/src/services/ingestion/speaker_identity.py`, `backend/src/test_moss_pipeline.py` | L | ☐ todo |
| 7 | Add engine option, setup/status/unload controls, segment/90-minute notice and queue badge; browser-verify save/reload, failure recovery and segment playback without word clocks. | `frontend/src/pages/Settings.tsx`, `frontend/src/components/SetupWizard.tsx`, `frontend/src/pages/JobQueue.tsx`, `frontend/e2e/moss.spec.ts` | M | ☐ todo |
| 8 | Run isolated real app-phase short/long audio, identity across recordings, sequential GPU release/staged reuse; run workflow checks, record evidence and prepare local closeout. | `docs/research/2026-10-18-moss-runtime.md`, this plan and spec | M | ☐ todo |

## Test plan

- **Unit:** strict complete parser, native labels, finite source-bounded clocks,
  overlapping/nested intervals, invalid/truncated output and context exhaustion.
  For moss-eof-v1: unchanged in-range endpoints; exactly 0.01s EOF overrun accepted
  and normalized; 0.010001s rejected; raw negative/reversed/nonfinite clocks,
  out-of-range starts and collapsed corrected spans rejected. Cover non-centisecond
  sample-derived duration, multiple overlapping EOF ends without accumulated
  allowance, invalid segments rejecting the whole output, intact raw provenance,
  and incomplete generation still rejected even with an otherwise eligible end.
- **Integration:** in-memory SQLite and temporary paths set before imports;
  production .env/DB/jobs untouched. Test settings/route controls, checkpoints,
  profile matching, missing voice model, mixed audio enrollment avoidance,
  cancellation, process cleanup and purge/engine switching. Existing Phonon,
  Parakeet, Whisper, cached-transcript and diarization tests remain green.
  Backend rejects tampered worker duration/correction evidence; completed native
  checkpoints preserve raw/normalized endpoints, policy and labels through
  serialization and identity retries without retranscription or second clipping.
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
Owner subsequently requested the explicit tiny-boundary refinement. The spec now
defines moss-eof-v1: end-only EOF normalization up to 0.01s inclusive, exact clock
comparison and preserved raw evidence; larger overruns and invalid spans fail.
Task 1 now passes the separate targeted rerun under that policy (1/8 complete).
Task 2 is complete; tasks 3–8 remain unstarted. Prior ordinary-audio, pins and exit evidence
are reused; the old failure is not relabeled as a pass. No repeat approval of
the original plan is needed.

## Adversarial review

2026-10-18: all four reviewer lenses applied inline to spec and plan; no subagents.
Spec findings and resolutions are recorded in its Adversarial review section.
Ordering gates runtime first, then parser/runtime, API, checkpoint, identity, UI
and real integration. Every tracker task has specific verification evidence.
Final planning review passed; owner approval received and task 1 executed.
Runtime acceptance now passes the separate EOF proof recorded below. The
planning review itself did not claim runtime feasibility.
Remember to follow the MSW deletion rule for all claims — no exceptions.

2026-10-18 paired refinement review applied architect, planner,
simplification-expert and code-reviewer inline. Resolved raw-evidence/backend
duration trust, exact comparison, collapsed spans and retry idempotence findings
in the spec and tasks 1/2/5. No necessary design finding remains open. The 0.01s
policy is derived from the recorded fixture failure, not an upstream precision
guarantee. No review subagents or product changes for this refinement.

## Verification log

- 2026-10-18 task 2 PASS, 2/8 complete. Accepted ADR-0002 and nonoverlapping
  MOSS registry mapping; preflight receipt read before parser edits. Pure stdlib
  parser retains labels/order, nested overlap, annotations, raw decimal tokens
  and JSON-safe exact rational clocks. Strict completion/EOS/context checks and
  backend recomputation against trusted sample dimensions reject tampered result
  and correction evidence. In-range zero spans retain approved raw <= semantics.
  SQLite environment set before tests; targeted parser suite: 63 PASS. Scoped
  Ruff PASS; SDD validation and diff whitespace PASS. No runtime/IPC or product
  integration; task 3 awaits user Do next.

- 2026-10-18 separate approved moss-eof-v1 proof PASS; task 1 complete, 1/8.
  Retained 9s PCM hash unchanged; native CUDA/BF16/SDPA, pinned default budget
  5120 bounded by context, 200 prompt/108 output tokens, EOS 151645. Four native
  segments fully parsed, index 3 raw 9.01 normalized to exact 9; three overlap
  pairs preserved, words=null. Exact Fraction normalizer: 18 boundary/rejection
  checks PASS; retained real raw output revalidated after strict decimal grammar
  tightening. Load/decode 53.435/29.205s; process-local peak allocated/reserved
  1,874,400,256/1,929,379,840 bytes; child 26664/launcher 46912 absent, shell 0.
  Global VRAM 3976→4020 MiB has unrelated-desktop caveat. Report includes full
  raw/normalized evidence, runnable source and exact commands. Reused existing
  pinned environment, eight complete ordinary decodes, cold/warm timings and
  successful/failed exit evidence; historical strict FAIL remains unchanged.
  `python scripts/check_sdd_docs.py validate` PASS; `git diff --check` PASS.
  No app tests or task 2 work. Plan stays In progress; spec is not Shipped.

- 2026-10-18 owner-requested MOSS/Parakeet timestamp comparison PASS as research.
  PCM identity proven for retained 120s result; fresh isolated actual-app 9s
  Parakeet CUDA cold/warm completed, child exit 0, no transcription fallback.
  Conversation: 9/31 unique exact phrases (97/392 MOSS tokens), 18 endpoints,
  median absolute delta 0.11s, mean 0.231667s, signed range -1.11 to +1.35s.
  Overlap: 0/4 exact phrases; separate unique endpoint anchors -0.16/+0.35/+0.13s.
  Lexical disagreement and repeated phrases explicitly excluded from full-phrase
  statistics. No ground-truth accuracy or uniform drift claim; native clocks
  remain unchanged. See docs/research/2026-10-18-moss-parakeet-timing.md.
  Task 1 remains in progress, 0/8 complete; EOF-policy proof remains pending.

- 2026-10-18 controlled official-helper/probe comparison PASS: same retained
  144000-sample fixture, default prompt, pinned native CUDA BF16 SDPA, greedy
  4096 budget, exact equality of all five complete input tensors and effective
  generation configurations. Both prompt/attention lengths 200, identical full
  output and 108 generated IDs/raw text, EOS 151645, raw terminal 9.01s.
  Child 36916 normal exit, shell exit 0, PID absent afterward. See research
  report for commands, script, tensors/hashes, raw IDs/text and measured times.
  This comparison does not explain the 9.01s output. Task 1 stays in progress;
  approved EOF-policy proof is a separate pending step. Tasks 2–8 unstarted.


- 2026-10-18 upstream follow-up: initial release, current pinned source and the
  fine-tuning-support commit retain the same marker arithmetic. Released
  fine-tuning collator loads checkpoint processor/config; report v7 describes
  time markers without exact rounding/calibration. Issue 51 reports other
  timestamp errors but provides no marker fix. No safe documented correction
  established; original pretraining intent remains unknown. No inference or
  parameter changes. Preserve upstream behavior; next targeted check compares
  official generation helper and probe raw output before applying EOF policy.
  See research report for primary-source links and access limitations.

- 2026-10-18 owner superseded immediate rerun with a workflow audit first. CPU
  loader/processor audit PASS: exact 144000/16000 duration, PCM-identical fixture
  reconstruction and loaded samples, clean pinned helper and matching cached
  custom code, raw literal 9.01 before parsing, exact 0.01s excess. Found nominal
  9.04s feature-token span and upstream marker-position truncation (5s marker at
  nominal 4.96s, 120s marker at 119.04s); causal effect on generated clocks is
  unproven. Research report distinguishes facts/inferences and records probe
  budget, abrupt fixture boundary, previous overlap-order reporting error and
  incomplete setup diagnostics. No generation, correction or source patch run;
  report audit before resuming. Product tasks remain unstarted.

- 2026-10-18 owner authorized an explicit tiny-overrun rule after explanation of
  the 9.01/9.00 failure. Read pinned helper README/parser: examples show two
  decimals but parser accepts arbitrary decimal precision. Refined existing
  spec and tasks 1/2/5 with measured 0.01s inclusive end-only EOF policy,
  exact duration/token comparisons, retained raw provenance, no collapsed spans
  and retry idempotence. Paired review passed inline. Targeted runtime rerun
  remains pending; prior failure and ordinary-audio proof remain unchanged.
  `python scripts/check_sdd_docs.py validate` PASS; `git diff --check` PASS.

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
