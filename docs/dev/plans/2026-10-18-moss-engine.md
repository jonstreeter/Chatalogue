# Optional MOSS Engine — Implementation Plan

**Date:** 2026-10-18
**Status:** In progress
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`

## Summary

Add an app-managed optional MOSS engine using Phonon's private-worker pattern.
Use native MOSS speaker turns and existing named voice-profile matching across
recordings. Tasks 1–8 proved the initial single-input implementation. The owner's
2026-10-19 correction supersedes recording rejection and whole-input retry:
tasks 9–13 add bounded, resumable chunks without imposing model call limits on
recording duration. That direction authorizes this refinement and implementation;
no repeat approval of the same intent is pending. Native CUDA feasibility remains
established; new chunk behavior needs its own isolated and real verification.

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
| 3 | Add isolated managed installer/worker using task-1 pins, process ownership, observable setup and offline cache; test concurrent setup, interruption, busy unload, worker death and parent-exit cleanup. | `backend/src/services/ingestion/moss_runtime.py`, `backend/src/services/ingestion/moss_worker.py`, `backend/src/services/ingestion/moss_worker_requirements.txt`, `backend/src/services/ingestion/phonon_process.py`, `backend/src/test_moss_runtime.py` | L | ✅ done: managed installation/worker, 110 isolated tests and real inference/exit proof PASS |
| 4 | Wire manager/GPU lifecycle, settings engine-test actions and explicit multilingual MOSS route; verify no default change or production job creation. | `backend/src/services/ingestion/service.py`, `backend/src/services/ingestion/runtime.py`, `backend/src/services/ingestion/cuda_memory.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/main.py`, `backend/src/schemas.py`, `backend/src/routers/settings.py`, `backend/src/test_moss_runtime.py` | M | ✅ done: app lifecycle/settings/routing; 193 isolated tests PASS |
| 5 | Add full-input transcription and atomic native-label checkpoint, duration/context rejection, provenance and retry semantics. Preserve raw clock tokens, source sample count/rate and moss-eof-v1 corrections across identity retries without cumulative normalization. Test old cache reuse, mixed-engine partial rejection, failure/cancellation and forced reprocessing. | `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline.py`, `backend/src/test_moss_pipeline.py` | M | ✅ done |
| 6 | Reuse identity/profile matching with explicit native speaker attribution and overlap-safe samples; retain standard save/cleanup. Test two recordings matching saved names, nested overlapping speakers, no clean sample and missing embedding model. | `backend/src/services/ingestion/pipeline_diarize.py`, `backend/src/services/ingestion/speaker_identity.py`, `backend/src/services/ingestion/transcription.py`, `backend/src/test_moss_pipeline.py` | L | ✅ done |
| 7 | Add engine option, setup/status/unload controls, segment/90-minute notice and queue badge; browser-verify save/reload, failure recovery and segment playback without word clocks. | `frontend/src/pages/Settings.tsx`, `frontend/src/components/SetupWizard.tsx`, `frontend/src/pages/JobQueue.tsx`, `frontend/e2e/moss.spec.ts` | M | ✅ done: managed UI and five Chromium checks PASS |
| 8 | Run isolated real app-phase short/long audio, identity across recordings, sequential GPU release/staged reuse; run workflow checks, record evidence and prepare local closeout. | `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/test_moss_pipeline.py`, `frontend/e2e/channels.spec.ts`, `frontend/e2e/pages/ChannelsPage.ts`, research, plan and spec | M | ✅ done — isolated browser/backend and actual CUDA/identity evidence PASS |



| 9 | Add sample-exact bounded chunk planning with existing VAD boundary hints and typed worker OOM/context failure classification; prove coverage and finite split semantics. | `backend/src/services/ingestion/moss_chunks.py`, `backend/src/services/ingestion/moss_worker.py`, `backend/src/services/ingestion/moss_runtime.py`, `backend/src/test_moss_chunks.py`, `backend/src/test_moss_runtime.py`, `docs/dev/contracts.json` | M | ✅ done: sample planner and typed resource failures verified |
| 10 | Stream normalized audio through bounded requests; durably commit complete native chunks, exact offsets, source fingerprints and frontier; support adaptive resource split, cancellation/resume and legacy reuse without rejecting long recordings. | `backend/src/services/ingestion/pipeline_transcribe.py`, `backend/src/services/ingestion/pipeline.py`, `backend/src/test_moss_pipeline.py` | L | ✅ done: bounded streaming and durable native chunks verified |
| 11 | Reconstruct chunk namespaces and reconcile real channel identities without native overlap loss; verify clean enrollment, swapped labels and identity-only retry. | `backend/src/services/ingestion/pipeline_diarize.py`, `backend/src/services/ingestion/speaker_identity.py`, `backend/src/test_moss_pipeline.py` | M | ✅ done: chunk identity enrollment, persistent names and retry verified |
| 12 | Replace recording-limit and whole-input retry UI with automatic batching/resume capability; expose chunk progress and browser-verify controls, persistence and segment playback. | `frontend/src/pages/Settings.tsx`, `frontend/src/components/SetupWizard.tsx`, `frontend/src/pages/JobQueue.tsx`, `frontend/e2e/moss.spec.ts` | M | ✅ done: automatic batching/resume notice and isolated Chromium proof |
| 13 | Verify isolated >90-minute coverage fixtures and real CUDA multi-chunk seams/identity/checkpoint reuse/GPU handoff; run full workflow and record measured limitations/local handoff. | `docs/research/2026-10-18-moss-runtime.md`, spec, plan, handoff; affected MOSS tests/paths only through explicit conductor repack | L | ✅ done: real three-chunk CUDA/identity/resume and full local verification |
| 14 | Correct Windows-checkout source checksum pins to exact immutable upstream download bytes; prove fresh-source download setup, tamper rejection, offline reuse and native CUDA load/inference without touching checkpoints. | `backend/src/services/ingestion/moss_runtime.py`, `backend/src/test_moss_runtime.py`, `docs/research/2026-10-18-moss-runtime.md`, plan and handoff | M | ✅ done: canonical fresh-source setup, offline reuse and real CUDA/exit verified |

| 15 | Diagnose the reported timestamp rejection without production reprocessing; distinguish invalid model output from GPU decode failure and log safe numeric boundary evidence. Preserve exact moss-eof-v1 acceptance and all checkpoint/retry behavior; report any policy decision needed separately. | `backend/src/services/ingestion/moss.py`, `backend/src/services/ingestion/moss_worker.py`, `backend/src/services/ingestion/moss_runtime.py`, `backend/src/test_moss.py`, `backend/src/test_moss_runtime.py`, research, plan, handoff | M | ✅ done: strict policy retained; numeric diagnostics and invalid-output classification verified |

| 16 | Prevent leading-VAD-silence-only microchunks by retaining leading silence with following speech; preserve bounded exact coverage, strict EOF validation and existing checkpoint resume. Reproduce the 256-frame leading-gap planner path and verify isolated pipeline requests plus real retained-audio CUDA inference. | `backend/src/services/ingestion/moss_chunks.py`, `backend/src/test_moss_chunks.py`, `backend/src/test_moss_pipeline.py`, research, plan, handoff | M | ✅ done: leading-gap regression, isolated pipeline and real CUDA/exit verified |

## Test plan

### Leading VAD gap regression

Task 16 addresses actual diagnostic duration 2/125 seconds (256 frames), not a
larger timestamp tolerance. Reproduce leading speech hints starting at frame256
with speech reaching the bounded-window end. Keep that leading silence inside
the following speech request; only gaps preceded by speech within the current
window should suggest an earlier seam. No arbitrary minimum chunk length, audio
padding, silence deletion or malformed-output retries. Verify exact coverage,
internal/trailing gaps, resumed nonzero cursors, EOF, VAD-unavailable and typed
resource splits. Run isolated parser/runtime/chunk/pipeline suite and one real
retained-audio CUDA check through repaired boundary planning with exact frame
accounting/strict parse/worker exit. Do not reprocess production audio. Scoped
Ruff, SDD validation, whitespace and normal commit hooks remain required.

### Timestamp rejection diagnostics

Task 15 preserves parser acceptance, private worker ownership and checkpoint
formats. Verify safe numeric diagnostics for beyond-allowance EOF rejection,
typed distinction from CUDA OOM/context errors, no automatic splitting/retry for
malformed output, no transcript leakage, unchanged accepted EOF/overlap cases,
and owned cleanup. Run isolated parser/runtime/chunk/pipeline tests, scoped Ruff,
SDD validation, whitespace and normal commit hooks. Use injected worker output
for rejected model generations; do not rerun production audio or repeat closed
CUDA performance proofs. Any wider timestamp policy requires a separate owner
decision after the measured diagnostic gap is recorded.

### Fresh-download checksum regression

Task 14 corrects an installation defect, not model/source revisions or a new
inference contract. Verify canonical immutable Git blob/download bytes, reject
tampered or truncated files, exercise downloader without preseeded source files,
and run actual isolated setup/native CUDA inference with the corrected source.
Retain old task-1/task-3 evidence unchanged and distinguish Windows checkout
CRLF hashes from upstream LF hashes. Existing completed chunks survive unchanged.

### Batching amendment tasks

- Task 9: pure sample-exact contiguous planner, optional VAD gap hints, measured
  120-second target, final fractional chunk, silence retained, and typed resource
  failures. Test OOM/context classification without allocating GPU memory and
  prove malformed output/worker crash never triggers automatic split.
- Task 10: isolated real normalized WAV longer than 5400 seconds with instrumented
  inference proves bounded calls and complete frame/PCM coverage; no long-input
  duration rejection. Exercise failed later chunk, zero ownership, append/fsync/
  metadata ordering, uncommitted tail, corrupt committed data, source fingerprint
  mismatch, split frontier/resume, forced processing and legacy complete reuse.
- Task 11: existing matcher/profile SQLite tests across chunk namespaces and
  recordings, swapped/reused local labels, absolute native overlap, clean-only
  sampling, unresolved voices and identity retry without new ASR calls.
- Task 12: browser option/save/reload, real controls, chunk progress, interrupted
  resume notice, words=null playback and no false recording limit.
- Task 13: actual CUDA multi-chunk research speech with a seam and real voice
  profiles; staged child reuse and verified exit before voice load, measured
  timing/memory, checkpoint retry and local full workflow. Over-90-minute
  instrumented coverage is not called a real 90-minute CUDA performance test.
  Production settings/recordings remain untouched.

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
Tasks 2–3 are complete; tasks 4–8 remain unstarted. Prior ordinary-audio, pins and exit evidence
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

- 2026-10-19 owner reported fresh setup failing source checksum. Exact immutable
  upstream reads identify four CRLF checkout pins incorrectly used for LF download
  bytes. For source/LICENSE: expected 11558 bytes versus upstream 11357; 201 CRLF
  sequences account for the difference. All four local files normalize byte-for-
  byte to their upstream LF content. Source/model revisions remain unchanged.
  Task 14 is the authorized root fix and fresh-download regression/proof.

- 2026-10-19 owner reported a real job rejected by the app's whole-recording
  guard and directed batching as the correct app responsibility. Revised existing
  spec/ADR and appended tasks 9–13; original 8/8 evidence stays historical.
  Paired inline architect/planner/simplification/code review passed: addressed
  native-label collisions, seam coverage, atomic retry, source fingerprint,
  measured sizing, typed resource recovery and legacy compatibility. No new
  product code or production job processing in this planning checkpoint.

- 2026-10-19 task 4 PASS, 4/8 complete. One lazy app MOSS manager, private
  runtime/temp paths, independent shutdown, competing GPU/voice handoff, busy
  controls/manual purge and retained-worker allocator recovery. Explicit MOSS
  settings/actions and multilingual route preserve defaults and avoid Whisper
  probing. Ready describes inference smoke completion, never silence accuracy.
  Packed SQLite suite (backend venv, five named test files): **193 PASS**, one
  existing pyannote warning, 115.79s; no production settings/jobs/media or actual
  GPU models. Scoped Ruff, SDD validation and diff whitespace PASS. Actual
  app-phase GPU evidence remains task 8; browser controls remain task 7.

- 2026-10-19 task 3 PASS, 3/8 complete. Separate hash-pinned private installer,
  source/model snapshots, Windows native SoundFile wheel, staged publication,
  interprocess setup lock and offline verified cache. Direct reuse of unchanged
  Phonon Windows ownership; actual setup interpreter handshake and cancellation.
  Isolated SQLite runtime/parser suite: 110 PASS; scoped Ruff PASS. Real managed
  RTX5090 CUDA12.8/Torch2.10/BF16/SDPA worker 19972 (launcher48444) loaded,
  generated actual readiness smoke output, decoded the retained 9s fixture with
  four native segments/three overlap pairs and exact EOF evidence, then exited
  (launcher0, actual interpreter exit verified, no temp files). Readiness silence
  hallucinated malformed transcript: it is explicitly a complete-generation
  smoke probe, never a claimed valid silence transcription. Fixture200/108tokens,
  EOS151645, actual context budget130872; load/decode3.575/6.922s. Private offline
  installed-file/67-wheel verification PASS with zero network/subprocess calls.
  Initial universal SoundFile attempt failed native verification before manifest;
  corrected exact-version Windows wheel passed normal installer. Full costs and
  process-local/global GPU evidence are in the research report. No shared Phonon
  edit, production settings/jobs/media, pipeline/UI integration or task4 work.
  SDD validation, whitespace and required commit gates recorded at checkpoint.

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
- 2026-10-19 task 5 PASS, 5/8 complete. Whole-input native MOSS checkpoint,
  exact decoded dimensions/90-minute rejection, strict complete/context evidence,
  native label/raw-token/rational/correction round trips, source/runtime provenance,
  identity-stage cache reuse, mixed partial refusal, cancellation/full retry and
  explicit force. Atomic fsync/replace preserves the prior complete checkpoint
  on failed forced publication. Packed backend venv SQLite suite:
  `python -m pytest src/test_moss_pipeline.py src/test_moss.py src/test_phonon.py -q`
  **116 PASS**, 45.34s. Scoped Ruff, SDD validation and diff whitespace PASS.
  No production DB/settings/jobs/media or GPU inference; identity integration is
  task 6, browser task 7 and actual inference/full regression task 8.

- 2026-10-19 task 6 PASS, 6/8 complete. Revalidated native checkpoint chooses explicit attribution; no clustering, RTTM attribution, midpoint mapping or turn consolidation. Other-label subtraction feeds existing sample thresholds; no-clean speakers remain distinct unresolved locals without profile enrollment, with job payload warning. Existing matcher, channel isolation, names/profile persistence, transcript files and checkpoint cleanup retained. Embedding-only mode reuses existing loader and owned MOSS exit barrier. Packed SQLite suite: **59 PASS**, 44.55s. Scoped Ruff, SDD validation and whitespace PASS; required hooks run at commit. Initial fixture assertion corrected to respect existing sample threshold; fixture model-load guards added after initial cached CPU model load. No production DB/settings changes. Real inference/identity evidence remains task 8.

- 2026-10-19 task 7 PASS, 7/8 complete. MOSS Settings/SetupWizard selection,
  managed lifecycle/status/progress/retry/unload, multilingual segment-timing and
  90-minute/whole-input retry notices, explicit save and queue provenance badge.
  Wizard MOSS setup does not save selection. Combined Chromium suite **5 PASS**,
  30.1s: lifecycle/save/reload, wizard setup/finish, queue and real PCM media
  segment seek/playback with two named overlapping words=null segments; existing
  Phonon regressions pass. `npx tsc -b` PASS; `npm run lint` PASS (0 errors,
  43 existing warnings); SDD/whitespace PASS. All API routes intercepted before
  navigation. Installed Chrome env override and owned Vite5175 polling avoid
  missing bundled browser and Windows watcher EBUSY. Initial fixture errors fixed
  with health/quality response shapes and HTTP206 media range support. Native
  seek allows100ms browser tolerance. Ignored output/playwright screenshots
  reviewed. Task8 retains actual app/runtime/identity/GPU/full regression evidence.

- 2026-10-19 task8 WIP, **7/8 complete; blocked**. Ordered static checks PASS:
  typecheck, root Ruff, frontend lint (0 errors/43 existing warnings). Full
  isolated backend suite **252 PASS**, one warning,124.31s; SQLite/fixture paths
  and dotenv disabled before imports, production data/env audit guard active.
  Full Chromium collection19 selected, **1 PASS/1 FAIL/17 not run**,10.7s,
  stopped at first failure. Fixture-only localhost8000 and owned Vites5173/5175;
  audio-download pass simulates state only. Existing channel-detail.spec.ts:18
  Open locator matches3 links; proposed test-only exact:true follow-up outside
  this phase ownership. No product/test edit or rerun. Actual MOSS app-phase,
  real voice identity, staged reuse and GPU handoff remain unrun. Evidence and
  commands in research Task8 checkpoint; resume:yes handoff records blocker.


- 2026-10-19 task8 resume: owner explicitly approved only test-only exact:true
  Open selectors in frontend/e2e/channel-detail.spec.ts; all four occurrences
  repaired. Scoped `npx eslint e2e/channel-detail.spec.ts` PASS. Backend Python3.11
  fixture bind succeeds after system Python3.14 WinError10013. Initial browser
  attempt failed infrastructure (HTML instead of JSON), not a selector regression.
  Full fixture-backed rerun:19 selected,2 PASS/1 FAIL/16 unrun,39.0s; first channel
  navigation now passes. Next test waits for absent Search tab at
  e2e/pages/ChannelDetailPage.ts:30; snapshot shows Videos/Transcripts/Speakers/
  Chats/Clips. Stop at first genuine failure; no further repair or real inference.
  Task8 remains incomplete. Earlier ignored screenshots were inadvertently removed
  by reused Playwright outputDir; original e2e.log survives, new failure screenshot
  and context retained. Owned API/Vites stopped and no listeners remain on
  8000/5173/5175. WIP checkpoint subject: docs(moss): checkpoint next browser blocker.

- 2026-10-19 task8 resumed with owner-approved tab-list repair only:
  channel-detail.spec.ts now visits Videos, Transcripts, Speakers, Chats, Clips.
  Scoped ESLint PASS. Full browser command selected all19; **1 PASS/1 FAIL/17
  unrun**,51.0s. First channel-detail test timed out navigating / before any tab
  assertion: page.goto net::ERR_ABORTED, BasePage.ts:11,30s load timeout.
  Root cause remains undiagnosed; no timeout/product workaround or rerun.
  First genuine new verification blocker stops task8; real inference/identity/
  staged reuse/GPU handoff remain unrun. Prior closed checks unchanged.
  Evidence task8/e2e-resume3.log and unique e2e-results-resume3 preserve earlier
  logs/assets. Fixture8000 PID67356 verified as child of owned venv launcher67052
  before HTTP requests; Vites58848/66484 returned200. All owned services stopped;
  netstat confirms no listeners8000/5173/5175. No production backend/data/env.
  WIP checkpoint subject: test(moss): checkpoint approved tab repair verification.
- 2026-10-19 task8 approved navigation diagnosis and same-root repair continuation:
  clean instrumented load PASS874ms/networkidle; Google Fonts denial nonblocking.
  Earlier Vite HTTP timeouts intermittent, no product root cause established.
  Scratch fixture delete-preview numeric response and synthetic youtube source
  corrected under conductor repack; no product changes or external scans.
  Full diag5:3PASS1FAIL15unrun13.1s (fixture preview shape); fixture6:
  5PASS1FAIL13unrun42.5s (manual fixture has no Scan); fixture7:
  5PASS1FAIL13unrun13.5s (same ambiguous Open selector).
  Existing owner exact Open approval extended by conductor to four identical
  callers: channel-refresh.spec.ts,channels.spec.ts,video-sort-order.spec.ts,
  pages/ChannelsPage.ts. Preflight no mapped contract; only exact:true added;
  scoped ESLint PASS. Full open8: **7 PASS/1 FAIL/11 unrun**,20.5s,19 selected.
  Current new blocker channels.spec.ts:17 expects inline add input before opening
  Add Channel wizard. Snapshot has Add Channel button; source shows source-type
  choice then input. No new test-flow repair authorized/performed. Task8 remains
  incomplete7/8; actual MOSS app-phase/identity/staged reuse/GPU handoff UNRUN.
  Evidence task8/e2e-open8.log,e2e-results-open8/; all completed reruns have unique
  outputDirs. Accidental earlier interrupted run reused resume3 dir after cwd
  preparation error (3 passes then stopped, not failure); old screenshot lost,
  original logs retained. Owned API63584/40500,Vites46652/29080 stopped; no listeners
  8000/5173/5175. No production backend/data/env or fallback.

- 2026-10-19 task8 COMPLETE, **8/8 tasks verified**; plan stays In progress and
  spec In Review pending local closeout. Owner-approved Add Channel test now opens
  Add Channel → YouTube Channel and asserts the empty current URL field; no submit.
  Scoped ESLint PASS. Full installed-Chrome browser suite **19 PASS**, 1.3m,
  unique task8/e2e-results-full12; no skips/exclusions. Earlier MOSS Ready timeout
  did not reproduce under timestamped diagnostics: status poll arrived4.680s after
  readiness flag, scoped3 PASS28.5s. Diagnostic logging removed; no product timing,
  Settings or MOSS test changes. Required Chrome env override retained in scratch.
  Actual app-phase uncovered FFmpeg LIST metadata rejected by strict44-byte WAV
  IPC. Conductor repacked only pipeline_transcribe.py/test_moss_pipeline.py:
  validated exact PCM is reserialized with stdlib wave; strict worker unchanged.
  Real FFmpeg LIST regression proves identical PCM144001samples/16000Hz/duration.
  Targeted20 PASS25.18s; full isolated backend **253 PASS**,1 warning,136.10s;
  root Ruff PASS. Prior typecheck/full frontend lint evidence reused.
  Actual app14 CUDA/BF16/SDPA sameworker62348:9s6.760s/4segments,
  10s5.370s/2segments,120s61.017s/31segments. Raw native overlap/words=null and
  exact9.01→9 moss-eof-v1 correction retained. Load14.057s end-to-end; verified
  unload0.194s/exit0/processfalse. Actual120s peakallocated2,506,159,616B,
  peakreserved2,699,034,624B; global VRAM includes desktop/other allocations.
  Real GPU embedding + existing channel matcher/profile persistence PASS:
  identity18 actual10s→120s namedprofile1 reused;11 clean-only crops. Shared
  source/opening speech disclosed, not accuracy/generalization evidence.
  identity19 actual clean120s S03 interval24.44–30.48 decoded as S01 in distinct
  6.04s input, matched persisted profile9/name Research native S03; no relabel or
  fabricated embedding. Same-service owned MOSS exit0/OS-dead verified before
  GPU voice load. MOSS reload after voice measured14.874s, finalexit0.
  Limitation:9s clean S02 interval5.83–9 failed existing embedding boundary check
  despite tensor[1,144000]@16000 and strictsamplecount144000; retained unresolved,
  no profile enrolled. Earlier short2.12s seed did not match120s at unchanged
  threshold. No cause inferred from rounded error; no crop/time/model workaround.
  All helpers/models cleaned; no production DB/env/settings/media or external
  scan/download. Longest120s and shared-source fixtures do not prove90-minute fit,
  general recognition accuracy or speed. Evidence/reproduction in research log.

- Task 9 (2026-10-19): pure next_chunk/split_chunk planner preserves exact contiguous
  normalized frames including silence and final fractional samples for >5400s totals.
  Initial ceiling120s comes from task8 proof; optional installed VAD sample hints
  choose latest gaps, explicit bounded fallback otherwise. VAD invocation remains
  task10. Only actual CUDA OOM / exhausted output budgets produce typed private
  resource errors; malformed output, early EOS, process errors remain ordinary
  failures. Cleanup/ID validation and strict moss-result-v1 remain intact.
  Packed backend suite: 163 PASS (final rerun recorded in checkpoint); scoped Ruff,
  SDD validation and git diff --check PASS. Required commit hooks run at checkpoint.
  Initial test-double close defect corrected; no product workaround. Worker crash,
  malformed JSON/clocks and protocol-ID failures explicitly remain non-resource.

- Task 10 (2026-10-19): bounded disk-normalized PCM fingerprint scanning and
  canonical WAV requests replace episode-duration rejection. Installed default
  CPU VAD hints select cuts only; unavailable hints record bounded fallback.
  Versioned moss-recording-v2 manifests retain source PCM hash/dimensions/stat,
  pinned per-chunk runtime, immutable native evidence, exact frame offsets and
  stable speaker namespace components. Initial ownership, append/fsync/atomic
  frontier, committed validation, tail-only truncation and atomic complete raw
  replacement preserve resume and prior completed artifacts. Typed resource
  bisection touches only failed intervals; successful reduced ceiling persists.
  Complete legacy raw reuse remains supported; v2 identity-only reconstruction
  does not normalize, infer or apply EOF correction again.
  Final packed command (backend cwd; DB_PROVIDER=sqlite DATABASE_URL=sqlite://):
  `.venv/Scripts/python.exe -m pytest src/test_moss_pipeline.py src/test_moss_chunks.py src/test_moss.py src/test_phonon.py -q`
  **162 PASS**, one existing Transformers cache deprecation warning,101.59s,
  actual process exit0. Earlier fixture-adaptation run15 failures preceded
  correcting legacy fixture provenance/format assumptions; adapted159 PASS,
  then final162 PASS including last three proofs. No product workaround.
  Generated real5401s+1frame WAV with real FFmpeg normalization/instrumented
  inference proves46 requests<=120s and exact PCM SHA256/frame coverage including
  silence. Crash/cancel committed resume, OOM/context split and learned ceiling,
  metadata tail recovery, corrupt/missing/gapped/mixed/source/model rejection,
  legacy/force behavior, TinyEOF local evidence/absolute offsets and final atomic
  recovery verified. This is separate from real CUDA/seam/recognition proof.
  Scoped Ruff, SDD validate and git diff --check PASS; normal required commit
  hooks run at checkpoint. Task11 must extend legacy-only native route detection
  to v2; existing identity regressions explicitly retain legacy coverage here.
  No production media/settings/DB, real inference, dependencies or UI changes.

- Task 11 (2026-10-19): validated v2 raw checkpoints now select the native
  diarization route alongside legacy moss_result. Existing namespace reconstruction,
  clean-span subtraction, channel matcher and transcript saves remain authoritative.
  Fixed first enrollment into an empty channel cache by initializing it before
  populated-cache dimension validation. No matching thresholds/schema changes.
  Real SQLite matcher/profile/save tests cover three chunks, equal labels for
  different voices, changed labels for one voice, cross-recording swapped labels
  with preserved Host/Guest names, channel isolation and no forced merge. Native
  nesting/overlap, absolute EOF/correction, words=None, clean-only crops/profiles,
  distinct unresolved locals, failed voice-model load and raw-preserving identity
  retry without normalization/VAD/ASR are verified for v2 and legacy routes.
  Final packed command (backend cwd; DB_PROVIDER=sqlite DATABASE_URL=sqlite://):
  `.venv/Scripts/python.exe -m pytest src/test_moss_pipeline.py src/test_speaker_loading_optimization.py src/test_transcript_diarization_rebuild.py src/test_phonon.py -q`
  **79 PASS**, one existing Transformers cache deprecation warning, 97.65s,
  actual process exit0. Initial existing-suite run76 PASS preceded new-case append;
  no test failures or workaround. Scoped Ruff, SDD validate and git diff --check
  PASS; normal required hooks run at checkpoint. Deterministic embeddings prove
  identity routing/persistence, not actual voice recognition; task13 keeps real
  multi-chunk CUDA/seam proof. No production DB/media/settings or GPU inference.


- Task 12 (2026-10-19): Settings and setup wizard now explain automatic bounded
  chunks and retries resuming completed chunks, retaining segment-only timing,
  local management and readiness-test limits. Removed recording-cap/whole-input
  retry copy. Existing queue status detail/progress/badge rendering needs no code
  change; browser fixture starts at resumed chunk3/240 of600s, then reloads at
  chunk4/360s and verifies existing40%/60% bars. No invented chunk count or ETA.
  Packed frontend command with installed Chrome executable environment:
  `npm run test:e2e -- e2e/moss.spec.ts e2e/phonon-settings.spec.ts --workers=1`
  **5 PASS**,28.8s, actual exit0. Explicit save/reload, setup/status polling,
  pending/error/retry/unload, wizard test without save, other engine options,
  MOSS badge and native words=null nested overlap/seek/real PCM playback pass.
  Final screenshots under ignored `output/playwright/task12-batching-final/`
  visually inspected (wizard capability notice, queue frontier/progress, native
  simultaneous highlights/playback). All APIs intercepted before navigation;
  owned Vite5175 used scratch envDir/polling; no production backend/settings/media.
  Initial5PASS screenshot exposed missing fixture stage-start timestamp, corrected
  with real existing field and added bar assertions; no product workaround.
  Typecheck PASS; lint PASS with43 existing warnings. SDD/whitespace/hooks run
  at checkpoint. Actual chunk seam/identity/GPU proof remains task13.

- Task13 (2026-10-19): **13/13 tasks verified**. Real default-VAD CUDA three-chunk
  proof on a distinct240s WAV made by repeating cached120s speech: exact normalized
  frame coverage0–1169152–3089152–3840000;73.072/120/46.928s requests;
  vad_gap/vad_gap/eof, no VAD fallback, unchanged120s ceiling. Actual childPID3208
  reused across all successful calls42.470/73.104/27.779s. Instrumented interruption
  after first durable publication retained exact first result/frontier; retry
  inferred only remaining chunks. Strict EOS/context,65 native turns, absolute
  offsets/namespaces/words=None and final46.93→46.928 localEOF correction retained.
  Initial scratch wrong manifest-field assertion corrected against completed raw
  without repeating inference; original failure evidence retained. Actual240s has
  no cross-label overlap; retained actual9s decoder overlap and v2 fixtures cover
  overlap separately. Seam-adjacent text inspected; repeated speech and recognition
  variation disclosed, no lossless or continuous-hard-cut accuracy claim.
  Real CUDA voice matching saved65rows/19profiles using22 clean-only crops:
  equal locals differed in identity and changed locals matched same real profile.
  OS worker absence verified before GPU voice load. Named raw-only identity retry
  passed65 matched rows,0 ASR/VAD, unchanged rawhash. Models/helpers cleaned.
  Ordered full workflow: typecheck PASS; root Ruff PASS; frontend lint PASS
  (0errors/43existing warnings); isolated backend **316PASS**,1 warning,179.58s,
  exit0; all installed-Chrome browser tests **19PASS**,49.6s,exit0, no exclusions.
  Full backend includes generated5401s+1frame coverage with real FFmpeg and46
  instrumented requests; not90-minute CUDA inference. Browser uses only mocked/
  fixture APIs, owned8000/5173/5175 and unique task13/e2e-results-full; no production
  scan/process/settings/media/DB. Existing runtime/deprecation warnings recorded
  in research; no dependencies/threshold changes. SDD/whitespace and normal commit
  hooks run at checkpoint. Plan remains In progress/spec In Review for local
  closeout; no push/deploy/installer rerun or feature archive performed.

- Task 13 required proof follow-up (2026-10-19): conductor review found that
  initial real cuts were all VAD gaps, leaving the spec's continuous-speech seam
  proof open. Closed with a 125s rotated cached-speech fixture: original second 27
  lies inside retained real speech 24.44–30.48, and PCM across the forced boundary
  is byte-identical to the original. Explicit optional-VAD-unavailable fault
  injection exercises the existing fallback, not synthetic VAD predictions.
  Two actual CUDA calls in child 61544 own 0–1920000–2000000 frames, 120s/5s,
  reasons vad_unavailable/eof. Strict EOS/context, 30 native turns, raw local
  clocks/namespaces/absolute offsets/words=None and final 5.01→5 correction PASS.
  Adjacent text splits a sentence and changes continuation wording; no obvious
  whole-turn replay. Recognition/timing accuracy remains unproven and no text
  deduplication or validation relaxation occurred. Model calls 51.884/3.830s,
  actual worker exit0/OS absence confirmed. Evidence task13/forced1/ and
  forced1.log/.exit. No product/identity changes; prior 316 backend/19 browser
  passes remain closed. SDD/whitespace/normal hooks run for the separate evidence
  commit; 622e7f437 is not amended. Task 13 and all 13 tasks are now fully verified.

- Task 14 (2026-10-19): corrected only four source size/SHA pins from Windows
  CRLF checkout bytes to unchanged immutable upstream LF downloads. Regular pin
  assertion catches the old values; opt-in real fresh-network regression covers
  all4 empty caches, offline verified reuse, truncated/altered/CRLF rejection.
  Isolated requested runtime/parser/chunk suite with MOSS_TEST_FRESH_SOURCE=1:
  168 PASS,0 skips,38.55s; scoped Ruff PASS. Actual normal standard-root ensure
  freshly downloaded all4 source files, verified/reused19 existing model files,
  and reused67 exact lock-matching dependency wheels after disk check. No model,
  revision, requirement, raw validation or checkpoint changes. Setup876.271s;
  offline verified reuse PASS. New version c19b2f905bd461f27f79731aa9745864128092d26a91f788b54980b5755ca98f.
  Installed helper AST equals retained original CRLF checkout, newline-only bytes.
  Actual retained9s CUDA decode:4 turns/3 native overlap pairs, strict EOS/context,
  raw9.01->9 moss-eof-v1; worker66300 absent after owned unload, launcherexit0.
  Evidence task14/proof.json/log/exit and tests.log/exit; old evidence untouched.
  SDD/whitespace and normal required hooks run at checkpoint. Prior baton archived
  unchanged. Restart backend before MOSS Load/Test; verified standard-root install
  is ready for reuse. No production jobs/settings/media/DB/checkpoints touched.
  All14 tasks verified; plan remains In progress/spec In Review; no push/deploy
  or feature archive.

- Task15 (2026-10-19): reported worker trace reaches EOF rejection, meaning end>D
  and either excess>0.01s or collapsed correction start==D; no endpoint numbers
  retained, so exact excess/recording cannot be identified. Parser now reports
  only reason/index and exact rational end/duration/excess. Private worker
  invalid_transcript distinguishes invalid generation from typed CUDA/context
  resource failures; manager avoids GPU-memory advice. Acceptance/checkpoints/
  retries/pins/lifecycle unchanged. No production reprocessing or CUDA rerun.
  Initial cwd mistake aborted edits then mistakenly launched unguarded tests;
  interrupted at about80 tests. Imports may have read dotenv; no mutation
  established, no cleanup attempted. Later full suite guarded before imports,
  SQLite/no dotenv/task15 fixtures; deliberate production .env open denied.
  Scoped Ruff, SDD validation and whitespace PASS. Parser final refinement65PASS.

- Final task15 guarded backend venv `python -m pytest src/test_moss.py
  src/test_moss_runtime.py src/test_moss_chunks.py src/test_moss_pipeline.py -q`
  ->208 PASS,4 opt-in network skips,1 existing TRANSFORMERS_CACHE warning,
  110.43s,exit0. Includes actual worker JSONL loop/fake Decoder classification,
  strict clock acceptance, and prior committed chunk retained on invalid output
  with exactly one failed attempt/no splitting. Final manager wording assertion
  ->5 PASS. Evidence: ignored task15/tests.log and tests.exit. Scoped Ruff,
  SDD validation and whitespace PASS; normal commit hooks run at checkpoint.


- Task16 (2026-10-19): reproduced the planner defect before editing: first
  speech at cursor+256 reaching the120s limit emitted a256-frame/16ms request,
  both at cursor0 and resumed cursor1920000. Gap candidates now require prior
  speech in the current window (`previous > cursor`); no minimum, padding,
  deletion, validator or checkpoint change. Internal/trailing gaps still cut.
  Guarded requested parser/runtime/chunk/pipeline suite ->212PASS,4 opt-in network
  skips,1 existing TRANSFORMERS_CACHE warning,112.38s,exit0. Guard loaded before
  app imports; deliberate production .env open denied. Exact PCM hash pipeline
  regression routes250s as120/120/10s through real VAD adapter with injected hints.
  Real retained9s audio plus256 leading zero frames ->144256 contiguous frames,
  nonfinal bounded9.016s planner request, one actual CUDA decode/4 turns/3 native
  overlaps, strict EOS/context, unchanged9.02->9.016 EOF correction (4ms).
  Real default VAD found start0/end144256; exact256-start hint is separate injected
  proof, not an assertion about actual default VAD or the production recording.
  Verified standard install hashes/cache via normal ensure with lock replaced by
  nullcontext to keep standard root read-only; downloads/rebuild forbidden.
  Manager reused that verified install with task16-owned log/IPC; worker67068
  launcherexit0/OSabsence. GPU free27909/32607MiB before load; no other process
  stopped. Evidence ignored task16/proof.py/json/log/exit and tests.log/exit.
  Scoped Ruff/SDD/whitespace PASS; normal hooks run at checkpoint. No production
  job, settings, DB, checkpoint or app-server actions. Prior HEAD0decaefe5;
  diagnostic baton archived byte-for-byte. Plan In progress/spec In Review;
  no push/deployment/feature archive.
