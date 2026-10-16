# Handoff: optional Phonon engine implementation

**Date:** 2026-10-15
**Spec:** `docs/dev/specs/2026-10-15-phonon-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-15-phonon-engine.md`
**Commit:** 76976c0 Phonon feature; 88e2517 separate frontend lint cleanup; 942ab4b process/authority checkpoint

## Current revision — 2026-10-16

Owner rejected operator-managed connection UX and explicitly chose "Revise spec
and plan for full integration". Revised existing spec/plan and ADR amendment now
propose an app-owned private native worker with managed setup/load/reuse/unload,
no URL/key UI and no Docker prerequisite. Inline paired artifact review passed;
revised owner implementation approval passed on 2026-10-16 (`d`). Original approval is historical
and must not be requested again as though the old plan were unapproved.

Tracker tasks 1–6 remain completed; tasks 7–13 are the proposed revision. Task 7
must prove native Windows CUDA using exact verified upstream 1.0.8 source and
cached weights before downstream product work. Task 7 native CUDA feasibility passed on 2026-10-16 with unmodified upstream
source, BF16/K16/six buckets, valid repeated inference and actual child exit.
Evidence and reproducible command are in the research report. Future gate failure requires owner decision,
not a silent CPU or Docker substitute. Current installed code is still HTTP mode. Tracker is now 7/13 complete.

Planning checkpoint committed as 02a8360. Next action: implement task 8, the
private optional installer/cache, then task 9 owned IPC worker. No product implementation changes,
production settings writes or media reprocessing occurred during this revision.
Historical status/next-step sections below describe the prior completed delivery
and are superseded by this current revision. Cached artifacts and verification
below remain useful, but prove the previous runtime only.

## Status

Owner approved spec/plan and then explicitly authorized the validator patch and
continuation. Do not ask for Phonon plan approval again. Super Looper was not
found; implementation/verification/review loop ran inline, with no subagents.
Tracker: 6/6 complete. Owner authorized separate lint cleanup; it is committed and the lint gate is clear.

Implemented optional settings/unsaved authenticated readiness; persistent local
HTTP adapter; bounded PCM uploads; absolute midpoint overlap ownership; upstream
segment grouping for diarization; atomic committed-offset checkpoints including
silence, unknown-provenance refusal, and interrupted-write recovery; queue
provenance; setup docs; isolated integrated benchmark mode. Existing engines,
language routing and defaults are preserved. Production settings were not saved,
engine was not activated globally, and existing media was not reprocessed.

Scoped paths: backend/src/services/ingestion/phonon.py, pipeline.py,
pipeline_transcribe.py, transcription.py, job_lifecycle.py; backend/src/schemas.py;
backend/src/routers/settings.py; backend/requirements-common.txt;
backend/src/test_phonon.py; frontend/src/pages/Settings.tsx and JobQueue.tsx;
frontend/e2e/phonon-settings.spec.ts; docs/phonon-engine.md; scripts/benchmark_asr.py;
Phonon spec/plan/ADR/contract registry and research report. The owner-authorized
validator repair is scripts/check_sdd_docs.py with tests/test_check_sdd_docs.py.

## Next step

Implementation and verification are complete and committed. Contract-impact
validation against 9482348 passed after the feature commit. Wait for owner
authorization to push/deploy. Do not reactivate the optional runtime or
change production engine settings automatically. Read the tracker verification
log for the latest results; historical failure notes below are superseded by the
successful reruns. Do not archive as shipped before the shipping lifecycle point.

## Verification state

- Validator regression suite: 46 passed using ordinary external temporary dirs.
  _repository_files now traverses only the three intended SDD artifact directories
  in Git mode, retaining all tracked files and original non-Git fallback behavior.
- Actual mapped contract preflight and full SDD validation: PASS after repair.
  Authority/registry and validator repair committed in 942ab4b. Actual commit-range
  impact check from 9482348 through 76976c0: PASS, with spec/ADR acknowledgement.
- Final stable full backend suite: 89 passed, one upstream deprecation warning. Additional final
  focused transport/settings/chunk/readiness suite: 47 passed. Isolated SQLite,
  scratch env/files, no workers. Includes >two-hour bounded mocked transport,
  failure/resume, pause/cancel, force re-transcription, original engine routing,
  silent checkpoints, atomic metadata failure and uncommitted tails.
- Ruff entire repository: PASS. Frontend TypeScript build: PASS. New browser test
  lint: PASS. Chromium error/hook/settings/queue checks: 4 passed; application API requests mocked or blocked;
  save/reload, unsaved key, unavailable/401 errors and actual engine badges checked.
  Screenshots: frontend/test-results/phonon-settings.png and phonon-queue.png.
- Frontend repository lint: PASS after separate owner-authorized cleanup, zero errors
  and 43 existing warnings. The original baseline had 205 errors. Baseline HEAD copies
  prove unchanged counts in touched pages: Settings 15 errors/4 warnings;
  JobQueue 11 errors/1 warning. Reports are ignored under backend/data/asr-benchmark:
  phonon-eslint.json, phonon-baseline-eslint.json, phonon-final-touched-eslint.json.
  Existing live-data E2E suite not run: it scans/deletes against an installed app;
  this task explicitly forbids production media/settings operations. No pass claimed.
- Live settings endpoint readiness against CUDA: ready PASS, wrong-key 401 PASS,
  unreachable loopback error PASS. Runtime response bodies/credentials not printed.
- Final actual integrated short app phases: 9.813 / 7.305 seconds, 384 words,
  41 upstream-preserving segments. Long phases: 46.257 / 62.129 seconds, 8709 words,
  1045 segments, five external chunks. Both repetitions have zero structural
  timestamp issues. Same runtime, BF16/K16 graphs/six buckets, increasing counters
  and uptime. No consistent large integrated speedup; WER/alignment unmeasured.
- Reports: backend/data/asr-benchmark/app-phonon-integrated-120s-final.json and
  app-phonon-integrated-2880s-final.json. Earlier non-final reports remain preserved.
  Generated verification key: ignored phonon-implementation.env in that directory.
  Native CPU/MLX performance not exercised on this host; documented upstream setup
  and CPU API shape tested. Installed browser version mismatch worked with
  PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH pointing to installed chromium-1217.

## Blockers

No implementation blocker remains. Push and deployment require owner authorization.
Keep unrelated dirty
work intact; the initial checkout already held 15 dirty paths. Do not commit those
unrelated docs/tooling trees. No paid action or unavailable CUDA access blocks this
feature. Temporary verification services were stopped at handoff; cache and
benchmark artifacts remain.

## Diagnosis worth keeping

Initial contract discovery stalled on generated-tree traversal; fixed with owner
approval, not by disabling validation. _repository_files used root.rglob before
filtering; now only intended artifact directories are walked in Git mode.

Code review resolved: preserve upstream segment boundaries for diarization; initial
zero-count engine ownership metadata distinguishes failed first writes from orphan
JSONL from unknown engines; strict loader validates committed records and removes
only uncommitted suffix. No silent cross-engine fallback.

The first test collection imported app defaults before SQLite isolation. PostgreSQL
bootstrap removed a stale PID file and failed to start; no media processing or
production settings write occurred. All subsequent tests/benchmarks explicitly
selected isolated SQLite before importing app modules. Keep this guard in place.


Final cleanup verification: guarded error extraction, concrete player/event/data
types, and conditional-hook fixes passed TypeScript and four isolated regressions.
Lint rules/config were preserved. Commit hooks passed on both completed commits.
The invalid backend run that overlapped a pre-commit stash was discarded; the
stable rerun passed all 89 tests. Feature committed in 76976c0; all hooks passed, including secrets, SDD and Ruff.
Only the nine unrelated dirty paths from the initial checkout remain. No push,
deployment, global engine activation or archival-as-shipped was performed.
