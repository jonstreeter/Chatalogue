# Handoff: optional Phonon engine implementation

**Date:** 2026-10-15
**Spec:** `docs/dev/specs/archive/2026-10-15-phonon-engine-design.md`
**Plan:** `docs/dev/plans/archive/2026-10-15-phonon-engine.md`
**Commit:** 699db74 verified integrated implementation; closeout commit follows this archived record

## Closed — 2026-10-16

Owner authorized continuation; implementation pushed through 699db74. Local build/cache updated and verified. This baton is archived with its completed plan/spec. No implementation work remains. Production engine selection/media were untouched.

## Final verification record — 2026-10-16

Owner explicitly approved the full integration revision; do not request plan
approval again. Tracker: 13/13 complete. This final record supersedes all historical
sections below. Implementation and verification are complete; push/shipping/archive were authorized and completed during closeout. No required owner decision blocks the implementation.

Checkpoints: planning 02a8360, native proof 0c5ad07, installer 5c35ea1,
owned worker 3245eae, app phases b8840f4, integrated settings 82e8139,
setup/benchmarks 0ec0c16; final tracker checkpoint follows this note.

Implemented app-owned hash-pinned private installation and persistent native worker;
no URL/key controls, Docker prerequisite or external inference service. Real setup,
load/test/unload, busy refusal, competing-GPU release and shutdown/parent-exit
ownership are verified. BF16/K16/six duration buckets retained. Existing bounded
chunks, absolute clocks/overlap, silent/interrupted checkpoints and queue provenance
remain; existing engines, routing and defaults are preserved. Native platform is
Windows x64/Python3.11/NVIDIA hardware BF16 only, tested RTX5090.

Verification: full backend108 PASS; frontend typecheck/build PASS; Ruff whole repo
PASS; full SDD validate PASS; actual committed-range impact from 02a8360 PASS with
spec/ADR acknowledgement. Frontend lint0errors/43unchanged warnings; plain command
hit EPERM on generated playwright-report, so same rules ran with only that generated
folder ignored. Two isolated Chromium settings/queue tests PASS and live Playwright
against isolated SQLite/scratch settings verified real model readiness/unload and
save/reload. The existing live-data E2E suite was not run: it scans/deletes installed
media. No gate bypass. Inline review resolved necessary findings; no subagents.

Measured actual app phases (routing disabled only in benchmark processes):
120s audio 6.021/7.136s, separate initial load46.664s; 2880s audio
125.762/128.842s, load46.025s. Same worker per pair, no structural timing issues;
383/8704 words,41/1043 segments; long5chunks. GPU release0.202s followed by
49.451s next short transcription including reload/newPID. Competing model inference,
download and queue wait excluded. Native long measurements are slower than prior
Docker app timings; no throughput improvement, WER or batch128 claim.

Local install update complete: frontend/dist rebuilt and verified standalone runtime
moved under locks into `backend/runtime/phonon/installs/b0806992a441ad3623fe35e78d30063ac845467f3f68289700a636ba19211b99`.
Manifest/model hashes, model/DLL imports, real GPU readiness and confirmed unload
passed at that final path. The ignored installed-path.txt pointer now names it.
Never promote the original probe venv with backend-sharing .pth. The broad Windows
updater was not run because it pulls and upgrades unrelated dependencies; this
checkout already contains the new source/dependencies, with its local assets/cache
updated directly. No production backend is running; start/restart the app and refresh
Settings to use the new build. Default selection/global activation was not changed.
Do not start queues, reprocess existing media, or push/deploy without authorization.

Evidence under ignored `backend/data/asr-benchmark/`:
- `app-phonon-managed-120s-final.json`, `app-phonon-managed-2880s-final.json`
- `phonon-local-install-final.json` (final path readiness, actual child stopped)
- `phonon-ui-evidence.json` (isolated settings, secret retained)
- `output/playwright/phonon-ready.png` is the reviewed live UI screenshot.
All owned test API/browser/Vite and GPU workers are stopped. Nine initial unrelated
dirty paths remain untouched; no stash remains. Super Looper was unavailable;
implement→verify→review→resolve→tracker/Handhold ran inline.

Implementation push CI passed: Lint, SDD and Secret Scan. Closeout validation
and remote CI results are reported by the finishing session;
this archived baton has no next implementation action.

Historical status/next-step sections below are superseded by this revision.

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
