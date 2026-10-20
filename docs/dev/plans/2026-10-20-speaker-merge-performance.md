# Speaker merge performance — Implementation Plan

**Date:** 2026-10-20
**Status:** Not started
**Spec:** `docs/dev/specs/2026-10-20-speaker-merge-performance-design.md`

## Summary

Measure and correct the causes of slow merge-target loading and the reported
roughly 30-second merge. Preserve existing API responses, picker eligibility,
and atomic reassignment. This is a tier-3 plan because it crosses UI, backend,
and storage boundaries. Planning does not authorize implementation or live merges.

## Affected modules

- `frontend/src/components/SpeakerModal.tsx` — picker lifecycle, request timing,
  measured UI corrections only; new lifecycle controls require causal evidence.
- `backend/src/routers/speakers.py` — list assembly, merge route and response costs.
- `backend/src/services/speaker_queries.py` — shared scope queries and cache behavior.
- `backend/src/services/speaker_merge.py` — atomic set-based ownership reassignment.
- `backend/src/db/database.py` — inspect existing metrics/indexes; change only if
  measured evidence requires it, using existing index/startup conventions.
- `backend/src/test_speaker_loading_optimization.py` — scope/query regressions.
- `backend/src/test_speaker_merge.py` — route and atomicity regressions.
- `backend/src/test_speaker_merge_benchmark.py` — bootstrap-safe isolation tests
  without application model imports at module load.
- `frontend/e2e/speaker-merge.spec.ts` — deterministic picker/merge browser tests.
- `scripts/benchmark_speaker_merge.py` — reproducible, disposable-store comparison
  harness, needed because a destructive merge cannot be repeated on unchanged data.
- `docs/dev/handoffs/2026-10-20-speaker-merge-loading.md` — measured diagnosis and
  command/results when pausing; current plan tracker remains progress authority.

Inspect callback consumers in `frontend/src/components/SpeakerList.tsx`,
`frontend/src/pages/SpeakerDetailPage.tsx`,
`frontend/src/pages/channel/ChannelSearch.tsx`,
`frontend/src/pages/video/VideoDetailPage.tsx`, and
`frontend/src/store/useSpeakersTabStore.ts`. Edit a consumer only if the baseline
locates redundant/stale work there; preflight the exact path before diagnosing it.

## Task tracker (source of truth for progress)

Update status as you go. Keep task numbering stable. Start after owner approval
and a scoped planning commit. Each task requires its verification before the next.
Task 2 is a local red-test checkpoint, not an independently committable slice:
record the expected failure, then pair each regression with its task 3 or 4 fix
and green checks in the same commit. Characterization tests must pass unchanged;
only a test for the actual corrected defect should fail before its fix.

| # | Task | Files | Size | Status |
| --- | --- | --- | --- | --- |
| 1 | Establish reproducible list and merge baselines; split request/SQL/commit/response/UI timings; record measured corrections and owner-approved end-to-end acceptance criteria | `scripts/benchmark_speaker_merge.py`, `backend/src/test_speaker_merge_benchmark.py`, `docs/dev/handoffs/2026-10-20-speaker-merge-loading.md`, this plan | M | ☐ todo |
| 2 | Add failing characterization/regression checks for the measured causes and preserved list/merge contracts | `backend/src/test_speaker_loading_optimization.py`, `backend/src/test_speaker_merge.py`, `frontend/e2e/speaker-merge.spec.ts` | M | ☐ todo |
| 3 | Apply the measured list-path correction; change picker lifecycle handling only if evidence requires it; verify shared callers | `backend/src/services/speaker_queries.py`, `backend/src/routers/speakers.py`, `frontend/src/components/SpeakerModal.tsx`, `backend/src/db/database.py` only if proven necessary | M | ☐ todo |
| 4 | Apply the measured merge-path correction while retaining transaction ownership, ownership integrity, response fields, and fresh parent state | `backend/src/services/speaker_merge.py`, `backend/src/routers/speakers.py`, `frontend/src/components/SpeakerModal.tsx`; implicated callback consumer only with recorded evidence | M | ☐ todo |
| 5 | Compare original and fixed requests on restored representative data, verify browser flow and rollback; run required verification and remove temporary probes | `scripts/benchmark_speaker_merge.py`, `frontend/e2e/speaker-merge.spec.ts`, this plan, `docs/dev/handoffs/2026-10-20-speaker-merge-loading.md` | M | ☐ todo |

### Execution and verification details

1. Confirm the running build and provider through existing app metrics/config
   without printing credentials. Capture browser/network timing for opening the
   picker, searching, confirming a merge, modal closure, and parent refresh.
   Inspect existing indexes and read-only plans. Build the harness so it requires
   an explicitly owned disposable database, rejects the configured live database, records
   schema/provider/row counts/cache/workload conditions, and restores equivalent
   data between merge runs. It must never start the app's live migration routine.
   Validate arguments and provision/verify isolation before importing any `src`
   module; `database.py` can start embedded Postgres at import. Bind an explicit
   disposable provider/URL before model/route imports and never load live defaults
   into the benchmark process. Resolve SQLite paths and verify Postgres
   server/database identity, not just URL string inequality. Operate only on a
   fresh benchmark-owned database; restore only that database. Bootstrap tests in
   `backend/src/test_speaker_merge_benchmark.py` must import the standalone harness
   without app imports and prove absent/live/alias/unowned destinations are rejected
   before application import, connection, or embedded-process startup. Keep these
   tests separate from existing merge tests that import app models at module load.
   Include isolated and representative contention runs on that copy. Use
   `backend/.venv/Scripts/python.exe scripts/benchmark_speaker_merge.py --help`
   from the root to document its exact invocation once implemented. Record
   before/after commands and results in the verification log. Do not advance if
   a slow phase cannot be reproduced or attributed; report evidence and the
   missing runtime access rather than selecting a speculative fix. Record measured
   end-to-end acceptance criteria for both symptoms and obtain owner agreement if
   acceptable residual latency is unclear before calling the fix complete.
2. Use in-memory fixtures for correctness and disposable provider-matched storage
   for timings. Check pre-fix failure for the actual regression; preserve existing
   five-second threshold, sorting, scope/search/pagination, response aggregates,
   all four ownership moves, matched profile references, and rollback. Mock network
   delays/errors in browser tests; these prove lifecycle behavior, not backend speed.
3. Run the list regression tests cold and after invalidation, plus the targeted
   browser picker cases. If an index is necessary, inspect its actual plan and
   verify repeated initialization and both providers before accepting it. Do not
   modify unrelated query consumers to compensate for changed results.
4. Run ownership/route tests, inject a failure after an earlier reassignment, and
   prove no partial commit. Test unchanged validation errors and lost-response
   reconciliation without automatic destructive retry. Verify post-merge lists,
   segments, duration and profile counts through each affected callback consumer.
5. Retain raw timing artifacts outside tracked source. Compare distributions and
   per-phase costs using equivalent pre-merge snapshots, cold/warm caches, provider
   and workload. Record the observed spread and measured improvement for each
   symptom; accept only when the reproduced expensive cause is removed, correctness
   is unchanged, improvement exceeds observed run variation, and both owner-approved
   end-to-end criteria from task 1 pass. Backend-only gains or a smaller but still
   prolonged wait cannot close the bug. The reported
   30 seconds is a symptom, not an invented SLA. If only one symptom improves,
   keep the other task open. Revert temporary logging and diagnostic-only code.

## Test plan

- **Unit:** Scope result equivalence and cold-cache query behavior in
  `test_speaker_loading_optimization.py`; pre-import harness isolation guards in
  `test_speaker_merge_benchmark.py`; transaction rollback and ownership preservation
  in `test_speaker_merge.py`.
- **Integration:** Real route calls against disposable storage, with accurate
  SpeakerRead fields, same-channel filtering, missing/self targets, cache
  invalidation, and concurrent writes. Verify the affected provider; SQLite-only
  unit success cannot validate a Postgres performance claim.
- **Manual / browser:** Open the affected video, Edit Speaker, open merge, search
  a known target, cancel, reopen, confirm a disposable merge and inspect refreshed
  transcript/list data. Exercise empty lists and failed POSTs. Add retry, context
  switching and disconnected-response cases only for lifecycle behavior changed
  by the measured correction; do not add a new recovery flow without that need.
  Never merge real speakers as a verification fixture.
- **Smallest checks first:** From `backend/`, `.venv/Scripts/python.exe -m pytest
  src/test_speaker_loading_optimization.py src/test_speaker_merge.py
  src/test_speaker_merge_benchmark.py -q`;
  from `frontend/`, `npm run test:e2e -- e2e/speaker-merge.spec.ts`.
- **Final implementation checks:** Follow `docs/dev/workflows/verify.md`:
  `npx tsc -b` and `npm run lint` from `frontend/`; `ruff check .` from root;
  `.venv/Scripts/python.exe -m pytest src -q` from `backend/`;
  `npm run test:e2e` from `frontend/`; `python scripts/check_sdd_docs.py validate`
  and `git diff --check` from root. Record blockers honestly; do not bypass gates.

## Risks

- Misattributing the 30-second symptom to an aggregate, missing index or lock
  timeout without timing — architect; task 1 gates implementation selection.
- Shared query optimization changes eligible targets or search results — code
  reviewer; result-equivalence and boundary fixtures before edits.
- Failed merge leaves partial ownership changes or UI reports premature success
  — code reviewer; rollback injection and commit-before-success contract.
- Benchmark changes live data or repeated runs measure shrinking work — planner;
  explicit disposable-store guard and restored baseline per run.
- New cache or asynchronous workflow hides the delay while adding stale state —
  simplification expert; preserve existing architecture and invalidation.
- Query/index correction benefits one provider but regresses another — architect;
  exercise the changed SQL/startup behavior on both, measure the affected runtime.

## Open questions

No owner decision remains for this scoped approach beyond implementation approval.
Task 1 resolves runtime facts; its output must identify the exact correction and
test before tasks 2–4 proceed. A required contract/architecture change returns to
spec review and owner approval; it is not authorized by a conditional task row.

## Adversarial review

2026-10-20: Final pair reviewed inline with architect, planner,
simplification-expert and code-reviewer lenses; no subagents spawned.

- **Architect — HIGH resolved:** Distinguish known expensive dependencies from
  unmeasured root cause. Task 1 explicitly gates all optimization choices; existing
  API/synchronous transaction chosen over new endpoint or materialized totals.
- **Planner — BLOCKER resolved:** Repeated merge benchmarks mutate the input.
  Require disposable-store safeguards, equivalent restoration per measurement,
  and phase-specific proof. Missing affected runtime blocks diagnosis, not evidence.
- **Simplification — HIGH resolved:** Avoid generic profiling infrastructure,
  arbitrary caps/TTLs and optimistic/background merges. Keep only the reproducible
  harness necessary for this destructive performance comparison.
- **Code reviewer — HIGH resolved:** Preserve same-channel route semantics,
  accurate totals, cache invalidation, rollback, and disconnected-response handling.
  Explicitly test existing indexes instead of inferring them from model metadata.

Consolidated outcome: no open BLOCKER/HIGH findings in the planning pair.
Resolution order was evidence -> safe reproducibility -> API/transaction parity ->
verification. Document review gate PASS; implementation not started or approved.
Runtime performance remains unverified and is not represented as fixed.

### Requested follow-up adversarial review

The earlier review PASS was incomplete. A fresh four-lens review found and
resolved C1 (BLOCKER: imports before storage isolation), A1 (HIGH: insufficient
end-to-end acceptance), P1 (HIGH: failing-test commit sequence), and S1 (MEDIUM:
unproven mandatory UI expansion). Full findings and resolutions are in
`docs/dev/audits/2026-10-20-speaker-merge-adversarial-review.md`.
Re-review of the changed provisions: PASS; no necessary review finding open.
This result supersedes the earlier review record, not the pending runtime gates.

## Verification log

- 2026-10-20: Owner approved revised plan with `d`. Planning checkpoint authorized;
  task 1 diagnosis follows. Review records above describe the pre-approval state.

- 2026-10-20: Requested follow-up adversarial review completed; corrected safety,
  acceptance, test sequencing and scope. `python scripts/check_sdd_docs.py validate`
  and `git diff --check` PASS. No application edits or runtime tests.

- 2026-10-20: Read relevant modal, list query, merge route/service, startup indexes,
  existing test fixtures and callback consumers. Contract preflight returned no
  mapped authority for the initial paths. Static review only; no live timings.
- 2026-10-20: Spec-only and final paired adversarial reviews completed inline;
  necessary findings resolved as recorded above. Application tests not run for
  this documentation-only change.
- 2026-10-20: `python scripts/check_sdd_docs.py validate` PASS;
  `git diff --check` PASS. Status reports this plan Not started, 0/5 done.
