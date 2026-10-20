# Speaker merge spec and plan — adversarial review

**Date:** 2026-10-20
**Scope:** `docs/dev/specs/2026-10-20-speaker-merge-performance-design.md` and
`docs/dev/plans/2026-10-20-speaker-merge-performance.md`
**Method:** All four repository lenses applied inline; no subagents spawned.
**Result:** PASS after the document corrections below; implementation remains
unapproved and runtime performance remains unverified.

Remember to follow the MSW deletion rule for all claims — no exceptions.

## Architect

**HIGH A1 — A measurable backend speedup could leave the reported wait unresolved.**
The acceptance rule was only removal of an expensive cause and improvement beyond
run variation. A substantially slower-than-acceptable picker or merge could still
pass it. Phase timings also do not capture parent refresh and rendering completion.

Resolved: task 1 now records owner-approved criteria for each end-to-end user
wait based on the actual baseline. Task 5 requires both criteria, not just SQL
gains. Unclear acceptable residual latency becomes an explicit owner decision
after measurements; no arbitrary SLA is introduced.

Alternatives considered: retain APIs and optimize measured cost; create a dedicated
picker endpoint; materialize totals or queue merges. Retain the first because no
runtime evidence yet supports a new API or persistence model. Evidence requiring
one of the alternatives must return to spec review. No ADR needed for this
retention of the existing architecture.

## Planner

**HIGH P1 — A red-only task conflicts with the verified-slice/commit sequence.**
Task 2 demanded failing tests, while the surrounding instructions required each
task verified before advancing. This could either block task 3 or encourage a
commit containing intentionally failing tests. Characterization tests were also
conflated with regressions that must fail before a fix.

Resolved: task 2 is explicitly a local red checkpoint. Characterizations pass
against current code; defect tests demonstrate the expected failure. Each defect
test and its task 3/4 fix land together after green checks. Task numbering retained.

## Simplification expert

**MEDIUM S1 — The plan mandates UI work before proving its relevance.**
New retry controls, stale-response protection, and a reconciliation flow were
unconditional despite the report being about latency. They may be useful, but
that does not establish necessity for either delay.

Resolved: removed these unconditional additions. Implement lifecycle changes
only when evidence ties them to the symptom or they are needed to preserve safety
of the chosen correction. Existing no-automatic-retry behavior is preserved.
Retain the task-specific benchmark because destructive comparisons need restored
data; reject a generic profiling platform and speculative caches.

## Code reviewer

**BLOCKER C1 — Benchmark safety checks could run after app imports start storage.**
`backend/src/db/database.py` resolves its global provider and engine at import;
without an explicit URL its Postgres path calls `ensure_embedded_postgres()`.
`speaker_merge.py` imports database models at module scope. A live-database
rejection performed after those imports is too late. Comparing URL strings is
also insufficient: aliases or differently formatted URLs can name the same store.
The previously designated test module itself imports application models at load.

Resolved: validate destination and provision/verify a fresh benchmark-owned store
before app imports; bind its explicit provider/URL before importing routes/models.
Resolve SQLite paths and check Postgres server/database identity before writes.
Reject missing/live/alias/unowned destinations without imports, connections, or
embedded process startup. Put these tests in an independent bootstrap-safe module
`backend/src/test_speaker_merge_benchmark.py`. No change to application code yet.

## Consolidated outcome

The three highest-priority findings were C1 (unsafe import order), A1 (insufficient
user-visible acceptance), and P1 (red-test commit sequencing). All were resolved
in the spec and/or plan, along with S1. The earlier paired-review PASS was
incomplete; this review supersedes it. No necessary review finding remains open.

The five runtime questions, in resolution order, are:

1. Which provider/build/data scope actually exhibits each reported wait?
2. Where does each end-to-end timeline spend its time, including parent refresh?
3. Do actual indexes, query plans, or contending transactions explain that cost?
4. What residual wait satisfies the owner for each measured baseline?
5. Does the smallest safe correction meet both criteria on equivalent restored data?

These are explicit task-1/task-5 evidence gates, not answered facts or authority
to implement an unreviewed architecture. Review the changed safety, acceptance,
and sequencing provisions again: they now have concrete tests/checkpoints and do
not conflict with the preserved API/atomicity contract. No additional necessary
finding arose. Mechanical validation is recorded in the plan verification log.
