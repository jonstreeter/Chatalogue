# Speaker merge performance — Design Spec

**Date:** 2026-10-20
**Status:** Approved for implementation
**Owner:** Jon
**Plan:** `docs/dev/plans/2026-10-20-speaker-merge-performance.md`
**Scope:** Diagnose and remove the measured causes of slow merge-target loading and slow speaker merging. Preserve speaker eligibility, response data, and atomic reassignment.

---

## Motivation

The user reports a prolonged Loading state in Edit Speaker's merge picker and
roughly 30 seconds after selecting a target. These are separate request paths;
neither has been timed on the affected installation yet.

`frontend/src/components/SpeakerModal.tsx` requests every channel speaker from
GET /speakers before rendering targets. `backend/src/services/speaker_queries.py`
aggregates channel transcript durations, excludes totals at or below five seconds,
and sorts by duration. `backend/src/routers/speakers.py` adds embedding counts.
The merge POST bulk-reassigns four child tables, deletes sources, commits,
invalidates caches, then queries target duration and profile count before replying.
Parent callbacks can subsequently reload lists and segments.

This identifies work on the critical paths, not its measured cost. Database startup
already ensures speaker lookup indexes in `backend/src/db/database.py`; absence of
`index=True` on a model field is not evidence of a missing deployed index. SQLite's
configured 30-second lock timeout is also not proof of the reported cause. The
default provider is Postgres; verify the affected runtime rather than assuming it.

## Design tenets

1. Remember to follow the MSW deletion rule for all claims — no exceptions.
2. Measure each wait independently before selecting its optimization. Reuse the
   existing SQLAlchemy metrics and browser network timings; temporary probes must
   separate query execution, connection acquisition, commit, serialization, and UI work.
3. Preserve synchronous, atomic merge semantics. Success means committed storage,
   not an optimistic animation or an enqueued background operation.
4. Preserve the current API and picker data. No arbitrary truncation, invented
   performance target, new cache TTL, or hardcoded retry/timeout policy.
5. Keep changes within the existing query, merge service, and modal boundaries;
   change only code justified by the captured baseline.

## Data flow

```
Open merge in SpeakerModal
  -> GET /speakers?channel_id=<source channel>
  -> read_speakers -> _query_speaker_page_rows
  -> duration scope query/cache -> _build_speaker_reads -> target list

Confirm target
  -> POST /speakers/merge {target_id, source_ids}
  -> validate channel/source IDs -> merge_speakers_in_session
  -> update segments/profiles/chunks/avatars -> delete source speakers
  -> commit -> invalidate query caches -> target aggregates -> SpeakerRead
  -> onMerge callback -> parent refresh; onClose -> modal removed
```

Capture a timeline at these boundaries. On a representative disposable copy of
the affected data, compare the original and changed code using identical data,
provider, cache state, and workload. Restore the copy between merge measurements
so each run moves the same rows. Never benchmark a destructive merge against
the user's real speakers. Read-only inspection of live timings, plans, indexes,
and locks is appropriate; production writes or index DDL are not diagnostic probes.
The benchmark must validate its destination before importing any `src` module:
`db/database.py` resolves its global engine at import and can start embedded
Postgres. Use a freshly created disposable database owned by the benchmark and
bind its explicit provider/URL before application imports. Fail closed on missing,
unowned, or live destinations; string inequality between connection URLs is not
proof of isolation. SQLite paths must be resolved and Postgres server/database
identity checked before writes. Restore only this owned destination. Bootstrap
tests must prove rejected destinations never reach app imports, connect, or spawn
an embedded database process. Do not print credentials in diagnostics.

### Selected implementation approach

Retain the existing endpoints and optimize their measured critical paths. First
record the provider, table and affected-row counts, installed indexes, request
timings, query plans, and whether other jobs contend for storage. Then apply the
smallest behavior-preserving query/transaction/UI correction supported by evidence:

- For list SQL cost, reshape the shared scope query only with result-equivalence
  tests for channel/video/search/pagination and cold-cache execution. Preserve the
  duration threshold and duration ordering. A LIMIT alone does not avoid aggregation.
- For merge cost, retain set-based reassignment and a single transaction. Remove
  demonstrated redundant ORM loads or queries, or correct a proven query plan or
  lock lifecycle problem at its owner. Inspect every caller before changing a helper.
- For response/refresh cost, avoid demonstrated duplicate fetches or synchronous
  rendering work while retaining fresh target counts and parent segments. Report
  POST completion separately from parent refresh completion.
- Add or repair an index only if installed-index inspection and query plans show
  it is necessary. Use `_ensure_indexes` for an idempotent additive index, verify
  SQLite and Postgres compatibility, and measure write/startup cost. No speculative
  schema edits and no running startup migrations against the live store as a test.

If the measurements instead require a new lightweight API, changed target
eligibility, denormalized totals, asynchronous merges, or unrelated subsystem work,
update and review this spec and plan before implementation of that design. The
measurement task can finish independently; an unconfirmed hypothesis is not a fix.

## Interface contracts

- **Client → API:** Keep GET /speakers parameters and List[SpeakerRead] response,
  including accurate duration, embedding count, and existing search/pagination.
  The modal continues to exclude its source and searches all returned targets.
  Keep POST /speakers/merge's MergeRequest, SpeakerRead result, same-channel
  validation, and existing error semantics. Do not introduce partial success.
- **Service → persistence:** `merge_speakers_in_session` remains a transaction
  participant; its caller owns commit/rollback. Move TranscriptSegment,
  SpeakerEmbedding, TranscriptChunkEmbedding, and Avatar ownership before source
  deletion; preserve matched profile IDs and unrelated records. Invalidate caches
  after successful commit. Query optimizations must retain mutation invalidation.
- **Persistence:** No table, column, storage format, or data migration is selected.
  Any evidence-required additive index belongs in the existing startup mechanism.
  Preserve both supported database providers.
- **UI:** Existing confirmation and pending state remain until a definitive POST
  result. Do not automatically retry an ambiguous merge after disconnect.
  New error/retry controls, stale-response machinery, or reconciliation UI require
  evidence that they cause the reported wait or are necessary to keep the selected
  correction safe; they are not unconditional additions to this performance fix.

## Edge cases & failure modes

- Preserve empty scopes, generated speaker names, extras, duration-boundary
  eligibility, source exclusion, and search across the complete eligible set.
- Preserve existing handling of absent targets, self-targets, invalid/mixed-channel
  sources, and sources removed by another operation; tests characterize current
  route behavior instead of silently redefining it.
- Failure during any reassignment/delete must roll back every ownership change.
  A failure after commit is not evidence that the merge rolled back; do not blindly
  replay it or claim unchanged data. Allow refresh to reconcile an uncertain result.
- Test a contending writer and concurrent target/source changes on disposable
  storage. Do not shorten lock timeouts to manufacture a faster response.
- Clearing caches after a merge makes the next list request cold: include this
  path in performance evidence, not just a warmed picker reopen.

## Adversarial review

Spec-only review, 2026-10-20, applied inline; no subagents spawned:

- **Architect — HIGH resolved:** A lightweight picker endpoint would change the
  contract before the bottleneck was established. Selected existing-endpoint
  optimization instead. Alternatives: dedicated lightweight endpoint (less aggregate
  work, changes displayed data/API); materialized totals/background merge (more
  state and failure modes). Neither is justified yet. No new architecture or ADR.
- **Planner — BLOCKER resolved:** No runtime reproduction exists. Task 1 must
  establish the baseline and record a concrete correction before later tasks run.
- **Simplification — HIGH resolved:** New TTL caches, arbitrary caps, background
  jobs and generic performance infrastructure are excluded.
- **Code reviewer — HIGH resolved:** Startup already creates relevant indexes;
  inspect actual storage and preserve commit-before-success and rollback tests.

Final paired review, 2026-10-20: architect, planner, simplification-expert and
code-reviewer applied inline to this spec and its linked plan; no subagents spawned.
The planner's HIGH finding that repeated merges shrink the benchmark input was
resolved with explicit disposable-store safeguards and equivalent restoration
between runs. The code-reviewer's HIGH finding about ambiguous post-commit failures
was resolved by prohibiting automatic destructive retries and requiring refresh
reconciliation. The architect's evidence gate is explicit in task 1; new architecture
requires renewed review and owner approval. The simplification review retained only
the task-specific harness and existing app boundaries.

Consolidated result: no open BLOCKER/HIGH findings; no unanswered product decisions.
Resolution order: phase evidence, safe reproduction, behavior/atomicity parity,
verification. Final document review gate PASS. Status remains In Review pending
owner approval for implementation; performance diagnosis is task 1, not a claim
that either runtime delay has already been reproduced or fixed.

### Requested follow-up adversarial review

The earlier paired-review PASS was incomplete. The requested fresh review found
and resolved benchmark import-order isolation (BLOCKER), end-to-end acceptance
(HIGH), red-test sequencing (HIGH), and unnecessary mandatory UI work (MEDIUM).
Full findings and re-review outcome:
`docs/dev/audits/2026-10-20-speaker-merge-adversarial-review.md`.
That review supersedes the earlier gate record: PASS after corrections; runtime
diagnosis and implementation approval still outstanding.

## Out of scope

Speaker recognition quality, merge suggestions, undo/history, new merge semantics,
tenancy/auth changes, provider migration, unrelated ingestion optimization, and a
general observability dashboard. No push, deployment, or live merge is authorized
by this planning request.

## Open questions

2026-10-20: Owner approved the revised plan with `d`; implementation may begin.

No unresolved product/API decisions for the selected behavior-preserving approach.
Runtime facts to resolve in task 1: affected provider and build; reproducible scope;
which phases consume the delay; contention and installed index state. Record
numeric acceptance evidence from that baseline rather than inventing a latency SLA.
Before optimizing, record an owner-approved acceptance criterion for each
end-to-end wait, informed by the measured baseline: picker-open to selectable
targets, and confirmation to committed response/modal close and fresh parent data.
Backend phase improvement alone cannot close either symptom. A residual long wait
requires continued diagnosis even when the speedup exceeds measurement variation.
If acceptable residual latency is unclear, obtain the owner's decision using the
measured timings; do not silently invent a numeric target or call the issue fixed.
If the affected dataset/runtime is unavailable, mark performance diagnosis blocked;
synthetic evidence cannot establish that the user's reported delay was fixed.
