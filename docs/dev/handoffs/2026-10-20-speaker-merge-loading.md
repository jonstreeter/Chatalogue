# Handoff: speaker merge list loading

**Date:** 2026-10-20
**Spec:** `docs/dev/specs/2026-10-20-speaker-merge-performance-design.md`
**Plan:** `docs/dev/plans/2026-10-20-speaker-merge-performance.md`

## Status

User reports the Edit Speaker merge list stays on Loading. Traced the request;
no application code changed. Owner chose spec+plan. Both documents are written
and paired review passed. Owner approved implementation with `d`. Planning
checkpoint is `f3804be0e`; task 1 is in progress. No application code changed.
Owner now confirms the speaker list is fast. Merge timing remains unverified.
Requested follow-up adversarial review corrected import-before-isolation safety,
end-to-end acceptance, red-test sequencing, and mandatory UI scope. Review report:
`docs/dev/audits/2026-10-20-speaker-merge-adversarial-review.md`.
Gate PASS after those corrections; it supersedes the earlier incomplete review.

## Next step

Resolve whether merge confirmation still takes roughly 30 seconds (question
pending). If persistent, identify a current source/target and build task 1's
disposable Postgres reproduction; never send a test merge against live speakers.
If both delays have disappeared, record that neither is currently reproducible
and close/park the investigation without manufacturing a patch or benchmark.

## Verification state

- Contract preflight for SpeakerModal.tsx, VideoDetailPage.tsx,
  routers/speakers.py and services/speaker_queries.py: no mapped authority.
- Static request trace complete; live request timing and database query plans
  have not been captured. No claim of a reproduced runtime bottleneck.
- No tests run; no implementation patch.
- Planning pair reviewed inline; SDD documentation validation and diff check PASS.
- Planning commit hooks PASS. Git's subsequent geometric repack failed; the
  planning commit succeeded. Do not conflate that maintenance error with a hook.
- Read-only runtime and browser diagnosis performed; see measurements below.

## Blockers

Implementation approval received with `d` on 2026-10-20.
Merge request timing and reproduction remain task 1; do not infer a root cause
from an aggregate query, model metadata, or the SQLite 30-second lock timeout.

## Diagnosis worth keeping

SpeakerModal.fetchChannelSpeakers requests GET /speakers with channel_id and
no limit. read_speakers calls _query_speaker_page_rows, whose unlimited path
calls _query_full_speaker_scope_rows. On a cache miss, that aggregates transcript
durations for the whole channel, filters totals above five seconds, sorts all
speakers by duration, then counts their embeddings in _build_speaker_reads.
The modal waits for all of this before showing any merge targets. This is a
concrete expensive dependency, but its share of the reported delay is unmeasured.
The list also displays duration, so bypassing aggregation changes visible data;
adding a limit alone still requires the aggregate and truncates local search.
Inspect backend/src/test_speaker_loading_optimization.py for existing fixtures.
Nine pre-existing dirty documentation/tooling paths were left untouched.

User additionally reports roughly 30 seconds after selecting a merge target.
POST /speakers/merge calls merge_speakers_in_session, which bulk-updates
TranscriptSegment, SpeakerEmbedding, TranscriptChunkEmbedding and Avatar,
deletes source speakers, and commits. The route then recalculates target
speaking time and embedding count before replying, and invalidates list caches.
SpeakerModal waits for the POST response before closing. The observed wait
therefore can include writes, commit/lock waits, and response aggregates;
their individual timings remain unmeasured. Investigate both delays together.

## Runtime evidence (2026-10-20)

- `/settings/db-health`: Postgres; pool size 5, no queued/running jobs at sampling.
  Read-only connection confirms database `chatalogue` on loopback port 55432.
  No blocked sessions were observed. Version endpoint reports checkout
  `f3804be0e`; this is not proof of the backend's loaded-code revision.
- Exact screenshot title maps to video 16980 / channel 9. Source label in the
  screenshot is a generated name, not a database ID. Initial probing of ID 957 /
  channel 1 was unrelated; do not use that timing as evidence for this video.
- GET `/speakers?channel_id=9`: 0.439 s then 0.050 s, 3,029 rows, 536,967 bytes.
  GET `/videos/16980/segments`: 0.020 s, 61 segments. Current assigned speakers
  are Dan McLellan (6188) and Speaker 3280 (17567); original merge is not replayable.
- The shared duration aggregation read 2,795,192 total-segment scope via parallel
  scan/join and completed in 391.304 ms in read-only EXPLAIN ANALYZE. All relevant
  speaker lookup indexes exist. EXPLAIN without ANALYZE for source 17567's segment
  UPDATE uses `idx_transcriptsegment_speaker_id`; no write was executed.
- Browser picker and search `Dan` worked. 3,119 page buttons were rendered. The
  4.932-second tool observation includes AX extraction and must not be reported
  as render latency. No browser merge, rename, or deletion performed.
- Owner replied `It is fast now` to the list question. Pending separate question
  about whether the actual merge is still slow. Do not optimize based on the
  earlier unmeasured assumptions. Safe replay harness is deferred until needed.
- Repeat list timing from PowerShell:
  `Measure-Command { $null = Invoke-WebRequest 'http://127.0.0.1:8011/speakers?channel_id=9' }`
