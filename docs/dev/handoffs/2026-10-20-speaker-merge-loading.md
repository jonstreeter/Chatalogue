# Handoff: speaker merge list loading

**Date:** 2026-10-20
**Spec:** `docs/dev/specs/2026-10-20-speaker-merge-performance-design.md`
**Plan:** `docs/dev/plans/2026-10-20-speaker-merge-performance.md`

## Status

User reports the Edit Speaker merge list stays on Loading. Traced the request;
no application code changed. Owner chose spec+plan. Both documents are written
and paired review passed. Owner approved implementation with `d`; all tasks todo
at the planning checkpoint.
Requested follow-up adversarial review corrected import-before-isolation safety,
end-to-end acceptance, red-test sequencing, and mandatory UI scope. Review report:
`docs/dev/audits/2026-10-20-speaker-merge-adversarial-review.md`.
Gate PASS after those corrections; it supersedes the earlier incomplete review.

## Next step

Checkpoint only these planning documents, then execute task 1's safe,
provider-matched performance diagnosis.

## Verification state

- Contract preflight for SpeakerModal.tsx, VideoDetailPage.tsx,
  routers/speakers.py and services/speaker_queries.py: no mapped authority.
- Static request trace complete; live request timing and database query plans
  have not been captured. No claim of a reproduced runtime bottleneck.
- No tests run; no implementation patch.
- Planning pair reviewed inline; SDD documentation validation and diff check PASS.

## Blockers

Implementation approval received with `d` on 2026-10-20.
Live request timing and reproduction remain task 1; do not infer a root cause
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
