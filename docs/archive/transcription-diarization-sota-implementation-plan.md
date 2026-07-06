# Transcription + Diarization SOTA Implementation Plan

This document defines a practical implementation plan for pushing PorchtimeIndex transcription and diarization quality closer to state of the art, while also improving the backlog of already-processed episodes.

It is based on:
- the current pipeline shape in [backend/src/services/ingestion.py](f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
- the current settings surface in [backend/src/main.py](f:/Projects/PorchtimeIndex/backend/src/main.py)
- inspection of recent real outputs from April 7, 2026
- adversarial review from ASR/diarization, backend/ops, and product/QA specialists

## Goal

Raise transcript quality in four areas:
- word accuracy
- punctuation and readability
- speaker assignment stability
- multilingual robustness

And do it for both:
- future processing
- existing transcripts already in the library

## Current Findings

Recent outputs show the pipeline is usable, but not state of the art.

Observed problems:
- proper nouns and names are often close but wrong
  - examples: `John DeLin`, `Frankie family`, `connections scandal`
- Spanish and code-switched content degrades sharply
  - recent Spanish sample was clearly below production quality
- diarization is too fragmentary
  - many single-word or very short `unknown` spans
  - many same-speaker interruptions where a tiny middle segment breaks an otherwise continuous speaker run
- output readability is decent overall, but not fully polished

High-confidence conclusion:
- English long-form monologue/interview content is decent but not SOTA
- multilingual and code-switched content is materially below SOTA
- diarization smoothing is the biggest immediate ROI area

## Success Criteria

We should treat "SOTA" here as measured product quality, not marketing language.

Primary product targets:
- fewer visible name/proper-noun errors
- fewer `unknown` speaker fragments
- fewer single-word or sub-2-second speaker flips
- better Spanish and mixed-language transcripts
- cleaner downstream search, clips, avatar prep, and speaker linking

Operational targets:
- no major slowdown in default processing for standard English episodes
- clear quality tiers so expensive reprocessing is used where it matters most
- safe reprocessing paths for existing transcripts

## Quality Profiles

Do not use one global quality target.

At minimum, define separate quality profiles for:
- English two-person interview
- English panel / multi-speaker discussion
- Spanish interview
- mixed-language / code-switched episode
- noisy remote or uploaded audio

Each profile should have its own acceptance gates for:
- unknown speaker rate
- same-speaker interruption rate
- named-entity precision
- human review score
- formatting/readability score

## Acceptance Gates

The plan should not be considered complete without measurable pass/fail gates.

Initial target structure:
- English interview smoothing gate:
  - reduce same-speaker interruptions by at least 60 to 70 percent
  - reduce `unknown` segment rate to under 3 percent on benchmark samples
  - keep reviewer-flagged bad merges under 2 percent
- Entity repair gate:
  - reviewed proper-noun precision at or above 95 percent
  - false-positive repairs are release blockers
- Multilingual routing gate:
  - Spanish and code-switched samples improve human review score versus current default
  - no destructive translation when transcript preservation is expected
- Backlog rollout gate:
  - dry-run report completed
  - preview sample set approved
  - pilot channel passes review
  - rollback verified before broader rollout

## Constraints

Important reality:
- it may not be possible to make every historical transcript truly SOTA because source audio quality varies
- some old outputs can be improved with post-processing only
- some will require full re-transcription
- some may still remain limited by audio quality, overlap, or language mixing

So the right goal is:
- best achievable quality per episode
- measured and prioritized by expected improvement

## Strategy

Use a 4-layer improvement stack:

1. Better transcript generation routing
- choose the right ASR path by language/content type

2. Better diarization post-processing
- smooth and repair fragmentation before final segment storage

3. Better transcript cleanup
- repair names/entities and improve formatting

4. Quality scoring + retrofit queue
- identify which existing transcripts justify cheap repair vs full reprocessing

## Major Workstreams

### 1. Quality Scoring Layer

Add transcript quality diagnostics per video.

Metrics to compute:
- segment count
- percent of `speaker_id is null`
- count of sub-1.5s segments
- count of same-speaker interruptions
- words per segment
- punctuation density
- repeated token anomalies
- language confidence
- code-switch likelihood
- named-entity confidence

Store per-video quality summary so we can:
- rank poor transcripts
- prioritize reprocessing
- show quality state in the UI later

Suggested implementation points:
- add a quality evaluator in [backend/src/services/ingestion.py](f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
- persist summary fields either on `video` or in a new transcript-quality table

Important requirement:
- store immutable before/after quality snapshots, not only current summary values
- the optimization controller must be able to compare runs historically

### 2. Diarization Smoothing Pass

Add a deterministic cleanup pass after diarization and before final segment persistence.

High-yield rules:
- absorb tiny `unknown` spans into adjacent same-speaker runs
- merge same-speaker segments separated by short gaps
- collapse A / tiny-B / A patterns when B is very short and low-confidence
- merge single-word fragments back into surrounding speaker where timing permits
- prevent punctuation-only or partial-word segments from becoming standalone turns

This should happen before the current transcript is treated as final.

Why this is first:
- it is cheaper than replacing the diarization model
- it improves many existing outputs even without re-transcription
- it directly addresses the most visible current artifact

Important distinction:
- this is speaker-label repair and transcript presentation repair
- it is not the same thing as improving the underlying diarization model
- overlapped speech, speaker-count drift, and true speaker confusion still require separate model/config evaluation

Suggested implementation points:
- extend the segment consolidation logic around `_consolidate_transcript_segments(...)` in [backend/src/services/ingestion.py](f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
- add a dedicated diarization repair pass rather than relying on generic consolidation alone

### 2B. Diarization Model / Config Evaluation

After repair heuristics are in place, run a separate evaluation of actual diarization quality improvement.

Candidate work:
- evaluate current pyannote path versus alternate configs/models already compatible with the stack
- test speaker-count stability
- test overlap-heavy samples separately
- evaluate whether threshold changes or profile matching changes reduce `unknown` without raising false assignments

Do not merge this with the repair pass in evaluation reporting. They solve different problems.

### 3. Language Detection + ASR Routing

Stop treating all episodes as equivalent.

Add language routing before transcription:
- English-only long-form: keep current fast default path if quality remains strong
- Spanish or multilingual episodes: route to a stronger multilingual ASR path
- code-switched episodes: use a multilingual model and preserve language-rich text

Needed behavior:
- detect likely language from title/description + short audio sample
- store language confidence and route reason
- allow manual override in settings or per-video later

Plan requirement:
- define candidate ASR/model routes explicitly
- benchmark Parakeet versus multilingual Whisper/faster-whisper or equivalent available paths
- define code-switch handling and preserve-language rules up front

Expected impact:
- biggest quality gain for Spanish and mixed-language episodes
- lower rate of nonsense substitutions and broken sentence structure

Suggested implementation points:
- add language classification near the transcription stage in [backend/src/services/ingestion.py](f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
- store on existing `video.transcript_language` and related metadata

### 4. Proper Noun / Entity Repair

Add a post-ASR cleanup pass for names and channel-specific entities.

Sources of truth:
- channel name
- video title
- description
- known speakers
- previously seen high-confidence entities from that channel

Repairs to target:
- host names
- guest names
- recurring channel vocabulary
- well-known brands/entities in titles

Guardrails:
- only repair when confidence is high
- preserve original text revision trail
- never silently over-rewrite uncertain phrases
- require phrase-level evidence from title/channel/speaker/entity memory
- keep repair decisions scored and reversible

This should reduce "almost correct" transcripts that still look low-quality.

### 5. Better Multilingual Punctuation / Formatting

Current punctuation is often acceptable in English, but not robust enough across languages.

Add cleanup for:
- sentence boundary repair
- quote/apostrophe normalization
- capitalization cleanup
- filler-collapse where appropriate
- language-specific punctuation conventions when practical

This should remain conservative and formatting-oriented, not semantic rewriting.

Hard rule:
- formatting cleanup must not become semantic rewriting
- any LLM-assisted cleanup should be review-gated or clearly separated from deterministic formatting repair

### 6. Model Evaluation Harness

Before changing defaults, build a repeatable evaluation set.

Need a small benchmark corpus:
- English interview
- English panel / multi-speaker overlap
- Spanish interview
- mixed Spanish/English
- low-quality audio
- high-value proper-noun-heavy episode
- phone/remote guest audio
- music/intro/outro interference
- short interjection-heavy panel exchanges

For each candidate pipeline, score:
- actual WER/CER on gold excerpts
- word accuracy proxy for broad triage only
- named-entity accuracy
- speaker fragmentation
- `unknown` rate
- human review score

Do not change the default production path without benchmark evidence.

## Gold Evaluation Corpus

The benchmark set needs hand-corrected reference material.

Minimum recommendation:
- at least 3 episodes per quality profile
- at least 10 fixed review windows per episode
- each window should intentionally cover:
  - speaker boundaries
  - names/entities
  - overlap or interruption risk
  - punctuation/readability
  - language switches when present

Gold corpus usage:
- use actual WER/CER only on these reviewed excerpts
- use heuristic proxies only for backlog prioritization

## Human Review Workflow

Before default enablement or bulk mutation, define a repeatable human review protocol.

Reviewer labels:
- better
- same
- worse
- bad merge
- bad speaker reassignment
- bad entity repair
- language regression

Review requirements:
- before/after diff on sampled windows
- pass/fail thresholds by quality profile
- explicit short-interjection preservation checks
- pilot signoff before channel-scale rollout

## Versioned Transcript Runs

Do not treat transcript state as just mutable rows on `video`.

Introduce a first-class transcript run model, for example:
- `TranscriptRun`
  - `id`
  - `video_id`
  - `input_run_id`
  - `pipeline_version`
  - `mode`
    - baseline
    - repair
    - diarization_rebuild
    - full_retranscribe
  - `status`
  - `started_at`
  - `completed_at`
  - `metrics_before_json`
  - `metrics_after_json`
  - `artifact_refs_json`
  - `rollback_state`
  - `model_provenance_json`

Why this is required:
- compare old vs new transcripts safely
- know which pass produced which output
- support rollback
- prevent ambiguous destructive rewrites

## Retrofit Plan For Existing Transcripts

Existing backlog should be upgraded in tiers, not all at once.

## Eligibility Checks Per Tier

Before assigning a video to a repair tier, validate prerequisites.

Examples:
- Tier A:
  - existing transcript present
  - low language-risk score
  - high confidence that repair is formatting/assignment only
- Tier B:
  - raw word timing data exists and is usable
  - transcript engine format is compatible with diarization-only rebuild
  - language confidence does not indicate likely ASR failure
- Tier C:
  - language-risk or transcript-quality score justifies full retranscription
  - capacity gate allows expensive work
  - rollback snapshot is ready
- Tier D:
  - value score justifies manual review
  - automated paths failed or confidence is too low

### Tier A. Cheap Repair For Almost-Good Transcripts

Apply to:
- English episodes with decent text but high speaker fragmentation
- episodes with low proper-noun error rate

Actions:
- run diarization smoothing
- run entity/name repair
- run punctuation/format cleanup

Additional requirement:
- preserve segment IDs where possible
- if merge/split occurs, emit an old-to-new segment mapping for downstream consumers

Do not re-transcribe.

Expected result:
- substantial visible improvement at low cost

### Tier B. Diarization-Only Rebuild

Apply to:
- episodes whose raw transcript words are mostly fine
- episodes with high `unknown` or speaker fragmentation

Actions:
- use existing raw transcript
- rerun diarization
- apply smoothing pass

Additional requirement:
- only eligible when raw timing/word data is present and trustworthy

Use existing channel/video redo-diarization flows already exposed in:
- [backend/src/main.py](f:/Projects/PorchtimeIndex/backend/src/main.py)

### Tier C. Full Re-Transcription

Apply to:
- Spanish episodes
- mixed-language episodes
- episodes with severe proper noun failure
- episodes with obviously degraded transcript text

Actions:
- rerun full transcription with better routing/model choice
- rerun diarization
- apply post-processing passes

Additional requirement:
- full retranscription must be capacity-gated and separated from user-facing live processing queues

Use existing purge/reprocess pipeline where possible.

### Tier D. Manual/High-Value Review Queue

Apply to:
- top channels
- highest-view videos
- avatar-critical videos
- clone-target videos
- transcripts flagged as still poor after automatic repair

Actions:
- prioritize for manual review tools
- expose revision diffs and confidence metrics

## Prioritization

Recommended implementation order:

1. Transcript quality scoring
2. Diarization smoothing / repair pass
3. Retrofit Tier A + Tier B queueing
4. Language detection + multilingual routing
5. Proper noun/entity repair
6. Benchmark harness and model comparison
7. Full backlog reprocessing controller

Why this order:
- steps 1 to 3 improve quality fastest with the lowest operational cost
- steps 4 to 6 improve the ceiling
- step 7 scales the improvements safely across the archive

## Backfill / Reprocessing Controller

Add a quality-driven maintenance job for historical upgrades.

Capabilities:
- scan all videos and compute quality score
- classify each video into Tier A/B/C/D
- batch queue repairs or reprocessing
- pause/resume safely
- store per-video upgrade outcome
- dry-run without transcript mutation
- idempotent queueing by transcript version + target optimization mode
- checkpoint progress at channel/video granularity

Useful filters:
- channel
- date range
- language
- popularity
- transcript quality score
- avatar relevance
- clone relevance

This should be implemented as an async background job, not a synchronous admin action.

## Backlog Job Semantics

Require queue correctness and idempotency.

Recommended unique key:
- `(video_id, optimization_tier, input_pipeline_version, target_pipeline_version)`

Required behavior:
- no duplicate expensive reruns for the same target state
- dry-run mode that only computes tiers and previews effects
- pause/resume support
- staged rollout by channel/profile
- explicit capacity limits for Tier C jobs

## Data Model Additions

Recommended additions:
- transcript quality summary
- language confidence / code-switch flag
- transcript pipeline version
- last transcript quality evaluation timestamp
- last transcript optimization pass timestamp
- optimization tier decision
- reprocess recommendation reason
- immutable transcript quality snapshot records
- transcript run / optimization run records
- segment mapping records when IDs change
- artifact invalidation state

This makes it possible to:
- compare old vs new outputs
- avoid repeated work
- selectively rerun only outdated pipeline versions

## Dependency Invalidation

Transcript mutation can stale downstream artifacts.

The plan must explicitly invalidate or refresh:
- transcript chunk embeddings / semantic index
- funny moments and humor summaries
- clip caption exports
- clip transcript-linked selections if they depend on segment identity
- avatar personality datasets
- speaker profile/sample derivations where timing or assignments changed
- clone context quality derived from transcript search

This should be tracked explicitly, either via:
- invalidation flags on `video`
- or a generic artifact invalidation table

## UI / Ops Recommendations

Recommended UI additions later:
- transcript quality badge on video rows
- "recommended upgrade" indicator
- bulk transcript optimization panel by channel
- per-video explanation of why quality is considered weak
- diff view between old and upgraded transcript
- pipeline version and rollback availability

Recommended operational controls:
- max concurrent full retranscriptions
- separate low-cost repair queue vs high-cost full ASR queue
- audit log of transcript upgrades
- rollback path using transcript revisions/backups

Hard requirement before broad backlog mutation:
- rollback must exist as a first-class workflow, not just a later recommendation

## Rollout Stages

Use a staged rollout, not an all-at-once backlog mutation.

Stage 0:
- score only
- no transcript mutation

Stage 1:
- repair preview on selected videos
- compare before/after diffs

Stage 2:
- pilot Tier A/B repair on one channel
- require human review signoff

Stage 3:
- broader Tier A/B repair
- continue immutable snapshot recording

Stage 4:
- selective Tier C full retranscription
- strictly capacity-gated

Stage 5:
- default-enable proven improvements for future processing

## Risks

Key risks:
- aggressive smoothing can incorrectly merge real short interjections
- entity repair can introduce false corrections
- multilingual routing can increase latency and cost
- full backlog re-transcription can create heavy GPU load
- changed transcripts may break existing speaker/avatar assumptions if not versioned carefully

Mitigations:
- keep conservative thresholds at first
- benchmark before default changes
- preserve prior transcript state for rollback
- separate cheap repair jobs from full reprocessing jobs
- version the transcript pipeline
- require human review gates before broader rollout
- treat no-op as a valid outcome when confidence is low

## Definition Of Done

This initiative is successful when:
- recent English episodes show materially fewer tiny `unknown` fragments
- Spanish/multilingual episodes stop producing obviously broken transcripts
- proper nouns are more reliable on major channels
- the app can score historical transcript quality and bulk-upgrade the backlog
- transcript upgrades measurably improve downstream search, clipping, speaker quality, and clone prep

## First Milestone

The first implementation milestone should be:

1. add transcript quality scoring with immutable snapshots
2. add a minimal evaluator and gold review set for before/after measurement
3. add diarization smoothing pass
4. define transcript run/version model and rollback contract
5. add a bulk "optimize existing transcripts" controller for Tier A/B repairs only in dry-run + pilot mode

This milestone gives the fastest visible quality improvement without committing yet to a wholesale ASR model switch.
