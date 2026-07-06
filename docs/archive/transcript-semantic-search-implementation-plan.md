# Transcript Semantic Search Implementation Plan

## Purpose

This document defines a practical implementation plan for adding semantic transcript search to PorchtimeIndex and using it to improve avatar personality dataset preparation.

The goal is intentionally two-sided:

- better user-facing transcript search features
- better avatar personality dataset curation and preparation

The system should be treated as a shared semantic retrieval layer that serves both product surfaces.

The payoff is:

- better retrieval of paraphrased ideas and arguments
- stronger de-duplication during dataset prep
- more balanced and semantically diverse training packages
- easier harvesting of recurring references, metaphors, and reasoning patterns

This plan is written against the current repo architecture:

- transcript search currently lives in [backend/src/main.py](/f:/Projects/PorchtimeIndex/backend/src/main.py)
- transcripts are stored in `TranscriptSegment` in [backend/src/db/database.py](/f:/Projects/PorchtimeIndex/backend/src/db/database.py)
- avatar dataset prep and lightweight clustering live in [backend/src/main.py](/f:/Projects/PorchtimeIndex/backend/src/main.py)
- the database stack is SQLModel on top of embedded Postgres by default in [backend/src/db/database.py](/f:/Projects/PorchtimeIndex/backend/src/db/database.py)

## Product Goals

For transcript search, the system should be able to:

1. find semantically similar passages even when wording differs
2. support the same metadata filters the current lexical search supports
3. combine keyword and semantic relevance where useful
4. return the surrounding transcript context needed to understand a hit

For avatar dataset prep, the system should be able to:

1. detect semantically duplicate or near-duplicate examples across episodes
2. cluster examples by idea, not just surface wording
3. retrieve recurring arguments, analogies, references, and topic frames
4. help enforce topic balance so one repeated anecdote does not dominate training

## Non-Goals For V1

- replacing the current lexical `/search` endpoint
- introducing an external vector database service
- real-time embedding generation during playback
- perfect semantic understanding for every query
- full RAG over all app data in the first pass

## Recommended Technical Direction

Use `Postgres + pgvector` inside the existing database stack.

Why this is the best fit:

- the app already uses embedded Postgres by default
- transcripts, metadata filters, and vectors can stay in one system
- `pgvector` is enough for the scale this app is likely to hit first
- it avoids operational complexity from adding `Qdrant`, `Elasticsearch`, or another service too early

Do not use the existing avatar hashing embeddings for transcript search.

Those vectors in the avatar dataset pipeline are lightweight clustering features designed for review balancing, not semantic retrieval quality.

## Retrieval Model

### V1 Retrieval Unit

Do not embed raw single diarization segments as the primary search unit.

Use `transcript chunks` built from adjacent segments:

- target chunk size: roughly `120-260` tokens
- overlap: about `20-40` tokens
- preserve source segment boundaries
- retain references to all included segment IDs

Why:

- single transcript segments are often too short
- semantic retrieval improves when the embedding sees a full thought
- arguments and analogies often span multiple diarized segments

### V1 Search Modes

Support three search modes:

1. `keyword`
   - existing lexical behavior
2. `semantic`
   - vector similarity only
3. `hybrid`
   - combine lexical and semantic ranking

`hybrid` should become the recommended default once it is stable.

## Data Model

Add a new table for transcript chunk embeddings.

Suggested model: `TranscriptChunkEmbedding`

Fields:

- `id`
- `channel_id`
- `video_id`
- `speaker_id` nullable
- `start_time`
- `end_time`
- `chunk_text`
- `chunk_token_estimate`
- `segment_ids_json`
- `embedding_model`
- `embedding_dim`
- `embedding` (pgvector `Vector(dim)` type)
- `chunk_text_tsv` (stored `tsvector` generated from `chunk_text`, for fast full-text search)
- `content_hash`
- `created_at`
- `updated_at`

Important notes:

- keep `chunk_text` in the table for debugging and search result rendering
- keep `content_hash` so unchanged chunks are not re-embedded
- store `speaker_id` only when the chunk is attributable to a single speaker, otherwise null
- `chunk_text_tsv` must be a stored column, not computed at query time, so that the GIN index is effective

## Database / Indexing

### Postgres Extension

Enable `pgvector` when Postgres is the provider.

Requirements:

- install or bundle the extension for the embedded Postgres runtime
- add startup/migration logic to ensure the extension exists

### Indexes

Add:

- HNSW vector index for similarity search: `USING hnsw (embedding vector_cosine_ops)`
  - HNSW is preferred over IVFFlat for this app: better recall, no periodic maintenance needed, set-and-forget — fits the local-first philosophy
  - IVFFlat would be cheaper to build but needs `VACUUM` and degrades without maintenance
- GIN index on `chunk_text_tsv` for fast full-text search in hybrid mode
- btree indexes on:
  - `channel_id`
  - `video_id`
  - `speaker_id`
  - `created_at` if useful for maintenance

For lexical fallback and hybrid:

- use `ts_rank()` against the stored `chunk_text_tsv` column for lexical scoring
- do not compute `to_tsvector(chunk_text)` at query time — the stored column + GIN index handles this
- optionally keep the current segment-level lexical search path for exact transcript editing workflows

## Embedding Model

Use a local embedding model with strong retrieval performance and a stable Hugging Face path.

Selection criteria:

- good semantic search quality on short-to-medium passages
- runs locally on CPU or GPU
- permissive enough for the project
- easy to batch offline during indexing jobs

Implementation rule:

- do not hard-code the first model forever
- store `embedding_model` and `embedding_dim` with every row
- make re-indexing possible when the embedding model changes

V1 default model: `BAAI/bge-small-en-v1.5`

- 384 dimensions, ~130MB download
- strong retrieval quality on short-to-medium English passages
- runs well on both CPU and GPU
- MIT license
- well-established Hugging Face path with stable weights

Why this over alternatives:

- `all-MiniLM-L6-v2` (~80MB) is lighter but slightly lower retrieval quality
- `nomic-embed-text-v1.5` (768-dim, ~550MB) is stronger but heavier and uses more storage per row
- `bge-small` hits the sweet spot of quality vs. resource cost for a local-first app

V1 model strategy:

- use `BAAI/bge-small-en-v1.5` as the default
- expose the model only in backend config, not main UI
- revisit only after retrieval quality is measured on real data

## Chunking Strategy

Create transcript chunks from `TranscriptSegment` rows using a deterministic merger.

Search and dataset prep have different needs, so use two chunking modes that share the same underlying infrastructure.

### Speaker-Pure Chunks (dataset prep + speaker-filtered search)

These are the primary chunk type. Used for avatar dataset prep, duplicate detection, clustering, and per-speaker semantic search.

Algorithm:

1. walk transcript segments in time order per video
2. **always break at speaker changes** — never merge segments from different speakers into one chunk
3. within a same-speaker run, start a new chunk at the first segment
4. append adjacent same-speaker segments until either:
   - token estimate exceeds target window
   - time gap exceeds a threshold
5. store overlapping windows for continuity within same-speaker runs only

Why speaker-pure chunking matters:

- the app is speaker-centric: avatar dataset prep, speaker filtering, and personality training all depend on speaker attribution
- mixing speakers in a chunk makes `speaker_id` null, which degrades filtered search quality
- keeping chunks speaker-pure means every chunk gets a populated `speaker_id`, enabling reliable per-speaker semantic search
- podcast dialogue has natural turn-taking boundaries that align well with thought boundaries

### Contextual Retrieval for Search

For user-facing search, a speaker-pure chunk alone may lack the conversational context needed to understand a hit (what question prompted this response?).

Rather than creating a second set of cross-speaker chunks, handle this at retrieval time:

- search always matches against speaker-pure chunks
- when returning results, fetch neighboring chunks (preceding question, following reply) from the same video by time range
- return these as `context_before` / `context_after` alongside the matched chunk

This keeps one chunk type in the database while giving search results the conversational framing users need. If retrieval-time neighbor expansion proves too slow or insufficient, a dedicated cross-speaker chunk type can be added later as an optimization.

### Shared Constraints

- avoid crossing very large time gaps
- preserve chunk-to-segment provenance
- keep chunk generation deterministic so hashes stay stable

Suggested initial split heuristics:

- max gap between merged segments: `8-12s`
- target chunk text length: `500-1200 chars`
- overlap by the last `1-2` short segments or similar token budget (within same-speaker runs only)

## Background Jobs

Add a dedicated indexing pipeline.

Suggested job stages:

1. `chunking`
2. `embedding`
3. `upsert`
4. `index complete`

Job triggers:

- when a transcript is first created
- when transcript text is edited materially
- when a transcript is deleted — delete all associated chunks
- manual `rebuild semantic index` action for a video/channel

### Deletion Cascade

Semantic index data must be cleaned up when parent entities are removed:

- video deleted → delete all `TranscriptChunkEmbedding` rows for that `video_id`
- channel deleted → cascade through all channel videos (plug into the existing channel cascade delete logic)
- transcript re-generated → delete old chunks for that video before re-indexing

V1 approach:

- run indexing asynchronously after transcription completes
- do not block transcription completion on semantic indexing

## API Plan

### New Endpoints

Add:

- `POST /search/semantic`
- `POST /channels/{id}/semantic-index/rebuild`
- `POST /videos/{id}/semantic-index/rebuild`
- `GET /semantic-index/status`

Index status response should include:

- `total_videos` — videos in scope for indexing
- `videos_indexed` — videos with completed embeddings
- `current_video` — video being processed now (null if idle)
- `chunks_pending` — estimated chunks remaining
- `is_running` — whether an indexing job is active

This parallels the existing transcription progress feedback and prevents users from thinking the app is hung during a channel-level rebuild.

Suggested semantic search request:

- `query`
- `channel_id` optional
- `video_id` optional
- `speaker_id` optional
- `year` optional
- `month` optional
- `mode` = `semantic` or `hybrid`
- `limit`
- `offset`

Suggested result payload:

- chunk hit info
- score
- source video
- speaker
- start/end time
- chunk text
- linked segment IDs
- `context_before` — preceding chunk(s) from the same video by time range (typically the question or prompt that led to this response)
- `context_after` — following chunk(s) for continuity

### Existing Search Endpoint

Do not immediately replace `/search`.

Safer path:

- keep `/search` as exact/lexical search
- add semantic search separately
- merge UI behavior later after quality is validated

## Frontend Plan

### Transcript Search UI

Add a mode selector:

- `Exact`
- `Semantic`
- `Hybrid`

Search results should clearly indicate:

- exact text match vs semantic match
- score or confidence
- matched video and timestamp
- active metadata filters

### Result Presentation

Semantic results need more context than exact search.

Show:

- snippet from chunk
- speaker name
- video title
- timestamp
- jump-to-video action
- neighboring conversational lines from `context_before` / `context_after`

### Search Workflow Updates

The search UX should change in a few concrete ways:

- keep the current exact transcript search as the default fallback
- let users switch modes without losing the current query and filters
- make it obvious when a result is semantic rather than an exact phrase hit
- show loading/progress messaging for semantic and hybrid search so slower vector queries do not feel hung

Suggested screen-level changes:

- update the existing transcript/channel search UI to include a search-mode control
- add a compact help tooltip:
  - `Exact` finds the same words
  - `Semantic` finds similar ideas
  - `Hybrid` combines both
- preserve deep-linkable query/filter state where practical

### Semantic Index Status UI

Users need visibility into whether semantic search is ready for the current scope.

Add a lightweight semantic index status panel or status row in the relevant search/admin surfaces showing:

- whether semantic indexing is ready for the selected scope
- videos indexed vs total
- whether indexing is currently running
- rebuild actions for:
  - current video
  - current channel

This matters in a local-first app because indexing can take time and users otherwise cannot tell whether the feature is unavailable or simply still processing.

### Avatar Dataset Review UI

Semantic retrieval should also surface directly inside the avatar workbench and dataset review workflow.

Add review affordances such as:

- `Show similar examples`
- `Find more like this`
- `Likely duplicates`
- semantic cluster/topic label where useful

For a selected dataset example, the review panel should be able to show:

- semantically similar approved examples
- semantically similar rejected examples
- likely duplicate candidates
- source video and timestamp for each similar example

This lets the user:

- keep the strongest example
- reject redundant paraphrases
- intentionally collect a family of related arguments or analogies

### Training Prep UI Feedback

Prepared-package feedback should eventually reflect semantic balancing decisions.

Recommended additions to the package review UI:

- semantic duplicate count removed
- semantic cluster count represented
- per-cluster cap effects
- optional summary of argument/reference-rich examples included

This keeps the semantic-search-backed dataset improvements visible rather than hidden.

## Dataset Prep Integration

This is the main training-related payoff.

### 1. Better Duplicate Detection

Current avatar dataset prep uses lightweight hashed vectors for duplicate detection and clustering.

Upgrade path:

- **augment** the current LSH-based duplicate detection with chunk embedding similarity, do not replace it immediately
- run both systems in parallel during Phase 3 and compare results on real data
- the LSH vectors and chunk embeddings operate on different units (training examples vs transcript chunks) — they may catch different kinds of duplicates
- use chunk embeddings to identify semantically similar responses across episodes
- treat very high similarity as likely duplicates
- keep the stronger or cleaner example
- note: the duplicate detection similarity threshold will need recalibration since real embedding models produce different similarity distributions than the current feature vectors (the current 0.89 cosine threshold was tuned for LSH vectors)
- only after measuring that chunk embeddings match or exceed LSH quality on the specific duplicate detection task should the LSH path be retired

Benefit:

- reduces repeated anecdotes and paraphrased reruns
- lowers memorization risk in personality LoRA training
- if chunk embeddings prove sufficient, eventually unifies the embedding infrastructure

### 2. Topic / Argument Clustering

Use semantic clustering to group examples by idea.

Examples:

- the speaker’s “liberty vs license” argument
- a recurring analogy
- a repeated historical reference

Then:

- cap how many examples are selected per semantic cluster
- ensure broad topic coverage in prepared packages

### 3. Reference / Reasoning Harvest

Let users or backend heuristics seed on an example and retrieve more semantically similar passages.

Use cases:

- “find more examples where this speaker argues about freedom”
- “find more analogies like this”
- “find more passages about this historical reference”

That is directly useful for the “substance + reasoning” direction of the personality pipeline.

### 4. Similar-Example Review Tools

Add a review affordance:

- click an example in dataset review
- fetch semantically similar examples
- approve one, reject redundant variants

This makes curation much faster and more defensible.

## Implementation Phases

### Phase 1: Semantic Index Foundation

Deliverables:

- `pgvector` enabled
- transcript chunk embedding table
- chunk builder
- offline indexer for a video
- manual rebuild endpoint

Acceptance:

- a transcripted video can be indexed end-to-end
- embeddings persist and can be queried by similarity

### Phase 2: Semantic Search API

Deliverables:

- `POST /search/semantic`
- channel/video/speaker filters
- result payload with timestamps and context

Acceptance:

- semantic search finds paraphrased ideas not captured by lexical search

### Phase 3: Dataset Prep Integration

This is one of the highest-value use cases, but it should not be treated as backend-only work. It needs review UI support in Avatar Studio to be fully useful.

Deliverables:

- semantic near-duplicate detector in avatar dataset prep (augmenting existing LSH vectors, running both in parallel)
- comparison tooling to evaluate chunk embeddings vs LSH on real duplicate detection cases
- recalibrated similarity threshold for the new embedding model
- cluster-aware dataset balancing
- avatar dataset review UI:
  - `show similar examples`
  - `find more like this`
  - `likely duplicates`
- decision gate: retire LSH path only after chunk embeddings are validated on real data

Acceptance:

- prepared datasets contain fewer semantically repetitive examples
- recurring argument structures are easier to collect intentionally
- embedding-based and LSH-based duplicate sets can be compared side by side
- users can act on semantic similarity directly from the review interface

### Phase 4: Hybrid Search UI

Deliverables:

- search mode selector
- semantic result rendering
- jump-to-video support
- semantic index status panel
- clear loading and empty-state messaging for semantic and hybrid modes

Acceptance:

- users can switch between exact and semantic search cleanly
- users can tell whether semantic search is unavailable, still indexing, or ready

### Phase 5: Advanced Tuning

Optional later work:

- hybrid reranking
- query embedding cache
- per-speaker semantic views
- semantic search over descriptions, clips, and notes

## Backend Integration Points

Likely files to touch:

- [backend/src/db/database.py](/f:/Projects/PorchtimeIndex/backend/src/db/database.py)
  - add model(s)
  - migration/index setup hooks
- [backend/src/main.py](/f:/Projects/PorchtimeIndex/backend/src/main.py)
  - semantic search endpoints
  - index management endpoints
  - dataset prep integration
- [backend/src/schemas.py](/f:/Projects/PorchtimeIndex/backend/src/schemas.py)
  - request/response models
- [backend/src/services/ingestion.py](/f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
  - trigger indexing after transcript generation completes

Suggested new module:

- `backend/src/services/semantic_search.py`

Use it for:

- chunk building
- embedding generation
- similarity query helpers
- index maintenance logic

## Frontend Integration Points

Likely files to touch:

- existing transcript/channel search pages in `frontend/src/pages/`
  - add mode selector
  - render semantic and hybrid hits
  - show semantic index status and rebuild actions where appropriate
- [frontend/src/pages/AvatarStudioPage.tsx](/f:/Projects/PorchtimeIndex/frontend/src/pages/AvatarStudioPage.tsx)
  - add semantic review tools in dataset preparation/review workflows
  - surface duplicate/similar-example actions and semantic package summaries
- [frontend/src/types.ts](/f:/Projects/PorchtimeIndex/frontend/src/types.ts)
  - semantic search request/response types
  - semantic index status types
  - dataset review similarity result types

Frontend implementation should keep the current exact-search experience intact while layering semantic and hybrid capabilities on top.

## Storage and Rebuild Strategy

Use deterministic chunk hashes so re-indexing is cheap.

For each chunk:

- compute normalized content hash from:
  - video ID
  - segment IDs
  - normalized text
- if hash unchanged, skip re-embedding
- if transcript changed, update only impacted chunks

This avoids rebuilding the entire corpus after every small edit.

## Ranking Strategy

V1 semantic ranking:

- cosine similarity on chunk embeddings

V1 hybrid ranking:

- fetch lexical candidates via `ts_rank` on `chunk_text_tsv`
- fetch semantic candidates via cosine similarity on `embedding`
- merge using Reciprocal Rank Fusion (RRF):

```
rrf_score(chunk) = sum over each list L where chunk appears:
    1 / (k + rank_in_L)
```

- default `k = 60` (standard RRF constant)
- each candidate list ranks its results independently (lexical by `ts_rank`, semantic by cosine similarity)
- chunks appearing in both lists get contributions from both ranks
- final results are sorted by `rrf_score` descending

Why RRF instead of raw score blending:

- `ts_rank` and cosine similarity are on fundamentally different scales
- normalizing `ts_rank` by max-in-set is fragile — the normalization shifts depending on what else is in the result set
- RRF only cares about rank position, not score magnitude, so it sidesteps the apples-to-oranges problem entirely
- RRF is well-established in information retrieval and simple to implement

Keep the first version simple.

Do not over-engineer learned rerankers before measuring retrieval quality on real transcripts.

## Risks

### 1. Embedding Cost

Large transcript libraries can take time to index.

Mitigation:

- background jobs
- batching
- incremental rebuilds
- video-level rebuilds before channel-level rebuilds

### 2. Over-Retrieval Of Generic Talk

Semantic search can over-match generic discussion segments.

Mitigation:

- chunking strategy
- metadata filters
- hybrid reranking
- later quality heuristics for low-information chunks

### 3. Model Drift

Changing the embedding model later can invalidate prior similarity assumptions.

Mitigation:

- store embedding model metadata
- support full rebuild

### 4. Embedded Postgres Packaging

Bundling `pgvector` with the embedded Postgres runtime may need platform-specific work.

Mitigation:

- verify extension support early
- if that blocks temporarily, use a non-persistent fallback only for prototyping, not as the final architecture

## Validation Plan

Use a small real benchmark set from your own channels.

Measure:

- queries that lexical search misses but semantic search should catch
- near-duplicate detection quality for dataset prep
- cluster diversity in prepared training packages
- indexing runtime per hour of transcripted content

Example evaluation tasks:

1. search for a concept stated with different wording
2. find repeated arguments across episodes
3. confirm duplicate anecdotes are grouped together
4. verify balanced dataset prep includes multiple semantic clusters

## Acceptance Criteria

This feature is successful when:

- semantic search retrieves paraphrased transcript passages that keyword search misses
- channel/video/speaker filtering still works cleanly
- avatar dataset prep can use semantic similarity to reduce repetition
- users can inspect semantically similar examples during review
- users can use semantic and hybrid transcript search from the interface without ambiguity about mode or index readiness
- indexing fits into the existing local-first architecture without adding an external service

## Recommended Build Order

Build in this order:

1. `pgvector` enablement
2. transcript chunk table and chunk builder (with speaker-aware chunking)
3. video-level indexing job
4. semantic search endpoint
5. dataset prep duplicate detection upgrade (augment LSH with chunk embeddings, run both in parallel, recalibrate threshold)
6. evaluate chunk embeddings vs LSH on real data — retire LSH only after validation
7. dataset review “similar examples” tooling
8. hybrid search UI with mode selector and semantic index status

Dataset prep integration remains a major value path, but the transcript search UI should be treated as a first-class deliverable rather than an afterthought.
