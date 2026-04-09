# Major Feature Optimization Review

Date: 2026-04-05

Scope:
- major app sections across channels, navigation, transcript workflow, clips, cleanup, reconstruction, AI features, avatars, jobs, and settings
- optimization recommendations intended for future implementation, not immediate bugfix-only triage

Context:
- the recent top-priority episode-clone fixes are already implemented:
  - `video.view_count` widened to `BIGINT`
  - clone generation moved to a queued background job
  - clone panel stale-response handling was tightened
- this document lists the remaining high-yield optimization work after those changes

Review method:
- multi-specialty review panel covering:
  - frontend/navigation and information architecture
  - transcript/speaker/clip workflow
  - AI/retrieval/search/cloning/avatar product quality
  - backend reliability/queueing/operations
  - media cleanup and reconstruction pipeline quality

## Priority Summary

### P0: Structural Reliability and Trust

1. Make queue claiming atomic and durable.
- current queue claiming is still `select` then `update`, which leaves double-claim risk under concurrent workers and blocks safe scaling
- add DB-backed claim semantics such as `FOR UPDATE SKIP LOCKED` or a compare-and-swap lease model
- also move duplicate-job prevention from app logic to DB-enforced uniqueness for active jobs
- references:
  - `backend/src/services/ingestion.py`
  - `backend/src/main.py`
  - `backend/src/db/database.py`

2. Fix reconstruction auxiliary job state handling.
- the reconstruction branch in auxiliary job state management is misplaced and can produce bad pause/cancel/clear behavior
- this is not just an optimization; it undermines trust in reconstruction lifecycle state
- references:
  - `backend/src/main.py`

3. Replace startup/runtime migration behavior with versioned migrations.
- schema mutation on startup increases boot fragility and complicates queue integrity changes
- move DDL changes to explicit versioned migrations before deeper queue work lands
- references:
  - `backend/src/db/database.py`

4. Surface real worker health and stuck-job telemetry.
- `/system/worker-status` is too shallow for operational confidence
- add per-queue lag, active-job age, last success, last failure, retry counts, and stuck-job detection
- references:
  - `backend/src/services/ingestion.py`
  - `backend/src/main.py`

### P1: Retrieval, AI Quality, and Grounding

5. Replace brute-force semantic retrieval with scalable indexing and reranking.
- current semantic and hybrid search load large candidate sets into Python memory and rank there
- this will degrade both latency and retrieval quality as corpus size grows
- move toward ANN or `pgvector`-style prefiltering, stronger lexical retrieval, and reranking
- references:
  - `backend/src/services/semantic_search.py`

6. Redesign chunking for meaning, not only attribution.
- speaker-pure chunking is good for speaker identity but harms semantic coherence for multi-speaker exchanges
- introduce a second retrieval-oriented chunk type or windowed semantic chunking that preserves conversational context
- references:
  - `backend/src/services/semantic_search.py`

7. Add real originality enforcement to episode cloning.
- cloning still relies mostly on prompt instructions instead of novelty checks
- add overlap scoring against source/context text, outline-first generation, regeneration on high overlap, and diversified retrieval before generation
- references:
  - `backend/src/services/episode_clone.py`

8. Ground avatar/personality inference with retrieval.
- current avatar chat/fit-check behavior is style-heavy but not fact-grounded
- add transcript retrieval injection, episode-aware memory, and source attribution for grounded responses
- references:
  - `backend/src/main.py`

9. Decouple YouTube metadata generation from local Ollama-only routing.
- metadata generation should use the same provider abstraction and model policy as other LLM features
- this removes local GPU pressure as a hidden product bottleneck
- references:
  - `backend/src/services/ingestion.py`

### P1: Workflow Throughput and UX

10. Preserve search context when handing off into the video workspace.
- channel search currently restores time position but not the matched segment/query/selection
- users lose context before clipping, transcript correction, or speaker cleanup
- pass query + result identity into the video route and restore focus/selection on load
- references:
  - `frontend/src/pages/channel/ChannelSearch.tsx`
  - `frontend/src/pages/video/VideoDetailPage.tsx`

11. Stop reloading entire video state for speaker cleanup operations.
- unknown-speaker assignment and merge flows still trigger broad refetches and modal detours
- move to local optimistic updates and scoped refreshes so long cleanup sessions keep position and momentum
- references:
  - `frontend/src/pages/video/VideoDetailPage.tsx`
  - `frontend/src/components/SpeakerList.tsx`
  - `frontend/src/components/SpeakerModal.tsx`

12. Speed up transcript correction for QA-heavy use.
- current transcript correction is accurate but click-heavy
- add plain-text edit mode, save-and-next, keyboard-driven review, and multi-segment cleanup actions
- references:
  - `frontend/src/pages/video/VideoDetailPage.tsx`

13. Unify clip creation and batch export around a faster path.
- clip creation/editing is fragmented and batch export is serialized on the client
- move batch render/export orchestration fully to queued backend jobs and improve transcript-to-clip precision with word-level bounds
- references:
  - `frontend/src/pages/video/VideoDetailPage.tsx`

14. Separate live queue management from queue history.
- `JobQueue` currently mixes active operations, queue controls, and large history views into one polling-heavy page
- split live queue from history and virtualize large lists
- references:
  - `frontend/src/pages/JobQueue.tsx`

### P1: Navigation and Information Architecture

15. Make the shell route-aware for major work areas.
- top-level navigation loses section context on detail routes like `/channel/:id`, `/video/:id`, `/speakers/:id`, and `/avatars/:id`
- add nested layouts, section-aware highlighting, and breadcrumbs
- references:
  - `frontend/src/components/Layout.tsx`
  - `frontend/src/App.tsx`
  - `frontend/src/pages/ChannelDetail.tsx`

16. Split `Settings` into nested routes instead of a single giant stateful page.
- current tab-local-state design hurts deep-linking, back/forward behavior, onboarding, and maintainability
- move to route-backed sections such as `/settings/transcription`, `/settings/llm`, `/settings/youtube`, `/settings/system`
- references:
  - `frontend/src/App.tsx`
  - `frontend/src/pages/Settings.tsx`

17. Add hierarchy and discoverability to channels and channel admin actions.
- channel admin actions currently compete visually with navigation, and channel browsing will get harder as the dataset grows
- add search/sort/filter on the channels page and push destructive/admin actions into a lower-emphasis management area
- references:
  - `frontend/src/pages/Channels.tsx`
  - `frontend/src/pages/ChannelDetail.tsx`
  - `frontend/src/pages/channel/ChannelVideos.tsx`

18. Standardize speaker interaction patterns across the app.
- speaker rows/cards behave differently depending on context and view
- route-back key speaker surfaces so state is shareable and predictable
- references:
  - `frontend/src/pages/Speakers.tsx`
  - `frontend/src/components/SpeakerList.tsx`

## Major Feature Sections

### Channels, Navigation, and Settings

High-yield improvements:
- route-backed settings sections
- route-aware shell navigation and breadcrumbs
- searchable/sortable channel index
- clearer separation between browse actions and destructive/admin actions
- route-backed speaker and avatar detail transitions

Why this matters:
- the app has become multi-workbench software, but the shell still behaves like a small single-surface tool
- improving route structure will raise usability and reduce local-state complexity across multiple pages

### Transcript, Search, Speakers, and Clips

High-yield improvements:
- preserve matched transcript context across route transitions
- avoid broad `fetchData()` refreshes during cleanup work
- add keyboard- and batch-oriented transcript QA tools
- use word-level timestamps for better initial clip bounds
- move batch clip exports fully into backend queue orchestration

Why this matters:
- these are high-frequency flows, so even small state and interaction friction compounds into large throughput losses

### Cleanup and Reconstruction

High-yield improvements:
- fix reconstruction auxiliary state correctness first
- invalidate stale cleaned/reconstructed artifacts whenever inputs change
- enforce stronger readiness rules before full reconstruction can queue
- separate speaker test clips from segment preview artifacts
- surface fallback ratios and degraded-output warnings to users

Why this matters:
- media enhancement features need stronger trust guarantees than ordinary analysis features because users listen to the outputs directly

### AI Search, Clone, Metadata, and Avatar Intelligence

High-yield improvements:
- scalable retrieval and reranking
- context-preserving chunk strategies
- originality/overlap gating in cloning
- provider-agnostic metadata generation
- retrieval-grounded avatar inference

Why this matters:
- the limiting factor here is system design quality, not just model quality
- better retrieval architecture improves multiple product areas at once

### Jobs, Queueing, and Operations

High-yield improvements:
- atomic queue claims
- DB-enforced active-job uniqueness
- durable worker lifecycle management on restart/update
- per-queue health metrics and stuck-job detection
- job-type-specific recovery policy instead of blanket requeue

Why this matters:
- the app already behaves like a local production system; queue semantics now need to match that reality

## Suggested Implementation Waves

### Wave 1
- atomic queue claim + duplicate-job DB invariants
- reconstruction auxiliary state fix
- worker health and stuck-job telemetry
- explicit stale-artifact invalidation for cleanup/reconstruction

### Wave 2
- scalable semantic retrieval architecture
- improved retrieval-oriented chunking
- clone originality gate and diversified retrieval
- YouTube metadata provider abstraction

### Wave 3
- route-backed settings and shell navigation refactor
- search-to-video context handoff
- local transcript/speaker optimistic updates
- queue-backed batch clip/export workflow

### Wave 4
- retrieval-grounded avatars
- channel index search/sort/filter improvements
- speaker interaction model standardization
- richer transcript QA tooling

## Notes

- This document mixes optimization work with a few correctness items because those correctness gaps directly affect trust in the corresponding feature sections.
- The highest-leverage shared investment is retrieval and queue infrastructure. Improving those foundations unlocks better behavior across search, cloning, metadata, avatars, and long-running media tasks.
