# Episode AI Chat Related-Scope Implementation Plan

## Goal

Add an explicit cross-episode retrieval mode to episode chat so a user can ask questions about the current episode while allowing the assistant to draw carefully bounded context from other episodes in the same channel.

This is not a silent expansion of chat memory. It is a scoped retrieval mode with visible provenance and guardrails.

## Product Positioning

### What this feature is

- An extension of episode chat
- Designed for recurring themes, callbacks, repeated claims, and creator-pattern analysis
- Centered on the current episode, with optional support from related episodes in the same channel

### What this feature is not

- A hidden channel-wide memory system
- A default mode for every chat turn
- A replacement for transcript search
- A web-search or general-research agent
- A mode that should override what the current episode actually says

## Core Product Principle

The current episode remains primary.

When related-episode context is enabled:

- answer the user's question about the current episode first
- use related episodes only to expand, compare, or contextualize
- never let related episodes overwrite or dilute a current-episode answer

## Recommended Scope Modes

### Mode 1: `Episode`

- Current default
- Uses only the current episode transcript and metadata
- Best for quote finding, fact lookup, passage explanation, and speaker-specific questions

### Mode 2: `Episode + Related`

- Uses the current episode plus a tightly capped set of semantically related chunks from other videos in the same channel
- Best for:
  - recurring themes
  - repeated claims
  - "has this channel discussed this before?"
  - "how does this episode fit the broader pattern?"

### Mode 3: `Channel`

- Future mode, not part of the first implementation slice
- Intended for broad creator-pattern research
- Much higher contamination risk
- Should not ship before `Episode + Related` proves trustworthy

## Recommended V1.5 Feature Set

### Workbench Controls

- Add a visible scope selector to the chat composer area:
  - `Episode`
  - `Episode + Related`
- Persist scope per thread
- Allow optional per-message override later, but not in the first rollout

### Reply Metadata

Each assistant reply should show:

- scope used
- provider/model
- prompt version
- retrieval mode

### Citation Groups

Assistant replies should distinguish:

- `Episode citations`
- `Related episode citations`

For related citations, show:

- source video title
- source timestamp
- source speaker when available

### User-Facing Behaviors

- User can ask:
  - "what are the major arguments in this episode?"
  - "does this channel return to this idea in other episodes?"
  - "how does this compare to related episodes?"
- User can click any citation to jump to the source episode/time
- Related-episode citations from other videos should open that episode in the correct context

## UX Recommendation

### Scope Selector

Place it in the episode chat workbench near the provider/model controls.

Recommended labels:

- `Episode Only`
- `Episode + Related`

### Reply Badges

Each assistant message should show a compact badge, for example:

- `Episode`
- `Episode + Related`

### Provenance Copy

When related material was used, include a small note such as:

- `This answer includes related context from other channel episodes.`

### Citation Layout

Inline citation blocks should be split into sections:

- `Current episode`
- `Related episodes`

This keeps the answer explainable.

## Retrieval Architecture

## Query Inputs

Build retrieval from:

- latest user message
- compact thread summary
- episode title
- episode-wide context map

## Current Episode Retrieval

Keep the current episode retrieval path:

- semantic query
- hybrid search
- episode-wide context map for synthesis questions
- top local evidence excerpts

## Related Episode Retrieval

For `Episode + Related`:

1. Run semantic retrieval across the same channel
2. Exclude the current episode
3. Deduplicate by video and chunk
4. Cap results aggressively
5. Prefer diversity across videos over many chunks from one video

Recommended initial caps:

- maximum related videos: `3`
- maximum related chunks total: `6`
- maximum chunks per related video: `2`

## Ranking Strategy

Use a hybrid ranking that favors:

- semantic similarity to the user question
- topical overlap with the current episode context map
- recency only as a weak tiebreaker

Do not rank by popularity for this feature.

Popularity is useful for clone candidate selection, not episode chat grounding.

## Prompt Strategy

Prompt should contain:

- system instruction
- thread scope mode
- episode title/description
- current-episode context map
- current-episode evidence excerpts
- related-episode evidence excerpts in a separate section
- explicit rules on source hierarchy

### Prompt Rules

- Current episode is the source of truth for episode-specific claims
- Related episodes may only provide supporting context, comparison, or recurring-theme evidence
- If related episodes conflict with the current episode, prefer the current episode
- Never imply related-episode content was said in the current episode
- When using related episodes, mention that clearly in the answer

## Response Contract

Extend assistant context payload to include:

- `scope_mode`
- `episode_citations`
- `related_citations`
- `related_video_ids`
- `related_video_titles`
- `used_related_context` boolean

Recommended response shape:

- `answer`
- `citation_indexes`
- `related_citation_indexes`
- `grounding_note`
- optional `scope_note`

## Backend Data Model Changes

### `EpisodeChatThread`

Add:

- `scope_mode`
  - `episode`
  - `episode_related`

### `EpisodeChatMessageContext`

Add:

- `scope_mode`
- `episode_citations_json`
- `related_citations_json`
- `related_video_ids_json`
- `used_related_context`
- `retrieval_mode`

Keep existing context fields for backward compatibility during migration.

## API Changes

### Thread APIs

Update:

- `POST /videos/{video_id}/episode-chat/threads`
- `PATCH /episode-chat/threads/{thread_id}`

Allow:

- `scope_mode`

### Message API

Update:

- `POST /episode-chat/threads/{thread_id}/messages`

Behavior:

- if thread scope is `episode`, use current flow
- if thread scope is `episode_related`, run dual retrieval

### Channel Chats View

Optional enhancement:

- show scope badge for the latest active thread on each episode chat row

## Frontend Changes

### Episode Chat Workbench

Add:

- scope selector
- scope badge in thread shelf
- reply badge in assistant messages
- split citation sections

### Citation Jump Behavior

For current-episode citations:

- keep current player + transcript jump behavior

For related-episode citations:

- open the cited episode directly with:
  - `tab=chat`
  - time or segment deep link

### Thread Shelf

Show:

- title
- last activity
- scope mode

## Guardrails

### Hard Requirements

- Scope must always be visible
- Related-episode usage must be disclosed in the answer
- Related citations must be visually separated from current-episode citations
- Current-episode evidence must always be present when answering an episode-specific question

### Retrieval Limits

- do not allow unlimited channel retrieval
- cap related context tightly
- diversify across episodes
- dedupe near-duplicate chunks

### Failure Mode

If related retrieval fails or semantic indexing is unavailable:

- fall back cleanly to `Episode`
- show a small note:
  - `Related-episode context was unavailable for this reply.`

## Evaluation Plan

### Success Cases

- The answer to a recurring-theme question is more complete than `Episode` alone
- The answer still distinguishes what was said in the current episode versus other episodes
- Citations are understandable and correctly grouped

### Failure Cases to Test

- related episodes overpower the current episode
- answer implies a related quote happened in the current episode
- too many citations from one video
- irrelevant related episodes pollute the answer
- latency becomes unacceptable

### Quality Metrics

- related-context usage rate
- average number of unique related videos cited
- answer latency delta versus episode-only mode
- manual trust review score
- contamination rate from evaluator review

## Rollout Plan

### Phase 1

- add `scope_mode` to thread model and APIs
- implement `Episode + Related` retrieval backend
- store related citation provenance
- show scope badges in UI

### Phase 2

- split citations visually into current vs related
- add related-episode jump behavior
- add fallback note when related retrieval is unavailable

### Phase 3

- add response evaluator for contamination checks
- add suggested prompt chips for related-mode questions
- add thread search/filter by scope

## Out of Scope

- full `Channel` mode
- web search
- cross-channel context
- autonomous multi-hop research
- silent memory blending

## Recommendation

Implement `Episode + Related` as a clearly labeled retrieval scope, not as invisible extra context.

This provides the real benefit of broader channel intelligence while preserving trust, source clarity, and transcript-grounded behavior.
