# Episode AI Chat Implementation Plan

## Goal

Add an episode-scoped AI chat workbench that lets a user ask questions about a single episode, uses the episode transcript as grounded context, and saves conversation threads for later review.

This is not a general chatbot. It is a transcript-grounded analysis tool.

## Multi-Specialty Review Summary

Reviewed against the current architecture using:

- UI/workbench patterns in `frontend/src/pages/video/VideoDetailPage.tsx`
- saved test-chat precedent in `frontend/src/pages/AvatarStudioPage.tsx`
- existing LLM provider routing in `backend/src/main.py`
- transcript chunk retrieval in `backend/src/services/semantic_search.py`
- provenance-heavy result patterns used by clone and transcript optimization flows

Consensus:

- The feature fits the product well.
- It should be a dedicated main-stage workbench tab, not a sidebar widget.
- It must be transcript-grounded and citation-first.
- It should save threads and assistant-turn retrieval context.
- V1 should stay episode-only and avoid channel-wide assistant scope.

## Product Positioning

### What it is

- A saved, episode-specific AI chat surface
- Backed by the episode transcript and episode metadata
- Designed for summary, fact lookup, claim tracing, quote finding, and speaker-specific questions

### What it is not

- A general-purpose assistant
- A web-search tool
- A channel-wide research agent
- A replacement for transcript search
- An avatar/personality chat mode

## Recommended V1 Feature Set

### Workbench

- New `Chat` icon tab in the episode sidebar
- Selecting `Chat` opens a main-area workbench
- Layout:
  - left rail: thread shelf
  - main stage: message timeline + composer
  - citation panel can be inline for v1

### Threading

- Create new thread per episode
- Rename thread
- Delete or archive thread
- Persist message history
- Restore the last-open thread when returning to the episode

### LLM Controls

- Default to the configured app LLM provider/model
- Allow per-thread or per-message provider/model override
- Show provider/model used on each assistant response

### Grounded QA

- Use transcript chunks from the current episode as retrieval context
- Include episode title and metadata in prompt context
- Every assistant answer should return:
  - answer text
  - cited transcript chunks
  - timestamps
  - segment jump targets

### User Actions

- Ask free-form questions
- Click citations to jump to the transcript
- Copy answer
- Start a fresh thread
- Resume prior thread

### Suggested Prompt Chips

- Summarize this episode
- What are the main themes?
- Find where they discuss X
- What did speaker Y say about X?
- List the strongest claims and supporting evidence

## V1 Constraints

These are important. They prevent the feature from turning into a vague low-trust chatbot.

- Episode-only retrieval
- No channel-wide semantic context in v1
- No web search
- No agentic tool use
- No voice chat
- No transcript editing by chat
- No autonomous follow-up chains
- No multimodal image/video reasoning

## UX Recommendation

### Why this should be a workbench

The episode page already uses the main stage for heavier tools like clone, cleanup, and optimization. Chat belongs in that same category.

A sidebar-only panel would fail because:

- thread history needs space
- citations need space
- model metadata needs to be visible
- multi-turn chat does not fit the transcript sidebar well

### Proposed Workbench Layout

#### Left rail

- `New Thread`
- thread list
- last updated
- message count
- active thread indicator

#### Main area

- episode title + scope badge
- provider/model badge
- chat timeline
- inline citation cards on assistant messages
- composer fixed at bottom

### Transcript integration

- Citation click should seek player and jump to the cited segment
- Optional future split view: transcript on one side, chat on the other

## Retrieval Architecture

### Existing assets to reuse

- `TranscriptChunkEmbedding`
- `semantic_search(...)` / `hybrid_search(...)`
- transcript segment jump behavior in episode detail

### Retrieval strategy for v1

Per assistant turn:

1. Build a retrieval query from:
   - latest user message
   - compact summary of prior thread turns
   - optional title/topic bias
2. Retrieve top transcript chunks from the current episode only
3. Deduplicate overlapping chunk hits
4. Add limited neighboring context for selected hits
5. Build grounded prompt with:
   - system instruction
   - concise thread summary
   - recent message history
   - retrieved transcript evidence
6. Require the answer to cite evidence

### Why not full transcript every turn

- poor latency on long episodes
- worse cost
- harder to keep answer grounded
- hard to persist and explain answer provenance

## Safety and Trust Guardrails

### Grounding rules

- Answers must be constrained to transcript evidence
- If evidence is weak, assistant should say so
- Citation payload should be first-class, not optional

### Prompt rules

- Transcript is source of truth
- Do not invent facts not supported by cited transcript evidence
- Distinguish direct evidence from inference
- Identify uncertainty explicitly

### User-facing affordances

- Scope badge: `Episode Transcript`
- Citation chips with timestamps
- Optional note when answer includes inference beyond direct quote

## Backend Domain Model

### New tables

#### `EpisodeChatThread`

- `id`
- `video_id`
- `channel_id`
- `title`
- `status` (`active`, `archived`)
- `provider`
- `model`
- `system_prompt` or `mode`
- `created_at`
- `updated_at`

#### `EpisodeChatMessage`

- `id`
- `thread_id`
- `role` (`user`, `assistant`)
- `content`
- `status` (`completed`, `running`, `failed`, `cancelled`)
- `parent_message_id` optional
- `error`
- `created_at`
- `completed_at`

#### `EpisodeChatMessageContext`

- `id`
- `message_id`
- `provider`
- `model`
- `temperature`
- `top_p`
- `semantic_query`
- `thread_summary`
- `retrieved_chunk_ids_json`
- `retrieved_segment_ids_json`
- `citations_json`
- `token_estimate`
- `latency_ms`
- `created_at`

## API Plan

### Thread endpoints

- `POST /videos/{video_id}/episode-chat/threads`
- `GET /videos/{video_id}/episode-chat/threads`
- `GET /episode-chat/threads/{thread_id}`
- `PATCH /episode-chat/threads/{thread_id}`
- `DELETE /episode-chat/threads/{thread_id}`

### Message endpoints

- `GET /episode-chat/threads/{thread_id}/messages`
- `POST /episode-chat/threads/{thread_id}/messages`
- `GET /episode-chat/messages/{message_id}`
- `POST /episode-chat/messages/{message_id}/retry`

### Optional async execution

- Use `Job` only as an execution wrapper for assistant reply generation
- Suggested `job_type`: `episode_chat_reply`

## Execution Model

### Recommended behavior

- User message persists immediately
- Assistant placeholder message is created with `status=running`
- Fast turns can complete inline
- Slow turns can run as a job and be polled

### Why hybrid sync/async is best

- avoids losing conversation state
- fits current job queue patterns
- works for both local and hosted LLMs
- supports future “deep answer” mode

## Prompt and Context Strategy

### Prompt inputs

- system instruction for grounded episode chat
- current thread summary
- a small number of recent turns
- retrieved transcript evidence
- episode metadata

### Response contract

- `answer`
- `citations`
- `confidence` or `grounding note`
- optional `follow_up_suggestions`

### Recommended citation structure

- `chunk_id`
- `segment_ids`
- `start_time`
- `end_time`
- `quoted_text` or `support_text`

## Performance Notes

### Risks

- repeated retrieval over long threads
- prompt growth
- provider cost on verbose evidence injection
- slow local models

### Mitigations

- rolling thread summary
- hard cap on recent turns included in prompt
- hard cap on retrieved chunks
- dedupe overlapping chunks
- episode-only retrieval in v1
- persist retrieval provenance per assistant turn

## Implementation Phases

### Phase 1: Backend foundation

- add DB models
- add schemas
- add thread/message APIs
- add retrieval builder for episode-only transcript grounding
- add sync assistant reply path

### Phase 2: Workbench UI

- add `Chat` tab to episode view
- add thread shelf
- add message timeline
- add composer
- add citation jump behavior

### Phase 3: Async hardening

- add `episode_chat_reply` jobs
- polling for running assistant messages
- retry failed assistant messages
- persist latency and retrieval provenance

### Phase 4: Trust and quality polish

- inference-vs-evidence labeling
- answer formatting improvements
- suggested prompts
- thread title generation

## Future Expansion

These should wait until v1 is stable:

- channel-context mode
- clip creation from chat answers
- export thread transcript
- collaborative/shared threads
- voice input/output
- web-connected research mode

## Recommended First Build Order

1. Episode-only persisted threads/messages
2. Citation-first retrieval and answer schema
3. Main-stage `Chat` workbench in episode view
4. Sync fast-path for assistant replies
5. Async job wrapper for slower replies

## Success Criteria

V1 is successful if:

- users can ask a question about a single episode
- answers come back grounded in transcript evidence
- citations jump directly to the relevant transcript moments
- threads persist and are easy to resume
- provider/model used is visible
- the feature does not degrade into a generic uncited chatbot
