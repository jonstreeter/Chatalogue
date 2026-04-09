# Avatar Workbench Implementation Plan

## Purpose

This document maps out a proposed `Avatar Workbench` / `Avatar Studio` feature set for PorchtimeIndex/Chatalogue so it can be implemented later without re-planning the architecture.

The workbench is intended to:

- derive avatars from existing `Channel -> Speaker -> TranscriptSegment` data
- treat avatar creation as three separate domains:
  - personality based on conversations
  - appearance based on images
  - voice based on audio
- mine diarized transcript data into high-quality personality-training datasets
- produce a speaker-specific personality LoRA bound to a chosen base model
- support reusable voice cloning assets and portrait-driven talking-head output
- launch a local, real-time conversational avatar runtime
- fit the current FastAPI + worker + React application rather than living as a separate throwaway script

This is a planning document, not a commitment to a specific model/tool version forever. Model/runtime choices should be re-verified when implementation begins.

## Product Shape

The correct UX is a dedicated `Avatar Workbench`, not a raw file import flow.

The workbench should be organized around three independent asset sections:

- `Personality`
- `Appearance`
- `Voice`

Each section should own its own:

- source assets
- review and approval workflow
- derived artifacts
- future training/build pipeline

The final avatar runtime should compose these three domains rather than treating the avatar as one opaque model.

Source of truth:

- `Channel` defines the corpus boundary
- `Speaker` defines the canonical source persona
- transcripts, speaker samples, embeddings, and approved workbench assets already in the app become the initial training material

High-level user flow:

1. Open a channel
2. Choose a speaker
3. Click `Open in Avatar Workbench`
4. Review personality, appearance, and voice source assets
5. Configure prompts, model choices, and artifact options per section
6. Run build stages for one or more sections
7. Launch a local avatar runtime

## Goals

- Reuse existing transcript and speaker data instead of requiring users to build datasets manually
- Make the workflow resumable, inspectable, and idempotent
- Separate build-time jobs from real-time runtime processes
- Keep the initial implementation local-only
- Provide clear extension points for future models, runtimes, and asset pipelines

## Non-Goals for V1

- Multi-avatar group chat
- Fully automatic portrait generation
- Cloud deployment
- Cross-machine distributed training
- Real-time avatar video on day one before text+voice is stable
- Fully autonomous fact synthesis from transcripts without retrieval controls

## Existing Repo Fit

This should integrate into the current app architecture:

- backend API layer in `backend/src/main.py`
- long-running staged jobs in `backend/src/services/ingestion.py`
- response models in `backend/src/schemas.py`
- React frontend pages in `frontend/src/pages/`

It should follow the same broad patterns already used for:

- cleanup workbench
- reconstruction workbench
- optional dependency installation/testing
- background progress reporting
- backend-managed local runtimes

## Proposed Domain Model

Add new persistent entities:

- `Avatar`
- `AvatarRun`
- `AvatarPersonalityProfile`
- `AvatarAppearanceProfile`
- `AvatarVoiceProfile`
- `AvatarDatasetArtifact`
- `AvatarVoiceArtifact`
- `AvatarPortraitArtifact`
- `AvatarModelArtifact`
- `AvatarRuntime`
- `AvatarConversationSession`

Suggested field outline:

### Avatar

- `id`
- `channel_id`
- `speaker_id`
- `name`
- `slug`
- `status`
- `base_model_id`
- `system_prompt`
- `persona_notes`
- `portrait_image_path`
- `voice_mode`
- `runtime_mode`
- `latest_run_id`
- `created_at`
- `updated_at`

### AvatarRun

- `id`
- `avatar_id`
- `status`
- `stage`
- `config_json`
- `input_fingerprint`
- `error`
- `started_at`
- `completed_at`

### AvatarPersonalityProfile

- `id`
- `avatar_id`
- `status`
- `base_model_id`
- `system_prompt`
- `persona_notes`
- `dataset_strategy`
- `latest_dataset_artifact_id`
- `latest_model_artifact_id`
- `metadata_json`

### AvatarAppearanceProfile

- `id`
- `avatar_id`
- `status`
- `primary_image_path`
- `approved_image_count`
- `appearance_notes`
- `latest_portrait_artifact_id`
- `latest_appearance_model_artifact_id`
- `metadata_json`

### AvatarVoiceProfile

- `id`
- `avatar_id`
- `status`
- `voice_mode`
- `primary_reference_audio_path`
- `approved_audio_count`
- `latest_voice_artifact_id`
- `metadata_json`

### AvatarDatasetArtifact

- `id`
- `avatar_id`
- `run_id`
- `path`
- `format`
- `example_count`
- `token_estimate`
- `source_video_count`
- `gold_count`
- `silver_count`
- `metadata_json`

### AvatarVoiceArtifact

- `id`
- `avatar_id`
- `run_id`
- `provider`
- `reference_audio_path`
- `embedding_path`
- `preview_manifest_json`
- `metadata_json`

### AvatarModelArtifact

- `id`
- `avatar_id`
- `run_id`
- `base_model_id`
- `adapter_path`
- `merged_model_path`
- `gguf_path`
- `metadata_json`

### AvatarRuntime

- `id`
- `avatar_id`
- `status`
- `runtime_type`
- `config_path`
- `log_path`
- `pid`
- `port`
- `started_at`
- `stopped_at`

### AvatarConversationSession

- `id`
- `avatar_id`
- `runtime_id`
- `started_at`
- `ended_at`
- `transcript_path`
- `metadata_json`

## Filesystem Layout

Store avatar artifacts under:

`backend/data/avatars/{avatar_slug_or_id}/`

Suggested structure:

```text
backend/data/avatars/{avatar}/
  config/
    avatar.yaml
    personality.yaml
    appearance.yaml
    voice.yaml
    runtime.yaml
  personality/
    datasets/
      dataset_gold.jsonl
      dataset_silver.jsonl
      metadata.yaml
      examples_preview.json
    training/
      logs/
      checkpoints/
      exports/
  appearance/
    original/
    approved/
    processed/
    previews/
    training/
      logs/
      checkpoints/
      exports/
  voice/
    references/
    approved/
    embedding/
    previews/
    training/
      logs/
      checkpoints/
      exports/
  runtime/
    pipecat/
    chroma/
    sessions/
    logs/
```

## Workbench Sections

The workbench should have three primary asset sections plus cross-cutting operational sections.

### 1. Avatar Identity

- Create avatar from an existing `Speaker`
- Editable avatar display name
- Shared avatar description and notes
- Runtime profile selection

### 2. Personality

Personality is based on conversation data and should be the only section that trains the main conversational LoRA.

- Pull transcript segments for a selected speaker
- Preview candidate examples
- Filter by:
  - source videos
  - date range
  - transcript length
  - approval state
  - diarization quality proxy
  - sample confidence tier
- Export gold/silver datasets
- Edit system prompt/persona framing
- Choose base model
- Launch personality LoRA training
- Show evaluation samples, metrics, checkpoints, and resulting artifacts

### 3. Appearance

Appearance is based on images and should own all portrait/face asset curation.

- Use uploaded portraits, extracted thumbnails, and future face crops
- Review image gallery
- Approve/reject source images
- Validate aspect ratio, face presence, and image quality
- Produce processed avatar render inputs
- Generate preview renders
- Reserve future hooks for appearance LoRA or portrait-specific model training

### 4. Voice

Voice is based on audio and should stay relatively lightweight compared with personality training.

- Use existing approved speaker samples where available
- Upload optional reference audio
- Auto-build a best-reference bundle
- Clone/generate reusable voice embedding
- Produce preview utterances
- Reserve optional future hooks for deeper voice fine-tuning, while treating one-shot cloning as the default

### 5. Builds

- Show artifact status across personality, appearance, and voice
- Surface current run stage and logs
- Support retry/resume
- Show compatibility warnings between artifacts and runtime settings

### 6. Runtime

- Build runtime package/config
- Launch/stop local runtime
- Show runtime health, logs, port, and current personality/appearance/voice artifacts
- Open the live local app

## Dataset Mining Strategy

The quality of the avatar depends heavily on curation. The system should not train on raw diarized output without selection.

This section applies specifically to the `Personality` domain.

### Candidate Extraction

For a target speaker:

- gather all transcript segments assigned to that speaker
- include neighboring turns from other speakers as conversational context
- merge nearby segments from the target speaker when they are clearly part of one answer
- attach source metadata: channel, video, date, timestamps, speaker identities, length

### Dataset Example Construction

Turn the data into conversational examples where:

- the target speaker becomes the `assistant`
- surrounding speakers become `user` or contextual prior turns
- a system prompt anchors tone and identity

Preferred formats:

- ShareGPT JSONL
- ChatML JSONL

### AI-Assisted Quality Improvement

Use local AI mostly to score and select data, not to rewrite the target speaker's words.

Suggested scoring dimensions:

- speaker assignment confidence
- transcript cleanliness
- response completeness
- style richness
- factual density
- conversational usefulness
- overlap/crosstalk suspicion
- likely diarization error

Suggested quality tiers:

- `gold`: safe for primary LoRA training
- `silver`: useful for retrieval or optional mixed training
- `bronze`: retained for archive only

### Useful AI Selection Tasks

- classify fragments vs complete thoughts
- identify filler-only turns
- detect quotational/reading voice instead of the speaker's own style
- cluster semantically similar responses to reduce redundancy
- detect strong style markers such as jokes, hedging, storytelling, preferred phrases, and argument patterns
- propose conversation windows around isolated good answers

### Important Rule

Preserve the target speaker's original transcript text whenever possible. Use AI to select, rank, and structure examples. Avoid heavy rewriting of the target responses.

## Training Strategy

This section refers to personality training unless explicitly noted otherwise.

### Artifact Model

The personality artifact should be a LoRA adapter attached to an explicitly chosen base model.

The workbench should always store:

- `base_model_id`
- `base_model_revision` if applicable
- LoRA config
- training config fingerprint
- dataset fingerprint

This is required because LoRAs are not portable across unrelated base models.

### Initial Training Shape

Recommended workflow:

- choose one default base model family
- train a speaker-specific LoRA
- save adapter artifacts first
- optionally export merged runtime artifacts later

Model/tool selection should be re-validated at build time, but the architecture should assume:

- a training backend capable of low-VRAM LoRA/QLoRA
- resumable subprocess-managed training
- structured logs and checkpoint export

### Training Stages

1. validate avatar config
2. finalize dataset
3. estimate tokens and sequence lengths
4. launch trainer subprocess
5. stream logs/progress to backend
6. register resulting artifacts
7. optionally export inference formats

### Resume/Idempotency

Training should be keyed by:

- avatar
- base model
- dataset fingerprint
- training config fingerprint

If all of these match a completed artifact, the stage should reuse prior outputs unless the user requests a rebuild.

## Voice Cloning Strategy

Voice and personality should be treated as separate artifacts.

The voice section should assume one-shot or few-shot cloning as the default path. Future voice training can be added later without changing the overall workbench shape.

Pipeline goals:

- accept optional uploaded reference audio
- optionally auto-assemble reference audio from approved speaker samples
- normalize/reference-clean the audio
- create a reusable voice embedding or speaker asset
- generate preview utterances

Previews should be generated automatically after voice artifact creation so the user can reject poor references before runtime launch.

## Appearance and Talking-Head Strategy

Appearance should be treated as its own asset pipeline, sourced from approved images, and should not block personality or voice work.

Stages:

1. upload/select images
2. approve a primary portrait set
3. validate and preprocess
4. generate still preview
5. generate offline talking-head preview clip
6. only then wire into real-time runtime

This keeps early milestones focused on text+voice quality before adding more GPU-heavy video synthesis.

## Real-Time Runtime Architecture

Do not run the full live avatar stack inside the main FastAPI backend process.

Instead:

- the main app builds and manages runtime configurations
- a separate local runtime process handles live conversation
- the backend starts/stops/monitors that process

Reasons:

- easier fault isolation
- safer GPU scheduling
- cleaner lifecycle management
- avoids destabilizing transcript processing workloads

### Runtime Responsibilities

- microphone input
- STT
- LLM inference
- retrieval injection
- TTS generation
- talking-head synthesis
- transcript/session logging

### Memory / Retrieval

Use retrieval separately from the LoRA:

- LoRA captures style, phrasing, and persona habits
- retrieval provides transcript-grounded facts and prior conversations

Suggested retrieval sources:

- source speaker transcript corpus
- optional channel-wide context corpus
- prior avatar conversations
- optional manual notes/lorebook entries

## Backend API Plan

Suggested endpoint groups:

### Avatar CRUD

- `GET /avatars`
- `POST /avatars`
- `GET /avatars/{avatar_id}`
- `PATCH /avatars/{avatar_id}`
- `DELETE /avatars/{avatar_id}`

### Workbench Data

- `GET /avatars/{avatar_id}/workbench`
- `GET /avatars/{avatar_id}/progress`
- `GET /avatars/{avatar_id}/dataset/preview`
- `POST /avatars/{avatar_id}/dataset/rebuild`
- `PATCH /avatars/{avatar_id}/dataset/filters`

### Personality

- `GET /avatars/{avatar_id}/personality`
- `PATCH /avatars/{avatar_id}/personality`
- `GET /avatars/{avatar_id}/personality/dataset/preview`
- `POST /avatars/{avatar_id}/personality/dataset/rebuild`
- `PATCH /avatars/{avatar_id}/personality/dataset/filters`
- `POST /avatars/{avatar_id}/personality/train`
- `POST /avatars/{avatar_id}/personality/train/cancel`
- `POST /avatars/{avatar_id}/personality/train/retry`
- `GET /avatars/{avatar_id}/personality/training/logs`
- `GET /avatars/{avatar_id}/personality/artifacts`

### Voice

- `GET /avatars/{avatar_id}/voice`
- `PATCH /avatars/{avatar_id}/voice`
- `POST /avatars/{avatar_id}/voice/reference`
- `POST /avatars/{avatar_id}/voice/build`
- `GET /avatars/{avatar_id}/voice/previews`
- `POST /avatars/{avatar_id}/voice/test`

### Appearance

- `GET /avatars/{avatar_id}/appearance`
- `PATCH /avatars/{avatar_id}/appearance`
- `POST /avatars/{avatar_id}/appearance/upload`
- `POST /avatars/{avatar_id}/appearance/process`
- `GET /avatars/{avatar_id}/appearance/previews`
- `PATCH /avatars/{avatar_id}/appearance/approval`

### Runtime

- `POST /avatars/{avatar_id}/runtime/build`
- `POST /avatars/{avatar_id}/runtime/launch`
- `POST /avatars/{avatar_id}/runtime/stop`
- `GET /avatars/{avatar_id}/runtime/status`
- `GET /avatars/{avatar_id}/runtime/logs`

## Job Types and Background Processing

Add explicit job types instead of implementing the whole workflow as one blocking call.

Suggested job types:

- `avatar_dataset_build`
- `avatar_dataset_score`
- `avatar_appearance_prepare`
- `avatar_voice_prepare`
- `avatar_personality_train`
- `avatar_appearance_train`
- `avatar_voice_train`
- `avatar_runtime_build`
- `avatar_runtime_launch`

Why stage it this way:

- easier retries
- clearer progress reporting
- artifact reuse
- safer cancellation
- better operator visibility

## Frontend Plan

Create a dedicated avatar workbench route set.

Suggested pages:

- `AvatarStudioIndex`
- `AvatarStudioDetail`
- `AvatarOverviewPanel`
- `AvatarPersonalityPanel`
- `AvatarAppearancePanel`
- `AvatarVoicePanel`
- `AvatarBuildsPanel`
- `AvatarRuntimePanel`

Suggested user entry points:

- speaker list page: `Create Avatar`
- speaker detail page: `Open Avatar Workbench`
- channel detail page: avatar counts / quick links

### Core UI Behaviors

- staged progress badges
- artifact previews
- logs panel
- sample approval toggles
- runtime open/stop controls
- warnings when data quality is insufficient

## Configuration Plan

Use a durable config file per avatar plus backend defaults.

Suggested config layers:

- global backend defaults
- avatar-level config
- per-run override config

Suggested config topics:

- personality dataset filters
- personality training hyperparameters
- appearance pipeline settings
- voice cloning settings
- runtime settings
- retrieval settings

## Dependency / Installation Plan

Do not install the full avatar stack by default for all users.

Use optional install groups and health checks for:

- training stack
- voice stack
- talking-head stack
- runtime stack

Installer/update flow should:

- validate torch/CUDA compatibility
- install only requested optional stacks
- write version manifests
- expose backend test endpoints for each major optional component

## Suggested Milestones

### Milestone 1: Workbench Skeleton

- DB schema for avatars and runs
- DB schema for personality/appearance/voice profiles
- basic CRUD APIs
- React workbench shell
- filesystem layout
- progress model

### Milestone 2: Dataset Mining

- speaker-based transcript extraction
- conversation reconstruction
- gold/silver dataset export
- personality dataset preview UI

### Milestone 3: AI-Assisted Dataset Quality

- scoring pipeline
- clustering / redundancy reduction
- candidate approval tools
- metadata and token estimation

### Milestone 4: Appearance Artifact Pipeline

- image upload and gallery
- approval workflow
- portrait processing
- preview generation

### Milestone 5: Voice Artifact Pipeline

- reference sample selection
- upload support
- voice artifact generation
- preview generation

### Milestone 6: Personality Training

- LoRA training job
- checkpoint/log surfacing
- artifact registration
- resume/retry support

### Milestone 7: Runtime Packaging

- runtime config builder
- retrieval corpus build
- runtime launch/stop lifecycle

### Milestone 8: Appearance Training and Talking-Head Integration

- appearance model hooks
- offline preview clips
- optional real-time video hookup

### Milestone 9: Hardening

- smoke tests
- end-to-end sample fixture
- docs
- dependency validation and repair flows

## Key Engineering Decisions

### Keep Build-Time and Run-Time Separate

Training, voice preparation, and portrait processing should be backend jobs.
Live conversation should run in a separate managed runtime process.

### Favor Speaker-Curated Inputs

Use the existing speaker model and approved samples as the seed for all avatar assets. Do not lead with free-form uploads unless needed.

### Use LoRA for Style, Retrieval for Facts

Do not try to force all knowledge into the LoRA. The LoRA should carry personality/style. Retrieval should supply transcript-grounded facts and context.

### Keep Personality, Appearance, and Voice Independent

The avatar should compose three artifacts:

- personality artifact
- appearance artifact
- voice artifact

Each should be buildable and replaceable independently.

### Make Every Stage Inspectable

If the system cannot show:

- what data was selected
- why it was selected
- what model/base was used
- where outputs were written

then debugging and iteration will be much harder later.

## Risks and Watchouts

- diarization errors can poison the dataset if unfiltered
- transcript fragments can create bad conversational training rows
- GPU contention between transcription, training, TTS, and runtime is likely
- talking-head latency may make live video much harder than text+voice
- optional dependency stacks will require version pinning and compatibility checks
- poor reference audio can degrade perceived avatar quality more than the LoRA itself

## Recommended Initial Scope

Start smaller than the full vision:

1. Avatar CRUD
2. Personality workbench from existing speaker transcripts
3. Appearance image gallery and approval flow
4. Voice sample selection and preview support
5. Training job scaffold with placeholder runner
6. Runtime package scaffold without live video

That sequence builds the foundation without committing immediately to the hardest real-time video components.

## Definition of Success

The initial implementation is successful when:

- a user can create an avatar from an existing speaker
- the system can mine and preview a usable personality dataset
- the system can curate appearance images
- the system can prepare voice inputs and previews
- the system can launch a personality training job and persist artifacts
- the system can start a local avatar runtime with clear status and logs
- every stage can be retried without manual cleanup

## Open Questions to Resolve Before Implementation

- Which exact base model family should be the initial default?
- Which training/export formats are actually required for the first runtime?
- Which runtime stack should ship first: text+voice only, or text+voice+video?
- How much of the transcript corpus should be allowed into retrieval by default?
- What operator review UI is required before training on mined data?
- How aggressively should the system auto-reuse existing artifacts vs force rebuilds?
