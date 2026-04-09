# AI Episode Localization & Dubbing Implementation Plan

## Purpose

This document defines a practical implementation plan for adding an `AI Episode Localization & Dubbing` workbench to Chatalogue.

The goal is to let a user take an existing episode, translate it into a target language, review the translation, assign target voices per speaker, generate dubbed dialogue, align it back onto the original timeline, and export alternate-language assets.

This should be built as a first-class workbench inside the current app, not as a loose script pipeline.

## Product Shape

The correct UX is a dedicated localization/dubbing workbench with staged checkpoints, not a one-click black box.

The workbench should be organized around four sections:

- `Source`
- `Translate`
- `Dub`
- `Mix & Export`

Each section should own:

- source artifacts
- editable intermediate outputs
- long-running jobs
- regeneration controls
- approval state

High-level user flow:

1. Open an episode
2. Launch `Localization & Dubbing`
3. Choose target language and project settings
4. Review source transcript blocks and speaker coverage
5. Generate and edit translated blocks
6. Assign voices or voice clones per speaker
7. Synthesize dubbed dialogue
8. Align and mix dubbed speech with preserved background audio
9. Export translated transcript, subtitles, dubbed audio, and optional remuxed video

## Goals

- Reuse the existing transcript, diarization, speaker, and job infrastructure
- Support user review before irreversible downstream generation
- Preserve speaker structure where diarization confidence is good
- Keep all major outputs editable and regenerable
- Separate translation quality from dubbing quality from mixing quality
- Make the system resumable, inspectable, and idempotent

## Non-Goals For V1

- Real-time speech-to-speech dubbing
- Fully automatic perfect lip-sync
- Production-grade cinematic source separation
- Automatic multilingual publishing to external platforms
- Cross-video voice identity learning beyond user-approved speaker samples
- Fully autonomous translation with no review path

## Why This Fits Chatalogue

The repo already has the core substrate needed for this feature:

- diarized transcripts
- canonical `Speaker` entities
- background job and progress infrastructure
- workbench UI patterns
- voice/reconstruction groundwork
- transcript optimization and review workflows

This feature should build on those primitives instead of introducing a separate parallel stack.

## Recommended System Architecture

Use a staged pipeline with editable checkpoints.

Core principle:

- the canonical transcript remains the source of truth
- localization creates derived, versioned artifacts
- each stage should be restartable without regenerating every later stage

Recommended pipeline:

1. `prepare`
   - derive localized translation blocks from transcript turns
   - validate speaker coverage
   - collect glossary/pronunciation rules
2. `translate`
   - generate target-language translation blocks
   - apply terminology and do-not-translate rules
3. `refine`
   - run a second-pass localization rewrite for natural spoken delivery
   - allow manual edits and approval
4. `voice_assign`
   - map each source speaker to a target voice strategy
5. `synthesize`
   - generate dubbed speech block by block
6. `align`
   - fit synthesized speech back onto the source timeline
7. `mix`
   - preserve background audio and replace or overlay dialogue
8. `export`
   - write translated transcript, subtitle assets, dubbed audio, and optional remuxed video

## Key Engineering Decisions

### 1. Translation Units Should Be Block-Based, Not Raw Segment-Based

Do not translate raw diarization rows one by one.

Instead, derive `localization blocks` from canonical transcript turns:

- usually one speaker turn per block
- optionally merge short adjacent turns for subtitle/translation coherence
- preserve references to original segment IDs

Why:

- translation quality improves when the model sees a complete thought
- timing can still be mapped back to source segments
- review is easier at a thought/block level than at a row level

### 2. Translation Review Must Be Explicit

Users should be able to edit translated blocks before dubbing begins.

This is critical because:

- untranslated names and terms need control
- literal translations are often unsuitable for spoken dubbing
- the cost of catching problems after synthesis is higher

### 3. Voice Assignment Must Be Per Speaker, With Fallback Modes

Do not assume high-confidence multi-speaker dubbing is always possible.

Support three voice modes:

- `clone_source_speaker`
- `mapped_synthetic_voice`
- `single_narrator_fallback`

If speaker quality is weak, degrade to narrator mode rather than faking precise multi-speaker dubbing.

### 4. Timing Alignment Must Be Its Own Stage

Do not treat TTS output as final once synthesized.

Speech alignment needs a separate stage because:

- translated speech often runs longer or shorter than source
- some blocks need retiming or alternate phrasings
- subtitle timing and dubbed timing should be inspectable independently

### 5. Background Preservation Should Be First-Class

The preferred output is not a fully synthetic audio replacement.

The preferred output is:

- preserved music/effects/room tone
- replaced or ducked dialogue
- exported dubbed master as a separate artifact

## Domain Model

Add new persistent entities.

### LocalizationProject

- `id`
- `video_id`
- `channel_id`
- `source_language`
- `target_language`
- `status`
- `title`
- `note`
- `translation_model`
- `tts_provider`
- `created_at`
- `updated_at`

### LocalizationBlock

- `id`
- `project_id`
- `speaker_id` nullable
- `block_index`
- `source_start_time`
- `source_end_time`
- `source_text`
- `translated_text`
- `refined_text`
- `approved_text`
- `segment_ids_json`
- `status`
- `timing_fit_status`
- `timing_fit_note`

### LocalizationGlossaryEntry

- `id`
- `project_id` nullable
- `channel_id` nullable
- `source_term`
- `target_term`
- `language_pair`
- `mode`
  - `preferred_translation`
  - `do_not_translate`
  - `pronunciation_hint`
- `notes`

### LocalizationSpeakerVoiceAssignment

- `id`
- `project_id`
- `speaker_id` nullable
- `mode`
- `voice_profile_id` nullable
- `avatar_voice_profile_id` nullable
- `tts_provider`
- `tts_model`
- `voice_ref`
- `approved`

### LocalizationSynthesisArtifact

- `id`
- `project_id`
- `block_id`
- `speaker_id` nullable
- `audio_path`
- `provider`
- `model`
- `voice_ref`
- `duration_seconds`
- `status`
- `error`

### LocalizationMixArtifact

- `id`
- `project_id`
- `kind`
  - `dubbed_dialogue_stem`
  - `background_stem`
  - `dubbed_master`
  - `muxed_video`
  - `srt`
  - `vtt`
  - `translated_transcript`
- `path`
- `status`
- `metadata_json`

### LocalizationRun

- `id`
- `project_id`
- `stage`
- `status`
- `provider`
- `model`
- `settings_json`
- `started_at`
- `completed_at`
- `error`

## Filesystem Layout

Suggested layout under `backend/data/localization/`:

- `project_<id>/source/`
- `project_<id>/translation/`
- `project_<id>/tts/`
- `project_<id>/aligned/`
- `project_<id>/mix/`
- `project_<id>/exports/`

Keep stage outputs separated so rollback and regeneration are obvious.

## Workbench Sections

## Source

Show:

- source episode metadata
- source language
- transcript readiness and quality
- diarization quality summary
- detected speakers
- target language selection

Actions:

- create project
- regenerate block segmentation
- downgrade to narrator mode when speaker quality is weak

## Translate

Show:

- block list with source text and translated text
- glossary and do-not-translate rules
- terminology warnings
- block approval state

Actions:

- translate all
- translate selected
- edit block
- approve block
- bulk apply glossary term
- mark project translation-ready

## Dub

Show:

- speaker list
- voice assignment per speaker
- source sample availability
- synthesis status per block
- duration mismatch indicators

Actions:

- assign voice mode
- test voice sample
- synthesize selected blocks
- synthesize all approved blocks
- regenerate failed blocks

## Mix & Export

Show:

- dubbed block coverage
- timeline fit summary
- background preservation settings
- export options

Actions:

- align blocks
- mix dubbed master
- export SRT/VTT
- export translated transcript
- export dubbed audio
- export muxed video

## Translation Strategy

Translation should be two-pass.

### Pass 1: Literal/Structured Translation

Purpose:

- preserve meaning
- preserve speaker/block mapping
- obey glossary rules

Output:

- `translated_text`

### Pass 2: Spoken Localization Refinement

Purpose:

- improve naturalness for speech
- preserve intent, not literal phrasing
- make blocks easier to dub within timing constraints

Output:

- `refined_text`

The user-approved text should be stored separately as `approved_text`.

## Timing Strategy

Timing fit should be explicit and inspectable.

Each block should carry:

- source duration
- synthesized duration
- delta ratio
- fit classification

Suggested fit classes:

- `good`
- `slightly_long`
- `slightly_short`
- `requires_rewrite`
- `requires_time_stretch`

Preferred order of correction:

1. better localized rewrite
2. punctuation/prosody adjustment
3. conservative time-stretch
4. narrator fallback

Do not rely on aggressive stretch as the primary fix.

## Speaker Strategy

Recommended speaker policy:

- use diarized speakers when transcript quality and speaker stability are good
- map each speaker to an approved target voice strategy
- if speaker assignment is weak, allow:
  - grouped speakers
  - single narrator mode

This feature should not pretend to be more certain than the underlying diarization quality.

## Audio / Mix Strategy

V1 should use a pragmatic mix pipeline:

- preserve original background when possible
- generate a dubbed dialogue stem
- duck original dialogue/background under dubbed speech
- export an intelligible localized master

Do not block V1 on perfect dialogue isolation.

If source separation is available later, it should slot into the `mix` stage as an optional enhancement.

## Backend Job Plan

Add new job types:

- `localization_prepare`
- `localization_translate`
- `localization_refine`
- `localization_voice_test`
- `localization_synthesize`
- `localization_align`
- `localization_mix`
- `localization_export`

Each job should:

- be idempotent
- record provider/model/settings in a run record
- support stage-specific regeneration
- avoid deleting downstream artifacts silently

## API Plan

Suggested backend routes:

### Project

- `POST /videos/{video_id}/localization-projects`
- `GET /videos/{video_id}/localization-projects`
- `GET /localization-projects/{project_id}`
- `DELETE /localization-projects/{project_id}`

### Blocks

- `GET /localization-projects/{project_id}/blocks`
- `PUT /localization-blocks/{block_id}`
- `POST /localization-projects/{project_id}/blocks/translate`
- `POST /localization-projects/{project_id}/blocks/refine`
- `POST /localization-projects/{project_id}/blocks/approve`

### Glossary

- `GET /localization-projects/{project_id}/glossary`
- `POST /localization-projects/{project_id}/glossary`
- `PUT /localization-glossary/{entry_id}`
- `DELETE /localization-glossary/{entry_id}`

### Voice Assignment

- `GET /localization-projects/{project_id}/voices`
- `PUT /localization-projects/{project_id}/voices/{speaker_id}`
- `POST /localization-projects/{project_id}/voices/test`

### Dubbing

- `POST /localization-projects/{project_id}/synthesize`
- `POST /localization-projects/{project_id}/align`
- `POST /localization-projects/{project_id}/mix`
- `POST /localization-projects/{project_id}/export`

### Artifacts

- `GET /localization-projects/{project_id}/artifacts`
- `GET /localization-projects/{project_id}/runs`

## Frontend Plan

Add a dedicated `Localize` or `Dub` tab on the episode detail page that takes over the main workbench area.

Do not place this in the transcript sidebar.

Recommended layout:

- left rail
  - project selector
  - stage navigation
  - speakers
  - glossary quick access
- main stage
  - stage-specific workbench content
- right utility rail optional
  - artifact preview
  - job state
  - warnings

The workbench should follow the same broad design patterns as:

- clone workbench
- reconstruction workbench
- transcript optimization workbench

## Settings / Dependency Plan

The feature should support pluggable providers for:

- translation
- TTS / voice cloning
- optional source separation

Initial provider abstraction:

- `translation_provider`
- `tts_provider`
- `alignment_strategy`
- `mix_strategy`

Do not hard-code one vendor path.

## Validation Strategy

Evaluate the system on:

- translation quality
- terminology consistency
- speaker consistency
- timing fit
- mix intelligibility

Create a representative benchmark set:

- single-speaker commentary
- two-person podcast/interview
- multi-speaker panel
- non-English source episode
- jargon-heavy episode

Review metrics:

- % blocks approved without edit
- % blocks requiring rewrite for timing
- mean synthesized/source duration ratio
- % blocks with pronunciation correction
- % projects that can stay multi-speaker instead of narrator fallback

## Risks

- weak diarization can make multi-speaker dubbing unreliable
- translated speech length can break timing badly
- voice cloning quality may vary heavily across providers
- pronunciation for names and technical terms will be a recurring issue
- background preservation may be “good enough” but not pristine in V1

## Recommended Initial Scope

V1:

- transcript-driven project creation
- block translation and refinement
- glossary and do-not-translate rules
- per-speaker voice assignment
- synthesized dubbed dialogue stem
- basic alignment
- translated transcript and subtitle export
- dubbed audio export

V1.5:

- background-preserving mixed master
- remuxed alternate-language video
- pronunciation rule library
- block-level re-synthesis controls

V2:

- source separation upgrade
- stronger timing adaptation
- better multi-speaker dubbing on weak diarization
- language-pair specific style templates

## Recommended Build Order

1. data model and project/block APIs
2. translation workbench with editable blocks
3. glossary and terminology controls
4. voice assignment model and test synthesis
5. block synthesis jobs
6. alignment and fit diagnostics
7. export pipeline
8. background-preserving mix stage

## Definition Of Success

The feature is successful when a user can:

1. open an episode and create a localization project
2. translate the transcript into a target language
3. review and correct the translation before dubbing
4. assign target voices per speaker
5. generate dubbed speech that maps back onto the episode timeline
6. export usable alternate-language subtitle and audio assets

The workbench should feel like an editable localization studio, not a hidden backend process.
