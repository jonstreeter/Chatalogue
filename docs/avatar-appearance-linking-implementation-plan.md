# Avatar Appearance Linking Implementation Plan

## Purpose

This document defines a practical implementation plan for automatically linking diarized speakers to visible faces in video so PorchtimeIndex can build high-quality appearance datasets for avatar training.

The goal is not just better diarization. The goal is to produce:

- reliable `speaker -> face` matches
- clean face crops for appearance LoRA training
- a reviewable approval workflow in `Avatar Studio`
- reusable per-video and cross-video face identity artifacts

This is a planning document. Tool choices should be re-verified when implementation begins.

## Product Goal

For a given existing `Speaker`, the system should be able to:

1. find that speaker's speaking segments from audio diarization
2. find visible face tracks that overlap those segments
3. score which face track is actually speaking
4. harvest high-quality face crops from that track
5. surface those crops in `Avatar Studio` for approval or rejection
6. accumulate approved crops across videos into an appearance training set

## Non-Goals For V1

- replacing the main audio diarization engine
- fully automatic zero-review appearance dataset creation
- real-time audiovisual diarization during playback
- end-to-end training of new vision models in-repo
- perfect handling of every multi-person or off-camera case

## Recommended Stack

Use a custom stack built around the current audio pipeline.

### 1. Audio Diarization

- `pyannote speaker-diarization-community-1` if adopted later
- otherwise keep the current `pyannote/speaker-diarization-3.1` path already in the repo

Why:

- audio diarization is already integrated and good enough to act as the source of speech windows
- this project needs face linking and crop harvesting more than a diarization replacement

### 2. Face Detection

- `InsightFace SCRFD`

Why:

- strong practical detector
- fast enough for offline batch processing
- supported inside the broader InsightFace ecosystem

### 3. Face Embeddings / Identity

- `InsightFace ArcFace` via the `buffalo_l` family or equivalent current package

Why:

- best practical open-source face embedding option for identity clustering
- works well for within-video and cross-video identity grouping

Important caveat:

- review InsightFace model licensing before productizing

### 4. Tracking

- `ByteTrack`

Why:

- strong practical tracker
- simpler and more stable for a first implementation than heavier alternatives
- good fit for offline batch extraction

### 5. Active Speaker Detection

- primary target: `TalkNCE` / `LoCoNet`
- fallback target: `TalkNet`

Why:

- `TalkNCE` is the stronger current open benchmark option
- `TalkNet` is older but easier to operationalize if the stronger model proves too research-shaped

Design rule:

- build the pipeline behind an internal `asd_backend` interface so the model can be swapped later

## Repo Fit

This feature should extend the existing avatar workbench shape instead of creating a separate subsystem.

Current relevant integration points:

- backend API layer: [backend/src/main.py](/f:/Projects/PorchtimeIndex/backend/src/main.py)
- background jobs / ML execution: [backend/src/services/ingestion.py](/f:/Projects/PorchtimeIndex/backend/src/services/ingestion.py)
- schemas: [backend/src/schemas.py](/f:/Projects/PorchtimeIndex/backend/src/schemas.py)
- avatar UI placeholder: [frontend/src/pages/AvatarStudioPage.tsx](/f:/Projects/PorchtimeIndex/frontend/src/pages/AvatarStudioPage.tsx#L2150)

Current state:

- the app already has speaker thumbnails
- the app already has diarized speakers and transcript segments
- the avatar appearance section is still a placeholder

This makes appearance linking a natural next asset pipeline for Avatar Studio.

## System Architecture

The system should treat audio and video evidence as separate layers:

1. `pyannote` provides diarized speech windows
2. face detection + tracking builds face tracks over the video timeline
3. face embeddings provide identity continuity
4. ASD scores whether a visible tracked face is actively speaking during a diarized segment
5. a fusion stage produces ranked `speaker -> face-track` matches
6. a crop-harvesting stage exports candidate images for review

This is a batch job pipeline, not a request-response path.

## Core Pipeline

### Stage 1: Prepare Video Inputs

Input:

- local video file
- diarized transcript segments

Outputs:

- normalized frame sampling plan
- extracted audio/video metadata

Implementation notes:

- use ffmpeg for deterministic video metadata and frame extraction
- keep original media path as the source of truth
- avoid storing every frame unless later profiling proves it necessary

### Stage 2: Face Detection

Input:

- sampled video frames

Outputs:

- face boxes
- landmarks
- detection confidence

Implementation notes:

- run SCRFD on sampled frames
- store raw detections as an artifact for debugging
- support lower sampling outside speech windows and denser sampling during speech windows later if needed

### Stage 3: Face Tracking

Input:

- frame detections

Outputs:

- per-video face tracks with stable temporary track IDs
- per-track start/end timestamps

Implementation notes:

- use ByteTrack to connect detections into tracks
- allow track fragmentation; do not assume one track per identity forever
- each track should retain frame-level provenance for later crop selection

### Stage 4: Face Embedding Extraction

Input:

- tracked face crops

Outputs:

- embedding vector per usable frame
- embedding centroid per track
- quality stats per track

Implementation notes:

- use InsightFace identity embeddings
- discard unusable crops before centroid creation:
  - too small
  - too blurry
  - extreme angle
  - low detector confidence

### Stage 5: Active Speaker Detection

Input:

- diarized speech segment
- overlapping face tracks
- corresponding audio/video windows

Outputs:

- ASD confidence per `segment, track` pair

Implementation notes:

- run ASD only on candidate tracks that overlap the diarized segment
- do not run ASD over the whole video if simple overlap filtering can reduce cost
- surface the chosen backend in artifacts for traceability

### Stage 6: Audio-Visual Match Fusion

Input:

- diarized speaker segment
- overlapping face tracks
- ASD confidence
- track quality
- face embedding consistency

Outputs:

- ranked `speaker_id -> track_id` match candidates
- confidence score
- reason metadata

Scoring dimensions:

- ASD score
- overlap duration
- face visibility duration
- track quality score
- consistency with prior matches in the same video
- consistency with seed identity images if available

V1 rule:

- produce candidates plus confidence, not irreversible hard assignments

### Stage 7: Crop Harvesting

Input:

- accepted or high-confidence matched face tracks

Outputs:

- cropped face images
- per-crop metadata
- deduped candidate gallery

Crop filters:

- minimum face size
- minimum sharpness
- detector confidence threshold
- frontalness threshold
- diversity limit per segment and per video

### Stage 8: Human Review

Input:

- appearance candidate gallery

Outputs:

- approved crops
- rejected crops
- optional primary reference portrait

Review behaviors:

- approve
- reject
- mark wrong speaker
- mark duplicate
- choose primary image

## Data Model Additions

Add persistent entities for the appearance pipeline.

### `SpeakerFaceTrack`

- `id`
- `video_id`
- `track_key`
- `start_time`
- `end_time`
- `frame_count`
- `quality_score`
- `embedding_path` or `embedding_json`
- `metadata_json`

### `SpeakerFaceTrackFrame`

- `id`
- `track_id`
- `timestamp`
- `frame_path`
- `crop_path`
- `bbox_json`
- `landmarks_json`
- `sharpness_score`
- `pose_score`
- `detector_confidence`
- `embedding_json` or reference

### `SpeakerFaceMatch`

- `id`
- `video_id`
- `speaker_id`
- `track_id`
- `segment_id`
- `asd_backend`
- `asd_score`
- `fusion_score`
- `status`
- `metadata_json`

### `AvatarAppearanceCandidate`

- `id`
- `avatar_id`
- `speaker_id`
- `video_id`
- `track_id`
- `segment_id`
- `image_path`
- `timestamp`
- `quality_score`
- `match_confidence`
- `approval_status`
- `rejection_reason`
- `metadata_json`

## Filesystem Layout

Store artifacts under the existing avatar/video data structure.

Suggested layout:

```text
backend/data/videos/{video_id}/appearance/
  detections/
    detections.json
  tracks/
    tracks.json
    embeddings.json
  asd/
    scores.json
  matches/
    speaker_face_matches.json
  crops/
    raw/
    candidates/
    debug_contact_sheets/

backend/data/avatars/{avatar_id}/appearance/
  candidates/
  approved/
  exports/
  manifests/
```

## Backend Job Plan

Add explicit staged jobs instead of one monolithic appearance command.

Suggested job types:

- `appearance_detect_faces`
- `appearance_track_faces`
- `appearance_embed_faces`
- `appearance_match_speakers`
- `appearance_harvest_crops`
- `appearance_build_dataset`

Alternative:

- one parent job `appearance_harvest` with internal stages and resumable artifacts

Recommendation:

- start with one parent job and clear internal stage reporting
- split into multiple queue types only if scheduler pressure makes that necessary

## API Plan

### Video-Level Endpoints

- `POST /videos/{video_id}/appearance/prepare`
- `GET /videos/{video_id}/appearance/status`
- `GET /videos/{video_id}/appearance/tracks`
- `GET /videos/{video_id}/appearance/matches`

### Speaker-Level Endpoints

- `GET /speakers/{speaker_id}/appearance/candidates`
- `POST /speakers/{speaker_id}/appearance/harvest`
- `PATCH /speakers/{speaker_id}/appearance/candidates/{candidate_id}`

### Avatar-Level Endpoints

- `GET /avatars/{avatar_id}/appearance/candidates`
- `POST /avatars/{avatar_id}/appearance/harvest`
- `PATCH /avatars/{avatar_id}/appearance/candidates/{candidate_id}`
- `POST /avatars/{avatar_id}/appearance/export`

## Frontend Plan

Implement this inside the existing appearance section of Avatar Studio.

### V1 Panels

- `Source Summary`
  - videos scanned
  - tracks found
  - matches proposed
  - candidates harvested

- `Candidate Gallery`
  - grid of crops
  - face quality badge
  - match confidence badge
  - source video and timestamp

- `Review Queue`
  - approve / reject / duplicate / wrong speaker
  - quick keyboard workflow later

- `Approved Set`
  - current appearance dataset
  - primary image selection

### Nice-To-Have Later

- per-video face-track timeline viewer
- side-by-side speaker segment and face-track audit
- contact sheet visualization for harvested sequences

## Confidence Policy

The system should not silently auto-approve most crops.

Suggested confidence bands:

- `high`: can be auto-added to candidate gallery
- `medium`: show but visually flag for review
- `low`: keep only in debug artifacts unless the user opts in

Auto-approval policy for V1:

- none

V1 should require human approval before an image becomes part of the training set.

## Incremental Milestones

### Milestone 1: Artifact Skeleton

- add schema/models for face tracks, matches, and appearance candidates
- add filesystem layout
- add placeholder backend endpoints
- add status card in Avatar Studio

### Milestone 2: Face Tracking Pipeline

- ffmpeg frame extraction
- SCRFD detection
- ByteTrack tracking
- debug artifact generation

Success criteria:

- one video can produce auditable face tracks with timestamps and crop previews

### Milestone 3: Identity Embeddings

- InsightFace embeddings per crop
- per-track centroids
- basic within-video track clustering diagnostics

Success criteria:

- same person across fragmented tracks groups closely enough for manual audit

### Milestone 4: Manual Harvest Without ASD

- overlap diarized segments with face tracks
- rank candidate tracks by overlap and quality only
- generate first reviewable appearance gallery

Success criteria:

- user can approve useful appearance crops even before ASD exists

### Milestone 5: ASD Integration

- add internal ASD backend abstraction
- implement TalkNCE path
- add TalkNet fallback if needed
- fuse ASD scores into ranking

Success criteria:

- multi-person scenes improve materially over overlap-only matching

### Milestone 6: Dataset Export

- export approved crops into avatar appearance dataset folders
- add dedupe and diversity controls
- wire approved count into Avatar Studio summaries

Success criteria:

- avatar has a clean appearance dataset ready for future LoRA training

### Milestone 7: Cross-Video Identity Consolidation

- use embeddings and approved seeds to merge identity evidence across episodes
- improve ranking based on historical approvals

Success criteria:

- repeated episodes of the same speaker require less manual review over time

## Best Practices

- preserve every intermediate artifact needed for debugging
- keep raw detections separate from curated outputs
- version every scoring config used to build candidates
- make every stage resumable and idempotent
- prefer conservative candidate generation to aggressive wrong-person linking
- never mix unapproved crops directly into training exports

## Validation Strategy

Evaluation should be product-driven, not just benchmark-driven.

Track at least:

- candidate precision at top-k
- approval rate by confidence bucket
- wrong-speaker rejection rate
- duplicate rate
- average approved crop count per speaker
- runtime per processed hour of video

Manual test set:

- create a small local benchmark of 10 to 20 episodes with different conditions:
  - solo talking head
  - two-person interview
  - panel discussion
  - heavy overlap
  - off-camera speech
  - slide-heavy / cutaway-heavy video

## Risks

- InsightFace model licensing may block production use
- ASD inference may be too research-shaped to integrate cleanly on Windows
- poor video quality may limit automatic face linking more than model choice
- off-camera speech and reaction shots will remain hard cases
- over-harvesting near-duplicate crops can poison appearance training

## Open Questions

- do we want one `appearance_harvest` job per video or per speaker-video pair?
- should face detection run on all videos automatically after diarization, or only on demand?
- do we want to store frame-level embeddings in DB or keep them in artifact JSON/NPY files?
- is InsightFace licensing acceptable for the intended use?
- do we need a CPU-only fallback, or is this feature GPU-optional and offline only?

## Recommended First Build Order

If development starts later, begin in this exact order:

1. artifact schema + filesystem layout
2. SCRFD detection + ByteTrack tracking
3. candidate gallery from overlap-only heuristics
4. approval UI in Avatar Studio
5. InsightFace embeddings for better ranking and dedupe
6. ASD integration
7. cross-video identity consolidation

That order gets useful appearance curation into the product before the hardest research dependency is added.
