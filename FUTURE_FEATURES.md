# Future Features

## 1. Speaker Wordcloud Tab
- Add an automated wordcloud generation tab on each speaker profile.
- Candidate implementation: [`minimaxir/stylecloud`](https://github.com/minimaxir/stylecloud) or a similar library.
- Goal: visualize a speaker's most frequent/defining terms from their transcript corpus.

## 2. Per-Episode Wordcloud
- Add per-episode wordcloud generation so each episode has its own term visualization.
- Reuse the same rendering pipeline/library approach as speaker wordclouds where possible.
- Goal: quickly identify episode themes and recurring terms at a glance.

## 3. Vector Embeddings + Semantic Search
- Add vector embedding generation for transcript content.
- Implement semantic search across episodes/transcripts using embedding similarity.
- Goal: support meaning-based discovery beyond exact keyword matches.

## 4. Preliminary YouTube-Caption Placeholder Transcript
- On new channel or episode ingest, fetch available YouTube captions and store a preliminary transcript as a searchable placeholder.
- Mark placeholder transcripts clearly in the UI until local high-quality transcription/diarization completes.
- Automatically replace placeholder content when local transcription finishes.

## 5. AI Voice Clone Per Speaker
- Add a feature to create an AI voice clone for a selected speaker profile.
- Define guardrails for consent/authorization and provenance tracking before enabling generation.
- Goal: enable speaker-specific synthetic voice workflows for approved use cases.

## 6. One-Shot Voice-Clone Source Snippet Builder
- Add a tool to generate a cleaned source snippet for one-shot voice cloning from a selected speaker sample.
- Include parameters for clip length, loudness normalization, background noise removal, and optional enhancement cleanup.
- Goal: produce high-quality, standardized input clips for downstream voice-cloning models.

## 7. Visual Speech + Face Pipeline
- Incorporate visual lip-reading signals alongside audio ASR to improve transcription robustness in noisy/crosstalk conditions.
- Detect and track speaker faces during playback to improve speaker identity linkage and auto-generate better speaker thumbnails.
- Aggregate speaker face crops across multiple videos to build a curated dataset for optional AI LoRA model training workflows.

## 8. Secure External Access Mode (Gradio-Style Sharing)
- Add an opt-in way to expose the local Chatalogue server externally, similar to Gradio share links.
- Support short-lived public URLs/tunnels with clear status, manual stop control, and expiry defaults.
- Include access controls (token/password/IP allowlist), audit logging, and explicit warning banners while sharing is enabled.
