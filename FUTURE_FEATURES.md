# Future Features

## Implemented Since Original Draft

### Preliminary YouTube-Caption Placeholder Transcript
- Status: implemented in March 2026.
- New channel refresh and manual episode ingest now attempt to fetch available YouTube captions and store them as searchable placeholder transcript segments.
- Placeholder transcripts are marked in the UI until local transcription/diarization completes.
- Final local transcription replaces the placeholder transcript automatically.

### Secure External Access Mode (Gradio-Style Sharing)
- Status: implemented in March 2026.
- Chatalogue can now run in LAN-share mode or public tunnel mode with short-lived share sessions.
- Share sessions support token/password protection, IP allowlists, expiry, audit logging, and operator controls in Settings.
- Remote sessions can access the main app and backend-streamed local media through the share URL.

### TikTok Channel Source Ingest
- Status: core implementation completed in March 2026; hardening remains.
- TikTok creator channels and individual TikTok videos can now be added and processed through the existing local-media pipeline.
- TikTok media is downloaded locally and played through the native media player instead of an embed, preserving word-level transcript sync and seek behavior.
- TikTok profile refresh now imports creator-feed videos, resolves creator artwork metadata, and supports placeholder captions when TikTok subtitle metadata is available.
- Remaining work is mostly platform hardening: rate-limit/auth edge cases, broader validation on live creator feeds, and further UI polish.

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

## 9. Intelligent Video Editing via Transcript Instructions
- Add support for AI-driven video editing directly from word-level diarized transcripts.
- Users input prompts like "cut filler words, repetitions, and off-topic parts; keep only the most important arguments."
- An integrated LLM proposes condensed segments with precise timestamps.
- Output: JSON/EDL or ready-to-run FFmpeg concat/trim commands for lossless local rendering.

## 10. AI Episode Cloning
- Add an AI-assisted episode analysis workflow that evaluates a source episode from a channel and produces a new original script based on the same core concept.
- Let the user direct the target rewrite style so the output can be adapted to different formats, tones, and presentation styles instead of being locked to a single template.
- Use semantic search across the rest of the channel's video library to pull in related themes, arguments, examples, and historical context from other episodes.
- Blend those semantically related references into the rewrite so the output is more original, more synthesized, and less dependent on a single source episode.
- Show channel and episode yield metrics including total views, popularity ranking, and views over time so users can identify the highest-performing candidates for cloning.
- Define guardrails for originality, attribution, and policy compliance before enabling script generation at scale.

## 11. AI Episode Localization & Dubbing
- Add a localization workflow that translates an episode transcript into a target language while preserving meaning, tone, and speaker intent.
- Let the user review and edit the translated transcript before any dubbing or synthesis step is finalized.
- Support speaker-preserving voice cloning or mapped synthetic voices so each diarized speaker can be dubbed with a consistent target-language voice.
- Generate dubbed audio aligned back onto the original episode timeline, with timing controls to handle translated speech that runs shorter or longer than the source.
- Export alternate-language subtitle tracks, translated transcripts, and dubbed episode audio as separate assets.
- Include pronunciation controls for names, technical terms, and channel-specific vocabulary to improve multilingual quality.
- Define consent, provenance, and voice-use guardrails before enabling production dubbing workflows.
