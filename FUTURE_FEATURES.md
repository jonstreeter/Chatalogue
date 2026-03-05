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
