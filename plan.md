# Chatalogue - Implementation Plan

## Phase 1: Core Backend (Current)
- [x] **Project Setup**: Basic FastAPI app and directory structure.
- [x] **Database**: SQLModel definitions for Channel, Video, Speaker.
- [x] **Ingestion Logic**: `yt-dlp` integration for channels/videos.
- [x] **ML Pipeline**: Whisper and Pyannote integration (Stubbed/Implemented).
- [x] **API**:/channels, /videos, /speakers endpoints.

## Phase 2: Refinement & UI (Current)
- [x] **Verification**: Ensure pipeline runs end-to-end on GPU.
- [ ] **Data Persistence**: Verify database handles real-world data correctly.
- [/] **Frontend**: Build a premium web UI using Vite + React + TailwindCSS.

## Phase 3: Advanced Features
- [x] **Search**: Full-text search on transcripts (Implemented by User).
- [ ] **Auto-Scheduling**: Background job to refresh channels periodically.
