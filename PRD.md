# Product Requirements Document (PRD)
## Chatalogue - YouTube Channel Indexer & Archiver

### 1. Overview
The **Chatalogue** is a local system designed to index, transcribe, and analyze content from specific YouTube channels. It serves as an archive and search engine, allowing users to track channel updates, identify speakers across videos, and extract specific video clips on demand.

### 2. Core Features

#### 2.1 Channel Management
- **Add Channel**: Users can register a YouTube channel by URL. The system fetches metadata (Name, ID).
- **Refresh Channel**: Users can trigger a scan for new videos. The system compares the live channel feed against the local database and queues new videos for processing.
- **List Channels**: View all tracked channels and their last updated status.

#### 2.2 Video Ingestion Pipeline
- **Audio Download**: Efficiently downloads only the audio track (m4a) for processing to save space.
- **Transcription**: Uses `faster-whisper` (model: `large-v3`) with CUDA acceleration for high-accuracy speech-to-text.
- **Diarization**: Uses `pyannote.audio` to distinguish different speakers and extract distinct voice embeddings (signatures).
- **Status Tracking**: Tracks the state of each video (`pending`, `downloading`, `transcribing`, `diarizing`, `completed`, `failed`).
- **Job Queue**: A FIFO queue manages processing to prevent resource exhaustion.
    - **Controls**: Pause All, Resume All, Clear Queue.
    - **Granular Status**: Real-time feedback on current processing stage.

#### 2.3 Speaker Identity Management
- **Speaker Recognition**: The system compares new voice segments against a database of known speaker embeddings (Cosine Similarity).
- ** persistent Identity**: Speakers are tracked across multiple videos within a channel.
- **Speaker Profiles**:
    - **Naming**: Users can rename identifying speakers (e.g., "Speaker 1" -> "Host Name").
    - **Thumbnail**: Users can upload or paste an image to visually identify a speaker.

#### 2.4 Content Retrieval & Clipping
- **Videos View**:
    - **Table Layout**: Sortable columns for Title, Date, Channel, Status.
    - **Search**: Real-time filtering by title and channel.
- **Search/Browse**: (Planned) Search transcripts for keywords.
- **On-Demand Clipping**: Users can define a start and end time for a specific video and generate a downloadable MP4 clip of that segment without downloading the entire original video.
    - **Mechanism**: Uses `yt-dlp`'s download sections capability to fetch only the required bytes from YouTube.

### 3. Technical Requirements

#### 3.1 Stack
- **Backend API**: FastAPI (Python)
- **Database**: SQLModel (SQLite)
- **AI/ML**:
    - `faster-whisper` (Transcription)
    - `pyannote.audio` (Diarization & Embeddings)
    - `scikit-learn` / `scipy` (Cosine Distance)
- **Media Processing**: `yt-dlp`, `ffmpeg`
- **Hardware**: NVIDIA GPU (CUDA) required for reasonable inference speeds.
- **Frontend**: React (Vite)

#### 3.2 Data Models
- **Channel**: `id`, `url`, `name`, `last_updated`, `status`
- **Video**: `id`, `channel_id`, `youtube_id`, `title`, `published_at`, `duration`, `status`, `muted`
- **Speaker**: `id`, `channel_id`, `name`, `embedding_blob`, `thumbnail_path`
- **TranscriptSegment**: `id`, `video_id`, `speaker_id`, `start_time`, `end_time`, `text`
- **Job**: `id`, `video_id`, `job_type`, `status` (`queued`, `running`, `paused`, `completed`, `failed`), `progress`, `created_at`

### 4. User Interaction Flow
1. **Setup**: User adds a channel via API/UI.
2. **Ingest**: System auto-crawls and processes videos (download -> transcribe -> diarize).
3. **Queue Management**: User monitors and controls processing via the Job Queue (Pause, Resume, Clear).
4.  **Curate**: User reviews identified speakers and assigns names/thumbnails.
5. **Utilize**: User searches for content or requests specific clips for download.

### 5. Development Workflow ("Ralph Loop")
- **Trigger**: User provides a task.
- **Step 1: Update PRD**: Reflect pending changes in `PRD.md` if necessary.
- **Step 2: Plan**: Analyze requirements and create an implementation plan (if complex).
- **Step 3: Execute**: Implement changes in code.
- **Step 4: Verify**: user confirms functionality or automated checks pass.
- **Loop**: Repeat for next task.

### 6. Constraints & Assumptions
- **Local Storage**: Audio files are stored locally.
- **GPU Dependency**: System assumes an NVIDIA GPU is available and configured.
- **YouTube limits**: Heavy usage of `yt-dlp` might trigger rate limits; refreshing should be done responsibly.
