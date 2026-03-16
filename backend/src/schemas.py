"""Pydantic / SQLModel schemas used across the API layer.

Extracted from main.py to reduce monolith size and allow reuse across routers.
"""
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
from sqlmodel import SQLModel

from .db.database import JobBase, Video


# ── Channel ──────────────────────────────────────────────────────────────────

class ChannelOverviewRead(BaseModel):
    id: int
    url: str
    name: str
    icon_url: Optional[str] = None
    header_image_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    status: str
    video_count: int = 0
    processed_count: int = 0
    speaker_count: int = 0
    total_duration_seconds: int = 0


class ChannelBatchPublishRequest(BaseModel):
    dry_run: bool = True
    confirm: bool = False
    push_to_youtube: Optional[bool] = None
    limit: Optional[int] = None


# ── Video ────────────────────────────────────────────────────────────────────

class VideoListItemRead(BaseModel):
    id: int
    youtube_id: str
    channel_id: Optional[int] = None
    title: str
    published_at: Optional[datetime] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    processed: bool = False
    muted: bool = False
    status: str


# ── Transcript / Search ─────────────────────────────────────────────────────

class TranscriptSearchItemRead(SQLModel):
    id: int
    video_id: int
    speaker_id: Optional[int] = None
    matched_profile_id: Optional[int] = None
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None


class TranscriptSearchPage(SQLModel):
    items: List[TranscriptSearchItemRead]
    total: int
    limit: int
    offset: int
    has_more: bool


class AssignSpeakerRequest(SQLModel):
    speaker_id: int


class SegmentTextUpdateRequest(SQLModel):
    text: str
    words: Optional[List[str]] = None


class SplitSegmentProfileRequest(BaseModel):
    profile_id: Optional[int] = None
    target_speaker_id: Optional[int] = None
    new_speaker_name: Optional[str] = None
    reassign_segment: bool = True


# ── Clips ────────────────────────────────────────────────────────────────────

class ClipCreate(SQLModel):
    start_time: float
    end_time: float
    title: str
    aspect_ratio: str = "source"
    crop_x: Optional[float] = None
    crop_y: Optional[float] = None
    crop_w: Optional[float] = None
    crop_h: Optional[float] = None
    portrait_split_enabled: bool = False
    portrait_top_crop_x: Optional[float] = None
    portrait_top_crop_y: Optional[float] = None
    portrait_top_crop_w: Optional[float] = None
    portrait_top_crop_h: Optional[float] = None
    portrait_bottom_crop_x: Optional[float] = None
    portrait_bottom_crop_y: Optional[float] = None
    portrait_bottom_crop_w: Optional[float] = None
    portrait_bottom_crop_h: Optional[float] = None
    script_edits_json: Optional[str] = None
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    burn_captions: bool = False
    caption_speaker_labels: bool = True


class ClipRead(ClipCreate):
    id: int
    video_id: int
    created_at: datetime


class ChannelClipRead(ClipRead):
    video_title: str
    video_youtube_id: str
    video_published_at: Optional[datetime] = None
    video_thumbnail_url: Optional[str] = None


class ClipCaptionExportRequest(SQLModel):
    format: str = "srt"
    speaker_labels: Optional[bool] = None


class ClipExportPresetRequest(SQLModel):
    burn_captions: Optional[bool] = None
    caption_speaker_labels: Optional[bool] = None
    aspect_ratio: Optional[str] = None
    crop_x: Optional[float] = None
    crop_y: Optional[float] = None
    crop_w: Optional[float] = None
    crop_h: Optional[float] = None
    portrait_split_enabled: Optional[bool] = None
    portrait_top_crop_x: Optional[float] = None
    portrait_top_crop_y: Optional[float] = None
    portrait_top_crop_w: Optional[float] = None
    portrait_top_crop_h: Optional[float] = None
    portrait_bottom_crop_x: Optional[float] = None
    portrait_bottom_crop_y: Optional[float] = None
    portrait_bottom_crop_w: Optional[float] = None
    portrait_bottom_crop_h: Optional[float] = None
    fade_in_sec: Optional[float] = None
    fade_out_sec: Optional[float] = None


class ClipYoutubeUploadRequest(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    privacy_status: str = "private"  # private|unlisted|public
    category_id: str = "22"  # People & Blogs
    made_for_kids: bool = False
    tags: Optional[List[str]] = None


class ClipBatchYoutubeUploadRequest(SQLModel):
    clip_ids: List[int]
    privacy_status: str = "private"  # private|unlisted|public
    category_id: str = "22"
    made_for_kids: bool = False
    tags: Optional[List[str]] = None


# ── Speakers ─────────────────────────────────────────────────────────────────

class SpeakerRead(SQLModel):
    id: int
    channel_id: int
    name: str
    thumbnail_path: Optional[str] = None
    is_extra: bool = False
    total_speaking_time: float = 0.0
    embedding_count: int = 0
    created_at: datetime


class SpeakerCountsRead(SQLModel):
    total: int
    identified: int
    unknown: int
    main: int
    extras: int


class SpeakerEpisodeAppearanceRead(SQLModel):
    video_id: int
    youtube_id: str
    title: str
    published_at: Optional[datetime] = None
    thumbnail_url: Optional[str] = None
    segment_count: int
    total_speaking_time: float
    first_start_time: float
    last_end_time: float


class SpeakerVoiceProfileRead(SQLModel):
    id: int
    speaker_id: int
    source_video_id: Optional[int] = None
    source_video_title: Optional[str] = None
    source_video_youtube_id: Optional[str] = None
    source_video_published_at: Optional[datetime] = None
    sample_start_time: Optional[float] = None
    sample_end_time: Optional[float] = None
    sample_text: Optional[str] = None
    created_at: datetime


class MoveSpeakerProfileRequest(BaseModel):
    target_speaker_id: Optional[int] = None
    new_speaker_name: Optional[str] = None


class SpeakerSample(SQLModel):
    id: int
    video_id: int
    start_time: float
    end_time: float
    text: str
    channel_id: int
    youtube_id: str


class ExtractThumbnailRequest(SQLModel):
    video_id: int
    timestamp: float
    crop_coords: dict  # {x, y, w, h}


class MergeRequest(BaseModel):
    target_id: int
    source_ids: List[int]


# ── Jobs ─────────────────────────────────────────────────────────────────────

class JobRead(JobBase):
    id: int
    video: Optional[Video] = None


class PipelineFocusRead(BaseModel):
    mode: Literal["transcribe", "diarize"]
    auto_diarize_ready: bool
    transcribe_active: int
    transcribe_queued: int
    diarize_active: int
    diarize_queued: int
    active_transcription_paused: int = 0


class PipelineFocusUpdate(BaseModel):
    mode: Literal["transcribe", "diarize"]
    pause_active_transcription: bool = False


# ── Settings ─────────────────────────────────────────────────────────────────

class Settings(BaseModel):
    hf_token: str
    transcription_engine: str = "auto"  # auto | whisper | parakeet
    transcription_model: str = "medium"
    transcription_compute_type: str = "int8_float16"
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    parakeet_batch_size: int = 16
    parakeet_batch_auto: bool = True
    parakeet_require_word_timestamps: bool = True
    parakeet_allow_whisper_fallback: bool = True
    parakeet_unload_after_transcribe: bool = False
    beam_size: int = 1
    vad_filter: bool = True
    batched_transcription: bool = True
    verbose_logging: bool = False
    llm_provider: str = "ollama"
    llm_enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_model_tier: str = "medium"
    ollama_enabled: bool = False
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com"
    nvidia_nim_model: str = "moonshotai/kimi-k2.5"
    nvidia_nim_api_key: str = ""
    nvidia_nim_thinking_mode: bool = False
    nvidia_nim_min_request_interval_seconds: float = 2.5
    openai_base_url: str = "https://api.openai.com"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_api_key: str = ""
    xai_base_url: str = "https://api.x.ai"
    xai_model: str = "grok-2"
    xai_api_key: str = ""
    youtube_oauth_client_id: str = ""
    youtube_oauth_client_secret: str = ""
    youtube_oauth_redirect_uri: str = "http://localhost:8000/auth/youtube/callback"
    youtube_publish_push_enabled: bool = False
    ytdlp_cookies_file: str = ""
    ytdlp_cookies_from_browser: str = ""
    diarization_sensitivity: str = "balanced"
    speaker_match_threshold: float = 0.5
    funny_moments_max_saved: int = 25
    funny_moments_explain_batch_limit: int = 12
    setup_wizard_completed: bool = False


class OllamaPullRequest(BaseModel):
    url: Optional[str] = None
    model: Optional[str] = None
    wait_for_completion: bool = False


class TranscriptionEngineTestRequest(BaseModel):
    engine: Optional[str] = None
