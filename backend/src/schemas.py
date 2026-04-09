"""Pydantic / SQLModel schemas used across the API layer.

Extracted from main.py to reduce monolith size and allow reuse across routers.
"""
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from sqlmodel import SQLModel

from .db.database import JobBase, Video


# ── Channel ──────────────────────────────────────────────────────────────────

class ChannelOverviewRead(BaseModel):
    id: int
    url: str
    name: str
    source_type: str = "youtube"
    icon_url: Optional[str] = None
    header_image_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    status: str
    actively_monitored: bool = False
    sync_status_detail: Optional[str] = None
    sync_progress: int = 0
    sync_total_items: int = 0
    sync_completed_items: int = 0
    video_count: int = 0
    processed_count: int = 0
    pending_video_count: int = 0
    active_job_count: int = 0
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
    media_source_type: str = "youtube"
    source_url: Optional[str] = None
    media_kind: Optional[str] = None
    manual_media_path: Optional[str] = None
    published_at: Optional[datetime] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    view_count: Optional[int] = None
    processed: bool = False
    muted: bool = False
    access_restricted: bool = False
    access_restriction_reason: Optional[str] = None
    status: str
    transcript_source: Optional[str] = None
    transcript_language: Optional[str] = None
    transcript_is_placeholder: bool = False
    last_pipeline_job_status: Optional[str] = None
    last_pipeline_job_type: Optional[str] = None


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


class TranscriptQualityRead(BaseModel):
    video_id: int
    title: str
    channel_id: Optional[int] = None
    quality_profile: str
    recommended_tier: str
    quality_score: float
    eligible_for_optimization: bool = False
    language: Optional[str] = None
    metrics: Dict[str, Any] = {}
    reasons: List[str] = []
    created_snapshot_id: Optional[int] = None
    snapshot_created_at: Optional[datetime] = None


class TranscriptQualitySnapshotRead(BaseModel):
    id: int
    video_id: int
    run_id: Optional[int] = None
    source: str
    quality_profile: str
    recommended_tier: str
    score: float
    metrics: Dict[str, Any] = {}
    reasons: List[str] = []
    created_at: datetime


class TranscriptRunRead(BaseModel):
    id: int
    video_id: int
    input_run_id: Optional[int] = None
    mode: str
    pipeline_version: str
    status: str
    quality_profile: Optional[str] = None
    recommended_tier: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None
    artifact_refs: Optional[Dict[str, Any]] = None
    rollback_state: Optional[str] = None
    model_provenance: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    created_at: datetime


class TranscriptRollbackOptionRead(BaseModel):
    run_id: int
    video_id: int
    mode: str
    pipeline_version: str
    note: Optional[str] = None
    created_at: datetime
    rollback_available: bool = False
    rollback_state: Optional[str] = None


class TranscriptRestoreResponse(BaseModel):
    video_id: int
    restored_from_run_id: int
    restore_run_id: int
    segment_count: int
    funny_moment_count: int = 0
    quality_profile: str
    recommended_tier: str
    quality_score: float


class TranscriptGoldWindowUpsertRequest(BaseModel):
    label: str = Field(default="window", max_length=160)
    quality_profile: Optional[str] = Field(default=None, max_length=80)
    language: Optional[str] = Field(default=None, max_length=32)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    reference_text: str = Field(min_length=1, max_length=12000)
    entities: List[str] = []
    notes: Optional[str] = Field(default=None, max_length=2000)
    active: bool = True


class TranscriptGoldWindowRead(BaseModel):
    id: int
    video_id: int
    label: str
    quality_profile: Optional[str] = None
    language: Optional[str] = None
    start_time: float
    end_time: float
    reference_text: str
    entities: List[str] = []
    notes: Optional[str] = None
    active: bool = True
    created_at: datetime
    updated_at: datetime


class TranscriptEvaluationResultRead(BaseModel):
    id: int
    gold_window_id: int
    video_id: int
    run_id: Optional[int] = None
    source: str
    candidate_text: str
    reference_text: str
    wer: float
    cer: float
    entity_accuracy: Optional[float] = None
    matched_entity_count: int = 0
    total_entity_count: int = 0
    segment_count: int = 0
    unknown_speaker_rate: float = 0.0
    punctuation_density_delta: float = 0.0
    metrics: Dict[str, Any] = {}
    created_at: datetime


class TranscriptEvaluationReviewRequest(BaseModel):
    reviewer: Optional[str] = Field(default=None, max_length=160)
    verdict: Literal["better", "same", "worse", "bad_merge", "bad_speaker_reassignment", "bad_entity_repair", "language_regression"]
    tags: List[str] = []
    notes: Optional[str] = Field(default=None, max_length=2000)


class TranscriptEvaluationReviewRead(BaseModel):
    id: int
    evaluation_result_id: int
    reviewer: Optional[str] = None
    verdict: str
    tags: List[str] = []
    notes: Optional[str] = None
    created_at: datetime


class TranscriptEvaluationBatchResponse(BaseModel):
    video_id: int
    run_id: Optional[int] = None
    total_windows: int
    average_wer: float
    average_cer: float
    average_entity_accuracy: Optional[float] = None
    average_unknown_speaker_rate: float = 0.0
    items: List[TranscriptEvaluationResultRead] = []


class TranscriptEvaluationSummaryRead(BaseModel):
    scope: str
    channel_id: Optional[int] = None
    total_gold_windows: int
    total_results: int
    total_reviewed_results: int
    average_wer: Optional[float] = None
    average_cer: Optional[float] = None
    average_entity_accuracy: Optional[float] = None
    average_unknown_speaker_rate: Optional[float] = None
    verdict_counts: Dict[str, int] = {}
    latest_result_at: Optional[datetime] = None


class TranscriptDiarizationConfigBenchmarkRead(BaseModel):
    label: str
    run_count: int
    average_wer: Optional[float] = None
    average_cer: Optional[float] = None
    average_unknown_speaker_rate: Optional[float] = None
    latest_run_id: Optional[int] = None
    latest_created_at: Optional[datetime] = None
    diarization_sensitivity: Optional[str] = None
    speaker_match_threshold: Optional[float] = None


class TranscriptOptimizeDryRunRequest(BaseModel):
    channel_id: Optional[int] = None
    video_id: Optional[int] = None
    limit: int = Field(default=50, ge=1, le=500)
    persist_snapshots: bool = False


class TranscriptOptimizeDryRunResponse(BaseModel):
    total_scanned: int
    total_eligible: int
    items: List[TranscriptQualityRead] = []


class TranscriptRepairQueueRequest(BaseModel):
    save_files: bool = True
    force: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptRepairQueueResponse(BaseModel):
    job_id: int
    video_id: int
    status: str
    recommended_tier: str
    quality_score: float
    queued: bool = True


class TranscriptRepairResultRead(BaseModel):
    video_id: int
    title: str
    before_count: int
    after_count: int
    merged_count: int
    reassigned_islands: int
    changed: bool
    backup_file: Optional[str] = None
    run_id: Optional[int] = None
    snapshot_id: Optional[int] = None
    quality_profile_before: str
    quality_profile_after: str
    recommended_tier_before: str
    recommended_tier_after: str
    quality_score_before: float
    quality_score_after: float


class TranscriptRepairBulkQueueRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    save_files: bool = True
    force_non_eligible: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptRepairBulkQueueResponse(BaseModel):
    channel_id: int
    queued: int
    skipped_active: int = 0
    skipped_no_segments: int = 0
    skipped_not_low_risk: int = 0
    jobs: List[TranscriptRepairQueueResponse] = []


class TranscriptDiarizationRebuildQueueRequest(BaseModel):
    force: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptDiarizationBenchmarkRequest(BaseModel):
    force: bool = True
    note: Optional[str] = Field(default=None, max_length=500)
    diarization_sensitivity: Literal["aggressive", "balanced", "conservative"] = "balanced"
    speaker_match_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class TranscriptDiarizationRebuildQueueResponse(BaseModel):
    job_id: int
    video_id: int
    status: str
    recommended_tier: str
    quality_score: float
    queued: bool = True


class TranscriptDiarizationRebuildBulkQueueRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    force_non_eligible: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptDiarizationRebuildBulkQueueResponse(BaseModel):
    channel_id: int
    queued: int
    skipped_active: int = 0
    skipped_no_raw_transcript: int = 0
    skipped_not_diarization_rebuild: int = 0
    skipped_unprocessed: int = 0
    skipped_muted: int = 0
    jobs: List[TranscriptDiarizationRebuildQueueResponse] = []


class TranscriptRetranscriptionQueueRequest(BaseModel):
    force: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptRetranscriptionQueueResponse(BaseModel):
    job_id: int
    video_id: int
    status: str
    recommended_tier: str
    quality_score: float
    queued: bool = True


class TranscriptRetranscriptionBulkQueueRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    force_non_eligible: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptRetranscriptionBulkQueueResponse(BaseModel):
    channel_id: int
    queued: int
    skipped_active: int = 0
    skipped_not_full_retranscription: int = 0
    skipped_unprocessed: int = 0
    skipped_muted: int = 0
    jobs: List[TranscriptRetranscriptionQueueResponse] = []


class TranscriptOptimizationCampaignCreateRequest(BaseModel):
    channel_id: Optional[int] = None
    limit: int = Field(default=100, ge=1, le=1000)
    tiers: List[Literal["low_risk_repair", "diarization_rebuild", "full_retranscription", "manual_review"]] = []
    force_non_eligible: bool = False
    note: Optional[str] = Field(default=None, max_length=500)


class TranscriptOptimizationCampaignRead(BaseModel):
    id: int
    channel_id: Optional[int] = None
    scope: str
    status: str
    tiers: List[str] = []
    limit: int
    force_non_eligible: bool = False
    queued_jobs: int = 0
    skipped_active: int = 0
    skipped_no_segments: int = 0
    skipped_not_eligible: int = 0
    skipped_other: int = 0
    note: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TranscriptOptimizationCampaignItemRead(BaseModel):
    id: int
    campaign_id: int
    video_id: int
    recommended_tier: str
    action_tier: str
    quality_score: float
    reason: Optional[str] = None
    status: str
    job_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class TranscriptOptimizationCampaignExecuteResponse(BaseModel):
    campaign_id: int
    status: str
    queued_jobs: int
    skipped_active: int = 0
    skipped_no_segments: int = 0
    skipped_not_eligible: int = 0
    skipped_other: int = 0


class TranscriptOptimizationCampaignDeleteResponse(BaseModel):
    campaign_id: int
    deleted_items: int
    detached_job_refs: int = 0
    status: str = "deleted"


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


class SpeakerOverviewRead(SQLModel):
    items: List[SpeakerRead]
    counts: SpeakerCountsRead
    total: int
    offset: int
    limit: Optional[int] = None


class SpeakerEpisodeAppearanceRead(SQLModel):
    video_id: int
    youtube_id: str
    title: str
    media_source_type: str = "youtube"
    media_kind: Optional[str] = None
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
    source_video_media_source_type: Optional[str] = None
    source_video_media_kind: Optional[str] = None
    source_video_published_at: Optional[datetime] = None
    sample_start_time: Optional[float] = None
    sample_end_time: Optional[float] = None
    sample_text: Optional[str] = None
    created_at: datetime


class SpeakerMergeSuggestionRead(SQLModel):
    source_speaker_id: int
    source_speaker_name: str
    target_speaker_id: int
    target_speaker_name: str
    distance: float
    source_embedding_count: int = 0
    target_embedding_count: int = 0
    source_segment_count: int = 0
    target_segment_count: int = 0


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
    media_source_type: str = "youtube"
    media_kind: Optional[str] = None


class ExtractThumbnailRequest(SQLModel):
    video_id: int
    timestamp: float
    crop_coords: dict  # {x, y, w, h}


class MergeRequest(BaseModel):
    target_id: int
    source_ids: List[int]


class AvatarCreateRequest(BaseModel):
    speaker_id: int
    name: Optional[str] = None
    description: Optional[str] = None


class AvatarUpdateRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    personality_system_prompt: Optional[str] = None
    personality_base_model_id: Optional[str] = None
    appearance_primary_image_path: Optional[str] = None
    voice_primary_reference_path: Optional[str] = None
    voice_provider: Optional[str] = None


class AvatarRead(BaseModel):
    id: int
    channel_id: int
    speaker_id: int
    name: str
    status: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AvatarSectionSummaryRead(BaseModel):
    status: str
    source_count: int = 0
    approved_count: int = 0
    artifact_ready: bool = False
    summary: Optional[str] = None
    artifact_path: Optional[str] = None
    last_built_at: Optional[datetime] = None


class AvatarWorkbenchSpeakerRead(BaseModel):
    id: int
    channel_id: int
    name: str
    thumbnail_path: Optional[str] = None
    total_speaking_time: float = 0.0
    embedding_count: int = 0
    appearance_count: int = 0


class AvatarWorkbenchRead(BaseModel):
    avatar: AvatarRead
    speaker: AvatarWorkbenchSpeakerRead
    personality: AvatarSectionSummaryRead
    appearance: AvatarSectionSummaryRead
    voice: AvatarSectionSummaryRead
    runtime_status: str = "not_ready"
    suggested_base_model: Optional[str] = None
    artifacts_dir: Optional[str] = None


class AvatarPersonalityDatasetExampleRead(BaseModel):
    example_id: int
    video_id: int
    video_title: str
    start_time: float
    end_time: float
    context_text: str
    response_text: str
    source_segment_ids: List[int] = []
    source_segment_count: int = 0
    context_turns: int = 0
    response_word_count: int = 0
    context_word_count: int = 0
    quality_score: int = 0
    completion_score: int = 0
    context_score: int = 0
    style_score: int = 0
    substance_score: int = 0
    cluster_id: Optional[int] = None
    cluster_size: int = 0
    diversity_score: int = 0
    duplicate_group_id: Optional[int] = None
    duplicate_group_size: int = 0
    duplicate_similarity: float = 0.0
    heuristic_label: Optional[Literal["gold", "silver", "reject"]] = None
    llm_label: Optional[Literal["gold", "silver", "reject"]] = None
    llm_confidence: int = 0
    llm_completion_score: int = 0
    llm_style_score: int = 0
    llm_usefulness_score: int = 0
    llm_rationale: Optional[str] = None
    llm_reasons: List[str] = []
    llm_model: Optional[str] = None
    llm_judged_at: Optional[datetime] = None
    auto_label: Literal["gold", "silver", "reject"] = "silver"
    manual_state: Optional[Literal["approved", "rejected"]] = None
    state: Literal["approved", "rejected"] = "approved"
    reject_reasons: List[str] = []


class AvatarPersonalityTrainingReadinessRead(BaseModel):
    status: Literal["insufficient", "borderline", "ready", "strong", "oversized"] = "insufficient"
    score: int = 0
    can_train_now: bool = False
    approved_duration_hours: float = 0.0
    approved_word_count: int = 0
    recommended_action: Optional[str] = None
    summary: Optional[str] = None
    caution: Optional[str] = None
    manual_review_roi: Literal["high", "medium", "low"] = "high"
    duplicate_pressure: int = 0
    hotspot_pressure: int = 0
    gold_ratio: float = 0.0
    reject_ratio: float = 0.0


class AvatarPersonalityDatasetRead(BaseModel):
    avatar_id: int
    speaker_id: int
    status: str
    system_prompt: Optional[str] = None
    base_model_id: Optional[str] = None
    dataset_path: Optional[str] = None
    metadata_path: Optional[str] = None
    example_count: int = 0
    gold_example_count: int = 0
    silver_example_count: int = 0
    auto_reject_count: int = 0
    needs_review_count: int = 0
    duplicate_example_count: int = 0
    cluster_count: int = 0
    hotspot_cluster_count: int = 0
    llm_judged_count: int = 0
    llm_promoted_count: int = 0
    llm_rejected_count: int = 0
    source_turn_count: int = 0
    discarded_turn_count: int = 0
    readiness: AvatarPersonalityTrainingReadinessRead = AvatarPersonalityTrainingReadinessRead()
    preview_examples: List[AvatarPersonalityDatasetExampleRead] = []
    generated_at: Optional[datetime] = None


class AvatarPersonalityDatasetPageRead(BaseModel):
    avatar_id: int
    total: int
    approved_count: int
    rejected_count: int
    gold_count: int
    silver_count: int
    auto_reject_count: int
    needs_review_count: int
    duplicate_count: int
    cluster_count: int
    hotspot_cluster_count: int
    llm_judged_count: int
    llm_promoted_count: int
    llm_rejected_count: int
    limit: int
    offset: int
    has_more: bool
    state_filter: Literal["all", "approved", "rejected", "gold", "silver", "auto_reject", "needs_review", "duplicate_risk"] = "all"
    items: List[AvatarPersonalityDatasetExampleRead] = []


class AvatarPersonalityExampleStateRequest(BaseModel):
    state: Literal["approved", "rejected", "inherit"]


class AvatarPersonalityJudgePassRequest(BaseModel):
    max_examples: int = 40
    overwrite_existing: bool = False
    target_filter: Literal["needs_review", "silver", "all"] = "needs_review"


class AvatarPersonalityJudgeFeedItemRead(BaseModel):
    example_id: int
    video_title: str
    llm_label: Literal["gold", "silver", "reject"]
    llm_confidence: int = 0
    llm_reasons: List[str] = []
    heuristic_label: Optional[Literal["gold", "silver", "reject"]] = None
    judged_at: Optional[datetime] = None


class AvatarPersonalityJudgeStatusRead(BaseModel):
    avatar_id: int
    status: Literal["idle", "running", "stopping", "completed", "stopped", "failed"] = "idle"
    active: bool = False
    stop_requested: bool = False
    model: Optional[str] = None
    target_filter: Literal["needs_review", "silver", "all"] = "needs_review"
    overwrite_existing: bool = False
    max_examples: int = 40
    total_candidates: int = 0
    processed_count: int = 0
    judged_count: int = 0
    promoted_count: int = 0
    rejected_count: int = 0
    current_example_id: Optional[int] = None
    current_video_title: Optional[str] = None
    current_stage: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    recent_results: List[AvatarPersonalityJudgeFeedItemRead] = []


class AvatarPersonalityLongFormSampleRead(BaseModel):
    sample_id: str
    video_id: int
    video_title: str
    start_time: float
    end_time: float
    duration_seconds: float = 0.0
    word_count: int = 0
    segment_count: int = 0
    text: str
    style_density: float = 0.0
    substance_density: float = 0.0
    state: Literal["included", "rejected"] = "included"


class AvatarPersonalityLongFormConfigRead(BaseModel):
    take_count: int = 0
    included_count: int = 0
    rejected_count: int = 0
    selected_count: int = 0


class AvatarPersonalityLongFormPageRead(BaseModel):
    avatar_id: int
    total: int
    included_count: int
    rejected_count: int
    selected_count: int
    take_count: int
    limit: int
    offset: int
    has_more: bool
    items: List[AvatarPersonalityLongFormSampleRead] = []


class AvatarPersonalityLongFormSampleStateRequest(BaseModel):
    state: Literal["included", "rejected"]


class AvatarPersonalityLongFormConfigUpdateRequest(BaseModel):
    take_count: int = 0


class AvatarPersonalityTrainingDatasetProfileRead(BaseModel):
    key: Literal["focused", "balanced", "broad", "exhaustive", "custom"] = "balanced"
    label: str
    summary: str
    conversation_target: int = 0
    long_form_target: int = 0
    pros: List[str] = []
    cons: List[str] = []
    recommended: bool = False


class AvatarPersonalityTrainingPlanRead(BaseModel):
    dataset_profile: Literal["focused", "balanced", "broad", "exhaustive", "custom"] = "balanced"
    dataset_profile_label: str = "Balanced"
    conversation_target: int = 0
    long_form_target: int = 0
    available_conversation_examples: int = 0
    available_long_form_examples: int = 0
    estimated_conversation_examples: int = 0
    estimated_long_form_examples: int = 0
    estimated_total_examples: int = 0
    estimated_train_examples: int = 0
    estimated_validation_examples: int = 0
    estimated_effective_batch_size: int = 0
    estimated_steps_per_epoch: int = 0
    estimated_total_steps: int = 0
    step_band: Literal["light", "ideal", "heavy", "aggressive"] = "ideal"
    headline: str = ""
    recommendation: str = ""
    snapshot_interval_suggestion: int = 0
    pros: List[str] = []
    cons: List[str] = []


class AvatarPersonalityTrainingConfigRead(BaseModel):
    base_model_id: str = "Qwen/Qwen3-8B"
    dataset_profile: Literal["focused", "balanced", "broad", "exhaustive", "custom"] = "balanced"
    training_strength: Literal["conservative", "balanced", "strong"] = "balanced"
    export_strategy: Literal["gold_only", "gold_balanced", "gold_plus_top_silver", "full_approved"] = "gold_balanced"
    validation_ratio: float = 0.10
    max_examples: int = 2500
    max_long_form_examples: int = 80
    include_long_form: bool = True
    training_mode: Literal["standard", "memory_optimized"] = "memory_optimized"
    snapshot_interval_steps: int = 0
    dataset_profiles: List[AvatarPersonalityTrainingDatasetProfileRead] = []
    training_plan: Optional[AvatarPersonalityTrainingPlanRead] = None


class AvatarPersonalityTrainingConfigUpdateRequest(BaseModel):
    base_model_id: Optional[str] = None
    dataset_profile: Optional[Literal["focused", "balanced", "broad", "exhaustive", "custom"]] = None
    training_strength: Optional[Literal["conservative", "balanced", "strong"]] = None
    export_strategy: Optional[Literal["gold_only", "gold_balanced", "gold_plus_top_silver", "full_approved"]] = None
    validation_ratio: Optional[float] = None
    max_examples: Optional[int] = None
    max_long_form_examples: Optional[int] = None
    include_long_form: Optional[bool] = None
    training_mode: Optional[Literal["standard", "memory_optimized"]] = None
    snapshot_interval_steps: Optional[int] = None


class AvatarPersonalityTrainingPackageRead(BaseModel):
    avatar_id: int
    status: Literal["not_prepared", "ready"] = "not_prepared"
    base_model_id: Optional[str] = None
    dataset_profile: Literal["focused", "balanced", "broad", "exhaustive", "custom"] = "balanced"
    training_strength: Literal["conservative", "balanced", "strong"] = "balanced"
    export_strategy: Literal["gold_only", "gold_balanced", "gold_plus_top_silver", "full_approved"] = "gold_balanced"
    validation_ratio: float = 0.10
    max_examples: int = 0
    max_long_form_examples: int = 0
    include_long_form: bool = True
    conversation_examples_selected: int = 0
    long_form_examples_selected: int = 0
    total_examples_selected: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    prompt: Optional[str] = None
    manifest_path: Optional[str] = None
    config_path: Optional[str] = None
    train_dataset_path: Optional[str] = None
    validation_dataset_path: Optional[str] = None
    command_preview: Optional[str] = None
    prepared_at: Optional[datetime] = None
    training_plan: Optional[AvatarPersonalityTrainingPlanRead] = None


class AvatarPersonalityTrainRequest(BaseModel):
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: Optional[int] = None
    max_seq_length: Optional[int] = None
    per_device_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    snapshot_interval_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    overwrite_output: bool = False
    training_mode: Literal["standard", "memory_optimized"] = "memory_optimized"


class AvatarPersonalitySnapshotRead(BaseModel):
    label: str
    kind: Literal["step", "epoch", "final"] = "step"
    adapter_path: str
    step: int = 0
    epoch: float = 0.0
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    created_at: Optional[datetime] = None
    selected: bool = False


class AvatarPersonalityTrainingStatusRead(BaseModel):
    avatar_id: int
    status: Literal["idle", "queued", "running", "stopping", "completed", "stopped", "failed"] = "idle"
    active: bool = False
    stop_requested: bool = False
    process_id: Optional[int] = None
    base_model_id: Optional[str] = None
    training_mode: Literal["standard", "memory_optimized"] = "memory_optimized"
    adapter_path: Optional[str] = None
    output_dir: Optional[str] = None
    current_stage: Optional[str] = None
    epoch: float = 0.0
    step: int = 0
    max_steps: int = 0
    snapshot_interval_steps: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    latest_loss: Optional[float] = None
    message: Optional[str] = None
    snapshots: List[AvatarPersonalitySnapshotRead] = []
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class AvatarPersonalityTestChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AvatarPersonalityTestChatRequest(BaseModel):
    message: str
    history: List[AvatarPersonalityTestChatTurn] = []
    adapter_path: Optional[str] = None
    max_new_tokens: int = 220
    temperature: float = 0.8
    top_p: float = 0.9


class AvatarPersonalityTestChatResponse(BaseModel):
    avatar_id: int
    reply: str
    model: Optional[str] = None
    adapter_path: Optional[str] = None
    snapshot_label: Optional[str] = None


class AvatarPersonalityFitCheckRequest(BaseModel):
    adapter_path: Optional[str] = None
    max_new_tokens: int = 160
    temperature: float = 0.75
    top_p: float = 0.9


class AvatarPersonalityFitCheckPromptResultRead(BaseModel):
    key: str
    prompt: str
    reply: str


class AvatarPersonalityFitCheckResponse(BaseModel):
    avatar_id: int
    model: Optional[str] = None
    judge_model: Optional[str] = None
    adapter_path: Optional[str] = None
    snapshot_label: Optional[str] = None
    classification: Literal["underfit", "balanced", "overfit", "unclear"] = "unclear"
    confidence: int = 0
    summary: str = ""
    strengths: List[str] = []
    concerns: List[str] = []
    recommendations: List[str] = []
    results: List[AvatarPersonalityFitCheckPromptResultRead] = []


class AvatarPersonalitySnapshotSelectRequest(BaseModel):
    adapter_path: str


class AvatarPersonalitySnapshotCleanupRequest(BaseModel):
    keep_adapter_path: str


class AvatarPersonalitySnapshotDeleteRequest(BaseModel):
    adapter_path: str


class AvatarPersonalityBaseModelCandidateRead(BaseModel):
    model_id: str
    label: str
    recommended: bool = False
    installed: bool = False


class AvatarPersonalityBaseModelSupportRead(BaseModel):
    selected_model_id: str
    recommended_model_id: Optional[str] = None
    installed: bool = False
    local_path: Optional[str] = None
    memory_optimized_available: bool = False
    memory_optimized_reason: Optional[str] = None
    downloading: bool = False
    download_status: Literal["idle", "running", "completed", "failed"] = "idle"
    download_message: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    rationale: Optional[str] = None
    candidates: List[AvatarPersonalityBaseModelCandidateRead] = []


class AvatarPersonalityBaseModelDownloadRequest(BaseModel):
    model_id: Optional[str] = None


# ── Jobs ─────────────────────────────────────────────────────────────────────

class JobRead(JobBase):
    id: int
    video: Optional[Video] = None


class PipelineFocusRead(BaseModel):
    mode: Literal["transcribe", "diarize"]
    execution_mode: Literal["sequential", "staged"] = "sequential"
    auto_diarize_ready: bool
    transcribe_active: int
    transcribe_queued: int
    diarize_active: int
    diarize_queued: int
    active_transcription_paused: int = 0
    diarize_auto_start_threshold: int = 0


class PipelineFocusUpdate(BaseModel):
    mode: Literal["transcribe", "diarize"]
    pause_active_transcription: bool = False


# ── Settings ─────────────────────────────────────────────────────────────────

class Settings(BaseModel):
    hf_token: str
    transcription_engine: str = "auto"  # auto | whisper | parakeet
    pipeline_execution_mode: Literal["sequential", "staged"] = "sequential"
    whisper_backend: str = "faster_whisper"
    transcription_model: str = "medium"
    transcription_compute_type: str = "int8_float16"
    multilingual_routing_enabled: bool = True
    multilingual_whisper_model: str = "large-v3"
    language_detection_sample_seconds: int = 45
    language_detection_confidence_threshold: float = 0.65
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
    youtube_data_api_key: str = ""
    youtube_oauth_client_id: str = ""
    youtube_oauth_client_secret: str = ""
    youtube_oauth_redirect_uri: str = "http://localhost:8000/auth/youtube/callback"
    youtube_publish_push_enabled: bool = False
    ytdlp_cookies_file: str = ""
    ytdlp_cookies_from_browser: str = ""
    diarization_sensitivity: str = "balanced"
    speaker_match_threshold: float = 0.35
    diarize_auto_start_threshold: int = 0
    funny_moments_max_saved: int = 25
    funny_moments_explain_batch_limit: int = 12
    setup_wizard_completed: bool = False


class OllamaPullRequest(BaseModel):
    url: Optional[str] = None
    model: Optional[str] = None
    wait_for_completion: bool = False


class TranscriptionEngineTestRequest(BaseModel):
    engine: Optional[str] = None
    whisper_backend: Optional[str] = None


class YouTubeDataApiTestRequest(BaseModel):
    api_key: Optional[str] = None


class YouTubeDataApiTestResult(BaseModel):
    status: str
    video_id: Optional[str] = None
    title: Optional[str] = None
    view_count: Optional[int] = None
    error: Optional[str] = None


class VoiceFixerUseCleanedRequest(BaseModel):
    enabled: bool


class ReconstructionUseForPlaybackRequest(BaseModel):
    enabled: bool


class UploadedPlaybackSourceRequest(BaseModel):
    source: str


class ReconstructionSettingsUpdateRequest(BaseModel):
    mode: str = "performance"
    instruction_template: Optional[str] = None


class ReconstructionWorkbenchSampleRead(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    text: str
    audio_url: Optional[str] = None
    cleaned_audio_url: Optional[str] = None
    rejected: bool = False
    selected: bool = False


class ReconstructionWorkbenchSegmentRead(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    text: str


class ReconstructionWorkbenchSpeakerRead(BaseModel):
    speaker_id: int
    speaker_name: str
    segment_count: int
    approved: bool = False
    selected_sample_segment_id: Optional[int] = None
    reference_text: Optional[str] = None
    reference_start_time: Optional[float] = None
    reference_end_time: Optional[float] = None
    reference_audio_url: Optional[str] = None
    samples: List[ReconstructionWorkbenchSampleRead] = []
    latest_test_audio_url: Optional[str] = None
    latest_test_text: Optional[str] = None
    latest_test_mode: Optional[str] = None
    can_add_sample: bool = False


class ReconstructionWorkbenchRead(BaseModel):
    mode: str = "performance"
    instruction_template: Optional[str] = None
    performance_supported: bool = True
    speaker_count: int = 0
    all_speakers_approved: bool = False
    speakers: List[ReconstructionWorkbenchSpeakerRead] = []


class ReconstructionSpeakerTestRequest(BaseModel):
    speaker_id: int
    segment_id: Optional[int] = None
    text: Optional[str] = None
    performance_mode: bool = True


class ReconstructionSpeakerTestResult(BaseModel):
    speaker_id: int
    mode: str
    segment_id: Optional[int] = None
    text: str
    audio_url: str
    detail: Optional[str] = None


class ReconstructionWorkbenchSampleCleanupRequest(BaseModel):
    speaker_id: int
    segment_id: int


class ReconstructionWorkbenchSampleStateRequest(BaseModel):
    speaker_id: int
    segment_id: int
    rejected: Optional[bool] = None
    selected: Optional[bool] = None
    clear_cleaned: Optional[bool] = None


class ReconstructionWorkbenchAddSampleRequest(BaseModel):
    speaker_id: int


class ReconstructionWorkbenchSpeakerApprovalRequest(BaseModel):
    speaker_id: int
    approved: bool


class ReconstructionSegmentPreviewRequest(BaseModel):
    segment_id: int
    performance_mode: Optional[bool] = True


class ReconstructionSegmentPreviewResult(BaseModel):
    segment_id: int
    speaker_id: Optional[int] = None
    mode: str
    text: str
    audio_url: str
    detail: Optional[str] = None


class WorkbenchTaskProgressRead(BaseModel):
    video_id: int
    area: Optional[str] = None
    task: Optional[str] = None
    status: str = "idle"
    stage: Optional[str] = None
    message: Optional[str] = None
    percent: Optional[int] = None
    current: Optional[int] = None
    total: Optional[int] = None


class CleanupWorkbenchAudioAnalysisRead(BaseModel):
    source_label: str = "Original upload"
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    peak: Optional[float] = None
    rms: Optional[float] = None
    clipped_ratio: Optional[float] = None


class CleanupWorkbenchCandidateRead(BaseModel):
    candidate_id: str
    stage: str = "enhancement"
    task: str = "speech_enhancement"
    model_name: str
    source_candidate_id: Optional[str] = None
    source_label: Optional[str] = None
    selected_for_processing: bool = False
    audio_url: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    peak: Optional[float] = None
    rms: Optional[float] = None


class CleanupWorkbenchRead(BaseModel):
    selected_candidate_id: Optional[str] = None
    selected_source_label: str = "Original upload"
    analysis: Optional[CleanupWorkbenchAudioAnalysisRead] = None
    candidates: List[CleanupWorkbenchCandidateRead] = []
    clearvoice_available: bool = False


class CleanupWorkbenchRunCandidateRequest(BaseModel):
    model_name: str
    stage: str = "enhancement"
    source_candidate_id: Optional[str] = None


class CleanupWorkbenchSelectCandidateRequest(BaseModel):
    candidate_id: Optional[str] = None


class VoiceFixerSettingsUpdateRequest(BaseModel):
    mode: int
    mix_ratio: float
    leveling_mode: str = "off"
    apply_scope: str = "none"


class ClearVoiceInstallInfo(BaseModel):
    installed: bool = False
    version: Optional[str] = None
    python_executable: Optional[str] = None
    package_name: str = "clearvoice"
    package_spec: str = "clearvoice==0.1.2"
    github_url: str = "https://github.com/modelscope/ClearerVoice-Studio"
    pypi_url: str = "https://pypi.org/project/clearvoice/"
    restart_required: bool = False
    runtime_ready: bool = False
    runtime_error: Optional[str] = None
    torch_version: Optional[str] = None
    torchaudio_version: Optional[str] = None
    message: Optional[str] = None


class ClearVoiceTestResult(BaseModel):
    status: str
    version: Optional[str] = None
    imported: bool = False
    class_available: bool = False
    torch_imported: bool = False
    torchaudio_imported: bool = False
    runtime_ready: bool = False
    torch_version: Optional[str] = None
    torchaudio_version: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None


class VoiceFixerInstallInfo(BaseModel):
    installed: bool = False
    version: Optional[str] = None
    python_executable: Optional[str] = None
    package_name: str = "voicefixer"
    package_spec: str = "voicefixer==0.1.3"
    github_url: str = "https://github.com/haoheliu/voicefixer"
    pypi_url: str = "https://pypi.org/project/voicefixer/"
    restart_required: bool = False
    message: Optional[str] = None


class VoiceFixerTestResult(BaseModel):
    status: str
    version: Optional[str] = None
    instantiated: bool = False
    error: Optional[str] = None
    detail: Optional[str] = None


class ReconstructionInstallInfo(BaseModel):
    installed: bool = False
    version: Optional[str] = None
    python_executable: Optional[str] = None
    package_name: str = "qwen-tts"
    package_spec: str = "qwen-tts==0.1.1"
    github_url: str = "https://github.com/QwenLM/Qwen3-TTS"
    pypi_url: str = "https://pypi.org/project/qwen-tts/"
    restart_required: bool = False
    message: Optional[str] = None


class ReconstructionTestResult(BaseModel):
    status: str
    version: Optional[str] = None
    imported: bool = False
    model_class_available: bool = False
    error: Optional[str] = None
    detail: Optional[str] = None


class ExternalShareStartRequest(BaseModel):
    enable_tunnel: bool = True
    frontend_port: int = 5173
    backend_port: int = 8011
    duration_minutes: int = 60
    password: str = ""
    ip_allowlist: str = ""


# ── Semantic Search ───────────────────────────────────────────────────────────

class ContextChunk(BaseModel):
    chunk_id: int
    speaker_id: Optional[int] = None
    start_time: float
    end_time: float
    chunk_text: str


class SemanticSearchRequest(BaseModel):
    query: str
    channel_id: Optional[int] = None
    video_id: Optional[int] = None
    speaker_id: Optional[int] = None
    year: Optional[int] = None
    month: Optional[int] = None
    mode: Literal["semantic", "hybrid"] = "semantic"
    limit: int = 20
    offset: int = 0


class SemanticSearchHit(BaseModel):
    id: int
    score: float
    video_id: int
    video_title: Optional[str] = None
    channel_id: int
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    start_time: float
    end_time: float
    chunk_text: str
    segment_ids: List[int] = []
    context_before: List[ContextChunk] = []
    context_after: List[ContextChunk] = []


class SemanticSearchPage(BaseModel):
    items: List[SemanticSearchHit]
    total: int
    limit: int
    offset: int


class SemanticIndexStatus(BaseModel):
    is_running: bool
    current_video_id: Optional[int] = None
    current_video_title: Optional[str] = None
    videos_completed: int = 0
    videos_total: int = 0
    started_at: Optional[str] = None
    last_finished_at: Optional[str] = None


class SemanticIndexRebuildResponse(BaseModel):
    started: bool
    message: str
    video_ids: List[int] = []


# Episode Clone

class EpisodeCloneCandidateRead(BaseModel):
    video_id: int
    title: str
    published_at: Optional[datetime] = None
    duration: Optional[int] = None
    view_count: Optional[int] = None
    age_days: Optional[float] = None
    views_per_day: Optional[float] = None
    transcript_segment_count: int = 0
    semantic_hit_count: int = 0
    semantic_score: Optional[float] = None


class EpisodeCloneContextHitRead(BaseModel):
    chunk_id: int
    score: float
    video_id: int
    video_title: Optional[str] = None
    speaker_name: Optional[str] = None
    start_time: float
    end_time: float
    chunk_text: str


class EpisodeCloneEngineRead(BaseModel):
    key: str
    label: str
    provider: str
    model: str
    is_default: bool = False
    available: bool = True
    disabled_reason: Optional[str] = None


class EpisodeCloneConceptsRequest(BaseModel):
    notes: Optional[str] = Field(default=None, max_length=2000)
    semantic_query: Optional[str] = Field(default=None, max_length=1200)
    related_limit: int = Field(default=8, ge=1, le=12)
    provider_override: Optional[str] = Field(default=None, max_length=40)
    model_override: Optional[str] = Field(default=None, max_length=160)


class EpisodeCloneConceptsResponse(BaseModel):
    video_id: int
    channel_id: int
    source_title: str
    source_metrics: EpisodeCloneCandidateRead
    semantic_query: str
    source_brief: str
    concepts: List[str] = []
    excluded_references: List[str] = []
    related_videos: List[EpisodeCloneCandidateRead] = []
    warnings: List[str] = []
    model: str = ""


class EpisodeCloneGenerateRequest(BaseModel):
    style_prompt: str = Field(min_length=1, max_length=4000)
    notes: Optional[str] = Field(default=None, max_length=2000)
    semantic_query: Optional[str] = Field(default=None, max_length=1200)
    related_limit: int = Field(default=8, ge=1, le=12)
    variant_label: Optional[str] = Field(default=None, max_length=120)
    provider_override: Optional[str] = Field(default=None, max_length=40)
    model_override: Optional[str] = Field(default=None, max_length=160)
    approved_concepts: List[str] = Field(default_factory=list, max_length=18)
    excluded_references: List[str] = Field(default_factory=list, max_length=24)


class EpisodeCloneGenerateResponse(BaseModel):
    video_id: int
    channel_id: int
    source_title: str
    source_metrics: EpisodeCloneCandidateRead
    style_prompt: str
    notes: Optional[str] = None
    semantic_query: str
    source_brief: str
    approved_concepts: List[str] = []
    excluded_references: List[str] = []
    related_videos: List[EpisodeCloneCandidateRead] = []
    context_hits: List[EpisodeCloneContextHitRead] = []
    suggested_title: str = ""
    opening_hook: str = ""
    angle_summary: str = ""
    originality_notes: List[str] = []
    script: str = ""
    warnings: List[str] = []
    model: str = ""


class EpisodeCloneJobRead(BaseModel):
    job_id: int
    video_id: int
    status: str
    progress: int = 0
    status_detail: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: EpisodeCloneGenerateRequest
    request_signature: str
    result: Optional[EpisodeCloneGenerateResponse] = None


class ExternalShareAuditEntry(BaseModel):
    at: str
    action: str
    allowed: bool
    reason: Optional[str] = None
    client_ip: Optional[str] = None
    path: Optional[str] = None


class ExternalShareStatus(BaseModel):
    active: bool = False
    mode: str = "off"
    enable_tunnel: bool = False
    tunnel_provider: Optional[str] = None
    started_at: Optional[str] = None
    expires_at: Optional[str] = None
    frontend_local_url: Optional[str] = None
    api_local_url: Optional[str] = None
    frontend_lan_url: Optional[str] = None
    api_lan_url: Optional[str] = None
    frontend_public_url: Optional[str] = None
    api_public_url: Optional[str] = None
    share_url: Optional[str] = None
    token_required: bool = True
    password_required: bool = False
    ip_allowlist: List[str] = []
    cloudflared_available: bool = False
    audit_log_path: Optional[str] = None
    audit_entries: List[ExternalShareAuditEntry] = []
