export interface Channel {
    id: number;
    url: string;
    name: string;
    source_type?: 'youtube' | 'manual' | 'tiktok' | string;
    icon_url?: string;
    header_image_url?: string;
    last_updated: string;
    status: string;
    actively_monitored?: boolean;
    sync_status_detail?: string;
    sync_progress?: number;
    sync_total_items?: number;
    sync_completed_items?: number;
}

export interface Video {
    id: number;
    channel_id: number;
    youtube_id: string;
    title: string;
    media_source_type?: 'youtube' | 'upload' | 'tiktok' | string;
    source_url?: string;
    media_kind?: 'audio' | 'video' | string;
    manual_media_path?: string;
    voicefixer_cleaned_path?: string;
    voicefixer_use_cleaned?: boolean;
    voicefixer_apply_scope?: 'none' | 'playback' | 'processing' | 'both' | string;
    voicefixer_mode?: number;
    voicefixer_mix_ratio?: number;
    voicefixer_leveling_mode?: 'off' | 'gentle' | 'balanced' | 'strong' | string;
    voicefixer_status?: string;
    voicefixer_error?: string;
    reconstruction_audio_path?: string;
    reconstruction_use_for_playback?: boolean;
    reconstruction_mode?: 'basic' | 'performance' | string;
    reconstruction_instruction_template?: string;
    reconstruction_status?: string;
    reconstruction_error?: string;
    reconstruction_model?: string;
    published_at?: string; // May be null if not available
    description?: string;
    thumbnail_url?: string;
    duration?: number; // seconds
    view_count?: number;
    processed: boolean;
    muted: boolean;
    access_restricted?: boolean;
    access_restriction_reason?: string;
    status: string;
    humor_context_summary?: string;
    humor_context_model?: string;
    humor_context_generated_at?: string;
    youtube_ai_summary?: string;
    youtube_ai_chapters_json?: string;
    youtube_ai_description_text?: string;
    youtube_ai_model?: string;
    youtube_ai_generated_at?: string;
    transcript_source?: string;
    transcript_language?: string;
    transcript_is_placeholder?: boolean;
    last_pipeline_job_status?: string;
    last_pipeline_job_type?: string;
}

export interface VideoChapterSuggestion {
    start_seconds: number;
    timestamp: string;
    title: string;
    description?: string;
}

export interface ReconstructionWorkbenchSegment {
    segment_id: number;
    start_time: number;
    end_time: number;
    duration: number;
    text: string;
}

export interface ReconstructionWorkbenchSample {
    segment_id: number;
    start_time: number;
    end_time: number;
    duration: number;
    text: string;
    audio_url?: string;
    cleaned_audio_url?: string;
    rejected: boolean;
    selected: boolean;
}

export interface ReconstructionWorkbenchSpeaker {
    speaker_id: number;
    speaker_name: string;
    segment_count: number;
    approved: boolean;
    selected_sample_segment_id?: number;
    reference_text?: string;
    reference_start_time?: number;
    reference_end_time?: number;
    reference_audio_url?: string;
    samples: ReconstructionWorkbenchSample[];
    latest_test_audio_url?: string;
    latest_test_text?: string;
    latest_test_mode?: string;
    can_add_sample: boolean;
}

export interface ReconstructionWorkbench {
    mode: 'basic' | 'performance' | string;
    instruction_template?: string;
    performance_supported: boolean;
    speaker_count: number;
    all_speakers_approved: boolean;
    speakers: ReconstructionWorkbenchSpeaker[];
}

export interface WorkbenchTaskProgress {
    video_id: number;
    area?: 'cleanup' | 'reconstruction' | string;
    task?: string;
    status: 'idle' | 'running' | 'completed' | 'error' | string;
    stage?: string | null;
    message?: string | null;
    percent?: number | null;
    current?: number | null;
    total?: number | null;
}

export interface CleanupWorkbenchAudioAnalysis {
    source_label: string;
    duration_seconds?: number | null;
    sample_rate?: number | null;
    channels?: number | null;
    peak?: number | null;
    rms?: number | null;
    clipped_ratio?: number | null;
}

export interface CleanupWorkbenchCandidate {
    candidate_id: string;
    stage: string;
    task: string;
    model_name: string;
    source_candidate_id?: string | null;
    source_label?: string | null;
    selected_for_processing: boolean;
    audio_url?: string | null;
    sample_rate?: number | null;
    duration_seconds?: number | null;
    peak?: number | null;
    rms?: number | null;
}

export interface CleanupWorkbench {
    selected_candidate_id?: string | null;
    selected_source_label: string;
    analysis?: CleanupWorkbenchAudioAnalysis | null;
    candidates: CleanupWorkbenchCandidate[];
    clearvoice_available: boolean;
}

export interface ClearVoiceInstallInfo {
    installed: boolean;
    version?: string | null;
    python_executable?: string | null;
    package_name: string;
    package_spec: string;
    github_url: string;
    pypi_url: string;
    restart_required: boolean;
    runtime_ready: boolean;
    runtime_error?: string | null;
    torch_version?: string | null;
    torchaudio_version?: string | null;
    message?: string | null;
}

export interface ClearVoiceTestResult {
    status: string;
    version?: string | null;
    imported: boolean;
    class_available: boolean;
    torch_imported: boolean;
    torchaudio_imported: boolean;
    runtime_ready: boolean;
    torch_version?: string | null;
    torchaudio_version?: string | null;
    error?: string | null;
    detail?: string | null;
}

export interface VideoDescriptionRevision {
    id: number;
    video_id: number;
    description_text: string;
    source: string;
    ai_model?: string;
    note?: string;
    created_at: string;
}

export interface TranscriptQuality {
    video_id: number;
    title: string;
    channel_id?: number;
    quality_profile: string;
    recommended_tier: 'low_risk_repair' | 'diarization_rebuild' | 'full_retranscription' | 'manual_review' | 'none' | string;
    quality_score: number;
    eligible_for_optimization: boolean;
    language?: string | null;
    metrics: Record<string, number | string | null>;
    reasons: string[];
    created_snapshot_id?: number | null;
    snapshot_created_at?: string | null;
}

export interface TranscriptRollbackOption {
    run_id: number;
    video_id: number;
    mode: string;
    pipeline_version: string;
    note?: string | null;
    created_at: string;
    rollback_available: boolean;
    rollback_state?: string | null;
}

export interface TranscriptRestoreResponse {
    video_id: number;
    restored_from_run_id: number;
    restore_run_id: number;
    segment_count: number;
    funny_moment_count: number;
    quality_profile: string;
    recommended_tier: string;
    quality_score: number;
}

export interface TranscriptOptimizeDryRunResponse {
    total_scanned: number;
    total_eligible: number;
    items: TranscriptQuality[];
}

export interface TranscriptGoldWindow {
    id: number;
    video_id: number;
    label: string;
    quality_profile?: string | null;
    language?: string | null;
    start_time: number;
    end_time: number;
    reference_text: string;
    entities: string[];
    notes?: string | null;
    active: boolean;
    created_at: string;
    updated_at: string;
}

export interface TranscriptEvaluationResult {
    id: number;
    gold_window_id: number;
    video_id: number;
    run_id?: number | null;
    source: string;
    candidate_text: string;
    reference_text: string;
    wer: number;
    cer: number;
    entity_accuracy?: number | null;
    matched_entity_count: number;
    total_entity_count: number;
    segment_count: number;
    unknown_speaker_rate: number;
    punctuation_density_delta: number;
    metrics: Record<string, number | string | null>;
    created_at: string;
}

export interface TranscriptEvaluationReview {
    id: number;
    evaluation_result_id: number;
    reviewer?: string | null;
    verdict: string;
    tags: string[];
    notes?: string | null;
    created_at: string;
}

export interface TranscriptEvaluationBatchResponse {
    video_id: number;
    run_id?: number | null;
    total_windows: number;
    average_wer: number;
    average_cer: number;
    average_entity_accuracy?: number | null;
    average_unknown_speaker_rate: number;
    items: TranscriptEvaluationResult[];
}

export interface TranscriptEvaluationSummary {
    scope: string;
    channel_id?: number | null;
    total_gold_windows: number;
    total_results: number;
    total_reviewed_results: number;
    average_wer?: number | null;
    average_cer?: number | null;
    average_entity_accuracy?: number | null;
    average_unknown_speaker_rate?: number | null;
    verdict_counts: Record<string, number>;
    latest_result_at?: string | null;
}

export interface TranscriptDiarizationConfigBenchmark {
    label: string;
    run_count: number;
    average_wer?: number | null;
    average_cer?: number | null;
    average_unknown_speaker_rate?: number | null;
    latest_run_id?: number | null;
    latest_created_at?: string | null;
    diarization_sensitivity?: string | null;
    speaker_match_threshold?: number | null;
}

export interface TranscriptRepairQueueResponse {
    job_id: number;
    video_id: number;
    status: string;
    recommended_tier: string;
    quality_score: number;
    queued: boolean;
}

export interface TranscriptRepairBulkQueueResponse {
    channel_id: number;
    queued: number;
    skipped_active: number;
    skipped_no_segments: number;
    skipped_not_low_risk: number;
    jobs: TranscriptRepairQueueResponse[];
}

export interface TranscriptDiarizationRebuildQueueResponse {
    job_id: number;
    video_id: number;
    status: string;
    recommended_tier: string;
    quality_score: number;
    queued: boolean;
}

export interface TranscriptDiarizationRebuildBulkQueueResponse {
    channel_id: number;
    queued: number;
    skipped_active: number;
    skipped_no_raw_transcript: number;
    skipped_not_diarization_rebuild: number;
    skipped_unprocessed: number;
    skipped_muted: number;
    jobs: TranscriptDiarizationRebuildQueueResponse[];
}

export interface TranscriptRetranscriptionQueueResponse {
    job_id: number;
    video_id: number;
    status: string;
    recommended_tier: string;
    quality_score: number;
    queued: boolean;
}

export interface TranscriptRetranscriptionBulkQueueResponse {
    channel_id: number;
    queued: number;
    skipped_active: number;
    skipped_not_full_retranscription: number;
    skipped_unprocessed: number;
    skipped_muted: number;
    jobs: TranscriptRetranscriptionQueueResponse[];
}

export interface TranscriptOptimizationCampaign {
    id: number;
    channel_id?: number | null;
    scope: string;
    status: string;
    tiers: string[];
    limit: number;
    force_non_eligible: boolean;
    queued_jobs: number;
    skipped_active: number;
    skipped_no_segments: number;
    skipped_not_eligible: number;
    skipped_other: number;
    note?: string | null;
    created_at: string;
    updated_at: string;
}

export interface TranscriptOptimizationCampaignItem {
    id: number;
    campaign_id: number;
    video_id: number;
    recommended_tier: string;
    action_tier: string;
    quality_score: number;
    reason?: string | null;
    status: string;
    job_id?: number | null;
    created_at: string;
    updated_at: string;
}

export interface TranscriptOptimizationCampaignExecuteResponse {
    campaign_id: number;
    status: string;
    queued_jobs: number;
    skipped_active: number;
    skipped_no_segments: number;
    skipped_not_eligible: number;
    skipped_other: number;
}

export interface TranscriptOptimizationCampaignDeleteResponse {
    campaign_id: number;
    deleted_items: number;
    detached_job_refs: number;
    status: string;
}

export interface Speaker {
    id: number;
    channel_id: number;
    name: string;
    thumbnail_path?: string;
    is_extra: boolean;
    total_speaking_time: number;
    embedding_count?: number;
    created_at: string;
}

export interface SpeakerSample {
    id?: number;
    video_id: number;
    start_time: number;
    end_time?: number;
    text: string;
    channel_id?: number;
    youtube_id?: string;
    media_source_type?: 'youtube' | 'upload' | 'tiktok' | string;
    media_kind?: 'audio' | 'video' | string;
}

export interface SpeakerCounts {
    total: number;
    identified: number;
    unknown: number;
    main: number;
    extras: number;
}

export interface SpeakerOverview {
    items: Speaker[];
    counts: SpeakerCounts;
    total: number;
    offset: number;
    limit?: number | null;
}

export interface SpeakerEpisodeAppearance {
    video_id: number;
    youtube_id: string;
    title: string;
    media_source_type?: 'youtube' | 'upload' | 'tiktok' | string;
    media_kind?: 'audio' | 'video' | string;
    published_at?: string;
    thumbnail_url?: string;
    segment_count: number;
    total_speaking_time: number;
    first_start_time: number;
    last_end_time: number;
}

export interface SpeakerVoiceProfile {
    id: number;
    speaker_id: number;
    source_video_id?: number;
    source_video_title?: string;
    source_video_youtube_id?: string;
    source_video_media_source_type?: 'youtube' | 'upload' | 'tiktok' | string;
    source_video_media_kind?: 'audio' | 'video' | string;
    source_video_published_at?: string;
    sample_start_time?: number;
    sample_end_time?: number;
    sample_text?: string;
    created_at: string;
}

export interface Avatar {
    id: number;
    channel_id: number;
    speaker_id: number;
    name: string;
    status: string;
    description?: string;
    created_at: string;
    updated_at: string;
}

export interface AvatarSectionSummary {
    status: string;
    source_count: number;
    approved_count: number;
    artifact_ready: boolean;
    summary?: string;
    artifact_path?: string;
    last_built_at?: string;
}

export interface AvatarWorkbenchSpeaker {
    id: number;
    channel_id: number;
    name: string;
    thumbnail_path?: string;
    total_speaking_time: number;
    embedding_count: number;
    appearance_count: number;
}

export interface AvatarWorkbench {
    avatar: Avatar;
    speaker: AvatarWorkbenchSpeaker;
    personality: AvatarSectionSummary;
    appearance: AvatarSectionSummary;
    voice: AvatarSectionSummary;
    runtime_status: string;
    suggested_base_model?: string;
    artifacts_dir?: string;
}

export interface AvatarPersonalityDatasetExample {
    example_id: number;
    video_id: number;
    video_title: string;
    start_time: number;
    end_time: number;
    context_text: string;
    response_text: string;
    source_segment_ids: number[];
    source_segment_count: number;
    context_turns: number;
    response_word_count: number;
    context_word_count: number;
    quality_score: number;
    completion_score: number;
    context_score: number;
    style_score: number;
    substance_score: number;
    cluster_id?: number | null;
    cluster_size: number;
    diversity_score: number;
    duplicate_group_id?: number | null;
    duplicate_group_size: number;
    duplicate_similarity: number;
    heuristic_label?: 'gold' | 'silver' | 'reject' | null;
    llm_label?: 'gold' | 'silver' | 'reject' | null;
    llm_confidence: number;
    llm_completion_score: number;
    llm_style_score: number;
    llm_usefulness_score: number;
    llm_rationale?: string | null;
    llm_reasons: string[];
    llm_model?: string | null;
    llm_judged_at?: string | null;
    auto_label: 'gold' | 'silver' | 'reject';
    manual_state?: 'approved' | 'rejected' | null;
    state: 'approved' | 'rejected';
    reject_reasons: string[];
}

export interface AvatarPersonalityTrainingReadiness {
    status: 'insufficient' | 'borderline' | 'ready' | 'strong' | 'oversized';
    score: number;
    can_train_now: boolean;
    approved_duration_hours: number;
    approved_word_count: number;
    recommended_action?: string | null;
    summary?: string | null;
    caution?: string | null;
    manual_review_roi: 'high' | 'medium' | 'low';
    duplicate_pressure: number;
    hotspot_pressure: number;
    gold_ratio: number;
    reject_ratio: number;
}

export interface AvatarPersonalityDataset {
    avatar_id: number;
    speaker_id: number;
    status: string;
    system_prompt?: string;
    base_model_id?: string;
    dataset_path?: string;
    metadata_path?: string;
    example_count: number;
    gold_example_count: number;
    silver_example_count: number;
    auto_reject_count: number;
    needs_review_count: number;
    duplicate_example_count: number;
    cluster_count: number;
    hotspot_cluster_count: number;
    llm_judged_count: number;
    llm_promoted_count: number;
    llm_rejected_count: number;
    source_turn_count: number;
    discarded_turn_count: number;
    readiness: AvatarPersonalityTrainingReadiness;
    preview_examples: AvatarPersonalityDatasetExample[];
    generated_at?: string;
}

export interface AvatarPersonalityDatasetPage {
    avatar_id: number;
    total: number;
    approved_count: number;
    rejected_count: number;
    gold_count: number;
    silver_count: number;
    auto_reject_count: number;
    needs_review_count: number;
    duplicate_count: number;
    cluster_count: number;
    hotspot_cluster_count: number;
    llm_judged_count: number;
    llm_promoted_count: number;
    llm_rejected_count: number;
    limit: number;
    offset: number;
    has_more: boolean;
    state_filter: 'all' | 'approved' | 'rejected' | 'gold' | 'silver' | 'auto_reject' | 'needs_review' | 'duplicate_risk';
    items: AvatarPersonalityDatasetExample[];
}

export interface AvatarPersonalityJudgeFeedItem {
    example_id: number;
    video_title: string;
    llm_label: 'gold' | 'silver' | 'reject';
    llm_confidence: number;
    llm_reasons: string[];
    heuristic_label?: 'gold' | 'silver' | 'reject' | null;
    judged_at?: string | null;
}

export interface AvatarPersonalityJudgeStatus {
    avatar_id: number;
    status: 'idle' | 'running' | 'stopping' | 'completed' | 'stopped' | 'failed';
    active: boolean;
    stop_requested: boolean;
    model?: string | null;
    target_filter: 'needs_review' | 'silver' | 'all';
    overwrite_existing: boolean;
    max_examples: number;
    total_candidates: number;
    processed_count: number;
    judged_count: number;
    promoted_count: number;
    rejected_count: number;
    current_example_id?: number | null;
    current_video_title?: string | null;
    current_stage?: string | null;
    started_at?: string | null;
    updated_at?: string | null;
    finished_at?: string | null;
    error?: string | null;
    recent_results: AvatarPersonalityJudgeFeedItem[];
}

export interface AvatarPersonalityLongFormSample {
    sample_id: string;
    video_id: number;
    video_title: string;
    start_time: number;
    end_time: number;
    duration_seconds: number;
    word_count: number;
    segment_count: number;
    text: string;
    style_density: number;
    substance_density: number;
    state: 'included' | 'rejected';
}

export interface AvatarPersonalityLongFormConfig {
    take_count: number;
    included_count: number;
    rejected_count: number;
    selected_count: number;
}

export interface AvatarPersonalityLongFormPage {
    avatar_id: number;
    total: number;
    included_count: number;
    rejected_count: number;
    selected_count: number;
    take_count: number;
    limit: number;
    offset: number;
    has_more: boolean;
    items: AvatarPersonalityLongFormSample[];
}

export interface AvatarPersonalityTrainingConfig {
    base_model_id: string;
    dataset_profile: 'focused' | 'balanced' | 'broad' | 'exhaustive' | 'custom';
    training_strength: 'conservative' | 'balanced' | 'strong';
    export_strategy: 'gold_only' | 'gold_balanced' | 'gold_plus_top_silver' | 'full_approved';
    validation_ratio: number;
    max_examples: number;
    max_long_form_examples: number;
    include_long_form: boolean;
    training_mode: 'standard' | 'memory_optimized';
    snapshot_interval_steps: number;
    dataset_profiles: AvatarPersonalityTrainingDatasetProfile[];
    training_plan?: AvatarPersonalityTrainingPlan | null;
}

export interface AvatarPersonalityTrainingDatasetProfile {
    key: 'focused' | 'balanced' | 'broad' | 'exhaustive' | 'custom';
    label: string;
    summary: string;
    conversation_target: number;
    long_form_target: number;
    pros: string[];
    cons: string[];
    recommended: boolean;
}

export interface AvatarPersonalityTrainingPlan {
    dataset_profile: 'focused' | 'balanced' | 'broad' | 'exhaustive' | 'custom';
    dataset_profile_label: string;
    conversation_target: number;
    long_form_target: number;
    available_conversation_examples: number;
    available_long_form_examples: number;
    estimated_conversation_examples: number;
    estimated_long_form_examples: number;
    estimated_total_examples: number;
    estimated_train_examples: number;
    estimated_validation_examples: number;
    estimated_effective_batch_size: number;
    estimated_steps_per_epoch: number;
    estimated_total_steps: number;
    step_band: 'light' | 'ideal' | 'heavy' | 'aggressive';
    headline: string;
    recommendation: string;
    snapshot_interval_suggestion: number;
    pros: string[];
    cons: string[];
}

export interface AvatarPersonalityTrainingPackage {
    avatar_id: number;
    status: 'not_prepared' | 'ready';
    base_model_id?: string | null;
    dataset_profile: 'focused' | 'balanced' | 'broad' | 'exhaustive' | 'custom';
    training_strength: 'conservative' | 'balanced' | 'strong';
    export_strategy: 'gold_only' | 'gold_balanced' | 'gold_plus_top_silver' | 'full_approved';
    validation_ratio: number;
    max_examples: number;
    max_long_form_examples: number;
    include_long_form: boolean;
    conversation_examples_selected: number;
    long_form_examples_selected: number;
    total_examples_selected: number;
    train_examples: number;
    validation_examples: number;
    prompt?: string | null;
    manifest_path?: string | null;
    config_path?: string | null;
    train_dataset_path?: string | null;
    validation_dataset_path?: string | null;
    command_preview?: string | null;
    prepared_at?: string | null;
    training_plan?: AvatarPersonalityTrainingPlan | null;
}

export interface AvatarPersonalityTrainingStatus {
    avatar_id: number;
    status: 'idle' | 'queued' | 'running' | 'stopping' | 'completed' | 'stopped' | 'failed';
    active: boolean;
    stop_requested: boolean;
    base_model_id?: string | null;
    training_mode: 'standard' | 'memory_optimized';
    adapter_path?: string | null;
    output_dir?: string | null;
    current_stage?: string | null;
    epoch: number;
    step: number;
    max_steps: number;
    snapshot_interval_steps: number;
    train_examples: number;
    validation_examples: number;
    latest_loss?: number | null;
    message?: string | null;
    snapshots: AvatarPersonalityTrainingSnapshot[];
    started_at?: string | null;
    updated_at?: string | null;
    finished_at?: string | null;
    error?: string | null;
}

export interface AvatarPersonalityTrainingSnapshot {
    label: string;
    kind: 'step' | 'epoch' | 'final';
    adapter_path: string;
    step: number;
    epoch: number;
    train_loss?: number | null;
    eval_loss?: number | null;
    created_at?: string | null;
    selected: boolean;
}

export interface AvatarPersonalityTestChatTurn {
    role: 'user' | 'assistant';
    content: string;
}

export interface AvatarPersonalityTestChatResponse {
    avatar_id: number;
    reply: string;
    model?: string | null;
    adapter_path?: string | null;
    snapshot_label?: string | null;
}

export interface AvatarPersonalityFitCheckPromptResult {
    key: string;
    prompt: string;
    reply: string;
}

export interface AvatarPersonalityFitCheckResponse {
    avatar_id: number;
    model?: string | null;
    judge_model?: string | null;
    adapter_path?: string | null;
    snapshot_label?: string | null;
    classification: 'underfit' | 'balanced' | 'overfit' | 'unclear';
    confidence: number;
    summary: string;
    strengths: string[];
    concerns: string[];
    recommendations: string[];
    results: AvatarPersonalityFitCheckPromptResult[];
}

export interface AvatarPersonalityBaseModelCandidate {
    model_id: string;
    label: string;
    recommended: boolean;
    installed: boolean;
}

export interface AvatarPersonalityBaseModelSupport {
    selected_model_id: string;
    recommended_model_id?: string | null;
    installed: boolean;
    local_path?: string | null;
    memory_optimized_available: boolean;
    memory_optimized_reason?: string | null;
    downloading: boolean;
    download_status: 'idle' | 'running' | 'completed' | 'failed';
    download_message?: string | null;
    gpu_name?: string | null;
    gpu_vram_gb?: number | null;
    rationale?: string | null;
    candidates: AvatarPersonalityBaseModelCandidate[];
}

export interface TranscriptSegment {
    id: number;
    video_id: number;
    speaker_id?: number;
    matched_profile_id?: number;
    start_time: number;
    end_time: number;
    text: string;
    words?: string; // JSON string from backend
    speaker?: string; // flattened name if joined
}

export interface TranscriptSegmentRevision {
    id: number;
    segment_id: number;
    video_id: number;
    old_text: string;
    new_text: string;
    source: string;
    created_at: string;
}

export interface Job {
    id: number;
    video_id: number;
    job_type: string;
    status: string; // queued, running, paused, completed, failed
    status_detail?: string; // Fine-grained status (e.g. "Converting audio...", "Loading models...")
    payload_json?: string;
    progress: number;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    error?: string;
    video?: Video;
}

export interface Clip {
    id: number;
    video_id: number;
    start_time: number;
    end_time: number;
    title: string;
    aspect_ratio?: string;
    crop_x?: number;
    crop_y?: number;
    crop_w?: number;
    crop_h?: number;
    portrait_split_enabled?: boolean;
    portrait_top_crop_x?: number;
    portrait_top_crop_y?: number;
    portrait_top_crop_w?: number;
    portrait_top_crop_h?: number;
    portrait_bottom_crop_x?: number;
    portrait_bottom_crop_y?: number;
    portrait_bottom_crop_w?: number;
    portrait_bottom_crop_h?: number;
    script_edits_json?: string;
    fade_in_sec?: number;
    fade_out_sec?: number;
    burn_captions?: boolean;
    caption_speaker_labels?: boolean;
    created_at: string;
}

export interface ChannelClip extends Clip {
    video_title: string;
    video_youtube_id: string;
    video_published_at?: string;
    video_thumbnail_url?: string;
}

export interface ClipExportArtifact {
    id: number;
    clip_id: number;
    video_id: number;
    artifact_type: string;
    format: string;
    file_path: string;
    file_name: string;
    file_size_bytes?: number;
    created_at: string;
}

export interface FunnyMoment {
    id: number;
    video_id: number;
    start_time: number;
    end_time: number;
    score: number;
    source: string;
    snippet?: string;
    humor_summary?: string;
    humor_confidence?: 'low' | 'medium' | 'high' | string;
    humor_model?: string;
    humor_explained_at?: string;
    created_at: string;
}

// ── Semantic Search ───────────────────────────────────────────────────────────

export interface ContextChunk {
    chunk_id: number;
    speaker_id?: number;
    start_time: number;
    end_time: number;
    chunk_text: string;
}

export interface SemanticSearchHit {
    id: number;
    score: number;
    video_id: number;
    video_title?: string;
    channel_id: number;
    speaker_id?: number;
    speaker_name?: string;
    start_time: number;
    end_time: number;
    chunk_text: string;
    segment_ids: number[];
    context_before: ContextChunk[];
    context_after: ContextChunk[];
}

export interface SemanticSearchPage {
    items: SemanticSearchHit[];
    total: number;
    limit: number;
    offset: number;
}

export interface SemanticIndexStatus {
    is_running: boolean;
    current_video_id?: number;
    current_video_title?: string;
    videos_completed: number;
    videos_total: number;
    started_at?: string;
    last_finished_at?: string;
}

export interface SemanticIndexRebuildResponse {
    started: boolean;
    message: string;
    video_ids: number[];
}

export interface EpisodeCloneCandidate {
    video_id: number;
    title: string;
    published_at?: string;
    duration?: number;
    view_count?: number;
    age_days?: number;
    views_per_day?: number;
    transcript_segment_count: number;
    semantic_hit_count: number;
    semantic_score?: number;
}

export interface EpisodeCloneContextHit {
    chunk_id: number;
    score: number;
    video_id: number;
    video_title?: string;
    speaker_name?: string;
    start_time: number;
    end_time: number;
    chunk_text: string;
}

export interface EpisodeCloneEngine {
    key: string;
    label: string;
    provider: string;
    model: string;
    is_default: boolean;
    available: boolean;
    disabled_reason?: string;
}

export interface EpisodeCloneConceptsResponse {
    video_id: number;
    channel_id: number;
    source_title: string;
    source_metrics: EpisodeCloneCandidate;
    semantic_query: string;
    source_brief: string;
    concepts: string[];
    excluded_references: string[];
    related_videos: EpisodeCloneCandidate[];
    warnings: string[];
    model: string;
}

export interface EpisodeCloneGenerateResponse {
    video_id: number;
    channel_id: number;
    source_title: string;
    source_metrics: EpisodeCloneCandidate;
    style_prompt: string;
    notes?: string;
    semantic_query: string;
    source_brief: string;
    approved_concepts: string[];
    excluded_references: string[];
    related_videos: EpisodeCloneCandidate[];
    context_hits: EpisodeCloneContextHit[];
    suggested_title: string;
    opening_hook: string;
    angle_summary: string;
    originality_notes: string[];
    script: string;
    warnings: string[];
    model: string;
}

export interface EpisodeCloneJob {
    job_id: number;
    video_id: number;
    status: string;
    progress: number;
    status_detail?: string;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    error?: string;
    request: {
        style_prompt: string;
        notes?: string;
        semantic_query?: string;
        related_limit: number;
        variant_label?: string;
        provider_override?: string;
        model_override?: string;
        approved_concepts: string[];
        excluded_references: string[];
    };
    request_signature: string;
    result?: EpisodeCloneGenerateResponse | null;
}
