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
    published_at?: string; // May be null if not available
    description?: string;
    thumbnail_url?: string;
    duration?: number; // seconds
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
}

export interface VideoChapterSuggestion {
    start_seconds: number;
    timestamp: string;
    title: string;
    description?: string;
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
