import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api, { toApiUrl } from '../lib/api';
import { useTranscriptStore } from './useTranscriptStore';
import type { TranscriptSegment, Video } from '../types';

type VideoUpdater = Video | null | ((video: Video | null) => Video | null);
type SegmentsUpdater = TranscriptSegment[] | ((segments: TranscriptSegment[]) => TranscriptSegment[]);
export type VideoUploadedPlaybackSource = 'original' | 'cleaned' | 'reconstructed';

export type VideoDetailDerivedMetadata = {
    mediaSourceType: string;
    isYoutubeMedia: boolean;
    isUploadedMedia: boolean;
    isTikTokMedia: boolean;
    isLocallyHostedMedia: boolean;
    isUploadedAudio: boolean;
    canShowCloneTab: boolean;
    canShowYoutubeTab: boolean;
    aiMetadataTabLabel: string;
    aiMetadataTabTitle: string;
    localMediaPending: boolean;
    transcriptJobActive: boolean;
    voiceFixerStatus: string;
    voiceFixerBusy: boolean;
    voiceFixerPaused: boolean;
    hasVoiceFixerCleaned: boolean;
    voiceFixerApplyScope: string;
    usingVoiceFixerForPlayback: boolean;
    usingVoiceFixerForProcessing: boolean;
    reconstructionStatus: string;
    reconstructionBusy: boolean;
    reconstructionPaused: boolean;
    hasReconstructionAudio: boolean;
    usingReconstructionForPlayback: boolean;
    currentUploadedPlaybackSource: VideoUploadedPlaybackSource;
    reconstructionAudioUrl: string;
};

const TRANSCRIPT_PIPELINE_STATUSES = ['queued', 'downloading', 'transcribing', 'diarizing'];

export const getVideoDetailDerivedMetadata = (video: Video | null): VideoDetailDerivedMetadata => {
    const mediaSourceType = String(video?.media_source_type || 'youtube').toLowerCase();
    const isYoutubeMedia = mediaSourceType === 'youtube';
    const isUploadedMedia = mediaSourceType === 'upload';
    const isTikTokMedia = mediaSourceType === 'tiktok';
    const isLocallyHostedMedia = isUploadedMedia || isTikTokMedia;
    const isUploadedAudio = isLocallyHostedMedia && String(video?.media_kind || '').toLowerCase() === 'audio';
    const canShowCloneTab = !!video?.channel_id;
    const canShowYoutubeTab = isYoutubeMedia || isUploadedMedia;
    const aiMetadataTabLabel = isYoutubeMedia ? 'YouTube' : 'Summary';
    const aiMetadataTabTitle = isYoutubeMedia
        ? 'AI-generated YouTube summary and chapters'
        : 'AI-generated episode summary and chapter index';
    const localMediaPending = isTikTokMedia && ['pending', 'queued'].includes(String(video?.status || '').toLowerCase());
    const transcriptJobActive = TRANSCRIPT_PIPELINE_STATUSES.includes(String(video?.status || '').toLowerCase());
    const voiceFixerStatus = String(video?.voicefixer_status || '').toLowerCase();
    const voiceFixerBusy = isUploadedMedia && (voiceFixerStatus === 'queued' || voiceFixerStatus === 'processing');
    const voiceFixerPaused = isUploadedMedia && voiceFixerStatus === 'paused';
    const hasVoiceFixerCleaned = isUploadedMedia && !!video?.voicefixer_cleaned_path;
    const voiceFixerApplyScope = String(video?.voicefixer_apply_scope || (video?.voicefixer_use_cleaned ? 'both' : 'none')).toLowerCase();
    const usingVoiceFixerForPlayback = isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'playback');
    const usingVoiceFixerForProcessing = isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'processing');
    const reconstructionStatus = String(video?.reconstruction_status || '').toLowerCase();
    const reconstructionBusy = isUploadedMedia && (reconstructionStatus === 'queued' || reconstructionStatus === 'processing');
    const reconstructionPaused = isUploadedMedia && reconstructionStatus === 'paused';
    const hasReconstructionAudio = isUploadedMedia && !!video?.reconstruction_audio_path;
    const usingReconstructionForPlayback = isUploadedMedia && !!video?.reconstruction_use_for_playback;
    const currentUploadedPlaybackSource: VideoUploadedPlaybackSource = usingReconstructionForPlayback
        ? 'reconstructed'
        : usingVoiceFixerForPlayback
            ? 'cleaned'
            : 'original';
    const reconstructionAudioUrl = video && isUploadedMedia && video.reconstruction_audio_path
        ? `${toApiUrl(`/videos/${video.id}/reconstruction/audio`)}?${new URLSearchParams({
            path: String(video.reconstruction_audio_path),
            status: reconstructionStatus || 'ready',
        }).toString()}`
        : '';

    return {
        mediaSourceType,
        isYoutubeMedia,
        isUploadedMedia,
        isTikTokMedia,
        isLocallyHostedMedia,
        isUploadedAudio,
        canShowCloneTab,
        canShowYoutubeTab,
        aiMetadataTabLabel,
        aiMetadataTabTitle,
        localMediaPending,
        transcriptJobActive,
        voiceFixerStatus,
        voiceFixerBusy,
        voiceFixerPaused,
        hasVoiceFixerCleaned,
        voiceFixerApplyScope,
        usingVoiceFixerForPlayback,
        usingVoiceFixerForProcessing,
        reconstructionStatus,
        reconstructionBusy,
        reconstructionPaused,
        hasReconstructionAudio,
        usingReconstructionForPlayback,
        currentUploadedPlaybackSource,
        reconstructionAudioUrl,
    };
};

export const selectVideoDetailDerivedMetadata = (state: VideoState) => getVideoDetailDerivedMetadata(state.video);

export interface VideoState {
    video: Video | null;
    segments: TranscriptSegment[];
    loading: boolean;
    error: string | null;

    setVideo: (value: VideoUpdater) => void;
    setSegments: (value: SegmentsUpdater) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    fetchVideoMeta: (videoId: string) => Promise<Video>;
    fetchVideoSegments: (videoId: string) => Promise<TranscriptSegment[]>;
    refreshVideoData: (videoId: string | number) => Promise<void>;
    resetVideoState: () => void;
}

export const useVideoStore = create<VideoState>()(
    devtools(
        (set, get) => ({
            video: null,
            segments: [],
            loading: true,
            error: null,

            setVideo: (value) => set((state) => ({
                video: typeof value === 'function' ? value(state.video) : value,
            }), false, 'setVideo'),
            setSegments: (value) => set((state) => ({
                segments: typeof value === 'function' ? value(state.segments) : value,
            }), false, 'setSegments'),
            setLoading: (loading) => set({ loading }, false, 'setLoading'),
            setError: (error) => set({ error }, false, 'setError'),
            fetchVideoMeta: async (videoId) => {
                set({ error: null }, false, 'fetchVideoMeta/pending');
                try {
                    const vidRes = await api.get<Video>(`/videos/${videoId}`);
                    set({ video: vidRes.data }, false, 'fetchVideoMeta/fulfilled');
                    return vidRes.data;
                } catch (e) {
                    set({ error: 'Failed to fetch video' }, false, 'fetchVideoMeta/rejected');
                    throw e;
                }
            },
            fetchVideoSegments: async (videoId) => {
                const segRes = await api.get<TranscriptSegment[]>(`/videos/${videoId}/segments`);
                set({ segments: segRes.data }, false, 'fetchVideoSegments/fulfilled');
                return segRes.data;
            },
            refreshVideoData: async (videoId) => {
                const numericVideoId = Number(videoId);
                const transcriptStore = useTranscriptStore.getState();

                try {
                    await get().fetchVideoMeta(String(videoId));
                } catch (e) {
                    console.error('Failed to fetch video:', e);
                    set({ error: 'Failed to fetch video', loading: false }, false, 'refreshVideoData/metaRejected');
                    return;
                }

                try {
                    const fetchedSegments = await get().fetchVideoSegments(String(videoId));
                    if (fetchedSegments.length > 0) {
                        void transcriptStore.fetchFunnyMoments(numericVideoId);
                        void transcriptStore.fetchTranscriptQuality(numericVideoId);
                    } else {
                        transcriptStore.setFunnyMoments([]);
                        useTranscriptStore.setState({ transcriptQuality: null, transcriptQualityError: null, loadingTranscriptQuality: false });
                    }
                } catch (e) {
                    console.error('Failed to fetch segments:', e);
                } finally {
                    set({ loading: false }, false, 'refreshVideoData/settled');
                }
            },
            resetVideoState: () => set({
                video: null,
                segments: [],
                loading: true,
                error: null,
            }, false, 'resetVideoState'),
        }),
        { name: 'VideoStore' },
    ),
);

