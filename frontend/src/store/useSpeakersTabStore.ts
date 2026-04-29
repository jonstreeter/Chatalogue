import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Speaker, SpeakerSample, TranscriptSegment, Video } from '../types';

type AssignPopup = {
    segmentId: number;
    x: number;
    y: number;
};

const ASSIGN_SPEAKER_PICKER_LIMIT = 100;
let _assignRequestId = 0;
const speakerDetailCache = new Map<number, Speaker>();

export interface SpeakersTabState {
    selectedSpeaker: Speaker | null;
    initialSample: SpeakerSample | null;
    assignPopup: AssignPopup | null;
    assignSpeakers: Speaker[];
    assignSearch: string;
    assignLoading: boolean;

    setAssignSearch: (search: string) => void;
    closeSpeakerModal: () => void;
    closeAssignPopup: () => void;
    openAssignPopup: (segmentId: number, x: number, y: number) => void;
    fetchAssignSpeakers: (channelId: number | null | undefined) => Promise<void>;
    assignSpeaker: (speakerId: number, onAssigned: () => void) => Promise<void>;
    openSpeaker: (speakerId: number, segment: TranscriptSegment | undefined, video: Video | null) => Promise<void>;
    handleSpeakerUpdated: (updatedSpeaker: Speaker, onSegmentsUpdated: (speakerId: number, name: string) => void) => void;
    handleSpeakerMerged: (videoId: number | string | null | undefined, onSegmentsLoaded: (segments: TranscriptSegment[]) => void) => Promise<void>;
    resetSpeakersTabState: () => void;
}

function buildSpeakerPlaceholder(speakerId: number, segment: TranscriptSegment | undefined, video: Video | null): Speaker | null {
    if (!video) return null;
    const cached = speakerDetailCache.get(speakerId);
    if (cached) return cached;
    return {
        id: speakerId,
        channel_id: video.channel_id,
        name: segment?.speaker || `Speaker ${speakerId}`,
        thumbnail_path: undefined,
        is_extra: false,
        total_speaking_time: 0,
        embedding_count: 0,
        created_at: '',
    };
}

function buildInitialSample(segment: TranscriptSegment | undefined, video: Video | null): SpeakerSample | null {
    if (!segment || !video) return null;
    return {
        youtube_id: video.youtube_id,
        video_id: video.id,
        start_time: segment.start_time,
        end_time: segment.end_time,
        text: segment.text,
        media_source_type: video.media_source_type,
        media_kind: video.media_kind,
    };
}

export const useSpeakersTabStore = create<SpeakersTabState>()(
    devtools(
        (set, get) => ({
            selectedSpeaker: null,
            initialSample: null,
            assignPopup: null,
            assignSpeakers: [],
            assignSearch: '',
            assignLoading: false,

            setAssignSearch: (search) => set({ assignSearch: search }, false, 'setAssignSearch'),

            closeSpeakerModal: () =>
                set({ selectedSpeaker: null, initialSample: null }, false, 'closeSpeakerModal'),

            closeAssignPopup: () => set({ assignPopup: null }, false, 'closeAssignPopup'),

            openAssignPopup: (segmentId, x, y) =>
                set(
                    {
                        assignSearch: '',
                        assignSpeakers: [],
                        assignPopup: { segmentId, x, y },
                    },
                    false,
                    'openAssignPopup'
                ),

            fetchAssignSpeakers: async (channelId) => {
                if (!get().assignPopup || !channelId) return;
                const requestId = ++_assignRequestId;
                set({ assignLoading: true }, false, 'fetchAssignSpeakers/pending');
                try {
                    const trimmedSearch = get().assignSearch.trim();
                    const res = await api.get<Speaker[]>('/speakers', {
                        params: {
                            channel_id: channelId,
                            limit: ASSIGN_SPEAKER_PICKER_LIMIT,
                            search: trimmedSearch || undefined,
                        },
                    });
                    if (requestId !== _assignRequestId) return;
                    set({ assignSpeakers: Array.isArray(res.data) ? res.data : [] }, false, 'fetchAssignSpeakers/fulfilled');
                } catch (e) {
                    if (requestId === _assignRequestId) {
                        console.error('Failed to fetch speakers', e);
                    }
                } finally {
                    if (requestId === _assignRequestId) {
                        set({ assignLoading: false }, false, 'fetchAssignSpeakers/settled');
                    }
                }
            },

            assignSpeaker: async (speakerId, onAssigned) => {
                const { assignPopup } = get();
                if (!assignPopup) return;
                try {
                    await api.patch(`/segments/${assignPopup.segmentId}/assign-speaker`, { speaker_id: speakerId });
                    set({ assignPopup: null }, false, 'assignSpeaker/fulfilled');
                    onAssigned();
                } catch (e) {
                    console.error('Failed to assign speaker', e);
                    alert('Failed to assign speaker');
                }
            },

            openSpeaker: async (speakerId, segment, video) => {
                set({ initialSample: buildInitialSample(segment, video) }, false, 'openSpeaker/sample');

                const placeholder = buildSpeakerPlaceholder(speakerId, segment, video);
                if (placeholder) {
                    set({ selectedSpeaker: placeholder }, false, 'openSpeaker/placeholder');
                }

                try {
                    const res = await api.get<Speaker>(`/speakers/${speakerId}`);
                    speakerDetailCache.set(speakerId, res.data);
                    set({ selectedSpeaker: res.data }, false, 'openSpeaker/fulfilled');
                } catch (e) {
                    console.error('Failed to fetch speaker details', e);
                    if (!placeholder) {
                        set({ selectedSpeaker: null }, false, 'openSpeaker/rejected');
                    }
                }
            },

            handleSpeakerUpdated: (updatedSpeaker, onSegmentsUpdated) => {
                speakerDetailCache.set(updatedSpeaker.id, updatedSpeaker);
                set({ selectedSpeaker: updatedSpeaker }, false, 'handleSpeakerUpdated');
                onSegmentsUpdated(updatedSpeaker.id, updatedSpeaker.name);
            },

            handleSpeakerMerged: async (videoId, onSegmentsLoaded) => {
                if (!videoId) return;
                const res = await api.get<TranscriptSegment[]>(`/videos/${videoId}/segments`);
                onSegmentsLoaded(res.data);
            },

            resetSpeakersTabState: () => {
                _assignRequestId += 1;
                set(
                    {
                        selectedSpeaker: null,
                        initialSample: null,
                        assignPopup: null,
                        assignSpeakers: [],
                        assignSearch: '',
                        assignLoading: false,
                    },
                    false,
                    'resetSpeakersTabState'
                );
            },
        }),
        { name: 'SpeakersTabStore' }
    )
);
