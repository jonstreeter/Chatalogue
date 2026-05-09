import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { ReconstructionWorkbench, Video } from '../types';

const DEFAULT_RECONSTRUCTION_TEST_TEXT = 'This is a test of the voice model. If you approve this test, then click the approve voice button below.';

export interface ReconstructionState {
    queueingReconstruction: boolean;
    switchingReconstructionPlayback: boolean;
    loadingReconstructionWorkbench: boolean;
    reconstructionWorkbench: ReconstructionWorkbench | null;
    savingReconstructionSettings: boolean;
    testingReconstructionSpeakerId: number | null;
    reconstructionInstructionDraft: string;
    reconstructionStudioTab: 'voices' | 'reconstruction';
    selectedReconstructionSpeakerId: number | null;
    reconstructionTestTextDrafts: Record<number, string>;
    cleaningReconstructionSampleKey: string | null;
    updatingReconstructionSampleKey: string | null;
    addingReconstructionSampleSpeakerId: number | null;
    approvingReconstructionSpeakerId: number | null;
    selectedReconstructionPreviewSegmentId: number | null;
    previewingReconstructionSegment: boolean;
    reconstructionPreviewAudioUrl: string;
    reconstructionPreviewText: string;

    setReconstructionInstructionDraft: (value: string) => void;
    setReconstructionStudioTab: (tab: 'voices' | 'reconstruction') => void;
    setSelectedReconstructionSpeakerId: (id: number | null | ((previous: number | null) => number | null)) => void;
    setReconstructionTestTextDrafts: (value: Record<number, string> | ((previous: Record<number, string>) => Record<number, string>)) => void;
    setSelectedReconstructionPreviewSegmentId: (id: number | null) => void;
    setReconstructionWorkbench: (workbench: ReconstructionWorkbench | null) => void;

    syncInstructionDraftFromVideo: (video: Video | null, isUploadedMedia: boolean) => void;
    syncWorkbenchSelection: () => void;
    resetReconstructionState: () => void;
    loadReconstructionWorkbench: (video: Video | null, isUploadedMedia: boolean, segmentsCount: number) => Promise<ReconstructionWorkbench | null>;
    queueReconstruction: (video: Video | null, isUploadedMedia: boolean, hasReconstructionAudio: boolean, force: boolean, onVideoUpdated: (video: Video) => void) => Promise<void>;
    saveReconstructionSettings: (video: Video | null, isUploadedMedia: boolean, onVideoUpdated: (video: Video) => void) => Promise<void>;
    testReconstructionSpeaker: (video: Video | null, isUploadedMedia: boolean, speakerId: number, segmentId?: number, options?: { performanceMode?: boolean; useSelectedSampleText?: boolean }) => Promise<void>;
    cleanupReconstructionSample: (video: Video | null, isUploadedMedia: boolean, speakerId: number, segmentId: number) => Promise<void>;
    updateReconstructionSampleState: (video: Video | null, isUploadedMedia: boolean, speakerId: number, segmentId: number, patch: { rejected?: boolean; selected?: boolean; clear_cleaned?: boolean }) => Promise<void>;
    addReconstructionSample: (video: Video | null, isUploadedMedia: boolean, speakerId: number) => Promise<void>;
    approveReconstructionSpeaker: (video: Video | null, isUploadedMedia: boolean, speakerId: number, approved: boolean) => Promise<void>;
    previewReconstructionSegment: (video: Video | null, isUploadedMedia: boolean, segmentId: number, resolveWorkbenchAudioUrl: (url?: string) => string) => Promise<void>;
    setUploadedPlaybackSource: (video: Video | null, isUploadedMedia: boolean, source: 'original' | 'cleaned' | 'reconstructed', currentSource: 'original' | 'cleaned' | 'reconstructed', onVideoUpdated: (video: Video) => void) => Promise<void>;
}

export const useReconstructionStore = create<ReconstructionState>()(
    devtools(
        (set, get) => ({
            queueingReconstruction: false,
            switchingReconstructionPlayback: false,
            loadingReconstructionWorkbench: false,
            reconstructionWorkbench: null,
            savingReconstructionSettings: false,
            testingReconstructionSpeakerId: null,
            reconstructionInstructionDraft: '',
            reconstructionStudioTab: 'voices',
            selectedReconstructionSpeakerId: null,
            reconstructionTestTextDrafts: {},
            cleaningReconstructionSampleKey: null,
            updatingReconstructionSampleKey: null,
            addingReconstructionSampleSpeakerId: null,
            approvingReconstructionSpeakerId: null,
            selectedReconstructionPreviewSegmentId: null,
            previewingReconstructionSegment: false,
            reconstructionPreviewAudioUrl: '',
            reconstructionPreviewText: '',

            setReconstructionInstructionDraft: (value) => set({ reconstructionInstructionDraft: value }, false, 'setReconstructionInstructionDraft'),
            setReconstructionStudioTab: (tab) => set({ reconstructionStudioTab: tab }, false, 'setReconstructionStudioTab'),
            setSelectedReconstructionSpeakerId: (id) => set((state) => ({ selectedReconstructionSpeakerId: typeof id === 'function' ? id(state.selectedReconstructionSpeakerId) : id }), false, 'setSelectedReconstructionSpeakerId'),
            setReconstructionTestTextDrafts: (value) => set((state) => ({ reconstructionTestTextDrafts: typeof value === 'function' ? value(state.reconstructionTestTextDrafts) : value }), false, 'setReconstructionTestTextDrafts'),
            setSelectedReconstructionPreviewSegmentId: (id) => set({ selectedReconstructionPreviewSegmentId: id }, false, 'setSelectedReconstructionPreviewSegmentId'),
            setReconstructionWorkbench: (workbench) => set({ reconstructionWorkbench: workbench }, false, 'setReconstructionWorkbench'),

            syncInstructionDraftFromVideo: (video, isUploadedMedia) => {
                if (!video || !isUploadedMedia) return;
                set({ reconstructionInstructionDraft: String(video.reconstruction_instruction_template || '') }, false, 'syncInstructionDraftFromVideo');
            },

            syncWorkbenchSelection: () => {
                const workbench = get().reconstructionWorkbench;
                if (!workbench?.speakers?.length) {
                    set({ selectedReconstructionSpeakerId: null }, false, 'syncWorkbenchSelection/empty');
                    return;
                }
                set((state) => {
                    const selectedReconstructionSpeakerId = state.selectedReconstructionSpeakerId && workbench.speakers.some((speaker) => speaker.speaker_id === state.selectedReconstructionSpeakerId)
                        ? state.selectedReconstructionSpeakerId
                        : workbench.speakers[0].speaker_id;
                    const reconstructionTestTextDrafts = { ...state.reconstructionTestTextDrafts };
                    for (const speaker of workbench.speakers) {
                        if (!reconstructionTestTextDrafts[speaker.speaker_id]) {
                            reconstructionTestTextDrafts[speaker.speaker_id] = speaker.latest_test_text || speaker.reference_text || DEFAULT_RECONSTRUCTION_TEST_TEXT;
                        }
                    }
                    return { selectedReconstructionSpeakerId, reconstructionTestTextDrafts };
                }, false, 'syncWorkbenchSelection');
            },

            resetReconstructionState: () => set({
                queueingReconstruction: false,
                switchingReconstructionPlayback: false,
                loadingReconstructionWorkbench: false,
                reconstructionWorkbench: null,
                savingReconstructionSettings: false,
                testingReconstructionSpeakerId: null,
                reconstructionInstructionDraft: '',
                reconstructionStudioTab: 'voices',
                selectedReconstructionSpeakerId: null,
                reconstructionTestTextDrafts: {},
                cleaningReconstructionSampleKey: null,
                updatingReconstructionSampleKey: null,
                addingReconstructionSampleSpeakerId: null,
                approvingReconstructionSpeakerId: null,
                selectedReconstructionPreviewSegmentId: null,
                previewingReconstructionSegment: false,
                reconstructionPreviewAudioUrl: '',
                reconstructionPreviewText: '',
            }, false, 'resetReconstructionState'),

            loadReconstructionWorkbench: async (video, isUploadedMedia, segmentsCount) => {
                if (!video || !isUploadedMedia || segmentsCount === 0) return null;
                set({ loadingReconstructionWorkbench: true }, false, 'loadReconstructionWorkbench/pending');
                try {
                    const res = await api.get<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench`);
                    set({ reconstructionWorkbench: res.data }, false, 'loadReconstructionWorkbench/fulfilled');
                    get().syncWorkbenchSelection();
                    return res.data;
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to load the reconstruction workbench');
                    return null;
                } finally {
                    set({ loadingReconstructionWorkbench: false }, false, 'loadReconstructionWorkbench/settled');
                }
            },

            queueReconstruction: async (video, isUploadedMedia, hasReconstructionAudio, force, onVideoUpdated) => {
                if (!video || !isUploadedMedia) return;
                const prompt = hasReconstructionAudio ? 'Rebuild the reconstructed conversation audio for this uploaded episode?' : 'Create reconstructed conversation audio for this uploaded episode?';
                if (!confirm(prompt)) return;
                set({ queueingReconstruction: true }, false, 'queueReconstruction/pending');
                try {
                    const res = await api.post<Video>(`/videos/${video.id}/reconstruct/queue`, null, { params: { force } });
                    onVideoUpdated(res.data);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to queue conversation reconstruction');
                } finally {
                    set({ queueingReconstruction: false }, false, 'queueReconstruction/settled');
                }
            },

            saveReconstructionSettings: async (video, isUploadedMedia, onVideoUpdated) => {
                if (!video || !isUploadedMedia) return;
                set({ savingReconstructionSettings: true }, false, 'saveReconstructionSettings/pending');
                try {
                    const res = await api.patch<Video>(`/videos/${video.id}/reconstruction/settings`, {
                        mode: 'performance',
                        instruction_template: get().reconstructionInstructionDraft,
                    });
                    onVideoUpdated(res.data);
                    await get().loadReconstructionWorkbench(res.data, isUploadedMedia, 1);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to save reconstruction settings');
                } finally {
                    set({ savingReconstructionSettings: false }, false, 'saveReconstructionSettings/settled');
                }
            },

            testReconstructionSpeaker: async (video, isUploadedMedia, speakerId, segmentId, options) => {
                if (!video || !isUploadedMedia) return;
                set({ testingReconstructionSpeakerId: speakerId }, false, 'testReconstructionSpeaker/pending');
                try {
                    const speakerCard = get().reconstructionWorkbench?.speakers.find(s => s.speaker_id === speakerId) || null;
                    const selectedSegment = speakerCard?.samples.find(seg => seg.segment_id === segmentId) || speakerCard?.samples.find(sample => sample.selected) || speakerCard?.samples[0] || null;
                    const requestedText = options?.useSelectedSampleText
                        ? (selectedSegment?.text || speakerCard?.reference_text || '')
                        : (get().reconstructionTestTextDrafts[speakerId] || selectedSegment?.text || speakerCard?.reference_text || '');
                    await api.post(`/videos/${video.id}/reconstruction/test-speaker`, {
                        speaker_id: speakerId,
                        segment_id: selectedSegment?.segment_id,
                        text: requestedText,
                        performance_mode: !!options?.performanceMode,
                    });
                    await get().loadReconstructionWorkbench(video, isUploadedMedia, 1);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to generate a reconstruction test for this speaker');
                } finally {
                    set({ testingReconstructionSpeakerId: null }, false, 'testReconstructionSpeaker/settled');
                }
            },

            cleanupReconstructionSample: async (video, isUploadedMedia, speakerId, segmentId) => {
                if (!video || !isUploadedMedia) return;
                const key = `${speakerId}:${segmentId}`;
                set({ cleaningReconstructionSampleKey: key }, false, 'cleanupReconstructionSample/pending');
                try {
                    const res = await api.post<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/sample-cleanup`, { speaker_id: speakerId, segment_id: segmentId });
                    set({ reconstructionWorkbench: res.data }, false, 'cleanupReconstructionSample/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to clean up this performance sample');
                } finally {
                    set({ cleaningReconstructionSampleKey: null }, false, 'cleanupReconstructionSample/settled');
                }
            },

            updateReconstructionSampleState: async (video, isUploadedMedia, speakerId, segmentId, patch) => {
                if (!video || !isUploadedMedia) return;
                const key = `${speakerId}:${segmentId}`;
                set({ updatingReconstructionSampleKey: key }, false, 'updateReconstructionSampleState/pending');
                try {
                    const res = await api.patch<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/sample-state`, { speaker_id: speakerId, segment_id: segmentId, ...patch });
                    set({ reconstructionWorkbench: res.data }, false, 'updateReconstructionSampleState/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to update this performance sample');
                } finally {
                    set({ updatingReconstructionSampleKey: null }, false, 'updateReconstructionSampleState/settled');
                }
            },

            addReconstructionSample: async (video, isUploadedMedia, speakerId) => {
                if (!video || !isUploadedMedia) return;
                set({ addingReconstructionSampleSpeakerId: speakerId }, false, 'addReconstructionSample/pending');
                try {
                    const res = await api.post<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/add-sample`, { speaker_id: speakerId });
                    set({ reconstructionWorkbench: res.data }, false, 'addReconstructionSample/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to add another performance sample');
                } finally {
                    set({ addingReconstructionSampleSpeakerId: null }, false, 'addReconstructionSample/settled');
                }
            },

            approveReconstructionSpeaker: async (video, isUploadedMedia, speakerId, approved) => {
                if (!video || !isUploadedMedia) return;
                set({ approvingReconstructionSpeakerId: speakerId }, false, 'approveReconstructionSpeaker/pending');
                try {
                    const res = await api.patch<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/speaker-approval`, { speaker_id: speakerId, approved });
                    set({ reconstructionWorkbench: res.data }, false, 'approveReconstructionSpeaker/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to update voice approval');
                } finally {
                    set({ approvingReconstructionSpeakerId: null }, false, 'approveReconstructionSpeaker/settled');
                }
            },

            previewReconstructionSegment: async (video, isUploadedMedia, segmentId, resolveWorkbenchAudioUrl) => {
                if (!video || !isUploadedMedia) return;
                set({ previewingReconstructionSegment: true, reconstructionPreviewAudioUrl: '', reconstructionPreviewText: '' }, false, 'previewReconstructionSegment/pending');
                try {
                    const res = await api.post(`/videos/${video.id}/reconstruction/preview-segment`, { segment_id: segmentId, performance_mode: true });
                    set({ reconstructionPreviewAudioUrl: resolveWorkbenchAudioUrl(String(res.data.audio_url || '')), reconstructionPreviewText: String(res.data.text || '') }, false, 'previewReconstructionSegment/fulfilled');
                } catch (e: any) {
                    set({ reconstructionPreviewAudioUrl: '', reconstructionPreviewText: '' }, false, 'previewReconstructionSegment/rejected');
                    alert(e?.response?.data?.detail || 'Failed to preview this reconstruction segment');
                } finally {
                    set({ previewingReconstructionSegment: false }, false, 'previewReconstructionSegment/settled');
                }
            },

            setUploadedPlaybackSource: async (video, isUploadedMedia, source, currentSource, onVideoUpdated) => {
                if (!video || !isUploadedMedia || source === currentSource) return;
                set({ switchingReconstructionPlayback: true }, false, 'setUploadedPlaybackSource/pending');
                try {
                    const res = await api.patch<Video>(`/videos/${video.id}/playback-source`, { source });
                    onVideoUpdated(res.data);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to switch playback source');
                } finally {
                    set({ switchingReconstructionPlayback: false }, false, 'setUploadedPlaybackSource/settled');
                }
            },
        }),
        { name: 'ReconstructionStore' }
    )
);
