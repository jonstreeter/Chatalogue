import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Video, ClearVoiceInstallInfo, ClearVoiceTestResult, CleanupWorkbench } from '../types';

interface CleanupState {
    queueingVoiceFixer: boolean;
    loadingCleanupWorkbench: boolean;
    cleanupWorkbench: CleanupWorkbench | null;
    analyzingCleanupWorkbench: boolean;
    runningClearVoiceModel: string | null;
    selectingCleanupCandidateId: string | null;
    clearVoiceInstallInfo: ClearVoiceInstallInfo | null;
    loadingClearVoiceInstallInfo: boolean;
    installingClearVoice: boolean;
    repairingClearVoice: boolean;
    testingClearVoice: boolean;
    clearVoiceTestResult: ClearVoiceTestResult | null;
    savingVoiceFixerSettings: boolean;
    voiceFixerModeDraft: number;
    voiceFixerMixDraft: number;
    voiceFixerLevelingDraft: 'off' | 'gentle' | 'balanced' | 'strong';
    voiceFixerApplyScopeDraft: 'none' | 'playback' | 'processing' | 'both';
    voiceFixerAction: 'start' | 'stop' | 'pause' | 'resume' | null;
    voiceFixerStatus: string | null;

    fetchCleanupWorkbench: (videoId: number) => Promise<void>;
    fetchClearVoiceInstallInfo: () => Promise<void>;
    handleQueueVoiceFixer: (
        videoId: number,
        hasVoiceFixerCleaned: boolean,
        onVideoUpdated: (v: Video) => void
    ) => Promise<void>;
    handleSaveVoiceFixerSettings: (videoId: number, onVideoUpdated: (v: Video) => void) => Promise<void>;
    handleAnalyzeCleanupWorkbench: (videoId: number) => Promise<void>;
    handleRunClearVoiceCandidate: (videoId: number, modelName: string) => Promise<void>;
    handleSelectCleanupCandidate: (videoId: number, candidateId: string | null) => Promise<void>;
    handleInstallClearVoice: () => Promise<void>;
    handleTestClearVoice: () => Promise<void>;
    handleRepairClearVoice: () => Promise<void>;
    handleVoiceFixer: (videoId: number, action: 'start' | 'stop' | 'pause' | 'resume') => Promise<void>;
    resetCleanupState: () => void;
}

export const useCleanupStore = create<CleanupState>()(
    devtools(
        (set, get) => ({
            queueingVoiceFixer: false,
            loadingCleanupWorkbench: false,
            cleanupWorkbench: null,
            analyzingCleanupWorkbench: false,
            runningClearVoiceModel: null,
            selectingCleanupCandidateId: null,
            clearVoiceInstallInfo: null,
            loadingClearVoiceInstallInfo: false,
            installingClearVoice: false,
            repairingClearVoice: false,
            testingClearVoice: false,
            clearVoiceTestResult: null,
            savingVoiceFixerSettings: false,
            voiceFixerModeDraft: 0,
            voiceFixerMixDraft: 1,
            voiceFixerLevelingDraft: 'off',
            voiceFixerApplyScopeDraft: 'none',
            voiceFixerAction: null,
            voiceFixerStatus: null,

            fetchCleanupWorkbench: async (videoId: number) => {
                set({ loadingCleanupWorkbench: true }, false, 'fetchCleanupWorkbench/pending');
                try {
                    const res = await api.get<CleanupWorkbench>(`/videos/${videoId}/cleanup/workbench`);
                    set({ cleanupWorkbench: res.data || null }, false, 'fetchCleanupWorkbench/fulfilled');
                } catch (e: any) {
                    console.error('Failed to fetch cleanup workbench', e);
                    // Set error state to null to indicate failure
                    set({ cleanupWorkbench: null }, false, 'fetchCleanupWorkbench/rejected');
                } finally {
                    set({ loadingCleanupWorkbench: false }, false, 'fetchCleanupWorkbench/settled');
                }
            },

            fetchClearVoiceInstallInfo: async () => {
                set({ loadingClearVoiceInstallInfo: true }, false, 'fetchClearVoiceInstallInfo/pending');
                try {
                    const res = await api.get<ClearVoiceInstallInfo>('/system/clearvoice/install-info');
                    set({ clearVoiceInstallInfo: res.data }, false, 'fetchClearVoiceInstallInfo/fulfilled');
                } catch (e) {
                    console.error('Failed to fetch ClearVoice install info', e);
                } finally {
                    set({ loadingClearVoiceInstallInfo: false }, false, 'fetchClearVoiceInstallInfo/settled');
                }
            },

            handleQueueVoiceFixer: async (videoId: number, hasVoiceFixerCleaned: boolean, onVideoUpdated: (v: Video) => void) => {
                const prompt = hasVoiceFixerCleaned
                    ? 'Rebuild the VoiceFixer-cleaned media for this uploaded episode?'
                    : 'Create a VoiceFixer-cleaned copy of this uploaded episode?';
                if (!confirm(prompt)) return;
                set({ queueingVoiceFixer: true }, false, 'handleQueueVoiceFixer/pending');
                try {
                    const res = await api.post<Video>(`/videos/${videoId}/voicefixer/queue`, null, {
                        params: { force: hasVoiceFixerCleaned },
                    });
                    onVideoUpdated(res.data);
                    set({ queueingVoiceFixer: false }, false, 'handleQueueVoiceFixer/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to queue VoiceFixer cleanup');
                    set({ queueingVoiceFixer: false }, false, 'handleQueueVoiceFixer/rejected');
                }
            },

            handleSaveVoiceFixerSettings: async (videoId: number, onVideoUpdated: (v: Video) => void) => {
                const { voiceFixerModeDraft, voiceFixerMixDraft, voiceFixerLevelingDraft, voiceFixerApplyScopeDraft } = get();
                set({ savingVoiceFixerSettings: true }, false, 'handleSaveVoiceFixerSettings/pending');
                try {
                    const res = await api.patch<Video>(`/videos/${videoId}/voicefixer/settings`, {
                        mode: voiceFixerModeDraft,
                        mix_ratio: voiceFixerMixDraft,
                        leveling_mode: voiceFixerLevelingDraft,
                        apply_scope: voiceFixerApplyScopeDraft,
                    });
                    onVideoUpdated(res.data);
                    set({ savingVoiceFixerSettings: false }, false, 'handleSaveVoiceFixerSettings/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to save VoiceFixer settings');
                    set({ savingVoiceFixerSettings: false }, false, 'handleSaveVoiceFixerSettings/rejected');
                }
            },

            handleAnalyzeCleanupWorkbench: async (videoId: number) => {
                set({ analyzingCleanupWorkbench: true }, false, 'handleAnalyzeCleanupWorkbench/pending');
                try {
                    const res = await api.post<CleanupWorkbench>(`/videos/${videoId}/cleanup/workbench/analyze`);
                    set({ cleanupWorkbench: res.data }, false, 'handleAnalyzeCleanupWorkbench/fulfilled');
                } catch (e: any) {
                    console.error('Failed to analyze cleanup workbench', e);
                    alert(e?.response?.data?.detail || 'Failed to analyze the uploaded audio');
                    set({ analyzingCleanupWorkbench: false }, false, 'handleAnalyzeCleanupWorkbench/rejected');
                }
            },

            handleRunClearVoiceCandidate: async (videoId: number, modelName: string) => {
                set({ runningClearVoiceModel: modelName }, false, 'handleRunClearVoiceCandidate/pending');
                try {
                    const res = await api.post<CleanupWorkbench>(
                        `/videos/${videoId}/cleanup/workbench/clearvoice-candidate`,
                        { stage: 'enhancement', model_name: modelName }
                    );
                    set({ cleanupWorkbench: res.data }, false, 'handleRunClearVoiceCandidate/fulfilled');
                } catch (e: any) {
                    console.error(`Failed to generate the ${modelName} candidate`, e);
                    alert(e?.response?.data?.detail || `Failed to generate the ${modelName} candidate`);
                    set({ runningClearVoiceModel: null }, false, 'handleRunClearVoiceCandidate/rejected');
                }
            },

            handleSelectCleanupCandidate: async (videoId: number, candidateId: string | null) => {
                set({ selectingCleanupCandidateId: candidateId || '__original__' }, false, 'handleSelectCleanupCandidate/pending');
                try {
                    const res = await api.patch<CleanupWorkbench>(
                        `/videos/${videoId}/cleanup/workbench/select-candidate`,
                        { candidate_id: candidateId }
                    );
                    set({ cleanupWorkbench: res.data }, false, 'handleSelectCleanupCandidate/fulfilled');
                } catch (e: any) {
                    console.error('Failed to update the selected pre-cleanup candidate', e);
                    alert(e?.response?.data?.detail || 'Failed to update the selected pre-cleanup candidate');
                    set({ selectingCleanupCandidateId: null }, false, 'handleSelectCleanupCandidate/rejected');
                }
            },

            handleInstallClearVoice: async () => {
                set({ installingClearVoice: true }, false, 'handleInstallClearVoice/pending');
                try {
                    const res = await api.post<ClearVoiceInstallInfo>('/system/clearvoice/install');
                    set({ clearVoiceInstallInfo: res.data, clearVoiceTestResult: null }, false, 'handleInstallClearVoice/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to install ClearVoice');
                    set({ installingClearVoice: false }, false, 'handleInstallClearVoice/rejected');
                }
            },

            handleTestClearVoice: async () => {
                set({ testingClearVoice: true }, false, 'handleTestClearVoice/pending');
                try {
                    const res = await api.post<ClearVoiceTestResult>('/system/clearvoice/test');
                    set({ clearVoiceTestResult: res.data }, false, 'handleTestClearVoice/fulfilled');
                } catch (e: any) {
                    console.error('Failed to test ClearVoice', e);
                    set({
                        clearVoiceTestResult: {
                            status: 'error',
                            imported: false,
                            class_available: false,
                            torch_imported: false,
                            torchaudio_imported: false,
                            runtime_ready: false,
                            error: e?.response?.data?.detail || 'Failed to test ClearVoice',
                            detail: 'The backend could not validate the local ClearVoice runtime.',
                        },
                    }, false, 'handleTestClearVoice/rejected');
                } finally {
                    set({ testingClearVoice: false }, false, 'handleTestClearVoice/settled');
                    await get().fetchClearVoiceInstallInfo();
                }
            },

            handleRepairClearVoice: async () => {
                if (!confirm('Repair the ClearVoice runtime now? This reinstalls torchaudio to match the backend torch build.')) return;
                set({ repairingClearVoice: true }, false, 'handleRepairClearVoice/pending');
                try {
                    await api.post('/system/clearvoice/repair');
                    await get().fetchClearVoiceInstallInfo();
                    await get().handleTestClearVoice();
                    set({ repairingClearVoice: false }, false, 'handleRepairClearVoice/fulfilled');
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to repair the ClearVoice runtime');
                    set({ repairingClearVoice: false }, false, 'handleRepairClearVoice/rejected');
                }
            },

            resetCleanupState: () => {
                set({
                    queueingVoiceFixer: false,
                    loadingCleanupWorkbench: false,
                    cleanupWorkbench: null,
                    analyzingCleanupWorkbench: false,
                    runningClearVoiceModel: null,
                    selectingCleanupCandidateId: null,
                    clearVoiceInstallInfo: null,
                    loadingClearVoiceInstallInfo: false,
                    installingClearVoice: false,
                    repairingClearVoice: false,
                    testingClearVoice: false,
                    clearVoiceTestResult: null,
                    savingVoiceFixerSettings: false,
                    voiceFixerModeDraft: 0,
                    voiceFixerMixDraft: 1,
                    voiceFixerLevelingDraft: 'off',
                    voiceFixerApplyScopeDraft: 'none',
                });
            },

            handleVoiceFixer: async (videoId: number, action: 'start' | 'stop' | 'pause' | 'resume') => {
                set({ voiceFixerAction: action }, false, 'handleVoiceFixer/pending');
                try {
                    const res = await api.post(`/videos/${videoId}/voicefixer/${action}`);
                    set({ voiceFixerStatus: res.data.status }, false, 'handleVoiceFixer/fulfilled');
                } catch (e: any) {
                    console.error(`Failed to ${action} VoiceFixer`, e);
                    alert(e?.response?.data?.detail || `Failed to ${action} VoiceFixer`);
                    set({ voiceFixerAction: null }, false, 'handleVoiceFixer/rejected');
                }
            },
        }),
        { name: 'CleanupStore' }
    )
);
