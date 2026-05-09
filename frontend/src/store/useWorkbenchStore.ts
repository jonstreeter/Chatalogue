import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Job, WorkbenchTaskProgress } from '../types';

interface WorkbenchState {
    auxiliaryJobs: Job[];
    workbenchTaskProgress: WorkbenchTaskProgress | null;
    loadingAuxiliaryJobs: boolean;

    loadAuxiliaryJobs: (videoId: number) => Promise<void>;
    fetchWorkbenchTaskProgress: (videoId: number) => Promise<void>;
    resetWorkbenchState: () => void;
}

export const useWorkbenchStore = create<WorkbenchState>()(
    devtools(
        (set) => ({
            auxiliaryJobs: [],
            workbenchTaskProgress: null,
            loadingAuxiliaryJobs: false,

            loadAuxiliaryJobs: async (videoId: number) => {
                set({ loadingAuxiliaryJobs: true });
                try {
                    const res = await api.get<Job[]>('/jobs', {
                        params: {
                            video_id: videoId,
                            status: 'queued,paused,running',
                            job_type: 'voicefixer_cleanup,conversation_reconstruct',
                            limit: 20,
                        },
                    });
                    set({ auxiliaryJobs: Array.isArray(res.data) ? res.data : [] });
                } catch (e) {
                    console.error('Failed to fetch auxiliary jobs:', e);
                } finally {
                    set({ loadingAuxiliaryJobs: false });
                }
            },

            fetchWorkbenchTaskProgress: async (videoId: number) => {
                try {
                    const res = await api.get<WorkbenchTaskProgress>(`/videos/${videoId}/workbench/progress`);
                    set({ workbenchTaskProgress: res.data || null });
                } catch (e) {
                    console.error('Failed to fetch workbench progress:', e);
                }
            },

            resetWorkbenchState: () => {
                set({
                    auxiliaryJobs: [],
                    workbenchTaskProgress: null,
                    loadingAuxiliaryJobs: false,
                });
            },
        }),
        { name: 'WorkbenchStore' }
    )
);
