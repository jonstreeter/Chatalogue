import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Video, VideoDescriptionRevision, VideoChapterSuggestion } from '../types';

export interface YoutubeState {
    generatingYoutubeAi: boolean;
    copiedYoutubeField: 'summary' | 'chapters' | 'description' | null;
    descriptionHistory: VideoDescriptionRevision[];
    loadingDescriptionHistory: boolean;
    publishingYoutubeDescription: boolean;
    restoringDescriptionRevisionId: number | null;

    fetchDescriptionHistory: (videoId: number) => Promise<void>;
    handleGenerateYoutubeAi: (videoId: number, force: boolean, onVideoUpdated: (v: Video) => void) => Promise<void>;
    handlePublishYoutubeDescription: (videoId: number, isYoutubeMedia: boolean, onVideoUpdated: (v: Video) => void) => Promise<void>;
    handleRestoreDescriptionRevision: (videoId: number, revision: VideoDescriptionRevision, onVideoUpdated: (v: Video) => void) => Promise<void>;
    handleCopyYoutubeField: (kind: 'summary' | 'chapters' | 'description', text: string) => Promise<void>;
    resetYoutubeState: () => void;
}

export const useYoutubeStore = create<YoutubeState>()(
    devtools(
        (set, get) => ({
            generatingYoutubeAi: false,
            copiedYoutubeField: null,
            descriptionHistory: [],
            loadingDescriptionHistory: false,
            publishingYoutubeDescription: false,
            restoringDescriptionRevisionId: null,

            fetchDescriptionHistory: async (videoId) => {
                set({ loadingDescriptionHistory: true }, false, 'fetchDescriptionHistory/pending');
                try {
                    const res = await api.get<VideoDescriptionRevision[]>(`/videos/${videoId}/description-history`);
                    set({ descriptionHistory: res.data }, false, 'fetchDescriptionHistory/fulfilled');
                } catch (e) {
                    console.error('Failed to fetch description history', e);
                } finally {
                    set({ loadingDescriptionHistory: false }, false, 'fetchDescriptionHistory/settled');
                }
            },

            handleGenerateYoutubeAi: async (videoId, force, onVideoUpdated) => {
                set({ generatingYoutubeAi: true }, false, 'handleGenerateYoutubeAi/pending');
                try {
                    const res = await api.post<Video>(`/videos/${videoId}/youtube-ai/generate`, null, { params: { force } });
                    onVideoUpdated(res.data);
                } catch (e: any) {
                    console.error('Failed to generate YouTube metadata', e);
                    alert(e?.response?.data?.detail || 'Failed to generate YouTube summary/chapters');
                } finally {
                    set({ generatingYoutubeAi: false }, false, 'handleGenerateYoutubeAi/settled');
                }
            },

            handlePublishYoutubeDescription: async (videoId, isYoutubeMedia, onVideoUpdated) => {
                const confirmMsg = isYoutubeMedia
                    ? 'Archive the current description and replace it with the AI-generated YouTube description draft?'
                    : 'Archive the current episode description and replace it with the AI-generated summary draft?';
                if (!confirm(confirmMsg)) return;
                set({ publishingYoutubeDescription: true }, false, 'handlePublishYoutubeDescription/pending');
                try {
                    const res = await api.post<Video>(`/videos/${videoId}/youtube-ai/publish-description`);
                    onVideoUpdated(res.data);
                    await get().fetchDescriptionHistory(videoId);
                } catch (e: any) {
                    console.error('Failed to publish AI description', e);
                    alert(e?.response?.data?.detail || 'Failed to publish AI description');
                } finally {
                    set({ publishingYoutubeDescription: false }, false, 'handlePublishYoutubeDescription/settled');
                }
            },

            handleRestoreDescriptionRevision: async (videoId, revision, onVideoUpdated) => {
                if (!revision?.id) return;
                if (!confirm(`Restore description from ${new Date(revision.created_at).toLocaleString()} (${revision.source})? The current description will be archived first.`)) return;
                set({ restoringDescriptionRevisionId: revision.id }, false, 'handleRestoreDescriptionRevision/pending');
                try {
                    const res = await api.post<Video>(`/videos/${videoId}/description-history/${revision.id}/restore`);
                    onVideoUpdated(res.data);
                    await get().fetchDescriptionHistory(videoId);
                } catch (e: any) {
                    console.error('Failed to restore description', e);
                    alert(e?.response?.data?.detail || 'Failed to restore description');
                } finally {
                    set({ restoringDescriptionRevisionId: null }, false, 'handleRestoreDescriptionRevision/settled');
                }
            },

            handleCopyYoutubeField: async (kind, text) => {
                try {
                    await navigator.clipboard.writeText(text);
                    set({ copiedYoutubeField: kind }, false, 'handleCopyYoutubeField/copied');
                    window.setTimeout(
                        () =>
                            set(
                                (s) => ({ copiedYoutubeField: s.copiedYoutubeField === kind ? null : s.copiedYoutubeField }),
                                false,
                                'handleCopyYoutubeField/reset'
                            ),
                        1500
                    );
                } catch {
                    alert('Failed to copy to clipboard');
                }
            },

            resetYoutubeState: () =>
                set(
                    {
                        generatingYoutubeAi: false,
                        copiedYoutubeField: null,
                        descriptionHistory: [],
                        loadingDescriptionHistory: false,
                        publishingYoutubeDescription: false,
                        restoringDescriptionRevisionId: null,
                    },
                    false,
                    'resetYoutubeState'
                ),
        }),
        { name: 'YoutubeStore' }
    )
);

export function parseYoutubeAiChapters(chaptersJson?: string): VideoChapterSuggestion[] {
    if (!chaptersJson) return [];
    try {
        const parsed = JSON.parse(chaptersJson);
        if (!Array.isArray(parsed)) return [];
        return parsed
            .filter(Boolean)
            .map((ch: any) => ({
                start_seconds: Number(ch.start_seconds ?? 0),
                timestamp: String(ch.timestamp ?? '0:00'),
                title: String(ch.title ?? '').trim(),
                description: ch.description ? String(ch.description) : undefined,
            }))
            .filter((ch: VideoChapterSuggestion) => ch.title);
    } catch {
        return [];
    }
}
