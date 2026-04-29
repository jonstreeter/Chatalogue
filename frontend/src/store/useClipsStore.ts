import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Clip, ClipExportArtifact } from '../types';

export type ClipToken = {
    key: string;
    start: number;
    end: number;
    word: string;
};

export type ClipCropTarget = 'main' | 'top' | 'bottom';

export type ClipDragRect = {
    target: ClipCropTarget;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
};

export type ClipTimelineDrag = {
    handle: 'start' | 'end';
};

export type ClipPreviewLoop = {
    start: number;
    end: number;
    clipId: number;
};

export type ClipBatchPresetKey = 'youtube_landscape' | 'shorts_vertical' | 'square_captioned' | 'audio_focus';
export type ClipUploadPrivacy = 'private' | 'unlisted' | 'public';

type SetStateValue<T> = T | ((previous: T) => T);

export const CLIP_BATCH_PRESETS: Record<ClipBatchPresetKey, { label: string; clipSettings: Partial<Clip>; exportSrt?: boolean; exportVtt?: boolean }> = {
    youtube_landscape: {
        label: 'YouTube Landscape',
        clipSettings: { aspect_ratio: '16:9', burn_captions: false, caption_speaker_labels: true },
        exportSrt: true,
    },
    shorts_vertical: {
        label: 'Shorts Vertical (Burned Captions)',
        clipSettings: { aspect_ratio: '9:16', burn_captions: true, caption_speaker_labels: false },
        exportSrt: true,
    },
    square_captioned: {
        label: 'Square Captioned',
        clipSettings: { aspect_ratio: '1:1', burn_captions: true, caption_speaker_labels: true },
        exportSrt: true,
        exportVtt: true,
    },
    audio_focus: {
        label: 'Podcast Promo (4:5 + Captions)',
        clipSettings: { aspect_ratio: '4:5', burn_captions: true, caption_speaker_labels: true },
        exportSrt: true,
    },
};

function resolveValue<T>(value: SetStateValue<T>, previous: T): T {
    return typeof value === 'function' ? (value as (previous: T) => T)(previous) : value;
}

function normalizeClip(clip: Clip): Clip {
    return {
        ...clip,
        aspect_ratio: clip.aspect_ratio || 'source',
        fade_in_sec: clip.fade_in_sec ?? 0,
        fade_out_sec: clip.fade_out_sec ?? 0,
        burn_captions: clip.burn_captions ?? false,
        caption_speaker_labels: clip.caption_speaker_labels ?? true,
    };
}

function downloadBlobResponse(blob: Blob, filename: string) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
}

function sanitizeClipFilename(value: string, fallback: string) {
    const cleaned = String(value || '')
        .replace(/[\\/:*?"<>|]+/g, '_')
        .replace(/\s+/g, ' ')
        .trim();
    return cleaned || fallback;
}

export interface ClipsState {
    clips: Clip[];
    clipExportArtifactsByClip: Record<number, ClipExportArtifact[]>;
    loadingClips: boolean;
    selectedClipIds: Set<number>;
    editingClipId: number | null;
    clipEditorDraft: Partial<Clip> | null;
    clipEditorTokens: ClipToken[];
    clipEditorRemovedWordKeys: Set<string>;
    clipEditorCropTarget: ClipCropTarget;
    clipEditorDragRect: ClipDragRect | null;
    clipTimelineDrag: ClipTimelineDrag | null;
    savingClipEdit: boolean;
    exportingClipIds: Set<number>;
    batchExporting: boolean;
    batchQueueingRenders: boolean;
    uploadingClipIds: Set<number>;
    batchUploadingClips: boolean;
    clipUploadPrivacy: ClipUploadPrivacy;
    clipPreviewLoop: ClipPreviewLoop | null;
    clipBatchPresetKey: ClipBatchPresetKey;
    clipTitle: string;
    creatingClip: boolean;

    setClips: (value: SetStateValue<Clip[]>) => void;
    setClipExportArtifactsByClip: (value: SetStateValue<Record<number, ClipExportArtifact[]>>) => void;
    setLoadingClips: (loading: boolean) => void;
    setSelectedClipIds: (value: SetStateValue<Set<number>>) => void;
    setEditingClipId: (id: number | null) => void;
    setClipEditorDraft: (value: SetStateValue<Partial<Clip> | null>) => void;
    setClipEditorTokens: (value: SetStateValue<ClipToken[]>) => void;
    setClipEditorRemovedWordKeys: (value: SetStateValue<Set<string>>) => void;
    setClipEditorCropTarget: (target: ClipCropTarget) => void;
    setClipEditorDragRect: (value: SetStateValue<ClipDragRect | null>) => void;
    setClipTimelineDrag: (value: SetStateValue<ClipTimelineDrag | null>) => void;
    setSavingClipEdit: (saving: boolean) => void;
    setExportingClipIds: (value: SetStateValue<Set<number>>) => void;
    setBatchExporting: (exporting: boolean) => void;
    setBatchQueueingRenders: (queueing: boolean) => void;
    setUploadingClipIds: (value: SetStateValue<Set<number>>) => void;
    setBatchUploadingClips: (uploading: boolean) => void;
    setClipUploadPrivacy: (privacy: ClipUploadPrivacy) => void;
    setClipPreviewLoop: (value: SetStateValue<ClipPreviewLoop | null>) => void;
    setClipBatchPresetKey: (key: ClipBatchPresetKey) => void;
    setClipTitle: (title: string) => void;

    fetchClips: (videoId: number) => Promise<void>;
    createClip: (videoId: number, input: { start: number; end: number; title: string }) => Promise<Clip | null>;
    fetchClipExportArtifacts: (videoId: number) => Promise<void>;
    deleteClip: (clipId: number) => Promise<void>;
    saveClipEdit: (clipId: number) => Promise<boolean>;
    downloadArchivedArtifact: (artifact: ClipExportArtifact) => Promise<void>;
    exportClipMp4: (clip: Clip) => Promise<void>;
    queueClipMp4: (clip: Clip) => Promise<void>;
    exportClipCaptions: (clip: Clip, format: 'srt' | 'vtt') => Promise<void>;
    uploadClipToYoutube: (clip: Clip) => Promise<void>;
    batchExportSelectedClips: () => Promise<void>;
    queueRenderSelectedClips: () => Promise<void>;
    batchUploadSelectedClips: () => Promise<void>;
    toggleClipPreviewLoop: (clip: Clip, onSeek: (seconds: number) => void) => void;
    cancelClipEdit: () => void;
    resetClipsState: () => void;
}

export const useClipsStore = create<ClipsState>()(
    devtools(
        (set, get) => ({
            clips: [],
            clipExportArtifactsByClip: {},
            loadingClips: false,
            selectedClipIds: new Set(),
            editingClipId: null,
            clipEditorDraft: null,
            clipEditorTokens: [],
            clipEditorRemovedWordKeys: new Set(),
            clipEditorCropTarget: 'main',
            clipEditorDragRect: null,
            clipTimelineDrag: null,
            savingClipEdit: false,
            exportingClipIds: new Set(),
            batchExporting: false,
            batchQueueingRenders: false,
            uploadingClipIds: new Set(),
            batchUploadingClips: false,
            clipUploadPrivacy: 'private',
            clipPreviewLoop: null,
            clipBatchPresetKey: 'youtube_landscape',
            clipTitle: '',
            creatingClip: false,

            setClips: (value) => set({ clips: resolveValue(value, get().clips) }, false, 'setClips'),
            setClipExportArtifactsByClip: (value) =>
                set(
                    { clipExportArtifactsByClip: resolveValue(value, get().clipExportArtifactsByClip) },
                    false,
                    'setClipExportArtifactsByClip'
                ),
            setLoadingClips: (loading) => set({ loadingClips: loading }, false, 'setLoadingClips'),
            setSelectedClipIds: (value) =>
                set({ selectedClipIds: resolveValue(value, get().selectedClipIds) }, false, 'setSelectedClipIds'),
            setEditingClipId: (id) => set({ editingClipId: id }, false, 'setEditingClipId'),
            setClipEditorDraft: (value) =>
                set({ clipEditorDraft: resolveValue(value, get().clipEditorDraft) }, false, 'setClipEditorDraft'),
            setClipEditorTokens: (value) =>
                set({ clipEditorTokens: resolveValue(value, get().clipEditorTokens) }, false, 'setClipEditorTokens'),
            setClipEditorRemovedWordKeys: (value) =>
                set(
                    { clipEditorRemovedWordKeys: resolveValue(value, get().clipEditorRemovedWordKeys) },
                    false,
                    'setClipEditorRemovedWordKeys'
                ),
            setClipEditorCropTarget: (target) => set({ clipEditorCropTarget: target }, false, 'setClipEditorCropTarget'),
            setClipEditorDragRect: (value) =>
                set({ clipEditorDragRect: resolveValue(value, get().clipEditorDragRect) }, false, 'setClipEditorDragRect'),
            setClipTimelineDrag: (value) =>
                set({ clipTimelineDrag: resolveValue(value, get().clipTimelineDrag) }, false, 'setClipTimelineDrag'),
            setSavingClipEdit: (saving) => set({ savingClipEdit: saving }, false, 'setSavingClipEdit'),
            setExportingClipIds: (value) =>
                set({ exportingClipIds: resolveValue(value, get().exportingClipIds) }, false, 'setExportingClipIds'),
            setBatchExporting: (exporting) => set({ batchExporting: exporting }, false, 'setBatchExporting'),
            setBatchQueueingRenders: (queueing) =>
                set({ batchQueueingRenders: queueing }, false, 'setBatchQueueingRenders'),
            setUploadingClipIds: (value) =>
                set({ uploadingClipIds: resolveValue(value, get().uploadingClipIds) }, false, 'setUploadingClipIds'),
            setBatchUploadingClips: (uploading) => set({ batchUploadingClips: uploading }, false, 'setBatchUploadingClips'),
            setClipUploadPrivacy: (privacy) => set({ clipUploadPrivacy: privacy }, false, 'setClipUploadPrivacy'),
            setClipPreviewLoop: (value) =>
                set({ clipPreviewLoop: resolveValue(value, get().clipPreviewLoop) }, false, 'setClipPreviewLoop'),
            setClipBatchPresetKey: (key) => set({ clipBatchPresetKey: key }, false, 'setClipBatchPresetKey'),
            setClipTitle: (title) => set({ clipTitle: title }, false, 'setClipTitle'),

            fetchClips: async (videoId) => {
                set({ loadingClips: true }, false, 'fetchClips/pending');
                try {
                    const res = await api.get<Clip[]>(`/videos/${videoId}/clips`);
                    set({ clips: res.data.map(normalizeClip) }, false, 'fetchClips/fulfilled');
                } catch (e) {
                    console.error(e);
                } finally {
                    set({ loadingClips: false }, false, 'fetchClips/settled');
                }
            },

            createClip: async (videoId, input) => {
                set({ creatingClip: true }, false, 'createClip/pending');
                try {
                    const res = await api.post<Clip>(`/videos/${videoId}/clips`, {
                        start_time: input.start,
                        end_time: input.end,
                        title: input.title,
                    });
                    const createdClip = normalizeClip(res.data);
                    set(
                        (state) => {
                            const withoutDuplicate = state.clips.filter((clip) => clip.id !== createdClip.id);
                            return {
                                clips: [...withoutDuplicate, createdClip].sort((a, b) => a.start_time - b.start_time),
                            };
                        },
                        false,
                        'createClip/fulfilled'
                    );
                    await get().fetchClips(videoId);
                    return createdClip;
                } catch (e) {
                    console.error(e);
                    alert((e as any)?.response?.data?.detail || 'Failed to save clip');
                    return null;
                } finally {
                    set({ creatingClip: false }, false, 'createClip/settled');
                }
            },

            fetchClipExportArtifacts: async (videoId) => {
                try {
                    const res = await api.get<ClipExportArtifact[]>(`/videos/${videoId}/clip-exports`);
                    const grouped: Record<number, ClipExportArtifact[]> = {};
                    for (const row of res.data || []) {
                        const clipId = Number(row.clip_id);
                        if (!grouped[clipId]) grouped[clipId] = [];
                        grouped[clipId].push(row);
                    }
                    set({ clipExportArtifactsByClip: grouped }, false, 'fetchClipExportArtifacts/fulfilled');
                } catch (e) {
                    console.error('Failed to fetch clip export artifacts:', e);
                    set({ clipExportArtifactsByClip: {} }, false, 'fetchClipExportArtifacts/rejected');
                }
            },

            deleteClip: async (clipId) => {
                if (!confirm('Are you sure you want to delete this clip?')) return;
                try {
                    await api.delete(`/clips/${clipId}`);
                    set(
                        (state) => ({
                            clips: state.clips.filter((clip) => clip.id !== clipId),
                            selectedClipIds: new Set([...state.selectedClipIds].filter((id) => id !== clipId)),
                            clipPreviewLoop: state.clipPreviewLoop?.clipId === clipId ? null : state.clipPreviewLoop,
                        }),
                        false,
                        'deleteClip/fulfilled'
                    );
                } catch (e) {
                    console.error(e);
                    alert('Failed to delete clip');
                }
            },

            saveClipEdit: async (clipId) => {
                const draft = get().clipEditorDraft;
                if (!draft) return false;
                if (!draft.title || !String(draft.title).trim()) {
                    alert('Clip title is required');
                    return false;
                }
                if ((draft.end_time ?? 0) <= (draft.start_time ?? 0)) {
                    alert('End time must be after start time');
                    return false;
                }
                set({ savingClipEdit: true }, false, 'saveClipEdit/pending');
                try {
                    const payload = {
                        start_time: Number(draft.start_time),
                        end_time: Number(draft.end_time),
                        title: String(draft.title),
                        aspect_ratio: draft.aspect_ratio || 'source',
                        crop_x: draft.crop_x ?? null,
                        crop_y: draft.crop_y ?? null,
                        crop_w: draft.crop_w ?? null,
                        crop_h: draft.crop_h ?? null,
                        portrait_split_enabled: !!draft.portrait_split_enabled,
                        portrait_top_crop_x: draft.portrait_top_crop_x ?? null,
                        portrait_top_crop_y: draft.portrait_top_crop_y ?? null,
                        portrait_top_crop_w: draft.portrait_top_crop_w ?? null,
                        portrait_top_crop_h: draft.portrait_top_crop_h ?? null,
                        portrait_bottom_crop_x: draft.portrait_bottom_crop_x ?? null,
                        portrait_bottom_crop_y: draft.portrait_bottom_crop_y ?? null,
                        portrait_bottom_crop_w: draft.portrait_bottom_crop_w ?? null,
                        portrait_bottom_crop_h: draft.portrait_bottom_crop_h ?? null,
                        script_edits_json: draft.script_edits_json ?? null,
                        fade_in_sec: Number(draft.fade_in_sec ?? 0),
                        fade_out_sec: Number(draft.fade_out_sec ?? 0),
                        burn_captions: !!draft.burn_captions,
                        caption_speaker_labels: !!draft.caption_speaker_labels,
                    };
                    const res = await api.patch<Clip>(`/clips/${clipId}`, payload);
                    const saved = normalizeClip(res.data);
                    set(
                        (state) => ({ clips: state.clips.map((clip) => (clip.id === clipId ? saved : clip)) }),
                        false,
                        'saveClipEdit/fulfilled'
                    );
                    get().cancelClipEdit();
                    return true;
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to save clip edits');
                    return false;
                } finally {
                    set({ savingClipEdit: false }, false, 'saveClipEdit/settled');
                }
            },

            downloadArchivedArtifact: async (artifact) => {
                try {
                    const response = await api.get(`/clip-exports/${artifact.id}/download`, { responseType: 'blob' });
                    downloadBlobResponse(response.data, artifact.file_name || `clip_export_${artifact.id}.${artifact.format || 'bin'}`);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to download archived export');
                }
            },

            exportClipMp4: async (clip) => {
                get().setExportingClipIds((prev) => {
                    const next = new Set(prev);
                    next.add(clip.id);
                    return next;
                });
                try {
                    const response = await api.post(`/clips/${clip.id}/export/mp4`, null, { responseType: 'blob' });
                    downloadBlobResponse(response.data, `${sanitizeClipFilename(clip.title || `clip_${clip.id}`, `clip_${clip.id}`)}.mp4`);
                    await get().fetchClipExportArtifacts(clip.video_id);
                } catch (e: any) {
                    const detail = e?.response?.data?.detail || 'Failed to export MP4';
                    try {
                        await api.post(`/clips/${clip.id}/export/mp4/queue`);
                        alert(`${detail}\n\nQueued a background render job for "${clip.title || `Clip #${clip.id}`}". You can download it from archived outputs when complete.`);
                    } catch {
                        alert(detail);
                    }
                } finally {
                    get().setExportingClipIds((prev) => {
                        const next = new Set(prev);
                        next.delete(clip.id);
                        return next;
                    });
                }
            },

            queueClipMp4: async (clip) => {
                get().setExportingClipIds((prev) => {
                    const next = new Set(prev);
                    next.add(clip.id);
                    return next;
                });
                try {
                    await api.post(`/clips/${clip.id}/export/mp4/queue`);
                    alert(`Queued render job for "${clip.title || `Clip #${clip.id}`}". Check Job Queue > Clip Export.`);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to queue MP4 export');
                } finally {
                    get().setExportingClipIds((prev) => {
                        const next = new Set(prev);
                        next.delete(clip.id);
                        return next;
                    });
                }
            },

            exportClipCaptions: async (clip, format) => {
                get().setExportingClipIds((prev) => {
                    const next = new Set(prev);
                    next.add(clip.id);
                    return next;
                });
                try {
                    const response = await api.post(`/clips/${clip.id}/export/captions`, { format }, { responseType: 'blob' });
                    downloadBlobResponse(response.data, `${sanitizeClipFilename(clip.title || `clip_${clip.id}`, `clip_${clip.id}`)}.${format}`);
                    await get().fetchClipExportArtifacts(clip.video_id);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || `Failed to export ${format.toUpperCase()}`);
                } finally {
                    get().setExportingClipIds((prev) => {
                        const next = new Set(prev);
                        next.delete(clip.id);
                        return next;
                    });
                }
            },

            uploadClipToYoutube: async (clip) => {
                get().setUploadingClipIds((prev) => {
                    const next = new Set(prev);
                    next.add(clip.id);
                    return next;
                });
                try {
                    const res = await api.post(`/clips/${clip.id}/youtube/upload`, {
                        privacy_status: get().clipUploadPrivacy,
                    });
                    const watchUrl = res.data?.uploaded_watch_url;
                    const title = res.data?.uploaded_title || clip.title;
                    if (watchUrl) {
                        if (confirm(`Uploaded "${title}". Open in YouTube Studio/watch page now?`)) {
                            window.open(watchUrl, '_blank', 'noopener,noreferrer');
                        }
                    } else {
                        alert(`Uploaded "${title}" successfully.`);
                    }
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to upload clip to YouTube');
                } finally {
                    get().setUploadingClipIds((prev) => {
                        const next = new Set(prev);
                        next.delete(clip.id);
                        return next;
                    });
                }
            },

            batchExportSelectedClips: async () => {
                const ids = Array.from(get().selectedClipIds);
                if (ids.length === 0) {
                    alert('Select one or more clips first');
                    return;
                }
                const presetKey = get().clipBatchPresetKey;
                const preset = CLIP_BATCH_PRESETS[presetKey];
                if (!confirm(`Apply preset "${preset.label}" and export ${ids.length} clip(s)?`)) return;
                set({ batchExporting: true }, false, 'batchExportSelectedClips/pending');
                try {
                    for (const clipId of ids) {
                        const res = await api.post<Clip>(`/clips/${clipId}/apply-export-preset`, preset.clipSettings);
                        const updated = normalizeClip(res.data);
                        set(
                            (state) => ({ clips: state.clips.map((clip) => (clip.id === clipId ? updated : clip)) }),
                            false,
                            'batchExportSelectedClips/applyPreset'
                        );
                        await get().exportClipMp4(updated);
                        if (preset.exportSrt) await get().exportClipCaptions(updated, 'srt');
                        if (preset.exportVtt) await get().exportClipCaptions(updated, 'vtt');
                    }
                } finally {
                    set({ batchExporting: false }, false, 'batchExportSelectedClips/settled');
                }
            },

            queueRenderSelectedClips: async () => {
                const ids = Array.from(get().selectedClipIds);
                if (ids.length === 0) {
                    alert('Select one or more clips first');
                    return;
                }
                if (!confirm(`Queue MP4 render jobs for ${ids.length} selected clip(s)?`)) return;
                set({ batchQueueingRenders: true }, false, 'queueRenderSelectedClips/pending');
                try {
                    let queued = 0;
                    for (const clipId of ids) {
                        try {
                            await api.post(`/clips/${clipId}/export/mp4/queue`);
                            queued += 1;
                        } catch {
                            // Continue queueing other clips; summarize at end.
                        }
                    }
                    alert(`Queued ${queued}/${ids.length} clip render job(s). Check Job Queue -> Clip Export.`);
                } finally {
                    set({ batchQueueingRenders: false }, false, 'queueRenderSelectedClips/settled');
                }
            },

            batchUploadSelectedClips: async () => {
                const ids = Array.from(get().selectedClipIds);
                if (ids.length === 0) {
                    alert('Select one or more clips first');
                    return;
                }
                const privacy = get().clipUploadPrivacy;
                if (!confirm(`Upload ${ids.length} selected clip(s) to your connected YouTube channel as ${privacy}?`)) return;
                set({ batchUploadingClips: true }, false, 'batchUploadSelectedClips/pending');
                try {
                    const res = await api.post('/clips/youtube/upload-batch', {
                        clip_ids: ids,
                        privacy_status: privacy,
                    });
                    const uploaded = Number(res.data?.uploaded || 0);
                    const failed = Number(res.data?.failed || 0);
                    alert(`Batch upload finished. Uploaded: ${uploaded}, Failed: ${failed}.`);
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to batch upload clips');
                } finally {
                    set({ batchUploadingClips: false }, false, 'batchUploadSelectedClips/settled');
                }
            },

            toggleClipPreviewLoop: (clip, onSeek) => {
                if (get().clipPreviewLoop?.clipId === clip.id) {
                    set({ clipPreviewLoop: null }, false, 'toggleClipPreviewLoop/stop');
                    return;
                }
                set(
                    { clipPreviewLoop: { start: clip.start_time, end: clip.end_time, clipId: clip.id } },
                    false,
                    'toggleClipPreviewLoop/start'
                );
                onSeek(clip.start_time);
            },

            cancelClipEdit: () =>
                set(
                    {
                        editingClipId: null,
                        clipEditorDraft: null,
                        clipEditorTokens: [],
                        clipEditorRemovedWordKeys: new Set(),
                        clipEditorCropTarget: 'main',
                        clipEditorDragRect: null,
                        clipTimelineDrag: null,
                    },
                    false,
                    'cancelClipEdit'
                ),

            resetClipsState: () =>
                set(
                    {
                        clips: [],
                        clipExportArtifactsByClip: {},
                        loadingClips: false,
                        selectedClipIds: new Set(),
                        editingClipId: null,
                        clipEditorDraft: null,
                        clipEditorTokens: [],
                        clipEditorRemovedWordKeys: new Set(),
                        clipEditorCropTarget: 'main',
                        clipEditorDragRect: null,
                        clipTimelineDrag: null,
                        savingClipEdit: false,
                        exportingClipIds: new Set(),
                        batchExporting: false,
                        batchQueueingRenders: false,
                        uploadingClipIds: new Set(),
                        batchUploadingClips: false,
                        clipUploadPrivacy: 'private',
                        clipPreviewLoop: null,
                        clipBatchPresetKey: 'youtube_landscape',
                        clipTitle: '',
                        creatingClip: false,
                    },
                    false,
                    'resetClipsState'
                ),
        }),
        { name: 'ClipsStore' }
    )
);
