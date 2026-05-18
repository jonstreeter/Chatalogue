import { useEffect } from 'react';
import { Clock, Download, Loader2, Pencil, Play, Scissors, Trash2, Upload } from 'lucide-react';
import { formatTime } from '../../../lib/formatters';
import {
    CLIP_BATCH_PRESETS,
    useClipsStore,
    type ClipBatchPresetKey,
    type ClipUploadPrivacy,
} from '../../../store/useClipsStore';

type Props = {
    videoId: number;
    isActive: boolean;
    onSeek: (seconds: number) => void;
};

function formatFileSize(bytes?: number) {
    if (!bytes || bytes <= 0) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let idx = 0;
    while (size >= 1024 && idx < units.length - 1) {
        size /= 1024;
        idx += 1;
    }
    return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

export function ClipsTab({ videoId, isActive, onSeek }: Props) {
    const clips = useClipsStore((s) => s.clips);
    const clipExportArtifactsByClip = useClipsStore((s) => s.clipExportArtifactsByClip);
    const loadingClips = useClipsStore((s) => s.loadingClips);
    const selectedClipIds = useClipsStore((s) => s.selectedClipIds);
    const editingClipId = useClipsStore((s) => s.editingClipId);
    const clipEditorDraft = useClipsStore((s) => s.clipEditorDraft);
    const exportingClipIds = useClipsStore((s) => s.exportingClipIds);
    const batchExporting = useClipsStore((s) => s.batchExporting);
    const batchQueueingRenders = useClipsStore((s) => s.batchQueueingRenders);
    const uploadingClipIds = useClipsStore((s) => s.uploadingClipIds);
    const batchUploadingClips = useClipsStore((s) => s.batchUploadingClips);
    const clipUploadPrivacy = useClipsStore((s) => s.clipUploadPrivacy);
    const clipPreviewLoop = useClipsStore((s) => s.clipPreviewLoop);
    const clipBatchPresetKey = useClipsStore((s) => s.clipBatchPresetKey);
    const setSelectedClipIds = useClipsStore((s) => s.setSelectedClipIds);
    const setClipUploadPrivacy = useClipsStore((s) => s.setClipUploadPrivacy);
    const setClipBatchPresetKey = useClipsStore((s) => s.setClipBatchPresetKey);
    const cancelClipEdit = useClipsStore((s) => s.cancelClipEdit);
    const deleteClip = useClipsStore((s) => s.deleteClip);
    const downloadArchivedArtifact = useClipsStore((s) => s.downloadArchivedArtifact);
    const exportClipMp4 = useClipsStore((s) => s.exportClipMp4);
    const queueClipMp4 = useClipsStore((s) => s.queueClipMp4);
    const exportClipCaptions = useClipsStore((s) => s.exportClipCaptions);
    const uploadClipToYoutube = useClipsStore((s) => s.uploadClipToYoutube);
    const batchExportSelectedClips = useClipsStore((s) => s.batchExportSelectedClips);
    const queueRenderSelectedClips = useClipsStore((s) => s.queueRenderSelectedClips);
    const batchUploadSelectedClips = useClipsStore((s) => s.batchUploadSelectedClips);
    const startClipEdit = useClipsStore((s) => s.startClipEdit);
    const toggleClipPreviewLoop = useClipsStore((s) => s.toggleClipPreviewLoop);

    useEffect(() => {
        if (!isActive) return;
        void useClipsStore.getState().fetchClips(videoId);
        void useClipsStore.getState().fetchClipExportArtifacts(videoId);
    }, [isActive, videoId]);

    if (!isActive) return null;

    const toggleClipSelected = (clipId: number) => {
        setSelectedClipIds((prev) => {
            const next = new Set(prev);
            if (next.has(clipId)) next.delete(clipId);
            else next.add(clipId);
            return next;
        });
    };

    return (
        <div className="h-full overflow-y-auto p-4 space-y-3">
            {clips.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-3 space-y-3 sticky top-0 z-10 shadow-sm">
                    <div className="flex items-center justify-between gap-2">
                        <div>
                            <div className="text-sm font-semibold text-slate-800">Batch Export + Presets</div>
                            <div className="text-xs text-slate-500">Select clips, apply a preset, export MP4 (+ caption sidecars).</div>
                        </div>
                        <button
                            onClick={() => setSelectedClipIds(new Set(clips.map((clip) => clip.id)))}
                            className="text-xs px-2 py-1 rounded-md bg-slate-100 text-slate-600 hover:bg-slate-200"
                        >
                            Select All
                        </button>
                    </div>
                    <div className="space-y-2">
                        <select
                            value={clipBatchPresetKey}
                            onChange={(e) => setClipBatchPresetKey(e.target.value as ClipBatchPresetKey)}
                            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                        >
                            {Object.entries(CLIP_BATCH_PRESETS).map(([key, preset]) => (
                                <option key={key} value={key}>{preset.label}</option>
                            ))}
                        </select>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            <button
                                onClick={() => void batchExportSelectedClips()}
                                disabled={batchExporting || batchUploadingClips || batchQueueingRenders || selectedClipIds.size === 0}
                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 whitespace-nowrap"
                            >
                                {batchExporting ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                                Export Selected ({selectedClipIds.size})
                            </button>
                            <button
                                onClick={() => void queueRenderSelectedClips()}
                                disabled={batchQueueingRenders || batchExporting || batchUploadingClips || selectedClipIds.size === 0}
                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
                                title="Queue server-side clip render jobs (parallel with other queues)"
                            >
                                {batchQueueingRenders ? <Loader2 size={13} className="animate-spin" /> : <Clock size={13} />}
                                Queue Renders ({selectedClipIds.size})
                            </button>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-[160px_minmax(0,1fr)] gap-2">
                            <select
                                value={clipUploadPrivacy}
                                onChange={(e) => setClipUploadPrivacy(e.target.value as ClipUploadPrivacy)}
                                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                title="YouTube upload privacy"
                            >
                                <option value="private">Upload Private</option>
                                <option value="unlisted">Upload Unlisted</option>
                                <option value="public">Upload Public</option>
                            </select>
                            <button
                                onClick={() => void batchUploadSelectedClips()}
                                disabled={batchUploadingClips || batchExporting || selectedClipIds.size === 0}
                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 whitespace-nowrap"
                                title="Upload selected clips directly to your connected YouTube channel"
                            >
                                {batchUploadingClips ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />}
                                Upload Selected ({selectedClipIds.size})
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {loadingClips && clips.length === 0 ? (
                <div className="flex justify-center p-8"><Loader2 className="animate-spin text-slate-300" /></div>
            ) : clips.length === 0 ? (
                <div className="text-center mt-10 p-8 border-2 border-dashed border-slate-200 rounded-xl">
                    <Scissors className="mx-auto text-slate-300 mb-2" size={32} />
                    <h3 className="text-slate-500 font-medium">No clips yet</h3>
                    <p className="text-sm text-slate-400 mt-1">Select text in the transcript to create a clip.</p>
                </div>
            ) : (
                clips.map((clip) => {
                    const isEditing = editingClipId === clip.id;
                    const draft = isEditing && clipEditorDraft ? clipEditorDraft : clip;
                    const isExporting = exportingClipIds.has(clip.id);
                    const isUploading = uploadingClipIds.has(clip.id);
                    const isLooping = clipPreviewLoop?.clipId === clip.id;
                    const artifacts = (clipExportArtifactsByClip[clip.id] || []).slice(0, 4);

                    return (
                        <div key={clip.id} className={`bg-white p-3 rounded-lg border shadow-sm ${isEditing ? 'border-purple-300 ring-1 ring-purple-200' : 'border-slate-200'}`}>
                            <div className="flex gap-3 items-start">
                                <input
                                    type="checkbox"
                                    checked={selectedClipIds.has(clip.id)}
                                    onChange={() => toggleClipSelected(clip.id)}
                                    className="mt-2 h-4 w-4 rounded border-slate-300 text-purple-600"
                                />
                                <div
                                    className="w-24 h-16 bg-slate-100 rounded overflow-hidden flex-shrink-0 relative cursor-pointer"
                                    onClick={() => onSeek((draft.start_time as number) ?? clip.start_time)}
                                    title="Jump to clip start in player"
                                >
                                    <div className="w-full h-full bg-gradient-to-br from-purple-100 to-indigo-100 flex items-center justify-center">
                                        <Play size={20} className="text-purple-400" />
                                    </div>
                                </div>
                                <div className="flex-1 min-w-0 space-y-2">
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="min-w-0">
                                            <h4 className="text-sm font-medium text-slate-800 line-clamp-1">{clip.title}</h4>
                                            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 font-mono mt-0.5">
                                                <span>{formatTime(clip.start_time)} - {formatTime(clip.end_time)}</span>
                                                <span className="w-1 h-1 bg-slate-300 rounded-full" />
                                                <span>{(clip.end_time - clip.start_time).toFixed(1)}s</span>
                                                <span className="w-1 h-1 bg-slate-300 rounded-full" />
                                                <span>{(clip.aspect_ratio || 'source').toUpperCase()}</span>
                                                {clip.burn_captions && <span className="px-1 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 text-[10px]">burned captions</span>}
                                                {clip.portrait_split_enabled && String(clip.aspect_ratio || '').toLowerCase() === '9:16' && (
                                                    <span className="px-1 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[10px]">split</span>
                                                )}
                                                {clip.script_edits_json && (
                                                    <span className="px-1 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px]">text-edited</span>
                                                )}
                                                {((clip.fade_in_sec || 0) > 0 || (clip.fade_out_sec || 0) > 0) && (
                                                    <span className="px-1 py-0.5 rounded bg-fuchsia-50 text-fuchsia-700 border border-fuchsia-200 text-[10px]">
                                                        fades {Number(clip.fade_in_sec || 0).toFixed(1)}/{Number(clip.fade_out_sec || 0).toFixed(1)}s
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={() => toggleClipPreviewLoop(clip, onSeek)}
                                                className={`px-2 py-1 text-xs rounded-md ${isLooping ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                                                title="Loop preview this clip in the main player"
                                            >
                                                {isLooping ? 'Stop Loop' : 'Loop'}
                                            </button>
                                            <button
                                                onClick={() => isEditing ? cancelClipEdit() : startClipEdit(clip)}
                                                className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded"
                                                title="Edit clip trim/export settings"
                                            >
                                                <Pencil size={14} />
                                            </button>
                                            <button
                                                onClick={() => void deleteClip(clip.id)}
                                                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded"
                                                title="Delete clip"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="flex flex-wrap gap-2">
                                        <button
                                            onClick={() => void queueClipMp4(clip)}
                                            disabled={isExporting || isUploading}
                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                        >
                                            {isExporting ? <Loader2 size={12} className="animate-spin" /> : <Clock size={12} />}
                                            Queue MP4
                                        </button>
                                        <button
                                            onClick={() => void exportClipMp4(clip)}
                                            disabled={isExporting || isUploading}
                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50"
                                        >
                                            {isExporting ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
                                            Download MP4
                                        </button>
                                        <button
                                            onClick={() => void uploadClipToYoutube(clip)}
                                            disabled={isExporting || isUploading || batchUploadingClips}
                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                                            title="Upload this clip to your connected YouTube channel"
                                        >
                                            {isUploading ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                                            Upload
                                        </button>
                                        <button
                                            onClick={() => void exportClipCaptions(clip, 'srt')}
                                            disabled={isExporting || isUploading}
                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                        >
                                            SRT
                                        </button>
                                        <button
                                            onClick={() => void exportClipCaptions(clip, 'vtt')}
                                            disabled={isExporting || isUploading}
                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                        >
                                            VTT
                                        </button>
                                    </div>

                                    {artifacts.length === 0 ? (
                                        <div className="text-[11px] text-slate-400">
                                            No archived exports yet. Export once to save for re-download.
                                        </div>
                                    ) : (
                                        <div className="space-y-1">
                                            <div className="text-[11px] font-medium text-slate-500">Archived outputs</div>
                                            <div className="flex flex-wrap gap-1.5">
                                                {artifacts.map((artifact) => (
                                                    <button
                                                        key={artifact.id}
                                                        onClick={() => void downloadArchivedArtifact(artifact)}
                                                        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] border border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                                                        title={`${artifact.file_name} - ${new Date(artifact.created_at).toLocaleString()}`}
                                                    >
                                                        <Download size={11} />
                                                        {artifact.format.toUpperCase()} {formatFileSize(artifact.file_size_bytes)}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {isEditing && (
                                        <div className="text-[11px] text-purple-700 bg-purple-50 border border-purple-200 rounded-lg px-2.5 py-2">
                                            Editing in main preview panel. Scroll/right pane to adjust trim, crop, split layout, and burned captions.
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })
            )}
        </div>
    );
}
