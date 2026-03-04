import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import api from '../../lib/api';
import type { ChannelClip } from '../../types';
import { Loader2, Scissors, Download, FileText, Video as VideoIcon, Upload } from 'lucide-react';

export function ChannelClips() {
    const { id } = useParams<{ id: string }>();
    const [clips, setClips] = useState<ChannelClip[]>([]);
    const [loading, setLoading] = useState(true);
    const [query, setQuery] = useState('');
    const [exportingIds, setExportingIds] = useState<Set<number>>(new Set());
    const [uploadingIds, setUploadingIds] = useState<Set<number>>(new Set());

    const fetchClips = async () => {
        if (!id) return;
        setLoading(true);
        try {
            const res = await api.get<ChannelClip[]>(`/channels/${id}/clips`);
            setClips(Array.isArray(res.data) ? res.data : []);
        } catch (e) {
            console.error('Failed to fetch channel clips', e);
            setClips([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void fetchClips();
    }, [id]);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return clips;
        return clips.filter(c =>
            (c.title || '').toLowerCase().includes(q) ||
            (c.video_title || '').toLowerCase().includes(q),
        );
    }, [clips, query]);

    const formatTime = (seconds: number) => {
        const total = Math.max(0, Math.floor(seconds));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        return h > 0
            ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
            : `${m}:${String(s).padStart(2, '0')}`;
    };

    const setBusy = (id: number, setFn: (updater: (prev: Set<number>) => Set<number>) => void, busy: boolean) => {
        setFn(prev => {
            const next = new Set(prev);
            if (busy) next.add(id); else next.delete(id);
            return next;
        });
    };

    const downloadBlob = (blob: Blob, filename: string) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    };

    const exportClipMp4 = async (clip: ChannelClip) => {
        setBusy(clip.id, setExportingIds, true);
        try {
            const res = await api.post(`/clips/${clip.id}/export/mp4`, null, { responseType: 'blob' });
            downloadBlob(res.data, `${(clip.title || `clip_${clip.id}`).replace(/[\\/:*?"<>|]/g, '_')}.mp4`);
        } catch (e: any) {
            const detail = e?.response?.data?.detail || 'Failed to export clip';
            try {
                await api.post(`/clips/${clip.id}/export/mp4/queue`);
                alert(`${detail}\n\nQueued a background render job for "${clip.title || `Clip #${clip.id}`}".`);
            } catch {
                alert(detail);
            }
        } finally {
            setBusy(clip.id, setExportingIds, false);
        }
    };

    const exportClipCaptions = async (clip: ChannelClip, format: 'srt' | 'vtt') => {
        setBusy(clip.id, setExportingIds, true);
        try {
            const res = await api.post(`/clips/${clip.id}/export/captions`, { format }, { responseType: 'blob' });
            downloadBlob(res.data, `${(clip.title || `clip_${clip.id}`).replace(/[\\/:*?"<>|]/g, '_')}.${format}`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || `Failed to export ${format.toUpperCase()}`);
        } finally {
            setBusy(clip.id, setExportingIds, false);
        }
    };

    const uploadClip = async (clip: ChannelClip) => {
        setBusy(clip.id, setUploadingIds, true);
        try {
            const res = await api.post(`/clips/${clip.id}/youtube/upload`, { privacy_status: 'private' });
            const url = res.data?.uploaded_watch_url;
            if (url && confirm('Uploaded clip to YouTube. Open it now?')) {
                window.open(url, '_blank', 'noopener,noreferrer');
            }
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to upload clip');
        } finally {
            setBusy(clip.id, setUploadingIds, false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-slate-400">
                <Loader2 className="animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div className="glass-panel rounded-xl p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <div>
                    <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                        <Scissors size={18} className="text-purple-500" />
                        Channel Clips
                    </h3>
                    <p className="text-sm text-slate-500">{clips.length} saved clip{clips.length === 1 ? '' : 's'}</p>
                </div>
                <div className="w-full md:w-80">
                    <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border border-slate-200 bg-white text-sm"
                        placeholder="Search clips or episode title..."
                    />
                </div>
            </div>

            {filtered.length === 0 ? (
                <div className="glass-panel rounded-xl p-10 text-center">
                    <Scissors size={34} className="mx-auto text-slate-300 mb-3" />
                    <div className="text-slate-600 font-medium">No clips found</div>
                    <div className="text-sm text-slate-400 mt-1">Create clips from an episode transcript, then they appear here.</div>
                </div>
            ) : (
                <div className="space-y-3">
                    {filtered.map((clip) => {
                        const exporting = exportingIds.has(clip.id);
                        const uploading = uploadingIds.has(clip.id);
                        return (
                            <div key={clip.id} className="glass-panel rounded-xl p-3.5 border border-slate-200 bg-white">
                                <div className="flex flex-col gap-3 sm:flex-row">
                                    <div className="w-full h-36 sm:w-28 sm:h-16 rounded overflow-hidden border border-slate-200 bg-slate-100 shrink-0">
                                        {clip.video_thumbnail_url ? (
                                            <img src={clip.video_thumbnail_url} alt="" className="w-full h-full object-cover" />
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center text-slate-300"><VideoIcon size={18} /></div>
                                        )}
                                    </div>
                                    <div className="flex-1 min-w-0 space-y-2">
                                        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                                            <div className="min-w-0">
                                                <div className="text-sm font-semibold text-slate-800 truncate">{clip.title || `Clip #${clip.id}`}</div>
                                                <div className="text-xs text-slate-500 truncate">{clip.video_title}</div>
                                                <div className="text-xs font-mono text-slate-400 mt-0.5">
                                                    {formatTime(clip.start_time)} - {formatTime(clip.end_time)} ({(clip.end_time - clip.start_time).toFixed(1)}s)
                                                </div>
                                            </div>
                                            <Link
                                                to={`/video/${clip.video_id}?t=${Math.floor(clip.start_time)}`}
                                                className="inline-flex min-h-9 self-start items-center text-xs px-3 py-1.5 rounded-md bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 shrink-0"
                                            >
                                                Open Episode
                                            </Link>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            <button
                                                onClick={() => void exportClipMp4(clip)}
                                                disabled={exporting || uploading}
                                                className="inline-flex min-h-9 items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50"
                                            >
                                                {exporting ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
                                                MP4
                                            </button>
                                            <button
                                                onClick={() => void exportClipCaptions(clip, 'srt')}
                                                disabled={exporting || uploading}
                                                className="inline-flex min-h-9 items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                            >
                                                <FileText size={12} /> SRT
                                            </button>
                                            <button
                                                onClick={() => void exportClipCaptions(clip, 'vtt')}
                                                disabled={exporting || uploading}
                                                className="inline-flex min-h-9 items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                            >
                                                <FileText size={12} /> VTT
                                            </button>
                                            <button
                                                onClick={() => void uploadClip(clip)}
                                                disabled={exporting || uploading}
                                                className="inline-flex min-h-9 items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                                            >
                                                {uploading ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                                                Upload
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
