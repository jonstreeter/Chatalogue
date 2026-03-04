import { useState, useEffect } from 'react';
import { AlertTriangle, Loader2, X } from 'lucide-react';
import api from '../lib/api';

interface DeletePreview {
    channel_name: string;
    video_count: number;
    segment_count: number;
    speaker_count: number;
    job_count: number;
    clip_count: number;
    active_jobs: number;
}

interface Props {
    channelId: number;
    channelName: string;
    onClose: () => void;
    onDeleted: () => void;
}

export function DeleteChannelModal({ channelId, channelName, onClose, onDeleted }: Props) {
    const [preview, setPreview] = useState<DeletePreview | null>(null);
    const [loading, setLoading] = useState(true);
    const [deleting, setDeleting] = useState(false);
    const [confirmText, setConfirmText] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        api.get<DeletePreview>(`/channels/${channelId}/delete-preview`)
            .then(res => setPreview(res.data))
            .catch(e => setError(e.response?.data?.detail || 'Failed to load preview'))
            .finally(() => setLoading(false));
    }, [channelId]);

    const canDelete = confirmText === channelName && preview && preview.active_jobs === 0;

    const handleDelete = async () => {
        setDeleting(true);
        setError(null);
        try {
            await api.delete(`/channels/${channelId}`);
            onDeleted();
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Failed to delete channel');
            setDeleting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className="bg-red-50 border-b border-red-100 px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                            <AlertTriangle size={20} className="text-red-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-red-900">Delete Channel</h2>
                            <p className="text-xs text-red-600">This action is irreversible</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-red-400 hover:text-red-600 transition-colors">
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="px-6 py-5 space-y-4">
                    {loading ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 size={24} className="animate-spin text-slate-400" />
                        </div>
                    ) : error && !preview ? (
                        <p className="text-red-600 text-sm">{error}</p>
                    ) : preview ? (
                        <>
                            <p className="text-sm text-slate-600">
                                You are about to permanently delete <span className="font-semibold text-slate-900">{channelName}</span> and all associated data:
                            </p>

                            <div className="bg-slate-50 rounded-xl p-4 space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Videos</span>
                                    <span className="font-medium text-slate-700">{preview.video_count}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Transcript segments</span>
                                    <span className="font-medium text-slate-700">{preview.segment_count.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Speakers</span>
                                    <span className="font-medium text-slate-700">{preview.speaker_count}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Clips</span>
                                    <span className="font-medium text-slate-700">{preview.clip_count}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Jobs</span>
                                    <span className="font-medium text-slate-700">{preview.job_count}</span>
                                </div>
                            </div>

                            {preview.active_jobs > 0 && (
                                <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-800">
                                    <span className="font-medium">Cannot delete:</span> {preview.active_jobs} active job{preview.active_jobs !== 1 ? 's' : ''} running. Cancel them first.
                                </div>
                            )}

                            <div>
                                <label className="block text-sm text-slate-600 mb-1.5">
                                    Type <span className="font-mono font-semibold text-slate-900">{channelName}</span> to confirm:
                                </label>
                                <input
                                    type="text"
                                    value={confirmText}
                                    onChange={e => setConfirmText(e.target.value)}
                                    className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-red-300 focus:border-red-300"
                                    placeholder="Channel name..."
                                    autoFocus
                                    disabled={preview.active_jobs > 0}
                                />
                            </div>

                            {error && <p className="text-red-600 text-sm">{error}</p>}
                        </>
                    ) : null}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors"
                        disabled={deleting}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleDelete}
                        disabled={!canDelete || deleting}
                        className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                    >
                        {deleting ? <Loader2 size={14} className="animate-spin" /> : null}
                        {deleting ? 'Deleting...' : 'Delete Permanently'}
                    </button>
                </div>
            </div>
        </div>
    );
}
