
import { NavLink, Outlet, useParams, useNavigate } from 'react-router-dom';
import { Video, FileText, Users, Scissors, ArrowLeft, Trash2, Download, Loader2, Link2, CheckCircle2, AlertCircle, AudioLines, RotateCcw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import api from '../lib/api';
import { DeleteChannelModal } from '../components/DeleteChannelModal';
import type { Channel, TranscriptRepairBulkQueueResponse, TranscriptDiarizationRebuildBulkQueueResponse, TranscriptRetranscriptionBulkQueueResponse } from '../types';

const tabs = [
    { path: '', label: 'Videos', icon: Video },
    { path: 'transcripts', label: 'Transcripts', icon: FileText },
    { path: 'speakers', label: 'Speakers', icon: Users },
    { path: 'clips', label: 'Clips', icon: Scissors },
];

export function ChannelDetail() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [channelName, setChannelName] = useState<string>('');
    const [channelIconUrl, setChannelIconUrl] = useState<string>('');
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [exporting, setExporting] = useState(false);
    const [showBatchPublishModal, setShowBatchPublishModal] = useState(false);
    const [consolidatingTranscripts, setConsolidatingTranscripts] = useState(false);
    const [queueingTranscriptRepairs, setQueueingTranscriptRepairs] = useState(false);
    const [queueingDiarizationRebuilds, setQueueingDiarizationRebuilds] = useState(false);
    const [queueingFullRetranscriptions, setQueueingFullRetranscriptions] = useState(false);

    useEffect(() => {
        if (id) {
            api.get<Channel>(`/channels/${id}`)
                .then(res => {
                    if (res.data?.name) setChannelName(res.data.name);
                    setChannelIconUrl(res.data?.icon_url || '');
                })
                .catch(() => { });
        }
    }, [id]);

    const channelInitial = (channelName?.trim()?.[0] || 'C').toUpperCase();

    const handleExport = async () => {
        setExporting(true);
        try {
            const res = await api.get(`/channels/${id}/export`);
            const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${channelName || 'channel'}_archive.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Export failed', e);
        } finally {
            setExporting(false);
        }
    };

    const handleConsolidateTranscripts = async () => {
        if (!id) return;
        if (!confirm('Post-process existing transcripts in this channel to merge same-speaker fragments and smooth tiny diarization cuts?')) return;
        setConsolidatingTranscripts(true);
        try {
            const res = await api.post(`/channels/${id}/consolidate-transcripts`);
            const changed = Number(res?.data?.counts?.changed || 0);
            const merged = Number(res?.data?.counts?.merged_segments || 0);
            const reassigned = Number(res?.data?.counts?.reassigned_islands || 0);
            alert(`Transcript consolidation complete. ${changed} video(s) changed, ${merged} segment merges, ${reassigned} short speaker-island reassignment${reassigned === 1 ? '' : 's'}.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to consolidate channel transcripts');
        } finally {
            setConsolidatingTranscripts(false);
        }
    };

    const handleQueueTranscriptRepairs = async () => {
        if (!id) return;
        if (!confirm('Queue low-risk transcript repairs for eligible processed episodes in this channel?')) return;
        setQueueingTranscriptRepairs(true);
        try {
            const res = await api.post<TranscriptRepairBulkQueueResponse>(`/channels/${id}/transcript-repair/queue`, { limit: 250 });
            alert(`Queued ${res.data.queued} low-risk repair job(s). Skipped ${res.data.skipped_active} active, ${res.data.skipped_no_segments} without transcripts, ${res.data.skipped_not_low_risk} not classified as low-risk repair.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue low-risk transcript repairs');
        } finally {
            setQueueingTranscriptRepairs(false);
        }
    };

    const handleQueueDiarizationRebuilds = async () => {
        if (!id) return;
        if (!confirm('Queue diarization rebuilds for eligible processed episodes in this channel?')) return;
        setQueueingDiarizationRebuilds(true);
        try {
            const res = await api.post<TranscriptDiarizationRebuildBulkQueueResponse>(`/channels/${id}/transcript-diarization-rebuild/queue`, { limit: 250 });
            alert(`Queued ${res.data.queued} diarization rebuild job(s). Skipped ${res.data.skipped_active} active, ${res.data.skipped_no_raw_transcript} without raw transcripts, ${res.data.skipped_not_diarization_rebuild} not classified for rebuild.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue diarization rebuilds');
        } finally {
            setQueueingDiarizationRebuilds(false);
        }
    };

    const handleQueueFullRetranscriptions = async () => {
        if (!id) return;
        if (!confirm('Queue full retranscriptions for eligible processed episodes in this channel? This forces a fresh ASR pass.')) return;
        setQueueingFullRetranscriptions(true);
        try {
            const res = await api.post<TranscriptRetranscriptionBulkQueueResponse>(`/channels/${id}/transcript-retranscribe/queue`, { limit: 250 });
            alert(`Queued ${res.data.queued} full retranscription job(s). Skipped ${res.data.skipped_active} active, ${res.data.skipped_unprocessed} unprocessed, ${res.data.skipped_muted} muted, ${res.data.skipped_not_full_retranscription} not classified for full retranscription.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue full retranscriptions');
        } finally {
            setQueueingFullRetranscriptions(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Back navigation + channel context + actions */}
            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="min-w-0">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-2 text-slate-500 hover:text-slate-700 transition-colors text-sm"
                    >
                        <ArrowLeft size={16} />
                        Back to Channels
                    </Link>
                    <div className="mt-2 flex items-center gap-3 min-w-0">
                        <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-white p-0.5 shadow-sm ring-1 ring-slate-200 overflow-hidden shrink-0">
                            {channelIconUrl ? (
                                <img
                                    src={channelIconUrl}
                                    alt=""
                                    className="w-full h-full object-cover rounded-[10px]"
                                    loading="lazy"
                                    referrerPolicy="no-referrer"
                                    onError={(e) => {
                                        (e.currentTarget as HTMLImageElement).style.display = 'none';
                                        const next = e.currentTarget.nextElementSibling as HTMLElement | null;
                                        if (next) next.style.display = 'flex';
                                    }}
                                />
                            ) : null}
                            <div className={`w-full h-full rounded-[10px] bg-gradient-to-br from-slate-100 to-slate-200 text-slate-600 font-bold text-sm items-center justify-center ${channelIconUrl ? 'hidden' : 'flex'}`}>
                                {channelInitial}
                            </div>
                        </div>
                        <h1 className="text-xl sm:text-2xl font-bold text-slate-800 truncate" title={channelName || `Channel ${id}`}>
                            {channelName || `Channel ${id}`}
                        </h1>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:justify-end">
                    <button
                        onClick={() => setShowBatchPublishModal(true)}
                        className="col-span-2 inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors sm:col-span-1"
                        title="Dry-run and publish AI-generated YouTube descriptions for processed videos in this channel"
                    >
                        <Link2 size={14} />
                        Publish AI Descriptions
                    </button>
                    <button
                        onClick={handleConsolidateTranscripts}
                        disabled={consolidatingTranscripts}
                        className="col-span-2 inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50 sm:col-span-1"
                        title="Merge same-speaker transcript fragments across this channel without re-running ASR or diarization"
                    >
                        {consolidatingTranscripts ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                        Consolidate Transcripts
                    </button>
                    <button
                        onClick={handleQueueTranscriptRepairs}
                        disabled={queueingTranscriptRepairs}
                        className="col-span-2 inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-emerald-700 bg-white border border-emerald-200 rounded-lg hover:bg-emerald-50 transition-colors disabled:opacity-50 sm:col-span-1"
                        title="Queue evaluator-approved low-risk transcript repairs across this channel"
                    >
                        {queueingTranscriptRepairs ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                        Queue Repairs
                    </button>
                    <button
                        onClick={handleQueueDiarizationRebuilds}
                        disabled={queueingDiarizationRebuilds}
                        className="col-span-2 inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-blue-700 bg-white border border-blue-200 rounded-lg hover:bg-blue-50 transition-colors disabled:opacity-50 sm:col-span-1"
                        title="Queue evaluator-approved diarization rebuilds across this channel"
                    >
                        {queueingDiarizationRebuilds ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
                        Queue Rebuilds
                    </button>
                    <button
                        onClick={handleQueueFullRetranscriptions}
                        disabled={queueingFullRetranscriptions}
                        className="col-span-2 inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-amber-700 bg-white border border-amber-200 rounded-lg hover:bg-amber-50 transition-colors disabled:opacity-50 sm:col-span-1"
                        title="Queue evaluator-approved full retranscriptions across this channel"
                    >
                        {queueingFullRetranscriptions ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                        Queue Retranscribe
                    </button>
                    <button
                        onClick={handleExport}
                        disabled={exporting}
                        className="inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
                    >
                        {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                        Export Archive
                    </button>
                    <button
                        onClick={() => setShowDeleteModal(true)}
                        className="inline-flex min-h-10 items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-red-600 bg-white border border-red-200 rounded-lg hover:bg-red-50 transition-colors"
                    >
                        <Trash2 size={14} />
                        Delete Channel
                    </button>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="glass-panel rounded-2xl p-2">
                <nav className="grid grid-cols-4 gap-1 sm:flex sm:flex-wrap">
                    {tabs.map((tab) => {
                        const Icon = tab.icon;
                        return (
                            <NavLink
                                key={tab.path}
                                to={tab.path === '' ? `/channel/${id}` : `/channel/${id}/${tab.path}`}
                                end={tab.path === ''}
                                className={({ isActive }) =>
                                    `inline-flex min-h-10 items-center justify-center gap-1.5 whitespace-nowrap px-2 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 sm:px-4 sm:text-sm ${isActive
                                        ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/25'
                                        : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
                                    }`
                                }
                            >
                                <Icon size={16} />
                                {tab.label}
                            </NavLink>
                        );
                    })}
                </nav>
            </div>

            {/* Tab Content */}
            <div className="animate-fade-in">
                <Outlet />
            </div>

            {/* Delete confirmation modal */}
            {showDeleteModal && id && (
                <DeleteChannelModal
                    channelId={Number(id)}
                    channelName={channelName}
                    onClose={() => setShowDeleteModal(false)}
                    onDeleted={() => navigate('/')}
                />
            )}

            {showBatchPublishModal && id && (
                <BatchPublishDescriptionsModal
                    channelId={Number(id)}
                    channelName={channelName}
                    onClose={() => setShowBatchPublishModal(false)}
                />
            )}
        </div>
    );
}

function BatchPublishDescriptionsModal({
    channelId,
    channelName,
    onClose,
}: {
    channelId: number;
    channelName: string;
    onClose: () => void;
}) {
    const [loadingDryRun, setLoadingDryRun] = useState(true);
    const [runningPublish, setRunningPublish] = useState(false);
    const [dryRun, setDryRun] = useState<any | null>(null);
    const [result, setResult] = useState<any | null>(null);
    const [pushToYouTube, setPushToYouTube] = useState<boolean | null>(null); // null = use backend setting

    const runDryRun = async () => {
        setLoadingDryRun(true);
        try {
            const res = await api.post(`/channels/${channelId}/youtube-ai/publish-descriptions`, {
                dry_run: true,
                confirm: false,
                push_to_youtube: pushToYouTube,
            });
            setDryRun(res.data);
            setResult(null);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Dry-run failed');
        } finally {
            setLoadingDryRun(false);
        }
    };

    useEffect(() => {
        void runDryRun();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (dryRun) {
            void runDryRun();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pushToYouTube]);

    const handleConfirmPublish = async () => {
        if (!dryRun) return;
        if ((dryRun?.counts?.eligible || 0) === 0) return;
        const pushLabel = dryRun.push_to_youtube ? 'This will also call YouTube Data API videos.update for each eligible video.' : 'This will update local stored descriptions only.';
        if (!confirm(`Publish AI descriptions for ${dryRun.counts.eligible} video(s) in ${channelName || 'this channel'}?\n\n${pushLabel}`)) return;
        setRunningPublish(true);
        try {
            const res = await api.post(`/channels/${channelId}/youtube-ai/publish-descriptions`, {
                dry_run: false,
                confirm: true,
                push_to_youtube: pushToYouTube,
            });
            setResult(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Batch publish failed');
        } finally {
            setRunningPublish(false);
        }
    };

    const display = result || dryRun;
    const items: any[] = Array.isArray(display?.items) ? display.items : [];
    const ownership = display?.ownership_check;
    const pushToYouTubeEffective = !!display?.push_to_youtube;
    const ownershipOkForPush = !pushToYouTubeEffective || ownership?.status === 'owned';

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-slate-900/40" onClick={onClose} />
            <div className="relative w-full max-w-3xl max-h-[85vh] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
                <div className="px-5 py-4 border-b border-slate-200 bg-gradient-to-r from-emerald-50 to-teal-50">
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="flex items-center gap-2 text-emerald-800 font-semibold">
                                <Link2 size={16} />
                                Publish AI Descriptions
                            </div>
                            <p className="text-xs text-emerald-700/80 mt-1">
                                Dry-run first, then confirm batch publish. Current descriptions are archived before replacement.
                            </p>
                        </div>
                        <button onClick={onClose} className="text-slate-500 hover:text-slate-700 text-sm">Close</button>
                    </div>
                </div>

                <div className="p-5 space-y-4 overflow-y-auto max-h-[calc(85vh-140px)]">
                    <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div>
                            <div className="text-sm font-medium text-slate-700">Push to YouTube</div>
                            <div className="text-xs text-slate-500">Use backend default setting, or override for this batch run.</div>
                        </div>
                        <select
                            value={pushToYouTube === null ? 'default' : pushToYouTube ? 'yes' : 'no'}
                            onChange={(e) => setPushToYouTube(
                                e.target.value === 'default' ? null : e.target.value === 'yes'
                            )}
                            className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                        >
                            <option value="default">Use Default Setting</option>
                            <option value="yes">Yes (push to YouTube)</option>
                            <option value="no">No (local only)</option>
                        </select>
                    </div>

                    {loadingDryRun ? (
                        <div className="flex items-center gap-2 text-sm text-slate-500">
                            <Loader2 size={16} className="animate-spin" />
                            Running dry-run...
                        </div>
                    ) : display ? (
                        <>
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                <StatPill label="Scanned" value={display.counts?.scanned ?? 0} />
                                <StatPill label="Eligible" value={display.counts?.eligible ?? 0} tone="green" />
                                <StatPill label="Missing Draft" value={display.counts?.missing_ai_draft ?? 0} />
                                <StatPill label="Already Match" value={display.counts?.already_matches_draft ?? 0} />
                                <StatPill label="Quota Est." value={display.estimated_youtube_quota_units ?? 0} tone="amber" />
                            </div>

                            {pushToYouTubeEffective && (
                                <div className={`rounded-xl border p-3 text-xs ${
                                    ownership?.status === 'owned'
                                        ? 'border-green-200 bg-green-50 text-green-800'
                                        : ownership?.status === 'not_owned'
                                            ? 'border-red-200 bg-red-50 text-red-800'
                                            : ownership?.status === 'unknown'
                                                ? 'border-amber-200 bg-amber-50 text-amber-800'
                                                : 'border-slate-200 bg-slate-50 text-slate-700'
                                }`}>
                                    <div className="font-semibold mb-1">YouTube Ownership Check</div>
                                    <div>
                                        Status: <span className="font-medium">{ownership?.status || 'unknown'}</span>
                                    </div>
                                    {ownership?.connected_channel_title && (
                                        <div>Connected OAuth channel: {ownership.connected_channel_title} ({ownership.connected_channel_id})</div>
                                    )}
                                    {ownership?.resolved_channel?.channel_title && (
                                        <div>Resolved app channel target: {ownership.resolved_channel.channel_title}{ownership?.resolved_channel?.channel_id ? ` (${ownership.resolved_channel.channel_id})` : ''}</div>
                                    )}
                                    {ownership?.resolved_channel?.method && (
                                        <div>Resolution method: {ownership.resolved_channel.method}</div>
                                    )}
                                    {ownership?.reason && (
                                        <div className="mt-1">{ownership.reason}</div>
                                    )}
                                </div>
                            )}

                            {result && (
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    <StatPill label="Published" value={result.counts?.published ?? 0} tone="green" />
                                    <StatPill label="Errors" value={result.counts?.errors ?? 0} tone="red" />
                                    <StatPill label="Skipped" value={result.counts?.skipped ?? 0} />
                                    <StatPill label="Push Mode" value={result.push_to_youtube ? 'YouTube' : 'Local'} />
                                </div>
                            )}

                            <div className="rounded-xl border border-slate-200 overflow-hidden">
                                <div className="px-3 py-2 bg-slate-50 border-b border-slate-200 text-xs font-medium text-slate-600">
                                    {result ? 'Batch Results' : 'Dry-run Preview'} ({items.length} items)
                                </div>
                                <div className="max-h-80 overflow-y-auto divide-y divide-slate-100">
                                    {items.map((item) => (
                                        <div key={item.video_id} className="px-3 py-2.5 text-xs">
                                            <div className="flex items-start justify-between gap-2">
                                                <div className="min-w-0">
                                                    <div className="font-medium text-slate-700 truncate">{item.title}</div>
                                                    <div className="text-slate-500 font-mono">{item.youtube_id}</div>
                                                </div>
                                                <div className="shrink-0 flex items-center gap-1.5">
                                                    {item.status === 'published' ? (
                                                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-green-50 text-green-700 border border-green-200"><CheckCircle2 size={11} /> published</span>
                                                    ) : item.status === 'error' ? (
                                                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-red-50 text-red-700 border border-red-200"><AlertCircle size={11} /> error</span>
                                                    ) : item.eligible ? (
                                                        <span className="px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">eligible</span>
                                                    ) : (
                                                        <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">{item.reason || 'skipped'}</span>
                                                    )}
                                                </div>
                                            </div>
                                            {item.error && (
                                                <div className="mt-1 text-red-600">{item.error}</div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </>
                    ) : null}
                </div>

                <div className="px-5 py-4 border-t border-slate-200 bg-white flex items-center justify-between gap-3">
                    <button
                        onClick={() => void runDryRun()}
                        disabled={loadingDryRun || runningPublish}
                        className="px-3 py-2 text-xs font-medium rounded-lg border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-50"
                    >
                        {loadingDryRun ? 'Refreshing...' : 'Refresh Dry-run'}
                    </button>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={onClose}
                            className="px-3 py-2 text-xs font-medium rounded-lg border border-slate-200 bg-white hover:bg-slate-50"
                        >
                            Close
                        </button>
                        <button
                            onClick={handleConfirmPublish}
                            disabled={runningPublish || loadingDryRun || !dryRun || (dryRun?.counts?.eligible ?? 0) === 0 || !ownershipOkForPush}
                            className="px-3 py-2 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50 inline-flex items-center gap-1.5"
                        >
                            {runningPublish ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                            Confirm Publish
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatPill({ label, value, tone = 'slate' }: { label: string; value: string | number; tone?: 'slate' | 'green' | 'amber' | 'red' }) {
    const toneClasses = tone === 'green'
        ? 'bg-green-50 border-green-200 text-green-700'
        : tone === 'amber'
            ? 'bg-amber-50 border-amber-200 text-amber-700'
            : tone === 'red'
                ? 'bg-red-50 border-red-200 text-red-700'
                : 'bg-slate-50 border-slate-200 text-slate-700';
    return (
        <div className={`rounded-lg border px-2.5 py-2 ${toneClasses}`}>
            <div className="text-[10px] uppercase tracking-wide opacity-80">{label}</div>
            <div className="text-sm font-semibold">{value}</div>
        </div>
    );
}
