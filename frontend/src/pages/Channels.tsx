import { useEffect, useState, useRef, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import type { Channel } from '../types';
import { RefreshCw, Plus, ExternalLink, ChevronRight, Loader2, Upload, X, Search, Activity, Radio, ArrowUpDown } from 'lucide-react';

interface ChannelStats {
    video_count: number;
    processed_count: number;
    pending_video_count: number;
    active_job_count: number;
    speaker_count: number;
    total_duration_seconds: number;
}

interface ChannelOverview extends Channel, ChannelStats {}

const CHANNEL_REFRESH_POLL_MS = 4000;

export function Channels() {
    const [channels, setChannels] = useState<ChannelOverview[]>([]);
    const [channelStats, setChannelStats] = useState<Record<number, ChannelStats>>({});
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [adding, setAdding] = useState(false);
    const [refreshingIds, setRefreshingIds] = useState<Set<number>>(new Set());
    const [monitorSavingIds, setMonitorSavingIds] = useState<Set<number>>(new Set());
    const [importing, setImporting] = useState(false);
    const [addWizardOpen, setAddWizardOpen] = useState(false);
    const [addWizardStep, setAddWizardStep] = useState<1 | 2>(1);
    const [addChannelType, setAddChannelType] = useState<'youtube' | 'tiktok' | 'manual' | null>(null);
    const [youtubeUrlInput, setYoutubeUrlInput] = useState('');
    const [tiktokCreatorInput, setTiktokCreatorInput] = useState('');
    const [manualNameInput, setManualNameInput] = useState('');
    const [searchQuery, setSearchQuery] = useState('');
    const [sourceFilter, setSourceFilter] = useState<'all' | 'youtube' | 'tiktok' | 'manual'>('all');
    const [activityFilter, setActivityFilter] = useState<'all' | 'attention' | 'monitored' | 'quiet'>('all');
    const [sortBy, setSortBy] = useState<'attention' | 'updated' | 'videos' | 'runtime' | 'name'>('attention');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const formatRuntime = (totalSeconds: number) => {
        const seconds = Math.max(0, Math.floor(totalSeconds || 0));
        const hours = Math.floor(seconds / 3600);
        if (hours >= 1000) {
            return `${(hours / 1000).toFixed(1)}k hrs`;
        }
        if (hours >= 100) {
            return `${hours} hrs`;
        }
        const days = Math.floor(hours / 24);
        const remHours = hours % 24;
        if (days >= 1) {
            return `${days}d ${remHours}h`;
        }
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    };

    const getSyncProgress = (channel: ChannelOverview) => {
        return Math.max(0, Math.min(100, Number(channel.sync_progress ?? 0)));
    };

    const formatLastUpdated = (value?: string | null) => {
        if (!value) return 'Never';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return 'Unknown';
        return parsed.toLocaleDateString();
    };

    const fetchChannels = async () => {
        setLoadError(null);
        try {
            const res = await api.get<ChannelOverview[]>('/channels/overview');
            setChannels(res.data);
            const statsByChannel: Record<number, ChannelStats> = {};
            for (const channel of res.data) {
                statsByChannel[channel.id] = {
                    video_count: channel.video_count ?? 0,
                    processed_count: channel.processed_count ?? 0,
                    pending_video_count: channel.pending_video_count ?? 0,
                    active_job_count: channel.active_job_count ?? 0,
                    speaker_count: channel.speaker_count ?? 0,
                    total_duration_seconds: channel.total_duration_seconds ?? 0,
                };
            }
            setChannelStats(statsByChannel);
        } catch (e) {
            console.error('Failed to fetch channels:', e);
            setLoadError('Unable to load channels. Confirm backend is running and reachable.');
            setChannels([]);
            setChannelStats({});
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchChannels();
    }, []);

    const visibleRefreshingIds = new Set<number>([
        ...Array.from(refreshingIds),
        ...channels
            .filter(channel => channel.status === 'refreshing' && (channel.source_type || 'youtube') !== 'manual')
            .map(channel => channel.id),
    ]);
    const refreshingChannelCount = visibleRefreshingIds.size;
    const monitoredBusyCount = channels.filter(channel => (
        channel.actively_monitored
        && (
            channel.status === 'refreshing'
            || (channel.pending_video_count ?? 0) > 0
            || (channel.active_job_count ?? 0) > 0
        )
    )).length;
    const totalVideoCount = channels.reduce((sum, channel) => sum + Number(channel.video_count ?? 0), 0);
    const totalProcessedCount = channels.reduce((sum, channel) => sum + Number(channel.processed_count ?? 0), 0);
    const totalSpeakerCount = channels.reduce((sum, channel) => sum + Number(channel.speaker_count ?? 0), 0);
    const monitoredChannelCount = channels.filter((channel) => !!channel.actively_monitored).length;
    const needsAttentionCount = channels.filter((channel) => (
        channel.status === 'refreshing'
        || Number(channel.pending_video_count ?? 0) > 0
        || Number(channel.active_job_count ?? 0) > 0
    )).length;

    const filteredChannels = channels
        .filter((channel) => {
            const source = String(channel.source_type || 'youtube').toLowerCase();
            if (sourceFilter !== 'all' && source !== sourceFilter) return false;

            const normalizedSearch = searchQuery.trim().toLowerCase();
            if (normalizedSearch) {
                const haystack = `${channel.name || ''} ${channel.url || ''}`.toLowerCase();
                if (!haystack.includes(normalizedSearch)) return false;
            }

            const needsAttention = channel.status === 'refreshing'
                || Number(channel.pending_video_count ?? 0) > 0
                || Number(channel.active_job_count ?? 0) > 0;
            const monitored = !!channel.actively_monitored;

            if (activityFilter === 'attention' && !needsAttention) return false;
            if (activityFilter === 'monitored' && !monitored) return false;
            if (activityFilter === 'quiet' && needsAttention) return false;
            return true;
        })
        .slice()
        .sort((a, b) => {
            if (sortBy === 'name') {
                return (a.name || '').localeCompare(b.name || '');
            }
            if (sortBy === 'videos') {
                return Number(b.video_count ?? 0) - Number(a.video_count ?? 0);
            }
            if (sortBy === 'runtime') {
                return Number(b.total_duration_seconds ?? 0) - Number(a.total_duration_seconds ?? 0);
            }
            if (sortBy === 'updated') {
                const aUpdated = new Date(a.last_updated || 0).getTime();
                const bUpdated = new Date(b.last_updated || 0).getTime();
                if (aUpdated !== bUpdated) return bUpdated - aUpdated;
                return (a.name || '').localeCompare(b.name || '');
            }
            const rank = (channel: ChannelOverview) => {
                const needsAttention = channel.status === 'refreshing'
                    || Number(channel.pending_video_count ?? 0) > 0
                    || Number(channel.active_job_count ?? 0) > 0;
                if (needsAttention) return 0;
                if (channel.actively_monitored) return 1;
                return 2;
            };
            const rankDiff = rank(a) - rank(b);
            if (rankDiff !== 0) return rankDiff;
            const aUpdated = new Date(a.last_updated || 0).getTime();
            const bUpdated = new Date(b.last_updated || 0).getTime();
            if (aUpdated !== bUpdated) return bUpdated - aUpdated;
            return (a.name || '').localeCompare(b.name || '');
        });

    useEffect(() => {
        if (loading || loadError || (refreshingChannelCount === 0 && monitoredBusyCount === 0 && !adding && refreshingIds.size === 0)) {
            return;
        }
        const timer = setInterval(() => {
            void fetchChannels();
        }, CHANNEL_REFRESH_POLL_MS);
        return () => clearInterval(timer);
    }, [adding, loadError, loading, monitoredBusyCount, refreshingChannelCount, refreshingIds]);

    const insertCreatedChannel = (created: Channel) => {
        setChannels(prev => {
            const existing = prev.find(channel => channel.id === created.id);
            const next = prev.filter(channel => channel.id !== created.id);
            next.push({
                ...created,
                video_count: existing?.video_count ?? 0,
                processed_count: existing?.processed_count ?? 0,
                pending_video_count: existing?.pending_video_count ?? 0,
                active_job_count: existing?.active_job_count ?? 0,
                speaker_count: existing?.speaker_count ?? 0,
                total_duration_seconds: existing?.total_duration_seconds ?? 0,
            });
            next.sort((a, b) => a.id - b.id);
            return next;
        });
        setChannelStats(prev => ({
            ...prev,
            [created.id]: prev[created.id] ?? {
                video_count: 0,
                processed_count: 0,
                pending_video_count: 0,
                active_job_count: 0,
                speaker_count: 0,
                total_duration_seconds: 0,
            },
        }));
    };

    const resetAddWizard = () => {
        setAddWizardOpen(false);
        setAddWizardStep(1);
        setAddChannelType(null);
        setYoutubeUrlInput('');
        setTiktokCreatorInput('');
        setManualNameInput('');
    };

    const parseTikTokCreatorInput = (raw: string): { name: string; url: string } => {
        const value = raw.trim();
        if (!value) {
            return { name: '', url: '' };
        }
        if (value.toLowerCase().includes('tiktok.com')) {
            const match = value.match(/tiktok\.com\/@([^/?#]+)/i);
            const handle = (match?.[1] || '').trim();
            const normalizedName = handle ? (handle.startsWith('@') ? handle : `@${handle}`) : '';
            return { name: normalizedName, url: value };
        }
        const compact = value.replace(/\s+/g, '');
        if (/^@?[A-Za-z0-9._]+$/.test(compact)) {
            const normalizedName = compact.startsWith('@') ? compact : `@${compact}`;
            return {
                name: normalizedName,
                url: `https://www.tiktok.com/${normalizedName}`,
            };
        }
        return { name: value, url: '' };
    };

    const handleCreateChannel = async (e?: FormEvent) => {
        e?.preventDefault();
        if (!addChannelType) return;
        setAdding(true);
        try {
            let created: Channel;
            if (addChannelType === 'youtube') {
                const url = youtubeUrlInput.trim();
                if (!url) {
                    alert('Channel URL is required.');
                    return;
                }
                const res = await api.post<Channel>('/channels', null, { params: { url } });
                created = res.data;
            } else if (addChannelType === 'tiktok') {
                const { name, url } = parseTikTokCreatorInput(tiktokCreatorInput);
                if (!name) {
                    alert('TikTok creator name or profile URL is required.');
                    return;
                }
                const res = await api.post<Channel>('/channels/tiktok', null, { params: { name, url } });
                created = res.data;
            } else {
                const name = manualNameInput.trim();
                if (!name) {
                    alert('Manual channel name is required.');
                    return;
                }
                const res = await api.post<Channel>('/channels/manual', null, { params: { name } });
                created = res.data;
            }
            insertCreatedChannel(created);
            resetAddWizard();
            void fetchChannels();
        } catch (e) {
            alert('Failed to add channel');
        } finally {
            setAdding(false);
        }
    };

    const handleRefresh = async (id: number) => {
        setRefreshingIds(prev => new Set(prev).add(id));
        try {
            await api.post(`/channels/${id}/refresh`);
            await fetchChannels();
            setRefreshingIds(prev => {
                const next = new Set(prev);
                next.delete(id);
                return next;
            });
        } catch (e) {
            alert('Failed to refresh');
            setRefreshingIds(prev => {
                const next = new Set(prev);
                next.delete(id);
                return next;
            });
        }
    };

    const handleToggleMonitoring = async (id: number, enabled: boolean) => {
        setMonitorSavingIds(prev => new Set(prev).add(id));
        try {
            const res = await api.patch<Channel>(`/channels/${id}/actively-monitored`, null, {
                params: { enabled },
            });
            setChannels(prev => prev.map(channel => (
                channel.id === id ? { ...channel, actively_monitored: res.data.actively_monitored } : channel
            )));
            await fetchChannels();
        } catch (e) {
            alert('Failed to update active monitoring');
        } finally {
            setMonitorSavingIds(prev => {
                const next = new Set(prev);
                next.delete(id);
                return next;
            });
        }
    };

    const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setImporting(true);
        try {
            const text = await file.text();
            const archive = JSON.parse(text);
            await api.post('/channels/import', archive);
            fetchChannels();
        } catch (err: any) {
            const detail = err.response?.data?.detail;
            alert(detail || 'Failed to import archive');
        } finally {
            setImporting(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    return (
        <div className="mx-auto max-w-7xl space-y-4">
            <div className="rounded-3xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
                <div className="flex flex-col gap-2.5 xl:flex-row xl:items-center xl:justify-between">
                    <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-blue-600">Library</p>
                            <h2 className="text-xl font-bold leading-none text-slate-900">All Channels</h2>
                        </div>
                        <p className="mt-1 max-w-4xl text-[12px] leading-5 text-slate-500">
                            Remote feeds and manual media collections, with attention-needed channels surfaced first.
                        </p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                        <button
                            onClick={() => void fetchChannels()}
                            disabled={loading}
                            className="inline-flex h-8 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-[13px] font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
                        >
                            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
                            Refresh
                        </button>
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={importing}
                            className="inline-flex h-8 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-[13px] font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
                        >
                            {importing ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
                            {importing ? 'Importing...' : 'Import Archive'}
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".json"
                            onChange={handleImport}
                            className="hidden"
                        />
                        <button
                            type="button"
                            onClick={() => setAddWizardOpen(true)}
                            disabled={adding}
                            className="inline-flex h-8 items-center justify-center gap-2 rounded-xl bg-blue-600 px-3 text-[13px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
                        >
                            <Plus size={15} />
                            {adding ? 'Adding...' : 'Add Channel'}
                        </button>
                    </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                    <div className="inline-flex min-w-[132px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Channels</span>
                        <span className="text-base font-bold text-slate-900">{channels.length}</span>
                        <span className="text-[11px] text-slate-500">{monitoredChannelCount} monitored</span>
                    </div>
                    <div className="inline-flex min-w-[132px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Attention</span>
                        <span className="text-base font-bold text-amber-700">{needsAttentionCount}</span>
                        <span className="text-[11px] text-slate-500">{refreshingChannelCount} syncing</span>
                    </div>
                    <div className="inline-flex min-w-[132px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Videos</span>
                        <span className="text-base font-bold text-slate-900">{totalVideoCount}</span>
                        <span className="text-[11px] text-slate-500">{totalProcessedCount} ready</span>
                    </div>
                    <div className="inline-flex min-w-[132px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Speakers</span>
                        <span className="text-base font-bold text-slate-900">{totalSpeakerCount}</span>
                        <span className="text-[11px] text-slate-500">indexed</span>
                    </div>
                    <div className="inline-flex min-w-[132px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Visible</span>
                        <span className="text-base font-bold text-slate-900">{filteredChannels.length}</span>
                        <span className="text-[11px] text-slate-500">current view</span>
                    </div>
                </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div className="relative max-w-xl flex-1">
                        <Search size={16} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search channels by name or source URL..."
                            className="w-full rounded-2xl border border-slate-200 bg-slate-50 py-3 pl-11 pr-4 text-sm text-slate-700 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-2 focus:ring-blue-500/20"
                        />
                    </div>
                    <div className="flex flex-col gap-3 sm:flex-row">
                        <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                            <Radio size={15} className="text-slate-400" />
                            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Source</span>
                            <select
                                value={sourceFilter}
                                onChange={(e) => setSourceFilter((e.target.value || 'all') as typeof sourceFilter)}
                                className="bg-transparent text-sm font-medium text-slate-700 outline-none"
                            >
                                <option value="all">All</option>
                                <option value="youtube">YouTube</option>
                                <option value="tiktok">TikTok</option>
                                <option value="manual">Manual</option>
                            </select>
                        </label>
                        <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                            <Activity size={15} className="text-slate-400" />
                            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Focus</span>
                            <select
                                value={activityFilter}
                                onChange={(e) => setActivityFilter((e.target.value || 'all') as typeof activityFilter)}
                                className="bg-transparent text-sm font-medium text-slate-700 outline-none"
                            >
                                <option value="all">All channels</option>
                                <option value="attention">Needs attention</option>
                                <option value="monitored">Monitored</option>
                                <option value="quiet">Quiet / stable</option>
                            </select>
                        </label>
                        <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                            <ArrowUpDown size={15} className="text-slate-400" />
                            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Sort</span>
                            <select
                                value={sortBy}
                                onChange={(e) => setSortBy((e.target.value || 'attention') as typeof sortBy)}
                                className="bg-transparent text-sm font-medium text-slate-700 outline-none"
                            >
                                <option value="attention">Attention first</option>
                                <option value="updated">Recently updated</option>
                                <option value="videos">Most videos</option>
                                <option value="runtime">Most runtime</option>
                                <option value="name">Name A-Z</option>
                            </select>
                        </label>
                    </div>
                </div>

                <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-sm">
                    <div className="text-slate-500">
                        Showing <span className="font-semibold text-slate-700">{filteredChannels.length}</span> of <span className="font-semibold text-slate-700">{channels.length}</span> channels
                    </div>
                    {(searchQuery || sourceFilter !== 'all' || activityFilter !== 'all' || sortBy !== 'attention') && (
                        <button
                            type="button"
                            onClick={() => {
                                setSearchQuery('');
                                setSourceFilter('all');
                                setActivityFilter('all');
                                setSortBy('attention');
                            }}
                            className="inline-flex h-9 items-center justify-center rounded-xl border border-slate-200 px-3 text-sm font-medium text-slate-600 hover:bg-slate-50"
                        >
                            Reset view
                        </button>
                    )}
                </div>
            </div>

            {(adding || refreshingChannelCount > 0) && (
                <div className="glass-panel rounded-2xl border border-amber-200 bg-amber-50/80 p-4 text-sm text-amber-900">
                    <div className="flex items-center gap-2 font-semibold">
                        <Loader2 size={16} className="animate-spin" />
                        {adding
                            ? 'Adding channel and starting initial scan...'
                            : `Refreshing ${refreshingChannelCount} channel${refreshingChannelCount === 1 ? '' : 's'}...`}
                    </div>
                    <p className="mt-1 text-amber-800/80">
                        Video counts appear first. Additional metadata can continue filling in after the initial scan finishes.
                    </p>
                </div>
            )}

            {addWizardOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 backdrop-blur-sm p-4" onClick={resetAddWizard}>
                    <div
                        className="w-full max-w-2xl rounded-3xl border border-slate-200 bg-white shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between border-b border-slate-100 px-6 py-5">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-600">Add Channel</p>
                                <h3 className="mt-1 text-xl font-bold text-slate-900">
                                    {addWizardStep === 1 ? 'Choose a channel type' : 'Enter channel details'}
                                </h3>
                            </div>
                            <button
                                type="button"
                                onClick={resetAddWizard}
                                className="rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                                aria-label="Close add channel wizard"
                            >
                                <X size={18} />
                            </button>
                        </div>

                        <div className="px-6 py-6">
                            <div className="mb-6 flex items-center gap-3">
                                <div className={`h-2 flex-1 rounded-full ${addWizardStep >= 1 ? 'bg-blue-600' : 'bg-slate-200'}`} />
                                <div className={`h-2 flex-1 rounded-full ${addWizardStep >= 2 ? 'bg-blue-600' : 'bg-slate-200'}`} />
                            </div>

                            {addWizardStep === 1 ? (
                                <div className="grid gap-3 md:grid-cols-3">
                                    {[
                                        {
                                            type: 'youtube' as const,
                                            title: 'YouTube Channel',
                                            body: 'Paste a YouTube channel, handle, or playlist-style channel URL and start a full scan.',
                                        },
                                        {
                                            type: 'tiktok' as const,
                                            title: 'TikTok Channel',
                                            body: 'Create a TikTok creator source and optionally attach a profile URL for feed refresh.',
                                        },
                                        {
                                            type: 'manual' as const,
                                            title: 'Manual Media Channel',
                                            body: 'Create a raw channel for uploaded audio or video files with no external feed.',
                                        },
                                    ].map((option) => (
                                        <button
                                            key={option.type}
                                            type="button"
                                            onClick={() => {
                                                setAddChannelType(option.type);
                                                setAddWizardStep(2);
                                            }}
                                            className="rounded-2xl border border-slate-200 bg-slate-50 p-5 text-left transition hover:border-blue-300 hover:bg-blue-50"
                                        >
                                            <div className="text-base font-semibold text-slate-900">{option.title}</div>
                                            <p className="mt-2 text-sm leading-6 text-slate-600">{option.body}</p>
                                        </button>
                                    ))}
                                </div>
                            ) : (
                                <form onSubmit={handleCreateChannel} className="space-y-5">
                                    {addChannelType === 'youtube' && (
                                        <div className="space-y-2">
                                            <label className="block text-sm font-semibold text-slate-700">YouTube channel URL</label>
                                            <input
                                                type="text"
                                                value={youtubeUrlInput}
                                                onChange={(e) => setYoutubeUrlInput(e.target.value)}
                                                placeholder="https://youtube.com/@channelname"
                                                className="w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20"
                                                autoFocus
                                            />
                                            <p className="text-sm text-slate-500">Use a channel URL or handle. The initial scan starts immediately after creation.</p>
                                        </div>
                                    )}

                                    {addChannelType === 'tiktok' && (
                                        <div className="space-y-2">
                                            <label className="block text-sm font-semibold text-slate-700">Creator name or TikTok profile URL</label>
                                            <input
                                                type="text"
                                                value={tiktokCreatorInput}
                                                onChange={(e) => setTiktokCreatorInput(e.target.value)}
                                                placeholder="@creatorname or https://www.tiktok.com/@creatorname"
                                                className="w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20"
                                                autoFocus
                                            />
                                            <p className="text-sm text-slate-500">Paste a TikTok creator URL and the name will be derived automatically. Typing a handle like <span className="font-medium text-slate-700">@creatorname</span> will also build the profile URL for refresh.</p>
                                        </div>
                                    )}

                                    {addChannelType === 'manual' && (
                                        <div className="space-y-2">
                                            <label className="block text-sm font-semibold text-slate-700">Manual channel name</label>
                                            <input
                                                type="text"
                                                value={manualNameInput}
                                                onChange={(e) => setManualNameInput(e.target.value)}
                                                placeholder="Conference Archive"
                                                className="w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20"
                                                autoFocus
                                            />
                                            <p className="text-sm text-slate-500">Manual channels are for uploaded media files and do not use an external feed.</p>
                                        </div>
                                    )}

                                    <div className="flex flex-col-reverse gap-3 border-t border-slate-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
                                        <button
                                            type="button"
                                            onClick={() => setAddWizardStep(1)}
                                            className="inline-flex h-11 items-center justify-center rounded-2xl border border-slate-200 px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                                        >
                                            Back
                                        </button>
                                        <div className="flex gap-3">
                                            <button
                                                type="button"
                                                onClick={resetAddWizard}
                                                className="inline-flex h-11 items-center justify-center rounded-2xl border border-slate-200 px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                                            >
                                                Cancel
                                            </button>
                                            <button
                                                type="submit"
                                                disabled={adding}
                                                className="inline-flex h-11 items-center justify-center gap-2 rounded-2xl bg-blue-600 px-5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
                                            >
                                                {adding ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
                                                Create Channel
                                            </button>
                                        </div>
                                    </div>
                                </form>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {loading ? (
                <div className="flex items-center justify-center h-64 text-slate-400 animate-pulse">Loading channels...</div>
            ) : loadError ? (
                <div className="glass-panel rounded-2xl p-6 text-sm text-red-600 border border-red-200 bg-red-50">
                    <div className="font-semibold mb-2">Channels failed to load</div>
                    <div>{loadError}</div>
                    <button
                        onClick={() => {
                            setLoading(true);
                            void fetchChannels();
                        }}
                        className="mt-4 inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-red-300 bg-white text-red-700 hover:bg-red-100"
                    >
                        <RefreshCw size={14} />
                        Retry
                    </button>
                </div>
            ) : filteredChannels.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center shadow-sm">
                    <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
                        <Search size={22} />
                    </div>
                    <h3 className="mt-4 text-lg font-semibold text-slate-800">No channels match these filters</h3>
                    <p className="mt-2 text-sm text-slate-500">
                        Try clearing the search text or relaxing the source and focus filters.
                    </p>
                    <div className="mt-5 flex justify-center gap-3">
                        <button
                            type="button"
                            onClick={() => {
                                setSearchQuery('');
                                setSourceFilter('all');
                                setActivityFilter('all');
                            }}
                            className="inline-flex h-11 items-center justify-center rounded-2xl border border-slate-200 px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                        >
                            Clear Filters
                        </button>
                        <button
                            type="button"
                            onClick={() => setAddWizardOpen(true)}
                            className="inline-flex h-11 items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 text-sm font-semibold text-white hover:bg-blue-700"
                        >
                            <Plus size={16} />
                            Add Channel
                        </button>
                    </div>
                </div>
            ) : (
                <div className="grid [grid-template-columns:repeat(auto-fit,minmax(244px,1fr))] gap-3 items-stretch">
                    {filteredChannels.map((channel, idx) => {
                        const stats = channelStats[channel.id];
                        const channelSourceType = (channel.source_type || 'youtube').toLowerCase();
                        const isManualChannel = channelSourceType === 'manual';
                        const isTikTokChannel = channelSourceType === 'tiktok';
                        const isMonitored = !!channel.actively_monitored;
                        const isRemoteChannel = !isManualChannel;
                        const canAutoMonitor = isRemoteChannel && (channelSourceType !== 'tiktok' || String(channel.url || '').toLowerCase().includes('tiktok.com'));
                        const isRefreshing = refreshingIds.has(channel.id) || (isRemoteChannel && channel.status === 'refreshing');
                        const monitorBusy = canAutoMonitor && isMonitored && (isRefreshing || (stats?.active_job_count ?? 0) > 0 || (stats?.pending_video_count ?? 0) > 0);
                        const monitorLabel = !isMonitored ? 'Off' : monitorBusy ? 'Processing' : 'Up to date';
                        const syncProgress = getSyncProgress(channel);
                        const initials = (channel.name || 'C')
                            .split(/\s+/)
                            .filter(Boolean)
                            .slice(0, 2)
                            .map(part => part[0]?.toUpperCase() || '')
                            .join('') || 'C';
                        return (
                            <div
                                key={channel.id}
                                className="glass-panel w-full min-w-0 p-3.5 rounded-xl hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300 group h-full flex flex-col"
                                style={{ animationDelay: `${idx * 100}ms` }}
                            >
                                <div className="-mx-3.5 -mt-3.5 mb-2">
                                    <div className="relative overflow-hidden rounded-t-xl border-b border-white/50 bg-slate-100/80 px-2 py-2">
                                        <Link
                                            to={`/channel/${channel.id}`}
                                            className="group/header block pl-8 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white rounded-lg"
                                            aria-label={`Open ${channel.name}`}
                                        >
                                            <div className="relative h-11 rounded-lg bg-gradient-to-r from-blue-100 via-indigo-100 to-slate-100">
                                                {channel.header_image_url ? (
                                                    <>
                                                        <img
                                                            src={channel.header_image_url}
                                                            alt=""
                                                            className="h-full w-full object-cover transition-transform duration-300 group-hover/header:scale-[1.02]"
                                                            loading="lazy"
                                                            referrerPolicy="no-referrer"
                                                            onError={(e) => {
                                                                (e.currentTarget as HTMLImageElement).style.display = 'none';
                                                            }}
                                                        />
                                                        <div className="absolute inset-0 bg-gradient-to-r from-slate-900/12 via-transparent to-slate-900/10" />
                                                    </>
                                                ) : (
                                                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,.22),transparent_50%),radial-gradient(circle_at_80%_30%,rgba(99,102,241,.2),transparent_45%),linear-gradient(120deg,#e2e8f0,#dbeafe,#eef2ff)]" />
                                                )}
                                            </div>
                                        </Link>

                                        <Link
                                            to={`/channel/${channel.id}`}
                                            className="absolute left-2 top-2 block shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white rounded-lg"
                                            aria-label={`Open ${channel.name}`}
                                        >
                                            <div className="h-11 w-11 rounded-lg bg-white p-0.5 shadow-sm ring-1 ring-slate-200 overflow-hidden">
                                                {channel.icon_url ? (
                                                    <img
                                                        src={channel.icon_url}
                                                        alt=""
                                                        className="w-full h-full object-cover rounded-lg"
                                                        loading="lazy"
                                                        referrerPolicy="no-referrer"
                                                        onError={(e) => {
                                                            (e.currentTarget as HTMLImageElement).style.display = 'none';
                                                            const next = e.currentTarget.nextElementSibling as HTMLElement | null;
                                                            if (next) next.style.display = 'flex';
                                                        }}
                                                    />
                                                ) : null}
                                                <div
                                                    className={`w-full h-full rounded-lg bg-gradient-to-br from-slate-100 to-slate-200 text-slate-600 font-bold text-sm items-center justify-center ${channel.icon_url ? 'hidden' : 'flex'}`}
                                                >
                                                    {initials}
                                                </div>
                                            </div>
                                        </Link>
                                    </div>
                                </div>

                                <div className="mb-2">
                                    <Link
                                        to={`/channel/${channel.id}`}
                                        className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white rounded-md"
                                    >
                                        <h3
                                            className="line-clamp-2 text-[15px] font-bold leading-5 text-slate-800"
                                            title={channel.name}
                                        >
                                            {channel.name}
                                        </h3>
                                    </Link>
                                    <div className="mt-1 flex items-center gap-1.5 flex-wrap">
                                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                                            isManualChannel
                                                ? 'bg-slate-100 text-slate-600'
                                                : isTikTokChannel
                                                    ? 'bg-pink-50 text-pink-600'
                                                    : 'bg-blue-50 text-blue-600'
                                        }`}>
                                            {isManualChannel ? 'Manual' : isTikTokChannel ? 'TikTok' : 'YouTube'}
                                        </span>
                                        <span className="text-[11px] text-slate-400">ID {channel.id}</span>
                                        {!isManualChannel && (
                                            <a
                                                href={channel.url}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-500 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-600"
                                            >
                                                <ExternalLink size={11} />
                                                Source
                                            </a>
                                        )}
                                    </div>
                                </div>

                                {/* Stats Section */}
                                {stats && (
                                    <div className="grid grid-cols-3 gap-1.5 mb-2 text-center">
                                        <Link
                                            to={`/channel/${channel.id}`}
                                            className="bg-slate-50 rounded-lg px-1.5 py-1 border border-slate-100 hover:border-blue-200 hover:bg-blue-50 transition-colors"
                                        >
                                            <p className="text-[13px] font-bold text-slate-700 leading-tight">{stats.video_count}</p>
                                            <p className="text-[10px] text-slate-400">Videos</p>
                                        </Link>
                                        <Link
                                            to={`/channel/${channel.id}/transcripts`}
                                            className="bg-slate-50 rounded-lg px-1.5 py-1 border border-slate-100 hover:border-green-200 hover:bg-green-50 transition-colors"
                                        >
                                            <p className="text-[13px] font-bold text-slate-700 leading-tight">{stats.processed_count}</p>
                                            <p className="text-[10px] text-slate-400">Transcripts</p>
                                        </Link>
                                        <Link
                                            to={`/channel/${channel.id}/speakers`}
                                            className="bg-slate-50 rounded-lg px-1.5 py-1 border border-slate-100 hover:border-purple-200 hover:bg-purple-50 transition-colors"
                                        >
                                            <p className="text-[13px] font-bold text-slate-700 leading-tight">{stats.speaker_count}</p>
                                            <p className="text-[10px] text-slate-400">Speakers</p>
                                        </Link>
                                    </div>
                                )}

                                {isRefreshing && (
                                    <div className="mb-2 rounded-lg border border-amber-200 bg-amber-50/85 p-2 text-[11px] text-amber-900">
                                        <div className="flex items-center gap-2 font-semibold">
                                            <Loader2 size={13} className="animate-spin" />
                                            Sync in progress
                                        </div>
                                        <div className="mt-1.5">
                                            <div className="mb-1 flex items-center justify-between text-[10px] font-medium uppercase tracking-[0.14em] text-amber-700/80">
                                                <span>Channel Sync</span>
                                                <span>{syncProgress}%</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-amber-100">
                                                <div
                                                    className="h-2 rounded-full bg-amber-500 transition-all duration-500"
                                                    style={{ width: `${Math.max(syncProgress, 3)}%` }}
                                                />
                                            </div>
                                            {channel.sync_total_items ? (
                                                <p className="mt-1 text-[10px] text-amber-700/75">
                                                    {channel.sync_completed_items ?? 0}/{channel.sync_total_items} items
                                                </p>
                                            ) : null}
                                        </div>
                                    </div>
                                )}

                                {canAutoMonitor && isMonitored && (
                                    <div className={`mb-2 rounded-lg border px-2 py-1.5 text-[11px] ${monitorBusy
                                        ? 'border-sky-200 bg-sky-50/85 text-sky-900'
                                        : 'border-emerald-200 bg-emerald-50/85 text-emerald-900'
                                        }`}>
                                        <div className="flex items-center gap-2 font-semibold">
                                            {monitorBusy ? <Loader2 size={13} className="animate-spin" /> : <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />}
                                            {monitorBusy ? 'Monitoring + processing' : 'Monitoring active'}
                                        </div>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-x-2.5 gap-y-1.5 mb-2 text-[12px]">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="text-slate-500">Status</span>
                                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${isRefreshing
                                            ? 'bg-amber-50 text-amber-800 border-amber-200'
                                            : channel.status === 'active'
                                                ? 'bg-green-50 text-green-700 border-green-200'
                                                : 'bg-red-50 text-red-700 border-red-200'
                                            }`}>
                                            {isRefreshing ? 'refreshing' : channel.status}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="text-slate-500">Updated</span>
                                        <span className="text-slate-700 font-medium">{formatLastUpdated(channel.last_updated)}</span>
                                    </div>
                                    {stats && (
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="text-slate-500">Runtime</span>
                                            <span className="text-slate-700 font-medium">{formatRuntime(stats.total_duration_seconds)}</span>
                                        </div>
                                    )}
                                    {canAutoMonitor && (
                                        <div className="col-span-2 flex items-center justify-between gap-2 rounded-lg border border-slate-100 bg-slate-50 px-2 py-1.5">
                                            <span className="text-slate-500">Active monitoring</span>
                                            <div className="flex items-center gap-2 min-w-0">
                                                <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${!isMonitored
                                                    ? 'border-slate-200 bg-slate-50 text-slate-600'
                                                    : monitorBusy
                                                        ? 'border-sky-200 bg-sky-50 text-sky-700'
                                                        : 'border-emerald-200 bg-emerald-50 text-emerald-700'
                                                    }`}>
                                                    {monitorLabel}
                                                </span>
                                                <button
                                                    type="button"
                                                    role="switch"
                                                    aria-checked={isMonitored}
                                                    onClick={() => handleToggleMonitoring(channel.id, !isMonitored)}
                                                    disabled={monitorSavingIds.has(channel.id)}
                                                    className={`relative inline-flex h-5 w-10 shrink-0 items-center rounded-full transition-colors disabled:opacity-60 ${isMonitored ? 'bg-blue-600' : 'bg-slate-300'}`}
                                                >
                                                    <span
                                                        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${isMonitored ? 'translate-x-5' : 'translate-x-1'}`}
                                                    />
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div className="flex gap-2 mt-auto">
                                                    {isRemoteChannel && (
                                        <button
                                            onClick={() => handleRefresh(channel.id)}
                                            disabled={refreshingIds.has(channel.id)}
                                            className="flex-1 bg-slate-50 text-slate-600 py-1.5 rounded-lg hover:bg-blue-50 hover:text-blue-600 hover:shadow-sm border border-slate-100 hover:border-blue-200 transition-all flex items-center justify-center gap-1.5 text-[13px] font-medium disabled:opacity-50"
                                        >
                                            {refreshingIds.has(channel.id) ? (
                                                <Loader2 size={14} className="animate-spin" />
                                            ) : (
                                                <RefreshCw size={14} className="group-hover:rotate-180 transition-transform duration-500" />
                                            )}
                                            {refreshingIds.has(channel.id) ? 'Scanning...' : isTikTokChannel ? 'Refresh' : 'Scan'}
                                        </button>
                                    )}
                                    <Link
                                        to={`/channel/${channel.id}`}
                                        className={`${!isRemoteChannel ? 'w-full' : 'flex-1'} bg-blue-600 text-white py-1.5 rounded-lg hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-500/25 transition-all flex items-center justify-center gap-1.5 text-[13px] font-semibold border border-blue-700/20`}
                                    >
                                        Open
                                        <ChevronRight size={14} />
                                    </Link>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
