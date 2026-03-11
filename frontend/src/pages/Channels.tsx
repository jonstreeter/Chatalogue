import { useEffect, useState, useRef, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import type { Channel } from '../types';
import { RefreshCw, Plus, ExternalLink, ChevronRight, Film, Users, FileText, Loader2, Upload } from 'lucide-react';

interface ChannelStats {
    video_count: number;
    processed_count: number;
    speaker_count: number;
    total_duration_seconds: number;
}

interface ChannelOverview extends Channel, ChannelStats {}

export function Channels() {
    const [channels, setChannels] = useState<Channel[]>([]);
    const [channelStats, setChannelStats] = useState<Record<number, ChannelStats>>({});
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [newUrl, setNewUrl] = useState('');
    const [adding, setAdding] = useState(false);
    const [refreshingIds, setRefreshingIds] = useState<Set<number>>(new Set());
    const [importing, setImporting] = useState(false);
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

    const handleAdd = async (e: FormEvent) => {
        e.preventDefault();
        if (!newUrl) return;
        setAdding(true);
        try {
            await api.post('/channels', null, { params: { url: newUrl } });
            setNewUrl('');
            fetchChannels();
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
            // Refresh stats after a delay to allow background task to complete
            setTimeout(() => {
                fetchChannels();
                setRefreshingIds(prev => {
                    const next = new Set(prev);
                    next.delete(id);
                    return next;
                });
            }, 3000);
        } catch (e) {
            alert('Failed to refresh');
            setRefreshingIds(prev => {
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
        <div className="space-y-6">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                <h2 className="text-2xl font-bold text-gray-800">Channels</h2>
                <div className="w-full xl:w-auto flex flex-col gap-2 lg:flex-row lg:items-center">
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={importing}
                        className="inline-flex h-11 items-center justify-center gap-2 bg-white text-slate-700 px-5 rounded-xl border border-slate-200 hover:bg-slate-50 text-sm font-semibold disabled:opacity-50 whitespace-nowrap shrink-0"
                    >
                        {importing ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                        {importing ? 'Importing...' : 'Import Archive'}
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".json"
                        onChange={handleImport}
                        className="hidden"
                    />
                    <form onSubmit={handleAdd} className="flex w-full flex-col gap-2 sm:flex-row sm:items-center">
                        <input
                            type="text"
                            value={newUrl}
                            onChange={(e) => setNewUrl(e.target.value)}
                            placeholder="https://youtube.com/@..."
                            className="w-full min-w-0 h-11 px-4 border rounded-xl focus:ring-2 focus:ring-blue-500 outline-none sm:w-72"
                        />
                        <button
                            type="submit"
                            disabled={adding}
                            className="inline-flex h-11 items-center justify-center gap-2 bg-blue-600 text-white px-5 rounded-xl hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap font-semibold"
                        >
                            <Plus size={18} />
                            {adding ? 'Adding...' : 'Add Channel'}
                        </button>
                    </form>
                </div>
            </div>

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
            ) : (
                <div className="grid [grid-template-columns:repeat(auto-fit,minmax(300px,1fr))] gap-x-5 gap-y-6 items-stretch">
                    {channels.map((channel, idx) => {
                        const stats = channelStats[channel.id];
                        const initials = (channel.name || 'C')
                            .split(/\s+/)
                            .filter(Boolean)
                            .slice(0, 2)
                            .map(part => part[0]?.toUpperCase() || '')
                            .join('') || 'C';
                        return (
                            <div
                                key={channel.id}
                                className="glass-panel w-full min-w-0 p-5 rounded-2xl hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 group h-full flex flex-col"
                                style={{ animationDelay: `${idx * 100}ms` }}
                            >
                                <div className="-mx-5 -mt-5 mb-4">
                                    <div className="relative h-32 rounded-t-2xl overflow-hidden border-b border-white/50 bg-gradient-to-r from-blue-100 via-indigo-100 to-slate-100">
                                        {channel.header_image_url ? (
                                            <>
                                                <img
                                                    src={channel.header_image_url}
                                                    alt=""
                                                    className="w-full h-full object-cover"
                                                    loading="lazy"
                                                    referrerPolicy="no-referrer"
                                                    onError={(e) => {
                                                        (e.currentTarget as HTMLImageElement).style.display = 'none';
                                                    }}
                                                />
                                                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/45 via-slate-900/10 to-transparent" />
                                            </>
                                        ) : (
                                            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,.22),transparent_50%),radial-gradient(circle_at_80%_30%,rgba(99,102,241,.2),transparent_45%),linear-gradient(120deg,#e2e8f0,#dbeafe,#eef2ff)]" />
                                        )}
                                    </div>
                                </div>

                                <div className="flex justify-between items-start mb-4 gap-3">
                                    <div className="flex items-start gap-3 min-w-0">
                                        <div className="-mt-14 relative z-10 shrink-0">
                                            <div className="w-[4.5rem] h-[4.5rem] rounded-2xl bg-white p-0.5 shadow-lg ring-1 ring-slate-200 overflow-hidden">
                                                {channel.icon_url ? (
                                                    <img
                                                        src={channel.icon_url}
                                                        alt=""
                                                        className="w-full h-full object-cover rounded-2xl"
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
                                                    className={`w-full h-full rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 text-slate-600 font-bold text-base items-center justify-center ${channel.icon_url ? 'hidden' : 'flex'}`}
                                                >
                                                    {initials}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="min-w-0 pt-0.5">
                                            <h3 className="font-bold text-lg text-slate-800 truncate" title={channel.name}>{channel.name}</h3>
                                            <p className="text-xs text-slate-400 mt-0.5">ID: {channel.id}</p>
                                        </div>
                                    </div>
                                    <a href={channel.url} target="_blank" rel="noreferrer" className="text-slate-300 hover:text-blue-500 transition-colors p-2 hover:bg-blue-50 rounded-full shrink-0">
                                        <ExternalLink size={18} />
                                    </a>
                                </div>

                                {/* Stats Section */}
                                {stats && (
                                    <div className="grid grid-cols-3 gap-2 mb-4 text-center">
                                        <Link
                                            to={`/channel/${channel.id}`}
                                            className="bg-slate-50 rounded-lg px-2 py-1.5 border border-slate-100 hover:border-blue-200 hover:bg-blue-50 transition-colors"
                                        >
                                            <div className="flex items-center justify-center gap-1 text-blue-500 mb-0.5">
                                                <Film size={14} />
                                            </div>
                                            <p className="text-base font-bold text-slate-700 leading-tight">{stats.video_count}</p>
                                            <p className="text-[11px] text-slate-400">Videos</p>
                                        </Link>
                                        <Link
                                            to={`/channel/${channel.id}/transcripts`}
                                            className="bg-slate-50 rounded-lg px-2 py-1.5 border border-slate-100 hover:border-green-200 hover:bg-green-50 transition-colors"
                                        >
                                            <div className="flex items-center justify-center gap-1 text-green-500 mb-0.5">
                                                <FileText size={14} />
                                            </div>
                                            <p className="text-base font-bold text-slate-700 leading-tight">{stats.processed_count}</p>
                                            <p className="text-[11px] text-slate-400">Transcripts</p>
                                        </Link>
                                        <Link
                                            to={`/channel/${channel.id}/speakers`}
                                            className="bg-slate-50 rounded-lg px-2 py-1.5 border border-slate-100 hover:border-purple-200 hover:bg-purple-50 transition-colors"
                                        >
                                            <div className="flex items-center justify-center gap-1 text-purple-500 mb-0.5">
                                                <Users size={14} />
                                            </div>
                                            <p className="text-base font-bold text-slate-700 leading-tight">{stats.speaker_count}</p>
                                            <p className="text-[11px] text-slate-400">Speakers</p>
                                        </Link>
                                    </div>
                                )}

                                <div className="space-y-4 mb-6">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="text-slate-500">Status</span>
                                        <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${channel.status === 'active'
                                            ? 'bg-green-50 text-green-700 border-green-200'
                                            : 'bg-red-50 text-red-700 border-red-200'
                                            }`}>
                                            {channel.status}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="text-slate-500">Last Updated</span>
                                        <span className="text-slate-700 font-medium">{new Date(channel.last_updated).toLocaleDateString()}</span>
                                    </div>
                                    {stats && (
                                        <div className="flex items-center justify-between text-sm">
                                            <span className="text-slate-500">Runtime</span>
                                            <span className="text-slate-700 font-medium">{formatRuntime(stats.total_duration_seconds)}</span>
                                        </div>
                                    )}
                                </div>

                                <div className="flex gap-2 mt-auto">
                                    <button
                                        onClick={() => handleRefresh(channel.id)}
                                        disabled={refreshingIds.has(channel.id)}
                                        className="flex-1 bg-slate-50 text-slate-600 py-2.5 rounded-xl hover:bg-blue-50 hover:text-blue-600 hover:shadow-sm border border-slate-100 hover:border-blue-200 transition-all flex items-center justify-center gap-2 text-sm font-medium disabled:opacity-50"
                                    >
                                        {refreshingIds.has(channel.id) ? (
                                            <Loader2 size={16} className="animate-spin" />
                                        ) : (
                                            <RefreshCw size={16} className="group-hover:rotate-180 transition-transform duration-500" />
                                        )}
                                        {refreshingIds.has(channel.id) ? 'Scanning...' : 'Scan'}
                                    </button>
                                    <Link
                                        to={`/channel/${channel.id}`}
                                        className="flex-1 bg-blue-600 text-white py-2.5 rounded-xl hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-500/25 transition-all flex items-center justify-center gap-2 text-sm font-semibold border border-blue-700/20"
                                    >
                                        Open
                                        <ChevronRight size={16} />
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
