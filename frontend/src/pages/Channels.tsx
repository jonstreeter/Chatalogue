import { useEffect, useState, useRef, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import type { Channel } from '../types';
import { RefreshCw, Plus, ExternalLink, ChevronRight, Film, Users, FileText, Loader2, Upload } from 'lucide-react';

interface ChannelStats {
    video_count: number;
    processed_count: number;
    speaker_count: number;
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
                <div className="w-full xl:w-auto flex flex-col gap-2 sm:flex-row sm:items-center">
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={importing}
                        className="inline-flex min-h-10 items-center justify-center gap-2 bg-white text-slate-600 px-4 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-sm font-medium disabled:opacity-50"
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
                            className="w-full min-w-0 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none sm:w-72"
                        />
                        <button
                            type="submit"
                            disabled={adding}
                            className="inline-flex min-h-10 items-center justify-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
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
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
                                className="glass-panel p-6 rounded-2xl hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 group"
                                style={{ animationDelay: `${idx * 100}ms` }}
                            >
                                <div className="-mx-6 -mt-6 mb-4">
                                    <div className="relative h-24 rounded-t-2xl overflow-hidden border-b border-white/50 bg-gradient-to-r from-blue-100 via-indigo-100 to-slate-100">
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
                                        <div className="-mt-10 relative z-10 shrink-0">
                                            <div className="w-14 h-14 rounded-xl bg-white p-0.5 shadow-lg ring-1 ring-slate-200 overflow-hidden">
                                                {channel.icon_url ? (
                                                    <img
                                                        src={channel.icon_url}
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
                                                <div
                                                    className={`w-full h-full rounded-[10px] bg-gradient-to-br from-slate-100 to-slate-200 text-slate-600 font-bold text-sm items-center justify-center ${channel.icon_url ? 'hidden' : 'flex'}`}
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
                                        <div className="bg-slate-50 rounded-lg p-2">
                                            <div className="flex items-center justify-center gap-1 text-blue-500 mb-1">
                                                <Film size={14} />
                                            </div>
                                            <p className="text-lg font-bold text-slate-700">{stats.video_count}</p>
                                            <p className="text-xs text-slate-400">Videos</p>
                                        </div>
                                        <div className="bg-slate-50 rounded-lg p-2">
                                            <div className="flex items-center justify-center gap-1 text-green-500 mb-1">
                                                <FileText size={14} />
                                            </div>
                                            <p className="text-lg font-bold text-slate-700">{stats.processed_count}</p>
                                            <p className="text-xs text-slate-400">Processed</p>
                                        </div>
                                        <div className="bg-slate-50 rounded-lg p-2">
                                            <div className="flex items-center justify-center gap-1 text-purple-500 mb-1">
                                                <Users size={14} />
                                            </div>
                                            <p className="text-lg font-bold text-slate-700">{stats.speaker_count}</p>
                                            <p className="text-xs text-slate-400">Speakers</p>
                                        </div>
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
                                </div>

                                <div className="flex gap-2">
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
                                        className="flex-1 bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-2.5 rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all flex items-center justify-center gap-2 text-sm font-medium"
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
