import { useEffect, useState } from 'react';
import api from '../lib/api';
import type { Video, Channel } from '../types';
import { Search, FileVideo, ArrowUpDown, ArrowUp, ArrowDown, Play } from 'lucide-react';
import { VideoStatusBadge } from '../components/VideoStatusBadge';

type SortField = 'published_at' | 'title' | 'duration' | 'status';
type SortOrder = 'asc' | 'desc';

export function Videos() {
    const [videos, setVideos] = useState<Video[]>([]);
    const [channels, setChannels] = useState<Channel[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [channelFilter, setChannelFilter] = useState<number | 'all'>('all');

    // Sorting state
    const [sortField, setSortField] = useState<SortField>('published_at');
    const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

    useEffect(() => {
        Promise.all([
            api.get<Video[]>('/videos/list'),
            api.get<Channel[]>('/channels')
        ]).then(([vidRes, chanRes]) => {
            setVideos(vidRes.data);
            setChannels(chanRes.data);
        }).catch(console.error).finally(() => setLoading(false));
    }, []);

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortOrder('desc'); // Default to desc for new field
        }
    };

    const getSortIcon = (field: SortField) => {
        if (sortField !== field) return <ArrowUpDown size={14} className="text-slate-300" />;
        return sortOrder === 'asc' ? <ArrowUp size={14} className="text-blue-500" /> : <ArrowDown size={14} className="text-blue-500" />;
    };

    const sortedVideos = [...videos].sort((a, b) => {
        let valA: any = a[sortField];
        let valB: any = b[sortField];

        if (sortField === 'published_at') {
            valA = new Date(a.published_at || 0).getTime();
            valB = new Date(b.published_at || 0).getTime();
        } else if (sortField === 'title') {
            valA = a.title.toLowerCase();
            valB = b.title.toLowerCase();
        }

        if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
        if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
        return 0;
    });

    const filtered = sortedVideos.filter(v => {
        const matchSearch = v.title.toLowerCase().includes(search.toLowerCase());
        const matchChannel = channelFilter === 'all' || v.channel_id === channelFilter;
        return matchSearch && matchChannel;
    });

    const getChannelName = (id: number) => channels.find(c => c.id === id)?.name || 'Unknown';

    return (
        <div className="h-[calc(100vh-2rem)] flex flex-col space-y-4">
            <div className="flex justify-between items-center shrink-0">
                <h2 className="text-2xl font-bold text-gray-800">Videos</h2>
                <div className="flex gap-4">
                    {/* Search & Filter Controls */}
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
                        <input
                            type="text"
                            placeholder="Filter titles..."
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                            className="pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none w-64"
                        />
                    </div>

                    <select
                        value={channelFilter}
                        onChange={e => setChannelFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                        className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                        <option value="all">All Channels</option>
                        {channels.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                    </select>
                </div>
            </div>

            {loading ? <div className="flex items-center justify-center h-64 text-slate-400 animate-pulse">Loading videos...</div> : (
                <div className="glass-panel rounded-xl overflow-hidden flex-1 flex flex-col">
                    <div className="overflow-auto flex-1">
                        <table className="w-full relative">
                            <thead className="bg-slate-50 sticky top-0 z-10 shadow-sm">
                                <tr className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                    <th className="px-6 py-4 w-20">Thumb</th>
                                    <th
                                        className="px-6 py-4 cursor-pointer hover:bg-slate-100 transition-colors"
                                        onClick={() => handleSort('title')}
                                    >
                                        <div className="flex items-center gap-2">
                                            Title {getSortIcon('title')}
                                        </div>
                                    </th>
                                    <th className="px-6 py-4">Channel</th>
                                    <th
                                        className="px-6 py-4 cursor-pointer hover:bg-slate-100 transition-colors"
                                        onClick={() => handleSort('published_at')}
                                    >
                                        <div className="flex items-center gap-2">
                                            Date {getSortIcon('published_at')}
                                        </div>
                                    </th>
                                    <th className="px-6 py-4">Status</th>
                                    <th className="px-6 py-4 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {filtered.map((video) => (
                                    <tr key={video.id} className="hover:bg-slate-50/50 transition-colors group">
                                        <td className="px-6 py-3">
                                            <div className="w-24 aspect-video bg-slate-200 rounded-md overflow-hidden relative">
                                                {video.thumbnail_url ? (
                                                    <img src={video.thumbnail_url} alt="" className="w-full h-full object-cover" />
                                                ) : (
                                                    <div className="flex items-center justify-center h-full text-slate-300">
                                                        <FileVideo size={20} />
                                                    </div>
                                                )}
                                            </div>
                                        </td>
                                        <td className="px-6 py-3">
                                            <div className="font-medium text-slate-800 line-clamp-2" title={video.title}>{video.title}</div>
                                            {video.description && (
                                                <div className="text-xs text-slate-400 line-clamp-1 mt-0.5">{video.description}</div>
                                            )}
                                        </td>
                                        <td className="px-6 py-3">
                                            <span className="text-sm text-slate-600">{getChannelName(video.channel_id)}</span>
                                        </td>
                                        <td className="px-6 py-3 text-sm text-slate-500">
                                            {video.published_at ? new Date(video.published_at).toLocaleDateString() : '-'}
                                        </td>
                                        <td className="px-6 py-3">
                                            <VideoStatusBadge
                                                status={video.status}
                                                processed={video.processed}
                                                className={video.muted ? 'bg-slate-100 text-slate-500' : ''}
                                            />
                                        </td>
                                        <td className="px-6 py-3 text-right">
                                            <button
                                                disabled={!video.processed}
                                                className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-30 disabled:hover:bg-transparent"
                                                title="Create Clip"
                                            >
                                                <Play size={16} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
