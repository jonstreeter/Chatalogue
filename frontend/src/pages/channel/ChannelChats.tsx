import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import api from '../../lib/api';
import type { EpisodeChatChannelItem } from '../../types';
import { Loader2, MessageSquareText, Search, Clock, Bot } from 'lucide-react';

const formatDateTime = (value?: string | null) => {
    if (!value) return 'No activity yet';
    const parsed = new Date(value);
    if (!Number.isFinite(parsed.getTime())) return 'No activity yet';
    return parsed.toLocaleString();
};

const formatPublishedDate = (value?: string | null) => {
    if (!value) return 'Unknown';
    const parsed = new Date(value);
    if (!Number.isFinite(parsed.getTime())) return 'Unknown';
    return parsed.toLocaleDateString();
};

export function ChannelChats() {
    const { id } = useParams<{ id: string }>();
    const [items, setItems] = useState<EpisodeChatChannelItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [query, setQuery] = useState('');

    const fetchItems = async () => {
        if (!id) return;
        setLoading(true);
        try {
            const res = await api.get<EpisodeChatChannelItem[]>(`/channels/${id}/episode-chat`);
            setItems(Array.isArray(res.data) ? res.data : []);
        } catch (e) {
            console.error('Failed to fetch channel chats', e);
            setItems([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void fetchItems();
    }, [id]);

    const filtered = useMemo(() => {
        const normalized = query.trim().toLowerCase();
        if (!normalized) return items;
        return items.filter((item) =>
            String(item.video_title || '').toLowerCase().includes(normalized) ||
            String(item.latest_thread_title || '').toLowerCase().includes(normalized) ||
            String(item.model || '').toLowerCase().includes(normalized) ||
            String(item.provider || '').toLowerCase().includes(normalized),
        );
    }, [items, query]);

    if (loading) {
        return (
            <div className="flex h-64 items-center justify-center text-slate-400">
                <Loader2 className="animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div className="glass-panel rounded-xl p-4">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-800">
                            <MessageSquareText size={18} className="text-indigo-500" />
                            Episode Chats
                        </h3>
                        <p className="mt-1 text-sm text-slate-500">
                            {items.length} episode{items.length === 1 ? '' : 's'} in this channel have saved AI chat threads.
                        </p>
                    </div>
                    <div className="relative w-full lg:w-96">
                        <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search by episode, thread, or model..."
                            className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm text-slate-700 outline-none transition focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
                        />
                    </div>
                </div>
            </div>

            {filtered.length === 0 ? (
                <div className="glass-panel rounded-xl p-10 text-center">
                    <MessageSquareText size={36} className="mx-auto mb-3 text-slate-300" />
                    <div className="text-slate-600 font-medium">No saved chats found</div>
                    <div className="mt-1 text-sm text-slate-400">
                        Start an episode chat from an episode page and it will appear here.
                    </div>
                </div>
            ) : (
                <div className="space-y-3">
                    {filtered.map((item) => (
                        <div key={item.video_id} className="glass-panel rounded-xl border border-slate-200 bg-white p-3.5">
                            <div className="flex flex-col gap-3 sm:flex-row">
                                <div className="h-20 w-full shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-slate-100 sm:h-16 sm:w-28">
                                    {item.video_thumbnail_url ? (
                                        <img src={item.video_thumbnail_url} alt="" className="h-full w-full object-cover" />
                                    ) : (
                                        <div className="flex h-full w-full items-center justify-center text-slate-300">
                                            <MessageSquareText size={18} />
                                        </div>
                                    )}
                                </div>
                                <div className="min-w-0 flex-1">
                                    <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
                                        <div className="min-w-0">
                                            <div className="truncate text-sm font-semibold text-slate-800" title={item.video_title}>
                                                {item.video_title}
                                            </div>
                                            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">
                                                    {item.thread_count} thread{item.thread_count === 1 ? '' : 's'}
                                                </span>
                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">
                                                    {item.message_count} message{item.message_count === 1 ? '' : 's'}
                                                </span>
                                                <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">
                                                    <Clock size={12} />
                                                    {formatPublishedDate(item.video_published_at)}
                                                </span>
                                            </div>
                                        </div>
                                        <Link
                                            to={`/video/${item.video_id}?tab=chat`}
                                            className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 transition hover:bg-indigo-100"
                                        >
                                            <MessageSquareText size={13} />
                                            Open Episode Chat
                                        </Link>
                                    </div>

                                    <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_240px]">
                                        <div className="rounded-xl bg-slate-50 px-3 py-3">
                                            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Latest thread</div>
                                            <div className="mt-1 text-sm font-medium text-slate-800">
                                                {item.latest_thread_title || 'New Chat'}
                                            </div>
                                            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                                <span className={`rounded-full px-2 py-0.5 ${item.latest_scope_mode === 'episode_related' ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>
                                                    {item.latest_scope_mode === 'episode_related' ? 'Episode + Related' : 'Episode'}
                                                </span>
                                                <span>
                                                Last activity: {formatDateTime(item.last_message_at)}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="rounded-xl bg-slate-50 px-3 py-3">
                                            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Model</div>
                                            <div className="mt-1 flex items-center gap-2 text-sm text-slate-800">
                                                <Bot size={14} className="text-slate-400" />
                                                <span className="truncate">{item.provider || 'default'}{item.model ? ` · ${item.model}` : ''}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
