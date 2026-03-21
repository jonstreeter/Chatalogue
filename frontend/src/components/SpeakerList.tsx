import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../lib/api';
import { toApiUrl } from '../lib/api';
import type { Speaker, SpeakerCounts } from '../types';
import { User, Edit2, Check, Mic2, ChevronDown, ChevronRight, Users, GitMerge, Brain, ExternalLink } from 'lucide-react';
import { SpeakerModal } from './SpeakerModal';

interface SpeakerListProps {
    channelId?: string;
    videoId?: number;
}

// Threshold in seconds - speakers with less total time are considered "extras"
const EXTRAS_THRESHOLD = 60; // 1 minute
const SPEAKER_CACHE_TTL_MS = 20_000;
const speakerListCache = new Map<string, { ts: number; items: Speaker[] }>();
const speakerCountsCache = new Map<string, { ts: number; counts: SpeakerCounts }>();
const speakerListInflight = new Map<string, Promise<Speaker[]>>();
const speakerCountsInflight = new Map<string, Promise<SpeakerCounts>>();

function formatTime(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

export function SpeakerList({ channelId, videoId }: SpeakerListProps) {
    const navigate = useNavigate();
    const [speakers, setSpeakers] = useState<Speaker[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingMore, setLoadingMore] = useState(false);
    const [hasMore, setHasMore] = useState(false);
    const [paginationEnabled, setPaginationEnabled] = useState(false);
    const [speakerCounts, setSpeakerCounts] = useState<SpeakerCounts | null>(null);
    const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
    const [showExtras, setShowExtras] = useState(false);
    const [speakerTab, setSpeakerTab] = useState<'identified' | 'unknown'>('identified');
    const [viewMode, setViewMode] = useState<'gallery' | 'list'>(
        channelId && !videoId ? 'gallery' : 'list'
    );
    const [thumbnailVersionBySpeakerId, setThumbnailVersionBySpeakerId] = useState<Record<number, number>>({});
    // Merge mode
    const [mergeMode, setMergeMode] = useState(false);
    const [mergeSelected, setMergeSelected] = useState<Set<number>>(new Set());
    const [merging, setMerging] = useState(false);
    const infiniteSentinelRef = useRef<HTMLDivElement | null>(null);
    const pageSize = 80;
    const canUsePagination = !videoId;
    const scopeKey = `channel:${channelId || 'all'}|video:${videoId ?? 'none'}`;

    const isFresh = (ts: number) => (Date.now() - ts) < SPEAKER_CACHE_TTL_MS;
    const clearScopeCache = () => {
        for (const key of speakerListCache.keys()) {
            if (key.startsWith(`${scopeKey}|`)) {
                speakerListCache.delete(key);
            }
        }
        speakerCountsCache.delete(scopeKey);
    };

    const fetchSpeakers = async (opts?: { append?: boolean; forceAll?: boolean }) => {
        const append = !!opts?.append;
        const forceAll = !!opts?.forceAll;
        try {
            const params: any = {};
            if (channelId) params.channel_id = channelId;
            if (videoId) params.video_id = videoId;
            const usePaging = canUsePagination && !forceAll;
            const offset = append ? speakers.length : 0;
            const limit = usePaging ? pageSize : null;
            const cacheKey = `${scopeKey}|offset:${offset}|limit:${limit ?? 'all'}`;

            if (usePaging) {
                params.offset = offset;
                params.limit = pageSize;
            }

            setPaginationEnabled(usePaging);
            if (append) setLoadingMore(true);
            else setLoading(true);

            if (!append) {
                const cached = speakerListCache.get(cacheKey);
                if (cached && isFresh(cached.ts)) {
                    setSpeakers(cached.items);
                    setHasMore(usePaging ? cached.items.length === pageSize : false);
                    setLoading(false);
                    return;
                }
            }

            let request = speakerListInflight.get(cacheKey);
            if (!request) {
                request = api.get<Speaker[]>('/speakers', { params })
                    .then((res) => (Array.isArray(res.data) ? res.data : []))
                    .finally(() => {
                        speakerListInflight.delete(cacheKey);
                    });
                speakerListInflight.set(cacheKey, request);
            }
            const items = await request;
            speakerListCache.set(cacheKey, { ts: Date.now(), items });
            if (append) {
                setSpeakers(prev => {
                    const seen = new Set(prev.map(s => s.id));
                    const next = [...prev];
                    for (const speaker of items) {
                        if (!seen.has(speaker.id)) next.push(speaker);
                    }
                    return next;
                });
            } else {
                setSpeakers(items);
            }
            setHasMore(usePaging ? items.length === pageSize : false);
        } catch (e) {
            console.error('Failed to fetch speakers:', e);
            if (!append) setSpeakers([]);
        } finally {
            setLoading(false);
            setLoadingMore(false);
        }
    };

    const fetchSpeakerCounts = async () => {
        try {
            const cached = speakerCountsCache.get(scopeKey);
            if (cached && isFresh(cached.ts)) {
                setSpeakerCounts(cached.counts);
                return;
            }

            const params: any = {};
            if (channelId) params.channel_id = channelId;
            if (videoId) params.video_id = videoId;

            let request = speakerCountsInflight.get(scopeKey);
            if (!request) {
                request = api.get<SpeakerCounts>('/speakers/stats', { params })
                    .then((res) => res.data)
                    .finally(() => {
                        speakerCountsInflight.delete(scopeKey);
                    });
                speakerCountsInflight.set(scopeKey, request);
            }

            const counts = await request;
            speakerCountsCache.set(scopeKey, { ts: Date.now(), counts });
            setSpeakerCounts(counts);
        } catch (e) {
            console.error('Failed to fetch speaker counts:', e);
            setSpeakerCounts(null);
        }
    };

    useEffect(() => {
        let cancelled = false;

        const load = async () => {
            setSpeakers([]);
            setHasMore(false);
            setLoading(true);
            await fetchSpeakers();
            if (!cancelled) {
                void fetchSpeakerCounts();
            }
        };

        void load();
        return () => {
            cancelled = true;
        };
    }, [channelId, videoId]);

    const isUnknownSpeaker = (speaker: Speaker) => {
        const name = (speaker.name || '').trim();
        if (!name) return true;
        if (/^unknown(\s+speaker)?$/i.test(name)) return true;
        if (/^speaker\s+\d+$/i.test(name)) return true;
        return false;
    };

    const identifiedSpeakers = speakers.filter(s => !isUnknownSpeaker(s));
    const unknownSpeakers = speakers.filter(s => isUnknownSpeaker(s));
    const identifiedCount = speakerCounts?.identified ?? identifiedSpeakers.length;
    const unknownCount = speakerCounts?.unknown ?? unknownSpeakers.length;

    // Separate main speakers from extras
    // A speaker is an "extra" if: is_extra is true OR total_speaking_time < threshold
    const mainSpeakers = identifiedSpeakers
        .filter(s => !s.is_extra && s.total_speaking_time >= EXTRAS_THRESHOLD)
        .sort((a, b) => b.total_speaking_time - a.total_speaking_time);

    const extraSpeakers = identifiedSpeakers
        .filter(s => s.is_extra || s.total_speaking_time < EXTRAS_THRESHOLD)
        .sort((a, b) => b.total_speaking_time - a.total_speaking_time);

    const sortedUnknownSpeakers = unknownSpeakers
        .slice()
        .sort((a, b) => b.total_speaking_time - a.total_speaking_time);
    const mainCount = speakerCounts?.main ?? mainSpeakers.length;
    const extrasCount = speakerCounts?.extras ?? extraSpeakers.length;
    const visibleTabLoadedCount = speakerTab === 'identified' ? identifiedSpeakers.length : sortedUnknownSpeakers.length;
    const visibleTabTotalCount = speakerTab === 'identified' ? identifiedCount : unknownCount;
    const visibleTabNeedsMore = paginationEnabled && visibleTabLoadedCount < visibleTabTotalCount;

    const loadMoreSpeakers = async () => {
        if (!paginationEnabled || loading || loadingMore || !hasMore || !visibleTabNeedsMore) return;
        await fetchSpeakers({ append: true });
    };

    useEffect(() => {
        if (!paginationEnabled || !hasMore || !visibleTabNeedsMore || loading || loadingMore) return;
        const node = infiniteSentinelRef.current;
        if (!node) return;

        const observer = new IntersectionObserver(
            (entries) => {
                const first = entries[0];
                if (first?.isIntersecting) {
                    void loadMoreSpeakers();
                }
            },
            { root: null, rootMargin: '500px 0px', threshold: 0.01 }
        );

        observer.observe(node);
        return () => observer.disconnect();
    }, [paginationEnabled, hasMore, visibleTabNeedsMore, loading, loadingMore, speakers.length, speakerTab, showExtras]);

    useEffect(() => {
        if (speakerTab === 'identified' && identifiedSpeakers.length === 0 && unknownSpeakers.length > 0) {
            setSpeakerTab('unknown');
        }
        if (speakerTab === 'unknown' && unknownSpeakers.length === 0 && identifiedSpeakers.length > 0) {
            setSpeakerTab('identified');
        }
    }, [speakerTab, identifiedSpeakers.length, unknownSpeakers.length]);

    useEffect(() => {
        if (speakerTab !== 'identified' && mergeMode) {
            setMergeMode(false);
            setMergeSelected(new Set());
        }
    }, [speakerTab, mergeMode]);

    // Calculate max speaking time for progress bars
    const maxSpeakingTime = Math.max(...speakers.map(s => s.total_speaking_time), 1);
    const showViewToggle = Boolean(channelId && !videoId);
    const galleryGridClass = 'grid [grid-template-columns:repeat(auto-fit,minmax(158px,196px))] justify-center gap-2 sm:gap-3';

    const handleOpenSpeakerEditor = (speaker: Speaker) => {
        setSelectedSpeaker(speaker);
    };

    const handleSpeakerRowClick = (speaker: Speaker) => {
        if (mergeMode) return;

        // In video context, keep the existing direct-edit modal behavior.
        if (videoId) {
            handleOpenSpeakerEditor(speaker);
            return;
        }

        // In the speaker page/channel speaker tab:
        // - Identified speakers open the new detail page by default
        // - Unknown speakers open the edit popup for quick assignment/merge cleanup
        if (speakerTab === 'identified' && !isUnknownSpeaker(speaker)) {
            navigate(`/speakers/${speaker.id}`);
            return;
        }

        handleOpenSpeakerEditor(speaker);
    };

    const handleModalClose = () => {
        setSelectedSpeaker(null);
    };

    const handleSpeakerUpdate = (updatedSpeaker: Speaker) => {
        clearScopeCache();
        setSpeakers(prev => prev.map(s => {
            if (s.id !== updatedSpeaker.id) return s;
            return {
                ...s,
                ...updatedSpeaker,
                // Be defensive: some mutation endpoints may return partial speaker payloads.
                total_speaking_time: (updatedSpeaker.total_speaking_time ?? 0) > 0
                    ? updatedSpeaker.total_speaking_time
                    : s.total_speaking_time,
                embedding_count: updatedSpeaker.embedding_count ?? s.embedding_count,
            };
        }));
        if (updatedSpeaker.thumbnail_path) {
            setThumbnailVersionBySpeakerId(prev => ({
                ...prev,
                [updatedSpeaker.id]: Date.now()
            }));
        }
        setSelectedSpeaker(updatedSpeaker);
    };

    const getSpeakerThumbSrc = (speaker: Speaker) => {
        if (!speaker.thumbnail_path) return null;
        const version = thumbnailVersionBySpeakerId[speaker.id];
        const base = toApiUrl(speaker.thumbnail_path);
        return version ? `${base}?v=${version}` : base;
    };

    const toggleMergeSelect = (id: number) => {
        setMergeSelected(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const handleMerge = async () => {
        if (mergeSelected.size < 2) return;

        // Pick the speaker with the most speaking time as the target
        const selectedSpeakers = speakers.filter(s => mergeSelected.has(s.id));
        selectedSpeakers.sort((a, b) => b.total_speaking_time - a.total_speaking_time);
        const target = selectedSpeakers[0];
        const sourceIds = selectedSpeakers.slice(1).map(s => s.id);

        const sourceNames = selectedSpeakers.slice(1).map(s => s.name).join(', ');
        if (!confirm(`Merge ${sourceNames} into "${target.name}"? This will combine their voice profiles and transcript segments.`)) return;

        setMerging(true);
        try {
            await api.post('/speakers/merge', {
                target_id: target.id,
                source_ids: sourceIds
            });
            setMergeMode(false);
            setMergeSelected(new Set());
            clearScopeCache();
            await Promise.all([fetchSpeakers(), fetchSpeakerCounts()]);
        } catch (e) {
            console.error('Failed to merge speakers:', e);
            alert('Failed to merge speakers');
        } finally {
            setMerging(false);
        }
    };

    const renderSpeakerRow = (speaker: Speaker, idx: number, isExtra: boolean = false) => {
        const percentage = Math.min(100, Math.max(1, (speaker.total_speaking_time / maxSpeakingTime) * 100));

        return (
            <div
                key={speaker.id}
                onClick={() => handleSpeakerRowClick(speaker)}
                className="group flex items-center gap-3 p-2 rounded-lg hover:bg-white hover:shadow-sm border border-transparent hover:border-slate-100 transition-all cursor-pointer"
            >
                {mergeMode && (
                    <div
                        className="flex-shrink-0 mr-1"
                        onClick={(e) => { e.stopPropagation(); toggleMergeSelect(speaker.id); }}
                    >
                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all cursor-pointer ${mergeSelected.has(speaker.id)
                            ? 'border-blue-500 bg-blue-500 text-white'
                            : 'border-slate-300 hover:border-blue-400'
                            }`}>
                            {mergeSelected.has(speaker.id) && <Check size={12} />}
                        </div>
                    </div>
                )}

                {/* Rank */}
                {!mergeMode && (
                    <div className="w-6 text-center text-xs font-mono text-slate-400">
                        {!isExtra ? idx + 1 : '-'}
                    </div>
                )}

                {/* Avatar */}
                <div className="relative w-10 h-10 flex-shrink-0 rounded-full bg-slate-100 overflow-hidden border border-slate-100">
                    {speaker.thumbnail_path ? (
                        <img
                            src={getSpeakerThumbSrc(speaker) || undefined}
                            alt={speaker.name}
                            className="w-full h-full object-cover"
                            loading="lazy"
                            decoding="async"
                        />
                    ) : (
                        <div className="flex items-center justify-center h-full">
                            <User size={16} className="text-slate-300" />
                        </div>
                    )}
                </div>

                {/* Info & Bar */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1.5 min-w-0">
                            <h3 className="text-sm font-medium text-slate-700 truncate" title={speaker.name}>
                                {speaker.name}
                            </h3>
                            <button
                                onClick={(e) => { e.stopPropagation(); handleOpenSpeakerEditor(speaker); }}
                                className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-blue-500 transition-opacity"
                                title="Edit speaker"
                            >
                                <Edit2 size={10} />
                            </button>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <Link
                                to={`/speakers/${speaker.id}`}
                                onClick={(e) => e.stopPropagation()}
                                className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-blue-600 transition-opacity p-0.5 rounded hover:bg-blue-50"
                                title="Open speaker detail page"
                            >
                                <ExternalLink size={12} />
                            </Link>
                            <span className="text-xs text-slate-500 font-mono">
                                {formatTime(speaker.total_speaking_time)}
                            </span>
                            {(speaker.embedding_count ?? 0) > 1 && (
                                <span className="text-[10px] text-blue-500 bg-blue-50 px-1.5 py-0.5 rounded-full flex items-center gap-0.5 font-medium" title={`${speaker.embedding_count} voice samples`}>
                                    <Brain size={9} />{speaker.embedding_count}
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Speaking Time Bar */}
                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full ${isExtra ? 'bg-slate-300' : 'bg-blue-400'} transition-all duration-500`}
                            style={{ width: `${percentage}%` }}
                        />
                    </div>
                </div>
            </div>
        );
    };

    const renderSpeakerCard = (speaker: Speaker, idx: number, isExtra: boolean = false) => {
        const percentage = Math.min(100, Math.max(1, (speaker.total_speaking_time / maxSpeakingTime) * 100));
        const unknown = isUnknownSpeaker(speaker);

        return (
            <div
                key={speaker.id}
                onClick={() => handleSpeakerRowClick(speaker)}
                className="group relative aspect-square w-full max-w-[196px] justify-self-center rounded-xl border border-slate-200 bg-white hover:shadow-md hover:border-blue-200 transition-all cursor-pointer p-2.5"
            >
                {mergeMode && (
                    <div
                        className="absolute top-2 left-2 z-10"
                        onClick={(e) => { e.stopPropagation(); toggleMergeSelect(speaker.id); }}
                    >
                        <div className={`w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all cursor-pointer shadow-sm ${mergeSelected.has(speaker.id)
                            ? 'border-blue-500 bg-blue-500 text-white'
                            : 'border-white/90 bg-white/90 text-slate-500 hover:border-blue-400'
                            }`}>
                            {mergeSelected.has(speaker.id) && <Check size={13} />}
                        </div>
                    </div>
                )}

                {!mergeMode && (
                    <div className="absolute top-2 left-2 z-10">
                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded-md border shadow-sm ${isExtra
                            ? 'bg-slate-50/95 text-slate-500 border-slate-200'
                            : 'bg-white/95 text-slate-600 border-slate-200'
                            }`}>
                            {isExtra ? 'EX' : `#${idx + 1}`}
                        </span>
                    </div>
                )}

                <div className="absolute top-2 right-2 z-10 flex items-center gap-1">
                    <button
                        onClick={(e) => { e.stopPropagation(); handleOpenSpeakerEditor(speaker); }}
                        className="p-1.5 rounded-md bg-white/90 border border-slate-200 text-slate-500 hover:text-blue-600 hover:border-blue-200 shadow-sm opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
                        title="Edit speaker"
                    >
                        <Edit2 size={12} />
                    </button>
                    {!unknown && (
                        <Link
                            to={`/speakers/${speaker.id}`}
                            onClick={(e) => e.stopPropagation()}
                            className="p-1.5 rounded-md bg-white/90 border border-slate-200 text-slate-500 hover:text-blue-600 hover:border-blue-200 shadow-sm opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
                            title="Open speaker detail page"
                        >
                            <ExternalLink size={12} />
                        </Link>
                    )}
                </div>

                <div className="flex h-full flex-col items-center text-center justify-between">
                    <div className="relative w-20 h-20 sm:w-24 sm:h-24 rounded-2xl bg-slate-100 overflow-hidden border border-slate-200 shadow-sm">
                        {speaker.thumbnail_path ? (
                            <img
                                src={getSpeakerThumbSrc(speaker) || undefined}
                                alt={speaker.name}
                                className="w-full h-full object-cover"
                                loading="lazy"
                                decoding="async"
                            />
                        ) : (
                            <div className="flex items-center justify-center h-full">
                                <User size={28} className="text-slate-300" />
                            </div>
                        )}
                    </div>

                    <div className="mt-2 w-full flex-1 flex flex-col">
                        <h3 className="text-[13px] sm:text-sm font-semibold text-slate-700 truncate" title={speaker.name}>
                            {speaker.name}
                        </h3>
                        <div className="mt-1 flex flex-wrap items-center justify-center gap-1 text-[11px] sm:text-xs">
                            <span className="text-slate-500 font-mono">{formatTime(speaker.total_speaking_time)}</span>
                            {(speaker.embedding_count ?? 0) > 1 && (
                                <span className="text-[10px] text-blue-500 bg-blue-50 px-1.5 py-0.5 rounded-full flex items-center gap-0.5 font-medium" title={`${speaker.embedding_count} voice samples`}>
                                    <Brain size={9} />{speaker.embedding_count}
                                </span>
                            )}
                        </div>
                        <div className="mt-auto pt-2 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full ${isExtra ? 'bg-slate-300' : 'bg-blue-400'} transition-all duration-500`}
                                style={{ width: `${percentage}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-32 text-slate-400 animate-pulse gap-2">
                <Loader2Pulse />
                <span>Loading speakers...</span>
            </div>
        );
    }

    return (
        <div className="space-y-6 pb-20">
            {speakers.length === 0 ? (
                <div className="text-center py-12 px-4 border-2 border-dashed border-slate-200 rounded-xl">
                    <Mic2 size={32} className="mx-auto text-slate-300 mb-3" />
                    <h3 className="text-slate-500 font-medium">No speakers identified</h3>
                    <p className="text-sm text-slate-400 mt-1">
                        We couldn't distinguish individual speakers in this video yet.
                    </p>
                </div>
            ) : (
                <>
                    {/* Identified vs Unknown Tabs */}
                    {(identifiedSpeakers.length > 0 || unknownSpeakers.length > 0) && (
                        <div className="glass-panel rounded-xl p-2">
                            <div className="flex items-center justify-between gap-2 flex-wrap">
                                <div className="flex gap-1.5 flex-wrap">
                                    <button
                                        type="button"
                                        onClick={() => setSpeakerTab('identified')}
                                        className={`px-2.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${speakerTab === 'identified'
                                            ? 'bg-blue-500 text-white shadow-sm'
                                            : 'text-slate-600 hover:bg-white'
                                            }`}
                                    >
                                        Identified ({identifiedCount})
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setSpeakerTab('unknown')}
                                        className={`px-2.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${speakerTab === 'unknown'
                                            ? 'bg-amber-500 text-white shadow-sm'
                                            : 'text-slate-600 hover:bg-white'
                                            }`}
                                    >
                                        Unknown ({unknownCount})
                                    </button>
                                </div>
                                {showViewToggle && (
                                    <div className="inline-flex gap-1 bg-white/80 rounded-lg p-1 border border-slate-200">
                                        <button
                                            type="button"
                                            onClick={() => setViewMode('gallery')}
                                            className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${viewMode === 'gallery'
                                                ? 'bg-slate-900 text-white'
                                                : 'text-slate-600 hover:bg-slate-100'
                                                }`}
                                        >
                                            Gallery
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setViewMode('list')}
                                            className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${viewMode === 'list'
                                                ? 'bg-slate-900 text-white'
                                                : 'text-slate-600 hover:bg-slate-100'
                                                }`}
                                        >
                                            List
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Main Speakers */}
                    {speakerTab === 'identified' && mainSpeakers.length > 0 && (
                        <div className="space-y-1">
                            <div className="flex items-center justify-between px-1.5 mb-1.5">
                                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Main Panel</h2>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-slate-400">{mainCount}</span>
                                    <button
                                        onClick={() => { setMergeMode(!mergeMode); setMergeSelected(new Set()); }}
                                        className={`text-xs px-2 py-0.5 rounded-full flex items-center gap-1 transition-colors font-medium ${mergeMode
                                            ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                            : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                                            }`}
                                    >
                                        <GitMerge size={11} />
                                        {mergeMode ? 'Cancel' : 'Merge'}
                                    </button>
                                </div>
                            </div>

                            {/* Merge action bar */}
                            {mergeMode && (
                                <div className="mx-2 mb-2 p-2.5 bg-blue-50 border border-blue-100 rounded-lg flex items-center justify-between">
                                    <span className="text-xs text-blue-700">
                                        {mergeSelected.size < 2
                                            ? 'Select 2 or more speakers to merge'
                                            : `${mergeSelected.size} speakers selected`}
                                    </span>
                                    <button
                                        disabled={mergeSelected.size < 2 || merging}
                                        onClick={handleMerge}
                                        className="px-3 py-1 bg-blue-600 text-white text-xs font-medium rounded-md hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-1"
                                    >
                                        <GitMerge size={12} />
                                        {merging ? 'Merging...' : 'Merge Selected'}
                                    </button>
                                </div>
                            )}

                            {viewMode === 'gallery' ? (
                                <div className={`${galleryGridClass} px-1`}>
                                    {mainSpeakers.map((speaker, idx) => renderSpeakerCard(speaker, idx))}
                                </div>
                            ) : (
                                mainSpeakers.map((speaker, idx) => renderSpeakerRow(speaker, idx))
                            )}
                        </div>
                    )}

                    {/* Extras Section */}
                    {speakerTab === 'identified' && extraSpeakers.length > 0 && (
                        <div className="pt-2 border-t border-slate-100">
                            <button
                                onClick={() => setShowExtras(!showExtras)}
                                className="w-full flex items-center justify-between px-2 py-2 text-slate-500 hover:text-slate-700 hover:bg-slate-50 rounded-lg transition-colors"
                            >
                                <div className="flex items-center gap-2">
                                    <Users size={16} />
                                    <span className="text-sm font-medium">Extras & Cleanup</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs bg-slate-100 px-1.5 py-0.5 rounded-full">{extrasCount}</span>
                                    {showExtras ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                </div>
                            </button>

                            {showExtras && (
                                <div className={`mt-1 animate-in slide-in-from-top-2 ${viewMode === 'gallery'
                                    ? ''
                                    : 'space-y-1 pl-2 border-l-2 border-slate-100 ml-4'
                                    }`}>
                                    {viewMode === 'gallery' ? (
                                        <div className={galleryGridClass}>
                                            {extraSpeakers.map((speaker, idx) => renderSpeakerCard(speaker, idx, true))}
                                        </div>
                                    ) : (
                                        extraSpeakers.map((speaker, idx) => renderSpeakerRow(speaker, idx, true))
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Unknown Speakers */}
                    {speakerTab === 'unknown' && (
                        <div className="space-y-2">
                            <div className="flex items-center justify-between px-2">
                                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Unknown Speakers</h2>
                                <span className="text-xs text-slate-400">{unknownCount}</span>
                            </div>
                            {sortedUnknownSpeakers.length === 0 ? (
                                <div className="text-center py-8 px-4 border border-dashed border-slate-200 rounded-xl text-slate-400 text-sm">
                                    No unknown speakers in this view.
                                </div>
                            ) : (
                                viewMode === 'gallery' ? (
                                    <div className={galleryGridClass}>
                                        {sortedUnknownSpeakers.map((speaker, idx) => renderSpeakerCard(speaker, idx, true))}
                                    </div>
                                ) : (
                                    <div className="space-y-1">
                                        {sortedUnknownSpeakers.map((speaker, idx) => renderSpeakerRow(speaker, idx, true))}
                                    </div>
                                )
                            )}
                        </div>
                    )}
                </>
            )}

            {paginationEnabled && speakers.length > 0 && (
                <div className="px-2">
                    <div
                        ref={infiniteSentinelRef}
                        className="rounded-xl border border-dashed border-slate-200 bg-white/70 px-4 py-3 text-center text-xs text-slate-500"
                    >
                        {loadingMore ? (
                            <span className="inline-flex items-center gap-2">
                                <Loader2Pulse />
                                Loading more speakers...
                            </span>
                        ) : !visibleTabNeedsMore ? (
                            speakerTab === 'identified'
                                ? `All identified speakers loaded (${identifiedCount}). Switch to Unknown to load more unknown speakers.`
                                : `All unknown speakers loaded (${unknownCount}).`
                        ) : hasMore ? (
                            `Loaded ${visibleTabLoadedCount} of ${visibleTabTotalCount} ${speakerTab} speakers. Scroll to load more.`
                        ) : (
                            `Loaded ${speakers.length} speakers. End of loaded speaker list.`
                        )}
                    </div>
                </div>
            )}

            {/* Speaker Modal */}
            {selectedSpeaker && (
                <SpeakerModal
                    speaker={selectedSpeaker}
                    onClose={handleModalClose}
                    onUpdate={handleSpeakerUpdate}
                    onMerge={() => {
                        setSelectedSpeaker(null);
                        clearScopeCache();
                        void Promise.all([fetchSpeakers(), fetchSpeakerCounts()]);
                    }}
                />
            )}
        </div>
    );
}

const Loader2Pulse = () => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="animate-spin"
    >
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
);
