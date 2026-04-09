import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type {
    ContextChunk,
    SemanticIndexRebuildResponse,
    SemanticIndexStatus,
    SemanticSearchHit,
    SemanticSearchPage,
    Speaker,
    SpeakerSample,
    TranscriptDiarizationRebuildQueueResponse,
    TranscriptOptimizeDryRunResponse,
    TranscriptQuality,
    TranscriptRepairQueueResponse,
    TranscriptRetranscriptionQueueResponse,
    Video,
} from '../../types';

interface PlayerController {
    seekTo(seconds: number, allowSeekAhead?: boolean): void;
    playVideo(): void;
    pauseVideo(): void;
    getCurrentTime?(): number;
}
import { SpeakerModal } from '../../components/SpeakerModal';
import {
    ChevronDown,
    ChevronUp,
    Clock,
    FileText,
    Layers,
    PlayCircle,
    RefreshCw,
    GitMerge,
    AudioLines,
    RotateCcw,
    Search,
    Tv,
    User,
    Zap,
} from 'lucide-react';

const PREVIEW_STOP_EARLY_SECONDS = 0.35;

type SearchMode = 'exact' | 'semantic' | 'hybrid';

interface SearchResult {
    id: number;
    video_id: number;
    speaker_id?: number;
    start_time: number;
    end_time: number;
    text: string;
    speaker?: string;
}

interface SearchResponse {
    items: SearchResult[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
}

// Unified type used by the preview panel regardless of search mode.
interface PreviewTarget {
    id: number;
    video_id: number;
    start_time: number;
    end_time: number;
    text: string;
    speaker?: string;
    speaker_id?: number;
    score?: number;
    context_before?: ContextChunk[];
    context_after?: ContextChunk[];
}

function normaliseExact(r: SearchResult): PreviewTarget {
    return { id: r.id, video_id: r.video_id, start_time: r.start_time, end_time: r.end_time, text: r.text, speaker: r.speaker, speaker_id: r.speaker_id };
}

function normaliseSemantic(r: SemanticSearchHit): PreviewTarget {
    return { id: r.id, video_id: r.video_id, start_time: r.start_time, end_time: r.end_time, text: r.chunk_text, speaker: r.speaker_name, speaker_id: r.speaker_id, score: r.score, context_before: r.context_before, context_after: r.context_after };
}

const MODE_LABELS: Record<SearchMode, string> = {
    exact: 'Exact',
    semantic: 'Semantic',
    hybrid: 'Hybrid',
};
const MODE_TIPS: Record<SearchMode, string> = {
    exact: 'Finds the same words in transcripts.',
    semantic: 'Finds similar ideas even when wording differs.',
    hybrid: 'Combines exact and semantic ranking for best coverage.',
};

export function ChannelSearch() {
    const { id } = useParams<{ id: string }>();

    // ── Search state ──────────────────────────────────────────────────────────
    const [searchMode, setSearchMode] = useState<SearchMode>('exact');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [semanticResults, setSemanticResults] = useState<SemanticSearchHit[]>([]);
    const [totalResults, setTotalResults] = useState(0);
    const [resultOffset, setResultOffset] = useState(0);
    const [resultLimit, setResultLimit] = useState(50);
    const [hasMoreResults, setHasMoreResults] = useState(false);
    const [searching, setSearching] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [yearFilter, setYearFilter] = useState('');
    const [monthFilter, setMonthFilter] = useState('');
    const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');

    // ── Videos + preview ─────────────────────────────────────────────────────
    const [videos, setVideos] = useState<Video[]>([]);
    const [loadingVideos, setLoadingVideos] = useState(true);
    const [selectedPreview, setSelectedPreview] = useState<PreviewTarget | null>(null);
    const [previewPlayer, setPreviewPlayer] = useState<PlayerController | null>(null);

    // ── Speaker modal ─────────────────────────────────────────────────────────
    const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
    const [initialSpeakerSample, setInitialSpeakerSample] = useState<SpeakerSample | null>(null);

    // ── Semantic index status ─────────────────────────────────────────────────
    const [indexStatus, setIndexStatus] = useState<SemanticIndexStatus | null>(null);
    const [rebuildLoading, setRebuildLoading] = useState(false);
    const [optimizationBacklog, setOptimizationBacklog] = useState<TranscriptQuality[]>([]);
    const [loadingOptimizationBacklog, setLoadingOptimizationBacklog] = useState(false);
    const [optimizationBacklogError, setOptimizationBacklogError] = useState<string | null>(null);
    const [optimizationTierFilter, setOptimizationTierFilter] = useState<'all' | 'low_risk_repair' | 'diarization_rebuild' | 'full_retranscription' | 'manual_review' | 'none'>('all');
    const [queueingOptimizationVideoId, setQueueingOptimizationVideoId] = useState<number | null>(null);

    // ── Context expansion ─────────────────────────────────────────────────────
    const [expandedContext, setExpandedContext] = useState<Set<number>>(new Set());

    // ── Scroll + hydration ────────────────────────────────────────────────────
    const [searchStateHydrated, setSearchStateHydrated] = useState(false);
    const resultsListRef = useRef<HTMLDivElement>(null);
    const savedScrollTopRef = useRef<number | null>(null);
    const nativePreviewRef = useRef<HTMLMediaElement | null>(null);

    const getPersistKey = () => (id ? `chatalogue:channel-search:${id}` : null);

    // ── State persistence ─────────────────────────────────────────────────────

    const persistSearchState = (scrollTopOverride?: number) => {
        const key = getPersistKey();
        if (!key) return;
        try {
            sessionStorage.setItem(key, JSON.stringify({
                query, searchMode, results, semanticResults, totalResults, resultOffset,
                resultLimit, hasMoreResults, hasSearched, yearFilter, monthFilter, sortOrder,
                selectedPreviewId: selectedPreview?.id ?? null,
                scrollTop: scrollTopOverride ?? resultsListRef.current?.scrollTop ?? 0,
            }));
        } catch { /* ignore */ }
    };

    useEffect(() => {
        const key = getPersistKey();
        if (!key) return;
        try {
            const saved = JSON.parse(sessionStorage.getItem(key) || 'null');
            if (!saved) return;
            const savedResults: SearchResult[] = Array.isArray(saved?.results) ? saved.results : [];
            const savedSemResults: SemanticSearchHit[] = Array.isArray(saved?.semanticResults) ? saved.semanticResults : [];
            setQuery(typeof saved?.query === 'string' ? saved.query : '');
            setSearchMode((['exact', 'semantic', 'hybrid'] as SearchMode[]).includes(saved?.searchMode) ? saved.searchMode : 'exact');
            setResults(savedResults);
            setSemanticResults(savedSemResults);
            setTotalResults(typeof saved?.totalResults === 'number' ? saved.totalResults : 0);
            setResultOffset(typeof saved?.resultOffset === 'number' ? saved.resultOffset : 0);
            setResultLimit(typeof saved?.resultLimit === 'number' ? saved.resultLimit : 50);
            setHasMoreResults(Boolean(saved?.hasMoreResults));
            setHasSearched(Boolean(saved?.hasSearched));
            setYearFilter(typeof saved?.yearFilter === 'string' ? saved.yearFilter : '');
            setMonthFilter(typeof saved?.monthFilter === 'string' ? saved.monthFilter : '');
            setSortOrder(saved?.sortOrder === 'oldest' ? 'oldest' : 'newest');
            savedScrollTopRef.current = typeof saved?.scrollTop === 'number' ? saved.scrollTop : null;
        } catch { /* ignore */ } finally {
            setSearchStateHydrated(true);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [id]);

    useEffect(() => {
        if (!hasSearched || !resultsListRef.current || savedScrollTopRef.current == null) return;
        const top = savedScrollTopRef.current;
        window.requestAnimationFrame(() => { if (resultsListRef.current) resultsListRef.current.scrollTop = top; });
        savedScrollTopRef.current = null;
    }, [hasSearched, results.length, semanticResults.length]);

    useEffect(() => {
        if (!searchStateHydrated) return;
        persistSearchState();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchStateHydrated, id, query, searchMode, hasSearched, results, semanticResults, totalResults, resultOffset, resultLimit, hasMoreResults, yearFilter, monthFilter, sortOrder, selectedPreview]);

    // ── Videos fetch ──────────────────────────────────────────────────────────

    useEffect(() => {
        let cancelled = false;
        const fetch = async () => {
            if (!id) return;
            setLoadingVideos(true);
            setPreviewPlayer(null);
            try {
                const res = await api.get<Video[]>('/videos/list', { params: { channel_id: id } });
                if (!cancelled) setVideos(res.data);
            } catch { if (!cancelled) setVideos([]); }
            finally { if (!cancelled) setLoadingVideos(false); }
        };
        void fetch();
        return () => { cancelled = true; };
    }, [id]);

    // ── Semantic index status ─────────────────────────────────────────────────

    const fetchIndexStatus = useCallback(async () => {
        try {
            const res = await api.get<SemanticIndexStatus>('/semantic-index/status');
            setIndexStatus(res.data);
        } catch { /* ignore */ }
    }, []);

    const fetchOptimizationBacklog = useCallback(async () => {
        if (!id) return;
        setLoadingOptimizationBacklog(true);
        setOptimizationBacklogError(null);
        try {
            const res = await api.post<TranscriptOptimizeDryRunResponse>('/transcripts/optimize/dry-run', {
                channel_id: Number(id),
                limit: 250,
                persist_snapshots: false,
            });
            setOptimizationBacklog(Array.isArray(res.data?.items) ? res.data.items : []);
        } catch (e: any) {
            console.error('Failed to load transcript optimization backlog:', e);
            setOptimizationBacklog([]);
            setOptimizationBacklogError(e?.response?.data?.detail || 'Failed to load transcript optimization backlog');
        } finally {
            setLoadingOptimizationBacklog(false);
        }
    }, [id]);

    useEffect(() => {
        void fetchIndexStatus();
    }, [fetchIndexStatus]);

    useEffect(() => {
        void fetchOptimizationBacklog();
    }, [fetchOptimizationBacklog]);

    useEffect(() => {
        if (!indexStatus?.is_running) return;
        const interval = window.setInterval(() => void fetchIndexStatus(), 3000);
        return () => window.clearInterval(interval);
    }, [indexStatus?.is_running, fetchIndexStatus]);

    const handleRebuildIndex = async () => {
        if (!id || rebuildLoading) return;
        setRebuildLoading(true);
        try {
            await api.post<SemanticIndexRebuildResponse>(`/channels/${id}/semantic-index/rebuild`);
            await fetchIndexStatus();
        } catch { /* ignore */ }
        finally { setRebuildLoading(false); }
    };

    const queueOptimizationForCandidate = async (candidate: TranscriptQuality) => {
        const videoId = Number(candidate.video_id);
        if (!videoId || queueingOptimizationVideoId != null) return;
        const tier = String(candidate.recommended_tier || 'none');
        const confirmText = tier === 'low_risk_repair'
            ? 'Queue the low-risk repair pass for this episode?'
            : tier === 'diarization_rebuild'
                ? 'Queue a diarization rebuild for this episode?'
                : tier === 'full_retranscription'
                    ? 'Queue a full retranscription for this episode?'
                    : '';
        if (!confirmText) return;
        if (!window.confirm(confirmText)) return;

        setQueueingOptimizationVideoId(videoId);
        try {
            if (tier === 'low_risk_repair') {
                const res = await api.post<TranscriptRepairQueueResponse>(`/videos/${videoId}/transcript-repair`, {});
                window.alert(`Low-risk repair queued as job ${res.data.job_id}.`);
            } else if (tier === 'diarization_rebuild') {
                const res = await api.post<TranscriptDiarizationRebuildQueueResponse>(`/videos/${videoId}/transcript-diarization-rebuild`, {});
                window.alert(`Diarization rebuild queued as job ${res.data.job_id}.`);
            } else if (tier === 'full_retranscription') {
                const res = await api.post<TranscriptRetranscriptionQueueResponse>(`/videos/${videoId}/transcript-retranscribe`, {});
                window.alert(`Full retranscription queued as job ${res.data.job_id}.`);
            }
            await fetchOptimizationBacklog();
        } catch (e: any) {
            window.alert(e?.response?.data?.detail || 'Failed to queue transcript optimization');
        } finally {
            setQueueingOptimizationVideoId(null);
        }
    };

    // ── Search ────────────────────────────────────────────────────────────────

    const runSearch = async (nextOffset: number = 0) => {
        if (!query.trim()) return;
        setSearching(true);
        setHasSearched(true);
        try {
            if (searchMode === 'exact') {
                const res = await api.get<SearchResponse>('/search', {
                    params: {
                        q: query, channel_id: id, offset: nextOffset, limit: resultLimit,
                        ...(yearFilter ? { year: Number(yearFilter) } : {}),
                        ...(monthFilter ? { month: Number(monthFilter) } : {}),
                        sort: sortOrder,
                    }
                });
                setResults(Array.isArray(res.data.items) ? res.data.items : []);
                setTotalResults(res.data.total ?? 0);
                setResultOffset(res.data.offset ?? nextOffset);
                setHasMoreResults(Boolean(res.data.has_more));
                setSemanticResults([]);
            } else {
                const res = await api.post<SemanticSearchPage>('/search/semantic', {
                    query, channel_id: Number(id), mode: searchMode, offset: nextOffset, limit: resultLimit,
                    ...(yearFilter ? { year: Number(yearFilter) } : {}),
                    ...(monthFilter ? { month: Number(monthFilter) } : {}),
                });
                setSemanticResults(Array.isArray(res.data.items) ? res.data.items : []);
                setTotalResults(res.data.total ?? 0);
                setResultOffset(nextOffset);
                setHasMoreResults(nextOffset + resultLimit < (res.data.total ?? 0));
                setResults([]);
            }
            setSelectedPreview(null);
            setExpandedContext(new Set());
            if (resultsListRef.current) resultsListRef.current.scrollTop = 0;
        } catch (e) {
            console.error('Search failed:', e);
        } finally {
            setSearching(false);
        }
    };

    const handleSearch = async (e: FormEvent) => {
        e.preventDefault();
        await runSearch(0);
    };

    // ── Preview ───────────────────────────────────────────────────────────────

    const playPreviewClip = useCallback((target: PreviewTarget) => {
        if (!previewPlayer) return;
        try {
            previewPlayer.seekTo(target.start_time, true);
            previewPlayer.playVideo();
        } catch { /* ignore */ }
    }, [previewPlayer]);

    const pausePreview = () => {
        try { previewPlayer?.pauseVideo?.(); } catch { /* ignore */ }
    };

    const buildNativePreviewPlayer = (media: HTMLMediaElement) => ({
        seekTo: (s: number) => { media.currentTime = Math.max(0, s || 0); },
        playVideo: async () => { try { await media.play(); } catch { /* ignore */ } },
        pauseVideo: () => media.pause(),
        getCurrentTime: () => media.currentTime || 0,
    });

    useEffect(() => {
        if (!selectedPreview || !previewPlayer) return;
        playPreviewClip(selectedPreview);
    }, [selectedPreview, previewPlayer, playPreviewClip]);

    useEffect(() => {
        if (!previewPlayer || !selectedPreview) return;
        const interval = window.setInterval(() => {
            try {
                const current = previewPlayer.getCurrentTime?.();
                const stopAt = Math.max(selectedPreview.start_time, selectedPreview.end_time - PREVIEW_STOP_EARLY_SECONDS);
                if (typeof current === 'number' && current >= stopAt) previewPlayer.pauseVideo?.();
            } catch { /* ignore */ }
        }, 200);
        return () => window.clearInterval(interval);
    }, [previewPlayer, selectedPreview]);

    const handleResultClick = (target: PreviewTarget) => {
        if (selectedPreview?.id === target.id) {
            playPreviewClip(target);
        } else {
            setSelectedPreview(target);
        }
    };

    // ── Speaker modal ─────────────────────────────────────────────────────────

    const handleSpeakerClick = async (target: PreviewTarget, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!target.speaker_id) return;
        pausePreview();
        const resultVideo = videoMap.get(target.video_id);
        setInitialSpeakerSample({
            youtube_id: resultVideo?.youtube_id,
            video_id: target.video_id,
            start_time: target.start_time,
            end_time: target.end_time,
            text: target.text,
            media_source_type: resultVideo?.media_source_type,
            media_kind: resultVideo?.media_kind,
        });
        try {
            const res = await api.get<Speaker>(`/speakers/${target.speaker_id}`);
            setSelectedSpeaker(res.data);
        } catch { /* ignore */ }
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    const formatTime = (seconds: number) => {
        const total = Math.max(0, Math.floor(seconds));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        return h > 0 ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}` : `${m}:${s.toString().padStart(2, '0')}`;
    };

    const formatPublished = (iso?: string | null) => {
        if (!iso) return 'Unknown date';
        const d = new Date(iso);
        return Number.isNaN(d.getTime()) ? 'Unknown date' : d.toLocaleDateString();
    };

    const videoMap = new Map(videos.map((v) => [v.id, v]));
    const previewVideo = selectedPreview ? videoMap.get(selectedPreview.video_id) : null;
    const previewMediaSourceType = String(previewVideo?.media_source_type || 'youtube').toLowerCase();
    const previewUsesYoutube = previewMediaSourceType === 'youtube';
    const previewUsesLocalMedia = previewMediaSourceType === 'upload' || previewMediaSourceType === 'tiktok';
    const previewIsAudioOnly = previewUsesLocalMedia && String(previewVideo?.media_kind || '').toLowerCase() === 'audio';
    const previewMediaUrl = previewVideo ? toApiUrl(`/videos/${previewVideo.id}/media`) : '';
    const buildPreviewEpisodeLink = (target: PreviewTarget) => {
        const params = new URLSearchParams({
            t: String(Math.max(0, Math.floor(target.start_time))),
            segment_id: String(target.id),
            search_mode: searchMode,
        });
        if (searchMode === 'exact' && query.trim()) {
            params.set('q', query.trim());
        }
        return `/video/${target.video_id}?${params.toString()}`;
    };
    const previewMediaPending = previewUsesLocalMedia && ['pending', 'queued'].includes(String(previewVideo?.status || '').toLowerCase());

    const totalPages = Math.max(1, Math.ceil(totalResults / Math.max(resultLimit, 1)));
    const currentPage = Math.floor(resultOffset / Math.max(resultLimit, 1)) + 1;
    const pageStart = totalResults === 0 ? 0 : resultOffset + 1;
    const activeResults = searchMode === 'exact' ? results : semanticResults;
    const pageEnd = Math.min(resultOffset + activeResults.length, totalResults);
    const showPagination = totalResults > resultLimit;

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const publishedVideoDates = videos
        .map((v) => ({ v, d: v.published_at ? new Date(v.published_at) : null }))
        .filter((x): x is { v: Video; d: Date } => !!x.d && !Number.isNaN(x.d.getTime()));
    const yearOptions = Array.from(new Set(publishedVideoDates.map(({ d }) => d.getFullYear()))).sort((a, b) => b - a);
    const monthOptions = Array.from(new Set(
        publishedVideoDates.filter(({ d }) => !yearFilter || d.getFullYear() === Number(yearFilter)).map(({ d }) => d.getMonth() + 1)
    )).sort((a, b) => a - b);

    const optimizationTierCounts = useMemo(() => {
        return optimizationBacklog.reduce<Record<string, number>>((acc, item) => {
            const key = String(item.recommended_tier || 'none');
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
    }, [optimizationBacklog]);

    const filteredOptimizationBacklog = useMemo(() => {
        const sorted = [...optimizationBacklog].sort((a, b) => {
            const rank = (tier: string) => (
                tier === 'full_retranscription' ? 0 :
                tier === 'diarization_rebuild' ? 1 :
                tier === 'low_risk_repair' ? 2 :
                tier === 'manual_review' ? 3 :
                4
            );
            const tierDiff = rank(String(a.recommended_tier || 'none')) - rank(String(b.recommended_tier || 'none'));
            if (tierDiff !== 0) return tierDiff;
            return Number(a.quality_score || 0) - Number(b.quality_score || 0);
        });
        if (optimizationTierFilter === 'all') return sorted;
        return sorted.filter((item) => String(item.recommended_tier || 'none') === optimizationTierFilter);
    }, [optimizationBacklog, optimizationTierFilter]);

    const formatOptimizationTierLabel = (tier: string) => (
        tier === 'low_risk_repair' ? 'Repair' :
        tier === 'diarization_rebuild' ? 'Rebuild' :
        tier === 'full_retranscription' ? 'Retranscribe' :
        tier === 'manual_review' ? 'Manual Review' :
        'No Auto'
    );

    const optimizationTierBadgeClass = (tier: string) => (
        tier === 'low_risk_repair' ? 'bg-emerald-100 text-emerald-700 border-emerald-200' :
        tier === 'diarization_rebuild' ? 'bg-blue-100 text-blue-700 border-blue-200' :
        tier === 'full_retranscription' ? 'bg-amber-100 text-amber-700 border-amber-200' :
        tier === 'manual_review' ? 'bg-rose-100 text-rose-700 border-rose-200' :
        'bg-slate-100 text-slate-600 border-slate-200'
    );

    const handleYearChange = (value: string) => { setYearFilter(value); setMonthFilter(''); };

    const toggleContext = (id: number) => {
        setExpandedContext((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id); else next.add(id);
            return next;
        });
    };

    // ── Score badge ───────────────────────────────────────────────────────────

    const ScoreBadge = ({ score }: { score: number }) => {
        const pct = Math.round(score * 100);
        const color = pct >= 80 ? 'bg-emerald-100 text-emerald-700' : pct >= 60 ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-500';
        return (
            <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium ${color}`}>
                <Zap size={9} />
                {pct}%
            </span>
        );
    };

    // ── Index status panel ────────────────────────────────────────────────────

    const IndexStatusPanel = () => {
        if (!indexStatus) return null;
        const { is_running, videos_completed, videos_total, current_video_title, last_finished_at } = indexStatus;
        const hasIndex = !is_running && last_finished_at;
        return (
            <div className="mt-4 pt-4 border-t border-slate-100">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-slate-600 flex items-center gap-1.5">
                        <Layers size={12} />
                        Semantic Index
                    </span>
                    <button
                        type="button"
                        onClick={() => void handleRebuildIndex()}
                        disabled={rebuildLoading || is_running}
                        className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                        title="Rebuild semantic index for this channel"
                    >
                        <RefreshCw size={10} className={rebuildLoading || is_running ? 'animate-spin' : ''} />
                        {is_running ? 'Indexing…' : 'Rebuild'}
                    </button>
                </div>
                {is_running ? (
                    <div className="text-xs text-slate-500">
                        <div className="flex justify-between mb-1">
                            <span>Indexing {videos_completed}/{videos_total} videos</span>
                        </div>
                        {current_video_title && (
                            <p className="truncate text-slate-400" title={current_video_title}>{current_video_title}</p>
                        )}
                        <div className="mt-1.5 h-1 rounded-full bg-slate-100 overflow-hidden">
                            <div
                                className="h-full bg-blue-400 rounded-full transition-all"
                                style={{ width: videos_total > 0 ? `${Math.round((videos_completed / videos_total) * 100)}%` : '0%' }}
                            />
                        </div>
                    </div>
                ) : hasIndex ? (
                    <p className="text-xs text-emerald-600">Ready — semantic search available</p>
                ) : (
                    <p className="text-xs text-slate-400">Not indexed. Click Rebuild to enable semantic search.</p>
                )}
            </div>
        );
    };

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(460px,560px)]">
            <div className="space-y-6 min-w-0">

                {/* Search form */}
                <form onSubmit={handleSearch} className="glass-panel rounded-xl p-3 sm:p-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
                        <div className="flex-1 relative">
                            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Search transcripts across this channel..."
                                className="w-full min-h-10 pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={searching}
                            className="px-5 min-h-10 py-2 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-lg hover:shadow-lg hover:shadow-blue-500/25 transition-all font-medium disabled:opacity-50"
                        >
                            {searching ? 'Searching…' : 'Search'}
                        </button>
                    </div>

                    {/* Mode selector */}
                    <div className="mt-2.5 flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg w-fit">
                        {(['exact', 'semantic', 'hybrid'] as SearchMode[]).map((mode) => (
                            <button
                                key={mode}
                                type="button"
                                onClick={() => setSearchMode(mode)}
                                title={MODE_TIPS[mode]}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                                    searchMode === mode
                                        ? 'bg-white shadow text-blue-700 font-semibold'
                                        : 'text-slate-500 hover:text-slate-700'
                                }`}
                            >
                                {MODE_LABELS[mode]}
                            </button>
                        ))}
                        <span className="ml-2 text-[11px] text-slate-400 hidden sm:inline">{MODE_TIPS[searchMode]}</span>
                    </div>

                    {/* Filters */}
                    <div className="mt-2.5 grid grid-cols-2 md:grid-cols-4 gap-2">
                        <select value={yearFilter} onChange={(e) => handleYearChange(e.target.value)}
                            className="w-full min-h-9 px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none">
                            <option value="">All years</option>
                            {yearOptions.map((y) => <option key={y} value={y}>{y}</option>)}
                        </select>
                        <select value={monthFilter} onChange={(e) => setMonthFilter(e.target.value)}
                            disabled={!yearFilter}
                            className="w-full min-h-9 px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50">
                            <option value="">All months</option>
                            {monthOptions.map((m) => <option key={m} value={m}>{monthNames[m - 1]}</option>)}
                        </select>
                        <div className="space-y-1">
                            <label className="text-[11px] text-slate-500">Per page</label>
                            <select value={resultLimit} onChange={(e) => setResultLimit(Number(e.target.value))}
                                className="w-full min-h-9 px-2 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none">
                                {[25, 50, 100].map((n) => <option key={n} value={n}>{n}</option>)}
                            </select>
                        </div>
                        {searchMode === 'exact' ? (
                            <div className="space-y-1 col-span-2 md:col-span-1">
                                <label className="text-[11px] text-slate-500">Sort</label>
                                <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value === 'oldest' ? 'oldest' : 'newest')}
                                    className="w-full min-h-9 px-2 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none">
                                    <option value="newest">Newest first</option>
                                    <option value="oldest">Chronological</option>
                                </select>
                            </div>
                        ) : (
                            <div className="col-span-2 md:col-span-1 flex items-end">
                                <p className="text-xs text-slate-400 leading-relaxed">Results ranked by relevance score</p>
                            </div>
                        )}
                    </div>
                </form>

                <div className="glass-panel rounded-xl p-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div className="min-w-0">
                            <div className="text-sm font-semibold text-slate-800">Transcript Optimization Backlog</div>
                            <div className="mt-1 text-xs text-slate-500">
                                Evaluator-ranked upgrade candidates for this channel. Review by tier, then queue repairs, diarization rebuilds, or full retranscriptions.
                            </div>
                            <div className="mt-3 flex flex-wrap gap-2">
                                {([
                                    ['all', `All (${optimizationBacklog.length})`],
                                    ['low_risk_repair', `Repair (${optimizationTierCounts.low_risk_repair || 0})`],
                                    ['diarization_rebuild', `Rebuild (${optimizationTierCounts.diarization_rebuild || 0})`],
                                    ['full_retranscription', `Retranscribe (${optimizationTierCounts.full_retranscription || 0})`],
                                    ['manual_review', `Manual (${optimizationTierCounts.manual_review || 0})`],
                                ] as const).map(([key, label]) => (
                                    <button
                                        key={key}
                                        type="button"
                                        onClick={() => setOptimizationTierFilter(key)}
                                        className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                                            optimizationTierFilter === key
                                                ? 'border-slate-300 bg-white text-slate-900 shadow-sm'
                                                : 'border-slate-200 bg-slate-50 text-slate-500 hover:bg-white'
                                        }`}
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={() => void fetchOptimizationBacklog()}
                            disabled={loadingOptimizationBacklog}
                            className="inline-flex items-center gap-2 self-start rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                        >
                            <RefreshCw size={14} className={loadingOptimizationBacklog ? 'animate-spin' : ''} />
                            Refresh Backlog
                        </button>
                    </div>

                    <div className="mt-4">
                        {loadingOptimizationBacklog ? (
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                                <RefreshCw size={14} className="animate-spin" />
                                Evaluating transcript backlog...
                            </div>
                        ) : optimizationBacklogError ? (
                            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-3 text-sm text-rose-700">
                                {optimizationBacklogError}
                            </div>
                        ) : filteredOptimizationBacklog.length === 0 ? (
                            <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
                                No candidates in this filter.
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {filteredOptimizationBacklog.slice(0, 40).map((item) => {
                                    const tier = String(item.recommended_tier || 'none');
                                    const isQueueing = queueingOptimizationVideoId === item.video_id;
                                    return (
                                        <div key={item.video_id} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
                                            <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                                                <div className="min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        <Link to={`/video/${item.video_id}`} className="text-sm font-semibold text-slate-800 hover:text-blue-600 hover:underline">
                                                            {item.title}
                                                        </Link>
                                                        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold ${optimizationTierBadgeClass(tier)}`}>
                                                            {formatOptimizationTierLabel(tier)}
                                                        </span>
                                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-600">
                                                            Score {Number(item.quality_score || 0).toFixed(1)}
                                                        </span>
                                                    </div>
                                                    <div className="mt-2 grid gap-2 text-xs text-slate-600 sm:grid-cols-4">
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2">Unknown rate: {Number(item.metrics?.unknown_speaker_rate || 0).toFixed(2)}</div>
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2">Micro segments: {Number(item.metrics?.micro_segment_count || 0)}</div>
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2">Interruptions: {Number(item.metrics?.same_speaker_interruptions || 0)}</div>
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2">Language: {String(item.language || item.metrics?.language || 'unknown')}</div>
                                                    </div>
                                                    {item.reasons.length > 0 && (
                                                        <div className="mt-2 text-xs leading-5 text-slate-500">
                                                            {item.reasons[0]}
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    <Link
                                                        to={`/video/${item.video_id}`}
                                                        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
                                                    >
                                                        Open Episode
                                                    </Link>
                                                    {tier !== 'manual_review' && tier !== 'none' && (
                                                        <button
                                                            type="button"
                                                            onClick={() => void queueOptimizationForCandidate(item)}
                                                            disabled={isQueueing}
                                                            className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-white disabled:opacity-50 ${
                                                                tier === 'low_risk_repair'
                                                                    ? 'bg-emerald-600 hover:bg-emerald-700'
                                                                    : tier === 'diarization_rebuild'
                                                                        ? 'bg-blue-600 hover:bg-blue-700'
                                                                        : 'bg-amber-600 hover:bg-amber-700'
                                                            }`}
                                                        >
                                                            {isQueueing ? <RefreshCw size={14} className="animate-spin" /> : tier === 'low_risk_repair' ? <GitMerge size={14} /> : tier === 'diarization_rebuild' ? <AudioLines size={14} /> : <RotateCcw size={14} />}
                                                            Queue {formatOptimizationTierLabel(tier)}
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>

                {/* Results area */}
                {!hasSearched ? (
                    <div className="glass-panel rounded-xl p-12 text-center">
                        <Search size={48} className="mx-auto text-slate-300 mb-4" />
                        <h3 className="text-lg font-semibold text-slate-700 mb-2">Search Transcripts</h3>
                        <p className="text-slate-500">
                            {searchMode === 'exact'
                                ? 'Enter a keyword to search across all transcribed episodes in this channel.'
                                : searchMode === 'semantic'
                                    ? 'Describe an idea or concept to find relevant passages even when the wording differs.'
                                    : 'Enter a query to combine keyword matching with semantic relevance.'}
                        </p>
                    </div>
                ) : searching ? (
                    <div className="flex items-center justify-center h-32 text-slate-400 animate-pulse">
                        {searchMode === 'exact' ? 'Searching…' : 'Thinking…'}
                    </div>
                ) : activeResults.length === 0 ? (
                    <div className="glass-panel rounded-xl p-8 text-center">
                        <FileText size={32} className="mx-auto text-slate-300 mb-3" />
                        <p className="text-slate-500">No results found for "{query}"</p>
                        {searchMode !== 'exact' && (
                            <p className="text-xs text-slate-400 mt-2">Make sure the semantic index has been built for this channel.</p>
                        )}
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="text-xs sm:text-sm text-slate-500 flex flex-wrap items-center gap-2">
                                <p className="px-2 py-1 rounded-full bg-white/80 border border-slate-200">
                                    Showing {pageStart}–{pageEnd} of {totalResults} results
                                </p>
                                <p className="px-2 py-1 rounded-full bg-white/80 border border-slate-200">
                                    Page {currentPage} of {totalPages}
                                </p>
                                {searchMode !== 'exact' && (
                                    <span className="px-2 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-xs font-medium capitalize">
                                        {searchMode}
                                    </span>
                                )}
                            </div>
                            {showPagination && (
                                <div className="flex items-center gap-2">
                                    <button type="button"
                                        onClick={() => void runSearch(Math.max(0, resultOffset - resultLimit))}
                                        disabled={searching || resultOffset <= 0}
                                        className="px-3 py-1.5 text-xs sm:text-sm rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50">
                                        Previous
                                    </button>
                                    <button type="button"
                                        onClick={() => void runSearch(resultOffset + resultLimit)}
                                        disabled={searching || !hasMoreResults}
                                        className="px-3 py-1.5 text-xs sm:text-sm rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50">
                                        Next
                                    </button>
                                </div>
                            )}
                        </div>

                        <div ref={resultsListRef}
                            onScroll={(e) => persistSearchState(e.currentTarget.scrollTop)}
                            className="glass-panel rounded-xl divide-y divide-slate-100 max-h-[70vh] overflow-y-auto">

                            {searchMode === 'exact'
                                ? results.map((result) => {
                                    const target = normaliseExact(result);
                                    const resultVideo = videoMap.get(result.video_id);
                                    const isActive = selectedPreview?.id === result.id;
                                    return (
                                        <div key={result.id} role="button" tabIndex={0}
                                            onClick={() => handleResultClick(target)}
                                            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleResultClick(target); } }}
                                            className={`w-full text-left p-3 sm:p-4 transition-colors cursor-pointer ${isActive ? 'bg-blue-50/70' : 'hover:bg-slate-50/50'}`}>
                                            <div className="flex items-start gap-3">
                                                <div className="flex-shrink-0 text-xs text-slate-400 font-mono pt-0.5">
                                                    <span className="flex items-center gap-1"><Clock size={12} />{formatTime(result.start_time)}</span>
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2 mb-1">
                                                        {result.speaker && (
                                                            <button type="button"
                                                                onClick={(e) => void handleSpeakerClick(target, e)}
                                                                className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline underline-offset-2">
                                                                <User size={10} />{result.speaker}
                                                            </button>
                                                        )}
                                                        {resultVideo && (
                                                            <span className="inline-flex items-center gap-1 text-xs text-slate-500 min-w-0">
                                                                <Tv size={10} />
                                                                <span className="truncate max-w-[420px]">{resultVideo.title}</span>
                                                                <span className="text-slate-400">• {formatPublished(resultVideo.published_at)}</span>
                                                                <span className="text-slate-400 font-mono">• {resultVideo.youtube_id}</span>
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="text-slate-700 text-sm leading-relaxed">{result.text}</p>
                                                </div>
                                                <PlayCircle size={16} className={`mt-0.5 flex-shrink-0 ${isActive ? 'text-blue-600' : 'text-slate-300'}`} />
                                            </div>
                                        </div>
                                    );
                                })
                                : semanticResults.map((hit) => {
                                    const target = normaliseSemantic(hit);
                                    const isActive = selectedPreview?.id === hit.id;
                                    const hasCtx = hit.context_before.length > 0 || hit.context_after.length > 0;
                                    const ctxOpen = expandedContext.has(hit.id);
                                    const resultVideo = videoMap.get(hit.video_id);
                                    return (
                                        <div key={hit.id} className={`p-3 sm:p-4 transition-colors ${isActive ? 'bg-indigo-50/70' : 'hover:bg-slate-50/50'}`}>
                                            <div className="flex items-start gap-3 cursor-pointer"
                                                role="button" tabIndex={0}
                                                onClick={() => handleResultClick(target)}
                                                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleResultClick(target); } }}>
                                                <div className="flex-shrink-0 text-xs text-slate-400 font-mono pt-0.5">
                                                    <span className="flex items-center gap-1"><Clock size={12} />{formatTime(hit.start_time)}</span>
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2 mb-1">
                                                        <ScoreBadge score={hit.score} />
                                                        {hit.speaker_name && (
                                                            <button type="button"
                                                                onClick={(e) => void handleSpeakerClick(target, e)}
                                                                className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline underline-offset-2">
                                                                <User size={10} />{hit.speaker_name}
                                                            </button>
                                                        )}
                                                        {(hit.video_title || resultVideo) && (
                                                            <span className="inline-flex items-center gap-1 text-xs text-slate-500 min-w-0">
                                                                <Tv size={10} />
                                                                <span className="truncate max-w-[380px]">{hit.video_title ?? resultVideo?.title}</span>
                                                                {resultVideo && <span className="text-slate-400">• {formatPublished(resultVideo.published_at)}</span>}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="text-slate-700 text-sm leading-relaxed">{hit.chunk_text}</p>
                                                </div>
                                                <PlayCircle size={16} className={`mt-0.5 flex-shrink-0 ${isActive ? 'text-indigo-600' : 'text-slate-300'}`} />
                                            </div>

                                            {/* Context toggle */}
                                            {hasCtx && (
                                                <div className="mt-2 pl-7">
                                                    <button type="button"
                                                        onClick={() => toggleContext(hit.id)}
                                                        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-600">
                                                        {ctxOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                                                        {ctxOpen ? 'Hide context' : 'Show context'}
                                                    </button>
                                                    {ctxOpen && (
                                                        <div className="mt-2 space-y-1.5 border-l-2 border-slate-100 pl-3">
                                                            {hit.context_before.map((c) => (
                                                                <div key={c.chunk_id} className="text-xs text-slate-500 leading-relaxed">
                                                                    <span className="font-mono text-slate-300 mr-1">{formatTime(c.start_time)}</span>
                                                                    {c.chunk_text}
                                                                </div>
                                                            ))}
                                                            <div className="text-xs font-medium text-indigo-500 pl-0.5">↑ match ↓</div>
                                                            {hit.context_after.map((c) => (
                                                                <div key={c.chunk_id} className="text-xs text-slate-500 leading-relaxed">
                                                                    <span className="font-mono text-slate-300 mr-1">{formatTime(c.start_time)}</span>
                                                                    {c.chunk_text}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })
                            }
                        </div>
                    </div>
                )}
            </div>

            {/* Sidebar */}
            <aside className="glass-panel rounded-xl p-4 xl:p-5 xl:sticky xl:top-24 self-start">
                <h3 className="text-sm font-semibold text-slate-700 mb-3">Preview</h3>
                {loadingVideos ? (
                    <div className="text-sm text-slate-400 py-8 text-center">Loading videos…</div>
                ) : !selectedPreview ? (
                    <div className="py-8 text-center text-slate-500">
                        <PlayCircle size={34} className="mx-auto text-slate-300 mb-3" />
                        <p className="text-sm">Click a search result to preview that clip.</p>
                    </div>
                ) : !previewVideo ? (
                    <div className="py-8 text-center text-slate-500">
                        <FileText size={32} className="mx-auto text-slate-300 mb-3" />
                        <p className="text-sm">Could not find the matching video for this result.</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="rounded-lg overflow-hidden bg-black">
                            {previewMediaPending ? (
                                <div className="flex h-[340px] items-center justify-center px-6 text-center text-sm text-slate-300">
                                    Local media is not ready yet for this episode.
                                </div>
                            ) : previewUsesYoutube ? (
                                <YouTube key={previewVideo.id} videoId={previewVideo.youtube_id}
                                    onReady={(event) => setPreviewPlayer(event.target as PlayerController)}
                                    opts={{ width: '100%', height: '340', playerVars: { rel: 0, modestbranding: 1 } }}
                                    className="w-full" />
                            ) : previewIsAudioOnly ? (
                                <div className="flex h-[340px] items-center justify-center px-6">
                                    <audio key={previewVideo.id} ref={nativePreviewRef} controls className="w-full" src={previewMediaUrl}
                                        onLoadedMetadata={(e) => setPreviewPlayer(buildNativePreviewPlayer(e.currentTarget))} />
                                </div>
                            ) : (
                                <video key={previewVideo.id} ref={nativePreviewRef as React.RefObject<HTMLVideoElement>} controls playsInline
                                    className="h-[340px] w-full bg-black object-contain" src={previewMediaUrl}
                                    onLoadedMetadata={(e) => setPreviewPlayer(buildNativePreviewPlayer(e.currentTarget))} />
                            )}
                        </div>
                        <div className="space-y-2">
                            <p className="text-base font-semibold text-slate-700 leading-snug">{previewVideo.title}</p>
                            <div className="text-xs text-slate-500 flex flex-wrap items-center gap-x-3 gap-y-1">
                                <span className="inline-flex items-center gap-1">
                                    <Clock size={12} />
                                    {formatTime(selectedPreview.start_time)} – {formatTime(selectedPreview.end_time)}
                                </span>
                                {selectedPreview.speaker && (
                                    <span className="inline-flex items-center gap-1"><User size={12} />{selectedPreview.speaker}</span>
                                )}
                                {selectedPreview.score !== undefined && (
                                    <ScoreBadge score={selectedPreview.score} />
                                )}
                            </div>
                            <p className="text-sm text-slate-600 leading-relaxed">{selectedPreview.text}</p>
                            <div className="flex flex-wrap items-center gap-2 pt-1">
                                <button type="button" onClick={() => playPreviewClip(selectedPreview)}
                                    className="text-xs text-blue-600 hover:text-blue-700 font-medium">
                                    Replay clip
                                </button>
                                <Link to={buildPreviewEpisodeLink(selectedPreview)}
                                    className="inline-flex items-center rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50">
                                    Open Episode At This Point
                                </Link>
                            </div>
                        </div>
                    </div>
                )}

                <IndexStatusPanel />
            </aside>

            {selectedSpeaker && (
                <SpeakerModal
                    speaker={selectedSpeaker}
                    initialSample={initialSpeakerSample || undefined}
                    onClose={() => { setSelectedSpeaker(null); setInitialSpeakerSample(null); }}
                    onUpdate={(updated) => {
                        setSelectedSpeaker(updated);
                        setResults((prev) => prev.map((r) => r.speaker_id === updated.id ? { ...r, speaker: updated.name } : r));
                        setSemanticResults((prev) => prev.map((r) => r.speaker_id === updated.id ? { ...r, speaker_name: updated.name } : r));
                    }}
                    onMerge={(merged) => {
                        if (!merged || !selectedSpeaker) return;
                        const src = selectedSpeaker.id;
                        setResults((prev) => prev.map((r) => r.speaker_id === src ? { ...r, speaker_id: merged.id, speaker: merged.name } : r));
                        setSemanticResults((prev) => prev.map((r) => r.speaker_id === src ? { ...r, speaker_id: merged.id, speaker_name: merged.name } : r));
                    }}
                />
            )}
        </div>
    );
}
