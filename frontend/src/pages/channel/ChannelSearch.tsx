import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type { Speaker, Video } from '../../types';
import { SpeakerModal } from '../../components/SpeakerModal';
import { Search, Clock, User, FileText, PlayCircle, Tv } from 'lucide-react';

const PREVIEW_STOP_EARLY_SECONDS = 0.35;

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

export function ChannelSearch() {
    const { id } = useParams<{ id: string }>();
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [totalResults, setTotalResults] = useState(0);
    const [resultOffset, setResultOffset] = useState(0);
    const [resultLimit, setResultLimit] = useState(50);
    const [hasMoreResults, setHasMoreResults] = useState(false);
    const [searching, setSearching] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [yearFilter, setYearFilter] = useState('');
    const [monthFilter, setMonthFilter] = useState('');
    const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');
    const [videos, setVideos] = useState<Video[]>([]);
    const [loadingVideos, setLoadingVideos] = useState(true);
    const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
    const [previewPlayer, setPreviewPlayer] = useState<any>(null);
    const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
    const [initialSpeakerSample, setInitialSpeakerSample] = useState<any>(null);
    const [searchStateHydrated, setSearchStateHydrated] = useState(false);
    const resultsListRef = useRef<HTMLDivElement>(null);
    const savedScrollTopRef = useRef<number | null>(null);
    const nativePreviewRef = useRef<HTMLMediaElement | null>(null);

    const getPersistKey = () => (id ? `chatalogue:channel-search:${id}` : null);

    const persistSearchState = (scrollTopOverride?: number) => {
        const key = getPersistKey();
        if (!key) return;
        try {
            const payload = {
                query,
                results,
                totalResults,
                resultOffset,
                resultLimit,
                hasMoreResults,
                hasSearched,
                yearFilter,
                monthFilter,
                sortOrder,
                selectedResultId: selectedResult?.id ?? null,
                scrollTop: scrollTopOverride ?? resultsListRef.current?.scrollTop ?? 0,
            };
            sessionStorage.setItem(key, JSON.stringify(payload));
        } catch (e) {
            console.warn('Failed to persist channel search state', e);
        }
    };

    useEffect(() => {
        const key = getPersistKey();
        if (!key) return;
        try {
            const raw = sessionStorage.getItem(key);
            if (!raw) return;
            const saved = JSON.parse(raw);
            const savedResults: SearchResult[] = Array.isArray(saved?.results) ? saved.results : [];
            setQuery(typeof saved?.query === 'string' ? saved.query : '');
            setResults(savedResults);
            setTotalResults(typeof saved?.totalResults === 'number' ? saved.totalResults : 0);
            setResultOffset(typeof saved?.resultOffset === 'number' ? saved.resultOffset : 0);
            setResultLimit(typeof saved?.resultLimit === 'number' ? saved.resultLimit : 50);
            setHasMoreResults(Boolean(saved?.hasMoreResults));
            setHasSearched(Boolean(saved?.hasSearched));
            setYearFilter(typeof saved?.yearFilter === 'string' ? saved.yearFilter : '');
            setMonthFilter(typeof saved?.monthFilter === 'string' ? saved.monthFilter : '');
            setSortOrder(saved?.sortOrder === 'oldest' ? 'oldest' : 'newest');
            const selectedId = typeof saved?.selectedResultId === 'number' ? saved.selectedResultId : null;
            setSelectedResult(selectedId ? (savedResults.find((r) => r.id === selectedId) ?? null) : null);
            savedScrollTopRef.current = typeof saved?.scrollTop === 'number' ? saved.scrollTop : null;
        } catch (e) {
            console.warn('Failed to restore channel search state', e);
        } finally {
            setSearchStateHydrated(true);
        }
        // Restore only when changing channels.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [id]);

    useEffect(() => {
        if (!hasSearched || !resultsListRef.current || savedScrollTopRef.current == null) return;
        const top = savedScrollTopRef.current;
        window.requestAnimationFrame(() => {
            if (resultsListRef.current) {
                resultsListRef.current.scrollTop = top;
            }
        });
        savedScrollTopRef.current = null;
    }, [hasSearched, results.length]);

    useEffect(() => {
        if (!searchStateHydrated) return;
        persistSearchState();
        // Persist whenever the visible search state changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchStateHydrated, id, query, hasSearched, results, totalResults, resultOffset, resultLimit, hasMoreResults, yearFilter, monthFilter, sortOrder, selectedResult]);

    useEffect(() => {
        let cancelled = false;

        const fetchVideos = async () => {
            if (!id) return;
            setLoadingVideos(true);
            setPreviewPlayer(null);

            try {
                const res = await api.get<Video[]>('/videos/list', {
                    params: { channel_id: id }
                });
                if (!cancelled) {
                    setVideos(res.data);
                }
            } catch (e) {
                console.error('Failed to fetch channel videos:', e);
                if (!cancelled) {
                    setVideos([]);
                }
            } finally {
                if (!cancelled) {
                    setLoadingVideos(false);
                }
            }
        };

        void fetchVideos();

        return () => {
            cancelled = true;
        };
    }, [id]);

    const playPreviewClip = (result: SearchResult) => {
        if (!previewPlayer) return;
        try {
            previewPlayer.seekTo(result.start_time, true);
            previewPlayer.playVideo?.();
        } catch (e) {
            console.warn('Failed to control preview player:', e);
        }
    };

    const pausePreview = () => {
        try {
            previewPlayer?.pauseVideo?.();
        } catch {
            // Ignore player state errors while iframe reloads.
        }
    };

    const buildNativePreviewPlayer = (media: HTMLMediaElement) => ({
        seekTo: (seconds: number) => {
            media.currentTime = Math.max(0, seconds || 0);
        },
        playVideo: async () => {
            try {
                await media.play();
            } catch {
                // Ignore autoplay restrictions in preview pane.
            }
        },
        pauseVideo: () => media.pause(),
        getCurrentTime: () => media.currentTime || 0,
    });

    const runSearch = async (nextOffset: number = 0) => {
        if (!query.trim()) return;
        setSearching(true);
        setHasSearched(true);
        try {
            const res = await api.get<SearchResponse>('/search', {
                params: {
                    q: query,
                    channel_id: id,
                    offset: nextOffset,
                    limit: resultLimit,
                    ...(yearFilter ? { year: Number(yearFilter) } : {}),
                    ...(monthFilter ? { month: Number(monthFilter) } : {}),
                    sort: sortOrder,
                }
            });
            setResults(Array.isArray(res.data.items) ? res.data.items : []);
            setTotalResults(res.data.total ?? 0);
            setResultOffset(res.data.offset ?? nextOffset);
            setHasMoreResults(Boolean(res.data.has_more));
            // Keep preview idle until the user explicitly clicks a result.
            setSelectedResult(null);
            if (resultsListRef.current) {
                resultsListRef.current.scrollTop = 0;
            }
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

    useEffect(() => {
        if (!selectedResult || !previewPlayer) return;
        playPreviewClip(selectedResult);
    }, [selectedResult, previewPlayer]);

    useEffect(() => {
        if (!previewPlayer || !selectedResult) return;

        const interval = window.setInterval(() => {
            try {
                const current = previewPlayer.getCurrentTime?.();
                const stopAt = Math.max(
                    selectedResult.start_time,
                    selectedResult.end_time - PREVIEW_STOP_EARLY_SECONDS
                );
                if (typeof current === 'number' && current >= stopAt) {
                    previewPlayer.pauseVideo?.();
                }
            } catch {
                // Ignore transient errors while the iframe/player is swapping.
            }
        }, 200);

        return () => window.clearInterval(interval);
    }, [previewPlayer, selectedResult]);

    const handleResultClick = (result: SearchResult) => {
        setSelectedResult(result);
        if (selectedResult?.id === result.id) {
            playPreviewClip(result);
        }
    };

    const handleSpeakerClick = async (result: SearchResult, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!result.speaker_id) return;

        pausePreview();

        const resultVideo = videoMap.get(result.video_id);
        setInitialSpeakerSample({
            youtube_id: resultVideo?.youtube_id,
            video_id: result.video_id,
            start_time: result.start_time,
            end_time: result.end_time,
            text: result.text,
            media_source_type: resultVideo?.media_source_type,
            media_kind: resultVideo?.media_kind,
        });

        try {
            const res = await api.get<Speaker>(`/speakers/${result.speaker_id}`);
            setSelectedSpeaker(res.data);
        } catch (err) {
            console.error('Failed to fetch speaker details', err);
        }
    };

    const formatTime = (seconds: number) => {
        const total = Math.max(0, Math.floor(seconds));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        return h > 0
            ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
            : `${m}:${s.toString().padStart(2, '0')}`;
    };
    const formatPublished = (iso?: string | null) => {
        if (!iso) return 'Unknown date';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return 'Unknown date';
        return d.toLocaleDateString();
    };

    const videoMap = new Map(videos.map((video) => [video.id, video]));
    const previewVideo = selectedResult ? videoMap.get(selectedResult.video_id) : null;
    const previewMediaSourceType = String(previewVideo?.media_source_type || 'youtube').toLowerCase();
    const previewUsesYoutube = previewMediaSourceType === 'youtube';
    const previewUsesLocalMedia = previewMediaSourceType === 'upload' || previewMediaSourceType === 'tiktok';
    const previewIsAudioOnly = previewUsesLocalMedia && String(previewVideo?.media_kind || '').toLowerCase() === 'audio';
    const previewMediaUrl = previewVideo ? toApiUrl(`/videos/${previewVideo.id}/media`) : '';
    const previewMediaPending = previewUsesLocalMedia && ['pending', 'queued'].includes(String(previewVideo?.status || '').toLowerCase());
    const totalPages = Math.max(1, Math.ceil(totalResults / Math.max(resultLimit, 1)));
    const currentPage = Math.floor(resultOffset / Math.max(resultLimit, 1)) + 1;
    const pageStart = totalResults === 0 ? 0 : resultOffset + 1;
    const pageEnd = Math.min(resultOffset + results.length, totalResults);
    const showPagination = totalResults > resultLimit;

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const publishedVideoDates = videos
        .map((v) => ({ v, d: v.published_at ? new Date(v.published_at) : null }))
        .filter((x): x is { v: Video; d: Date } => !!x.d && !Number.isNaN(x.d.getTime()));
    const yearOptions = Array.from(new Set(publishedVideoDates.map(({ d }) => d.getFullYear())))
        .sort((a, b) => b - a);
    const monthOptions = Array.from(
        new Set(
            publishedVideoDates
                .filter(({ d }) => !yearFilter || d.getFullYear() === Number(yearFilter))
                .map(({ d }) => d.getMonth() + 1)
        )
    ).sort((a, b) => a - b);

    const handleYearChange = (value: string) => {
        setYearFilter(value);
        setMonthFilter('');
    };

        return (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(460px,560px)]">
            <div className="space-y-6 min-w-0">
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
                            {searching ? 'Searching...' : 'Search'}
                        </button>
                    </div>
                    <div className="mt-2.5 grid grid-cols-2 md:grid-cols-4 gap-2">
                        <select
                            value={yearFilter}
                            onChange={(e) => handleYearChange(e.target.value)}
                            className="w-full min-h-9 px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                        >
                            <option value="">All years</option>
                            {yearOptions.map((year) => (
                                <option key={year} value={year}>{year}</option>
                            ))}
                        </select>
                        <select
                            value={monthFilter}
                            onChange={(e) => setMonthFilter(e.target.value)}
                            disabled={!yearFilter}
                            className="w-full min-h-9 px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50"
                        >
                            <option value="">All months</option>
                            {monthOptions.map((month) => (
                                <option key={month} value={month}>{monthNames[month - 1]}</option>
                            ))}
                        </select>
                        <div className="space-y-1">
                            <label className="text-[11px] text-slate-500">Per page</label>
                            <select
                                value={resultLimit}
                                onChange={(e) => setResultLimit(Number(e.target.value))}
                                className="w-full min-h-9 px-2 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                            >
                                {[25, 50, 100].map((n) => (
                                    <option key={n} value={n}>{n}</option>
                                ))}
                            </select>
                        </div>
                        <div className="space-y-1 col-span-2 md:col-span-1">
                            <label className="text-[11px] text-slate-500">Sort</label>
                            <select
                                value={sortOrder}
                                onChange={(e) => setSortOrder(e.target.value === 'oldest' ? 'oldest' : 'newest')}
                                className="w-full min-h-9 px-2 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                            >
                                <option value="newest">Newest first</option>
                                <option value="oldest">Chronological</option>
                            </select>
                        </div>
                    </div>
                </form>

                {!hasSearched ? (
                    <div className="glass-panel rounded-xl p-12 text-center">
                        <Search size={48} className="mx-auto text-slate-300 mb-4" />
                        <h3 className="text-lg font-semibold text-slate-700 mb-2">Search Transcripts</h3>
                        <p className="text-slate-500">Enter a keyword to search across all transcribed episodes in this channel.</p>
                    </div>
                ) : searching ? (
                    <div className="flex items-center justify-center h-32 text-slate-400 animate-pulse">Searching...</div>
                ) : results.length === 0 ? (
                    <div className="glass-panel rounded-xl p-8 text-center">
                        <FileText size={32} className="mx-auto text-slate-300 mb-3" />
                        <p className="text-slate-500">No results found for "{query}"</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="text-xs sm:text-sm text-slate-500 flex flex-wrap items-center gap-2">
                                <p className="px-2 py-1 rounded-full bg-white/80 border border-slate-200">
                                    Showing {pageStart}-{pageEnd} of {totalResults} results
                                </p>
                                <p className="px-2 py-1 rounded-full bg-white/80 border border-slate-200">
                                    Page {currentPage} of {totalPages}
                                </p>
                            </div>
                            {showPagination && (
                                <div className="flex items-center gap-2">
                                    <button
                                        type="button"
                                        onClick={() => void runSearch(Math.max(0, resultOffset - resultLimit))}
                                        disabled={searching || resultOffset <= 0}
                                        className="px-3 py-1.5 text-xs sm:text-sm rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                    >
                                        Previous
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => void runSearch(resultOffset + resultLimit)}
                                        disabled={searching || !hasMoreResults}
                                        className="px-3 py-1.5 text-xs sm:text-sm rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                    >
                                        Next
                                    </button>
                                </div>
                            )}
                        </div>
                        <div
                            ref={resultsListRef}
                            onScroll={(e) => persistSearchState(e.currentTarget.scrollTop)}
                            className="glass-panel rounded-xl divide-y divide-slate-100 max-h-[70vh] overflow-y-auto"
                        >
                            {results.map((result) => {
                                const resultVideo = videoMap.get(result.video_id);
                                const isActive = selectedResult?.id === result.id;

                                return (
                                    <div
                                        key={result.id}
                                        role="button"
                                        tabIndex={0}
                                        onClick={() => handleResultClick(result)}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter' || e.key === ' ') {
                                                e.preventDefault();
                                                handleResultClick(result);
                                            }
                                        }}
                                        className={`w-full text-left p-3 sm:p-4 transition-colors cursor-pointer ${isActive ? 'bg-blue-50/70' : 'hover:bg-slate-50/50'}`}
                                    >
                                        <div className="flex items-start gap-3">
                                            <div className="flex-shrink-0 text-xs text-slate-400 font-mono pt-0.5">
                                                <span className="flex items-center gap-1">
                                                    <Clock size={12} />
                                                    {formatTime(result.start_time)}
                                                </span>
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex flex-wrap items-center gap-2 mb-1">
                                                    {result.speaker && (
                                                        <button
                                                            type="button"
                                                            onClick={(e) => void handleSpeakerClick(result, e)}
                                                            className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline underline-offset-2"
                                                            title={result.speaker_id ? 'Open speaker identification' : undefined}
                                                        >
                                                            <User size={10} />
                                                            {result.speaker}
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
                                            <PlayCircle size={16} className={`mt-0.5 ${isActive ? 'text-blue-600' : 'text-slate-300'}`} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            <aside className="glass-panel rounded-xl p-4 xl:p-5 xl:sticky xl:top-24 self-start">
                <h3 className="text-sm font-semibold text-slate-700 mb-3">Preview</h3>
                {loadingVideos ? (
                    <div className="text-sm text-slate-400 py-8 text-center">Loading videos...</div>
                ) : !selectedResult ? (
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
                                <YouTube
                                    key={previewVideo.id}
                                    videoId={previewVideo.youtube_id}
                                    onReady={(event: any) => setPreviewPlayer(event.target)}
                                    opts={{
                                        width: '100%',
                                        height: '340',
                                        playerVars: {
                                            rel: 0,
                                            modestbranding: 1,
                                        },
                                    }}
                                    className="w-full"
                                />
                            ) : previewIsAudioOnly ? (
                                <div className="flex h-[340px] items-center justify-center px-6">
                                    <audio
                                        key={previewVideo.id}
                                        ref={nativePreviewRef}
                                        controls
                                        className="w-full"
                                        src={previewMediaUrl}
                                        onLoadedMetadata={(event) => {
                                            const controller = buildNativePreviewPlayer(event.currentTarget);
                                            setPreviewPlayer(controller);
                                        }}
                                    />
                                </div>
                            ) : (
                                <video
                                    key={previewVideo.id}
                                    ref={nativePreviewRef}
                                    controls
                                    playsInline
                                    className="h-[340px] w-full bg-black object-contain"
                                    src={previewMediaUrl}
                                    onLoadedMetadata={(event) => {
                                        const controller = buildNativePreviewPlayer(event.currentTarget);
                                        setPreviewPlayer(controller);
                                    }}
                                />
                            )}
                        </div>

                        <div className="space-y-2">
                            <p className="text-base font-semibold text-slate-700 leading-snug">{previewVideo.title}</p>
                            <div className="text-xs text-slate-500 flex flex-wrap items-center gap-x-3 gap-y-1">
                                <span className="inline-flex items-center gap-1">
                                    <Clock size={12} />
                                    {formatTime(selectedResult.start_time)} - {formatTime(selectedResult.end_time)}
                                </span>
                                {selectedResult.speaker && (
                                    <span className="inline-flex items-center gap-1">
                                        <User size={12} />
                                        {selectedResult.speaker}
                                    </span>
                                )}
                            </div>
                            <p className="text-sm text-slate-600 leading-relaxed">{selectedResult.text}</p>
                            <div className="flex flex-wrap items-center gap-2 pt-1">
                                <button
                                    type="button"
                                    onClick={() => playPreviewClip(selectedResult)}
                                    className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                                >
                                    Replay clip
                                </button>
                                <Link
                                    to={`/video/${previewVideo.id}?t=${Math.floor(selectedResult.start_time)}`}
                                    className="inline-flex items-center rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                                >
                                    Open Episode At This Point
                                </Link>
                            </div>
                        </div>
                    </div>
                )}
            </aside>

            {selectedSpeaker && (
                <SpeakerModal
                    speaker={selectedSpeaker}
                    initialSample={initialSpeakerSample}
                    onClose={() => {
                        setSelectedSpeaker(null);
                        setInitialSpeakerSample(null);
                    }}
                    onUpdate={(updatedSpeaker) => {
                        setSelectedSpeaker(updatedSpeaker);
                        setResults(prev => prev.map(r =>
                            r.speaker_id === updatedSpeaker.id ? { ...r, speaker: updatedSpeaker.name } : r
                        ));
                    }}
                    onMerge={(mergedIntoSpeaker) => {
                        if (!mergedIntoSpeaker || !selectedSpeaker) return;
                        const sourceSpeakerId = selectedSpeaker.id;
                        setResults(prev => prev.map(r =>
                            r.speaker_id === sourceSpeakerId
                                ? { ...r, speaker_id: mergedIntoSpeaker.id, speaker: mergedIntoSpeaker.name }
                                : r
                        ));
                    }}
                />
            )}
        </div>
    );
}
