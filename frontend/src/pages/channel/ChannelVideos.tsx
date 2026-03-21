import { useEffect, useState, useMemo, useRef, type MouseEvent } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import api from '../../lib/api';
import type { Channel, Video } from '../../types';
import { Play, Loader2, Eye, EyeOff, RefreshCw, Download, Grid, List, Search, X, ArrowUpDown, ArrowUp, ArrowDown, Plus, Filter, Upload } from 'lucide-react';
import { VideoStatusBadge } from '../../components/VideoStatusBadge';

type SortField = 'published_at' | 'title' | 'duration' | 'status';
type SortOrder = 'asc' | 'desc';
const VIDEO_PAGE_SIZE = 250;
const DATE_BACKFILL_POLL_MS = 5000;
const DATE_BACKFILL_MAX_POLLS = 24;

export function ChannelVideos() {
    const { id } = useParams<{ id: string }>();
    const [searchParams] = useSearchParams();
    const [channel, setChannel] = useState<Channel | null>(null);
    const [videos, setVideos] = useState<Video[]>([]);
    const [loading, setLoading] = useState(true);
    const [missingDateCount, setMissingDateCount] = useState(0);
    const [processingIds, setProcessingIds] = useState<Set<number>>(new Set());
    const [processingAll, setProcessingAll] = useState(false);
    const [fetching, setFetching] = useState(false);
    const [viewMode, setViewMode] = useState<'compact' | 'grid'>('compact');
    const [searchQuery, setSearchQuery] = useState('');
    const [mutingFiltered, setMutingFiltered] = useState(false);
    const [addingVideo, setAddingVideo] = useState(false);
    const [uploadingMedia, setUploadingMedia] = useState(false);
    const navigate = useNavigate();
    const uploadInputRef = useRef<HTMLInputElement>(null);

    // Status filter state
    const [statusFilters, setStatusFilters] = useState<Set<string>>(new Set());

    const toggleStatusFilter = (status: string) => {
        setStatusFilters(prev => {
            const next = new Set(prev);
            if (next.has(status)) next.delete(status);
            else next.add(status);
            return next;
        });
    };

    // Sorting state
    const [sortField, setSortField] = useState<SortField>('published_at');
    const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
    const [visibleCount, setVisibleCount] = useState(VIDEO_PAGE_SIZE);
    const backfillPollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const backfillPollCountRef = useRef(0);

    const clearBackfillPollTimer = () => {
        if (backfillPollTimerRef.current) {
            clearTimeout(backfillPollTimerRef.current);
            backfillPollTimerRef.current = null;
        }
    };

    const fetchChannel = async () => {
        if (!id) return;
        try {
            const res = await api.get<Channel>(`/channels/${id}`);
            setChannel(res.data);
        } catch (e) {
            console.error('Failed to fetch channel:', e);
        }
    };

    const getPlaceholderLabel = (video: Video) => {
        const source = String(video.transcript_source || '').toLowerCase();
        if (source === 'youtube_auto_captions' || source === 'tiktok_auto_captions') return 'Auto-captions';
        if (source === 'youtube_subtitles') return 'YT captions';
        if (source === 'tiktok_subtitles') return 'TikTok captions';
        return 'Placeholder captions';
    };

    const getAccessRestrictionLabel = (video: Video) => {
        const reason = String(video.access_restriction_reason || '').toLowerCase();
        if (reason.includes('members-only') || reason.includes('members only')) return 'Members only';
        if (reason.includes('private')) return 'Private';
        if (reason.includes('sign in') || reason.includes('auth')) return 'Sign-in required';
        return 'Access restricted';
    };
    const getManualMediaLabel = (video: Video) => (
        String(video.media_kind || '').toLowerCase() === 'audio' ? 'Uploaded audio' : 'Uploaded video'
    );
    const getSourceMediaLabel = (video: Video) => {
        const mediaSourceType = String(video.media_source_type || 'youtube').toLowerCase();
        if (mediaSourceType === 'upload') return getManualMediaLabel(video);
        if (mediaSourceType === 'tiktok') return 'TikTok';
        return null;
    };

    const isProcessableVideo = (video: Video) => !video.processed && !video.muted && !video.access_restricted;
    const channelSourceType = (channel?.source_type || 'youtube').toLowerCase();
    const isManualChannel = channelSourceType === 'manual';
    const isTikTokChannel = channelSourceType === 'tiktok';
    const isYoutubeChannel = channelSourceType === 'youtube';

    const getChannelSyncProgress = () => Math.max(0, Math.min(100, Number(channel?.sync_progress ?? 0)));

    const fetchVideos = async (polling: boolean = false) => {
        try {
            const res = await api.get<Video[]>('/videos/list', { params: { channel_id: id } });
            setVideos(res.data);
            const nextMissingDateCount = res.data.reduce((count, video) => (
                count + (video.published_at ? 0 : 1)
            ), 0);
            setMissingDateCount(nextMissingDateCount);
            if (!polling && nextMissingDateCount === 0) {
                backfillPollCountRef.current = 0;
            }
        } catch (e) {
            console.error('Failed to fetch videos:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        clearBackfillPollTimer();
        backfillPollCountRef.current = 0;
        void fetchChannel();
        void fetchVideos();
        return () => {
            clearBackfillPollTimer();
        };
    }, [id]);

    useEffect(() => {
        clearBackfillPollTimer();
        if (!id) {
            return;
        }

        const shouldPollRefresh = channel?.status === 'refreshing';
        const shouldPollDates = missingDateCount > 0 && backfillPollCountRef.current < DATE_BACKFILL_MAX_POLLS;
        if (!shouldPollRefresh && !shouldPollDates) {
            if (missingDateCount === 0) {
                backfillPollCountRef.current = 0;
            }
            return;
        }

        backfillPollTimerRef.current = setTimeout(() => {
            if (!shouldPollRefresh && shouldPollDates) {
                backfillPollCountRef.current += 1;
            }
            void fetchVideos(true);
            if (shouldPollRefresh || !isManualChannel) {
                void fetchChannel();
            }
        }, DATE_BACKFILL_POLL_MS);

        return () => {
            clearBackfillPollTimer();
        };
    }, [channel?.status, id, missingDateCount]);

    // Handle deep linking to video via URL params
    useEffect(() => {
        const videoIdParam = searchParams.get('videoId');
        if (videoIdParam && videos.length > 0) {
            navigate(`/video/${videoIdParam}`);
        }
    }, [videos, searchParams]);

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortOrder('desc');
        }
    };

    const getSortIcon = (field: SortField) => {
        if (sortField !== field) return <ArrowUpDown size={14} className="text-slate-300" />;
        return sortOrder === 'asc' ? <ArrowUp size={14} className="text-blue-500" /> : <ArrowDown size={14} className="text-blue-500" />;
    };

    // Filter and Sort videos
    const processedVideos = useMemo(() => {
        // 1. Filter by search
        let result = videos;
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            result = result.filter(v =>
                v.title.toLowerCase().includes(query) ||
                (v.description && v.description.toLowerCase().includes(query))
            );
        }

        // 2. Filter by status categories
        if (statusFilters.size > 0) {
            result = result.filter(v => {
                const effective = v.processed ? 'completed' : (v.status || 'pending');
                if (statusFilters.has('completed') && v.processed) return true;
                if (statusFilters.has('failed') && effective === 'failed') return true;
                if (statusFilters.has('unprocessed') && !v.processed && effective !== 'failed' && !v.access_restricted) return true;
                return false;
            });
        }

        // 3. Sort
        return [...result].sort((a, b) => {
            let valA: any = a[sortField];
            let valB: any = b[sortField];

            if (sortField === 'published_at') {
                valA = new Date(a.published_at || 0).getTime();
                valB = new Date(b.published_at || 0).getTime();
            } else if (sortField === 'title') {
                valA = a.title.toLowerCase();
                valB = b.title.toLowerCase();
            } else if (sortField === 'status') {
                // Sort priority: processed > transcribing > downloading > pending > failed > restricted
                const statusOrder: Record<string, number> = { processed: 5, transcribing: 4, downloading: 3, pending: 2, failed: 1, access_restricted: 0 };
                valA = statusOrder[a.access_restricted ? 'access_restricted' : (a.processed ? 'processed' : (a.status || 'pending'))] || 0;
                valB = statusOrder[b.access_restricted ? 'access_restricted' : (b.processed ? 'processed' : (b.status || 'pending'))] || 0;
            }

            if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
            if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
            return 0;
        });
    }, [videos, searchQuery, statusFilters, sortField, sortOrder]);

    const visibleVideos = useMemo(
        () => processedVideos.slice(0, visibleCount),
        [processedVideos, visibleCount]
    );
    const canLoadMore = visibleVideos.length < processedVideos.length;

    useEffect(() => {
        setVisibleCount(VIDEO_PAGE_SIZE);
    }, [searchQuery, statusFilters, sortField, sortOrder, viewMode, videos.length]);

    const handleFetchFromYouTube = async () => {
        setFetching(true);
        try {
            await api.post(`/channels/${id}/refresh`);
            await fetchChannel();
            await fetchVideos();
        } catch (e) {
            console.error('Failed to fetch videos:', e);
            alert('Failed to start video fetch');
        } finally {
            setFetching(false);
        }
    };

    const handleAddVideo = async () => {
        const url = prompt(isTikTokChannel ? 'Paste a TikTok video URL:' : 'Paste a YouTube video URL:');
        if (!url?.trim()) return;
        setAddingVideo(true);
        try {
            await api.post(`/channels/${id}/add-video`, null, { params: { url: url.trim() } });
            await fetchVideos();
        } catch (e: any) {
            console.error('Failed to add video:', e);
            alert(e?.response?.data?.detail || 'Failed to add video');
        } finally {
            setAddingVideo(false);
        }
    };

    const handleUploadMedia = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file || !id) return;
        const suggestedTitle = window.prompt('Episode title:', file.name.replace(/\.[^.]+$/, ''))?.trim() || '';
        const formData = new FormData();
        formData.append('file', file);
        if (suggestedTitle) {
            formData.append('title', suggestedTitle);
        }
        setUploadingMedia(true);
        try {
            await api.post(`/channels/${id}/upload-media`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            await fetchVideos();
        } catch (e: any) {
            console.error('Failed to upload media:', e);
            alert(e?.response?.data?.detail || 'Failed to upload media');
        } finally {
            setUploadingMedia(false);
            if (uploadInputRef.current) uploadInputRef.current.value = '';
        }
    };

    const handleProcess = async (videoId: number) => {
        setProcessingIds(prev => new Set(prev).add(videoId));
        try {
            await api.post(`/videos/${videoId}/process`);
            // Update local state immediately for better UX
            setVideos(prev => prev.map(v =>
                v.id === videoId ? { ...v, status: 'queued', processed: false } : v
            ));
        } catch (e: any) {
            console.error('Failed to start processing:', e);
            alert(e?.response?.data?.detail || 'Failed to start processing');
        } finally {
            setProcessingIds(prev => {
                const next = new Set(prev);
                next.delete(videoId);
                return next;
            });
        }
    };

    const handleProcessAll = async () => {
        setProcessingAll(true);
        try {
            const res = await api.post(`/channels/${id}/process-all`);
            alert(`Queued ${res.data.queued} videos for processing`);
            fetchVideos();
        } catch (e) {
            console.error('Failed to process all:', e);
            alert('Failed to queue videos');
        } finally {
            setProcessingAll(false);
        }
    };



    const handleToggleMute = async (videoId: number, e?: MouseEvent) => {
        e?.stopPropagation();
        try {
            const res = await api.patch<Video>(`/videos/${videoId}/mute`);
            setVideos(prev => prev.map(v => v.id === videoId ? res.data : v));
        } catch (e) {
            console.error('Failed to toggle mute:', e);
        }
    };

    const handleMuteFiltered = async () => {
        const unmutedFiltered = processedVideos.filter(v => !v.muted && !v.access_restricted);
        if (unmutedFiltered.length === 0) return;

        if (!confirm(`Mute ${unmutedFiltered.length} filtered videos?`)) return;

        setMutingFiltered(true);
        try {
            for (const video of unmutedFiltered) {
                await api.patch<Video>(`/videos/${video.id}/mute`);
            }
            await fetchVideos();
        } catch (e) {
            console.error('Failed to mute videos:', e);
        } finally {
            setMutingFiltered(false);
        }
    };

    const formatDuration = (seconds?: number) => {
        if (!seconds) return '--:--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = seconds % 60;
        if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    const unprocessedCount = videos.filter(isProcessableVideo).length;
    const unmutedFilteredCount = processedVideos.filter(v => !v.muted && !v.access_restricted).length;
    const isRefreshingChannel = !isManualChannel && (channel?.status === 'refreshing' || fetching);
    const isDateBackfillActive = missingDateCount > 0 && (isRefreshingChannel || backfillPollCountRef.current < DATE_BACKFILL_MAX_POLLS);
    const syncProgress = getChannelSyncProgress();
    const activityMessage = isRefreshingChannel
        ? videos.length > 0
            ? (channel?.sync_status_detail || (missingDateCount > 0
                ? `${isTikTokChannel ? 'Scanning TikTok for more videos' : 'Scanning YouTube for more videos'}. Publication dates are still being filled in for ${missingDateCount} video${missingDateCount === 1 ? '' : 's'}.`
                : `${isTikTokChannel ? 'Scanning TikTok for more videos' : 'Scanning YouTube for more videos'}. This list updates automatically as new metadata arrives.`))
            : (channel?.sync_status_detail || `Scanning the channel and importing the initial video list from ${isTikTokChannel ? 'TikTok' : 'YouTube'}. This page updates automatically.`)
        : isDateBackfillActive
            ? `Backfilling publication dates for ${missingDateCount} video${missingDateCount === 1 ? '' : 's'}.`
            : null;

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-slate-400 animate-pulse">
                Loading videos...
            </div>
        );
    }

    if (videos.length === 0 && isRefreshingChannel) {
        return (
            <div className="glass-panel rounded-2xl p-12 text-center">
                <Loader2 size={48} className="mx-auto mb-4 text-amber-400 animate-spin" />
                <h3 className="text-lg font-semibold text-slate-700 mb-2">Importing Channel Videos</h3>
                <p className="text-slate-500 mb-2">
                    {channel?.sync_status_detail || `${isTikTokChannel ? 'TikTok' : 'YouTube'} metadata is still coming in for this channel.`}
                </p>
                <div className="mx-auto mt-5 max-w-md text-left">
                    <div className="mb-1 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.16em] text-amber-700/80">
                        <span>Channel Sync</span>
                        <span>{syncProgress}%</span>
                    </div>
                    <div className="h-2.5 rounded-full bg-amber-100">
                        <div
                            className="h-2.5 rounded-full bg-amber-500 transition-all duration-500"
                            style={{ width: `${Math.max(syncProgress, 4)}%` }}
                        />
                    </div>
                    {(channel?.sync_total_items ?? 0) > 0 ? (
                        <p className="mt-2 text-xs text-amber-700">
                            {channel?.sync_completed_items ?? 0}/{channel?.sync_total_items ?? 0} items complete
                        </p>
                    ) : (
                        <p className="mt-2 text-xs text-amber-700">
                            The video list appears first, and publication dates can continue filling in after that.
                        </p>
                    )}
                </div>
            </div>
        );
    }

    if (videos.length === 0) {
        return (
            <div className="glass-panel rounded-2xl p-12 text-center">
                <Download size={48} className="mx-auto text-slate-300 mb-4" />
                <h3 className="text-lg font-semibold text-slate-700 mb-2">{isManualChannel ? 'No Episodes Yet' : 'No Videos Yet'}</h3>
                <p className="text-slate-500 mb-6">
                    {isManualChannel
                        ? 'Upload an audio or video file to create the first episode in this manual channel.'
                        : isTikTokChannel
                            ? 'Add a TikTok video URL, or refresh the creator profile to pull the full feed.'
                            : 'Fetch the video list from YouTube to get started.'}
                </p>
                {isManualChannel ? (
                    <>
                        <input
                            ref={uploadInputRef}
                            type="file"
                            accept="audio/*,video/*"
                            className="hidden"
                            onChange={handleUploadMedia}
                        />
                        <button
                            onClick={() => uploadInputRef.current?.click()}
                            disabled={uploadingMedia}
                            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all font-medium disabled:opacity-50"
                        >
                            {uploadingMedia ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
                            {uploadingMedia ? 'Uploading...' : 'Upload Media'}
                        </button>
                    </>
                ) : isTikTokChannel ? (
                    <div className="flex flex-wrap items-center justify-center gap-3">
                        <button
                            onClick={handleAddVideo}
                            disabled={addingVideo}
                            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-pink-500 to-rose-500 text-white rounded-xl hover:shadow-lg hover:shadow-pink-500/25 transition-all font-medium disabled:opacity-50"
                        >
                            {addingVideo ? <Loader2 size={18} className="animate-spin" /> : <Plus size={18} />}
                            {addingVideo ? 'Adding...' : 'Add TikTok Video'}
                        </button>
                        {String(channel?.url || '').toLowerCase().includes('tiktok.com') && (
                            <button
                                onClick={handleFetchFromYouTube}
                                disabled={fetching}
                                className="inline-flex items-center gap-2 px-6 py-3 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition-all font-medium disabled:opacity-50"
                            >
                                {fetching ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
                                {fetching ? 'Refreshing...' : 'Refresh Profile'}
                            </button>
                        )}
                    </div>
                ) : (
                    <button
                        onClick={handleFetchFromYouTube}
                        disabled={fetching}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all font-medium disabled:opacity-50"
                    >
                        {fetching ? (
                            <>
                                <Loader2 size={18} className="animate-spin" />
                                Fetching Videos...
                            </>
                        ) : (
                            <>
                                <Download size={18} />
                                Fetch Video List
                            </>
                        )}
                    </button>
                )}
            </div>
        );
    }

    return (
        <div className="flex gap-4 h-[calc(100vh-200px)]">
            {/* Video list section */}
            <div className="flex-1 flex flex-col space-y-4 transition-all">
                {/* Header with search and action buttons */}
                <div className="flex flex-col gap-3 shrink-0">
                    {activityMessage && (
                        <div className="rounded-xl border border-amber-200 bg-amber-50/85 px-4 py-3 text-sm text-amber-900">
                            <div className="flex items-center gap-2 font-semibold">
                                <Loader2 size={15} className={`${isRefreshingChannel || isDateBackfillActive ? 'animate-spin' : ''}`} />
                                {isRefreshingChannel ? 'Channel sync in progress' : 'Metadata backfill in progress'}
                            </div>
                            <p className="mt-1 text-amber-800/80">{activityMessage}</p>
                            {isRefreshingChannel && (
                                <div className="mt-3">
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
                                    {(channel?.sync_total_items ?? 0) > 0 && (
                                        <p className="mt-1 text-[10px] text-amber-700/75">
                                            {channel?.sync_completed_items ?? 0}/{channel?.sync_total_items ?? 0} items
                                        </p>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Search bar */}
                    <div className="relative">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                            type="text"
                            placeholder="Search title or description..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-9 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400"
                        />
                        {searchQuery && (
                            <button
                                onClick={() => setSearchQuery('')}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                            >
                                <X size={16} />
                            </button>
                        )}
                    </div>

                    {/* Status filter chips */}
                    <div className="flex items-center gap-2">
                        <Filter size={14} className="text-slate-400 shrink-0" />
                        {[
                            { key: 'unprocessed', label: 'Unprocessed', color: 'slate' },
                            { key: 'completed', label: 'Completed', color: 'green' },
                            { key: 'failed', label: 'Failed', color: 'red' },
                        ].map(({ key, label, color }) => {
                            const active = statusFilters.has(key);
                            const colorMap: Record<string, string> = {
                                slate: active ? 'bg-slate-500 text-white' : 'bg-slate-100 text-slate-500 hover:bg-slate-200',
                                green: active ? 'bg-green-500 text-white' : 'bg-green-50 text-green-600 hover:bg-green-100',
                                red: active ? 'bg-red-500 text-white' : 'bg-red-50 text-red-600 hover:bg-red-100',
                            };
                            return (
                                <button
                                    key={key}
                                    onClick={() => toggleStatusFilter(key)}
                                    className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${colorMap[color]}`}
                                >
                                    {label}
                                </button>
                            );
                        })}
                        {statusFilters.size > 0 && (
                            <button
                                onClick={() => setStatusFilters(new Set())}
                                className="text-xs text-slate-400 hover:text-slate-600 ml-1"
                            >
                                Clear
                            </button>
                        )}
                    </div>

                    {/* Actions row */}
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-4">
                            <p className="text-sm text-slate-500">
                                {(searchQuery || statusFilters.size > 0) ? `${processedVideos.length} of ${videos.length}` : `${videos.length}`} video{videos.length !== 1 ? 's' : ''}
                                {unprocessedCount > 0 && !searchQuery && ` | ${unprocessedCount} ready`}
                            </p>
                            {/* View toggle */}
                            <div className="flex items-center bg-slate-100 rounded-lg p-0.5">
                                <button
                                    onClick={() => setViewMode('compact')}
                                    className={`p-1.5 rounded ${viewMode === 'compact' ? 'bg-white shadow-sm text-blue-600' : 'text-slate-400'}`}
                                    title="Compact view"
                                >
                                    <List size={16} />
                                </button>
                                <button
                                    onClick={() => setViewMode('grid')}
                                    className={`p-1.5 rounded ${viewMode === 'grid' ? 'bg-white shadow-sm text-blue-600' : 'text-slate-400'}`}
                                    title="Grid view"
                                >
                                    <Grid size={16} />
                                </button>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            {/* Mute filtered button */}
                            {searchQuery && unmutedFilteredCount > 0 && (
                                <button
                                    onClick={handleMuteFiltered}
                                    disabled={mutingFiltered}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-all text-sm font-medium disabled:opacity-50"
                                    title="Mute all filtered videos"
                                >
                                    {mutingFiltered ? <Loader2 size={14} className="animate-spin" /> : <EyeOff size={14} />}
                                    Mute ({unmutedFilteredCount})
                                </button>
                            )}
                            {isManualChannel ? (
                                <>
                                    <input
                                        ref={uploadInputRef}
                                        type="file"
                                        accept="audio/*,video/*"
                                        className="hidden"
                                        onChange={handleUploadMedia}
                                    />
                                    <button
                                        onClick={() => uploadInputRef.current?.click()}
                                        disabled={uploadingMedia}
                                        className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-all text-sm font-medium disabled:opacity-50"
                                        title="Upload an audio or video file"
                                    >
                                        {uploadingMedia ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
                                        {uploadingMedia ? 'Uploading...' : 'Upload Media'}
                                    </button>
                                </>
                            ) : isTikTokChannel ? (
                                <>
                                    <button
                                        onClick={handleAddVideo}
                                        disabled={addingVideo}
                                        className="flex items-center gap-2 px-3 py-1.5 bg-pink-100 text-pink-700 rounded-lg hover:bg-pink-200 transition-all text-sm font-medium disabled:opacity-50"
                                        title="Add a video by TikTok URL"
                                    >
                                        {addingVideo ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                                        {addingVideo ? 'Adding...' : 'Add TikTok'}
                                    </button>
                                    {String(channel?.url || '').toLowerCase().includes('tiktok.com') && (
                                        <button
                                            onClick={handleFetchFromYouTube}
                                            disabled={fetching}
                                            className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-all text-sm font-medium disabled:opacity-50"
                                        >
                                            {fetching ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                            {fetching ? 'Refreshing...' : 'Refresh'}
                                        </button>
                                    )}
                                </>
                            ) : (
                                <>
                                    <button
                                        onClick={handleAddVideo}
                                        disabled={addingVideo}
                                        className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-all text-sm font-medium disabled:opacity-50"
                                        title="Add a video by YouTube URL"
                                    >
                                        {addingVideo ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                                        {addingVideo ? 'Adding...' : 'Add Video'}
                                    </button>
                                    <button
                                        onClick={handleFetchFromYouTube}
                                        disabled={fetching}
                                        className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-all text-sm font-medium disabled:opacity-50"
                                    >
                                        {fetching ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                        {fetching ? 'Fetching...' : 'Refresh'}
                                    </button>
                                </>
                            )}
                            {unprocessedCount > 0 && !searchQuery && (
                                <button
                                    onClick={handleProcessAll}
                                    disabled={processingAll}
                                    className="flex items-center gap-2 px-4 py-1.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-lg hover:shadow-lg transition-all text-sm font-medium disabled:opacity-50"
                                >
                                    {processingAll ? <Loader2 size={14} className="animate-spin" /> : null}
                                    {processingAll ? 'Queueing...' : `Process All (${unprocessedCount})`}
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* No results message */}
                {searchQuery && processedVideos.length === 0 && (
                    <div className="glass-panel rounded-xl p-8 text-center text-slate-500">
                        No videos match "{searchQuery}"
                    </div>
                )}

                {/* Compact List View */}
                {viewMode === 'compact' && processedVideos.length > 0 && (
                    <div className="glass-panel rounded-xl overflow-hidden flex-1 flex flex-col">
                        <div className="overflow-auto flex-1">
                            <table className="w-full text-sm relative">
                                <thead className="bg-slate-50 sticky top-0 z-10 shadow-sm">
                                    <tr className="text-left text-slate-500 text-xs uppercase tracking-wider">
                                        <th className="px-2 py-3 w-10"></th>
                                        <th
                                            className="px-2 py-3 cursor-pointer hover:bg-slate-100 transition-colors"
                                            onClick={() => handleSort('title')}
                                        >
                                            <div className="flex items-center gap-1">
                                                Title {getSortIcon('title')}
                                            </div>
                                        </th>
                                        <th
                                            className="px-2 py-3 w-28 text-center cursor-pointer hover:bg-slate-100 transition-colors"
                                            onClick={() => handleSort('published_at')}
                                        >
                                            <div className="flex items-center justify-center gap-1">
                                                Date {getSortIcon('published_at')}
                                            </div>
                                        </th>
                                        <th
                                            className="px-2 py-3 w-20 text-center cursor-pointer hover:bg-slate-100 transition-colors"
                                            onClick={() => handleSort('duration')}
                                        >
                                            <div className="flex items-center justify-center gap-1">
                                                Dur {getSortIcon('duration')}
                                            </div>
                                        </th>
                                        <th
                                            className="px-2 py-3 w-32 text-center cursor-pointer hover:bg-slate-100 transition-colors"
                                            onClick={() => handleSort('status')}
                                        >
                                            <div className="flex items-center justify-center gap-1">
                                                Status {getSortIcon('status')}
                                            </div>
                                        </th>
                                        <th className="px-2 py-3 w-20 text-center">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {visibleVideos.map((video) => (
                                        <tr
                                            key={video.id}
                                            onClick={() => navigate(`/video/${video.id}`)}
                                            className={`transition-colors cursor-pointer group ${video.access_restricted ? 'bg-slate-50/90 opacity-65' : 'hover:bg-slate-50'} ${video.muted ? 'opacity-50 bg-slate-50' : ''}`}
                                        >
                                            {/* Thumbnail */}
                                            <td className="px-2 py-2">
                                                <div className="w-12 h-8 bg-slate-200 rounded overflow-hidden flex-shrink-0 relative">
                                                    {video.thumbnail_url ? (
                                                        <img src={video.thumbnail_url} alt="" className="w-full h-full object-cover" />
                                                    ) : (
                                                        <div className="flex items-center justify-center h-full">
                                                            <Play size={10} className="text-slate-400" />
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                            {/* Title */}
                                            <td className="px-2 py-2">
                                                <div className="flex flex-col gap-1">
                                                    <span className={`font-medium text-slate-700 line-clamp-2 text-xs leading-snug ${video.muted ? 'line-through' : ''}`} title={video.title}>
                                                        {video.title}
                                                    </span>
                                                    {video.access_restricted && (
                                                        <span
                                                            className="inline-flex w-fit items-center rounded-full bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-700"
                                                            title={video.access_restriction_reason || 'This video is not accessible with the current YouTube session.'}
                                                        >
                                                            {getAccessRestrictionLabel(video)}
                                                        </span>
                                                    )}
                                                    {video.transcript_is_placeholder && (
                                                        <span className="inline-flex w-fit items-center rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                                                            {getPlaceholderLabel(video)}
                                                        </span>
                                                    )}
                                                    {getSourceMediaLabel(video) && (
                                                        <span className={`inline-flex w-fit items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                                                            video.media_source_type === 'tiktok'
                                                                ? 'bg-pink-100 text-pink-800'
                                                                : 'bg-blue-100 text-blue-800'
                                                        }`}>
                                                            {getSourceMediaLabel(video)}
                                                        </span>
                                                    )}
                                                </div>
                                            </td>
                                            {/* Date */}
                                            <td className="px-2 py-2 text-center text-slate-500 text-xs whitespace-nowrap">
                                                {video.published_at ? (
                                                    new Date(video.published_at).toLocaleDateString()
                                                ) : isDateBackfillActive ? (
                                                    <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-800">
                                                        Fetching...
                                                    </span>
                                                ) : '-'}
                                            </td>
                                            {/* Duration */}
                                            <td className="px-2 py-2 text-center text-slate-500 text-xs font-mono">
                                                {formatDuration(video.duration)}
                                            </td>
                                            {/* Status */}
                                            <td className="px-2 py-2">
                                                <div className="flex justify-center">
                                                    <VideoStatusBadge
                                                        status={video.status}
                                                        processed={video.processed}
                                                        accessRestricted={video.access_restricted}
                                                        className={video.muted || video.access_restricted ? 'opacity-50' : ''}
                                                    />
                                                </div>
                                            </td>
                                            {/* Actions */}

                                            <td className="px-2 py-2">
                                                <div className="flex items-center justify-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
                                                    <button
                                                        onClick={(e) => handleToggleMute(video.id, e)}
                                                        className={`p-1.5 rounded transition-colors ${video.muted ? 'bg-red-100 text-red-600 hover:bg-red-200' : 'hover:bg-slate-200 text-slate-400 hover:text-slate-600'
                                                            }`}
                                                        title={video.muted ? 'Unmute' : 'Mute'}
                                                    >
                                                        {video.muted ? <EyeOff size={14} /> : <Eye size={14} />}
                                                    </button>
                                                    {!video.processed && !video.muted && !video.access_restricted && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); handleProcess(video.id); }}
                                                            disabled={processingIds.has(video.id)}
                                                            className="p-1.5 rounded bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors disabled:opacity-50"
                                                            title="Process"
                                                        >
                                                            {processingIds.has(video.id) ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                                                        </button>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {canLoadMore && (
                            <div className="border-t border-slate-100 p-3 flex justify-center">
                                <button
                                    onClick={() => setVisibleCount(prev => prev + VIDEO_PAGE_SIZE)}
                                    className="px-3 py-1.5 text-xs font-medium bg-white border border-slate-200 rounded-lg hover:bg-slate-50"
                                >
                                    Load more videos ({processedVideos.length - visibleVideos.length} remaining)
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {/* Grid View */}
                {viewMode === 'grid' && processedVideos.length > 0 && (
                    <div className="flex-1 overflow-y-auto pr-2">
                        <div className={`grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3`}>
                            {visibleVideos.map((video) => (
                                <div
                                    key={video.id}
                                    onClick={() => navigate(`/video/${video.id}`)}
                                    className={`glass-panel rounded-lg overflow-hidden transition-all duration-300 group cursor-pointer ${video.access_restricted ? 'opacity-65 saturate-0' : video.muted ? 'opacity-50 grayscale' : 'hover:shadow-lg hover:-translate-y-0.5'
                                        }`}
                                >
                                    {/* Thumbnail */}
                                    <div className="aspect-video bg-slate-200 relative overflow-hidden">
                                        {video.thumbnail_url ? (
                                            <img src={video.thumbnail_url} alt={video.title} className="w-full h-full object-cover" />
                                        ) : (
                                            <div className="flex items-center justify-center h-full">
                                                <Play size={20} className="text-slate-400" />
                                            </div>
                                        )}
                                        {/* Mute button overlay */}
                                        <button
                                            onClick={(e) => handleToggleMute(video.id, e)}
                                            className={`absolute top-1 right-1 p-1 rounded-full transition-all ${video.muted ? 'bg-red-500 text-white' : 'bg-black/50 text-white/80 opacity-0 group-hover:opacity-100'
                                                }`}
                                            title={video.muted ? 'Unmute' : 'Mute'}
                                        >
                                            {video.muted ? <EyeOff size={10} /> : <Eye size={10} />}
                                        </button>
                                        {video.duration && (
                                            <span className="absolute bottom-1 right-1 bg-black/70 text-white px-1 py-0.5 rounded text-[9px]">
                                                {formatDuration(video.duration)}
                                            </span>
                                        )}
                                    </div>
                                    {/* Content */}
                                    <div className="p-1.5">
                                        <h3 className={`text-[10px] font-medium text-slate-700 line-clamp-2 leading-tight ${video.muted ? 'line-through' : ''}`} title={video.title}>
                                            {video.title}
                                        </h3>
                                        {video.access_restricted && (
                                            <div
                                                className="mt-1 inline-flex items-center rounded-full bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-700"
                                                title={video.access_restriction_reason || 'This video is not accessible with the current YouTube session.'}
                                            >
                                                {getAccessRestrictionLabel(video)}
                                            </div>
                                        )}
                                        {video.transcript_is_placeholder && (
                                            <div className="mt-1 inline-flex items-center rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                                                {getPlaceholderLabel(video)}
                                            </div>
                                        )}
                                        {getSourceMediaLabel(video) && (
                                            <div className={`mt-1 inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                                                video.media_source_type === 'tiktok'
                                                    ? 'bg-pink-100 text-pink-800'
                                                    : 'bg-blue-100 text-blue-800'
                                            }`}>
                                                {getSourceMediaLabel(video)}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                        {canLoadMore && (
                            <div className="pt-3 pb-1 flex justify-center">
                                <button
                                    onClick={() => setVisibleCount(prev => prev + VIDEO_PAGE_SIZE)}
                                    className="px-3 py-1.5 text-xs font-medium bg-white border border-slate-200 rounded-lg hover:bg-slate-50"
                                >
                                    Load more videos ({processedVideos.length - visibleVideos.length} remaining)
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
