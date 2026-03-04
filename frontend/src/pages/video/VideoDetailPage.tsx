import { useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type { Video, TranscriptSegment, Clip, Speaker, FunnyMoment, VideoChapterSuggestion, VideoDescriptionRevision, ClipExportArtifact } from '../../types';
import { Loader2, ArrowLeft, FileText, Scissors, Users, X, CheckCircle2, Play, Pause, Plus, Trash2, Mic, Search, ChevronUp, ChevronDown, GitMerge, RotateCcw, Eraser, AudioLines, Smile, RefreshCw, Bot, Copy, Pencil, Save, XCircle, Download, Upload, Clock } from 'lucide-react';
import { SpeakerList } from '../../components/SpeakerList';
import { SpeakerModal } from '../../components/SpeakerModal';

export function VideoDetailPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();

    // Data State
    const [video, setVideo] = useState<Video | null>(null);
    const [segments, setSegments] = useState<TranscriptSegment[]>([]);
    const [funnyMoments, setFunnyMoments] = useState<FunnyMoment[]>([]);
    const [loading, setLoading] = useState(true);

    // UI State
    const [activeTab, setActiveTab] = useState<'transcript' | 'clips' | 'speakers' | 'youtube'>('transcript');
    const transcriptRef = useRef<HTMLDivElement>(null);
    const cropPreviewRef = useRef<HTMLDivElement>(null);
    const clipTimelineRef = useRef<HTMLDivElement>(null);
    const initialJumpDoneRef = useRef(false);
    const initialSeekDoneRef = useRef(false);
    const lastAutoScrollSegIdRef = useRef<number | null>(null);

    // Player State
    const [player, setPlayer] = useState<any>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const playerClockRef = useRef<{
        mediaTime: number;
        wallTimeMs: number;
        playbackRate: number;
        playerState: number;
    }>({
        mediaTime: 0,
        wallTimeMs: 0,
        playbackRate: 1,
        playerState: -1,
    });

    // Clipping State
    const [selection, setSelection] = useState<{ start: number; end: number; defaultTitle: string } | null>(null);
    const [clipTitle, setClipTitle] = useState('');
    const [creatingClip, setCreatingClip] = useState(false);

    // Speaker Modal State
    const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
    const [initialSample, setInitialSample] = useState<any>(null);

    // Clips State
    const [clips, setClips] = useState<Clip[]>([]);
    const [clipExportArtifactsByClip, setClipExportArtifactsByClip] = useState<Record<number, ClipExportArtifact[]>>({});
    const [loadingClips, setLoadingClips] = useState(false);
    const [startingTranscription, setStartingTranscription] = useState(false);

    // Search State
    const [searchQuery, setSearchQuery] = useState('');
    const [searchMatchIndex, setSearchMatchIndex] = useState(0);
    const [followPlayback, setFollowPlayback] = useState(true);
    const [loadingFunnyMoments, setLoadingFunnyMoments] = useState(false);
    const [detectingFunnyMoments, setDetectingFunnyMoments] = useState(false);
    const [funnyDrawerOpen, setFunnyDrawerOpen] = useState(false);
    const [explainingFunnyMoments, setExplainingFunnyMoments] = useState(false);
    const [showGlobalHumorContext, setShowGlobalHumorContext] = useState(false);
    const [expandedFunnySummaryIds, setExpandedFunnySummaryIds] = useState<Set<number>>(new Set());
    const [funnyTaskProgress, setFunnyTaskProgress] = useState<{
        video_id: number;
        task?: 'detect' | 'explain' | string;
        status: 'idle' | 'running' | 'completed' | 'error' | string;
        stage?: string | null;
        message?: string | null;
        percent?: number | null;
        current?: number | null;
        total?: number | null;
    } | null>(null);
    const [generatingYoutubeAi, setGeneratingYoutubeAi] = useState(false);
    const [copiedYoutubeField, setCopiedYoutubeField] = useState<'summary' | 'chapters' | 'description' | null>(null);
    const [descriptionHistory, setDescriptionHistory] = useState<VideoDescriptionRevision[]>([]);
    const [loadingDescriptionHistory, setLoadingDescriptionHistory] = useState(false);
    const [publishingYoutubeDescription, setPublishingYoutubeDescription] = useState(false);
    const [restoringDescriptionRevisionId, setRestoringDescriptionRevisionId] = useState<number | null>(null);
    const [editingSegmentId, setEditingSegmentId] = useState<number | null>(null);
    const [editingSegmentWords, setEditingSegmentWords] = useState<string[]>([]);
    const [editingLoopSegment, setEditingLoopSegment] = useState(false);
    const [savingSegmentEdit, setSavingSegmentEdit] = useState(false);
    const [selectedClipIds, setSelectedClipIds] = useState<Set<number>>(new Set());
    const [editingClipId, setEditingClipId] = useState<number | null>(null);
    const [clipEditorDraft, setClipEditorDraft] = useState<Partial<Clip> | null>(null);
    const [clipEditorTokens, setClipEditorTokens] = useState<Array<{ key: string; start: number; end: number; word: string }>>([]);
    const [clipEditorRemovedWordKeys, setClipEditorRemovedWordKeys] = useState<Set<string>>(new Set());
    const [clipEditorCropTarget, setClipEditorCropTarget] = useState<'main' | 'top' | 'bottom'>('main');
    const [clipEditorDragRect, setClipEditorDragRect] = useState<{
        target: 'main' | 'top' | 'bottom';
        startX: number;
        startY: number;
        currentX: number;
        currentY: number;
    } | null>(null);
    const [clipTimelineDrag, setClipTimelineDrag] = useState<{
        handle: 'start' | 'end';
    } | null>(null);
    const [savingClipEdit, setSavingClipEdit] = useState(false);
    const [exportingClipIds, setExportingClipIds] = useState<Set<number>>(new Set());
    const [batchExporting, setBatchExporting] = useState(false);
    const [batchQueueingRenders, setBatchQueueingRenders] = useState(false);
    const [uploadingClipIds, setUploadingClipIds] = useState<Set<number>>(new Set());
    const [batchUploadingClips, setBatchUploadingClips] = useState(false);
    const [clipUploadPrivacy, setClipUploadPrivacy] = useState<'private' | 'unlisted' | 'public'>('private');
    const [clipPreviewLoop, setClipPreviewLoop] = useState<{ start: number; end: number; clipId: number } | null>(null);
    const [clipBatchPresetKey, setClipBatchPresetKey] = useState<'youtube_landscape' | 'shorts_vertical' | 'square_captioned' | 'audio_focus'>('youtube_landscape');

    // Assign Speaker State (for segments with no speaker_id)
    const [assignPopup, setAssignPopup] = useState<{
        segmentId: number;
        x: number;
        y: number;
    } | null>(null);
    const [assignSpeakers, setAssignSpeakers] = useState<Speaker[]>([]);
    const [assignSearch, setAssignSearch] = useState('');
    const [assignLoading, setAssignLoading] = useState(false);
    const [splittingSegmentId, setSplittingSegmentId] = useState<number | null>(null);
    const activeEditingClip = editingClipId != null ? (clips.find(c => c.id === editingClipId) || null) : null;
    const showClipEditorMain = activeTab === 'clips' && !!activeEditingClip && !!clipEditorDraft;

    const fetchFunnyMoments = async () => {
        if (!id) return;
        setLoadingFunnyMoments(true);
        try {
            const res = await api.get<FunnyMoment[]>(`/videos/${id}/funny-moments`);
            setFunnyMoments(res.data);
        } catch (e) {
            console.error('Failed to fetch funny moments:', e);
            setFunnyMoments([]);
        } finally {
            setLoadingFunnyMoments(false);
        }
    };

    const fetchFunnyTaskProgress = async () => {
        if (!id) return;
        try {
            const res = await api.get(`/videos/${id}/funny-moments/progress`);
            setFunnyTaskProgress(res.data || null);
        } catch {
            // Ignore transient polling failures.
        }
    };

    const fetchVideoMeta = async () => {
        if (!id) return null;
        const vidRes = await api.get<Video>(`/videos/${id}`);
        setVideo(vidRes.data);
        return vidRes.data;
    };

    const fetchData = async () => {
        try {
            await fetchVideoMeta();
        } catch (e) {
            console.error('Failed to fetch video:', e);
            setLoading(false);
            return;
        }
        try {
            const segRes = await api.get<TranscriptSegment[]>(`/videos/${id}/segments`);
            setSegments(segRes.data);
            if (segRes.data.length > 0) {
                void fetchFunnyMoments();
            } else {
                setFunnyMoments([]);
            }
        } catch (e) {
            console.error('Failed to fetch segments:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (id) fetchData();
    }, [id]);

    useEffect(() => {
        // Reset deep-link jump state when navigating to a different video.
        initialJumpDoneRef.current = false;
        initialSeekDoneRef.current = false;
        setExpandedFunnySummaryIds(new Set());
    }, [id]);

    const toggleFunnySummaryExpanded = (momentId: number) => {
        setExpandedFunnySummaryIds(prev => {
            const next = new Set(prev);
            if (next.has(momentId)) {
                next.delete(momentId);
            } else {
                next.add(momentId);
            }
            return next;
        });
    };

    const samplePlayerClock = (ytPlayer: any) => {
        if (!ytPlayer) return;
        try {
            const mediaTimeRaw = ytPlayer.getCurrentTime?.();
            const stateRaw = ytPlayer.getPlayerState?.();
            const rateRaw = ytPlayer.getPlaybackRate?.();
            const mediaTime = Number(mediaTimeRaw);
            const state = Number(stateRaw);
            const rate = Number(rateRaw);
            if (!Number.isFinite(mediaTime)) return;

            const nextRate = Number.isFinite(rate) && rate > 0 ? rate : playerClockRef.current.playbackRate;
            const nextState = Number.isFinite(state) ? state : playerClockRef.current.playerState;

            playerClockRef.current = {
                mediaTime,
                wallTimeMs: performance.now(),
                playbackRate: nextRate,
                playerState: nextState,
            };
            setCurrentTime(prev => (Math.abs(prev - mediaTime) >= 0.015 ? mediaTime : prev));
        } catch {
            // Ignore transient iframe/player API failures.
        }
    };

    useEffect(() => {
        if (!player) return;

        samplePlayerClock(player);
        // Use YouTube player time as the single source of truth.
        // This avoids drift/overshoot from extrapolation when iframe timing events stall.
        const pollId = window.setInterval(() => samplePlayerClock(player), 33);

        return () => {
            window.clearInterval(pollId);
        };
    }, [player]);

    useEffect(() => {
        if (!clipPreviewLoop) return;
        if (currentTime >= Math.max(clipPreviewLoop.start, clipPreviewLoop.end - 0.35)) {
            try {
                if (player && typeof player.seekTo === 'function') {
                    player.seekTo(clipPreviewLoop.start, true);
                }
                if (player && typeof player.playVideo === 'function') {
                    player.playVideo();
                }
            } catch {}
        }
    }, [currentTime, clipPreviewLoop, player]);

    useEffect(() => {
        if (!editingSegmentId || !editingLoopSegment || !player) return;
        const seg = segments.find(s => s.id === editingSegmentId);
        if (!seg) return;

        try {
            player.seekTo?.(seg.start_time, true);
            player.playVideo?.();
        } catch {
            // ignore
        }

        const loopId = window.setInterval(() => {
            try {
                const t = Number(player.getCurrentTime?.());
                if (!Number.isFinite(t)) return;
                if (t >= Math.max(seg.start_time, seg.end_time - 0.12)) {
                    player.seekTo?.(seg.start_time, true);
                    player.playVideo?.();
                }
            } catch {
                // ignore transient iframe failures
            }
        }, 100);

        return () => window.clearInterval(loopId);
    }, [editingSegmentId, editingLoopSegment, player, segments]);

    useEffect(() => {
        if (!clipEditorDragRect) return;
        const onMove = (evt: PointerEvent) => {
            if (!cropPreviewRef.current) return;
            const bounds = cropPreviewRef.current.getBoundingClientRect();
            if (bounds.width <= 0 || bounds.height <= 0) return;
            const nx = clampNorm((evt.clientX - bounds.left) / bounds.width);
            const ny = clampNorm((evt.clientY - bounds.top) / bounds.height);
            setClipEditorDragRect(prev => prev ? { ...prev, currentX: nx, currentY: ny } : prev);
        };
        const onUp = () => {
            setClipEditorDragRect(prev => {
                if (!prev) return null;
                const x = Math.min(prev.startX, prev.currentX);
                const y = Math.min(prev.startY, prev.currentY);
                const w = Math.max(0.01, Math.abs(prev.currentX - prev.startX));
                const h = Math.max(0.01, Math.abs(prev.currentY - prev.startY));
                setDraftCropRect(prev.target, { x, y, w, h });
                return null;
            });
        };
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        return () => {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
        };
    }, [clipEditorDragRect]);

    useEffect(() => {
        if (!clipTimelineDrag) return;
        const onMove = (evt: PointerEvent) => {
            if (!clipEditorDraft) return;
            const duration = getEditorMediaDuration();
            const t = timelineEventToTime(evt, duration);
            const start = Number(clipEditorDraft.start_time ?? 0);
            const end = Number(clipEditorDraft.end_time ?? 0);
            if (clipTimelineDrag.handle === 'start') {
                updateClipDraftField('start_time', Number(Math.max(0, Math.min(end - 0.05, t)).toFixed(3)));
            } else {
                updateClipDraftField('end_time', Number(Math.max(start + 0.05, t).toFixed(3)));
            }
        };
        const onUp = () => setClipTimelineDrag(null);
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        return () => {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
        };
    }, [clipTimelineDrag, clipEditorDraft]);

    useEffect(() => {
        const splitEnabled = String(clipEditorDraft?.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft?.portrait_split_enabled;
        if (!splitEnabled && clipEditorCropTarget !== 'main') {
            setClipEditorCropTarget('main');
        }
    }, [clipEditorDraft?.aspect_ratio, clipEditorDraft?.portrait_split_enabled, clipEditorCropTarget]);

    useEffect(() => {
        if (!showClipEditorMain) return;
        const onKeyDown = (evt: KeyboardEvent) => {
            const target = evt.target as HTMLElement | null;
            const tag = (target?.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable) return;
            const frame = 1 / 30;
            if (evt.key.toLowerCase() === 'i') {
                evt.preventDefault();
                setClipBoundaryFromPlayhead('start');
                return;
            }
            if (evt.key.toLowerCase() === 'o') {
                evt.preventDefault();
                setClipBoundaryFromPlayhead('end');
                return;
            }
            if (evt.altKey && evt.key === 'ArrowLeft') {
                evt.preventDefault();
                nudgeClipBoundary(evt.shiftKey ? 'end' : 'start', -frame);
                return;
            }
            if (evt.altKey && evt.key === 'ArrowRight') {
                evt.preventDefault();
                nudgeClipBoundary(evt.shiftKey ? 'end' : 'start', frame);
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [showClipEditorMain, currentTime, clipEditorDraft]);

    const tParam = searchParams.get('t');
    const requestedJumpTime = tParam ? Number(tParam) : NaN;

    useEffect(() => {
        if (initialJumpDoneRef.current) return;
        if (!Number.isFinite(requestedJumpTime) || requestedJumpTime < 0) return;
        if (segments.length === 0) return;

        initialJumpDoneRef.current = true;
        const timer = window.setTimeout(() => {
            try {
                scrollTranscriptToTime(requestedJumpTime);
            } catch (e) {
                console.warn('Initial transcript scroll failed', e);
            }
        }, 80);

        return () => window.clearTimeout(timer);
    }, [requestedJumpTime, segments.length]);

    const onPlayerReady = (event: any) => {
        setPlayer(event.target);
        samplePlayerClock(event.target);
        if (!initialSeekDoneRef.current && Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0) {
            initialSeekDoneRef.current = true;
            window.setTimeout(() => {
                try {
                    if (typeof event.target.seekTo === 'function') {
                        event.target.seekTo(requestedJumpTime, true);
                    }
                    // Do not autoplay when opening episode detail (including deep links).
                    if (typeof event.target.pauseVideo === 'function') {
                        event.target.pauseVideo();
                    }
                    samplePlayerClock(event.target);
                } catch (e) {
                    console.warn('Initial timestamp seek failed', e);
                    initialSeekDoneRef.current = false;
                }
            }, 150);
        }
    };

    const onPlayerStateChange = (event: any) => {
        const state = Number(event?.data);
        if (!Number.isFinite(state)) return;
        const now = performance.now();
        let mediaTime = playerClockRef.current.mediaTime;
        try {
            const t = Number(event?.target?.getCurrentTime?.());
            if (Number.isFinite(t)) mediaTime = t;
        } catch {
            // Ignore and keep last known media time.
        }
        playerClockRef.current = {
            ...playerClockRef.current,
            playerState: state,
            mediaTime,
            wallTimeMs: now,
        };
        setCurrentTime(mediaTime);
    };

    const onPlayerPlaybackRateChange = (event: any) => {
        const rate = Number(event?.data ?? event?.target?.getPlaybackRate?.());
        if (!Number.isFinite(rate) || rate <= 0) return;
        const now = performance.now();
        let mediaTime = playerClockRef.current.mediaTime;
        try {
            const t = Number(event?.target?.getCurrentTime?.());
            if (Number.isFinite(t)) mediaTime = t;
        } catch {
            // Ignore and keep last known media time.
        }
        playerClockRef.current = {
            ...playerClockRef.current,
            playbackRate: rate,
            mediaTime,
            wallTimeMs: now,
        };
        setCurrentTime(mediaTime);
    };

    const handleSeek = (time: number) => {
        if (!player) return;
        try {
            if (typeof player.seekTo === 'function') {
                player.seekTo(time, true);
            }
            if (typeof player.playVideo === 'function') {
                player.playVideo();
            }
            const rate = Number(player?.getPlaybackRate?.());
            playerClockRef.current = {
                ...playerClockRef.current,
                mediaTime: time,
                wallTimeMs: performance.now(),
                playerState: 1,
                playbackRate: Number.isFinite(rate) && rate > 0 ? rate : playerClockRef.current.playbackRate,
            };
            setCurrentTime(time);
        } catch (e) {
            console.warn('Failed to seek video player', e);
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

    const parseSegmentWords = (seg: TranscriptSegment): Array<{ start: number; end: number; word: string }> => {
        if (!seg.words) return [];
        let words = [] as Array<{ start: number; end: number; word: string }>;
        try {
            const parsed = JSON.parse(seg.words);
            if (!Array.isArray(parsed)) return [];
            words = parsed
                .map((w: any) => ({
                    start: Number(w?.start),
                    end: Number(w?.end),
                    word: String(w?.word || '').trim(),
                }))
                .filter((w) => Number.isFinite(w.start) && Number.isFinite(w.end) && w.end > w.start && !!w.word);
        } catch {
            return [];
        }
        if (words.length === 0) return [];

        words.sort((a, b) => a.start - b.start);
        const segStart = Number(seg.start_time);
        const segEnd = Number(seg.end_time);
        const segDuration = Number.isFinite(segStart) && Number.isFinite(segEnd) ? Math.max(0.01, segEnd - segStart) : 0.01;

        let minStart = Math.min(...words.map(w => w.start));
        let maxEnd = Math.max(...words.map(w => w.end));

        const looksMsAbsolute = Number.isFinite(segEnd) && maxEnd > Math.max(segEnd * 5, 1000);
        const looksMsRelative =
            minStart >= -0.5 &&
            minStart < Math.max(2, segDuration * 2) &&
            maxEnd > Math.max(1000, segDuration * 20);

        if (looksMsAbsolute || looksMsRelative) {
            words = words.map(w => ({ ...w, start: w.start / 1000, end: w.end / 1000 }));
            minStart = Math.min(...words.map(w => w.start));
            maxEnd = Math.max(...words.map(w => w.end));
        }

        const looksRelative = minStart >= -0.5 && maxEnd <= segDuration + 1.5;
        if (looksRelative && Number.isFinite(segStart)) {
            words = words.map(w => ({ ...w, start: w.start + segStart, end: w.end + segStart }));
        }

        if (Number.isFinite(segStart) && Number.isFinite(segEnd) && segEnd > segStart) {
            words = words
                .map(w => {
                    const s = Math.max(segStart, w.start);
                    const e = Math.min(segEnd, Math.max(w.end, s + 0.001));
                    return { ...w, start: s, end: e };
                })
                .filter(w => w.end > w.start + 0.0005);
        }

        words.sort((a, b) => a.start - b.start);
        return words;
    };

    const normalizedWordsBySegmentId = useMemo(() => {
        const map = new Map<number, Array<{ start: number; end: number; word: string }>>();
        for (const seg of segments) {
            if (typeof seg.id !== 'number') continue;
            map.set(seg.id, parseSegmentWords(seg));
        }
        return map;
    }, [segments]);

    const scrollTranscriptToTime = (time: number) => {
        if (segments.length === 0) return;

        const target =
            segments.find(s => time >= s.start_time && time < s.end_time) ||
            segments.find(s => s.start_time >= time) ||
            segments[segments.length - 1];

        if (!target) return;

        const doScroll = () => {
            const el = document.getElementById(`seg-${target.id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };

        if (activeTab !== 'transcript') {
            setActiveTab('transcript');
            window.setTimeout(doScroll, 80);
        } else {
            doScroll();
        }
    };

    const handleFunnyMomentJump = (moment: FunnyMoment) => {
        handleSeek(moment.start_time);
        scrollTranscriptToTime(moment.start_time);
    };

    const getDisplayHumorSummary = (raw?: string) => {
        if (!raw) return raw;
        let text = raw.trim();

        // Strip inline reasoning/thinking wrappers if a model leaked them into content.
        text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
        text = text.replace(/^\s*```(?:thinking|reasoning)\s*[\s\S]*?```\s*/i, '').trim();

        // Strip markdown code fences if the LLM returned fenced JSON/prose.
        text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

        // If JSON (or JSON-ish) leaked into the cached summary, extract "summary".
        try {
            const parsed = JSON.parse(text);
            if (parsed && typeof parsed === 'object' && typeof (parsed as any).summary === 'string') {
                return (parsed as any).summary.trim();
            }
        } catch {
            const m = text.match(/"summary"\s*:\s*"((?:\\.|[^"\\])*)"/i);
            if (m?.[1]) {
                try {
                    return JSON.parse(`"${m[1]}"`).trim();
                } catch {
                    return m[1];
                }
            }
        }

        // Heuristic cleanup for reasoning-style preambles (e.g. "The user wants me to...").
        const lowered = text.toLowerCase();
        if (
            lowered.startsWith('the user wants me to') ||
            lowered.startsWith('the user asked me to') ||
            lowered.startsWith('first, i need to') ||
            lowered.startsWith('i need to analyze') ||
            lowered.startsWith('let me analyze')
        ) {
            const cues = [
                /\blikely joke\b/i,
                /\bthe joke likely\b/i,
                /\bthis laugh is likely\b/i,
                /\bthe humor is likely\b/i,
                /\bsummary\s*:/i,
                /\bmost likely\b/i,
            ];
            let cueIndex = -1;
            for (const cue of cues) {
                const m = cue.exec(text);
                if (m && m.index > 40 && (cueIndex === -1 || m.index < cueIndex)) {
                    cueIndex = m.index;
                }
            }
            if (cueIndex > 0) {
                text = text.slice(cueIndex).replace(/^[:\-\s]+/, '').trim();
            }
        }

        return text;
    };

    const pauseMainPreview = () => {
        try {
            if (player && typeof player.pauseVideo === 'function') {
                player.pauseVideo();
            }
        } catch (e) {
            console.warn('Failed to pause main preview player', e);
        }
    };

    // Auto-scroll transcript
    useEffect(() => {
        if (activeTab === 'transcript' && followPlayback && segments.length > 0 && !selection && !searchQuery && !editingSegmentId) {
            // Don't auto-scroll while selecting text or searching, it's annoying
            const activeSeg = segments.find(s => currentTime >= s.start_time && currentTime < s.end_time);
            if (activeSeg) {
                if (lastAutoScrollSegIdRef.current !== activeSeg.id) {
                    lastAutoScrollSegIdRef.current = activeSeg.id;
                    const activeEl = document.getElementById(`seg-${activeSeg.id}`);
                    if (activeEl) activeEl.scrollIntoView({ behavior: 'auto', block: 'center' });
                }
            }
        }
    }, [currentTime, activeTab, followPlayback, segments, selection, searchQuery, editingSegmentId]);

    // Search: filter segments and compute matches
    const searchLower = searchQuery.toLowerCase().trim();
    const filteredSegments = useMemo(() => (
        searchLower
            ? segments.filter(seg => seg.text.toLowerCase().includes(searchLower))
            : segments
    ), [segments, searchLower]);
    const totalMatches = filteredSegments.length;

    // Navigate between search results
    useEffect(() => {
        if (searchLower && filteredSegments.length > 0 && searchMatchIndex < filteredSegments.length) {
            const seg = filteredSegments[searchMatchIndex];
            const el = document.getElementById(`seg-${seg.id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [searchMatchIndex, searchQuery]);

    // Reset match index when query changes
    useEffect(() => {
        setSearchMatchIndex(0);
    }, [searchQuery]);

    // Highlight helper: wraps matched text in a <mark>
    const highlightText = (text: string) => {
        if (!searchLower) return text;
        const idx = text.toLowerCase().indexOf(searchLower);
        if (idx === -1) return text;
        const parts: (string | React.ReactNode)[] = [];
        let lastIdx = 0;
        let i = text.toLowerCase().indexOf(searchLower, lastIdx);
        while (i !== -1) {
            if (i > lastIdx) parts.push(text.slice(lastIdx, i));
            parts.push(<mark key={i} className="bg-yellow-200 text-yellow-900 rounded px-0.5">{text.slice(i, i + searchLower.length)}</mark>);
            lastIdx = i + searchLower.length;
            i = text.toLowerCase().indexOf(searchLower, lastIdx);
        }
        if (lastIdx < text.length) parts.push(text.slice(lastIdx));
        return <>{parts}</>;
    };

    const handleMouseUp = () => {
        if (activeTab !== 'transcript') return;

        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) {
            return;
        }

        // Helper to find segment div from text node
        const getSegmentDiv = (node: Node | null): HTMLElement | null => {
            let curr: any = node;
            while (curr && curr !== transcriptRef.current) {
                if (curr.dataset && curr.dataset.start) return curr;
                curr = curr.parentNode;
            }
            return null;
        };

        const startEl = getSegmentDiv(sel.anchorNode);
        const endEl = getSegmentDiv(sel.focusNode);

        if (startEl && endEl) {
            const t1 = parseFloat(startEl.dataset.start!);
            const t2 = parseFloat(endEl.dataset.end!);
            // Handle reverse selection (drag bottom-to-top)
            const start = Math.min(t1, parseFloat(endEl.dataset.start!));
            const end = Math.max(t2, parseFloat(startEl.dataset.end!));

            // Clean text
            let text = sel.toString().replace(/\s+/g, ' ').trim();
            if (text.length > 60) text = text.substring(0, 60) + '...';

            setSelection({ start, end, defaultTitle: text });
            setClipTitle(text);
        }
    };

    const handleCreateClip = async () => {
        if (!selection || !video) return;
        setCreatingClip(true);
        try {
            const res = await api.post<Clip>(`/videos/${video.id}/clips`, {
                start_time: selection.start,
                end_time: selection.end,
                title: clipTitle || selection.defaultTitle
            }); // Omit 'id' and 'created_at' as backend handles them, cast not strictly needed if shape matches

            const createdClip: Clip = {
                ...res.data,
                aspect_ratio: res.data.aspect_ratio || 'source',
                fade_in_sec: res.data.fade_in_sec ?? 0,
                fade_out_sec: res.data.fade_out_sec ?? 0,
                burn_captions: res.data.burn_captions ?? false,
                caption_speaker_labels: res.data.caption_speaker_labels ?? true,
            };
            setClips(prev => {
                const withoutDuplicate = prev.filter(c => c.id !== createdClip.id);
                return [...withoutDuplicate, createdClip].sort((a, b) => a.start_time - b.start_time);
            });
            setActiveTab('clips');
            setSelection(null);
            void fetchClips();
        } catch (e) {
            console.error(e);
            alert((e as any)?.response?.data?.detail || "Failed to save clip");
        } finally {
            setCreatingClip(false);
        }
    };

    const fetchClips = async () => {
        if (!video) return;
        setLoadingClips(true);
        try {
            const res = await api.get<Clip[]>(`/videos/${video.id}/clips`);
            setClips(res.data.map(c => ({
                ...c,
                aspect_ratio: c.aspect_ratio || 'source',
                fade_in_sec: c.fade_in_sec ?? 0,
                fade_out_sec: c.fade_out_sec ?? 0,
                burn_captions: c.burn_captions ?? false,
                caption_speaker_labels: c.caption_speaker_labels ?? true,
            })));
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingClips(false);
        }
    };

    const fetchClipExportArtifacts = async () => {
        if (!video) return;
        try {
            const res = await api.get<ClipExportArtifact[]>(`/videos/${video.id}/clip-exports`);
            const grouped: Record<number, ClipExportArtifact[]> = {};
            for (const row of (res.data || [])) {
                const cid = Number(row.clip_id);
                if (!grouped[cid]) grouped[cid] = [];
                grouped[cid].push(row);
            }
            setClipExportArtifactsByClip(grouped);
        } catch (e) {
            console.error('Failed to fetch clip export artifacts:', e);
            setClipExportArtifactsByClip({});
        }
    };

    const handleDeleteClip = async (clipId: number) => {
        if (!confirm("Are you sure you want to delete this clip?")) return;
        try {
            await api.delete(`/clips/${clipId}`);
            setClips(prev => prev.filter(c => c.id !== clipId));
        } catch (e) {
            console.error(e);
            alert("Failed to delete clip");
        }
    };

    const handleSpeakerClick = async (speakerId: number, segment?: TranscriptSegment) => {
        pauseMainPreview();

        if (segment && video) {
            const sample = {
                youtube_id: video.youtube_id,
                video_id: video.id,
                start_time: segment.start_time,
                text: segment.text
            };
            setInitialSample(sample);
        } else {
            setInitialSample(null);
        }

        try {
            const res = await api.get<Speaker>(`/speakers/${speakerId}`);
            setSelectedSpeaker(res.data);
        } catch (e) {
            console.error("Failed to fetch speaker details", e);
        }
    };

    const [purging, setPurging] = useState(false);
    const [redoing, setRedoing] = useState(false);

    const handlePurgeTranscript = async () => {
        if (!video) return;
        if (!confirm('Purge all transcript and diarization data for this video? This cannot be undone.')) return;
        setPurging(true);
        try {
            await api.post(`/videos/${video.id}/purge`);
            setSegments([]);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to purge');
        } finally {
            setPurging(false);
        }
    };

    const handleRedoTranscript = async () => {
        if (!video) return;
        if (!confirm('Re-run transcription for this video? This will also re-run diarization after transcription completes.')) return;
        setRedoing(true);
        try {
            await api.post(`/videos/${video.id}/redo-transcription`);
            setSegments([]);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to redo transcription');
        } finally {
            setRedoing(false);
        }
    };

    const [redoingDiarization, setRedoingDiarization] = useState(false);

    const handleRedoDiarization = async () => {
        if (!video) return;
        if (!confirm('Re-run diarization using improved speaker profiles? Existing transcript segments will be re-split and re-assigned.')) return;
        setRedoingDiarization(true);
        try {
            await api.post(`/videos/${video.id}/redo-diarization`);
            setSegments([]);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to redo diarization');
        } finally {
            setRedoingDiarization(false);
        }
    };

    const handleUnknownSpeakerClick = async (segmentId: number, e: React.MouseEvent) => {
        e.stopPropagation();
        pauseMainPreview();
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        setAssignPopup({
            segmentId,
            x: rect.left + window.scrollX,
            y: rect.bottom + window.scrollY
        });

        // Fetch speakers for assignment
        setAssignLoading(true);
        try {
            // Get speakers from this channel
            if (!video) return;
            const res = await api.get<Speaker[]>('/speakers', { params: { channel_id: video.channel_id } });
            setAssignSpeakers(res.data);
        } catch (e) {
            console.error("Failed to fetch speakers", e);
        } finally {
            setAssignLoading(false);
        }
    };

    const handleAssignSpeaker = async (speakerId: number) => {
        if (!assignPopup) return;
        try {
            await api.patch(`/segments/${assignPopup.segmentId}/assign-speaker`, { speaker_id: speakerId });
            setAssignPopup(null);
            // Refresh segments
            fetchData();
        } catch (e) {
            console.error("Failed to assign speaker", e);
            alert("Failed to assign speaker");
        }
    };

    const handleSplitSegmentProfile = async (seg: TranscriptSegment) => {
        if (!seg.speaker_id) return;

        const defaultName = seg.speaker ? `${seg.speaker} - split` : 'New Speaker';
        const enteredName = window.prompt(
            'Create a standalone speaker from this segment profile.\n\nEnter new speaker name (blank = auto-name):',
            defaultName
        );
        if (enteredName === null) return;

        const payload: any = {};
        const trimmed = enteredName.trim();
        if (trimmed) payload.new_speaker_name = trimmed;

        setSplittingSegmentId(seg.id);
        try {
            const res = await api.post(`/segments/${seg.id}/split-profile`, payload);
            const targetSpeakerId = Number(res.data?.segment_speaker_id ?? res.data?.target_speaker_id);
            const targetSpeakerName = String(res.data?.target_speaker_name || seg.speaker || 'Speaker');
            const matchedProfileId = Number(res.data?.segment_matched_profile_id ?? res.data?.profile_id);

            setSegments(prev => prev.map(s => (
                s.id === seg.id
                    ? {
                        ...s,
                        speaker_id: Number.isFinite(targetSpeakerId) ? targetSpeakerId : s.speaker_id,
                        speaker: targetSpeakerName,
                        matched_profile_id: Number.isFinite(matchedProfileId) ? matchedProfileId : s.matched_profile_id,
                    }
                    : s
            )));

            let msg = `Created standalone speaker "${targetSpeakerName}" from this segment.`;
            if (Number.isFinite(matchedProfileId) && matchedProfileId > 0) {
                const doBulk = confirm(
                    `Created "${targetSpeakerName}".\n\nReassign all diarization segments already linked to this profile across the channel now?`
                );
                if (doBulk) {
                    const bulkRes = await api.post(`/profiles/${matchedProfileId}/reassign-segments`);
                    const updated = Number(bulkRes.data?.updated_segments ?? 0);
                    const total = Number(bulkRes.data?.matched_segments ?? 0);
                    msg += `\n\nBulk reassignment complete: ${updated} of ${total} matched segments reassigned.`;
                }
            }
            alert(msg);
        } catch (e: any) {
            console.error('Failed to split segment profile', e);
            alert(e?.response?.data?.detail || 'Failed to split this segment into a standalone profile');
        } finally {
            setSplittingSegmentId(null);
        }
    };

    const beginSegmentEdit = (seg: TranscriptSegment) => {
        pauseMainPreview();
        setEditingSegmentId(seg.id);
        const baseWords = parseSegmentWords(seg).map(w => w.word.trim()).filter(Boolean);
        const fallbackWords = String(seg.text || '').split(/\s+/).map(w => w.trim()).filter(Boolean);
        const tokens = baseWords.length > 0 ? baseWords : fallbackWords;
        setEditingSegmentWords(tokens);
        setEditingLoopSegment(false);
    };

    const updateEditingWord = (index: number, value: string) => {
        setEditingSegmentWords(prev => {
            const next = [...prev];
            next[index] = value;
            return next;
        });
    };

    const removeEditingWord = (index: number) => {
        setEditingSegmentWords(prev => prev.filter((_, i) => i !== index));
    };

    const addEditingWord = () => {
        setEditingSegmentWords(prev => [...prev, '']);
    };

    const saveSegmentEdit = async (segmentId: number) => {
        const words = editingSegmentWords.map(w => w.trim()).filter(Boolean);
        const text = words.join(' ').trim();
        if (!text) {
            alert('Transcript text cannot be empty');
            return;
        }
        setSavingSegmentEdit(true);
        try {
            const res = await api.patch<TranscriptSegment>(`/segments/${segmentId}/text`, { text, words });
            setSegments(prev => prev.map(s => (
                s.id === segmentId
                    ? { ...s, text: res.data.text, words: res.data.words ?? s.words }
                    : s
            )));
            setEditingLoopSegment(false);
            pauseMainPreview();
            setEditingSegmentId(null);
            setEditingSegmentWords([]);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save transcript correction');
        } finally {
            setSavingSegmentEdit(false);
        }
    };

    const toggleClipSelected = (clipId: number) => {
        setSelectedClipIds(prev => {
            const next = new Set(prev);
            if (next.has(clipId)) next.delete(clipId); else next.add(clipId);
            return next;
        });
    };

    const startClipEdit = (clip: Clip) => {
        const nextDraft: Partial<Clip> = {
            ...clip,
            aspect_ratio: clip.aspect_ratio || 'source',
            portrait_split_enabled: clip.portrait_split_enabled ?? false,
            fade_in_sec: clip.fade_in_sec ?? 0,
            fade_out_sec: clip.fade_out_sec ?? 0,
            burn_captions: clip.burn_captions ?? false,
            caption_speaker_labels: clip.caption_speaker_labels ?? true,
        };
        setEditingClipId(clip.id);
        setClipEditorDraft(nextDraft);
        setClipEditorCropTarget('main');
        setClipEditorDragRect(null);
        hydrateClipEditorText(nextDraft.start_time ?? clip.start_time, nextDraft.end_time ?? clip.end_time, nextDraft.script_edits_json);
    };

    const cancelClipEdit = () => {
        setEditingClipId(null);
        setClipEditorDraft(null);
        setClipEditorTokens([]);
        setClipEditorRemovedWordKeys(new Set());
        setClipEditorCropTarget('main');
        setClipEditorDragRect(null);
        setClipTimelineDrag(null);
    };

    const updateClipDraftField = (field: keyof Clip, value: any) => {
        setClipEditorDraft(prev => ({ ...(prev || {}), [field]: value }));
    };

    const parseClipEditorKeptRanges = (raw?: string | null): Array<[number, number]> => {
        if (!raw) return [];
        try {
            const parsed = JSON.parse(raw);
            if (!parsed || !Array.isArray(parsed.kept_ranges)) return [];
            const ranges: Array<[number, number]> = [];
            for (const r of parsed.kept_ranges) {
                if (!Array.isArray(r) || r.length < 2) continue;
                const s = Number(r[0]);
                const e = Number(r[1]);
                if (Number.isFinite(s) && Number.isFinite(e) && e > s + 0.01) ranges.push([s, e]);
            }
            ranges.sort((a, b) => a[0] - b[0]);
            return ranges;
        } catch {
            return [];
        }
    };

    const buildClipEditorTokens = (clipStart: number, clipEnd: number) => {
        const out: Array<{ key: string; start: number; end: number; word: string }> = [];
        for (const seg of segments) {
            if (seg.end_time <= clipStart || seg.start_time >= clipEnd) continue;
            let words: Array<{ start: number; end: number; word: string }> = parseSegmentWords(seg);
            if (words.length === 0 && seg.text?.trim()) {
                const textWords = seg.text.trim().split(/\s+/);
                const duration = Math.max(0.05, seg.end_time - seg.start_time);
                const step = duration / Math.max(textWords.length, 1);
                words = textWords.map((w, idx) => ({
                    start: seg.start_time + (idx * step),
                    end: seg.start_time + ((idx + 1) * step),
                    word: w,
                }));
            }
            for (const w of words) {
                if (w.end <= clipStart || w.start >= clipEnd) continue;
                const s = Math.max(clipStart, w.start);
                const e = Math.min(clipEnd, w.end);
                if (e <= s + 0.005) continue;
                out.push({
                    key: `${s.toFixed(3)}-${e.toFixed(3)}-${out.length}`,
                    start: s,
                    end: e,
                    word: w.word,
                });
            }
        }
        out.sort((a, b) => a.start - b.start);
        return out;
    };

    const buildKeptRangesFromTokenState = (
        tokens: Array<{ key: string; start: number; end: number; word: string }>,
        removedKeys: Set<string>,
        clipStart: number,
        clipEnd: number,
    ): Array<[number, number]> => {
        const kept = tokens
            .filter(t => !removedKeys.has(t.key))
            .map(t => [Math.max(clipStart, t.start), Math.min(clipEnd, t.end)] as [number, number])
            .filter(r => r[1] > r[0] + 0.005)
            .sort((a, b) => a[0] - b[0]);
        if (kept.length === 0) return [];
        const merged: Array<[number, number]> = [kept[0]];
        for (let i = 1; i < kept.length; i++) {
            const [s, e] = kept[i];
            const last = merged[merged.length - 1];
            if (s <= last[1] + 0.22) {
                last[1] = Math.max(last[1], e);
            } else {
                merged.push([s, e]);
            }
        }
        return merged;
    };

    const hydrateClipEditorText = (clipStart: number, clipEnd: number, scriptEditsJson?: string | null) => {
        const tokens = buildClipEditorTokens(clipStart, clipEnd);
        const keptRanges = parseClipEditorKeptRanges(scriptEditsJson);
        const removed = new Set<string>();
        if (keptRanges.length > 0) {
            for (const t of tokens) {
                const mid = (t.start + t.end) / 2;
                const inKept = keptRanges.some(([s, e]) => mid >= s && mid <= e);
                if (!inKept) removed.add(t.key);
            }
        }
        setClipEditorTokens(tokens);
        setClipEditorRemovedWordKeys(removed);
    };

    const rebuildClipEditorTextWindow = () => {
        if (!clipEditorDraft) return;
        const clipStart = Number(clipEditorDraft.start_time ?? 0);
        const clipEnd = Number(clipEditorDraft.end_time ?? 0);
        if (!Number.isFinite(clipStart) || !Number.isFinite(clipEnd) || clipEnd <= clipStart) {
            alert('Set a valid start/end first, then refresh transcript window.');
            return;
        }
        hydrateClipEditorText(clipStart, clipEnd, clipEditorDraft.script_edits_json);
    };

    const persistClipEditorRemovedWords = (nextRemoved: Set<string>) => {
        if (!clipEditorDraft) return;
        const clipStart = Number(clipEditorDraft.start_time ?? 0);
        const clipEnd = Number(clipEditorDraft.end_time ?? 0);
        const keptRanges = buildKeptRangesFromTokenState(clipEditorTokens, nextRemoved, clipStart, clipEnd);
        if (nextRemoved.size > 0 && keptRanges.length === 0) {
            alert('Cannot remove every word from the clip. Keep at least one word.');
            return;
        }
        setClipEditorRemovedWordKeys(nextRemoved);
        if (nextRemoved.size === 0) {
            updateClipDraftField('script_edits_json', null);
            return;
        }
        updateClipDraftField('script_edits_json', JSON.stringify({
            version: 1,
            mode: 'keep_ranges',
            source: 'text_editor',
            kept_ranges: keptRanges,
            removed_word_count: nextRemoved.size,
            total_word_count: clipEditorTokens.length,
            updated_at: new Date().toISOString(),
        }));
    };

    const toggleClipEditorWord = (tokenKey: string) => {
        const next = new Set(clipEditorRemovedWordKeys);
        if (next.has(tokenKey)) next.delete(tokenKey); else next.add(tokenKey);
        persistClipEditorRemovedWords(next);
    };

    const restoreAllClipEditorWords = () => {
        persistClipEditorRemovedWords(new Set());
    };

    const autoRemoveClipEditorFillers = () => {
        const filler = new Set(['um', 'uh', 'erm', 'hmm', 'ah', 'like']);
        const next = new Set(clipEditorRemovedWordKeys);
        for (const t of clipEditorTokens) {
            const w = t.word.toLowerCase().replace(/[^\w']/g, '');
            if (filler.has(w)) next.add(t.key);
        }
        persistClipEditorRemovedWords(next);
    };

    const clampNorm = (val: number) => Math.max(0, Math.min(1, val));

    const normalizeCropRect = (
        x?: number | null,
        y?: number | null,
        w?: number | null,
        h?: number | null,
        fallback?: { x: number; y: number; w: number; h: number },
    ) => {
        if (x == null || y == null || w == null || h == null) {
            return fallback || { x: 0, y: 0, w: 1, h: 1 };
        }
        const nx = clampNorm(Number(x));
        const ny = clampNorm(Number(y));
        const nw = Math.max(0.01, clampNorm(Number(w)));
        const nh = Math.max(0.01, clampNorm(Number(h)));
        return {
            x: nx,
            y: ny,
            w: nx + nw > 1 ? Math.max(0.01, 1 - nx) : nw,
            h: ny + nh > 1 ? Math.max(0.01, 1 - ny) : nh,
        };
    };

    const applyPortraitSplitDefaults = () => {
        setClipEditorDraft(prev => {
            const next: Partial<Clip> = { ...(prev || {}), portrait_split_enabled: true };
            if (next.portrait_top_crop_x == null) next.portrait_top_crop_x = 0;
            if (next.portrait_top_crop_y == null) next.portrait_top_crop_y = 0;
            if (next.portrait_top_crop_w == null) next.portrait_top_crop_w = 1;
            if (next.portrait_top_crop_h == null) next.portrait_top_crop_h = 0.5;
            if (next.portrait_bottom_crop_x == null) next.portrait_bottom_crop_x = 0;
            if (next.portrait_bottom_crop_y == null) next.portrait_bottom_crop_y = 0.5;
            if (next.portrait_bottom_crop_w == null) next.portrait_bottom_crop_w = 1;
            if (next.portrait_bottom_crop_h == null) next.portrait_bottom_crop_h = 0.5;
            return next;
        });
    };

    const getDraftCropRect = (target: 'main' | 'top' | 'bottom') => {
        if (!clipEditorDraft) return { x: 0, y: 0, w: 1, h: 1 };
        if (target === 'top') {
            return normalizeCropRect(
                clipEditorDraft.portrait_top_crop_x,
                clipEditorDraft.portrait_top_crop_y,
                clipEditorDraft.portrait_top_crop_w,
                clipEditorDraft.portrait_top_crop_h,
                { x: 0, y: 0, w: 1, h: 0.5 },
            );
        }
        if (target === 'bottom') {
            return normalizeCropRect(
                clipEditorDraft.portrait_bottom_crop_x,
                clipEditorDraft.portrait_bottom_crop_y,
                clipEditorDraft.portrait_bottom_crop_w,
                clipEditorDraft.portrait_bottom_crop_h,
                { x: 0, y: 0.5, w: 1, h: 0.5 },
            );
        }
        return normalizeCropRect(
            clipEditorDraft.crop_x,
            clipEditorDraft.crop_y,
            clipEditorDraft.crop_w,
            clipEditorDraft.crop_h,
        );
    };

    const setDraftCropRect = (target: 'main' | 'top' | 'bottom', rect: { x: number; y: number; w: number; h: number }) => {
        const x = Number(clampNorm(rect.x).toFixed(4));
        const y = Number(clampNorm(rect.y).toFixed(4));
        const w = Number(Math.max(0.01, Math.min(1 - x, rect.w)).toFixed(4));
        const h = Number(Math.max(0.01, Math.min(1 - y, rect.h)).toFixed(4));
        if (target === 'top') {
            updateClipDraftField('portrait_top_crop_x', x);
            updateClipDraftField('portrait_top_crop_y', y);
            updateClipDraftField('portrait_top_crop_w', w);
            updateClipDraftField('portrait_top_crop_h', h);
            return;
        }
        if (target === 'bottom') {
            updateClipDraftField('portrait_bottom_crop_x', x);
            updateClipDraftField('portrait_bottom_crop_y', y);
            updateClipDraftField('portrait_bottom_crop_w', w);
            updateClipDraftField('portrait_bottom_crop_h', h);
            return;
        }
        updateClipDraftField('crop_x', x);
        updateClipDraftField('crop_y', y);
        updateClipDraftField('crop_w', w);
        updateClipDraftField('crop_h', h);
    };

    const nudgeClipBoundary = (which: 'start' | 'end', deltaSec: number) => {
        if (!clipEditorDraft) return;
        const start = Number(clipEditorDraft.start_time ?? 0);
        const end = Number(clipEditorDraft.end_time ?? 0);
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        if (which === 'start') {
            const next = Math.max(0, Math.min(end - 0.05, start + deltaSec));
            updateClipDraftField('start_time', Number(next.toFixed(3)));
        } else {
            const next = Math.max(start + 0.05, end + deltaSec);
            updateClipDraftField('end_time', Number(next.toFixed(3)));
        }
    };

    const setClipBoundaryFromPlayhead = (which: 'start' | 'end') => {
        if (!clipEditorDraft || !Number.isFinite(currentTime)) return;
        const start = Number(clipEditorDraft.start_time ?? 0);
        const end = Number(clipEditorDraft.end_time ?? 0);
        if (which === 'start') {
            updateClipDraftField('start_time', Number(Math.max(0, Math.min(currentTime, end - 0.05)).toFixed(3)));
        } else {
            updateClipDraftField('end_time', Number(Math.max(start + 0.05, currentTime).toFixed(3)));
        }
    };

    const getEditorMediaDuration = () => {
        const byVideo = Number(video?.duration || 0);
        if (Number.isFinite(byVideo) && byVideo > 1) return byVideo;
        const segMax = segments.reduce((m, s) => Math.max(m, Number(s.end_time || 0)), 0);
        if (Number.isFinite(segMax) && segMax > 1) return segMax;
        const clipEnd = Number(clipEditorDraft?.end_time || 0);
        return Math.max(clipEnd + 1, 60);
    };

    const timelineTimeToPct = (t: number, duration: number) => {
        if (!Number.isFinite(duration) || duration <= 0) return 0;
        return clampNorm(t / duration);
    };

    const timelineEventToTime = (evt: { clientX: number }, duration: number) => {
        if (!clipTimelineRef.current) return 0;
        const bounds = clipTimelineRef.current.getBoundingClientRect();
        if (bounds.width <= 0) return 0;
        const ratio = clampNorm((evt.clientX - bounds.left) / bounds.width);
        return ratio * duration;
    };

    const beginClipTimelineDrag = (handle: 'start' | 'end', e: any) => {
        e.preventDefault();
        e.stopPropagation();
        setClipTimelineDrag({ handle });
    };

    const handleTimelineScrub = (e: any) => {
        if (!clipEditorDraft) return;
        const duration = getEditorMediaDuration();
        const t = timelineEventToTime(e, duration);
        handleSeek(t);
    };

    const handleCropPreviewPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!clipEditorDraft || !cropPreviewRef.current) return;
        if (clipEditorCropTarget !== 'main') {
            const splitEnabled = String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled;
            if (!splitEnabled) return;
        }
        const bounds = cropPreviewRef.current.getBoundingClientRect();
        if (bounds.width <= 0 || bounds.height <= 0) return;
        const nx = clampNorm((e.clientX - bounds.left) / bounds.width);
        const ny = clampNorm((e.clientY - bounds.top) / bounds.height);
        setClipEditorDragRect({
            target: clipEditorCropTarget,
            startX: nx,
            startY: ny,
            currentX: nx,
            currentY: ny,
        });
    };

    const saveClipEdit = async (clipId: number) => {
        if (!clipEditorDraft) return;
        if (!clipEditorDraft.title || !String(clipEditorDraft.title).trim()) {
            alert('Clip title is required');
            return;
        }
        if ((clipEditorDraft.end_time ?? 0) <= (clipEditorDraft.start_time ?? 0)) {
            alert('End time must be after start time');
            return;
        }
        setSavingClipEdit(true);
        try {
            const payload = {
                start_time: Number(clipEditorDraft.start_time),
                end_time: Number(clipEditorDraft.end_time),
                title: String(clipEditorDraft.title),
                aspect_ratio: clipEditorDraft.aspect_ratio || 'source',
                crop_x: clipEditorDraft.crop_x ?? null,
                crop_y: clipEditorDraft.crop_y ?? null,
                crop_w: clipEditorDraft.crop_w ?? null,
                crop_h: clipEditorDraft.crop_h ?? null,
                portrait_split_enabled: !!clipEditorDraft.portrait_split_enabled,
                portrait_top_crop_x: clipEditorDraft.portrait_top_crop_x ?? null,
                portrait_top_crop_y: clipEditorDraft.portrait_top_crop_y ?? null,
                portrait_top_crop_w: clipEditorDraft.portrait_top_crop_w ?? null,
                portrait_top_crop_h: clipEditorDraft.portrait_top_crop_h ?? null,
                portrait_bottom_crop_x: clipEditorDraft.portrait_bottom_crop_x ?? null,
                portrait_bottom_crop_y: clipEditorDraft.portrait_bottom_crop_y ?? null,
                portrait_bottom_crop_w: clipEditorDraft.portrait_bottom_crop_w ?? null,
                portrait_bottom_crop_h: clipEditorDraft.portrait_bottom_crop_h ?? null,
                script_edits_json: clipEditorDraft.script_edits_json ?? null,
                fade_in_sec: Number(clipEditorDraft.fade_in_sec ?? 0),
                fade_out_sec: Number(clipEditorDraft.fade_out_sec ?? 0),
                burn_captions: !!clipEditorDraft.burn_captions,
                caption_speaker_labels: !!clipEditorDraft.caption_speaker_labels,
            };
            const res = await api.patch<Clip>(`/clips/${clipId}`, payload);
            setClips(prev => prev.map(c => c.id === clipId ? res.data : c));
            cancelClipEdit();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save clip edits');
        } finally {
            setSavingClipEdit(false);
        }
    };

    const downloadBlobResponse = (blob: Blob, filename: string) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    };

    const setClipExporting = (clipId: number, exporting: boolean) => {
        setExportingClipIds(prev => {
            const next = new Set(prev);
            if (exporting) next.add(clipId); else next.delete(clipId);
            return next;
        });
    };

    const formatFileSize = (bytes?: number) => {
        if (!bytes || bytes <= 0) return '';
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let idx = 0;
        while (size >= 1024 && idx < units.length - 1) {
            size /= 1024;
            idx += 1;
        }
        return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
    };

    const downloadArchivedArtifact = async (artifact: ClipExportArtifact) => {
        try {
            const response = await api.get(`/clip-exports/${artifact.id}/download`, { responseType: 'blob' });
            downloadBlobResponse(response.data, artifact.file_name || `clip_export_${artifact.id}.${artifact.format || 'bin'}`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to download archived export');
        }
    };

    const setClipUploading = (clipId: number, uploading: boolean) => {
        setUploadingClipIds(prev => {
            const next = new Set(prev);
            if (uploading) next.add(clipId); else next.delete(clipId);
            return next;
        });
    };

    const exportClipMp4 = async (clip: Clip) => {
        setClipExporting(clip.id, true);
        try {
            const response = await api.post(`/clips/${clip.id}/export/mp4`, null, { responseType: 'blob' });
            const ext = 'mp4';
            downloadBlobResponse(response.data, `${(clip.title || `clip_${clip.id}`).replace(/[\\\\/:*?\"<>|]/g, '_')}.${ext}`);
            await fetchClipExportArtifacts();
        } catch (e: any) {
            const detail = e?.response?.data?.detail || 'Failed to export MP4';
            try {
                await api.post(`/clips/${clip.id}/export/mp4/queue`);
                alert(`${detail}\n\nQueued a background render job for "${clip.title || `Clip #${clip.id}`}". You can download it from archived outputs when complete.`);
            } catch {
                alert(detail);
            }
        } finally {
            setClipExporting(clip.id, false);
        }
    };

    const queueClipMp4 = async (clip: Clip) => {
        setClipExporting(clip.id, true);
        try {
            await api.post(`/clips/${clip.id}/export/mp4/queue`);
            alert(`Queued render job for "${clip.title || `Clip #${clip.id}`}". Check Job Queue > Clip Export.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue MP4 export');
        } finally {
            setClipExporting(clip.id, false);
        }
    };

    const exportClipCaptions = async (clip: Clip, format: 'srt' | 'vtt') => {
        setClipExporting(clip.id, true);
        try {
            const response = await api.post(`/clips/${clip.id}/export/captions`, { format }, { responseType: 'blob' });
            downloadBlobResponse(response.data, `${(clip.title || `clip_${clip.id}`).replace(/[\\\\/:*?\"<>|]/g, '_')}.${format}`);
            await fetchClipExportArtifacts();
        } catch (e: any) {
            alert(e?.response?.data?.detail || `Failed to export ${format.toUpperCase()}`);
        } finally {
            setClipExporting(clip.id, false);
        }
    };

    const uploadClipToYoutube = async (clip: Clip) => {
        setClipUploading(clip.id, true);
        try {
            const res = await api.post(`/clips/${clip.id}/youtube/upload`, {
                privacy_status: clipUploadPrivacy,
            });
            const watchUrl = res.data?.uploaded_watch_url;
            const title = res.data?.uploaded_title || clip.title;
            if (watchUrl) {
                if (confirm(`Uploaded "${title}". Open in YouTube Studio/watch page now?`)) {
                    window.open(watchUrl, '_blank', 'noopener,noreferrer');
                }
            } else {
                alert(`Uploaded "${title}" successfully.`);
            }
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to upload clip to YouTube');
        } finally {
            setClipUploading(clip.id, false);
        }
    };

    const CLIP_BATCH_PRESETS: Record<string, { label: string; clipSettings: Partial<Clip>; exportSrt?: boolean; exportVtt?: boolean }> = {
        youtube_landscape: {
            label: 'YouTube Landscape',
            clipSettings: { aspect_ratio: '16:9', burn_captions: false, caption_speaker_labels: true },
            exportSrt: true,
        },
        shorts_vertical: {
            label: 'Shorts Vertical (Burned Captions)',
            clipSettings: { aspect_ratio: '9:16', burn_captions: true, caption_speaker_labels: false },
            exportSrt: true,
        },
        square_captioned: {
            label: 'Square Captioned',
            clipSettings: { aspect_ratio: '1:1', burn_captions: true, caption_speaker_labels: true },
            exportSrt: true,
            exportVtt: true,
        },
        audio_focus: {
            label: 'Podcast Promo (4:5 + Captions)',
            clipSettings: { aspect_ratio: '4:5', burn_captions: true, caption_speaker_labels: true },
            exportSrt: true,
        },
    };

    const applyPresetToClip = async (clipId: number, presetKey: keyof typeof CLIP_BATCH_PRESETS) => {
        const preset = CLIP_BATCH_PRESETS[presetKey];
        const res = await api.post<Clip>(`/clips/${clipId}/apply-export-preset`, preset.clipSettings);
        setClips(prev => prev.map(c => (c.id === clipId ? res.data : c)));
        return res.data;
    };

    const batchExportSelectedClips = async () => {
        const ids = Array.from(selectedClipIds);
        if (ids.length === 0) {
            alert('Select one or more clips first');
            return;
        }
        const preset = CLIP_BATCH_PRESETS[clipBatchPresetKey];
        if (!confirm(`Apply preset "${preset.label}" and export ${ids.length} clip(s)?`)) return;
        setBatchExporting(true);
        try {
            for (const clipId of ids) {
                const updated = await applyPresetToClip(clipId, clipBatchPresetKey);
                await exportClipMp4(updated);
                if (preset.exportSrt) await exportClipCaptions(updated, 'srt');
                if (preset.exportVtt) await exportClipCaptions(updated, 'vtt');
            }
        } finally {
            setBatchExporting(false);
        }
    };

    const queueRenderSelectedClips = async () => {
        const ids = Array.from(selectedClipIds);
        if (ids.length === 0) {
            alert('Select one or more clips first');
            return;
        }
        if (!confirm(`Queue MP4 render jobs for ${ids.length} selected clip(s)?`)) return;
        setBatchQueueingRenders(true);
        try {
            let queued = 0;
            for (const clipId of ids) {
                try {
                    await api.post(`/clips/${clipId}/export/mp4/queue`);
                    queued += 1;
                } catch {
                    // Continue queueing other clips; summarize at end.
                }
            }
            alert(`Queued ${queued}/${ids.length} clip render job(s). Check Job Queue -> Clip Export.`);
        } finally {
            setBatchQueueingRenders(false);
        }
    };

    const batchUploadSelectedClips = async () => {
        const ids = Array.from(selectedClipIds);
        if (ids.length === 0) {
            alert('Select one or more clips first');
            return;
        }
        if (!confirm(`Upload ${ids.length} selected clip(s) to your connected YouTube channel as ${clipUploadPrivacy}?`)) return;
        setBatchUploadingClips(true);
        try {
            const res = await api.post('/clips/youtube/upload-batch', {
                clip_ids: ids,
                privacy_status: clipUploadPrivacy,
            });
            const uploaded = Number(res.data?.uploaded || 0);
            const failed = Number(res.data?.failed || 0);
            alert(`Batch upload finished. Uploaded: ${uploaded}, Failed: ${failed}.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to batch upload clips');
        } finally {
            setBatchUploadingClips(false);
        }
    };

    const toggleClipPreviewLoop = (clip: Clip) => {
        if (clipPreviewLoop?.clipId === clip.id) {
            setClipPreviewLoop(null);
            return;
        }
        setClipPreviewLoop({ start: clip.start_time, end: clip.end_time, clipId: clip.id });
        handleSeek(clip.start_time);
    };

    const handleDetectFunnyMoments = async (force = true) => {
        if (!id) return;
        setDetectingFunnyMoments(true);
        setFunnyTaskProgress(prev => ({
            video_id: Number(id),
            status: 'running',
            task: 'detect',
            stage: prev?.stage ?? 'loading',
            message: 'Starting funny-moment scan...',
            percent: 1,
            current: null,
            total: null,
        }));
        try {
            const res = await api.post<FunnyMoment[]>(`/videos/${id}/funny-moments/detect`, null, {
                params: { force }
            });
            setFunnyMoments(res.data);
        } catch (e: any) {
            console.error('Failed to detect funny moments', e);
            alert(e?.response?.data?.detail || 'Failed to detect funny moments');
        } finally {
            void fetchFunnyTaskProgress();
            setDetectingFunnyMoments(false);
        }
    };

    const handleExplainFunnyMoments = async (force = false) => {
        if (!id) return;
        setExplainingFunnyMoments(true);
        setFunnyTaskProgress(prev => ({
            video_id: Number(id),
            status: 'running',
            task: 'explain',
            stage: prev?.stage ?? 'loading',
            message: force ? 'Starting re-explain...' : 'Starting explain...',
            percent: 1,
            current: 0,
            total: null,
        }));
        try {
            const res = await api.post<FunnyMoment[]>(`/videos/${id}/funny-moments/explain`, null, {
                params: { force }
            });
            setFunnyMoments(res.data);
            await fetchVideoMeta();
        } catch (e: any) {
            console.error('Failed to explain funny moments', e);
            alert(e?.response?.data?.detail || 'Failed to generate AI explanations');
        } finally {
            void fetchFunnyTaskProgress();
            setExplainingFunnyMoments(false);
        }
    };

    const handleGenerateYoutubeAi = async (force = false) => {
        if (!id) return;
        setGeneratingYoutubeAi(true);
        try {
            const res = await api.post<Video>(`/videos/${id}/youtube-ai/generate`, null, { params: { force } });
            setVideo(res.data);
            setActiveTab('youtube');
        } catch (e: any) {
            console.error('Failed to generate YouTube metadata', e);
            alert(e?.response?.data?.detail || 'Failed to generate YouTube summary/chapters');
        } finally {
            setGeneratingYoutubeAi(false);
        }
    };

    const parseYoutubeAiChapters = (chaptersJson?: string): VideoChapterSuggestion[] => {
        if (!chaptersJson) return [];
        try {
            const parsed = JSON.parse(chaptersJson);
            if (!Array.isArray(parsed)) return [];
            return parsed.filter(Boolean).map((ch: any) => ({
                start_seconds: Number(ch.start_seconds ?? 0),
                timestamp: String(ch.timestamp ?? '0:00'),
                title: String(ch.title ?? '').trim(),
                description: ch.description ? String(ch.description) : undefined,
            })).filter((ch: VideoChapterSuggestion) => ch.title);
        } catch {
            return [];
        }
    };

    const copyToClipboard = async (text: string, kind: 'summary' | 'chapters' | 'description') => {
        try {
            await navigator.clipboard.writeText(text);
            setCopiedYoutubeField(kind);
            window.setTimeout(() => setCopiedYoutubeField(prev => (prev === kind ? null : prev)), 1500);
        } catch {
            alert('Failed to copy to clipboard');
        }
    };

    const fetchDescriptionHistory = async () => {
        if (!id) return;
        setLoadingDescriptionHistory(true);
        try {
            const res = await api.get<VideoDescriptionRevision[]>(`/videos/${id}/description-history`);
            setDescriptionHistory(res.data);
        } catch (e) {
            console.error('Failed to fetch description history', e);
        } finally {
            setLoadingDescriptionHistory(false);
        }
    };

    const handlePublishYoutubeDescription = async () => {
        if (!id || !video?.youtube_ai_description_text) return;
        if (!confirm('Archive the current description and replace it with the AI-generated YouTube description draft?')) return;
        setPublishingYoutubeDescription(true);
        try {
            const res = await api.post<Video>(`/videos/${id}/youtube-ai/publish-description`);
            setVideo(res.data);
            await fetchDescriptionHistory();
        } catch (e: any) {
            console.error('Failed to publish AI description', e);
            alert(e?.response?.data?.detail || 'Failed to publish AI description');
        } finally {
            setPublishingYoutubeDescription(false);
        }
    };

    const handleRestoreDescriptionRevision = async (revision: VideoDescriptionRevision) => {
        if (!id || !revision?.id) return;
        if (!confirm(`Restore description from ${new Date(revision.created_at).toLocaleString()} (${revision.source})? The current description will be archived first.`)) return;
        setRestoringDescriptionRevisionId(revision.id);
        try {
            const res = await api.post<Video>(`/videos/${id}/description-history/${revision.id}/restore`);
            setVideo(res.data);
            await fetchDescriptionHistory();
        } catch (e: any) {
            console.error('Failed to restore description', e);
            alert(e?.response?.data?.detail || 'Failed to restore description');
        } finally {
            setRestoringDescriptionRevisionId(null);
        }
    };

    useEffect(() => {
        if (!id) return;
        if (!detectingFunnyMoments && !explainingFunnyMoments) return;
        const timer = window.setInterval(() => {
            void fetchFunnyTaskProgress();
        }, 700);
        void fetchFunnyTaskProgress();
        return () => window.clearInterval(timer);
    }, [id, detectingFunnyMoments, explainingFunnyMoments]);

    useEffect(() => {
        if (activeTab === 'clips' && video) {
            fetchClips();
            void fetchClipExportArtifacts();
        }
    }, [activeTab, video?.id]);

    useEffect(() => {
        if (activeTab === 'youtube' && id) {
            void fetchDescriptionHistory();
        }
    }, [activeTab, id]);
    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-slate-50">
                <Loader2 className="animate-spin text-slate-400" size={32} />
            </div>
        );
    }

    if (!video) {
        return (
            <div className="flex items-center justify-center h-screen bg-slate-50">
                <div className="text-center">
                    <h3 className="text-xl font-semibold text-slate-700">Video not found</h3>
                    <button onClick={() => navigate(-1)} className="mt-4 text-blue-600 hover:underline">
                        Go Back
                    </button>
                </div>
            </div>
        );
    }

    const explainedFunnyMoments = funnyMoments.filter(m => !!m.humor_summary);
    const youtubeAiChapters = parseYoutubeAiChapters(video.youtube_ai_chapters_json);
    const hasYoutubeAiMetadata = !!(video.youtube_ai_summary || video.youtube_ai_description_text || youtubeAiChapters.length);
    const explainedModelNames = Array.from(new Set(
        explainedFunnyMoments
            .map(m => (m.humor_model || '').trim())
            .filter(Boolean)
    ));
    const latestFunnyExplainAt = explainedFunnyMoments
        .map(m => m.humor_explained_at ? new Date(m.humor_explained_at).getTime() : 0)
        .filter(ts => Number.isFinite(ts) && ts > 0)
        .reduce((max, ts) => Math.max(max, ts), 0);
    const funnyExplainHeaderModelLabel =
        explainedModelNames.length === 0
            ? null
            : explainedModelNames.length === 1
                ? explainedModelNames[0]
                : `Mixed models (${explainedModelNames.length})`;
    const hasExistingFunnyExplanations = explainedFunnyMoments.length > 0 || !!video?.humor_context_summary;
    const funnyTaskIsRunning = !!(detectingFunnyMoments || explainingFunnyMoments);
    const funnyTaskPercent =
        funnyTaskIsRunning && typeof funnyTaskProgress?.percent === 'number'
            ? Math.max(0, Math.min(100, funnyTaskProgress.percent))
            : null;
    const funnyTaskCurrent = typeof funnyTaskProgress?.current === 'number' ? funnyTaskProgress.current : null;
    const funnyTaskTotal = typeof funnyTaskProgress?.total === 'number' ? funnyTaskProgress.total : null;
    const funnyDrawerTaskLabel = detectingFunnyMoments
        ? (funnyTaskProgress?.message || 'Scanning transcript/audio for laughter and funny moments...')
        : explainingFunnyMoments
            ? (funnyTaskProgress?.message || (hasExistingFunnyExplanations
                ? 'Re-generating global humor context and joke explanations...'
                : 'Generating global humor context and joke explanations...'))
            : null;
    return (
        <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-slate-50">
            {/* Left Column: Tools */}
            <div className="w-[450px] flex flex-col bg-white border-r border-slate-200 shrink-0 shadow-xl z-10 transition-all relative">
                {/* Tabs */}
                <div className="flex border-b border-slate-100 bg-white sticky top-0 z-10">
                    <button
                        onClick={() => setActiveTab('transcript')}
                        className={`py-3 px-3 text-sm font-medium flex items-center justify-center gap-2 border-b-2 transition-colors ${activeTab === 'transcript' ? 'border-blue-500 text-blue-600' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'} flex-[1.1]`}
                    >
                        <FileText size={16} /> Transcript
                    </button>
                    <button
                        onClick={() => setActiveTab('clips')}
                        className={`py-3 px-3 text-sm font-medium flex items-center justify-center gap-2 border-b-2 transition-colors ${activeTab === 'clips' ? 'border-purple-500 text-purple-600' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'} flex-1`}
                    >
                        <Scissors size={16} /> Clips
                    </button>
                    <button
                        onClick={() => setActiveTab('speakers')}
                        className={`py-3 px-3 text-sm font-medium flex items-center justify-center gap-2 border-b-2 transition-colors ${activeTab === 'speakers' ? 'border-orange-500 text-orange-600' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'} flex-1`}
                    >
                        <Users size={16} /> Speakers
                    </button>
                    <button
                        onClick={() => setActiveTab('youtube')}
                        className={`py-3 px-3 text-sm font-medium flex items-center justify-center gap-2 border-b-2 transition-colors ${activeTab === 'youtube' ? 'border-emerald-500 text-emerald-600' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'} flex-1`}
                        title="AI-generated YouTube summary and chapters"
                    >
                        <Bot size={16} /> YouTube
                    </button>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-hidden relative bg-slate-50/50">
                    {activeTab === 'transcript' && (
                        <div className="h-full flex flex-col">
                            {/* Search Bar */}
                            {segments.length > 0 && (
                                <div className="p-2 border-b border-slate-100 bg-white/80 backdrop-blur-sm shrink-0">
                                    <div className="relative flex items-center">
                                        <Search size={14} className="absolute left-2.5 text-slate-400" />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search transcript..."
                                            className="w-full pl-8 pr-24 py-1.5 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors"
                                        />
                                        {searchQuery && (
                                            <div className="absolute right-1 flex items-center gap-0.5">
                                                <span className="text-[10px] text-slate-400 font-mono mr-1">
                                                    {totalMatches > 0 ? `${searchMatchIndex + 1}/${totalMatches}` : '0'}
                                                </span>
                                                <button
                                                    onClick={() => setSearchMatchIndex(prev => (prev - 1 + totalMatches) % totalMatches)}
                                                    disabled={totalMatches === 0}
                                                    className="p-0.5 text-slate-400 hover:text-slate-600 disabled:opacity-30 rounded"
                                                ><ChevronUp size={14} /></button>
                                                <button
                                                    onClick={() => setSearchMatchIndex(prev => (prev + 1) % totalMatches)}
                                                    disabled={totalMatches === 0}
                                                    className="p-0.5 text-slate-400 hover:text-slate-600 disabled:opacity-30 rounded"
                                                ><ChevronDown size={14} /></button>
                                                <button
                                                    onClick={() => setSearchQuery('')}
                                                    className="p-0.5 text-slate-400 hover:text-slate-600 rounded"
                                                ><X size={14} /></button>
                                            </div>
                                        )}
                                    </div>
                                    <div className="mt-2 flex items-center justify-end">
                                        <label className="inline-flex items-center gap-2.5 text-xs text-slate-600 select-none cursor-pointer">
                                            <span className="font-medium">Follow playback</span>
                                            <span className="relative inline-flex items-center">
                                                <input
                                                    type="checkbox"
                                                    checked={followPlayback}
                                                    onChange={(e) => setFollowPlayback(e.target.checked)}
                                                    className="sr-only peer"
                                                />
                                                <span className="w-10 h-5 bg-slate-200 rounded-full transition-colors peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300/40 peer-checked:bg-blue-500" />
                                                <span className="absolute left-[2px] top-[2px] h-4 w-4 rounded-full bg-white border border-slate-300 shadow-sm transition-transform peer-checked:translate-x-5 peer-checked:border-white" />
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            )}
                            <div
                                ref={transcriptRef}
                                className="flex-1 overflow-y-auto p-4 space-y-2 select-text pb-40"
                                onMouseUp={handleMouseUp}
                            >
                                {segments.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center mt-16 px-6">
                                        <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-4">
                                            <FileText size={28} className="text-slate-300" />
                                        </div>
                                        <h3 className="text-sm font-semibold text-slate-600 mb-1">No transcript available</h3>
                                        {video && !video.processed && video.status !== 'queued' && video.status !== 'running' && video.status !== 'downloading' && video.status !== 'transcribing' && video.status !== 'diarizing' ? (
                                            <>
                                                <p className="text-xs text-slate-400 mb-5 text-center">Start a transcription job to generate the transcript for this episode.</p>
                                                <button
                                                    onClick={async () => {
                                                        setStartingTranscription(true);
                                                        try {
                                                            await api.post(`/videos/${video.id}/process`);
                                                            setVideo({ ...video, status: 'queued' });
                                                        } catch (e) {
                                                            console.error('Failed to start transcription:', e);
                                                            alert('Failed to start transcription');
                                                        } finally {
                                                            setStartingTranscription(false);
                                                        }
                                                    }}
                                                    disabled={startingTranscription}
                                                    className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all font-medium text-sm disabled:opacity-50"
                                                >
                                                    {startingTranscription ? <Loader2 size={16} className="animate-spin" /> : <Mic size={16} />}
                                                    {startingTranscription ? 'Starting...' : 'Start Transcription'}
                                                </button>
                                            </>
                                        ) : video && (video.status === 'queued' || video.status === 'running' || video.status === 'downloading' || video.status === 'transcribing' || video.status === 'diarizing') ? (
                                            <div className="flex items-center gap-2 mt-2 text-xs text-blue-500">
                                                <Loader2 size={14} className="animate-spin" />
                                                <span>{video.status === 'queued' ? 'Transcription queued' : video.status === 'diarizing' ? 'Diarization in progress' : 'Transcription in progress'}...</span>
                                            </div>
                                        ) : (
                                            <p className="text-xs text-slate-400">This episode has not been transcribed yet.</p>
                                        )}
                                    </div>
                                ) : (
                                    filteredSegments.map((seg, filteredIdx) => {
                                        // Highlight logic
                                        const isActiveSegment = currentTime >= seg.start_time && currentTime < seg.end_time;
                                        const isActiveMatch = searchLower && filteredIdx === searchMatchIndex;
                                        const wordsFn = typeof seg.id === 'number'
                                            ? (normalizedWordsBySegmentId.get(seg.id) || [])
                                            : parseSegmentWords(seg);

                                        return (
                                            <div
                                                key={seg.id}
                                                id={`seg-${seg.id}`}
                                                data-start={seg.start_time}
                                                data-end={seg.end_time}
                                                className={`p-3 rounded-lg text-sm transition-all cursor-pointer border relative group ${isActiveMatch
                                                    ? 'bg-yellow-50 border-yellow-300 ring-1 ring-yellow-200 shadow-sm'
                                                    : isActiveSegment
                                                        ? 'bg-blue-50 border-blue-200 shadow-sm ring-1 ring-blue-100'
                                                        : 'bg-white border-transparent hover:border-slate-200 hover:bg-white'}`}
                                                onClick={(e) => {
                                                    // If clicking the container (not a word span), seek to segment start
                                                    // But check if selection exists
                                                    if (window.getSelection()?.toString().length === 0) {
                                                        // Only seek if target wasn't a specific word (handled by word click)
                                                        if ((e.target as HTMLElement).tagName !== 'SPAN') {
                                                            handleSeek(seg.start_time);
                                                        }
                                                    }
                                                }}
                                            >
                                                <div className="flex justify-between items-center mb-1 text-xs text-slate-400 select-none">
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        <span
                                                            className="font-medium text-slate-500 hover:text-blue-600 hover:underline cursor-pointer truncate"
                                                            onClick={(e) => {
                                                                if (seg.speaker_id) {
                                                                    e.stopPropagation();
                                                                    handleSpeakerClick(seg.speaker_id, seg);
                                                                } else {
                                                                    handleUnknownSpeakerClick(seg.id, e);
                                                                }
                                                            }}
                                                        >
                                                            {seg.speaker || "Unknown"}
                                                        </span>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                beginSegmentEdit(seg);
                                                            }}
                                                            className="opacity-0 group-hover:opacity-100 p-1 rounded text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition"
                                                            title="Edit transcript text"
                                                        >
                                                            <Pencil size={12} />
                                                        </button>
                                                        {seg.speaker_id && (
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    void handleSplitSegmentProfile(seg);
                                                                }}
                                                                disabled={splittingSegmentId === seg.id}
                                                                className="opacity-0 group-hover:opacity-100 p-1 rounded text-slate-400 hover:text-purple-600 hover:bg-purple-50 transition disabled:opacity-50"
                                                                title="Split this segment into a standalone speaker profile"
                                                            >
                                                                {splittingSegmentId === seg.id ? (
                                                                    <Loader2 size={12} className="animate-spin" />
                                                                ) : (
                                                                    <Scissors size={12} />
                                                                )}
                                                            </button>
                                                        )}
                                                    </div>
                                                    <span className="font-mono shrink-0">{new Date(seg.start_time * 1000).toISOString().substr(14, 5)}</span>
                                                </div>
                                                {editingSegmentId === seg.id ? (
                                                    <div className="space-y-2">
                                                        <div
                                                            onClick={(e) => e.stopPropagation()}
                                                            className="rounded-lg border border-blue-200 bg-white p-2.5 space-y-2"
                                                        >
                                                            <div className="flex flex-wrap items-center gap-1.5">
                                                                {editingSegmentWords.map((word, idx) => (
                                                                    <div key={`${seg.id}-${idx}`} className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-1.5 py-1">
                                                                        <input
                                                                            value={word}
                                                                            onChange={(e) => updateEditingWord(idx, e.target.value)}
                                                                            className="min-w-[2.5ch] max-w-[22ch] bg-transparent text-sm text-slate-700 focus:outline-none"
                                                                            style={{ width: `${Math.max(2.5, Math.min(22, (word || '').length + 1.5))}ch` }}
                                                                        />
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => removeEditingWord(idx)}
                                                                            className="text-slate-400 hover:text-red-600"
                                                                            title="Remove word"
                                                                        >
                                                                            <X size={11} />
                                                                        </button>
                                                                    </div>
                                                                ))}
                                                                <button
                                                                    type="button"
                                                                    onClick={addEditingWord}
                                                                    className="inline-flex items-center gap-1 rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100"
                                                                    title="Add word"
                                                                >
                                                                    <Plus size={11} />
                                                                    Add
                                                                </button>
                                                            </div>
                                                            <div className="text-[11px] text-slate-500">
                                                                Per-word timing is preserved when possible.
                                                            </div>
                                                        </div>
                                                        <div className="flex items-center justify-end gap-2">
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    if (!editingLoopSegment) {
                                                                        setEditingLoopSegment(true);
                                                                    } else {
                                                                        setEditingLoopSegment(false);
                                                                        pauseMainPreview();
                                                                    }
                                                                }}
                                                                className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border ${editingLoopSegment
                                                                    ? 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100'
                                                                    : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
                                                                    }`}
                                                                title="Loop this segment while editing"
                                                            >
                                                                {editingLoopSegment ? <Pause size={12} /> : <Play size={12} />}
                                                                {editingLoopSegment ? 'Stop Loop' : 'Loop Segment'}
                                                            </button>
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    setEditingLoopSegment(false);
                                                                    pauseMainPreview();
                                                                    setEditingSegmentId(null);
                                                                    setEditingSegmentWords([]);
                                                                }}
                                                                className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-slate-100 text-slate-600 hover:bg-slate-200"
                                                            >
                                                                <XCircle size={12} /> Cancel
                                                            </button>
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); void saveSegmentEdit(seg.id); }}
                                                                disabled={savingSegmentEdit}
                                                                className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                                            >
                                                                {savingSegmentEdit ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                                                                Save
                                                            </button>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <p className="text-slate-700 leading-relaxed">
                                                        {wordsFn.length > 0 && isActiveSegment ? (
                                                            wordsFn.map((w: any, idx: number) => {
                                                                const isWordActive = currentTime >= w.start && currentTime < w.end;
                                                                return (
                                                                    <span
                                                                        key={idx}
                                                                        className={`inline-block mr-1 px-0.5 rounded transition-colors duration-100 ${isWordActive ? 'bg-blue-200/90 text-blue-900 shadow-[inset_0_0_0_1px_rgba(59,130,246,0.35)]' : 'hover:bg-slate-200'}`}
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            handleSeek(w.start);
                                                                        }}
                                                                    >
                                                                        {searchLower ? highlightText(w.word) : w.word}
                                                                    </span>
                                                                );
                                                            })
                                                        ) : (
                                                            searchLower ? highlightText(seg.text) : seg.text
                                                        )}
                                                    </p>
                                                )}
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        </div>
                    )}
                    {activeTab === 'clips' && (
                        <div className="h-full overflow-y-auto p-4 space-y-3">
                            {clips.length > 0 && (
                                <div className="rounded-xl border border-slate-200 bg-white p-3 space-y-3 sticky top-0 z-10 shadow-sm">
                                    <div className="flex items-center justify-between gap-2">
                                        <div>
                                            <div className="text-sm font-semibold text-slate-800">Batch Export + Presets</div>
                                            <div className="text-xs text-slate-500">Select clips, apply a preset, export MP4 (+ caption sidecars).</div>
                                        </div>
                                        <button
                                            onClick={() => setSelectedClipIds(new Set(clips.map(c => c.id)))}
                                            className="text-xs px-2 py-1 rounded-md bg-slate-100 text-slate-600 hover:bg-slate-200"
                                        >
                                            Select All
                                        </button>
                                    </div>
                                    <div className="space-y-2">
                                        <select
                                            value={clipBatchPresetKey}
                                            onChange={(e) => setClipBatchPresetKey(e.target.value as any)}
                                            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                        >
                                            {Object.entries(CLIP_BATCH_PRESETS).map(([key, preset]) => (
                                                <option key={key} value={key}>{preset.label}</option>
                                            ))}
                                        </select>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                            <button
                                                onClick={() => void batchExportSelectedClips()}
                                                disabled={batchExporting || batchUploadingClips || batchQueueingRenders || selectedClipIds.size === 0}
                                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 whitespace-nowrap"
                                            >
                                                {batchExporting ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                                                Export Selected ({selectedClipIds.size})
                                            </button>
                                            <button
                                                onClick={() => void queueRenderSelectedClips()}
                                                disabled={batchQueueingRenders || batchExporting || batchUploadingClips || selectedClipIds.size === 0}
                                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
                                                title="Queue server-side clip render jobs (parallel with other queues)"
                                            >
                                                {batchQueueingRenders ? <Loader2 size={13} className="animate-spin" /> : <Clock size={13} />}
                                                Queue Renders ({selectedClipIds.size})
                                            </button>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-[160px_minmax(0,1fr)] gap-2">
                                            <select
                                                value={clipUploadPrivacy}
                                                onChange={(e) => setClipUploadPrivacy(e.target.value as 'private' | 'unlisted' | 'public')}
                                                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                                title="YouTube upload privacy"
                                            >
                                                <option value="private">Upload Private</option>
                                                <option value="unlisted">Upload Unlisted</option>
                                                <option value="public">Upload Public</option>
                                            </select>
                                            <button
                                                onClick={() => void batchUploadSelectedClips()}
                                                disabled={batchUploadingClips || batchExporting || selectedClipIds.size === 0}
                                                className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 whitespace-nowrap"
                                                title="Upload selected clips directly to your connected YouTube channel"
                                            >
                                                {batchUploadingClips ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />}
                                                Upload Selected ({selectedClipIds.size})
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {loadingClips && clips.length === 0 ? (
                                <div className="flex justify-center p-8"><Loader2 className="animate-spin text-slate-300" /></div>
                            ) : clips.length === 0 ? (
                                <div className="text-center mt-10 p-8 border-2 border-dashed border-slate-200 rounded-xl">
                                    <Scissors className="mx-auto text-slate-300 mb-2" size={32} />
                                    <h3 className="text-slate-500 font-medium">No clips yet</h3>
                                    <p className="text-sm text-slate-400 mt-1">Select text in the transcript to create a clip.</p>
                                </div>
                            ) : (
                                clips.map(clip => {
                                    const isEditing = editingClipId === clip.id;
                                    const draft = isEditing && clipEditorDraft ? clipEditorDraft : clip;
                                    const isExporting = exportingClipIds.has(clip.id);
                                    const isUploading = uploadingClipIds.has(clip.id);
                                    const isLooping = clipPreviewLoop?.clipId === clip.id;
                                    return (
                                        <div key={clip.id} className={`bg-white p-3 rounded-lg border shadow-sm ${isEditing ? 'border-purple-300 ring-1 ring-purple-200' : 'border-slate-200'}`}>
                                            <div className="flex gap-3 items-start">
                                                <input
                                                    type="checkbox"
                                                    checked={selectedClipIds.has(clip.id)}
                                                    onChange={() => toggleClipSelected(clip.id)}
                                                    className="mt-2 h-4 w-4 rounded border-slate-300 text-purple-600"
                                                />
                                                <div
                                                    className="w-24 h-16 bg-slate-100 rounded overflow-hidden flex-shrink-0 relative cursor-pointer"
                                                    onClick={() => handleSeek((draft.start_time as number) ?? clip.start_time)}
                                                    title="Jump to clip start in player"
                                                >
                                                    <div className="w-full h-full bg-gradient-to-br from-purple-100 to-indigo-100 flex items-center justify-center">
                                                        <Play size={20} className="text-purple-400" />
                                                    </div>
                                                </div>
                                                <div className="flex-1 min-w-0 space-y-2">
                                                    <div className="flex items-start justify-between gap-2">
                                                        <div className="min-w-0">
                                                            <h4 className="text-sm font-medium text-slate-800 line-clamp-1">{clip.title}</h4>
                                                            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 font-mono mt-0.5">
                                                                <span>{formatTime(clip.start_time)} - {formatTime(clip.end_time)}</span>
                                                                <span className="w-1 h-1 bg-slate-300 rounded-full" />
                                                                <span>{(clip.end_time - clip.start_time).toFixed(1)}s</span>
                                                                <span className="w-1 h-1 bg-slate-300 rounded-full" />
                                                                <span>{(clip.aspect_ratio || 'source').toUpperCase()}</span>
                                                                {clip.burn_captions && <span className="px-1 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 text-[10px]">burned captions</span>}
                                                                {clip.portrait_split_enabled && String(clip.aspect_ratio || '').toLowerCase() === '9:16' && (
                                                                    <span className="px-1 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[10px]">split</span>
                                                                )}
                                                                {clip.script_edits_json && (
                                                                    <span className="px-1 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px]">text-edited</span>
                                                                )}
                                                                {((clip.fade_in_sec || 0) > 0 || (clip.fade_out_sec || 0) > 0) && (
                                                                    <span className="px-1 py-0.5 rounded bg-fuchsia-50 text-fuchsia-700 border border-fuchsia-200 text-[10px]">
                                                                        fades {Number(clip.fade_in_sec || 0).toFixed(1)}/{Number(clip.fade_out_sec || 0).toFixed(1)}s
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div className="flex items-center gap-1">
                                                            <button
                                                                onClick={() => toggleClipPreviewLoop(clip)}
                                                                className={`px-2 py-1 text-xs rounded-md ${isLooping ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                                                                title="Loop preview this clip in the main player"
                                                            >
                                                                {isLooping ? 'Stop Loop' : 'Loop'}
                                                            </button>
                                                            <button
                                                                onClick={() => isEditing ? cancelClipEdit() : startClipEdit(clip)}
                                                                className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded"
                                                                title="Edit clip trim/export settings"
                                                            >
                                                                <Pencil size={14} />
                                                            </button>
                                                            <button
                                                                onClick={() => handleDeleteClip(clip.id)}
                                                                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded"
                                                                title="Delete clip"
                                                            >
                                                                <Trash2 size={14} />
                                                            </button>
                                                        </div>
                                                    </div>

                                                    <div className="flex flex-wrap gap-2">
                                                        <button
                                                            onClick={() => void queueClipMp4(clip)}
                                                            disabled={isExporting || isUploading}
                                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                                        >
                                                            {isExporting ? <Loader2 size={12} className="animate-spin" /> : <Clock size={12} />}
                                                            Queue MP4
                                                        </button>
                                                        <button
                                                            onClick={() => void exportClipMp4(clip)}
                                                            disabled={isExporting || isUploading}
                                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50"
                                                        >
                                                            {isExporting ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
                                                            Download MP4
                                                        </button>
                                                        <button
                                                            onClick={() => void uploadClipToYoutube(clip)}
                                                            disabled={isExporting || isUploading || batchUploadingClips}
                                                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                                                            title="Upload this clip to your connected YouTube channel"
                                                        >
                                                            {isUploading ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                                                            Upload
                                                        </button>
                                                        <button
                                                            onClick={() => void exportClipCaptions(clip, 'srt')}
                                                            disabled={isExporting || isUploading}
                                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                                        >
                                                            SRT
                                                        </button>
                                                        <button
                                                            onClick={() => void exportClipCaptions(clip, 'vtt')}
                                                            disabled={isExporting || isUploading}
                                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                                                        >
                                                            VTT
                                                        </button>
                                                    </div>

                                                    {(() => {
                                                        const artifacts = (clipExportArtifactsByClip[clip.id] || []).slice(0, 4);
                                                        if (artifacts.length === 0) {
                                                            return (
                                                                <div className="text-[11px] text-slate-400">
                                                                    No archived exports yet. Export once to save for re-download.
                                                                </div>
                                                            );
                                                        }
                                                        return (
                                                            <div className="space-y-1">
                                                                <div className="text-[11px] font-medium text-slate-500">Archived outputs</div>
                                                                <div className="flex flex-wrap gap-1.5">
                                                                    {artifacts.map((art) => (
                                                                        <button
                                                                            key={art.id}
                                                                            onClick={() => void downloadArchivedArtifact(art)}
                                                                            className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] border border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                                                                            title={`${art.file_name} • ${new Date(art.created_at).toLocaleString()}`}
                                                                        >
                                                                            <Download size={11} />
                                                                            {art.format.toUpperCase()} {formatFileSize(art.file_size_bytes)}
                                                                        </button>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        );
                                                    })()}

                                                    {isEditing && (
                                                        <div className="text-[11px] text-purple-700 bg-purple-50 border border-purple-200 rounded-lg px-2.5 py-2">
                                                            Editing in main preview panel. Scroll/right pane to adjust trim, crop, split layout, and burned captions.
                                                        </div>
                                                    )}

                                                    {false && isEditing && draft && (
                                                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-3">
                                                            <div className="grid grid-cols-1 gap-4">
                                                                <div className="space-y-3">
                                                                <div>
                                                                    <label className="block text-[11px] font-medium text-slate-600 mb-1">Title</label>
                                                                    <input
                                                                        type="text"
                                                                        value={String(draft.title || '')}
                                                                        onChange={(e) => updateClipDraftField('title', e.target.value)}
                                                                        className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                                                    />
                                                                </div>
                                                                <div className="grid grid-cols-2 gap-3">
                                                                    <div>
                                                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Start (sec)</label>
                                                                        <input type="number" step="0.1" value={Number(draft.start_time ?? clip.start_time)} onChange={(e) => updateClipDraftField('start_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                                                                    </div>
                                                                    <div>
                                                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">End (sec)</label>
                                                                        <input type="number" step="0.1" value={Number(draft.end_time ?? clip.end_time)} onChange={(e) => updateClipDraftField('end_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                                                                    </div>
                                                                </div>
                                                                <div className="grid grid-cols-2 gap-3">
                                                                    <div>
                                                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Aspect Ratio</label>
                                                                        <select value={String(draft.aspect_ratio || 'source')} onChange={(e) => updateClipDraftField('aspect_ratio', e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white">
                                                                            <option value="source">Source</option>
                                                                            <option value="16:9">16:9</option>
                                                                            <option value="9:16">9:16</option>
                                                                            <option value="1:1">1:1</option>
                                                                            <option value="4:5">4:5</option>
                                                                        </select>
                                                                    </div>
                                                                    <div className="flex items-end">
                                                                        <button
                                                                            onClick={() => {
                                                                                updateClipDraftField('crop_x', null);
                                                                                updateClipDraftField('crop_y', null);
                                                                                updateClipDraftField('crop_w', null);
                                                                                updateClipDraftField('crop_h', null);
                                                                            }}
                                                                            className="w-full px-3 py-2 text-xs rounded-lg bg-white border border-slate-300 text-slate-600 hover:bg-slate-50"
                                                                        >
                                                                            Reset Crop
                                                                        </button>
                                                                    </div>
                                                                </div>
                                                                <div>
                                                                    <label className="block text-[11px] font-medium text-slate-600 mb-1">Crop / Reframe (normalized 0-1)</label>
                                                                    <div className="grid grid-cols-4 gap-2">
                                                                        {(['crop_x','crop_y','crop_w','crop_h'] as const).map((field) => (
                                                                            <input
                                                                                key={field}
                                                                                type="number"
                                                                                min={0}
                                                                                max={1}
                                                                                step={0.01}
                                                                                value={draft[field] == null ? '' : Number(draft[field])}
                                                                                placeholder={field.replace('crop_','')}
                                                                                onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                                                className="px-2 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                                                                            />
                                                                        ))}
                                                                    </div>
                                                                    <p className="mt-1 text-[10px] text-slate-500">Set `x y w h` to crop before aspect-ratio scaling/padding. Leave blank for full frame.</p>
                                                                </div>
                                                                {String(draft.aspect_ratio || 'source') === '9:16' && (
                                                                    <div className="rounded-lg border border-indigo-200 bg-indigo-50/50 p-2.5 space-y-2">
                                                                        <label className="inline-flex items-center gap-2 text-xs text-indigo-700 font-medium">
                                                                            <input
                                                                                type="checkbox"
                                                                                checked={!!draft.portrait_split_enabled}
                                                                                onChange={(e) => {
                                                                                    if (e.target.checked) {
                                                                                        applyPortraitSplitDefaults();
                                                                                    } else {
                                                                                        updateClipDraftField('portrait_split_enabled', false);
                                                                                    }
                                                                                }}
                                                                                className="h-4 w-4 rounded border-indigo-300 text-indigo-600"
                                                                            />
                                                                            Portrait split mode (top/lower stacked)
                                                                        </label>
                                                                        {!!draft.portrait_split_enabled && (
                                                                            <>
                                                                                <div className="grid grid-cols-4 gap-2">
                                                                                    {(['portrait_top_crop_x', 'portrait_top_crop_y', 'portrait_top_crop_w', 'portrait_top_crop_h'] as const).map((field) => (
                                                                                        <input
                                                                                            key={field}
                                                                                            type="number"
                                                                                            min={0}
                                                                                            max={1}
                                                                                            step={0.01}
                                                                                            value={draft[field] == null ? '' : Number(draft[field])}
                                                                                            placeholder={field.replace('portrait_top_crop_', 'top_')}
                                                                                            onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                                                            className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white"
                                                                                        />
                                                                                    ))}
                                                                                </div>
                                                                                <div className="grid grid-cols-4 gap-2">
                                                                                    {(['portrait_bottom_crop_x', 'portrait_bottom_crop_y', 'portrait_bottom_crop_w', 'portrait_bottom_crop_h'] as const).map((field) => (
                                                                                        <input
                                                                                            key={field}
                                                                                            type="number"
                                                                                            min={0}
                                                                                            max={1}
                                                                                            step={0.01}
                                                                                            value={draft[field] == null ? '' : Number(draft[field])}
                                                                                            placeholder={field.replace('portrait_bottom_crop_', 'low_')}
                                                                                            onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                                                            className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white"
                                                                                        />
                                                                                    ))}
                                                                                </div>
                                                                            </>
                                                                        )}
                                                                    </div>
                                                                )}
                                                                <div className="grid grid-cols-2 gap-3">
                                                                    <label className="inline-flex items-center gap-2 text-xs text-slate-600">
                                                                        <input type="checkbox" checked={!!draft.burn_captions} onChange={(e) => updateClipDraftField('burn_captions', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />
                                                                        Burn captions into MP4
                                                                    </label>
                                                                    <label className="inline-flex items-center gap-2 text-xs text-slate-600">
                                                                        <input type="checkbox" checked={!!draft.caption_speaker_labels} onChange={(e) => updateClipDraftField('caption_speaker_labels', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />
                                                                        Speaker labels in captions
                                                                    </label>
                                                                </div>
                                                                <div className="space-y-2">
                                                                    <div className="text-[11px] font-semibold tracking-wide text-slate-600">Preview (guide)</div>
                                                                    <div
                                                                        className="relative w-full rounded-lg overflow-hidden border border-slate-300 bg-slate-900"
                                                                        style={{
                                                                            aspectRatio:
                                                                                String(draft.aspect_ratio || 'source') === '1:1'
                                                                                    ? '1 / 1'
                                                                                    : String(draft.aspect_ratio || 'source') === '4:5'
                                                                                        ? '4 / 5'
                                                                                        : String(draft.aspect_ratio || 'source') === '9:16'
                                                                                            ? '9 / 16'
                                                                                            : '16 / 9',
                                                                        }}
                                                                    >
                                                                        {video?.thumbnail_url ? (
                                                                            <img src={video?.thumbnail_url || ''} alt="" className="absolute inset-0 w-full h-full object-cover opacity-70" />
                                                                        ) : (
                                                                            <div className="absolute inset-0 bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900" />
                                                                        )}
                                                                        {!(String(draft.aspect_ratio || 'source') === '9:16' && !!draft.portrait_split_enabled) && (
                                                                            <div
                                                                                className="absolute border-2 border-cyan-300/90 bg-cyan-300/10"
                                                                                style={{
                                                                                    left: `${normalizeCropRect(draft.crop_x, draft.crop_y, draft.crop_w, draft.crop_h).x * 100}%`,
                                                                                    top: `${normalizeCropRect(draft.crop_x, draft.crop_y, draft.crop_w, draft.crop_h).y * 100}%`,
                                                                                    width: `${normalizeCropRect(draft.crop_x, draft.crop_y, draft.crop_w, draft.crop_h).w * 100}%`,
                                                                                    height: `${normalizeCropRect(draft.crop_x, draft.crop_y, draft.crop_w, draft.crop_h).h * 100}%`,
                                                                                }}
                                                                            />
                                                                        )}
                                                                        {String(draft.aspect_ratio || 'source') === '9:16' && !!draft.portrait_split_enabled && (
                                                                            <>
                                                                                <div
                                                                                    className="absolute border-2 border-amber-300/90 bg-amber-300/15"
                                                                                    style={{
                                                                                        left: `${normalizeCropRect(draft.portrait_top_crop_x, draft.portrait_top_crop_y, draft.portrait_top_crop_w, draft.portrait_top_crop_h, { x: 0, y: 0, w: 1, h: 0.5 }).x * 100}%`,
                                                                                        top: `${normalizeCropRect(draft.portrait_top_crop_x, draft.portrait_top_crop_y, draft.portrait_top_crop_w, draft.portrait_top_crop_h, { x: 0, y: 0, w: 1, h: 0.5 }).y * 100}%`,
                                                                                        width: `${normalizeCropRect(draft.portrait_top_crop_x, draft.portrait_top_crop_y, draft.portrait_top_crop_w, draft.portrait_top_crop_h, { x: 0, y: 0, w: 1, h: 0.5 }).w * 100}%`,
                                                                                        height: `${normalizeCropRect(draft.portrait_top_crop_x, draft.portrait_top_crop_y, draft.portrait_top_crop_w, draft.portrait_top_crop_h, { x: 0, y: 0, w: 1, h: 0.5 }).h * 100}%`,
                                                                                    }}
                                                                                />
                                                                                <div
                                                                                    className="absolute border-2 border-lime-300/90 bg-lime-300/15"
                                                                                    style={{
                                                                                        left: `${normalizeCropRect(draft.portrait_bottom_crop_x, draft.portrait_bottom_crop_y, draft.portrait_bottom_crop_w, draft.portrait_bottom_crop_h, { x: 0, y: 0.5, w: 1, h: 0.5 }).x * 100}%`,
                                                                                        top: `${normalizeCropRect(draft.portrait_bottom_crop_x, draft.portrait_bottom_crop_y, draft.portrait_bottom_crop_w, draft.portrait_bottom_crop_h, { x: 0, y: 0.5, w: 1, h: 0.5 }).y * 100}%`,
                                                                                        width: `${normalizeCropRect(draft.portrait_bottom_crop_x, draft.portrait_bottom_crop_y, draft.portrait_bottom_crop_w, draft.portrait_bottom_crop_h, { x: 0, y: 0.5, w: 1, h: 0.5 }).w * 100}%`,
                                                                                        height: `${normalizeCropRect(draft.portrait_bottom_crop_x, draft.portrait_bottom_crop_y, draft.portrait_bottom_crop_w, draft.portrait_bottom_crop_h, { x: 0, y: 0.5, w: 1, h: 0.5 }).h * 100}%`,
                                                                                    }}
                                                                                />
                                                                                <div className="absolute inset-x-0 top-1/2 border-t border-white/70 border-dashed" />
                                                                            </>
                                                                        )}
                                                                        {!!draft.burn_captions && (
                                                                            <div className="absolute inset-x-2 bottom-2 px-2 py-1.5 rounded bg-black/55 text-[11px] text-white text-center">
                                                                                [Speaker] Sample burned caption preview
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                    <p className="text-[10px] text-slate-500">Preview is approximate and shows crop + burn-overlay placement.</p>
                                                                </div>
                                                            </div>
                                                            </div>
                                                            <div className="flex items-center justify-end gap-2">
                                                                <button onClick={cancelClipEdit} className="px-3 py-2 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-600 hover:bg-slate-50">Cancel</button>
                                                                <button
                                                                    onClick={() => void saveClipEdit(clip.id)}
                                                                    disabled={savingClipEdit}
                                                                    className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                                                >
                                                                    {savingClipEdit ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                                                                    Save Clip Settings
                                                                </button>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })
                            )}
                        </div>
                    )}
                    {activeTab === 'speakers' && (
                        <div className="h-full overflow-y-auto p-4">
                            {video ? (
                                <SpeakerList videoId={video.id} channelId={undefined} />
                            ) : (
                                <Loader2 className="animate-spin" />
                            )}
                        </div>
                    )}
                    {activeTab === 'youtube' && (
                        <div className="h-full overflow-y-auto p-4 space-y-4">
                            <div className="rounded-xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50 p-4">
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-800">
                                            <Bot size={15} className="text-emerald-600" />
                                            YouTube Summary + Chapters
                                        </div>
                                        <p className="mt-1 text-xs text-emerald-700/80">
                                            Generate a YouTube-style episode description summary and chapter timestamps/descriptions from the transcript using the current LLM provider.
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => handleGenerateYoutubeAi(hasYoutubeAiMetadata)}
                                        disabled={generatingYoutubeAi || segments.length === 0}
                                        className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                        title={segments.length === 0 ? 'Transcript required first' : 'Generate or re-generate YouTube summary + chapters'}
                                    >
                                        {generatingYoutubeAi ? <Loader2 size={13} className="animate-spin" /> : (hasYoutubeAiMetadata ? <RefreshCw size={13} /> : <Bot size={13} />)}
                                        {hasYoutubeAiMetadata ? 'Re-generate' : 'Generate'}
                                    </button>
                                </div>
                                <div className="mt-2 flex items-center gap-2">
                                    <button
                                        onClick={handlePublishYoutubeDescription}
                                        disabled={publishingYoutubeDescription || !video.youtube_ai_description_text}
                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                        title={video.youtube_ai_description_text ? 'Archive current description and apply the AI draft to this video record' : 'Generate a draft first'}
                                    >
                                        {publishingYoutubeDescription ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                                        Publish Draft (Archive Current)
                                    </button>
                                    <span className="text-[11px] text-emerald-800/80">
                                        Updates the app’s stored video description and preserves restorable history.
                                    </span>
                                </div>
                                {video.youtube_ai_model && (
                                    <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px]">
                                        <span className="px-2 py-0.5 rounded bg-white/80 border border-emerald-200 text-emerald-700">
                                            {video.youtube_ai_model}
                                        </span>
                                        {video.youtube_ai_generated_at && (
                                            <span className="text-emerald-800/80">
                                                {new Date(video.youtube_ai_generated_at).toLocaleString()}
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>

                            {segments.length === 0 ? (
                                <div className="rounded-xl border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">
                                    Transcript required first. Run transcription/diarization before generating YouTube metadata.
                                </div>
                            ) : !hasYoutubeAiMetadata ? (
                                <div className="rounded-xl border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">
                                    No generated summary/chapters yet. Click <span className="font-medium">Generate</span> to create a YouTube-ready draft description and chapter list.
                                </div>
                            ) : (
                                <>
                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">Current Video Description (Stored)</h3>
                                            {video.description && (
                                                <button
                                                    onClick={() => void copyToClipboard(video.description || '', 'description')}
                                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                                >
                                                    <Copy size={12} />
                                                    Copy Current
                                                </button>
                                            )}
                                        </div>
                                        <pre className="text-xs text-slate-700 bg-slate-50 border border-slate-100 rounded-lg p-3 whitespace-pre-wrap break-words font-mono leading-relaxed max-h-56 overflow-y-auto">
                                            {video.description || 'No description stored.'}
                                        </pre>
                                    </div>

                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">Episode Summary</h3>
                                            {video.youtube_ai_summary && (
                                                <button
                                                    onClick={() => void copyToClipboard(video.youtube_ai_summary || '', 'summary')}
                                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                                >
                                                    <Copy size={12} />
                                                    {copiedYoutubeField === 'summary' ? 'Copied' : 'Copy'}
                                                </button>
                                            )}
                                        </div>
                                        <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                                            {video.youtube_ai_summary || 'No summary generated.'}
                                        </p>
                                    </div>

                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">Chapters (YouTube-style)</h3>
                                            {youtubeAiChapters.length > 0 && (
                                                <button
                                                    onClick={() => void copyToClipboard(youtubeAiChapters.map(ch => `${ch.timestamp} ${ch.title}`).join('\n'), 'chapters')}
                                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                                >
                                                    <Copy size={12} />
                                                    {copiedYoutubeField === 'chapters' ? 'Copied' : 'Copy Lines'}
                                                </button>
                                            )}
                                        </div>
                                        {youtubeAiChapters.length === 0 ? (
                                            <p className="text-sm text-slate-500">No chapter timestamps generated.</p>
                                        ) : (
                                            <div className="space-y-2">
                                                {youtubeAiChapters.map((ch, idx) => (
                                                    <button
                                                        key={`${ch.timestamp}-${idx}`}
                                                        onClick={() => handleSeek(ch.start_seconds)}
                                                        className="w-full text-left rounded-lg border border-slate-100 hover:border-emerald-200 hover:bg-emerald-50/40 p-2.5 transition-colors"
                                                        title="Jump preview to chapter timestamp"
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <span className="font-mono text-xs text-emerald-700 min-w-[46px]">{ch.timestamp}</span>
                                                            <span className="text-sm font-medium text-slate-800">{ch.title}</span>
                                                        </div>
                                                        {ch.description && (
                                                            <p className="mt-1 ml-[54px] text-xs text-slate-600 leading-relaxed">
                                                                {ch.description}
                                                            </p>
                                                        )}
                                                    </button>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">YouTube Description Draft (Copy/Paste)</h3>
                                            {video.youtube_ai_description_text && (
                                                <button
                                                    onClick={() => void copyToClipboard(video.youtube_ai_description_text || '', 'description')}
                                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                                >
                                                    <Copy size={12} />
                                                    {copiedYoutubeField === 'description' ? 'Copied' : 'Copy Full'}
                                                </button>
                                            )}
                                        </div>
                                        <pre className="text-xs text-slate-700 bg-slate-50 border border-slate-100 rounded-lg p-3 whitespace-pre-wrap break-words font-mono leading-relaxed">
                                            {video.youtube_ai_description_text || 'No description draft generated yet.'}
                                        </pre>
                                    </div>

                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">Description History (Restore)</h3>
                                            {loadingDescriptionHistory && <Loader2 size={14} className="animate-spin text-slate-400" />}
                                        </div>
                                        {descriptionHistory.length === 0 ? (
                                            <p className="text-sm text-slate-500">No archived descriptions yet. Publishing a draft will archive the current description first.</p>
                                        ) : (
                                            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                                                {descriptionHistory.map((rev) => (
                                                    <div key={rev.id} className="rounded-lg border border-slate-100 p-2.5 bg-slate-50/60">
                                                        <div className="flex items-start justify-between gap-2">
                                                            <div className="min-w-0">
                                                                <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                                                                    <span className="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-700 font-medium">
                                                                        {rev.source}
                                                                    </span>
                                                                    <span className="text-slate-500">
                                                                        {new Date(rev.created_at).toLocaleString()}
                                                                    </span>
                                                                    {rev.ai_model && (
                                                                        <span className="px-1.5 py-0.5 rounded bg-purple-50 border border-purple-100 text-purple-700">
                                                                            {rev.ai_model}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                                {rev.note && (
                                                                    <p className="mt-1 text-[11px] text-slate-500">{rev.note}</p>
                                                                )}
                                                            </div>
                                                            <button
                                                                onClick={() => handleRestoreDescriptionRevision(rev)}
                                                                disabled={restoringDescriptionRevisionId === rev.id}
                                                                className="shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 disabled:opacity-50"
                                                            >
                                                                {restoringDescriptionRevisionId === rev.id ? <Loader2 size={11} className="animate-spin" /> : <RotateCcw size={11} />}
                                                                Restore
                                                            </button>
                                                        </div>
                                                        <pre className="mt-2 text-[11px] text-slate-700 whitespace-pre-wrap break-words font-mono leading-relaxed max-h-24 overflow-y-auto">
                                                            {rev.description_text}
                                                        </pre>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>

                {/* Clip Creation Panel (Slide up) */}
                {selection && (
                    <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 shadow-lg animate-in slide-in-from-bottom-10 z-20">
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                                    <Scissors size={14} className="text-purple-500" />
                                    Create Clip
                                </h3>
                                <p className="text-xs text-slate-500 font-mono mt-1">
                                    {new Date(selection.start * 1000).toISOString().substr(14, 5)} - {new Date(selection.end * 1000).toISOString().substr(14, 5)}
                                    <span className="mx-2">•</span>
                                    {(selection.end - selection.start).toFixed(1)}s
                                </p>
                            </div>
                            <button
                                onClick={() => setSelection(null)}
                                className="text-slate-400 hover:text-slate-600"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <div className="space-y-3">
                            <input
                                type="text"
                                value={clipTitle}
                                onChange={(e) => setClipTitle(e.target.value)}
                                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                                placeholder="Clip Title..."
                                autoFocus
                            />
                            <button
                                onClick={handleCreateClip}
                                disabled={creatingClip || !clipTitle.trim()}
                                className="w-full flex items-center justify-center gap-2 bg-purple-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors disabled:opacity-50"
                            >
                                {creatingClip ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                                Save Clip
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Right Column: Video Stage */}
            <div className="flex-1 bg-slate-100 flex flex-col min-w-0 relative">
                {/* Header */}
                <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center gap-4 shadow-sm z-0">
                    <button onClick={() => navigate(-1)} className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 hover:text-slate-700 transition-colors">
                        <ArrowLeft size={20} />
                    </button>
                    <div className="flex-1 min-w-0">
                        <h1 className="font-semibold text-slate-800 line-clamp-1">{video.title}</h1>
                        <p className="text-xs text-slate-500">{new Date(video.published_at || '').toLocaleDateString()}</p>
                    </div>
                    {(() => {
                        const activeStatuses = ['queued', 'downloading', 'transcribing', 'diarizing'];
                        const jobActive = activeStatuses.includes(video.status);
                        const busy = purging || redoing || redoingDiarization || jobActive;
                        return (
                            <div className="flex items-center gap-2 shrink-0">
                                {jobActive && (
                                    <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-500 bg-slate-100 rounded-lg">
                                        <Loader2 size={14} className="animate-spin" />
                                        {video.status.charAt(0).toUpperCase() + video.status.slice(1)}...
                                    </span>
                                )}
                                <button
                                    onClick={handleRedoDiarization}
                                    disabled={busy}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    title={jobActive ? `Cannot redo while ${video.status}` : "Re-run speaker diarization using improved speaker profiles"}
                                >
                                    {redoingDiarization ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
                                    Redo Diarization
                                </button>
                                <button
                                    onClick={handleRedoTranscript}
                                    disabled={busy}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    title={jobActive ? `Cannot redo while ${video.status}` : "Re-run transcription and then diarization"}
                                >
                                    {redoing ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                                    Redo Transcription
                                </button>
                                <button
                                    onClick={handlePurgeTranscript}
                                    disabled={busy}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    title={jobActive ? `Cannot purge while ${video.status}` : "Purge transcript & diarization data"}
                                >
                                    {purging ? <Loader2 size={14} className="animate-spin" /> : <Eraser size={14} />}
                                    Purge
                                </button>
                            </div>
                        );
                    })()}
                </div>

                {/* Video Container / Clip Editor Workspace */}
                {showClipEditorMain && activeEditingClip && clipEditorDraft ? (
                    <div className="flex-1 overflow-y-auto p-6">
                        <div className="mx-auto max-w-6xl grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-6">
                            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-3">
                                <div className="flex items-center justify-between gap-2">
                                    <div>
                                        <div className="text-sm font-semibold text-slate-800">Clip Editor</div>
                                        <div className="text-xs text-slate-500">Editing: {activeEditingClip.title || `Clip #${activeEditingClip.id}`}</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button onClick={cancelClipEdit} className="px-3 py-2 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-600 hover:bg-slate-50">Close</button>
                                        <button
                                            onClick={() => void saveClipEdit(activeEditingClip.id)}
                                            disabled={savingClipEdit}
                                            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                        >
                                            {savingClipEdit ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                                            Save
                                        </button>
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Title</label>
                                        <input
                                            type="text"
                                            value={String(clipEditorDraft.title || '')}
                                            onChange={(e) => updateClipDraftField('title', e.target.value)}
                                            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Aspect Ratio</label>
                                        <select value={String(clipEditorDraft.aspect_ratio || 'source')} onChange={(e) => updateClipDraftField('aspect_ratio', e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white">
                                            <option value="source">Source</option>
                                            <option value="16:9">16:9</option>
                                            <option value="9:16">9:16</option>
                                            <option value="1:1">1:1</option>
                                            <option value="4:5">4:5</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Start (sec)</label>
                                        <input type="number" step="0.1" value={Number(clipEditorDraft.start_time ?? activeEditingClip.start_time)} onChange={(e) => updateClipDraftField('start_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                                    </div>
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">End (sec)</label>
                                        <input type="number" step="0.1" value={Number(clipEditorDraft.end_time ?? activeEditingClip.end_time)} onChange={(e) => updateClipDraftField('end_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                                    </div>
                                </div>

                                <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 space-y-2">
                                    <div className="text-[11px] font-semibold text-slate-600">Fast Trim Controls</div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                        <button onClick={() => setClipBoundaryFromPlayhead('start')} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Set In @ Playhead (I)</button>
                                        <button onClick={() => setClipBoundaryFromPlayhead('end')} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Set Out @ Playhead (O)</button>
                                        <button onClick={() => handleSeek(Number(clipEditorDraft.start_time ?? activeEditingClip.start_time))} className="px-2 py-1.5 text-[11px] rounded-md border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100">Jump In</button>
                                        <button onClick={() => handleSeek(Number(clipEditorDraft.end_time ?? activeEditingClip.end_time))} className="px-2 py-1.5 text-[11px] rounded-md border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100">Jump Out</button>
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                        <button onClick={() => nudgeClipBoundary('start', -(1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">In -1f</button>
                                        <button onClick={() => nudgeClipBoundary('start', (1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">In +1f</button>
                                        <button onClick={() => nudgeClipBoundary('end', -(1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Out -1f</button>
                                        <button onClick={() => nudgeClipBoundary('end', (1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Out +1f</button>
                                    </div>
                                    <div className="text-[10px] text-slate-500">Keyboard: `I`/`O` set in/out at playhead, `Alt+←/→` nudges In, `Shift+Alt+←/→` nudges Out.</div>
                                </div>

                                <div className="rounded-lg border border-blue-200 bg-blue-50/40 p-2.5 space-y-2">
                                    <div className="flex items-center justify-between gap-2">
                                        <div className="text-[11px] font-semibold text-blue-700">Mini Timeline</div>
                                        <div className="text-[10px] text-blue-700/80">Drag handles or click timeline to seek</div>
                                    </div>
                                    {(() => {
                                        const duration = getEditorMediaDuration();
                                        const startSec = Number(clipEditorDraft.start_time ?? activeEditingClip.start_time);
                                        const endSec = Number(clipEditorDraft.end_time ?? activeEditingClip.end_time);
                                        const startPct = timelineTimeToPct(startSec, duration) * 100;
                                        const endPct = timelineTimeToPct(endSec, duration) * 100;
                                        const playPct = timelineTimeToPct(currentTime, duration) * 100;
                                        return (
                                            <>
                                                <div className="flex items-center justify-between text-[10px] text-blue-700/80 font-mono">
                                                    <span>0:00</span>
                                                    <span>{formatTime(duration / 4)}</span>
                                                    <span>{formatTime(duration / 2)}</span>
                                                    <span>{formatTime((duration * 3) / 4)}</span>
                                                    <span>{formatTime(duration)}</span>
                                                </div>
                                                <div
                                                    ref={clipTimelineRef}
                                                    onClick={handleTimelineScrub}
                                                    className="relative h-10 rounded-md border border-blue-200 bg-white cursor-pointer overflow-hidden"
                                                >
                                                    <div className="absolute inset-y-0 left-0 w-full bg-gradient-to-r from-slate-100 via-blue-50 to-slate-100" />
                                                    <div
                                                        className="absolute inset-y-1 rounded bg-blue-500/25 border border-blue-300"
                                                        style={{ left: `${startPct}%`, width: `${Math.max(0.2, endPct - startPct)}%` }}
                                                    />
                                                    <div
                                                        className="absolute top-0 bottom-0 w-[2px] bg-red-500/80"
                                                        style={{ left: `${playPct}%` }}
                                                    />
                                                    <button
                                                        type="button"
                                                        onPointerDown={(e) => beginClipTimelineDrag('start', e)}
                                                        className="absolute top-0 bottom-0 -ml-1 w-2 rounded bg-blue-700 hover:bg-blue-800 cursor-ew-resize"
                                                        style={{ left: `${startPct}%` }}
                                                        title="Drag In point"
                                                    />
                                                    <button
                                                        type="button"
                                                        onPointerDown={(e) => beginClipTimelineDrag('end', e)}
                                                        className="absolute top-0 bottom-0 -ml-1 w-2 rounded bg-blue-700 hover:bg-blue-800 cursor-ew-resize"
                                                        style={{ left: `${endPct}%` }}
                                                        title="Drag Out point"
                                                    />
                                                </div>
                                                <div className="flex items-center justify-between text-[10px] text-blue-700/80 font-mono">
                                                    <span>IN {formatTime(startSec)}</span>
                                                    <span>{(Math.max(0, endSec - startSec)).toFixed(2)}s</span>
                                                    <span>OUT {formatTime(endSec)}</span>
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>

                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <label className="block text-[11px] font-medium text-slate-600">Main Crop / Reframe (x y w h, 0-1)</label>
                                        <button
                                            onClick={() => {
                                                updateClipDraftField('crop_x', null);
                                                updateClipDraftField('crop_y', null);
                                                updateClipDraftField('crop_w', null);
                                                updateClipDraftField('crop_h', null);
                                            }}
                                            className="px-2.5 py-1 text-[11px] rounded-md bg-white border border-slate-300 text-slate-600 hover:bg-slate-50"
                                        >
                                            Reset Main Crop
                                        </button>
                                    </div>
                                    <div className="grid grid-cols-4 gap-2">
                                        {(['crop_x', 'crop_y', 'crop_w', 'crop_h'] as const).map((field) => (
                                            <input
                                                key={field}
                                                type="number"
                                                min={0}
                                                max={1}
                                                step={0.01}
                                                value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])}
                                                placeholder={field.replace('crop_', '')}
                                                onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                className="px-2 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                                            />
                                        ))}
                                    </div>
                                </div>

                                {String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && (
                                    <div className="rounded-lg border border-indigo-200 bg-indigo-50/50 p-2.5 space-y-2">
                                        <label className="inline-flex items-center gap-2 text-xs text-indigo-700 font-medium">
                                            <input
                                                type="checkbox"
                                                checked={!!clipEditorDraft.portrait_split_enabled}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        applyPortraitSplitDefaults();
                                                    } else {
                                                        updateClipDraftField('portrait_split_enabled', false);
                                                    }
                                                }}
                                                className="h-4 w-4 rounded border-indigo-300 text-indigo-600"
                                            />
                                            Portrait split mode (top/lower stacked)
                                        </label>
                                        {!!clipEditorDraft.portrait_split_enabled && (
                                            <>
                                                <div className="grid grid-cols-4 gap-2">
                                                    {(['portrait_top_crop_x', 'portrait_top_crop_y', 'portrait_top_crop_w', 'portrait_top_crop_h'] as const).map((field) => (
                                                        <input
                                                            key={field}
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step={0.01}
                                                            value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])}
                                                            placeholder={field.replace('portrait_top_crop_', 'top_')}
                                                            onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                            className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white"
                                                        />
                                                    ))}
                                                </div>
                                                <div className="grid grid-cols-4 gap-2">
                                                    {(['portrait_bottom_crop_x', 'portrait_bottom_crop_y', 'portrait_bottom_crop_w', 'portrait_bottom_crop_h'] as const).map((field) => (
                                                        <input
                                                            key={field}
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step={0.01}
                                                            value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])}
                                                            placeholder={field.replace('portrait_bottom_crop_', 'low_')}
                                                            onChange={(e) => updateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))}
                                                            className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white"
                                                        />
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-3">
                                    <label className="inline-flex items-center gap-2 text-xs text-slate-600">
                                        <input type="checkbox" checked={!!clipEditorDraft.burn_captions} onChange={(e) => updateClipDraftField('burn_captions', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />
                                        Burn captions into MP4
                                    </label>
                                    <label className="inline-flex items-center gap-2 text-xs text-slate-600">
                                        <input type="checkbox" checked={!!clipEditorDraft.caption_speaker_labels} onChange={(e) => updateClipDraftField('caption_speaker_labels', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />
                                        Speaker labels in captions
                                    </label>
                                </div>

                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Fade In (sec)</label>
                                        <input
                                            type="number"
                                            min={0}
                                            step={0.05}
                                            value={Number(clipEditorDraft.fade_in_sec ?? 0)}
                                            onChange={(e) => updateClipDraftField('fade_in_sec', Math.max(0, Number(e.target.value || 0)))}
                                            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-[11px] font-medium text-slate-600 mb-1">Fade Out (sec)</label>
                                        <input
                                            type="number"
                                            min={0}
                                            step={0.05}
                                            value={Number(clipEditorDraft.fade_out_sec ?? 0)}
                                            onChange={(e) => updateClipDraftField('fade_out_sec', Math.max(0, Number(e.target.value || 0)))}
                                            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                                        />
                                    </div>
                                </div>

                                <div className="rounded-lg border border-violet-200 bg-violet-50/50 p-2.5 space-y-1.5">
                                    <div className="text-[11px] font-semibold text-violet-700">Composable Export Stack</div>
                                    <div className="flex flex-wrap gap-1.5 text-[10px]">
                                        <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">Source Ranges</span>
                                        {clipEditorDraft.script_edits_json && <span className="px-1.5 py-0.5 rounded bg-emerald-100 border border-emerald-200 text-emerald-700">Text Keep-Ranges</span>}
                                        {(clipEditorDraft.crop_x != null || clipEditorDraft.crop_y != null || clipEditorDraft.crop_w != null || clipEditorDraft.crop_h != null) && (
                                            <span className="px-1.5 py-0.5 rounded bg-cyan-100 border border-cyan-200 text-cyan-700">Crop/Reframe</span>
                                        )}
                                        {String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled && (
                                            <span className="px-1.5 py-0.5 rounded bg-indigo-100 border border-indigo-200 text-indigo-700">Portrait Split</span>
                                        )}
                                        <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">Aspect {String(clipEditorDraft.aspect_ratio || 'source').toUpperCase()}</span>
                                        {!!clipEditorDraft.burn_captions && <span className="px-1.5 py-0.5 rounded bg-amber-100 border border-amber-200 text-amber-700">Burn Captions</span>}
                                        <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">H264/AAC Render</span>
                                    </div>
                                </div>

                                <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3 space-y-2.5">
                                    <div className="flex items-center justify-between gap-2">
                                        <div>
                                            <div className="text-xs font-semibold text-emerald-800">Text-Based Edit (Phase 1)</div>
                                            <div className="text-[11px] text-emerald-700/80">Click words to remove/restore. Export stitches only kept transcript ranges.</div>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <button
                                                onClick={rebuildClipEditorTextWindow}
                                                className="px-2 py-1 text-[11px] rounded-md border border-emerald-300 bg-white text-emerald-700 hover:bg-emerald-50"
                                            >
                                                Refresh Window
                                            </button>
                                            <button
                                                onClick={autoRemoveClipEditorFillers}
                                                disabled={clipEditorTokens.length === 0}
                                                className="px-2 py-1 text-[11px] rounded-md border border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100 disabled:opacity-50"
                                            >
                                                Remove Fillers
                                            </button>
                                            <button
                                                onClick={restoreAllClipEditorWords}
                                                disabled={clipEditorRemovedWordKeys.size === 0}
                                                className="px-2 py-1 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                            >
                                                Restore All
                                            </button>
                                        </div>
                                    </div>
                                    <div className="text-[11px] text-emerald-900/80">
                                        {(() => {
                                            const clipStart = Number(clipEditorDraft.start_time ?? activeEditingClip.start_time);
                                            const clipEnd = Number(clipEditorDraft.end_time ?? activeEditingClip.end_time);
                                            const kept = buildKeptRangesFromTokenState(clipEditorTokens, clipEditorRemovedWordKeys, clipStart, clipEnd);
                                            const dur = kept.reduce((acc, [s, e]) => acc + (e - s), 0);
                                            return `${clipEditorTokens.length} words • ${clipEditorRemovedWordKeys.size} removed • ${kept.length} kept ranges • est. ${dur.toFixed(1)}s output`;
                                        })()}
                                    </div>
                                    <div className="max-h-48 overflow-y-auto rounded-lg border border-emerald-100 bg-white p-2">
                                        {clipEditorTokens.length === 0 ? (
                                            <div className="text-[11px] text-slate-500">
                                                No word-level transcript tokens in this trim window. Adjust start/end and click Refresh Window.
                                            </div>
                                        ) : (
                                            <div className="flex flex-wrap gap-1.5">
                                                {clipEditorTokens.map((tok) => {
                                                    const removed = clipEditorRemovedWordKeys.has(tok.key);
                                                    return (
                                                        <button
                                                            key={tok.key}
                                                            type="button"
                                                            onClick={() => toggleClipEditorWord(tok.key)}
                                                            className={`px-1.5 py-0.5 rounded text-[11px] border transition-colors ${removed
                                                                ? 'bg-rose-50 border-rose-200 text-rose-700 line-through'
                                                                : 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100'
                                                                }`}
                                                            title={`${formatTime(tok.start)} - ${formatTime(tok.end)}`}
                                                        >
                                                            {tok.word}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="w-full bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video">
                                    <YouTube
                                        videoId={video?.youtube_id || ''}
                                        className="w-full h-full"
                                        iframeClassName="w-full h-full"
                                        onReady={onPlayerReady}
                                        onStateChange={onPlayerStateChange}
                                        onPlaybackRateChange={onPlayerPlaybackRateChange}
                                        opts={{
                                            height: '100%',
                                            width: '100%',
                                            playerVars: {
                                                autoplay: 0,
                                                modestbranding: 1,
                                                rel: 0,
                                                ...(Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0
                                                    ? { start: Math.floor(requestedJumpTime) }
                                                    : {}),
                                            },
                                        }}
                                    />
                                </div>
                                <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                                    <div className="flex items-center justify-between gap-2 mb-2">
                                        <div className="text-[11px] font-semibold tracking-wide text-slate-600">Burn/Crop Preview (draw to set crop)</div>
                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={() => setClipEditorCropTarget('main')}
                                                className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'main' ? 'bg-cyan-100 border-cyan-300 text-cyan-700' : 'bg-white border-slate-300 text-slate-600'}`}
                                            >
                                                Main
                                            </button>
                                            {String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled && (
                                                <>
                                                    <button
                                                        onClick={() => setClipEditorCropTarget('top')}
                                                        className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'top' ? 'bg-amber-100 border-amber-300 text-amber-700' : 'bg-white border-slate-300 text-slate-600'}`}
                                                    >
                                                        Top
                                                    </button>
                                                    <button
                                                        onClick={() => setClipEditorCropTarget('bottom')}
                                                        className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'bottom' ? 'bg-lime-100 border-lime-300 text-lime-700' : 'bg-white border-slate-300 text-slate-600'}`}
                                                    >
                                                        Lower
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                    <div
                                        className="relative w-full rounded-lg overflow-hidden border border-slate-300 bg-slate-900 cursor-crosshair"
                                        ref={cropPreviewRef}
                                        onPointerDown={handleCropPreviewPointerDown}
                                        style={{
                                            aspectRatio:
                                                String(clipEditorDraft.aspect_ratio || 'source') === '1:1'
                                                    ? '1 / 1'
                                                    : String(clipEditorDraft.aspect_ratio || 'source') === '4:5'
                                                        ? '4 / 5'
                                                        : String(clipEditorDraft.aspect_ratio || 'source') === '9:16'
                                                            ? '9 / 16'
                                                            : '16 / 9',
                                        }}
                                    >
                                        {video?.thumbnail_url ? (
                                            <img src={video?.thumbnail_url || ''} alt="" className="absolute inset-0 w-full h-full object-cover opacity-70" />
                                        ) : (
                                            <div className="absolute inset-0 bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900" />
                                        )}
                                        {!(String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled) && (
                                            <div
                                                className="absolute border-2 border-cyan-300/90 bg-cyan-300/10"
                                                style={{
                                                    left: `${getDraftCropRect('main').x * 100}%`,
                                                    top: `${getDraftCropRect('main').y * 100}%`,
                                                    width: `${getDraftCropRect('main').w * 100}%`,
                                                    height: `${getDraftCropRect('main').h * 100}%`,
                                                }}
                                            />
                                        )}
                                        {String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled && (
                                            <>
                                                <div
                                                    className="absolute border-2 border-amber-300/90 bg-amber-300/15"
                                                    style={{
                                                        left: `${getDraftCropRect('top').x * 100}%`,
                                                        top: `${getDraftCropRect('top').y * 100}%`,
                                                        width: `${getDraftCropRect('top').w * 100}%`,
                                                        height: `${getDraftCropRect('top').h * 100}%`,
                                                    }}
                                                />
                                                <div
                                                    className="absolute border-2 border-lime-300/90 bg-lime-300/15"
                                                    style={{
                                                        left: `${getDraftCropRect('bottom').x * 100}%`,
                                                        top: `${getDraftCropRect('bottom').y * 100}%`,
                                                        width: `${getDraftCropRect('bottom').w * 100}%`,
                                                        height: `${getDraftCropRect('bottom').h * 100}%`,
                                                    }}
                                                />
                                                <div className="absolute inset-x-0 top-1/2 border-t border-white/70 border-dashed" />
                                            </>
                                        )}
                                        {clipEditorDragRect && (
                                            <div
                                                className="absolute border-2 border-white border-dashed bg-white/10 pointer-events-none"
                                                style={{
                                                    left: `${Math.min(clipEditorDragRect.startX, clipEditorDragRect.currentX) * 100}%`,
                                                    top: `${Math.min(clipEditorDragRect.startY, clipEditorDragRect.currentY) * 100}%`,
                                                    width: `${Math.max(0.01, Math.abs(clipEditorDragRect.currentX - clipEditorDragRect.startX)) * 100}%`,
                                                    height: `${Math.max(0.01, Math.abs(clipEditorDragRect.currentY - clipEditorDragRect.startY)) * 100}%`,
                                                }}
                                            />
                                        )}
                                        {!!clipEditorDraft.burn_captions && (
                                            <div className="absolute inset-x-2 bottom-2 px-2 py-1.5 rounded bg-black/55 text-[11px] text-white text-center">
                                                [Speaker] Sample burned caption preview
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
                        <div className="w-full max-w-5xl bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video">
                            <YouTube
                                videoId={video?.youtube_id || ''}
                                className="w-full h-full"
                                iframeClassName="w-full h-full"
                                onReady={onPlayerReady}
                                onStateChange={onPlayerStateChange}
                                onPlaybackRateChange={onPlayerPlaybackRateChange}
                                opts={{
                                    height: '100%',
                                    width: '100%',
                                    playerVars: {
                                        autoplay: 0,
                                        modestbranding: 1,
                                        rel: 0,
                                        ...(Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0
                                            ? { start: Math.floor(requestedJumpTime) }
                                            : {}),
                                    },
                                }}
                            />
                        </div>
                    </div>
                )}

                {!showClipEditorMain && (
                <>
                <button
                    onClick={() => setFunnyDrawerOpen(v => !v)}
                    className="absolute right-4 top-20 z-20 flex items-center gap-2 px-3 py-2 rounded-xl border border-amber-200 bg-white/95 hover:bg-white shadow-lg text-amber-800 text-sm font-medium"
                    title="Open funny moments drawer"
                >
                    <Smile size={15} className="text-amber-600" />
                    {funnyDrawerOpen ? 'Hide Funny Moments' : 'Funny Moments'}
                </button>

                <div
                    className={`absolute right-4 top-34 bottom-4 z-20 w-[380px] max-w-[calc(100%-2rem)] rounded-2xl border border-slate-200 bg-white/95 backdrop-blur-sm shadow-2xl overflow-hidden transition-transform duration-200 ${funnyDrawerOpen ? 'translate-x-0' : 'translate-x-[110%]'}`}
                >
                    <div className="h-full flex flex-col">
                        <div className="px-4 py-3 border-b border-slate-200 bg-gradient-to-r from-amber-50 to-yellow-50">
                            <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                    <div className="flex items-center gap-2 text-amber-800 font-semibold text-sm">
                                        <Smile size={15} className="text-amber-600" />
                                        Funny Moments
                                    </div>
                                    <p className="text-xs text-amber-700/80 mt-0.5">
                                        Click to jump video and transcript to the laugh moment.
                                    </p>
                                </div>
                                <button
                                    onClick={() => setFunnyDrawerOpen(false)}
                                    className="p-1.5 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-white/80"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                            <div className="mt-3 flex items-center gap-2">
                                <div className="grid grid-cols-2 gap-2 w-full">
                                    <button
                                        onClick={() => handleDetectFunnyMoments(true)}
                                        disabled={detectingFunnyMoments || !video?.processed}
                                        className="h-10 px-2 rounded-lg text-xs font-medium bg-amber-100 text-amber-800 hover:bg-amber-200 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                                        title={video?.processed ? 'Analyze transcript/audio for funny moments' : 'Transcribe the episode first'}
                                    >
                                        {detectingFunnyMoments ? <Loader2 size={13} className="animate-spin" /> : <Smile size={13} />}
                                        {funnyMoments.length > 0 ? 'Rescan' : 'Find'}
                                    </button>
                                    <button
                                        onClick={() => handleExplainFunnyMoments(true)}
                                        disabled={explainingFunnyMoments || funnyMoments.length === 0}
                                        className="h-10 px-2 rounded-lg text-xs font-medium bg-purple-100 text-purple-700 hover:bg-purple-200 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                                        title="Force-regenerate global humor context and moment explanations with the current LLM provider/model"
                                    >
                                        {explainingFunnyMoments ? <Loader2 size={13} className="animate-spin" /> : (hasExistingFunnyExplanations ? <RefreshCw size={13} /> : <Search size={13} />)}
                                        {hasExistingFunnyExplanations ? 'Re-explain' : 'Explain'}
                                    </button>
                                </div>
                            </div>
                            {funnyDrawerTaskLabel && (
                                <div className="mt-2 rounded-lg border border-slate-200/80 bg-white/80 px-2.5 py-2">
                                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-600">
                                        <div className="flex items-center gap-2 min-w-0">
                                        <Loader2 size={12} className="animate-spin text-amber-600 shrink-0" />
                                            <span className="truncate">{funnyDrawerTaskLabel}</span>
                                        </div>
                                        {funnyTaskCurrent != null && funnyTaskTotal != null && funnyTaskTotal > 0 && (
                                            <span className="shrink-0 font-mono text-slate-500">
                                                {funnyTaskCurrent}/{funnyTaskTotal}
                                            </span>
                                        )}
                                    </div>
                                    <div className="mt-2 h-1.5 bg-slate-100 rounded-full overflow-hidden relative">
                                        {funnyTaskPercent != null ? (
                                            <div
                                                className="h-full bg-amber-500 transition-all duration-300"
                                                style={{ width: `${funnyTaskPercent}%` }}
                                            />
                                        ) : (
                                            <div className="absolute inset-0 bg-amber-500/15">
                                                <div className="h-full w-1/3 bg-amber-500 animate-[shimmer_1.5s_infinite] relative overflow-hidden">
                                                    <div className="absolute inset-0 bg-white/35 skew-x-12" />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                                <span className="px-2 py-0.5 rounded bg-white/80 border border-amber-200 text-amber-800">
                                    {funnyMoments.length > 0 ? `${funnyMoments.length} saved moments` : 'No saved moments'}
                                </span>
                                {funnyExplainHeaderModelLabel && (
                                    <span className="px-2 py-0.5 rounded bg-purple-100 text-purple-700 border border-purple-200">
                                        {funnyExplainHeaderModelLabel}
                                    </span>
                                )}
                                {latestFunnyExplainAt > 0 && (
                                    <span className="text-slate-600">
                                        {new Date(latestFunnyExplainAt).toLocaleString()}
                                    </span>
                                )}
                                {explainedFunnyMoments.length > 0 && (
                                    <span className="text-slate-500">
                                        {explainedFunnyMoments.length} explained
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-3 space-y-2 bg-slate-50/50">
                            {segments.length > 0 && (
                                <div className="rounded-xl border border-slate-200 bg-white px-3 py-2.5 shadow-sm">
                                    <div className="flex items-center justify-between gap-2">
                                        <button
                                            type="button"
                                            onClick={() => setShowGlobalHumorContext(v => !v)}
                                            className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-600 font-semibold hover:text-slate-800"
                                            title={showGlobalHumorContext ? 'Hide global humor context' : 'Show global humor context'}
                                        >
                                            {showGlobalHumorContext ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                            Global Humor Context
                                        </button>
                                        <div className="flex items-center gap-1.5">
                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                                                Stage 1
                                            </span>
                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-50 text-slate-500 border border-slate-200">
                                                Episode-wide context
                                            </span>
                                        </div>
                                    </div>
                                    {showGlobalHumorContext ? (
                                        video?.humor_context_summary ? (
                                            <div className="mt-1.5">
                                                <p className="text-xs text-slate-700 leading-relaxed whitespace-pre-wrap">
                                                    {getDisplayHumorSummary(video.humor_context_summary)}
                                                </p>
                                                <div className="mt-2 text-[10px] text-slate-500 flex flex-wrap items-center gap-x-2 gap-y-1">
                                                    {video.humor_context_model && (
                                                        <span className="px-1.5 py-0.5 rounded bg-purple-50 text-purple-700">
                                                            {video.humor_context_model}
                                                        </span>
                                                    )}
                                                    {video.humor_context_generated_at && (
                                                        <span>
                                                            {new Date(video.humor_context_generated_at).toLocaleString()}
                                                        </span>
                                                    )}
                                                    <span>Used to inform per-moment explanations</span>
                                                </div>
                                            </div>
                                        ) : explainingFunnyMoments ? (
                                            <div className="mt-1.5 text-xs text-slate-600 flex items-center gap-2">
                                                <Loader2 size={13} className="animate-spin" />
                                                Building episode-wide humor context summary...
                                            </div>
                                        ) : (
                                            <p className="mt-1.5 text-xs text-slate-500">
                                                Run <span className="font-medium">Explain</span> to generate an episode-wide humor context summary, then per-moment joke summaries.
                                            </p>
                                        )
                                    ) : (
                                        <p className="mt-1.5 text-xs text-slate-500">
                                            Optional episode-wide context for callbacks/running bits. Expand if you want extra background while reviewing individual moments.
                                        </p>
                                    )}
                                </div>
                            )}

                            {funnyMoments.length > 0 && (
                                <div className="px-1 pt-1 pb-0.5 flex items-center justify-between">
                                    <div className="text-[10px] uppercase tracking-wide text-slate-600 font-semibold">
                                        Moment Explanations
                                    </div>
                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">
                                        Stage 2
                                    </span>
                                </div>
                            )}

                            {(loadingFunnyMoments || detectingFunnyMoments) && funnyMoments.length === 0 ? (
                                <div className="text-xs text-slate-600 flex items-center gap-2 py-2 px-2">
                                    <Loader2 size={13} className="animate-spin" />
                                    Analyzing episode for laughter...
                                </div>
                            ) : funnyMoments.length > 0 ? (
                                funnyMoments.map((moment) => {
                                    const summaryText = moment.humor_summary ? getDisplayHumorSummary(moment.humor_summary) : '';
                                    const isExpanded = expandedFunnySummaryIds.has(moment.id);
                                    const canExpand = summaryText.length > 220;
                                    return (
                                    <button
                                        key={moment.id}
                                        onClick={() => handleFunnyMomentJump(moment)}
                                        className="w-full text-left rounded-xl border border-amber-200/60 bg-white hover:bg-amber-50/40 px-3 py-2.5 transition-colors shadow-sm"
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <div className="font-mono text-xs text-amber-900">
                                                {formatTime(moment.start_time)} - {formatTime(moment.end_time)}
                                            </div>
                                            <div className="flex items-center gap-2 shrink-0">
                                                <span className="text-[10px] uppercase tracking-wide text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded">
                                                    {moment.source}
                                                </span>
                                                <span className="text-[10px] text-amber-700 font-semibold">
                                                    {(moment.score * 100).toFixed(0)}
                                                </span>
                                            </div>
                                        </div>
                                        {moment.humor_summary ? (
                                            <div className="mt-1.5">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className="text-[10px] uppercase tracking-wide text-slate-600">Likely joke</span>
                                                    {moment.humor_confidence && (
                                                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${moment.humor_confidence === 'high'
                                                            ? 'bg-emerald-100 text-emerald-700'
                                                            : moment.humor_confidence === 'medium'
                                                                ? 'bg-blue-100 text-blue-700'
                                                                : 'bg-slate-100 text-slate-600'
                                                            }`}>
                                                            {moment.humor_confidence}
                                                        </span>
                                                    )}
                                                </div>
                                                <p className={`text-xs text-slate-700 ${isExpanded ? '' : 'line-clamp-4'}`}>
                                                    {summaryText}
                                                </p>
                                                {canExpand && (
                                                    <span
                                                        role="button"
                                                        tabIndex={0}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            toggleFunnySummaryExpanded(moment.id);
                                                        }}
                                                        onKeyDown={(e) => {
                                                            if (e.key === 'Enter' || e.key === ' ') {
                                                                e.preventDefault();
                                                                e.stopPropagation();
                                                                toggleFunnySummaryExpanded(moment.id);
                                                            }
                                                        }}
                                                        className="mt-1 inline-flex text-[11px] font-medium text-amber-700 hover:text-amber-800 underline underline-offset-2 cursor-pointer"
                                                    >
                                                        {isExpanded ? 'Show less' : 'Show more'}
                                                    </span>
                                                )}
                                            </div>
                                        ) : moment.snippet ? (
                                            <p className="mt-1.5 text-xs text-slate-700 line-clamp-3">
                                                {moment.snippet}
                                            </p>
                                        ) : null}
                                    </button>
                                )})
                            ) : (
                                <div className="rounded-lg border border-dashed border-slate-200 bg-white p-4 text-xs text-slate-500">
                                    {segments.length === 0
                                        ? 'Transcript required first. Start transcription to analyze funny moments.'
                                        : 'No funny moments detected yet. Click Find to analyze this episode.'}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                </>
                )}
            </div>

            {/* Speaker Modal */}
            {selectedSpeaker && (
                <SpeakerModal
                    speaker={selectedSpeaker}
                    initialSample={initialSample}
                    onClose={() => { setSelectedSpeaker(null); setInitialSample(null); }}
                    onUpdate={(updatedSpeaker) => {
                        setSelectedSpeaker(updatedSpeaker);
                        // Update segments to reflect new name
                        setSegments(prev => prev.map(s =>
                            s.speaker_id === updatedSpeaker.id
                                ? { ...s, speaker: updatedSpeaker.name }
                                : s
                        ));
                    }}
                    onMerge={() => {
                        // Re-fetch segments to reflect merged speaker assignments
                        if (id) {
                            api.get<TranscriptSegment[]>(`/videos/${id}/segments`).then(res => setSegments(res.data));
                        }
                    }}
                />
            )}

            {/* Assign Speaker Popup (for segments with no speaker) */}
            {assignPopup && (
                <>
                    <div className="fixed inset-0 z-40" onClick={() => setAssignPopup(null)} />
                    <div
                        className="fixed z-50 bg-white rounded-xl shadow-2xl border border-slate-200 w-72 overflow-hidden"
                        style={{ left: assignPopup.x, top: assignPopup.y }}
                    >
                        <div className="p-3 border-b border-slate-100 bg-slate-50">
                            <p className="text-xs font-semibold text-slate-600 mb-2 flex items-center gap-1.5">
                                <GitMerge size={12} className="text-purple-500" />
                                Assign Speaker
                            </p>
                            <div className="relative">
                                <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-400" />
                                <input
                                    type="text"
                                    value={assignSearch}
                                    onChange={(e) => setAssignSearch(e.target.value)}
                                    placeholder="Search speakers..."
                                    className="w-full pl-7 pr-3 py-1.5 text-sm bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-400"
                                    autoFocus
                                />
                            </div>
                        </div>
                        <div className="max-h-52 overflow-y-auto divide-y divide-slate-100">
                            {assignLoading ? (
                                <div className="flex items-center justify-center p-4 text-slate-400">
                                    <Loader2 size={16} className="animate-spin mr-2" /> Loading...
                                </div>
                            ) : (
                                assignSpeakers
                                    .filter(s => s.name.toLowerCase().includes(assignSearch.toLowerCase()))
                                    .map(s => (
                                        <button
                                            key={s.id}
                                            onClick={() => handleAssignSpeaker(s.id)}
                                            className="w-full flex items-center gap-2.5 px-3 py-2 hover:bg-purple-50 transition-colors text-left"
                                        >
                                            {s.thumbnail_path ? (
                                                <img
                                                    src={toApiUrl(s.thumbnail_path)}
                                                    className="w-7 h-7 rounded-full object-cover border border-slate-200"
                                                />
                                            ) : (
                                                <div className="w-7 h-7 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 text-[10px] font-medium border border-slate-200">
                                                    {s.name.charAt(0).toUpperCase()}
                                                </div>
                                            )}
                                            <span className="text-sm font-medium text-slate-700 truncate">{s.name}</span>
                                        </button>
                                    ))
                            )}
                            {!assignLoading && assignSpeakers.filter(s => s.name.toLowerCase().includes(assignSearch.toLowerCase())).length === 0 && (
                                <div className="p-4 text-center text-sm text-slate-400">
                                    {assignSearch ? 'No matching speakers' : 'No speakers available'}
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
