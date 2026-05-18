import type { ReactNode } from 'react';
import { useState, useEffect, useRef, useMemo } from 'react';
import {
    Search, ChevronUp, ChevronDown, X, Clock, FileText, Loader2, Mic,
    Pencil, Plus, XCircle, Play, Pause, Save, Smile, RefreshCw, CheckCircle2,
} from 'lucide-react';
import api from '../../../lib/api';
import type { FunnyMoment, TranscriptSegment } from '../../../types';
import { useTranscriptStore } from '../../../store/useTranscriptStore';
import { useSpeakersTabStore } from '../../../store/useSpeakersTabStore';
import { useVideoStore } from '../../../store/useVideoStore';
import { usePlayerStore } from '../../../store/usePlayerStore';
import { useClipsStore } from '../../../store/useClipsStore';
import { formatTime } from '../../../lib/formatters';
import { useSearchResultScroll } from '../useSearchResultScroll';

const TRANSCRIPT_HIGHLIGHT_LEAD_SECONDS = 0.08;
const TRANSCRIPT_SEGMENT_TRAIL_SECONDS = 0.05;
const TRANSCRIPT_WORD_GAP_BRIDGE_SECONDS = 0.85;
const TRANSCRIPT_MIN_WORD_HIGHLIGHT_SECONDS = 0.12;

type Props = {
    videoId: number;
    onSeek: (time: number) => void;
    onSelectionCreated: (start: number, end: number, defaultTitle: string) => void;
    onRefreshVideo: () => Promise<void>;
    onNavigateToOptimize: () => void;
};

export function TranscriptTab({
    videoId,
    onSeek,
    onSelectionCreated,
    onRefreshVideo,
    onNavigateToOptimize,
}: Props) {
    // Store reads (replacing props)
    const video = useVideoStore((s) => s.video);
    const segments = useVideoStore((s) => s.segments);
    const currentTime = usePlayerStore((s) => s.currentTime);
    const clips = useClipsStore((s) => s.clips);
    const clipSelection = useClipsStore((s) => s.clipSelection);
    const editingClipId = useClipsStore((s) => s.editingClipId);
    const clipEditorDraft = useClipsStore((s) => s.clipEditorDraft);
    const activeEditingClip = editingClipId != null
        ? (clips.find((c) => c.id === editingClipId) || null)
        : null;
    const showClipEditorMain = !!activeEditingClip && !!clipEditorDraft;
    const transcriptRef = useRef<HTMLDivElement>(null);
    const lastAutoScrollSegIdRef = useRef<number | null>(null);
    const [startingTranscription, setStartingTranscription] = useState(false);

    // Store state
    const searchQuery = useTranscriptStore((s) => s.searchQuery);
    const deepLinkedSegmentId = useTranscriptStore((s) => s.deepLinkedSegmentId);
    const searchMatchIndex = useTranscriptStore((s) => s.searchMatchIndex);
    const followPlayback = useTranscriptStore((s) => s.followPlayback);
    const detectingFunnyMoments = useTranscriptStore((s) => s.detectingFunnyMoments);
    const funnyDrawerOpen = useTranscriptStore((s) => s.funnyDrawerOpen);
    const explainingFunnyMoments = useTranscriptStore((s) => s.explainingFunnyMoments);
    const showGlobalHumorContext = useTranscriptStore((s) => s.showGlobalHumorContext);
    const expandedFunnySummaryIds = useTranscriptStore((s) => s.expandedFunnySummaryIds);
    const funnyTaskProgress = useTranscriptStore((s) => s.funnyTaskProgress);
    const funnyMoments = useTranscriptStore((s) => s.funnyMoments);
    const loadingFunnyMoments = useTranscriptStore((s) => s.loadingFunnyMoments);
    const editingSegmentId = useTranscriptStore((s) => s.editingSegmentId);
    const editingSegmentWords = useTranscriptStore((s) => s.editingSegmentWords);
    const editingLoopSegment = useTranscriptStore((s) => s.editingLoopSegment);
    const savingSegmentEdit = useTranscriptStore((s) => s.savingSegmentEdit);
    const transcriptQuality = useTranscriptStore((s) => s.transcriptQuality);
    const loadingTranscriptQuality = useTranscriptStore((s) => s.loadingTranscriptQuality);
    const transcriptQualityError = useTranscriptStore((s) => s.transcriptQualityError);

    const setSearchQuery = useTranscriptStore((s) => s.setSearchQuery);
    const setSearchMatchIndex = useTranscriptStore((s) => s.setSearchMatchIndex);
    const setFollowPlayback = useTranscriptStore((s) => s.setFollowPlayback);
    const setFunnyDrawerOpen = useTranscriptStore((s) => s.setFunnyDrawerOpen);
    const setShowGlobalHumorContext = useTranscriptStore((s) => s.setShowGlobalHumorContext);
    const setExpandedFunnySummaryIds = useTranscriptStore((s) => s.setExpandedFunnySummaryIds);
    const setEditingSegmentId = useTranscriptStore((s) => s.setEditingSegmentId);
    const setEditingSegmentWords = useTranscriptStore((s) => s.setEditingSegmentWords);
    const setEditingLoopSegment = useTranscriptStore((s) => s.setEditingLoopSegment);

    // Derived playback time
    const transcriptPlaybackTime = currentTime + TRANSCRIPT_HIGHLIGHT_LEAD_SECONDS;

    const { searchLower, filteredSegments, totalMatches } = useSearchResultScroll({ segments });

    // Word-level timing for each segment
    const parseSegmentWords = (
        seg: TranscriptSegment,
    ): Array<{ start: number; end: number; displayEnd: number; word: string }> => {
        if (!seg.words) return [];
        let words = [] as Array<{ start: number; end: number; word: string }>;
        try {
            const parsed = JSON.parse(seg.words);
            if (!Array.isArray(parsed)) return [];
            words = parsed
                .map((w: any) => ({
                    start: Number(w?.start),
                    end: Number.isFinite(Number(w?.end)) ? Number(w?.end) : Number(w?.start),
                    word: String(w?.word || '').trim(),
                }))
                .filter((w) => Number.isFinite(w.start) && Number.isFinite(w.end) && !!w.word);
        } catch {
            return [];
        }
        if (words.length === 0) return [];

        words.sort((a, b) => a.start - b.start);
        const segStart = Number(seg.start_time);
        const segEnd = Number(seg.end_time);
        const segDuration =
            Number.isFinite(segStart) && Number.isFinite(segEnd) ? Math.max(0.01, segEnd - segStart) : 0.01;

        let minStart = Math.min(...words.map((w) => w.start));
        let maxEnd = Math.max(...words.map((w) => w.end));

        const looksMsAbsolute = Number.isFinite(segEnd) && maxEnd > Math.max(segEnd * 5, 1000);
        const looksMsRelative =
            minStart >= -0.5 &&
            minStart < Math.max(2, segDuration * 2) &&
            maxEnd > Math.max(1000, segDuration * 20);

        if (looksMsAbsolute || looksMsRelative) {
            words = words.map((w) => ({ ...w, start: w.start / 1000, end: w.end / 1000 }));
            minStart = Math.min(...words.map((w) => w.start));
            maxEnd = Math.max(...words.map((w) => w.end));
        }

        const looksRelative = minStart >= -0.5 && maxEnd <= segDuration + 1.5;
        if (looksRelative && Number.isFinite(segStart)) {
            words = words.map((w) => ({ ...w, start: w.start + segStart, end: w.end + segStart }));
        }

        if (Number.isFinite(segStart) && Number.isFinite(segEnd) && segEnd > segStart) {
            words = words
                .map((w) => {
                    const s = Math.max(segStart, w.start);
                    const e = Math.min(segEnd, Math.max(w.end, s));
                    return { ...w, start: s, end: e };
                })
                .filter((w) => w.end >= w.start);
        }

        words.sort((a, b) => a.start - b.start);
        return words.map((w, idx) => {
            const next = words[idx + 1];
            const naturalEnd = Math.max(w.end, w.start);
            let displayEnd = naturalEnd;

            const minHighlightEnd = w.start + TRANSCRIPT_MIN_WORD_HIGHLIGHT_SECONDS;
            if (displayEnd < minHighlightEnd) {
                if (next && next.start > w.start) {
                    displayEnd = Math.min(minHighlightEnd, next.start);
                } else if (Number.isFinite(segEnd)) {
                    displayEnd = Math.min(segEnd, minHighlightEnd);
                } else {
                    displayEnd = minHighlightEnd;
                }
            }

            if (next) {
                const gapToNext = next.start - naturalEnd;
                if (gapToNext > 0 && gapToNext <= TRANSCRIPT_WORD_GAP_BRIDGE_SECONDS) {
                    displayEnd = next.start;
                }
            } else if (Number.isFinite(segEnd)) {
                displayEnd = Math.min(segEnd, Math.max(displayEnd, naturalEnd + TRANSCRIPT_SEGMENT_TRAIL_SECONDS));
            }
            if (displayEnd <= w.start) displayEnd = w.start + 0.01;
            return { ...w, displayEnd };
        });
    };

    const normalizedWordsBySegmentId = useMemo(() => {
        const map = new Map<number, Array<{ start: number; end: number; displayEnd: number; word: string }>>();
        for (const seg of segments) {
            if (typeof seg.id !== 'number') continue;
            map.set(seg.id, parseSegmentWords(seg));
        }
        return map;
    }, [segments]);

    // Placeholder transcript labels
    const placeholderTranscriptSourceLabel = (() => {
        const source = String(video?.transcript_source || '').toLowerCase();
        if (source === 'youtube_auto_captions') return 'YouTube auto-captions';
        if (source === 'youtube_subtitles') return 'YouTube captions';
        if (source === 'tiktok_auto_captions') return 'TikTok auto-captions';
        if (source === 'tiktok_subtitles') return 'TikTok captions';
        return 'Preliminary captions';
    })();
    const placeholderTranscriptLanguage = String(video?.transcript_language || '').trim();
    const isPlaceholderTranscript = !!video?.transcript_is_placeholder && segments.length > 0;
    const accessRestrictionReason = String(video?.access_restriction_reason || '').trim();
    const accessRestrictionLabel = (() => {
        const reason = accessRestrictionReason.toLowerCase();
        if (reason.includes('members-only') || reason.includes('members only')) return 'Members-only video';
        if (reason.includes('private')) return 'Private video';
        if (reason.includes('sign in') || reason.includes('auth')) return 'Sign-in required';
        return 'Access restricted';
    })();

    // Transcript quality / snapshot card
    const recommendedOptimizationTier = String(transcriptQuality?.recommended_tier || 'none');
    const recommendedOptimizationLabel =
        recommendedOptimizationTier === 'low_risk_repair'
            ? 'Low-Risk Repair'
            : recommendedOptimizationTier === 'diarization_rebuild'
                ? 'Diarization Rebuild'
                : recommendedOptimizationTier === 'full_retranscription'
                    ? 'Full Retranscription'
                    : recommendedOptimizationTier === 'manual_review'
                        ? 'Manual Review'
                        : 'No Automatic Optimization';

    // Funny moments computed values (post-render, so placed before return)
    const explainedFunnyMoments = funnyMoments.filter((m) => !!m.humor_summary);
    const explainedModelNames = Array.from(
        new Set(explainedFunnyMoments.map((m) => (m.humor_model || '').trim()).filter(Boolean)),
    );
    const latestFunnyExplainAt = explainedFunnyMoments
        .map((m) => (m.humor_explained_at ? new Date(m.humor_explained_at).getTime() : 0))
        .filter((ts) => Number.isFinite(ts) && ts > 0)
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
    const funnyTaskCurrent =
        typeof funnyTaskProgress?.current === 'number' ? funnyTaskProgress.current : null;
    const funnyTaskTotal =
        typeof funnyTaskProgress?.total === 'number' ? funnyTaskProgress.total : null;
    const funnyDrawerTaskLabel = detectingFunnyMoments
        ? (funnyTaskProgress?.message || 'Scanning transcript/audio for laughter and funny moments...')
        : explainingFunnyMoments
            ? (funnyTaskProgress?.message ||
                (hasExistingFunnyExplanations
                    ? 'Re-generating global humor context and joke explanations...'
                    : 'Generating global humor context and joke explanations...'))
            : null;

    // Auto-scroll transcript to active segment during playback
    useEffect(() => {
        if (!followPlayback || segments.length === 0 || clipSelection || searchQuery || editingSegmentId) return;
        const activeSeg = segments.find(
            (s) =>
                transcriptPlaybackTime >= s.start_time &&
                transcriptPlaybackTime < s.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS,
        );
        if (activeSeg && lastAutoScrollSegIdRef.current !== activeSeg.id) {
            lastAutoScrollSegIdRef.current = activeSeg.id;
            const el = document.getElementById(`seg-${activeSeg.id}`);
            if (el) el.scrollIntoView({ behavior: 'auto', block: 'center' });
        }
    }, [transcriptPlaybackTime, followPlayback, segments, clipSelection, searchQuery, editingSegmentId]);

    // Poll funny task progress while detection/explanation is running
    useEffect(() => {
        if (!detectingFunnyMoments && !explainingFunnyMoments) return;
        void useTranscriptStore.getState().fetchFunnyTaskProgress(videoId);
        const timer = window.setInterval(
            () => void useTranscriptStore.getState().fetchFunnyTaskProgress(videoId),
            700,
        );
        return () => window.clearInterval(timer);
    }, [videoId, detectingFunnyMoments, explainingFunnyMoments]);

    // Handlers
    const handleMouseUp = () => {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) return;

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
            const start = Math.min(t1, parseFloat(endEl.dataset.start!));
            const end = Math.max(t2, parseFloat(startEl.dataset.end!));
            let text = sel.toString().replace(/\s+/g, ' ').trim();
            if (text.length > 60) text = text.substring(0, 60) + '...';
            onSelectionCreated(start, end, text);
        }
    };

    const pausePlayer = () => {
        usePlayerStore.getState().player?.pauseVideo?.();
    };

    const handleSpeakerClick = (speakerId: number, seg?: TranscriptSegment) => {
        pausePlayer();
        void useSpeakersTabStore.getState().openSpeaker(speakerId, seg, video);
    };

    const handleUnknownSpeakerClick = (segmentId: number, e: React.MouseEvent) => {
        e.stopPropagation();
        pausePlayer();
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        useSpeakersTabStore.getState().openAssignPopup(segmentId, rect.left + window.scrollX, rect.bottom + window.scrollY);
    };

    const beginSegmentEdit = (seg: TranscriptSegment) => {
        pausePlayer();
        setEditingSegmentId(seg.id);
        const baseWords = parseSegmentWords(seg)
            .map((w) => w.word.trim())
            .filter(Boolean);
        const fallbackWords = String(seg.text || '')
            .split(/\s+/)
            .map((w) => w.trim())
            .filter(Boolean);
        const tokens = baseWords.length > 0 ? baseWords : fallbackWords;
        setEditingSegmentWords(tokens);
        setEditingLoopSegment(false);
    };

    const updateEditingWord = (index: number, value: string) => {
        setEditingSegmentWords((prev) => {
            const next = [...prev];
            next[index] = value;
            return next;
        });
    };

    const removeEditingWord = (index: number) => {
        setEditingSegmentWords((prev) => prev.filter((_, i) => i !== index));
    };

    const addEditingWord = () => {
        setEditingSegmentWords((prev) => [...prev, '']);
    };

    const saveSegmentEdit = async (segmentId: number) => {
        await useTranscriptStore.getState().saveSegmentEdit(
            segmentId,
            (updatedSegment) => {
                useVideoStore.getState().setSegments((prev) =>
                    prev.map((s) =>
                        s.id === segmentId
                            ? { ...s, text: updatedSegment.text, words: updatedSegment.words ?? s.words }
                            : s,
                    ),
                );
            },
            pausePlayer,
        );
    };

    const highlightText = (text: string): string | ReactNode => {
        if (!searchLower) return text;
        const parts: (string | ReactNode)[] = [];
        let lastIdx = 0;
        let i = text.toLowerCase().indexOf(searchLower, lastIdx);
        while (i !== -1) {
            if (i > lastIdx) parts.push(text.slice(lastIdx, i));
            parts.push(
                <mark key={i} className="bg-yellow-200 text-yellow-900 rounded-sm">
                    {text.slice(i, i + searchLower.length)}
                </mark>,
            );
            lastIdx = i + searchLower.length;
            i = text.toLowerCase().indexOf(searchLower, lastIdx);
        }
        if (lastIdx < text.length) parts.push(text.slice(lastIdx));
        return <>{parts}</>;
    };

    // Funny moments helpers
    const scrollToTime = (time: number) => {
        const target =
            segments.find(
                (s) => time >= s.start_time && time < s.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS,
            ) ||
            segments.find((s) => s.start_time >= time) ||
            segments[segments.length - 1];
        if (target) {
            const el = document.getElementById(`seg-${target.id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };

    const handleFunnyMomentJump = (moment: FunnyMoment) => {
        onSeek(moment.start_time);
        scrollToTime(moment.start_time);
    };

    const handleDetectFunnyMoments = (force = true) => {
        void useTranscriptStore.getState().detectFunnyMoments(videoId, force);
    };

    const handleExplainFunnyMoments = (force = false) => {
        void useTranscriptStore.getState().explainFunnyMoments(videoId, force, onRefreshVideo);
    };

    const getDisplayHumorSummary = (raw?: string) => {
        if (!raw) return raw;
        let text = raw.trim();
        text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
        text = text.replace(/^\s*```(?:thinking|reasoning)\s*[\s\S]*?```\s*/i, '').trim();
        text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
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
                if (m && m.index > 40 && (cueIndex === -1 || m.index < cueIndex)) cueIndex = m.index;
            }
            if (cueIndex > 0) text = text.slice(cueIndex).replace(/^[:\-\s]+/, '').trim();
        }
        return text;
    };

    const toggleFunnySummaryExpanded = (momentId: number) => {
        setExpandedFunnySummaryIds((prev) => {
            const next = new Set(prev);
            if (next.has(momentId)) next.delete(momentId);
            else next.add(momentId);
            return next;
        });
    };

    const renderSnapshotCard = () => (
        <div className="rounded-2xl border px-4 py-4 shadow-sm border-slate-200 bg-slate-50/90">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        Transcript Status
                    </div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">
                        Optimization has its own workbench now
                    </div>
                    <div className="mt-1 text-xs leading-5 text-slate-600">
                        Use the Optimize tab for repair, diarization rebuild, retranscription, rollback, and transcript
                        benchmarking.
                    </div>
                </div>
                <div className="shrink-0">
                    <button
                        type="button"
                        onClick={onNavigateToOptimize}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-200 bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700"
                    >
                        <CheckCircle2 size={14} />
                        Open Optimize
                    </button>
                </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
                {loadingTranscriptQuality ? (
                    <span className="inline-flex items-center gap-2 rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                        <Loader2 size={13} className="animate-spin" />
                        Evaluating
                    </span>
                ) : transcriptQuality ? (
                    <>
                        <span className="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                            {recommendedOptimizationLabel}
                        </span>
                        <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                            Score {transcriptQuality.quality_score.toFixed(1)}
                        </span>
                        <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                            {String(transcriptQuality.quality_profile || 'unknown').replaceAll('_', ' ')}
                        </span>
                    </>
                ) : (
                    <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                        {transcriptQualityError || 'No quality assessment yet'}
                    </span>
                )}
            </div>
            {transcriptQuality && (
                <div className="mt-3 grid gap-2 text-xs text-slate-600 sm:grid-cols-3">
                    <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                        Unknown speaker rate:{' '}
                        {Number(transcriptQuality.metrics?.unknown_speaker_rate || 0).toFixed(2)}
                    </div>
                    <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                        Micro segments: {Number(transcriptQuality.metrics?.micro_segment_count || 0)}
                    </div>
                    <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                        Interruptions: {Number(transcriptQuality.metrics?.same_speaker_interruptions || 0)}
                    </div>
                </div>
            )}
            {transcriptQuality?.reasons?.[0] && (
                <div className="mt-3 text-xs leading-5 text-slate-500">{transcriptQuality.reasons[0]}</div>
            )}
        </div>
    );

    return (
        <>
            {/* Transcript Sidebar */}
            <div className="h-full flex flex-col">
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
                                        onClick={() =>
                                            setSearchMatchIndex(
                                                (prev) => (prev - 1 + totalMatches) % totalMatches,
                                            )
                                        }
                                        disabled={totalMatches === 0}
                                        className="p-0.5 text-slate-400 hover:text-slate-600 disabled:opacity-30 rounded"
                                    >
                                        <ChevronUp size={14} />
                                    </button>
                                    <button
                                        onClick={() =>
                                            setSearchMatchIndex((prev) => (prev + 1) % totalMatches)
                                        }
                                        disabled={totalMatches === 0}
                                        className="p-0.5 text-slate-400 hover:text-slate-600 disabled:opacity-30 rounded"
                                    >
                                        <ChevronDown size={14} />
                                    </button>
                                    <button
                                        onClick={() => setSearchQuery('')}
                                        className="p-0.5 text-slate-400 hover:text-slate-600 rounded"
                                    >
                                        <X size={14} />
                                    </button>
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
                        {isPlaceholderTranscript && (
                            <div className="mt-2 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                                <Clock size={14} className="mt-0.5 shrink-0 text-amber-600" />
                                <div>
                                    <div className="font-semibold">
                                        {placeholderTranscriptSourceLabel}
                                        {placeholderTranscriptLanguage
                                            ? ` (${placeholderTranscriptLanguage})`
                                            : ''}
                                    </div>
                                    <div className="mt-0.5 text-amber-800/90">
                                        This searchable transcript is a temporary placeholder and will be replaced
                                        automatically after local transcription and diarization finish.
                                    </div>
                                </div>
                            </div>
                        )}
                        {segments.length > 0 && (
                            <div className="mt-3">{renderSnapshotCard()}</div>
                        )}
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
                            {video?.access_restricted ? (
                                <div className="w-full max-w-md rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm text-slate-700">
                                    <div className="font-semibold text-slate-800">{accessRestrictionLabel}</div>
                                    <div className="mt-1 text-xs leading-relaxed text-slate-500">
                                        {accessRestrictionReason ||
                                            'This episode is not accessible with the current YouTube session, so it will be skipped instead of being downloaded or processed.'}
                                    </div>
                                </div>
                            ) : video &&
                                !video.processed &&
                                video.status !== 'queued' &&
                                video.status !== 'running' &&
                                video.status !== 'downloading' &&
                                video.status !== 'transcribing' &&
                                video.status !== 'diarizing' ? (
                                <>
                                    <p className="text-xs text-slate-400 mb-5 text-center">
                                        Start a transcription job to generate the transcript for this episode.
                                    </p>
                                    <button
                                        onClick={async () => {
                                            if (!video) return;
                                            setStartingTranscription(true);
                                            try {
                                                await api.post(`/videos/${video.id}/process`);
                                                // Refresh video metadata so status updates
                                                await onRefreshVideo();
                                            } catch (e: any) {
                                                console.error('Failed to start transcription:', e);
                                                alert(e?.response?.data?.detail || 'Failed to start transcription');
                                            } finally {
                                                setStartingTranscription(false);
                                            }
                                        }}
                                        disabled={startingTranscription}
                                        className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all font-medium text-sm disabled:opacity-50"
                                    >
                                        {startingTranscription ? (
                                            <Loader2 size={16} className="animate-spin" />
                                        ) : (
                                            <Mic size={16} />
                                        )}
                                        {startingTranscription ? 'Starting...' : 'Start Transcription'}
                                    </button>
                                </>
                            ) : video &&
                                (video.status === 'queued' ||
                                    video.status === 'running' ||
                                    video.status === 'downloading' ||
                                    video.status === 'transcribing' ||
                                    video.status === 'diarizing') ? (
                                <div className="flex items-center gap-2 mt-2 text-xs text-blue-500">
                                    <Loader2 size={14} className="animate-spin" />
                                    <span>
                                        {video.status === 'queued'
                                            ? 'Transcription queued'
                                            : video.status === 'diarizing'
                                                ? 'Diarization in progress'
                                                : 'Transcription in progress'}
                                        ...
                                    </span>
                                </div>
                            ) : (
                                <p className="text-xs text-slate-400">
                                    This episode has not been transcribed yet.
                                </p>
                            )}
                        </div>
                    ) : (
                        filteredSegments.map((seg, filteredIdx) => {
                            const isActiveSegment =
                                transcriptPlaybackTime >= seg.start_time &&
                                transcriptPlaybackTime < seg.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS;
                            const isActiveMatch = searchLower && filteredIdx === searchMatchIndex;
                            const isDeepLinkedSegment = !searchLower && deepLinkedSegmentId === seg.id;
                            const wordsFn =
                                typeof seg.id === 'number'
                                    ? (normalizedWordsBySegmentId.get(seg.id) || [])
                                    : parseSegmentWords(seg);

                            return (
                                <div
                                    key={seg.id}
                                    id={`seg-${seg.id}`}
                                    data-start={seg.start_time}
                                    data-end={seg.end_time}
                                    className={`p-3 rounded-lg text-sm transition-colors cursor-pointer border relative group ${isActiveMatch
                                        ? 'bg-yellow-50 border-yellow-300 ring-1 ring-yellow-200 shadow-sm'
                                        : isDeepLinkedSegment
                                            ? 'bg-amber-50 border-amber-300 ring-1 ring-amber-200 shadow-sm'
                                            : isActiveSegment
                                                ? 'bg-blue-50 border-blue-200 shadow-sm ring-1 ring-blue-100'
                                                : 'bg-white border-transparent hover:border-slate-200 hover:bg-white'
                                        }`}
                                    onClick={() => {
                                        if (window.getSelection()?.toString().length === 0) {
                                            onSeek(seg.start_time);
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
                                                {seg.speaker || 'Unknown'}
                                            </span>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    beginSegmentEdit(seg);
                                                }}
                                                className="opacity-100 lg:opacity-0 lg:group-hover:opacity-100 p-1 rounded text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition"
                                                title="Edit transcript text"
                                            >
                                                <Pencil size={12} />
                                            </button>
                                        </div>
                                        <span className="font-mono shrink-0">
                                            {new Date(seg.start_time * 1000).toISOString().substr(14, 5)}
                                        </span>
                                    </div>
                                    {editingSegmentId === seg.id ? (
                                        <div className="space-y-2">
                                            <div
                                                onClick={(e) => e.stopPropagation()}
                                                className="rounded-lg border border-blue-200 bg-white p-2.5 space-y-2"
                                            >
                                                <div className="flex flex-wrap items-center gap-1.5">
                                                    {editingSegmentWords.map((word, idx) => (
                                                        <div
                                                            key={`${seg.id}-${idx}`}
                                                            className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-1.5 py-1"
                                                        >
                                                            <input
                                                                value={word}
                                                                onChange={(e) =>
                                                                    updateEditingWord(idx, e.target.value)
                                                                }
                                                                className="min-w-[2.5ch] max-w-[22ch] bg-transparent text-sm text-slate-700 focus:outline-none"
                                                                style={{
                                                                    width: `${Math.max(2.5, Math.min(22, (word || '').length + 1.5))}ch`,
                                                                }}
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
                                                            pausePlayer();
                                                        }
                                                    }}
                                                    className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border ${editingLoopSegment
                                                        ? 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100'
                                                        : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
                                                        }`}
                                                    title="Loop this segment while editing"
                                                >
                                                    {editingLoopSegment ? (
                                                        <Pause size={12} />
                                                    ) : (
                                                        <Play size={12} />
                                                    )}
                                                    {editingLoopSegment ? 'Stop Loop' : 'Loop Segment'}
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setEditingLoopSegment(false);
                                                        pausePlayer();
                                                        setEditingSegmentId(null);
                                                        setEditingSegmentWords([]);
                                                    }}
                                                    className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-slate-100 text-slate-600 hover:bg-slate-200"
                                                >
                                                    <XCircle size={12} /> Cancel
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        void saveSegmentEdit(seg.id);
                                                    }}
                                                    disabled={savingSegmentEdit}
                                                    className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                                                >
                                                    {savingSegmentEdit ? (
                                                        <Loader2 size={12} className="animate-spin" />
                                                    ) : (
                                                        <Save size={12} />
                                                    )}
                                                    Save
                                                </button>
                                            </div>
                                        </div>
                                    ) : (
                                        <p className="text-slate-700 leading-relaxed whitespace-pre-wrap break-words">
                                            {wordsFn.length > 0 ? (
                                                wordsFn.map((w: any, idx: number) => {
                                                    const isWordActive =
                                                        transcriptPlaybackTime >= w.start &&
                                                        transcriptPlaybackTime < (w.displayEnd ?? w.end);
                                                    return (
                                                        <span
                                                            key={idx}
                                                            className={`inline align-baseline transition-colors duration-100 ${isWordActive ? 'bg-blue-200/80 text-blue-900 rounded-sm' : ''}`}
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                onSeek(w.start);
                                                            }}
                                                        >
                                                            {searchLower ? highlightText(w.word) : w.word}
                                                            {idx < wordsFn.length - 1 ? ' ' : ''}
                                                        </span>
                                                    );
                                                })
                                            ) : searchLower ? (
                                                highlightText(seg.text)
                                            ) : (
                                                seg.text
                                            )}
                                        </p>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Funny Moments Overlay (absolutely positioned within the sidebar content area) */}
            {!showClipEditorMain && (
                <>
                    <button
                        onClick={() => setFunnyDrawerOpen((v) => !v)}
                        className="absolute right-4 bottom-4 z-20 flex items-center gap-2 px-3 py-2 rounded-xl border border-amber-200 bg-white/95 hover:bg-white shadow-lg text-amber-800 text-sm font-medium"
                        title="Open funny moments drawer"
                    >
                        <Smile size={15} className="text-amber-600" />
                        {funnyDrawerOpen ? 'Hide Funny Moments' : 'Funny Moments'}
                    </button>

                    <div
                        className={`absolute right-4 top-4 bottom-20 z-20 w-[380px] max-w-[calc(100%-2rem)] rounded-2xl border border-slate-200 bg-white/95 backdrop-blur-sm shadow-2xl overflow-hidden transition-transform duration-200 ${funnyDrawerOpen ? 'translate-x-0' : 'translate-x-[110%]'}`}
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
                                            title={
                                                video?.processed
                                                    ? 'Analyze transcript/audio for funny moments'
                                                    : 'Transcribe the episode first'
                                            }
                                        >
                                            {detectingFunnyMoments ? (
                                                <Loader2 size={13} className="animate-spin" />
                                            ) : (
                                                <Smile size={13} />
                                            )}
                                            {funnyMoments.length > 0 ? 'Rescan' : 'Find'}
                                        </button>
                                        <button
                                            onClick={() => handleExplainFunnyMoments(true)}
                                            disabled={explainingFunnyMoments || funnyMoments.length === 0}
                                            className="h-10 px-2 rounded-lg text-xs font-medium bg-purple-100 text-purple-700 hover:bg-purple-200 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                                            title="Force-regenerate global humor context and moment explanations with the current LLM provider/model"
                                        >
                                            {explainingFunnyMoments ? (
                                                <Loader2 size={13} className="animate-spin" />
                                            ) : hasExistingFunnyExplanations ? (
                                                <RefreshCw size={13} />
                                            ) : (
                                                <Search size={13} />
                                            )}
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
                                            {funnyTaskCurrent != null &&
                                                funnyTaskTotal != null &&
                                                funnyTaskTotal > 0 && (
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
                                        {funnyMoments.length > 0
                                            ? `${funnyMoments.length} saved moments`
                                            : 'No saved moments'}
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
                                                onClick={() => setShowGlobalHumorContext((v) => !v)}
                                                className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-600 font-semibold hover:text-slate-800"
                                                title={
                                                    showGlobalHumorContext
                                                        ? 'Hide global humor context'
                                                        : 'Show global humor context'
                                                }
                                            >
                                                {showGlobalHumorContext ? (
                                                    <ChevronUp size={12} />
                                                ) : (
                                                    <ChevronDown size={12} />
                                                )}
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
                                                                {new Date(
                                                                    video.humor_context_generated_at,
                                                                ).toLocaleString()}
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
                                                    Run <span className="font-medium">Explain</span> to generate an
                                                    episode-wide humor context summary, then per-moment joke summaries.
                                                </p>
                                            )
                                        ) : (
                                            <p className="mt-1.5 text-xs text-slate-500">
                                                Optional episode-wide context for callbacks/running bits. Expand if you
                                                want extra background while reviewing individual moments.
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
                                        const summaryText = moment.humor_summary
                                            ? getDisplayHumorSummary(moment.humor_summary)
                                            : '';
                                        const isExpanded = expandedFunnySummaryIds.has(moment.id);
                                        const canExpand = (summaryText?.length ?? 0) > 220;
                                        return (
                                            <button
                                                key={moment.id}
                                                onClick={() => handleFunnyMomentJump(moment)}
                                                className="w-full text-left rounded-xl border border-amber-200/60 bg-white hover:bg-amber-50/40 px-3 py-2.5 transition-colors shadow-sm"
                                            >
                                                <div className="flex items-center justify-between gap-2">
                                                    <div className="font-mono text-xs text-amber-900">
                                                        {formatTime(moment.start_time)} -{' '}
                                                        {formatTime(moment.end_time)}
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
                                                            <span className="text-[10px] uppercase tracking-wide text-slate-600">
                                                                Likely joke
                                                            </span>
                                                            {moment.humor_confidence && (
                                                                <span
                                                                    className={`text-[10px] px-1.5 py-0.5 rounded ${moment.humor_confidence === 'high'
                                                                        ? 'bg-emerald-100 text-emerald-700'
                                                                        : moment.humor_confidence === 'medium'
                                                                            ? 'bg-blue-100 text-blue-700'
                                                                            : 'bg-slate-100 text-slate-600'
                                                                        }`}
                                                                >
                                                                    {moment.humor_confidence}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <p
                                                            className={`text-xs text-slate-700 ${isExpanded ? '' : 'line-clamp-4'}`}
                                                        >
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
                                        );
                                    })
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
        </>
    );
}
