import { useEffect, useRef, type ReactNode } from 'react';
import { Loader2, Save } from 'lucide-react';
import { formatTime } from '../../../lib/formatters';
import type { Clip, TranscriptSegment, Video } from '../../../types';
import { useClipsStore, type ClipCropTarget, type ClipToken } from '../../../store/useClipsStore';

type Props = {
    activeEditingClip: Clip;
    currentTime: number;
    video: Video | null;
    segments: TranscriptSegment[];
    playerNode: ReactNode;
    onSeek: (seconds: number) => void;
};

const TRANSCRIPT_WORD_GAP_BRIDGE_SECONDS = 0.85;
const TRANSCRIPT_MIN_WORD_HIGHLIGHT_SECONDS = 0.12;
const TRANSCRIPT_SEGMENT_TRAIL_SECONDS = 0.05;

function parseSegmentWords(seg: TranscriptSegment): Array<{ start: number; end: number; displayEnd: number; word: string }> {
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
    const segDuration = Number.isFinite(segStart) && Number.isFinite(segEnd) ? Math.max(0.01, segEnd - segStart) : 0.01;

    let minStart = Math.min(...words.map(w => w.start));
    let maxEnd = Math.max(...words.map(w => w.end));

    const looksMsAbsolute = Number.isFinite(segEnd) && maxEnd > Math.max(segEnd * 5, 1000);
    const looksMsRelative = minStart >= -0.5 && minStart < Math.max(2, segDuration * 2) && maxEnd > Math.max(1000, segDuration * 20);
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
                const e = Math.min(segEnd, Math.max(w.end, s));
                return { ...w, start: s, end: e };
            })
            .filter(w => w.end >= w.start);
    }

    words.sort((a, b) => a.start - b.start);
    return words.map((w, idx) => {
        const next = words[idx + 1];
        const naturalEnd = Math.max(w.end, w.start);
        let displayEnd = naturalEnd;
        const minHighlightEnd = w.start + TRANSCRIPT_MIN_WORD_HIGHLIGHT_SECONDS;
        if (displayEnd < minHighlightEnd) {
            if (next && next.start > w.start) displayEnd = Math.min(minHighlightEnd, next.start);
            else if (Number.isFinite(segEnd)) displayEnd = Math.min(segEnd, minHighlightEnd);
            else displayEnd = minHighlightEnd;
        }
        if (next) {
            const gapToNext = next.start - naturalEnd;
            if (gapToNext > 0 && gapToNext <= TRANSCRIPT_WORD_GAP_BRIDGE_SECONDS) displayEnd = next.start;
        } else if (Number.isFinite(segEnd)) {
            displayEnd = Math.min(segEnd, Math.max(displayEnd, naturalEnd + TRANSCRIPT_SEGMENT_TRAIL_SECONDS));
        }
        if (displayEnd <= w.start) displayEnd = w.start + 0.01;
        return { ...w, displayEnd };
    });
}

export function ClipEditorWorkspace({
    activeEditingClip,
    currentTime,
    video,
    segments,
    playerNode,
    onSeek,
}: Props) {
    const cropPreviewRef = useRef<HTMLDivElement>(null);
    const clipTimelineRef = useRef<HTMLDivElement>(null);
    const clipEditorDraft = useClipsStore((s) => s.clipEditorDraft);
    const clipEditorTokens = useClipsStore((s) => s.clipEditorTokens);
    const clipEditorRemovedWordKeys = useClipsStore((s) => s.clipEditorRemovedWordKeys);
    const clipEditorCropTarget = useClipsStore((s) => s.clipEditorCropTarget);
    const clipEditorDragRect = useClipsStore((s) => s.clipEditorDragRect);
    const clipTimelineDrag = useClipsStore((s) => s.clipTimelineDrag);
    const savingClipEdit = useClipsStore((s) => s.savingClipEdit);
    const setClipEditorDraft = useClipsStore((s) => s.setClipEditorDraft);
    const setClipEditorTokens = useClipsStore((s) => s.setClipEditorTokens);
    const setClipEditorRemovedWordKeys = useClipsStore((s) => s.setClipEditorRemovedWordKeys);
    const setClipEditorCropTarget = useClipsStore((s) => s.setClipEditorCropTarget);
    const setClipEditorDragRect = useClipsStore((s) => s.setClipEditorDragRect);
    const setClipTimelineDrag = useClipsStore((s) => s.setClipTimelineDrag);

    if (!clipEditorDraft) return null;

    const isPortraitSplit = String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && !!clipEditorDraft.portrait_split_enabled;

    const onCancelClipEdit = () => useClipsStore.getState().cancelClipEdit();
    const onSaveClipEdit = (clipId: number) => void useClipsStore.getState().saveClipEdit(clipId);
    const onUpdateClipDraftField = (field: keyof Clip, value: any) => setClipEditorDraft(prev => ({ ...(prev || {}), [field]: value }));

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
        const out: ClipToken[] = [];
        for (const seg of segments) {
            if (seg.end_time <= clipStart || seg.start_time >= clipEnd) continue;
            let words: Array<{ start: number; end: number; word: string }> = parseSegmentWords(seg);
            if (words.length === 0 && seg.text?.trim()) {
                const textWords = seg.text.trim().split(/\s+/);
                const duration = Math.max(0.05, seg.end_time - seg.start_time);
                const step = duration / Math.max(textWords.length, 1);
                words = textWords.map((w, idx) => ({ start: seg.start_time + (idx * step), end: seg.start_time + ((idx + 1) * step), word: w }));
            }
            for (const w of words) {
                if (w.end <= clipStart || w.start >= clipEnd) continue;
                const s = Math.max(clipStart, w.start);
                const e = Math.min(clipEnd, w.end);
                if (e <= s + 0.005) continue;
                out.push({ key: `${s.toFixed(3)}-${e.toFixed(3)}-${out.length}`, start: s, end: e, word: w.word });
            }
        }
        out.sort((a, b) => a.start - b.start);
        return out;
    };

    const buildKeptRangesFromTokenState = (tokens: ClipToken[], removedKeys: Set<string>, clipStart: number, clipEnd: number): Array<[number, number]> => {
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
            if (s <= last[1] + 0.22) last[1] = Math.max(last[1], e);
            else merged.push([s, e]);
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

    const onRebuildClipEditorTextWindow = () => {
        const clipStart = Number(clipEditorDraft.start_time ?? 0);
        const clipEnd = Number(clipEditorDraft.end_time ?? 0);
        if (!Number.isFinite(clipStart) || !Number.isFinite(clipEnd) || clipEnd <= clipStart) {
            alert('Set a valid start/end first, then refresh transcript window.');
            return;
        }
        hydrateClipEditorText(clipStart, clipEnd, clipEditorDraft.script_edits_json);
    };

    const persistClipEditorRemovedWords = (nextRemoved: Set<string>) => {
        const clipStart = Number(clipEditorDraft.start_time ?? 0);
        const clipEnd = Number(clipEditorDraft.end_time ?? 0);
        const keptRanges = buildKeptRangesFromTokenState(clipEditorTokens, nextRemoved, clipStart, clipEnd);
        if (nextRemoved.size > 0 && keptRanges.length === 0) {
            alert('Cannot remove every word from the clip. Keep at least one word.');
            return;
        }
        setClipEditorRemovedWordKeys(nextRemoved);
        if (nextRemoved.size === 0) {
            onUpdateClipDraftField('script_edits_json', null);
            return;
        }
        onUpdateClipDraftField('script_edits_json', JSON.stringify({
            version: 1,
            mode: 'keep_ranges',
            source: 'text_editor',
            kept_ranges: keptRanges,
            removed_word_count: nextRemoved.size,
            total_word_count: clipEditorTokens.length,
            updated_at: new Date().toISOString(),
        }));
    };

    const onToggleClipEditorWord = (tokenKey: string) => {
        const next = new Set(clipEditorRemovedWordKeys);
        if (next.has(tokenKey)) next.delete(tokenKey); else next.add(tokenKey);
        persistClipEditorRemovedWords(next);
    };

    const onRestoreAllClipEditorWords = () => persistClipEditorRemovedWords(new Set());
    const onAutoRemoveClipEditorFillers = () => {
        const filler = new Set(['um', 'uh', 'erm', 'hmm', 'ah', 'like']);
        const next = new Set(clipEditorRemovedWordKeys);
        for (const t of clipEditorTokens) {
            const w = t.word.toLowerCase().replace(/[^\w']/g, '');
            if (filler.has(w)) next.add(t.key);
        }
        persistClipEditorRemovedWords(next);
    };

    const clampNorm = (val: number) => Math.max(0, Math.min(1, val));
    const normalizeCropRect = (x?: number | null, y?: number | null, w?: number | null, h?: number | null, fallback?: { x: number; y: number; w: number; h: number }) => {
        if (x == null || y == null || w == null || h == null) return fallback || { x: 0, y: 0, w: 1, h: 1 };
        const nx = clampNorm(Number(x));
        const ny = clampNorm(Number(y));
        const nw = Math.max(0.01, clampNorm(Number(w)));
        const nh = Math.max(0.01, clampNorm(Number(h)));
        return { x: nx, y: ny, w: nx + nw > 1 ? Math.max(0.01, 1 - nx) : nw, h: ny + nh > 1 ? Math.max(0.01, 1 - ny) : nh };
    };
    const onApplyPortraitSplitDefaults = () => {
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
    const getDraftCropRect = (target: ClipCropTarget) => {
        if (target === 'top') return normalizeCropRect(clipEditorDraft.portrait_top_crop_x, clipEditorDraft.portrait_top_crop_y, clipEditorDraft.portrait_top_crop_w, clipEditorDraft.portrait_top_crop_h, { x: 0, y: 0, w: 1, h: 0.5 });
        if (target === 'bottom') return normalizeCropRect(clipEditorDraft.portrait_bottom_crop_x, clipEditorDraft.portrait_bottom_crop_y, clipEditorDraft.portrait_bottom_crop_w, clipEditorDraft.portrait_bottom_crop_h, { x: 0, y: 0.5, w: 1, h: 0.5 });
        return normalizeCropRect(clipEditorDraft.crop_x, clipEditorDraft.crop_y, clipEditorDraft.crop_w, clipEditorDraft.crop_h);
    };
    const setDraftCropRect = (target: ClipCropTarget, rect: { x: number; y: number; w: number; h: number }) => {
        const x = Number(clampNorm(rect.x).toFixed(4));
        const y = Number(clampNorm(rect.y).toFixed(4));
        const w = Number(Math.max(0.01, Math.min(1 - x, rect.w)).toFixed(4));
        const h = Number(Math.max(0.01, Math.min(1 - y, rect.h)).toFixed(4));
        if (target === 'top') {
            onUpdateClipDraftField('portrait_top_crop_x', x);
            onUpdateClipDraftField('portrait_top_crop_y', y);
            onUpdateClipDraftField('portrait_top_crop_w', w);
            onUpdateClipDraftField('portrait_top_crop_h', h);
            return;
        }
        if (target === 'bottom') {
            onUpdateClipDraftField('portrait_bottom_crop_x', x);
            onUpdateClipDraftField('portrait_bottom_crop_y', y);
            onUpdateClipDraftField('portrait_bottom_crop_w', w);
            onUpdateClipDraftField('portrait_bottom_crop_h', h);
            return;
        }
        onUpdateClipDraftField('crop_x', x);
        onUpdateClipDraftField('crop_y', y);
        onUpdateClipDraftField('crop_w', w);
        onUpdateClipDraftField('crop_h', h);
    };
    const onNudgeClipBoundary = (which: 'start' | 'end', deltaSec: number) => {
        const start = Number(clipEditorDraft.start_time ?? 0);
        const end = Number(clipEditorDraft.end_time ?? 0);
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        if (which === 'start') onUpdateClipDraftField('start_time', Number(Math.max(0, Math.min(end - 0.05, start + deltaSec)).toFixed(3)));
        else onUpdateClipDraftField('end_time', Number(Math.max(start + 0.05, end + deltaSec).toFixed(3)));
    };
    const onSetClipBoundaryFromPlayhead = (which: 'start' | 'end') => {
        if (!Number.isFinite(currentTime)) return;
        const start = Number(clipEditorDraft.start_time ?? 0);
        const end = Number(clipEditorDraft.end_time ?? 0);
        if (which === 'start') onUpdateClipDraftField('start_time', Number(Math.max(0, Math.min(currentTime, end - 0.05)).toFixed(3)));
        else onUpdateClipDraftField('end_time', Number(Math.max(start + 0.05, currentTime).toFixed(3)));
    };
    const getEditorMediaDuration = () => {
        const byVideo = Number(video?.duration || 0);
        if (Number.isFinite(byVideo) && byVideo > 1) return byVideo;
        const segMax = segments.reduce((m, s) => Math.max(m, Number(s.end_time || 0)), 0);
        if (Number.isFinite(segMax) && segMax > 1) return segMax;
        const clipEnd = Number(clipEditorDraft?.end_time || 0);
        return Math.max(clipEnd + 1, 60);
    };
    const timelineTimeToPct = (t: number, duration: number) => Number.isFinite(duration) && duration > 0 ? clampNorm(t / duration) : 0;
    const timelineEventToTime = (evt: { clientX: number }, duration: number) => {
        if (!clipTimelineRef.current) return 0;
        const bounds = clipTimelineRef.current.getBoundingClientRect();
        if (bounds.width <= 0) return 0;
        return clampNorm((evt.clientX - bounds.left) / bounds.width) * duration;
    };
    const onBeginClipTimelineDrag = (handle: 'start' | 'end', e: any) => {
        e.preventDefault();
        e.stopPropagation();
        setClipTimelineDrag({ handle });
    };
    const onTimelineScrub = (e: any) => onSeek(timelineEventToTime(e, getEditorMediaDuration()));
    const onCropPreviewPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!cropPreviewRef.current) return;
        if (clipEditorCropTarget !== 'main' && !isPortraitSplit) return;
        const bounds = cropPreviewRef.current.getBoundingClientRect();
        if (bounds.width <= 0 || bounds.height <= 0) return;
        const nx = clampNorm((e.clientX - bounds.left) / bounds.width);
        const ny = clampNorm((e.clientY - bounds.top) / bounds.height);
        setClipEditorDragRect({ target: clipEditorCropTarget, startX: nx, startY: ny, currentX: nx, currentY: ny });
    };

    useEffect(() => {
        hydrateClipEditorText(activeEditingClip.start_time, activeEditingClip.end_time, activeEditingClip.script_edits_json);
    }, [activeEditingClip.id, segments]);

    useEffect(() => {
        if (!clipEditorDragRect) return;
        const onMove = (evt: PointerEvent) => {
            if (!cropPreviewRef.current) return;
            const bounds = cropPreviewRef.current.getBoundingClientRect();
            if (bounds.width <= 0 || bounds.height <= 0) return;
            setClipEditorDragRect(prev => prev ? { ...prev, currentX: clampNorm((evt.clientX - bounds.left) / bounds.width), currentY: clampNorm((evt.clientY - bounds.top) / bounds.height) } : prev);
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
    }, [clipEditorDragRect, clipEditorDraft]);

    useEffect(() => {
        if (!clipTimelineDrag) return;
        const onMove = (evt: PointerEvent) => {
            const t = timelineEventToTime(evt, getEditorMediaDuration());
            const start = Number(clipEditorDraft.start_time ?? 0);
            const end = Number(clipEditorDraft.end_time ?? 0);
            if (clipTimelineDrag.handle === 'start') onUpdateClipDraftField('start_time', Number(Math.max(0, Math.min(end - 0.05, t)).toFixed(3)));
            else onUpdateClipDraftField('end_time', Number(Math.max(start + 0.05, t).toFixed(3)));
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
        if (!isPortraitSplit && clipEditorCropTarget !== 'main') setClipEditorCropTarget('main');
    }, [isPortraitSplit, clipEditorCropTarget]);

    useEffect(() => {
        const onKeyDown = (evt: KeyboardEvent) => {
            const target = evt.target as HTMLElement | null;
            const tag = (target?.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable) return;
            const frame = 1 / 30;
            if (evt.key.toLowerCase() === 'i') {
                evt.preventDefault();
                onSetClipBoundaryFromPlayhead('start');
                return;
            }
            if (evt.key.toLowerCase() === 'o') {
                evt.preventDefault();
                onSetClipBoundaryFromPlayhead('end');
                return;
            }
            if (evt.altKey && evt.key === 'ArrowLeft') {
                evt.preventDefault();
                onNudgeClipBoundary(evt.shiftKey ? 'end' : 'start', -frame);
                return;
            }
            if (evt.altKey && evt.key === 'ArrowRight') {
                evt.preventDefault();
                onNudgeClipBoundary(evt.shiftKey ? 'end' : 'start', frame);
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [currentTime, clipEditorDraft]);

    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="mx-auto max-w-6xl grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-6">
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-3">
                    <div className="flex items-center justify-between gap-2">
                        <div>
                            <div className="text-sm font-semibold text-slate-800">Clip Editor</div>
                            <div className="text-xs text-slate-500">Editing: {activeEditingClip.title || `Clip #${activeEditingClip.id}`}</div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button onClick={onCancelClipEdit} className="px-3 py-2 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-600 hover:bg-slate-50">Close</button>
                            <button
                                onClick={() => onSaveClipEdit(activeEditingClip.id)}
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
                            <input type="text" value={String(clipEditorDraft.title || '')} onChange={(e) => onUpdateClipDraftField('title', e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-slate-600 mb-1">Aspect Ratio</label>
                            <select value={String(clipEditorDraft.aspect_ratio || 'source')} onChange={(e) => onUpdateClipDraftField('aspect_ratio', e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white">
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
                            <input type="number" step="0.1" value={Number(clipEditorDraft.start_time ?? activeEditingClip.start_time)} onChange={(e) => onUpdateClipDraftField('start_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-slate-600 mb-1">End (sec)</label>
                            <input type="number" step="0.1" value={Number(clipEditorDraft.end_time ?? activeEditingClip.end_time)} onChange={(e) => onUpdateClipDraftField('end_time', Number(e.target.value))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" />
                        </div>
                    </div>

                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 space-y-2">
                        <div className="text-[11px] font-semibold text-slate-600">Fast Trim Controls</div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <button onClick={() => onSetClipBoundaryFromPlayhead('start')} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Set In @ Playhead (I)</button>
                            <button onClick={() => onSetClipBoundaryFromPlayhead('end')} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Set Out @ Playhead (O)</button>
                            <button onClick={() => onSeek(Number(clipEditorDraft.start_time ?? activeEditingClip.start_time))} className="px-2 py-1.5 text-[11px] rounded-md border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100">Jump In</button>
                            <button onClick={() => onSeek(Number(clipEditorDraft.end_time ?? activeEditingClip.end_time))} className="px-2 py-1.5 text-[11px] rounded-md border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100">Jump Out</button>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <button onClick={() => onNudgeClipBoundary('start', -(1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">In -1f</button>
                            <button onClick={() => onNudgeClipBoundary('start', (1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">In +1f</button>
                            <button onClick={() => onNudgeClipBoundary('end', -(1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Out -1f</button>
                            <button onClick={() => onNudgeClipBoundary('end', (1 / 30))} className="px-2 py-1.5 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50">Out +1f</button>
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
                                        <span>0:00</span><span>{formatTime(duration / 4)}</span><span>{formatTime(duration / 2)}</span><span>{formatTime((duration * 3) / 4)}</span><span>{formatTime(duration)}</span>
                                    </div>
                                    <div ref={clipTimelineRef} onClick={onTimelineScrub} className="relative h-10 rounded-md border border-blue-200 bg-white cursor-pointer overflow-hidden">
                                        <div className="absolute inset-y-0 left-0 w-full bg-gradient-to-r from-slate-100 via-blue-50 to-slate-100" />
                                        <div className="absolute inset-y-1 rounded bg-blue-500/25 border border-blue-300" style={{ left: `${startPct}%`, width: `${Math.max(0.2, endPct - startPct)}%` }} />
                                        <div className="absolute top-0 bottom-0 w-[2px] bg-red-500/80" style={{ left: `${playPct}%` }} />
                                        <button type="button" onPointerDown={(e) => onBeginClipTimelineDrag('start', e)} className="absolute top-0 bottom-0 -ml-1 w-2 rounded bg-blue-700 hover:bg-blue-800 cursor-ew-resize" style={{ left: `${startPct}%` }} title="Drag In point" />
                                        <button type="button" onPointerDown={(e) => onBeginClipTimelineDrag('end', e)} className="absolute top-0 bottom-0 -ml-1 w-2 rounded bg-blue-700 hover:bg-blue-800 cursor-ew-resize" style={{ left: `${endPct}%` }} title="Drag Out point" />
                                    </div>
                                    <div className="flex items-center justify-between text-[10px] text-blue-700/80 font-mono">
                                        <span>IN {formatTime(startSec)}</span><span>{(Math.max(0, endSec - startSec)).toFixed(2)}s</span><span>OUT {formatTime(endSec)}</span>
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
                                    onUpdateClipDraftField('crop_x', null);
                                    onUpdateClipDraftField('crop_y', null);
                                    onUpdateClipDraftField('crop_w', null);
                                    onUpdateClipDraftField('crop_h', null);
                                }}
                                className="px-2.5 py-1 text-[11px] rounded-md bg-white border border-slate-300 text-slate-600 hover:bg-slate-50"
                            >
                                Reset Main Crop
                            </button>
                        </div>
                        <div className="grid grid-cols-4 gap-2">
                            {(['crop_x', 'crop_y', 'crop_w', 'crop_h'] as const).map((field) => (
                                <input key={field} type="number" min={0} max={1} step={0.01} value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])} placeholder={field.replace('crop_', '')} onChange={(e) => onUpdateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))} className="px-2 py-2 text-xs border border-slate-300 rounded-lg bg-white" />
                            ))}
                        </div>
                    </div>

                    {String(clipEditorDraft.aspect_ratio || 'source') === '9:16' && (
                        <div className="rounded-lg border border-indigo-200 bg-indigo-50/50 p-2.5 space-y-2">
                            <label className="inline-flex items-center gap-2 text-xs text-indigo-700 font-medium">
                                <input type="checkbox" checked={!!clipEditorDraft.portrait_split_enabled} onChange={(e) => e.target.checked ? onApplyPortraitSplitDefaults() : onUpdateClipDraftField('portrait_split_enabled', false)} className="h-4 w-4 rounded border-indigo-300 text-indigo-600" />
                                Portrait split mode (top/lower stacked)
                            </label>
                            {!!clipEditorDraft.portrait_split_enabled && (
                                <>
                                    <div className="grid grid-cols-4 gap-2">
                                        {(['portrait_top_crop_x', 'portrait_top_crop_y', 'portrait_top_crop_w', 'portrait_top_crop_h'] as const).map((field) => (
                                            <input key={field} type="number" min={0} max={1} step={0.01} value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])} placeholder={field.replace('portrait_top_crop_', 'top_')} onChange={(e) => onUpdateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))} className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white" />
                                        ))}
                                    </div>
                                    <div className="grid grid-cols-4 gap-2">
                                        {(['portrait_bottom_crop_x', 'portrait_bottom_crop_y', 'portrait_bottom_crop_w', 'portrait_bottom_crop_h'] as const).map((field) => (
                                            <input key={field} type="number" min={0} max={1} step={0.01} value={clipEditorDraft[field] == null ? '' : Number(clipEditorDraft[field])} placeholder={field.replace('portrait_bottom_crop_', 'low_')} onChange={(e) => onUpdateClipDraftField(field, e.target.value === '' ? null : Number(e.target.value))} className="px-2 py-2 text-xs border border-indigo-200 rounded-lg bg-white" />
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-3">
                        <label className="inline-flex items-center gap-2 text-xs text-slate-600"><input type="checkbox" checked={!!clipEditorDraft.burn_captions} onChange={(e) => onUpdateClipDraftField('burn_captions', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />Burn captions into MP4</label>
                        <label className="inline-flex items-center gap-2 text-xs text-slate-600"><input type="checkbox" checked={!!clipEditorDraft.caption_speaker_labels} onChange={(e) => onUpdateClipDraftField('caption_speaker_labels', e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-purple-600" />Speaker labels in captions</label>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div><label className="block text-[11px] font-medium text-slate-600 mb-1">Fade In (sec)</label><input type="number" min={0} step={0.05} value={Number(clipEditorDraft.fade_in_sec ?? 0)} onChange={(e) => onUpdateClipDraftField('fade_in_sec', Math.max(0, Number(e.target.value || 0)))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" /></div>
                        <div><label className="block text-[11px] font-medium text-slate-600 mb-1">Fade Out (sec)</label><input type="number" min={0} step={0.05} value={Number(clipEditorDraft.fade_out_sec ?? 0)} onChange={(e) => onUpdateClipDraftField('fade_out_sec', Math.max(0, Number(e.target.value || 0)))} className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white" /></div>
                    </div>

                    <div className="rounded-lg border border-violet-200 bg-violet-50/50 p-2.5 space-y-1.5">
                        <div className="text-[11px] font-semibold text-violet-700">Composable Export Stack</div>
                        <div className="flex flex-wrap gap-1.5 text-[10px]">
                            <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">Source Ranges</span>
                            {clipEditorDraft.script_edits_json && <span className="px-1.5 py-0.5 rounded bg-emerald-100 border border-emerald-200 text-emerald-700">Text Keep-Ranges</span>}
                            {(clipEditorDraft.crop_x != null || clipEditorDraft.crop_y != null || clipEditorDraft.crop_w != null || clipEditorDraft.crop_h != null) && <span className="px-1.5 py-0.5 rounded bg-cyan-100 border border-cyan-200 text-cyan-700">Crop/Reframe</span>}
                            {isPortraitSplit && <span className="px-1.5 py-0.5 rounded bg-indigo-100 border border-indigo-200 text-indigo-700">Portrait Split</span>}
                            <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">Aspect {String(clipEditorDraft.aspect_ratio || 'source').toUpperCase()}</span>
                            {!!clipEditorDraft.burn_captions && <span className="px-1.5 py-0.5 rounded bg-amber-100 border border-amber-200 text-amber-700">Burn Captions</span>}
                            <span className="px-1.5 py-0.5 rounded bg-white border border-violet-200 text-violet-700">H264/AAC Render</span>
                        </div>
                    </div>

                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3 space-y-2.5">
                        <div className="flex items-center justify-between gap-2">
                            <div><div className="text-xs font-semibold text-emerald-800">Text-Based Edit (Phase 1)</div><div className="text-[11px] text-emerald-700/80">Click words to remove/restore. Export stitches only kept transcript ranges.</div></div>
                            <div className="flex items-center gap-1.5">
                                <button onClick={onRebuildClipEditorTextWindow} className="px-2 py-1 text-[11px] rounded-md border border-emerald-300 bg-white text-emerald-700 hover:bg-emerald-50">Refresh Window</button>
                                <button onClick={onAutoRemoveClipEditorFillers} disabled={clipEditorTokens.length === 0} className="px-2 py-1 text-[11px] rounded-md border border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100 disabled:opacity-50">Remove Fillers</button>
                                <button onClick={onRestoreAllClipEditorWords} disabled={clipEditorRemovedWordKeys.size === 0} className="px-2 py-1 text-[11px] rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50">Restore All</button>
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
                            {clipEditorTokens.length === 0 ? <div className="text-[11px] text-slate-500">No word-level transcript tokens in this trim window. Adjust start/end and click Refresh Window.</div> : (
                                <div className="flex flex-wrap gap-1.5">
                                    {clipEditorTokens.map((tok) => {
                                        const removed = clipEditorRemovedWordKeys.has(tok.key);
                                        return <button key={tok.key} type="button" onClick={() => onToggleClipEditorWord(tok.key)} className={`px-1.5 py-0.5 rounded text-[11px] border transition-colors ${removed ? 'bg-rose-50 border-rose-200 text-rose-700 line-through' : 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100'}`} title={`${formatTime(tok.start)} - ${formatTime(tok.end)}`}>{tok.word}</button>;
                                    })}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    {playerNode}
                    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <div className="text-[11px] font-semibold tracking-wide text-slate-600">Burn/Crop Preview (draw to set crop)</div>
                            <div className="flex items-center gap-1">
                                <button onClick={() => setClipEditorCropTarget('main')} className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'main' ? 'bg-cyan-100 border-cyan-300 text-cyan-700' : 'bg-white border-slate-300 text-slate-600'}`}>Main</button>
                                {isPortraitSplit && <><button onClick={() => setClipEditorCropTarget('top')} className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'top' ? 'bg-amber-100 border-amber-300 text-amber-700' : 'bg-white border-slate-300 text-slate-600'}`}>Top</button><button onClick={() => setClipEditorCropTarget('bottom')} className={`px-2 py-1 rounded text-[10px] border ${clipEditorCropTarget === 'bottom' ? 'bg-lime-100 border-lime-300 text-lime-700' : 'bg-white border-slate-300 text-slate-600'}`}>Lower</button></>}
                            </div>
                        </div>
                        <div
                            className="relative w-full rounded-lg overflow-hidden border border-slate-300 bg-slate-900 cursor-crosshair"
                            ref={cropPreviewRef}
                            onPointerDown={onCropPreviewPointerDown}
                            style={{ aspectRatio: String(clipEditorDraft.aspect_ratio || 'source') === '1:1' ? '1 / 1' : String(clipEditorDraft.aspect_ratio || 'source') === '4:5' ? '4 / 5' : String(clipEditorDraft.aspect_ratio || 'source') === '9:16' ? '9 / 16' : '16 / 9' }}
                        >
                            {video?.thumbnail_url ? <img src={video?.thumbnail_url || ''} alt="" className="absolute inset-0 w-full h-full object-cover opacity-70" /> : <div className="absolute inset-0 bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900" />}
                            {!isPortraitSplit && <div className="absolute border-2 border-cyan-300/90 bg-cyan-300/10" style={{ left: `${getDraftCropRect('main').x * 100}%`, top: `${getDraftCropRect('main').y * 100}%`, width: `${getDraftCropRect('main').w * 100}%`, height: `${getDraftCropRect('main').h * 100}%` }} />}
                            {isPortraitSplit && <><div className="absolute border-2 border-amber-300/90 bg-amber-300/15" style={{ left: `${getDraftCropRect('top').x * 100}%`, top: `${getDraftCropRect('top').y * 100}%`, width: `${getDraftCropRect('top').w * 100}%`, height: `${getDraftCropRect('top').h * 100}%` }} /><div className="absolute border-2 border-lime-300/90 bg-lime-300/15" style={{ left: `${getDraftCropRect('bottom').x * 100}%`, top: `${getDraftCropRect('bottom').y * 100}%`, width: `${getDraftCropRect('bottom').w * 100}%`, height: `${getDraftCropRect('bottom').h * 100}%` }} /><div className="absolute inset-x-0 top-1/2 border-t border-white/70 border-dashed" /></>}
                            {clipEditorDragRect && <div className="absolute border-2 border-white border-dashed bg-white/10 pointer-events-none" style={{ left: `${Math.min(clipEditorDragRect.startX, clipEditorDragRect.currentX) * 100}%`, top: `${Math.min(clipEditorDragRect.startY, clipEditorDragRect.currentY) * 100}%`, width: `${Math.max(0.01, Math.abs(clipEditorDragRect.currentX - clipEditorDragRect.startX)) * 100}%`, height: `${Math.max(0.01, Math.abs(clipEditorDragRect.currentY - clipEditorDragRect.startY)) * 100}%` }} />}
                            {!!clipEditorDraft.burn_captions && <div className="absolute inset-x-2 bottom-2 px-2 py-1.5 rounded bg-black/55 text-[11px] text-white text-center">[Speaker] Sample burned caption preview</div>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
