import { useEffect } from 'react';
import { Bot, CheckCircle2, Copy, Loader2, RefreshCw, RotateCcw } from 'lucide-react';
import { useYoutubeStore, parseYoutubeAiChapters } from '../../../store/useYoutubeStore';
import type { Video, TranscriptSegment } from '../../../types';

type Props = {
    video: Video;
    videoId: number;
    segments: TranscriptSegment[];
    isYoutubeMedia: boolean;
    isActive: boolean;
    onVideoUpdated: (v: Video) => void;
    onSeek: (seconds: number) => void;
};

export function YoutubeTab({ video, videoId, segments, isYoutubeMedia, isActive, onVideoUpdated, onSeek }: Props) {
    const store = useYoutubeStore();

    const youtubeAiChapters = parseYoutubeAiChapters(video.youtube_ai_chapters_json);
    const hasYoutubeAiMetadata = !!(video.youtube_ai_summary || video.youtube_ai_description_text || youtubeAiChapters.length);

    // Computed labels (isYoutubeMedia-derived strings kept local — no store needed)
    const panelTitle = isYoutubeMedia ? 'YouTube Summary + Chapters' : 'Episode Summary + Chapters';
    const panelDescription = isYoutubeMedia
        ? 'Generate a YouTube-style episode description summary and chapter timestamps/descriptions from the transcript using the current LLM provider.'
        : 'Generate a readable episode summary and chapter-style conversation index from the transcript using the current LLM provider.';
    const generateTitle = isYoutubeMedia
        ? 'Generate or re-generate YouTube summary + chapters'
        : 'Generate or re-generate episode summary + chapters';
    const emptyText = isYoutubeMedia
        ? 'No generated summary/chapters yet. Click Generate to create a YouTube-ready draft description and chapter list.'
        : 'No generated summary/chapters yet. Click Generate to create a readable summary and chapter-style conversation index.';
    const currentDescriptionLabel = isYoutubeMedia ? 'Current Video Description (Stored)' : 'Current Episode Description (Stored)';
    const chaptersLabel = isYoutubeMedia ? 'Chapters (YouTube-style)' : 'Conversation Index';
    const descriptionLabel = isYoutubeMedia ? 'YouTube Description Draft (Copy/Paste)' : 'Episode Description Draft';
    const publishLabel = isYoutubeMedia ? 'Publish Draft (Archive Current)' : 'Apply Draft (Archive Current)';
    const publishHelp = isYoutubeMedia
        ? 'Updates the app’s stored video description and preserves restorable history.'
        : 'Updates the app’s stored episode description and preserves restorable history.';

    useEffect(() => {
        if (!isActive) return;
        void store.fetchDescriptionHistory(videoId);
    }, [isActive, videoId]);

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4">
            {/* Header card — generate + publish controls */}
            <div className="rounded-xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50 p-4">
                <div className="flex items-start justify-between gap-3">
                    <div>
                        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-800">
                            <Bot size={15} className="text-emerald-600" />
                            {panelTitle}
                        </div>
                        <p className="mt-1 text-xs text-emerald-700/80">
                            {panelDescription}
                        </p>
                    </div>
                    <button
                        onClick={() => void store.handleGenerateYoutubeAi(videoId, hasYoutubeAiMetadata, onVideoUpdated)}
                        disabled={store.generatingYoutubeAi || segments.length === 0}
                        className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        title={segments.length === 0 ? 'Transcript required first' : generateTitle}
                    >
                        {store.generatingYoutubeAi ? (
                            <Loader2 size={13} className="animate-spin" />
                        ) : hasYoutubeAiMetadata ? (
                            <RefreshCw size={13} />
                        ) : (
                            <Bot size={13} />
                        )}
                        {hasYoutubeAiMetadata ? 'Re-generate' : 'Generate'}
                    </button>
                </div>
                <div className="mt-2 flex items-center gap-2">
                    <button
                        onClick={() => void store.handlePublishYoutubeDescription(videoId, isYoutubeMedia, onVideoUpdated)}
                        disabled={store.publishingYoutubeDescription || !video.youtube_ai_description_text}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        title={video.youtube_ai_description_text ? publishHelp : 'Generate a draft first'}
                    >
                        {store.publishingYoutubeDescription ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                        {publishLabel}
                    </button>
                    <span className="text-[11px] text-emerald-800/80">{publishHelp}</span>
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

            {/* Empty states */}
            {segments.length === 0 ? (
                <div className="rounded-xl border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">
                    Transcript required first. Run transcription/diarization before generating{' '}
                    {isYoutubeMedia ? 'AI summary metadata' : 'episode summary metadata'}.
                </div>
            ) : !hasYoutubeAiMetadata ? (
                <div className="rounded-xl border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">
                    {emptyText}
                </div>
            ) : (
                <>
                    {/* Current stored description */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <h3 className="text-sm font-semibold text-slate-800">{currentDescriptionLabel}</h3>
                            {video.description && (
                                <button
                                    onClick={() => void store.handleCopyYoutubeField('description', video.description || '')}
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

                    {/* Episode summary */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <h3 className="text-sm font-semibold text-slate-800">Episode Summary</h3>
                            {video.youtube_ai_summary && (
                                <button
                                    onClick={() => void store.handleCopyYoutubeField('summary', video.youtube_ai_summary || '')}
                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                >
                                    <Copy size={12} />
                                    {store.copiedYoutubeField === 'summary' ? 'Copied' : 'Copy'}
                                </button>
                            )}
                        </div>
                        <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                            {video.youtube_ai_summary || 'No summary generated.'}
                        </p>
                    </div>

                    {/* Chapters */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <h3 className="text-sm font-semibold text-slate-800">{chaptersLabel}</h3>
                            {youtubeAiChapters.length > 0 && (
                                <button
                                    onClick={() =>
                                        void store.handleCopyYoutubeField(
                                            'chapters',
                                            youtubeAiChapters.map((ch) => `${ch.timestamp} ${ch.title}`).join('\n')
                                        )
                                    }
                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                >
                                    <Copy size={12} />
                                    {store.copiedYoutubeField === 'chapters' ? 'Copied' : 'Copy Lines'}
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
                                        onClick={() => onSeek(ch.start_seconds)}
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

                    {/* Description draft */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <h3 className="text-sm font-semibold text-slate-800">{descriptionLabel}</h3>
                            {video.youtube_ai_description_text && (
                                <button
                                    onClick={() =>
                                        void store.handleCopyYoutubeField('description', video.youtube_ai_description_text || '')
                                    }
                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200"
                                >
                                    <Copy size={12} />
                                    {store.copiedYoutubeField === 'description' ? 'Copied' : 'Copy Full'}
                                </button>
                            )}
                        </div>
                        <pre className="text-xs text-slate-700 bg-slate-50 border border-slate-100 rounded-lg p-3 whitespace-pre-wrap break-words font-mono leading-relaxed">
                            {video.youtube_ai_description_text || 'No description draft generated yet.'}
                        </pre>
                    </div>

                    {/* Description history */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-2">
                            <h3 className="text-sm font-semibold text-slate-800">Description History (Restore)</h3>
                            {store.loadingDescriptionHistory && <Loader2 size={14} className="animate-spin text-slate-400" />}
                        </div>
                        {store.descriptionHistory.length === 0 ? (
                            <p className="text-sm text-slate-500">
                                No archived descriptions yet. Publishing a draft will archive the current description first.
                            </p>
                        ) : (
                            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                                {store.descriptionHistory.map((rev) => (
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
                                                onClick={() => void store.handleRestoreDescriptionRevision(videoId, rev, onVideoUpdated)}
                                                disabled={store.restoringDescriptionRevisionId === rev.id}
                                                className="shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 disabled:opacity-50"
                                            >
                                                {store.restoringDescriptionRevisionId === rev.id ? (
                                                    <Loader2 size={11} className="animate-spin" />
                                                ) : (
                                                    <RotateCcw size={11} />
                                                )}
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
    );
}
