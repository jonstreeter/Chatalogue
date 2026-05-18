import { useEffect } from 'react';
import {
    AudioLines,
    CheckCircle2,
    CircleHelp,
    FileText,
    GitMerge,
    Loader2,
    RefreshCw,
    RotateCcw,
    Save,
    Scissors,
} from 'lucide-react';
import { useTranscriptStore } from '../../../store/useTranscriptStore';

const transcriptOptimizationHelp: Record<string, string> = {
    repair: 'Low-risk repair merges tiny same-speaker fragments, absorbs short unknown interruptions, applies conservative entity repair, and cleans transcript formatting without re-running ASR.',
    rebuild: 'Diarization rebuild keeps the raw transcript words but recomputes speaker turns and speaker matching. Use it when labeling is unstable but the wording is mostly correct.',
    retranscribe: 'Full retranscription discards the current transcript and runs a fresh transcription plus diarization pass. Use it for multilingual failures or broadly inaccurate text.',
};

type SelectionRange = { start: number; end: number; defaultTitle: string } | null;

type Props = {
    hasTranscript: boolean;
    videoId: number;
    isActive: boolean;
    episodeBusy: boolean;
    selection: SelectionRange;
    transcriptLanguage?: string | null;
    onSeek: (time: number) => void;
    onNavigateToTranscript: () => void;
    onRefreshVideo: () => Promise<void>;
};

export function OptimizeTab({
    hasTranscript,
    videoId,
    isActive,
    episodeBusy,
    selection,
    transcriptLanguage,
    onSeek,
    onNavigateToTranscript,
    onRefreshVideo,
}: Props) {
    const transcriptQuality = useTranscriptStore((s) => s.transcriptQuality);
    const loadingTranscriptQuality = useTranscriptStore((s) => s.loadingTranscriptQuality);
    const transcriptQualityError = useTranscriptStore((s) => s.transcriptQualityError);
    const transcriptRollbackOptions = useTranscriptStore((s) => s.transcriptRollbackOptions);
    const loadingTranscriptRollbackOptions = useTranscriptStore((s) => s.loadingTranscriptRollbackOptions);
    const restoringTranscriptRunId = useTranscriptStore((s) => s.restoringTranscriptRunId);
    const transcriptGoldWindows = useTranscriptStore((s) => s.transcriptGoldWindows);
    const loadingTranscriptGoldWindows = useTranscriptStore((s) => s.loadingTranscriptGoldWindows);
    const transcriptGoldWindowsError = useTranscriptStore((s) => s.transcriptGoldWindowsError);
    const savingTranscriptGoldWindow = useTranscriptStore((s) => s.savingTranscriptGoldWindow);
    const evaluatingTranscript = useTranscriptStore((s) => s.evaluatingTranscript);
    const transcriptEvaluationSummary = useTranscriptStore((s) => s.transcriptEvaluationSummary);
    const transcriptEvaluationResults = useTranscriptStore((s) => s.transcriptEvaluationResults);
    const loadingTranscriptEvaluationResults = useTranscriptStore((s) => s.loadingTranscriptEvaluationResults);
    const transcriptEvaluationError = useTranscriptStore((s) => s.transcriptEvaluationError);
    const reviewingEvaluationResultId = useTranscriptStore((s) => s.reviewingEvaluationResultId);
    const evaluationReviewsByResultId = useTranscriptStore((s) => s.evaluationReviewsByResultId);
    const goldWindowLabelDraft = useTranscriptStore((s) => s.goldWindowLabelDraft);
    const goldWindowStartDraft = useTranscriptStore((s) => s.goldWindowStartDraft);
    const goldWindowEndDraft = useTranscriptStore((s) => s.goldWindowEndDraft);
    const goldWindowReferenceDraft = useTranscriptStore((s) => s.goldWindowReferenceDraft);
    const goldWindowEntitiesDraft = useTranscriptStore((s) => s.goldWindowEntitiesDraft);
    const goldWindowNotesDraft = useTranscriptStore((s) => s.goldWindowNotesDraft);
    const evaluationReviewVerdictDrafts = useTranscriptStore((s) => s.evaluationReviewVerdictDrafts);
    const evaluationReviewNotesDrafts = useTranscriptStore((s) => s.evaluationReviewNotesDrafts);
    const evaluationReviewReviewerDrafts = useTranscriptStore((s) => s.evaluationReviewReviewerDrafts);
    const queueingTranscriptRepair = useTranscriptStore((s) => s.queueingTranscriptRepair);
    const queueingDiarizationRebuild = useTranscriptStore((s) => s.queueingDiarizationRebuild);
    const queueingDiarizationBenchmark = useTranscriptStore((s) => s.queueingDiarizationBenchmark);
    const queueingFullRetranscription = useTranscriptStore((s) => s.queueingFullRetranscription);
    const diarizationBenchmarkSensitivity = useTranscriptStore((s) => s.diarizationBenchmarkSensitivity);
    const diarizationBenchmarkThreshold = useTranscriptStore((s) => s.diarizationBenchmarkThreshold);
    const setGoldWindowLabelDraft = useTranscriptStore((s) => s.setGoldWindowLabelDraft);
    const setGoldWindowStartDraft = useTranscriptStore((s) => s.setGoldWindowStartDraft);
    const setGoldWindowEndDraft = useTranscriptStore((s) => s.setGoldWindowEndDraft);
    const setGoldWindowReferenceDraft = useTranscriptStore((s) => s.setGoldWindowReferenceDraft);
    const setGoldWindowEntitiesDraft = useTranscriptStore((s) => s.setGoldWindowEntitiesDraft);
    const setGoldWindowNotesDraft = useTranscriptStore((s) => s.setGoldWindowNotesDraft);
    const setEvaluationReviewVerdictDrafts = useTranscriptStore((s) => s.setEvaluationReviewVerdictDrafts);
    const setEvaluationReviewNotesDrafts = useTranscriptStore((s) => s.setEvaluationReviewNotesDrafts);
    const setEvaluationReviewReviewerDrafts = useTranscriptStore((s) => s.setEvaluationReviewReviewerDrafts);
    const setDiarizationBenchmarkSensitivity = useTranscriptStore((s) => s.setDiarizationBenchmarkSensitivity);
    const setDiarizationBenchmarkThreshold = useTranscriptStore((s) => s.setDiarizationBenchmarkThreshold);

    const recommendedOptimizationTier = String(transcriptQuality?.recommended_tier || 'none');
    const recommendedOptimizationLabel = recommendedOptimizationTier === 'low_risk_repair'
        ? 'Low-Risk Repair'
        : recommendedOptimizationTier === 'diarization_rebuild'
            ? 'Diarization Rebuild'
            : recommendedOptimizationTier === 'full_retranscription'
                ? 'Full Retranscription'
                : recommendedOptimizationTier === 'manual_review'
                    ? 'Manual Review'
                    : 'No Automatic Optimization';

    useEffect(() => {
        if (!isActive || !videoId || !hasTranscript) return;
        const controller = new AbortController();
        const store = useTranscriptStore.getState();
        void store.fetchTranscriptRollbackOptions(videoId, controller.signal);
        void store.fetchTranscriptGoldWindows(videoId, controller.signal);
        void store.fetchTranscriptEvaluationResults(videoId, controller.signal);
        return () => controller.abort();
    }, [hasTranscript, isActive, videoId]);

    const refreshAssessment = () => {
        if (!videoId) return;
        void useTranscriptStore.getState().fetchTranscriptQuality(videoId);
    };

    const useCurrentSelectionForGoldWindow = () => {
        if (!selection) {
            alert('Create a transcript or clip selection first, then use it as the benchmark window range.');
            return;
        }
        setGoldWindowStartDraft(selection.start.toFixed(2));
        setGoldWindowEndDraft(selection.end.toFixed(2));
        setGoldWindowLabelDraft((current) => (current && current !== 'Gold Window' ? current : selection.defaultTitle || 'Gold Window'));
    };

    const createTranscriptGoldWindow = async () => {
        await useTranscriptStore.getState().createTranscriptGoldWindow(videoId, transcriptLanguage);
    };

    const runTranscriptEvaluation = async () => {
        await useTranscriptStore.getState().runTranscriptEvaluation(videoId);
    };

    const submitTranscriptEvaluationReview = async (resultId: number) => {
        await useTranscriptStore.getState().submitTranscriptEvaluationReview(resultId);
    };

    const queueTranscriptRepairJob = async () => {
        await useTranscriptStore.getState().queueTranscriptRepairJob(videoId, onRefreshVideo);
    };

    const queueTranscriptDiarizationRebuildJob = async () => {
        await useTranscriptStore.getState().queueTranscriptDiarizationRebuildJob(videoId, onRefreshVideo);
    };

    const queueTranscriptDiarizationBenchmarkJob = async () => {
        await useTranscriptStore.getState().queueTranscriptDiarizationBenchmarkJob(videoId, onRefreshVideo);
    };

    const queueTranscriptRetranscriptionJob = async () => {
        await useTranscriptStore.getState().queueTranscriptRetranscriptionJob(videoId, onRefreshVideo);
    };

    const restoreTranscriptFromRun = async (runId: number) => {
        await useTranscriptStore.getState().restoreTranscriptFromRun(videoId, runId, onRefreshVideo);
    };

    const renderSnapshot = () => (
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-4 shadow-sm">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Optimization Navigator</div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">Transcript Optimization Workbench</div>
                    <div className="mt-1 text-xs leading-5 text-slate-600">
                        Benchmarking, rollback, repair, rebuild, and retranscription controls live in the main stage. Use the transcript tab for reading and editing only.
                    </div>
                </div>
                <div className="flex shrink-0 gap-2">
                    <button
                        type="button"
                        onClick={onNavigateToTranscript}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/80 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                    >
                        <FileText size={14} />
                        Transcript
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
                        Unknown speaker rate: {Number(transcriptQuality.metrics?.unknown_speaker_rate || 0).toFixed(2)}
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
            <div className="mt-3">
                <button
                    type="button"
                    onClick={refreshAssessment}
                    disabled={loadingTranscriptQuality}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                >
                    {loadingTranscriptQuality ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                    Refresh Assessment
                </button>
            </div>
        </div>
    );

    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="mx-auto max-w-7xl space-y-6">
                {renderSnapshot()}

                {!hasTranscript ? (
                    <div className="rounded-3xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center shadow-sm">
                        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-slate-100 text-slate-400">
                            <CheckCircle2 size={24} />
                        </div>
                        <div className="mt-4 text-lg font-semibold text-slate-900">Transcript required</div>
                        <div className="mx-auto mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                            Optimization, rollback, and benchmarking all depend on a transcript. Generate or restore the transcript first, then return to this workbench.
                        </div>
                        <button
                            type="button"
                            onClick={onNavigateToTranscript}
                            className="mt-5 inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
                        >
                            <FileText size={15} />
                            Go to Transcript
                        </button>
                    </div>
                ) : (
                    <>
                        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                                <div className="min-w-0">
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Optimization Actions</div>
                                    <div className="mt-1 text-lg font-semibold text-slate-900">Repair, rebuild, and rollback</div>
                                    <div className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
                                        Use repair for conservative cleanup, rebuild for speaker-turn correction, and retranscribe only when the wording itself is unreliable.
                                    </div>
                                </div>
                                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                                    <div className="group relative">
                                        <button
                                            onClick={queueTranscriptRepairJob}
                                            disabled={episodeBusy || recommendedOptimizationTier !== 'low_risk_repair'}
                                            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-3 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                                        >
                                            {queueingTranscriptRepair ? <Loader2 size={15} className="animate-spin" /> : <GitMerge size={15} />}
                                            Queue Repair
                                            <CircleHelp size={13} className="opacity-80" />
                                        </button>
                                        <div className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-72 -translate-x-1/2 rounded-lg border border-slate-200 bg-slate-900 px-3 py-2 text-[11px] leading-5 text-white shadow-xl group-hover:block">
                                            {transcriptOptimizationHelp.repair}
                                        </div>
                                    </div>
                                    <div className="group relative">
                                        <button
                                            onClick={queueTranscriptDiarizationRebuildJob}
                                            disabled={episodeBusy || recommendedOptimizationTier !== 'diarization_rebuild'}
                                            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                                        >
                                            {queueingDiarizationRebuild ? <Loader2 size={15} className="animate-spin" /> : <AudioLines size={15} />}
                                            Queue Rebuild
                                            <CircleHelp size={13} className="opacity-80" />
                                        </button>
                                        <div className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-72 -translate-x-1/2 rounded-lg border border-slate-200 bg-slate-900 px-3 py-2 text-[11px] leading-5 text-white shadow-xl group-hover:block">
                                            {transcriptOptimizationHelp.rebuild}
                                        </div>
                                    </div>
                                    <div className="group relative">
                                        <button
                                            onClick={queueTranscriptRetranscriptionJob}
                                            disabled={episodeBusy || recommendedOptimizationTier !== 'full_retranscription'}
                                            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-600 px-4 py-3 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
                                        >
                                            {queueingFullRetranscription ? <Loader2 size={15} className="animate-spin" /> : <RotateCcw size={15} />}
                                            Queue Retranscribe
                                            <CircleHelp size={13} className="opacity-80" />
                                        </button>
                                        <div className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-72 -translate-x-1/2 rounded-lg border border-slate-200 bg-slate-900 px-3 py-2 text-[11px] leading-5 text-white shadow-xl group-hover:block">
                                            {transcriptOptimizationHelp.retranscribe}
                                        </div>
                                    </div>
                                    <button
                                        onClick={onNavigateToTranscript}
                                        className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-100"
                                    >
                                        <FileText size={15} />
                                        Back to Transcript
                                    </button>
                                </div>
                            </div>

                            <div className="mt-5 grid gap-4 xl:grid-cols-[minmax(320px,0.9fr)_minmax(0,1fr)]">
                                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Diarization Benchmark</div>
                                    <div className="mt-1 text-sm leading-6 text-slate-600">
                                        Queue a benchmark variant with explicit sensitivity and speaker-match threshold so you can compare configurations without changing the default pipeline.
                                    </div>
                                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                                        <label className="text-xs text-slate-600">
                                            <span className="mb-1 block font-medium">Sensitivity</span>
                                            <select
                                                value={diarizationBenchmarkSensitivity}
                                                onChange={(e) => setDiarizationBenchmarkSensitivity(e.target.value as 'aggressive' | 'balanced' | 'conservative')}
                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                            >
                                                <option value="aggressive">Aggressive</option>
                                                <option value="balanced">Balanced</option>
                                                <option value="conservative">Conservative</option>
                                            </select>
                                        </label>
                                        <label className="text-xs text-slate-600">
                                            <span className="mb-1 block font-medium">Match Threshold</span>
                                            <input
                                                value={diarizationBenchmarkThreshold}
                                                onChange={(e) => setDiarizationBenchmarkThreshold(e.target.value)}
                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                placeholder="0.35"
                                            />
                                        </label>
                                    </div>
                                    <button
                                        onClick={queueTranscriptDiarizationBenchmarkJob}
                                        disabled={episodeBusy}
                                        className="mt-4 inline-flex items-center justify-center gap-2 rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                                    >
                                        {queueingDiarizationBenchmark ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
                                        Queue Benchmark Variant
                                    </button>
                                </div>

                                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                                    <div className="flex items-center justify-between gap-2">
                                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Rollback</div>
                                        {loadingTranscriptRollbackOptions && <Loader2 size={14} className="animate-spin text-slate-400" />}
                                    </div>
                                    <div className="mt-1 text-sm leading-6 text-slate-600">
                                        Restore a prior optimization run if a repair, rebuild, or retranscription regresses quality. The current transcript is backed up before restore.
                                    </div>
                                    <div className="mt-4 space-y-2">
                                        {transcriptRollbackOptions.length === 0 ? (
                                            <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">
                                                No rollback snapshots recorded for this episode yet.
                                            </div>
                                        ) : transcriptRollbackOptions.map((option) => (
                                            <div key={option.run_id} className="rounded-xl border border-slate-200 bg-white px-3 py-3">
                                                <div className="flex items-start justify-between gap-3">
                                                    <div className="min-w-0">
                                                        <div className="text-sm font-medium text-slate-800">Run {option.run_id} · {option.mode.replaceAll('_', ' ')}</div>
                                                        <div className="mt-0.5 text-[11px] text-slate-500">{new Date(option.created_at).toLocaleString()} · {option.pipeline_version}</div>
                                                        {option.note && <div className="mt-1 text-xs text-slate-600">{option.note}</div>}
                                                    </div>
                                                    <button
                                                        onClick={() => restoreTranscriptFromRun(option.run_id)}
                                                        disabled={!option.rollback_available || episodeBusy}
                                                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                                    >
                                                        {restoringTranscriptRunId === option.run_id ? 'Restoring...' : 'Restore'}
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                            <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                                <div className="min-w-0">
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Transcript Benchmark</div>
                                    <div className="mt-1 text-lg font-semibold text-slate-900">Gold windows and evaluation</div>
                                    <div className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
                                        Define hand-corrected gold windows for this episode, run deterministic scoring, then attach reviewer verdicts. Selection ranges still come from the transcript tab.
                                    </div>
                                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                        <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                            {transcriptGoldWindows.length} gold window{transcriptGoldWindows.length === 1 ? '' : 's'}
                                        </span>
                                        <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                            {transcriptEvaluationResults.length} evaluation result{transcriptEvaluationResults.length === 1 ? '' : 's'}
                                        </span>
                                        {transcriptEvaluationSummary && (
                                            <>
                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">Avg WER {transcriptEvaluationSummary.average_wer.toFixed(3)}</span>
                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">Avg CER {transcriptEvaluationSummary.average_cer.toFixed(3)}</span>
                                            </>
                                        )}
                                    </div>
                                </div>
                                <div className="grid gap-2 sm:grid-cols-2">
                                    <button
                                        onClick={useCurrentSelectionForGoldWindow}
                                        disabled={!selection}
                                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                        title="Use the current transcript or clip selection as the benchmark window range"
                                    >
                                        <Scissors size={14} />
                                        Use Selection Range
                                    </button>
                                    <button
                                        onClick={runTranscriptEvaluation}
                                        disabled={evaluatingTranscript || transcriptGoldWindows.length === 0}
                                        className="inline-flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                                    >
                                        {evaluatingTranscript ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                        Run Evaluation
                                    </button>
                                </div>
                            </div>

                            <div className="mt-5 grid gap-4 xl:grid-cols-[minmax(320px,0.95fr)_minmax(0,1.35fr)]">
                                <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                                    <div className="text-sm font-semibold text-slate-800">Gold Windows</div>
                                    <div className="grid gap-2 sm:grid-cols-2">
                                        <div className="sm:col-span-2">
                                            <label className="mb-1 block text-xs font-medium text-slate-600">Label</label>
                                            <input value={goldWindowLabelDraft} onChange={(e) => setGoldWindowLabelDraft(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Window label" />
                                        </div>
                                        <div>
                                            <label className="mb-1 block text-xs font-medium text-slate-600">Start</label>
                                            <input value={goldWindowStartDraft} onChange={(e) => setGoldWindowStartDraft(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="0.00" />
                                        </div>
                                        <div>
                                            <label className="mb-1 block text-xs font-medium text-slate-600">End</label>
                                            <input value={goldWindowEndDraft} onChange={(e) => setGoldWindowEndDraft(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="15.00" />
                                        </div>
                                        <div className="sm:col-span-2">
                                            <label className="mb-1 block text-xs font-medium text-slate-600">Reference Transcript</label>
                                            <textarea value={goldWindowReferenceDraft} onChange={(e) => setGoldWindowReferenceDraft(e.target.value)} rows={5} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Paste the hand-corrected reference transcript for this window." />
                                        </div>
                                        <div className="sm:col-span-2">
                                            <label className="mb-1 block text-xs font-medium text-slate-600">Entities</label>
                                            <input value={goldWindowEntitiesDraft} onChange={(e) => setGoldWindowEntitiesDraft(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Comma-separated entities to track" />
                                        </div>
                                        <div className="sm:col-span-2">
                                            <label className="mb-1 block text-xs font-medium text-slate-600">Notes</label>
                                            <textarea value={goldWindowNotesDraft} onChange={(e) => setGoldWindowNotesDraft(e.target.value)} rows={2} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Speaker boundaries, entity focus, overlap risk, punctuation notes..." />
                                        </div>
                                    </div>
                                    <button onClick={createTranscriptGoldWindow} disabled={savingTranscriptGoldWindow} className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">
                                        {savingTranscriptGoldWindow ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                                        Save Gold Window
                                    </button>
                                    {loadingTranscriptGoldWindows ? (
                                        <div className="inline-flex items-center gap-2 text-xs text-slate-500"><Loader2 size={14} className="animate-spin" />Loading benchmark windows...</div>
                                    ) : transcriptGoldWindowsError ? (
                                        <div className="text-xs text-rose-600">{transcriptGoldWindowsError}</div>
                                    ) : transcriptGoldWindows.length === 0 ? (
                                        <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">No gold windows yet. Create one from a selected range or enter a benchmark window manually.</div>
                                    ) : (
                                        <div className="space-y-2">
                                            {transcriptGoldWindows.map((window) => (
                                                <div key={window.id} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                                    <div className="flex items-start justify-between gap-2">
                                                        <div className="min-w-0">
                                                            <div className="text-sm font-medium text-slate-800">{window.label}</div>
                                                            <div className="mt-0.5 text-xs text-slate-500">{window.start_time.toFixed(2)}s to {window.end_time.toFixed(2)}s{window.language ? ` • ${window.language}` : ''}</div>
                                                        </div>
                                                        <button onClick={() => onSeek(window.start_time)} className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100">Jump</button>
                                                    </div>
                                                    <div className="mt-2 line-clamp-3 text-xs leading-5 text-slate-600">{window.reference_text}</div>
                                                    {window.entities.length > 0 && (
                                                        <div className="mt-2 flex flex-wrap gap-1">
                                                            {window.entities.map((entity) => (
                                                                <span key={`${window.id}-${entity}`} className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 ring-1 ring-slate-200">{entity}</span>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                                    <div className="flex items-center justify-between gap-3">
                                        <div>
                                            <div className="text-sm font-semibold text-slate-800">Evaluation Results</div>
                                            <div className="text-xs text-slate-500">WER and CER come from the stored reference windows. Reviewer verdicts capture the human judgment layer.</div>
                                        </div>
                                    </div>
                                    {loadingTranscriptEvaluationResults ? (
                                        <div className="inline-flex items-center gap-2 text-xs text-slate-500"><Loader2 size={14} className="animate-spin" />Loading evaluation results...</div>
                                    ) : transcriptEvaluationError ? (
                                        <div className="text-xs text-rose-600">{transcriptEvaluationError}</div>
                                    ) : transcriptEvaluationResults.length === 0 ? (
                                        <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">No evaluation results yet. Run evaluation after defining at least one gold window.</div>
                                    ) : (
                                        <div className="space-y-3">
                                            {transcriptEvaluationResults.map((result) => {
                                                const reviews = evaluationReviewsByResultId[result.id] || [];
                                                const verdict = evaluationReviewVerdictDrafts[result.id] || 'same';
                                                const reviewNotes = evaluationReviewNotesDrafts[result.id] || '';
                                                const reviewer = evaluationReviewReviewerDrafts[result.id] || '';
                                                return (
                                                    <div key={result.id} className="rounded-lg border border-slate-200 bg-white p-3">
                                                        <div className="flex flex-wrap items-center gap-2">
                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-700 ring-1 ring-slate-200">WER {result.wer.toFixed(3)}</span>
                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">CER {result.cer.toFixed(3)}</span>
                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">Unknown {result.unknown_speaker_rate.toFixed(2)}</span>
                                                            {result.entity_accuracy != null && <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">Entity {result.entity_accuracy.toFixed(2)}</span>}
                                                        </div>
                                                        <div className="mt-3 grid gap-3 lg:grid-cols-2">
                                                            <div>
                                                                <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Reference</div>
                                                                <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">{result.reference_text}</div>
                                                            </div>
                                                            <div>
                                                                <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Candidate</div>
                                                                <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">{result.candidate_text}</div>
                                                            </div>
                                                        </div>
                                                        <div className="mt-3 grid gap-2 lg:grid-cols-[140px_140px_minmax(0,1fr)_auto]">
                                                            <input value={reviewer} onChange={(e) => setEvaluationReviewReviewerDrafts((current) => ({ ...current, [result.id]: e.target.value }))} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Reviewer" />
                                                            <select value={verdict} onChange={(e) => setEvaluationReviewVerdictDrafts((current) => ({ ...current, [result.id]: e.target.value }))} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200">
                                                                <option value="better">Better</option>
                                                                <option value="same">Same</option>
                                                                <option value="worse">Worse</option>
                                                                <option value="bad_merge">Bad merge</option>
                                                                <option value="bad_speaker_reassignment">Bad speaker reassignment</option>
                                                                <option value="bad_entity_repair">Bad entity repair</option>
                                                                <option value="language_regression">Language regression</option>
                                                            </select>
                                                            <input value={reviewNotes} onChange={(e) => setEvaluationReviewNotesDrafts((current) => ({ ...current, [result.id]: e.target.value }))} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200" placeholder="Review notes" />
                                                            <button onClick={() => submitTranscriptEvaluationReview(result.id)} disabled={reviewingEvaluationResultId === result.id} className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50">
                                                                {reviewingEvaluationResultId === result.id ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                                                                Save Review
                                                            </button>
                                                        </div>
                                                        {reviews.length > 0 && (
                                                            <div className="mt-3 space-y-2">
                                                                {reviews.slice(0, 3).map((review) => (
                                                                    <div key={review.id} className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                                                                        <span className="font-semibold text-slate-700">{review.verdict.replaceAll('_', ' ')}</span>
                                                                        {review.reviewer ? ` by ${review.reviewer}` : ''}
                                                                        {review.notes ? ` • ${review.notes}` : ''}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

// ── OptimizeSidebar ────────────────────────────────────────────────────────────
// Rendered in the left sidebar column of VideoDetailPage when the optimize tab is active.
// Owns its own store reads so VDP can use it without threading extra props.

type OptimizeSidebarProps = {
    videoId: number;
    onNavigateToTranscript: () => void;
};

export function OptimizeSidebar({ videoId, onNavigateToTranscript }: OptimizeSidebarProps) {
    const transcriptQuality = useTranscriptStore((s) => s.transcriptQuality);
    const loadingTranscriptQuality = useTranscriptStore((s) => s.loadingTranscriptQuality);
    const transcriptQualityError = useTranscriptStore((s) => s.transcriptQualityError);

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

    const refreshAssessment = () => {
        void useTranscriptStore.getState().fetchTranscriptQuality(videoId);
    };

    return (
        <div className="h-full overflow-y-auto p-4">
            <div className="space-y-4">
                {/* Snapshot card */}
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-4 shadow-sm">
                    <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                                Optimization Navigator
                            </div>
                            <div className="mt-1 text-sm font-semibold text-slate-900">
                                Transcript Optimization Workbench
                            </div>
                            <div className="mt-1 text-xs leading-5 text-slate-600">
                                Benchmarking, rollback, repair, rebuild, and retranscription controls live in the main
                                stage. Use the transcript tab for reading and editing only.
                            </div>
                        </div>
                        <div className="flex shrink-0 gap-2">
                            <button
                                type="button"
                                onClick={onNavigateToTranscript}
                                className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/80 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                            >
                                <FileText size={14} />
                                Transcript
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
                        <div className="mt-3 text-xs leading-5 text-slate-500">
                            {transcriptQuality.reasons[0]}
                        </div>
                    )}
                    <div className="mt-3">
                        <button
                            type="button"
                            onClick={refreshAssessment}
                            disabled={loadingTranscriptQuality}
                            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                        >
                            {loadingTranscriptQuality ? (
                                <Loader2 size={14} className="animate-spin" />
                            ) : (
                                <RefreshCw size={14} />
                            )}
                            Refresh Assessment
                        </button>
                    </div>
                </div>

                {/* Workbench notes */}
                <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        Workbench Notes
                    </div>
                    <div className="mt-2 text-sm font-semibold text-slate-900">
                        Use transcript selections as benchmark ranges
                    </div>
                    <div className="mt-1 text-xs leading-5 text-slate-600">
                        The benchmark tools still use the current transcript selection when you click{' '}
                        <span className="font-medium">Use Selection Range</span>. Open the transcript tab whenever you
                        need to inspect or select a passage, then return here to run the evaluation.
                    </div>
                    <button
                        type="button"
                        onClick={onNavigateToTranscript}
                        className="mt-3 inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                    >
                        <FileText size={14} />
                        Open Transcript
                    </button>
                </div>
            </div>
        </div>
    );
}
