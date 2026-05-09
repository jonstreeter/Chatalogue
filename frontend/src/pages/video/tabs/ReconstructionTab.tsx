import { Bot, CheckCircle2, Clock, Download, Eraser, FileText, Loader2, PlayCircle, Plus, RefreshCw, RotateCcw, Save, Sparkles, Users, XCircle } from 'lucide-react';
import type { ReactNode } from 'react';
import type { ReconstructionWorkbench } from '../../../types';\nimport { formatTime } from '../../../lib/formatters';

type ReconstructionSidebarProps = {
    isActive: boolean;
    segmentsCount: number;
    reconstructionWorkbench: ReconstructionWorkbench | null;
    loadingReconstructionWorkbench: boolean;
    episodeBusy: boolean;
    hasReconstructionAudio: boolean;
    queueingReconstruction: boolean;
    reconstructionBusy: boolean;
    onRefreshWorkbench: () => void;
    onQueueReconstruction: () => void;
    onNavigateToTranscript: () => void;
};

export function ReconstructionSidebarTab({
    isActive,
    segmentsCount,
    reconstructionWorkbench,
    loadingReconstructionWorkbench,
    episodeBusy,
    hasReconstructionAudio,
    queueingReconstruction,
    reconstructionBusy,
    onRefreshWorkbench,
    onQueueReconstruction,
    onNavigateToTranscript,
}: ReconstructionSidebarProps) {
    if (!isActive) return null;

    return (
        <div className="h-full overflow-y-auto p-4">
            <div className="space-y-4">
                <div className="rounded-2xl border border-violet-200 bg-gradient-to-br from-violet-50 via-white to-fuchsia-50 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-violet-600">Reconstruction Studio</div>
                    <div className="mt-2 text-lg font-semibold text-slate-900">Central rebuild workspace</div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                        The main stage now becomes a dedicated reconstruction workbench, separate from transcript review. Use this tab when you want to audition speakers and rebuild the conversation.
                    </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Current State</div>
                    <div className="mt-3 space-y-2 text-sm text-slate-600">
                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                            <span>Transcript segments</span>
                            <span className="font-semibold text-slate-800">{segmentsCount}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                            <span>Workbench speakers</span>
                            <span className="font-semibold text-slate-800">{reconstructionWorkbench?.speaker_count ?? 0}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                            <span>Mode</span>
                            <span className="font-semibold text-slate-800">Performance-driven</span>
                        </div>
                    </div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Actions</div>
                    <div className="mt-3 flex flex-col gap-2">
                        <button
                            type="button"
                            onClick={onRefreshWorkbench}
                            disabled={loadingReconstructionWorkbench || segmentsCount === 0}
                            className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                        >
                            {loadingReconstructionWorkbench ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                            Refresh Workbench
                        </button>
                        <button
                            type="button"
                            onClick={onQueueReconstruction}
                            disabled={episodeBusy || segmentsCount === 0}
                            className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                        >
                            {queueingReconstruction || reconstructionBusy ? <Loader2 size={14} className="animate-spin" /> : <Bot size={14} />}
                            {hasReconstructionAudio ? 'Rebuild Reconstruction' : 'Reconstruct Audio'}
                        </button>
                        <button
                            type="button"
                            onClick={onNavigateToTranscript}
                            className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                        >
                            <FileText size={14} />
                            Back to Transcript
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

type ReconstructionStageProps = {
    isActive: boolean;
    reconstructionWorkbench: ReconstructionWorkbench | null;
    loadingReconstructionWorkbench: boolean;
    segmentsCount: number;
    reconstructionStudioTab: 'voices' | 'reconstruction';
    reconstructionStatus: string;
    reconstructionPaused: boolean;
    reconstructionBusy: boolean;
    hasReconstructionAudio: boolean;
    usingReconstructionForPlayback: boolean;
    reconstructionError?: string;
    reconstructionWorkbenchProgressNode: ReactNode;
    reconstructionWorkbenchActivityNode: ReactNode;
    reconstructionJobNode: ReactNode;
    ctx: any;
    onRefreshWorkbench: () => void;
    onSetStudioTab: (tab: 'voices' | 'reconstruction') => void;
};

export function ReconstructionTab({
    isActive,
    reconstructionWorkbench,
    loadingReconstructionWorkbench,
    segmentsCount,
    reconstructionStudioTab,
    reconstructionStatus,
    reconstructionPaused,
    reconstructionBusy,
    hasReconstructionAudio,
    usingReconstructionForPlayback,
    reconstructionError,
    reconstructionWorkbenchProgressNode,
    reconstructionWorkbenchActivityNode,
    reconstructionJobNode,
    ctx,
    onRefreshWorkbench,
    onSetStudioTab,
}: ReconstructionStageProps) {
    if (!isActive) return null;

    const speakerCount = reconstructionWorkbench?.speaker_count ?? 0;
    const approvedCount = reconstructionWorkbench?.speakers.filter((speaker) => speaker.approved).length ?? 0;
    const pendingCount = Math.max(0, speakerCount - approvedCount);

    const {
        selectedReconstructionSpeaker, resolveWorkbenchAudioUrl, segments, setSelectedReconstructionSpeakerId, selectedReconstructionSpeakerId,
        handleAddReconstructionSample, addingReconstructionSampleSpeakerId, handleApproveReconstructionSpeaker, approvingReconstructionSpeakerId,
        handleUpdateReconstructionSampleState, updatingReconstructionSampleKey, cleaningReconstructionSampleKey, handleCleanupReconstructionSample,
        reconstructionTestTextDrafts, setReconstructionTestTextDrafts, episodeBusy, handleTestReconstructionSpeaker, testingReconstructionSpeakerId,
        hasReconstructionAudio, reconstructionAudioUrl, video, handleSetReconstructionPlayback, usingReconstructionForPlayback, switchingReconstructionPlayback,
        selectedReconstructionPreviewSegmentId, setSelectedReconstructionPreviewSegmentId, reconstructionPreviewAudioUrl, reconstructionPreviewText,
        handlePreviewReconstructionSegment, queueingReconstruction, reconstructionBusy, handleQueueReconstruction, reconstructionInstructionDraft,
        setReconstructionInstructionDraft, savingReconstructionSettings, handleSaveReconstructionSettings, setReconstructionStudioTab,
    } = ctx;

    const renderReconstructionVoiceReview = () => {
        const selectedSpeaker = selectedReconstructionSpeaker;
        const selectedSpeakerSamples = selectedSpeaker?.samples ?? [];
        const selectedSpeakerPerformanceSample =
            selectedSpeaker?.samples.find((sample) => sample.selected) ||
            selectedSpeaker?.samples[0] ||
            null;
        const selectedSpeakerReferenceAudio = resolveWorkbenchAudioUrl(selectedSpeaker?.reference_audio_url);
        const selectedSpeakerLatestTestAudio = resolveWorkbenchAudioUrl(selectedSpeaker?.latest_test_audio_url);

        return (
            <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
                <div className="rounded-[24px] border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between gap-3">
                        <div>
                            <div className="text-sm font-semibold text-slate-900">Voice Queue</div>
                            <div className="mt-1 text-xs text-slate-500">Work through each speaker until all cloned voices are approved.</div>
                        </div>
                        {reconstructionWorkbench && (
                            <div className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-600">
                                {reconstructionWorkbench.speaker_count} total
                            </div>
                        )}
                    </div>

                    {!reconstructionWorkbench ? (
                        <div className="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
                            {segments.length === 0
                                ? 'Transcript and diarization are required before the reconstruction workbench can prepare voices.'
                                : loadingReconstructionWorkbench
                                    ? 'Preparing speaker references...'
                                    : 'Load the workbench to inspect diarized voices.'}
                        </div>
                    ) : (
                        <div className="mt-4 space-y-2">
                            {reconstructionWorkbench.speakers.map((speaker) => (
                                <button
                                    key={speaker.speaker_id}
                                    type="button"
                                    onClick={() => setSelectedReconstructionSpeakerId(speaker.speaker_id)}
                                    className={`w-full rounded-2xl border px-4 py-3 text-left transition-colors ${selectedReconstructionSpeakerId === speaker.speaker_id
                                        ? 'border-violet-300 bg-violet-50 shadow-sm'
                                        : 'border-slate-200 bg-slate-50 hover:border-violet-200 hover:bg-violet-50/70'
                                        }`}
                                >
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="min-w-0">
                                            <div className="truncate text-sm font-semibold text-slate-900">{speaker.speaker_name}</div>
                                            <div className="mt-1 text-xs text-slate-500">{speaker.segment_count} diarized segments</div>
                                        </div>
                                        <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${speaker.approved
                                            ? 'bg-emerald-100 text-emerald-700'
                                            : 'bg-amber-100 text-amber-700'
                                            }`}>
                                            {speaker.approved ? 'Approved' : 'Needs review'}
                                        </span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="space-y-6">
                    {!selectedSpeaker ? (
                        <div className="rounded-[24px] border border-dashed border-slate-200 bg-white px-6 py-16 text-center shadow-sm">
                            <div className="mx-auto max-w-xl">
                                <div className="text-lg font-semibold text-slate-900">Select a voice to review</div>
                                <p className="mt-2 text-sm leading-6 text-slate-600">
                                    Use the voice queue to review source clips, clean noisy performance samples, audition test TTS, and approve each model before reconstruction.
                                </p>
                            </div>
                        </div>
                    ) : (
                        <>
                            <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                                    <div>
                                        <div className="flex flex-wrap items-center gap-2">
                                            <div className="text-xl font-semibold text-slate-900">{selectedSpeaker.speaker_name}</div>
                                            <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${selectedSpeaker.approved
                                                ? 'bg-emerald-100 text-emerald-700'
                                                : 'bg-amber-100 text-amber-700'
                                                }`}>
                                                {selectedSpeaker.approved ? 'Voice approved' : 'Voice review pending'}
                                            </span>
                                        </div>
                                        <div className="mt-2 text-sm text-slate-500">
                                            {selectedSpeaker.segment_count} diarized segments, {selectedSpeakerSamples.length} active performance sample{selectedSpeakerSamples.length === 1 ? '' : 's'}
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => void handleAddReconstructionSample(selectedSpeaker.speaker_id)}
                                            disabled={addingReconstructionSampleSpeakerId === selectedSpeaker.speaker_id || !selectedSpeaker.can_add_sample}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                        >
                                            {addingReconstructionSampleSpeakerId === selectedSpeaker.speaker_id ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
                                            Add Performance Sample
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleApproveReconstructionSpeaker(selectedSpeaker.speaker_id, !selectedSpeaker.approved)}
                                            disabled={approvingReconstructionSpeakerId === selectedSpeaker.speaker_id}
                                            className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50 ${selectedSpeaker.approved
                                                ? 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                                                : 'bg-emerald-600 text-white hover:bg-emerald-700'
                                                }`}
                                        >
                                            {approvingReconstructionSpeakerId === selectedSpeaker.speaker_id ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                                            {selectedSpeaker.approved ? 'Mark as Needs Review' : 'Approve Voice Model'}
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)]">
                                <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                                    <div className="text-sm font-semibold text-slate-900">Source Material</div>
                                    <div className="mt-1 text-xs text-slate-500">Preview the clean timbre reference and review candidate performance samples for this speaker.</div>

                                    <div className="mt-5">
                                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                            <div className="mb-2 flex items-center justify-between gap-2">
                                                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Timbre Reference</div>
                                                {selectedSpeaker.reference_start_time != null && selectedSpeaker.reference_end_time != null && (
                                                    <span className="text-[11px] font-medium text-slate-500">
                                                        {formatTime(selectedSpeaker.reference_start_time)}-{formatTime(selectedSpeaker.reference_end_time)}
                                                    </span>
                                                )}
                                            </div>
                                            {selectedSpeakerReferenceAudio ? (
                                                <audio controls preload="none" src={selectedSpeakerReferenceAudio} className="w-full" />
                                            ) : (
                                                <div className="rounded-xl border border-dashed border-slate-200 bg-white px-3 py-5 text-center text-xs text-slate-500">
                                                    No clean reference clip is ready yet.
                                                </div>
                                            )}
                                            {selectedSpeaker.reference_text && (
                                                <div className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-600">
                                                    {selectedSpeaker.reference_text}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="mt-5">
                                        <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">Performance Samples</div>
                                        {selectedSpeakerSamples.length === 0 ? (
                                            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
                                                No active performance samples yet for this speaker.
                                            </div>
                                        ) : (
                                            <div className="space-y-4">
                                                {selectedSpeakerSamples.map((sample) => {
                                                    const sampleKey = `${selectedSpeaker.speaker_id}:${sample.segment_id}`;
                                                    const sampleAudioUrl = resolveWorkbenchAudioUrl(sample.audio_url);
                                                    const cleanedAudioUrl = resolveWorkbenchAudioUrl(sample.cleaned_audio_url);
                                                    const sampleBusy = updatingReconstructionSampleKey === sampleKey || cleaningReconstructionSampleKey === sampleKey;
                                                    return (
                                                        <div
                                                            key={sample.segment_id}
                                                            className={`rounded-2xl border p-4 ${sample.selected ? 'border-violet-300 bg-violet-50/70' : sample.rejected ? 'border-slate-200 bg-slate-50 opacity-85' : 'border-slate-200 bg-white'}`}
                                                        >
                                                            <div className="space-y-3">
                                                                <div className="flex flex-wrap items-center gap-2">
                                                                    <span className="text-sm font-semibold text-slate-900">Sample {sample.segment_id}</span>
                                                                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">{formatTime(sample.start_time)}-{formatTime(sample.end_time)}</span>
                                                                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">{sample.duration.toFixed(1)}s</span>
                                                                    {sample.selected && <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-medium text-violet-700">Selected</span>}
                                                                    {sample.rejected && <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-700">Rejected</span>}
                                                                    {cleanedAudioUrl && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-medium text-sky-700">Cleaned</span>}
                                                                </div>
                                                                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm leading-7 text-slate-700 break-words">
                                                                    {sample.text}
                                                                </div>
                                                                <div className="flex flex-wrap items-center gap-2">
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handleUpdateReconstructionSampleState(selectedSpeaker.speaker_id, sample.segment_id, { selected: true, rejected: false })}
                                                                        disabled={sampleBusy || sample.rejected}
                                                                        className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                                                                    >
                                                                        <CheckCircle2 size={13} />
                                                                        {sample.selected ? 'Selected Reference' : 'Use for Voice Model'}
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handleCleanupReconstructionSample(selectedSpeaker.speaker_id, sample.segment_id)}
                                                                        disabled={sampleBusy}
                                                                        className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                                                    >
                                                                        {cleaningReconstructionSampleKey === sampleKey ? <Loader2 size={13} className="animate-spin" /> : <Eraser size={13} />}
                                                                        {cleanedAudioUrl ? 'Re-clean Audio' : 'Clean Audio'}
                                                                    </button>
                                                                    {cleanedAudioUrl && (
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => void handleUpdateReconstructionSampleState(selectedSpeaker.speaker_id, sample.segment_id, { clear_cleaned: true })}
                                                                            disabled={sampleBusy}
                                                                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                                                        >
                                                                            <RotateCcw size={13} />
                                                                            Remove Cleaned
                                                                        </button>
                                                                    )}
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handleUpdateReconstructionSampleState(selectedSpeaker.speaker_id, sample.segment_id, sample.rejected ? { rejected: false } : { rejected: true })}
                                                                        disabled={sampleBusy}
                                                                        className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                                                    >
                                                                        <XCircle size={13} />
                                                                        {sample.rejected ? 'Restore Sample' : 'Reject Sample'}
                                                                    </button>
                                                                </div>
                                                            </div>

                                                            <div className="mt-4 grid gap-4 lg:grid-cols-2">
                                                                <div>
                                                                    <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Original Performance Clip</div>
                                                                    {sampleAudioUrl ? (
                                                                        <audio controls preload="none" src={sampleAudioUrl} className="w-full" />
                                                                    ) : (
                                                                        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-center text-xs text-slate-500">
                                                                            Original clip unavailable.
                                                                        </div>
                                                                    )}
                                                                </div>
                                                                <div>
                                                                    <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Cleaned Performance Clip</div>
                                                                    {cleanedAudioUrl ? (
                                                                        <audio controls preload="none" src={cleanedAudioUrl} className="w-full" />
                                                                    ) : (
                                                                        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-center text-xs text-slate-500">
                                                                            Run cleanup if you want a cleaner performance reference for this sample.
                                                                        </div>
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

                                <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                                    <div className="text-sm font-semibold text-slate-900">Voice Model Test</div>
                                    <div className="mt-1 text-xs text-slate-500">Compare a plain TTS sample against a performance-guided sample before approving this speaker.</div>

                                    <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-6 text-slate-600">
                                        Basic TTS checks timbre on arbitrary text. Performance-guided testing uses the selected performance sample as the delivery guide, which is usually the better check when cadence or speech rate feels off.
                                    </div>

                                    <label className="mt-4 block text-xs text-slate-600">
                                        <div className="mb-2 font-medium text-slate-700">Custom text for Basic TTS</div>
                                        <textarea
                                            value={reconstructionTestTextDrafts[selectedSpeaker.speaker_id] || ''}
                                            onChange={(e) => setReconstructionTestTextDrafts((prev) => ({ ...prev, [selectedSpeaker.speaker_id]: e.target.value }))}
                                            rows={7}
                                            disabled={episodeBusy}
                                            className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                                            placeholder="Enter a short sentence to audition the voice without performance guidance."
                                        />
                                    </label>

                                    <div className="mt-3 rounded-2xl border border-violet-200 bg-violet-50/60 px-4 py-3 text-xs leading-6 text-violet-900">
                                        <div className="text-[11px] font-semibold uppercase tracking-wide text-violet-700">Performance-Guided Source</div>
                                        {selectedSpeakerPerformanceSample ? (
                                            <div className="mt-1">
                                                Sample {selectedSpeakerPerformanceSample.segment_id} at {formatTime(selectedSpeakerPerformanceSample.start_time)}-{formatTime(selectedSpeakerPerformanceSample.end_time)}.
                                                The performance-guided test uses this sample's original transcript and delivery as the prosody guide.
                                            </div>
                                        ) : (
                                            <div className="mt-1">Select at least one performance sample to unlock the performance-guided voice test.</div>
                                        )}
                                    </div>

                                    <div className="mt-4 flex flex-wrap items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => void handleTestReconstructionSpeaker(
                                                selectedSpeaker.speaker_id,
                                                selectedSpeakerPerformanceSample?.segment_id,
                                                { performanceMode: false, useSelectedSampleText: false }
                                            )}
                                            disabled={testingReconstructionSpeakerId === selectedSpeaker.speaker_id}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                        >
                                            {testingReconstructionSpeakerId === selectedSpeaker.speaker_id ? <Loader2 size={14} className="animate-spin" /> : <Bot size={14} />}
                                            Run Basic TTS
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleTestReconstructionSpeaker(
                                                selectedSpeaker.speaker_id,
                                                selectedSpeakerPerformanceSample?.segment_id,
                                                { performanceMode: true, useSelectedSampleText: true }
                                            )}
                                            disabled={testingReconstructionSpeakerId === selectedSpeaker.speaker_id || !selectedSpeakerPerformanceSample}
                                            className="inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                                        >
                                            {testingReconstructionSpeakerId === selectedSpeaker.speaker_id ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                                            Run Performance-Guided Test
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleApproveReconstructionSpeaker(selectedSpeaker.speaker_id, true)}
                                            disabled={approvingReconstructionSpeakerId === selectedSpeaker.speaker_id}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-sm font-medium text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                                        >
                                            {approvingReconstructionSpeakerId === selectedSpeaker.speaker_id ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                                            Approve Voice
                                        </button>
                                    </div>

                                    <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Latest Voice Test</div>
                                        {selectedSpeakerLatestTestAudio ? (
                                            <>
                                                <audio controls preload="none" src={selectedSpeakerLatestTestAudio} className="mt-3 w-full" />
                                                <div className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-600">
                                                    <div className="font-medium text-slate-700">Last prompt</div>
                                                    <div className="mt-1 whitespace-pre-wrap">{selectedSpeaker.latest_test_text || 'No prompt saved.'}</div>
                                                    <div className="mt-2 text-[11px] uppercase tracking-wide text-violet-600">
                                                        {selectedSpeaker.latest_test_mode === 'performance' ? 'performance guided' : 'basic tts'}
                                                    </div>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="mt-3 rounded-xl border border-dashed border-slate-200 bg-white px-3 py-5 text-center text-xs text-slate-500">
                                                No TTS test has been generated for this speaker yet.
                                            </div>
                                        )}
                                    </div>

                                    <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Approval Checklist</div>
                                        <ul className="mt-3 space-y-2 text-sm text-slate-600">
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 size={14} className={`mt-0.5 ${selectedSpeaker.samples.some((sample) => sample.selected) ? 'text-emerald-600' : 'text-slate-300'}`} />
                                                <span>A performance sample is selected for this speaker.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 size={14} className={`mt-0.5 ${!!selectedSpeaker.latest_test_audio_url ? 'text-emerald-600' : 'text-slate-300'}`} />
                                                <span>You have listened to at least one TTS test output.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 size={14} className={`mt-0.5 ${selectedSpeaker.approved ? 'text-emerald-600' : 'text-slate-300'}`} />
                                                <span>The speaker voice model is approved.</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        );
    };

    const renderReconstructionBuildSuite = () => {
        const speakerCount = reconstructionWorkbench?.speaker_count ?? 0;
        const approvedCount = reconstructionWorkbench?.speakers.filter((speaker) => speaker.approved).length ?? 0;
        const pendingCount = Math.max(0, speakerCount - approvedCount);
        const previewCandidates = segments
            .filter((seg) => seg.speaker_id != null && String(seg.text || '').trim())
            .slice(0, 120);
        const selectedPreviewSegment =
            previewCandidates.find((seg) => seg.id === selectedReconstructionPreviewSegmentId) ||
            previewCandidates[0] ||
            null;

        if (!reconstructionWorkbench?.all_speakers_approved) {
            return (
                <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-6 shadow-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <div className="text-sm font-semibold text-amber-800">Voice approval required before reconstruction</div>
                            <div className="mt-2 text-sm leading-6 text-amber-900/80">
                                Review and approve all speaker voices first. {pendingCount} voice model{pendingCount === 1 ? '' : 's'} still need approval before the reconstruction tab can be used.
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={() => setReconstructionStudioTab('voices')}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-amber-300 bg-white px-4 py-2.5 text-sm font-medium text-amber-800 hover:bg-amber-100"
                        >
                            <Users size={14} />
                            Go back to Voices
                        </button>
                    </div>
                </div>
            );
        }

        return (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(340px,1.05fr)]">
                <div className="space-y-6">
                    <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <div className="text-sm font-semibold text-slate-900">Reconstruction Settings</div>
                                <div className="mt-1 text-xs text-slate-500">Configure cloning mode and performance-driven prosody before previewing or rebuilding.</div>
                            </div>
                            <button
                                type="button"
                                onClick={() => void handleSaveReconstructionSettings()}
                                disabled={episodeBusy || savingReconstructionSettings}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                            >
                                {savingReconstructionSettings ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
                                Save Settings
                            </button>
                        </div>
                        <div className="mt-4 space-y-4">
                            <div className="rounded-2xl border border-violet-200 bg-violet-50 px-4 py-3 text-xs leading-6 text-violet-900">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-violet-700">Mode</div>
                                <div className="mt-1 text-sm font-medium text-violet-900">Performance-driven reconstruction</div>
                                <div className="mt-1 text-violet-900/80">
                                    Reconstruction always uses the original segment as a prosody guide and the approved voice reference for timbre consistency.
                                </div>
                            </div>
                            <label className="block text-xs text-slate-600">
                                <div className="mb-1 font-medium text-slate-700">Performance Instruction</div>
                                <textarea
                                    value={reconstructionInstructionDraft}
                                    onChange={(e) => setReconstructionInstructionDraft(e.target.value)}
                                    disabled={episodeBusy}
                                    rows={7}
                                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                                    placeholder="Speak with the exact same intonation, emotion, rhythm, breathing, pauses, and emphasis as the reference audio..."
                                />
                            </label>
                            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-6 text-slate-600">
                                Performance mode uses the original utterance as a prosody guide and the approved speaker reference for timbre.
                            </div>
                        </div>
                    </div>

                    {hasReconstructionAudio && reconstructionAudioUrl && (
                        <div className="rounded-[24px] border border-violet-200 bg-white p-5 shadow-sm">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-sm font-semibold text-slate-900">Reconstructed Audio</div>
                                    <div className="mt-1 text-xs text-slate-500">Preview the current rebuilt WAV and choose whether playback should follow it.</div>
                                </div>
                                {video?.reconstruction_model && (
                                    <span className="inline-flex items-center rounded-full bg-violet-100 px-2.5 py-1 text-[11px] font-medium text-violet-700">
                                        {video.reconstruction_model}
                                    </span>
                                )}
                            </div>
                            <audio controls preload="none" src={reconstructionAudioUrl} className="mt-4 w-full" />
                            <div className="mt-4 flex flex-wrap items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() => void handleSetReconstructionPlayback(!usingReconstructionForPlayback)}
                                    disabled={episodeBusy || switchingReconstructionPlayback}
                                    className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50 ${usingReconstructionForPlayback
                                        ? 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                                        : 'border-violet-200 bg-violet-50 text-violet-700 hover:bg-violet-100'
                                        }`}
                                >
                                    {switchingReconstructionPlayback ? <Loader2 size={13} className="animate-spin" /> : <PlayCircle size={13} />}
                                    {usingReconstructionForPlayback ? 'Use Original Media for Playback' : 'Use Reconstruction for Playback'}
                                </button>
                                <a
                                    href={reconstructionAudioUrl}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-medium text-violet-700 hover:bg-violet-100"
                                >
                                    <Download size={13} />
                                    Open / Download WAV
                                </a>
                            </div>
                        </div>
                    )}
                </div>

                <div className="space-y-6">
                    <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <div className="text-sm font-semibold text-slate-900">Short Segment Preview</div>
                                <div className="mt-1 text-xs text-slate-500">Render a small sample with the current settings before running the full reconstruction job.</div>
                            </div>
                            <button
                                type="button"
                                onClick={() => selectedPreviewSegment && void handlePreviewReconstructionSegment(selectedPreviewSegment.id)}
                                disabled={previewingReconstructionSegment || !selectedPreviewSegment}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                            >
                                {previewingReconstructionSegment ? <Loader2 size={13} className="animate-spin" /> : <PlayCircle size={13} />}
                                Preview Segment
                            </button>
                        </div>

                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Approved voices</div>
                                <div className="mt-1 text-2xl font-semibold text-slate-900">{approvedCount}</div>
                            </div>
                            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Previewable segments</div>
                                <div className="mt-1 text-2xl font-semibold text-slate-900">{previewCandidates.length}</div>
                            </div>
                        </div>

                        <label className="mt-4 block text-xs text-slate-600">
                            <div className="mb-1 font-medium text-slate-700">Segment</div>
                            <select
                                value={selectedPreviewSegment?.id ?? ''}
                                onChange={(e) => setSelectedReconstructionPreviewSegmentId(Number(e.target.value))}
                                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                            >
                                {previewCandidates.map((seg) => (
                                    <option key={seg.id} value={seg.id}>
                                        {formatTime(seg.start_time)} - {String(seg.speaker || seg.speaker_id || 'Speaker')} - {String(seg.text || '').slice(0, 64)}
                                    </option>
                                ))}
                            </select>
                        </label>

                        {selectedPreviewSegment && (
                            <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-700">
                                {selectedPreviewSegment.text}
                            </div>
                        )}

                        {reconstructionPreviewAudioUrl && (
                            <div className="mt-4 rounded-2xl border border-violet-200 bg-violet-50/60 p-4">
                                <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-violet-700">
                                    <span>Preview Ready</span>
                                    <span className="rounded-full bg-white px-2 py-0.5 text-violet-700">performance mode</span>
                                </div>
                                <audio controls preload="none" src={reconstructionPreviewAudioUrl} className="mt-3 w-full" />
                                {reconstructionPreviewText && (
                                    <div className="mt-3 rounded-xl border border-violet-200 bg-white px-3 py-2 text-xs leading-5 text-slate-600">
                                        {reconstructionPreviewText}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <div className="rounded-[24px] border border-violet-200 bg-violet-50/60 p-5 shadow-sm">
                        <div className="text-sm font-semibold text-slate-900">Full Reconstruction</div>
                        <div className="mt-1 text-xs leading-6 text-slate-600">
                            Once the short preview sounds right, run the full conversation reconstruction across the diarized transcript timeline.
                        </div>
                        <div className="mt-4 flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                onClick={() => void handleQueueReconstruction(hasReconstructionAudio)}
                                disabled={episodeBusy || segments.length === 0}
                                className="inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                            >
                                {queueingReconstruction || reconstructionBusy ? <Loader2 size={15} className="animate-spin" /> : <Bot size={15} />}
                                {hasReconstructionAudio ? 'Rebuild Reconstruction' : 'Run Full Reconstruction'}
                            </button>
                            <div className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-white px-3 py-2 text-xs font-medium text-violet-700">
                                <Clock size={13} />
                                Preview first, then commit
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };



    return (
        <div className="flex-1 overflow-y-auto bg-[radial-gradient(circle_at_top,rgba(139,92,246,.12),transparent_38%),linear-gradient(180deg,#f8fafc,#f1f5f9)] p-6">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
                <div className="rounded-[28px] border border-violet-200 bg-white/90 p-6 shadow-[0_24px_60px_rgba(88,28,135,0.08)] backdrop-blur-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                        <div className="max-w-3xl">
                            <div className="text-xs font-semibold uppercase tracking-[0.28em] text-violet-600">Reconstruction Studio</div>
                            <h2 className="mt-2 text-2xl font-semibold text-slate-900">Voice review before full rebuild</h2>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                Approve each diarized speaker voice first, then move into reconstruction mode for short previews and the final studio-quality conversation rebuild.
                            </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                onClick={onRefreshWorkbench}
                                disabled={loadingReconstructionWorkbench || segmentsCount === 0}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-4 py-2.5 text-sm font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                            >
                                {loadingReconstructionWorkbench ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                                Refresh Workbench
                            </button>
                            <div className="inline-flex rounded-2xl border border-violet-200 bg-violet-50 p-1">
                                <button
                                    type="button"
                                    onClick={() => onSetStudioTab('voices')}
                                    className={`inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${reconstructionStudioTab === 'voices' ? 'bg-white text-violet-700 shadow-sm' : 'text-violet-700/80 hover:text-violet-800'}`}
                                >
                                    <Users size={15} />
                                    Voices
                                </button>
                                <button
                                    type="button"
                                    onClick={() => onSetStudioTab('reconstruction')}
                                    disabled={!reconstructionWorkbench?.all_speakers_approved}
                                    className={`inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${reconstructionStudioTab === 'reconstruction' ? 'bg-white text-violet-700 shadow-sm' : 'text-violet-700/80 hover:text-violet-800'}`}
                                >
                                    <Bot size={15} />
                                    Reconstruction
                                </button>
                            </div>
                        </div>
                    </div>
                    <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${reconstructionStatus === 'failed' ? 'border-red-200 bg-red-50 text-red-700' : reconstructionPaused ? 'border-amber-200 bg-amber-50 text-amber-700' : hasReconstructionAudio ? 'border-violet-200 bg-violet-50/70 text-violet-900' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
                        {reconstructionBusy
                            ? `Conversation reconstruction is ${reconstructionStatus === 'queued' ? 'queued' : 'building the reconstructed audio'}...`
                            : reconstructionPaused
                                ? 'Conversation reconstruction is paused.'
                                : reconstructionStatus === 'failed'
                                    ? String(reconstructionError || 'Conversation reconstruction failed.').slice(0, 240)
                                    : usingReconstructionForPlayback
                                        ? 'Reconstructed audio is currently driving local playback, so transcript-follow uses the rebuilt conversation track.'
                                        : hasReconstructionAudio
                                            ? 'Reconstructed audio is ready for preview, download, and optional playback handoff.'
                                            : 'Approve each voice model first, then move into the reconstruction tab for short previews and the final full build.'}
                    </div>
                    {reconstructionWorkbenchProgressNode ? (
                        <div className="mt-4">{reconstructionWorkbenchProgressNode}</div>
                    ) : reconstructionWorkbenchActivityNode ? (
                        <div className="mt-4">{reconstructionWorkbenchActivityNode}</div>
                    ) : null}
                    {reconstructionJobNode}
                    <div className="mt-4 grid gap-3 md:grid-cols-3">
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Voices</div>
                            <div className="mt-1 text-2xl font-semibold text-slate-900">{speakerCount}</div>
                        </div>
                        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-700">Approved</div>
                            <div className="mt-1 text-2xl font-semibold text-emerald-800">{approvedCount}</div>
                        </div>
                        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-amber-700">Still to review</div>
                            <div className="mt-1 text-2xl font-semibold text-amber-800">{pendingCount}</div>
                        </div>
                    </div>
                </div>

                {reconstructionStudioTab === 'voices'
                    ? renderReconstructionVoiceReview()
                    : renderReconstructionBuildSuite()}
            </div>
        </div>
    );
}
