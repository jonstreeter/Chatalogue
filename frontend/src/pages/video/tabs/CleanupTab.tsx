import { useEffect } from 'react';
import {
    Loader2,
    Save,
    AudioLines,
    RefreshCw,
    Download,
    CheckCircle2,
    Sparkles,
    FileText,
} from 'lucide-react';
import type { Video, Job, WorkbenchTaskProgress, CleanupWorkbench, ClearVoiceInstallInfo, ClearVoiceTestResult } from '../../../types';
import { useCleanupStore } from '../../../store/useCleanupStore';
import { useWorkbenchStore } from '../../../store/useWorkbenchStore';

// ── local types ────────────────────────────────────────────────────────────────
type WorkbenchActivity = {
    label: string;
    detail: string;
    tone: 'sky' | 'violet';
};

// ── helpers ────────────────────────────────────────────────────────────────────
const clampPercent = (value: number) =>
    Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));

const auxiliaryJobSortTime = (job: Job) => {
    const started = job.started_at ? Date.parse(job.started_at) : NaN;
    if (Number.isFinite(started)) return started;
    const created = job.created_at ? Date.parse(job.created_at) : NaN;
    return Number.isFinite(created) ? created : 0;
};

const latestAuxiliaryJobByType = (
    auxiliaryJobs: Job[],
    jobType: 'voicefixer_cleanup' | 'conversation_reconstruct'
) =>
    auxiliaryJobs
        .filter((job) => String(job.job_type || '').toLowerCase() === jobType)
        .sort((a, b) => auxiliaryJobSortTime(b) - auxiliaryJobSortTime(a))[0] || null;

const getAuxiliaryStageData = (job: Job | null, kind: 'voicefixer' | 'reconstruction') => {
    const rawProgress = clampPercent(Number(job?.progress || 0));
    const detail = String(job?.status_detail || '').toLowerCase();
    const isCompleted = String(job?.status || '').toLowerCase() === 'completed';

    if (kind === 'reconstruction') {
        const referencesActive = detail.includes('extracting speaker references');
        const modelLoadActive = detail.includes('loading reconstruction tts model');
        const synthActive =
            detail.includes('reconstructing segment') ||
            detail.includes('reconstructing long segment');
        const assembleActive = detail.includes('writing reconstructed');

        return {
            progress: rawProgress,
            detail: String(job?.status_detail || ''),
            stages: [
                {
                    key: 'references',
                    label: 'References',
                    state: (
                        rawProgress >= 15 || modelLoadActive || synthActive || assembleActive
                            ? 'completed'
                            : referencesActive || rawProgress > 0
                                ? 'active'
                                : 'pending'
                    ) as 'pending' | 'active' | 'completed',
                    percent:
                        referencesActive || rawProgress > 0
                            ? Math.max(10, (rawProgress / 15) * 100)
                            : rawProgress >= 15
                                ? 100
                                : 0,
                },
                {
                    key: 'model',
                    label: 'Model',
                    state: (
                        synthActive || assembleActive || rawProgress >= 25
                            ? 'completed'
                            : modelLoadActive || (rawProgress >= 15 && rawProgress < 25)
                                ? 'active'
                                : 'pending'
                    ) as 'pending' | 'active' | 'completed',
                    percent:
                        modelLoadActive || (rawProgress >= 15 && rawProgress < 25)
                            ? Math.max(10, ((rawProgress - 15) / 10) * 100)
                            : rawProgress >= 25
                                ? 100
                                : 0,
                },
                {
                    key: 'synth',
                    label: 'Synthesis',
                    state: (
                        assembleActive || rawProgress >= 94
                            ? 'completed'
                            : synthActive || (rawProgress >= 20 && rawProgress < 94)
                                ? 'active'
                                : 'pending'
                    ) as 'pending' | 'active' | 'completed',
                    percent:
                        synthActive || (rawProgress >= 20 && rawProgress < 94)
                            ? Math.max(5, ((rawProgress - 20) / 74) * 100)
                            : rawProgress >= 94
                                ? 100
                                : 0,
                },
                {
                    key: 'assemble',
                    label: 'Assemble',
                    state: (
                        isCompleted
                            ? 'completed'
                            : assembleActive || rawProgress >= 94
                                ? 'active'
                                : 'pending'
                    ) as 'pending' | 'active' | 'completed',
                    percent: isCompleted
                        ? 100
                        : assembleActive || rawProgress >= 94
                            ? Math.max(10, ((rawProgress - 94) / 6) * 100)
                            : 0,
                },
            ],
        };
    }

    const prepareActive = detail.includes('preparing media');
    const restoreActive = detail.includes('voicefixer restoration');
    const blendActive = detail.includes('blending restored');
    const levelActive = detail.includes('voice leveling');
    const mergeActive = detail.includes('merging restored') || detail.includes('replacing');

    return {
        progress: rawProgress,
        detail: String(job?.status_detail || ''),
        stages: [
            {
                key: 'prepare',
                label: 'Prepare',
                state: (
                    rawProgress >= 45 || restoreActive || blendActive || levelActive || mergeActive
                        ? 'completed'
                        : prepareActive || rawProgress > 0
                            ? 'active'
                            : 'pending'
                ) as 'pending' | 'active' | 'completed',
                percent:
                    prepareActive || rawProgress > 0
                        ? Math.max(10, (rawProgress / 45) * 100)
                        : rawProgress >= 45
                            ? 100
                            : 0,
            },
            {
                key: 'restore',
                label: 'Restore',
                state: (
                    rawProgress >= 62 || blendActive || levelActive || mergeActive
                        ? 'completed'
                        : restoreActive || (rawProgress >= 45 && rawProgress < 62)
                            ? 'active'
                            : 'pending'
                ) as 'pending' | 'active' | 'completed',
                percent:
                    restoreActive || (rawProgress >= 45 && rawProgress < 62)
                        ? Math.max(10, ((rawProgress - 45) / 17) * 100)
                        : rawProgress >= 62
                            ? 100
                            : 0,
            },
            {
                key: 'finish',
                label: 'Finish',
                state: (
                    isCompleted
                        ? 'completed'
                        : blendActive || levelActive || mergeActive || rawProgress >= 62
                            ? 'active'
                            : 'pending'
                ) as 'pending' | 'active' | 'completed',
                percent: isCompleted
                    ? 100
                    : blendActive || levelActive || mergeActive || rawProgress >= 62
                        ? Math.max(10, ((rawProgress - 62) / 38) * 100)
                        : 0,
            },
        ],
    };
};

// ── render helpers ─────────────────────────────────────────────────────────────
function renderAuxiliaryProgressCard(
    kind: 'voicefixer' | 'reconstruction',
    job: Job | null,
    options?: { compact?: boolean; className?: string }
) {
    if (!job) return null;
    const compact = !!options?.compact;
    const { progress, detail, stages } = getAuxiliaryStageData(job, kind);
    const accent =
        kind === 'voicefixer'
            ? { ring: 'text-sky-600', active: 'bg-sky-500', soft: 'bg-sky-100 text-sky-700 border-sky-200' }
            : { ring: 'text-violet-600', active: 'bg-violet-500', soft: 'bg-violet-100 text-violet-700 border-violet-200' };
    const size = compact ? 56 : 68;
    const center = compact ? 28 : 34;
    const radius = compact ? 22 : 28;
    const stroke = compact ? 6 : 7;
    const circumference = 2 * Math.PI * radius;
    const dashOffset = circumference - (clampPercent(progress) / 100) * circumference;

    return (
        <div className={`rounded-2xl border border-slate-200 bg-white/85 p-3 ${options?.className || ''}`}>
            <div className={`flex ${compact ? 'items-center gap-3' : 'items-start gap-4'}`}>
                <div className="relative shrink-0">
                    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={accent.ring}>
                        <circle cx={center} cy={center} r={radius} fill="none" stroke="currentColor" strokeOpacity="0.12" strokeWidth={stroke} />
                        <circle
                            cx={center}
                            cy={center}
                            r={radius}
                            fill="none"
                            stroke="currentColor"
                            strokeWidth={stroke}
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            strokeDashoffset={dashOffset}
                            transform={`rotate(-90 ${center} ${center})`}
                        />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center text-[11px] font-semibold text-slate-700">
                        {Math.round(progress)}%
                    </div>
                </div>
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${accent.soft}`}>
                            {kind === 'voicefixer' ? 'VoiceFixer' : 'Reconstruction'}
                        </span>
                        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                            {String(job.status || 'running')}
                        </span>
                    </div>
                    <div className="mt-2 text-sm font-medium text-slate-800">
                        {detail || (kind === 'voicefixer' ? 'Preparing cleanup job...' : 'Preparing reconstruction job...')}
                    </div>
                    <div className="mt-3 space-y-2">
                        {stages.map((stage) => (
                            <div key={stage.key}>
                                <div className="mb-1 flex items-center justify-between text-[11px]">
                                    <span className={stage.state === 'pending' ? 'text-slate-500' : 'text-slate-700'}>{stage.label}</span>
                                    <span className="text-slate-400">
                                        {stage.state === 'completed'
                                            ? 'done'
                                            : stage.state === 'active'
                                                ? `${Math.round(clampPercent(stage.percent))}%`
                                                : 'pending'}
                                    </span>
                                </div>
                                <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                                    <div
                                        className={`h-full transition-all duration-500 ${stage.state === 'pending' ? 'bg-slate-200' : accent.active}`}
                                        style={{ width: `${stage.state === 'completed' ? 100 : clampPercent(stage.percent)}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

function renderWorkbenchTaskProgressCard(progress: WorkbenchTaskProgress | null) {
    if (!progress || String(progress.status || '').toLowerCase() === 'idle') return null;
    const area = String(progress.area || '').toLowerCase();
    const tone = area === 'cleanup' ? 'sky' : 'violet';
    const palette =
        tone === 'sky'
            ? { shell: 'border-sky-200 bg-sky-50/85', badge: 'border-sky-200 bg-white text-sky-700', bar: 'bg-sky-500' }
            : { shell: 'border-violet-200 bg-violet-50/85', badge: 'border-violet-200 bg-white text-violet-700', bar: 'bg-violet-500' };
    const status = String(progress.status || 'running').toLowerCase();
    const pct = typeof progress.percent === 'number' ? clampPercent(progress.percent) : null;
    const meta = progress.current != null && progress.total != null ? `${progress.current}/${progress.total}` : null;
    const stageLabel = String(progress.stage || '').trim().replace(/[_-]+/g, ' ');

    return (
        <div className={`rounded-2xl border p-4 shadow-sm ${palette.shell}`}>
            <div className="flex items-start justify-between gap-3">
                <div>
                    <div className="text-sm font-semibold text-slate-900">{progress.message || 'Working...'}</div>
                    <div className="mt-1 text-xs leading-6 text-slate-600">
                        {stageLabel ? `Stage: ${stageLabel}` : 'Task is in progress.'}
                        {meta ? ` (${meta})` : ''}
                    </div>
                </div>
                <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${palette.badge}`}>
                    {status}
                </span>
            </div>
            {pct !== null ? (
                <>
                    <div className="mt-3 flex items-center justify-between text-[11px] font-medium text-slate-500">
                        <span>{meta || 'Progress'}</span>
                        <span>{Math.round(pct)}%</span>
                    </div>
                    <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-white/80">
                        <div className={`h-full rounded-full transition-all duration-300 ${palette.bar}`} style={{ width: `${pct}%` }} />
                    </div>
                </>
            ) : (
                <div className="mt-3 flex items-center gap-2 text-xs font-medium text-slate-600">
                    <Loader2 size={14} className="animate-spin" />
                    Waiting for task progress...
                </div>
            )}
        </div>
    );
}

function renderWorkbenchActivityCard(activity: WorkbenchActivity | null) {
    if (!activity) return null;
    const palette =
        activity.tone === 'sky'
            ? { shell: 'border-sky-200 bg-sky-50/85', badge: 'border-sky-200 bg-white text-sky-700' }
            : { shell: 'border-violet-200 bg-violet-50/85', badge: 'border-violet-200 bg-white text-violet-700' };

    return (
        <div className={`rounded-2xl border p-4 shadow-sm ${palette.shell}`}>
            <div className="flex items-start justify-between gap-3">
                <div>
                    <div className="text-sm font-semibold text-slate-900">{activity.label}</div>
                    <div className="mt-1 text-xs leading-6 text-slate-600">{activity.detail}</div>
                </div>
                <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${palette.badge}`}>
                    Working
                </span>
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs font-medium text-slate-600">
                <Loader2 size={14} className="animate-spin" />
                Working...
            </div>
        </div>
    );
}

// ── props ──────────────────────────────────────────────────────────────────────
interface CleanupTabProps {
    video: Video;
    videoId: number;
    isActive: boolean;
    onVideoUpdated: (v: Video) => void;
    episodeBusy: boolean;
    renderMainPlayer: (containerClassName: string) => React.ReactNode;
    onNavigateToTranscript: () => void;
}

// ── component ──────────────────────────────────────────────────────────────────
export function CleanupTab({
    video,
    videoId,
    isActive,
    onVideoUpdated,
    episodeBusy,
    renderMainPlayer,
    onNavigateToTranscript,
}: CleanupTabProps) {
    // derived from video prop
    const mediaSourceType = String(video?.media_source_type || 'upload').toLowerCase();
    const isUploadedMedia = mediaSourceType === 'upload';
    const voiceFixerStatus = String(video?.voicefixer_status || '').toLowerCase();
    const voiceFixerBusy = isUploadedMedia && (voiceFixerStatus === 'queued' || voiceFixerStatus === 'processing');
    const voiceFixerPaused = isUploadedMedia && voiceFixerStatus === 'paused';
    const hasVoiceFixerCleaned = isUploadedMedia && !!video?.voicefixer_cleaned_path;
    const voiceFixerApplyScope = String(
        video?.voicefixer_apply_scope || (video?.voicefixer_use_cleaned ? 'both' : 'none')
    ).toLowerCase();
    const usingVoiceFixerForPlayback =
        isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'playback');
    const usingVoiceFixerForProcessing =
        isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'processing');
    const voiceFixerStatusTone =
        voiceFixerStatus === 'failed'
            ? 'border-red-200 bg-red-50 text-red-700'
            : voiceFixerPaused
                ? 'border-amber-200 bg-amber-50 text-amber-700'
                : usingVoiceFixerForPlayback || usingVoiceFixerForProcessing
                    ? 'border-sky-200 bg-sky-50 text-sky-700'
                    : 'border-slate-200 bg-slate-50 text-slate-600';
    const voiceFixerStatusMessage = voiceFixerBusy
        ? `VoiceFixer is ${voiceFixerStatus === 'queued' ? 'queued' : 'cleaning this uploaded media'}...`
        : voiceFixerPaused
            ? 'VoiceFixer cleanup is paused.'
            : voiceFixerStatus === 'failed'
                ? String(video?.voicefixer_error || 'VoiceFixer cleanup failed.').slice(0, 240)
                : usingVoiceFixerForPlayback && usingVoiceFixerForProcessing
                    ? 'Using VoiceFixer-cleaned media for playback and processing.'
                    : usingVoiceFixerForPlayback
                        ? 'Using VoiceFixer-cleaned media for playback only. Processing still uses the original upload.'
                        : usingVoiceFixerForProcessing
                            ? 'Using VoiceFixer-cleaned media for processing only. Playback still uses the original upload.'
                            : hasVoiceFixerCleaned
                                ? 'VoiceFixer-cleaned media is available, but playback and processing are currently using the original upload.'
                                : 'Tune cleanup settings, rebuild the cleaned pass, and decide where it should apply.';

    // store state
    const {
        queueingVoiceFixer,
        loadingCleanupWorkbench,
        cleanupWorkbench,
        analyzingCleanupWorkbench,
        runningClearVoiceModel,
        selectingCleanupCandidateId,
        clearVoiceInstallInfo,
        loadingClearVoiceInstallInfo,
        installingClearVoice,
        repairingClearVoice,
        testingClearVoice,
        clearVoiceTestResult,
        savingVoiceFixerSettings,
        voiceFixerModeDraft,
        voiceFixerMixDraft,
        voiceFixerLevelingDraft,
        voiceFixerApplyScopeDraft,
        fetchCleanupWorkbench,
        fetchClearVoiceInstallInfo,
        handleQueueVoiceFixer,
        handleSaveVoiceFixerSettings,
        handleAnalyzeCleanupWorkbench,
        handleRunClearVoiceCandidate,
        handleSelectCleanupCandidate,
        handleInstallClearVoice,
        handleTestClearVoice,
        handleRepairClearVoice,
    } = useCleanupStore();

    const { auxiliaryJobs, workbenchTaskProgress } = useWorkbenchStore();

    // derived workbench state
    const currentWorkbenchProgress =
        workbenchTaskProgress && String(workbenchTaskProgress.status || '').toLowerCase() !== 'idle'
            ? workbenchTaskProgress
            : null;
    const cleanupWorkbenchProgress =
        currentWorkbenchProgress &&
        String(currentWorkbenchProgress.area || '').toLowerCase() === 'cleanup'
            ? currentWorkbenchProgress
            : null;

    const voiceFixerJob = latestAuxiliaryJobByType(auxiliaryJobs, 'voicefixer_cleanup');

    const cleanupWorkbenchActivity: WorkbenchActivity | null = loadingCleanupWorkbench
        ? { label: 'Loading cleanup workbench', detail: 'Refreshing the ClearVoice prep area and current pre-cleanup candidates.', tone: 'sky' }
        : analyzingCleanupWorkbench
            ? { label: 'Analyzing uploaded audio', detail: 'Inspecting the original upload so the pre-cleanup bench can suggest the right enhancement tools.', tone: 'sky' }
            : runningClearVoiceModel !== null
                ? { label: 'Generating ClearVoice candidate', detail: `Running ${runningClearVoiceModel} before the existing VoiceFixer cleanup step.`, tone: 'sky' }
                : selectingCleanupCandidateId !== null
                    ? { label: 'Selecting pre-cleanup source', detail: 'Updating which ClearVoice candidate should feed the existing VoiceFixer cleanup pass.', tone: 'sky' }
                    : installingClearVoice
                        ? { label: 'Installing ClearVoice', detail: 'Adding the ClearVoice runtime to the backend environment.', tone: 'sky' }
                        : testingClearVoice
                            ? { label: 'Testing ClearVoice', detail: 'Verifying the ClearVoice runtime before generating enhancement candidates.', tone: 'sky' }
                            : savingVoiceFixerSettings
                                ? { label: 'Saving cleanup settings', detail: 'Updating the cleanup recipe for this episode before the next rebuild.', tone: 'sky' }
                                : queueingVoiceFixer
                                    ? { label: 'Starting cleanup job', detail: 'Queueing the VoiceFixer pass now. The stage breakdown appears below once the job is registered.', tone: 'sky' }
                                    : null;

    // fetch on activation
    useEffect(() => {
        if (!isActive) return;
        void fetchCleanupWorkbench(videoId);
        void fetchClearVoiceInstallInfo();
    }, [isActive, videoId, fetchCleanupWorkbench, fetchClearVoiceInstallInfo]);

    // sync draft state from video when video changes
    useEffect(() => {
        useCleanupStore.setState({
            voiceFixerModeDraft: video?.voicefixer_mode ?? 0,
            voiceFixerMixDraft: video?.voicefixer_mix_ratio ?? 1,
            voiceFixerLevelingDraft: (video?.voicefixer_leveling_mode as 'off' | 'gentle' | 'balanced' | 'strong') ?? 'off',
            voiceFixerApplyScopeDraft: (video?.voicefixer_apply_scope as 'none' | 'playback' | 'processing' | 'both') ?? 'none',
        });
    }, [video?.id]);

    // render helpers using local fns
    const clearVoiceEnhancementModels = [
        { model: 'FRCRN_SE_16K', label: 'FRCRN 16k', detail: 'Focused on rough, narrowband speech cleanup.' },
        { model: 'MossFormerGAN_SE_16K', label: 'MossFormerGAN 16k', detail: 'Stronger denoising pass for difficult speech.' },
        { model: 'MossFormer2_SE_48K', label: 'MossFormer2 48k', detail: 'Full-band enhancement for higher fidelity sources.' },
    ];
    const clearVoiceInstalled = !!clearVoiceInstallInfo?.installed;
    const clearVoiceNeedsRestart = !!clearVoiceInstallInfo?.restart_required;
    const clearVoiceRuntimeReady = !!clearVoiceInstallInfo?.runtime_ready;
    const selectedPreCleanupLabel = cleanupWorkbench?.selected_source_label || 'Original upload';
    const desiredApplyScopeLabel =
        voiceFixerApplyScopeDraft === 'both'
            ? 'Playback + processing'
            : voiceFixerApplyScopeDraft === 'playback'
                ? 'Playback only'
                : voiceFixerApplyScopeDraft === 'processing'
                    ? 'Processing only'
                    : 'Original only';

    return (
        <div className="flex-1 overflow-y-auto bg-[radial-gradient(circle_at_top,rgba(14,165,233,.12),transparent_38%),linear-gradient(180deg,#f8fafc,#eff6ff)] p-6">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
                <div className="rounded-[28px] border border-sky-200 bg-white/90 p-6 shadow-[0_24px_60px_rgba(14,116,144,0.08)] backdrop-blur-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                        <div className="max-w-3xl">
                            <div className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-600">Cleanup Workbench</div>
                            <h2 className="mt-2 text-2xl font-semibold text-slate-900">Tune, rebuild, and route the cleaned pass</h2>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                VoiceFixer cleanup now lives in its own studio. Dial in the pass, rebuild the cleaned media, then decide whether playback and downstream processing should use it.
                            </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                onClick={() => void handleSaveVoiceFixerSettings(videoId, onVideoUpdated)}
                                disabled={episodeBusy || savingVoiceFixerSettings}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-4 py-2.5 text-sm font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                            >
                                {savingVoiceFixerSettings ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                                Save Settings
                            </button>
                            <button
                                type="button"
                                onClick={() => void handleQueueVoiceFixer(videoId, hasVoiceFixerCleaned, onVideoUpdated)}
                                disabled={episodeBusy}
                                className="inline-flex items-center gap-1.5 rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-sky-700 disabled:opacity-50"
                            >
                                {queueingVoiceFixer || voiceFixerBusy ? <Loader2 size={15} className="animate-spin" /> : <AudioLines size={15} />}
                                {hasVoiceFixerCleaned ? 'Rebuild Cleanup' : 'Run Cleanup'}
                            </button>
                        </div>
                    </div>
                    <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${voiceFixerStatusTone}`}>
                        {voiceFixerStatusMessage}
                    </div>
                    {cleanupWorkbenchActivity && (
                        <div className="mt-4">
                            {renderWorkbenchActivityCard(cleanupWorkbenchActivity)}
                        </div>
                    )}
                    {voiceFixerJob && renderAuxiliaryProgressCard('voicefixer', voiceFixerJob, { className: 'mt-4 border-sky-200 bg-sky-50/50' })}
                    <div className="mt-4 grid gap-3 md:grid-cols-3">
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Cleanup mode</div>
                            <div className="mt-1 text-2xl font-semibold text-slate-900">Mode {voiceFixerModeDraft}</div>
                        </div>
                        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-sky-700">Playback route</div>
                            <div className="mt-1 text-2xl font-semibold text-sky-900">{usingVoiceFixerForPlayback ? 'Cleaned' : 'Original'}</div>
                        </div>
                        <div className="rounded-2xl border border-indigo-200 bg-indigo-50 px-4 py-3">
                            <div className="text-[11px] font-semibold uppercase tracking-wide text-indigo-700">Processing route</div>
                            <div className="mt-1 text-2xl font-semibold text-indigo-900">{usingVoiceFixerForProcessing ? 'Cleaned' : 'Original'}</div>
                        </div>
                    </div>
                </div>

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1.08fr)_380px]">
                    <div className="space-y-6">
                        <div className="rounded-[24px] border border-sky-200 bg-white p-5 shadow-sm">
                            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                <div className="max-w-3xl">
                                    <div className="text-sm font-semibold text-slate-900">Pre-Cleanup Bench</div>
                                    <div className="mt-1 text-xs leading-6 text-slate-500">
                                        Run ClearVoice enhancement passes before the existing VoiceFixer cleanup. The selected candidate becomes the input source for the next VoiceFixer rebuild.
                                    </div>
                                </div>
                                <div className="flex flex-wrap items-center gap-2">
                                    <button
                                        type="button"
                                        onClick={() => void handleAnalyzeCleanupWorkbench(videoId)}
                                        disabled={episodeBusy || analyzingCleanupWorkbench}
                                        className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                    >
                                        {analyzingCleanupWorkbench ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                                        Analyze Upload
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => void fetchCleanupWorkbench(videoId)}
                                        disabled={loadingCleanupWorkbench}
                                        className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                    >
                                        {loadingCleanupWorkbench ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                                        Refresh Bench
                                    </button>
                                </div>
                            </div>

                            <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">VoiceFixer input source</div>
                                <div className="mt-1 text-lg font-semibold text-slate-900">{selectedPreCleanupLabel}</div>
                                <div className="mt-1 text-xs text-slate-500">
                                    Use a ClearVoice candidate here when the original upload needs denoising before the existing VoiceFixer cleanup pass.
                                </div>
                            </div>

                            {cleanupWorkbenchProgress ? (
                                <div className="mt-4">
                                    {renderWorkbenchTaskProgressCard(cleanupWorkbenchProgress)}
                                </div>
                            ) : cleanupWorkbenchActivity ? (
                                <div className="mt-4">
                                    {renderWorkbenchActivityCard(cleanupWorkbenchActivity)}
                                </div>
                            ) : null}

                            <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm leading-6 text-slate-700 ${clearVoiceInstalled && !clearVoiceNeedsRestart && clearVoiceRuntimeReady ? 'border-emerald-200 bg-emerald-50/70' : 'border-amber-200 bg-amber-50/70'}`}>
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div>
                                        <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">ClearVoice Runtime</div>
                                        <div className="mt-1">
                                            {clearVoiceInstalled
                                                ? (clearVoiceNeedsRestart
                                                    ? (clearVoiceInstallInfo?.message || 'ClearVoice is installed but the backend needs a restart before it can use it.')
                                                    : (clearVoiceInstallInfo?.message || (clearVoiceInstallInfo?.runtime_ready ? 'ClearVoice is ready for pre-cleanup candidate generation.' : 'ClearVoice is installed, but its runtime still needs repair.')))
                                                : (loadingClearVoiceInstallInfo
                                                    ? 'Checking ClearVoice availability...'
                                                    : (clearVoiceInstallInfo?.message || 'ClearVoice is not installed yet. Install it to generate enhancement candidates before VoiceFixer.'))}
                                        </div>
                                        {clearVoiceInstalled && (
                                            <div className="mt-1 text-[11px] text-slate-500">
                                                Torch: {clearVoiceInstallInfo?.torch_version || 'not detected'} | Torchaudio: {clearVoiceInstallInfo?.torchaudio_version || 'not detected'}
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => void handleInstallClearVoice()}
                                            disabled={installingClearVoice || clearVoiceInstalled || loadingClearVoiceInstallInfo}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                        >
                                            {installingClearVoice ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                                            {clearVoiceInstalled ? 'Installed' : 'Install ClearVoice'}
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleTestClearVoice()}
                                            disabled={testingClearVoice || !clearVoiceInstalled || clearVoiceNeedsRestart || loadingClearVoiceInstallInfo}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                        >
                                            {testingClearVoice ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                                            Self-Test
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleRepairClearVoice()}
                                            disabled={repairingClearVoice || !clearVoiceInstalled || loadingClearVoiceInstallInfo}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800 hover:bg-amber-100 disabled:opacity-50"
                                        >
                                            {repairingClearVoice ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                                            Repair Runtime
                                        </button>
                                    </div>
                                </div>
                                {clearVoiceTestResult && (
                                    <div className={`mt-3 rounded-xl border px-3 py-2 text-xs ${clearVoiceTestResult.status === 'ok' ? 'border-emerald-200 bg-white text-emerald-700' : 'border-red-200 bg-white text-red-700'}`}>
                                        <div>{clearVoiceTestResult.detail || clearVoiceTestResult.error || 'ClearVoice test finished.'}</div>
                                        {(clearVoiceTestResult.torch_version || clearVoiceTestResult.torchaudio_version) && (
                                            <div className="mt-1 text-[11px]">
                                                Torch: {clearVoiceTestResult.torch_version || 'not detected'} | Torchaudio: {clearVoiceTestResult.torchaudio_version || 'not detected'}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <div className="mt-4 grid gap-3 md:grid-cols-4">
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Analyzed source</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">{cleanupWorkbench?.analysis?.source_label || 'Original upload'}</div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Duration</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                        {cleanupWorkbench?.analysis?.duration_seconds != null ? `${cleanupWorkbench.analysis.duration_seconds.toFixed(1)}s` : 'Not analyzed'}
                                    </div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Sample rate</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                        {cleanupWorkbench?.analysis?.sample_rate != null ? `${cleanupWorkbench.analysis.sample_rate} Hz` : 'Not analyzed'}
                                    </div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Channels</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                        {cleanupWorkbench?.analysis?.channels != null ? cleanupWorkbench.analysis.channels : 'Not analyzed'}
                                    </div>
                                </div>
                            </div>

                            <div className="mt-5">
                                <div className="flex items-center justify-between gap-3">
                                    <div>
                                        <div className="text-sm font-semibold text-slate-900">Enhancement Candidates</div>
                                        <div className="mt-1 text-xs text-slate-500">Generate one or more ClearVoice candidates, audition them, then choose the one that should feed the existing VoiceFixer cleanup step.</div>
                                    </div>
                                </div>
                                <div className="mt-3 grid gap-3 md:grid-cols-3">
                                    {clearVoiceEnhancementModels.map((entry) => (
                                        <button
                                            key={entry.model}
                                            type="button"
                                            onClick={() => void handleRunClearVoiceCandidate(videoId, entry.model)}
                                            disabled={!clearVoiceInstalled || clearVoiceNeedsRestart || !clearVoiceRuntimeReady || runningClearVoiceModel !== null}
                                            className="rounded-2xl border border-sky-200 bg-sky-50/70 p-4 text-left transition-colors hover:bg-sky-100 disabled:opacity-50"
                                        >
                                            <div className="flex items-center gap-2 text-sky-700">
                                                {runningClearVoiceModel === entry.model ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                                                <span className="text-sm font-semibold">{entry.label}</span>
                                            </div>
                                            <div className="mt-2 text-xs leading-6 text-slate-600">{entry.detail}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="mt-5 space-y-3">
                                {cleanupWorkbench?.candidates?.length ? cleanupWorkbench.candidates.map((candidate) => (
                                    <div key={candidate.candidate_id} className={`rounded-2xl border p-4 ${candidate.selected_for_processing ? 'border-sky-300 bg-sky-50/70' : 'border-slate-200 bg-white'}`}>
                                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                            <div className="min-w-0">
                                                <div className="flex flex-wrap items-center gap-2">
                                                    <div className="text-sm font-semibold text-slate-900">{candidate.model_name}</div>
                                                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">{candidate.stage}</span>
                                                    {candidate.selected_for_processing && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-medium text-sky-700">Feeds VoiceFixer</span>}
                                                </div>
                                                <div className="mt-1 text-xs text-slate-500">Source: {candidate.source_label || 'Original upload'}</div>
                                            </div>
                                            <div className="flex flex-wrap items-center gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => void handleSelectCleanupCandidate(videoId, candidate.selected_for_processing ? null : candidate.candidate_id)}
                                                    disabled={selectingCleanupCandidateId !== null}
                                                    className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50 ${candidate.selected_for_processing ? 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50' : 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100'}`}
                                                >
                                                    {selectingCleanupCandidateId === candidate.candidate_id ? <Loader2 size={13} className="animate-spin" /> : <AudioLines size={13} />}
                                                    {candidate.selected_for_processing ? 'Use Original Upload' : 'Use Before VoiceFixer'}
                                                </button>
                                            </div>
                                        </div>
                                        {candidate.audio_url && (
                                            <audio controls preload="none" src={candidate.audio_url} className="mt-3 w-full" />
                                        )}
                                        <div className="mt-3 grid gap-2 sm:grid-cols-4">
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">Duration: {candidate.duration_seconds != null ? `${candidate.duration_seconds.toFixed(1)}s` : 'n/a'}</div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">Sample rate: {candidate.sample_rate != null ? `${candidate.sample_rate} Hz` : 'n/a'}</div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">Peak: {candidate.peak != null ? candidate.peak.toFixed(3) : 'n/a'}</div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">RMS: {candidate.rms != null ? candidate.rms.toFixed(4) : 'n/a'}</div>
                                        </div>
                                    </div>
                                )) : (
                                    <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
                                        No ClearVoice candidates yet. Analyze the upload, then generate one or more enhancement passes before VoiceFixer.
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                <div>
                                    <div className="text-sm font-semibold text-slate-900">Audition Current Episode Media</div>
                                    <div className="mt-1 text-xs text-slate-500">
                                        Use the player source switcher below to compare the original upload, cleanup pass, and reconstruction without leaving the workbench.
                                    </div>
                                </div>
                                <span className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-600">
                                    Draft route: {desiredApplyScopeLabel}
                                </span>
                            </div>
                            <div className="mt-4 overflow-hidden rounded-[24px] border border-slate-200 bg-black">
                                {renderMainPlayer('h-[360px] w-full')}
                            </div>
                            <div className="mt-4 grid gap-3 md:grid-cols-3">
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">1. Prep</div>
                                    <div className="mt-1">Generate ClearVoice candidates first when the raw upload needs denoising before VoiceFixer.</div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">2. Rebuild</div>
                                    <div className="mt-1">Run the existing VoiceFixer cleanup after you choose which pre-cleanup source should feed it.</div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">3. Route</div>
                                    <div className="mt-1">Choose whether playback, downstream processing, or both should use the final cleaned pass.</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-sm font-semibold text-slate-900">VoiceFixer Tuning</div>
                                    <div className="mt-1 text-xs text-slate-500">These settings are saved on the episode and used for the next rebuild.</div>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => void handleSaveVoiceFixerSettings(videoId, onVideoUpdated)}
                                    disabled={episodeBusy || savingVoiceFixerSettings}
                                    className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                >
                                    {savingVoiceFixerSettings ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
                                    Save
                                </button>
                            </div>

                            <div className="mt-4 space-y-4">
                                <label className="text-xs text-slate-600">
                                    <div className="font-medium text-slate-700 mb-1">Cleanup Mode</div>
                                    <select
                                        value={voiceFixerModeDraft}
                                        onChange={(e) => useCleanupStore.setState({ voiceFixerModeDraft: Number(e.target.value) })}
                                        disabled={episodeBusy}
                                        className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                                    >
                                        <option value={0}>Mode 0: Balanced</option>
                                        <option value={1}>Mode 1: Smoother / darker</option>
                                        <option value={2}>Mode 2: Aggressive repair</option>
                                    </select>
                                </label>
                                <label className="text-xs text-slate-600">
                                    <div className="mb-1 flex items-center justify-between gap-2">
                                        <span className="font-medium text-slate-700">Clean Mix</span>
                                        <span>{Math.round(voiceFixerMixDraft * 100)}%</span>
                                    </div>
                                    <input
                                        type="range"
                                        min={0}
                                        max={100}
                                        step={5}
                                        value={Math.round(voiceFixerMixDraft * 100)}
                                        onChange={(e) => useCleanupStore.setState({ voiceFixerMixDraft: Number(e.target.value) / 100 })}
                                        disabled={episodeBusy}
                                        className="w-full accent-sky-600"
                                    />
                                    <div className="mt-1 text-[11px] text-slate-500">Lower values keep more of the original natural tone.</div>
                                </label>
                                <label className="text-xs text-slate-600">
                                    <div className="font-medium text-slate-700 mb-1">Voice Leveling</div>
                                    <select
                                        value={voiceFixerLevelingDraft}
                                        onChange={(e) => useCleanupStore.setState({ voiceFixerLevelingDraft: e.target.value as 'off' | 'gentle' | 'balanced' | 'strong' })}
                                        disabled={episodeBusy}
                                        className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                                    >
                                        <option value="off">Off</option>
                                        <option value="gentle">Gentle</option>
                                        <option value="balanced">Balanced</option>
                                        <option value="strong">Strong</option>
                                    </select>
                                </label>
                                <label className="text-xs text-slate-600">
                                    <div className="font-medium text-slate-700 mb-1">Apply Cleaned Audio To</div>
                                    <select
                                        value={voiceFixerApplyScopeDraft}
                                        onChange={(e) => useCleanupStore.setState({ voiceFixerApplyScopeDraft: e.target.value as 'none' | 'playback' | 'processing' | 'both' })}
                                        disabled={episodeBusy || (!hasVoiceFixerCleaned && voiceFixerApplyScopeDraft !== 'none')}
                                        className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm"
                                    >
                                        <option value="none">Original only</option>
                                        <option value="playback">Playback only</option>
                                        <option value="processing">Processing only</option>
                                        <option value="both">Playback + processing</option>
                                    </select>
                                </label>
                            </div>
                        </div>

                        <div className="rounded-[24px] border border-sky-200 bg-sky-50/70 p-5 shadow-sm">
                            <div className="text-sm font-semibold text-slate-900">Cleanup Workflow</div>
                            <div className="mt-1 text-xs leading-6 text-slate-600">
                                The player on this page always reflects the saved routing, not the unsaved draft. Save settings first, then rebuild if you changed the repair recipe.
                            </div>
                            <div className="mt-4 space-y-3">
                                <div className="rounded-2xl border border-white/70 bg-white/80 px-4 py-3 text-sm text-slate-700">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-sky-700">Saved routing</div>
                                    <div className="mt-1">Playback: {usingVoiceFixerForPlayback ? 'Cleaned media' : 'Original upload'}</div>
                                    <div className="mt-1">Processing: {usingVoiceFixerForProcessing ? 'Cleaned media' : 'Original upload'}</div>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => void handleQueueVoiceFixer(videoId, hasVoiceFixerCleaned, onVideoUpdated)}
                                    disabled={episodeBusy}
                                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-white px-4 py-2.5 text-sm font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                >
                                    {queueingVoiceFixer || voiceFixerBusy ? <Loader2 size={15} className="animate-spin" /> : <AudioLines size={15} />}
                                    {hasVoiceFixerCleaned ? 'Rebuild Cleaned Pass' : 'Create Cleaned Pass'}
                                </button>
                                <button
                                    type="button"
                                    onClick={onNavigateToTranscript}
                                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                                >
                                    <FileText size={15} />
                                    Return to Transcript
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
