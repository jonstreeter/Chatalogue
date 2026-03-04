import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import { toApiUrl } from '../lib/api';
import type { Job } from '../types';
import { Pause, Play, Trash2, Clock, CheckCircle2, DownloadCloud, FileText, Users, Video as VideoIcon, RefreshCw, ArrowUp, Smile, Bot, Scissors } from 'lucide-react';
import axios from 'axios';

const ACTIVE_STATUSES = ['downloading', 'transcribing', 'diarizing', 'running'];
type QueueName = 'pipeline' | 'funny' | 'youtube' | 'clip' | 'other';

const queueLabel: Record<QueueName, string> = {
    pipeline: 'Pipeline',
    funny: 'Funny Moments',
    youtube: 'Summary/Chapters',
    clip: 'Clip Export',
    other: 'Other',
};

const queueOrder: QueueName[] = ['pipeline', 'funny', 'youtube', 'clip', 'other'];

const queueIcon: Record<QueueName, React.ElementType> = {
    pipeline: FileText,
    funny: Smile,
    youtube: Bot,
    clip: Scissors,
    other: Clock,
};

const getQueueNameForJob = (job: Job): QueueName => {
    const jt = (job.job_type || '').toLowerCase();
    if (jt === 'process') return 'pipeline';
    if (jt === 'funny_detect' || jt === 'funny_explain') return 'funny';
    if (jt === 'youtube_metadata') return 'youtube';
    if (jt === 'clip_export_mp4' || jt === 'clip_export_captions') return 'clip';
    return 'other';
};

const getJobTypeLabel = (jobType: string): string => {
    const jt = (jobType || '').toLowerCase();
    if (jt === 'process') return 'Pipeline';
    if (jt === 'funny_detect') return 'Funny Scan';
    if (jt === 'funny_explain') return 'Funny Explain';
    if (jt === 'youtube_metadata') return 'Summary/Chapters';
    if (jt === 'clip_export_mp4') return 'Clip MP4';
    if (jt === 'clip_export_captions') return 'Clip Captions';
    return jobType || 'Job';
};

type QueueSummary = Record<QueueName, { queued: number; running: number; paused: number; completed: number; failed: number; total: number }>;

export function JobQueue() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [queueSummary, setQueueSummary] = useState<QueueSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [workerStatus, setWorkerStatus] = useState<'online' | 'offline' | 'stalled'>('offline');
    const [lowerTab, setLowerTab] = useState<'pending' | 'history'>('pending');

    const fetchJobs = async () => {
        try {
            const res = await api.get<Job[]>('/jobs', { params: { limit: 2000 } });
            setJobs(Array.isArray(res.data) ? res.data : []);
        } catch (e) {
            console.error('Failed to fetch jobs:', e);
            setJobs([]);
        } finally {
            setLoading(false);
        }
    };

    const fetchQueueSummary = async () => {
        try {
            const res = await api.get<QueueSummary>('/jobs/queues/summary');
            setQueueSummary(res.data || null);
        } catch (e) {
            console.error('Failed to fetch queue summary:', e);
            setQueueSummary(null);
        }
    };

    const checkWorkerStatus = async () => {
        try {
            const res = await api.get('/system/worker-status');
            setWorkerStatus(res.data.status);
        } catch {
            setWorkerStatus('offline');
        }
    };

    useEffect(() => {
        fetchJobs();
        fetchQueueSummary();
        checkWorkerStatus();
        const jobInterval = setInterval(fetchJobs, 2000);
        const summaryInterval = setInterval(fetchQueueSummary, 2500);
        const workerInterval = setInterval(checkWorkerStatus, 5000);
        return () => {
            clearInterval(jobInterval);
            clearInterval(summaryInterval);
            clearInterval(workerInterval);
        };
    }, []);

    const handlePause = async (jobId: number) => {
        try { await api.post(`/jobs/${jobId}/pause`); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleResume = async (jobId: number) => {
        try { await api.post(`/jobs/${jobId}/resume`); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleMoveToTop = async (jobId: number) => {
        try { await api.post(`/jobs/${jobId}/move-to-top`); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleCancel = async (jobId: number) => {
        if (!confirm('Cancel this job?')) return;
        try { await api.delete(`/jobs/${jobId}`); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handlePauseAll = async () => {
        if (!confirm('Pause all queued jobs? Running jobs will finish.')) return;
        try { await api.post('/jobs/pause-all'); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleResumeAll = async () => {
        try { await api.post('/jobs/resume-all'); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleClearQueue = async () => {
        if (!confirm('Are you sure you want to delete all queued/paused jobs? This cannot be undone.')) return;
        try {
            const res = await api.delete<{ deleted?: number }>('/jobs/queue');
            const deleted = Number(res.data?.deleted || 0);
            await Promise.all([fetchJobs(), fetchQueueSummary()]);
            alert(`Cleared ${deleted} queued/paused job${deleted === 1 ? '' : 's'}. Running jobs are not canceled.`);
        } catch (e) {
            console.error(e);
            const detail = axios.isAxiosError(e)
                ? String((e.response?.data as any)?.detail || e.message || 'Unknown error')
                : String(e || 'Unknown error');
            alert(`Failed to clear queue: ${detail}`);
        }
    };

    const handleClearHistory = async () => {
        if (!confirm('Clear completed and failed jobs history?')) return;
        try { await api.delete('/jobs/history'); fetchJobs(); } catch (e) { console.error(e); }
    };

    const handleResubmit = async (jobId: number) => {
        try { await api.post(`/jobs/${jobId}/resubmit`); fetchJobs(); } catch (e) { console.error(e); }
    };

    const getThumbnailUrl = (job: Job) => {
        if (!job.video?.thumbnail_url) return null;
        if (job.video.thumbnail_url.startsWith('http')) return job.video.thumbnail_url;
        return toApiUrl(job.video.thumbnail_url);
    };

    const byAscCreated = (a: Job, b: Job) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
    const byDescCreated = (a: Job, b: Job) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    const byDescStarted = (a: Job, b: Job) => {
        const aTime = new Date(a.started_at || a.created_at).getTime();
        const bTime = new Date(b.started_at || b.created_at).getTime();
        return bTime - aTime;
    };
    const getJobDurationSeconds = (job: Job): number | null => {
        const started = job.started_at ? new Date(job.started_at).getTime() : NaN;
        const completed = job.completed_at ? new Date(job.completed_at).getTime() : NaN;
        if (!Number.isFinite(started) || !Number.isFinite(completed)) return null;
        const seconds = (completed - started) / 1000;
        return seconds >= 0 ? seconds : null;
    };
    const getJobElapsedSeconds = (job: Job): number | null => {
        const created = job.created_at ? new Date(job.created_at).getTime() : NaN;
        const completed = job.completed_at ? new Date(job.completed_at).getTime() : NaN;
        if (!Number.isFinite(created) || !Number.isFinite(completed)) return null;
        const seconds = (completed - created) / 1000;
        return seconds >= 0 ? seconds : null;
    };
    const formatDuration = (seconds: number | null): string => {
        if (seconds == null) return '-';
        if (seconds < 1) return '<1s';
        if (seconds < 10) return `${seconds.toFixed(1)}s`;
        if (seconds < 60) return `${Math.round(seconds)}s`;
        const rounded = Math.round(seconds);
        const h = Math.floor(rounded / 3600);
        const m = Math.floor((rounded % 3600) / 60);
        const s = rounded % 60;
        if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        return `${m}:${String(s).padStart(2, '0')}`;
    };

    const getStatusPillClasses = (status: string): string => {
        const s = (status || '').toLowerCase();
        if (s === 'completed') return 'bg-green-50 text-green-700 border-green-200';
        if (s === 'failed') return 'bg-red-50 text-red-700 border-red-200';
        if (s === 'paused') return 'bg-amber-50 text-amber-700 border-amber-200';
        if (ACTIVE_STATUSES.includes(s)) return 'bg-blue-50 text-blue-700 border-blue-200';
        if (s === 'queued') return 'bg-slate-50 text-slate-600 border-slate-200';
        return 'bg-slate-50 text-slate-600 border-slate-200';
    };

const getStatusLabel = (status: string): string => {
        const s = (status || '').toLowerCase();
        if (!s) return 'unknown';
        if (s === 'downloading') return 'downloading';
        if (s === 'transcribing') return 'transcribing';
        if (s === 'diarizing') return 'diarizing';
        return s;
    };

    const getJobPayload = (job: Job): Record<string, unknown> => {
        if (!job.payload_json) return {};
        try {
            const parsed = JSON.parse(job.payload_json);
            return parsed && typeof parsed === 'object' ? parsed as Record<string, unknown> : {};
        } catch {
            return {};
        }
    };

    const getTranscriptionEngineUsed = (job: Job): 'parakeet' | 'whisper' | null => {
        if ((job.job_type || '').toLowerCase() !== 'process') return null;
        const payload = getJobPayload(job);
        const engine = String(payload.transcription_engine_used || '').toLowerCase();
        if (engine === 'parakeet' || engine === 'whisper') return engine;
        const detail = (job.status_detail || '').toLowerCase();
        if (detail.includes('parakeet')) return 'parakeet';
        if (detail.includes('whisper')) return 'whisper';
        return null;
    };

    const EngineBadge = ({ engine }: { engine: 'parakeet' | 'whisper' }) => (
        <span className={`text-[11px] px-2 py-0.5 rounded border font-medium ${
            engine === 'parakeet'
                ? 'bg-violet-50 text-violet-700 border-violet-200'
                : 'bg-cyan-50 text-cyan-700 border-cyan-200'
        }`}>
            {engine === 'parakeet' ? 'Parakeet' : 'Whisper'}
        </span>
    );

    const activeJobs = jobs.filter(j => ACTIVE_STATUSES.includes(j.status)).sort(byDescStarted);
    const activeByQueue: Record<QueueName, Job[]> = {
        pipeline: [],
        funny: [],
        youtube: [],
        clip: [],
        other: [],
    };
    activeJobs.forEach(j => activeByQueue[getQueueNameForJob(j)].push(j));
    const currentJob = activeByQueue.pipeline[0] ?? null;
    const activeByQueueExcludingCurrent: Record<QueueName, Job[]> = {
        pipeline: currentJob ? activeByQueue.pipeline.filter(j => j.id !== currentJob.id) : [...activeByQueue.pipeline],
        funny: [...activeByQueue.funny],
        youtube: [...activeByQueue.youtube],
        clip: [...activeByQueue.clip],
        other: [...activeByQueue.other],
    };
    const additionalActiveCount = queueOrder.reduce((sum, q) => sum + activeByQueueExcludingCurrent[q].length, 0);
    const queuedJobs = jobs.filter(j => j.status === 'queued').sort(byAscCreated);
    const pausedJobs = jobs.filter(j => j.status === 'paused').sort(byAscCreated);
    const historyJobs = jobs.filter(j => ['completed', 'failed'].includes(j.status)).sort(byDescCreated);
    const pendingByQueue: Record<QueueName, { queued: Job[]; paused: Job[]; running: Job[] }> = {
        pipeline: { queued: [], paused: [], running: [] },
        funny: { queued: [], paused: [], running: [] },
        youtube: { queued: [], paused: [], running: [] },
        clip: { queued: [], paused: [], running: [] },
        other: { queued: [], paused: [], running: [] },
    };
    queuedJobs.forEach(j => pendingByQueue[getQueueNameForJob(j)].queued.push(j));
    pausedJobs.forEach(j => pendingByQueue[getQueueNameForJob(j)].paused.push(j));
    activeJobs.forEach(j => {
        const q = getQueueNameForJob(j);
        if (q === 'pipeline' && currentJob && j.id === currentJob.id) return; // already shown in Current Job card
        pendingByQueue[q].running.push(j);
    });
    const pendingCount = queueOrder.reduce((sum, q) => (
        sum
        + pendingByQueue[q].queued.length
        + pendingByQueue[q].paused.length
        + pendingByQueue[q].running.length
    ), 0);

    const ActiveJobCard = ({ job }: { job: Job }) => {
        const thumb = getThumbnailUrl(job);
        const transcriptionEngineUsed = getTranscriptionEngineUsed(job);
        const detail = (job.status_detail || '').toLowerCase();
        const funnyScanActive = detail.includes('funny moments') || detail.includes('laughter');
        const videoPipelineComplete = job.video?.processed || job.video?.status === 'completed';
        const showFinalizingBadge = job.status === 'diarizing' && (funnyScanActive || videoPipelineComplete);
        const statusBadgeLabel = showFinalizingBadge ? 'finalizing' : job.status;

        const getStageState = (stage: 'download' | 'transcribe' | 'diarize' | 'funny') => {
            const status = job.status;
            const inFunnyTail = status === 'diarizing' && videoPipelineComplete;
            if (status === 'completed') return 'completed';
            if (status === 'failed') return 'failed';

            if (stage === 'funny') {
                if (funnyScanActive || inFunnyTail) return 'active';
                // If we are still in diarization and not scanning funny moments yet, keep pending.
                if (status === 'diarizing') return 'pending';
                // If we're past the main phases (or queue worker flips completed), treat as done.
                if (['running', 'downloading', 'transcribing'].includes(status)) return 'pending';
                return 'pending';
            }

            const stages = ['downloading', 'transcribing', 'diarizing'] as const;
            if (status === 'running') {
                return stage === 'download' ? 'active' : 'pending';
            }
            const currentStageIndex = stages.indexOf(status as any);

            let targetStageIndex = -1;
            if (stage === 'download') targetStageIndex = 0;
            if (stage === 'transcribe') targetStageIndex = 1;
            if (stage === 'diarize') targetStageIndex = 2;

            if (stage === 'diarize' && inFunnyTail) return 'completed';
            if (currentStageIndex > targetStageIndex) return 'completed';
            if (currentStageIndex === targetStageIndex) return 'active';
            return 'pending';
        };

        const renderProgressBar = (label: string, stage: 'download' | 'transcribe' | 'diarize' | 'funny', Icon: React.ElementType) => {
            const state = getStageState(stage);
            const percent = state === 'completed' ? 100 : state === 'active' ? job.progress : 0;
            const isIndeterminate = state === 'active' && ((stage === 'diarize' || stage === 'funny') || percent <= 0);

            return (
                <div className="space-y-1.5">
                    <div className="flex justify-between items-center text-xs text-slate-500 uppercase tracking-wide font-semibold">
                        <div className="flex items-center gap-1.5">
                            <Icon size={12} className={state === 'active' ? 'text-blue-500' : ''} />
                            <span>{label}</span>
                        </div>
                        {state === 'completed' && <CheckCircle2 size={14} className="text-green-500" />}
                        {state === 'active' && <span className="text-blue-600 animate-pulse">Processing...</span>}
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden relative">
                        {state === 'active' && isIndeterminate ? (
                            <div className="absolute inset-0 bg-blue-500/20">
                                <div className="h-full w-1/3 bg-blue-500 animate-[shimmer_1.5s_infinite] relative overflow-hidden">
                                    <div className="absolute inset-0 bg-white/30 skew-x-12" />
                                </div>
                            </div>
                        ) : (
                            <div
                                className={`h-full transition-all duration-500 ${state === 'completed' ? 'bg-green-500' : 'bg-blue-500'}`}
                                style={{ width: `${Math.max(percent, 0)}%` }}
                            />
                        )}
                    </div>
                </div>
            );
        };

        return (
            <div className="bg-white p-3.5 sm:p-4 rounded-xl border border-blue-100 shadow-sm flex flex-col lg:flex-row gap-4 items-start">
                <div className="w-full lg:w-44 h-28 lg:h-24 bg-slate-100 rounded-lg overflow-hidden shrink-0 border border-slate-200 relative">
                    {thumb ? (
                        <img src={thumb} className="w-full h-full object-cover" alt="" />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center text-slate-300">
                            <VideoIcon size={32} />
                        </div>
                    )}
                    <div className="absolute inset-0 bg-black/10 flex items-center justify-center">
                        <div className="bg-white/90 backdrop-blur px-3 py-1 rounded-full text-xs font-bold shadow-sm uppercase tracking-wide">
                            {statusBadgeLabel}
                        </div>
                    </div>
                </div>

                <div className="flex-1 min-w-0 space-y-3 w-full">
                    <div className="flex flex-col md:flex-row md:justify-between md:items-start gap-3">
                        <div className="min-w-0">
                            <h4 className="text-base sm:text-lg font-bold text-slate-800 leading-tight mb-1">
                                {job.video?.title || `Video #${job.video_id}`}
                            </h4>
                            <div className="flex flex-wrap items-center gap-2 text-xs sm:text-sm text-slate-500">
                                <span className="flex items-center gap-1"><VideoIcon size={14} /> ID: {job.video_id}</span>
                                <span className="font-mono text-xs bg-slate-50 px-2 py-0.5 rounded border border-slate-200">{job.job_type}</span>
                                {transcriptionEngineUsed && <EngineBadge engine={transcriptionEngineUsed} />}
                                {job.started_at && (
                                    <span className="text-xs">Started {new Date(job.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                )}
                            </div>
                        </div>
                        <div className="flex gap-2 shrink-0">
                            <button onClick={() => handlePause(job.id)} className="px-3 py-1.5 bg-amber-50 text-amber-700 rounded hover:bg-amber-100 flex items-center gap-1 transition-colors text-xs font-medium">
                                <Pause size={14} /> Pause
                            </button>
                            <button onClick={() => handleCancel(job.id)} className="px-3 py-1.5 bg-red-50 text-red-700 rounded hover:bg-red-100 flex items-center gap-1 transition-colors text-xs font-medium">
                                <Trash2 size={14} /> Cancel
                            </button>
                        </div>
                    </div>

                    <div className="space-y-2.5">
                        {renderProgressBar('Downloading', 'download', DownloadCloud)}
                        {renderProgressBar('Transcribing', 'transcribe', FileText)}
                        {renderProgressBar('Diarizing', 'diarize', Users)}
                        {renderProgressBar('Funny Moments', 'funny', Smile)}
                    </div>

                    {job.status_detail && (
                        <div className="flex items-center gap-2 text-xs sm:text-sm text-blue-600 bg-blue-50 px-3 py-2 rounded-lg border border-blue-100">
                            <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                            <span className="font-medium">{job.status_detail}</span>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    const PendingQueueItem = ({ job, index, mode }: { job: Job; index?: number; mode: 'queued' | 'paused' | 'running' }) => {
        const thumb = getThumbnailUrl(job);
        const transcriptionEngineUsed = getTranscriptionEngineUsed(job);
        const paused = mode === 'paused';
        const running = mode === 'running';
        const canMoveToTop = mode === 'queued' && getQueueNameForJob(job) === 'pipeline';
        const isPrefetched = job.video?.status === 'downloaded';
        const jobType = (job.job_type || '').toLowerCase();
        const rawProgress = Number.isFinite(job.progress) ? Math.max(0, Math.min(100, Number(job.progress))) : 0;
        const stepDetail = (job.status_detail || '').trim();

        const jobTypeLabel = (() => {
            if (jobType === 'process') return 'Pipeline';
            if (jobType === 'clip_export_mp4') return 'Clip MP4';
            if (jobType === 'clip_export_captions') return 'Clip Captions';
            if (jobType === 'funny_detect') return 'Funny Scan';
            if (jobType === 'funny_explain') return 'Funny Explain';
            if (jobType === 'youtube_metadata') return 'Summary/Chapters';
            return job.job_type || 'Job';
        })();

        const stepText = (() => {
            if (stepDetail) return stepDetail;
            if (paused) return 'Paused by user';
            if (!running) return 'Waiting in queue';
            if (jobType === 'process') {
                if (job.status === 'downloading') return 'Downloading source audio...';
                if (job.status === 'transcribing') return 'Transcribing audio...';
                if (job.status === 'diarizing') return 'Running diarization...';
                return 'Processing...';
            }
            if (jobType === 'clip_export_mp4') return 'Rendering MP4 clip...';
            if (jobType === 'clip_export_captions') return 'Generating caption sidecar...';
            if (jobType === 'funny_detect') return 'Scanning for funny moments...';
            if (jobType === 'funny_explain') return 'Generating humor explanations...';
            if (jobType === 'youtube_metadata') return 'Generating summary + chapters...';
            return 'Running...';
        })();

        const indeterminate =
            running &&
            rawProgress <= 5 &&
            (
                jobType === 'clip_export_mp4' ||
                jobType === 'clip_export_captions' ||
                /rendering|analyzing|processing|generating/i.test(stepText)
            );
        const barPercent = paused ? rawProgress : running ? (indeterminate ? Math.max(10, rawProgress) : rawProgress) : 0;

        return (
            <div className={`group p-2.5 rounded-lg border transition-all ${paused ? 'bg-amber-50/40 border-amber-100' : running ? 'bg-blue-50/40 border-blue-100' : 'bg-white border-slate-100 hover:border-blue-200'}`}>
                <div className="flex items-center gap-2.5">
                <div className="w-7 text-center text-[10px] font-mono text-slate-400 shrink-0">
                    {paused ? '||' : running ? 'RUN' : String((index ?? 0) + 1).padStart(2, '0')}
                </div>
                <div className="w-14 h-9 bg-slate-100 rounded overflow-hidden shrink-0 border border-slate-200">
                    {thumb ? <img src={thumb} className="w-full h-full object-cover" alt="" /> : <div className="w-full h-full" />}
                </div>
                <div className="flex-1 min-w-0">
                    <div className="truncate text-sm font-medium text-slate-700" title={job.video?.title}>
                        {job.video?.title || `Video #${job.video_id}`}
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5 mt-1 text-[11px]">
                        <span className="text-slate-500 bg-slate-50 px-2 py-0.5 rounded border border-slate-200">{jobTypeLabel}</span>
                        {transcriptionEngineUsed && <EngineBadge engine={transcriptionEngineUsed} />}
                        {isPrefetched && (
                            <span className="text-green-700 bg-green-50 px-2 py-0.5 rounded border border-green-200">
                                Prefetched
                            </span>
                        )}
                        {paused ? (
                            <span className="text-amber-700 bg-amber-100 px-2 py-0.5 rounded border border-amber-200">Paused</span>
                        ) : running ? (
                            <span className="text-blue-700 bg-blue-100 px-2 py-0.5 rounded border border-blue-200">Running</span>
                        ) : (
                            <span className="text-slate-400">Queued</span>
                        )}
                    </div>
                </div>
                <div className="flex gap-1 shrink-0">
                    {canMoveToTop && (
                        <button
                            onClick={() => handleMoveToTop(job.id)}
                            className="px-2 py-1 text-[11px] rounded border border-blue-200 text-blue-700 bg-blue-50 hover:bg-blue-100 flex items-center gap-1"
                            title="Move to top of queue"
                        >
                            <ArrowUp size={12} /> Top
                        </button>
                    )}
                    {paused ? (
                        <button onClick={() => handleResume(job.id)} className="p-1.5 bg-green-50 text-green-600 hover:bg-green-100 rounded" title="Resume">
                            <Play size={14} />
                        </button>
                    ) : (
                        <button onClick={() => handlePause(job.id)} className="p-1.5 hover:bg-amber-50 text-slate-400 hover:text-amber-600 rounded" title="Pause">
                            <Pause size={14} />
                        </button>
                    )}
                    <button onClick={() => handleCancel(job.id)} className="p-1.5 hover:bg-red-50 text-slate-400 hover:text-red-500 rounded" title="Cancel">
                        <Trash2 size={14} />
                    </button>
                </div>
                </div>
                <div className="mt-2 pl-[calc(1.75rem+3.5rem+0.625rem)]">
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                        <span className={`${running ? 'text-blue-700' : paused ? 'text-amber-700' : 'text-slate-500'} truncate`}>{stepText}</span>
                        <span className="text-slate-400 tabular-nums shrink-0">{running || paused ? `${Math.round(barPercent)}%` : 'queued'}</span>
                    </div>
                    <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden mt-1.5 relative">
                        {indeterminate ? (
                            <div className="absolute inset-0 bg-blue-500/15">
                                <div className="h-full w-1/3 bg-blue-500 animate-[shimmer_1.5s_infinite] relative overflow-hidden">
                                    <div className="absolute inset-0 bg-white/30 skew-x-12" />
                                </div>
                            </div>
                        ) : (
                            <div
                                className={`h-full transition-all duration-500 ${paused ? 'bg-amber-400' : running ? 'bg-blue-500' : 'bg-slate-300'}`}
                                style={{ width: `${Math.max(0, Math.min(100, barPercent))}%` }}
                            />
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-8 max-w-6xl mx-auto pb-20">
            <div className="flex flex-col lg:flex-row lg:justify-between lg:items-end gap-4 pb-4 border-b border-slate-200">
                <div>
                    <h1 className="text-2xl sm:text-3xl font-bold text-slate-800 flex flex-wrap items-center gap-3">
                        Job Queue
                        <span className={`text-sm px-3 py-1 rounded-full border font-mono ${workerStatus === 'online'
                            ? 'bg-green-50 text-green-600 border-green-200'
                            : workerStatus === 'stalled'
                                ? 'bg-amber-50 text-amber-600 border-amber-200'
                                : 'bg-red-50 text-red-600 border-red-200'
                            }`}>
                            WORKER: {workerStatus.toUpperCase()}
                        </span>
                    </h1>
                    <p className="text-slate-500 mt-1">Current job on top, pending queue in the middle, history log at the bottom.</p>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:flex sm:flex-wrap">
                    <button onClick={handlePauseAll} className="px-4 py-2.5 bg-amber-50 text-amber-700 rounded-lg hover:bg-amber-100 border border-amber-200 transition-colors inline-flex items-center justify-center gap-2 text-sm font-medium">
                        <Pause size={16} /> Pause All
                    </button>
                    <button onClick={handleResumeAll} className="px-4 py-2.5 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 border border-green-200 transition-colors inline-flex items-center justify-center gap-2 text-sm font-medium">
                        <Play size={16} /> Resume All
                    </button>
                    <button onClick={handleClearQueue} className="px-4 py-2.5 bg-white text-slate-600 rounded-lg hover:bg-red-50 hover:text-red-600 border border-slate-200 hover:border-red-200 transition-colors inline-flex items-center justify-center gap-2 text-sm font-medium">
                        <Trash2 size={16} /> Clear Queue
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="py-20 flex flex-col items-center justify-center text-slate-400 gap-4">
                    <div className="w-8 h-8 border-2 border-slate-200 border-t-blue-500 rounded-full animate-spin" />
                    <p>Connecting to backend...</p>
                </div>
            ) : (
                <>
                    <section className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                        {queueOrder.map((q) => {
                            const Icon = queueIcon[q];
                            const s = queueSummary?.[q];
                            const queued = s?.queued ?? pendingByQueue[q].queued.length;
                            const running = s?.running ?? activeByQueue[q].length;
                            return (
                                <div key={q} className="rounded-xl border border-slate-200 bg-white px-2.5 py-2">
                                    <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                                        <Icon size={13} />
                                        <span className="font-semibold uppercase tracking-wide">{queueLabel[q]}</span>
                                    </div>
                                    <div className="mt-1 text-xs sm:text-sm text-slate-700">
                                        <span className="font-semibold">{running}</span> running
                                        <span className="mx-1.5 text-slate-300">|</span>
                                        <span className="font-semibold">{queued}</span> queued
                                    </div>
                                </div>
                            );
                        })}
                    </section>

                    <section className="space-y-3">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                            <h2 className="text-xl font-bold text-slate-700 flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${activeJobs.length > 0 ? 'bg-blue-500 animate-pulse' : 'bg-slate-300'}`} />
                                Current Job
                            </h2>
                            {additionalActiveCount > 0 && (
                                <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-full px-2 py-1">
                                    {additionalActiveCount} additional active job{additionalActiveCount === 1 ? '' : 's'}
                                </span>
                            )}
                        </div>

                        {currentJob ? (
                            <>
                                <ActiveJobCard job={currentJob} />
                            </>
                        ) : (
                            additionalActiveCount === 0 ? (
                                <div className="bg-slate-50/60 rounded-xl border border-dashed border-slate-200 p-8 text-center text-slate-400">
                                    No job currently running.
                                </div>
                            ) : (
                                <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-2.5 text-xs sm:text-sm text-blue-800">
                                    Pipeline queue is idle, but other queues are actively running below.
                                </div>
                            )
                        )}
                        {additionalActiveCount > 0 && (
                            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-2.5 sm:p-3">
                                <div className="space-y-3">
                                    {queueOrder.map((q) => {
                                        const qRunning = activeByQueueExcludingCurrent[q];
                                        if (qRunning.length === 0) return null;
                                        const Icon = queueIcon[q];
                                        return (
                                            <div key={q} className="space-y-1.5">
                                                <div className="flex items-center justify-between px-1">
                                                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-1.5">
                                                        <Icon size={12} />
                                                        {queueLabel[q]}
                                                    </div>
                                                    <span className="text-[11px] text-slate-400">{qRunning.length} running</span>
                                                </div>
                                                {qRunning.map(job => (
                                                    <PendingQueueItem key={job.id} job={job} mode="running" />
                                                ))}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </section>

                    <section className="space-y-4 pt-2">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <div className="inline-flex w-full md:w-auto rounded-xl bg-slate-100 p-1 border border-slate-200 overflow-x-auto">
                                <button
                                    onClick={() => setLowerTab('pending')}
                                    className={`px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors ${
                                        lowerTab === 'pending'
                                            ? 'bg-white text-slate-800 shadow-sm'
                                            : 'text-slate-500 hover:text-slate-700'
                                    }`}
                                >
                                    <Clock size={16} className="text-slate-400" />
                                    Pending Queue
                                    <span className={`text-[11px] px-1.5 py-0.5 rounded-full border ${
                                        lowerTab === 'pending'
                                            ? 'bg-blue-50 text-blue-700 border-blue-200'
                                            : 'bg-white text-slate-500 border-slate-200'
                                    }`}>
                                        {pendingCount}
                                    </span>
                                </button>
                                <button
                                    onClick={() => setLowerTab('history')}
                                    className={`px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors ${
                                        lowerTab === 'history'
                                            ? 'bg-white text-slate-800 shadow-sm'
                                            : 'text-slate-500 hover:text-slate-700'
                                    }`}
                                >
                                    <CheckCircle2 size={16} className="text-slate-400" />
                                    History Log
                                    <span className={`text-[11px] px-1.5 py-0.5 rounded-full border ${
                                        lowerTab === 'history'
                                            ? 'bg-slate-50 text-slate-700 border-slate-200'
                                            : 'bg-white text-slate-500 border-slate-200'
                                    }`}>
                                        {historyJobs.length}
                                    </span>
                                </button>
                            </div>

                            {lowerTab === 'pending' ? (
                                <div className="text-xs text-slate-500 flex flex-wrap items-center gap-2">
                                    <span className="px-2 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700">Running: {activeJobs.length}</span>
                                    <span className="px-2 py-1 rounded-full bg-slate-100 border border-slate-200">Queued: {queuedJobs.length}</span>
                                    <span className="px-2 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700">Paused: {pausedJobs.length}</span>
                                    <span className="px-2 py-1 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-700">Total: {pendingCount}</span>
                                </div>
                            ) : historyJobs.length > 0 ? (
                                <button onClick={handleClearHistory} className="text-xs text-slate-400 hover:text-red-500 underline decoration-slate-300 hover:decoration-red-300 underline-offset-2">
                                    Clear History
                                </button>
                            ) : (
                                <div />
                            )}
                        </div>

                        {lowerTab === 'pending' ? (
                            pendingCount === 0 ? (
                                <div className="py-10 text-center bg-slate-50/50 rounded-xl border border-dashed border-slate-200">
                                    <p className="text-slate-400">No pending jobs</p>
                                </div>
                            ) : (
                                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-2.5 sm:p-3">
                                    <div className="max-h-[46vh] overflow-y-auto pr-1 space-y-3">
                                        {queueOrder.map((q) => {
                                            const qRunning = pendingByQueue[q].running;
                                            const qQueued = pendingByQueue[q].queued;
                                            const qPaused = pendingByQueue[q].paused;
                                            const total = qRunning.length + qQueued.length + qPaused.length;
                                            if (total === 0) return null;
                                            const Icon = queueIcon[q];
                                            return (
                                                <div key={q} className="space-y-1.5">
                                                    <div className="flex items-center justify-between px-1">
                                                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-1.5">
                                                            <Icon size={12} />
                                                            {queueLabel[q]}
                                                        </div>
                                                        <span className="text-[11px] text-slate-400">{total} job{total === 1 ? '' : 's'}</span>
                                                    </div>
                                                    {qRunning.map(job => (
                                                        <PendingQueueItem key={job.id} job={job} mode="running" />
                                                    ))}
                                                    {qQueued.map((job, index) => (
                                                        <PendingQueueItem key={job.id} job={job} index={index} mode="queued" />
                                                    ))}
                                                    {qPaused.map(job => (
                                                        <PendingQueueItem key={job.id} job={job} mode="paused" />
                                                    ))}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )
                        ) : (
                            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                                {historyJobs.length === 0 ? (
                                    <div className="p-8 text-center text-slate-400 text-sm">No history available</div>
                                ) : (
                                    <div className="max-h-[44vh] overflow-auto">
                                        <table className="w-full text-sm">
                                            <thead className="bg-slate-50 border-b border-slate-100 sticky top-0 z-10">
                                                <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                                                    <th className="p-3 font-medium w-16">Img</th>
                                                    <th className="p-3 font-medium">Video / Job</th>
                                                    <th className="p-3 font-medium min-w-64">Status</th>
                                                    <th className="p-3 font-medium w-32 text-right">Duration</th>
                                                    <th className="p-3 font-medium w-20 text-center">Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-100">
                                                {historyJobs.map(job => {
                                                    const transcriptionEngineUsed = getTranscriptionEngineUsed(job);
                                                    return (
                                                    <tr key={job.id} className="hover:bg-slate-50 transition-colors">
                                                        <td className="p-3">
                                                            <div className="w-12 h-8 bg-slate-100 rounded overflow-hidden border border-slate-200">
                                                                {getThumbnailUrl(job) && <img src={getThumbnailUrl(job)!} className="w-full h-full object-cover" alt="" />}
                                                            </div>
                                                        </td>
                                                        <td className="p-3">
                                                            <Link to={`/video/${job.video_id}`} className="font-medium text-slate-700 hover:text-blue-600 transition-colors" title={job.video?.title}>
                                                                {job.video?.title || `Video #${job.video_id}`}
                                                            </Link>
                                                            <div className="text-xs text-slate-400 font-mono mt-0.5">{job.job_type}</div>
                                                        </td>
                                                        <td className="p-3">
                                                            <div className="flex flex-wrap items-center gap-1.5">
                                                                <span className="text-[11px] px-2 py-0.5 rounded border bg-slate-50 text-slate-600 border-slate-200">
                                                                    {getJobTypeLabel(job.job_type)}
                                                                </span>
                                                                {transcriptionEngineUsed && <EngineBadge engine={transcriptionEngineUsed} />}
                                                                <span className={`text-[11px] px-2 py-0.5 rounded border font-medium ${getStatusPillClasses(job.status)}`}>
                                                                    {getStatusLabel(job.status)}
                                                                </span>
                                                            </div>
                                                            {job.status === 'failed' && job.error && (
                                                                <div className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded border border-red-100 break-words max-w-xs">
                                                                    {job.error}
                                                                </div>
                                                            )}
                                                        </td>
                                                        <td className="p-3 text-right tabular-nums align-top">
                                                            <div className="text-slate-700 font-medium">
                                                                {formatDuration(getJobDurationSeconds(job))}
                                                            </div>
                                                            {(() => {
                                                                const runtime = getJobDurationSeconds(job);
                                                                const elapsed = getJobElapsedSeconds(job);
                                                                if (runtime == null || elapsed == null) return null;
                                                                if (elapsed - runtime < 2) return null;
                                                                return (
                                                                    <div className="text-[11px] text-slate-400 mt-0.5">
                                                                        total {formatDuration(elapsed)}
                                                                    </div>
                                                                );
                                                            })()}
                                                            {job.started_at && (
                                                                <div className="text-[11px] text-slate-400 mt-0.5">
                                                                    started {new Date(job.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                                </div>
                                                            )}
                                                        </td>
                                                        <td className="p-3 text-center align-top">
                                                            <button
                                                                onClick={() => handleResubmit(job.id)}
                                                                className="p-1.5 hover:bg-blue-50 text-slate-400 hover:text-blue-600 rounded transition-colors"
                                                                title="Resubmit job"
                                                            >
                                                                <RefreshCw size={15} />
                                                            </button>
                                                        </td>
                                                    </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}
                    </section>
                </>
            )}
        </div>
    );
}
