import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type { Speaker, Job, WorkbenchTaskProgress } from '../../types';
import { Loader2, ArrowLeft, FileText, Scissors, Users, CheckCircle2, Search, GitMerge, RotateCcw, Eraser, AudioLines, Bot, Download, Sparkles, Clapperboard, MessageSquareText, type LucideIcon } from 'lucide-react';
import { SpeakerModal } from '../../components/SpeakerModal';
import { EpisodeChatWorkbench } from '../../components/video/EpisodeChatWorkbench';
import { VideoPlayer, type UploadedPlaybackSource } from '../../components/video/VideoPlayer';
import { CloneTab } from './tabs/CloneTab';
import { YoutubeTab } from './tabs/YoutubeTab';
import { SpeakersTab } from './tabs/SpeakersTab';
import { ClipsTab } from './tabs/ClipsTab';
import { ClipCreationPanel } from './tabs/ClipCreationPanel';
import { CleanupTab } from './tabs/CleanupTab';
import { ClipEditorWorkspace } from './tabs/ClipEditorWorkspace';
import { OptimizeTab, OptimizeSidebar } from './tabs/OptimizeTab';
import { TranscriptTab } from './tabs/TranscriptTab';
import { ReconstructionSidebarTab, ReconstructionTab } from './tabs/ReconstructionTab';
import { useCloneStore, useCloneUsesOllama } from '../../store/useCloneStore';
import { useYoutubeStore } from '../../store/useYoutubeStore';
import { useSpeakersTabStore } from '../../store/useSpeakersTabStore';
import { useClipsStore } from '../../store/useClipsStore';
import { useWorkbenchStore } from '../../store/useWorkbenchStore';
import { useCleanupStore } from '../../store/useCleanupStore';
import { useReconstructionStore } from '../../store/useReconstructionStore';
import { useTranscriptStore } from '../../store/useTranscriptStore';
import { usePlayerStore } from '../../store/usePlayerStore';
import { selectVideoDetailDerivedMetadata, useVideoStore } from '../../store/useVideoStore';
import { useVideoMetadataPolling } from '../../hooks/useVideoMetadataPolling';
import { useClipPreviewLoop } from './useClipPreviewLoop';
import { useTranscriptSegmentEditLoop } from './useTranscriptSegmentEditLoop';
import { useVideoSeek, type VideoSidebarTab } from './useVideoSeek';

type SidebarTabConfig = {
    id: VideoSidebarTab;
    label: string;
    title: string;
    icon: LucideIcon;
    activeClassName: string;
    inactiveClassName: string;
};

type WorkbenchActivity = {
    label: string;
    detail: string;
    tone: 'sky' | 'violet';
};

export function VideoDetailPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();

    // Data State
    const video = useVideoStore((s) => s.video);
    const setVideo = useVideoStore((s) => s.setVideo);
    const segments = useVideoStore((s) => s.segments);
    const setSegments = useVideoStore((s) => s.setSegments);
    const loading = useVideoStore((s) => s.loading);
    const refreshVideoData = useVideoStore((s) => s.refreshVideoData);

    // UI State
    const [activeTab, setActiveTab] = useState<VideoSidebarTab>('transcript');

    // Player State
    const currentTime = usePlayerStore((s) => s.currentTime);
    const playbackRate = usePlayerStore((s) => s.playbackRate);

    const clips = useClipsStore((s) => s.clips);
    const clipSelection = useClipsStore((s) => s.clipSelection);
    const setClipSelection = useClipsStore((s) => s.setClipSelection);
    // Transcript store — setters still used by VDP-level effects / URL restoration.
    const detectingFunnyMoments = useTranscriptStore((s) => s.detectingFunnyMoments);
    const explainingFunnyMoments = useTranscriptStore((s) => s.explainingFunnyMoments);
    const setSearchQuery = useTranscriptStore((s) => s.setSearchQuery);
    const setDeepLinkedSegmentId = useTranscriptStore((s) => s.setDeepLinkedSegmentId);
    const setSearchMatchIndex = useTranscriptStore((s) => s.setSearchMatchIndex);
    const setFollowPlayback = useTranscriptStore((s) => s.setFollowPlayback);
    const setExpandedFunnySummaryIds = useTranscriptStore((s) => s.setExpandedFunnySummaryIds);

    // Clone store — only the values VideoDetailPage itself needs (chat sidebar + EpisodeChatWorkbench).
    const cloneEngines = useCloneStore((s) => s.cloneEngines);
    const cloneOllamaModels = useCloneStore((s) => s.cloneOllamaModels);
    const cloneUsesOllama = useCloneUsesOllama();
    const cloneEngineKey = useCloneStore((s) => s.cloneEngineKey);

    const editingClipId = useClipsStore((s) => s.editingClipId);
    const clipEditorDraft = useClipsStore((s) => s.clipEditorDraft);

    const selectedSpeaker = useSpeakersTabStore((s) => s.selectedSpeaker);
    const initialSample = useSpeakersTabStore((s) => s.initialSample);
    const assignPopup = useSpeakersTabStore((s) => s.assignPopup);
    const assignSpeakers = useSpeakersTabStore((s) => s.assignSpeakers);
    const assignSearch = useSpeakersTabStore((s) => s.assignSearch);
    const assignLoading = useSpeakersTabStore((s) => s.assignLoading);
    const setAssignSearch = useSpeakersTabStore((s) => s.setAssignSearch);
    const closeSpeakerModal = useSpeakersTabStore((s) => s.closeSpeakerModal);
    const closeAssignPopup = useSpeakersTabStore((s) => s.closeAssignPopup);
    const fetchAssignSpeakers = useSpeakersTabStore((s) => s.fetchAssignSpeakers);
    const assignSpeaker = useSpeakersTabStore((s) => s.assignSpeaker);
    const handleStoreSpeakerUpdated = useSpeakersTabStore((s) => s.handleSpeakerUpdated);
    const handleStoreSpeakerMerged = useSpeakersTabStore((s) => s.handleSpeakerMerged);
    const [purging, setPurging] = useState(false);
    const [redoing, setRedoing] = useState(false);
    const [redoingDiarization, setRedoingDiarization] = useState(false);
    const [consolidatingTranscript, setConsolidatingTranscript] = useState(false);
    const fetchStoreFunnyTaskProgress = useTranscriptStore((s) => s.fetchFunnyTaskProgress);
    const queueingVoiceFixer = useCleanupStore((s) => s.queueingVoiceFixer);
    const queueingReconstruction = useReconstructionStore((s) => s.queueingReconstruction);
    const auxiliaryJobs = useWorkbenchStore((s) => s.auxiliaryJobs);
    const workbenchTaskProgress = useWorkbenchStore((s) => s.workbenchTaskProgress);
    const switchingReconstructionPlayback = useReconstructionStore((s) => s.switchingReconstructionPlayback);
    const loadingReconstructionWorkbench = useReconstructionStore((s) => s.loadingReconstructionWorkbench);
    const reconstructionWorkbench = useReconstructionStore((s) => s.reconstructionWorkbench);
    const savingReconstructionSettings = useReconstructionStore((s) => s.savingReconstructionSettings);
    const testingReconstructionSpeakerId = useReconstructionStore((s) => s.testingReconstructionSpeakerId);
    const reconstructionStudioTab = useReconstructionStore((s) => s.reconstructionStudioTab);
    const cleaningReconstructionSampleKey = useReconstructionStore((s) => s.cleaningReconstructionSampleKey);
    const updatingReconstructionSampleKey = useReconstructionStore((s) => s.updatingReconstructionSampleKey);
    const addingReconstructionSampleSpeakerId = useReconstructionStore((s) => s.addingReconstructionSampleSpeakerId);
    const approvingReconstructionSpeakerId = useReconstructionStore((s) => s.approvingReconstructionSpeakerId);
    const previewingReconstructionSegment = useReconstructionStore((s) => s.previewingReconstructionSegment);
    const setReconstructionStudioTab = useReconstructionStore((s) => s.setReconstructionStudioTab);
    const savingVoiceFixerSettings = useCleanupStore((s) => s.savingVoiceFixerSettings);
    const activeEditingClip = editingClipId != null ? (clips.find(c => c.id === editingClipId) || null) : null;
    const showClipEditorMain = activeTab === 'clips' && !!activeEditingClip && !!clipEditorDraft;
    const {
        isYoutubeMedia,
        isUploadedMedia,
        isTikTokMedia,
        isLocallyHostedMedia,
        isUploadedAudio,
        canShowCloneTab,
        canShowYoutubeTab,
        aiMetadataTabLabel,
        aiMetadataTabTitle,
        localMediaPending,
        transcriptJobActive,
        voiceFixerBusy,
        voiceFixerPaused,
        hasVoiceFixerCleaned,
        voiceFixerApplyScope,
        usingVoiceFixerForPlayback,
        usingVoiceFixerForProcessing,
        reconstructionStatus,
        reconstructionBusy,
        reconstructionPaused,
        hasReconstructionAudio,
        usingReconstructionForPlayback,
        currentUploadedPlaybackSource,
        reconstructionAudioUrl,
    } = useVideoStore(selectVideoDetailDerivedMetadata);
    useVideoMetadataPolling(id);
    const sidebarTabs = useMemo<SidebarTabConfig[]>(() => {
        const tabs: SidebarTabConfig[] = [
            {
                id: 'transcript',
                label: 'Transcript',
                title: 'Transcript',
                icon: FileText,
                activeClassName: 'border-blue-200 bg-blue-50 text-blue-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            },
            {
                id: 'optimize',
                label: 'Optimize',
                title: 'Open the transcript optimization workbench',
                icon: CheckCircle2,
                activeClassName: 'border-emerald-200 bg-emerald-50 text-emerald-700 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            },
            {
                id: 'clips',
                label: 'Clips',
                title: 'Clips',
                icon: Scissors,
                activeClassName: 'border-purple-200 bg-purple-50 text-purple-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            },
            {
                id: 'speakers',
                label: 'Speakers',
                title: 'Speakers',
                icon: Users,
                activeClassName: 'border-orange-200 bg-orange-50 text-orange-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            },
        ];

        if (isUploadedMedia) {
            tabs.push({
                id: 'cleanup',
                label: 'Cleanup',
                title: 'Open the cleanup workbench',
                icon: AudioLines,
                activeClassName: 'border-sky-200 bg-sky-50 text-sky-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            });
            tabs.push({
                id: 'reconstruction',
                label: 'Rebuild',
                title: 'Open the conversation reconstruction studio',
                icon: Bot,
                activeClassName: 'border-violet-200 bg-violet-50 text-violet-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            });
        }

        if (canShowYoutubeTab) {
            tabs.push({
                id: 'youtube',
                label: aiMetadataTabLabel,
                title: aiMetadataTabTitle,
                icon: Sparkles,
                activeClassName: 'border-emerald-200 bg-emerald-50 text-emerald-600 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            });
        }

        if (canShowCloneTab) {
            tabs.push({
                id: 'clone',
                label: 'Clone',
                title: 'Generate an AI episode clone draft',
                icon: Clapperboard,
                activeClassName: 'border-fuchsia-200 bg-fuchsia-50 text-fuchsia-700 shadow-sm',
                inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
            });
        }

        tabs.push({
            id: 'chat',
            label: 'Chat',
            title: 'Ask transcript-grounded questions about this episode',
            icon: MessageSquareText,
            activeClassName: 'border-indigo-200 bg-indigo-50 text-indigo-700 shadow-sm',
            inactiveClassName: 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-700',
        });

        return tabs;
    }, [aiMetadataTabLabel, aiMetadataTabTitle, canShowCloneTab, canShowYoutubeTab, isUploadedMedia]);
    const resolveWorkbenchAudioUrl = (url?: string) => {
        if (!url) return '';
        return /^https?:\/\//i.test(url) ? url : toApiUrl(url);
    };
    const fetchFunnyTaskProgress = async () => {
        if (!id) return;
        await fetchStoreFunnyTaskProgress(Number(id));
    };

    const fetchData = async () => {
        if (!id) return;
        await refreshVideoData(id);
    };

    useEffect(() => {
        useVideoStore.getState().resetVideoState();
        useCloneStore.getState().resetCloneState();
        useYoutubeStore.getState().resetYoutubeState();
        useSpeakersTabStore.getState().resetSpeakersTabState();
        useClipsStore.getState().resetClipsState();
        useWorkbenchStore.getState().resetWorkbenchState();
        useCleanupStore.getState().resetCleanupState();
        useTranscriptStore.getState().resetTranscriptState();
        usePlayerStore.getState().resetPlayerState();
        if (id) fetchData();
    }, [id]);

    // Fetch clone engines for EpisodeChatWorkbench when the chat tab is active.
    // (The clone tab manages its own fetches internally via CloneTab.)
    useEffect(() => {
        if (activeTab !== 'chat') return;
        const controller = new AbortController();
        void useCloneStore.getState().fetchCloneEngines(controller.signal);
        return () => controller.abort();
    }, [activeTab, id]);

    useEffect(() => {
        if (activeTab !== 'chat' || !cloneUsesOllama) return;
        const controller = new AbortController();
        void useCloneStore.getState().fetchCloneOllamaModels(controller.signal);
        return () => controller.abort();
    }, [activeTab, cloneUsesOllama, cloneEngineKey]);

    useEffect(() => {
        if (activeTab !== 'optimize' || !id || segments.length === 0) return;
        const controller = new AbortController();
        const store = useTranscriptStore.getState();
        void store.fetchTranscriptRollbackOptions(Number(id), controller.signal);
        void store.fetchTranscriptGoldWindows(Number(id), controller.signal);
        void store.fetchTranscriptEvaluationResults(Number(id), controller.signal);
        store.fetchTranscriptEvaluationResults(Number(id), controller.signal).then(() => {
            const results = useTranscriptStore.getState().transcriptEvaluationResults;
            const reviews = useTranscriptStore.getState().evaluationReviewsByResultId;
            results.slice(0, 8).forEach((item) => {
                if (!reviews[item.id]) void store.fetchEvaluationReviews(item.id);
            });
        });
        return () => controller.abort();
    }, [activeTab, id, segments.length]);

    useEffect(() => {
        if (!isUploadedMedia || !video?.id) {
            useWorkbenchStore.getState().resetWorkbenchState();
            return;
        }
        void useWorkbenchStore.getState().loadAuxiliaryJobs(video.id);
        if (!(voiceFixerBusy || voiceFixerPaused || queueingVoiceFixer || reconstructionBusy || reconstructionPaused || queueingReconstruction)) {
            return;
        }
        const timer = window.setInterval(() => {
            void useWorkbenchStore.getState().loadAuxiliaryJobs(video.id);
        }, 2500);
        return () => window.clearInterval(timer);
    }, [
        isUploadedMedia,
        queueingReconstruction,
        queueingVoiceFixer,
        reconstructionBusy,
        reconstructionPaused,
        video?.id,
        voiceFixerBusy,
        voiceFixerPaused,
    ]);

    const workbenchTaskIsRunning = String(workbenchTaskProgress?.status || '').toLowerCase() === 'running';

    useEffect(() => {
        if (!id || !isUploadedMedia) {
            useWorkbenchStore.setState({ workbenchTaskProgress: null });
            return;
        }
        const onWorkbenchTab = activeTab === 'cleanup' || activeTab === 'reconstruction';
        const shouldPoll =
            onWorkbenchTab ||
            loadingReconstructionWorkbench ||
            addingReconstructionSampleSpeakerId !== null ||
            cleaningReconstructionSampleKey !== null ||
            updatingReconstructionSampleKey !== null ||
            testingReconstructionSpeakerId !== null ||
            approvingReconstructionSpeakerId !== null ||
            previewingReconstructionSegment ||
            switchingReconstructionPlayback ||
            queueingReconstruction ||
            queueingVoiceFixer ||
            workbenchTaskIsRunning;
        if (!shouldPoll) {
            return;
        }

        void useWorkbenchStore.getState().fetchWorkbenchTaskProgress(Number(id));
        const intervalMs = workbenchTaskIsRunning ? 700 : 1800;
        const timer = window.setInterval(() => {
            void useWorkbenchStore.getState().fetchWorkbenchTaskProgress(Number(id));
        }, intervalMs);
        return () => window.clearInterval(timer);
    }, [
        activeTab,
        addingReconstructionSampleSpeakerId,
        approvingReconstructionSpeakerId,
        cleaningReconstructionSampleKey,
        id,
        isUploadedMedia,
        loadingReconstructionWorkbench,
        previewingReconstructionSegment,
        queueingReconstruction,
        queueingVoiceFixer,
        switchingReconstructionPlayback,
        testingReconstructionSpeakerId,
        updatingReconstructionSampleKey,
        workbenchTaskIsRunning,
    ]);

    useEffect(() => {
        if (!video || !isUploadedMedia || segments.length === 0) return;
        if (loadingReconstructionWorkbench || reconstructionWorkbench) return;
        void loadReconstructionWorkbench();
    }, [video?.id, isUploadedMedia, segments.length]);


    useEffect(() => {
        useReconstructionStore.getState().syncInstructionDraftFromVideo(video, isUploadedMedia);
    }, [isUploadedMedia, video]);

    useEffect(() => {
        // Reset deep-link jump state when navigating to a different video.
        setSearchQuery('');
        setSearchMatchIndex(0);
        setDeepLinkedSegmentId(null);
        setExpandedFunnySummaryIds(new Set());
        useReconstructionStore.getState().resetReconstructionState();
    }, [id]);

    useEffect(() => {
        useReconstructionStore.getState().syncWorkbenchSelection();
    }, [reconstructionWorkbench]);

    useEffect(() => {
        if (reconstructionWorkbench?.all_speakers_approved) return;
        if (reconstructionStudioTab === 'reconstruction') {
            setReconstructionStudioTab('voices');
        }
    }, [reconstructionStudioTab, reconstructionWorkbench?.all_speakers_approved]);

    const auxiliaryJobSortTime = (job: Job) => {
        const started = job.started_at ? Date.parse(job.started_at) : NaN;
        if (Number.isFinite(started)) return started;
        const created = job.created_at ? Date.parse(job.created_at) : NaN;
        return Number.isFinite(created) ? created : 0;
    };

    const latestAuxiliaryJobByType = (jobType: 'voicefixer_cleanup' | 'conversation_reconstruct') =>
        auxiliaryJobs
            .filter((job) => String(job.job_type || '').toLowerCase() === jobType)
            .sort((a, b) => auxiliaryJobSortTime(b) - auxiliaryJobSortTime(a))[0] || null;

    const reconstructionJob = latestAuxiliaryJobByType('conversation_reconstruct');

    useClipPreviewLoop();
    useTranscriptSegmentEditLoop();

    const tParam = searchParams.get('t');
    const requestedTabParam = String(searchParams.get('tab') || '').trim().toLowerCase();
    const segmentParam = searchParams.get('segment_id');
    const searchQueryParam = searchParams.get('q');
    const searchModeParam = String(searchParams.get('search_mode') || '').trim().toLowerCase();
    const requestedJumpTime = tParam ? Number(tParam) : NaN;
    const requestedSegmentId = segmentParam ? Number(segmentParam) : NaN;
    const requestedSearchQuery = String(searchQueryParam || '').trim();
    const requestedSearchMode = searchModeParam === 'exact' ? 'exact' : '';

    const {
        handleSeek,
        handleCitationClick,
        syncPlayerClockSnapshot,
    } = useVideoSeek({
        videoId: id,
        activeTab,
        setActiveTab,
        segments,
        requestedJumpTime,
        requestedSegmentId,
    });

    useEffect(() => {
        if (!id) return;
        const hasRequestedSegment = Number.isInteger(requestedSegmentId) && requestedSegmentId > 0;
        const shouldRestoreSearch = requestedSearchMode === 'exact' && !!requestedSearchQuery;
        const requestedTab = (() => {
            const value = requestedTabParam;
            if (value === 'chat') return 'chat';
            if (value === 'clone') return 'clone';
            if (value === 'optimize') return 'optimize';
            if (value === 'clips') return 'clips';
            if (value === 'speakers') return 'speakers';
            if (value === 'youtube') return 'youtube';
            if (value === 'cleanup' && isUploadedMedia) return 'cleanup';
            if (value === 'reconstruction' && isUploadedMedia) return 'reconstruction';
            return 'transcript';
        })();
        setDeepLinkedSegmentId(hasRequestedSegment ? requestedSegmentId : null);
        setSearchQuery(shouldRestoreSearch ? requestedSearchQuery : '');
        setSearchMatchIndex(0);
        setActiveTab(requestedTab);
        if (hasRequestedSegment || shouldRestoreSearch) {
            setFollowPlayback(false);
            setActiveTab('transcript');
        }
    }, [id, isUploadedMedia, requestedSearchMode, requestedSearchQuery, requestedSegmentId, requestedTabParam]);

    useEffect(() => {
        if (isLocallyHostedMedia && activeTab === 'youtube') {
            setActiveTab('transcript');
        }
        if (!isUploadedMedia && (activeTab === 'cleanup' || activeTab === 'reconstruction')) {
            setActiveTab('transcript');
        }
    }, [activeTab, isLocallyHostedMedia, isUploadedMedia]);

    const episodeBusy =
        purging ||
        redoing ||
        redoingDiarization ||
        consolidatingTranscript ||
        transcriptJobActive ||
        voiceFixerBusy ||
        reconstructionBusy ||
        queueingVoiceFixer ||
        queueingReconstruction ||
        savingVoiceFixerSettings ||
        savingReconstructionSettings;

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

    const handleConsolidateTranscript = async () => {
        if (!video) return;
        if (!confirm('Post-process this transcript to merge same-speaker fragments and smooth tiny diarization cuts?')) return;
        setConsolidatingTranscript(true);
        try {
            const res = await api.post(`/videos/${video.id}/consolidate-transcript`);
            setSegments([]);
            fetchData();
            const merged = Number(res?.data?.merged_count || 0);
            const reassigned = Number(res?.data?.reassigned_islands || 0);
            alert(`Transcript consolidated. ${merged} segment merges, ${reassigned} short speaker-island reassignment${reassigned === 1 ? '' : 's'}.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to consolidate transcript');
        } finally {
            setConsolidatingTranscript(false);
        }
    };

    const handleSetUploadedPlaybackSource = async (source: UploadedPlaybackSource) => {
        await useReconstructionStore.getState().setUploadedPlaybackSource(video, isUploadedMedia, source, currentUploadedPlaybackSource, setVideo);
    };

    const handleQueueReconstruction = async (force = false) => {
        await useReconstructionStore.getState().queueReconstruction(video, isUploadedMedia, hasReconstructionAudio, force, setVideo);
    };

    const loadReconstructionWorkbench = async () => {
        await useReconstructionStore.getState().loadReconstructionWorkbench(video, isUploadedMedia, segments.length);
    };

    const handleSetReconstructionPlayback = async (enabled: boolean) => {
        await handleSetUploadedPlaybackSource(enabled ? 'reconstructed' : 'original');
    };

    const clampPercent = (value: number) => Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));

    const getAuxiliaryStageData = (job: Job | null, kind: 'voicefixer' | 'reconstruction') => {
        const rawProgress = clampPercent(Number(job?.progress || 0));
        const detail = String(job?.status_detail || '').toLowerCase();
        const isCompleted = String(job?.status || '').toLowerCase() === 'completed';

        if (kind === 'reconstruction') {
            const referencesActive = detail.includes('extracting speaker references');
            const modelLoadActive = detail.includes('loading reconstruction tts model');
            const synthActive = detail.includes('reconstructing segment') || detail.includes('reconstructing long segment');
            const assembleActive = detail.includes('writing reconstructed');

            return {
                progress: rawProgress,
                detail: String(job?.status_detail || ''),
                stages: [
                    {
                        key: 'references',
                        label: 'References',
                        state: (rawProgress >= 15 || modelLoadActive || synthActive || assembleActive ? 'completed' : referencesActive || rawProgress > 0 ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                        percent: referencesActive || rawProgress > 0 ? Math.max(10, (rawProgress / 15) * 100) : rawProgress >= 15 ? 100 : 0,
                    },
                    {
                        key: 'model',
                        label: 'Model',
                        state: (synthActive || assembleActive || rawProgress >= 25 ? 'completed' : modelLoadActive || (rawProgress >= 15 && rawProgress < 25) ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                        percent: modelLoadActive || (rawProgress >= 15 && rawProgress < 25) ? Math.max(10, ((rawProgress - 15) / 10) * 100) : rawProgress >= 25 ? 100 : 0,
                    },
                    {
                        key: 'synth',
                        label: 'Synthesis',
                        state: (assembleActive || rawProgress >= 94 ? 'completed' : synthActive || (rawProgress >= 20 && rawProgress < 94) ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                        percent: synthActive || (rawProgress >= 20 && rawProgress < 94) ? Math.max(5, ((rawProgress - 20) / 74) * 100) : rawProgress >= 94 ? 100 : 0,
                    },
                    {
                        key: 'assemble',
                        label: 'Assemble',
                        state: (isCompleted ? 'completed' : assembleActive || rawProgress >= 94 ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                        percent: isCompleted ? 100 : assembleActive || rawProgress >= 94 ? Math.max(10, ((rawProgress - 94) / 6) * 100) : 0,
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
                    state: (rawProgress >= 45 || restoreActive || blendActive || levelActive || mergeActive ? 'completed' : prepareActive || rawProgress > 0 ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                    percent: prepareActive || rawProgress > 0 ? Math.max(10, (rawProgress / 45) * 100) : rawProgress >= 45 ? 100 : 0,
                },
                {
                    key: 'restore',
                    label: 'Restore',
                    state: (rawProgress >= 62 || blendActive || levelActive || mergeActive ? 'completed' : restoreActive || (rawProgress >= 45 && rawProgress < 62) ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                    percent: restoreActive || (rawProgress >= 45 && rawProgress < 62) ? Math.max(10, ((rawProgress - 45) / 17) * 100) : rawProgress >= 62 ? 100 : 0,
                },
                {
                    key: 'finish',
                    label: 'Finish',
                    state: (isCompleted ? 'completed' : blendActive || levelActive || mergeActive || rawProgress >= 62 ? 'active' : 'pending') as 'pending' | 'active' | 'completed',
                    percent: isCompleted ? 100 : blendActive || levelActive || mergeActive || rawProgress >= 62 ? Math.max(10, ((rawProgress - 62) / 38) * 100) : 0,
                },
            ],
        };
    };

    const renderAuxiliaryProgressCard = (
        kind: 'voicefixer' | 'reconstruction',
        job: Job | null,
        options?: { compact?: boolean; className?: string }
    ) => {
        if (!job) return null;
        const compact = !!options?.compact;
        const { progress, detail, stages } = getAuxiliaryStageData(job, kind);
        const accent = kind === 'voicefixer'
            ? {
                ring: 'text-sky-600',
                active: 'bg-sky-500',
                soft: 'bg-sky-100 text-sky-700 border-sky-200',
            }
            : {
                ring: 'text-violet-600',
                active: 'bg-violet-500',
                soft: 'bg-violet-100 text-violet-700 border-violet-200',
            };
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
                                        <span className="text-slate-400">{stage.state === 'completed' ? 'done' : stage.state === 'active' ? `${Math.round(clampPercent(stage.percent))}%` : 'pending'}</span>
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
    };

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

    useEffect(() => {
        if (!assignPopup || !video) return;
        const timeoutId = window.setTimeout(() => {
            void fetchAssignSpeakers(video.channel_id);
        }, 200);

        return () => {
            window.clearTimeout(timeoutId);
        };
    }, [assignPopup, assignSearch, fetchAssignSpeakers, video]);

    const handleAssignSpeaker = async (speakerId: number) => {
        await assignSpeaker(speakerId, fetchData);
    };

    const handleSpeakerListUpdated = (updatedSpeaker: Speaker) => {
        handleStoreSpeakerUpdated(updatedSpeaker, (speakerId, name) => {
            setSegments(prev => prev.map(segment =>
                segment.speaker_id === speakerId
                    ? { ...segment, speaker: name }
                    : segment
            ));
        });
    };

    const handleSpeakerListMerged = () => {
        void handleStoreSpeakerMerged(id, setSegments);
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

    const sanitizeFilename = (value: string, fallback: string) => {
        const cleaned = String(value || '')
            .replace(/[\\/:*?"<>|]+/g, '_')
            .replace(/\s+/g, ' ')
            .trim();
        return cleaned || fallback;
    };

    const buildEpisodeScriptText = () => {
        if (!video || segments.length === 0) return '';

        const lines: string[] = [];
        lines.push(video.title || `Episode ${video.id}`);
        const v = video as any;
        if (v.channel_name) lines.push(`Channel: ${v.channel_name}`);
        if (video.published_at) {
            lines.push(`Published: ${new Date(video.published_at).toLocaleDateString()}`);
        }
        lines.push('');

        let currentSpeaker = '';
        let currentParts: string[] = [];

        const flushCurrent = () => {
            if (!currentSpeaker || currentParts.length === 0) return;
            lines.push(`${currentSpeaker}:`);
            lines.push(currentParts.join(' ').replace(/\s+/g, ' ').trim());
            lines.push('');
            currentSpeaker = '';
            currentParts = [];
        };

        for (const segment of segments) {
            const speakerName = (segment.speaker || '').trim() || 'Unknown Speaker';
            const text = String(segment.text || '').replace(/\s+/g, ' ').trim();
            if (!text) continue;

            if (speakerName !== currentSpeaker) {
                flushCurrent();
                currentSpeaker = speakerName;
            }
            currentParts.push(text);
        }

        flushCurrent();

        return lines.join('\n').trim() + '\n';
    };

    const handleExportEpisodeScript = () => {
        if (!video || segments.length === 0) return;
        const scriptText = buildEpisodeScriptText();
        if (!scriptText.trim()) {
            alert('No transcript text is available to export.');
            return;
        }
        const filename = `${sanitizeFilename(video.title || `episode_${video.id}`, `episode_${video.id}`)} - script.txt`;
        downloadBlobResponse(new Blob([scriptText], { type: 'text/plain;charset=utf-8' }), filename);
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

    const renderMainPlayer = (containerClassName: string) => (
        <VideoPlayer
            video={video}
            containerClassName={containerClassName}
            isLocallyHostedMedia={isLocallyHostedMedia}
            isUploadedMedia={isUploadedMedia}
            isTikTokMedia={isTikTokMedia}
            localMediaPending={localMediaPending}
            isUploadedAudio={isUploadedAudio}
            requestedJumpTime={requestedJumpTime}
            playbackRate={playbackRate}
            currentUploadedPlaybackSource={currentUploadedPlaybackSource}
            usingReconstructionForPlayback={usingReconstructionForPlayback}
            reconstructionStatus={reconstructionStatus}
            usingVoiceFixerForPlayback={usingVoiceFixerForPlayback}
            voiceFixerApplyScope={voiceFixerApplyScope}
            hasVoiceFixerCleaned={hasVoiceFixerCleaned}
            hasReconstructionAudio={hasReconstructionAudio}
            usingVoiceFixerForProcessing={usingVoiceFixerForProcessing}
            episodeBusy={episodeBusy}
            switchingReconstructionPlayback={switchingReconstructionPlayback}
            onUploadedPlaybackSourceChange={handleSetUploadedPlaybackSource}
            onPlayerClock={syncPlayerClockSnapshot}
        />
    );

    const currentWorkbenchProgress = workbenchTaskProgress && String(workbenchTaskProgress.status || '').toLowerCase() !== 'idle'
        ? workbenchTaskProgress
        : null;
    const reconstructionWorkbenchProgress = currentWorkbenchProgress && String(currentWorkbenchProgress.area || '').toLowerCase() === 'reconstruction'
        ? currentWorkbenchProgress
        : null;

    const renderWorkbenchTaskProgressCard = (progress: WorkbenchTaskProgress | null) => {
        if (!progress || String(progress.status || '').toLowerCase() === 'idle') return null;
        const area = String(progress.area || '').toLowerCase();
        const tone = area === 'cleanup' ? 'sky' : 'violet';
        const palette = tone === 'sky'
            ? {
                shell: 'border-sky-200 bg-sky-50/85',
                badge: 'border-sky-200 bg-white text-sky-700',
                bar: 'bg-sky-500',
            }
            : {
                shell: 'border-violet-200 bg-violet-50/85',
                badge: 'border-violet-200 bg-white text-violet-700',
                bar: 'bg-violet-500',
            };
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
    };

    const renderWorkbenchActivityCard = (activity: WorkbenchActivity | null) => {
        if (!activity) return null;
        const palette = activity.tone === 'sky'
            ? {
                shell: 'border-sky-200 bg-sky-50/85',
                badge: 'border-sky-200 bg-white text-sky-700',
            }
            : {
                shell: 'border-violet-200 bg-violet-50/85',
                badge: 'border-violet-200 bg-white text-violet-700',
            };

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
    };

    const reconstructionWorkbenchActivity: WorkbenchActivity | null = loadingReconstructionWorkbench
        ? {
            label: 'Refreshing reconstruction workbench',
            detail: 'Loading the latest voice references, performance samples, and approval state.',
            tone: 'violet',
        }
        : addingReconstructionSampleSpeakerId !== null
            ? {
                label: 'Adding performance sample',
                detail: 'Finding another candidate segment for the selected speaker.',
                tone: 'violet',
            }
            : cleaningReconstructionSampleKey !== null
                ? {
                    label: 'Cleaning performance sample',
                    detail: 'Running VoiceFixer on the chosen sample so you can compare a cleaner prosody reference.',
                    tone: 'violet',
                }
                : updatingReconstructionSampleKey !== null
                    ? {
                        label: 'Updating sample state',
                        detail: 'Saving the selection or rejection state for this performance sample.',
                        tone: 'violet',
                    }
                    : testingReconstructionSpeakerId !== null
                        ? {
                            label: 'Generating voice test',
                            detail: 'Synthesizing a fresh speaker test clip with the current reconstruction settings.',
                            tone: 'violet',
                        }
                        : approvingReconstructionSpeakerId !== null
                            ? {
                                label: 'Saving voice approval',
                                detail: 'Updating the review state for this speaker before reconstruction.',
                                tone: 'violet',
                            }
                            : savingReconstructionSettings
                                ? {
                                    label: 'Saving reconstruction settings',
                                    detail: 'Applying the current instruction template and reconstruction mode.',
                                    tone: 'violet',
                                }
                                : previewingReconstructionSegment
                                    ? {
                                        label: 'Rendering segment preview',
                                        detail: 'Generating a short reconstruction sample for the selected transcript segment.',
                                        tone: 'violet',
                                    }
                                    : switchingReconstructionPlayback
                                        ? {
                                            label: 'Switching playback source',
                                            detail: 'Updating whether the main player follows the original upload, the cleanup pass, or the reconstructed track.',
                                            tone: 'violet',
                                        }
                                        : queueingReconstruction
                                            ? {
                                                label: 'Starting full reconstruction',
                                                detail: 'Queueing the rebuild job now. The stage breakdown appears below once the job is registered.',
                                                tone: 'violet',
                                            }
                                            : null;



    return (
        <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-slate-50">
            {/* Left Column: Tools */}
            <div className="w-[450px] flex flex-col bg-white border-r border-slate-200 shrink-0 shadow-xl z-10 transition-all relative">
                {/* Tabs */}
                <div className="sticky top-0 z-10 border-b border-slate-100 bg-white">
                    <div className="flex items-center gap-1 overflow-x-auto px-2 py-2 [scrollbar-width:none]" aria-label="Episode sidebar sections">
                        {sidebarTabs.map((tab) => {
                            const Icon = tab.icon;
                            const isActive = activeTab === tab.id;

                            return (
                                <button
                                    key={tab.id}
                                    type="button"
                                    onClick={() => setActiveTab(tab.id)}
                                    title={tab.title}
                                    aria-label={tab.label}
                                    aria-pressed={isActive}
                                    className={`group inline-flex h-10 shrink-0 items-center rounded-xl border px-2.5 text-sm font-medium transition-all ${isActive ? tab.activeClassName : tab.inactiveClassName}`}
                                >
                                    <Icon size={16} className="shrink-0" />
                                    <span
                                        className={`overflow-hidden whitespace-nowrap text-left transition-all ${isActive ? 'ml-2 max-w-28 opacity-100' : 'ml-0 max-w-0 opacity-0'}`}
                                    >
                                        {tab.label}
                                    </span>
                                    {!isActive && <span className="sr-only">{tab.label}</span>}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-hidden relative bg-slate-50/50">
                    {activeTab === 'transcript' && (
                        <TranscriptTab
                            videoId={Number(id)}
                            onSeek={handleSeek}
                            onSelectionCreated={(start, end, title) => setClipSelection({ start, end, defaultTitle: title })}
                            onRefreshVideo={fetchData}
                            onNavigateToOptimize={() => setActiveTab('optimize')}
                        />
                    )}
                    {activeTab === 'optimize' && (
                        <OptimizeSidebar
                            videoId={Number(id)}
                            onNavigateToTranscript={() => setActiveTab('transcript')}
                        />
                    )}
                    {activeTab === 'clips' && (
                        <ClipsTab
                            videoId={Number(id)}
                            isActive={activeTab === 'clips'}
                            onSeek={handleSeek}
                        />
                    )}
                    {activeTab === 'chat' && (
                        <div className="h-full overflow-y-auto p-4">
                            <div className="space-y-4">
                                <div className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-white to-sky-50 p-4 shadow-sm">
                                    <div className="flex items-start gap-3">
                                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-indigo-600 shadow-sm">
                                            <MessageSquareText size={18} />
                                        </div>
                                        <div>
                                            <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-indigo-600">Episode Chat</div>
                                            <div className="mt-1 text-lg font-semibold text-slate-900">Transcript-grounded conversation</div>
                                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                                Use the main stage to ask questions about this episode. Replies are grounded in the transcript and include jump-back citations.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Chat Readiness</div>
                                    <div className="mt-3 space-y-2 text-sm text-slate-600">
                                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                                            <span>Transcript segments</span>
                                            <span className="font-semibold text-slate-800">{segments.length}</span>
                                        </div>
                                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                                            <span>Transcript language</span>
                                            <span className="font-semibold text-slate-800">{video?.transcript_language || 'Unknown'}</span>
                                        </div>
                                        <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2">
                                            <span>Model routing</span>
                                            <span className="font-semibold text-slate-800">{cloneEngines[0]?.label || 'Configured default'}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">How It Works</div>
                                    <div className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                                        <div className="rounded-xl bg-slate-50 px-3 py-2">Saved threads are scoped to this episode only.</div>
                                        <div className="rounded-xl bg-slate-50 px-3 py-2">The assistant pulls relevant transcript context before answering.</div>
                                        <div className="rounded-xl bg-slate-50 px-3 py-2">Citation chips in the reply jump back into the transcript timeline.</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'speakers' && (
                        <SpeakersTab
                            video={video}
                            videoId={Number(id)}
                            isActive={activeTab === 'speakers'}
                            onSegmentsUpdated={setSegments}
                            onSegmentsLoaded={setSegments}
                        />
                    )}
                    {activeTab === 'reconstruction' && isUploadedMedia && (
                        <ReconstructionSidebarTab
                            isActive={activeTab === 'reconstruction'}
                            segmentsCount={segments.length}
                            reconstructionWorkbench={reconstructionWorkbench}
                            loadingReconstructionWorkbench={loadingReconstructionWorkbench}
                            episodeBusy={episodeBusy}
                            hasReconstructionAudio={hasReconstructionAudio}
                            queueingReconstruction={queueingReconstruction}
                            reconstructionBusy={reconstructionBusy}
                            onRefreshWorkbench={() => void loadReconstructionWorkbench()}
                            onQueueReconstruction={() => void handleQueueReconstruction(hasReconstructionAudio)}
                            onNavigateToTranscript={() => setActiveTab('transcript')}
                        />
                    )}
                    {activeTab === 'youtube' && video && (
                        <YoutubeTab
                            video={video}
                            videoId={Number(id)}
                            segments={segments}
                            isYoutubeMedia={isYoutubeMedia}
                            isActive={activeTab === 'youtube'}
                            onVideoUpdated={setVideo}
                            onSeek={handleSeek}
                        />
                    )}
                </div>

                <ClipCreationPanel videoId={video.id} onClipCreated={() => setActiveTab('clips')} />
            </div>

            {/* Right Column: Video Stage */}
            <div className="flex-1 bg-slate-100 flex flex-col min-w-0 relative">
                {/* Header */}
                <div className="bg-white border-b border-slate-200 px-4 sm:px-6 py-3 flex flex-col gap-3 shadow-sm z-0">
                    <div className="flex items-center gap-3 min-w-0">
                        <button onClick={() => navigate(-1)} className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 hover:text-slate-700 transition-colors shrink-0">
                            <ArrowLeft size={20} />
                        </button>
                        <div className="flex-1 min-w-0">
                            <h1 className="font-semibold text-slate-800 line-clamp-1">{video.title}</h1>
                            <p className="text-xs text-slate-500">{new Date(video.published_at || '').toLocaleDateString()}</p>
                        </div>
                    </div>
                    {(() => {
                        return (
                            <div className="flex flex-col gap-2 w-full">
                                <div className="flex flex-wrap items-stretch gap-2 w-full">
                                    {transcriptJobActive && (
                                        <span className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-500 bg-slate-100 rounded-lg sm:min-h-9">
                                            <Loader2 size={14} className="animate-spin" />
                                            {video.status.charAt(0).toUpperCase() + video.status.slice(1)}...
                                        </span>
                                    )}
                                    <button
                                        onClick={handleExportEpisodeScript}
                                        disabled={segments.length === 0}
                                        className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                        title={segments.length === 0 ? 'Transcript required before exporting a script' : 'Export a plain-text script with speaker labels'}
                                    >
                                        <Download size={14} />
                                        Export Script
                                    </button>
                                    <button
                                        onClick={handleRedoDiarization}
                                        disabled={episodeBusy}
                                        className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                        title={transcriptJobActive ? `Cannot redo while ${video.status}` : "Re-run speaker diarization using improved speaker profiles"}
                                    >
                                        {redoingDiarization ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
                                        Redo Diarization
                                    </button>
                                    <button
                                        onClick={handleConsolidateTranscript}
                                        disabled={episodeBusy}
                                        className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                        title={transcriptJobActive ? `Cannot consolidate while ${video.status}` : "Merge same-speaker transcript fragments without re-running ASR or diarization"}
                                    >
                                        {consolidatingTranscript ? <Loader2 size={14} className="animate-spin" /> : <GitMerge size={14} />}
                                        Consolidate Transcript
                                    </button>
                                    <button
                                        onClick={handleRedoTranscript}
                                        disabled={episodeBusy}
                                        className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                        title={transcriptJobActive ? `Cannot redo while ${video.status}` : "Re-run transcription and then diarization"}
                                    >
                                        {redoing ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                                        Redo Transcription
                                    </button>
                                    <button
                                        onClick={handlePurgeTranscript}
                                        disabled={episodeBusy}
                                        className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                        title={transcriptJobActive ? `Cannot purge while ${video.status}` : "Purge transcript & diarization data"}
                                    >
                                        {purging ? <Loader2 size={14} className="animate-spin" /> : <Eraser size={14} />}
                                        Purge
                                    </button>
                                    {isUploadedMedia && (
                                        <>
                                            <button
                                                onClick={() => setActiveTab('cleanup')}
                                                className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-sky-700 bg-sky-50 hover:bg-sky-100 rounded-lg transition-colors min-h-9 max-sm:flex-1"
                                                title="Open the cleanup workbench"
                                            >
                                                <AudioLines size={14} />
                                                Cleanup Studio
                                            </button>
                                            <button
                                                onClick={() => setActiveTab('reconstruction')}
                                                disabled={segments.length === 0}
                                                className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-violet-700 bg-violet-50 hover:bg-violet-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-9 max-sm:flex-1"
                                                title={segments.length === 0 ? 'Transcript and diarization are required before reconstruction.' : 'Open the reconstruction studio'}
                                            >
                                                <Bot size={14} />
                                                Rebuild Studio
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>
                        );
                    })()}
                </div>

                {/* Video Container / Cleanup / Reconstruction / Clip Editor Workspace */}
                {activeTab === 'cleanup' && isUploadedMedia ? (
                    <CleanupTab
                        video={video}
                        videoId={Number(id)}
                        isActive={activeTab === 'cleanup'}
                        onVideoUpdated={setVideo}
                        episodeBusy={episodeBusy}
                        playerNode={renderMainPlayer('h-[360px] w-full')}
                        onNavigateToTranscript={() => setActiveTab('transcript')}
                    />
                ) : activeTab === 'optimize' ? (
                    <OptimizeTab
                        hasTranscript={segments.length > 0}
                        videoId={Number(id)}
                        isActive={activeTab === 'optimize'}
                        episodeBusy={episodeBusy}
                        selection={clipSelection}
                        transcriptLanguage={video?.transcript_language}
                        onSeek={handleSeek}
                        onNavigateToTranscript={() => setActiveTab('transcript')}
                        onRefreshVideo={fetchData}
                    />
                ) : activeTab === 'clone' ? (
                    <CloneTab
                        video={video}
                        segmentsCount={segments.length}
                        videoId={Number(id)}
                        isActive={activeTab === 'clone'}
                    />
                ) : activeTab === 'chat' ? (
                    <EpisodeChatWorkbench
                        videoId={Number(id)}
                        cloneEngines={cloneEngines}
                        cloneOllamaModels={cloneOllamaModels}
                        onCitationClick={handleCitationClick}
                    />
                ) : activeTab === 'reconstruction' && isUploadedMedia ? (
                    <ReconstructionTab
                        isActive={activeTab === 'reconstruction'}
                        reconstructionWorkbench={reconstructionWorkbench}
                        loadingReconstructionWorkbench={loadingReconstructionWorkbench}
                        segmentsCount={segments.length}
                        reconstructionStudioTab={reconstructionStudioTab}
                        reconstructionStatus={reconstructionStatus}
                        reconstructionPaused={reconstructionPaused}
                        reconstructionBusy={reconstructionBusy}
                        hasReconstructionAudio={hasReconstructionAudio}
                        usingReconstructionForPlayback={usingReconstructionForPlayback}
                        reconstructionError={video?.reconstruction_error}
                        reconstructionWorkbenchProgressNode={reconstructionWorkbenchProgress ? renderWorkbenchTaskProgressCard(reconstructionWorkbenchProgress) : null}
                        reconstructionWorkbenchActivityNode={reconstructionWorkbenchActivity ? renderWorkbenchActivityCard(reconstructionWorkbenchActivity) : null}
                        reconstructionJobNode={reconstructionJob ? renderAuxiliaryProgressCard('reconstruction', reconstructionJob, { className: 'mt-4 border-violet-200 bg-violet-50/50' }) : null}
                        video={video}
                        segments={segments}
                        isUploadedMedia={isUploadedMedia}
                        episodeBusy={episodeBusy}
                        reconstructionAudioUrl={reconstructionAudioUrl}
                        resolveWorkbenchAudioUrl={resolveWorkbenchAudioUrl}
                        setVideo={setVideo}
                        onSetReconstructionPlayback={(enabled) => void handleSetReconstructionPlayback(enabled)}
                        onRefreshWorkbench={() => void loadReconstructionWorkbench()}
                        onSetStudioTab={setReconstructionStudioTab}
                    />
                ) : showClipEditorMain && activeEditingClip && clipEditorDraft ? (
                    <ClipEditorWorkspace
                        activeEditingClip={activeEditingClip}
                        currentTime={currentTime}
                        video={video}
                        segments={segments}
                        playerNode={renderMainPlayer('w-full bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video')}
                        onSeek={handleSeek}
                    />
                ) : (
                    <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
                        <div className="w-full max-w-5xl">
                            {renderMainPlayer("w-full bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video")}
                        </div>
                    </div>
                )}
            </div>

            {/* Speaker Modal */}
            {selectedSpeaker && (
                <SpeakerModal
                    speaker={selectedSpeaker}
                    initialSample={initialSample || undefined}
                    onClose={closeSpeakerModal}
                    onUpdate={(updatedSpeaker) => {
                        handleSpeakerListUpdated(updatedSpeaker);
                    }}
                    onMerge={() => {
                        handleSpeakerListMerged();
                    }}
                />
            )}

            {/* Assign Speaker Popup (for segments with no speaker) */}
            {assignPopup && (
                <>
                    <div className="fixed inset-0 z-40" onClick={closeAssignPopup} />
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
