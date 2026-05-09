import { startTransition, useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type { Video, TranscriptSegment, Clip, Speaker, FunnyMoment, ReconstructionWorkbench, Job, WorkbenchTaskProgress, CleanupWorkbench, ClearVoiceInstallInfo, ClearVoiceTestResult, EpisodeChatCitation, TranscriptQuality, TranscriptRollbackOption, TranscriptRestoreResponse, TranscriptGoldWindow, TranscriptEvaluationResult, TranscriptEvaluationReview, TranscriptEvaluationBatchResponse, TranscriptRepairQueueResponse, TranscriptDiarizationRebuildQueueResponse, TranscriptRetranscriptionQueueResponse } from '../../types';
import { Loader2, ArrowLeft, FileText, Scissors, Users, X, CheckCircle2, Play, Pause, Plus, Mic, Search, ChevronUp, ChevronDown, GitMerge, RotateCcw, Eraser, AudioLines, Smile, RefreshCw, Bot, Pencil, Save, XCircle, Download, PlayCircle, Clock, Sparkles, Clapperboard, CircleHelp, MessageSquareText, type LucideIcon } from 'lucide-react';
import { SpeakerModal } from '../../components/SpeakerModal';
import { EpisodeChatWorkbench } from '../../components/video/EpisodeChatWorkbench';
import { CloneTab } from './tabs/CloneTab';
import { YoutubeTab } from './tabs/YoutubeTab';
import { SpeakersTab } from './tabs/SpeakersTab';
import { ClipsTab } from './tabs/ClipsTab';
import { CleanupTab } from './tabs/CleanupTab';
import { ClipEditorWorkspace } from './tabs/ClipEditorWorkspace';
import { OptimizeTab } from './tabs/OptimizeTab';
import { ReconstructionSidebarTab, ReconstructionTab } from './tabs/ReconstructionTab';
import { useCloneStore, useCloneUsesOllama } from '../../store/useCloneStore';
import { useYoutubeStore } from '../../store/useYoutubeStore';
import { useSpeakersTabStore } from '../../store/useSpeakersTabStore';
import { useClipsStore } from '../../store/useClipsStore';
import { useWorkbenchStore } from '../../store/useWorkbenchStore';
import { useCleanupStore } from '../../store/useCleanupStore';
import { useReconstructionStore } from '../../store/useReconstructionStore';
import { useTranscriptStore } from '../../store/useTranscriptStore';
import { formatTime } from '../../lib/formatters';

type UnifiedPlayer = {
    getCurrentTime?: () => number;
    seekTo?: (seconds: number, allowSeekAhead?: boolean) => void;
    playVideo?: () => void | Promise<void>;
    pauseVideo?: () => void;
    getPlaybackRate?: () => number;
    setPlaybackRate?: (rate: number) => void;
    getPlayerState?: () => number;
};

type VideoSidebarTab = 'transcript' | 'optimize' | 'clips' | 'speakers' | 'cleanup' | 'reconstruction' | 'clone' | 'chat' | 'youtube';

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

type UploadedPlaybackSource = 'original' | 'cleaned' | 'reconstructed';

const PLAYER_UI_UPDATE_INTERVAL_MS = 120;
const PLAYER_UI_MIN_DELTA_SECONDS = 0.12;

const transcriptOptimizationHelp: Record<string, string> = {
    repair: 'Low-risk repair merges tiny same-speaker fragments, absorbs short unknown interruptions, applies conservative entity repair, and cleans transcript formatting without re-running ASR.',
    rebuild: 'Diarization rebuild keeps the raw transcript words but recomputes speaker turns and speaker matching. Use it when labeling is unstable but the wording is mostly correct.',
    retranscribe: 'Full retranscription discards the current transcript and runs a fresh transcription plus diarization pass. Use it for multilingual failures or broadly inaccurate text.',
};

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
    const [activeTab, setActiveTab] = useState<VideoSidebarTab>('transcript');
    const transcriptRef = useRef<HTMLDivElement>(null);
    const initialJumpDoneRef = useRef(false);
    const initialSeekDoneRef = useRef(false);
    const lastAutoScrollSegIdRef = useRef<number | null>(null);

    // Player State
    const [player, setPlayer] = useState<UnifiedPlayer | null>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [playbackRate, setPlaybackRate] = useState(1);
    const nativeMediaRef = useRef<HTMLMediaElement | null>(null);
    const pendingNativeSourceRestoreRef = useRef<{ currentTime: number; wasPlaying: boolean; playbackRate: number } | null>(null);
    const previousLocalMediaUrlRef = useRef('');
    const lastCurrentTimeUiUpdateMsRef = useRef(0);
    const lastPublishedCurrentTimeRef = useRef(0);
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

    const clips = useClipsStore((s) => s.clips);
    const clipTitle = useClipsStore((s) => s.clipTitle);
    const creatingClip = useClipsStore((s) => s.creatingClip);
    const setClipTitle = useClipsStore((s) => s.setClipTitle);
    const [startingTranscription, setStartingTranscription] = useState(false);

    // Search State
    const [searchQuery, setSearchQuery] = useState('');
    const [deepLinkedSegmentId, setDeepLinkedSegmentId] = useState<number | null>(null);
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

    // Clone store — only the values VideoDetailPage itself needs (chat sidebar + EpisodeChatWorkbench).
    const cloneEngines = useCloneStore((s) => s.cloneEngines);
    const cloneOllamaModels = useCloneStore((s) => s.cloneOllamaModels);
    const cloneUsesOllama = useCloneUsesOllama();
    const cloneEngineKey = useCloneStore((s) => s.cloneEngineKey);

    const [editingSegmentId, setEditingSegmentId] = useState<number | null>(null);
    const [editingSegmentWords, setEditingSegmentWords] = useState<string[]>([]);
    const [editingLoopSegment, setEditingLoopSegment] = useState(false);
    const [savingSegmentEdit, setSavingSegmentEdit] = useState(false);
    const editingClipId = useClipsStore((s) => s.editingClipId);
    const clipEditorDraft = useClipsStore((s) => s.clipEditorDraft);
    const clipPreviewLoop = useClipsStore((s) => s.clipPreviewLoop);
    const setEditingClipId = useClipsStore((s) => s.setEditingClipId);
    const setClipEditorDraft = useClipsStore((s) => s.setClipEditorDraft);
    const setClipEditorCropTarget = useClipsStore((s) => s.setClipEditorCropTarget);
    const setClipEditorDragRect = useClipsStore((s) => s.setClipEditorDragRect);

    const selectedSpeaker = useSpeakersTabStore((s) => s.selectedSpeaker);
    const initialSample = useSpeakersTabStore((s) => s.initialSample);
    const assignPopup = useSpeakersTabStore((s) => s.assignPopup);
    const assignSpeakers = useSpeakersTabStore((s) => s.assignSpeakers);
    const assignSearch = useSpeakersTabStore((s) => s.assignSearch);
    const assignLoading = useSpeakersTabStore((s) => s.assignLoading);
    const setAssignSearch = useSpeakersTabStore((s) => s.setAssignSearch);
    const closeSpeakerModal = useSpeakersTabStore((s) => s.closeSpeakerModal);
    const closeAssignPopup = useSpeakersTabStore((s) => s.closeAssignPopup);
    const openAssignPopup = useSpeakersTabStore((s) => s.openAssignPopup);
    const fetchAssignSpeakers = useSpeakersTabStore((s) => s.fetchAssignSpeakers);
    const assignSpeaker = useSpeakersTabStore((s) => s.assignSpeaker);
    const openSpeaker = useSpeakersTabStore((s) => s.openSpeaker);
    const handleStoreSpeakerUpdated = useSpeakersTabStore((s) => s.handleSpeakerUpdated);
    const handleStoreSpeakerMerged = useSpeakersTabStore((s) => s.handleSpeakerMerged);
    const [purging, setPurging] = useState(false);
    const [redoing, setRedoing] = useState(false);
    const [redoingDiarization, setRedoingDiarization] = useState(false);
    const [consolidatingTranscript, setConsolidatingTranscript] = useState(false);
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
    const setTranscriptQuality = useTranscriptStore((s) => s.setTranscriptQuality);
    const setLoadingTranscriptQuality = useTranscriptStore((s) => s.setLoadingTranscriptQuality);
    const setTranscriptQualityError = useTranscriptStore((s) => s.setTranscriptQualityError);
    const setTranscriptRollbackOptions = useTranscriptStore((s) => s.setTranscriptRollbackOptions);
    const setLoadingTranscriptRollbackOptions = useTranscriptStore((s) => s.setLoadingTranscriptRollbackOptions);
    const setRestoringTranscriptRunId = useTranscriptStore((s) => s.setRestoringTranscriptRunId);
    const setTranscriptGoldWindows = useTranscriptStore((s) => s.setTranscriptGoldWindows);
    const setLoadingTranscriptGoldWindows = useTranscriptStore((s) => s.setLoadingTranscriptGoldWindows);
    const setTranscriptGoldWindowsError = useTranscriptStore((s) => s.setTranscriptGoldWindowsError);
    const setSavingTranscriptGoldWindow = useTranscriptStore((s) => s.setSavingTranscriptGoldWindow);
    const setEvaluatingTranscript = useTranscriptStore((s) => s.setEvaluatingTranscript);
    const setTranscriptEvaluationSummary = useTranscriptStore((s) => s.setTranscriptEvaluationSummary);
    const setTranscriptEvaluationResults = useTranscriptStore((s) => s.setTranscriptEvaluationResults);
    const setLoadingTranscriptEvaluationResults = useTranscriptStore((s) => s.setLoadingTranscriptEvaluationResults);
    const setTranscriptEvaluationError = useTranscriptStore((s) => s.setTranscriptEvaluationError);
    const setReviewingEvaluationResultId = useTranscriptStore((s) => s.setReviewingEvaluationResultId);
    const setEvaluationReviewsByResultId = useTranscriptStore((s) => s.setEvaluationReviewsByResultId);
    const setGoldWindowLabelDraft = useTranscriptStore((s) => s.setGoldWindowLabelDraft);
    const setGoldWindowStartDraft = useTranscriptStore((s) => s.setGoldWindowStartDraft);
    const setGoldWindowEndDraft = useTranscriptStore((s) => s.setGoldWindowEndDraft);
    const setGoldWindowReferenceDraft = useTranscriptStore((s) => s.setGoldWindowReferenceDraft);
    const setGoldWindowEntitiesDraft = useTranscriptStore((s) => s.setGoldWindowEntitiesDraft);
    const setGoldWindowNotesDraft = useTranscriptStore((s) => s.setGoldWindowNotesDraft);
    const setEvaluationReviewVerdictDrafts = useTranscriptStore((s) => s.setEvaluationReviewVerdictDrafts);
    const setEvaluationReviewNotesDrafts = useTranscriptStore((s) => s.setEvaluationReviewNotesDrafts);
    const setEvaluationReviewReviewerDrafts = useTranscriptStore((s) => s.setEvaluationReviewReviewerDrafts);
    const setQueueingTranscriptRepair = useTranscriptStore((s) => s.setQueueingTranscriptRepair);
    const setQueueingDiarizationRebuild = useTranscriptStore((s) => s.setQueueingDiarizationRebuild);
    const setQueueingDiarizationBenchmark = useTranscriptStore((s) => s.setQueueingDiarizationBenchmark);
    const setQueueingFullRetranscription = useTranscriptStore((s) => s.setQueueingFullRetranscription);
    const setDiarizationBenchmarkSensitivity = useTranscriptStore((s) => s.setDiarizationBenchmarkSensitivity);
    const setDiarizationBenchmarkThreshold = useTranscriptStore((s) => s.setDiarizationBenchmarkThreshold);
    const queueingVoiceFixer = useCleanupStore((s) => s.queueingVoiceFixer);
    const queueingReconstruction = useReconstructionStore((s) => s.queueingReconstruction);
    const auxiliaryJobs = useWorkbenchStore((s) => s.auxiliaryJobs);
    const workbenchTaskProgress = useWorkbenchStore((s) => s.workbenchTaskProgress);
    const switchingReconstructionPlayback = useReconstructionStore((s) => s.switchingReconstructionPlayback);
    const loadingReconstructionWorkbench = useReconstructionStore((s) => s.loadingReconstructionWorkbench);
    const reconstructionWorkbench = useReconstructionStore((s) => s.reconstructionWorkbench);
    const savingReconstructionSettings = useReconstructionStore((s) => s.savingReconstructionSettings);
    const testingReconstructionSpeakerId = useReconstructionStore((s) => s.testingReconstructionSpeakerId);
    const reconstructionInstructionDraft = useReconstructionStore((s) => s.reconstructionInstructionDraft);
    const reconstructionStudioTab = useReconstructionStore((s) => s.reconstructionStudioTab);
    const selectedReconstructionSpeakerId = useReconstructionStore((s) => s.selectedReconstructionSpeakerId);
    const reconstructionTestTextDrafts = useReconstructionStore((s) => s.reconstructionTestTextDrafts);
    const cleaningReconstructionSampleKey = useReconstructionStore((s) => s.cleaningReconstructionSampleKey);
    const updatingReconstructionSampleKey = useReconstructionStore((s) => s.updatingReconstructionSampleKey);
    const addingReconstructionSampleSpeakerId = useReconstructionStore((s) => s.addingReconstructionSampleSpeakerId);
    const approvingReconstructionSpeakerId = useReconstructionStore((s) => s.approvingReconstructionSpeakerId);
    const selectedReconstructionPreviewSegmentId = useReconstructionStore((s) => s.selectedReconstructionPreviewSegmentId);
    const previewingReconstructionSegment = useReconstructionStore((s) => s.previewingReconstructionSegment);
    const reconstructionPreviewAudioUrl = useReconstructionStore((s) => s.reconstructionPreviewAudioUrl);
    const reconstructionPreviewText = useReconstructionStore((s) => s.reconstructionPreviewText);
    const setReconstructionInstructionDraft = useReconstructionStore((s) => s.setReconstructionInstructionDraft);
    const setReconstructionStudioTab = useReconstructionStore((s) => s.setReconstructionStudioTab);
    const setSelectedReconstructionSpeakerId = useReconstructionStore((s) => s.setSelectedReconstructionSpeakerId);
    const setReconstructionTestTextDrafts = useReconstructionStore((s) => s.setReconstructionTestTextDrafts);
    const setSelectedReconstructionPreviewSegmentId = useReconstructionStore((s) => s.setSelectedReconstructionPreviewSegmentId);
    const savingVoiceFixerSettings = useCleanupStore((s) => s.savingVoiceFixerSettings);
    const selectedReconstructionSpeaker = useMemo(
        () => reconstructionWorkbench?.speakers.find((speaker) => speaker.speaker_id === selectedReconstructionSpeakerId) || null,
        [reconstructionWorkbench, selectedReconstructionSpeakerId]
    );
    const activeEditingClip = editingClipId != null ? (clips.find(c => c.id === editingClipId) || null) : null;
    const showClipEditorMain = activeTab === 'clips' && !!activeEditingClip && !!clipEditorDraft;
    const mediaSourceType = String(video?.media_source_type || 'youtube').toLowerCase();
    const isYoutubeMedia = mediaSourceType === 'youtube';
    const isUploadedMedia = mediaSourceType === 'upload';
    const isTikTokMedia = mediaSourceType === 'tiktok';
    const isLocallyHostedMedia = isUploadedMedia || isTikTokMedia;
    const isUploadedAudio = isLocallyHostedMedia && String(video?.media_kind || '').toLowerCase() === 'audio';
    const canShowCloneTab = !!video?.channel_id;
    const canShowYoutubeTab = isYoutubeMedia || isUploadedMedia;
    const aiMetadataTabLabel = isYoutubeMedia ? 'YouTube' : 'Summary';
    const aiMetadataTabTitle = isYoutubeMedia
        ? 'AI-generated YouTube summary and chapters'
        : 'AI-generated episode summary and chapter index';
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
    const localMediaPending = isTikTokMedia && ['pending', 'queued'].includes(String(video?.status || '').toLowerCase());
    const voiceFixerStatus = String(video?.voicefixer_status || '').toLowerCase();
    const voiceFixerBusy = isUploadedMedia && (voiceFixerStatus === 'queued' || voiceFixerStatus === 'processing');
    const voiceFixerPaused = isUploadedMedia && voiceFixerStatus === 'paused';
    const hasVoiceFixerCleaned = isUploadedMedia && !!video?.voicefixer_cleaned_path;
    const voiceFixerApplyScope = String(video?.voicefixer_apply_scope || (video?.voicefixer_use_cleaned ? 'both' : 'none')).toLowerCase();
    const usingVoiceFixerForPlayback = isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'playback');
    const usingVoiceFixerForProcessing = isUploadedMedia && (voiceFixerApplyScope === 'both' || voiceFixerApplyScope === 'processing');
    const voiceFixerStatusTone = voiceFixerStatus === 'failed'
        ? 'border-red-200 bg-red-50 text-red-700'
        : voiceFixerPaused
            ? 'border-amber-200 bg-amber-50 text-amber-700'
            : (usingVoiceFixerForPlayback || usingVoiceFixerForProcessing)
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
    const reconstructionStatus = String(video?.reconstruction_status || '').toLowerCase();
    const reconstructionBusy = isUploadedMedia && (reconstructionStatus === 'queued' || reconstructionStatus === 'processing');
    const reconstructionPaused = isUploadedMedia && reconstructionStatus === 'paused';
    const hasReconstructionAudio = isUploadedMedia && !!video?.reconstruction_audio_path;
    const usingReconstructionForPlayback = isUploadedMedia && !!video?.reconstruction_use_for_playback;
    const currentUploadedPlaybackSource: UploadedPlaybackSource = usingReconstructionForPlayback
        ? 'reconstructed'
        : usingVoiceFixerForPlayback
            ? 'cleaned'
            : 'original';
    const reconstructionAudioUrl = useMemo(() => {
        if (!video || !isUploadedMedia || !video.reconstruction_audio_path) return '';
        const params = new URLSearchParams({
            path: String(video.reconstruction_audio_path),
            status: reconstructionStatus || 'ready',
        });
        return `${toApiUrl(`/videos/${video.id}/reconstruction/audio`)}?${params.toString()}`;
    }, [isUploadedMedia, reconstructionStatus, video]);
    const resolveWorkbenchAudioUrl = (url?: string) => {
        if (!url) return '';
        return /^https?:\/\//i.test(url) ? url : toApiUrl(url);
    };
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


    const fetchTranscriptQuality = async (videoId: number, signal?: AbortSignal) => {
        setLoadingTranscriptQuality(true);
        setTranscriptQualityError(null);
        try {
            const res = await api.get<TranscriptQuality>(`/videos/${videoId}/transcript-quality`, { signal });
            if (signal?.aborted) return;
            setTranscriptQuality(res.data);
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch transcript quality:', e);
            setTranscriptQuality(null);
            setTranscriptQualityError(e?.response?.data?.detail || 'Failed to evaluate transcript quality');
        } finally {
            if (!signal?.aborted) {
                setLoadingTranscriptQuality(false);
            }
        }
    };

    const fetchTranscriptRollbackOptions = async (videoId: number, signal?: AbortSignal) => {
        setLoadingTranscriptRollbackOptions(true);
        try {
            const res = await api.get<TranscriptRollbackOption[]>(`/videos/${videoId}/transcript-rollback-options`, { signal });
            if (signal?.aborted) return;
            setTranscriptRollbackOptions(res.data || []);
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch transcript rollback options:', e);
            setTranscriptRollbackOptions([]);
        } finally {
            if (!signal?.aborted) {
                setLoadingTranscriptRollbackOptions(false);
            }
        }
    };

    const fetchTranscriptGoldWindows = async (videoId: number, signal?: AbortSignal) => {
        setLoadingTranscriptGoldWindows(true);
        setTranscriptGoldWindowsError(null);
        try {
            const res = await api.get<TranscriptGoldWindow[]>(`/videos/${videoId}/transcript-gold-windows`, { signal });
            if (signal?.aborted) return;
            setTranscriptGoldWindows(res.data || []);
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch transcript gold windows:', e);
            setTranscriptGoldWindows([]);
            setTranscriptGoldWindowsError(e?.response?.data?.detail || 'Failed to load transcript benchmark windows');
        } finally {
            if (!signal?.aborted) {
                setLoadingTranscriptGoldWindows(false);
            }
        }
    };

    const fetchTranscriptEvaluationResults = async (videoId: number, signal?: AbortSignal) => {
        setLoadingTranscriptEvaluationResults(true);
        setTranscriptEvaluationError(null);
        try {
            const res = await api.get<TranscriptEvaluationResult[]>(`/videos/${videoId}/transcript-evaluation-results`, { signal });
            if (signal?.aborted) return;
            setTranscriptEvaluationResults(res.data || []);
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch transcript evaluation results:', e);
            setTranscriptEvaluationResults([]);
            setTranscriptEvaluationError(e?.response?.data?.detail || 'Failed to load transcript evaluation results');
        } finally {
            if (!signal?.aborted) {
                setLoadingTranscriptEvaluationResults(false);
            }
        }
    };

    const fetchEvaluationReviews = async (resultId: number) => {
        try {
            const res = await api.get<TranscriptEvaluationReview[]>(`/transcript-evaluation-results/${resultId}/reviews`);
            setEvaluationReviewsByResultId((current) => ({ ...current, [resultId]: res.data || [] }));
        } catch (e: any) {
            console.error('Failed to fetch transcript evaluation reviews:', e);
        }
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
        useCloneStore.getState().resetCloneState();
        useYoutubeStore.getState().resetYoutubeState();
        useSpeakersTabStore.getState().resetSpeakersTabState();
        useClipsStore.getState().resetClipsState();
        useWorkbenchStore.getState().resetWorkbenchState();
        useCleanupStore.getState().resetCleanupState();
        useTranscriptStore.getState().resetTranscriptState();
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
        if (!id || segments.length === 0) {
            setTranscriptQuality(null);
            setTranscriptQualityError(null);
            setLoadingTranscriptQuality(false);
            return;
        }
        const controller = new AbortController();
        void fetchTranscriptQuality(Number(id), controller.signal);
        return () => controller.abort();
    }, [id, segments.length]);

    useEffect(() => {
        if (activeTab !== 'optimize' || !id || segments.length === 0) {
            return;
        }
        const controller = new AbortController();
        void fetchTranscriptRollbackOptions(Number(id), controller.signal);
        void fetchTranscriptGoldWindows(Number(id), controller.signal);
        void fetchTranscriptEvaluationResults(Number(id), controller.signal);
        return () => controller.abort();
    }, [activeTab, id, segments.length]);

    useEffect(() => {
        if (!id || (!voiceFixerBusy && !reconstructionBusy)) return;
        const timer = window.setInterval(() => {
            void fetchVideoMeta();
        }, 4000);
        return () => window.clearInterval(timer);
    }, [id, reconstructionBusy, voiceFixerBusy]);

    useEffect(() => {
        transcriptEvaluationResults.slice(0, 8).forEach((item) => {
            if (!evaluationReviewsByResultId[item.id]) {
                void fetchEvaluationReviews(item.id);
            }
        });
    }, [transcriptEvaluationResults, evaluationReviewsByResultId]);

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
        initialJumpDoneRef.current = false;
        initialSeekDoneRef.current = false;
        setSearchQuery('');
        setSearchMatchIndex(0);
        setDeepLinkedSegmentId(null);
        setExpandedFunnySummaryIds(new Set());
        useReconstructionStore.getState().resetReconstructionState();
        previousLocalMediaUrlRef.current = '';
        pendingNativeSourceRestoreRef.current = null;
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

    const voiceFixerJob = latestAuxiliaryJobByType('voicefixer_cleanup');
    const reconstructionJob = latestAuxiliaryJobByType('conversation_reconstruct');

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

    const publishCurrentTime = (mediaTime: number, options?: { force?: boolean }) => {
        if (!Number.isFinite(mediaTime)) return;
        const now = performance.now();
        const force = !!options?.force;
        const lastUpdateMs = lastCurrentTimeUiUpdateMsRef.current;
        const lastPublished = lastPublishedCurrentTimeRef.current;

        if (!force) {
            if ((now - lastUpdateMs) < PLAYER_UI_UPDATE_INTERVAL_MS) return;
            if (Math.abs(mediaTime - lastPublished) < PLAYER_UI_MIN_DELTA_SECONDS) return;
        }

        lastCurrentTimeUiUpdateMsRef.current = now;
        lastPublishedCurrentTimeRef.current = mediaTime;
        startTransition(() => {
            setCurrentTime(prev => (Math.abs(prev - mediaTime) >= 0.01 ? mediaTime : prev));
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
            publishCurrentTime(mediaTime);
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
            } catch { }
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

    const tParam = searchParams.get('t');
    const requestedTabParam = String(searchParams.get('tab') || '').trim().toLowerCase();
    const segmentParam = searchParams.get('segment_id');
    const searchQueryParam = searchParams.get('q');
    const searchModeParam = String(searchParams.get('search_mode') || '').trim().toLowerCase();
    const requestedJumpTime = tParam ? Number(tParam) : NaN;
    const requestedSegmentId = segmentParam ? Number(segmentParam) : NaN;
    const requestedSearchQuery = String(searchQueryParam || '').trim();
    const requestedSearchMode = searchModeParam === 'exact' ? 'exact' : '';

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
        if (initialJumpDoneRef.current) return;
        if (segments.length === 0) return;
        const hasRequestedSegment = Number.isInteger(requestedSegmentId) && requestedSegmentId > 0;
        const hasRequestedJumpTime = Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0;
        if (!hasRequestedSegment && !hasRequestedJumpTime) return;

        initialJumpDoneRef.current = true;
        const timer = window.setTimeout(() => {
            try {
                if (hasRequestedSegment && segments.some((seg) => seg.id === requestedSegmentId)) {
                    scrollTranscriptToSegment(requestedSegmentId);
                } else if (hasRequestedJumpTime) {
                    scrollTranscriptToTime(requestedJumpTime);
                }
            } catch (e) {
                console.warn('Initial transcript scroll failed', e);
            }
        }, 80);

        return () => window.clearTimeout(timer);
    }, [requestedJumpTime, requestedSegmentId, segments]);

    const onPlayerReady = (event: any) => {
        setPlayer(event.target);
        samplePlayerClock(event.target);
        setPlaybackRate(Number(event?.target?.getPlaybackRate?.()) || 1);
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
        publishCurrentTime(mediaTime, { force: true });
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
        setPlaybackRate(rate);
        publishCurrentTime(mediaTime, { force: true });
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
            publishCurrentTime(time, { force: true });
        } catch (e) {
            console.warn('Failed to seek video player', e);
        }
    };

    useEffect(() => {
        if (isLocallyHostedMedia && activeTab === 'youtube') {
            setActiveTab('transcript');
        }
        if (!isUploadedMedia && (activeTab === 'cleanup' || activeTab === 'reconstruction')) {
            setActiveTab('transcript');
        }
    }, [activeTab, isLocallyHostedMedia, isUploadedMedia]);

    const buildNativePlayerAdapter = (element: HTMLMediaElement): UnifiedPlayer => ({
        getCurrentTime: () => Number(element.currentTime || 0),
        seekTo: (seconds: number) => {
            element.currentTime = Math.max(0, Number(seconds || 0));
        },
        playVideo: () => {
            void element.play().catch(() => { });
        },
        pauseVideo: () => {
            element.pause();
        },
        getPlaybackRate: () => Number(element.playbackRate || 1),
        setPlaybackRate: (rate: number) => {
            element.playbackRate = rate;
        },
        getPlayerState: () => {
            if (element.ended) return 0;
            return element.paused ? 2 : 1;
        },
    });

    const syncNativePlayerClock = (element: HTMLMediaElement) => {
        const nextRate = Number(element.playbackRate || 1);
        const nextState = element.ended ? 0 : (element.paused ? 2 : 1);
        const mediaTime = Number(element.currentTime || 0);
        playerClockRef.current = {
            mediaTime,
            wallTimeMs: performance.now(),
            playbackRate: nextRate,
            playerState: nextState,
        };
        setPlaybackRate(nextRate);
        publishCurrentTime(mediaTime, { force: true });
    };

    const handleNativeMediaReady = (element: HTMLMediaElement) => {
        nativeMediaRef.current = element;
        const adapter = buildNativePlayerAdapter(element);
        setPlayer(adapter);
        syncNativePlayerClock(element);
        const pendingRestore = pendingNativeSourceRestoreRef.current;
        if (pendingRestore) {
            pendingNativeSourceRestoreRef.current = null;
            window.setTimeout(() => {
                try {
                    element.currentTime = Math.max(0, pendingRestore.currentTime);
                    element.playbackRate = pendingRestore.playbackRate;
                    syncNativePlayerClock(element);
                    if (pendingRestore.wasPlaying) {
                        const playAttempt = element.play();
                        if (playAttempt && typeof (playAttempt as Promise<void>).catch === 'function') {
                            void (playAttempt as Promise<void>).catch(() => { });
                        }
                    }
                } catch (e) {
                    console.warn('Playback source restore failed', e);
                }
            }, 120);
            return;
        }
        if (!initialSeekDoneRef.current && Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0) {
            initialSeekDoneRef.current = true;
            window.setTimeout(() => {
                try {
                    element.currentTime = Math.max(0, requestedJumpTime);
                    element.pause();
                    syncNativePlayerClock(element);
                } catch (e) {
                    console.warn('Initial timestamp seek failed', e);
                    initialSeekDoneRef.current = false;
                }
            }, 120);
        }
    };

    const handlePlayerRateChange = (rate: number) => {
        const nextRate = Number(rate);
        if (!Number.isFinite(nextRate) || nextRate <= 0 || !player) return;
        try {
            player.setPlaybackRate?.(nextRate);
            if (nativeMediaRef.current) {
                nativeMediaRef.current.playbackRate = nextRate;
                syncNativePlayerClock(nativeMediaRef.current);
            }
            setPlaybackRate(nextRate);
        } catch (e) {
            console.warn('Failed to change playback rate', e);
        }
    };


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
    const localMediaUrl = useMemo(() => {
        if (!video || !isLocallyHostedMedia) return '';
        const params = new URLSearchParams({
            reconstruction_variant: usingReconstructionForPlayback ? 'reconstructed' : 'source',
            reconstruction_enabled: usingReconstructionForPlayback ? 'true' : 'false',
            reconstruction_status: reconstructionStatus || 'none',
            media_variant: usingVoiceFixerForPlayback ? 'clean' : 'original',
            media_scope: voiceFixerApplyScope,
            media_status: String(video.voicefixer_status || 'none'),
        });
        return `${toApiUrl(`/videos/${video.id}/media`)}?${params.toString()}`;
    }, [isLocallyHostedMedia, reconstructionStatus, usingReconstructionForPlayback, usingVoiceFixerForPlayback, video, voiceFixerApplyScope]);

    useEffect(() => {
        if (!isLocallyHostedMedia) {
            previousLocalMediaUrlRef.current = '';
            pendingNativeSourceRestoreRef.current = null;
            return;
        }
        if (!localMediaUrl) return;
        if (previousLocalMediaUrlRef.current && previousLocalMediaUrlRef.current !== localMediaUrl) {
            const element = nativeMediaRef.current;
            pendingNativeSourceRestoreRef.current = {
                currentTime: Math.max(0, Number(playerClockRef.current.mediaTime || 0)),
                wasPlaying: !!element && !element.paused && !element.ended,
                playbackRate: Math.max(0.75, Number(playerClockRef.current.playbackRate || 1)),
            };
        }
        previousLocalMediaUrlRef.current = localMediaUrl;
    }, [isLocallyHostedMedia, localMediaUrl]);

    const TRANSCRIPT_HIGHLIGHT_LEAD_SECONDS = 0.08;
    const TRANSCRIPT_SEGMENT_TRAIL_SECONDS = 0.05;
    const TRANSCRIPT_WORD_GAP_BRIDGE_SECONDS = 0.85;
    const TRANSCRIPT_MIN_WORD_HIGHLIGHT_SECONDS = 0.12;

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
            if (displayEnd <= w.start) {
                displayEnd = w.start + 0.01;
            }
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

    const transcriptPlaybackTime = currentTime + TRANSCRIPT_HIGHLIGHT_LEAD_SECONDS;

    const scrollTranscriptToSegment = (segmentId: number) => {
        if (!segmentId) return;

        const doScroll = () => {
            const el = document.getElementById(`seg-${segmentId}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };

        if (activeTab !== 'transcript') {
            setActiveTab('transcript');
            window.setTimeout(doScroll, 80);
        } else {
            doScroll();
        }
    };

    const scrollTranscriptToTime = (time: number) => {
        if (segments.length === 0) return;

        const target =
            segments.find(s => time >= s.start_time && time < s.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS) ||
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
            const activeSeg = segments.find(
                s => transcriptPlaybackTime >= s.start_time && transcriptPlaybackTime < s.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS,
            );
            if (activeSeg) {
                if (lastAutoScrollSegIdRef.current !== activeSeg.id) {
                    lastAutoScrollSegIdRef.current = activeSeg.id;
                    const activeEl = document.getElementById(`seg-${activeSeg.id}`);
                    if (activeEl) activeEl.scrollIntoView({ behavior: 'auto', block: 'center' });
                }
            }
        }
    }, [transcriptPlaybackTime, activeTab, followPlayback, segments, selection, searchQuery, editingSegmentId]);

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
            parts.push(<mark key={i} className="bg-yellow-200 text-yellow-900 rounded-sm">{text.slice(i, i + searchLower.length)}</mark>);
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
        const createdClip = await useClipsStore.getState().createClip(video.id, {
            start: selection.start,
            end: selection.end,
            title: clipTitle || selection.defaultTitle,
        });
        if (createdClip) {
            setActiveTab('clips');
            setSelection(null);
        }
    };

    const fetchClips = async () => {
        if (!video) return;
        await useClipsStore.getState().fetchClips(video.id);
    };

    const fetchClipExportArtifacts = async () => {
        if (!video) return;
        await useClipsStore.getState().fetchClipExportArtifacts(video.id);
    };

    const handleSpeakerClick = async (speakerId: number, segment?: TranscriptSegment) => {
        pauseMainPreview();
        await openSpeaker(speakerId, segment, video);
    };

    const transcriptPipelineStatuses = ['queued', 'downloading', 'transcribing', 'diarizing'];
    const transcriptJobActive = transcriptPipelineStatuses.includes(String(video?.status || '').toLowerCase());
    const episodeBusy =
        purging ||
        redoing ||
        redoingDiarization ||
        consolidatingTranscript ||
        queueingTranscriptRepair ||
        queueingDiarizationRebuild ||
        queueingDiarizationBenchmark ||
        queueingFullRetranscription ||
        restoringTranscriptRunId !== null ||
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
        if (!video) return;
        const startTime = Number(goldWindowStartDraft);
        const endTime = Number(goldWindowEndDraft);
        if (!Number.isFinite(startTime) || !Number.isFinite(endTime) || endTime <= startTime) {
            alert('Set a valid gold window start/end range.');
            return;
        }
        const referenceText = goldWindowReferenceDraft.trim();
        if (!referenceText) {
            alert('Reference transcript text is required for a gold window.');
            return;
        }
        setSavingTranscriptGoldWindow(true);
        try {
            await api.post<TranscriptGoldWindow>(`/videos/${video.id}/transcript-gold-windows`, {
                label: goldWindowLabelDraft.trim() || 'Gold Window',
                quality_profile: transcriptQuality?.quality_profile || null,
                language: video.transcript_language || transcriptQuality?.language || null,
                start_time: startTime,
                end_time: endTime,
                reference_text: referenceText,
                entities: goldWindowEntitiesDraft
                    .split(',')
                    .map((item) => item.trim())
                    .filter(Boolean),
                notes: goldWindowNotesDraft.trim() || null,
                active: true,
            });
            await fetchTranscriptGoldWindows(video.id);
            setGoldWindowLabelDraft('Gold Window');
            setGoldWindowStartDraft('');
            setGoldWindowEndDraft('');
            setGoldWindowReferenceDraft('');
            setGoldWindowEntitiesDraft('');
            setGoldWindowNotesDraft('');
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save transcript gold window');
        } finally {
            setSavingTranscriptGoldWindow(false);
        }
    };

    const runTranscriptEvaluation = async () => {
        if (!video) return;
        setEvaluatingTranscript(true);
        setTranscriptEvaluationError(null);
        try {
            const res = await api.post<TranscriptEvaluationBatchResponse>(`/videos/${video.id}/transcript-evaluation`);
            setTranscriptEvaluationSummary(res.data);
            setTranscriptEvaluationResults(res.data.items || []);
            for (const item of res.data.items || []) {
                void fetchEvaluationReviews(item.id);
            }
        } catch (e: any) {
            console.error('Failed to evaluate transcript against gold windows:', e);
            setTranscriptEvaluationSummary(null);
            setTranscriptEvaluationError(e?.response?.data?.detail || 'Failed to evaluate transcript');
            alert(e?.response?.data?.detail || 'Failed to evaluate transcript');
        } finally {
            setEvaluatingTranscript(false);
        }
    };

    const submitTranscriptEvaluationReview = async (resultId: number) => {
        const verdict = String(evaluationReviewVerdictDrafts[resultId] || 'same');
        setReviewingEvaluationResultId(resultId);
        try {
            await api.post<TranscriptEvaluationReview>(`/transcript-evaluation-results/${resultId}/review`, {
                reviewer: (evaluationReviewReviewerDrafts[resultId] || '').trim() || null,
                verdict,
                tags: [],
                notes: (evaluationReviewNotesDrafts[resultId] || '').trim() || null,
            });
            await fetchEvaluationReviews(resultId);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save transcript evaluation review');
        } finally {
            setReviewingEvaluationResultId(null);
        }
    };

    const queueTranscriptRepairJob = async () => {
        if (!video) return;
        if (!confirm('Queue the low-risk transcript repair pass for this episode?')) return;
        setQueueingTranscriptRepair(true);
        try {
            const res = await api.post<TranscriptRepairQueueResponse>(`/videos/${video.id}/transcript-repair`, {});
            alert(`Low-risk repair queued as job ${res.data.job_id}.`);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue transcript repair');
        } finally {
            setQueueingTranscriptRepair(false);
        }
    };

    const queueTranscriptDiarizationRebuildJob = async () => {
        if (!video) return;
        if (!confirm('Queue a diarization rebuild for this episode? This will reuse the raw transcript but replace speaker segmentation and assignments.')) return;
        setQueueingDiarizationRebuild(true);
        try {
            const res = await api.post<TranscriptDiarizationRebuildQueueResponse>(`/videos/${video.id}/transcript-diarization-rebuild`, {});
            alert(`Diarization rebuild queued as job ${res.data.job_id}.`);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue diarization rebuild');
        } finally {
            setQueueingDiarizationRebuild(false);
        }
    };

    const queueTranscriptDiarizationBenchmarkJob = async () => {
        if (!video) return;
        const threshold = Number(diarizationBenchmarkThreshold);
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
            alert('Set a valid speaker match threshold between 0.0 and 1.0.');
            return;
        }
        if (!confirm(`Queue a diarization benchmark run using ${diarizationBenchmarkSensitivity} sensitivity and threshold ${threshold.toFixed(2)}?`)) return;
        setQueueingDiarizationBenchmark(true);
        try {
            const res = await api.post<TranscriptDiarizationRebuildQueueResponse>(`/videos/${video.id}/transcript-diarization-benchmark`, {
                force: true,
                diarization_sensitivity: diarizationBenchmarkSensitivity,
                speaker_match_threshold: threshold,
            });
            alert(`Diarization benchmark queued as job ${res.data.job_id}.`);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue diarization benchmark');
        } finally {
            setQueueingDiarizationBenchmark(false);
        }
    };

    const queueTranscriptRetranscriptionJob = async () => {
        if (!video) return;
        if (!confirm('Queue a full retranscription for this episode? This will force a fresh transcription pass before diarization.')) return;
        setQueueingFullRetranscription(true);
        try {
            const res = await api.post<TranscriptRetranscriptionQueueResponse>(`/videos/${video.id}/transcript-retranscribe`, {});
            alert(`Full retranscription queued as job ${res.data.job_id}.`);
            fetchData();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue full retranscription');
        } finally {
            setQueueingFullRetranscription(false);
        }
    };

    const restoreTranscriptFromRun = async (runId: number) => {
        if (!video) return;
        if (!confirm('Restore the transcript from this saved optimization run? The current transcript will be backed up first.')) return;
        setRestoringTranscriptRunId(runId);
        try {
            const res = await api.post<TranscriptRestoreResponse>(`/videos/${video.id}/transcript-runs/${runId}/restore`);
            await fetchData();
            await fetchTranscriptQuality(video.id);
            await fetchTranscriptRollbackOptions(video.id);
            alert(`Transcript restored from run ${res.data.restored_from_run_id}. New restore run ${res.data.restore_run_id} recorded.`);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to restore transcript run');
        } finally {
            setRestoringTranscriptRunId(null);
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

    const handleSaveReconstructionSettings = async () => {
        await useReconstructionStore.getState().saveReconstructionSettings(video, isUploadedMedia, setVideo);
    };

    const handleTestReconstructionSpeaker = async (
        speakerId: number,
        segmentId?: number,
        options?: { performanceMode?: boolean; useSelectedSampleText?: boolean }
    ) => {
        await useReconstructionStore.getState().testReconstructionSpeaker(video, isUploadedMedia, speakerId, segmentId, options);
    };

    const handleSetReconstructionPlayback = async (enabled: boolean) => {
        await handleSetUploadedPlaybackSource(enabled ? 'reconstructed' : 'original');
    };

    const handleCleanupReconstructionSample = async (speakerId: number, segmentId: number) => {
        await useReconstructionStore.getState().cleanupReconstructionSample(video, isUploadedMedia, speakerId, segmentId);
    };

    const handleUpdateReconstructionSampleState = async (
        speakerId: number,
        segmentId: number,
        patch: { rejected?: boolean; selected?: boolean; clear_cleaned?: boolean }
    ) => {
        await useReconstructionStore.getState().updateReconstructionSampleState(video, isUploadedMedia, speakerId, segmentId, patch);
    };

    const handleAddReconstructionSample = async (speakerId: number) => {
        await useReconstructionStore.getState().addReconstructionSample(video, isUploadedMedia, speakerId);
    };

    const handleApproveReconstructionSpeaker = async (speakerId: number, approved: boolean) => {
        await useReconstructionStore.getState().approveReconstructionSpeaker(video, isUploadedMedia, speakerId, approved);
    };

    const handlePreviewReconstructionSegment = async (segmentId: number) => {
        await useReconstructionStore.getState().previewReconstructionSegment(video, isUploadedMedia, segmentId, resolveWorkbenchAudioUrl);
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

    const handleUnknownSpeakerClick = async (segmentId: number, e: React.MouseEvent) => {
        e.stopPropagation();
        pauseMainPreview();
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        openAssignPopup(segmentId, rect.left + window.scrollX, rect.bottom + window.scrollY);
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
    const renderUploadedPlaybackSourceSwitcher = () => {
        if (!isUploadedMedia) return null;

        const options: Array<{
            id: UploadedPlaybackSource;
            label: string;
            detail: string;
            icon: LucideIcon;
            available: boolean;
            activeClassName: string;
            inactiveClassName: string;
        }> = [
                {
                    id: 'original',
                    label: 'Original',
                    detail: 'Uploaded media',
                    icon: PlayCircle,
                    available: true,
                    activeClassName: 'border-slate-300 bg-slate-900 text-white shadow-sm',
                    inactiveClassName: 'border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50',
                },
                {
                    id: 'cleaned',
                    label: 'Cleanup',
                    detail: hasVoiceFixerCleaned ? 'VoiceFixer pass' : 'Run cleanup first',
                    icon: AudioLines,
                    available: hasVoiceFixerCleaned,
                    activeClassName: 'border-sky-200 bg-sky-600 text-white shadow-sm',
                    inactiveClassName: 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100',
                },
                {
                    id: 'reconstructed',
                    label: 'Rebuild',
                    detail: hasReconstructionAudio ? 'Conversation rebuild' : 'Run reconstruction first',
                    icon: Sparkles,
                    available: hasReconstructionAudio,
                    activeClassName: 'border-violet-200 bg-violet-600 text-white shadow-sm',
                    inactiveClassName: 'border-violet-200 bg-violet-50 text-violet-700 hover:bg-violet-100',
                },
            ];

        return (
            <div className="mt-2 rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs shadow-sm">
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                    <div className="min-w-0">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Playback Source</div>
                        <div className="mt-1 text-slate-700">
                            Switch the player between the original upload and generated variants while you listen.
                        </div>
                        <div className="mt-1 text-[11px] text-slate-500">
                            Processing stays on {usingVoiceFixerForProcessing ? 'the cleaned pass' : 'the original upload'}.
                        </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {options.map((option) => {
                            const Icon = option.icon;
                            const active = currentUploadedPlaybackSource === option.id;
                            return (
                                <button
                                    key={option.id}
                                    type="button"
                                    onClick={() => void handleSetUploadedPlaybackSource(option.id)}
                                    disabled={!option.available || episodeBusy || switchingReconstructionPlayback}
                                    title={!option.available ? option.detail : `Use ${option.label.toLowerCase()} audio for playback`}
                                    className={`inline-flex min-w-[122px] items-center gap-2 rounded-xl border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${active ? option.activeClassName : option.inactiveClassName
                                        }`}
                                >
                                    {switchingReconstructionPlayback && active ? <Loader2 size={14} className="animate-spin" /> : <Icon size={14} />}
                                    <span className="min-w-0">
                                        <span className="block text-[11px] font-semibold uppercase tracking-wide">{option.label}</span>
                                        <span className={`block truncate text-[11px] ${active ? 'text-white/80' : 'text-slate-500'}`}>{option.detail}</span>
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>
        );
    };
    const renderPlaybackRateControl = () => (
        <div className="mt-2">
            <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-sm">
                <span>
                    {isUploadedMedia
                        ? (isUploadedAudio ? 'Uploaded audio' : 'Uploaded video')
                        : isTikTokMedia
                            ? 'TikTok local media'
                            : 'YouTube player'}
                </span>
                <div className="flex items-center gap-2">
                    <span>Speed</span>
                    <select
                        value={String(playbackRate || 1)}
                        onChange={(e) => handlePlayerRateChange(Number(e.target.value))}
                        className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    >
                        {[0.75, 1, 1.25, 1.5, 1.75, 2].map(rate => (
                            <option key={rate} value={rate}>{rate}x</option>
                        ))}
                    </select>
                </div>
            </div>
            {renderUploadedPlaybackSourceSwitcher()}
        </div>
    );

    const renderMainPlayer = (containerClassName: string) => (
        <>
            <div className={containerClassName}>
                {isLocallyHostedMedia ? (
                    localMediaPending ? (
                        <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_top,rgba(236,72,153,.22),transparent_45%),linear-gradient(135deg,#111827,#1f2937)] px-8 text-white">
                            <div className="max-w-lg rounded-2xl border border-white/10 bg-white/10 p-6 text-center backdrop-blur-sm">
                                <div className="text-sm font-semibold">Local TikTok media is not ready yet</div>
                                <div className="mt-2 text-xs leading-relaxed text-white/75">
                                    Playback switches to the native player after the TikTok file has been downloaded locally. Start processing or wait for the download stage to complete.
                                </div>
                            </div>
                        </div>
                    ) : (
                        isUploadedAudio ? (
                            <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_top,rgba(59,130,246,.24),transparent_45%),linear-gradient(135deg,#0f172a,#1e293b)] px-8">
                                <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-white/10 p-6 text-white backdrop-blur-sm">
                                    <div className="mb-4 flex items-center gap-3">
                                        <AudioLines size={18} className="text-blue-200" />
                                        <div>
                                            <div className="text-sm font-semibold">Audio Episode</div>
                                            <div className="text-xs text-blue-100/80">{video?.title}</div>
                                        </div>
                                    </div>
                                    <audio
                                        ref={(element) => {
                                            nativeMediaRef.current = element;
                                        }}
                                        src={localMediaUrl}
                                        controls
                                        preload="metadata"
                                        className="w-full"
                                        onLoadedMetadata={(e) => handleNativeMediaReady(e.currentTarget)}
                                        onTimeUpdate={(e) => syncNativePlayerClock(e.currentTarget)}
                                        onPlay={(e) => syncNativePlayerClock(e.currentTarget)}
                                        onPause={(e) => syncNativePlayerClock(e.currentTarget)}
                                        onRateChange={(e) => syncNativePlayerClock(e.currentTarget)}
                                        onEnded={(e) => syncNativePlayerClock(e.currentTarget)}
                                    />
                                </div>
                            </div>
                        ) : (
                            <video
                                ref={(element) => {
                                    nativeMediaRef.current = element;
                                }}
                                src={localMediaUrl}
                                controls
                                preload="metadata"
                                className="h-full w-full bg-black"
                                onLoadedMetadata={(e) => handleNativeMediaReady(e.currentTarget)}
                                onTimeUpdate={(e) => syncNativePlayerClock(e.currentTarget)}
                                onPlay={(e) => syncNativePlayerClock(e.currentTarget)}
                                onPause={(e) => syncNativePlayerClock(e.currentTarget)}
                                onRateChange={(e) => syncNativePlayerClock(e.currentTarget)}
                                onEnded={(e) => syncNativePlayerClock(e.currentTarget)}
                            />
                        )
                    )
                ) : (
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
                )}
            </div>
            {renderPlaybackRateControl()}
        </>
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

    const renderTranscriptOptimizationSnapshotCard = (variant: 'transcript' | 'optimize' = 'transcript') => (
        <div className={`rounded-2xl border px-4 py-4 shadow-sm ${variant === 'optimize' ? 'border-emerald-200 bg-emerald-50/70' : 'border-slate-200 bg-slate-50/90'}`}>
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        {variant === 'optimize' ? 'Optimization Navigator' : 'Transcript Status'}
                    </div>
                    <div className="mt-1 text-sm font-semibold text-slate-900">
                        {variant === 'optimize' ? 'Transcript Optimization Workbench' : 'Optimization has its own workbench now'}
                    </div>
                    <div className="mt-1 text-xs leading-5 text-slate-600">
                        {variant === 'optimize'
                            ? 'Benchmarking, rollback, repair, rebuild, and retranscription controls live in the main stage. Use the transcript tab for reading and editing only.'
                            : 'Use the Optimize tab for repair, diarization rebuild, retranscription, rollback, and transcript benchmarking.'}
                    </div>
                </div>
                <div className="flex shrink-0 gap-2">
                    {variant === 'optimize' ? (
                        <button
                            type="button"
                            onClick={() => setActiveTab('transcript')}
                            className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/80 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                        >
                            <FileText size={14} />
                            Transcript
                        </button>
                    ) : (
                        <button
                            type="button"
                            onClick={() => setActiveTab('optimize')}
                            className="inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-200 bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700"
                        >
                            <CheckCircle2 size={14} />
                            Open Optimize
                        </button>
                    )}
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
                <div className="mt-3 text-xs leading-5 text-slate-500">
                    {transcriptQuality.reasons[0]}
                </div>
            )}

            {variant === 'optimize' && (
                <div className="mt-3">
                    <button
                        type="button"
                        onClick={() => {
                            if (!id) return;
                            void fetchTranscriptQuality(Number(id));
                        }}
                        disabled={loadingTranscriptQuality}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                    >
                        {loadingTranscriptQuality ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                        Refresh Assessment
                    </button>
                </div>
            )}
        </div>
    );

    const renderTranscriptOptimizationWorkbench = () => {
        const hasTranscript = segments.length > 0;

        return (
            <div className="flex-1 overflow-y-auto p-6">
                <div className="mx-auto max-w-7xl space-y-6">
                    {renderTranscriptOptimizationSnapshotCard('optimize')}

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
                                onClick={() => setActiveTab('transcript')}
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
                                            onClick={() => setActiveTab('transcript')}
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
                                                            <div className="text-sm font-medium text-slate-800">
                                                                Run {option.run_id} · {option.mode.replaceAll('_', ' ')}
                                                            </div>
                                                            <div className="mt-0.5 text-[11px] text-slate-500">
                                                                {new Date(option.created_at).toLocaleString()} · {option.pipeline_version}
                                                            </div>
                                                            {option.note && (
                                                                <div className="mt-1 text-xs text-slate-600">{option.note}</div>
                                                            )}
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
                                                    <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                        Avg WER {transcriptEvaluationSummary.average_wer.toFixed(3)}
                                                    </span>
                                                    <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                        Avg CER {transcriptEvaluationSummary.average_cer.toFixed(3)}
                                                    </span>
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
                                                <input
                                                    value={goldWindowLabelDraft}
                                                    onChange={(e) => setGoldWindowLabelDraft(e.target.value)}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="Window label"
                                                />
                                            </div>
                                            <div>
                                                <label className="mb-1 block text-xs font-medium text-slate-600">Start</label>
                                                <input
                                                    value={goldWindowStartDraft}
                                                    onChange={(e) => setGoldWindowStartDraft(e.target.value)}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="0.00"
                                                />
                                            </div>
                                            <div>
                                                <label className="mb-1 block text-xs font-medium text-slate-600">End</label>
                                                <input
                                                    value={goldWindowEndDraft}
                                                    onChange={(e) => setGoldWindowEndDraft(e.target.value)}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="15.00"
                                                />
                                            </div>
                                            <div className="sm:col-span-2">
                                                <label className="mb-1 block text-xs font-medium text-slate-600">Reference Transcript</label>
                                                <textarea
                                                    value={goldWindowReferenceDraft}
                                                    onChange={(e) => setGoldWindowReferenceDraft(e.target.value)}
                                                    rows={5}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="Paste the hand-corrected reference transcript for this window."
                                                />
                                            </div>
                                            <div className="sm:col-span-2">
                                                <label className="mb-1 block text-xs font-medium text-slate-600">Entities</label>
                                                <input
                                                    value={goldWindowEntitiesDraft}
                                                    onChange={(e) => setGoldWindowEntitiesDraft(e.target.value)}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="Comma-separated entities to track"
                                                />
                                            </div>
                                            <div className="sm:col-span-2">
                                                <label className="mb-1 block text-xs font-medium text-slate-600">Notes</label>
                                                <textarea
                                                    value={goldWindowNotesDraft}
                                                    onChange={(e) => setGoldWindowNotesDraft(e.target.value)}
                                                    rows={2}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                    placeholder="Speaker boundaries, entity focus, overlap risk, punctuation notes..."
                                                />
                                            </div>
                                        </div>
                                        <button
                                            onClick={createTranscriptGoldWindow}
                                            disabled={savingTranscriptGoldWindow}
                                            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                                        >
                                            {savingTranscriptGoldWindow ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                                            Save Gold Window
                                        </button>
                                        {loadingTranscriptGoldWindows ? (
                                            <div className="inline-flex items-center gap-2 text-xs text-slate-500">
                                                <Loader2 size={14} className="animate-spin" />
                                                Loading benchmark windows...
                                            </div>
                                        ) : transcriptGoldWindowsError ? (
                                            <div className="text-xs text-rose-600">{transcriptGoldWindowsError}</div>
                                        ) : transcriptGoldWindows.length === 0 ? (
                                            <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">
                                                No gold windows yet. Create one from a selected range or enter a benchmark window manually.
                                            </div>
                                        ) : (
                                            <div className="space-y-2">
                                                {transcriptGoldWindows.map((window) => (
                                                    <div key={window.id} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                                        <div className="flex items-start justify-between gap-2">
                                                            <div className="min-w-0">
                                                                <div className="text-sm font-medium text-slate-800">{window.label}</div>
                                                                <div className="mt-0.5 text-xs text-slate-500">
                                                                    {window.start_time.toFixed(2)}s to {window.end_time.toFixed(2)}s
                                                                    {window.language ? ` • ${window.language}` : ''}
                                                                </div>
                                                            </div>
                                                            <button
                                                                onClick={() => handleSeek(window.start_time)}
                                                                className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
                                                            >
                                                                Jump
                                                            </button>
                                                        </div>
                                                        <div className="mt-2 line-clamp-3 text-xs leading-5 text-slate-600">
                                                            {window.reference_text}
                                                        </div>
                                                        {window.entities.length > 0 && (
                                                            <div className="mt-2 flex flex-wrap gap-1">
                                                                {window.entities.map((entity) => (
                                                                    <span key={`${window.id}-${entity}`} className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                        {entity}
                                                                    </span>
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
                                                <div className="text-xs text-slate-500">
                                                    WER and CER come from the stored reference windows. Reviewer verdicts capture the human judgment layer.
                                                </div>
                                            </div>
                                        </div>
                                        {loadingTranscriptEvaluationResults ? (
                                            <div className="inline-flex items-center gap-2 text-xs text-slate-500">
                                                <Loader2 size={14} className="animate-spin" />
                                                Loading evaluation results...
                                            </div>
                                        ) : transcriptEvaluationError ? (
                                            <div className="text-xs text-rose-600">{transcriptEvaluationError}</div>
                                        ) : transcriptEvaluationResults.length === 0 ? (
                                            <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">
                                                No evaluation results yet. Run evaluation after defining at least one gold window.
                                            </div>
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
                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-700 ring-1 ring-slate-200">
                                                                    WER {result.wer.toFixed(3)}
                                                                </span>
                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                    CER {result.cer.toFixed(3)}
                                                                </span>
                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                    Unknown {result.unknown_speaker_rate.toFixed(2)}
                                                                </span>
                                                                {result.entity_accuracy != null && (
                                                                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                        Entity {result.entity_accuracy.toFixed(2)}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <div className="mt-3 grid gap-3 lg:grid-cols-2">
                                                                <div>
                                                                    <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Reference</div>
                                                                    <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                                                                        {result.reference_text}
                                                                    </div>
                                                                </div>
                                                                <div>
                                                                    <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Candidate</div>
                                                                    <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                                                                        {result.candidate_text}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div className="mt-3 grid gap-2 lg:grid-cols-[140px_140px_minmax(0,1fr)_auto]">
                                                                <input
                                                                    value={reviewer}
                                                                    onChange={(e) => setEvaluationReviewReviewerDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                    placeholder="Reviewer"
                                                                />
                                                                <select
                                                                    value={verdict}
                                                                    onChange={(e) => setEvaluationReviewVerdictDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                >
                                                                    <option value="better">Better</option>
                                                                    <option value="same">Same</option>
                                                                    <option value="worse">Worse</option>
                                                                    <option value="bad_merge">Bad merge</option>
                                                                    <option value="bad_speaker_reassignment">Bad speaker reassignment</option>
                                                                    <option value="bad_entity_repair">Bad entity repair</option>
                                                                    <option value="language_regression">Language regression</option>
                                                                </select>
                                                                <input
                                                                    value={reviewNotes}
                                                                    onChange={(e) => setEvaluationReviewNotesDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                    placeholder="Review notes"
                                                                />
                                                                <button
                                                                    onClick={() => submitTranscriptEvaluationReview(result.id)}
                                                                    disabled={reviewingEvaluationResultId === result.id}
                                                                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                                                                >
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
    };

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
                                    {isPlaceholderTranscript && (
                                        <div className="mt-2 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                                            <Clock size={14} className="mt-0.5 shrink-0 text-amber-600" />
                                            <div>
                                                <div className="font-semibold">
                                                    {placeholderTranscriptSourceLabel}
                                                    {placeholderTranscriptLanguage ? ` (${placeholderTranscriptLanguage})` : ''}
                                                </div>
                                                <div className="mt-0.5 text-amber-800/90">
                                                    This searchable transcript is a temporary placeholder and will be replaced automatically after local transcription and diarization finish.
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    {segments.length > 0 && (
                                        <div className="mt-3">
                                            {renderTranscriptOptimizationSnapshotCard('transcript')}
                                        </div>
                                    )}
                                    {false && segments.length > 0 && (
                                        <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50/90 px-3 py-3">
                                            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                                <div className="min-w-0">
                                                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Transcript Optimization</div>
                                                    {loadingTranscriptQuality ? (
                                                        <div className="mt-2 inline-flex items-center gap-2 text-sm text-slate-500">
                                                            <Loader2 size={14} className="animate-spin" />
                                                            Evaluating transcript quality...
                                                        </div>
                                                    ) : transcriptQuality ? (
                                                        <>
                                                            <div className="mt-1 flex flex-wrap items-center gap-2">
                                                                <span className="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                                                    {recommendedOptimizationLabel}
                                                                </span>
                                                                <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                                                                    Score {transcriptQuality?.quality_score?.toFixed(1) ?? '0.0'}
                                                                </span>
                                                                <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                                                                    {String(transcriptQuality?.quality_profile || 'unknown').replaceAll('_', ' ')}
                                                                </span>
                                                            </div>
                                                            <div className="mt-2 grid gap-2 text-xs text-slate-600 sm:grid-cols-3">
                                                                <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                                                                    Unknown speaker rate: {Number(transcriptQuality?.metrics?.unknown_speaker_rate || 0).toFixed(2)}
                                                                </div>
                                                                <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                                                                    Micro segments: {Number(transcriptQuality?.metrics?.micro_segment_count || 0)}
                                                                </div>
                                                                <div className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
                                                                    Interruptions: {Number(transcriptQuality?.metrics?.same_speaker_interruptions || 0)}
                                                                </div>
                                                            </div>
                                                            {Boolean(transcriptQuality?.reasons?.length) && (
                                                                <div className="mt-2 text-xs leading-5 text-slate-500">
                                                                    {transcriptQuality?.reasons?.[0]}
                                                                </div>
                                                            )}
                                                        </>
                                                    ) : (
                                                        <div className="mt-2 text-sm text-slate-500">
                                                            {transcriptQualityError || 'Transcript quality has not been evaluated yet.'}
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                                                    <button
                                                        onClick={() => {
                                                            if (!id) return;
                                                            void fetchTranscriptQuality(Number(id));
                                                        }}
                                                        disabled={loadingTranscriptQuality}
                                                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                                    >
                                                        {loadingTranscriptQuality ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                                        Refresh Assessment
                                                    </button>
                                                    <div className="group relative">
                                                        <button
                                                            onClick={queueTranscriptRepairJob}
                                                            disabled={episodeBusy || recommendedOptimizationTier !== 'low_risk_repair'}
                                                            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                                                        >
                                                            {queueingTranscriptRepair ? <Loader2 size={14} className="animate-spin" /> : <GitMerge size={14} />}
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
                                                            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                                                        >
                                                            {queueingDiarizationRebuild ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
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
                                                            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-amber-600 px-3 py-2 text-xs font-medium text-white hover:bg-amber-700 disabled:opacity-50"
                                                        >
                                                            {queueingFullRetranscription ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                                                            Queue Retranscribe
                                                            <CircleHelp size={13} className="opacity-80" />
                                                        </button>
                                                        <div className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-72 -translate-x-1/2 rounded-lg border border-slate-200 bg-slate-900 px-3 py-2 text-[11px] leading-5 text-white shadow-xl group-hover:block">
                                                            {transcriptOptimizationHelp.retranscribe}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(260px,0.9fr)]">
                                                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3">
                                                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Diarization Benchmark</div>
                                                    <div className="mt-1 text-xs leading-5 text-slate-500">
                                                        Queue a one-off benchmark variant with explicit diarization sensitivity and speaker-match threshold so you can compare results in the benchmark dashboard.
                                                    </div>
                                                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
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
                                                        className="mt-3 inline-flex items-center justify-center gap-2 rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                                                    >
                                                        {queueingDiarizationBenchmark ? <Loader2 size={14} className="animate-spin" /> : <AudioLines size={14} />}
                                                        Queue Benchmark Variant
                                                    </button>
                                                </div>
                                                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3">
                                                    <div className="flex items-center justify-between gap-2">
                                                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Rollback</div>
                                                        {loadingTranscriptRollbackOptions && <Loader2 size={14} className="animate-spin text-slate-400" />}
                                                    </div>
                                                    <div className="mt-1 text-xs leading-5 text-slate-500">
                                                        Restore a prior optimization run if a repair, rebuild, or retranscription regresses quality. The current transcript is backed up before restore.
                                                    </div>
                                                    <div className="mt-3 space-y-2">
                                                        {transcriptRollbackOptions.length === 0 ? (
                                                            <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-3 text-xs text-slate-500">
                                                                No rollback snapshots recorded for this episode yet.
                                                            </div>
                                                        ) : transcriptRollbackOptions.slice(0, 4).map((option) => (
                                                            <div key={option.run_id} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                                                                <div className="flex items-start justify-between gap-3">
                                                                    <div className="min-w-0">
                                                                        <div className="text-sm font-medium text-slate-800">
                                                                            Run {option.run_id} · {option.mode.replaceAll('_', ' ')}
                                                                        </div>
                                                                        <div className="mt-0.5 text-[11px] text-slate-500">
                                                                            {new Date(option.created_at).toLocaleString()} · {option.pipeline_version}
                                                                        </div>
                                                                        {option.note && (
                                                                            <div className="mt-1 text-xs text-slate-600 line-clamp-2">{option.note}</div>
                                                                        )}
                                                                    </div>
                                                                    <button
                                                                        onClick={() => restoreTranscriptFromRun(option.run_id)}
                                                                        disabled={!option.rollback_available || episodeBusy}
                                                                        className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
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
                                    )}
                                    {false && segments.length > 0 && (
                                        <div className="mt-3 rounded-xl border border-slate-200 bg-white/95 px-3 py-3">
                                            <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                                                <div className="min-w-0">
                                                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Transcript Benchmark</div>
                                                    <div className="mt-1 text-sm text-slate-600">
                                                        Define hand-corrected gold windows for this episode, run deterministic scoring, then attach reviewer verdicts.
                                                    </div>
                                                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                                        <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                            {transcriptGoldWindows.length} gold window{transcriptGoldWindows.length === 1 ? '' : 's'}
                                                        </span>
                                                        <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                            {transcriptEvaluationResults.length} evaluation result{transcriptEvaluationResults.length === 1 ? '' : 's'}
                                                        </span>
                                                        {transcriptEvaluationSummary && (
                                                            <>
                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                                    Avg WER {transcriptEvaluationSummary?.average_wer?.toFixed(3) ?? '0.000'}
                                                                </span>
                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 ring-1 ring-slate-200">
                                                                    Avg CER {transcriptEvaluationSummary?.average_cer?.toFixed(3) ?? '0.000'}
                                                                </span>
                                                            </>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="grid gap-2 sm:grid-cols-2">
                                                    <button
                                                        onClick={useCurrentSelectionForGoldWindow}
                                                        disabled={!selection}
                                                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                                                        title="Use the current transcript or clip selection as the benchmark window range"
                                                    >
                                                        <Scissors size={14} />
                                                        Use Selection Range
                                                    </button>
                                                    <button
                                                        onClick={runTranscriptEvaluation}
                                                        disabled={evaluatingTranscript || transcriptGoldWindows.length === 0}
                                                        className="inline-flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                                                    >
                                                        {evaluatingTranscript ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                                        Run Evaluation
                                                    </button>
                                                </div>
                                            </div>

                                            <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(320px,0.95fr)_minmax(0,1.35fr)]">
                                                <div className="space-y-3 rounded-xl border border-slate-200 bg-slate-50/80 p-3">
                                                    <div className="text-sm font-semibold text-slate-800">Gold Windows</div>
                                                    <div className="grid gap-2 sm:grid-cols-2">
                                                        <div className="sm:col-span-2">
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">Label</label>
                                                            <input
                                                                value={goldWindowLabelDraft}
                                                                onChange={(e) => setGoldWindowLabelDraft(e.target.value)}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="Window label"
                                                            />
                                                        </div>
                                                        <div>
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">Start</label>
                                                            <input
                                                                value={goldWindowStartDraft}
                                                                onChange={(e) => setGoldWindowStartDraft(e.target.value)}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="0.00"
                                                            />
                                                        </div>
                                                        <div>
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">End</label>
                                                            <input
                                                                value={goldWindowEndDraft}
                                                                onChange={(e) => setGoldWindowEndDraft(e.target.value)}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="15.00"
                                                            />
                                                        </div>
                                                        <div className="sm:col-span-2">
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">Reference Transcript</label>
                                                            <textarea
                                                                value={goldWindowReferenceDraft}
                                                                onChange={(e) => setGoldWindowReferenceDraft(e.target.value)}
                                                                rows={5}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="Paste the hand-corrected reference transcript for this window."
                                                            />
                                                        </div>
                                                        <div className="sm:col-span-2">
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">Entities</label>
                                                            <input
                                                                value={goldWindowEntitiesDraft}
                                                                onChange={(e) => setGoldWindowEntitiesDraft(e.target.value)}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="Comma-separated entities to track"
                                                            />
                                                        </div>
                                                        <div className="sm:col-span-2">
                                                            <label className="mb-1 block text-xs font-medium text-slate-600">Notes</label>
                                                            <textarea
                                                                value={goldWindowNotesDraft}
                                                                onChange={(e) => setGoldWindowNotesDraft(e.target.value)}
                                                                rows={2}
                                                                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                placeholder="Speaker boundaries, entity focus, overlap risk, punctuation notes..."
                                                            />
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={createTranscriptGoldWindow}
                                                        disabled={savingTranscriptGoldWindow}
                                                        className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                                                    >
                                                        {savingTranscriptGoldWindow ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                                                        Save Gold Window
                                                    </button>
                                                    {loadingTranscriptGoldWindows ? (
                                                        <div className="inline-flex items-center gap-2 text-xs text-slate-500">
                                                            <Loader2 size={14} className="animate-spin" />
                                                            Loading benchmark windows...
                                                        </div>
                                                    ) : transcriptGoldWindowsError ? (
                                                        <div className="text-xs text-rose-600">{transcriptGoldWindowsError}</div>
                                                    ) : transcriptGoldWindows.length === 0 ? (
                                                        <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">
                                                            No gold windows yet. Create one from a selected range or enter a benchmark window manually.
                                                        </div>
                                                    ) : (
                                                        <div className="space-y-2">
                                                            {transcriptGoldWindows.map((window) => (
                                                                <div key={window.id} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                                                    <div className="flex items-start justify-between gap-2">
                                                                        <div className="min-w-0">
                                                                            <div className="text-sm font-medium text-slate-800">{window.label}</div>
                                                                            <div className="mt-0.5 text-xs text-slate-500">
                                                                                {window.start_time.toFixed(2)}s to {window.end_time.toFixed(2)}s
                                                                                {window.language ? ` • ${window.language}` : ''}
                                                                            </div>
                                                                        </div>
                                                                        <button
                                                                            onClick={() => handleSeek(window.start_time)}
                                                                            className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
                                                                        >
                                                                            Jump
                                                                        </button>
                                                                    </div>
                                                                    <div className="mt-2 line-clamp-3 text-xs leading-5 text-slate-600">
                                                                        {window.reference_text}
                                                                    </div>
                                                                    {window.entities.length > 0 && (
                                                                        <div className="mt-2 flex flex-wrap gap-1">
                                                                            {window.entities.map((entity) => (
                                                                                <span key={`${window.id}-${entity}`} className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                                    {entity}
                                                                                </span>
                                                                            ))}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>

                                                <div className="space-y-3 rounded-xl border border-slate-200 bg-slate-50/80 p-3">
                                                    <div className="flex items-center justify-between gap-3">
                                                        <div>
                                                            <div className="text-sm font-semibold text-slate-800">Evaluation Results</div>
                                                            <div className="text-xs text-slate-500">
                                                                WER and CER come from the stored reference windows. Reviewer verdicts capture the human judgment layer.
                                                            </div>
                                                        </div>
                                                    </div>
                                                    {loadingTranscriptEvaluationResults ? (
                                                        <div className="inline-flex items-center gap-2 text-xs text-slate-500">
                                                            <Loader2 size={14} className="animate-spin" />
                                                            Loading evaluation results...
                                                        </div>
                                                    ) : transcriptEvaluationError ? (
                                                        <div className="text-xs text-rose-600">{transcriptEvaluationError}</div>
                                                    ) : transcriptEvaluationResults.length === 0 ? (
                                                        <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-xs text-slate-500">
                                                            No evaluation results yet. Run evaluation after defining at least one gold window.
                                                        </div>
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
                                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-700 ring-1 ring-slate-200">
                                                                                WER {result.wer.toFixed(3)}
                                                                            </span>
                                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                                CER {result.cer.toFixed(3)}
                                                                            </span>
                                                                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                                Unknown {result.unknown_speaker_rate.toFixed(2)}
                                                                            </span>
                                                                            {result.entity_accuracy != null && (
                                                                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] text-slate-600 ring-1 ring-slate-200">
                                                                                    Entity {result.entity_accuracy.toFixed(2)}
                                                                                </span>
                                                                            )}
                                                                        </div>
                                                                        <div className="mt-3 grid gap-3 lg:grid-cols-2">
                                                                            <div>
                                                                                <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Reference</div>
                                                                                <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                                                                                    {result.reference_text}
                                                                                </div>
                                                                            </div>
                                                                            <div>
                                                                                <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Candidate</div>
                                                                                <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                                                                                    {result.candidate_text}
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                        <div className="mt-3 grid gap-2 lg:grid-cols-[140px_140px_minmax(0,1fr)_auto]">
                                                                            <input
                                                                                value={reviewer}
                                                                                onChange={(e) => setEvaluationReviewReviewerDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                                placeholder="Reviewer"
                                                                            />
                                                                            <select
                                                                                value={verdict}
                                                                                onChange={(e) => setEvaluationReviewVerdictDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                            >
                                                                                <option value="better">Better</option>
                                                                                <option value="same">Same</option>
                                                                                <option value="worse">Worse</option>
                                                                                <option value="bad_merge">Bad merge</option>
                                                                                <option value="bad_speaker_reassignment">Bad speaker reassignment</option>
                                                                                <option value="bad_entity_repair">Bad entity repair</option>
                                                                                <option value="language_regression">Language regression</option>
                                                                            </select>
                                                                            <input
                                                                                value={reviewNotes}
                                                                                onChange={(e) => setEvaluationReviewNotesDrafts((current) => ({ ...current, [result.id]: e.target.value }))}
                                                                                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                                                                                placeholder="Review notes"
                                                                            />
                                                                            <button
                                                                                onClick={() => submitTranscriptEvaluationReview(result.id)}
                                                                                disabled={reviewingEvaluationResultId === result.id}
                                                                                className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                                                                            >
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
                                                    {accessRestrictionReason || 'This episode is not accessible with the current YouTube session, so it will be skipped instead of being downloaded or processed.'}
                                                </div>
                                            </div>
                                        ) : video && !video.processed && video.status !== 'queued' && video.status !== 'running' && video.status !== 'downloading' && video.status !== 'transcribing' && video.status !== 'diarizing' ? (
                                            <>
                                                <p className="text-xs text-slate-400 mb-5 text-center">Start a transcription job to generate the transcript for this episode.</p>
                                                <button
                                                    onClick={async () => {
                                                        setStartingTranscription(true);
                                                        try {
                                                            await api.post(`/videos/${video.id}/process`);
                                                            setVideo({ ...video, status: 'queued' });
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
                                        const isActiveSegment =
                                            transcriptPlaybackTime >= seg.start_time &&
                                            transcriptPlaybackTime < seg.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS;
                                        const isActiveMatch = searchLower && filteredIdx === searchMatchIndex;
                                        const isDeepLinkedSegment = !searchLower && deepLinkedSegmentId === seg.id;
                                        const wordsFn = typeof seg.id === 'number'
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
                                                            : 'bg-white border-transparent hover:border-slate-200 hover:bg-white'}`}
                                                onClick={() => {
                                                    // Seek to segment start on click (word spans use stopPropagation so they won't trigger this)
                                                    if (window.getSelection()?.toString().length === 0) {
                                                        handleSeek(seg.start_time);
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
                                                            className="opacity-100 lg:opacity-0 lg:group-hover:opacity-100 p-1 rounded text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition"
                                                            title="Edit transcript text"
                                                        >
                                                            <Pencil size={12} />
                                                        </button>
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
                                                                            handleSeek(w.start);
                                                                        }}
                                                                    >
                                                                        {searchLower ? highlightText(w.word) : w.word}
                                                                        {idx < wordsFn.length - 1 ? ' ' : ''}
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
                    {activeTab === 'optimize' && (
                        <div className="h-full overflow-y-auto p-4">
                            <div className="space-y-4">
                                {renderTranscriptOptimizationSnapshotCard('optimize')}
                                <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm">
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Workbench Notes</div>
                                    <div className="mt-2 text-sm font-semibold text-slate-900">Use transcript selections as benchmark ranges</div>
                                    <div className="mt-1 text-xs leading-5 text-slate-600">
                                        The benchmark tools still use the current transcript selection when you click <span className="font-medium">Use Selection Range</span>. Open the transcript tab whenever you need to inspect or select a passage, then return here to run the evaluation.
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => setActiveTab('transcript')}
                                        className="mt-3 inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                                    >
                                        <FileText size={14} />
                                        Open Transcript
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'clips' && (
                        <ClipsTab
                            isActive={activeTab === 'clips'}
                            onSeek={handleSeek}
                            onStartClipEdit={startClipEdit}
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
                        renderMainPlayer={renderMainPlayer}
                        onNavigateToTranscript={() => setActiveTab('transcript')}
                    />
                ) : activeTab === 'optimize' ? (
                    <OptimizeTab
                        hasTranscript={segments.length > 0}
                        renderWorkbench={renderTranscriptOptimizationWorkbench}
                        renderSnapshot={() => renderTranscriptOptimizationSnapshotCard('optimize')}
                        onNavigateToTranscript={() => setActiveTab('transcript')}
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
                        onCitationClick={(citation: EpisodeChatCitation) => {
                            const citationVideoId = Number(citation.video_id || 0);
                            const jumpTime = Number(citation.start_time || 0);
                            const primarySegmentId = Array.isArray(citation.segment_ids) ? Number(citation.segment_ids[0] || 0) : 0;
                            if (citationVideoId > 0 && citationVideoId !== Number(id)) {
                                const params = new URLSearchParams();
                                params.set('tab', 'chat');
                                if (Number.isFinite(jumpTime) && jumpTime >= 0) {
                                    params.set('t', String(Math.floor(jumpTime)));
                                }
                                if (primarySegmentId > 0) {
                                    params.set('segment_id', String(primarySegmentId));
                                }
                                navigate(`/video/${citationVideoId}?${params.toString()}`);
                                return;
                            }
                            handleSeek(jumpTime);
                            if (primarySegmentId > 0) {
                                scrollTranscriptToSegment(primarySegmentId);
                            } else {
                                scrollTranscriptToTime(jumpTime);
                            }
                        }}
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
                        ctx={{
                            selectedReconstructionSpeaker,
                            resolveWorkbenchAudioUrl,
                            segments,
                            setSelectedReconstructionSpeakerId,
                            selectedReconstructionSpeakerId,
                            handleAddReconstructionSample,
                            addingReconstructionSampleSpeakerId,
                            handleApproveReconstructionSpeaker,
                            approvingReconstructionSpeakerId,
                            handleUpdateReconstructionSampleState,
                            updatingReconstructionSampleKey,
                            cleaningReconstructionSampleKey,
                            handleCleanupReconstructionSample,
                            reconstructionTestTextDrafts,
                            setReconstructionTestTextDrafts,
                            episodeBusy,
                            handleTestReconstructionSpeaker,
                            testingReconstructionSpeakerId,
                            hasReconstructionAudio,
                            reconstructionAudioUrl,
                            video,
                            handleSetReconstructionPlayback,
                            usingReconstructionForPlayback,
                            switchingReconstructionPlayback,
                            selectedReconstructionPreviewSegmentId,
                            setSelectedReconstructionPreviewSegmentId,
                            reconstructionPreviewAudioUrl,
                            reconstructionPreviewText,
                            handlePreviewReconstructionSegment,
                            queueingReconstruction,
                            reconstructionBusy,
                            handleQueueReconstruction,
                            reconstructionInstructionDraft,
                            setReconstructionInstructionDraft,
                            savingReconstructionSettings,
                            handleSaveReconstructionSettings,
                            setReconstructionStudioTab,
                        }}
                        onRefreshWorkbench={() => void loadReconstructionWorkbench()}
                        onSetStudioTab={setReconstructionStudioTab}
                    />
                ) : showClipEditorMain && activeEditingClip && clipEditorDraft ? (
                    <ClipEditorWorkspace
                        activeEditingClip={activeEditingClip}
                        currentTime={currentTime}
                        video={video}
                        segments={segments}
                        renderMainPlayer={renderMainPlayer}
                        onSeek={handleSeek}
                    />
                ) : (
                    <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
                        <div className="w-full max-w-5xl">
                            {renderMainPlayer("w-full bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video")}
                        </div>
                    </div>
                )}

                {!showClipEditorMain && activeTab === 'transcript' && (
                    <>
                        <button
                            onClick={() => setFunnyDrawerOpen(v => !v)}
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
                                            )
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
