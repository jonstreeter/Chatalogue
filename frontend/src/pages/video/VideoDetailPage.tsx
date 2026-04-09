import { startTransition, useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../../lib/api';
import { toApiUrl } from '../../lib/api';
import type { Video, TranscriptSegment, Clip, Speaker, SpeakerSample, FunnyMoment, VideoChapterSuggestion, VideoDescriptionRevision, ClipExportArtifact, ReconstructionWorkbench, Job, WorkbenchTaskProgress, CleanupWorkbench, ClearVoiceInstallInfo, ClearVoiceTestResult, EpisodeCloneConceptsResponse, EpisodeCloneEngine, EpisodeCloneJob, TranscriptQuality, TranscriptRollbackOption, TranscriptRestoreResponse, TranscriptGoldWindow, TranscriptEvaluationResult, TranscriptEvaluationReview, TranscriptEvaluationBatchResponse, TranscriptRepairQueueResponse, TranscriptDiarizationRebuildQueueResponse, TranscriptRetranscriptionQueueResponse } from '../../types';
import { Loader2, ArrowLeft, FileText, Scissors, Users, X, CheckCircle2, Play, Pause, Plus, Trash2, Mic, Search, ChevronUp, ChevronDown, GitMerge, RotateCcw, Eraser, AudioLines, Smile, RefreshCw, Bot, Copy, Pencil, Save, XCircle, Download, Upload, PlayCircle, Clock, Sparkles, Clapperboard, CircleHelp, type LucideIcon } from 'lucide-react';
import { SpeakerList } from '../../components/SpeakerList';
import { SpeakerModal } from '../../components/SpeakerModal';
import { CloneWorkbenchPanel } from '../../components/video/CloneWorkbenchPanel';

type UnifiedPlayer = {
    getCurrentTime?: () => number;
    seekTo?: (seconds: number, allowSeekAhead?: boolean) => void;
    playVideo?: () => void | Promise<void>;
    pauseVideo?: () => void;
    getPlaybackRate?: () => number;
    setPlaybackRate?: (rate: number) => void;
    getPlayerState?: () => number;
};

type VideoSidebarTab = 'transcript' | 'optimize' | 'clips' | 'speakers' | 'cleanup' | 'reconstruction' | 'clone' | 'youtube';

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

type OllamaLocalModel = {
    name: string;
    size_bytes?: number;
    modified_at?: string;
    parameter_size?: string;
    quantization_level?: string;
    families?: string[] | null;
};

type OllamaLocalModelsResponse = {
    status: string;
    ollama_url?: string;
    current_model?: string;
    models: OllamaLocalModel[];
    error?: string;
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
    const cropPreviewRef = useRef<HTMLDivElement>(null);
    const clipTimelineRef = useRef<HTMLDivElement>(null);
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
    const [clipTitle, setClipTitle] = useState('');
    const [creatingClip, setCreatingClip] = useState(false);

    // Speaker Modal State
    const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
    const [initialSample, setInitialSample] = useState<SpeakerSample | null>(null);
    const speakerDetailCacheRef = useRef<Map<number, Speaker>>(new Map());

    // Clips State
    const [clips, setClips] = useState<Clip[]>([]);
    const [clipExportArtifactsByClip, setClipExportArtifactsByClip] = useState<Record<number, ClipExportArtifact[]>>({});
    const [loadingClips, setLoadingClips] = useState(false);
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
    const [generatingYoutubeAi, setGeneratingYoutubeAi] = useState(false);
    const [copiedYoutubeField, setCopiedYoutubeField] = useState<'summary' | 'chapters' | 'description' | null>(null);
    const [descriptionHistory, setDescriptionHistory] = useState<VideoDescriptionRevision[]>([]);
    const [loadingDescriptionHistory, setLoadingDescriptionHistory] = useState(false);
    const [publishingYoutubeDescription, setPublishingYoutubeDescription] = useState(false);
    const [restoringDescriptionRevisionId, setRestoringDescriptionRevisionId] = useState<number | null>(null);
    const [cloneEngines, setCloneEngines] = useState<EpisodeCloneEngine[]>([]);
    const [loadingCloneEngines, setLoadingCloneEngines] = useState(false);
    const [cloneEnginesError, setCloneEnginesError] = useState<string | null>(null);
    const [cloneEngineKey, setCloneEngineKey] = useState('default');
    const [cloneOllamaModels, setCloneOllamaModels] = useState<OllamaLocalModel[]>([]);
    const [loadingCloneOllamaModels, setLoadingCloneOllamaModels] = useState(false);
    const [cloneOllamaModelsError, setCloneOllamaModelsError] = useState<string | null>(null);
    const [cloneOllamaModel, setCloneOllamaModel] = useState('');
    const [cloneStylePrompt, setCloneStylePrompt] = useState('Create a fresh, original script with a clear hook, stronger structure, and a more polished delivery than the source.');
    const [cloneNotes, setCloneNotes] = useState('');
    const [cloneConcepts, setCloneConcepts] = useState<EpisodeCloneConceptsResponse | null>(null);
    const [cloneConceptsText, setCloneConceptsText] = useState('');
    const [cloneExcludedReferencesText, setCloneExcludedReferencesText] = useState('');
    const [detectingCloneConcepts, setDetectingCloneConcepts] = useState(false);
    const [cloneBatchSize, setCloneBatchSize] = useState(1);
    const [generatingClone, setGeneratingClone] = useState(false);
    const [cloneJobs, setCloneJobs] = useState<EpisodeCloneJob[]>([]);
    const [loadingCloneJobs, setLoadingCloneJobs] = useState(false);
    const [cloneJobsError, setCloneJobsError] = useState<string | null>(null);
    const [selectedCloneJobId, setSelectedCloneJobId] = useState<number | null>(null);
    const [copiedCloneScript, setCopiedCloneScript] = useState(false);
    const cloneGenerateRequestRef = useRef(0);
    const cloneJobsRequestRef = useRef(0);
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
    const assignSpeakerPickerLimit = 100;
    const [assignSpeakers, setAssignSpeakers] = useState<Speaker[]>([]);
    const [assignSearch, setAssignSearch] = useState('');
    const [assignLoading, setAssignLoading] = useState(false);
    const [purging, setPurging] = useState(false);
    const [redoing, setRedoing] = useState(false);
    const [redoingDiarization, setRedoingDiarization] = useState(false);
    const [consolidatingTranscript, setConsolidatingTranscript] = useState(false);
    const [transcriptQuality, setTranscriptQuality] = useState<TranscriptQuality | null>(null);
    const [loadingTranscriptQuality, setLoadingTranscriptQuality] = useState(false);
    const [transcriptQualityError, setTranscriptQualityError] = useState<string | null>(null);
    const [transcriptRollbackOptions, setTranscriptRollbackOptions] = useState<TranscriptRollbackOption[]>([]);
    const [loadingTranscriptRollbackOptions, setLoadingTranscriptRollbackOptions] = useState(false);
    const [restoringTranscriptRunId, setRestoringTranscriptRunId] = useState<number | null>(null);
    const [transcriptGoldWindows, setTranscriptGoldWindows] = useState<TranscriptGoldWindow[]>([]);
    const [loadingTranscriptGoldWindows, setLoadingTranscriptGoldWindows] = useState(false);
    const [transcriptGoldWindowsError, setTranscriptGoldWindowsError] = useState<string | null>(null);
    const [savingTranscriptGoldWindow, setSavingTranscriptGoldWindow] = useState(false);
    const [evaluatingTranscript, setEvaluatingTranscript] = useState(false);
    const [transcriptEvaluationSummary, setTranscriptEvaluationSummary] = useState<TranscriptEvaluationBatchResponse | null>(null);
    const [transcriptEvaluationResults, setTranscriptEvaluationResults] = useState<TranscriptEvaluationResult[]>([]);
    const [loadingTranscriptEvaluationResults, setLoadingTranscriptEvaluationResults] = useState(false);
    const [transcriptEvaluationError, setTranscriptEvaluationError] = useState<string | null>(null);
    const [reviewingEvaluationResultId, setReviewingEvaluationResultId] = useState<number | null>(null);
    const [evaluationReviewsByResultId, setEvaluationReviewsByResultId] = useState<Record<number, TranscriptEvaluationReview[]>>({});
    const [goldWindowLabelDraft, setGoldWindowLabelDraft] = useState('Gold Window');
    const [goldWindowStartDraft, setGoldWindowStartDraft] = useState('');
    const [goldWindowEndDraft, setGoldWindowEndDraft] = useState('');
    const [goldWindowReferenceDraft, setGoldWindowReferenceDraft] = useState('');
    const [goldWindowEntitiesDraft, setGoldWindowEntitiesDraft] = useState('');
    const [goldWindowNotesDraft, setGoldWindowNotesDraft] = useState('');
    const [evaluationReviewVerdictDrafts, setEvaluationReviewVerdictDrafts] = useState<Record<number, string>>({});
    const [evaluationReviewNotesDrafts, setEvaluationReviewNotesDrafts] = useState<Record<number, string>>({});
    const [evaluationReviewReviewerDrafts, setEvaluationReviewReviewerDrafts] = useState<Record<number, string>>({});
    const [queueingTranscriptRepair, setQueueingTranscriptRepair] = useState(false);
    const [queueingDiarizationRebuild, setQueueingDiarizationRebuild] = useState(false);
    const [queueingDiarizationBenchmark, setQueueingDiarizationBenchmark] = useState(false);
    const [queueingFullRetranscription, setQueueingFullRetranscription] = useState(false);
    const [diarizationBenchmarkSensitivity, setDiarizationBenchmarkSensitivity] = useState<'aggressive' | 'balanced' | 'conservative'>('balanced');
    const [diarizationBenchmarkThreshold, setDiarizationBenchmarkThreshold] = useState('0.35');
    const [queueingVoiceFixer, setQueueingVoiceFixer] = useState(false);
    const [queueingReconstruction, setQueueingReconstruction] = useState(false);
    const [auxiliaryJobs, setAuxiliaryJobs] = useState<Job[]>([]);
    const [workbenchTaskProgress, setWorkbenchTaskProgress] = useState<WorkbenchTaskProgress | null>(null);
    const [loadingCleanupWorkbench, setLoadingCleanupWorkbench] = useState(false);
    const [cleanupWorkbench, setCleanupWorkbench] = useState<CleanupWorkbench | null>(null);
    const [analyzingCleanupWorkbench, setAnalyzingCleanupWorkbench] = useState(false);
    const [runningClearVoiceModel, setRunningClearVoiceModel] = useState<string | null>(null);
    const [selectingCleanupCandidateId, setSelectingCleanupCandidateId] = useState<string | null>(null);
    const [clearVoiceInstallInfo, setClearVoiceInstallInfo] = useState<ClearVoiceInstallInfo | null>(null);
    const [loadingClearVoiceInstallInfo, setLoadingClearVoiceInstallInfo] = useState(false);
    const [installingClearVoice, setInstallingClearVoice] = useState(false);
    const [repairingClearVoice, setRepairingClearVoice] = useState(false);
    const [testingClearVoice, setTestingClearVoice] = useState(false);
    const [clearVoiceTestResult, setClearVoiceTestResult] = useState<ClearVoiceTestResult | null>(null);
    const [switchingReconstructionPlayback, setSwitchingReconstructionPlayback] = useState(false);
    const [loadingReconstructionWorkbench, setLoadingReconstructionWorkbench] = useState(false);
    const [reconstructionWorkbench, setReconstructionWorkbench] = useState<ReconstructionWorkbench | null>(null);
    const [savingReconstructionSettings, setSavingReconstructionSettings] = useState(false);
    const [testingReconstructionSpeakerId, setTestingReconstructionSpeakerId] = useState<number | null>(null);
    const [reconstructionInstructionDraft, setReconstructionInstructionDraft] = useState('');
    const [reconstructionStudioTab, setReconstructionStudioTab] = useState<'voices' | 'reconstruction'>('voices');
    const [selectedReconstructionSpeakerId, setSelectedReconstructionSpeakerId] = useState<number | null>(null);
    const [reconstructionTestTextDrafts, setReconstructionTestTextDrafts] = useState<Record<number, string>>({});
    const [cleaningReconstructionSampleKey, setCleaningReconstructionSampleKey] = useState<string | null>(null);
    const [updatingReconstructionSampleKey, setUpdatingReconstructionSampleKey] = useState<string | null>(null);
    const [addingReconstructionSampleSpeakerId, setAddingReconstructionSampleSpeakerId] = useState<number | null>(null);
    const [approvingReconstructionSpeakerId, setApprovingReconstructionSpeakerId] = useState<number | null>(null);
    const [selectedReconstructionPreviewSegmentId, setSelectedReconstructionPreviewSegmentId] = useState<number | null>(null);
    const [previewingReconstructionSegment, setPreviewingReconstructionSegment] = useState(false);
    const [reconstructionPreviewAudioUrl, setReconstructionPreviewAudioUrl] = useState('');
    const [reconstructionPreviewText, setReconstructionPreviewText] = useState('');
    const [savingVoiceFixerSettings, setSavingVoiceFixerSettings] = useState(false);
    const [voiceFixerModeDraft, setVoiceFixerModeDraft] = useState(0);
    const [voiceFixerMixDraft, setVoiceFixerMixDraft] = useState(1);
    const [voiceFixerLevelingDraft, setVoiceFixerLevelingDraft] = useState<'off' | 'gentle' | 'balanced' | 'strong'>('off');
    const [voiceFixerApplyScopeDraft, setVoiceFixerApplyScopeDraft] = useState<'none' | 'playback' | 'processing' | 'both'>('none');
    const defaultReconstructionTestText = 'This is a test of the voice model. If you approve this test, then click the approve voice button below.';
    const selectedReconstructionSpeaker = useMemo(
        () => reconstructionWorkbench?.speakers.find((speaker) => speaker.speaker_id === selectedReconstructionSpeakerId) || null,
        [reconstructionWorkbench, selectedReconstructionSpeakerId]
    );
    const selectedCloneJob = useMemo(
        () => cloneJobs.find((job) => job.job_id === selectedCloneJobId) || cloneJobs[0] || null,
        [cloneJobs, selectedCloneJobId]
    );
    const cloneDraftResult = selectedCloneJob?.result || null;
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
    const aiMetadataPanelTitle = isYoutubeMedia ? 'YouTube Summary + Chapters' : 'Episode Summary + Chapters';
    const aiMetadataPanelDescription = isYoutubeMedia
        ? 'Generate a YouTube-style episode description summary and chapter timestamps/descriptions from the transcript using the current LLM provider.'
        : 'Generate a readable episode summary and chapter-style conversation index from the transcript using the current LLM provider.';
    const aiMetadataGenerateTitle = isYoutubeMedia
        ? 'Generate or re-generate YouTube summary + chapters'
        : 'Generate or re-generate episode summary + chapters';
    const aiMetadataEmptyText = isYoutubeMedia
        ? 'No generated summary/chapters yet. Click Generate to create a YouTube-ready draft description and chapter list.'
        : 'No generated summary/chapters yet. Click Generate to create a readable summary and chapter-style conversation index.';
    const aiMetadataCurrentDescriptionLabel = isYoutubeMedia ? 'Current Video Description (Stored)' : 'Current Episode Description (Stored)';
    const aiMetadataChaptersLabel = isYoutubeMedia ? 'Chapters (YouTube-style)' : 'Conversation Index';
    const aiMetadataDescriptionLabel = isYoutubeMedia ? 'YouTube Description Draft (Copy/Paste)' : 'Episode Description Draft';
    const aiMetadataPublishLabel = isYoutubeMedia ? 'Publish Draft (Archive Current)' : 'Apply Draft (Archive Current)';
    const aiMetadataPublishHelp = isYoutubeMedia
        ? 'Updates the app’s stored video description and preserves restorable history.'
        : 'Updates the app’s stored episode description and preserves restorable history.';
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
    const loadAuxiliaryJobs = async () => {
        if (!video?.id || !isUploadedMedia) {
            setAuxiliaryJobs([]);
            return;
        }
        try {
            const res = await api.get<Job[]>('/jobs', {
                params: {
                    video_id: video.id,
                    status: 'queued,paused,running',
                    job_type: 'voicefixer_cleanup,conversation_reconstruct',
                    limit: 20,
                },
            });
            setAuxiliaryJobs(Array.isArray(res.data) ? res.data : []);
        } catch (e) {
            console.error('Failed to fetch auxiliary jobs:', e);
        }
    };
    const fetchWorkbenchTaskProgress = async () => {
        if (!id || !isUploadedMedia) {
            setWorkbenchTaskProgress(null);
            return;
        }
        try {
            const res = await api.get<WorkbenchTaskProgress>(`/videos/${id}/workbench/progress`);
            setWorkbenchTaskProgress(res.data || null);
        } catch (e) {
            console.error('Failed to fetch workbench progress:', e);
        }
    };
    const fetchCleanupWorkbench = async () => {
        if (!id || !isUploadedMedia) {
            setCleanupWorkbench(null);
            return;
        }
        setLoadingCleanupWorkbench(true);
        try {
            const res = await api.get<CleanupWorkbench>(`/videos/${id}/cleanup/workbench`);
            setCleanupWorkbench(res.data || null);
        } catch (e) {
            console.error('Failed to fetch cleanup workbench:', e);
        } finally {
            setLoadingCleanupWorkbench(false);
        }
    };
    const fetchClearVoiceInstallInfo = async () => {
        if (!isUploadedMedia) {
            setClearVoiceInstallInfo(null);
            return;
        }
        setLoadingClearVoiceInstallInfo(true);
        try {
            const res = await api.get<ClearVoiceInstallInfo>('/system/clearvoice/install-info');
            setClearVoiceInstallInfo(res.data);
        } catch (e) {
            console.error('Failed to fetch ClearVoice install info:', e);
        } finally {
            setLoadingClearVoiceInstallInfo(false);
        }
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

    const normalizeCloneTextList = (value: string) => {
        return value
            .split(/\r?\n/)
            .map((item) => item.trim())
            .filter(Boolean);
    };

    const selectedCloneEngine = cloneEngines.find((engine) => engine.key === cloneEngineKey) || cloneEngines[0] || null;
    const cloneUsesOllama = (selectedCloneEngine?.provider || '') === 'ollama';

    const cloneListsMatch = (a: string[], b: string[]) => {
        if (a.length !== b.length) return false;
        return a.every((item, index) => item === b[index]);
    };

    const resolveCloneEngineOverride = () => {
        const selected = selectedCloneEngine;
        const overrideModel = cloneUsesOllama ? (cloneOllamaModel.trim() || selected?.model || '') : (selected?.model || '');
        if (!selected || selected.key === 'default') {
            return { provider_override: undefined as string | undefined, model_override: overrideModel || undefined };
        }
        return {
            provider_override: selected.provider || undefined,
            model_override: overrideModel || undefined,
        };
    };

    const cloneJobMatchesVisibleInputs = (job: EpisodeCloneJob | null | undefined) => {
        if (!job) return false;
        const currentEngine = resolveCloneEngineOverride();
        return String(job.request?.style_prompt || '').trim() === cloneStylePrompt.trim()
            && String(job.request?.notes || '').trim() === cloneNotes.trim()
            && String(job.request?.provider_override || '') === String(currentEngine.provider_override || '')
            && String(job.request?.model_override || '') === String(currentEngine.model_override || '')
            && cloneListsMatch(job.request?.approved_concepts || [], normalizeCloneTextList(cloneConceptsText))
            && cloneListsMatch(job.request?.excluded_references || [], normalizeCloneTextList(cloneExcludedReferencesText));
    };

    const fetchCloneEngines = async (signal?: AbortSignal) => {
        setLoadingCloneEngines(true);
        setCloneEnginesError(null);
        try {
            const res = await api.get<EpisodeCloneEngine[]>('/episode-clone/engines', { signal });
            if (signal?.aborted) return;
            const engines = Array.isArray(res.data) ? res.data : [];
            setCloneEngines(engines);
            setCloneEngineKey((current) => {
                if (current && engines.some((engine) => engine.key === current)) return current;
                return engines[0]?.key || 'default';
            });
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch clone engines:', e);
            setCloneEngines([]);
            setCloneEnginesError(e?.response?.data?.detail || 'Failed to load clone engines');
        } finally {
            if (!signal?.aborted) {
                setLoadingCloneEngines(false);
            }
        }
    };

    const fetchCloneOllamaModels = async (signal?: AbortSignal) => {
        setLoadingCloneOllamaModels(true);
        setCloneOllamaModelsError(null);
        try {
            const res = await api.get<OllamaLocalModelsResponse>('/settings/ollama/models', { signal });
            if (signal?.aborted) return;
            if (res.data?.status !== 'ok') {
                setCloneOllamaModels([]);
                setCloneOllamaModelsError(res.data?.error || 'Failed to load local Ollama models.');
                return;
            }
            const models = Array.isArray(res.data?.models) ? res.data.models.filter((model) => !!model?.name) : [];
            setCloneOllamaModels(models);
            setCloneOllamaModel((current) => {
                if (current && models.some((model) => model.name === current)) return current;
                if (selectedCloneEngine?.model && models.some((model) => model.name === selectedCloneEngine.model)) return selectedCloneEngine.model;
                return res.data?.current_model || models[0]?.name || '';
            });
        } catch (e: any) {
            if (signal?.aborted) return;
            console.error('Failed to fetch Ollama models for clone workbench:', e);
            setCloneOllamaModels([]);
            setCloneOllamaModelsError(e?.response?.data?.detail || 'Failed to load local Ollama models.');
        } finally {
            if (!signal?.aborted) {
                setLoadingCloneOllamaModels(false);
            }
        }
    };

    const fetchCloneJobs = async (
        videoId: number,
        signal?: AbortSignal,
        options?: { preferredJobId?: number | null; silent?: boolean }
    ) => {
        const requestId = ++cloneJobsRequestRef.current;
        if (!options?.silent) {
            setLoadingCloneJobs(true);
        }
        setCloneJobsError(null);
        try {
            const res = await api.get<EpisodeCloneJob[]>(`/videos/${videoId}/episode-clone/jobs`, {
                params: { limit: 16 },
                signal,
            });
            if (requestId !== cloneJobsRequestRef.current || signal?.aborted) return [];
            const jobs = Array.isArray(res.data) ? res.data : [];
            setCloneJobs(jobs);
            setGeneratingClone(jobs.some((job) => ['queued', 'running'].includes(String(job.status || '').toLowerCase())));
            setSelectedCloneJobId((current) => {
                if (options?.preferredJobId && jobs.some((job) => job.job_id === options.preferredJobId)) {
                    return options.preferredJobId;
                }
                if (current && jobs.some((job) => job.job_id === current)) {
                    return current;
                }
                return jobs[0]?.job_id ?? null;
            });
            return jobs;
        } catch (e: any) {
            if (signal?.aborted) return null;
            console.error('Failed to fetch episode clone jobs:', e);
            setCloneJobsError(e?.response?.data?.detail || 'Failed to load clone workbench history');
            setGeneratingClone(false);
            return [];
        } finally {
            if (requestId === cloneJobsRequestRef.current && !signal?.aborted && !options?.silent) {
                setLoadingCloneJobs(false);
            }
        }
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
        cloneGenerateRequestRef.current += 1;
        cloneJobsRequestRef.current += 1;
        setCloneJobs([]);
        setCloneJobsError(null);
        setSelectedCloneJobId(null);
        setTranscriptQuality(null);
        setTranscriptQualityError(null);
        setTranscriptGoldWindows([]);
        setTranscriptGoldWindowsError(null);
        setTranscriptEvaluationSummary(null);
        setTranscriptEvaluationResults([]);
        setTranscriptEvaluationError(null);
        setEvaluationReviewsByResultId({});
        setGoldWindowLabelDraft('Gold Window');
        setGoldWindowStartDraft('');
        setGoldWindowEndDraft('');
        setGoldWindowReferenceDraft('');
        setGoldWindowEntitiesDraft('');
        setGoldWindowNotesDraft('');
        setCloneEngines([]);
        setCloneEnginesError(null);
        setCloneEngineKey('default');
        setCloneOllamaModels([]);
        setCloneOllamaModelsError(null);
        setCloneOllamaModel('');
        setCloneConcepts(null);
        setCloneConceptsText('');
        setCloneExcludedReferencesText('');
        setDetectingCloneConcepts(false);
        setGeneratingClone(false);
        setCopiedCloneScript(false);
        if (id) fetchData();
    }, [id]);

    useEffect(() => {
        if (activeTab !== 'clone') return;
        const controller = new AbortController();
        void fetchCloneEngines(controller.signal);
        if (id) {
            void fetchCloneJobs(Number(id), controller.signal);
        }
        return () => controller.abort();
    }, [activeTab, id]);

    useEffect(() => {
        if (activeTab !== 'clone' || !cloneUsesOllama) return;
        const controller = new AbortController();
        void fetchCloneOllamaModels(controller.signal);
        return () => controller.abort();
    }, [activeTab, cloneUsesOllama, cloneEngineKey]);

    useEffect(() => {
        if (activeTab !== 'clone' || cloneJobs.length === 0) return;
        const hasActiveCloneJob = cloneJobs.some((job) => ['queued', 'running'].includes(String(job.status || '').toLowerCase()));
        if (!hasActiveCloneJob || !id) return;
        const interval = window.setInterval(() => {
            void fetchCloneJobs(Number(id), undefined, { silent: true });
        }, 2500);
        return () => window.clearInterval(interval);
    }, [activeTab, cloneJobs, id]);

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
            setAuxiliaryJobs([]);
            return;
        }
        void loadAuxiliaryJobs();
        if (!(voiceFixerBusy || voiceFixerPaused || queueingVoiceFixer || reconstructionBusy || reconstructionPaused || queueingReconstruction)) {
            return;
        }
        const timer = window.setInterval(() => {
            void loadAuxiliaryJobs();
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
            setWorkbenchTaskProgress(null);
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

        void fetchWorkbenchTaskProgress();
        const intervalMs = workbenchTaskIsRunning ? 700 : 1800;
        const timer = window.setInterval(() => {
            void fetchWorkbenchTaskProgress();
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
        if (!video || !isUploadedMedia) return;
        setVoiceFixerModeDraft(Number(video.voicefixer_mode ?? 0));
        setVoiceFixerMixDraft(Math.max(0, Math.min(1, Number(video.voicefixer_mix_ratio ?? 1))));
        const leveling = String(video.voicefixer_leveling_mode || 'off').toLowerCase();
        setVoiceFixerLevelingDraft(
            leveling === 'gentle' || leveling === 'balanced' || leveling === 'strong' ? leveling : 'off'
        );
        const scope = String(video.voicefixer_apply_scope || (video.voicefixer_use_cleaned ? 'both' : 'none')).toLowerCase();
        setVoiceFixerApplyScopeDraft(
            scope === 'playback' || scope === 'processing' || scope === 'both' ? scope : 'none'
        );
    }, [isUploadedMedia, video]);

    useEffect(() => {
        if (!video || !isUploadedMedia) return;
        setReconstructionInstructionDraft(String(video.reconstruction_instruction_template || ''));
    }, [isUploadedMedia, video]);

    useEffect(() => {
        // Reset deep-link jump state when navigating to a different video.
        initialJumpDoneRef.current = false;
        initialSeekDoneRef.current = false;
        setSearchQuery('');
        setSearchMatchIndex(0);
        setDeepLinkedSegmentId(null);
        setExpandedFunnySummaryIds(new Set());
        setReconstructionWorkbench(null);
        setCleanupWorkbench(null);
        setSelectedReconstructionSpeakerId(null);
        setReconstructionPreviewAudioUrl('');
        setReconstructionPreviewText('');
        setWorkbenchTaskProgress(null);
        setClearVoiceInstallInfo(null);
        setClearVoiceTestResult(null);
        previousLocalMediaUrlRef.current = '';
        pendingNativeSourceRestoreRef.current = null;
    }, [id]);

    useEffect(() => {
        if (!reconstructionWorkbench?.speakers?.length) {
            setSelectedReconstructionSpeakerId(null);
            return;
        }
        setSelectedReconstructionSpeakerId((prev) => {
            if (prev && reconstructionWorkbench.speakers.some((speaker) => speaker.speaker_id === prev)) return prev;
            return reconstructionWorkbench.speakers[0].speaker_id;
        });
        setReconstructionTestTextDrafts((prev) => {
            const next = { ...prev };
            for (const speaker of reconstructionWorkbench.speakers) {
                if (!next[speaker.speaker_id]) {
                    next[speaker.speaker_id] =
                        speaker.latest_test_text ||
                        defaultReconstructionTestText;
                }
            }
            return next;
        });
    }, [defaultReconstructionTestText, reconstructionWorkbench]);

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
        setDeepLinkedSegmentId(hasRequestedSegment ? requestedSegmentId : null);
        setSearchQuery(shouldRestoreSearch ? requestedSearchQuery : '');
        setSearchMatchIndex(0);
        if (hasRequestedSegment || shouldRestoreSearch) {
            setFollowPlayback(false);
            setActiveTab('transcript');
        }
    }, [id, requestedSearchMode, requestedSearchQuery, requestedSegmentId]);

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
            void element.play().catch(() => {});
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
                            void (playAttempt as Promise<void>).catch(() => {});
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

    const formatTime = (seconds: number) => {
        const total = Math.max(0, Math.floor(seconds));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        return h > 0
            ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
            : `${m}:${s.toString().padStart(2, '0')}`;
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

    const buildSpeakerPlaceholder = (speakerId: number, segment?: TranscriptSegment): Speaker | null => {
        if (!video) return null;
        const cached = speakerDetailCacheRef.current.get(speakerId);
        if (cached) return cached;
        return {
            id: speakerId,
            channel_id: video.channel_id,
            name: segment?.speaker || `Speaker ${speakerId}`,
            thumbnail_path: undefined,
            is_extra: false,
            total_speaking_time: 0,
            embedding_count: 0,
            created_at: '',
        };
    };

    const handleSpeakerClick = async (speakerId: number, segment?: TranscriptSegment) => {
        pauseMainPreview();

        if (segment && video) {
            const sample: SpeakerSample = {
                youtube_id: video.youtube_id,
                video_id: video.id,
                start_time: segment.start_time,
                end_time: segment.end_time,
                text: segment.text,
                media_source_type: video.media_source_type,
                media_kind: video.media_kind,
            };
            setInitialSample(sample);
        } else {
            setInitialSample(null);
        }

        const placeholder = buildSpeakerPlaceholder(speakerId, segment);
        if (placeholder) {
            setSelectedSpeaker(placeholder);
        }

        try {
            const res = await api.get<Speaker>(`/speakers/${speakerId}`);
            speakerDetailCacheRef.current.set(speakerId, res.data);
            setSelectedSpeaker(res.data);
        } catch (e) {
            console.error("Failed to fetch speaker details", e);
            if (!placeholder) {
                setSelectedSpeaker(null);
            }
        }
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

    const handleQueueVoiceFixer = async (force = false) => {
        if (!video || !isUploadedMedia) return;
        const prompt = hasVoiceFixerCleaned
            ? 'Rebuild the VoiceFixer-cleaned media for this uploaded episode?'
            : 'Create a VoiceFixer-cleaned copy of this uploaded episode?';
        if (!confirm(prompt)) return;
        setQueueingVoiceFixer(true);
        try {
            const res = await api.post<Video>(`/videos/${video.id}/voicefixer/queue`, null, { params: { force } });
            setVideo(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue VoiceFixer cleanup');
        } finally {
            setQueueingVoiceFixer(false);
        }
    };

    const handleSaveVoiceFixerSettings = async () => {
        if (!video || !isUploadedMedia) return;
        setSavingVoiceFixerSettings(true);
        try {
            const res = await api.patch<Video>(`/videos/${video.id}/voicefixer/settings`, {
                mode: voiceFixerModeDraft,
                mix_ratio: voiceFixerMixDraft,
                leveling_mode: voiceFixerLevelingDraft,
                apply_scope: voiceFixerApplyScopeDraft,
            });
            setVideo(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save VoiceFixer settings');
        } finally {
            setSavingVoiceFixerSettings(false);
        }
    };

    const handleAnalyzeCleanupWorkbench = async () => {
        if (!video || !isUploadedMedia) return;
        setAnalyzingCleanupWorkbench(true);
        try {
            const res = await api.post<CleanupWorkbench>(`/videos/${video.id}/cleanup/workbench/analyze`);
            setCleanupWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to analyze the uploaded audio');
        } finally {
            setAnalyzingCleanupWorkbench(false);
        }
    };

    const handleRunClearVoiceCandidate = async (modelName: string) => {
        if (!video || !isUploadedMedia) return;
        setRunningClearVoiceModel(modelName);
        try {
            const res = await api.post<CleanupWorkbench>(`/videos/${video.id}/cleanup/workbench/clearvoice-candidate`, {
                stage: 'enhancement',
                model_name: modelName,
            });
            setCleanupWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || `Failed to generate the ${modelName} candidate`);
        } finally {
            setRunningClearVoiceModel(null);
        }
    };

    const handleSelectCleanupCandidate = async (candidateId: string | null) => {
        if (!video || !isUploadedMedia) return;
        setSelectingCleanupCandidateId(candidateId || '__original__');
        try {
            const res = await api.patch<CleanupWorkbench>(`/videos/${video.id}/cleanup/workbench/select-candidate`, {
                candidate_id: candidateId,
            });
            setCleanupWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to update the selected pre-cleanup candidate');
        } finally {
            setSelectingCleanupCandidateId(null);
        }
    };

    const handleInstallClearVoice = async () => {
        if (!isUploadedMedia) return;
        setInstallingClearVoice(true);
        try {
            const res = await api.post<ClearVoiceInstallInfo>('/system/clearvoice/install');
            setClearVoiceInstallInfo(res.data);
            setClearVoiceTestResult(null);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to install ClearVoice');
        } finally {
            setInstallingClearVoice(false);
        }
    };

    const handleTestClearVoice = async () => {
        if (!isUploadedMedia) return;
        setTestingClearVoice(true);
        try {
            const res = await api.post<ClearVoiceTestResult>('/system/clearvoice/test');
            setClearVoiceTestResult(res.data);
        } catch (e: any) {
            setClearVoiceTestResult({
                status: 'error',
                imported: false,
                class_available: false,
                torch_imported: false,
                torchaudio_imported: false,
                runtime_ready: false,
                error: e?.response?.data?.detail || 'Failed to test ClearVoice',
                detail: 'The backend could not validate the local ClearVoice runtime.',
            });
        } finally {
            setTestingClearVoice(false);
            await fetchClearVoiceInstallInfo();
        }
    };

    const handleRepairClearVoice = async () => {
        if (!isUploadedMedia) return;
        if (!confirm('Repair the ClearVoice runtime now? This reinstalls torchaudio to match the backend torch build.')) return;
        setRepairingClearVoice(true);
        try {
            await api.post('/system/clearvoice/repair');
            await fetchClearVoiceInstallInfo();
            await handleTestClearVoice();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to repair the ClearVoice runtime');
        } finally {
            setRepairingClearVoice(false);
        }
    };

    const handleSetUploadedPlaybackSource = async (source: UploadedPlaybackSource) => {
        if (!video || !isUploadedMedia || source === currentUploadedPlaybackSource) return;
        setSwitchingReconstructionPlayback(true);
        try {
            const res = await api.patch<Video>(`/videos/${video.id}/playback-source`, { source });
            setVideo(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to switch playback source');
        } finally {
            setSwitchingReconstructionPlayback(false);
        }
    };

    const handleQueueReconstruction = async (force = false) => {
        if (!video || !isUploadedMedia) return;
        const prompt = hasReconstructionAudio
            ? 'Rebuild the reconstructed conversation audio for this uploaded episode?'
            : 'Create reconstructed conversation audio for this uploaded episode?';
        if (!confirm(prompt)) return;
        setQueueingReconstruction(true);
        try {
            const res = await api.post<Video>(`/videos/${video.id}/reconstruct/queue`, null, { params: { force } });
            setVideo(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to queue conversation reconstruction');
        } finally {
            setQueueingReconstruction(false);
        }
    };

    const loadReconstructionWorkbench = async () => {
        if (!video || !isUploadedMedia || segments.length === 0) return;
        setLoadingReconstructionWorkbench(true);
        try {
            const res = await api.get<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench`);
            setReconstructionWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to load the reconstruction workbench');
        } finally {
            setLoadingReconstructionWorkbench(false);
        }
    };

    const handleSaveReconstructionSettings = async () => {
        if (!video || !isUploadedMedia) return;
        setSavingReconstructionSettings(true);
        try {
            const res = await api.patch<Video>(`/videos/${video.id}/reconstruction/settings`, {
                mode: 'performance',
                instruction_template: reconstructionInstructionDraft,
            });
            setVideo(res.data);
            await loadReconstructionWorkbench();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to save reconstruction settings');
        } finally {
            setSavingReconstructionSettings(false);
        }
    };

    const handleTestReconstructionSpeaker = async (
        speakerId: number,
        segmentId?: number,
        options?: { performanceMode?: boolean; useSelectedSampleText?: boolean }
    ) => {
        if (!video || !isUploadedMedia) return;
        setTestingReconstructionSpeakerId(speakerId);
        try {
            const speakerCard = reconstructionWorkbench?.speakers.find(s => s.speaker_id === speakerId) || null;
            const selectedSegment = speakerCard?.samples.find(seg => seg.segment_id === segmentId) || speakerCard?.samples.find(sample => sample.selected) || speakerCard?.samples[0] || null;
            const performanceMode = !!options?.performanceMode;
            const requestedText = options?.useSelectedSampleText
                ? (selectedSegment?.text || speakerCard?.reference_text || '')
                : (reconstructionTestTextDrafts[speakerId] || selectedSegment?.text || speakerCard?.reference_text || '');
            await api.post(`/videos/${video.id}/reconstruction/test-speaker`, {
                speaker_id: speakerId,
                segment_id: selectedSegment?.segment_id,
                text: requestedText,
                performance_mode: performanceMode,
            });
            await loadReconstructionWorkbench();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to generate a reconstruction test for this speaker');
        } finally {
            setTestingReconstructionSpeakerId(null);
        }
    };

    const handleSetReconstructionPlayback = async (enabled: boolean) => {
        await handleSetUploadedPlaybackSource(enabled ? 'reconstructed' : 'original');
    };

    const handleCleanupReconstructionSample = async (speakerId: number, segmentId: number) => {
        if (!video || !isUploadedMedia) return;
        const key = `${speakerId}:${segmentId}`;
        setCleaningReconstructionSampleKey(key);
        try {
            const res = await api.post<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/sample-cleanup`, {
                speaker_id: speakerId,
                segment_id: segmentId,
            });
            setReconstructionWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to clean up this performance sample');
        } finally {
            setCleaningReconstructionSampleKey(null);
        }
    };

    const handleUpdateReconstructionSampleState = async (
        speakerId: number,
        segmentId: number,
        patch: { rejected?: boolean; selected?: boolean; clear_cleaned?: boolean }
    ) => {
        if (!video || !isUploadedMedia) return;
        const key = `${speakerId}:${segmentId}`;
        setUpdatingReconstructionSampleKey(key);
        try {
            const res = await api.patch<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/sample-state`, {
                speaker_id: speakerId,
                segment_id: segmentId,
                ...patch,
            });
            setReconstructionWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to update this performance sample');
        } finally {
            setUpdatingReconstructionSampleKey(null);
        }
    };

    const handleAddReconstructionSample = async (speakerId: number) => {
        if (!video || !isUploadedMedia) return;
        setAddingReconstructionSampleSpeakerId(speakerId);
        try {
            const res = await api.post<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/add-sample`, {
                speaker_id: speakerId,
            });
            setReconstructionWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to add another performance sample');
        } finally {
            setAddingReconstructionSampleSpeakerId(null);
        }
    };

    const handleApproveReconstructionSpeaker = async (speakerId: number, approved: boolean) => {
        if (!video || !isUploadedMedia) return;
        setApprovingReconstructionSpeakerId(speakerId);
        try {
            const res = await api.patch<ReconstructionWorkbench>(`/videos/${video.id}/reconstruction/workbench/speaker-approval`, {
                speaker_id: speakerId,
                approved,
            });
            setReconstructionWorkbench(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to update voice approval');
        } finally {
            setApprovingReconstructionSpeakerId(null);
        }
    };

    const handlePreviewReconstructionSegment = async (segmentId: number) => {
        if (!video || !isUploadedMedia) return;
        setPreviewingReconstructionSegment(true);
        setReconstructionPreviewAudioUrl('');
        setReconstructionPreviewText('');
        try {
            const res = await api.post(`/videos/${video.id}/reconstruction/preview-segment`, {
                segment_id: segmentId,
                performance_mode: true,
            });
            setReconstructionPreviewAudioUrl(resolveWorkbenchAudioUrl(String(res.data.audio_url || '')));
            setReconstructionPreviewText(String(res.data.text || ''));
        } catch (e: any) {
            setReconstructionPreviewAudioUrl('');
            setReconstructionPreviewText('');
            alert(e?.response?.data?.detail || 'Failed to preview this reconstruction segment');
        } finally {
            setPreviewingReconstructionSegment(false);
        }
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
        setAssignSearch('');
        setAssignSpeakers([]);
        setAssignPopup({
            segmentId,
            x: rect.left + window.scrollX,
            y: rect.bottom + window.scrollY
        });
    };

    useEffect(() => {
        if (!assignPopup || !video) return;
        let cancelled = false;
        const timeoutId = window.setTimeout(() => {
            const fetchAssignSpeakers = async () => {
                setAssignLoading(true);
                try {
                    const trimmedSearch = assignSearch.trim();
                    const res = await api.get<Speaker[]>('/speakers', {
                        params: {
                            channel_id: video.channel_id,
                            limit: assignSpeakerPickerLimit,
                            search: trimmedSearch || undefined,
                        },
                    });
                    if (!cancelled) {
                        setAssignSpeakers(Array.isArray(res.data) ? res.data : []);
                    }
                } catch (e) {
                    if (!cancelled) {
                        console.error("Failed to fetch speakers", e);
                    }
                } finally {
                    if (!cancelled) {
                        setAssignLoading(false);
                    }
                }
            };
            void fetchAssignSpeakers();
        }, 200);

        return () => {
            cancelled = true;
            window.clearTimeout(timeoutId);
        };
    }, [assignPopup, assignSearch, video?.channel_id]);

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

    const handleSpeakerListUpdated = (updatedSpeaker: Speaker) => {
        speakerDetailCacheRef.current.set(updatedSpeaker.id, updatedSpeaker);
        setSegments(prev => prev.map(segment =>
            segment.speaker_id === updatedSpeaker.id
                ? { ...segment, speaker: updatedSpeaker.name }
                : segment
        ));
    };

    const handleSpeakerListMerged = () => {
        if (!id) return;
        void api.get<TranscriptSegment[]>(`/videos/${id}/segments`).then(res => setSegments(res.data));
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

    const formatViewMetric = (value?: number | null, fractionDigits: number = 0) => {
        if (value == null || Number.isNaN(value)) return 'Unknown';
        return new Intl.NumberFormat(undefined, {
            maximumFractionDigits: fractionDigits,
            minimumFractionDigits: fractionDigits > 0 ? fractionDigits : 0,
        }).format(value);
    };

    const getCloneVariantNumberSeed = () => {
        let maxVariant = 0;
        for (const job of cloneJobs) {
            const match = String(job.request?.variant_label || '').match(/(\d+)\s*$/);
            if (match) {
                maxVariant = Math.max(maxVariant, Number(match[1] || 0));
            }
        }
        return maxVariant + 1;
    };

    const loadCloneVariantInputs = (job: EpisodeCloneJob) => {
        setCloneStylePrompt(String(job.request?.style_prompt || ''));
        setCloneNotes(String(job.request?.notes || ''));
        const requestProvider = String(job.request?.provider_override || '').trim();
        const providerMatch = cloneEngines.find((engine) => (
            requestProvider
                ? String(engine.provider || '') === requestProvider
                : engine.key === 'default'
        ));
        setCloneEngineKey(providerMatch?.key || (requestProvider ? cloneEngineKey : 'default'));
        setCloneOllamaModel(String(job.request?.model_override || ''));
        setCloneConceptsText((job.request?.approved_concepts || []).join('\n'));
        setCloneExcludedReferencesText((job.request?.excluded_references || []).join('\n'));
    };

    const handleDetectCloneConcepts = async () => {
        if (!id) return;
        const engineOverride = resolveCloneEngineOverride();
        setDetectingCloneConcepts(true);
        setCloneEnginesError(null);
        try {
            const res = await api.post<EpisodeCloneConceptsResponse>(`/videos/${id}/episode-clone/concepts`, {
                notes: cloneNotes.trim() || undefined,
                related_limit: 8,
                provider_override: engineOverride.provider_override,
                model_override: engineOverride.model_override,
            });
            setCloneConcepts(res.data);
            setCloneConceptsText((Array.isArray(res.data?.concepts) ? res.data.concepts : []).join('\n'));
            setCloneExcludedReferencesText((Array.isArray(res.data?.excluded_references) ? res.data.excluded_references : []).join('\n'));
        } catch (e: any) {
            console.error('Failed to detect clone concepts:', e);
            alert(e?.response?.data?.detail || 'Failed to detect clone concepts');
        } finally {
            setDetectingCloneConcepts(false);
        }
    };

    const handleGenerateEpisodeClone = async () => {
        if (!id) return;
        if (!cloneStylePrompt.trim()) {
            alert('Enter a target style prompt first.');
            return;
        }
        const approvedConcepts = normalizeCloneTextList(cloneConceptsText);
        if (approvedConcepts.length === 0) {
            alert('Detect and approve at least one concept before generating a clone.');
            return;
        }
        const engineOverride = resolveCloneEngineOverride();
        const requestId = ++cloneGenerateRequestRef.current;
        setGeneratingClone(true);
        setCloneJobsError(null);
        setCopiedCloneScript(false);
        try {
            const nextVariantSeed = getCloneVariantNumberSeed();
            const jobs: EpisodeCloneJob[] = [];
            for (let idx = 0; idx < cloneBatchSize; idx += 1) {
                const res = await api.post<EpisodeCloneJob>(`/videos/${id}/episode-clone/generate`, {
                    style_prompt: cloneStylePrompt.trim(),
                    notes: cloneNotes.trim() || undefined,
                    related_limit: 8,
                    variant_label: `Variant ${nextVariantSeed + idx}`,
                    provider_override: engineOverride.provider_override,
                    model_override: engineOverride.model_override,
                    approved_concepts: approvedConcepts,
                    excluded_references: normalizeCloneTextList(cloneExcludedReferencesText),
                });
                jobs.push(res.data);
            }
            if (requestId !== cloneGenerateRequestRef.current) return;
            const preferredJobId = jobs[0]?.job_id ?? null;
            setSelectedCloneJobId(preferredJobId);
            setCloneJobs((prev) => {
                const merged = [...jobs, ...prev].filter(
                    (job, index, array) => array.findIndex((candidate) => candidate.job_id === job.job_id) === index
                );
                return merged;
            });
            await fetchCloneJobs(Number(id), undefined, { preferredJobId, silent: true });
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to generate episode clone');
            setGeneratingClone(false);
        }
    };

    const handleCopyCloneScript = async () => {
        if (!cloneDraftResult?.script) return;
        try {
            await navigator.clipboard.writeText(cloneDraftResult.script);
            setCopiedCloneScript(true);
            window.setTimeout(() => setCopiedCloneScript(false), 1500);
        } catch {
            alert('Failed to copy clone script');
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
        if (!confirm(isYoutubeMedia
            ? 'Archive the current description and replace it with the AI-generated YouTube description draft?'
            : 'Archive the current episode description and replace it with the AI-generated summary draft?'
        )) return;
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

    useEffect(() => {
        if (activeTab !== 'cleanup' || !id || !isUploadedMedia) return;
        void fetchCleanupWorkbench();
        void fetchClearVoiceInstallInfo();
    }, [activeTab, id, isUploadedMedia]);
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
                                    className={`inline-flex min-w-[122px] items-center gap-2 rounded-xl border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
                                        active ? option.activeClassName : option.inactiveClassName
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
                                    className={`w-full rounded-2xl border px-4 py-3 text-left transition-colors ${
                                        selectedReconstructionSpeakerId === speaker.speaker_id
                                            ? 'border-violet-300 bg-violet-50 shadow-sm'
                                            : 'border-slate-200 bg-slate-50 hover:border-violet-200 hover:bg-violet-50/70'
                                    }`}
                                >
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="min-w-0">
                                            <div className="truncate text-sm font-semibold text-slate-900">{speaker.speaker_name}</div>
                                            <div className="mt-1 text-xs text-slate-500">{speaker.segment_count} diarized segments</div>
                                        </div>
                                        <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                                            speaker.approved
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
                                            <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                                                selectedSpeaker.approved
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
                                            className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50 ${
                                                selectedSpeaker.approved
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
                                    className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50 ${
                                        usingReconstructionForPlayback
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

    const currentWorkbenchProgress = workbenchTaskProgress && String(workbenchTaskProgress.status || '').toLowerCase() !== 'idle'
        ? workbenchTaskProgress
        : null;
    const cleanupWorkbenchProgress = currentWorkbenchProgress && String(currentWorkbenchProgress.area || '').toLowerCase() === 'cleanup'
        ? currentWorkbenchProgress
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

    const cleanupWorkbenchActivity: WorkbenchActivity | null = loadingCleanupWorkbench
        ? {
            label: 'Loading cleanup workbench',
            detail: 'Refreshing the ClearVoice prep area and current pre-cleanup candidates.',
            tone: 'sky',
        }
        : analyzingCleanupWorkbench
            ? {
                label: 'Analyzing uploaded audio',
                detail: 'Inspecting the original upload so the pre-cleanup bench can suggest the right enhancement tools.',
                tone: 'sky',
            }
            : runningClearVoiceModel !== null
                ? {
                    label: 'Generating ClearVoice candidate',
                    detail: `Running ${runningClearVoiceModel} before the existing VoiceFixer cleanup step.`,
                    tone: 'sky',
                }
                : selectingCleanupCandidateId !== null
                    ? {
                        label: 'Selecting pre-cleanup source',
                        detail: 'Updating which ClearVoice candidate should feed the existing VoiceFixer cleanup pass.',
                        tone: 'sky',
                    }
                    : installingClearVoice
                        ? {
                            label: 'Installing ClearVoice',
                            detail: 'Adding the ClearVoice runtime to the backend environment.',
                            tone: 'sky',
                        }
                        : testingClearVoice
                            ? {
                                label: 'Testing ClearVoice',
                                detail: 'Verifying the ClearVoice runtime before generating enhancement candidates.',
                                tone: 'sky',
                            }
                            : savingVoiceFixerSettings
        ? {
            label: 'Saving cleanup settings',
            detail: 'Updating the cleanup recipe for this episode before the next rebuild.',
            tone: 'sky',
        }
        : queueingVoiceFixer
            ? {
                label: 'Starting cleanup job',
                detail: 'Queueing the VoiceFixer pass now. The stage breakdown appears below once the job is registered.',
                tone: 'sky',
            }
            : null;

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

    const renderCleanupStudio = () => {
        const clearVoiceEnhancementModels = [
            { model: 'FRCRN_SE_16K', label: 'FRCRN 16k', detail: 'Focused on rough, narrowband speech cleanup.' },
            { model: 'MossFormerGAN_SE_16K', label: 'MossFormerGAN 16k', detail: 'Stronger denoising pass for difficult speech.' },
            { model: 'MossFormer2_SE_48K', label: 'MossFormer2 48k', detail: 'Full-band enhancement for higher fidelity sources.' },
        ];
        const clearVoiceInstalled = !!clearVoiceInstallInfo?.installed;
        const clearVoiceNeedsRestart = !!clearVoiceInstallInfo?.restart_required;
        const clearVoiceRuntimeReady = !!clearVoiceInstallInfo?.runtime_ready;
        const selectedPreCleanupLabel = cleanupWorkbench?.selected_source_label || 'Original upload';
        const desiredApplyScopeLabel = voiceFixerApplyScopeDraft === 'both'
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
                                    onClick={() => void handleSaveVoiceFixerSettings()}
                                    disabled={episodeBusy || savingVoiceFixerSettings}
                                    className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-4 py-2.5 text-sm font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                >
                                    {savingVoiceFixerSettings ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                                    Save Settings
                                </button>
                                <button
                                    type="button"
                                    onClick={() => void handleQueueVoiceFixer(hasVoiceFixerCleaned)}
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
                                            onClick={() => void handleAnalyzeCleanupWorkbench()}
                                            disabled={episodeBusy || analyzingCleanupWorkbench}
                                            className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                        >
                                            {analyzingCleanupWorkbench ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                                            Analyze Upload
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void fetchCleanupWorkbench()}
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
                                                onClick={() => void handleRunClearVoiceCandidate(entry.model)}
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
                                                        onClick={() => void handleSelectCleanupCandidate(candidate.selected_for_processing ? null : candidate.candidate_id)}
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
                                        onClick={() => void handleSaveVoiceFixerSettings()}
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
                                            onChange={(e) => setVoiceFixerModeDraft(Number(e.target.value))}
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
                                            onChange={(e) => setVoiceFixerMixDraft(Number(e.target.value) / 100)}
                                            disabled={episodeBusy}
                                            className="w-full accent-sky-600"
                                        />
                                        <div className="mt-1 text-[11px] text-slate-500">Lower values keep more of the original natural tone.</div>
                                    </label>
                                    <label className="text-xs text-slate-600">
                                        <div className="font-medium text-slate-700 mb-1">Voice Leveling</div>
                                        <select
                                            value={voiceFixerLevelingDraft}
                                            onChange={(e) => setVoiceFixerLevelingDraft(e.target.value as 'off' | 'gentle' | 'balanced' | 'strong')}
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
                                            onChange={(e) => setVoiceFixerApplyScopeDraft(e.target.value as 'none' | 'playback' | 'processing' | 'both')}
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
                                        onClick={() => void handleQueueVoiceFixer(hasVoiceFixerCleaned)}
                                        disabled={episodeBusy}
                                        className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-white px-4 py-2.5 text-sm font-medium text-sky-700 hover:bg-sky-100 disabled:opacity-50"
                                    >
                                        {queueingVoiceFixer || voiceFixerBusy ? <Loader2 size={15} className="animate-spin" /> : <AudioLines size={15} />}
                                        {hasVoiceFixerCleaned ? 'Rebuild Cleaned Pass' : 'Create Cleaned Pass'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setActiveTab('transcript')}
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
    };

    const renderReconstructionStudio = () => {
        const speakerCount = reconstructionWorkbench?.speaker_count ?? 0;
        const approvedCount = reconstructionWorkbench?.speakers.filter((speaker) => speaker.approved).length ?? 0;
        const pendingCount = Math.max(0, speakerCount - approvedCount);

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
                                    onClick={() => void loadReconstructionWorkbench()}
                                    disabled={loadingReconstructionWorkbench || segments.length === 0}
                                    className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-4 py-2.5 text-sm font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                                >
                                    {loadingReconstructionWorkbench ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                                    Refresh Workbench
                                </button>
                                <div className="inline-flex rounded-2xl border border-violet-200 bg-violet-50 p-1">
                                    <button
                                        type="button"
                                        onClick={() => setReconstructionStudioTab('voices')}
                                        className={`inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${reconstructionStudioTab === 'voices' ? 'bg-white text-violet-700 shadow-sm' : 'text-violet-700/80 hover:text-violet-800'}`}
                                    >
                                        <Users size={15} />
                                        Voices
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setReconstructionStudioTab('reconstruction')}
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
                                        ? String(video?.reconstruction_error || 'Conversation reconstruction failed.').slice(0, 240)
                                        : usingReconstructionForPlayback
                                            ? 'Reconstructed audio is currently driving local playback, so transcript-follow uses the rebuilt conversation track.'
                                            : hasReconstructionAudio
                                                ? 'Reconstructed audio is ready for preview, download, and optional playback handoff.'
                                                : 'Approve each voice model first, then move into the reconstruction tab for short previews and the final full build.'}
                        </div>
                        {reconstructionWorkbenchProgress ? (
                            <div className="mt-4">
                                {renderWorkbenchTaskProgressCard(reconstructionWorkbenchProgress)}
                            </div>
                        ) : reconstructionWorkbenchActivity ? (
                            <div className="mt-4">
                                {renderWorkbenchActivityCard(reconstructionWorkbenchActivity)}
                            </div>
                        ) : null}
                        {reconstructionJob && renderAuxiliaryProgressCard('reconstruction', reconstructionJob, { className: 'mt-4 border-violet-200 bg-violet-50/50' })}
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
                                <SpeakerList
                                    videoId={video.id}
                                    channelId={undefined}
                                    onSpeakerUpdated={handleSpeakerListUpdated}
                                    onSpeakerMerged={handleSpeakerListMerged}
                                />
                            ) : (
                                <Loader2 className="animate-spin" />
                            )}
                        </div>
                    )}
                    {activeTab === 'reconstruction' && isUploadedMedia && (
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
                                            <span className="font-semibold text-slate-800">{segments.length}</span>
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
                                            onClick={() => void loadReconstructionWorkbench()}
                                            disabled={loadingReconstructionWorkbench || segments.length === 0}
                                            className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                                        >
                                            {loadingReconstructionWorkbench ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                            Refresh Workbench
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleQueueReconstruction(hasReconstructionAudio)}
                                            disabled={episodeBusy || segments.length === 0}
                                            className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                                        >
                                            {queueingReconstruction || reconstructionBusy ? <Loader2 size={14} className="animate-spin" /> : <Bot size={14} />}
                                            {hasReconstructionAudio ? 'Rebuild Reconstruction' : 'Reconstruct Audio'}
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setActiveTab('transcript')}
                                            className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                                        >
                                            <FileText size={14} />
                                            Back to Transcript
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'youtube' && (
                        <div className="h-full overflow-y-auto p-4 space-y-4">
                            <div className="rounded-xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50 p-4">
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-800">
                                            <Bot size={15} className="text-emerald-600" />
                                            {aiMetadataPanelTitle}
                                        </div>
                                        <p className="mt-1 text-xs text-emerald-700/80">
                                            {aiMetadataPanelDescription}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => handleGenerateYoutubeAi(hasYoutubeAiMetadata)}
                                        disabled={generatingYoutubeAi || segments.length === 0}
                                        className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                        title={segments.length === 0 ? 'Transcript required first' : aiMetadataGenerateTitle}
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
                                        title={video.youtube_ai_description_text ? aiMetadataPublishHelp : 'Generate a draft first'}
                                    >
                                        {publishingYoutubeDescription ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                                        {aiMetadataPublishLabel}
                                    </button>
                                    <span className="text-[11px] text-emerald-800/80">
                                        {aiMetadataPublishHelp}
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
                                    Transcript required first. Run transcription/diarization before generating {isYoutubeMedia ? 'AI summary metadata' : 'episode summary metadata'}.
                                </div>
                            ) : !hasYoutubeAiMetadata ? (
                                <div className="rounded-xl border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">
                                    {aiMetadataEmptyText}
                                </div>
                            ) : (
                                <>
                                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                                        <div className="flex items-center justify-between gap-2 mb-2">
                                            <h3 className="text-sm font-semibold text-slate-800">{aiMetadataCurrentDescriptionLabel}</h3>
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
                                            <h3 className="text-sm font-semibold text-slate-800">{aiMetadataChaptersLabel}</h3>
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
                                            <h3 className="text-sm font-semibold text-slate-800">{aiMetadataDescriptionLabel}</h3>
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
                    renderCleanupStudio()
                ) : activeTab === 'optimize' ? (
                    renderTranscriptOptimizationWorkbench()
                ) : activeTab === 'clone' ? (
                    <CloneWorkbenchPanel
                        video={video}
                        segmentsCount={segments.length}
                        cloneEngineKey={cloneEngineKey}
                        onCloneEngineKeyChange={setCloneEngineKey}
                        cloneEngines={cloneEngines}
                        loadingCloneEngines={loadingCloneEngines}
                        cloneEnginesError={cloneEnginesError}
                        cloneUsesOllama={cloneUsesOllama}
                        cloneOllamaModel={cloneOllamaModel}
                        onCloneOllamaModelChange={setCloneOllamaModel}
                        cloneOllamaModels={cloneOllamaModels}
                        loadingCloneOllamaModels={loadingCloneOllamaModels}
                        cloneOllamaModelsError={cloneOllamaModelsError}
                        detectingCloneConcepts={detectingCloneConcepts}
                        onDetectConcepts={() => void handleDetectCloneConcepts()}
                        cloneConcepts={cloneConcepts}
                        cloneConceptsText={cloneConceptsText}
                        onCloneConceptsTextChange={setCloneConceptsText}
                        cloneExcludedReferencesText={cloneExcludedReferencesText}
                        onCloneExcludedReferencesTextChange={setCloneExcludedReferencesText}
                        cloneStylePrompt={cloneStylePrompt}
                        onCloneStylePromptChange={setCloneStylePrompt}
                        cloneNotes={cloneNotes}
                        onCloneNotesChange={setCloneNotes}
                        cloneBatchSize={cloneBatchSize}
                        onCloneBatchSizeChange={setCloneBatchSize}
                        generatingClone={generatingClone}
                        onGenerate={() => void handleGenerateEpisodeClone()}
                        cloneJobs={cloneJobs}
                        loadingCloneJobs={loadingCloneJobs}
                        cloneJobsError={cloneJobsError}
                        selectedCloneJobId={selectedCloneJobId}
                        onSelectCloneJob={setSelectedCloneJobId}
                        onLoadCloneVariantInputs={loadCloneVariantInputs}
                        cloneDraft={cloneDraftResult}
                        copiedCloneScript={copiedCloneScript}
                        onCopyCloneScript={() => void handleCopyCloneScript()}
                        cloneJobMatchesVisibleInputs={cloneJobMatchesVisibleInputs}
                        formatViewMetric={formatViewMetric}
                        formatTime={formatTime}
                    />
                ) : activeTab === 'reconstruction' && isUploadedMedia ? (
                    renderReconstructionStudio()
                ) : showClipEditorMain && activeEditingClip && clipEditorDraft ? (
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
                                {renderMainPlayer("w-full bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video")}
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
                    initialSample={initialSample || undefined}
                    onClose={() => { setSelectedSpeaker(null); setInitialSample(null); }}
                    onUpdate={(updatedSpeaker) => {
                        speakerDetailCacheRef.current.set(updatedSpeaker.id, updatedSpeaker);
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
