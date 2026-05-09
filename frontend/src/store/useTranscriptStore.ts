import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
    FunnyMoment,
    TranscriptQuality,
    TranscriptRollbackOption,
    TranscriptGoldWindow,
    TranscriptEvaluationResult,
    TranscriptEvaluationReview,
    TranscriptEvaluationBatchResponse,
} from '../types';

type SetStateValue<T> = T | ((previous: T) => T);

function resolveValue<T>(value: SetStateValue<T>, previous: T): T {
    return typeof value === 'function' ? (value as (previous: T) => T)(previous) : value;
}

export interface TranscriptState {
    funnyMoments: FunnyMoment[];
    searchQuery: string;
    deepLinkedSegmentId: number | null;
    searchMatchIndex: number;
    followPlayback: boolean;
    loadingFunnyMoments: boolean;
    detectingFunnyMoments: boolean;
    funnyDrawerOpen: boolean;
    explainingFunnyMoments: boolean;
    showGlobalHumorContext: boolean;
    expandedFunnySummaryIds: Set<number>;
    funnyTaskProgress: {
        video_id: number;
        task?: 'detect' | 'explain' | string;
        status: 'idle' | 'running' | 'completed' | 'error' | string;
        stage?: string | null;
        message?: string | null;
        percent?: number | null;
        current?: number | null;
        total?: number | null;
    } | null;
    editingSegmentId: number | null;
    editingSegmentWords: string[];
    editingLoopSegment: boolean;
    savingSegmentEdit: boolean;
    transcriptQuality: TranscriptQuality | null;
    loadingTranscriptQuality: boolean;
    transcriptQualityError: string | null;
    transcriptRollbackOptions: TranscriptRollbackOption[];
    loadingTranscriptRollbackOptions: boolean;
    restoringTranscriptRunId: number | null;
    transcriptGoldWindows: TranscriptGoldWindow[];
    loadingTranscriptGoldWindows: boolean;
    transcriptGoldWindowsError: string | null;
    savingTranscriptGoldWindow: boolean;
    evaluatingTranscript: boolean;
    transcriptEvaluationSummary: TranscriptEvaluationBatchResponse | null;
    transcriptEvaluationResults: TranscriptEvaluationResult[];
    loadingTranscriptEvaluationResults: boolean;
    transcriptEvaluationError: string | null;
    reviewingEvaluationResultId: number | null;
    evaluationReviewsByResultId: Record<number, TranscriptEvaluationReview[]>;
    goldWindowLabelDraft: string;
    goldWindowStartDraft: string;
    goldWindowEndDraft: string;
    goldWindowReferenceDraft: string;
    goldWindowEntitiesDraft: string;
    goldWindowNotesDraft: string;
    evaluationReviewVerdictDrafts: Record<number, string>;
    evaluationReviewNotesDrafts: Record<number, string>;
    evaluationReviewReviewerDrafts: Record<number, string>;
    queueingTranscriptRepair: boolean;
    queueingDiarizationRebuild: boolean;
    queueingDiarizationBenchmark: boolean;
    queueingFullRetranscription: boolean;
    diarizationBenchmarkSensitivity: 'aggressive' | 'balanced' | 'conservative';
    diarizationBenchmarkThreshold: string;

    setFunnyMoments: (value: FunnyMoment[]) => void;
    setSearchQuery: (value: string) => void;
    setDeepLinkedSegmentId: (value: number | null) => void;
    setSearchMatchIndex: (value: SetStateValue<number>) => void;
    setFollowPlayback: (value: boolean) => void;
    setLoadingFunnyMoments: (value: boolean) => void;
    setDetectingFunnyMoments: (value: boolean) => void;
    setFunnyDrawerOpen: (value: SetStateValue<boolean>) => void;
    setExplainingFunnyMoments: (value: boolean) => void;
    setShowGlobalHumorContext: (value: SetStateValue<boolean>) => void;
    setExpandedFunnySummaryIds: (value: SetStateValue<Set<number>>) => void;
    setFunnyTaskProgress: (value: TranscriptState['funnyTaskProgress']) => void;
    setEditingSegmentId: (value: number | null) => void;
    setEditingSegmentWords: (value: SetStateValue<string[]>) => void;
    setEditingLoopSegment: (value: boolean) => void;
    setSavingSegmentEdit: (value: boolean) => void;
    setTranscriptQuality: (value: TranscriptQuality | null) => void;
    setLoadingTranscriptQuality: (value: boolean) => void;
    setTranscriptQualityError: (value: string | null) => void;
    setTranscriptRollbackOptions: (value: TranscriptRollbackOption[]) => void;
    setLoadingTranscriptRollbackOptions: (value: boolean) => void;
    setRestoringTranscriptRunId: (value: number | null) => void;
    setTranscriptGoldWindows: (value: TranscriptGoldWindow[]) => void;
    setLoadingTranscriptGoldWindows: (value: boolean) => void;
    setTranscriptGoldWindowsError: (value: string | null) => void;
    setSavingTranscriptGoldWindow: (value: boolean) => void;
    setEvaluatingTranscript: (value: boolean) => void;
    setTranscriptEvaluationSummary: (value: TranscriptEvaluationBatchResponse | null) => void;
    setTranscriptEvaluationResults: (value: TranscriptEvaluationResult[]) => void;
    setLoadingTranscriptEvaluationResults: (value: boolean) => void;
    setTranscriptEvaluationError: (value: string | null) => void;
    setReviewingEvaluationResultId: (value: number | null) => void;
    setEvaluationReviewsByResultId: (value: SetStateValue<Record<number, TranscriptEvaluationReview[]>>) => void;
    setGoldWindowLabelDraft: (value: SetStateValue<string>) => void;
    setGoldWindowStartDraft: (value: string) => void;
    setGoldWindowEndDraft: (value: string) => void;
    setGoldWindowReferenceDraft: (value: string) => void;
    setGoldWindowEntitiesDraft: (value: string) => void;
    setGoldWindowNotesDraft: (value: string) => void;
    setEvaluationReviewVerdictDrafts: (value: SetStateValue<Record<number, string>>) => void;
    setEvaluationReviewNotesDrafts: (value: SetStateValue<Record<number, string>>) => void;
    setEvaluationReviewReviewerDrafts: (value: SetStateValue<Record<number, string>>) => void;
    setQueueingTranscriptRepair: (value: boolean) => void;
    setQueueingDiarizationRebuild: (value: boolean) => void;
    setQueueingDiarizationBenchmark: (value: boolean) => void;
    setQueueingFullRetranscription: (value: boolean) => void;
    setDiarizationBenchmarkSensitivity: (value: 'aggressive' | 'balanced' | 'conservative') => void;
    setDiarizationBenchmarkThreshold: (value: string) => void;
    resetTranscriptState: () => void;
}

const initialTranscriptState = {
    funnyMoments: [],
    searchQuery: '',
    deepLinkedSegmentId: null,
    searchMatchIndex: 0,
    followPlayback: true,
    loadingFunnyMoments: false,
    detectingFunnyMoments: false,
    funnyDrawerOpen: false,
    explainingFunnyMoments: false,
    showGlobalHumorContext: false,
    expandedFunnySummaryIds: new Set<number>(),
    funnyTaskProgress: null,
    editingSegmentId: null,
    editingSegmentWords: [],
    editingLoopSegment: false,
    savingSegmentEdit: false,
    transcriptQuality: null,
    loadingTranscriptQuality: false,
    transcriptQualityError: null,
    transcriptRollbackOptions: [],
    loadingTranscriptRollbackOptions: false,
    restoringTranscriptRunId: null,
    transcriptGoldWindows: [],
    loadingTranscriptGoldWindows: false,
    transcriptGoldWindowsError: null,
    savingTranscriptGoldWindow: false,
    evaluatingTranscript: false,
    transcriptEvaluationSummary: null,
    transcriptEvaluationResults: [],
    loadingTranscriptEvaluationResults: false,
    transcriptEvaluationError: null,
    reviewingEvaluationResultId: null,
    evaluationReviewsByResultId: {},
    goldWindowLabelDraft: 'Gold Window',
    goldWindowStartDraft: '',
    goldWindowEndDraft: '',
    goldWindowReferenceDraft: '',
    goldWindowEntitiesDraft: '',
    goldWindowNotesDraft: '',
    evaluationReviewVerdictDrafts: {},
    evaluationReviewNotesDrafts: {},
    evaluationReviewReviewerDrafts: {},
    queueingTranscriptRepair: false,
    queueingDiarizationRebuild: false,
    queueingDiarizationBenchmark: false,
    queueingFullRetranscription: false,
    diarizationBenchmarkSensitivity: 'balanced' as const,
    diarizationBenchmarkThreshold: '0.35',
};

export const useTranscriptStore = create<TranscriptState>()(
    devtools(
        (set, get) => ({
            ...initialTranscriptState,
            setFunnyMoments: (value) => set({ funnyMoments: value }, false, 'setFunnyMoments'),
            setSearchQuery: (value) => set({ searchQuery: value }, false, 'setSearchQuery'),
            setDeepLinkedSegmentId: (value) => set({ deepLinkedSegmentId: value }, false, 'setDeepLinkedSegmentId'),
            setSearchMatchIndex: (value) => set({ searchMatchIndex: resolveValue(value, get().searchMatchIndex) }, false, 'setSearchMatchIndex'),
            setFollowPlayback: (value) => set({ followPlayback: value }, false, 'setFollowPlayback'),
            setLoadingFunnyMoments: (value) => set({ loadingFunnyMoments: value }, false, 'setLoadingFunnyMoments'),
            setDetectingFunnyMoments: (value) => set({ detectingFunnyMoments: value }, false, 'setDetectingFunnyMoments'),
            setFunnyDrawerOpen: (value) => set({ funnyDrawerOpen: resolveValue(value, get().funnyDrawerOpen) }, false, 'setFunnyDrawerOpen'),
            setExplainingFunnyMoments: (value) => set({ explainingFunnyMoments: value }, false, 'setExplainingFunnyMoments'),
            setShowGlobalHumorContext: (value) => set({ showGlobalHumorContext: resolveValue(value, get().showGlobalHumorContext) }, false, 'setShowGlobalHumorContext'),
            setExpandedFunnySummaryIds: (value) => set({ expandedFunnySummaryIds: resolveValue(value, get().expandedFunnySummaryIds) }, false, 'setExpandedFunnySummaryIds'),
            setFunnyTaskProgress: (value) => set({ funnyTaskProgress: value }, false, 'setFunnyTaskProgress'),
            setEditingSegmentId: (value) => set({ editingSegmentId: value }, false, 'setEditingSegmentId'),
            setEditingSegmentWords: (value) => set({ editingSegmentWords: resolveValue(value, get().editingSegmentWords) }, false, 'setEditingSegmentWords'),
            setEditingLoopSegment: (value) => set({ editingLoopSegment: value }, false, 'setEditingLoopSegment'),
            setSavingSegmentEdit: (value) => set({ savingSegmentEdit: value }, false, 'setSavingSegmentEdit'),
            setTranscriptQuality: (value) => set({ transcriptQuality: value }, false, 'setTranscriptQuality'),
            setLoadingTranscriptQuality: (value) => set({ loadingTranscriptQuality: value }, false, 'setLoadingTranscriptQuality'),
            setTranscriptQualityError: (value) => set({ transcriptQualityError: value }, false, 'setTranscriptQualityError'),
            setTranscriptRollbackOptions: (value) => set({ transcriptRollbackOptions: value }, false, 'setTranscriptRollbackOptions'),
            setLoadingTranscriptRollbackOptions: (value) => set({ loadingTranscriptRollbackOptions: value }, false, 'setLoadingTranscriptRollbackOptions'),
            setRestoringTranscriptRunId: (value) => set({ restoringTranscriptRunId: value }, false, 'setRestoringTranscriptRunId'),
            setTranscriptGoldWindows: (value) => set({ transcriptGoldWindows: value }, false, 'setTranscriptGoldWindows'),
            setLoadingTranscriptGoldWindows: (value) => set({ loadingTranscriptGoldWindows: value }, false, 'setLoadingTranscriptGoldWindows'),
            setTranscriptGoldWindowsError: (value) => set({ transcriptGoldWindowsError: value }, false, 'setTranscriptGoldWindowsError'),
            setSavingTranscriptGoldWindow: (value) => set({ savingTranscriptGoldWindow: value }, false, 'setSavingTranscriptGoldWindow'),
            setEvaluatingTranscript: (value) => set({ evaluatingTranscript: value }, false, 'setEvaluatingTranscript'),
            setTranscriptEvaluationSummary: (value) => set({ transcriptEvaluationSummary: value }, false, 'setTranscriptEvaluationSummary'),
            setTranscriptEvaluationResults: (value) => set({ transcriptEvaluationResults: value }, false, 'setTranscriptEvaluationResults'),
            setLoadingTranscriptEvaluationResults: (value) => set({ loadingTranscriptEvaluationResults: value }, false, 'setLoadingTranscriptEvaluationResults'),
            setTranscriptEvaluationError: (value) => set({ transcriptEvaluationError: value }, false, 'setTranscriptEvaluationError'),
            setReviewingEvaluationResultId: (value) => set({ reviewingEvaluationResultId: value }, false, 'setReviewingEvaluationResultId'),
            setEvaluationReviewsByResultId: (value) =>
                set({ evaluationReviewsByResultId: resolveValue(value, get().evaluationReviewsByResultId) }, false, 'setEvaluationReviewsByResultId'),
            setGoldWindowLabelDraft: (value) => set({ goldWindowLabelDraft: resolveValue(value, get().goldWindowLabelDraft) }, false, 'setGoldWindowLabelDraft'),
            setGoldWindowStartDraft: (value) => set({ goldWindowStartDraft: value }, false, 'setGoldWindowStartDraft'),
            setGoldWindowEndDraft: (value) => set({ goldWindowEndDraft: value }, false, 'setGoldWindowEndDraft'),
            setGoldWindowReferenceDraft: (value) => set({ goldWindowReferenceDraft: value }, false, 'setGoldWindowReferenceDraft'),
            setGoldWindowEntitiesDraft: (value) => set({ goldWindowEntitiesDraft: value }, false, 'setGoldWindowEntitiesDraft'),
            setGoldWindowNotesDraft: (value) => set({ goldWindowNotesDraft: value }, false, 'setGoldWindowNotesDraft'),
            setEvaluationReviewVerdictDrafts: (value) =>
                set({ evaluationReviewVerdictDrafts: resolveValue(value, get().evaluationReviewVerdictDrafts) }, false, 'setEvaluationReviewVerdictDrafts'),
            setEvaluationReviewNotesDrafts: (value) =>
                set({ evaluationReviewNotesDrafts: resolveValue(value, get().evaluationReviewNotesDrafts) }, false, 'setEvaluationReviewNotesDrafts'),
            setEvaluationReviewReviewerDrafts: (value) =>
                set({ evaluationReviewReviewerDrafts: resolveValue(value, get().evaluationReviewReviewerDrafts) }, false, 'setEvaluationReviewReviewerDrafts'),
            setQueueingTranscriptRepair: (value) => set({ queueingTranscriptRepair: value }, false, 'setQueueingTranscriptRepair'),
            setQueueingDiarizationRebuild: (value) => set({ queueingDiarizationRebuild: value }, false, 'setQueueingDiarizationRebuild'),
            setQueueingDiarizationBenchmark: (value) => set({ queueingDiarizationBenchmark: value }, false, 'setQueueingDiarizationBenchmark'),
            setQueueingFullRetranscription: (value) => set({ queueingFullRetranscription: value }, false, 'setQueueingFullRetranscription'),
            setDiarizationBenchmarkSensitivity: (value) => set({ diarizationBenchmarkSensitivity: value }, false, 'setDiarizationBenchmarkSensitivity'),
            setDiarizationBenchmarkThreshold: (value) => set({ diarizationBenchmarkThreshold: value }, false, 'setDiarizationBenchmarkThreshold'),
            resetTranscriptState: () => set({ ...initialTranscriptState }, false, 'resetTranscriptState'),
        }),
        { name: 'TranscriptStore' }
    )
);
