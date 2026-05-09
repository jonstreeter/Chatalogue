import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
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
