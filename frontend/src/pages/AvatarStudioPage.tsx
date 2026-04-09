import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import api, { toApiUrl } from '../lib/api';
import type { AvatarPersonalityBaseModelSupport, AvatarPersonalityDataset, AvatarPersonalityDatasetExample, AvatarPersonalityDatasetPage, AvatarPersonalityFitCheckResponse, AvatarPersonalityJudgeStatus, AvatarPersonalityLongFormConfig, AvatarPersonalityLongFormPage, AvatarPersonalityTestChatResponse, AvatarPersonalityTestChatTurn, AvatarPersonalityTrainingConfig, AvatarPersonalityTrainingPackage, AvatarPersonalityTrainingPlan, AvatarPersonalityTrainingStatus, AvatarWorkbench, SemanticSearchHit } from '../types';
import { AlertTriangle, ArrowLeft, Brain, Check, Clock, Image as ImageIcon, Layers, Loader2, Mic2, Radio, RefreshCw, Save, Search, Square, X, Zap } from 'lucide-react';

type ExampleFilter = AvatarPersonalityDatasetPage['state_filter'];

const RECOMMENDED_TRAINING_CONFIG: AvatarPersonalityTrainingConfig = {
    base_model_id: 'Qwen/Qwen3-8B',
    dataset_profile: 'balanced',
    training_strength: 'balanced',
    export_strategy: 'gold_balanced',
    validation_ratio: 0.10,
    max_examples: 2500,
    max_long_form_examples: 80,
    include_long_form: true,
    training_mode: 'memory_optimized',
    snapshot_interval_steps: 0,
    dataset_profiles: [],
    training_plan: null,
};

const RECOMMENDED_LONG_FORM_TAKE_COUNT = 80;
const TRAINING_STRENGTH_OPTIONS: Array<{
    value: AvatarPersonalityTrainingConfig['training_strength'];
    label: string;
    summary: string;
}> = [
    {
        value: 'conservative',
        label: 'Conservative',
        summary: 'Safer first pass with lower overfitting risk and weaker style push.',
    },
    {
        value: 'balanced',
        label: 'Balanced',
        summary: 'Recommended default for most personality runs.',
    },
    {
        value: 'strong',
        label: 'Strong',
        summary: 'More aggressive style adaptation when the speaker still feels too generic.',
    },
];
const SNAPSHOT_INTERVAL_OPTIONS = [
    { value: 0, label: 'Auto (~5 snapshots per run)' },
    { value: 100, label: 'Every 100 steps' },
    { value: 250, label: 'Every 250 steps' },
    { value: 500, label: 'Every 500 steps' },
    { value: 1000, label: 'Every 1000 steps' },
    { value: 2000, label: 'Every 2000 steps' },
];

const DEFAULT_DATASET_PROFILE_OPTIONS: AvatarPersonalityTrainingConfig['dataset_profiles'] = [
    {
        key: 'focused',
        label: 'Focused',
        summary: 'Smaller, faster package with lower memorization risk.',
        conversation_target: 1000,
        long_form_target: 32,
        pros: [],
        cons: [],
        recommended: false,
    },
    {
        key: 'balanced',
        label: 'Balanced',
        summary: 'Recommended default with enough breadth for style and reasoning.',
        conversation_target: 2500,
        long_form_target: 80,
        pros: [],
        cons: [],
        recommended: true,
    },
    {
        key: 'broad',
        label: 'Broad',
        summary: 'Wider package for speakers with more varied topics and references.',
        conversation_target: 4000,
        long_form_target: 120,
        pros: [],
        cons: [],
        recommended: false,
    },
    {
        key: 'exhaustive',
        label: 'Exhaustive',
        summary: 'Largest preset. Best reserved for very clean datasets.',
        conversation_target: 6000,
        long_form_target: 160,
        pros: [],
        cons: [],
        recommended: false,
    },
    {
        key: 'custom',
        label: 'Custom',
        summary: 'Manual conversation and long-form caps.',
        conversation_target: 0,
        long_form_target: 0,
        pros: [],
        cons: [],
        recommended: false,
    },
];

function formatDuration(seconds: number): string {
    const total = Math.max(0, Math.round(seconds || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatClock(seconds: number): string {
    const total = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatHoursMinutes(seconds: number | null): string {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) {
        return '—';
    }
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
}

function statusTone(status: string): string {
    const normalized = String(status || '').toLowerCase();
    if (normalized.includes('ready')) return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    if (normalized.includes('needs')) return 'bg-amber-50 text-amber-700 border-amber-200';
    if (normalized.includes('draft')) return 'bg-slate-100 text-slate-700 border-slate-200';
    return 'bg-blue-50 text-blue-700 border-blue-200';
}

function autoLabelTone(label: string): string {
    if (label === 'gold') return 'border-amber-200 bg-amber-50 text-amber-700';
    if (label === 'reject') return 'border-rose-200 bg-rose-50 text-rose-700';
    return 'border-blue-200 bg-blue-50 text-blue-700';
}

function readinessTone(status: string): string {
    if (status === 'strong') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    if (status === 'ready') return 'border-blue-200 bg-blue-50 text-blue-700';
    if (status === 'borderline') return 'border-amber-200 bg-amber-50 text-amber-700';
    if (status === 'oversized') return 'border-fuchsia-200 bg-fuchsia-50 text-fuchsia-700';
    return 'border-rose-200 bg-rose-50 text-rose-700';
}

function manualReviewTone(roi: string): string {
    if (roi === 'low') return 'border-slate-200 bg-slate-100 text-slate-700';
    if (roi === 'medium') return 'border-blue-200 bg-blue-50 text-blue-700';
    return 'border-amber-200 bg-amber-50 text-amber-700';
}

function reasonLabel(reason: string): string {
    return String(reason || '')
        .replaceAll('_', ' ')
        .replace(/\b\w/g, (match) => match.toUpperCase());
}

function trainingStrategyLabel(strategy: string): string {
    if (strategy === 'gold_only') return 'Gold Only';
    if (strategy === 'gold_plus_top_silver') return 'Gold + Top Silver';
    if (strategy === 'full_approved') return 'Full Approved';
    return 'Gold Balanced';
}

function trainingStrengthLearningRate(strength: AvatarPersonalityTrainingConfig['training_strength'] | string | undefined): number {
    if (strength === 'conservative') return 0.00003;
    if (strength === 'strong') return 0.00008;
    return 0.00005;
}

function trainingStrengthLabel(strength: AvatarPersonalityTrainingConfig['training_strength'] | string | undefined): string {
    if (strength === 'conservative') return 'Conservative';
    if (strength === 'strong') return 'Strong';
    return 'Balanced';
}

function trainingStrengthTone(strength: AvatarPersonalityTrainingConfig['training_strength'] | string | undefined): string {
    if (strength === 'conservative') return 'border-sky-200 bg-sky-50 text-sky-700';
    if (strength === 'strong') return 'border-amber-200 bg-amber-50 text-amber-700';
    return 'border-emerald-200 bg-emerald-50 text-emerald-700';
}

function trainingStepBandTone(stepBand: AvatarPersonalityTrainingPlan['step_band'] | string | undefined): string {
    if (stepBand === 'light') return 'border-sky-200 bg-sky-50 text-sky-700';
    if (stepBand === 'heavy') return 'border-amber-200 bg-amber-50 text-amber-700';
    if (stepBand === 'aggressive') return 'border-rose-200 bg-rose-50 text-rose-700';
    return 'border-emerald-200 bg-emerald-50 text-emerald-700';
}

function trainingStepBandLabel(stepBand: AvatarPersonalityTrainingPlan['step_band'] | string | undefined): string {
    if (stepBand === 'light') return 'Light';
    if (stepBand === 'heavy') return 'Heavy';
    if (stepBand === 'aggressive') return 'Aggressive';
    return 'Ideal';
}

function fitCheckTone(classification: AvatarPersonalityFitCheckResponse['classification'] | string | undefined): string {
    if (classification === 'underfit') return 'border-sky-200 bg-sky-50 text-sky-700';
    if (classification === 'overfit') return 'border-rose-200 bg-rose-50 text-rose-700';
    if (classification === 'balanced') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    return 'border-slate-200 bg-slate-50 text-slate-700';
}

function fitCheckLabel(classification: AvatarPersonalityFitCheckResponse['classification'] | string | undefined): string {
    if (classification === 'underfit') return 'Likely Underfit';
    if (classification === 'overfit') return 'Likely Overfit';
    if (classification === 'balanced') return 'Likely Balanced';
    return 'Unclear';
}

function escapeHtml(value: string): string {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

interface ParsedDatasetContextTurn {
    speakerName: string;
    text: string;
}

interface ParsedDatasetContext {
    episodeTitle: string;
    turns: ParsedDatasetContextTurn[];
}

function parseDatasetContext(raw: string): ParsedDatasetContext {
    const lines = String(raw || '')
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);

    let episodeTitle = '';
    const turns: ParsedDatasetContextTurn[] = [];

    for (const line of lines) {
        if (line.startsWith('Podcast episode:')) {
            episodeTitle = line.replace('Podcast episode:', '').trim();
            continue;
        }
        if (line === 'Conversation context:') continue;
        const separatorIndex = line.indexOf(':');
        if (separatorIndex > 0) {
            turns.push({
                speakerName: line.slice(0, separatorIndex).trim(),
                text: line.slice(separatorIndex + 1).trim(),
            });
            continue;
        }
        turns.push({
            speakerName: 'Context',
            text: line,
        });
    }

    return { episodeTitle, turns };
}

function ExampleConversationCard({
    example,
    responseSpeakerName,
}: {
    example: AvatarPersonalityDatasetPage['items'][number] | AvatarPersonalityDataset['preview_examples'][number];
    responseSpeakerName: string;
}) {
    const parsed = parseDatasetContext(example.context_text);

    return (
        <div className={`rounded-2xl border p-4 ${example.state === 'rejected' ? 'border-rose-200 bg-rose-50/50' : 'border-slate-200 bg-white'}`}>
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
                <div className="space-y-3">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Episode</div>
                        <div className="mt-1 text-sm font-semibold text-slate-900">{parsed.episodeTitle || example.video_title}</div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Conversation Context</div>
                        <div className="mt-3 space-y-2">
                            {parsed.turns.map((turn, index) => (
                                <div key={`${turn.speakerName}-${index}`} className="rounded-lg border border-slate-200 bg-white px-3 py-2.5">
                                    <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{turn.speakerName}</div>
                                    <div className="mt-1 whitespace-pre-wrap text-sm leading-6 text-slate-700">{turn.text}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="space-y-3">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Response Speaker</div>
                        <div className="mt-1 text-sm font-semibold text-slate-900">{responseSpeakerName}</div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-white p-4">
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Target Response</div>
                        <div className="mt-2 whitespace-pre-wrap text-[15px] leading-7 text-slate-900">{example.response_text}</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

interface SectionCardProps {
    title: string;
    icon: React.ReactNode;
    status: string;
    sourceLabel: string;
    sourceCount: number;
    approvedLabel: string;
    approvedCount: number;
    artifactReady: boolean;
    summary?: string;
    compactStats?: boolean;
    children?: React.ReactNode;
}

function SectionCard({
    title,
    icon,
    status,
    sourceLabel,
    sourceCount,
    approvedLabel,
    approvedCount,
    artifactReady,
    summary,
    compactStats,
    children,
}: SectionCardProps) {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
                        {icon}
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
                        <p className="text-sm text-slate-500">{summary || 'Configuration pending.'}</p>
                    </div>
                </div>
                <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${statusTone(status)}`}>
                    {status.replaceAll('_', ' ')}
                </span>
            </div>

            {compactStats ? (
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                    <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                        <span className="font-semibold text-slate-900">{sourceCount}</span> {sourceLabel.toLowerCase()}
                    </div>
                    <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                        <span className="font-semibold text-slate-900">{approvedCount}</span> {approvedLabel.toLowerCase()}
                    </div>
                    <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                        Artifact: <span className="font-semibold text-slate-900">{artifactReady ? 'Ready' : 'Not built'}</span>
                    </div>
                </div>
            ) : (
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-500">{sourceLabel}</div>
                        <div className="mt-1 text-2xl font-semibold text-slate-900">{sourceCount}</div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-500">{approvedLabel}</div>
                        <div className="mt-1 text-2xl font-semibold text-slate-900">{approvedCount}</div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-500">Artifact</div>
                        <div className="mt-1 text-sm font-semibold text-slate-900">{artifactReady ? 'Ready' : 'Not Built'}</div>
                    </div>
                </div>
            )}

            {children && <div className="mt-4">{children}</div>}
        </section>
    );
}

export function AvatarStudioPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [workbench, setWorkbench] = useState<AvatarWorkbench | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [nameDraft, setNameDraft] = useState('');
    const [descriptionDraft, setDescriptionDraft] = useState('');
    const [saving, setSaving] = useState(false);
    const [dataset, setDataset] = useState<AvatarPersonalityDataset | null>(null);
    const [loadingDataset, setLoadingDataset] = useState(false);
    const [buildingDataset, setBuildingDataset] = useState(false);
    const [judgingDataset, setJudgingDataset] = useState(false);
    const [judgeStatus, setJudgeStatus] = useState<AvatarPersonalityJudgeStatus | null>(null);
    const [datasetPage, setDatasetPage] = useState<AvatarPersonalityDatasetPage | null>(null);
    const [longFormPage, setLongFormPage] = useState<AvatarPersonalityLongFormPage | null>(null);
    const [longFormConfig, setLongFormConfig] = useState<AvatarPersonalityLongFormConfig | null>(null);
    const [trainingConfig, setTrainingConfig] = useState<AvatarPersonalityTrainingConfig | null>(null);
    const [trainingPackage, setTrainingPackage] = useState<AvatarPersonalityTrainingPackage | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<AvatarPersonalityTrainingStatus | null>(null);
    const [baseModelSupport, setBaseModelSupport] = useState<AvatarPersonalityBaseModelSupport | null>(null);
    const [showAdvancedTraining, setShowAdvancedTraining] = useState(false);
    const [loadingExamples, setLoadingExamples] = useState(false);
    const [loadingLongForm, setLoadingLongForm] = useState(false);
    const [loadingTrainingConfig, setLoadingTrainingConfig] = useState(false);
    const [loadingTrainingPackage, setLoadingTrainingPackage] = useState(false);
    const [exampleStateFilter, setExampleStateFilter] = useState<ExampleFilter>('needs_review');
    const [exampleOffset, setExampleOffset] = useState(0);
    const [updatingExampleId, setUpdatingExampleId] = useState<number | null>(null);
    const [selectedExampleId, setSelectedExampleId] = useState<number | null>(null);
    const [similarPassages, setSimilarPassages] = useState<SemanticSearchHit[] | null>(null);
    const [loadingSimilar, setLoadingSimilar] = useState(false);
    const [similarError, setSimilarError] = useState<string | null>(null);
    const [duplicateGroupExamples, setDuplicateGroupExamples] = useState<AvatarPersonalityDatasetExample[] | null>(null);
    const [loadingDupGroup, setLoadingDupGroup] = useState(false);
    const [selectedLongFormId, setSelectedLongFormId] = useState<string | null>(null);
    const [updatingLongFormId, setUpdatingLongFormId] = useState<string | null>(null);
    const [savingLongFormCount, setSavingLongFormCount] = useState(false);
    const [savingTrainingConfig, setSavingTrainingConfig] = useState(false);
    const [preparingTrainingPackage, setPreparingTrainingPackage] = useState(false);
    const [resettingRecommended, setResettingRecommended] = useState(false);
    const [downloadingBaseModel, setDownloadingBaseModel] = useState(false);
    const [startingTraining, setStartingTraining] = useState(false);
    const [stoppingTraining, setStoppingTraining] = useState(false);
    const [, setTrainingBaseModelDraft] = useState('');
    const [selectedSnapshotPath, setSelectedSnapshotPath] = useState('');
    const [promotingSnapshot, setPromotingSnapshot] = useState(false);
    const [cleaningSnapshots, setCleaningSnapshots] = useState(false);
    const [deletingSnapshot, setDeletingSnapshot] = useState(false);
    const [chatInput, setChatInput] = useState('');
    const [chatHistory, setChatHistory] = useState<AvatarPersonalityTestChatTurn[]>([]);
    const [sendingChat, setSendingChat] = useState(false);
    const [runningFitCheck, setRunningFitCheck] = useState(false);
    const [fitCheckResult, setFitCheckResult] = useState<AvatarPersonalityFitCheckResponse | null>(null);
    const [activeTab, setActiveTab] = useState<'personality' | 'appearance' | 'voice'>('personality');
    const [personalityTaskTab, setPersonalityTaskTab] = useState<'prep' | 'manual' | 'llm' | 'long_form' | 'training'>('prep');
    const examplePageSize = 6;

    const loadWorkbench = async (foreground = false) => {
        if (!id) return;
        const shouldShowPageLoader = foreground || !workbench;
        if (shouldShowPageLoader) {
            setLoading(true);
        }
        setError(null);
        try {
            const res = await api.get<AvatarWorkbench>(`/avatars/${id}/workbench`);
            setWorkbench(res.data);
            setNameDraft(res.data.avatar.name || '');
            setDescriptionDraft(res.data.avatar.description || '');
        } catch (e: any) {
            console.error('Failed to load avatar workbench', e);
            setError(e?.response?.data?.detail || 'Failed to load avatar studio');
        } finally {
            if (shouldShowPageLoader) {
                setLoading(false);
            }
        }
    };

    const loadDatasetPreview = async () => {
        if (!id) return;
        setLoadingDataset(true);
        try {
            const res = await api.get<AvatarPersonalityDataset>(`/avatars/${id}/personality/dataset-preview`);
            setDataset(res.data);
        } catch (e: any) {
            if (e?.response?.status === 404) {
                setDataset(null);
            } else {
                console.error('Failed to load avatar dataset preview', e);
            }
        } finally {
            setLoadingDataset(false);
        }
    };

    const loadDatasetExamples = async (nextOffset = exampleOffset, nextState = exampleStateFilter) => {
        if (!id) return;
        setLoadingExamples(true);
        try {
            const res = await api.get<AvatarPersonalityDatasetPage>(`/avatars/${id}/personality/examples`, {
                params: {
                    offset: nextOffset,
                    limit: examplePageSize,
                    state: nextState,
                },
            });
            setDatasetPage(res.data);
        } catch (e: any) {
            console.error('Failed to load avatar dataset examples', e);
            setDatasetPage(null);
        } finally {
            setLoadingExamples(false);
        }
    };

    const loadLongFormSamples = async () => {
        if (!id) return;
        setLoadingLongForm(true);
        try {
            const [pageRes, configRes] = await Promise.all([
                api.get<AvatarPersonalityLongFormPage>(`/avatars/${id}/personality/long-form-samples`, {
                    params: { offset: 0, limit: 30, state: 'all' },
                }),
                api.get<AvatarPersonalityLongFormConfig>(`/avatars/${id}/personality/long-form-config`),
            ]);
            setLongFormPage(pageRes.data);
            setLongFormConfig(configRes.data);
        } catch (e: any) {
            console.error('Failed to load long-form samples', e);
            setLongFormPage(null);
            setLongFormConfig(null);
        } finally {
            setLoadingLongForm(false);
        }
    };

    const loadTrainingPackageState = async () => {
        if (!id) return;
        setLoadingTrainingConfig(true);
        setLoadingTrainingPackage(true);
        try {
            const [configRes, packageRes] = await Promise.all([
                api.get<AvatarPersonalityTrainingConfig>(`/avatars/${id}/personality/training-config`),
                api.get<AvatarPersonalityTrainingPackage>(`/avatars/${id}/personality/training-package`),
            ]);
            setTrainingConfig(configRes.data);
            setTrainingPackage(packageRes.data);
        } catch (e: any) {
            console.error('Failed to load training package state', e);
            setTrainingConfig(null);
            setTrainingPackage(null);
        } finally {
            setLoadingTrainingConfig(false);
            setLoadingTrainingPackage(false);
        }
    };

    const loadTrainingStatus = async () => {
        if (!id) return null;
        try {
            const res = await api.get<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/training-status`);
            setTrainingStatus(res.data);
            const selectedSnapshot = res.data.snapshots?.find((snapshot) => snapshot.selected)?.adapter_path
                || res.data.adapter_path
                || '';
            setSelectedSnapshotPath((current) => {
                if (current && res.data.snapshots?.some((snapshot) => snapshot.adapter_path === current)) {
                    return current;
                }
                return selectedSnapshot;
            });
            return res.data;
        } catch (e: any) {
            if (e?.response?.status !== 404) {
                console.error('Failed to load training status', e);
            }
            setTrainingStatus(null);
            setSelectedSnapshotPath('');
            return null;
        }
    };

    const loadBaseModelSupport = async (modelId?: string) => {
        if (!id) return null;
        try {
            const res = await api.get<AvatarPersonalityBaseModelSupport>(`/avatars/${id}/personality/base-model-support`, {
                params: modelId ? { model_id: modelId } : undefined,
            });
            setBaseModelSupport(res.data);
            return res.data;
        } catch (e: any) {
            console.error('Failed to load base model support', e);
            setBaseModelSupport(null);
            return null;
        }
    };

    const loadJudgeStatus = async () => {
        if (!id) return;
        try {
            const res = await api.get<AvatarPersonalityJudgeStatus>(`/avatars/${id}/personality/judge-status`);
            setJudgeStatus(res.data);
            return res.data;
        } catch (e: any) {
            if (e?.response?.status !== 404) {
                console.error('Failed to load avatar judge status', e);
            }
            setJudgeStatus(null);
            return null;
        }
    };

    useEffect(() => {
        void loadWorkbench(true);
        void loadDatasetPreview();
        void loadDatasetExamples(0, 'needs_review');
        void loadJudgeStatus();
        void loadTrainingStatus();
    }, [id]);

    useEffect(() => {
        if (activeTab === 'personality' && (personalityTaskTab === 'long_form' || personalityTaskTab === 'training') && !loadingLongForm) {
            void loadLongFormSamples();
        }
    }, [activeTab, personalityTaskTab]);

    useEffect(() => {
        if (activeTab === 'personality' && personalityTaskTab === 'training' && !loadingTrainingConfig && !loadingTrainingPackage) {
            void loadTrainingPackageState();
            void loadTrainingStatus();
            void loadBaseModelSupport(trainingConfig?.base_model_id);
        }
    }, [activeTab, personalityTaskTab, trainingConfig?.base_model_id]);

    useEffect(() => {
        if (!(trainingStatus?.active)) return;
        const timer = window.setInterval(() => {
            void loadTrainingStatus().then((status) => {
                if (status && !status.active) {
                    void loadTrainingPackageState();
                    void loadWorkbench();
                }
            });
        }, 3000);
        return () => window.clearInterval(timer);
    }, [trainingStatus?.active]);

    useEffect(() => {
        if (!baseModelSupport?.downloading) return;
        const timer = window.setInterval(() => {
            void loadBaseModelSupport(trainingConfig?.base_model_id);
        }, 4000);
        return () => window.clearInterval(timer);
    }, [baseModelSupport?.downloading, trainingConfig?.base_model_id]);

    useEffect(() => {
        if (!id || !judgeStatus?.active) return;
        const timer = window.setInterval(() => {
            void loadJudgeStatus().then((status) => {
                if (status && !status.active) {
                    void loadDatasetPreview();
                    void loadDatasetExamples(exampleOffset, exampleStateFilter);
                    void loadWorkbench();
                }
            });
        }, 2000);
        return () => window.clearInterval(timer);
    }, [id, judgeStatus?.active, exampleOffset, exampleStateFilter]);

    useEffect(() => {
        if (!datasetPage?.items.length) {
            setSelectedExampleId(null);
            return;
        }
        if (selectedExampleId == null || !datasetPage.items.some((example) => example.example_id === selectedExampleId)) {
            setSelectedExampleId(datasetPage.items[0].example_id);
        }
    }, [datasetPage, selectedExampleId]);

    useEffect(() => {
        if (!longFormPage?.items.length) {
            setSelectedLongFormId(null);
            return;
        }
        if (selectedLongFormId == null || !longFormPage.items.some((sample) => sample.sample_id === selectedLongFormId)) {
            setSelectedLongFormId(longFormPage.items[0].sample_id);
        }
    }, [longFormPage, selectedLongFormId]);

    useEffect(() => {
        if (trainingConfig?.base_model_id) {
            setTrainingBaseModelDraft(trainingConfig.base_model_id);
        }
    }, [trainingConfig?.base_model_id]);

    useEffect(() => {
        setFitCheckResult(null);
    }, [selectedSnapshotPath]);

    const hasChanges = useMemo(() => {
        if (!workbench) return false;
        return nameDraft.trim() !== (workbench.avatar.name || '').trim()
            || descriptionDraft.trim() !== (workbench.avatar.description || '').trim();
    }, [workbench, nameDraft, descriptionDraft]);

    const trainingProgress = useMemo(() => {
        const step = Math.max(0, trainingStatus?.step ?? 0);
        const maxSteps = Math.max(0, trainingStatus?.max_steps ?? 0);
        const progress = maxSteps > 0 ? Math.min(100, Math.max(0, (step / maxSteps) * 100)) : 0;
        const startedAt = trainingStatus?.started_at ? new Date(trainingStatus.started_at).getTime() : null;
        const finishedAt = trainingStatus?.finished_at ? new Date(trainingStatus.finished_at).getTime() : null;
        const referenceTime = trainingStatus?.active ? Date.now() : finishedAt;
        const elapsedSeconds = startedAt && referenceTime ? Math.max(0, (referenceTime - startedAt) / 1000) : null;
        const stepsPerMinute = elapsedSeconds && step > 0 ? (step / elapsedSeconds) * 60 : null;
        const etaSeconds = elapsedSeconds && step > 0 && maxSteps > step ? ((elapsedSeconds / step) * (maxSteps - step)) : null;
        return {
            step,
            maxSteps,
            progress,
            elapsedSeconds,
            etaSeconds,
            stepsPerMinute,
        };
    }, [trainingStatus]);

    const activeTrainingPlan = trainingPackage?.training_plan || trainingConfig?.training_plan || null;
    const datasetProfileOptions = trainingConfig?.dataset_profiles?.length ? trainingConfig.dataset_profiles : DEFAULT_DATASET_PROFILE_OPTIONS;

    const handleSave = async () => {
        if (!workbench || !id) return;
        const trimmedName = nameDraft.trim();
        if (!trimmedName) return;
        setSaving(true);
        try {
            await api.patch(`/avatars/${id}`, {
                name: trimmedName,
                description: descriptionDraft.trim() || null,
            });
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to update avatar', e);
            alert(e?.response?.data?.detail || 'Failed to update avatar');
        } finally {
            setSaving(false);
        }
    };

    const handleBuildDataset = async () => {
        if (!id) return;
        setBuildingDataset(true);
        try {
            const res = await api.post<AvatarPersonalityDataset>(`/avatars/${id}/personality/build-dataset`);
            setDataset(res.data);
            setExampleOffset(0);
            await loadWorkbench();
            await loadDatasetExamples(0, exampleStateFilter);
        } catch (e: any) {
            console.error('Failed to build avatar personality dataset', e);
            alert(e?.response?.data?.detail || 'Failed to build personality dataset');
        } finally {
            setBuildingDataset(false);
        }
    };

    const handleChangeExampleFilter = (nextFilter: ExampleFilter) => {
        setExampleStateFilter(nextFilter);
        setExampleOffset(0);
        void loadDatasetExamples(0, nextFilter);
    };

    const handleRunJudgePass = async () => {
        if (!id) return;
        setJudgingDataset(true);
        try {
            const res = await api.post<AvatarPersonalityJudgeStatus>(`/avatars/${id}/personality/start-judge-pass`, {
                max_examples: 40,
                overwrite_existing: false,
                target_filter: 'needs_review',
            });
            setJudgeStatus(res.data);
        } catch (e: any) {
            console.error('Failed to run avatar personality judge pass', e);
            alert(e?.response?.data?.detail || 'Failed to run local judge pass');
        } finally {
            setJudgingDataset(false);
        }
    };

    const handleStopJudgePass = async () => {
        if (!id) return;
        try {
            const res = await api.post<AvatarPersonalityJudgeStatus>(`/avatars/${id}/personality/stop-judge-pass`);
            setJudgeStatus(res.data);
        } catch (e: any) {
            console.error('Failed to stop avatar personality judge pass', e);
            alert(e?.response?.data?.detail || 'Failed to stop local judge pass');
        }
    };

    const handleUpdateLongFormState = async (sampleId: string, state: 'included' | 'rejected') => {
        if (!id) return;
        setUpdatingLongFormId(sampleId);
        try {
            await api.patch(`/avatars/${id}/personality/long-form-samples/${sampleId}`, { state });
            await loadLongFormSamples();
        } catch (e: any) {
            console.error('Failed to update long-form sample state', e);
            alert(e?.response?.data?.detail || 'Failed to update long-form sample state');
        } finally {
            setUpdatingLongFormId(null);
        }
    };

    const handleUpdateLongFormTakeCount = async (takeCount: number) => {
        if (!id) return;
        setSavingLongFormCount(true);
        try {
            const res = await api.patch<AvatarPersonalityLongFormConfig>(`/avatars/${id}/personality/long-form-config`, {
                take_count: Math.max(0, takeCount),
            });
            setLongFormConfig(res.data);
            await loadLongFormSamples();
        } catch (e: any) {
            console.error('Failed to update long-form training count', e);
            alert(e?.response?.data?.detail || 'Failed to update long-form training count');
        } finally {
            setSavingLongFormCount(false);
        }
    };

    const handleUpdateTrainingConfig = async (patch: Partial<AvatarPersonalityTrainingConfig>) => {
        if (!id) return;
        setSavingTrainingConfig(true);
        try {
            const res = await api.patch<AvatarPersonalityTrainingConfig>(`/avatars/${id}/personality/training-config`, patch);
            setTrainingConfig(res.data);
            await loadTrainingPackageState();
            await loadBaseModelSupport(res.data.base_model_id);
        } catch (e: any) {
            console.error('Failed to update training config', e);
            alert(e?.response?.data?.detail || 'Failed to update training config');
        } finally {
            setSavingTrainingConfig(false);
        }
    };

    const handlePrepareTrainingPackage = async () => {
        if (!id) return;
        setPreparingTrainingPackage(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingPackage>(`/avatars/${id}/personality/prepare-training-package`);
            setTrainingPackage(res.data);
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to prepare training package', e);
            alert(e?.response?.data?.detail || 'Failed to prepare training package');
        } finally {
            setPreparingTrainingPackage(false);
        }
    };

    const handleResetToRecommended = async () => {
        if (!id) return;
        setResettingRecommended(true);
        try {
            const [configRes, longFormRes] = await Promise.all([
                api.patch<AvatarPersonalityTrainingConfig>(`/avatars/${id}/personality/training-config`, {
                    base_model_id: RECOMMENDED_TRAINING_CONFIG.base_model_id,
                    dataset_profile: RECOMMENDED_TRAINING_CONFIG.dataset_profile,
                    training_strength: RECOMMENDED_TRAINING_CONFIG.training_strength,
                    export_strategy: RECOMMENDED_TRAINING_CONFIG.export_strategy,
                    validation_ratio: RECOMMENDED_TRAINING_CONFIG.validation_ratio,
                    max_examples: RECOMMENDED_TRAINING_CONFIG.max_examples,
                    max_long_form_examples: RECOMMENDED_TRAINING_CONFIG.max_long_form_examples,
                    include_long_form: RECOMMENDED_TRAINING_CONFIG.include_long_form,
                    training_mode: RECOMMENDED_TRAINING_CONFIG.training_mode,
                    snapshot_interval_steps: RECOMMENDED_TRAINING_CONFIG.snapshot_interval_steps,
                }),
                api.patch<AvatarPersonalityLongFormConfig>(`/avatars/${id}/personality/long-form-config`, {
                    take_count: RECOMMENDED_LONG_FORM_TAKE_COUNT,
                }),
            ]);
            setTrainingConfig(configRes.data);
            setTrainingBaseModelDraft(configRes.data.base_model_id);
            setLongFormConfig(longFormRes.data);
            await Promise.all([loadTrainingPackageState(), loadLongFormSamples(), loadBaseModelSupport(configRes.data.base_model_id)]);
        } catch (e: any) {
            console.error('Failed to reset recommended training settings', e);
            alert(e?.response?.data?.detail || 'Failed to reset recommended training settings');
        } finally {
            setResettingRecommended(false);
        }
    };

    const handleDownloadBaseModel = async (modelId?: string) => {
        if (!id) return;
        setDownloadingBaseModel(true);
        try {
            const res = await api.post<AvatarPersonalityBaseModelSupport>(`/avatars/${id}/personality/base-model-download`, {
                model_id: modelId || trainingConfig?.base_model_id,
            });
            setBaseModelSupport(res.data);
        } catch (e: any) {
            console.error('Failed to start base model download', e);
            alert(e?.response?.data?.detail || 'Failed to start base model download');
        } finally {
            setDownloadingBaseModel(false);
        }
    };

    const handleStartTraining = async () => {
        if (!id) return;
        setStartingTraining(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/start-training`, {
                training_mode: trainingConfig?.training_mode || 'memory_optimized',
                epochs: 1,
                learning_rate: trainingStrengthLearningRate(trainingConfig?.training_strength),
                snapshot_interval_steps: trainingConfig?.snapshot_interval_steps ?? 0,
                warmup_ratio: 0.03,
                overwrite_output: true,
            });
            setTrainingStatus(res.data);
            setSelectedSnapshotPath('');
            setChatHistory([]);
            setFitCheckResult(null);
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to start training', e);
            alert(e?.response?.data?.detail || 'Failed to start training');
        } finally {
            setStartingTraining(false);
        }
    };

    const handleStopTraining = async () => {
        if (!id) return;
        setStoppingTraining(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/stop-training`);
            setTrainingStatus(res.data);
        } catch (e: any) {
            console.error('Failed to stop training', e);
            alert(e?.response?.data?.detail || 'Failed to stop training');
        } finally {
            setStoppingTraining(false);
        }
    };

    const handlePromoteSnapshot = async () => {
        if (!id || !selectedSnapshotPath) return;
        setPromotingSnapshot(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/promote-snapshot`, {
                adapter_path: selectedSnapshotPath,
            });
            setTrainingStatus(res.data);
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to promote snapshot', e);
            alert(e?.response?.data?.detail || 'Failed to promote snapshot');
        } finally {
            setPromotingSnapshot(false);
        }
    };

    const handleDeleteOtherSnapshots = async () => {
        if (!id || !selectedSnapshotPath) return;
        setCleaningSnapshots(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/delete-other-snapshots`, {
                keep_adapter_path: selectedSnapshotPath,
            });
            setTrainingStatus(res.data);
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to delete other snapshots', e);
            alert(e?.response?.data?.detail || 'Failed to delete other snapshots');
        } finally {
            setCleaningSnapshots(false);
        }
    };

    const handleDeleteSnapshot = async () => {
        if (!id || !selectedSnapshotPath) return;
        if (!confirm('Delete the currently selected snapshot? This removes that adapter checkpoint from disk.')) return;
        setDeletingSnapshot(true);
        try {
            const res = await api.post<AvatarPersonalityTrainingStatus>(`/avatars/${id}/personality/delete-snapshot`, {
                adapter_path: selectedSnapshotPath,
            });
            setTrainingStatus(res.data);
            setChatHistory([]);
            setFitCheckResult(null);
            await loadWorkbench();
        } catch (e: any) {
            console.error('Failed to delete snapshot', e);
            alert(e?.response?.data?.detail || 'Failed to delete snapshot');
        } finally {
            setDeletingSnapshot(false);
        }
    };

    const handleSendTestChat = async () => {
        if (!id || !chatInput.trim()) return;
        const message = chatInput.trim();
        setSendingChat(true);
        try {
            const res = await api.post<AvatarPersonalityTestChatResponse>(`/avatars/${id}/personality/test-chat`, {
                message,
                history: chatHistory,
                adapter_path: selectedSnapshotPath || undefined,
                max_new_tokens: 220,
                temperature: 0.8,
                top_p: 0.9,
            });
            setChatHistory((current) => [
                ...current,
                { role: 'user', content: message },
                { role: 'assistant', content: res.data.reply },
            ]);
            setChatInput('');
        } catch (e: any) {
            console.error('Failed to send test chat', e);
            alert(e?.response?.data?.detail || 'Failed to generate test reply');
        } finally {
            setSendingChat(false);
        }
    };

    const handleRunFitCheck = async () => {
        if (!id || !selectedSnapshotPath) return;
        setRunningFitCheck(true);
        try {
            const res = await api.post<AvatarPersonalityFitCheckResponse>(`/avatars/${id}/personality/fit-check`, {
                adapter_path: selectedSnapshotPath,
                max_new_tokens: 160,
                temperature: 0.75,
                top_p: 0.9,
            });
            setFitCheckResult(res.data);
        } catch (e: any) {
            console.error('Failed to run fit check', e);
            alert(e?.response?.data?.detail || 'Failed to run personality fit check');
        } finally {
            setRunningFitCheck(false);
        }
    };

    // Clear similarity state whenever the selected example changes
    useEffect(() => {
        setSimilarPassages(null);
        setSimilarError(null);
        setDuplicateGroupExamples(null);
    }, [selectedExampleId]);

    const handleFindSimilar = async (exampleId: number) => {
        if (!id || loadingSimilar) return;
        setLoadingSimilar(true);
        setSimilarError(null);
        setSimilarPassages(null);
        try {
            const res = await api.get<{ items: SemanticSearchHit[]; total: number }>(
                `/avatars/${id}/personality/examples/${exampleId}/find-similar`,
                { params: { limit: 8 } },
            );
            setSimilarPassages(res.data.items);
        } catch (e: unknown) {
            const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setSimilarError(msg ?? 'Semantic search unavailable — build the semantic index first.');
        } finally {
            setLoadingSimilar(false);
        }
    };

    const handleLoadDuplicateGroup = async (groupId: number) => {
        if (!id || loadingDupGroup) return;
        setLoadingDupGroup(true);
        setDuplicateGroupExamples(null);
        try {
            const res = await api.get<AvatarPersonalityDatasetExample[]>(
                `/avatars/${id}/personality/duplicate-group/${groupId}`,
            );
            setDuplicateGroupExamples(res.data);
        } catch {
            setDuplicateGroupExamples([]);
        } finally {
            setLoadingDupGroup(false);
        }
    };

    const handleUpdateExampleState = async (exampleId: number, state: 'approved' | 'rejected' | 'inherit') => {
        if (!id) return;
        setUpdatingExampleId(exampleId);
        try {
            const res = await api.patch<AvatarPersonalityDataset>(`/avatars/${id}/personality/examples/${exampleId}`, { state });
            setDataset(res.data);
            await loadWorkbench();
            await loadDatasetExamples(exampleOffset, exampleStateFilter);
        } catch (e: any) {
            console.error('Failed to update avatar dataset example state', e);
            alert(e?.response?.data?.detail || 'Failed to update dataset example state');
        } finally {
            setUpdatingExampleId(null);
        }
    };

    if (loading) {
        return (
            <div className="py-20 flex flex-col items-center justify-center gap-4 text-slate-400">
                <Loader2 className="w-8 h-8 animate-spin" />
                <p>Loading avatar studio...</p>
            </div>
        );
    }

    if (error || !workbench) {
        return (
            <div className="space-y-4">
                <button onClick={() => navigate(-1)} className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700">
                    <ArrowLeft size={16} /> Back
                </button>
                <div className="rounded-2xl border border-red-200 bg-red-50 p-8 text-center text-red-700">
                    {error || 'Avatar not found'}
                </div>
            </div>
        );
    }

    const portraitUrl = workbench.speaker.thumbnail_path ? toApiUrl(workbench.speaker.thumbnail_path) : null;
    const selectedExample = datasetPage?.items.find((example) => example.example_id === selectedExampleId) ?? datasetPage?.items[0] ?? null;
    const selectedLongForm = longFormPage?.items.find((sample) => sample.sample_id === selectedLongFormId) ?? longFormPage?.items[0] ?? null;
    const baseModelOptions = (() => {
        const candidates = baseModelSupport?.candidates || [];
        const selected = (trainingConfig?.base_model_id || '').trim();
        if (selected && !candidates.some((candidate) => candidate.model_id === selected)) {
            return [
                {
                    model_id: selected,
                    label: `${selected} (Custom)`,
                    recommended: false,
                    installed: !!baseModelSupport?.installed,
                },
                ...candidates,
            ];
        }
        return candidates;
    })();
    const openLongFormAudioPopout = (sample: NonNullable<typeof selectedLongForm>) => {
        const params = new URLSearchParams({
            start: String(sample.start_time),
            end: String(sample.end_time),
            audio_only: 'true',
        });
        const audioUrl = toApiUrl(`/videos/${sample.video_id}/clip?${params.toString()}`);
        const popup = window.open('', `avatar-long-form-${sample.sample_id}`, 'popup=yes,width=560,height=320');
        if (!popup) {
            alert('Popup blocked. Allow popups for localhost and try again.');
            return;
        }
        const title = escapeHtml(sample.video_title);
        const timeRange = escapeHtml(`${formatClock(sample.start_time)}-${formatClock(sample.end_time)}`);
        const duration = escapeHtml(formatDuration(sample.duration_seconds));
        const wordCount = escapeHtml(`${sample.word_count} words`);
        popup.document.title = `${sample.video_title} audio preview`;
        popup.document.body.innerHTML = `
            <div style="font-family: Segoe UI, Arial, sans-serif; padding: 18px; color: #0f172a; background: #f8fafc;">
                <div style="font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: #64748b;">Long Form Audio</div>
                <div style="margin-top: 6px; font-size: 20px; font-weight: 700; line-height: 1.3;">${title}</div>
                <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px; color: #475569;">
                    <span style="padding: 6px 10px; border-radius: 999px; border: 1px solid #cbd5e1; background: white;">${timeRange}</span>
                    <span style="padding: 6px 10px; border-radius: 999px; border: 1px solid #cbd5e1; background: white;">${duration}</span>
                    <span style="padding: 6px 10px; border-radius: 999px; border: 1px solid #cbd5e1; background: white;">${wordCount}</span>
                </div>
                <audio controls autoplay preload="metadata" src="${audioUrl}" style="margin-top: 16px; width: 100%;"></audio>
                <div style="margin-top: 14px; font-size: 12px; color: #64748b;">This popout plays the selected long-form segment only.</div>
            </div>
        `;
        popup.document.close();
        popup.focus();
    };
    const exampleFilters: Array<{ key: ExampleFilter; label: string; count: number }> = [
        { key: 'needs_review', label: 'Needs Review', count: datasetPage?.needs_review_count ?? 0 },
        { key: 'gold', label: 'Gold', count: datasetPage?.gold_count ?? 0 },
        { key: 'silver', label: 'Silver', count: datasetPage?.silver_count ?? 0 },
        { key: 'auto_reject', label: 'Auto Reject', count: datasetPage?.auto_reject_count ?? 0 },
        { key: 'duplicate_risk', label: 'Duplicate Risk', count: datasetPage?.duplicate_count ?? 0 },
        { key: 'approved', label: 'Included', count: datasetPage?.approved_count ?? 0 },
        { key: 'rejected', label: 'Excluded', count: datasetPage?.rejected_count ?? 0 },
        { key: 'all', label: 'All', count: (datasetPage?.approved_count ?? 0) + (datasetPage?.rejected_count ?? 0) },
    ];
    const personalityTaskTabs: Array<{ key: 'prep' | 'manual' | 'llm' | 'long_form' | 'training'; label: string; summary: string }> = [
        { key: 'prep', label: 'Dataset Prep', summary: 'Build the dataset and check whether you have enough signal.' },
        { key: 'manual', label: 'Manual Grading', summary: 'Review examples directly and override automation when needed.' },
        { key: 'llm', label: 'LLM Grading', summary: 'Run the local judge and monitor its live verdict feed.' },
        { key: 'long_form', label: 'Long Form', summary: 'Review the longest uninterrupted speaker samples and exclude weak monologues.' },
        { key: 'training', label: 'Training', summary: 'Decide whether to train now and what subset to use.' },
    ];
    const studioTabs = [
        {
            key: 'personality' as const,
            title: 'Personality',
            icon: <Brain size={18} />,
            status: workbench.personality.status,
            sourceLabel: 'Conversation Turns',
            sourceCount: workbench.personality.source_count,
            approvedLabel: 'Approved Examples',
            approvedCount: workbench.personality.approved_count,
            artifactReady: workbench.personality.artifact_ready,
            summary: workbench.personality.summary,
        },
        {
            key: 'appearance' as const,
            title: 'Appearance',
            icon: <ImageIcon size={18} />,
            status: workbench.appearance.status,
            sourceLabel: 'Image Sources',
            sourceCount: workbench.appearance.source_count,
            approvedLabel: 'Approved Images',
            approvedCount: workbench.appearance.approved_count,
            artifactReady: workbench.appearance.artifact_ready,
            summary: workbench.appearance.summary,
        },
        {
            key: 'voice' as const,
            title: 'Voice',
            icon: <Mic2 size={18} />,
            status: workbench.voice.status,
            sourceLabel: 'Reference Clips',
            sourceCount: workbench.voice.source_count,
            approvedLabel: 'Approved Profiles',
            approvedCount: workbench.voice.approved_count,
            artifactReady: workbench.voice.artifact_ready,
            summary: workbench.voice.summary,
        },
    ];
    return (
        <div className="space-y-6 pb-20">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <button onClick={() => navigate(-1)} className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700">
                    <ArrowLeft size={16} /> Back
                </button>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                    <Link
                        to={`/speakers/${workbench.speaker.id}`}
                        className="flex-1 sm:flex-none text-center rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-slate-600 hover:bg-slate-50 whitespace-nowrap"
                    >
                        Open Speaker
                    </Link>
                    <Link
                        to={`/channel/${workbench.speaker.channel_id}/speakers`}
                        className="flex-1 sm:flex-none text-center rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-slate-600 hover:bg-slate-50 whitespace-nowrap"
                    >
                        Channel Speakers
                    </Link>
                </div>
            </div>

            <section className="rounded-3xl border border-slate-200 bg-gradient-to-br from-white via-slate-50 to-blue-50 p-6 shadow-sm">
                <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
                    <div className="flex flex-col items-center rounded-2xl border border-slate-200 bg-white/85 p-5">
                        <div className="h-36 w-36 overflow-hidden rounded-3xl border border-slate-200 bg-slate-100">
                            {portraitUrl ? (
                                <img src={portraitUrl} alt={workbench.avatar.name} className="h-full w-full object-cover" />
                            ) : (
                                <div className="flex h-full w-full items-center justify-center text-slate-300">
                                    <ImageIcon size={34} />
                                </div>
                            )}
                        </div>
                        <div className="mt-4 text-center">
                            <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Avatar Studio</div>
                            <div className="mt-1 text-lg font-semibold text-slate-900">{workbench.avatar.name}</div>
                            <div className="mt-2 inline-flex rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600">
                                {workbench.avatar.status}
                            </div>
                        </div>
                    </div>

                    <div className="space-y-5">
                        <div>
                            <div className="text-sm font-medium text-slate-500">Source Speaker</div>
                            <div className="mt-1 text-3xl font-bold text-slate-950">{workbench.speaker.name}</div>
                            <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-600">
                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                    Speaking time: {formatDuration(workbench.speaker.total_speaking_time)}
                                </span>
                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                    Voice profiles: {workbench.speaker.embedding_count}
                                </span>
                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                    Episode appearances: {workbench.speaker.appearance_count}
                                </span>
                                <span className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-blue-700">
                                    Suggested base model: {workbench.suggested_base_model || 'Qwen/Qwen3-14B'}
                                </span>
                            </div>
                        </div>

                        <div className="grid gap-4 md:grid-cols-2">
                            <label className="block">
                                <div className="mb-1.5 text-sm font-medium text-slate-700">Avatar Name</div>
                                <input
                                    value={nameDraft}
                                    onChange={(e) => setNameDraft(e.target.value)}
                                    className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/15"
                                    placeholder="Avatar name"
                                />
                            </label>
                            <label className="block md:col-span-2">
                                <div className="mb-1.5 text-sm font-medium text-slate-700">Studio Notes</div>
                                <textarea
                                    value={descriptionDraft}
                                    onChange={(e) => setDescriptionDraft(e.target.value)}
                                    className="min-h-[96px] w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/15"
                                    placeholder="Describe the intended avatar, constraints, or future training notes."
                                />
                            </label>
                        </div>

                        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3">
                            <div>
                                <div className="text-sm font-medium text-slate-800">Runtime</div>
                                <div className="text-xs text-slate-500">Current state: {workbench.runtime_status}</div>
                            </div>
                            <button
                                type="button"
                                onClick={() => void handleSave()}
                                disabled={!hasChanges || saving || !nameDraft.trim()}
                                className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                                Save Avatar
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
                <div className="grid gap-2 md:grid-cols-3">
                    {studioTabs.map((tab) => (
                        <button
                            key={tab.key}
                            type="button"
                            onClick={() => setActiveTab(tab.key)}
                            className={`rounded-2xl border px-4 py-4 text-left transition ${activeTab === tab.key
                                ? 'border-slate-900 bg-slate-900 text-white shadow-sm'
                                : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300 hover:bg-white'
                                }`}
                        >
                            <div className="flex items-start justify-between gap-3">
                                <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${activeTab === tab.key ? 'bg-white/10 text-white' : 'bg-white text-slate-700'}`}>
                                    {tab.icon}
                                </div>
                                <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold ${activeTab === tab.key ? 'border-white/20 bg-white/10 text-white' : statusTone(tab.status)}`}>
                                    {tab.status.replaceAll('_', ' ')}
                                </span>
                            </div>
                            <div className="mt-4">
                                <div className="text-base font-semibold">{tab.title}</div>
                                <div className={`mt-1 text-sm ${activeTab === tab.key ? 'text-slate-200' : 'text-slate-500'}`}>{tab.summary || 'Configuration pending.'}</div>
                            </div>
                            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                                <div className={`rounded-xl border px-3 py-2 ${activeTab === tab.key ? 'border-white/15 bg-white/10 text-white' : 'border-slate-200 bg-white text-slate-700'}`}>
                                    <div className={`${activeTab === tab.key ? 'text-slate-200' : 'text-slate-500'}`}>{tab.sourceLabel}</div>
                                    <div className="mt-1 text-lg font-semibold">{tab.sourceCount}</div>
                                </div>
                                <div className={`rounded-xl border px-3 py-2 ${activeTab === tab.key ? 'border-white/15 bg-white/10 text-white' : 'border-slate-200 bg-white text-slate-700'}`}>
                                    <div className={`${activeTab === tab.key ? 'text-slate-200' : 'text-slate-500'}`}>{tab.approvedLabel}</div>
                                    <div className="mt-1 text-lg font-semibold">{tab.approvedCount}</div>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>
            </section>

            <div className="space-y-5">
                <div className={activeTab === 'personality' ? '' : 'hidden'}>
                <SectionCard
                    title="Personality"
                    icon={<Brain size={18} />}
                    status={workbench.personality.status}
                    sourceLabel="Conversation Turns"
                    sourceCount={workbench.personality.source_count}
                    approvedLabel="Approved Examples"
                    approvedCount={workbench.personality.approved_count}
                    artifactReady={workbench.personality.artifact_ready}
                    summary={workbench.personality.summary}
                    compactStats
                >
                    <div className="space-y-3">
                        <div className="grid gap-2 md:grid-cols-4">
                            {personalityTaskTabs.map((tab) => (
                                <button
                                    key={tab.key}
                                    type="button"
                                    onClick={() => setPersonalityTaskTab(tab.key)}
                                    className={`rounded-xl border px-3 py-3 text-left transition ${personalityTaskTab === tab.key
                                        ? 'border-slate-900 bg-slate-900 text-white'
                                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300 hover:bg-white'
                                        }`}
                                >
                                    <div className="text-sm font-semibold">{tab.label}</div>
                                    <div className={`mt-1 text-xs leading-5 ${personalityTaskTab === tab.key ? 'text-slate-200' : 'text-slate-500'}`}>
                                        {tab.summary}
                                    </div>
                                </button>
                            ))}
                        </div>

                        {personalityTaskTab === 'prep' && (
                        <>
                        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
                            <div className="text-sm text-slate-600">
                                Build the transcript-derived dataset and confirm whether the corpus is ready for training.
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() => void handleBuildDataset()}
                                    disabled={buildingDataset || !!judgeStatus?.active}
                                    className="inline-flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {buildingDataset ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                    {dataset?.example_count ? 'Rebuild Dataset' : 'Build Dataset'}
                                </button>
                            </div>
                        </div>

                        {(loadingDataset || dataset) && (
                            <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div>
                                        <div className="text-xs uppercase tracking-wide text-slate-500">Dataset Snapshot</div>
                                        <div className="mt-1 text-sm text-slate-600">
                                            {loadingDataset
                                                ? 'Loading latest dataset snapshot...'
                                                : dataset?.generated_at
                                                    ? `Generated ${new Date(dataset.generated_at).toLocaleString()}`
                                                    : 'No dataset has been built yet.'}
                                        </div>
                                    </div>
                                    {dataset?.dataset_path && (
                                        <div className="max-w-[280px] truncate text-right text-[11px] font-mono text-slate-500" title={dataset.dataset_path}>
                                            {dataset.dataset_path}
                                        </div>
                                    )}
                                </div>

                                {dataset && (
                                    <>
                                        <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.8fr)]">
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <div className="flex flex-wrap items-start justify-between gap-2">
                                                    <div>
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Training Readiness</div>
                                                        <div className="mt-1 text-sm font-semibold text-slate-900">
                                                            {dataset.readiness.can_train_now ? 'Enough signal to train now' : 'More curation recommended before training'}
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-wrap gap-2">
                                                        <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold ${readinessTone(dataset.readiness.status)}`}>
                                                            {dataset.readiness.status}
                                                        </span>
                                                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-700">
                                                            score {dataset.readiness.score}
                                                        </span>
                                                        <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold ${manualReviewTone(dataset.readiness.manual_review_roi)}`}>
                                                            manual review ROI {dataset.readiness.manual_review_roi}
                                                        </span>
                                                    </div>
                                                </div>
                                                {dataset.readiness.summary && (
                                                    <div className="mt-2 text-sm leading-6 text-slate-600">
                                                        {dataset.readiness.summary}
                                                    </div>
                                                )}
                                                {dataset.readiness.recommended_action && (
                                                    <div className="mt-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                                                        {dataset.readiness.recommended_action}
                                                    </div>
                                                )}
                                                {dataset.readiness.caution && (
                                                    <div className="mt-2 text-xs leading-5 text-slate-500">
                                                        {dataset.readiness.caution}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="rounded-xl border border-slate-200 bg-white p-3">
                                                <div className="text-[11px] uppercase tracking-wide text-slate-500">Readiness Inputs</div>
                                                <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                                                    <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                                                        <span>Included examples</span>
                                                        <span className="font-semibold text-slate-900">{workbench.personality.approved_count}</span>
                                                    </div>
                                                    <div className="flex items-center justify-between rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                                                        <span>Gold examples</span>
                                                        <span className="font-semibold">{dataset.gold_example_count}</span>
                                                    </div>
                                                    <div className="flex items-center justify-between rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-700">
                                                        <span>Approved response hours</span>
                                                        <span className="font-semibold">{dataset.readiness.approved_duration_hours.toFixed(1)}h</span>
                                                    </div>
                                                    <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                                                        <span>Approved words</span>
                                                        <span className="font-semibold text-slate-900">{dataset.readiness.approved_word_count.toLocaleString()}</span>
                                                    </div>
                                                    <div className="flex items-center justify-between rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-xs text-violet-700">
                                                        <span>Still needs review</span>
                                                        <span className="font-semibold">{dataset.needs_review_count}</span>
                                                    </div>
                                                    <div className="flex items-center justify-between rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">
                                                        <span>Duplicate pressure</span>
                                                        <span className="font-semibold">{dataset.readiness.duplicate_pressure}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="mt-3 flex flex-wrap gap-2 text-xs">
                                            <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                                                <span className="font-semibold text-slate-900">{dataset.cluster_count}</span> clusters
                                            </div>
                                            <div className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1.5 text-sky-700">
                                                <span className="font-semibold">{dataset.duplicate_example_count}</span> duplicate risk
                                            </div>
                                            <div className="rounded-full border border-fuchsia-200 bg-fuchsia-50 px-3 py-1.5 text-fuchsia-700">
                                                <span className="font-semibold">{dataset.llm_judged_count}</span> llm judged
                                            </div>
                                            <div className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-emerald-700">
                                                <span className="font-semibold">{dataset.llm_promoted_count}</span> llm promoted
                                            </div>
                                            <div className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1.5 text-rose-700">
                                                <span className="font-semibold">{dataset.llm_rejected_count}</span> llm rejected
                                            </div>
                                            <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                                                <span className="font-semibold text-slate-900">{dataset.source_turn_count}</span> source turns
                                            </div>
                                            <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-slate-600">
                                                Base: <span className="font-semibold text-slate-900">{dataset.base_model_id || workbench.suggested_base_model || 'Qwen/Qwen3-14B'}</span>
                                            </div>
                                        </div>

                                        {dataset.system_prompt && (
                                            <div className="mt-2 text-xs text-slate-500">
                                                <span className="font-semibold text-slate-700">Prompt:</span> {dataset.system_prompt}
                                            </div>
                                        )}

                                        {false && dataset!.preview_examples.length > 0 && (
                                            <div className="hidden">
                                                {dataset!.preview_examples.slice(0, 3).map((example, index) => (
                                                    <div key={`preview-card-${example.video_id}-${example.start_time}-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                                            <div>
                                                                <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Preview Example</div>
                                                                <div className="mt-1 text-sm font-semibold text-slate-900">{example.video_title}</div>
                                                            </div>
                                                            <div className="flex flex-wrap gap-2 text-xs text-slate-600">
                                                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                                                    {formatClock(example.start_time)}-{formatClock(example.end_time)}
                                                                </span>
                                                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                                                    {example.context_turns} context turn{example.context_turns === 1 ? '' : 's'}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div className="mt-4">
                                                            <ExampleConversationCard example={example} responseSpeakerName={workbench!.speaker.name} />
                                                        </div>
                                                    </div>
                                                ))}
                                            <div className="hidden">
                                                {dataset!.preview_examples.slice(0, 3).map((example, index) => (
                                                    <div key={`${example.video_id}-${example.start_time}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">
                                                            {example.video_title} â€¢ {example.context_turns} context turn{example.context_turns === 1 ? '' : 's'}
                                                        </div>
                                                        <div className="mt-2">
                                                            <div className="text-xs font-semibold text-slate-500">Prompt</div>
                                                            <div className="mt-1 whitespace-pre-wrap text-sm text-slate-700">{example.context_text}</div>
                                                        </div>
                                                        <div className="mt-3">
                                                            <div className="text-xs font-semibold text-slate-500">Response</div>
                                                            <div className="mt-1 whitespace-pre-wrap text-sm text-slate-900">{example.response_text}</div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        )}
                        </>
                        )}

                        {personalityTaskTab === 'llm' && (
                        <>
                        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-violet-200 bg-violet-50/60 px-3 py-2.5">
                            <div className="text-sm text-violet-900">
                                Use the local model to grade the silver band one example at a time and watch the live feed below.
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                                {judgeStatus?.active ? (
                                    <button
                                        type="button"
                                        onClick={() => void handleStopJudgePass()}
                                        className="inline-flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100"
                                    >
                                        <Square size={14} />
                                        Stop Judge
                                    </button>
                                ) : (
                                    <button
                                        type="button"
                                        onClick={() => void handleRunJudgePass()}
                                        disabled={judgingDataset || !dataset?.needs_review_count}
                                        className="inline-flex items-center gap-2 rounded-lg border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-violet-700 hover:bg-violet-100 disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                        {judgingDataset ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
                                        Run Local Judge
                                    </button>
                                )}
                            </div>
                        </div>

                        {judgeStatus && judgeStatus.status !== 'idle' ? (
                            <div className="rounded-xl border border-violet-200 bg-violet-50/60 p-3">
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div>
                                        <div className="text-[11px] uppercase tracking-wide text-violet-700">Local Judge Feed</div>
                                        <div className="mt-1 text-sm font-semibold text-violet-900">
                                            {judgeStatus.active
                                                ? `Processing ${judgeStatus.processed_count} of ${judgeStatus.total_candidates || judgeStatus.max_examples}`
                                                : `Last run ${judgeStatus.status}`}
                                        </div>
                                        <div className="mt-1 text-xs text-violet-700">
                                            {judgeStatus.model || 'Local model'} â€¢ target {judgeStatus.target_filter.replaceAll('_', ' ')}
                                            {judgeStatus.current_video_title ? ` â€¢ ${judgeStatus.current_video_title}` : ''}
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap gap-2 text-xs">
                                        <span className="rounded-full border border-violet-200 bg-white px-3 py-1 font-semibold text-violet-700">
                                            judged {judgeStatus.judged_count}
                                        </span>
                                        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 font-semibold text-emerald-700">
                                            promoted {judgeStatus.promoted_count}
                                        </span>
                                        <span className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1 font-semibold text-rose-700">
                                            rejected {judgeStatus.rejected_count}
                                        </span>
                                        {judgeStatus.current_stage && (
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1 font-semibold text-slate-700">
                                                {judgeStatus.current_stage.replaceAll('_', ' ')}
                                            </span>
                                        )}
                                    </div>
                                </div>

                                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/80">
                                    <div
                                        className="h-full rounded-full bg-violet-500 transition-all"
                                        style={{ width: `${Math.max(4, Math.min(100, Math.round(((judgeStatus.processed_count || 0) / Math.max(1, judgeStatus.total_candidates || judgeStatus.max_examples || 1)) * 100)))}%` }}
                                    />
                                </div>

                                {judgeStatus.error && (
                                    <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                                        {judgeStatus.error}
                                    </div>
                                )}

                                <div className="mt-3 grid gap-2 lg:grid-cols-2">
                                    {judgeStatus.recent_results.length > 0 ? judgeStatus.recent_results.map((item) => (
                                        <div key={`judge-feed-${item.example_id}-${item.judged_at || item.video_title}`} className="rounded-lg border border-white bg-white/90 px-3 py-2.5">
                                            <div className="flex items-start justify-between gap-2">
                                                <div className="min-w-0">
                                                    <div className="truncate text-xs font-semibold uppercase tracking-wide text-slate-500">
                                                        Example #{item.example_id}
                                                    </div>
                                                    <div className="truncate text-sm font-semibold text-slate-900">{item.video_title}</div>
                                                </div>
                                                <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${autoLabelTone(item.llm_label)}`}>
                                                    {item.llm_label}
                                                </span>
                                            </div>
                                            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-600">
                                                <span>{item.llm_confidence} confidence</span>
                                                {item.heuristic_label && <span>heuristic {item.heuristic_label}</span>}
                                            </div>
                                            {item.llm_reasons.length > 0 && (
                                                <div className="mt-2 flex flex-wrap gap-1.5">
                                                    {item.llm_reasons.slice(0, 3).map((reason) => (
                                                        <span key={`judge-feed-reason-${item.example_id}-${reason}`} className="rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600">
                                                            {reasonLabel(reason)}
                                                        </span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )) : (
                                        <div className="rounded-lg border border-dashed border-violet-200 bg-white/70 px-3 py-4 text-sm text-violet-700">
                                            The local judge feed will populate as examples are scored.
                                        </div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <div className="rounded-xl border border-dashed border-violet-200 bg-violet-50/40 px-4 py-6 text-sm text-violet-700">
                                No judge run is active yet. Start a local pass to see live progress, verdicts, and reject reasons here.
                            </div>
                        )}
                        </>
                        )}

                        {personalityTaskTab === 'manual' && (
                            <div className="rounded-xl border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div>
                                        <div className="text-xs uppercase tracking-wide text-slate-500">Curation Queue</div>
                                        <div className="mt-1 text-sm text-slate-600">Review examples directly and override automation when needed.</div>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                        {exampleFilters.map((filterOption) => (
                                            <button
                                                key={filterOption.key}
                                                type="button"
                                                onClick={() => handleChangeExampleFilter(filterOption.key)}
                                                className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${exampleStateFilter === filterOption.key
                                                    ? 'border-slate-900 bg-slate-900 text-white'
                                                    : 'border-slate-200 bg-slate-50 text-slate-600 hover:bg-white'
                                                    }`}
                                            >
                                                {filterOption.label} ({filterOption.count})
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {loadingExamples ? (
                                    <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
                                        <Loader2 size={15} className="animate-spin" />
                                        Loading examples...
                                    </div>
                                ) : !datasetPage || datasetPage.items.length === 0 ? (
                                    <div className="mt-4 rounded-lg border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
                                        No examples found for this filter yet.
                                    </div>
                                ) : (
                                    <div className="mt-4 grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
                                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-2">
                                            <div className="max-h-[780px] space-y-2 overflow-y-auto pr-1">
                                                {datasetPage.items.map((example) => (
                                                    <button
                                                        key={`queue-item-${example.example_id}`}
                                                        type="button"
                                                        onClick={() => setSelectedExampleId(example.example_id)}
                                                        className={`w-full rounded-xl border px-3 py-3 text-left transition ${selectedExample?.example_id === example.example_id
                                                            ? 'border-slate-900 bg-white shadow-sm'
                                                            : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                                                            }`}
                                                    >
                                                        <div className="flex items-start justify-between gap-2">
                                                            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Example #{example.example_id}</div>
                                                            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${autoLabelTone(example.auto_label)}`}>
                                                                {example.auto_label === 'reject' ? 'auto reject' : example.auto_label}
                                                            </span>
                                                        </div>
                                                        <div className="mt-2 line-clamp-2 text-sm font-semibold leading-5 text-slate-900">{example.video_title}</div>
                                                        <div className="mt-2 line-clamp-3 text-sm leading-6 text-slate-600">{example.response_text}</div>
                                                        {example.duplicate_group_size > 1 && (
                                                            <div className="mt-1.5 inline-flex items-center gap-1 text-[10px] text-amber-600">
                                                                <AlertTriangle size={9} />
                                                                {example.duplicate_similarity > 0
                                                                    ? `Dup group · ${Math.round(example.duplicate_similarity * 100)}% similar`
                                                                    : `Group leader · ${example.duplicate_group_size} in group`}
                                                            </div>
                                                        )}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                            {selectedExample ? (
                                                <div className="space-y-4">
                                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                                        <div>
                                                            <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Selected Example</div>
                                                            <div className="mt-1 text-lg font-semibold text-slate-900">{selectedExample.video_title}</div>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            {selectedExample.manual_state && (
                                                                <button
                                                                    type="button"
                                                                    onClick={() => void handleUpdateExampleState(selectedExample.example_id, 'inherit')}
                                                                    disabled={updatingExampleId === selectedExample.example_id}
                                                                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                                                >
                                                                    {updatingExampleId === selectedExample.example_id ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                                    Reset to auto
                                                                </button>
                                                            )}
                                                            <button
                                                                type="button"
                                                                onClick={() => void handleUpdateExampleState(selectedExample.example_id, 'approved')}
                                                                disabled={updatingExampleId === selectedExample.example_id || selectedExample.state === 'approved'}
                                                                className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                                                            >
                                                                {updatingExampleId === selectedExample.example_id && selectedExample.state !== 'approved' ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                                                                Approve
                                                            </button>
                                                            <button
                                                                type="button"
                                                                onClick={() => void handleUpdateExampleState(selectedExample.example_id, 'rejected')}
                                                                disabled={updatingExampleId === selectedExample.example_id || selectedExample.state === 'rejected'}
                                                                className="inline-flex items-center gap-1.5 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100 disabled:opacity-50"
                                                            >
                                                                {updatingExampleId === selectedExample.example_id && selectedExample.state !== 'rejected' ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                                                                Reject
                                                            </button>
                                                        </div>
                                                    </div>

                                                    <div className="grid gap-2 md:grid-cols-4">
                                                        <div className="rounded-xl border border-slate-200 bg-white p-3"><div className="text-[11px] uppercase tracking-wide text-slate-500">Quality</div><div className="mt-1 text-lg font-semibold text-slate-900">{selectedExample.quality_score}</div></div>
                                                        <div className="rounded-xl border border-slate-200 bg-white p-3"><div className="text-[11px] uppercase tracking-wide text-slate-500">Completion</div><div className="mt-1 text-lg font-semibold text-slate-900">{selectedExample.completion_score}</div></div>
                                                        <div className="rounded-xl border border-slate-200 bg-white p-3"><div className="text-[11px] uppercase tracking-wide text-slate-500">Context</div><div className="mt-1 text-lg font-semibold text-slate-900">{selectedExample.context_score}</div></div>
                                                        <div className="rounded-xl border border-slate-200 bg-white p-3"><div className="text-[11px] uppercase tracking-wide text-slate-500">Style</div><div className="mt-1 text-lg font-semibold text-slate-900">{selectedExample.style_score}</div></div>
                                                    </div>

                                                    {/* ── Duplicate / Similarity info ─────────────────── */}
                                                    {(selectedExample.duplicate_group_size > 1 || selectedExample.duplicate_similarity > 0) && (
                                                        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 space-y-2">
                                                            <div className="flex flex-wrap items-center gap-2">
                                                                <AlertTriangle size={13} className="text-amber-500 flex-shrink-0" />
                                                                <span className="text-xs font-semibold text-amber-700">
                                                                    Likely duplicate
                                                                </span>
                                                                {selectedExample.duplicate_similarity > 0 && (
                                                                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-600">
                                                                        <Zap size={9} />
                                                                        {Math.round(selectedExample.duplicate_similarity * 100)}% similar
                                                                    </span>
                                                                )}
                                                                {selectedExample.duplicate_group_size > 1 && (
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => selectedExample.duplicate_group_id != null
                                                                            ? void handleLoadDuplicateGroup(selectedExample.duplicate_group_id)
                                                                            : undefined}
                                                                        disabled={loadingDupGroup}
                                                                        className="inline-flex items-center gap-1 text-[11px] text-amber-600 hover:underline disabled:opacity-50"
                                                                    >
                                                                        <Layers size={10} />
                                                                        {loadingDupGroup ? 'Loading…' : `Show ${selectedExample.duplicate_group_size} in group`}
                                                                    </button>
                                                                )}
                                                            </div>
                                                            {duplicateGroupExamples && duplicateGroupExamples.length > 0 && (
                                                                <div className="mt-2 space-y-1.5 border-t border-amber-100 pt-2">
                                                                    {duplicateGroupExamples
                                                                        .filter((ex) => ex.example_id !== selectedExample.example_id)
                                                                        .map((ex) => (
                                                                            <button
                                                                                key={ex.example_id}
                                                                                type="button"
                                                                                onClick={() => setSelectedExampleId(ex.example_id)}
                                                                                className="w-full text-left rounded-lg border border-amber-100 bg-white px-2.5 py-2 hover:bg-amber-50/50 transition"
                                                                            >
                                                                                <div className="text-[10px] text-amber-500 font-mono mb-0.5">#{ex.example_id} • {ex.video_title}</div>
                                                                                <p className="text-xs text-slate-700 line-clamp-2">{ex.response_text}</p>
                                                                            </button>
                                                                        ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}

                                                    {/* ── Cluster info ────────────────────────────────── */}
                                                    {selectedExample.cluster_id != null && (
                                                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
                                                            <span className="inline-flex items-center gap-1">
                                                                <Layers size={11} />
                                                                Cluster {selectedExample.cluster_id} · {selectedExample.cluster_size} members
                                                            </span>
                                                            <span className="inline-flex items-center gap-1">
                                                                Diversity {selectedExample.diversity_score}/100
                                                                {selectedExample.diversity_score < 30 && (
                                                                    <span className="text-amber-500 font-medium">(saturated topic)</span>
                                                                )}
                                                            </span>
                                                        </div>
                                                    )}

                                                    {/* ── Find similar passages ────────────────────────── */}
                                                    <div className="space-y-2">
                                                        <button
                                                            type="button"
                                                            onClick={() => void handleFindSimilar(selectedExample.example_id)}
                                                            disabled={loadingSimilar}
                                                            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                                        >
                                                            {loadingSimilar
                                                                ? <Loader2 size={12} className="animate-spin" />
                                                                : <Search size={12} />}
                                                            {loadingSimilar ? 'Searching…' : 'Find similar passages'}
                                                        </button>

                                                        {similarError && (
                                                            <p className="text-xs text-slate-400">{similarError}</p>
                                                        )}

                                                        {similarPassages && similarPassages.length === 0 && !similarError && (
                                                            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-1">
                                                                <p className="text-xs font-medium text-slate-500">No similar passages found</p>
                                                                <p className="text-[11px] text-slate-400">The semantic index may not be built yet for this channel. Go to the channel's Transcripts tab and click "Rebuild" on the Semantic Index panel to enable this feature.</p>
                                                            </div>
                                                        )}

                                                        {similarPassages && similarPassages.length > 0 && (
                                                            <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-3 space-y-2">
                                                                <div className="text-[11px] font-semibold text-indigo-600 flex items-center gap-1.5">
                                                                    <Search size={11} />
                                                                    Similar passages ({similarPassages.length})
                                                                </div>
                                                                {similarPassages.map((hit) => {
                                                                    const pct = Math.round(hit.score * 100);
                                                                    return (
                                                                        <div key={hit.id} className="rounded-lg border border-indigo-100 bg-white p-2.5 space-y-1">
                                                                            <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                                                                                <span className="inline-flex items-center gap-1 text-indigo-600 font-medium">
                                                                                    <Zap size={9} />
                                                                                    {pct}%
                                                                                </span>
                                                                                {hit.video_title && (
                                                                                    <span className="truncate max-w-[280px]">{hit.video_title}</span>
                                                                                )}
                                                                                <span className="inline-flex items-center gap-1 font-mono">
                                                                                    <Clock size={9} />
                                                                                    {Math.floor(hit.start_time / 60)}:{String(Math.floor(hit.start_time % 60)).padStart(2, '0')}
                                                                                </span>
                                                                                {hit.speaker_name && <span>{hit.speaker_name}</span>}
                                                                            </div>
                                                                            <p className="text-xs text-slate-700 leading-relaxed">{hit.chunk_text}</p>
                                                                        </div>
                                                                    );
                                                                })}
                                                            </div>
                                                        )}
                                                    </div>

                                                    <ExampleConversationCard example={selectedExample} responseSpeakerName={workbench.speaker.name} />
                                                </div>
                                            ) : (
                                                <div className="flex min-h-[360px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white text-sm text-slate-500">
                                                    Select an example from the queue to review it.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {personalityTaskTab === 'long_form' && (
                            <div className="rounded-xl border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div>
                                        <div className="text-xs uppercase tracking-wide text-slate-500">Long Form Samples</div>
                                        <div className="mt-1 text-sm text-slate-600">
                                            These are sorted from longest to shortest. Reject weak monologues, then choose how many of the longest included ones should flow into training.
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                                            Take top <span className="font-semibold text-slate-900">{longFormConfig?.take_count ?? 0}</span>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => void handleUpdateLongFormTakeCount(Math.max(0, (longFormConfig?.take_count ?? 0) - 1))}
                                            disabled={savingLongFormCount}
                                            className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                        >
                                            -1
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => void handleUpdateLongFormTakeCount((longFormConfig?.take_count ?? 0) + 1)}
                                            disabled={savingLongFormCount}
                                            className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                        >
                                            +1
                                        </button>
                                    </div>
                                </div>

                                {loadingLongForm ? (
                                    <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
                                        <Loader2 size={15} className="animate-spin" />
                                        Loading long-form samples...
                                    </div>
                                ) : !longFormPage || longFormPage.items.length === 0 ? (
                                    <div className="mt-4 rounded-lg border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
                                        No long-form samples were found for this speaker yet.
                                    </div>
                                ) : (
                                    <div className="mt-4 space-y-3">
                                        <div className="grid gap-2 sm:grid-cols-4">
                                            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                                                <div className="text-[11px] uppercase tracking-wide text-slate-500">Included</div>
                                                <div className="mt-1 text-lg font-semibold text-slate-900">{longFormPage.included_count}</div>
                                            </div>
                                            <div className="rounded-lg border border-rose-200 bg-rose-50 p-2.5">
                                                <div className="text-[11px] uppercase tracking-wide text-rose-700">Rejected</div>
                                                <div className="mt-1 text-lg font-semibold text-rose-700">{longFormPage.rejected_count}</div>
                                            </div>
                                            <div className="rounded-lg border border-blue-200 bg-blue-50 p-2.5">
                                                <div className="text-[11px] uppercase tracking-wide text-blue-700">Selected for Training</div>
                                                <div className="mt-1 text-lg font-semibold text-blue-700">{longFormPage.selected_count}</div>
                                            </div>
                                            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                                                <div className="text-[11px] uppercase tracking-wide text-slate-500">Available</div>
                                                <div className="mt-1 text-lg font-semibold text-slate-900">{longFormPage.total}</div>
                                            </div>
                                        </div>

                                        <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
                                            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-2">
                                                <div className="max-h-[780px] space-y-2 overflow-y-auto pr-1">
                                                    {longFormPage.items.map((sample, index) => (
                                                        <button
                                                            key={sample.sample_id}
                                                            type="button"
                                                            onClick={() => setSelectedLongFormId(sample.sample_id)}
                                                            className={`w-full rounded-xl border px-3 py-3 text-left transition ${selectedLongForm?.sample_id === sample.sample_id
                                                                ? 'border-slate-900 bg-white shadow-sm'
                                                                : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                                                                }`}
                                                        >
                                                            <div className="flex items-start justify-between gap-2">
                                                                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">#{index + 1}</div>
                                                                <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${sample.state === 'rejected' ? 'border-rose-200 bg-rose-50 text-rose-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'}`}>
                                                                    {sample.state}
                                                                </span>
                                                            </div>
                                                            <div className="mt-2 line-clamp-2 text-sm font-semibold leading-5 text-slate-900">{sample.video_title}</div>
                                                            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
                                                                <span>{formatDuration(sample.duration_seconds)}</span>
                                                                <span>{sample.word_count} words</span>
                                                                <span>{sample.segment_count} segments</span>
                                                            </div>
                                                            <div className="mt-2 line-clamp-4 text-sm leading-6 text-slate-600">{sample.text}</div>
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                {selectedLongForm ? (
                                                    <div className="space-y-4">
                                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                                            <div>
                                                                <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Selected Long Form</div>
                                                                <div className="mt-1 text-lg font-semibold text-slate-900">{selectedLongForm.video_title}</div>
                                                                <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-600">
                                                                    <span className="rounded-full border border-slate-200 bg-white px-3 py-1">{formatClock(selectedLongForm.start_time)}-{formatClock(selectedLongForm.end_time)}</span>
                                                                    <span className="rounded-full border border-slate-200 bg-white px-3 py-1">{formatDuration(selectedLongForm.duration_seconds)}</span>
                                                                    <span className="rounded-full border border-slate-200 bg-white px-3 py-1">{selectedLongForm.word_count} words</span>
                                                                </div>
                                                            </div>
                                                            <div className="flex items-center gap-2">
                                                                <button
                                                                    type="button"
                                                                    onClick={() => openLongFormAudioPopout(selectedLongForm)}
                                                                    className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700 hover:bg-blue-100"
                                                                >
                                                                    <Radio size={12} />
                                                                    Popout Audio
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => void handleUpdateLongFormState(selectedLongForm.sample_id, 'included')}
                                                                    disabled={updatingLongFormId === selectedLongForm.sample_id || selectedLongForm.state === 'included'}
                                                                    className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                                                                >
                                                                    {updatingLongFormId === selectedLongForm.sample_id && selectedLongForm.state !== 'included' ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                                                                    Include
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => void handleUpdateLongFormState(selectedLongForm.sample_id, 'rejected')}
                                                                    disabled={updatingLongFormId === selectedLongForm.sample_id || selectedLongForm.state === 'rejected'}
                                                                    className="inline-flex items-center gap-1.5 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100 disabled:opacity-50"
                                                                >
                                                                    {updatingLongFormId === selectedLongForm.sample_id && selectedLongForm.state !== 'rejected' ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                                                                    Disapprove
                                                                </button>
                                                            </div>
                                                        </div>

                                                        <div className="rounded-xl border border-slate-200 bg-white p-4">
                                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Transcript</div>
                                                            <div className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-900">{selectedLongForm.text}</div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div className="flex min-h-[360px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white text-sm text-slate-500">
                                                        Select a long-form sample to review it.
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {personalityTaskTab === 'training' && (
                            <div className="space-y-3">
                                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div>
                                            <div className="text-xs uppercase tracking-wide text-slate-500">Training Readiness</div>
                                            <div className="mt-1 text-lg font-semibold text-slate-900">
                                                {dataset?.readiness.can_train_now ? 'Ready to train a personality LoRA' : 'More curation still recommended'}
                                            </div>
                                            <div className="mt-2 text-sm text-slate-600">
                                                {dataset?.readiness.recommended_action || 'Build the dataset first to assess training readiness.'}
                                            </div>
                                        </div>
                                        {dataset && (
                                            <div className="flex flex-wrap gap-2 text-xs">
                                                <span className={`rounded-full border px-3 py-1 font-semibold ${readinessTone(dataset.readiness.status)}`}>
                                                    {dataset.readiness.status}
                                                </span>
                                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 font-semibold text-slate-700">
                                                    score {dataset.readiness.score}
                                                </span>
                                                <span className={`rounded-full border px-3 py-1 font-semibold ${manualReviewTone(dataset.readiness.manual_review_roi)}`}>
                                                    review ROI {dataset.readiness.manual_review_roi}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="grid gap-3 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                                    <div className="rounded-xl border border-slate-200 bg-white p-4">
                                        <div className="flex items-center justify-between gap-3">
                                            <div>
                                                <div className="text-xs uppercase tracking-wide text-slate-500">Training Package</div>
                                                <div className="mt-1 text-sm text-slate-600">Pick a model, dataset breadth, and training strength, then prepare the final train/validation package.</div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => void handleResetToRecommended()}
                                                    disabled={resettingRecommended}
                                                    className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                                                >
                                                    {resettingRecommended ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                                    Reset to Recommended
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => void handlePrepareTrainingPackage()}
                                                    disabled={preparingTrainingPackage || !dataset?.readiness.can_train_now}
                                                    className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                                                >
                                                    {preparingTrainingPackage ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                                    Prepare Training Package
                                                </button>
                                            </div>
                                        </div>

                                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                                            <label className="space-y-1.5 text-sm md:col-span-2">
                                                <div className="text-xs uppercase tracking-wide text-slate-500">Base Model</div>
                                                <select
                                                    value={trainingConfig?.base_model_id || ''}
                                                    onChange={(e) => {
                                                        const modelId = e.target.value;
                                                        setTrainingBaseModelDraft(modelId);
                                                        void handleUpdateTrainingConfig({ base_model_id: modelId });
                                                    }}
                                                    disabled={savingTrainingConfig || !baseModelOptions.length}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                >
                                                    {baseModelOptions.length === 0 ? (
                                                        <option value="">No base models available</option>
                                                    ) : null}
                                                    {baseModelOptions.map((candidate) => (
                                                        <option key={candidate.model_id} value={candidate.model_id}>
                                                            {candidate.label}
                                                            {candidate.recommended ? ' • Recommended' : ''}
                                                            {candidate.installed ? ' • Installed' : ''}
                                                        </option>
                                                    ))}
                                                </select>
                                                <div className="flex flex-wrap items-center gap-2 text-xs">
                                                    {baseModelSupport?.recommended_model_id && (
                                                        <span className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 font-semibold text-blue-700">
                                                            recommended {baseModelSupport.recommended_model_id}
                                                        </span>
                                                    )}
                                                    <span className={`rounded-full border px-3 py-1 font-semibold ${baseModelSupport?.installed ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'}`}>
                                                        {baseModelSupport?.installed ? 'installed locally' : 'not installed'}
                                                    </span>
                                                    {baseModelSupport?.downloading && (
                                                        <span className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 font-semibold text-indigo-700">
                                                            downloading
                                                        </span>
                                                    )}
                                                    <span className={`rounded-full border px-3 py-1 font-semibold ${baseModelSupport?.memory_optimized_available ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-100 text-slate-600'}`}>
                                                        {baseModelSupport?.memory_optimized_available ? '4-bit QLoRA available' : '4-bit QLoRA unavailable'}
                                                    </span>
                                                </div>
                                                <div className="text-xs text-slate-500">
                                                    This exact Hugging Face model will be downloaded and used for personality training.
                                                </div>
                                                {baseModelSupport?.gpu_name && (
                                                    <div className="text-xs text-slate-500">
                                                        {baseModelSupport.gpu_name}{baseModelSupport.gpu_vram_gb ? ` • ${baseModelSupport.gpu_vram_gb.toFixed(1)} GB VRAM` : ''}
                                                    </div>
                                                )}
                                                {baseModelSupport?.rationale && (
                                                    <div className="text-xs text-slate-500">{baseModelSupport.rationale}</div>
                                                )}
                                                {baseModelSupport?.download_message && (
                                                    <div className={`text-xs ${baseModelSupport.download_status === 'failed' ? 'text-rose-600' : 'text-slate-500'}`}>
                                                        {baseModelSupport.download_message}
                                                    </div>
                                                )}
                                                {baseModelSupport?.memory_optimized_reason && (
                                                    <div className={`text-xs ${baseModelSupport.memory_optimized_available ? 'text-slate-500' : 'text-amber-700'}`}>
                                                        {baseModelSupport.memory_optimized_reason}
                                                    </div>
                                                )}
                                                <div className="flex flex-wrap gap-2">
                                                    <button
                                                        type="button"
                                                        onClick={() => {
                                                            const modelId = baseModelSupport?.recommended_model_id || RECOMMENDED_TRAINING_CONFIG.base_model_id;
                                                            setTrainingBaseModelDraft(modelId);
                                                            void handleUpdateTrainingConfig({ base_model_id: modelId });
                                                        }}
                                                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                                                    >
                                                        Use Recommended
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() => void handleDownloadBaseModel(trainingConfig?.base_model_id || baseModelSupport?.recommended_model_id || undefined)}
                                                        disabled={downloadingBaseModel || baseModelSupport?.downloading}
                                                        className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
                                                    >
                                                        {downloadingBaseModel || baseModelSupport?.downloading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                        Download Selected Model
                                                    </button>
                                                </div>
                                                {baseModelSupport?.candidates?.length ? (
                                                    <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                                                        {baseModelSupport.candidates.map((candidate) => (
                                                            <button
                                                                key={candidate.model_id}
                                                                type="button"
                                                                onClick={() => {
                                                                    setTrainingBaseModelDraft(candidate.model_id);
                                                                    void handleUpdateTrainingConfig({ base_model_id: candidate.model_id });
                                                                }}
                                                                className={`rounded-full border px-3 py-1 ${candidate.recommended ? 'border-blue-200 bg-blue-50 text-blue-700' : candidate.installed ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-600'}`}
                                                            >
                                                                {candidate.label}
                                                            </button>
                                                        ))}
                                                    </div>
                                                ) : null}
                                            </label>

                                            <div className="md:col-span-2">
                                                <button
                                                    type="button"
                                                    onClick={() => setShowAdvancedTraining((current) => !current)}
                                                    className="flex w-full items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-left"
                                                >
                                                    <div>
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Advanced</div>
                                                        <div className="mt-1 text-sm text-slate-600">Training mode, export strategy, evaluation holdout, snapshot cadence, and custom caps.</div>
                                                    </div>
                                                    <span className="text-xs font-semibold text-slate-700">{showAdvancedTraining ? 'Hide' : 'Show'}</span>
                                                </button>
                                            </div>

                                            {showAdvancedTraining && (
                                                <>
                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Training Mode</div>
                                                        <select
                                                            value={trainingConfig?.training_mode || 'memory_optimized'}
                                                            onChange={(e) => void handleUpdateTrainingConfig({ training_mode: e.target.value as AvatarPersonalityTrainingConfig['training_mode'] })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        >
                                                            <option value="memory_optimized">Memory Optimized 4-bit QLoRA</option>
                                                            <option value="standard">Standard LoRA</option>
                                                        </select>
                                                    </label>

                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Export Strategy</div>
                                                        <select
                                                            value={trainingConfig?.export_strategy || 'gold_balanced'}
                                                            onChange={(e) => void handleUpdateTrainingConfig({ export_strategy: e.target.value as AvatarPersonalityTrainingConfig['export_strategy'] })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        >
                                                            <option value="gold_balanced">Gold Balanced</option>
                                                            <option value="gold_only">Gold Only</option>
                                                            <option value="gold_plus_top_silver">Gold + Top Silver</option>
                                                            <option value="full_approved">Full Approved</option>
                                                        </select>
                                                    </label>
                                                </>
                                            )}

                                            <label className="space-y-1.5 text-sm">
                                                <div className="text-xs uppercase tracking-wide text-slate-500">Dataset Breadth</div>
                                                <select
                                                    value={trainingConfig?.dataset_profile || 'balanced'}
                                                    onChange={(e) => void handleUpdateTrainingConfig({ dataset_profile: e.target.value as AvatarPersonalityTrainingConfig['dataset_profile'] })}
                                                    disabled={savingTrainingConfig}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                >
                                                    {datasetProfileOptions.map((option) => (
                                                        <option key={option.key} value={option.key}>
                                                            {option.label}
                                                            {option.recommended ? ' - Recommended' : ''}
                                                        </option>
                                                    ))}
                                                </select>
                                                <div className="text-xs text-slate-500">
                                                    {datasetProfileOptions.find((option) => option.key === (trainingConfig?.dataset_profile || 'balanced'))?.summary || 'Choose how broad the training package should be.'}
                                                </div>
                                            </label>

                                            <label className="space-y-1.5 text-sm">
                                                <div className="text-xs uppercase tracking-wide text-slate-500">Training Strength</div>
                                                <select
                                                    value={trainingConfig?.training_strength || 'balanced'}
                                                    onChange={(e) => void handleUpdateTrainingConfig({ training_strength: e.target.value as AvatarPersonalityTrainingConfig['training_strength'] })}
                                                    disabled={savingTrainingConfig}
                                                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                >
                                                    {TRAINING_STRENGTH_OPTIONS.map((option) => (
                                                        <option key={option.value} value={option.value}>
                                                            {option.label}
                                                        </option>
                                                    ))}
                                                </select>
                                                <div className="text-xs text-slate-500">
                                                    {TRAINING_STRENGTH_OPTIONS.find((option) => option.value === (trainingConfig?.training_strength || 'balanced'))?.summary}
                                                    {' '}Uses learning rate {trainingStrengthLearningRate(trainingConfig?.training_strength).toFixed(5)}.
                                                </div>
                                            </label>

                                            {showAdvancedTraining && trainingConfig?.dataset_profile === 'custom' ? (
                                                <>
                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Custom Conversation Cap</div>
                                                        <input
                                                            type="number"
                                                            min={0}
                                                            step={100}
                                                            value={trainingConfig?.max_examples ?? 2500}
                                                            onChange={(e) => void handleUpdateTrainingConfig({
                                                                dataset_profile: 'custom',
                                                                max_examples: Number(e.target.value || 0),
                                                            })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        />
                                                    </label>

                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Custom Long-Form Cap</div>
                                                        <input
                                                            type="number"
                                                            min={0}
                                                            step={10}
                                                            value={trainingConfig?.max_long_form_examples ?? 80}
                                                            onChange={(e) => void handleUpdateTrainingConfig({
                                                                dataset_profile: 'custom',
                                                                max_long_form_examples: Number(e.target.value || 0),
                                                            })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        />
                                                    </label>
                                                </>
                                            ) : (
                                                <div className="grid gap-2 sm:grid-cols-2">
                                                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Conversation Cap</div>
                                                        <div className="mt-1 font-semibold text-slate-900">{trainingConfig?.max_examples ?? 0}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-sm text-indigo-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-indigo-600">Long-Form Cap</div>
                                                        <div className="mt-1 font-semibold">{trainingConfig?.max_long_form_examples ?? 0}</div>
                                                    </div>
                                                </div>
                                            )}

                                            {showAdvancedTraining && (
                                                <>
                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Hold Out For Evaluation</div>
                                                        <input
                                                            type="number"
                                                            min={0.01}
                                                            max={0.2}
                                                            step={0.01}
                                                            value={trainingConfig?.validation_ratio ?? 0.10}
                                                            onChange={(e) => void handleUpdateTrainingConfig({ validation_ratio: Number(e.target.value || 0.10) })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        />
                                                        <div className="text-xs text-slate-500">
                                                            Reserved for evaluation only. Higher values give cleaner checkpoint comparison but fewer training rows.
                                                        </div>
                                                    </label>

                                                    <label className="space-y-1.5 text-sm">
                                                        <div className="text-xs uppercase tracking-wide text-slate-500">Snapshot Cadence</div>
                                                        <select
                                                            value={trainingConfig?.snapshot_interval_steps ?? 0}
                                                            onChange={(e) => void handleUpdateTrainingConfig({ snapshot_interval_steps: Number(e.target.value || 0) })}
                                                            disabled={savingTrainingConfig}
                                                            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                        >
                                                            {SNAPSHOT_INTERVAL_OPTIONS.map((option) => (
                                                                <option key={option.value} value={option.value}>
                                                                    {option.label}
                                                                </option>
                                                            ))}
                                                        </select>
                                                        <div className="text-xs text-slate-500">
                                                            `Auto` resolves to roughly five snapshots across the run, plus epoch/final adapters.
                                                        </div>
                                                        {activeTrainingPlan && (trainingConfig?.snapshot_interval_steps ?? 0) > 0 && (trainingConfig?.snapshot_interval_steps ?? 0) >= activeTrainingPlan.estimated_total_steps && activeTrainingPlan.estimated_total_steps > 0 && (
                                                            <div className="text-xs text-amber-700">
                                                                This interval is larger than the estimated run length. The trainer will fall back to a smaller auto cadence.
                                                            </div>
                                                        )}
                                                    </label>
                                                </>
                                            )}

                                            <label className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-700">
                                                <span>Include long-form reasoning samples</span>
                                                <input
                                                    type="checkbox"
                                                    checked={trainingConfig?.include_long_form ?? true}
                                                    onChange={(e) => void handleUpdateTrainingConfig({ include_long_form: e.target.checked })}
                                                    disabled={savingTrainingConfig}
                                                    className="h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500"
                                                />
                                            </label>

                                            {activeTrainingPlan && (
                                                <div className={`rounded-xl border px-3 py-3 text-sm ${trainingStepBandTone(activeTrainingPlan.step_band)}`}>
                                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                                        <div className="font-semibold">{activeTrainingPlan.headline}</div>
                                                        <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${trainingStepBandTone(activeTrainingPlan.step_band)}`}>
                                                            {trainingStepBandLabel(activeTrainingPlan.step_band)}
                                                        </span>
                                                    </div>
                                                    <div className="mt-2 text-xs leading-5">
                                                        {activeTrainingPlan.recommendation}
                                                    </div>
                                                    <div className="mt-3 grid gap-2 sm:grid-cols-3">
                                                        <div className="rounded-lg border border-white/60 bg-white/70 px-3 py-2">
                                                            <div className="text-[11px] uppercase tracking-wide opacity-70">Estimated Package</div>
                                                            <div className="mt-1 font-semibold">{activeTrainingPlan.estimated_total_examples} rows</div>
                                                        </div>
                                                        <div className="rounded-lg border border-white/60 bg-white/70 px-3 py-2">
                                                            <div className="text-[11px] uppercase tracking-wide opacity-70">1-Epoch Steps</div>
                                                            <div className="mt-1 font-semibold">{activeTrainingPlan.estimated_steps_per_epoch}</div>
                                                        </div>
                                                        <div className="rounded-lg border border-white/60 bg-white/70 px-3 py-2">
                                                            <div className="text-[11px] uppercase tracking-wide opacity-70">Effective Batch</div>
                                                            <div className="mt-1 font-semibold">{activeTrainingPlan.estimated_effective_batch_size}</div>
                                                        </div>
                                                    </div>
                                                    <div className="mt-3 text-xs leading-5">
                                                        Snapshot suggestion: {activeTrainingPlan.snapshot_interval_suggestion > 0 ? `every ${activeTrainingPlan.snapshot_interval_suggestion} steps` : 'use Auto'}.
                                                    </div>
                                                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
                                                        <div className="rounded-lg border border-white/60 bg-white/70 px-3 py-2">
                                                            <div className="text-[11px] uppercase tracking-wide opacity-70">Pros</div>
                                                            <div className="mt-1 space-y-1 text-xs">
                                                                {activeTrainingPlan.pros.map((item) => (
                                                                    <div key={item}>{item}</div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg border border-white/60 bg-white/70 px-3 py-2">
                                                            <div className="text-[11px] uppercase tracking-wide opacity-70">Tradeoffs</div>
                                                            <div className="mt-1 space-y-1 text-xs">
                                                                {activeTrainingPlan.cons.map((item) => (
                                                                    <div key={item}>{item}</div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        <div className="mt-4 space-y-2 text-sm text-slate-700">
                                            <div className={`flex items-center justify-between rounded-lg border px-3 py-2 ${trainingConfig?.training_mode === 'memory_optimized' ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
                                                <span>Training mode</span>
                                                <span className="font-semibold">{trainingConfig?.training_mode === 'memory_optimized' ? 'Memory Optimized 4-bit QLoRA' : 'Standard LoRA'}</span>
                                            </div>
                                            <div className={`flex items-center justify-between rounded-lg border px-3 py-2 ${trainingStrengthTone(trainingConfig?.training_strength)}`}>
                                                <span>Training strength</span>
                                                <span className="font-semibold">{trainingStrengthLabel(trainingConfig?.training_strength)} • LR {trainingStrengthLearningRate(trainingConfig?.training_strength).toFixed(5)}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                                                <span>Base model</span>
                                                <span className="font-semibold text-slate-900">{trainingConfig?.base_model_id || dataset?.base_model_id || workbench.suggested_base_model || 'Qwen/Qwen3-8B'}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-700">
                                                <span>Gold examples</span>
                                                <span className="font-semibold">{dataset?.gold_example_count ?? 0}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-blue-700">
                                                <span>Included examples</span>
                                                <span className="font-semibold">{workbench.personality.approved_count}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                                                <span>Approved hours</span>
                                                <span className="font-semibold text-slate-900">{dataset?.readiness.approved_duration_hours?.toFixed(1) || '0.0'}h</span>
                                            </div>
                                            <div className={`flex items-center justify-between rounded-lg border px-3 py-2 ${(trainingConfig?.include_long_form ?? true) ? 'border-indigo-200 bg-indigo-50 text-indigo-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
                                                <span>Reasoning samples</span>
                                                <span className="font-semibold">{trainingConfig?.include_long_form ? `top ${longFormConfig?.take_count ?? 0} / ${longFormConfig?.included_count ?? 0}` : 'disabled'}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                                                <span>Dataset profile</span>
                                                <span className="font-semibold text-slate-900">{activeTrainingPlan?.dataset_profile_label || 'Balanced'}</span>
                                            </div>
                                            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                                                <span>Snapshot cadence</span>
                                                <span className="font-semibold text-slate-900">
                                                    {(trainingConfig?.snapshot_interval_steps ?? 0) > 0
                                                        ? `every ${trainingConfig?.snapshot_interval_steps} steps`
                                                        : 'auto'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                                        <div className="flex items-center justify-between gap-3">
                                            <div>
                                                <div className="text-xs uppercase tracking-wide text-slate-500">Prepared Export</div>
                                                <div className="mt-1 text-sm text-slate-600">
                                                    {loadingTrainingPackage ? 'Loading prepared package...' : trainingPackage?.status === 'ready' ? 'Train and validation files are ready.' : 'No training package has been prepared yet.'}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {trainingPackage?.status === 'ready' && (
                                                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
                                                        ready
                                                    </span>
                                                )}
                                                <button
                                                    type="button"
                                                    onClick={() => void handleStartTraining()}
                                                    disabled={
                                                        startingTraining
                                                        || trainingStatus?.active
                                                        || trainingPackage?.status !== 'ready'
                                                        || !baseModelSupport?.installed
                                                        || (trainingConfig?.training_mode === 'memory_optimized' && !baseModelSupport?.memory_optimized_available)
                                                    }
                                                    className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                                                >
                                                    {startingTraining ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
                                                    Start Training
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => void handleStopTraining()}
                                                    disabled={!trainingStatus?.active || stoppingTraining}
                                                    className="inline-flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                                                >
                                                    {stoppingTraining ? <Loader2 size={14} className="animate-spin" /> : <Square size={14} />}
                                                    Stop
                                                </button>
                                            </div>
                                        </div>

                                            <div className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                <div className="flex flex-wrap items-center gap-2">
                                                    <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${statusTone(trainingStatus?.status || 'draft')}`}>
                                                        {trainingStatus?.status || 'idle'}
                                                    </span>
                                                    <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${(trainingStatus?.training_mode || trainingConfig?.training_mode) === 'memory_optimized' ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-100 text-slate-700'}`}>
                                                        {(trainingStatus?.training_mode || trainingConfig?.training_mode) === 'memory_optimized' ? 'memory optimized' : 'standard'}
                                                    </span>
                                                    {trainingStatus?.current_stage && (
                                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                                                            {trainingStatus.current_stage}
                                                        </span>
                                                    )}
                                                </div>
                                            <div className="mt-3">
                                                <div className="flex items-center justify-between text-xs text-slate-500">
                                                    <span>Progress</span>
                                                    <span>{trainingProgress.progress.toFixed(1)}%</span>
                                                </div>
                                                <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-100">
                                                    <div
                                                        className="h-full rounded-full bg-slate-900 transition-all duration-300"
                                                        style={{ width: `${trainingProgress.progress}%` }}
                                                    />
                                                </div>
                                            </div>
                                            <div className="mt-2 grid gap-2 sm:grid-cols-4">
                                                <div>Epoch: <span className="font-semibold text-slate-900">{trainingStatus?.epoch?.toFixed(2) || '0.00'}</span></div>
                                                <div>Step: <span className="font-semibold text-slate-900">{trainingStatus?.step ?? 0}/{trainingStatus?.max_steps ?? 0}</span></div>
                                                <div>Loss: <span className="font-semibold text-slate-900">{trainingStatus?.latest_loss != null ? trainingStatus.latest_loss.toFixed(4) : '—'}</span></div>
                                                <div>Snapshots: <span className="font-semibold text-slate-900">{trainingStatus?.snapshot_interval_steps ? `every ${trainingStatus.snapshot_interval_steps}` : 'auto'}</span></div>
                                            </div>
                                            <div className="mt-2 grid gap-2 sm:grid-cols-3">
                                                <div>Elapsed: <span className="font-semibold text-slate-900">{formatHoursMinutes(trainingProgress.elapsedSeconds)}</span></div>
                                                <div>Remaining: <span className="font-semibold text-slate-900">{formatHoursMinutes(trainingProgress.etaSeconds)}</span></div>
                                                <div>Rate: <span className="font-semibold text-slate-900">{trainingProgress.stepsPerMinute ? `${trainingProgress.stepsPerMinute.toFixed(1)} steps/min` : '—'}</span></div>
                                            </div>
                                            {(trainingStatus?.message || trainingStatus?.error) && (
                                                <div className={`mt-2 text-sm ${trainingStatus?.error ? 'text-rose-700' : 'text-slate-600'}`}>
                                                    {trainingStatus?.error || trainingStatus?.message}
                                                </div>
                                            )}
                                            </div>
                                            {trainingConfig?.training_mode === 'memory_optimized' && !baseModelSupport?.memory_optimized_available && (
                                                <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                                                    Memory optimized mode is selected, but 4-bit QLoRA is not currently available in the backend environment.
                                                </div>
                                            )}

                                        {trainingPackage?.status === 'ready' ? (
                                                <div className="mt-4 space-y-3">
                                                    <div className="grid gap-2 sm:grid-cols-2">
                                                        <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Dataset Profile</div>
                                                            <div className="mt-1 font-semibold text-slate-900">{trainingPackage.training_plan?.dataset_profile_label || activeTrainingPlan?.dataset_profile_label || 'Balanced'}</div>
                                                        </div>
                                                        <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Strategy</div>
                                                            <div className="mt-1 font-semibold text-slate-900">{trainingStrategyLabel(trainingPackage.export_strategy)}</div>
                                                        </div>
                                                    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Split</div>
                                                        <div className="mt-1 font-semibold text-slate-900">{trainingPackage.train_examples} train / {trainingPackage.validation_examples} val</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Conversation Rows</div>
                                                        <div className="mt-1 font-semibold text-slate-900">{trainingPackage.conversation_examples_selected}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-sm text-indigo-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-indigo-600">Long Form Rows</div>
                                                        <div className="mt-1 font-semibold">{trainingPackage.long_form_examples_selected}</div>
                                                    </div>
                                                    </div>

                                                    {trainingPackage.training_plan && (
                                                        <div className={`rounded-lg border px-3 py-3 text-sm ${trainingStepBandTone(trainingPackage.training_plan.step_band)}`}>
                                                            <div className="flex flex-wrap items-center justify-between gap-2">
                                                                <div className="font-semibold">{trainingPackage.training_plan.headline}</div>
                                                                <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${trainingStepBandTone(trainingPackage.training_plan.step_band)}`}>
                                                                    {trainingStepBandLabel(trainingPackage.training_plan.step_band)}
                                                                </span>
                                                            </div>
                                                            <div className="mt-2 grid gap-2 sm:grid-cols-4 text-xs">
                                                                <div>Rows: <span className="font-semibold">{trainingPackage.training_plan.estimated_total_examples}</span></div>
                                                                <div>Train: <span className="font-semibold">{trainingPackage.training_plan.estimated_train_examples}</span></div>
                                                                <div>Val: <span className="font-semibold">{trainingPackage.training_plan.estimated_validation_examples}</span></div>
                                                                <div>1-epoch steps: <span className="font-semibold">{trainingPackage.training_plan.estimated_steps_per_epoch}</span></div>
                                                            </div>
                                                        </div>
                                                    )}

                                                <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Manifest</div>
                                                    <div className="mt-1 break-all font-mono text-xs text-slate-700">{trainingPackage.manifest_path}</div>
                                                </div>
                                                <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Train Dataset</div>
                                                    <div className="mt-1 break-all font-mono text-xs text-slate-700">{trainingPackage.train_dataset_path}</div>
                                                </div>
                                                <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Validation Dataset</div>
                                                    <div className="mt-1 break-all font-mono text-xs text-slate-700">{trainingPackage.validation_dataset_path}</div>
                                                </div>
                                                {trainingPackage.command_preview && (
                                                    <div className="rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-sm text-slate-700">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">CLI Preflight</div>
                                                        <div className="mt-1 break-all font-mono text-xs text-slate-700">{trainingPackage.command_preview}</div>
                                                    </div>
                                                )}

                                                {(trainingStatus?.adapter_path || trainingPackage?.status === 'ready') && (
                                                    <div className="rounded-lg border border-slate-200 bg-white p-3">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Test Interaction</div>
                                                        {trainingStatus?.snapshots?.length ? (
                                                            <div className="mt-3 grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 md:grid-cols-[minmax(0,1fr)_auto]">
                                                                <div className="space-y-1.5">
                                                                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Snapshot</div>
                                                                    <select
                                                                        value={selectedSnapshotPath}
                                                                        onChange={(e) => setSelectedSnapshotPath(e.target.value)}
                                                                        disabled={trainingStatus?.active || sendingChat}
                                                                        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                                    >
                                                                        {trainingStatus.snapshots.map((snapshot) => (
                                                                            <option key={snapshot.adapter_path} value={snapshot.adapter_path}>
                                                                                {snapshot.label}
                                                                                {snapshot.selected ? ' • selected' : ''}
                                                                                {snapshot.eval_loss != null ? ` • eval ${snapshot.eval_loss.toFixed(4)}` : ''}
                                                                            </option>
                                                                        ))}
                                                                    </select>
                                                                    <div className="text-xs text-slate-500">
                                                                        Switch checkpoints here to compare style strength before promoting one as the canonical adapter.
                                                                    </div>
                                                                </div>
                                                                <div className="flex flex-wrap items-end gap-2">
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handlePromoteSnapshot()}
                                                                        disabled={
                                                                            promotingSnapshot
                                                                            || trainingStatus?.active
                                                                            || !selectedSnapshotPath
                                                                            || selectedSnapshotPath === (trainingStatus?.adapter_path || '')
                                                                        }
                                                                        className="inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
                                                                    >
                                                                        {promotingSnapshot ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                                                                        Promote
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handleDeleteOtherSnapshots()}
                                                                        disabled={cleaningSnapshots || trainingStatus?.active || !selectedSnapshotPath || (trainingStatus?.snapshots?.length || 0) <= 1}
                                                                        className="inline-flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                                                                    >
                                                                        {cleaningSnapshots ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                                                                        Delete Others
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => void handleDeleteSnapshot()}
                                                                        disabled={deletingSnapshot || trainingStatus?.active || !selectedSnapshotPath || (trainingStatus?.snapshots?.length || 0) <= 1}
                                                                        className="inline-flex items-center gap-2 rounded-lg border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50"
                                                                    >
                                                                        {deletingSnapshot ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                                                                        Delete Snapshot
                                                                    </button>
                                                                </div>
                                                            </div>
                                                        ) : null}
                                                        <div className="mt-3 max-h-[280px] space-y-2 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3">
                                                            {chatHistory.length === 0 ? (
                                                                <div className="text-sm text-slate-500">Start a short conversation once training completes.</div>
                                                            ) : (
                                                                chatHistory.map((turn, index) => (
                                                                    <div key={`${turn.role}-${index}`} className={`rounded-lg px-3 py-2 text-sm ${turn.role === 'assistant' ? 'bg-white text-slate-800' : 'bg-blue-50 text-blue-900'}`}>
                                                                        <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">{turn.role}</div>
                                                                        <div className="whitespace-pre-wrap leading-6">{turn.content}</div>
                                                                    </div>
                                                                ))
                                                            )}
                                                        </div>
                                                        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                                                            <div>
                                                                Fit check runs a fixed prompt battery against this snapshot and asks the local judge model whether it looks underfit, balanced, or overfit.
                                                            </div>
                                                            <button
                                                                type="button"
                                                                onClick={() => void handleRunFitCheck()}
                                                                disabled={runningFitCheck || trainingStatus?.status !== 'completed' || !selectedSnapshotPath}
                                                                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
                                                            >
                                                                {runningFitCheck ? <Loader2 size={12} className="animate-spin" /> : <Brain size={12} />}
                                                                Run Fit Check
                                                            </button>
                                                        </div>
                                                        {fitCheckResult && (
                                                            <div className={`mt-3 rounded-lg border px-3 py-3 text-sm ${fitCheckTone(fitCheckResult.classification)}`}>
                                                                <div className="flex flex-wrap items-center justify-between gap-2">
                                                                    <div className="font-semibold">{fitCheckLabel(fitCheckResult.classification)}</div>
                                                                    <div className="flex flex-wrap items-center gap-2 text-xs">
                                                                        <span className={`rounded-full border px-2.5 py-1 font-semibold ${fitCheckTone(fitCheckResult.classification)}`}>
                                                                            {fitCheckResult.confidence}% confidence
                                                                        </span>
                                                                        {fitCheckResult.judge_model && (
                                                                            <span className="rounded-full border border-white/70 bg-white/70 px-2.5 py-1 font-medium text-slate-700">
                                                                                Judge: {fitCheckResult.judge_model}
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                                {fitCheckResult.summary && (
                                                                    <div className="mt-2 text-sm leading-6">{fitCheckResult.summary}</div>
                                                                )}
                                                                <div className="mt-3 grid gap-2 md:grid-cols-3">
                                                                    <div className="rounded-lg border border-white/70 bg-white/70 px-3 py-2">
                                                                        <div className="text-[11px] uppercase tracking-wide opacity-70">Strengths</div>
                                                                        <div className="mt-1 space-y-1 text-xs">
                                                                            {fitCheckResult.strengths.length ? fitCheckResult.strengths.map((item) => (
                                                                                <div key={item}>{item}</div>
                                                                            )) : <div>None flagged.</div>}
                                                                        </div>
                                                                    </div>
                                                                    <div className="rounded-lg border border-white/70 bg-white/70 px-3 py-2">
                                                                        <div className="text-[11px] uppercase tracking-wide opacity-70">Concerns</div>
                                                                        <div className="mt-1 space-y-1 text-xs">
                                                                            {fitCheckResult.concerns.length ? fitCheckResult.concerns.map((item) => (
                                                                                <div key={item}>{item}</div>
                                                                            )) : <div>No major issues flagged.</div>}
                                                                        </div>
                                                                    </div>
                                                                    <div className="rounded-lg border border-white/70 bg-white/70 px-3 py-2">
                                                                        <div className="text-[11px] uppercase tracking-wide opacity-70">Recommendations</div>
                                                                        <div className="mt-1 space-y-1 text-xs">
                                                                            {fitCheckResult.recommendations.length ? fitCheckResult.recommendations.map((item) => (
                                                                                <div key={item}>{item}</div>
                                                                            )) : <div>No recommendation returned.</div>}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                                <div className="mt-3 space-y-2">
                                                                    {fitCheckResult.results.map((result) => (
                                                                        <div key={result.key} className="rounded-lg border border-white/70 bg-white/70 px-3 py-2 text-xs text-slate-700">
                                                                            <div className="font-semibold text-slate-900">{result.prompt}</div>
                                                                            <div className="mt-1 whitespace-pre-wrap leading-5">{result.reply}</div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}
                                                        <div className="mt-3 flex gap-2">
                                                            <textarea
                                                                value={chatInput}
                                                                onChange={(e) => setChatInput(e.target.value)}
                                                                placeholder="Say something to the trained avatar..."
                                                                rows={3}
                                                                className="min-h-[76px] flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                                                            />
                                                            <button
                                                                type="button"
                                                                onClick={() => void handleSendTestChat()}
                                                                disabled={sendingChat || !chatInput.trim() || trainingStatus?.status !== 'completed' || !selectedSnapshotPath}
                                                                className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                                                            >
                                                                {sendingChat ? <Loader2 size={14} className="animate-spin" /> : 'Send'}
                                                            </button>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-white px-3 py-3 text-sm text-slate-600">
                                                Build the final export after you are satisfied with curation. This package becomes the handoff point for the LoRA trainer.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </SectionCard>
                </div>

                <div className={activeTab === 'appearance' ? '' : 'hidden'}>
                <SectionCard
                    title="Appearance"
                    icon={<ImageIcon size={18} />}
                    status={workbench.appearance.status}
                    sourceLabel="Image Sources"
                    sourceCount={workbench.appearance.source_count}
                    approvedLabel="Approved Images"
                    approvedCount={workbench.appearance.approved_count}
                    artifactReady={workbench.appearance.artifact_ready}
                    summary={workbench.appearance.summary}
                >
                    <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
                        This section will manage portrait selection, face validation, reference galleries, and the future appearance training/export path.
                    </div>
                </SectionCard>
                </div>

                <div className={activeTab === 'voice' ? '' : 'hidden'}>
                <SectionCard
                    title="Voice"
                    icon={<Mic2 size={18} />}
                    status={workbench.voice.status}
                    sourceLabel="Reference Clips"
                    sourceCount={workbench.voice.source_count}
                    approvedLabel="Approved Profiles"
                    approvedCount={workbench.voice.approved_count}
                    artifactReady={workbench.voice.artifact_ready}
                    summary={workbench.voice.summary}
                >
                    <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
                        This section will own reference-clip selection, one-shot cloning, preview generation, and future advanced voice asset workflows.
                    </div>
                </SectionCard>
                </div>
            </div>

            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
                        <Radio size={18} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-slate-900">Workbench State</h2>
                        <p className="text-sm text-slate-500">This first slice establishes the speaker-based avatar record and the three independent domains.</p>
                    </div>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                        <div className="text-xs uppercase tracking-wide text-slate-500">Artifacts Directory</div>
                        <div className="mt-1 break-all font-mono text-xs text-slate-600">{workbench.artifacts_dir || 'Pending'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                        <div className="text-xs uppercase tracking-wide text-slate-500">Next Build Steps</div>
                        <div className="mt-1">Dataset curation, image gallery approval, and voice-reference selection can now be layered onto this studio record.</div>
                    </div>
                </div>
            </section>
        </div>
    );
}
