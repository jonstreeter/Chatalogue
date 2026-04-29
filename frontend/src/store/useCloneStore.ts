import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type {
    EpisodeCloneEngine,
    EpisodeCloneJob,
    EpisodeCloneConceptsResponse,
    EpisodeCloneGenerateResponse,
} from '../types';

// These types live only in the VideoDetailPage module currently — duplicated here
// until they can be shared via types.ts.
export type OllamaLocalModel = {
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

// Module-level counters replace useRef — mutated without triggering re-renders.
let _generateRequestId = 0;
let _jobsRequestId = 0;

const DEFAULT_STYLE_PROMPT =
    'Create a fresh, original script with a clear hook, stronger structure, and a more polished delivery than the source.';

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------
export interface CloneState {
    // Engine selection
    cloneEngines: EpisodeCloneEngine[];
    loadingCloneEngines: boolean;
    cloneEnginesError: string | null;
    cloneEngineKey: string;

    // Ollama model selection (only relevant when selected engine provider === 'ollama')
    cloneOllamaModels: OllamaLocalModel[];
    loadingCloneOllamaModels: boolean;
    cloneOllamaModelsError: string | null;
    cloneOllamaModel: string;

    // Generation inputs
    cloneStylePrompt: string;
    cloneNotes: string;
    cloneConcepts: EpisodeCloneConceptsResponse | null;
    cloneConceptsText: string;
    cloneExcludedReferencesText: string;
    cloneBatchSize: number;

    // Async status flags
    detectingCloneConcepts: boolean;
    generatingClone: boolean;
    copiedCloneScript: boolean;

    // Job history
    cloneJobs: EpisodeCloneJob[];
    loadingCloneJobs: boolean;
    cloneJobsError: string | null;
    selectedCloneJobId: number | null;

    // ---------------------------------------------------------------------------
    // Simple setters
    // ---------------------------------------------------------------------------
    setCloneEngineKey: (key: string) => void;
    setCloneOllamaModel: (model: string) => void;
    setCloneStylePrompt: (prompt: string) => void;
    setCloneNotes: (notes: string) => void;
    setCloneConceptsText: (text: string) => void;
    setCloneExcludedReferencesText: (text: string) => void;
    setCloneBatchSize: (size: number) => void;
    setSelectedCloneJobId: (id: number | null) => void;

    // ---------------------------------------------------------------------------
    // Async data-fetching actions
    // ---------------------------------------------------------------------------
    fetchCloneEngines: (signal?: AbortSignal) => Promise<void>;
    fetchCloneOllamaModels: (signal?: AbortSignal) => Promise<void>;
    fetchCloneJobs: (
        videoId: number,
        signal?: AbortSignal,
        options?: { preferredJobId?: number | null; silent?: boolean }
    ) => Promise<EpisodeCloneJob[] | null>;

    // ---------------------------------------------------------------------------
    // Business-logic actions
    // ---------------------------------------------------------------------------
    loadCloneVariantInputs: (job: EpisodeCloneJob) => void;
    handleDetectCloneConcepts: (videoId: number) => Promise<void>;
    handleGenerateEpisodeClone: (videoId: number) => Promise<void>;
    handleCopyCloneScript: () => Promise<void>;
    cloneJobMatchesVisibleInputs: (job: EpisodeCloneJob | null | undefined) => boolean;

    // ---------------------------------------------------------------------------
    // Reset (call when the viewed video changes)
    // ---------------------------------------------------------------------------
    resetCloneState: () => void;
}

// ---------------------------------------------------------------------------
// Internal helpers (pure functions — no store access needed)
// ---------------------------------------------------------------------------
function normalizeTextList(value: string): string[] {
    return value
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function listsMatch(a: string[], b: string[]): boolean {
    return a.length === b.length && a.every((item, i) => item === b[i]);
}

function resolveEngineOverride(
    engines: EpisodeCloneEngine[],
    engineKey: string,
    ollamaModel: string
): { provider_override: string | undefined; model_override: string | undefined } {
    const selected = engines.find((e) => e.key === engineKey) || engines[0] || null;
    const usesOllama = (selected?.provider || '') === 'ollama';
    const overrideModel = usesOllama
        ? ollamaModel.trim() || selected?.model || ''
        : selected?.model || '';

    if (!selected || selected.key === 'default') {
        return { provider_override: undefined, model_override: overrideModel || undefined };
    }
    return {
        provider_override: selected.provider || undefined,
        model_override: overrideModel || undefined,
    };
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------
export const useCloneStore = create<CloneState>()(
    devtools(
        (set, get) => ({
            // -----------------------------------------------------------------
            // Initial state
            // -----------------------------------------------------------------
            cloneEngines: [],
            loadingCloneEngines: false,
            cloneEnginesError: null,
            cloneEngineKey: 'default',

            cloneOllamaModels: [],
            loadingCloneOllamaModels: false,
            cloneOllamaModelsError: null,
            cloneOllamaModel: '',

            cloneStylePrompt: DEFAULT_STYLE_PROMPT,
            cloneNotes: '',
            cloneConcepts: null,
            cloneConceptsText: '',
            cloneExcludedReferencesText: '',
            cloneBatchSize: 1,

            detectingCloneConcepts: false,
            generatingClone: false,
            copiedCloneScript: false,

            cloneJobs: [],
            loadingCloneJobs: false,
            cloneJobsError: null,
            selectedCloneJobId: null,

            // -----------------------------------------------------------------
            // Simple setters
            // -----------------------------------------------------------------
            setCloneEngineKey: (key) => set({ cloneEngineKey: key }, false, 'setCloneEngineKey'),
            setCloneOllamaModel: (model) => set({ cloneOllamaModel: model }, false, 'setCloneOllamaModel'),
            setCloneStylePrompt: (prompt) => set({ cloneStylePrompt: prompt }, false, 'setCloneStylePrompt'),
            setCloneNotes: (notes) => set({ cloneNotes: notes }, false, 'setCloneNotes'),
            setCloneConceptsText: (text) => set({ cloneConceptsText: text }, false, 'setCloneConceptsText'),
            setCloneExcludedReferencesText: (text) =>
                set({ cloneExcludedReferencesText: text }, false, 'setCloneExcludedReferencesText'),
            setCloneBatchSize: (size) => set({ cloneBatchSize: size }, false, 'setCloneBatchSize'),
            setSelectedCloneJobId: (id) => set({ selectedCloneJobId: id }, false, 'setSelectedCloneJobId'),

            // -----------------------------------------------------------------
            // fetchCloneEngines
            // -----------------------------------------------------------------
            fetchCloneEngines: async (signal) => {
                set({ loadingCloneEngines: true, cloneEnginesError: null }, false, 'fetchCloneEngines/pending');
                try {
                    const res = await api.get<EpisodeCloneEngine[]>('/episode-clone/engines', { signal });
                    if (signal?.aborted) return;
                    const engines = Array.isArray(res.data) ? res.data : [];
                    set(
                        (state) => ({
                            cloneEngines: engines,
                            cloneEngineKey: engines.some((e) => e.key === state.cloneEngineKey)
                                ? state.cloneEngineKey
                                : engines[0]?.key || 'default',
                        }),
                        false,
                        'fetchCloneEngines/fulfilled'
                    );
                } catch (e: any) {
                    if (signal?.aborted) return;
                    console.error('Failed to fetch clone engines:', e);
                    set(
                        { cloneEngines: [], cloneEnginesError: e?.response?.data?.detail || 'Failed to load clone engines' },
                        false,
                        'fetchCloneEngines/rejected'
                    );
                } finally {
                    if (!signal?.aborted) set({ loadingCloneEngines: false }, false, 'fetchCloneEngines/settled');
                }
            },

            // -----------------------------------------------------------------
            // fetchCloneOllamaModels
            // -----------------------------------------------------------------
            fetchCloneOllamaModels: async (signal) => {
                set({ loadingCloneOllamaModels: true, cloneOllamaModelsError: null }, false, 'fetchCloneOllamaModels/pending');
                try {
                    const res = await api.get<OllamaLocalModelsResponse>('/settings/ollama/models', { signal });
                    if (signal?.aborted) return;
                    if (res.data?.status !== 'ok') {
                        set(
                            {
                                cloneOllamaModels: [],
                                cloneOllamaModelsError: res.data?.error || 'Failed to load local Ollama models.',
                                loadingCloneOllamaModels: false,
                            },
                            false,
                            'fetchCloneOllamaModels/rejected'
                        );
                        return;
                    }
                    const models = Array.isArray(res.data?.models)
                        ? res.data.models.filter((m) => !!m?.name)
                        : [];
                    const { cloneEngines, cloneEngineKey, cloneOllamaModel } = get();
                    const selectedEngine = cloneEngines.find((e) => e.key === cloneEngineKey) || cloneEngines[0] || null;
                    let nextModel = cloneOllamaModel;
                    if (!nextModel || !models.some((m) => m.name === nextModel)) {
                        if (selectedEngine?.model && models.some((m) => m.name === selectedEngine.model)) {
                            nextModel = selectedEngine.model;
                        } else {
                            nextModel = res.data?.current_model || models[0]?.name || '';
                        }
                    }
                    set({ cloneOllamaModels: models, cloneOllamaModel: nextModel }, false, 'fetchCloneOllamaModels/fulfilled');
                } catch (e: any) {
                    if (signal?.aborted) return;
                    console.error('Failed to fetch Ollama models for clone workbench:', e);
                    set(
                        { cloneOllamaModels: [], cloneOllamaModelsError: e?.response?.data?.detail || 'Failed to load local Ollama models.' },
                        false,
                        'fetchCloneOllamaModels/rejected'
                    );
                } finally {
                    if (!signal?.aborted) set({ loadingCloneOllamaModels: false }, false, 'fetchCloneOllamaModels/settled');
                }
            },

            // -----------------------------------------------------------------
            // fetchCloneJobs
            // -----------------------------------------------------------------
            fetchCloneJobs: async (videoId, signal, options) => {
                const requestId = ++_jobsRequestId;
                if (!options?.silent) set({ loadingCloneJobs: true }, false, 'fetchCloneJobs/pending');
                set({ cloneJobsError: null }, false, 'fetchCloneJobs/clear-error');
                try {
                    const res = await api.get<EpisodeCloneJob[]>(`/videos/${videoId}/episode-clone/jobs`, {
                        params: { limit: 16 },
                        signal,
                    });
                    if (requestId !== _jobsRequestId || signal?.aborted) return [];
                    const jobs = Array.isArray(res.data) ? res.data : [];
                    set(
                        (state) => {
                            let newSelectedId = state.selectedCloneJobId;
                            if (options?.preferredJobId && jobs.some((j) => j.job_id === options.preferredJobId)) {
                                newSelectedId = options.preferredJobId;
                            } else if (!newSelectedId || !jobs.some((j) => j.job_id === newSelectedId)) {
                                newSelectedId = jobs[0]?.job_id ?? null;
                            }
                            return {
                                cloneJobs: jobs,
                                generatingClone: jobs.some((j) =>
                                    ['queued', 'running'].includes(String(j.status || '').toLowerCase())
                                ),
                                selectedCloneJobId: newSelectedId,
                            };
                        },
                        false,
                        'fetchCloneJobs/fulfilled'
                    );
                    return jobs;
                } catch (e: any) {
                    if (signal?.aborted) return null;
                    console.error('Failed to fetch episode clone jobs:', e);
                    set(
                        {
                            cloneJobsError: e?.response?.data?.detail || 'Failed to load clone workbench history',
                            generatingClone: false,
                        },
                        false,
                        'fetchCloneJobs/rejected'
                    );
                    return [];
                } finally {
                    if (requestId === _jobsRequestId && !signal?.aborted && !options?.silent) {
                        set({ loadingCloneJobs: false }, false, 'fetchCloneJobs/settled');
                    }
                }
            },

            // -----------------------------------------------------------------
            // loadCloneVariantInputs
            // -----------------------------------------------------------------
            loadCloneVariantInputs: (job) => {
                const { cloneEngines, cloneEngineKey } = get();
                const requestProvider = String(job.request?.provider_override || '').trim();
                const providerMatch = cloneEngines.find((engine) =>
                    requestProvider
                        ? String(engine.provider || '') === requestProvider
                        : engine.key === 'default'
                );
                set(
                    {
                        cloneStylePrompt: String(job.request?.style_prompt || ''),
                        cloneNotes: String(job.request?.notes || ''),
                        cloneEngineKey: providerMatch?.key || (requestProvider ? cloneEngineKey : 'default'),
                        cloneOllamaModel: String(job.request?.model_override || ''),
                        cloneConceptsText: (job.request?.approved_concepts || []).join('\n'),
                        cloneExcludedReferencesText: (job.request?.excluded_references || []).join('\n'),
                    },
                    false,
                    'loadCloneVariantInputs'
                );
            },

            // -----------------------------------------------------------------
            // handleDetectCloneConcepts
            // -----------------------------------------------------------------
            handleDetectCloneConcepts: async (videoId) => {
                const { cloneEngines, cloneEngineKey, cloneOllamaModel, cloneNotes } = get();
                const engineOverride = resolveEngineOverride(cloneEngines, cloneEngineKey, cloneOllamaModel);
                set({ detectingCloneConcepts: true, cloneEnginesError: null }, false, 'handleDetectCloneConcepts/pending');
                try {
                    const res = await api.post<EpisodeCloneConceptsResponse>(`/videos/${videoId}/episode-clone/concepts`, {
                        notes: cloneNotes.trim() || undefined,
                        related_limit: 8,
                        provider_override: engineOverride.provider_override,
                        model_override: engineOverride.model_override,
                    });
                    set(
                        {
                            cloneConcepts: res.data,
                            cloneConceptsText: (Array.isArray(res.data?.concepts) ? res.data.concepts : []).join('\n'),
                            cloneExcludedReferencesText: (
                                Array.isArray(res.data?.excluded_references) ? res.data.excluded_references : []
                            ).join('\n'),
                        },
                        false,
                        'handleDetectCloneConcepts/fulfilled'
                    );
                } catch (e: any) {
                    console.error('Failed to detect clone concepts:', e);
                    alert(e?.response?.data?.detail || 'Failed to detect clone concepts');
                } finally {
                    set({ detectingCloneConcepts: false }, false, 'handleDetectCloneConcepts/settled');
                }
            },

            // -----------------------------------------------------------------
            // handleGenerateEpisodeClone
            // -----------------------------------------------------------------
            handleGenerateEpisodeClone: async (videoId) => {
                const state = get();
                if (!state.cloneStylePrompt.trim()) {
                    alert('Enter a target style prompt first.');
                    return;
                }
                const approvedConcepts = normalizeTextList(state.cloneConceptsText);
                if (approvedConcepts.length === 0) {
                    alert('Detect and approve at least one concept before generating a clone.');
                    return;
                }
                const engineOverride = resolveEngineOverride(state.cloneEngines, state.cloneEngineKey, state.cloneOllamaModel);
                const excludedReferences = normalizeTextList(state.cloneExcludedReferencesText);
                const requestId = ++_generateRequestId;
                set({ generatingClone: true, cloneJobsError: null, copiedCloneScript: false }, false, 'handleGenerateEpisodeClone/pending');
                try {
                    let maxVariant = 0;
                    for (const job of state.cloneJobs) {
                        const match = String(job.request?.variant_label || '').match(/(\d+)\s*$/);
                        if (match) maxVariant = Math.max(maxVariant, Number(match[1] || 0));
                    }
                    const nextVariantSeed = maxVariant + 1;
                    const jobs: EpisodeCloneJob[] = [];
                    for (let idx = 0; idx < state.cloneBatchSize; idx += 1) {
                        const res = await api.post<EpisodeCloneJob>(`/videos/${videoId}/episode-clone/generate`, {
                            style_prompt: state.cloneStylePrompt.trim(),
                            notes: state.cloneNotes.trim() || undefined,
                            related_limit: 8,
                            variant_label: `Variant ${nextVariantSeed + idx}`,
                            provider_override: engineOverride.provider_override,
                            model_override: engineOverride.model_override,
                            approved_concepts: approvedConcepts,
                            excluded_references: excludedReferences,
                        });
                        jobs.push(res.data);
                    }
                    if (requestId !== _generateRequestId) return;
                    const preferredJobId = jobs[0]?.job_id ?? null;
                    set(
                        (prev) => {
                            const merged = [...jobs, ...prev.cloneJobs].filter(
                                (job, index, array) =>
                                    array.findIndex((c) => c.job_id === job.job_id) === index
                            );
                            return { selectedCloneJobId: preferredJobId, cloneJobs: merged };
                        },
                        false,
                        'handleGenerateEpisodeClone/fulfilled'
                    );
                    await get().fetchCloneJobs(videoId, undefined, { preferredJobId, silent: true });
                } catch (e: any) {
                    alert(e?.response?.data?.detail || 'Failed to generate episode clone');
                    set({ generatingClone: false }, false, 'handleGenerateEpisodeClone/rejected');
                }
            },

            // -----------------------------------------------------------------
            // handleCopyCloneScript
            // -----------------------------------------------------------------
            handleCopyCloneScript: async () => {
                const { cloneJobs, selectedCloneJobId } = get();
                const selectedJob = cloneJobs.find((j) => j.job_id === selectedCloneJobId) || cloneJobs[0] || null;
                const script = (selectedJob?.result as EpisodeCloneGenerateResponse | null | undefined)?.script;
                if (!script) return;
                try {
                    await navigator.clipboard.writeText(script);
                    set({ copiedCloneScript: true }, false, 'handleCopyCloneScript/copied');
                    window.setTimeout(
                        () => set({ copiedCloneScript: false }, false, 'handleCopyCloneScript/reset'),
                        1500
                    );
                } catch {
                    alert('Failed to copy clone script');
                }
            },

            // -----------------------------------------------------------------
            // cloneJobMatchesVisibleInputs
            // -----------------------------------------------------------------
            cloneJobMatchesVisibleInputs: (job) => {
                if (!job) return false;
                const { cloneEngines, cloneEngineKey, cloneOllamaModel, cloneStylePrompt, cloneNotes, cloneConceptsText, cloneExcludedReferencesText } =
                    get();
                const currentEngine = resolveEngineOverride(cloneEngines, cloneEngineKey, cloneOllamaModel);
                return (
                    String(job.request?.style_prompt || '').trim() === cloneStylePrompt.trim() &&
                    String(job.request?.notes || '').trim() === cloneNotes.trim() &&
                    String(job.request?.provider_override || '') === String(currentEngine.provider_override || '') &&
                    String(job.request?.model_override || '') === String(currentEngine.model_override || '') &&
                    listsMatch(job.request?.approved_concepts || [], normalizeTextList(cloneConceptsText)) &&
                    listsMatch(job.request?.excluded_references || [], normalizeTextList(cloneExcludedReferencesText))
                );
            },

            // -----------------------------------------------------------------
            // resetCloneState — call when video ID changes
            // -----------------------------------------------------------------
            resetCloneState: () => {
                _generateRequestId += 1;
                _jobsRequestId += 1;
                set(
                    {
                        cloneJobs: [],
                        cloneJobsError: null,
                        selectedCloneJobId: null,
                        cloneEngines: [],
                        cloneEnginesError: null,
                        cloneEngineKey: 'default',
                        cloneOllamaModels: [],
                        cloneOllamaModelsError: null,
                        cloneOllamaModel: '',
                        cloneConcepts: null,
                        cloneConceptsText: '',
                        cloneExcludedReferencesText: '',
                        detectingCloneConcepts: false,
                        generatingClone: false,
                        copiedCloneScript: false,
                    },
                    false,
                    'resetCloneState'
                );
            },
        }),
        { name: 'CloneStore' }
    )
);

// ---------------------------------------------------------------------------
// Selector hooks — use these in components to avoid prop drilling
// ---------------------------------------------------------------------------

export const useSelectedCloneJob = () =>
    useCloneStore((state) =>
        state.cloneJobs.find((j) => j.job_id === state.selectedCloneJobId) || state.cloneJobs[0] || null
    );

export const useCloneDraftResult = () =>
    useCloneStore((state) => {
        const job = state.cloneJobs.find((j) => j.job_id === state.selectedCloneJobId) || state.cloneJobs[0] || null;
        return (job?.result as EpisodeCloneGenerateResponse | null | undefined) || null;
    });

export const useSelectedCloneEngine = () =>
    useCloneStore((state) =>
        state.cloneEngines.find((e) => e.key === state.cloneEngineKey) || state.cloneEngines[0] || null
    );

export const useCloneUsesOllama = () =>
    useCloneStore((state) => {
        const selected = state.cloneEngines.find((e) => e.key === state.cloneEngineKey) || state.cloneEngines[0] || null;
        return (selected?.provider || '') === 'ollama';
    });
