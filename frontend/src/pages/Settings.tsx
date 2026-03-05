import { useState, useEffect, type FormEvent } from 'react';
import { Save, Key, CheckCircle2, AlertCircle, Zap, ExternalLink, Loader2, RefreshCw, Terminal, Bot, AudioLines, Power, Mic, Smile, Link2, Database } from 'lucide-react';
import api from '../lib/api';

interface ModelStatus {
    accessible: boolean;
    error: string | null;
    url?: string;
}

interface TokenValidation {
    valid: boolean;
    token_set: boolean;
    error?: string;
    models: Record<string, ModelStatus>;
}

type SettingsTab = 'transcription' | 'diarization' | 'llm' | 'youtube' | 'funny' | 'system';
type TranscriptionEngine = 'auto' | 'whisper' | 'parakeet';
type LlmProvider = 'ollama' | 'nvidia_nim' | 'openai' | 'anthropic' | 'gemini' | 'groq' | 'openrouter' | 'xai';
type OllamaModelTier = 'lite' | 'medium' | 'q8' | 'custom';
type OllamaPreference = 'speed' | 'balanced' | 'capability';
type UnslothQuant = 'Q4_K_M' | 'Q5_K_M' | 'Q6_K' | 'Q8_0';
type UnslothFamilyKey = 'qwen35_35b_a3b' | 'qwen35_27b' | 'qwen3_14b' | 'qwen3_8b';

interface LlmConnectionTestResult {
    status: string;
    error?: string;
    available_models?: string[];
    test_response?: string;
    model?: string;
    provider?: string;
    thinking_mode?: boolean;
    latency_ms?: number;
    generation_latency_ms?: number;
    tags_latency_ms?: number;
}

interface TranscriptionEngineTestResult {
    status: 'ok' | 'error' | string;
    requested_engine: 'auto' | 'whisper' | 'parakeet' | string;
    resolved_engine?: 'whisper' | 'parakeet' | string | null;
    device?: string;
    parakeet_dependencies_available?: boolean;
    whisper_model?: string;
    whisper_compute_type?: string | null;
    parakeet_model?: string;
    parakeet_batch_size_requested?: number;
    parakeet_batch_auto?: boolean;
    parakeet_effective_batch_size?: number;
    parakeet_batch_hard_max?: number;
    parakeet_max_gpu_memory_fraction?: number;
    parakeet_unload_after_transcribe?: boolean | string;
    cuda_memory?: {
        free_gb?: number;
        total_gb?: number;
        allocated_gb?: number;
        reserved_gb?: number;
    };
    fallback_used?: boolean;
    error?: string | null;
}

interface YouTubeOAuthStatus {
    configured: boolean;
    connected: boolean;
    channel_id?: string | null;
    channel_title?: string | null;
    redirect_uri?: string;
    scope?: string;
    token_expires_at?: string | null;
    push_enabled?: boolean;
}

interface OllamaHardwareRecommendationResponse {
    status: string;
    hardware?: {
        gpu_name?: string | null;
        gpu_vendor?: string | null;
        gpu_vram_gb?: number | null;
        gpu_count?: number | null;
        detection_method?: string | null;
    };
    recommendation?: {
        objective?: OllamaPreference;
        base_model: string;
        tier: 'lite' | 'medium' | 'q8';
        model_tag: string;
        estimated_size_gb?: number | null;
        size_source?: 'ollama_exact' | 'estimated' | 'unknown' | string;
        quant_level?: string;
        quant_bits_estimate?: number | null;
        fallback_tag: string;
        reason: string;
    };
    error?: string;
}

interface OllamaLocalModel {
    name: string;
    size_bytes?: number;
    modified_at?: string;
    parameter_size?: string;
    quantization_level?: string;
    families?: string[] | null;
}

interface OllamaLocalModelsResponse {
    status: string;
    ollama_url?: string;
    current_model?: string;
    models: OllamaLocalModel[];
    error?: string;
}

interface OllamaPullJob {
    status?: string;
    started_at?: number;
    updated_at?: number;
    completed_at?: number | null;
    error?: string | null;
    ollama_response?: Record<string, unknown> | null;
    available_models?: string[];
    pull_event_status?: string;
    pull_completed?: number | null;
    pull_total?: number | null;
    pull_percent?: number | null;
}

interface OllamaPullStatusResponse {
    status: string;
    model?: string;
    job?: OllamaPullJob | null;
    job_status?: string;
    elapsed_seconds?: number | null;
}

interface OllamaPullProgress {
    running: boolean;
    statusText: string;
    elapsedSeconds: number;
    percent: number | null;
    completed: number | null;
    total: number | null;
}

interface DbHealthResponse {
    timestamp: string;
    database: {
        provider: string;
        database_url: string;
        is_postgres: boolean;
        pool: {
            size: number | null;
            checked_out: number | null;
            checked_in: number | null;
            overflow: number | null;
        };
        connections: {
            total: number | null;
            active: number | null;
            max: number | null;
        };
        query_metrics: {
            started_at: string;
            total_queries: number;
            total_time_ms: number;
            avg_ms: number;
            slow_queries: number;
            slow_threshold_ms: number;
            error_queries: number;
            recent_count: number;
            recent_avg_ms: number;
            recent_p95_ms: number;
            recent_p99_ms: number;
        };
    };
    queue_depth: {
        running: number;
        queued: number;
        paused: number;
        total_active: number;
        by_queue: Record<string, { queued: number; running: number; paused: number }>;
    };
}

const HOSTED_PROVIDER_LABELS: Record<Exclude<LlmProvider, 'ollama'>, string> = {
    nvidia_nim: 'NVIDIA NIM',
    openai: 'ChatGPT (OpenAI)',
    anthropic: 'Claude (Anthropic)',
    gemini: 'Gemini (Google)',
    groq: 'Groq',
    openrouter: 'OpenRouter',
    xai: 'xAI (Grok)',
};

const NVIDIA_NIM_MODEL_PRESETS: Array<{
    id: string;
    label: string;
    note: string;
    modelCardUrl: string;
}> = [
    {
        id: 'moonshotai/kimi-k2.5',
        label: 'Kimi K2.5',
        note: 'Fast and strong for humor explanations.',
        modelCardUrl: 'https://build.nvidia.com/moonshotai/kimi-k2.5/modelcard',
    },
    {
        id: 'qwen/qwen3.5-397b-a17b',
        label: 'Qwen3.5-397B-A17B',
        note: 'Higher-capability MoE option, typically higher latency/cost.',
        modelCardUrl: 'https://build.nvidia.com/qwen/qwen3.5-397b-a17b/modelcard',
    },
];

function inferOllamaTier(modelTag: string): OllamaModelTier {
    const tag = (modelTag || '').trim().toLowerCase();
    if (!tag) return 'medium';
    if (/-q8_0$/.test(tag) || /:q8_0$/.test(tag)) return 'q8';
    if (/-q4_k_m$/.test(tag) || /:q4_k_m$/.test(tag)) return 'lite';
    if (/-q\d/.test(tag)) return 'custom';
    if (/:q\d/.test(tag)) return 'custom';
    return 'medium';
}

function formatBytesAsGb(bytes?: number): string {
    if (!bytes || bytes <= 0) return 'unknown';
    return `${(bytes / (1024 ** 3)).toFixed(1)} GB`;
}

function ollamaModelNameMatches(localModelName: string, requestedModelName: string): boolean {
    const local = (localModelName || '').trim().toLowerCase();
    const req = (requestedModelName || '').trim().toLowerCase();
    if (!local || !req) return false;
    if (local === req || local.startsWith(`${req}:`)) return true;

    const toSig = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '');
    const localBase = local.replace(/:latest$/, '');
    const reqBase = req.replace(/:latest$/, '');
    if (localBase === reqBase) return true;

    const localSig = toSig(localBase);
    const reqSig = toSig(reqBase);
    if (localSig && reqSig && (localSig === reqSig || localSig.startsWith(reqSig) || reqSig.startsWith(localSig))) {
        return true;
    }

    if (reqBase.startsWith('hf.co/')) {
        const noHf = reqBase.slice('hf.co/'.length);
        const sepIdx = noHf.lastIndexOf(':');
        const repoPath = sepIdx >= 0 ? noHf.slice(0, sepIdx) : noHf;
        const reqQuant = sepIdx >= 0 ? noHf.slice(sepIdx + 1) : '';
        const repoTail = (repoPath.split('/').pop() || '').replace(/-gguf$/, '');
        const tailSig = toSig(repoTail);
        const quantSig = toSig(reqQuant);
        if (tailSig && localSig.includes(tailSig)) {
            if (!quantSig || localSig.includes(quantSig)) return true;
        }
    }

    return false;
}

const UNSLOTH_QUANTS: { value: UnslothQuant; label: string; note: string }[] = [
    { value: 'Q4_K_M', label: 'Q4_K_M (Lite)', note: 'Lowest VRAM, fastest.' },
    { value: 'Q5_K_M', label: 'Q5_K_M (Balanced)', note: 'Good quality/speed default.' },
    { value: 'Q6_K', label: 'Q6_K (Quality)', note: 'Higher quality, more VRAM.' },
    { value: 'Q8_0', label: 'Q8_0 (Max Quality)', note: 'Highest quality, highest VRAM.' },
];

const UNSLOTH_FAMILIES: { key: UnslothFamilyKey; label: string; repo: string; note: string; paramsB: number }[] = [
    {
        key: 'qwen35_35b_a3b',
        label: 'Qwen3.5-35B-A3B',
        repo: 'hf.co/unsloth/Qwen3.5-35B-A3B-GGUF',
        note: 'Highest capability, highest VRAM.',
        paramsB: 35,
    },
    {
        key: 'qwen35_27b',
        label: 'Qwen3.5-27B',
        repo: 'hf.co/unsloth/Qwen3.5-27B-GGUF',
        note: 'Strong quality with lower VRAM than 35B.',
        paramsB: 27,
    },
    {
        key: 'qwen3_14b',
        label: 'Qwen3-14B',
        repo: 'hf.co/unsloth/Qwen3-14B-GGUF',
        note: 'Mid-tier balance for local systems.',
        paramsB: 14,
    },
    {
        key: 'qwen3_8b',
        label: 'Qwen3-8B',
        repo: 'hf.co/unsloth/Qwen3-8B-GGUF',
        note: 'Fastest, lowest VRAM among these presets.',
        paramsB: 8,
    },
];

function resolveUnslothQuantTag(familyKey: UnslothFamilyKey, quant: UnslothQuant): string {
    // Unsloth Qwen3.5-35B-A3B repo uses UD-* quant tag names.
    if (familyKey === 'qwen35_35b_a3b') {
        if (quant === 'Q4_K_M') return 'UD-Q4_K_M';
        if (quant === 'Q5_K_M') return 'UD-Q5_K_XL';
        if (quant === 'Q6_K') return 'UD-Q6_K_S';
        return 'UD-Q8_K_XL';
    }
    return quant;
}

function buildUnslothTag(familyKey: UnslothFamilyKey, quant: UnslothQuant): string {
    const family = UNSLOTH_FAMILIES.find(f => f.key === familyKey) || UNSLOTH_FAMILIES[0];
    return `${family.repo}:${resolveUnslothQuantTag(family.key, quant)}`;
}

function estimateUnslothSizeGb(familyKey: UnslothFamilyKey, quant: UnslothQuant): number {
    const family = UNSLOTH_FAMILIES.find(f => f.key === familyKey) || UNSLOTH_FAMILIES[0];
    const bits = quant === 'Q4_K_M' ? 4.5 : quant === 'Q5_K_M' ? 5.5 : quant === 'Q6_K' ? 6.5 : 8.5;
    return Math.round((family.paramsB * (bits / 8) * 1.15) * 10) / 10;
}

export function Settings() {
    const [activeTab, setActiveTab] = useState<SettingsTab>('transcription');
    const [token, setToken] = useState('');
    const [transcriptionEngine, setTranscriptionEngine] = useState<TranscriptionEngine>('auto');
    const [transcriptionModel, setTranscriptionModel] = useState('medium');
    const [computeType, setComputeType] = useState('int8_float16');
    const [parakeetModel, setParakeetModel] = useState('nvidia/parakeet-tdt-0.6b-v2');
    const [parakeetBatchSize, setParakeetBatchSize] = useState(16);
    const [parakeetBatchAuto, setParakeetBatchAuto] = useState(true);
    const [parakeetRequireWordTimestamps, setParakeetRequireWordTimestamps] = useState(true);
    const [parakeetUnloadAfterTranscribe, setParakeetUnloadAfterTranscribe] = useState(false);
    const [testingTranscriptionEngine, setTestingTranscriptionEngine] = useState(false);
    const [transcriptionEngineTestResult, setTranscriptionEngineTestResult] = useState<TranscriptionEngineTestResult | null>(null);
    const [beamSize, setBeamSize] = useState(1);
    const [vadFilter, setVadFilter] = useState(true);
    const [batchedTranscription, setBatchedTranscription] = useState(true);
    const [verboseLogging, setVerboseLogging] = useState(false);
    const [llmProvider, setLlmProvider] = useState<LlmProvider>('ollama');
    const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
    const [ollamaModel, setOllamaModel] = useState('mistral');
    const [ollamaModelTier, setOllamaModelTier] = useState<OllamaModelTier>('medium');
    const [ollamaPreference, setOllamaPreference] = useState<OllamaPreference>('balanced');
    const [ollamaEnabled, setOllamaEnabled] = useState(false);
    const [ollamaHardwareRecommendation, setOllamaHardwareRecommendation] = useState<OllamaHardwareRecommendationResponse | null>(null);
    const [detectingOllamaHardware, setDetectingOllamaHardware] = useState(false);
    const [ollamaLocalModels, setOllamaLocalModels] = useState<OllamaLocalModel[]>([]);
    const [unslothFamily, setUnslothFamily] = useState<UnslothFamilyKey>('qwen35_35b_a3b');
    const [unslothQuant, setUnslothQuant] = useState<UnslothQuant>('Q5_K_M');
    const [loadingOllamaLocalModels, setLoadingOllamaLocalModels] = useState(false);
    const [ollamaLocalModelsError, setOllamaLocalModelsError] = useState<string | null>(null);
    const [nvidiaNimBaseUrl, setNvidiaNimBaseUrl] = useState('https://integrate.api.nvidia.com');
    const [nvidiaNimModel, setNvidiaNimModel] = useState('moonshotai/kimi-k2.5');
    const [nvidiaNimApiKey, setNvidiaNimApiKey] = useState('');
    const [nvidiaNimThinkingMode, setNvidiaNimThinkingMode] = useState(false);
    const [nvidiaNimMinRequestIntervalSeconds, setNvidiaNimMinRequestIntervalSeconds] = useState(2.5);
    const [openaiBaseUrl, setOpenaiBaseUrl] = useState('https://api.openai.com');
    const [openaiModel, setOpenaiModel] = useState('gpt-4o-mini');
    const [openaiApiKey, setOpenaiApiKey] = useState('');
    const [anthropicBaseUrl, setAnthropicBaseUrl] = useState('https://api.anthropic.com');
    const [anthropicModel, setAnthropicModel] = useState('claude-3-5-sonnet-latest');
    const [anthropicApiKey, setAnthropicApiKey] = useState('');
    const [geminiBaseUrl, setGeminiBaseUrl] = useState('https://generativelanguage.googleapis.com');
    const [geminiModel, setGeminiModel] = useState('gemini-2.5-flash');
    const [geminiApiKey, setGeminiApiKey] = useState('');
    const [groqBaseUrl, setGroqBaseUrl] = useState('https://api.groq.com/openai');
    const [groqModel, setGroqModel] = useState('llama-3.3-70b-versatile');
    const [groqApiKey, setGroqApiKey] = useState('');
    const [openrouterBaseUrl, setOpenrouterBaseUrl] = useState('https://openrouter.ai/api');
    const [openrouterModel, setOpenrouterModel] = useState('openai/gpt-4o-mini');
    const [openrouterApiKey, setOpenrouterApiKey] = useState('');
    const [xaiBaseUrl, setXaiBaseUrl] = useState('https://api.x.ai');
    const [xaiModel, setXaiModel] = useState('grok-2');
    const [xaiApiKey, setXaiApiKey] = useState('');
    const [youtubeOauthClientId, setYoutubeOauthClientId] = useState('');
    const [youtubeOauthClientSecret, setYoutubeOauthClientSecret] = useState('');
    const [youtubeOauthRedirectUri, setYoutubeOauthRedirectUri] = useState('http://localhost:8000/auth/youtube/callback');
    const [youtubePublishPushEnabled, setYoutubePublishPushEnabled] = useState(false);
    const [ytdlpCookiesFile, setYtdlpCookiesFile] = useState('');
    const [ytdlpCookiesFromBrowser, setYtdlpCookiesFromBrowser] = useState('');
    const [youtubeOauthStatus, setYoutubeOauthStatus] = useState<YouTubeOAuthStatus | null>(null);
    const [loadingYoutubeOauthStatus, setLoadingYoutubeOauthStatus] = useState(false);
    const [testingYouTubeOauth, setTestingYouTubeOauth] = useState(false);
    const [youtubeOauthTestResult, setYoutubeOauthTestResult] = useState<{ status: string; channel_id?: string; channel_title?: string; error?: string } | null>(null);
    const [connectingYouTube, setConnectingYouTube] = useState(false);
    const [disconnectingYouTube, setDisconnectingYouTube] = useState(false);
    const [diarizationSensitivity, setDiarizationSensitivity] = useState('balanced');
    const [speakerMatchThreshold, setSpeakerMatchThreshold] = useState(0.5);
    const [funnyMomentsMaxSaved, setFunnyMomentsMaxSaved] = useState(25);
    const [funnyMomentsExplainBatchLimit, setFunnyMomentsExplainBatchLimit] = useState(12);
    const [ollamaTest, setOllamaTest] = useState<LlmConnectionTestResult | null>(null);
    const [testingOllama, setTestingOllama] = useState(false);
    const [nvidiaNimTest, setNvidiaNimTest] = useState<LlmConnectionTestResult | null>(null);
    const [testingNvidiaNim, setTestingNvidiaNim] = useState(false);
    const [pullingOllamaModel, setPullingOllamaModel] = useState(false);
    const [ollamaPullStatus, setOllamaPullStatus] = useState<{ status: string; message: string } | null>(null);
    const [ollamaPullProgress, setOllamaPullProgress] = useState<OllamaPullProgress | null>(null);
    const [optimisticDownloadedModel, setOptimisticDownloadedModel] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [tokenValidation, setTokenValidation] = useState<TokenValidation | null>(null);
    const [validating, setValidating] = useState(false);
    const [restarting, setRestarting] = useState(false);
    const [dbHealth, setDbHealth] = useState<DbHealthResponse | null>(null);
    const [loadingDbHealth, setLoadingDbHealth] = useState(false);
    const [dbHealthError, setDbHealthError] = useState<string | null>(null);

    useEffect(() => {
        loadSettings();
    }, []);

    useEffect(() => {
        const inferredTier = inferOllamaTier(ollamaModel);
        if (inferredTier !== ollamaModelTier) {
            setOllamaModelTier(inferredTier);
        }
    }, [ollamaModel]);

    useEffect(() => {
        if (activeTab !== 'llm' || llmProvider !== 'ollama') return;
        if (
            ollamaHardwareRecommendation?.status === 'ok' &&
            ollamaHardwareRecommendation?.recommendation?.objective === ollamaPreference
        ) {
            return;
        }
        if (detectingOllamaHardware) return;
        void detectOllamaHardwareRecommendation();
    }, [activeTab, llmProvider, ollamaPreference]);

    useEffect(() => {
        if (activeTab !== 'llm' || llmProvider !== 'ollama') return;
        if (loadingOllamaLocalModels) return;
        void loadOllamaLocalModels();
    }, [activeTab, llmProvider, ollamaUrl]);

    useEffect(() => {
        setTranscriptionEngineTestResult(null);
    }, [transcriptionEngine, transcriptionModel, computeType, parakeetModel, parakeetBatchSize, parakeetBatchAuto, parakeetRequireWordTimestamps, parakeetUnloadAfterTranscribe]);

    const loadSettings = async () => {
        try {
            const res = await api.get('/settings');
            setToken(res.data.hf_token);
            {
                const rawEngine = String(res.data.transcription_engine || 'auto').toLowerCase();
                const engine: TranscriptionEngine = rawEngine === 'whisper' || rawEngine === 'parakeet' ? rawEngine : 'auto';
                setTranscriptionEngine(engine);
            }
            setTranscriptionModel(res.data.transcription_model || 'tiny');
            setComputeType(res.data.transcription_compute_type || 'float16');
            setParakeetModel(res.data.parakeet_model || 'nvidia/parakeet-tdt-0.6b-v2');
            setParakeetBatchSize(res.data.parakeet_batch_size ?? 16);
            setParakeetBatchAuto(res.data.parakeet_batch_auto ?? true);
            setParakeetRequireWordTimestamps(res.data.parakeet_require_word_timestamps ?? true);
            setParakeetUnloadAfterTranscribe(res.data.parakeet_unload_after_transcribe ?? false);
            setBeamSize(res.data.beam_size ?? 1);
            setVadFilter(res.data.vad_filter ?? true);
            setBatchedTranscription(res.data.batched_transcription ?? true);
            setVerboseLogging(res.data.verbose_logging ?? false);
            {
                const rawProvider = String(res.data.llm_provider || 'ollama') as LlmProvider;
                const allowed: LlmProvider[] = ['ollama', 'nvidia_nim', 'openai', 'anthropic', 'gemini', 'groq', 'openrouter', 'xai'];
                setLlmProvider(allowed.includes(rawProvider) ? rawProvider : 'ollama');
            }
            setOllamaUrl(res.data.ollama_url || 'http://localhost:11434');
            const loadedModel = res.data.ollama_model || 'mistral';
            const loadedTier = (res.data.ollama_model_tier || inferOllamaTier(loadedModel)) as OllamaModelTier;
            setOllamaModel(loadedModel);
            setOllamaModelTier(loadedTier);
            setOllamaEnabled(res.data.llm_enabled ?? res.data.ollama_enabled ?? false);
            setNvidiaNimBaseUrl(res.data.nvidia_nim_base_url || 'https://integrate.api.nvidia.com');
            setNvidiaNimModel(res.data.nvidia_nim_model || 'moonshotai/kimi-k2.5');
            setNvidiaNimApiKey(res.data.nvidia_nim_api_key || '');
            setNvidiaNimThinkingMode(res.data.nvidia_nim_thinking_mode ?? false);
            setNvidiaNimMinRequestIntervalSeconds(res.data.nvidia_nim_min_request_interval_seconds ?? 2.5);
            setOpenaiBaseUrl(res.data.openai_base_url || 'https://api.openai.com');
            setOpenaiModel(res.data.openai_model || 'gpt-4o-mini');
            setOpenaiApiKey(res.data.openai_api_key || '');
            setAnthropicBaseUrl(res.data.anthropic_base_url || 'https://api.anthropic.com');
            setAnthropicModel(res.data.anthropic_model || 'claude-3-5-sonnet-latest');
            setAnthropicApiKey(res.data.anthropic_api_key || '');
            setGeminiBaseUrl(res.data.gemini_base_url || 'https://generativelanguage.googleapis.com');
            setGeminiModel(res.data.gemini_model || 'gemini-2.5-flash');
            setGeminiApiKey(res.data.gemini_api_key || '');
            setGroqBaseUrl(res.data.groq_base_url || 'https://api.groq.com/openai');
            setGroqModel(res.data.groq_model || 'llama-3.3-70b-versatile');
            setGroqApiKey(res.data.groq_api_key || '');
            setOpenrouterBaseUrl(res.data.openrouter_base_url || 'https://openrouter.ai/api');
            setOpenrouterModel(res.data.openrouter_model || 'openai/gpt-4o-mini');
            setOpenrouterApiKey(res.data.openrouter_api_key || '');
            setXaiBaseUrl(res.data.xai_base_url || 'https://api.x.ai');
            setXaiModel(res.data.xai_model || 'grok-2');
            setXaiApiKey(res.data.xai_api_key || '');
            setYoutubeOauthClientId(res.data.youtube_oauth_client_id || '');
            setYoutubeOauthClientSecret(res.data.youtube_oauth_client_secret || '');
            setYoutubeOauthRedirectUri(res.data.youtube_oauth_redirect_uri || 'http://localhost:8000/auth/youtube/callback');
            setYoutubePublishPushEnabled(res.data.youtube_publish_push_enabled ?? false);
            setYtdlpCookiesFile(res.data.ytdlp_cookies_file || '');
            setYtdlpCookiesFromBrowser(res.data.ytdlp_cookies_from_browser || '');
            setDiarizationSensitivity(res.data.diarization_sensitivity || 'balanced');
            setSpeakerMatchThreshold(res.data.speaker_match_threshold ?? 0.5);
            setFunnyMomentsMaxSaved(res.data.funny_moments_max_saved ?? 25);
            setFunnyMomentsExplainBatchLimit(res.data.funny_moments_explain_batch_limit ?? 12);
            if (res.data.hf_token) {
                validateToken();
            }
        } catch (e) {
            console.error('Failed to load settings:', e);
        } finally {
            setLoading(false);
        }
        void loadYouTubeOAuthStatus();
    };

    const loadDbHealth = async (silent = false) => {
        if (!silent) setLoadingDbHealth(true);
        try {
            const res = await api.get<DbHealthResponse>('/settings/db-health');
            setDbHealth(res.data);
            setDbHealthError(null);
        } catch (e: any) {
            const message =
                e?.response?.data?.detail ||
                e?.message ||
                'Failed to load database health';
            setDbHealthError(String(message));
        } finally {
            if (!silent) setLoadingDbHealth(false);
        }
    };

    useEffect(() => {
        if (activeTab !== 'system') return;
        void loadDbHealth();
        const interval = setInterval(() => {
            void loadDbHealth(true);
        }, 5000);
        return () => clearInterval(interval);
    }, [activeTab]);

    const loadYouTubeOAuthStatus = async () => {
        setLoadingYoutubeOauthStatus(true);
        try {
            const res = await api.get('/youtube/oauth/status');
            setYoutubeOauthStatus(res.data);
        } catch (e) {
            console.error('Failed to load YouTube OAuth status:', e);
            setYoutubeOauthStatus(null);
        } finally {
            setLoadingYoutubeOauthStatus(false);
        }
    };

    const validateToken = async () => {
        setValidating(true);
        try {
            const res = await api.post('/settings/validate-token');
            setTokenValidation(res.data);
        } catch (e) {
            console.error('Failed to validate token:', e);
            setTokenValidation({ valid: false, token_set: false, error: 'Validation failed', models: {} });
        } finally {
            setValidating(false);
        }
    };

    const testOllama = async () => {
        setTestingOllama(true);
        setOllamaTest(null);
        try {
            const res = await api.post('/settings/test-ollama');
            setOllamaTest(res.data);
            await loadOllamaLocalModels();
        } catch (e) {
            setOllamaTest({ status: 'error', error: 'Failed to reach backend' });
        } finally {
            setTestingOllama(false);
        }
    };

    const loadOllamaLocalModels = async () => {
        setLoadingOllamaLocalModels(true);
        setOllamaLocalModelsError(null);
        try {
            const encodedUrl = encodeURIComponent(ollamaUrl.trim() || 'http://localhost:11434');
            const res = await api.get<OllamaLocalModelsResponse>(`/settings/ollama/models?url=${encodedUrl}`);
            if (res.data?.status !== 'ok') {
                setOllamaLocalModels([]);
                setOllamaLocalModelsError(res.data?.error || 'Failed to load local Ollama models.');
                return;
            }
            const models = Array.isArray(res.data?.models) ? res.data.models : [];
            const cleaned = models.filter(m => !!m?.name);
            setOllamaLocalModels(cleaned);
            if (optimisticDownloadedModel) {
                const nowPresent = cleaned.some(m => ollamaModelNameMatches(m.name || '', optimisticDownloadedModel));
                if (nowPresent) {
                    setOptimisticDownloadedModel(null);
                }
            }
        } catch {
            setOllamaLocalModels([]);
            setOllamaLocalModelsError('Failed to load local Ollama models.');
        } finally {
            setLoadingOllamaLocalModels(false);
        }
    };

    const detectOllamaHardwareRecommendation = async () => {
        setDetectingOllamaHardware(true);
        try {
            const res = await api.get<OllamaHardwareRecommendationResponse>(
                `/settings/ollama/hardware-recommendation?objective=${ollamaPreference}`
            );
            setOllamaHardwareRecommendation(res.data);
        } catch {
            setOllamaHardwareRecommendation({
                status: 'error',
                error: 'Failed to detect GPU hardware. You can still set a model manually.'
            });
        } finally {
            setDetectingOllamaHardware(false);
        }
    };

    const testHostedLlm = async () => {
        setTestingNvidiaNim(true);
        setNvidiaNimTest(null);
        try {
            const res = await api.post('/settings/test-hosted-llm');
            setNvidiaNimTest(res.data);
        } catch {
            setNvidiaNimTest({ status: 'error', error: 'Failed to reach backend' });
        } finally {
            setTestingNvidiaNim(false);
        }
    };

    const getHostedBaseUrl = (): string => {
        switch (llmProvider) {
            case 'nvidia_nim': return nvidiaNimBaseUrl;
            case 'openai': return openaiBaseUrl;
            case 'anthropic': return anthropicBaseUrl;
            case 'gemini': return geminiBaseUrl;
            case 'groq': return groqBaseUrl;
            case 'openrouter': return openrouterBaseUrl;
            case 'xai': return xaiBaseUrl;
            default: return '';
        }
    };

    const setHostedBaseUrl = (value: string) => {
        switch (llmProvider) {
            case 'nvidia_nim': setNvidiaNimBaseUrl(value); break;
            case 'openai': setOpenaiBaseUrl(value); break;
            case 'anthropic': setAnthropicBaseUrl(value); break;
            case 'gemini': setGeminiBaseUrl(value); break;
            case 'groq': setGroqBaseUrl(value); break;
            case 'openrouter': setOpenrouterBaseUrl(value); break;
            case 'xai': setXaiBaseUrl(value); break;
            default: break;
        }
    };

    const getHostedModel = (): string => {
        switch (llmProvider) {
            case 'nvidia_nim': return nvidiaNimModel;
            case 'openai': return openaiModel;
            case 'anthropic': return anthropicModel;
            case 'gemini': return geminiModel;
            case 'groq': return groqModel;
            case 'openrouter': return openrouterModel;
            case 'xai': return xaiModel;
            default: return '';
        }
    };

    const setHostedModel = (value: string) => {
        switch (llmProvider) {
            case 'nvidia_nim': setNvidiaNimModel(value); break;
            case 'openai': setOpenaiModel(value); break;
            case 'anthropic': setAnthropicModel(value); break;
            case 'gemini': setGeminiModel(value); break;
            case 'groq': setGroqModel(value); break;
            case 'openrouter': setOpenrouterModel(value); break;
            case 'xai': setXaiModel(value); break;
            default: break;
        }
    };

    const getHostedApiKey = (): string => {
        switch (llmProvider) {
            case 'nvidia_nim': return nvidiaNimApiKey;
            case 'openai': return openaiApiKey;
            case 'anthropic': return anthropicApiKey;
            case 'gemini': return geminiApiKey;
            case 'groq': return groqApiKey;
            case 'openrouter': return openrouterApiKey;
            case 'xai': return xaiApiKey;
            default: return '';
        }
    };

    const setHostedApiKey = (value: string) => {
        switch (llmProvider) {
            case 'nvidia_nim': setNvidiaNimApiKey(value); break;
            case 'openai': setOpenaiApiKey(value); break;
            case 'anthropic': setAnthropicApiKey(value); break;
            case 'gemini': setGeminiApiKey(value); break;
            case 'groq': setGroqApiKey(value); break;
            case 'openrouter': setOpenrouterApiKey(value); break;
            case 'xai': setXaiApiKey(value); break;
            default: break;
        }
    };

    const testYouTubeOAuth = async () => {
        setTestingYouTubeOauth(true);
        setYoutubeOauthTestResult(null);
        try {
            const res = await api.post('/youtube/oauth/test');
            setYoutubeOauthTestResult(res.data);
            await loadYouTubeOAuthStatus();
        } catch {
            setYoutubeOauthTestResult({ status: 'error', error: 'Failed to reach backend' });
        } finally {
            setTestingYouTubeOauth(false);
        }
    };

    const connectYouTubeOAuth = async () => {
        setConnectingYouTube(true);
        setYoutubeOauthTestResult(null);
        try {
            const res = await api.post('/youtube/oauth/start');
            const authUrl = res.data?.auth_url;
            if (!authUrl) throw new Error('Missing auth URL');
            const popup = window.open(authUrl, 'youtube-oauth', 'popup=yes,width=620,height=760');
            if (!popup) {
                throw new Error('Popup blocked. Allow popups for localhost and try again.');
            }
            for (let i = 0; i < 90; i++) {
                await new Promise(r => setTimeout(r, 2000));
                await loadYouTubeOAuthStatus();
                if (youtubeOauthStatus?.connected) break;
                try {
                    const statusRes = await api.get('/youtube/oauth/status');
                    setYoutubeOauthStatus(statusRes.data);
                    if (statusRes.data?.connected) {
                        try { popup.close(); } catch {}
                        break;
                    }
                } catch {
                    // ignore polling errors
                }
            }
        } catch (e: any) {
            alert(e?.response?.data?.detail || e?.message || 'Failed to start YouTube OAuth flow');
        } finally {
            setConnectingYouTube(false);
        }
    };

    const disconnectYouTubeOAuth = async () => {
        if (!confirm('Disconnect the saved YouTube OAuth tokens for this app?')) return;
        setDisconnectingYouTube(true);
        try {
            await api.post('/youtube/oauth/disconnect');
            setYoutubeOauthTestResult(null);
            await loadYouTubeOAuthStatus();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to disconnect YouTube OAuth');
        } finally {
            setDisconnectingYouTube(false);
        }
    };

    const applyDetectedOllamaRecommendation = (pullAfter = false) => {
        const rec = ollamaHardwareRecommendation?.recommendation;
        if (!rec?.model_tag) return;
        setOllamaModel(rec.model_tag);
        setOllamaModelTier(rec.tier);
        setOllamaPullStatus({
            status: 'info',
            message: `Applied recommended model: ${rec.model_tag}`
        });
        if (pullAfter) {
            void pullOllamaModel(rec.model_tag);
        }
    };

    const applyUnslothPreset = (pullAfter = false) => {
        const tag = buildUnslothTag(unslothFamily, unslothQuant);
        setOllamaModel(tag);
        setOllamaModelTier(inferOllamaTier(tag));
        setOllamaPullStatus({
            status: 'info',
            message: `Applied Unsloth preset: ${tag}`
        });
        if (pullAfter) {
            void pullOllamaModel(tag);
        }
    };

    const waitForOllamaPullCompletion = async (modelToPull: string): Promise<{ ok: boolean; message: string; verified: boolean }> => {
        const encodedUrl = encodeURIComponent(ollamaUrl.trim() || 'http://localhost:11434');
        const encodedModel = encodeURIComponent(modelToPull);
        const startedMs = Date.now();
        const timeoutMs = 2 * 60 * 60 * 1000; // 2h safety timeout for very large pulls.
        let notFoundCount = 0;
        let pollCount = 0;

        while ((Date.now() - startedMs) < timeoutMs) {
            try {
                const res = await api.get<OllamaPullStatusResponse>(`/settings/ollama/pull-status?url=${encodedUrl}&model=${encodedModel}`);
                const jobStatus = String(res.data?.job_status || res.data?.job?.status || '').toLowerCase();
                const apiStatus = String(res.data?.status || '').toLowerCase();
                const elapsedSec = typeof res.data?.elapsed_seconds === 'number'
                    ? Math.max(0, Math.floor(res.data.elapsed_seconds))
                    : Math.floor((Date.now() - startedMs) / 1000);
                const pullPercentRaw = res.data?.job?.pull_percent;
                const pullPercent = typeof pullPercentRaw === 'number'
                    ? Math.max(0, Math.min(100, pullPercentRaw))
                    : null;
                const displayPercent = (jobStatus === 'running' && pullPercent !== null && pullPercent >= 100)
                    ? 99.0
                    : pullPercent;
                const pullStatusText = String(res.data?.job?.pull_event_status || '').trim() || 'downloading';
                const pullCompletedRaw = res.data?.job?.pull_completed;
                const pullTotalRaw = res.data?.job?.pull_total;
                const pullCompleted = typeof pullCompletedRaw === 'number' ? pullCompletedRaw : null;
                const pullTotal = typeof pullTotalRaw === 'number' ? pullTotalRaw : null;

                if (jobStatus === 'completed' || jobStatus === 'completed_unverified') {
                    setOllamaPullProgress({
                        running: false,
                        statusText: 'completed',
                        elapsedSeconds: elapsedSec,
                        percent: 100,
                        completed: pullCompleted,
                        total: pullTotal,
                    });
                    if (jobStatus === 'completed') {
                        return { ok: true, message: `Downloaded "${modelToPull}" successfully.`, verified: true };
                    }
                    return {
                        ok: true,
                        message: `Download finished for "${modelToPull}", but local model list has not confirmed the name yet. Refresh list in a few seconds.`,
                        verified: false,
                    };
                }
                if (jobStatus === 'failed') {
                    const err = String(res.data?.job?.error || 'Model download failed.');
                    setOllamaPullProgress({
                        running: false,
                        statusText: 'failed',
                        elapsedSeconds: elapsedSec,
                        percent: pullPercent,
                        completed: pullCompleted,
                        total: pullTotal,
                    });
                    return { ok: false, message: err, verified: false };
                }

                if (apiStatus === 'not_found' || jobStatus === 'not_found') {
                    notFoundCount += 1;
                    if (notFoundCount > 20) {
                        return {
                            ok: false,
                            message: `Download status for "${modelToPull}" was not found. Try "Download Current Model" again.`,
                            verified: false,
                        };
                    }
                } else {
                    notFoundCount = 0;
                    setOllamaPullProgress({
                        running: true,
                        statusText: pullStatusText,
                        elapsedSeconds: elapsedSec,
                        percent: displayPercent,
                        completed: pullCompleted,
                        total: pullTotal,
                    });
                    setOllamaPullStatus({
                        status: 'info',
                        message: displayPercent !== null
                            ? `Downloading "${modelToPull}"... ${displayPercent.toFixed(1)}%`
                            : `Downloading "${modelToPull}"... ${elapsedSec}s elapsed.`,
                    });
                }

                pollCount += 1;
                if (pollCount % 4 === 0) {
                    await loadOllamaLocalModels();
                }
                await new Promise(resolve => setTimeout(resolve, 1500));
            } catch {
                // Transient polling error: continue until timeout.
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }

        return {
            ok: false,
            message: `Timed out waiting for "${modelToPull}" to finish downloading.`,
            verified: false,
        };
    };

    const pullOllamaModel = async (modelOverride?: string) => {
        const modelToPull = (modelOverride ?? ollamaModel).trim();
        if (!modelToPull) return;
        setPullingOllamaModel(true);
        setOllamaPullStatus(null);
        setOllamaPullProgress(null);
        try {
            if (modelOverride) {
                setOllamaModel(modelToPull);
            }
            let pullSucceeded = false;
            const res = await api.post('/settings/ollama/pull-model', {
                url: ollamaUrl,
                model: modelToPull
            });
            const status = res.data?.status || 'ok';

            if (status === 'already_installed') {
                setOllamaPullStatus({ status: 'success', message: `Model "${modelToPull}" is already installed.` });
                setOptimisticDownloadedModel(modelToPull);
                pullSucceeded = true;
            } else if (status === 'pulling_started' || status === 'already_running') {
                setOllamaPullStatus({
                    status: 'info',
                    message: `Started downloading "${modelToPull}" in background...`,
                });
                setOllamaPullProgress({
                    running: true,
                    statusText: 'queued',
                    elapsedSeconds: 0,
                    percent: null,
                    completed: null,
                    total: null,
                });
                const result = await waitForOllamaPullCompletion(modelToPull);
                setOllamaPullStatus({
                    status: result.ok ? 'success' : 'error',
                    message: result.message,
                });
                if (result.ok && result.verified) {
                    setOptimisticDownloadedModel(modelToPull);
                }
                pullSucceeded = result.ok;
            } else if (status === 'pulled' || status === 'pull_completed_unverified') {
                setOllamaPullStatus({
                    status: status === 'pulled' ? 'success' : 'info',
                    message: status === 'pulled'
                        ? `Downloaded "${modelToPull}" successfully.`
                        : `Download finished for "${modelToPull}", but local model list has not confirmed the name yet.`,
                });
                setOllamaPullProgress({
                    running: false,
                    statusText: 'completed',
                    elapsedSeconds: 0,
                    percent: 100,
                    completed: null,
                    total: null,
                });
                if (status === 'pulled') {
                    setOptimisticDownloadedModel(modelToPull);
                    pullSucceeded = true;
                }
            } else {
                setOllamaPullStatus({ status: 'info', message: `Ollama pull finished for "${modelToPull}" (${status}).` });
            }

            await loadOllamaLocalModels();
            if (pullSucceeded) {
                await testOllama();
            }
        } catch (e: any) {
            const detail = e?.response?.data?.detail || 'Failed to download model in Ollama.';
            setOllamaPullStatus({ status: 'error', message: detail });
        } finally {
            setPullingOllamaModel(false);
        }
    };

    const handleRestart = async (reloadFrontendAfter = false) => {
        const prompt = reloadFrontendAfter
            ? 'Restart the backend and reload the app? Active jobs will be interrupted.'
            : 'Restart the backend server? Active jobs will be interrupted.';
        if (!confirm(prompt)) return;
        setRestarting(true);
        try {
            await api.post('/system/restart');
        } catch {
            // Server will disconnect as it restarts  -  that's expected
        }
        const poll = async () => {
            for (let i = 0; i < 30; i++) {
                await new Promise(r => setTimeout(r, 2000));
                try {
                    await api.get('/settings');
                    setRestarting(false);
                    if (reloadFrontendAfter) {
                        window.location.reload();
                    }
                    return;
                } catch {
                    // still restarting
                }
            }
            setRestarting(false);
            alert('Server did not come back within 60 seconds.');
        };
        poll();
    };

    const testTranscriptionEngine = async () => {
        setTestingTranscriptionEngine(true);
        setTranscriptionEngineTestResult(null);
        try {
            const res = await api.post<TranscriptionEngineTestResult>('/settings/test-transcription-engine', {
                engine: transcriptionEngine,
            });
            setTranscriptionEngineTestResult(res.data);
        } catch (e: any) {
            const detail = String(e?.response?.data?.detail || e?.message || 'Unknown error');
            setTranscriptionEngineTestResult({
                status: 'error',
                requested_engine: transcriptionEngine,
                error: detail,
            });
        } finally {
            setTestingTranscriptionEngine(false);
        }
    };

    const handleSave = async (e: FormEvent) => {
        e.preventDefault();
        setSaving(true);
        setStatus('idle');
        try {
            await api.post('/settings', {
                hf_token: token,
                transcription_engine: transcriptionEngine,
                transcription_model: transcriptionModel,
                transcription_compute_type: computeType,
                parakeet_model: parakeetModel,
                parakeet_batch_size: parakeetBatchSize,
                parakeet_batch_auto: parakeetBatchAuto,
                parakeet_require_word_timestamps: parakeetRequireWordTimestamps,
                parakeet_unload_after_transcribe: parakeetUnloadAfterTranscribe,
                beam_size: beamSize,
                vad_filter: vadFilter,
                batched_transcription: batchedTranscription,
                verbose_logging: verboseLogging,
                llm_provider: llmProvider,
                llm_enabled: ollamaEnabled,
                ollama_url: ollamaUrl,
                ollama_model: ollamaModel,
                ollama_model_tier: ollamaModelTier,
                ollama_enabled: ollamaEnabled,
                nvidia_nim_base_url: nvidiaNimBaseUrl,
                nvidia_nim_model: nvidiaNimModel,
                nvidia_nim_api_key: nvidiaNimApiKey,
                nvidia_nim_thinking_mode: nvidiaNimThinkingMode,
                nvidia_nim_min_request_interval_seconds: nvidiaNimMinRequestIntervalSeconds,
                openai_base_url: openaiBaseUrl,
                openai_model: openaiModel,
                openai_api_key: openaiApiKey,
                anthropic_base_url: anthropicBaseUrl,
                anthropic_model: anthropicModel,
                anthropic_api_key: anthropicApiKey,
                gemini_base_url: geminiBaseUrl,
                gemini_model: geminiModel,
                gemini_api_key: geminiApiKey,
                groq_base_url: groqBaseUrl,
                groq_model: groqModel,
                groq_api_key: groqApiKey,
                openrouter_base_url: openrouterBaseUrl,
                openrouter_model: openrouterModel,
                openrouter_api_key: openrouterApiKey,
                xai_base_url: xaiBaseUrl,
                xai_model: xaiModel,
                xai_api_key: xaiApiKey,
                youtube_oauth_client_id: youtubeOauthClientId,
                youtube_oauth_client_secret: youtubeOauthClientSecret,
                youtube_oauth_redirect_uri: youtubeOauthRedirectUri,
                youtube_publish_push_enabled: youtubePublishPushEnabled,
                ytdlp_cookies_file: ytdlpCookiesFile,
                ytdlp_cookies_from_browser: ytdlpCookiesFromBrowser,
                diarization_sensitivity: diarizationSensitivity,
                speaker_match_threshold: speakerMatchThreshold,
                funny_moments_max_saved: funnyMomentsMaxSaved,
                funny_moments_explain_batch_limit: funnyMomentsExplainBatchLimit,
            });
            setStatus('success');
            if (token) {
                validateToken();
            } else {
                setTokenValidation(null);
            }
            setTimeout(() => setStatus('idle'), 3000);
        } catch (e) {
            console.error('Failed to save settings:', e);
            setStatus('error');
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return <div className="p-8">Loading settings...</div>;
    }

    const requiredModels = [
        { id: 'pyannote/speaker-diarization-3.1', name: 'Speaker Diarization 3.1' },
        { id: 'pyannote/speaker-diarization-community-1', name: 'Speaker Diarization Community 1' },
        { id: 'pyannote/segmentation-3.0', name: 'Segmentation 3.0' },
        { id: 'pyannote/embedding', name: 'Speaker Embedding' }
    ];
    const ollamaSelectableModels = ollamaLocalModels.filter(model => {
        const lowerName = (model.name || '').toLowerCase();
        if (lowerName.includes('embed') || lowerName.includes('embedding')) return false;
        const fams = Array.isArray(model.families) ? model.families : [];
        if (fams.some(f => String(f || '').toLowerCase().includes('embed'))) return false;
        return true;
    });
    const isSelectedOllamaModelDownloadedByList = ollamaSelectableModels.some(
        m => ollamaModelNameMatches(m.name || '', ollamaModel || '')
    );
    const isSelectedOllamaModelDownloadedByOptimistic = !!optimisticDownloadedModel && ollamaModelNameMatches(optimisticDownloadedModel, ollamaModel || '');
    const isSelectedOllamaModelPendingRefresh = !isSelectedOllamaModelDownloadedByList && isSelectedOllamaModelDownloadedByOptimistic;
    const isSelectedOllamaModelDownloaded = isSelectedOllamaModelDownloadedByList;
    const shouldRenderSelectedFallbackOption = !!ollamaModel.trim() && !isSelectedOllamaModelDownloadedByList;
    const tabs: { key: SettingsTab; label: string; icon: React.ReactNode }[] = [
        { key: 'transcription', label: 'Transcription', icon: <Mic size={16} /> },
        { key: 'diarization', label: 'Diarization', icon: <AudioLines size={16} /> },
        { key: 'llm', label: 'LLM', icon: <Bot size={16} /> },
        { key: 'youtube', label: 'YouTube', icon: <Link2 size={16} /> },
        { key: 'funny', label: 'Funny', icon: <Smile size={16} /> },
        { key: 'system', label: 'System', icon: <Terminal size={16} /> },
    ];

    return (
        <div className="max-w-2xl">
            <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <div className="p-2 bg-slate-100 rounded-lg">
                    <Key className="w-6 h-6 text-slate-700" />
                </div>
                Settings
            </h1>

            {/* Tab Navigation */}
            <div className="flex gap-1 mb-6 bg-slate-100 p-1 rounded-xl overflow-hidden">
                {tabs.map(tab => (
                    <button
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        className={`flex-1 min-w-0 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                            activeTab === tab.key
                                ? 'bg-white text-slate-800 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'
                        }`}
                    >
                        {tab.icon}
                        {tab.label}
                    </button>
                ))}
            </div>

            <form onSubmit={handleSave}>
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-4">
                    <div className="p-6 space-y-6">

                        {/* === TRANSCRIPTION TAB === */}
                        {activeTab === 'transcription' && (
                            <>
                                <div>
                                    <h3 className="font-semibold text-slate-800 mb-4">Model Configuration</h3>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Transcription Engine
                                            </label>
                                            <select
                                                value={transcriptionEngine}
                                                onChange={(e) => setTranscriptionEngine((e.target.value || 'auto') as TranscriptionEngine)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                            >
                                                <option value="auto">Auto (prefer Parakeet when available, fallback to Whisper)</option>
                                                <option value="parakeet">NVIDIA Parakeet first (fallback to Whisper)</option>
                                                <option value="whisper">Whisper only</option>
                                            </select>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Auto keeps cross-platform behavior: Parakeet is used when supported on host/runtime, otherwise Whisper runs automatically.
                                            </p>
                                        </div>

                                        {transcriptionEngine === 'parakeet' && (
                                            <div className="rounded-lg border border-indigo-100 bg-indigo-50/40 p-3 space-y-3">
                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        Parakeet Model
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={parakeetModel}
                                                        onChange={(e) => setParakeetModel(e.target.value)}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                                        placeholder="nvidia/parakeet-tdt-0.6b-v2"
                                                    />
                                                </div>
                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                                    <div>
                                                        <label className="block text-sm font-medium text-slate-700 mb-2">
                                                            Parakeet Batch Size
                                                        </label>
                                                        <input
                                                            type="number"
                                                            min={1}
                                                            max={64}
                                                            value={parakeetBatchSize}
                                                            onChange={(e) => setParakeetBatchSize(Number(e.target.value || 16))}
                                                            disabled={parakeetBatchAuto}
                                                            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                                        />
                                                        <p className="mt-1 text-xs text-slate-500">
                                                            {parakeetBatchAuto
                                                                ? 'Auto mode adjusts batch size to available VRAM and free GPU memory.'
                                                                : 'Manual override. Higher batch can increase speed but may trigger shared-memory spill and slowdown.'}
                                                        </p>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200">
                                                        <div className="flex-1">
                                                            <label className="block text-sm font-medium text-slate-700">
                                                                Auto Batch (VRAM-aware)
                                                            </label>
                                                            <p className="text-xs text-slate-500 mt-0.5">
                                                                Automatically reduces Parakeet batch size on lower-VRAM or memory-pressured GPUs.
                                                            </p>
                                                        </div>
                                                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                            <input type="checkbox" checked={parakeetBatchAuto} onChange={(e) => setParakeetBatchAuto(e.target.checked)} className="sr-only peer" />
                                                            <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                                        </label>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 sm:col-span-2">
                                                        <div className="flex-1">
                                                            <label className="block text-sm font-medium text-slate-700">
                                                                Require Word Timestamps
                                                            </label>
                                                            <p className="text-xs text-slate-500 mt-0.5">
                                                                If Parakeet output lacks word timings, fallback to Whisper.
                                                            </p>
                                                        </div>
                                                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                            <input type="checkbox" checked={parakeetRequireWordTimestamps} onChange={(e) => setParakeetRequireWordTimestamps(e.target.checked)} className="sr-only peer" />
                                                            <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                                        </label>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 sm:col-span-2">
                                                        <div className="flex-1">
                                                            <label className="block text-sm font-medium text-slate-700">
                                                                Release Parakeet After Transcribe
                                                            </label>
                                                            <p className="text-xs text-slate-500 mt-0.5">
                                                                Frees GPU memory before diarization/other tasks. Recommended on 16GB-and-below cards.
                                                            </p>
                                                        </div>
                                                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                            <input type="checkbox" checked={parakeetUnloadAfterTranscribe} onChange={(e) => setParakeetUnloadAfterTranscribe(e.target.checked)} className="sr-only peer" />
                                                            <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                                        </label>
                                                    </div>
                                                </div>
                                                <p className="text-xs text-slate-500">
                                                    Install optional Parakeet dependencies in the app venv via <code>backend/requirements-parakeet.txt</code>. If unavailable or unsupported, Whisper fallback is automatic.
                                                </p>
                                            </div>
                                        )}

                                        {transcriptionEngine === 'whisper' && (
                                            <>
                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        Transcription Accuracy (Whisper Model)
                                                    </label>
                                                    <select
                                                        value={transcriptionModel}
                                                        onChange={(e) => setTranscriptionModel(e.target.value)}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                                    >
                                                        <option value="tiny">Tiny  -  Fastest, lowest accuracy (&lt;1GB VRAM)</option>
                                                        <option value="base">Base  -  Fast, basic accuracy (~1GB VRAM)</option>
                                                        <option value="small">Small  -  Balanced speed & accuracy (~2GB VRAM)</option>
                                                        <option value="medium">Medium  -  Slower, good accuracy (~5GB VRAM)</option>
                                                        <option value="large-v3">Large-v3  -  Slowest, best accuracy (~10GB VRAM)</option>
                                                    </select>
                                                    <p className="mt-2 text-xs text-slate-500">
                                                        Larger models produce more accurate transcriptions but run slower and require more VRAM.
                                                    </p>
                                                </div>

                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        Compute Type (Precision)
                                                    </label>
                                                    <select
                                                        value={computeType}
                                                        onChange={(e) => setComputeType(e.target.value)}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                                    >
                                                        <option value="float16">float16  -  Fast on GPU, recommended default</option>
                                                        <option value="int8_float16">int8_float16  -  Faster, ~half the VRAM of float16</option>
                                                        <option value="int8">int8  -  Fastest, best for CPU-only systems</option>
                                                        <option value="float32">float32  -  Slowest, full precision (debug/reference)</option>
                                                    </select>
                                                    <p className="mt-2 text-xs text-slate-500">
                                                        Lower precision runs faster and uses less VRAM with negligible quality loss.
                                                    </p>
                                                </div>
                                            </>
                                        )}

                                        {transcriptionEngine === 'auto' && (
                                            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs text-slate-600">
                                                    Auto mode does not expose engine-specific options here. It will prefer Parakeet on supported CUDA hosts and fallback to Whisper automatically.
                                                </p>
                                            </div>
                                        )}

                                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-3">
                                            <div className="flex items-center justify-between">
                                                <h4 className="text-sm font-medium text-slate-800">Transcription Engine Test</h4>
                                                <button
                                                    type="button"
                                                    onClick={testTranscriptionEngine}
                                                    disabled={testingTranscriptionEngine}
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border border-slate-300 bg-white text-slate-700 hover:bg-slate-100 disabled:opacity-60"
                                                >
                                                    {testingTranscriptionEngine ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                                    Test Selected Engine
                                                </button>
                                            </div>
                                            <p className="text-xs text-slate-500">
                                                Validates that the selected engine can load with current settings and shows which engine is actually usable.
                                            </p>
                                            {transcriptionEngineTestResult && (
                                                <div className={`rounded-lg border p-3 text-sm ${
                                                    transcriptionEngineTestResult.status === 'ok'
                                                        ? 'border-green-200 bg-green-50 text-green-800'
                                                        : 'border-red-200 bg-red-50 text-red-800'
                                                }`}>
                                                    <div className="font-medium">
                                                        {transcriptionEngineTestResult.status === 'ok'
                                                            ? 'Engine test passed'
                                                            : 'Engine test failed'}
                                                    </div>
                                                    <div className="mt-1 text-xs">
                                                        Requested: {transcriptionEngineTestResult.requested_engine} | Resolved: {transcriptionEngineTestResult.resolved_engine || 'unknown'} | Device: {transcriptionEngineTestResult.device || 'unknown'}
                                                    </div>
                                                    {typeof transcriptionEngineTestResult.parakeet_effective_batch_size === 'number' && (
                                                        <div className="mt-1 text-xs">
                                                            Parakeet batch: {transcriptionEngineTestResult.parakeet_effective_batch_size}
                                                            {typeof transcriptionEngineTestResult.parakeet_batch_size_requested === 'number'
                                                                ? ` (requested ${transcriptionEngineTestResult.parakeet_batch_size_requested})`
                                                                : ''}
                                                            {typeof transcriptionEngineTestResult.parakeet_batch_auto === 'boolean'
                                                                ? ` • auto ${transcriptionEngineTestResult.parakeet_batch_auto ? 'on' : 'off'}`
                                                                : ''}
                                                        </div>
                                                    )}
                                                    {transcriptionEngineTestResult.cuda_memory && (
                                                        <div className="mt-1 text-xs">
                                                            CUDA mem: {transcriptionEngineTestResult.cuda_memory.free_gb ?? '?'}GB free / {transcriptionEngineTestResult.cuda_memory.total_gb ?? '?'}GB total
                                                            {typeof transcriptionEngineTestResult.cuda_memory.reserved_gb === 'number'
                                                                ? ` • reserved ${transcriptionEngineTestResult.cuda_memory.reserved_gb}GB`
                                                                : ''}
                                                        </div>
                                                    )}
                                                    {transcriptionEngineTestResult.fallback_used && (
                                                        <div className="mt-1 text-xs">Fallback used: yes</div>
                                                    )}
                                                    {typeof transcriptionEngineTestResult.parakeet_dependencies_available === 'boolean' && (
                                                        <div className="mt-1 text-xs">
                                                            Parakeet deps: {transcriptionEngineTestResult.parakeet_dependencies_available ? 'available' : 'missing'}
                                                        </div>
                                                    )}
                                                    {transcriptionEngineTestResult.error && (
                                                        <div className="mt-1 text-xs">{transcriptionEngineTestResult.error}</div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <div className="flex items-center gap-2 mb-4">
                                        <Zap className="w-5 h-5 text-amber-500" />
                                        <h3 className="font-semibold text-slate-800">Speed Optimization</h3>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                            <div className="flex-1">
                                                <label className="block text-sm font-medium text-slate-700">
                                                    Skip Silent Portions (VAD Filter)
                                                </label>
                                                <p className="text-xs text-slate-500 mt-0.5">
                                                    Uses voice activity detection to skip silence, intros, and dead air.
                                                </p>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                <input type="checkbox" checked={vadFilter} onChange={(e) => setVadFilter(e.target.checked)} className="sr-only peer" />
                                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                            </label>
                                        </div>

                                        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                            <div className="flex-1">
                                                <label className="block text-sm font-medium text-slate-700">
                                                    Batched Transcription (GPU Only)
                                                </label>
                                                <p className="text-xs text-slate-500 mt-0.5">
                                                    Processes multiple audio segments in parallel. Can be 2-3x faster. Requires CUDA GPU.
                                                </p>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                <input type="checkbox" checked={batchedTranscription} onChange={(e) => setBatchedTranscription(e.target.checked)} className="sr-only peer" />
                                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                            </label>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Beam Size (Accuracy vs Speed)
                                            </label>
                                            <select
                                                value={beamSize}
                                                onChange={(e) => setBeamSize(Number(e.target.value))}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                            >
                                                <option value={1}>1  -  Fastest (greedy decoding, recommended)</option>
                                                <option value={2}>2  -  Fast</option>
                                                <option value={3}>3  -  Balanced</option>
                                                <option value={5}>5  -  Accurate (original Whisper default)</option>
                                            </select>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Higher values are more accurate but significantly slower. For podcasts, beam_size=1 usually works well.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* === DIARIZATION TAB === */}
                        {activeTab === 'diarization' && (
                            <>
                                <div>
                                    <div className="flex items-center gap-2 mb-4">
                                        <Key className="w-5 h-5 text-blue-500" />
                                        <h3 className="font-semibold text-slate-800">Hugging Face Token</h3>
                                    </div>

                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                API Token
                                            </label>
                                            <div className="relative">
                                                <input
                                                    type="password"
                                                    value={token}
                                                    onChange={(e) => setToken(e.target.value)}
                                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                                    placeholder="hf_..."
                                                />
                                                {token && (
                                                    <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 font-mono bg-slate-100 px-1.5 py-0.5 rounded">
                                                        HIDDEN
                                                    </div>
                                                )}
                                            </div>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Get a token at{' '}
                                                <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                                                    huggingface.co/settings/tokens
                                                </a>
                                            </p>
                                        </div>

                                        <div className="bg-slate-50 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <h4 className="font-medium text-slate-700 text-sm">Required Model Access</h4>
                                                <button
                                                    type="button"
                                                    onClick={validateToken}
                                                    disabled={validating || !token}
                                                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 disabled:text-slate-400"
                                                >
                                                    {validating ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                    Check Access
                                                </button>
                                            </div>

                                            <p className="text-xs text-slate-500 mb-3">
                                                Speaker diarization requires accepting license agreements for these pyannote models:
                                            </p>

                                            <div className="space-y-2">
                                                {requiredModels.map(model => {
                                                    const modelStatus = tokenValidation?.models[model.id];
                                                    const isAccessible = modelStatus?.accessible;
                                                    const isPending = !tokenValidation || validating;

                                                    return (
                                                        <div key={model.id} className="flex items-center justify-between py-1.5 px-2 bg-white rounded border border-slate-200">
                                                            <div className="flex items-center gap-2">
                                                                {isPending ? (
                                                                    <div className="w-4 h-4 rounded-full bg-slate-200" />
                                                                ) : isAccessible ? (
                                                                    <CheckCircle2 size={16} className="text-green-500" />
                                                                ) : (
                                                                    <AlertCircle size={16} className="text-amber-500" />
                                                                )}
                                                                <span className="text-xs font-medium text-slate-700">{model.name}</span>
                                                            </div>
                                                            {!isPending && !isAccessible && (
                                                                <a
                                                                    href={`https://huggingface.co/${model.id}`}
                                                                    target="_blank"
                                                                    rel="noreferrer"
                                                                    className="flex items-center gap-1 text-xs text-blue-600 hover:underline"
                                                                >
                                                                    Accept <ExternalLink size={10} />
                                                                </a>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>

                                            {tokenValidation && !tokenValidation.valid && tokenValidation.token_set && (
                                                <div className="mt-3 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
                                                    <strong>Action Required:</strong> Click "Accept" on each model page above, then click "Check Access" to verify.
                                                </div>
                                            )}

                                            {tokenValidation?.valid && (
                                                <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-xs text-green-800 flex items-center gap-2">
                                                    <CheckCircle2 size={14} />
                                                    All models accessible  -  diarization ready!
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <div className="flex items-center gap-2 mb-4">
                                        <AudioLines className="w-5 h-5 text-indigo-500" />
                                        <h3 className="font-semibold text-slate-800">Speaker Detection</h3>
                                    </div>

                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Speaker Change Sensitivity
                                            </label>
                                            <select
                                                value={diarizationSensitivity}
                                                onChange={(e) => setDiarizationSensitivity(e.target.value)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                                            >
                                                <option value="conservative">Conservative  -  fewer splits, may merge distinct speakers</option>
                                                <option value="balanced">Balanced  -  pyannote defaults (recommended)</option>
                                                <option value="aggressive">Aggressive  -  more splits, better for rapid back-and-forth</option>
                                            </select>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Controls how sensitive pyannote is to speaker changes. Requires re-running diarization to take effect.
                                            </p>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Speaker Match Threshold (cosine distance: 0-1)
                                            </label>
                                            <div className="flex items-center gap-4">
                                                <input
                                                    type="range"
                                                    min="0.1"
                                                    max="0.9"
                                                    step="0.05"
                                                    value={speakerMatchThreshold}
                                                    onChange={(e) => setSpeakerMatchThreshold(parseFloat(e.target.value))}
                                                    className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                                                />
                                                <span className="text-sm font-mono text-slate-700 w-12 text-right">{speakerMatchThreshold.toFixed(2)}</span>
                                            </div>
                                            <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-0.5">
                                                <span>Strict (more new speakers)</span>
                                                <span>Lenient (more matches)</span>
                                            </div>
                                            <p className="mt-2 text-xs text-slate-500">
                                                How similar a voice must sound to an existing speaker to be matched. Default: 0.50.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* === LLM TAB === */}
                        {activeTab === 'llm' && (
                            <>
                                <div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <Bot className="w-5 h-5 text-purple-500" />
                                        <h3 className="font-semibold text-slate-800">LLM Provider (Humor Explanations + Future Speaker ID)</h3>
                                    </div>
                                    <p className="text-xs text-slate-500 mb-4">
                                        Configure the AI provider used for funny-moment explanations (global humor context + per-laugh summaries). This will also be reused for future speaker-identification prompts.
                                    </p>

                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                            <div className="flex-1">
                                                <label className="block text-sm font-medium text-slate-700">
                                                    Enable LLM Features
                                                </label>
                                                <p className="text-xs text-slate-500 mt-0.5">
                                                    Enables humor explanation generation (and future LLM-based speaker identification) using the selected provider.
                                                </p>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                <input type="checkbox" checked={ollamaEnabled} onChange={(e) => setOllamaEnabled(e.target.checked)} className="sr-only peer" />
                                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                            </label>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Provider
                                            </label>
                                            <select
                                                value={llmProvider}
                                                onChange={(e) => {
                                                    setLlmProvider(e.target.value as LlmProvider);
                                                    setOllamaTest(null);
                                                    setNvidiaNimTest(null);
                                                }}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                            >
                                                <option value="ollama">Ollama (local)</option>
                                                <option value="openai">ChatGPT (OpenAI API)</option>
                                                <option value="anthropic">Claude (Anthropic API)</option>
                                                <option value="gemini">Gemini (Google API)</option>
                                                <option value="nvidia_nim">NVIDIA NIM (hosted API)</option>
                                                <option value="groq">Groq (hosted API)</option>
                                                <option value="openrouter">OpenRouter (hosted API)</option>
                                                <option value="xai">xAI Grok (hosted API)</option>
                                            </select>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Ollama runs locally. Hosted providers use API keys. The selected provider is used for all non-local LLM tasks.
                                            </p>
                                        </div>

                                        {llmProvider === 'ollama' && (
                                            <>
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Ollama URL
                                            </label>
                                            <input
                                                type="text"
                                                value={ollamaUrl}
                                                onChange={(e) => setOllamaUrl(e.target.value)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                placeholder="http://localhost:11434"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Model
                                            </label>
                                            <div className="space-y-2">
                                                <div className="flex items-center justify-between">
                                                    <p className="text-xs text-slate-500">
                                                        Downloaded local chat models: <span className="font-semibold text-slate-700">{ollamaSelectableModels.length}</span>
                                                        {ollamaLocalModels.length !== ollamaSelectableModels.length && (
                                                            <span className="text-slate-400"> ({ollamaLocalModels.length} total)</span>
                                                        )}
                                                    </p>
                                                    <button
                                                        type="button"
                                                        onClick={() => void loadOllamaLocalModels()}
                                                        disabled={loadingOllamaLocalModels}
                                                        className="px-2 py-1 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1"
                                                    >
                                                        {loadingOllamaLocalModels ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                                                        Refresh list
                                                    </button>
                                                </div>
                                                <select
                                                    value={ollamaModel}
                                                    onChange={(e) => setOllamaModel(e.target.value)}
                                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                >
                                                    {shouldRenderSelectedFallbackOption && (
                                                        <option value={ollamaModel}>
                                                            {ollamaModel}
                                                            {isSelectedOllamaModelPendingRefresh ? ' (verifying local list)' : ' (not downloaded)'}
                                                        </option>
                                                    )}
                                                    {ollamaSelectableModels.map(model => (
                                                        <option key={model.name} value={model.name}>
                                                            {model.name}
                                                            {typeof model.size_bytes === 'number' && model.size_bytes > 0 ? ` • ${formatBytesAsGb(model.size_bytes)}` : ''}
                                                            {model.quantization_level ? ` • ${model.quantization_level}` : ''}
                                                        </option>
                                                    ))}
                                                    {ollamaSelectableModels.length === 0 && (
                                                        <option value={ollamaModel || 'mistral'}>
                                                            {ollamaModel || 'mistral'}
                                                        </option>
                                                    )}
                                                </select>
                                                {ollamaLocalModelsError && (
                                                    <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                                                        {ollamaLocalModelsError}
                                                    </div>
                                                )}
                                                <div className={`text-xs ${
                                                    isSelectedOllamaModelDownloaded
                                                        ? 'text-green-700'
                                                        : isSelectedOllamaModelPendingRefresh
                                                            ? 'text-blue-700'
                                                            : 'text-amber-700'
                                                }`}>
                                                    {isSelectedOllamaModelDownloaded
                                                        ? 'Selected model is downloaded locally.'
                                                        : isSelectedOllamaModelPendingRefresh
                                                            ? 'Download finished. Waiting for local model list to confirm exact model name...'
                                                            : ollamaPullProgress?.running
                                                                ? 'Download in progress for selected model...'
                                                                : 'Selected model is not downloaded yet. Use "Download Current Model".'}
                                                </div>
                                            </div>
                                            <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2">
                                                <div className="flex items-center justify-between gap-2">
                                                    <div>
                                                        <div className="text-xs font-semibold text-slate-700">Unsloth GGUF Presets (Qwen families)</div>
                                                        <p className="text-[11px] text-slate-500 mt-0.5">
                                                            Uses Hugging Face GGUF tags via Ollama (`hf.co/...`), including Unsloth quants.
                                                        </p>
                                                    </div>
                                                    <a
                                                        href={`https://huggingface.co/${(UNSLOTH_FAMILIES.find(f => f.key === unslothFamily)?.repo || UNSLOTH_FAMILIES[0].repo).replace(/^hf\.co\//i, '')}`}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        className="text-xs text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1 whitespace-nowrap"
                                                    >
                                                        Model page <ExternalLink size={11} />
                                                    </a>
                                                </div>
                                                <div className="flex flex-col gap-2">
                                                    <select
                                                        value={unslothFamily}
                                                        onChange={(e) => setUnslothFamily(e.target.value as UnslothFamilyKey)}
                                                        className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all bg-white"
                                                    >
                                                        {UNSLOTH_FAMILIES.map(family => (
                                                            <option key={family.key} value={family.key}>
                                                                {family.label} • {family.note}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                                <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
                                                    <select
                                                        value={unslothQuant}
                                                        onChange={(e) => setUnslothQuant(e.target.value as UnslothQuant)}
                                                        className="flex-1 min-w-0 px-3 py-1.5 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all bg-white"
                                                    >
                                                        {UNSLOTH_QUANTS.map(opt => (
                                                            <option key={opt.value} value={opt.value}>
                                                                {opt.label} • ~{estimateUnslothSizeGb(unslothFamily, opt.value).toFixed(1)} GB • {opt.note}
                                                            </option>
                                                        ))}
                                                    </select>
                                                    <div className="flex gap-2">
                                                        <button
                                                            type="button"
                                                            onClick={() => applyUnslothPreset(false)}
                                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-indigo-50 text-indigo-700 hover:bg-indigo-100"
                                                        >
                                                            Use Preset
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => applyUnslothPreset(true)}
                                                            disabled={pullingOllamaModel || !ollamaUrl.trim()}
                                                            className="px-2.5 py-1.5 rounded-md text-xs font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed"
                                                        >
                                                            Use + Download
                                                        </button>
                                                    </div>
                                                </div>
                                                <div className="text-[11px] text-slate-500">
                                                    Selected tag: <code className="bg-white border border-slate-200 px-1.5 py-0.5 rounded">{buildUnslothTag(unslothFamily, unslothQuant)}</code>
                                                </div>
                                                {unslothFamily === 'qwen35_35b_a3b' && (
                                                    <div className="text-[11px] text-slate-500">
                                                        35B-A3B note: this family uses Unsloth <code className="bg-white border border-slate-200 px-1.5 py-0.5 rounded">UD-*</code> quant tags for Ollama pulls.
                                                    </div>
                                                )}
                                                <div className="text-[11px] text-slate-500">
                                                    Estimated size: <span className="font-medium text-slate-700">~{estimateUnslothSizeGb(unslothFamily, unslothQuant).toFixed(1)} GB</span>
                                                </div>
                                            </div>
                                            <div className="mt-3">
                                                <label className="block text-xs font-semibold text-slate-600 mb-2">
                                                    Optimization Preference
                                                </label>
                                                <div className="inline-flex w-full rounded-lg border border-slate-200 bg-slate-50 p-1">
                                                    {[
                                                        { key: 'speed', label: 'Speed' },
                                                        { key: 'balanced', label: 'Balanced' },
                                                        { key: 'capability', label: 'Capability' },
                                                    ].map(option => (
                                                        <button
                                                            key={option.key}
                                                            type="button"
                                                            onClick={() => setOllamaPreference(option.key as OllamaPreference)}
                                                            className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition ${
                                                                ollamaPreference === option.key
                                                                    ? 'bg-white text-slate-900 shadow-sm'
                                                                    : 'text-slate-600 hover:text-slate-800'
                                                            }`}
                                                        >
                                                            {option.label}
                                                        </button>
                                                    ))}
                                                </div>
                                                <p className="mt-1 text-[11px] text-slate-500">
                                                    Picks the best model+GGUF quant tag for your VRAM based on your speed vs capability preference.
                                                </p>
                                            </div>
                                            <div className="mt-2 flex flex-wrap items-center gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => void detectOllamaHardwareRecommendation()}
                                                    disabled={detectingOllamaHardware}
                                                    className="px-2.5 py-1 rounded-md text-xs font-medium bg-purple-50 text-purple-700 hover:bg-purple-100 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
                                                >
                                                    {detectingOllamaHardware ? <Loader2 size={12} className="animate-spin" /> : <Zap size={12} />}
                                                    Detect Hardware + Recommend
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => applyDetectedOllamaRecommendation(false)}
                                                    disabled={!ollamaHardwareRecommendation?.recommendation}
                                                    className="px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed"
                                                >
                                                    Apply Recommended
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => applyDetectedOllamaRecommendation(true)}
                                                    disabled={!ollamaHardwareRecommendation?.recommendation || pullingOllamaModel || !ollamaUrl.trim()}
                                                    className="px-2.5 py-1 rounded-md text-xs font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed"
                                                >
                                                    Apply + Download
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => void pullOllamaModel()}
                                                    disabled={pullingOllamaModel || !ollamaModel.trim() || !ollamaUrl.trim()}
                                                    className="px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                                                    title="Download/install the configured model in Ollama if missing"
                                                >
                                                    {pullingOllamaModel ? <Loader2 size={12} className="animate-spin" /> : <Bot size={12} />}
                                                    Download Current Model
                                                </button>
                                            </div>
                                            <p className="mt-2 text-xs text-slate-500">
                                                One local model is used for all local LLM tasks. The recommender picks a model + GGUF quant tag that fits your VRAM and preference.
                                            </p>
                                            <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2.5">
                                                <div className="text-xs font-semibold text-slate-700">Hardware Recommendation</div>
                                                {ollamaHardwareRecommendation?.status === 'error' ? (
                                                    <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                                                        {ollamaHardwareRecommendation.error || 'Could not detect hardware. Set model manually.'}
                                                    </div>
                                                ) : (
                                                    <>
                                                        <div className="text-xs text-slate-600">
                                                            <span className="font-medium">Detected GPU:</span>{' '}
                                                            {ollamaHardwareRecommendation?.hardware?.gpu_name
                                                                ? `${ollamaHardwareRecommendation.hardware.gpu_name}${ollamaHardwareRecommendation.hardware.gpu_vram_gb ? ` • ${ollamaHardwareRecommendation.hardware.gpu_vram_gb} GB VRAM` : ''}`
                                                                : 'Not detected (CPU-only or unavailable)'}
                                                        </div>
                                                        <div className="text-xs text-slate-600">
                                                            <span className="font-medium">Recommended model:</span>{' '}
                                                            <code className="bg-white border border-slate-200 px-1.5 py-0.5 rounded">
                                                                {ollamaHardwareRecommendation?.recommendation?.model_tag || '—'}
                                                            </code>
                                                        </div>
                                                        {typeof ollamaHardwareRecommendation?.recommendation?.estimated_size_gb === 'number' && (
                                                            <div className="text-xs text-slate-600">
                                                                <span className="font-medium">Model size:</span>{' '}
                                                                {ollamaHardwareRecommendation.recommendation.estimated_size_gb.toFixed(1)} GB
                                                                {ollamaHardwareRecommendation?.recommendation?.size_source === 'ollama_exact' ? ' (exact from Ollama)' : ' (estimate)'}
                                                            </div>
                                                        )}
                                                        {ollamaHardwareRecommendation?.recommendation?.quant_level && (
                                                            <div className="text-xs text-slate-600">
                                                                <span className="font-medium">GGUF quant level:</span>{' '}
                                                                {ollamaHardwareRecommendation.recommendation.quant_level}
                                                                {typeof ollamaHardwareRecommendation?.recommendation?.quant_bits_estimate === 'number' && (
                                                                    <> ({ollamaHardwareRecommendation.recommendation.quant_bits_estimate.toFixed(1)}-bit est.)</>
                                                                )}
                                                            </div>
                                                        )}
                                                        <div className="text-xs text-slate-600">
                                                            <span className="font-medium">Tradeoff:</span>{' '}
                                                            {(ollamaHardwareRecommendation?.recommendation?.objective || ollamaPreference)}
                                                            {' • '}
                                                            <span className="font-medium">GGUF tier:</span>{' '}
                                                            {(ollamaHardwareRecommendation?.recommendation?.tier || '—').toUpperCase()}
                                                        </div>
                                                        {ollamaHardwareRecommendation?.recommendation?.reason && (
                                                            <p className="text-[11px] text-slate-500 leading-relaxed">
                                                                {ollamaHardwareRecommendation.recommendation.reason}
                                                            </p>
                                                        )}
                                                    </>
                                                )}
                                            </div>
                                            {ollamaPullProgress && (
                                                <div className="mt-2 p-3 rounded border border-blue-200 bg-blue-50">
                                                    <div className="flex items-center justify-between gap-2 text-xs text-blue-800">
                                                        <span className="font-medium truncate">
                                                            {ollamaPullProgress.running ? 'Downloading model' : 'Download complete'}
                                                            {ollamaPullProgress.statusText ? ` • ${ollamaPullProgress.statusText}` : ''}
                                                        </span>
                                                        <span className="shrink-0">
                                                            {typeof ollamaPullProgress.percent === 'number'
                                                                ? `${ollamaPullProgress.percent.toFixed(1)}%`
                                                                : `${ollamaPullProgress.elapsedSeconds}s`}
                                                        </span>
                                                    </div>
                                                    <div className="mt-2 h-2 rounded-full bg-blue-100 overflow-hidden">
                                                        <div
                                                            className={`h-full rounded-full bg-blue-500 ${typeof ollamaPullProgress.percent === 'number' ? '' : 'animate-pulse'}`}
                                                            style={{
                                                                width: typeof ollamaPullProgress.percent === 'number'
                                                                    ? `${Math.max(2, Math.min(100, ollamaPullProgress.percent))}%`
                                                                    : '35%',
                                                            }}
                                                        />
                                                    </div>
                                                    {typeof ollamaPullProgress.completed === 'number' && typeof ollamaPullProgress.total === 'number' && ollamaPullProgress.total > 0 && (
                                                        <p className="mt-1 text-[11px] text-blue-700">
                                                            {(ollamaPullProgress.completed / (1024 ** 3)).toFixed(1)} GB / {(ollamaPullProgress.total / (1024 ** 3)).toFixed(1)} GB
                                                        </p>
                                                    )}
                                                </div>
                                            )}
                                            {ollamaPullStatus && (
                                                <div className={`mt-2 p-2 rounded text-xs border ${
                                                    ollamaPullStatus.status === 'success'
                                                        ? 'bg-green-50 border-green-200 text-green-800'
                                                        : ollamaPullStatus.status === 'error'
                                                            ? 'bg-red-50 border-red-200 text-red-800'
                                                            : 'bg-blue-50 border-blue-200 text-blue-800'
                                                }`}>
                                                    {ollamaPullStatus.message}
                                                </div>
                                            )}
                                        </div>

                                        <div className="bg-slate-50 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <h4 className="font-medium text-slate-700 text-sm">Connection Test</h4>
                                                <button
                                                    type="button"
                                                    onClick={testOllama}
                                                    disabled={testingOllama}
                                                    className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-700 disabled:text-slate-400"
                                                >
                                                    {testingOllama ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                    Test Connection
                                                </button>
                                            </div>

                                            {testingOllama && (
                                                <div className="flex items-center gap-2 text-sm text-slate-500">
                                                    <Loader2 size={14} className="animate-spin" /> Testing connection & model...
                                                </div>
                                            )}

                                            {ollamaTest && ollamaTest.status === 'ok' && (
                                                <div className="space-y-2">
                                                    <div className="p-2 bg-green-50 border border-green-200 rounded text-xs text-green-800 flex items-center gap-2">
                                                        <CheckCircle2 size={14} />
                                                        Connected  -  model "{ollamaTest.model}" responded: "{ollamaTest.test_response}"
                                                    </div>
                                                    {(typeof ollamaTest.latency_ms === 'number' || typeof ollamaTest.tags_latency_ms === 'number') && (
                                                        <p className="text-xs text-slate-500">
                                                            {typeof ollamaTest.latency_ms === 'number' && (
                                                                <span>Generation latency: {(ollamaTest.latency_ms / 1000).toFixed(2)}s</span>
                                                            )}
                                                            {typeof ollamaTest.latency_ms === 'number' && typeof ollamaTest.tags_latency_ms === 'number' && '  |  '}
                                                            {typeof ollamaTest.tags_latency_ms === 'number' && (
                                                                <span>Model list check: {(ollamaTest.tags_latency_ms / 1000).toFixed(2)}s</span>
                                                            )}
                                                        </p>
                                                    )}
                                                    {ollamaTest.available_models && ollamaTest.available_models.length > 0 && (
                                                        <p className="text-xs text-slate-500">
                                                            Available models: {ollamaTest.available_models.join(', ')}
                                                        </p>
                                                    )}
                                                </div>
                                            )}

                                            {ollamaTest && ollamaTest.status === 'model_not_found' && (
                                                <div className="space-y-2">
                                                    <div className="p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
                                                        <strong>Model not found:</strong> {ollamaTest.error}
                                                    </div>
                                                    {typeof ollamaTest.tags_latency_ms === 'number' && (
                                                        <p className="text-xs text-slate-500">
                                                            Model list check: {(ollamaTest.tags_latency_ms / 1000).toFixed(2)}s
                                                        </p>
                                                    )}
                                                    {ollamaTest.available_models && ollamaTest.available_models.length > 0 && (
                                                        <p className="text-xs text-slate-500">
                                                            Available models: {ollamaTest.available_models.join(', ')}
                                                        </p>
                                                    )}
                                                </div>
                                            )}

                                            {ollamaTest && ollamaTest.status === 'error' && (
                                                <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-800">
                                                    <strong>Error:</strong> {ollamaTest.error}
                                                </div>
                                            )}

                                            {ollamaTest && ollamaTest.status === 'generation_failed' && (
                                                <div className="p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
                                                    <strong>Generation failed:</strong> {ollamaTest.error}
                                                </div>
                                            )}
                                        </div>
                                            </>
                                        )}

                                        {llmProvider !== 'ollama' && (
                                            <>
                                                <div className="rounded-lg border border-blue-200 bg-blue-50/60 p-4">
                                                    <h4 className="text-sm font-semibold text-blue-900">
                                                        {HOSTED_PROVIDER_LABELS[llmProvider as Exclude<LlmProvider, 'ollama'>]} Setup
                                                    </h4>
                                                    <p className="mt-1 text-xs text-blue-800">
                                                        Hosted API provider (no local download). Set base URL, model, API key, save, then run connection test.
                                                    </p>
                                                    <div className="mt-3 grid gap-2 text-xs">
                                                        {llmProvider === 'openai' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://platform.openai.com/docs/quickstart" target="_blank" rel="noreferrer">
                                                                OpenAI API quickstart <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                        {llmProvider === 'anthropic' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://docs.anthropic.com/en/api/getting-started" target="_blank" rel="noreferrer">
                                                                Anthropic API docs <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                        {llmProvider === 'gemini' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://ai.google.dev/gemini-api/docs/quickstart" target="_blank" rel="noreferrer">
                                                                Gemini API quickstart <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                        {llmProvider === 'nvidia_nim' && (
                                                            <>
                                                                {NVIDIA_NIM_MODEL_PRESETS.map((preset) => (
                                                                    <a
                                                                        key={preset.id}
                                                                        className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1"
                                                                        href={preset.modelCardUrl}
                                                                        target="_blank"
                                                                        rel="noreferrer"
                                                                    >
                                                                        {preset.label} model card <ExternalLink size={12} />
                                                                    </a>
                                                                ))}
                                                                <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://docs.api.nvidia.com/nim/docs/api-quickstart" target="_blank" rel="noreferrer">
                                                                    NVIDIA API quickstart <ExternalLink size={12} />
                                                                </a>
                                                            </>
                                                        )}
                                                        {llmProvider === 'groq' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://console.groq.com/docs/quickstart" target="_blank" rel="noreferrer">
                                                                Groq API quickstart <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                        {llmProvider === 'openrouter' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://openrouter.ai/docs/quick-start" target="_blank" rel="noreferrer">
                                                                OpenRouter quickstart <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                        {llmProvider === 'xai' && (
                                                            <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://docs.x.ai/docs/overview" target="_blank" rel="noreferrer">
                                                                xAI API docs <ExternalLink size={12} />
                                                            </a>
                                                        )}
                                                    </div>
                                                </div>

                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        Base URL
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={getHostedBaseUrl()}
                                                        onChange={(e) => setHostedBaseUrl(e.target.value)}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                        placeholder="Provider API base URL"
                                                    />
                                                </div>

                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        Model
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={getHostedModel()}
                                                        onChange={(e) => setHostedModel(e.target.value)}
                                                        list={llmProvider === 'nvidia_nim' ? 'nvidia-nim-model-presets' : undefined}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                        placeholder="Provider model ID"
                                                    />
                                                    {llmProvider === 'nvidia_nim' && (
                                                        <>
                                                            <datalist id="nvidia-nim-model-presets">
                                                                {NVIDIA_NIM_MODEL_PRESETS.map((preset) => (
                                                                    <option key={preset.id} value={preset.id}>
                                                                        {preset.label}
                                                                    </option>
                                                                ))}
                                                            </datalist>
                                                            <div className="mt-2 flex flex-wrap gap-2">
                                                                {NVIDIA_NIM_MODEL_PRESETS.map((preset) => {
                                                                    const selected = nvidiaNimModel.trim() === preset.id;
                                                                    return (
                                                                        <button
                                                                            key={preset.id}
                                                                            type="button"
                                                                            onClick={() => setNvidiaNimModel(preset.id)}
                                                                            className={`rounded-lg border px-3 py-1 text-xs transition-colors ${selected ? 'border-blue-300 bg-blue-50 text-blue-700' : 'border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:text-blue-700'}`}
                                                                            title={preset.note}
                                                                        >
                                                                            Use {preset.label}
                                                                        </button>
                                                                    );
                                                                })}
                                                            </div>
                                                        </>
                                                    )}
                                                </div>

                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">
                                                        API Key
                                                    </label>
                                                    <input
                                                        type="password"
                                                        value={getHostedApiKey()}
                                                        onChange={(e) => setHostedApiKey(e.target.value)}
                                                        className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                        placeholder="Paste API key"
                                                        autoComplete="off"
                                                    />
                                                    <p className="mt-2 text-xs text-slate-500">
                                                        Stored in backend <code className="bg-slate-100 px-1 rounded">.env</code> for the selected provider.
                                                    </p>
                                                </div>

                                                {llmProvider === 'nvidia_nim' && (
                                                    <>
                                                        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                                            <div className="flex-1">
                                                                <label className="block text-sm font-medium text-slate-700">
                                                                    NIM Thinking Mode (optional)
                                                                </label>
                                                                <p className="text-xs text-slate-500 mt-0.5">
                                                                    Sends <code className="bg-slate-100 px-1 rounded">chat_template_kwargs.thinking=true</code> in NIM requests. Some models ignore this.
                                                                </p>
                                                            </div>
                                                            <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                                <input type="checkbox" checked={nvidiaNimThinkingMode} onChange={(e) => setNvidiaNimThinkingMode(e.target.checked)} className="sr-only peer" />
                                                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                                            </label>
                                                        </div>

                                                        <div>
                                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                                NIM Request Spacing (seconds)
                                                            </label>
                                                            <input
                                                                type="number"
                                                                min={0}
                                                                max={30}
                                                                step={0.5}
                                                                value={nvidiaNimMinRequestIntervalSeconds}
                                                                onChange={(e) => {
                                                                    const parsed = Number.parseFloat(e.target.value);
                                                                    setNvidiaNimMinRequestIntervalSeconds(Number.isFinite(parsed) ? parsed : 0);
                                                                }}
                                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                                            />
                                                            <p className="mt-2 text-xs text-slate-500">
                                                                Minimum delay between NIM requests to reduce 429 rate limits during chunked analysis.
                                                            </p>
                                                        </div>
                                                    </>
                                                )}

                                                <div className="bg-slate-50 rounded-lg p-4">
                                                    <div className="flex items-center justify-between mb-3">
                                                        <h4 className="font-medium text-slate-700 text-sm">
                                                            {HOSTED_PROVIDER_LABELS[llmProvider as Exclude<LlmProvider, 'ollama'>]} Connection Test
                                                        </h4>
                                                        <button
                                                            type="button"
                                                            onClick={testHostedLlm}
                                                            disabled={testingNvidiaNim}
                                                            className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-700 disabled:text-slate-400"
                                                        >
                                                            {testingNvidiaNim ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                            Test Connection
                                                        </button>
                                                    </div>

                                                    {testingNvidiaNim && (
                                                        <div className="flex items-center gap-2 text-sm text-slate-500">
                                                            <Loader2 size={14} className="animate-spin" /> Testing hosted chat completions...
                                                        </div>
                                                    )}

                                                    {nvidiaNimTest && nvidiaNimTest.status === 'ok' && (
                                                        <div className="p-2 bg-green-50 border border-green-200 rounded text-xs text-green-800 flex items-center gap-2">
                                                            <CheckCircle2 size={14} />
                                                            Connected  -  model "{nvidiaNimTest.model}" responded: "{nvidiaNimTest.test_response}"
                                                        </div>
                                                    )}
                                                    {nvidiaNimTest && nvidiaNimTest.status === 'ok' && typeof nvidiaNimTest.latency_ms === 'number' && (
                                                        <p className="text-xs text-slate-500">
                                                            Request latency: {(nvidiaNimTest.latency_ms / 1000).toFixed(2)}s
                                                        </p>
                                                    )}
                                                    {nvidiaNimTest && nvidiaNimTest.status !== 'ok' && (
                                                        <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-800">
                                                            <strong>Error:</strong> {nvidiaNimTest.error || 'Unknown error'}
                                                        </div>
                                                    )}
                                                    {nvidiaNimTest && nvidiaNimTest.status !== 'ok' && typeof nvidiaNimTest.latency_ms === 'number' && (
                                                        <p className="text-xs text-slate-500">
                                                            Request latency: {(nvidiaNimTest.latency_ms / 1000).toFixed(2)}s
                                                        </p>
                                                    )}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </>
                        )}

                        {/* === YOUTUBE TAB === */}
                        {activeTab === 'youtube' && (
                            <>
                                <div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <Link2 className="w-5 h-5 text-red-500" />
                                        <h3 className="font-semibold text-slate-800">YouTube Publishing (OAuth)</h3>
                                    </div>
                                    <p className="text-xs text-slate-500 mb-4">
                                        Connect a YouTube account/channel you control to push AI-generated description updates via the YouTube Data API. Local description archiving and restore history is always preserved.
                                    </p>

                                    <div className="space-y-4">
                                        <div className="rounded-lg border border-red-100 bg-red-50/40 p-4">
                                            <div className="flex items-center justify-between gap-3">
                                                <div className="min-w-0">
                                                    <div className="text-sm font-semibold text-slate-800">Connection Status</div>
                                                    {loadingYoutubeOauthStatus ? (
                                                        <div className="mt-1 text-xs text-slate-500 flex items-center gap-2">
                                                            <Loader2 size={12} className="animate-spin" /> Checking OAuth status...
                                                        </div>
                                                    ) : youtubeOauthStatus ? (
                                                        <div className="mt-1 space-y-1 text-xs">
                                                            <div className={youtubeOauthStatus.connected ? 'text-green-700' : 'text-slate-600'}>
                                                                {youtubeOauthStatus.connected
                                                                    ? `Connected${youtubeOauthStatus.channel_title ? `: ${youtubeOauthStatus.channel_title}` : ''}`
                                                                    : youtubeOauthStatus.configured
                                                                        ? 'Configured but not connected'
                                                                        : 'Not configured'}
                                                            </div>
                                                            {youtubeOauthStatus.channel_id && (
                                                                <div className="text-slate-500">Channel ID: <code className="bg-white px-1 rounded">{youtubeOauthStatus.channel_id}</code></div>
                                                            )}
                                                            {youtubeOauthStatus.token_expires_at && (
                                                                <div className="text-slate-500">Access token expires: {new Date(youtubeOauthStatus.token_expires_at).toLocaleString()}</div>
                                                            )}
                                                        </div>
                                                    ) : (
                                                        <div className="mt-1 text-xs text-slate-500">Status unavailable.</div>
                                                    )}
                                                </div>
                                                <div className="flex flex-col gap-2 shrink-0">
                                                    <button
                                                        type="button"
                                                        onClick={connectYouTubeOAuth}
                                                        disabled={connectingYouTube}
                                                        className="px-3 py-2 rounded-lg text-xs font-medium bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                                                    >
                                                        {connectingYouTube ? 'Connecting...' : 'Connect YouTube'}
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={testYouTubeOAuth}
                                                        disabled={testingYouTubeOauth}
                                                        className="px-3 py-2 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50 inline-flex items-center justify-center gap-1"
                                                    >
                                                        {testingYouTubeOauth ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                        Test
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={disconnectYouTubeOAuth}
                                                        disabled={disconnectingYouTube || !(youtubeOauthStatus?.connected)}
                                                        className="px-3 py-2 rounded-lg text-xs font-medium bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 disabled:opacity-50"
                                                    >
                                                        {disconnectingYouTube ? 'Disconnecting...' : 'Disconnect'}
                                                    </button>
                                                </div>
                                            </div>
                                            {youtubeOauthTestResult && (
                                                <div className={`mt-3 p-2 rounded text-xs border ${
                                                    youtubeOauthTestResult.status === 'ok'
                                                        ? 'bg-green-50 border-green-200 text-green-800'
                                                        : 'bg-red-50 border-red-200 text-red-800'
                                                }`}>
                                                    {youtubeOauthTestResult.status === 'ok'
                                                        ? `Connected to "${youtubeOauthTestResult.channel_title}" (${youtubeOauthTestResult.channel_id})`
                                                        : `Error: ${youtubeOauthTestResult.error || 'Unknown error'}`}
                                                </div>
                                            )}
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                YouTube OAuth Client ID
                                            </label>
                                            <input
                                                type="text"
                                                value={youtubeOauthClientId}
                                                onChange={(e) => setYoutubeOauthClientId(e.target.value)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all"
                                                placeholder="Google OAuth client ID"
                                                autoComplete="off"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                YouTube OAuth Client Secret
                                            </label>
                                            <input
                                                type="password"
                                                value={youtubeOauthClientSecret}
                                                onChange={(e) => setYoutubeOauthClientSecret(e.target.value)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all"
                                                placeholder="Google OAuth client secret"
                                                autoComplete="off"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                OAuth Redirect URI
                                            </label>
                                            <input
                                                type="text"
                                                value={youtubeOauthRedirectUri}
                                                onChange={(e) => setYoutubeOauthRedirectUri(e.target.value)}
                                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all"
                                                placeholder="http://localhost:8000/auth/youtube/callback"
                                            />
                                            <p className="mt-2 text-xs text-slate-500">
                                                Add this exact URI to your Google Cloud OAuth client's authorized redirect URIs.
                                            </p>
                                        </div>

                                        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                            <div className="flex-1">
                                                <label className="block text-sm font-medium text-slate-700">
                                                    Push to YouTube on Publish (default)
                                                </label>
                                                <p className="text-xs text-slate-500 mt-0.5">
                                                    When enabled, publishing an AI description from the episode YouTube tab (or channel batch publish) also calls YouTube Data API <code className="bg-white px-1 rounded">videos.update</code>.
                                                </p>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer ml-4">
                                                <input type="checkbox" checked={youtubePublishPushEnabled} onChange={(e) => setYoutubePublishPushEnabled(e.target.checked)} className="sr-only peer" />
                                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-red-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-red-500"></div>
                                            </label>
                                        </div>

                                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-4">
                                            <div>
                                                <h4 className="text-sm font-semibold text-slate-800">yt-dlp Authentication (Age-Restricted Videos)</h4>
                                                <p className="mt-1 text-xs text-slate-500">
                                                    Use one method below when YouTube requires sign-in (for age-gated videos). This is used for scans, prefetch, processing, clip render input, and thumbnail extraction.
                                                </p>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">
                                                    Cookie File Path (recommended on servers)
                                                </label>
                                                <input
                                                    type="text"
                                                    value={ytdlpCookiesFile}
                                                    onChange={(e) => setYtdlpCookiesFile(e.target.value)}
                                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all"
                                                    placeholder="C:\\path\\to\\youtube_cookies.txt"
                                                    autoComplete="off"
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">
                                                    Cookies From Browser (optional)
                                                </label>
                                                <input
                                                    type="text"
                                                    value={ytdlpCookiesFromBrowser}
                                                    onChange={(e) => setYtdlpCookiesFromBrowser(e.target.value)}
                                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all"
                                                    placeholder="chrome OR chrome:Default OR firefox:default-release"
                                                    autoComplete="off"
                                                />
                                                <p className="mt-2 text-xs text-slate-500">
                                                    Format: <code className="bg-white px-1 rounded">browser</code> or <code className="bg-white px-1 rounded">browser:profile</code>. Leave blank if using a cookie file.
                                                </p>
                                            </div>

                                            <div className="text-xs text-slate-600 space-y-1">
                                                <div>After saving settings, queued prefetch retries are automatically throttled for auth-gated videos to reduce log noise.</div>
                                                <div className="pt-1">
                                                    <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies" target="_blank" rel="noreferrer">
                                                        yt-dlp cookie export guide <ExternalLink size={12} />
                                                    </a>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600 space-y-1.5">
                                            <div className="font-semibold text-slate-700">Setup (Google Cloud)</div>
                                            <div>1. Create an OAuth client in Google Cloud Console (Web application).</div>
                                            <div>2. Enable YouTube Data API v3 for the project.</div>
                                            <div>3. Add the Redirect URI shown above.</div>
                                            <div>4. Save settings, then click <span className="font-medium">Connect YouTube</span>.</div>
                                            <div className="pt-1 flex flex-col gap-1">
                                                <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://developers.google.com/youtube/v3/guides/auth/server-side-web-apps" target="_blank" rel="noreferrer">
                                                    YouTube OAuth guide (server-side web apps) <ExternalLink size={12} />
                                                </a>
                                                <a className="text-blue-700 hover:text-blue-800 underline inline-flex items-center gap-1" href="https://developers.google.com/youtube/v3/docs/videos/update" target="_blank" rel="noreferrer">
                                                    YouTube Data API videos.update reference <ExternalLink size={12} />
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* === FUNNY MOMENTS TAB === */}
                        {activeTab === 'funny' && (
                            <>
                                <div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <Smile className="w-5 h-5 text-amber-500" />
                                        <h3 className="font-semibold text-slate-800">Funny Moments Analysis</h3>
                                    </div>
                                    <p className="text-xs text-slate-500 mb-4">
                                        Tune how many funny moments are saved after detection and how many are explained per Explain/Re-explain run.
                                    </p>

                                    <div className="space-y-5">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Saved Funny Moments (Top-N)
                                            </label>
                                            <div className="flex items-center gap-4">
                                                <input
                                                    type="range"
                                                    min="5"
                                                    max="100"
                                                    step="5"
                                                    value={funnyMomentsMaxSaved}
                                                    onChange={(e) => setFunnyMomentsMaxSaved(Number(e.target.value))}
                                                    className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                                                />
                                                <span className="text-sm font-mono text-slate-700 w-12 text-right">{funnyMomentsMaxSaved}</span>
                                            </div>
                                            <p className="mt-2 text-xs text-slate-500">
                                                After detection, the app keeps the highest-ranked moments (then displays them chronologically). Higher values capture more possible laughs but may include weaker candidates.
                                            </p>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                                Explain/Re-explain Batch Size
                                            </label>
                                            <div className="flex items-center gap-4">
                                                <input
                                                    type="range"
                                                    min="1"
                                                    max="100"
                                                    step="1"
                                                    value={funnyMomentsExplainBatchLimit}
                                                    onChange={(e) => setFunnyMomentsExplainBatchLimit(Number(e.target.value))}
                                                    className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
                                                />
                                                <span className="text-sm font-mono text-slate-700 w-12 text-right">{funnyMomentsExplainBatchLimit}</span>
                                            </div>
                                            <p className="mt-2 text-xs text-slate-500">
                                                Maximum number of saved funny moments explained per click. Higher values take longer, especially with remote models like Kimi (one LLM call per moment, plus global context generation).
                                            </p>
                                        </div>

                                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
                                            <div className="font-medium text-slate-700 mb-1">How this is used</div>
                                            <ul className="space-y-1 list-disc pl-4">
                                                <li><span className="font-medium">Rescan</span> saves up to the configured Top-N moments.</li>
                                                <li><span className="font-medium">Explain/Re-explain</span> processes moments serially up to the configured batch size.</li>
                                                <li>If batch size exceeds saved moments, it simply explains all saved moments.</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* === SYSTEM TAB === */}
                        {activeTab === 'system' && (
                            <>
                                <div>
                                    <h3 className="font-semibold text-slate-800 mb-4">Logging</h3>
                                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                                        <div className="flex-1">
                                            <label className="block text-sm font-medium text-slate-700 flex items-center gap-2">
                                                <Terminal size={16} />
                                                Verbose Logging
                                            </label>
                                            <p className="text-xs text-slate-500 mt-0.5">
                                                Show detailed debug output in the terminal. Disable for cleaner logs.
                                            </p>
                                        </div>
                                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                                            <input type="checkbox" checked={verboseLogging} onChange={(e) => setVerboseLogging(e.target.checked)} className="sr-only peer" />
                                            <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                        </label>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                                            <Database size={16} />
                                            Database Health
                                        </h3>
                                        <button
                                            type="button"
                                            onClick={() => void loadDbHealth()}
                                            disabled={loadingDbHealth}
                                            className="px-3 py-1.5 text-xs rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 disabled:opacity-50 inline-flex items-center gap-1"
                                        >
                                            {loadingDbHealth ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                            Refresh
                                        </button>
                                    </div>

                                    {dbHealthError && (
                                        <div className="mb-3 text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                                            {dbHealthError}
                                        </div>
                                    )}

                                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
                                        {!dbHealth ? (
                                            <div className="text-xs text-slate-500">Loading database metrics...</div>
                                        ) : (
                                            <>
                                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Connections</div>
                                                        <div className="mt-1 text-sm font-semibold text-slate-800">
                                                            {dbHealth.database.connections.total ?? '-'}
                                                            {dbHealth.database.connections.max ? ` / ${dbHealth.database.connections.max}` : ''}
                                                        </div>
                                                        <div className="text-[11px] text-slate-500">
                                                            active: {dbHealth.database.connections.active ?? '-'}
                                                        </div>
                                                    </div>

                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Slow Queries</div>
                                                        <div className="mt-1 text-sm font-semibold text-slate-800">
                                                            {dbHealth.database.query_metrics.slow_queries}
                                                        </div>
                                                        <div className="text-[11px] text-slate-500">
                                                            threshold: {dbHealth.database.query_metrics.slow_threshold_ms} ms
                                                        </div>
                                                    </div>

                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Queue Depth</div>
                                                        <div className="mt-1 text-sm font-semibold text-slate-800">
                                                            {dbHealth.queue_depth.total_active}
                                                        </div>
                                                        <div className="text-[11px] text-slate-500">
                                                            run {dbHealth.queue_depth.running} | q {dbHealth.queue_depth.queued} | p {dbHealth.queue_depth.paused}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-slate-500">Avg query</div>
                                                        <div className="font-medium text-slate-700">{dbHealth.database.query_metrics.avg_ms} ms</div>
                                                    </div>
                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-slate-500">P95 / P99</div>
                                                        <div className="font-medium text-slate-700">
                                                            {dbHealth.database.query_metrics.recent_p95_ms} / {dbHealth.database.query_metrics.recent_p99_ms} ms
                                                        </div>
                                                    </div>
                                                    <div className="rounded-md bg-white border border-slate-200 px-3 py-2">
                                                        <div className="text-slate-500">Pool checked out</div>
                                                        <div className="font-medium text-slate-700">
                                                            {dbHealth.database.pool.checked_out ?? '-'}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="text-[11px] text-slate-500">
                                                    {dbHealth.database.provider} · {dbHealth.database.database_url}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <h3 className="font-semibold text-slate-800 mb-4">Server</h3>
                                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                            <button
                                                type="button"
                                                onClick={() => void handleRestart(false)}
                                                disabled={restarting}
                                                className="min-h-[56px] px-4 py-2.5 rounded-lg text-slate-700 bg-white border border-slate-200 hover:bg-slate-100 font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2"
                                            >
                                                {restarting ? (
                                                    <><Loader2 size={16} className="animate-spin" /> Restarting...</>
                                                ) : (
                                                    <><Power size={16} /> Restart Backend</>
                                                )}
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => void handleRestart(true)}
                                                disabled={restarting}
                                                className="min-h-[56px] px-4 py-2.5 rounded-lg text-white bg-blue-600 hover:bg-blue-700 border border-blue-600 font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2"
                                            >
                                                {restarting ? (
                                                    <><Loader2 size={16} className="animate-spin" /> Restarting...</>
                                                ) : (
                                                    <><RefreshCw size={16} /> Restart + Reload UI</>
                                                )}
                                            </button>
                                        </div>
                                        <p className="text-xs text-slate-500 leading-relaxed">
                                            Restarts the backend process and interrupts active jobs. <span className="font-medium text-slate-600">Restart + Reload UI</span> also refreshes this app after the backend is back.
                                        </p>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                </div>

                {/* Save bar  -  always visible */}
                <div className="flex items-center gap-4">
                    <button
                        type="submit"
                        disabled={saving}
                        className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-white font-medium transition-colors ${saving ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                    >
                        {saving ? (
                            <>Saving...</>
                        ) : (
                            <>
                                <Save size={18} />
                                Save Changes
                            </>
                        )}
                    </button>

                    {status === 'success' && (
                        <div className="flex items-center gap-2 text-green-600 text-sm font-medium animate-in fade-in slide-in-from-left-2">
                            <CheckCircle2 size={18} />
                            Settings saved & models reloading...
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="flex items-center gap-2 text-red-600 text-sm font-medium animate-in fade-in slide-in-from-left-2">
                            <AlertCircle size={18} />
                            Failed to save settings
                        </div>
                    )}
                </div>
            </form>
        </div>
    );
}



