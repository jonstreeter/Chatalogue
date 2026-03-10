import { useState, useEffect } from 'react';
import { X, Eye, EyeOff, CheckCircle2, AlertTriangle, Loader2, ExternalLink, ChevronRight, ChevronLeft, Cpu, Zap, Globe, Server } from 'lucide-react';
import api from '../lib/api';

type WizardStep = 'welcome' | 'hf_token' | 'transcription' | 'llm' | 'complete';

const STEPS: WizardStep[] = ['welcome', 'hf_token', 'transcription', 'llm', 'complete'];
const STEP_LABELS: Record<WizardStep, string> = {
  welcome: 'Welcome',
  hf_token: 'Diarization',
  transcription: 'Transcription',
  llm: 'LLM',
  complete: 'Done',
};

type LlmProvider = 'ollama' | 'nvidia_nim' | 'openai' | 'anthropic' | 'gemini' | 'groq' | 'openrouter' | 'xai';

const PROVIDER_INFO: Record<LlmProvider, { label: string; icon: typeof Server; local: boolean; keyUrl: string }> = {
  ollama: { label: 'Ollama', icon: Server, local: true, keyUrl: 'https://ollama.com/download' },
  nvidia_nim: { label: 'NVIDIA NIM', icon: Zap, local: false, keyUrl: 'https://build.nvidia.com/' },
  openai: { label: 'OpenAI', icon: Globe, local: false, keyUrl: 'https://platform.openai.com/api-keys' },
  anthropic: { label: 'Anthropic', icon: Globe, local: false, keyUrl: 'https://console.anthropic.com/settings/keys' },
  gemini: { label: 'Google Gemini', icon: Globe, local: false, keyUrl: 'https://aistudio.google.com/apikey' },
  groq: { label: 'Groq', icon: Zap, local: false, keyUrl: 'https://console.groq.com/keys' },
  openrouter: { label: 'OpenRouter', icon: Globe, local: false, keyUrl: 'https://openrouter.ai/keys' },
  xai: { label: 'xAI (Grok)', icon: Globe, local: false, keyUrl: 'https://console.x.ai/' },
};

interface Settings {
  hf_token: string;
  transcription_engine: string;
  transcription_model: string;
  transcription_compute_type: string;
  parakeet_model: string;
  parakeet_batch_size: number;
  parakeet_batch_auto: boolean;
  parakeet_require_word_timestamps: boolean;
  parakeet_unload_after_transcribe: boolean;
  beam_size: number;
  vad_filter: boolean;
  batched_transcription: boolean;
  verbose_logging: boolean;
  llm_provider: string;
  llm_enabled: boolean;
  ollama_url: string;
  ollama_model: string;
  ollama_model_tier: string;
  ollama_enabled: boolean;
  nvidia_nim_base_url: string;
  nvidia_nim_model: string;
  nvidia_nim_api_key: string;
  nvidia_nim_thinking_mode: boolean;
  nvidia_nim_min_request_interval_seconds: number;
  openai_base_url: string;
  openai_model: string;
  openai_api_key: string;
  anthropic_base_url: string;
  anthropic_model: string;
  anthropic_api_key: string;
  gemini_base_url: string;
  gemini_model: string;
  gemini_api_key: string;
  groq_base_url: string;
  groq_model: string;
  groq_api_key: string;
  openrouter_base_url: string;
  openrouter_model: string;
  openrouter_api_key: string;
  xai_base_url: string;
  xai_model: string;
  xai_api_key: string;
  youtube_oauth_client_id: string;
  youtube_oauth_client_secret: string;
  youtube_oauth_redirect_uri: string;
  youtube_publish_push_enabled: boolean;
  ytdlp_cookies_file: string;
  ytdlp_cookies_from_browser: string;
  diarization_sensitivity: string;
  speaker_match_threshold: number;
  funny_moments_max_saved: number;
  funny_moments_explain_batch_limit: number;
  setup_wizard_completed: boolean;
}

interface Props {
  onClose: () => void;
  onComplete: () => void;
}

export function SetupWizard({ onClose, onComplete }: Props) {
  const [step, setStep] = useState<WizardStep>('welcome');
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);

  // Step-specific state
  const [gpuInfo, setGpuInfo] = useState<{ gpu_name?: string; vram_gb?: number } | null>(null);
  const [hfToken, setHfToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [tokenValidation, setTokenValidation] = useState<{ valid?: boolean; error?: string; models?: Record<string, any> } | null>(null);
  const [validating, setValidating] = useState(false);
  const [hfSkipped, setHfSkipped] = useState(false);

  const [engine, setEngine] = useState<'auto' | 'whisper' | 'parakeet'>('auto');
  const [whisperModel, setWhisperModel] = useState('medium');
  const [engineTest, setEngineTest] = useState<{ success?: boolean; error?: string; device?: string } | null>(null);
  const [testingEngine, setTestingEngine] = useState(false);

  const [llmProvider, setLlmProvider] = useState<LlmProvider>('ollama');
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [llmSkipped, setLlmSkipped] = useState(false);
  const [llmKey, setLlmKey] = useState('');
  const [llmModel, setLlmModel] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [llmTest, setLlmTest] = useState<{ success?: boolean; error?: string; model?: string } | null>(null);
  const [testingLlm, setTestingLlm] = useState(false);

  const [saving, setSaving] = useState(false);

  useEffect(() => {
    Promise.all([
      api.get<Settings>('/settings').then(r => r.data),
      api.get('/settings/ollama/hardware-recommendation').then(r => r.data).catch(() => null),
    ]).then(([s, hw]) => {
      setSettings(s);
      setHfToken(s.hf_token || '');
      setEngine((s.transcription_engine || 'auto') as any);
      setWhisperModel(s.transcription_model || 'medium');
      setLlmProvider((s.llm_provider || 'ollama') as LlmProvider);
      setLlmEnabled(s.llm_enabled);
      setOllamaUrl(s.ollama_url || 'http://localhost:11434');
      if (hw?.hardware) {
        setGpuInfo({
          gpu_name: hw.hardware.gpu_name,
          vram_gb: hw.hardware.vram_total_gb,
        });
      }
    }).finally(() => setLoading(false));
  }, []);

  const currentIndex = STEPS.indexOf(step);

  const saveSettings = async (overrides: Partial<Settings> = {}) => {
    if (!settings) return;
    const merged = { ...settings, ...overrides };
    await api.post('/settings', merged);
    setSettings(merged);
  };

  const goNext = () => {
    const next = STEPS[currentIndex + 1];
    if (next) setStep(next);
  };
  const goBack = () => {
    const prev = STEPS[currentIndex - 1];
    if (prev) setStep(prev);
  };

  const handleValidateToken = async () => {
    if (!hfToken.trim()) return;
    setValidating(true);
    setTokenValidation(null);
    try {
      await saveSettings({ hf_token: hfToken });
      const res = await api.post('/settings/validate-token');
      setTokenValidation(res.data);
    } catch (e: any) {
      setTokenValidation({ valid: false, error: e.response?.data?.detail || 'Validation failed' });
    } finally {
      setValidating(false);
    }
  };

  const handleTestEngine = async () => {
    setTestingEngine(true);
    setEngineTest(null);
    try {
      await saveSettings({
        transcription_engine: engine,
        transcription_model: whisperModel,
      });
      const res = await api.post('/settings/test-transcription-engine', { engine });
      setEngineTest({
        success: res.data.available,
        device: res.data.device,
        error: res.data.available ? undefined : res.data.error || 'Engine not available',
      });
    } catch (e: any) {
      setEngineTest({ success: false, error: e.response?.data?.detail || 'Test failed' });
    } finally {
      setTestingEngine(false);
    }
  };

  const getProviderKeyField = (p: LlmProvider): string => {
    const map: Record<string, string> = {
      nvidia_nim: 'nvidia_nim_api_key',
      openai: 'openai_api_key',
      anthropic: 'anthropic_api_key',
      gemini: 'gemini_api_key',
      groq: 'groq_api_key',
      openrouter: 'openrouter_api_key',
      xai: 'xai_api_key',
    };
    return map[p] || '';
  };

  const getProviderModelField = (p: LlmProvider): string => {
    const map: Record<string, string> = {
      ollama: 'ollama_model',
      nvidia_nim: 'nvidia_nim_model',
      openai: 'openai_model',
      anthropic: 'anthropic_model',
      gemini: 'gemini_model',
      groq: 'groq_model',
      openrouter: 'openrouter_model',
      xai: 'xai_model',
    };
    return map[p] || '';
  };

  const getDefaultModel = (p: LlmProvider): string => {
    const map: Record<string, string> = {
      ollama: 'mistral',
      nvidia_nim: 'moonshotai/kimi-k2.5',
      openai: 'gpt-4o-mini',
      anthropic: 'claude-3-5-sonnet-latest',
      gemini: 'gemini-2.5-flash',
      groq: 'llama-3.3-70b-versatile',
      openrouter: 'openai/gpt-4o-mini',
      xai: 'grok-2',
    };
    return map[p] || '';
  };

  const handleProviderSelect = (p: LlmProvider) => {
    setLlmProvider(p);
    setLlmTest(null);
    setLlmKey(settings ? (settings as any)[getProviderKeyField(p)] || '' : '');
    setLlmModel(settings ? (settings as any)[getProviderModelField(p)] || getDefaultModel(p) : getDefaultModel(p));
    if (p === 'ollama') {
      setOllamaUrl(settings?.ollama_url || 'http://localhost:11434');
    }
  };

  const handleTestLlm = async () => {
    setTestingLlm(true);
    setLlmTest(null);
    try {
      const overrides: Partial<Settings> = {
        llm_provider: llmProvider,
        llm_enabled: true,
      };
      if (llmProvider === 'ollama') {
        overrides.ollama_url = ollamaUrl;
        overrides.ollama_model = llmModel;
        overrides.ollama_enabled = true;
      } else {
        const keyField = getProviderKeyField(llmProvider);
        const modelField = getProviderModelField(llmProvider);
        if (keyField) (overrides as any)[keyField] = llmKey;
        if (modelField) (overrides as any)[modelField] = llmModel;
      }
      await saveSettings(overrides);

      let res;
      if (llmProvider === 'ollama') {
        res = await api.post('/settings/test-ollama');
      } else if (llmProvider === 'nvidia_nim') {
        res = await api.post('/settings/test-nvidia-nim');
      } else {
        res = await api.post('/settings/test-hosted-llm');
      }
      const d = res.data;
      setLlmTest({
        success: d.success || d.status === 'ok',
        model: d.model || d.model_used,
        error: d.success === false ? (d.error || 'Connection failed') : undefined,
      });
    } catch (e: any) {
      setLlmTest({ success: false, error: e.response?.data?.detail || 'Test failed' });
    } finally {
      setTestingLlm(false);
    }
  };

  const handleFinish = async () => {
    setSaving(true);
    try {
      const overrides: Partial<Settings> = {
        hf_token: hfSkipped ? '' : hfToken,
        transcription_engine: engine,
        transcription_model: whisperModel,
        llm_provider: llmProvider,
        llm_enabled: llmEnabled && !llmSkipped,
        setup_wizard_completed: true,
      };
      if (llmEnabled && !llmSkipped) {
        if (llmProvider === 'ollama') {
          overrides.ollama_url = ollamaUrl;
          overrides.ollama_model = llmModel;
          overrides.ollama_enabled = true;
        } else {
          const keyField = getProviderKeyField(llmProvider);
          const modelField = getProviderModelField(llmProvider);
          if (keyField) (overrides as any)[keyField] = llmKey;
          if (modelField) (overrides as any)[modelField] = llmModel;
        }
      }
      await saveSettings(overrides);
      onComplete();
    } catch {
      // Still close on error — settings were partially saved
      onComplete();
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <Loader2 className="w-8 h-8 animate-spin text-rose-500 mx-auto" />
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header with progress */}
        <div className="px-6 pt-5 pb-4 border-b border-slate-100">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-slate-900">Setup Wizard</h2>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">
              <X size={20} />
            </button>
          </div>
          <div className="flex items-center gap-1">
            {STEPS.map((s, i) => (
              <div key={s} className="flex items-center flex-1">
                <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-semibold transition-colors ${
                  i < currentIndex ? 'bg-green-500 text-white' :
                  i === currentIndex ? 'bg-rose-500 text-white' :
                  'bg-slate-200 text-slate-500'
                }`}>
                  {i < currentIndex ? <CheckCircle2 size={16} /> : i + 1}
                </div>
                <span className={`ml-1.5 text-xs font-medium hidden sm:inline ${
                  i === currentIndex ? 'text-rose-600' : 'text-slate-400'
                }`}>{STEP_LABELS[s]}</span>
                {i < STEPS.length - 1 && (
                  <div className={`flex-1 h-0.5 mx-2 rounded ${i < currentIndex ? 'bg-green-300' : 'bg-slate-200'}`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {/* WELCOME */}
          {step === 'welcome' && (
            <div className="text-center py-6">
              <div className="text-5xl mb-4">&#127925;</div>
              <h3 className="text-2xl font-bold text-slate-900 mb-2">Welcome to Chatalogue</h3>
              <p className="text-slate-600 mb-6 max-w-md mx-auto">
                Podcast transcription & speaker diarization made easy.
                Let's get your system configured in a few quick steps.
              </p>
              {gpuInfo?.gpu_name && (
                <div className="inline-flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg px-4 py-2 text-sm text-green-800">
                  <Cpu size={16} />
                  <span>Detected: <strong>{gpuInfo.gpu_name}</strong></span>
                  {gpuInfo.vram_gb != null && <span>({gpuInfo.vram_gb.toFixed(1)} GB VRAM)</span>}
                </div>
              )}
              {gpuInfo === null && !loading && (
                <div className="inline-flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 text-sm text-amber-800">
                  <Cpu size={16} />
                  <span>No GPU detected &mdash; CPU mode will be used</span>
                </div>
              )}
            </div>
          )}

          {/* HF TOKEN */}
          {step === 'hf_token' && (
            <div>
              <h3 className="text-lg font-bold text-slate-900 mb-1">Speaker Diarization</h3>
              <p className="text-sm text-slate-600 mb-4">
                Diarization identifies <em>who</em> is speaking. It requires a free HuggingFace token
                and accepting two model agreements.
              </p>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4 text-sm text-blue-800 space-y-1">
                <p className="font-semibold">Three steps to enable diarization:</p>
                <ol className="list-decimal ml-5 space-y-1">
                  <li>
                    <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="underline inline-flex items-center gap-1">
                      Create a HuggingFace token <ExternalLink size={12} />
                    </a>
                  </li>
                  <li>
                    Accept the{' '}
                    <a href="https://huggingface.co/pyannote/speaker-diarization-3.1" target="_blank" rel="noopener noreferrer" className="underline inline-flex items-center gap-1">
                      speaker-diarization-3.1 <ExternalLink size={12} />
                    </a>{' '}
                    model agreement
                  </li>
                  <li>
                    Accept the{' '}
                    <a href="https://huggingface.co/pyannote/embedding" target="_blank" rel="noopener noreferrer" className="underline inline-flex items-center gap-1">
                      embedding <ExternalLink size={12} />
                    </a>{' '}
                    model agreement
                  </li>
                </ol>
              </div>

              <label className="block text-sm font-medium text-slate-700 mb-1">HuggingFace Token</label>
              <div className="relative mb-3">
                <input
                  type={showToken ? 'text' : 'password'}
                  value={hfToken}
                  onChange={e => { setHfToken(e.target.value); setTokenValidation(null); }}
                  placeholder="hf_..."
                  className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={() => setShowToken(!showToken)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showToken ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>

              <button
                onClick={handleValidateToken}
                disabled={!hfToken.trim() || validating}
                className="px-4 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {validating ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                Validate Token
              </button>

              {tokenValidation && (
                <div className={`mt-3 p-3 rounded-lg text-sm ${tokenValidation.valid ? 'bg-green-50 border border-green-200 text-green-800' : 'bg-red-50 border border-red-200 text-red-800'}`}>
                  {tokenValidation.valid ? (
                    <div className="flex items-center gap-2">
                      <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
                      <span>Token is valid! Model access confirmed.</span>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2">
                      <AlertTriangle size={16} className="text-red-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="font-medium">Validation failed.</span>
                        {tokenValidation.error && <p className="mt-1">{tokenValidation.error}</p>}
                        {tokenValidation.models && Object.entries(tokenValidation.models).map(([model, info]: [string, any]) => (
                          info.accessible === false && (
                            <p key={model} className="mt-1">
                              <strong>{model}</strong>: {info.error || 'Access denied — accept the model agreement'}
                            </p>
                          )
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
                <div className="flex items-start gap-2">
                  <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
                  <span>You can skip this step, but all speech will be attributed to a single speaker.</span>
                </div>
              </div>
            </div>
          )}

          {/* TRANSCRIPTION */}
          {step === 'transcription' && (
            <div>
              <h3 className="text-lg font-bold text-slate-900 mb-1">Transcription Engine</h3>
              <p className="text-sm text-slate-600 mb-4">
                Choose how audio is converted to text. No API keys needed &mdash; all engines run locally.
              </p>

              <div className="space-y-2 mb-4">
                {([
                  { value: 'auto' as const, label: 'Auto (Recommended)', desc: 'Automatically picks the best available engine' },
                  { value: 'whisper' as const, label: 'Whisper', desc: 'OpenAI Whisper — reliable, well-tested, supports many languages' },
                  { value: 'parakeet' as const, label: 'Parakeet', desc: 'NVIDIA NeMo — fast with word-level timestamps, English-focused' },
                ]).map(opt => (
                  <label
                    key={opt.value}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      engine === opt.value ? 'border-rose-300 bg-rose-50' : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <input
                      type="radio"
                      name="engine"
                      value={opt.value}
                      checked={engine === opt.value}
                      onChange={() => { setEngine(opt.value); setEngineTest(null); }}
                      className="mt-0.5 accent-rose-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-slate-900">{opt.label}</div>
                      <div className="text-xs text-slate-500">{opt.desc}</div>
                    </div>
                  </label>
                ))}
              </div>

              {(engine === 'whisper' || engine === 'auto') && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-slate-700 mb-1">Whisper Model Size</label>
                  <select
                    value={whisperModel}
                    onChange={e => { setWhisperModel(e.target.value); setEngineTest(null); }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
                  >
                    <option value="tiny">tiny (~1 GB VRAM)</option>
                    <option value="base">base (~1 GB VRAM)</option>
                    <option value="small">small (~2 GB VRAM)</option>
                    <option value="medium">medium (~5 GB VRAM) — recommended</option>
                    <option value="large-v3">large-v3 (~10 GB VRAM) — most accurate</option>
                  </select>
                </div>
              )}

              <button
                onClick={handleTestEngine}
                disabled={testingEngine}
                className="px-4 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {testingEngine ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
                Test Engine
              </button>

              {engineTest && (
                <div className={`mt-3 p-3 rounded-lg text-sm ${engineTest.success ? 'bg-green-50 border border-green-200 text-green-800' : 'bg-red-50 border border-red-200 text-red-800'}`}>
                  {engineTest.success ? (
                    <div className="flex items-center gap-2">
                      <CheckCircle2 size={16} className="text-green-600" />
                      <span>Engine ready! Device: <strong>{engineTest.device || 'detected'}</strong></span>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2">
                      <AlertTriangle size={16} className="text-red-600 mt-0.5" />
                      <span>{engineTest.error}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* LLM */}
          {step === 'llm' && (
            <div>
              <h3 className="text-lg font-bold text-slate-900 mb-1">LLM Provider</h3>
              <p className="text-sm text-slate-600 mb-4">
                An LLM powers humor detection explanations, video summaries, and AI chapters.
                This is optional &mdash; core transcription works without it.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
                {(Object.entries(PROVIDER_INFO) as [LlmProvider, typeof PROVIDER_INFO[LlmProvider]][]).map(([key, info]) => {
                  const Icon = info.icon;
                  return (
                    <button
                      key={key}
                      onClick={() => handleProviderSelect(key)}
                      className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border text-center transition-colors ${
                        llmProvider === key && llmEnabled ? 'border-rose-300 bg-rose-50' : 'border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      <Icon size={20} className={llmProvider === key && llmEnabled ? 'text-rose-600' : 'text-slate-400'} />
                      <span className="text-xs font-medium text-slate-700">{info.label}</span>
                      {info.local && <span className="text-[10px] text-green-600 font-medium">LOCAL/FREE</span>}
                    </button>
                  );
                })}
              </div>

              {!llmEnabled && (
                <button
                  onClick={() => { setLlmEnabled(true); setLlmSkipped(false); handleProviderSelect(llmProvider); }}
                  className="w-full mb-4 px-4 py-2 text-sm font-medium text-rose-600 border border-rose-300 rounded-lg hover:bg-rose-50 transition-colors"
                >
                  Enable LLM Features
                </button>
              )}

              {llmEnabled && (
                <div className="space-y-3">
                  {llmProvider === 'ollama' ? (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Ollama URL</label>
                        <input
                          type="text"
                          value={ollamaUrl}
                          onChange={e => { setOllamaUrl(e.target.value); setLlmTest(null); }}
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
                        <input
                          type="text"
                          value={llmModel}
                          onChange={e => { setLlmModel(e.target.value); setLlmTest(null); }}
                          placeholder="mistral"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">
                          API Key
                          <a href={PROVIDER_INFO[llmProvider].keyUrl} target="_blank" rel="noopener noreferrer" className="ml-2 text-rose-500 text-xs inline-flex items-center gap-0.5 hover:underline">
                            Get key <ExternalLink size={10} />
                          </a>
                        </label>
                        <input
                          type="password"
                          value={llmKey}
                          onChange={e => { setLlmKey(e.target.value); setLlmTest(null); }}
                          placeholder="sk-..."
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
                        <input
                          type="text"
                          value={llmModel}
                          onChange={e => { setLlmModel(e.target.value); setLlmTest(null); }}
                          placeholder={getDefaultModel(llmProvider)}
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
                        />
                      </div>
                    </>
                  )}

                  <button
                    onClick={handleTestLlm}
                    disabled={testingLlm}
                    className="px-4 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  >
                    {testingLlm ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
                    Test Connection
                  </button>

                  {llmTest && (
                    <div className={`p-3 rounded-lg text-sm ${llmTest.success ? 'bg-green-50 border border-green-200 text-green-800' : 'bg-red-50 border border-red-200 text-red-800'}`}>
                      {llmTest.success ? (
                        <div className="flex items-center gap-2">
                          <CheckCircle2 size={16} className="text-green-600" />
                          <span>Connected! {llmTest.model && <>Model: <strong>{llmTest.model}</strong></>}</span>
                        </div>
                      ) : (
                        <div className="flex items-start gap-2">
                          <AlertTriangle size={16} className="text-red-600 mt-0.5" />
                          <span>{llmTest.error}</span>
                        </div>
                      )}
                    </div>
                  )}

                  <button
                    onClick={() => { setLlmEnabled(false); setLlmSkipped(true); setLlmTest(null); }}
                    className="text-xs text-slate-500 hover:text-slate-700 underline"
                  >
                    Disable LLM features
                  </button>
                </div>
              )}

              {!llmEnabled && (
                <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-600">
                  LLM features are disabled. You can enable them later from the Settings page.
                </div>
              )}
            </div>
          )}

          {/* COMPLETE */}
          {step === 'complete' && (
            <div className="text-center py-4">
              <div className="text-5xl mb-4">&#9989;</div>
              <h3 className="text-2xl font-bold text-slate-900 mb-2">You're all set!</h3>
              <p className="text-slate-600 mb-6">Here's a summary of your configuration:</p>

              <div className="text-left max-w-sm mx-auto space-y-3 mb-6">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-50">
                  {hfToken && !hfSkipped ? (
                    <CheckCircle2 size={20} className="text-green-600 flex-shrink-0" />
                  ) : (
                    <AlertTriangle size={20} className="text-amber-500 flex-shrink-0" />
                  )}
                  <div>
                    <div className="text-sm font-medium text-slate-900">Speaker Diarization</div>
                    <div className="text-xs text-slate-500">
                      {hfToken && !hfSkipped ? 'HuggingFace token configured' : 'Skipped — single speaker mode'}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-50">
                  <CheckCircle2 size={20} className="text-green-600 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium text-slate-900">Transcription</div>
                    <div className="text-xs text-slate-500">
                      Engine: {engine}{engine === 'whisper' || engine === 'auto' ? ` / Model: ${whisperModel}` : ''}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-50">
                  {llmEnabled && !llmSkipped ? (
                    <CheckCircle2 size={20} className="text-green-600 flex-shrink-0" />
                  ) : (
                    <span className="w-5 h-5 flex items-center justify-center text-slate-400 flex-shrink-0">&mdash;</span>
                  )}
                  <div>
                    <div className="text-sm font-medium text-slate-900">LLM Provider</div>
                    <div className="text-xs text-slate-500">
                      {llmEnabled && !llmSkipped ? PROVIDER_INFO[llmProvider].label : 'Skipped'}
                    </div>
                  </div>
                </div>
              </div>

              <p className="text-xs text-slate-500 mb-4">
                You can change these settings anytime from the Settings page.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between">
          <div>
            {currentIndex > 0 && step !== 'complete' && (
              <button
                onClick={goBack}
                className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors flex items-center gap-1"
              >
                <ChevronLeft size={16} /> Back
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {step === 'hf_token' && (
              <button
                onClick={() => { setHfSkipped(true); goNext(); }}
                className="px-4 py-2 text-sm font-medium text-slate-500 hover:text-slate-700 transition-colors"
              >
                Skip
              </button>
            )}
            {step === 'llm' && (
              <button
                onClick={() => { setLlmSkipped(true); setLlmEnabled(false); goNext(); }}
                className="px-4 py-2 text-sm font-medium text-slate-500 hover:text-slate-700 transition-colors"
              >
                Skip
              </button>
            )}

            {step === 'welcome' && (
              <button
                onClick={goNext}
                className="px-5 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 transition-colors flex items-center gap-1"
              >
                Get Started <ChevronRight size={16} />
              </button>
            )}
            {step === 'hf_token' && (
              <button
                onClick={() => { setHfSkipped(false); goNext(); }}
                className="px-5 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 transition-colors flex items-center gap-1"
              >
                Next <ChevronRight size={16} />
              </button>
            )}
            {step === 'transcription' && (
              <button
                onClick={goNext}
                className="px-5 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 transition-colors flex items-center gap-1"
              >
                Next <ChevronRight size={16} />
              </button>
            )}
            {step === 'llm' && (
              <button
                onClick={() => { setLlmSkipped(false); goNext(); }}
                className="px-5 py-2 text-sm font-medium text-white bg-rose-600 rounded-lg hover:bg-rose-700 transition-colors flex items-center gap-1"
              >
                Next <ChevronRight size={16} />
              </button>
            )}
            {step === 'complete' && (
              <button
                onClick={handleFinish}
                disabled={saving}
                className="px-6 py-2.5 text-sm font-semibold text-white bg-rose-600 rounded-lg hover:bg-rose-700 disabled:opacity-50 transition-colors flex items-center gap-2"
              >
                {saving ? <Loader2 size={16} className="animate-spin" /> : null}
                Open Chatalogue
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
