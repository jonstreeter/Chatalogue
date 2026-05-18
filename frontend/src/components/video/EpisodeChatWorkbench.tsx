import { useState, useEffect, useRef, useMemo } from 'react';
import api from '../../lib/api';
import type { 
  EpisodeChatCitation,
  EpisodeChatThread, 
  EpisodeChatThreadDetail, 
  EpisodeChatSendResponse,
  EpisodeCloneEngine
} from '../../types';
import { Loader2, Plus, MessageCircle, Send, Settings, Trash2 } from 'lucide-react';
import { EpisodeChatMessageItem } from './EpisodeChatMessageItem';

// Define the model for Ollama we pass in
type OllamaLocalModel = {
  name: string;
};

type Props = {
  videoId: number;
  cloneEngines: EpisodeCloneEngine[];
  cloneOllamaModels: OllamaLocalModel[];
  onCitationClick: (citation: EpisodeChatCitation) => void;
};

const formatTimestamp = (value?: string | null) => {
  if (!value) return 'No activity yet';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'No activity yet';
  return date.toLocaleString();
};

export function EpisodeChatWorkbench({ videoId, cloneEngines, cloneOllamaModels, onCitationClick }: Props) {
  const [threads, setThreads] = useState<EpisodeChatThread[]>([]);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [threadsError, setThreadsError] = useState<string | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<number | null>(null);
  
  const [activeThread, setActiveThread] = useState<EpisodeChatThreadDetail | null>(null);
  const [loadingActiveThread, setLoadingActiveThread] = useState(false);
  const [activeThreadError, setActiveThreadError] = useState<string | null>(null);
  
  const [composerText, setComposerText] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [creatingThread, setCreatingThread] = useState(false);
  const [deletingThreadId, setDeletingThreadId] = useState<number | null>(null);

  const [engineKey, setEngineKey] = useState('default');
  const [ollamaModel, setOllamaModel] = useState('');
  const [scopeMode, setScopeMode] = useState<'episode' | 'episode_related'>('episode');
  const [availableOllamaModels, setAvailableOllamaModels] = useState<OllamaLocalModel[]>(cloneOllamaModels);
  const [loadingOllamaModels, setLoadingOllamaModels] = useState(false);
  const [ollamaModelsError, setOllamaModelsError] = useState<string | null>(null);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const temporaryMessageIdRef = useRef(-1);
  const selectedEngine = useMemo(() => cloneEngines.find((engine) => engine.key === engineKey) || cloneEngines[0] || null, [cloneEngines, engineKey]);
  const chatUsesOllama = (selectedEngine?.provider || '').toLowerCase() === 'ollama';
  const scopeLabel = scopeMode === 'episode_related' ? 'Episode + Related' : 'Episode Only';

  const syncThreadEngineSelection = (thread: EpisodeChatThreadDetail | EpisodeChatThread | null) => {
    if (!thread) return;
    const provider = String(thread.provider || '').trim().toLowerCase();
    const model = String(thread.model || '').trim();
    const directMatch = cloneEngines.find((engine) => {
      return String(engine.provider || '').trim().toLowerCase() === provider && String(engine.model || '').trim() === model;
    });
    const providerMatch = directMatch || cloneEngines.find((engine) => String(engine.provider || '').trim().toLowerCase() === provider);
    setEngineKey(providerMatch?.key || 'default');
    setOllamaModel(provider === 'ollama' ? model : '');
    setScopeMode(thread.scope_mode === 'episode_related' ? 'episode_related' : 'episode');
  };

  const fetchThreads = async () => {
    setLoadingThreads(true);
    setThreadsError(null);
    try {
      const res = await api.get<EpisodeChatThread[]>(`/videos/${videoId}/episode-chat/threads`);
      setThreads(res.data);
      setActiveThreadId((current) => {
        if (current && res.data.some((thread) => thread.id === current)) return current;
        return res.data[0]?.id ?? null;
      });
      if (res.data.length === 0) {
        setActiveThread(null);
      }
    } catch (e: any) {
      console.error("Failed to fetch chat threads:", e);
      setThreadsError(e?.response?.data?.detail || 'Failed to load saved chats.');
    } finally {
      setLoadingThreads(false);
    }
  };

  const fetchThreadDetail = async (threadId: number) => {
    setLoadingActiveThread(true);
    setSendError(null);
    setActiveThreadError(null);
    try {
      const res = await api.get<EpisodeChatThreadDetail>(`/episode-chat/threads/${threadId}`);
      setActiveThread(res.data);
      syncThreadEngineSelection(res.data);
    } catch (e: any) {
      console.error("Failed to fetch thread detail:", e);
      setActiveThreadError(e?.response?.data?.detail || 'Failed to load this chat thread.');
    } finally {
      setLoadingActiveThread(false);
    }
  };

  const createThread = async () => {
    setCreatingThread(true);
    setThreadsError(null);
    try {
      const selected = cloneEngines.find((engine) => engine.key === engineKey) || cloneEngines[0] || null;
      const providerOverride = selected && selected.key !== 'default' ? selected.provider : undefined;
      const modelOverride = providerOverride === 'ollama'
        ? (ollamaModel || selected?.model || undefined)
        : (selected && selected.key !== 'default' ? selected.model : undefined);
      const payload = {
         title: 'New Chat',
         scope_mode: scopeMode,
         provider_override: providerOverride,
         model_override: modelOverride,
      };
      const res = await api.post<EpisodeChatThread>(`/videos/${videoId}/episode-chat/threads`, payload);
      setThreads((current) => [res.data, ...current.filter((thread) => thread.id !== res.data.id)]);
      setActiveThreadId(res.data.id);
      syncThreadEngineSelection(res.data);
    } catch (e: any) {
      console.error("Failed to create thread:", e);
      setThreadsError(e?.response?.data?.detail || 'Failed to create a new chat thread.');
    } finally {
      setCreatingThread(false);
    }
  };

  const deleteThread = async (id: number) => {
    setDeletingThreadId(id);
    try {
      await api.delete(`/episode-chat/threads/${id}`);
      const remaining = threads.filter((thread) => thread.id !== id);
      setThreads(remaining);
      if (activeThreadId === id) {
        setActiveThreadId(remaining[0]?.id ?? null);
        setActiveThread(null);
      }
    } catch (e: any) {
      console.error("Failed to delete thread:", e);
      setThreadsError(e?.response?.data?.detail || 'Failed to delete this chat thread.');
    } finally {
      setDeletingThreadId(null);
    }
  };

  const updateThreadScope = async (nextScope: 'episode' | 'episode_related') => {
    setScopeMode(nextScope);
    if (!activeThreadId) return;
    try {
      const res = await api.patch<EpisodeChatThread>(`/episode-chat/threads/${activeThreadId}`, {
        scope_mode: nextScope,
      });
      setThreads((current) => current.map((thread) => thread.id === res.data.id ? { ...thread, ...res.data } : thread));
      setActiveThread((current) => current ? { ...current, scope_mode: res.data.scope_mode } : current);
    } catch (e: any) {
      console.error("Failed to update chat scope:", e);
      setThreadsError(e?.response?.data?.detail || 'Failed to update chat scope.');
      if (activeThread) {
        setScopeMode(activeThread.scope_mode === 'episode_related' ? 'episode_related' : 'episode');
      }
    }
  };

  const sendMessage = async () => {
    if (!composerText.trim() || !activeThreadId) return;
    
    // Resolve provider overrides
    const selectedEngine = cloneEngines.find(e => e.key === engineKey);
    const resolvedProvider = (!selectedEngine || engineKey === 'default') ? undefined : selectedEngine.provider;
    const resolvedModel = resolvedProvider === 'ollama' ? ollamaModel : (selectedEngine?.model || undefined);
    
    setSending(true);
    setSendError(null);
    const text = composerText.trim();
    setComposerText('');
    
    const optimisticUserId = temporaryMessageIdRef.current--;
    if (activeThread) {
       setActiveThread({
         ...activeThread,
         messages: [
           ...activeThread.messages, 
           {
             id: optimisticUserId,
             thread_id: activeThreadId,
             role: 'user',
             content: text,
             status: 'completed',
             created_at: new Date().toISOString(),
           }
         ]
       });
       setTimeout(scrollToBottom, 50);
    }

    try {
      const res = await api.post<EpisodeChatSendResponse>(`/episode-chat/threads/${activeThreadId}/messages`, {
        message: text,
        max_context_chunks: 10,
        provider_override: resolvedProvider,
        model_override: resolvedModel,
      });
      
      // Merge results
      setActiveThread(current => {
        if (!current) return null;
        const filtered = current.messages.filter(m => m.id !== optimisticUserId);
        return {
          ...current,
          messages: [...filtered, res.data.user_message, res.data.assistant_message]
        };
      });
      setThreads((current) => current.map((thread) => thread.id === res.data.thread.id ? res.data.thread : thread));
      setTimeout(scrollToBottom, 50);
    } catch (e: any) {
      console.error("Failed to send message:", e);
      setSendError(e?.response?.data?.detail || "Failed to generate reply.");
      setComposerText(text); // Restore text
      void fetchThreadDetail(activeThreadId); // Rebuild true state
    } finally {
      setSending(false);
    }
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    void fetchThreads();
  }, [videoId]);

  useEffect(() => {
    if (activeThreadId) {
      void fetchThreadDetail(activeThreadId);
    }
  }, [activeThreadId]);

  useEffect(() => {
    setAvailableOllamaModels(cloneOllamaModels);
  }, [cloneOllamaModels]);

  useEffect(() => {
    if (activeThread && cloneEngines.length > 0) {
      syncThreadEngineSelection(activeThread);
    }
  }, [activeThread, cloneEngines]);

  useEffect(() => {
    if (!chatUsesOllama || availableOllamaModels.length > 0) return;
    const controller = new AbortController();
    const load = async () => {
      setLoadingOllamaModels(true);
      setOllamaModelsError(null);
      try {
        const res = await api.get<{ models: OllamaLocalModel[]; error?: string }>('/settings/ollama/models', { signal: controller.signal });
        if (Array.isArray(res.data?.models)) {
          setAvailableOllamaModels(res.data.models);
        }
        if (res.data?.error) {
          setOllamaModelsError(res.data.error);
        }
      } catch (e: any) {
        if (controller.signal.aborted) return;
        setOllamaModelsError(e?.response?.data?.detail || 'Failed to load local Ollama models.');
      } finally {
        if (!controller.signal.aborted) {
          setLoadingOllamaModels(false);
        }
      }
    };
    void load();
    return () => controller.abort();
  }, [chatUsesOllama, availableOllamaModels.length]);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeThread?.messages.length]);

  return (
    <div className="flex h-full bg-slate-50 overflow-hidden">
      {/* Left Rail: Thread Shelf */}
      <div className="w-72 border-r border-slate-200 bg-white flex flex-col flex-shrink-0">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <span className="font-semibold text-slate-800 flex items-center gap-2">
            <MessageCircle size={18} className="text-indigo-600" />
            Chat History
          </span>
          <button 
            onClick={createThread}
            disabled={creatingThread}
            className="rounded-xl border border-slate-200 bg-slate-50 p-2 text-slate-600 hover:text-indigo-600 hover:bg-white hover:border-indigo-200 transition"
            title="New Thread"
          >
            {creatingThread ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
          </button>
        </div>
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/80">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Thread model</div>
          <div className="mt-1 text-sm font-medium text-slate-800">{selectedEngine?.label || 'Configured default'}</div>
          {chatUsesOllama && (
            <div className="mt-1 text-xs text-slate-500">
              {ollamaModel || 'Default Ollama model'}
            </div>
          )}
          <div className="mt-1 text-xs text-slate-500">{scopeLabel}</div>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {loadingThreads ? (
             <div className="flex justify-center p-6 text-slate-400"><Loader2 size={18} className="animate-spin" /></div>
          ) : threads.length === 0 ? (
             <div className="text-sm text-center text-slate-500 py-6 border border-dashed border-slate-200 rounded-xl">No prior chats.</div>
          ) : (
            threads.map(thread => (
              <div 
                key={thread.id} 
                className={`group flex items-start justify-between rounded-xl p-3 cursor-pointer border transition ${activeThreadId === thread.id ? 'bg-indigo-50 border-indigo-200 shadow-sm' : 'bg-transparent border-transparent hover:bg-slate-50'}`}
                onClick={() => setActiveThreadId(thread.id)}
              >
                <div className="flex-1 overflow-hidden">
                   <div className={`text-sm font-medium truncate ${activeThreadId === thread.id ? 'text-indigo-900' : 'text-slate-700'}`}>
                     {thread.title || 'New Chat'}
                   </div>
                   <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-400">
                     <span>{new Date(thread.last_message_at || thread.created_at).toLocaleDateString()}</span>
                     <span>&middot;</span>
                     <span>{thread.message_count} msg{thread.message_count === 1 ? '' : 's'}</span>
                     <span className={`rounded-full px-1.5 py-0.5 ${thread.scope_mode === 'episode_related' ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>
                       {thread.scope_mode === 'episode_related' ? 'Episode + Related' : 'Episode'}
                     </span>
                   </div>
                </div>
                <button 
                  disabled={deletingThreadId === thread.id}
                  className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-rose-600 transition"
                  onClick={(e) => { e.stopPropagation(); deleteThread(thread.id); }}
                >
                   {deletingThreadId === thread.id ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                </button>
              </div>
            ))
          )}
          {threadsError && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {threadsError}
            </div>
          )}
        </div>
      </div>

      {/* Main Stage: Timeline + Composer */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        {!activeThreadId ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-4">
             <MessageCircle size={48} className="text-slate-300 opacity-50" />
             <div>Select a thread or create a new one to ask questions.</div>
             <button onClick={createThread} className="rounded-xl px-4 py-2 bg-indigo-600 text-white font-medium hover:bg-indigo-700">Start New Chat</button>
          </div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto p-6" ref={scrollRef}>
               {loadingActiveThread ? (
                 <div className="flex justify-center py-12 text-slate-400"><Loader2 size={24} className="animate-spin" /></div>
               ) : activeThreadError ? (
                 <div className="mx-auto max-w-2xl rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4 text-sm text-rose-700">
                   {activeThreadError}
                 </div>
               ) : activeThread?.messages.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto opacity-70">
                    <MessageCircle size={32} className="text-slate-300 mb-4" />
                    <p className="text-sm text-slate-500">Ask a question about this episode. The AI has access to the full transcript and will cite the exact moments it references.</p>
                 </div>
               ) : (
                 <div className="max-w-4xl mx-auto flex flex-col w-full">
                   {activeThread && (
                     <div className="mb-5 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm">
                       <div className="flex flex-wrap items-center justify-between gap-3">
                         <div>
                           <div className="text-sm font-semibold text-slate-900">{activeThread.title || 'New Chat'}</div>
                           <div className="mt-1 text-xs text-slate-500">
                             {activeThread.message_count} message{activeThread.message_count === 1 ? '' : 's'} • {formatTimestamp(activeThread.last_message_at || activeThread.updated_at)}
                           </div>
                         </div>
                         <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                           <span className={`rounded-full border px-2.5 py-1 ${activeThread.scope_mode === 'episode_related' ? 'border-amber-200 bg-amber-50 text-amber-700' : 'border-slate-200 bg-slate-50'}`}>
                             {activeThread.scope_mode === 'episode_related' ? 'Episode + Related' : 'Episode Only'}
                           </span>
                           <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">{activeThread.provider || 'default'}</span>
                           <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">{activeThread.model || 'configured default model'}</span>
                         </div>
                       </div>
                     </div>
                   )}
                   {activeThread?.messages.map((msg) => (
                     <EpisodeChatMessageItem 
                       key={msg.id} 
                       message={msg} 
                       onCitationClick={onCitationClick} 
                     />
                   ))}
                   {sendError && (
                     <div className="p-4 rounded-xl border border-rose-200 bg-rose-50 text-rose-700 text-sm mb-6 max-w-4xl mx-auto w-full">
                       {sendError}
                     </div>
                   )}
                 </div>
               )}
            </div>
            
            {/* Composer */}
            <div className="w-full bg-white border-t border-slate-200 p-4">
               <div className="max-w-4xl mx-auto flex flex-col gap-3">
                 {/* Generation Params Row */}
                 <div className="flex items-center gap-3 w-full">
                   <Settings size={14} className="text-slate-400" />
                   <select
                     value={scopeMode}
                     onChange={(e) => void updateThreadScope(e.target.value === 'episode_related' ? 'episode_related' : 'episode')}
                     className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 outline-none focus:border-indigo-300 text-slate-600"
                   >
                     <option value="episode">Episode Only</option>
                     <option value="episode_related">Episode + Related</option>
                   </select>
                   <select 
                     value={engineKey} 
                     onChange={(e) => setEngineKey(e.target.value)} 
                     className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 outline-none focus:border-indigo-300 text-slate-600"
                   >
                     {cloneEngines.map(e => <option key={e.key} value={e.key} disabled={!e.available}>{e.label}</option>)}
                   </select>

                   {chatUsesOllama && (
                     <select 
                       value={ollamaModel} 
                       onChange={(e) => setOllamaModel(e.target.value)} 
                       disabled={loadingOllamaModels}
                       className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 outline-none focus:border-indigo-300 text-slate-600 max-w-[200px] truncate"
                     >
                       <option value="">{loadingOllamaModels ? 'Loading Ollama models...' : 'Default Ollama'}</option>
                       {availableOllamaModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                     </select>
                   )}
                 </div>
                 
                 {/* Input Box */}
                 <div className="flex items-end gap-3 w-full border border-slate-300 rounded-2xl bg-white p-2 shadow-sm focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-100 transition">
                    <textarea 
                      value={composerText}
                      onChange={(e) => setComposerText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                           e.preventDefault();
                           sendMessage();
                        }
                      }}
                      disabled={sending}
                      placeholder="Ask about this episode... (Shift+Enter for new line)"
                      className="flex-1 max-h-32 min-h-11 bg-transparent border-none resize-none p-2 outline-none text-sm text-slate-800 disabled:opacity-50"
                      rows={1}
                    />
                    <button 
                      onClick={sendMessage}
                      disabled={sending || !composerText.trim()}
                      className="mb-1 mr-1 flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:bg-slate-300 disabled:text-slate-500 transition shadow-sm"
                    >
                      {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                    </button>
                 </div>
                 <div className="flex items-center justify-between gap-3 px-2 text-[11px] text-slate-500">
                   <div>
                     {scopeMode === 'episode_related'
                       ? 'Current episode stays primary. Related citations open the source episode.'
                       : 'Transcript-grounded replies only. Citations jump back into the episode.'}
                   </div>
                   <div>{selectedEngine?.label || 'Configured default model'}</div>
                 </div>
                 {ollamaModelsError && (
                   <div className="px-2 text-[11px] text-rose-600">{ollamaModelsError}</div>
                 )}
               </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
