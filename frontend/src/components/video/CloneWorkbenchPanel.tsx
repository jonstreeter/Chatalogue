import type { EpisodeCloneConceptsResponse, EpisodeCloneEngine, EpisodeCloneGenerateResponse, EpisodeCloneJob, Video } from '../../types';
import { Bot, Clapperboard, Copy, Loader2, Radar, RotateCcw, ShieldBan, Sparkles } from 'lucide-react';

type Props = {
  video: Video | null;
  segmentsCount: number;
  cloneEngineKey: string;
  onCloneEngineKeyChange: (value: string) => void;
  cloneEngines: EpisodeCloneEngine[];
  loadingCloneEngines: boolean;
  cloneEnginesError: string | null;
  cloneUsesOllama: boolean;
  cloneOllamaModel: string;
  onCloneOllamaModelChange: (value: string) => void;
  cloneOllamaModels: Array<{ name: string; size_bytes?: number; parameter_size?: string; quantization_level?: string }>;
  loadingCloneOllamaModels: boolean;
  cloneOllamaModelsError: string | null;
  cloneNotes: string;
  onCloneNotesChange: (value: string) => void;
  detectingCloneConcepts: boolean;
  onDetectConcepts: () => void;
  cloneConcepts: EpisodeCloneConceptsResponse | null;
  cloneConceptsText: string;
  onCloneConceptsTextChange: (value: string) => void;
  cloneExcludedReferencesText: string;
  onCloneExcludedReferencesTextChange: (value: string) => void;
  cloneStylePrompt: string;
  onCloneStylePromptChange: (value: string) => void;
  cloneBatchSize: number;
  onCloneBatchSizeChange: (value: number) => void;
  generatingClone: boolean;
  onGenerate: () => void;
  cloneJobs: EpisodeCloneJob[];
  loadingCloneJobs: boolean;
  cloneJobsError: string | null;
  selectedCloneJobId: number | null;
  onSelectCloneJob: (jobId: number) => void;
  onLoadCloneVariantInputs: (job: EpisodeCloneJob) => void;
  cloneDraft: EpisodeCloneGenerateResponse | null;
  copiedCloneScript: boolean;
  onCopyCloneScript: () => void;
  cloneJobMatchesVisibleInputs: (job: EpisodeCloneJob | null | undefined) => boolean;
  formatViewMetric: (value?: number | null, fractionDigits?: number) => string;
  formatTime: (seconds: number) => string;
};

const splitLines = (value: string) => value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);

export function CloneWorkbenchPanel({
  video, segmentsCount, cloneEngineKey, onCloneEngineKeyChange, cloneEngines, loadingCloneEngines, cloneEnginesError,
  cloneUsesOllama, cloneOllamaModel, onCloneOllamaModelChange, cloneOllamaModels, loadingCloneOllamaModels, cloneOllamaModelsError,
  cloneNotes, onCloneNotesChange, detectingCloneConcepts, onDetectConcepts, cloneConcepts, cloneConceptsText,
  onCloneConceptsTextChange, cloneExcludedReferencesText, onCloneExcludedReferencesTextChange, cloneStylePrompt,
  onCloneStylePromptChange, cloneBatchSize, onCloneBatchSizeChange, generatingClone, onGenerate, cloneJobs,
  loadingCloneJobs, cloneJobsError, selectedCloneJobId, onSelectCloneJob, onLoadCloneVariantInputs, cloneDraft,
  copiedCloneScript, onCopyCloneScript, cloneJobMatchesVisibleInputs, formatViewMetric, formatTime,
}: Props) {
  const selectedJob = cloneJobs.find((job) => job.job_id === selectedCloneJobId) || cloneJobs[0] || null;
  const selectedEngine = cloneEngines.find((engine) => engine.key === cloneEngineKey) || cloneEngines[0] || null;
  const approvedConceptCount = splitLines(cloneConceptsText).length;
  const forbiddenReferenceCount = splitLines(cloneExcludedReferencesText).length;
  const labelFor = (job: EpisodeCloneJob, index: number) => String(job.request?.variant_label || '').trim() || `Variant ${Math.max(1, cloneJobs.length - index)}`;
  const statusTone = (status: string) => {
    const v = String(status || '').toLowerCase();
    if (v === 'completed') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    if (v === 'failed' || v === 'cancelled') return 'border-rose-200 bg-rose-50 text-rose-700';
    if (v === 'running') return 'border-fuchsia-200 bg-fuchsia-50 text-fuchsia-700';
    return 'border-amber-200 bg-amber-50 text-amber-700';
  };

  return <div className="h-full overflow-y-auto bg-slate-50/80 p-5"><div className="space-y-5">
    <div className="rounded-[28px] border border-fuchsia-100 bg-gradient-to-br from-fuchsia-50 via-rose-50 to-amber-50 p-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-5">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/90 text-fuchsia-700 shadow-sm"><Clapperboard size={20} /></div>
          <div><div className="text-[11px] font-semibold uppercase tracking-[0.26em] text-fuchsia-500">Clone Workbench</div><div className="mt-1 text-2xl font-semibold text-slate-950">AI Episode Cloning</div><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Detect reusable concepts first, clear source contamination, then generate independent variants with the engine you choose for this run.</p></div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">{[['Views', formatViewMetric(video?.view_count)], ['Published', video?.published_at ? new Date(video.published_at).toLocaleDateString() : 'Unknown'], ['Transcript Segments', String(segmentsCount)]].map(([label, value]) => <div key={label} className="rounded-2xl border border-white/80 bg-white/85 px-4 py-3 text-xs text-slate-600"><div className="font-semibold uppercase tracking-wide text-slate-500">{label}</div><div className="mt-1 text-lg font-semibold text-slate-900">{value}</div></div>)}</div>
      </div>
    </div>

    <div className="grid gap-5 2xl:grid-cols-2">
      <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-3"><div><div className="flex items-center gap-2 text-sm font-semibold text-slate-900"><Radar size={16} className="text-sky-600" />Phase 1: Detect Concepts</div><div className="mt-1 text-xs text-slate-500">Extract the portable ideas first, then remove source-specific references before generation.</div></div>{loadingCloneEngines && <Loader2 size={15} className="animate-spin text-slate-400" />}</div>
        <div className="mt-4 grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
          <div className="space-y-4">
            <label className="block"><div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Clone Engine</div><select value={cloneEngineKey} onChange={(e) => onCloneEngineKeyChange(e.target.value)} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800 outline-none focus:border-sky-300 focus:ring-2 focus:ring-sky-200">{cloneEngines.map((engine) => <option key={engine.key} value={engine.key} disabled={!engine.available}>{engine.label}{!engine.available && engine.disabled_reason ? ` - ${engine.disabled_reason}` : ''}</option>)}</select></label>
            {cloneUsesOllama && <label className="block"><div className="mb-1.5 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wide text-slate-500"><span>Ollama Model</span>{loadingCloneOllamaModels && <Loader2 size={13} className="animate-spin text-slate-400" />}</div><select value={cloneOllamaModel} onChange={(e) => onCloneOllamaModelChange(e.target.value)} disabled={loadingCloneOllamaModels || cloneOllamaModels.length === 0} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800 outline-none focus:border-sky-300 focus:ring-2 focus:ring-sky-200 disabled:cursor-not-allowed disabled:opacity-60"><option value="">{loadingCloneOllamaModels ? 'Loading local models...' : cloneOllamaModels.length === 0 ? 'No local models found' : 'Use configured default model'}</option>{cloneOllamaModels.map((model) => <option key={model.name} value={model.name}>{model.name}{model.parameter_size ? ` · ${model.parameter_size}` : ''}{model.quantization_level ? ` · ${model.quantization_level}` : ''}</option>)}</select>{cloneOllamaModelsError && <div className="mt-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">{cloneOllamaModelsError}</div>}</label>}
            <label className="block"><div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Filtering Notes</div><textarea value={cloneNotes} onChange={(e) => onCloneNotesChange(e.target.value)} rows={6} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800 outline-none focus:border-sky-300 focus:ring-2 focus:ring-sky-200" placeholder="Ignore tangents, live-chat detours, housekeeping, or off-topic references." /></label>
            <button onClick={onDetectConcepts} disabled={segmentsCount === 0 || detectingCloneConcepts || !selectedEngine?.available} className="inline-flex items-center gap-2 rounded-2xl bg-sky-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50">{detectingCloneConcepts ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}{detectingCloneConcepts ? 'Detecting...' : 'Detect Themes'}</button>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-600"><div className="font-semibold text-slate-700">Detection Snapshot</div><div className="mt-1">{cloneConcepts ? `${cloneConcepts.concepts.length} concepts | ${cloneConcepts.excluded_references.length} forbidden references` : 'No concept pass yet.'}</div>{cloneConcepts?.model && <div className="mt-1">Last run with: <span className="font-medium text-slate-800">{cloneConcepts.model}</span></div>}{cloneConcepts?.semantic_query && <div className="mt-1">Semantic query: <span className="font-medium text-slate-800">{cloneConcepts.semantic_query}</span></div>}{cloneEnginesError && <div className="mt-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-rose-700">{cloneEnginesError}</div>}</div>
          </div>
          <div className="space-y-4">
            <label className="block"><div className="mb-1.5 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wide text-slate-500"><span>Approved Concepts</span><span className="rounded-full bg-sky-100 px-2 py-0.5 text-[11px] text-sky-700">{approvedConceptCount}</span></div><textarea value={cloneConceptsText} onChange={(e) => onCloneConceptsTextChange(e.target.value)} rows={10} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-800 outline-none focus:border-sky-300 focus:ring-2 focus:ring-sky-200" placeholder="One approved concept per line." /></label>
            <label className="block"><div className="mb-1.5 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wide text-slate-500"><span>Forbidden References</span><span className="rounded-full bg-rose-100 px-2 py-0.5 text-[11px] text-rose-700">{forbiddenReferenceCount}</span></div><textarea value={cloneExcludedReferencesText} onChange={(e) => onCloneExcludedReferencesTextChange(e.target.value)} rows={6} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-800 outline-none focus:border-rose-300 focus:ring-2 focus:ring-rose-200" placeholder="One forbidden reference per line. Example: source episode title, channel name, Q&A session." /></label>
            {cloneConcepts?.warnings && cloneConcepts.warnings.length > 0 && <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4"><div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-amber-700"><ShieldBan size={14} />Detection Warnings</div><div className="mt-2 space-y-1 text-sm text-amber-900">{cloneConcepts.warnings.map((warning, idx) => <div key={`${idx}-${warning}`}>- {warning}</div>)}</div></div>}
          </div>
        </div>
      </section>

      <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-3"><div><div className="flex items-center gap-2 text-sm font-semibold text-slate-900"><Bot size={16} className="text-fuchsia-600" />Phase 2: Generate Variants</div><div className="mt-1 text-xs text-slate-500">Use the approved concept list plus style instructions to create independent clone drafts.</div></div>{loadingCloneJobs && <Loader2 size={15} className="animate-spin text-slate-400" />}</div>
        <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
          <div className="space-y-4">
            <label className="block"><div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Style Instructions</div><textarea value={cloneStylePrompt} onChange={(e) => onCloneStylePromptChange(e.target.value)} rows={10} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-800 outline-none focus:border-fuchsia-300 focus:ring-2 focus:ring-fuchsia-200" placeholder="Describe the target format, tone, hook, pacing, and structure." /></label>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-600"><div className="font-semibold text-slate-700">Generation Inputs</div><div className="mt-1">{approvedConceptCount} approved concept{approvedConceptCount === 1 ? '' : 's'} ready</div><div className="mt-1">{forbiddenReferenceCount} forbidden reference{forbiddenReferenceCount === 1 ? '' : 's'} enforced</div><div className="mt-1">Selected engine: <span className="font-medium text-slate-800">{selectedEngine?.label || 'Unknown'}</span></div>{cloneJobsError && <div className="mt-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-rose-700">{cloneJobsError}</div>}</div>
          </div>
          <div className="space-y-4 rounded-3xl border border-fuchsia-100 bg-gradient-to-br from-fuchsia-50 via-white to-rose-50 p-5">
            <label className="block"><div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Variants</div><select value={cloneBatchSize} onChange={(e) => onCloneBatchSizeChange(Number(e.target.value) || 1)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none focus:border-fuchsia-300 focus:ring-2 focus:ring-fuchsia-200"><option value={1}>1</option><option value={2}>2</option><option value={3}>3</option><option value={4}>4</option></select></label>
            <button onClick={onGenerate} disabled={segmentsCount === 0 || !cloneStylePrompt.trim() || approvedConceptCount === 0 || !selectedEngine?.available} className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-fuchsia-600 px-4 py-3 text-sm font-semibold text-white shadow-sm hover:bg-fuchsia-700 disabled:cursor-not-allowed disabled:opacity-50">{generatingClone ? <Loader2 size={15} className="animate-spin" /> : <Clapperboard size={15} />}{generatingClone ? 'Queueing...' : cloneBatchSize > 1 ? `Queue ${cloneBatchSize} Variants` : 'Queue Variant'}</button>
            <div className="rounded-2xl border border-white/80 bg-white/80 px-4 py-3 text-xs text-slate-600"><div className="font-semibold text-slate-700">Workbench State</div><div className="mt-1">{cloneJobs.length === 0 ? 'No variants yet.' : `${cloneJobs.length} variant${cloneJobs.length === 1 ? '' : 's'} on shelf | ${cloneJobs.filter((job) => ['queued', 'running'].includes(String(job.status || '').toLowerCase())).length} active`}</div></div>
          </div>
        </div>
      </section>
    </div>

    <div className="grid gap-5 2xl:grid-cols-[360px_minmax(0,1fr)]">
      <section className="space-y-5">
        <div className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold text-slate-900">Variant Shelf</div><div className="mt-1 text-xs text-slate-500">Keep prior clone attempts and reload their inputs.</div></div><div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">{cloneJobs.length} total</div></div><div className="mt-4 space-y-3">{cloneJobs.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-sm text-slate-500">No clone variants yet.</div> : cloneJobs.map((job, index) => <button key={job.job_id} type="button" onClick={() => onSelectCloneJob(job.job_id)} className={`w-full text-left rounded-2xl border px-4 py-4 transition ${selectedJob?.job_id === job.job_id ? 'border-fuchsia-300 bg-fuchsia-50 shadow-sm' : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-white'}`}><div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold text-slate-900">{labelFor(job, index)}</div><div className="mt-1 text-xs text-slate-500">Job #{job.job_id} | {new Date(job.created_at).toLocaleString()}</div></div><span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusTone(job.status)}`}>{String(job.status || '').toLowerCase() || 'unknown'}</span></div><div className="mt-2 text-xs text-slate-500">Engine: {job.result?.model || (job.request.model_override ? `${job.request.provider_override || 'engine'}:${job.request.model_override}` : 'default engine')}</div><div className="mt-2 line-clamp-2 text-sm text-slate-700">{job.request.style_prompt}</div><div className="mt-3 flex flex-wrap gap-2 text-[11px]"><span className="rounded-full bg-white px-2 py-0.5 text-slate-600">{job.progress}% complete</span><span className="rounded-full bg-white px-2 py-0.5 text-slate-600">{job.request.approved_concepts?.length || 0} concepts</span>{cloneJobMatchesVisibleInputs(job) && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-emerald-700">Matches current inputs</span>}</div></button>)}</div></div>
      </section>

      <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm">
        {!selectedJob ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-12 text-center text-sm text-slate-500">Select a clone variant from the shelf to review it.</div> : <>
          <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><div className="text-xl font-semibold text-slate-950">{labelFor(selectedJob, cloneJobs.findIndex((job) => job.job_id === selectedJob.job_id))}</div><span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusTone(selectedJob.status)}`}>{String(selectedJob.status || '').toLowerCase() || 'unknown'}</span></div><div className="mt-1 text-xs text-slate-500">Job #{selectedJob.job_id} | {new Date(selectedJob.created_at).toLocaleString()} | {selectedJob.progress}% complete</div></div><div className="flex flex-wrap gap-2"><button type="button" onClick={() => onLoadCloneVariantInputs(selectedJob)} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"><RotateCcw size={13} />Load Inputs</button><button onClick={onCopyCloneScript} disabled={!cloneDraft?.script} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"><Copy size={13} />{copiedCloneScript ? 'Copied' : 'Copy Script'}</button></div></div>
          <div className="mt-4 grid gap-3 xl:grid-cols-3">{[['Engine Used', cloneDraft?.model || (selectedJob.request.model_override ? `${selectedJob.request.provider_override || 'engine'}:${selectedJob.request.model_override}` : 'Default engine')], ['Approved Concepts', String(cloneDraft?.approved_concepts?.length ?? selectedJob.request.approved_concepts?.length ?? 0)], ['Forbidden References', String(cloneDraft?.excluded_references?.length ?? selectedJob.request.excluded_references?.length ?? 0)]].map(([label, value]) => <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50 p-4"><div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</div><div className="mt-2 text-sm font-medium text-slate-900">{value}</div></div>)}</div>
          <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700"><div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Prompt Snapshot</div><div className="mt-2 leading-6">{selectedJob.request.style_prompt}</div>{selectedJob.request.notes && <div className="mt-2 text-xs text-slate-500">Notes: {selectedJob.request.notes}</div>}{!cloneJobMatchesVisibleInputs(selectedJob) && <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">The current composer inputs differ from this variant. Use Load Inputs to continue from it.</div>}{selectedJob.status_detail && <div className="mt-2 text-xs text-slate-500">Status detail: {selectedJob.status_detail}</div>}{selectedJob.error && <div className="mt-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">{selectedJob.error}</div>}</div>
          <div className="mt-3 grid gap-4 xl:grid-cols-2"><div className="rounded-3xl border border-slate-200 bg-white p-4"><div className="text-sm font-semibold text-slate-900">Concept Snapshot</div><div className="mt-1 text-xs text-slate-500">These are the sanitized ideas the generator was allowed to use.</div><div className="mt-3 space-y-2">{(cloneDraft?.approved_concepts || selectedJob.request.approved_concepts || []).length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-sm text-slate-500">No approved concepts were stored for this variant.</div> : (cloneDraft?.approved_concepts || selectedJob.request.approved_concepts || []).map((concept, idx) => <div key={`${idx}-${concept}`} className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-800">{concept}</div>)}</div></div><div className="rounded-3xl border border-slate-200 bg-white p-4"><div className="text-sm font-semibold text-slate-900">Forbidden References</div><div className="mt-1 text-xs text-slate-500">If these show up in output, the variant is contaminated.</div><div className="mt-3 space-y-2">{(cloneDraft?.excluded_references || selectedJob.request.excluded_references || []).length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-sm text-slate-500">No forbidden references were stored for this variant.</div> : (cloneDraft?.excluded_references || selectedJob.request.excluded_references || []).map((item, idx) => <div key={`${idx}-${item}`} className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-3 text-sm text-rose-800">{item}</div>)}</div></div></div>
          {!cloneDraft ? <div className="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-12 text-center text-sm text-slate-500">{['queued', 'running'].includes(String(selectedJob.status || '').toLowerCase()) ? 'This variant is still being generated.' : 'This variant finished without a draft payload.'}</div> : <>
            <div className="mt-4 grid gap-3 xl:grid-cols-3">{[['Suggested Title', cloneDraft.suggested_title || 'None'], ['Opening Hook', cloneDraft.opening_hook || 'None'], ['Semantic Query', cloneDraft.semantic_query || 'None']].map(([label, value]) => <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50 p-4"><div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</div><div className="mt-2 text-sm text-slate-900">{value}</div></div>)}</div>
            <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 p-4"><div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Angle Summary</div><div className="mt-2 text-sm leading-6 text-slate-800">{cloneDraft.angle_summary || 'None'}</div></div>
            {cloneDraft.originality_notes.length > 0 && <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-4"><div className="text-[11px] font-semibold uppercase tracking-wide text-amber-700">Originality Notes</div><div className="mt-2 space-y-1 text-sm text-amber-900">{cloneDraft.originality_notes.map((note, idx) => <div key={`${idx}-${note}`}>- {note}</div>)}</div></div>}
            {cloneDraft.warnings.length > 0 && <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-4"><div className="text-[11px] font-semibold uppercase tracking-wide text-rose-700">Warnings</div><div className="mt-2 space-y-1 text-sm text-rose-900">{cloneDraft.warnings.map((warning, idx) => <div key={`${idx}-${warning}`}>- {warning}</div>)}</div></div>}
            <div className="mt-3 rounded-3xl border border-slate-200 bg-slate-950 p-4"><div className="mb-2 flex items-center justify-between gap-3"><div className="text-[11px] font-semibold uppercase tracking-wide text-slate-300">Script Review</div><div className="text-[11px] text-slate-400">{cloneDraft.script ? `${cloneDraft.script.length.toLocaleString()} chars` : 'No script'}</div></div><pre className="max-h-[520px] overflow-y-auto whitespace-pre-wrap break-words text-sm leading-6 text-slate-100">{cloneDraft.script || 'No script returned.'}</pre></div>
            <div className="mt-4 grid gap-4 xl:grid-cols-2"><div className="rounded-3xl border border-slate-200 bg-white p-4"><div className="text-sm font-semibold text-slate-900">Related Episodes Used</div><div className="mt-1 text-xs text-slate-500">Nearby episodes pulled through semantic search before concept sanitization.</div><div className="mt-3 space-y-2">{cloneDraft.related_videos.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-sm text-slate-500">No related episodes were incorporated into this variant.</div> : cloneDraft.related_videos.map((item) => <div key={item.video_id} className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3"><div className="font-medium text-slate-800">{item.title}</div><div className="mt-1 text-xs text-slate-500">{formatViewMetric(item.view_count)} views | {item.views_per_day != null ? `${formatViewMetric(item.views_per_day, 2)} views/day` : 'views/day unknown'} | semantic score {item.semantic_score != null ? item.semantic_score.toFixed(4) : 'n/a'} | {item.semantic_hit_count} hit{item.semantic_hit_count === 1 ? '' : 's'}</div></div>)}</div></div><div className="rounded-3xl border border-slate-200 bg-white p-4"><div className="text-sm font-semibold text-slate-900">Context Hits</div><div className="mt-1 text-xs text-slate-500">Transcript chunks that informed theme detection before clone generation.</div><div className="mt-3 space-y-2">{cloneDraft.context_hits.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-sm text-slate-500">No semantic context hits were attached to this variant.</div> : cloneDraft.context_hits.slice(0, 8).map((hit) => <div key={`${hit.chunk_id}-${hit.video_id}`} className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3"><div className="font-medium text-slate-800">{hit.video_title || `Video ${hit.video_id}`}</div><div className="mt-1 text-xs text-slate-500">{formatTime(hit.start_time)} - {formatTime(hit.end_time)} | score {hit.score.toFixed(4)}{hit.speaker_name ? ` | ${hit.speaker_name}` : ''}</div><div className="mt-2 line-clamp-4 text-sm leading-6 text-slate-700">{hit.chunk_text}</div></div>)}</div></div></div>
          </>}
        </>}
      </section>
    </div>
  </div></div>;
}
