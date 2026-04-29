import { useEffect, useRef } from 'react';
import { CloneWorkbenchPanel } from '../../../components/video/CloneWorkbenchPanel';
import { useCloneStore, useCloneDraftResult, useCloneUsesOllama } from '../../../store/useCloneStore';
import type { Video } from '../../../types';

type Props = {
    video: Video | null;
    segmentsCount: number;
    /** Pass the numeric video ID so the tab can fetch its own data. */
    videoId: number;
    /** Whether this tab is currently visible (drives fetch triggers). */
    isActive: boolean;
};

export function CloneTab({ video, segmentsCount, videoId, isActive }: Props) {
    const store = useCloneStore();
    const cloneDraft = useCloneDraftResult();
    const cloneUsesOllama = useCloneUsesOllama();

    // Stable ref so effects can read the latest value without re-running.
    const cloneEngineKeyRef = useRef(store.cloneEngineKey);
    cloneEngineKeyRef.current = store.cloneEngineKey;

    // Fetch engines + jobs whenever the tab becomes active.
    useEffect(() => {
        if (!isActive) return;
        const controller = new AbortController();
        void store.fetchCloneEngines(controller.signal);
        void store.fetchCloneJobs(videoId, controller.signal);
        return () => controller.abort();
    }, [isActive, videoId]);

    // Fetch Ollama models when the engine switches to an Ollama provider.
    useEffect(() => {
        if (!isActive || !cloneUsesOllama) return;
        const controller = new AbortController();
        void store.fetchCloneOllamaModels(controller.signal);
        return () => controller.abort();
    }, [isActive, cloneUsesOllama, store.cloneEngineKey]);

    // Poll while a job is active.
    useEffect(() => {
        if (!isActive || store.cloneJobs.length === 0) return;
        const hasActive = store.cloneJobs.some((j) =>
            ['queued', 'running'].includes(String(j.status || '').toLowerCase())
        );
        if (!hasActive) return;
        const interval = window.setInterval(() => {
            void store.fetchCloneJobs(videoId, undefined, { silent: true });
        }, 2500);
        return () => window.clearInterval(interval);
    }, [isActive, store.cloneJobs, videoId]);

    return (
        <CloneWorkbenchPanel
            video={video}
            segmentsCount={segmentsCount}
            cloneEngineKey={store.cloneEngineKey}
            onCloneEngineKeyChange={store.setCloneEngineKey}
            cloneEngines={store.cloneEngines}
            loadingCloneEngines={store.loadingCloneEngines}
            cloneEnginesError={store.cloneEnginesError}
            cloneUsesOllama={cloneUsesOllama}
            cloneOllamaModel={store.cloneOllamaModel}
            onCloneOllamaModelChange={store.setCloneOllamaModel}
            cloneOllamaModels={store.cloneOllamaModels}
            loadingCloneOllamaModels={store.loadingCloneOllamaModels}
            cloneOllamaModelsError={store.cloneOllamaModelsError}
            detectingCloneConcepts={store.detectingCloneConcepts}
            onDetectConcepts={() => void store.handleDetectCloneConcepts(videoId)}
            cloneConcepts={store.cloneConcepts}
            cloneConceptsText={store.cloneConceptsText}
            onCloneConceptsTextChange={store.setCloneConceptsText}
            cloneExcludedReferencesText={store.cloneExcludedReferencesText}
            onCloneExcludedReferencesTextChange={store.setCloneExcludedReferencesText}
            cloneStylePrompt={store.cloneStylePrompt}
            onCloneStylePromptChange={store.setCloneStylePrompt}
            cloneNotes={store.cloneNotes}
            onCloneNotesChange={store.setCloneNotes}
            cloneBatchSize={store.cloneBatchSize}
            onCloneBatchSizeChange={store.setCloneBatchSize}
            generatingClone={store.generatingClone}
            onGenerate={() => void store.handleGenerateEpisodeClone(videoId)}
            cloneJobs={store.cloneJobs}
            loadingCloneJobs={store.loadingCloneJobs}
            cloneJobsError={store.cloneJobsError}
            selectedCloneJobId={store.selectedCloneJobId}
            onSelectCloneJob={store.setSelectedCloneJobId}
            onLoadCloneVariantInputs={store.loadCloneVariantInputs}
            cloneDraft={cloneDraft}
            copiedCloneScript={store.copiedCloneScript}
            onCopyCloneScript={() => void store.handleCopyCloneScript()}
            cloneJobMatchesVisibleInputs={store.cloneJobMatchesVisibleInputs}
        />
    );
}
