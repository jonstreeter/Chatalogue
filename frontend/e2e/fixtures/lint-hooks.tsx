import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ClipEditorWorkspace } from '../../src/pages/video/tabs/ClipEditorWorkspace';
import { ReconstructionTab } from '../../src/pages/video/tabs/ReconstructionTab';
import { useClipsStore } from '../../src/store/useClipsStore';
import type { Clip, TranscriptSegment } from '../../src/types';

const clip: Clip = { id: 1, video_id: 1, title: 'Regression clip', start_time: 0, end_time: 10, created_at: '2026-10-15' };
const segments: TranscriptSegment[] = [];
const noop = () => {};

export function Fixture() {
    const [active, setActive] = useState(false);
    return <>
        <button onClick={() => useClipsStore.setState({ clipEditorDraft: { ...clip } })}>Open draft</button>
        <button onClick={() => useClipsStore.setState({ clipEditorDraft: null })}>Close draft</button>
        <button onClick={() => setActive(value => !value)}>Toggle reconstruction</button>
        <ClipEditorWorkspace activeEditingClip={clip} currentTime={1} video={null} segments={segments} playerNode={null} onSeek={noop} />
        <ReconstructionTab isActive={active} reconstructionWorkbench={null} loadingReconstructionWorkbench={false}
            segmentsCount={0} reconstructionStudioTab="voices" reconstructionStatus="idle" reconstructionPaused={false}
            reconstructionBusy={false} hasReconstructionAudio={false} usingReconstructionForPlayback={false}
            reconstructionWorkbenchProgressNode={null} reconstructionWorkbenchActivityNode={null} reconstructionJobNode={null}
            video={null} segments={segments} isUploadedMedia={false} episodeBusy={false} reconstructionAudioUrl=""
            resolveWorkbenchAudioUrl={value => value || ''} setVideo={noop} onSetReconstructionPlayback={noop}
            onRefreshWorkbench={noop} onSetStudioTab={noop} />
    </>;
}

createRoot(document.getElementById('root')!).render(<StrictMode><Fixture /></StrictMode>);
