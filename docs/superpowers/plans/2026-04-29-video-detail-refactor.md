# VideoDetailPage Strangler Fig Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incrementally decompose `VideoDetailPage.tsx` (8 000+ lines) and other large page files into isolated Zustand stores + focused tab/page components using the Strangler Fig pattern — one feature at a time, never breaking the running app.

**Architecture:** Each sidebar tab gets its own Zustand store in `src/store/` and a self-contained component in `src/pages/video/tabs/`. The store owns all state, async fetches, and business logic for that tab. `VideoDetailPage` shrinks to a thin shell that renders `<ActiveTab />` and holds only truly shared state (video meta, segments, player, active tab routing). Other large pages follow the same pattern.

**Tech Stack:** React 18, TypeScript, Zustand (`zustand` + `zustand/middleware`), Vite, Axios (`api` from `src/lib/api.ts`), Lucide icons

---

## Established Conventions — Read Before Any Task

These patterns were set in Phase 1 (Clone tab). All subsequent phases must follow them exactly.

### Store pattern (`src/store/use<Feature>Store.ts`)
```ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { SomeType } from '../types';

// Module-level counters replace useRef — mutated without triggering re-renders
let _requestId = 0;

export interface FeatureState {
  // state fields ...
  // actions ...
  resetFeatureState: () => void;
}

export const useFeatureStore = create<FeatureState>()(
  devtools(
    (set, get) => ({
      // implementation ...
    }),
    { name: 'FeatureStore' }
  )
);

// Selector hooks for derived/computed state
export const useDerivedValue = () => useFeatureStore((s) => /* compute */ s.field);
```

### Tab component pattern (`src/pages/video/tabs/<Feature>Tab.tsx`)
```tsx
import { useEffect } from 'react';
import { useFeatureStore } from '../../../store/useFeatureStore';
import type { Video } from '../../../types';

type Props = { video: Video | null; videoId: number; isActive: boolean; /* minimal */ };

export function FeatureTab({ video, videoId, isActive }: Props) {
  const store = useFeatureStore();
  useEffect(() => {
    if (!isActive) return;
    const c = new AbortController();
    void store.fetchSomething(videoId, c.signal);
    return () => c.abort();
  }, [isActive, videoId]);
  return <SomeWorkbenchPanel /* bind store fields */ />;
}
```

### Integration into VideoDetailPage
- Call `useFeatureStore.getState().resetFeatureState()` inside the `[id]` useEffect.
- Replace `<OldInlineJsx />` with `<FeatureTab video={video} videoId={Number(id)} isActive={activeTab === 'feature'} />`.
- Read only the minimum store fields needed for cross-tab concerns (e.g. `cloneEngines` for the chat sidebar).
- Delete extracted useState/useRef/functions/types from VideoDetailPage.
- Shared pure utilities go in `src/lib/formatters.ts` (already exists) or `src/lib/utils.ts`.

### Verify after every phase
```bash
cd frontend && npx tsc --noEmit
```
Zero errors = done.

---

## Status Dashboard

| Phase | Feature | Store | Tab Component | Integrated | Done |
|---|---|---|---|---|---|
| 1 | AI Clone tab | `useCloneStore.ts` ✅ | `CloneTab.tsx` ✅ | ✅ | ✅ |
| 2 | YouTube/Summary metadata tab | `useYoutubeStore.ts` ✅ | `YoutubeTab.tsx` ✅ | ✅ | ✅ |
| 3 | Speakers tab | `useSpeakersTabStore.ts` ✅ | `SpeakersTab.tsx` ✅ | ✅ | ✅ |
| 4 | Clips tab | `useClipsStore.ts` ✅ | `ClipsTab.tsx` ✅ + `ClipEditorWorkspace.tsx` ✅ | ✅ | ⚠️ |
| 5 | Cleanup tab | `useCleanupStore.ts` ✅ | `CleanupTab.tsx` ✅ | ✅ | ⚠️ |
| 6 | Reconstruction tab | `useReconstructionStore.ts` ✅ | `ReconstructionTab.tsx` ✅ | ✅ | ⚠️ |
| 7 | Transcript + Optimize tabs | `useTranscriptStore.ts` ✅ | `TranscriptTab.tsx` ✅, `OptimizeTab.tsx` ✅ | ✅ | ✅ |
| 8 | Player + shell | — | thin shell | | |
| 9 | Settings.tsx | multiple stores | settings section components | | |
| 10 | JobQueue.tsx | `useJobQueueStore.ts` | — | | |

---

## Phase 2 — YouTube / Summary Metadata Tab

**Target state extracted from VideoDetailPage (lines ~131–136, ~3425–3540, ~7044–7350):**

```ts
// useState to remove:
const [generatingYoutubeAi, setGeneratingYoutubeAi] = useState(false);
const [copiedYoutubeField, setCopiedYoutubeField] = useState<'summary'|'chapters'|'description'|null>(null);
const [descriptionHistory, setDescriptionHistory] = useState<VideoDescriptionRevision[]>([]);
const [loadingDescriptionHistory, setLoadingDescriptionHistory] = useState(false);
const [publishingYoutubeDescription, setPublishingYoutubeDescription] = useState(false);
const [restoringDescriptionRevisionId, setRestoringDescriptionRevisionId] = useState<number|null>(null);

// functions to remove:
handleGenerateYoutubeAi, parseYoutubeAiChapters, fetchDescriptionHistory,
handlePublishYoutubeDescription, handleCopyYoutubeField, handleRestoreDescriptionRevision

// computed to remove:
youtubeAiChapters, hasYoutubeAiMetadata
// (isYoutubeMedia, aiMetadataTabLabel, etc. stay — they inform tab routing)
```

### Task 2.1 — Create `useYoutubeStore.ts`

**Files:**
- Create: `frontend/src/store/useYoutubeStore.ts`

- [x] Create the store file with this exact content:

```ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import api from '../lib/api';
import type { Video, VideoDescriptionRevision, VideoChapterSuggestion } from '../types';

export interface YoutubeState {
  generatingYoutubeAi: boolean;
  copiedYoutubeField: 'summary' | 'chapters' | 'description' | null;
  descriptionHistory: VideoDescriptionRevision[];
  loadingDescriptionHistory: boolean;
  publishingYoutubeDescription: boolean;
  restoringDescriptionRevisionId: number | null;

  fetchDescriptionHistory: (videoId: number) => Promise<void>;
  handleGenerateYoutubeAi: (videoId: number, force: boolean, onVideoUpdated: (v: Video) => void) => Promise<void>;
  handlePublishYoutubeDescription: (videoId: number, isYoutubeMedia: boolean, onVideoUpdated: (v: Video) => void) => Promise<void>;
  handleRestoreDescriptionRevision: (videoId: number, revisionId: number, onVideoUpdated: (v: Video) => void) => Promise<void>;
  handleCopyYoutubeField: (kind: 'summary' | 'chapters' | 'description', text: string) => Promise<void>;
  resetYoutubeState: () => void;
}

export const useYoutubeStore = create<YoutubeState>()(
  devtools(
    (set, get) => ({
      generatingYoutubeAi: false,
      copiedYoutubeField: null,
      descriptionHistory: [],
      loadingDescriptionHistory: false,
      publishingYoutubeDescription: false,
      restoringDescriptionRevisionId: null,

      fetchDescriptionHistory: async (videoId) => {
        set({ loadingDescriptionHistory: true }, false, 'fetchDescriptionHistory/pending');
        try {
          const res = await api.get<VideoDescriptionRevision[]>(`/videos/${videoId}/description-history`);
          set({ descriptionHistory: res.data }, false, 'fetchDescriptionHistory/fulfilled');
        } catch (e) {
          console.error('Failed to fetch description history', e);
        } finally {
          set({ loadingDescriptionHistory: false }, false, 'fetchDescriptionHistory/settled');
        }
      },

      handleGenerateYoutubeAi: async (videoId, force, onVideoUpdated) => {
        set({ generatingYoutubeAi: true }, false, 'handleGenerateYoutubeAi/pending');
        try {
          const res = await api.post<Video>(`/videos/${videoId}/youtube-ai/generate`, null, { params: { force } });
          onVideoUpdated(res.data);
        } catch (e: any) {
          console.error('Failed to generate YouTube metadata', e);
          alert(e?.response?.data?.detail || 'Failed to generate YouTube summary/chapters');
        } finally {
          set({ generatingYoutubeAi: false }, false, 'handleGenerateYoutubeAi/settled');
        }
      },

      handlePublishYoutubeDescription: async (videoId, isYoutubeMedia, onVideoUpdated) => {
        const confirmMsg = isYoutubeMedia
          ? 'Archive the current description and replace it with the AI-generated YouTube description draft?'
          : 'Archive the current description and replace it with the AI-generated episode description draft?';
        if (!confirm(confirmMsg)) return;
        set({ publishingYoutubeDescription: true }, false, 'handlePublishYoutubeDescription/pending');
        try {
          const res = await api.post<Video>(`/videos/${videoId}/youtube-ai/publish-description`);
          onVideoUpdated(res.data);
          await get().fetchDescriptionHistory(videoId);
        } catch (e: any) {
          console.error('Failed to publish description', e);
          alert(e?.response?.data?.detail || 'Failed to publish description');
        } finally {
          set({ publishingYoutubeDescription: false }, false, 'handlePublishYoutubeDescription/settled');
        }
      },

      handleRestoreDescriptionRevision: async (videoId, revisionId, onVideoUpdated) => {
        if (!confirm('Restore this archived description as the current stored description?')) return;
        set({ restoringDescriptionRevisionId: revisionId }, false, 'handleRestoreDescriptionRevision/pending');
        try {
          const res = await api.post<Video>(`/videos/${videoId}/description-history/${revisionId}/restore`);
          onVideoUpdated(res.data);
          await get().fetchDescriptionHistory(videoId);
        } catch (e: any) {
          console.error('Failed to restore description revision', e);
          alert(e?.response?.data?.detail || 'Failed to restore description revision');
        } finally {
          set({ restoringDescriptionRevisionId: null }, false, 'handleRestoreDescriptionRevision/settled');
        }
      },

      handleCopyYoutubeField: async (kind, text) => {
        try {
          await navigator.clipboard.writeText(text);
          set({ copiedYoutubeField: kind }, false, 'handleCopyYoutubeField/copied');
          window.setTimeout(
            () => set((s) => ({ copiedYoutubeField: s.copiedYoutubeField === kind ? null : s.copiedYoutubeField }), false, 'handleCopyYoutubeField/reset'),
            1500
          );
        } catch {
          alert('Failed to copy');
        }
      },

      resetYoutubeState: () =>
        set(
          {
            generatingYoutubeAi: false,
            copiedYoutubeField: null,
            descriptionHistory: [],
            loadingDescriptionHistory: false,
            publishingYoutubeDescription: false,
            restoringDescriptionRevisionId: null,
          },
          false,
          'resetYoutubeState'
        ),
    }),
    { name: 'YoutubeStore' }
  )
);

export function parseYoutubeAiChapters(chaptersJson?: string): VideoChapterSuggestion[] {
  if (!chaptersJson) return [];
  try {
    const parsed = JSON.parse(chaptersJson);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}
```

- [x] Run `cd frontend && npx tsc --noEmit` — expect zero errors.

### Task 2.2 — Create `YoutubeTab.tsx`

**Files:**
- Create: `frontend/src/pages/video/tabs/YoutubeTab.tsx`

- [x] Find the YouTube sidebar JSX block in VideoDetailPage. It starts at the line matching `{activeTab === 'youtube' && (` (currently ~line 7044) and closes ~300 lines later. Read it with `Read offset=7044 limit=320`.

- [x] Create `YoutubeTab.tsx`. The component receives `video`, `videoId`, `segments`, `isYoutubeMedia`, `isActive` as props. It reads store state via `useYoutubeStore()` and calls store actions. Move the entire JSX block from `{activeTab === 'youtube' && ( ... )}` into the return value, replacing all inline `setState` calls with store actions. The `onVideoUpdated` callback passed to store actions calls the parent's `setVideo` — receive it as a prop `onVideoUpdated: (v: Video) => void`.

  Skeleton:
  ```tsx
  import { useEffect } from 'react';
  import { useYoutubeStore, parseYoutubeAiChapters } from '../../../store/useYoutubeStore';
  import { formatViewMetric } from '../../../lib/formatters';
  import type { Video, TranscriptSegment } from '../../../types';
  // import Lucide icons used in the JSX block

  type Props = {
    video: Video;
    videoId: number;
    segments: TranscriptSegment[];
    isYoutubeMedia: boolean;
    isActive: boolean;
    onVideoUpdated: (v: Video) => void;
  };

  export function YoutubeTab({ video, videoId, segments, isYoutubeMedia, isActive, onVideoUpdated }: Props) {
    const store = useYoutubeStore();
    const youtubeAiChapters = parseYoutubeAiChapters(video.youtube_ai_chapters_json);
    const hasYoutubeAiMetadata = !!(video.youtube_ai_summary || video.youtube_ai_description_text || youtubeAiChapters.length);

    useEffect(() => {
      if (!isActive) return;
      void store.fetchDescriptionHistory(videoId);
    }, [isActive, videoId]);

    // Paste full sidebar JSX here, replacing inline setState with store actions
    return ( /* JSX */ );
  }
```

- [x] Run `cd frontend && npx tsc --noEmit` — expect zero errors.

### Task 2.3 — Integrate into VideoDetailPage

**Files:**
- Modify: `frontend/src/pages/video/VideoDetailPage.tsx`

- [x] Add import at top of file:
  ```ts
  import { YoutubeTab } from './tabs/YoutubeTab';
  import { useYoutubeStore } from '../../store/useYoutubeStore';
  ```

- [x] Remove these 6 `useState` declarations (search exact text):
  ```
  const [generatingYoutubeAi, ...
  const [copiedYoutubeField, ...
  const [descriptionHistory, ...
  const [loadingDescriptionHistory, ...
  const [publishingYoutubeDescription, ...
  const [restoringDescriptionRevisionId, ...
  ```

- [x] In the `[id]` useEffect reset block, replace all six `set*` clone calls with:
  ```ts
  useYoutubeStore.getState().resetYoutubeState();
  ```

- [x] Remove the youtube `useEffect` that calls `fetchDescriptionHistory` when `activeTab === 'youtube'` (around line 3533).

- [x] Remove functions: `handleGenerateYoutubeAi`, `parseYoutubeAiChapters`, `fetchDescriptionHistory`, `handlePublishYoutubeDescription`, `handleCopyYoutubeField` (search by name and delete each block), `handleRestoreDescriptionRevision` if present.

- [x] Remove computed vars `youtubeAiChapters` and `hasYoutubeAiMetadata` (they now live in `YoutubeTab`).

- [x] Replace the `{activeTab === 'youtube' && ( ... )}` sidebar block (the one that was just moved) with:
  ```tsx
  {activeTab === 'youtube' && video && (
    <YoutubeTab
      video={video}
      videoId={Number(id)}
      segments={segments}
      isYoutubeMedia={isYoutubeMedia}
      isActive={activeTab === 'youtube'}
      onVideoUpdated={setVideo}
    />
  )}
  ```

- [x] Remove `VideoDescriptionRevision` from the type import line if it's no longer used elsewhere (grep first).

- [x] Run `cd frontend && npx tsc --noEmit` — expect zero errors.

- [ ] Commit:
  ```bash
  git add frontend/src/store/useYoutubeStore.ts \
          frontend/src/pages/video/tabs/YoutubeTab.tsx \
          frontend/src/pages/video/VideoDetailPage.tsx
  git commit -m "refactor: extract YouTube metadata tab to YoutubeTab + useYoutubeStore"
  ```

---

## Phase 3 — Speakers Tab

**Target state extracted from VideoDetailPage (lines ~100–102, ~175–184):**

```ts
// useState to remove (speakers-tab-specific):
const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
const [initialSample, setInitialSample] = useState<SpeakerSample | null>(null);
const [assignPopup, setAssignPopup] = useState<{...} | null>(null);
const [assignSpeakers, setAssignSpeakers] = useState<Speaker[]>([]);
const [assignSearch, setAssignSearch] = useState('');
const [assignLoading, setAssignLoading] = useState(false);
// useRef to remove:
const speakerDetailCacheRef = useRef<Map<number, Speaker>>(new Map());
```

Note: `SpeakerModal` and `SpeakerList` are already separate components. The speakers-tab state controls which speaker is selected in the sidebar panel and the assignment popup.

### Task 3.1 — Create `useSpeakersTabStore.ts`

**Files:**
- Create: `frontend/src/store/useSpeakersTabStore.ts`

- [x] Identify all speaker-tab-specific functions in VideoDetailPage by grepping:
  ```bash
  grep -n "assignPopup\|assignSpeakers\|assignSearch\|assignLoading\|selectedSpeaker\|initialSample\|speakerDetailCache" \
    frontend/src/pages/video/VideoDetailPage.tsx
  ```

- [x] Read each function body (use line numbers from grep output) and replicate them as store actions following the established pattern. The `speakerDetailCacheRef` becomes a module-level `Map`.

- [x] Create `useSpeakersTabStore.ts` with `devtools` middleware, all state fields, all actions, and `resetSpeakersTabState`.

- [x] Run `npx tsc --noEmit` — zero errors.

### Task 3.2 — Create `SpeakersTab.tsx`

**Files:**
- Create: `frontend/src/pages/video/tabs/SpeakersTab.tsx`

- [x] Grep for `{activeTab === 'speakers' && (` to find the sidebar JSX block start line.
- [x] Read that block and move it into `SpeakersTab.tsx`, binding all state via `useSpeakersTabStore`.
- [x] Props: `video`, `videoId`, `segments`, `isActive`, `onVideoUpdated`.
- [x] Run `npx tsc --noEmit` — zero errors.

### Task 3.3 — Integrate into VideoDetailPage

- [x] Add import, replace JSX block, remove state/functions, add `resetSpeakersTabState()` to `[id]` effect.
- [x] Run `npx tsc --noEmit` — zero errors.
- [ ] Commit.

---

## Phase 4 — Clips Tab

**Complexity: HIGH** — The clips tab has ~20 useState declarations, an inline clip editor with drag state, batch export/upload logic, and it reads `currentTime` from the player for timeline scrubbing.

**Target state (lines ~95–172):**
```ts
const [selection, setSelection] = useState<...>
const [clipTitle, setClipTitle] = useState('')
const [creatingClip, setCreatingClip] = useState(false)
const [clips, setClips] = useState<Clip[]>([])
const [clipExportArtifactsByClip, setClipExportArtifactsByClip] = useState<...>
const [loadingClips, setLoadingClips] = useState(false)
const [selectedClipIds, setSelectedClipIds] = useState<Set<number>>(new Set())
const [editingClipId, setEditingClipId] = useState<number | null>(null)
const [clipEditorDraft, setClipEditorDraft] = useState<Partial<Clip> | null>(null)
const [clipEditorTokens, setClipEditorTokens] = useState<...>
const [clipEditorRemovedWordKeys, setClipEditorRemovedWordKeys] = useState<Set<string>>(new Set())
const [clipEditorCropTarget, setClipEditorCropTarget] = useState<...>
const [clipEditorDragRect, setClipEditorDragRect] = useState<...>
const [clipTimelineDrag, setClipTimelineDrag] = useState<...>
const [savingClipEdit, setSavingClipEdit] = useState(false)
const [exportingClipIds, setExportingClipIds] = useState<Set<number>>(new Set())
const [batchExporting, setBatchExporting] = useState(false)
const [batchQueueingRenders, setBatchQueueingRenders] = useState(false)
const [uploadingClipIds, setUploadingClipIds] = useState<Set<number>>(new Set())
const [batchUploadingClips, setBatchUploadingClips] = useState(false)
const [clipUploadPrivacy, setClipUploadPrivacy] = useState<...>
const [clipPreviewLoop, setClipPreviewLoop] = useState<...>
const [clipBatchPresetKey, setClipBatchPresetKey] = useState<...>
```

**Special concern:** `clipPreviewLoop` is consumed by the player. After extraction, `ClipsTab` must expose a callback or the player must subscribe to `useClipsStore` directly. Recommended: player reads `useClipsStore((s) => s.clipPreviewLoop)`.

### Task 4.1 — Create `useClipsStore.ts`

- [x] Grep for all clips-related functions:
  ```bash
  grep -n "const.*[Cc]lip\|fetchClips\|createClip\|deleteClip\|exportClip\|uploadClip\|handleClip\|saveClip" \
    frontend/src/pages/video/VideoDetailPage.tsx | head -60
  ```
- [x] Read each function body and move it to the store as an action.
- [x] Create `frontend/src/store/useClipsStore.ts` with `devtools`, all state, all actions, `resetClipsState`.
- [x] Run `npx tsc --noEmit` — zero errors.

### Task 4.2 — Create `ClipsTab.tsx`

- [x] Grep for `{activeTab === 'clips' && (` to find the sidebar JSX block.
- [x] Grep for `renderClipEditor` or `showClipEditorMain` to find the main-area clip editor block.
- [x] Move both JSX blocks into `ClipsTab.tsx`. Props: `video`, `videoId`, `segments`, `currentTime`, `isActive`, `onSeek`, `onVideoUpdated`.
  - Verified 2026-05-09: sidebar clips list lives in `ClipsTab.tsx`; the main editor workspace is extracted as `ClipEditorWorkspace.tsx` and rendered by `VideoDetailPage` when `showClipEditorMain` is true.
  - Remaining deviation: transcript text selection and the slide-up "Create Clip" panel still live in `VideoDetailPage.tsx` because clip creation is triggered from transcript selection rather than the clips tab.
- [x] Run `npx tsc --noEmit` — zero errors.
  - Verified 2026-05-09 with `cd frontend && npx tsc --noEmit`.

### Task 4.3 — Integrate into VideoDetailPage

- [x] Replace JSX, remove state/functions, wire `resetClipsState`, update player section to read `useClipsStore((s) => s.clipPreviewLoop)`.
  - Verified 2026-05-09: `VideoDetailPage.tsx` imports and renders `ClipsTab`, `ClipEditorWorkspace`, and reads `clipPreviewLoop` from `useClipsStore`.
  - Remaining cleanup: `selection`, `handleMouseUp`, `handleCreateClip`, `fetchClips`, and `fetchClipExportArtifacts` still live in `VideoDetailPage.tsx` for transcript-selection-driven clip creation/fetch orchestration.
- [x] Run `npx tsc --noEmit` — zero errors.
  - Verified 2026-05-09 with `cd frontend && npx tsc --noEmit`.
- [ ] Commit.

---

## Phase 5 — Cleanup Tab

**Complexity: MEDIUM-HIGH** — Cleanup owns the VoiceFixer workbench, ClearVoice models, auxiliary job polling.

**Target state (lines ~220–230):**
```ts
const [queueingVoiceFixer, setQueueingVoiceFixer]
const [auxiliaryJobs, setAuxiliaryJobs]
const [workbenchTaskProgress, setWorkbenchTaskProgress]
const [loadingCleanupWorkbench, setLoadingCleanupWorkbench]
const [cleanupWorkbench, setCleanupWorkbench]
const [analyzingCleanupWorkbench, setAnalyzingCleanupWorkbench]
const [runningClearVoiceModel, setRunningClearVoiceModel]
const [selectingCleanupCandidateId, setSelectingCleanupCandidateId]
const [clearVoiceInstallInfo, setClearVoiceInstallInfo]
const [loadingClearVoiceInstallInfo, setLoadingClearVoiceInstallInfo]
// + voiceFixer busy/paused flags
```

**Special concern:** `auxiliaryJobs` and `workbenchTaskProgress` are also read by the Reconstruction tab. Extract a shared `useWorkbenchStore` that both Cleanup and Reconstruction tabs subscribe to, OR have one store import the other. Recommended: single `useWorkbenchStore` for polling/progress, separate stores for each tab's UI state.

### Task 5.1 — Audit shared state between Cleanup and Reconstruction

- [x] Grep for every variable used in BOTH `renderCleanupStudio()` and `renderReconstructionStudio()`:
  ```bash
  grep -n "voiceFixerBusy\|voiceFixerPaused\|reconstructionBusy\|reconstructionPaused\|workbenchTaskProgress\|auxiliaryJobs\|loadAuxiliaryJobs" \
    frontend/src/pages/video/VideoDetailPage.tsx
  ```
- [x] List which variables are shared, create `useWorkbenchStore.ts` for them.
  - Verified 2026-05-09: `useWorkbenchStore.ts` exists and `VideoDetailPage.tsx` reads shared `auxiliaryJobs` / `workbenchTaskProgress` from it.

### Task 5.2 — Create `useWorkbenchStore.ts` (shared)

- [x] Create `frontend/src/store/useWorkbenchStore.ts` with polling logic for `auxiliaryJobs`, `workbenchTaskProgress`, voiceFixer and reconstruction busy/paused flags.
  - Verified 2026-05-09: shared store exists and TypeScript passes.

### Task 5.3 — Create `useCleanupStore.ts`

- [x] Move Cleanup-specific state and `renderCleanupStudio` logic into `frontend/src/store/useCleanupStore.ts`.
  - Verified 2026-05-09: cleanup-specific store exists and `CleanupTab.tsx` uses `useCleanupStore()`.
  - Remaining cleanup: `VideoDetailPage.tsx` still contains an inline `renderCleanupStudio()` function and stale cleanup reset/fetch references (`setCleanupWorkbench`, `setClearVoiceInstallInfo`, `setClearVoiceTestResult`, `fetchCleanupWorkbench`, `fetchClearVoiceInstallInfo`) that appear to be dead/stale code paths after extraction but still compile.

### Task 5.4 — Create `CleanupTab.tsx`

- [x] Move `renderCleanupStudio()` JSX into `frontend/src/pages/video/tabs/CleanupTab.tsx`.
  - Verified 2026-05-09: main cleanup stage renders `<CleanupTab />` from `VideoDetailPage.tsx`, and `CleanupTab.tsx` owns the cleanup UI.

### Task 5.5 — Integrate

- [x] Replace inline JSX, remove state/functions, commit.
  - Verified 2026-05-09: `VideoDetailPage.tsx` renders `<CleanupTab />` for the cleanup main stage.
  - Remaining cleanup before marking phase truly done: delete stale inline `renderCleanupStudio()` and any obsolete cleanup reset/fetch references from `VideoDetailPage.tsx`.
- [x] Run `npx tsc --noEmit` — zero errors.
  - Verified 2026-05-09 with `cd frontend && npx tsc --noEmit`.
- [ ] Commit.

---

## Phase 6 — Reconstruction Tab

Depends on `useWorkbenchStore` from Phase 5.

### Task 6.1 — Create `useReconstructionStore.ts`

- [x] Grep Reconstruction-specific state: `reconstructionWorkbench`, `selectedReconstructionSpeakerId`, `addingReconstructionSampleSpeakerId`, etc.
- [x] Move into `frontend/src/store/useReconstructionStore.ts`.
  - Verified 2026-05-09: reconstruction-specific store exists and `VideoDetailPage.tsx` reads reconstruction state/actions from it.

### Task 6.2 — Create `ReconstructionTab.tsx`

- [x] Move `renderReconstructionStudio()` (which calls `renderReconstructionVoiceReview()` and `renderReconstructionBuildSuite()`) into `frontend/src/pages/video/tabs/ReconstructionTab.tsx`. Convert the three render functions to local components or inline JSX.
  - Verified 2026-05-09: `ReconstructionTab.tsx` exports both `ReconstructionSidebarTab` and `ReconstructionTab`, and contains local reconstruction voice/build render functions.
  - Remaining deviation: `ReconstructionTab` receives a broad `ctx: any` prop with many callbacks/state values still assembled in `VideoDetailPage.tsx`; further cleanup should move more orchestration into `useReconstructionStore` or typed props.

### Task 6.3 — Integrate

- [x] Replace inline JSX, remove state/functions, commit.
  - Verified 2026-05-09: `VideoDetailPage.tsx` renders `ReconstructionSidebarTab` and `ReconstructionTab` for reconstruction routes.
  - Remaining cleanup before marking phase truly done: remove root-level reconstruction handler wrappers where possible and replace `ctx: any` with typed props or store-driven actions.
- [x] Run `npx tsc --noEmit` — zero errors.
  - Verified 2026-05-09 with `cd frontend && npx tsc --noEmit`.
- [ ] Commit.

---

## Phase 7 — Transcript + Optimize Tabs

**Complexity: VERY HIGH** — These two tabs share most state and are deeply intertwined with the player (follow-playback, segment editing, segment search, gold windows, evaluation). Extract together.

**Target state (lines ~60–61, ~111–119, ~186–218):** ~40 useState + several useRef declarations.

**Key shared concerns:**
- `segments` array — currently VideoDetailPage-level state; must stay there (or move to a root video store) since player and all tabs read it.
- `currentTime` — player-driven; must remain at root or be published via a tiny `usePlayerStore`.
- Segment edit state (`editingSegmentId`, `editingSegmentWords`, etc.) — safe to extract to transcript store.
- Gold window drafts — safe to extract to transcript store.
- Funny moments (`funnyMoments`, `funnyDrawerOpen`, etc.) — safe to extract to transcript store.

### Task 7.1 — Create `useTranscriptStore.ts`

- [x] Move all transcript-tab-specific state (segment editing, search, follow-playback, funny moments, gold windows, evaluation, transcript quality, rollback options) into `frontend/src/store/useTranscriptStore.ts`.
- [x] Keep `segments` in VideoDetailPage (or a new `useVideoStore`).

### Task 7.2 — Create `TranscriptTab.tsx`

- [x] Find the `{activeTab === 'transcript' && (` sidebar block and `{!showClipEditorMain && activeTab === 'transcript' && (` main block.
- [x] Move into `frontend/src/pages/video/tabs/TranscriptTab.tsx`. Props: `video`, `videoId`, `segments`, `currentTime`, `isActive`, `onSeek`.

### Task 7.3 — Create `OptimizeTab.tsx`

- [x] Move `renderTranscriptOptimizationWorkbench()` JSX (~line 4683) into `frontend/src/pages/video/tabs/OptimizeTab.tsx`.
  - Completed 2026-05-10: `OptimizeTab.tsx` owns the optimize fetch side effect, snapshot card, rollback controls, repair/rebuild/retranscribe actions, benchmark gold-window form, and evaluation review UI.

### Task 7.4 — Integrate both

- [x] Replace inline JSX blocks, remove state/functions, commit.
  - Completed 2026-05-10: `VideoDetailPage.tsx` now passes only video/selection/player callbacks into `TranscriptTab.tsx` and `OptimizeTab.tsx`; `segments` remains root-level for shared player/tab usage.

---

## Phase 8 — VideoDetailPage Shell

After all tabs are extracted, VideoDetailPage should contain only:
- Video meta fetch + `video` / `segments` state
- Player state (`currentTime`, `playbackRate`, `player`, `nativeMediaRef`, refs)
- Active tab routing (`activeTab`, `tabs` config, URL params)
- The `<TabNav>` + `<ActiveTab />` render

### Task 8.1 — Extract player logic

- [ ] Create `frontend/src/store/usePlayerStore.ts` for `currentTime`, `playbackRate`, `player`.
- [ ] Move `renderMainPlayer()`, `renderPlaybackRateControl()`, `renderUploadedPlaybackSourceSwitcher()` into `frontend/src/components/video/VideoPlayer.tsx`.
  - Progress 2026-05-10: created `frontend/src/components/video/VideoPlayer.tsx` and moved the main player render body for YouTube/native audio/video playback out of `VideoDetailPage.tsx`. Playback rate controls and uploaded playback source switcher still remain in `VideoDetailPage.tsx` and are passed as `playbackControls` pending the next Phase 8 increment.

### Task 8.2 — Extract video meta

- [ ] Create `frontend/src/store/useVideoStore.ts` for `video`, `segments`, `loading`, fetch logic.

### Task 8.3 — Reduce VideoDetailPage to shell

- [ ] VideoDetailPage becomes ~100 lines: reads `useVideoStore`, renders tab nav, renders active tab component.
- [ ] Commit.

---

## Phase 9 — Settings.tsx (4,152 lines)

Settings.tsx has 6+ independent sections that can each be isolated.

### Task 9.1 — Audit Settings sections

- [ ] Grep for section headings or `activeTab` equivalents in Settings.tsx to list all sections.

### Task 9.2 — Per section: store + component + integrate

For each section (repeat pattern):
- [ ] Create `frontend/src/store/useSettings<Section>Store.ts`
- [ ] Create `frontend/src/pages/settings/<Section>Settings.tsx`
- [ ] Replace inline JSX in Settings.tsx
- [ ] Run `npx tsc --noEmit`, commit

---

## Phase 10 — JobQueue.tsx (2,220 lines)

### Task 10.1 — Create `useJobQueueStore.ts`

- [ ] Grep for useState in `frontend/src/pages/JobQueue.tsx` to list all state.
- [ ] Move fetch logic and filter/sort state into `frontend/src/store/useJobQueueStore.ts`.

### Task 10.2 — Extract `JobRow.tsx`

- [ ] Identify the per-job render block and move it into `frontend/src/components/jobs/JobRow.tsx`.

### Task 10.3 — Integrate

- [ ] Replace inline JSX, remove state, run `npx tsc --noEmit`, commit.

---

## Quick Reference — File Map

```
frontend/src/
  lib/
    api.ts                    — Axios instance (import as `api`)
    formatters.ts             — formatTime, formatViewMetric ✅
  store/
    useCloneStore.ts          ✅ Phase 1
    useYoutubeStore.ts        Phase 2
    useSpeakersTabStore.ts    Phase 3
    useClipsStore.ts          Phase 4
    useWorkbenchStore.ts      Phase 5 (shared)
    useCleanupStore.ts        Phase 5
    useReconstructionStore.ts Phase 6
    useTranscriptStore.ts     Phase 7
    usePlayerStore.ts         Phase 8
    useVideoStore.ts          Phase 8
  pages/
    video/
      VideoDetailPage.tsx     shrinks each phase
      tabs/
        CloneTab.tsx          ✅ Phase 1
        YoutubeTab.tsx        Phase 2
        SpeakersTab.tsx       Phase 3
        ClipsTab.tsx          Phase 4
        CleanupTab.tsx        Phase 5
        ReconstructionTab.tsx Phase 6
        TranscriptTab.tsx     Phase 7
        OptimizeTab.tsx       Phase 7
    settings/
      <Section>Settings.tsx  Phase 9
  components/
    video/
      VideoPlayer.tsx         Phase 8
      CloneWorkbenchPanel.tsx ✅ already extracted
      EpisodeChatWorkbench.tsx ✅ already extracted
    jobs/
      JobRow.tsx              Phase 10
  types.ts                   — shared types (do not split)
```

---

## Resuming After a Context Limit

If a new agent picks up this plan mid-phase:

1. Read this file completely.
2. Check the Status Dashboard table to see which phase is in progress.
3. Read `frontend/src/pages/video/VideoDetailPage.tsx` lines 59–230 to see which useState declarations remain — this tells you exactly how far extraction has gotten.
4. Run `cd frontend && npx tsc --noEmit` — if errors exist, fix before proceeding.
5. Continue from the first unchecked `- [ ]` in the active phase.
