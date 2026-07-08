# Ingestion service split — status and architecture

`backend/src/services/ingestion.py` was a single 17,756-line module holding
`IngestionService` (422 methods) plus its exceptions and runtime config. It
is now the package `backend/src/services/ingestion/`, split by domain into
mixin classes that `IngestionService` composes. **The split is complete**
(phases 1–9, July 2026).

## Architecture

`service.py` is the composition root: it holds `__init__` (all instance
state) and the pipeline focus-mode accessors, and assembles the class:

```python
class IngestionService(
    DiarizePhaseMixin, TranscribePhaseMixin, PipelineMixin, ClipsMixin,
    YoutubeMetadataMixin, LlmProviderMixin, FunnyMomentsMixin,
    ReconstructionMixin, AudioCleanupMixin, MediaFilesMixin,
    ChannelSyncMixin, YoutubeDownloadMixin, SpeakerIdentityMixin,
    JobLifecycleMixin, TranscriptionEngineMixin, CudaMemoryMixin,
    TranscriptQualityMixin,
):
```

Method names are unique across all mixins (verified at extraction time), so
MRO order carries no behavior. Mixins freely call each other's methods via
`self` — the split changes code location only, not the object model.

| Module | Mixin | Owns |
|---|---|---|
| `runtime.py` | — | Directory constants, env helpers, job-type sets, **rebindable DB globals** |
| `exceptions.py` | — | `JobPaused/JobCancelled/JobDeferred/JobNoticeException` |
| `service.py` | — | `__init__` state, focus-mode accessors, class assembly |
| `transcription.py` | TranscriptionEngineMixin | Whisper backends, Parakeet, language routing, model loading; `TransformersWhisperCompatModel` |
| `transcript_quality.py` | TranscriptQualityMixin | Consolidation, repair, gold windows, evaluation, campaigns, runs, rollback |
| `reconstruction.py` | ReconstructionMixin | Conversation-reconstruction TTS pipeline + workbench |
| `job_lifecycle.py` | JobLifecycleMixin | Job claim/progress/stage, worker loops, dispatch handlers, orphan cleanup, redo backups |
| `channels.py` | ChannelSyncMixin | Channel CRUD/refresh, YouTube/TikTok metadata, monitor loop, backfill |
| `cuda_memory.py` | CudaMemoryMixin | CUDA health/reset/auto-restart, memory accounting, model residency |
| `llm.py` | LlmProviderMixin | Provider selection, VRAM guard, text generation (all providers) |
| `pipeline.py` | PipelineMixin | `process_video`, download phase, checkpoints, transcript persistence |
| `pipeline_transcribe.py` | TranscribePhaseMixin | `_process_transcribe_phase` |
| `pipeline_diarize.py` | DiarizePhaseMixin | `_process_diarize_phase` + pyannote progress/adaptive batch |
| `media_files.py` | MediaFilesMixin | ffmpeg/ffprobe, audio paths, download, validation, slicing |
| `funny_moments.py` | FunnyMomentsMixin | Laughter detection + explanation |
| `clips.py` | ClipsMixin | Clip creation, captions, export rendering |
| `audio_cleanup.py` | AudioCleanupMixin | VoiceFixer runs + cleanup workbench |
| `youtube_metadata.py` | YoutubeMetadataMixin | Humor summaries, chapters, description suggestions |
| `youtube_download.py` | YoutubeDownloadMixin | yt-dlp auth/notices, placeholder captions |
| `speaker_identity.py` | SpeakerIdentityMixin | Embedding profiles, match cache, identification |
| `__init__.py` | — | Public re-exports (`IngestionService`, exceptions, dirs) |

## Rules that keep this working

1. **DB access goes through `runtime`.** Service code writes
   `Session(runtime.engine)` — never `from ...db.database import engine`.
   The call-time attribute lookup gives tests exactly one patch point:
   `monkeypatch.setattr(ingestion_runtime, "engine", test_engine)` and
   `...("create_db_and_tables", lambda: None)` work no matter which module
   the code lives in. Tests import it as
   `from src.services.ingestion import runtime as ingestion_rt`.
2. **Mixin modules never import `service.py`** (circular). Shared
   module-level things live in `runtime.py` / `exceptions.py`.
3. **Instance state is declared only in `service.py`'s `__init__`.** A mixin
   needing new state adds it there, not in its own module.
4. **New methods must not reuse a name defined in another mixin** — name
   collisions across bases silently resolve by MRO. Check with:
   `grep -rn "def <name>" backend/src/services/ingestion/`.
5. **`__file__`-relative paths**: package modules are one level deeper than
   the old `ingestion.py`; always anchor on `runtime.BACKEND_DIR` instead of
   `Path(__file__).parent...` chains.

## Verification loop (used after every phase; still the standard)

```
.venv/Scripts/python -m ruff check .                    # F821 catches names an
                                                        # extraction missed
.venv/Scripts/python -c "from src.main import app; print(len(app.routes))"  # 248
.venv/Scripts/python -m pytest src -q                   # 42 tests
```

## How it was done (for the next split of this size)

Line-range hand cuts were rejected in favor of an AST-scripted extractor:
parse the file, map each method (with decorators) to a target module via an
explicit range table, emit `class XxxMixin:` files with a superset import
header, delete methods bottom-up, register the mixin base + import in
service.py. Then `ruff check --fix` (F401) strips each file's unused header
imports and plain `ruff check` (F821) proves no method lost a name it needs.
Route-count + full pytest close each phase before commit.

## Possible follow-ups (not scheduled)

- The per-module import headers were pruned mechanically; a human pass could
  group/sort them.
- Cross-mixin `self._x()` calls are invisible to static tooling; if coupling
  becomes a problem, introduce `typing.Protocol` interfaces per mixin or
  promote heavy domains (reconstruction, transcription) to standalone
  services owning their own state.
