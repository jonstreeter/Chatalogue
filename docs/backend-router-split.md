# Backend router split — status and continuation guide

`backend/src/main.py` began as a single file holding all API routes (242 of
them) plus their helpers. It is being split into domain routers under
`backend/src/routers/`. This documents the pattern so the remaining domains
can be extracted the same way.

## Current state

| Module | Owns |
|---|---|
| `src/routers/jobs.py` | 14 job-queue + pipeline-focus endpoints |
| `src/routers/settings.py` | 12 settings, LLM-provider-test, and Ollama endpoints |
| `src/routers/system.py` | 21 system endpoints: CUDA health, component installers, version/restart/update, worker status |
| `src/routers/youtube_auth.py` | 6 YouTube OAuth + data-api-test endpoints (owns the pending-state dict) |
| `src/routers/speakers.py` | 17 speaker profile/sample/thumbnail/merge endpoints |
| `src/routers/channels.py` | 17 core channel CRUD/ingest/refresh/upload/export endpoints |
| `src/routers/clips.py` | 14 clip CRUD/export/YouTube-upload endpoints |
| `src/routers/share.py` | 6 external-share session endpoints |
| `src/routers/episode_chat.py` | 9 episode-chat thread/message endpoints (+ serialization helpers) |
| `src/routers/transcript_ops.py` | 29 transcript quality/run/evaluation/campaign/repair/rebuild endpoints |
| `src/routers/episode_clone.py` | 6 episode cloning endpoints |
| `src/routers/video_media.py` | 26 media, VoiceFixer, cleanup- and reconstruction-workbench endpoints |
| `src/routers/videos.py` | 19 core video endpoints (list, process, funny moments, YouTube-AI) |
| `src/routers/search.py` | 5 keyword/semantic search + semantic-index endpoints |
| `src/routers/segments.py` | 4 segment editing endpoints |
| `src/routers/video_maintenance.py` | 7 purge/redo/consolidate/mute endpoints |
| `src/deps.py` | `get_session` dependency, `get_ingestion_service()` lazy accessor |
| `src/job_utils.py` | Job-state helpers + `PIPELINE_ACTIVE_STATUSES` constants |
| `src/env_utils.py` | `ENV_PATH` + `_set_env_persist` (.env persistence) |
| `src/paths.py` | Data/runtime directory constants (`IMAGES_DIR`, `AVATARS_DIR`, ...) |
| `src/youtube_utils.py` | YouTube API/OAuth URL constants |
| `src/video_utils.py` | Shared video helpers: `_enqueue_unique_job`, queue builders, remote info fetch |
| `src/main.py` | Avatars domain (33 routes + large helper web), YouTube metadata helpers, share/middleware helpers, app lifespan (~8.2k lines) |

Routers are registered at the **end** of `main.py` via `app.include_router(...)`.

## Remaining domains to extract (largest first)

Only `/avatars` (33 routes) remains, plus its large helper web
(personality scoring, dataset building, training orchestration —
several thousand lines). Budget a full session; consider moving the
helper web to `src/services/avatar_personality.py` rather than into the
router. After that, migrate remaining `_main().x` call sites
(`grep -rn "_main()." src/routers/ src/video_utils.py`) to direct
imports as helpers find permanent homes.

Caution from phase 8-9: never cut two ranges where one lies inside the
other — the bottom-up deletion shifts the outer range and eats an extra
line (a route decorator, in that case). The route-count assertion
(`len(app.routes) == 248`) catches this class of error; run it after every
extraction.

When a test fails after an extraction with
`module 'src.main' has no attribute '<route_fn>'`, update the test to import
the route function from its router module (helpers it monkeypatches on
`src.main` keep working — routers resolve them via `_main()` at call time).

## The extraction pattern

1. **Cut route blocks** out of `main.py` into `src/routers/<domain>.py` with
   `router = APIRouter()`, replacing `@app.` with `@router.`.
2. **Shared helpers** used by both the router and remaining `main.py` code
   move to a neutral module (`job_utils.py`-style) that imports only from
   `db`/`schemas` — never from `main`.
3. **`ingestion_service`** (mutable singleton owned by `main`'s lifespan) is
   reached via `get_ingestion_service()` from `deps.py`.
4. **Helpers still living in `main.py`** are reached through a transitional
   call-time accessor inside the router module:

   ```python
   def _main():
       from .. import main
       return main
   ```

   Use `_main().<helper>()` at call sites. This is deliberate: routers must
   NEVER import `main` at module level (circular import). As helpers migrate
   into services/util modules, replace `_main().x` with direct imports —
   `grep -rn "_main()." src/routers/` lists the remaining debt.
5. **Register** at the end of `main.py`:
   `from .routers import <domain> as <domain>_routes` + `app.include_router(...)`.

## Verification loop (all steps required)

```
.venv/Scripts/python -m ruff check backend      # F821 catches any name the
                                                # extraction missed — fix until clean
.venv/Scripts/python -c "from src.main import app; print(len(app.routes))"
                                                # route count must not change (248)
.venv/Scripts/python -m pytest src -q
```

The scripted extraction used for jobs/settings (anchor-asserted line-range
cuts) is the safest method at this file size; hand-editing invites silent
duplication. Fresh anchors must be re-checked each pass since line numbers
shift.

## Route-ordering caveat

FastAPI matches routes in registration order. Routers are included at the end
of `main.py`, so any route remaining in `main.py` that shares a path prefix
with an extracted parametric route (e.g. `/jobs/{job_id}/episode-clone` in
main vs `/jobs/{job_id}` in the router) resolves correctly today because
main's routes register first. When extracting a domain, move ALL routes that
share its parametric prefixes in the same pass, or re-verify ordering.
