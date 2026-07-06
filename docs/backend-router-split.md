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
| `src/deps.py` | `get_session` dependency, `get_ingestion_service()` lazy accessor |
| `src/job_utils.py` | Job-state helpers + `PIPELINE_ACTIVE_STATUSES` constants |
| `src/env_utils.py` | `ENV_PATH` + `_set_env_persist` (.env persistence) |
| `src/main.py` | Everything else (~15.7k lines, shrinking) |

Routers are registered at the **end** of `main.py` via `app.include_router(...)`.

## Remaining domains to extract (largest first)

`/videos` (67 routes), `/avatars` (33), `/channels` (29), `/system` (21),
`/speakers` (16), `/clips` (10), `/share`+`/auth` (7), `/episode-chat` (6),
`/youtube` OAuth cluster (5), transcript-optimization (11), search (3).

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
