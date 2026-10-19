# MOSS task8 verification — blocked checkpoint

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task8 incomplete;7/8 complete. Plan Blocked, spec not Shipped. No product or
test changes. Static checks and full backend tests passed; full browser run
stopped on first existing channel selector failure. No real task8 inference
was started. Prior task7 baton archived unchanged as
`archive/2026-10/2026-10-18-moss-engine-ui.md`.

## Next step

Conductor must obtain explicit ownership/approval for the unrelated test-only
selector repair at frontend/e2e/channel-detail.spec.ts:18. Proposed change:
`getByRole('link', { name: 'Open', exact: true })`. Do not silently broaden MOSS
scope. Root belongs to channel UI/test subsystem; options are spec+plan,
labeled stopgap, or patch anyway with an explicitly narrow repack. After that,
resume task8 verification; do not restart implementation tasks1–7.

## Verification state

- frontend `npx tsc -b` PASS.
- root `backend/.venv/Scripts/python.exe -m ruff check .` PASS.
- frontend `npm run lint` PASS,0 errors/43 existing warnings.
- backend `.venv/Scripts/python.exe -m pytest src -q --basetemp=runtime/moss/task8/pytest-tmp`
  PASS252,1 warning,124.31s. Before imports PYTHONPATH includes task8 bootstrap
  and backend; DB_PROVIDER=sqlite,DATABASE_URL=sqlite://. Scratch sitecustomize
  disables dotenv, redirects runtime/data/env paths and denies production
  backend/data or .env access. See research Task8 checkpoint for exact command
  and hashes. No tests skipped/assertions changed.
- frontend `npm run test:e2e -- --config F:/Projects/PorchtimeIndex/backend/runtime/moss/task8/playwright.config.cjs`
  FAIL:19 selected,1 pass,1 fail,17 not run,10.7s. Installed Chrome,workers1,
  maxFailures1 stops at first failure; no test excluded. Port8000 bind free
  before owned fixture API; owned Vites5173/5175 polling/scratch envDir.
  Synthetic API process/scan transitions create no jobs and make no external
  calls. Audio-download PASS is mock browser evidence, not downloader proof.
- SDD validation and whitespace checked at checkpoint; required commit hooks
  apply. No push/deploy/archive completion.

## Blockers

frontend/e2e/channel-detail.spec.ts:18 partial accessible-name Open matches
three links: two aria-label Open Chatalogue fixture and exact Open. Strict
locator violation before channel navigation. This is a concrete existing
regression outside MOSS edit ownership. Do not rerun unchanged failing phase
or workaround product markup. No further full-check success is claimed.

## Diagnosis worth keeping

Backend import defaults read .env/create backend/data directories; setting
SQLite alone does not isolate files. Read-only safety expansion included
src.paths,src.env_utils,services.logger,services.external_share,routers.system/
avatars and exact frontend channel/speaker/API consumers. Bootstrap source
rewrites only directory/env constants in memory; product files unchanged.

Ignored evidence in backend/runtime/moss/task8/: sitecustomize.py,pytest.log,
e2e.log,fixture_api.py,fixture-api.jsonl,vite.config.mjs,playwright.config.cjs,
and e2e-results/channel-detail-Channel-Det-90ab1-s-export-and-delete-buttons-chromium/
(test-failed-1.png,error-context.md). API/Vite processes stopped via owned
sessions; no global process-name kill. No production settings/media/DB reads.
Warnings: TRANSFORMERS_CACHE deprecation,post-exit swigvarlink warning,Node
module.register deprecation,NO_COLOR/FORCE_COLOR notice,old Browserslist data.

Next worker prompt: task8 only, resume:yes. Keep task3 manager environment and
original evidence immutable. After approved selector fix and full checks,
prove actual isolated MOSS short9s/ordinary10s/long120s app phases, real GPU
voice embedding/matcher named profiles across two fixture recordings with
changed labels, native overlap/clean enrollment preservation, same staged
MOSS child reuse, exit-before-voice barrier,sequential reload timing and app
shutdown. None is yet task8 evidence. Never use production recordings/.env,
spoof embeddings,claim accuracy or extrapolate120s to90min. Preserve baseline
unrelated dirty paths. PLAN_COMPLETE only when full acceptance actually passes;
plan must not become Done/spec Shipped in this phase.
