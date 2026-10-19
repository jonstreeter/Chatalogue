# MOSS UI — task 7 handoff

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task7 complete; 7/8. Checkpoint subject:
`feat(moss): expose managed engine controls and verify segment playback`.
Plan remains In progress. Owner dd authorizes continued sequential work.

## Next step

Obtain task8 context pack for isolated actual app/runtime, cross-recording
identity, GPU release and full regressions. Do not mark plan Done/spec Shipped.

## Verification state

From frontend: `npx tsc -b` PASS; `npm run lint` PASS (0 errors, 43 existing
warnings). With PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH set to installed
C:/Program Files/Google/Chrome/Application/chrome.exe,
`npm run test:e2e -- e2e/moss.spec.ts e2e/phonon-settings.spec.ts --workers=1`
PASS: 5 Chromium tests, 30.1s. All API requests intercepted before navigation.
Real native audio decodes synthetic15-second PCM WAV fixture. Clicking each
words=null segment verifies near-start seek and playing state, both named
nested overlapping segments highlighted, and no word spans. Native seek uses
100ms browser tolerance. Existing Phonon lifecycle/save/routing badge PASS.

`python scripts/check_sdd_docs.py validate` and `git diff --check` PASS;
required hooks run with Git usr/bin on PATH. Screenshots visually reviewed in
ignored output/playwright/: moss-settings.png, moss-wizard.png, moss-queue.png,
moss-segment-playback.png. Nested settings scroll means page screenshot captures
selection/routing; assertions prove capability notice/progress/actions further
down. No generated artifacts committed. No model-quality claim.

## Blockers

None. Task-owned Vite servers stopped before return. No push/deployment/installer
refresh. Actual app/runtime evidence and full checks remain task8.

## Diagnosis worth keeping

MOSS reuses Phonon lifecycle panel, choosing moss_action for status/load/unload.
No URL/key/server controls. Before saving, disclose multilingual segment timing,
no word highlighting, initial90-minute cap and whole-input restart. Wizard MOSS
setup skips legacy save-before-test; explicit Save/finish persists selection.
Wizard blanket word-timing claim excludes MOSS; routing text says unused by MOSS.

Missing bundled Playwright browser resolved using installed Chrome env override.
Vite watch hit Windows EBUSY on generated playwright-report/index.html;
CHOKIDAR_USEPOLLING=true on owned Vite5175 resolves it. Conductor approved5175
because existing Phonon tests hardcode it. Initial API fixtures lacked db-health
and transcript-quality shapes and crashed rendering; corrected fixtures. PCM
fixture supports Range with206. No product playback edit needed.

## Workspace and next worker rules

Preserve unrelated baseline dirt. Task7 owns Settings.tsx, SetupWizard.tsx,
JobQueue.tsx, e2e/moss.spec.ts, plan and baton. Prior baton preserved unchanged:
docs/dev/handoffs/archive/2026-10/2026-10-18-moss-engine-identity.md.
Approved read-only playback repack opened VideoDetailPage.tsx, useVideoStore.ts,
usePlayerStore.ts, components/video/VideoPlayer.tsx and pages/video/useVideoSeek.ts.
No production settings/jobs/media requests or backend/model-dependency edits.

Next worker prompt: task8 only, fresh context pack/preflight, resume:no. Plan
and spec above. UI fixtures establish client behavior only; retain distinction
from actual runtime, cross-recording identity and GPU/full regression proof.
