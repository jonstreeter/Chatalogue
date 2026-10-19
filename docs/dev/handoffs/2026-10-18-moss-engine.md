# MOSS native identity — task 6 handoff

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task 6 complete; 6/8. Checkpoint subject:
`feat(moss): preserve native speaker identities with clean voice samples`.
Plan remains In progress. Owner `dd` authorizes continued sequential work.

## Next step

Obtain task-7 context pack and preflight its paths, starting with
`frontend/src/pages/Settings.tsx` MOSS engine controls. Task 7 owns browser
verification, task 8 actual app-phase runtime and identity verification.

## Verification state

`DB_PROVIDER=sqlite DATABASE_URL=sqlite:// backend/.venv/Scripts/python -m pytest src/test_moss_pipeline.py src/test_speaker_loading_optimization.py src/test_transcript_diarization_rebuild.py src/test_phonon.py -q`
(from backend) → **59 PASS**, 44.55s. Scoped Ruff, SDD validation and whitespace
PASS; required hooks recorded by commit. Deterministic embedding fixtures use
real matcher, temporary SQLite, profile enrollment and standard transcript save.
Initial run was 58 pass/1 failed because a fixture incorrectly expected the
existing selector to choose a 1s tail despite its 2s preferred sample. Corrected
and added direct subtraction assertion. Initial fixture also allowed cached CPU
embedding load after transcription released sentinels; guards now prohibit model
load. No production DB/settings/jobs/media changes or GPU inference.

## Blockers

None. No push/deployment/installer refresh/final archive closeout.

## Diagnosis worth keeping

Only revalidated `moss_result` evidence in the title-based raw checkpoint selects
native identity; current engine setting and stale RTTM cannot do so. Rebuild
segment objects from immutable evidence. Construct native annotation tracks,
stop owned MOSS, and use existing model loader with `embedding_only=True`.
Subtract every other label from each voice timeline before existing profile
sample selection. Real matcher and enrollment persist known channel names and
profiles across changed native labels; native turns bypass single-speaker mapping
and consolidation, preserving order, endpoints and words=null. Standard text
cleanup, file save and temp cleanup remain.

No usable clean embedding creates a distinct unresolved local Speaker with
pickle(None), no SpeakerEmbedding profile, and `moss_identity_warning` job payload.
Missing identity model raises before prior transcript rows are cleared and leaves
complete raw checkpoint intact. Task 8 must verify actual cross-recording voice
identity and owned GPU release; deterministic tests do not establish model quality.

## Workspace and next worker rules

Same checkout; preserve unrelated baseline dirt. Task-6 ownership includes
pipeline_diarize.py, speaker_identity.py, transcription.py (narrow loader repack),
test_moss_pipeline.py, plan and baton. Prior task-5 baton preserved unchanged in
`docs/dev/handoffs/archive/2026-10/2026-10-18-moss-engine-checkpoint.md`.
Task 7 expected files: Settings.tsx, SetupWizard.tsx, JobQueue.tsx, e2e/moss.spec.ts;
obtain full fresh pack before editing. Do not reinterpret segment clocks as word
clocks. Browser acceptance includes option/save/reload, setup/status/unload,
segment/90-minute notice, failure recovery, queue badge and no-word playback.
