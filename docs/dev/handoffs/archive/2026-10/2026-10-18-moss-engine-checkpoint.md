# MOSS whole-input checkpoint — task 5 handoff

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task 5 complete; 5/8. Checkpoint subject:
`feat(moss): persist whole-input native transcription checkpoints`.
Plan remains In progress. Owner `dd` authorizes continued sequential work.

## Next step

Obtain task-6 context pack and preflight its paths. Native identity matching
and overlap-safe samples belong in
`backend/src/services/ingestion/pipeline_diarize.py`; use normalized endpoints
and preserve labels through standard save/cleanup.

## Verification state

Packed backend venv SQLite suite:
`python -m pytest src/test_moss_pipeline.py src/test_moss.py src/test_phonon.py -q`
→ 116 PASS, 45.34s. Scoped Ruff, SDD validation, whitespace and required commit
gates recorded in plan. Temporary SQLite/files only; no production DB/settings,
jobs/media or GPU inference. Actual inference/full regression remains task 8;
browser controls remain task 7.

## Blockers

None. No push, deployment, installer refresh or final archive closeout.

## Diagnosis worth keeping

MOSS normalizes one complete PCM16 mono 16kHz WAV, checks exact sample dimensions
and <=5400 seconds before inference, releases competing models, then strictly
validates runtime output against manager-owned dimensions. Existing raw JSON adds
`moss_result` (strict moss-result-v1, intact), `moss_runtime` (model/source/runtime
provenance), `moss_source` (source size/mtime_ns), plus float adapter `segments`.
Segment objects carry `.speaker` native S labels; words=null. Strict native result
owns raw decimal tokens, overlaps/order, rational duration and corrections;
float adapters and legacy RTTM are never their source.

Both reuse entry points validate native evidence and restore labels without
cumulative correction. Source stat mismatch rejects reuse with deliberate
retranscription guidance. Completed legacy cache retains existing reuse semantics.
Incomplete inference/cancellation has no partial; retry runs the whole input.
Other-engine partials/orphan records refuse MOSS unless force was explicit.
Force bypasses complete cache and removes relevant partials; prior complete raw
checkpoint survives failed inference/publication and is replaced only after a
fully validated complete generation. Context exhaustion/malformed output fails.

Task 6 owns native identity, overlap-safe enrollment, persistence and cleanup
integration. Task 5 does not claim those paths finished.

## Workspace and next worker rules

Same checkout; no new worktree/stash. Preserve unrelated baseline dirt from the
context pack. Stage explicit phase paths, inspect cached and committed names,
use required hooks without bypass. Fresh ownership pack before next edits.
