# MOSS app lifecycle — task 4 handoff

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task 4 complete; 4/8. Checkpoint commit subject:
`feat(moss): wire app lifecycle and explicit engine controls`.
Plan remains In progress; task 5 awaits owner Do next.

One lazy manager per service under backend/runtime/moss and app temp/moss-ipc.
Settings supports explicit moss selection and status/load/unload actions. Status
never sets up/downloads/loads; Test does not save selection or enqueue media.
Ready describes complete-generation inference readiness, never silence accuracy.
Existing auto/default behavior is preserved. Explicit multilingual MOSS bypasses
Whisper language probing and rerouting. Full-input processing is task 5.

MOSS load releases Phonon/Whisper/Parakeet/voice models; voice-model load stops
MOSS before ML initialization and waits for owned exit. Other GPU loaders also
release MOSS. Busy user load/unload/manual purge refuses pipeline work; resident
MOSS is exempt from global free-VRAM-drop recovery. App shutdown independently
attempts MOSS cleanup even if Phonon cleanup fails.

## Next step

Obtain the task-5 context pack and preflight its paths. Open
`backend/src/services/ingestion/pipeline_transcribe.py` for full-input routing and
atomic native-label checkpoint/retry work; do not begin task 6 identity or task 7 UI.

## Verification state

Packed SQLite suite, scoped Ruff, SDD validation, whitespace and commit gates
are recorded in the task-4 plan verification log. All controls/settings tests use
fake managers, temporary .env and fixture environment/DB; shutdown uses a scratch
backend lock and no-op workers. No production settings/jobs/media or GPU benchmark
was used. Actual app-phase CUDA handoff remains task 8; browser controls task 7.

## Blockers

None. No push, deployment, installer refresh or final archive closeout performed.

## Diagnosis worth keeping

Task-3 readiness remains complete-generation smoke only: one-second silence
produced a malformed 23-token hallucinated apology. Raw evidence remains under
runtime.readiness_probe; actual transcription strictly validates moss-result-v1.
MossRuntime.transcribe returns sibling result and runtime/provenance objects.
Preserve raw decimal tokens, exact rational source duration and moss-eof-v1
correction evidence; never cumulatively renormalize on checkpoint retries.

Task 5 must add whole-input (<=5400s) transcription and atomic completed raw
checkpoint, reject mixed-engine partials, preserve old complete-cache reuse,
provenance and native speaker labels, and honor forced retranscription. Task 6
owns identity matching/overlap-safe samples and standard save/cleanup. No DB schema
change, generic engine framework, default change, pin/dependency change or benchmark
rerun. Existing task-3 GPU/process proof is archived in the managed-worker baton;
source evidence remains docs/research/2026-10-18-moss-runtime.md and ignored runtime.

## Workspace and next worker rules

Same main checkout; no new worktree or stash. Preserve unrelated dirt:
docs/dev/README.md, .kilo/, docs/CONSTITUTION.md, docs/dev/audits/, ASR handoff,
semantic-search plan/spec, tech-debt workflow and docs/templates/. Stage explicit
phase paths only, inspect cached names and committed names, use required hooks;
no bypass, push or deployment. Get a fresh phase allowlist before edits. Run
session status and preflight; read returned contracts/refinements in order.
Next worker resumes task 5 only after owner Do next; no task-5 product edits exist.
