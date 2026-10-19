# Handoff: MOSS bounded planner complete

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Tasks 1–9 complete; plan In progress and spec In Review. Task 9 checkpoint
subject: `feat(moss): plan bounded sample chunks and type resource failures`.
The prior design baton is preserved unchanged at
`archive/2026-10/2026-10-18-moss-engine-batching-design.md`.
The pipeline still has its initial whole-recording guard: task 9 supplies the
building blocks and does not yet fix the owner's reported long-job rejection.

## Next step

Task 10: open `backend/src/services/ingestion/pipeline_transcribe.py` and wire
bounded normalized frame streaming and durable per-chunk checkpoints. Owner
direction already authorizes tasks 10–13; no repeated intent approval needed.

## Verification state

Packed isolated suite runs from backend with DB_PROVIDER=sqlite and
DATABASE_URL=sqlite:// set before imports:
`.venv/Scripts/python.exe -m pytest src/test_moss_chunks.py src/test_moss_runtime.py src/test_moss.py -q`.
Final result and scoped Ruff/SDD/whitespace/hooks recorded in the plan.
No production settings, recordings, DB, external inference or GPU allocation.

## Blockers

None. Tasks 10–13 remain pending; no pipeline/identity/UI code changed here.

## Diagnosis worth keeping

`moss_chunks.next_chunk(total_frames, cursor=0, ceiling=INITIAL_FRAME_CEILING,
speech=None)` returns immutable `Chunk(start, end, reason)` or None at EOF.
Coordinates are absolute normalized 16kHz PCM sample indices; ceiling defaults
to measured 120s and cannot exceed the 5400s worker-request ceiling. Optional
speech hints use installed faster-whisper `{start,end}` sample dictionaries,
ordered, covering current-window speech and offset into absolute coordinates.
Task 10 runs VAD on bounded waveform windows with installed default VadOptions;
task 9 neither loads ML nor runs VAD. None hints mean unavailable, [] means
VAD found no speech. Reasons are eof, vad_gap, hard_cut, vad_unavailable.
Gap ends select cuts only; every silence sample remains owned exactly once.

`split_chunk(chunk)` returns two contiguous positive smaller Chunks, with
resource_split reason; raises ValueError at one frame. No guessed retry count.
Task 10 invokes it only for `MossResourceError` from transcribe, not general
MossRuntimeError, and carries the successful reduced ceiling to later input.
MossCudaOOMError and MossContextBudgetError preserve type after verified worker
cleanup. A cleanup failure becomes ordinary MossRuntimeError so it cannot split.
Private worker codes cuda_oom/context_exhausted arise from actual CUDA OOM type
or pinned context budget exhaustion. Load/probe errors, malformed generation,
early EOS, impossible over-budget output, invalid clocks/JSON and crashes do
not authorize splitting. Per-request moss-result-v1 stays unchanged.

## Next worker prompt

Fresh task 10 only, resume:no. Request narrow repack for pipeline_transcribe.py,
pipeline.py and test_moss_pipeline.py plus SDD baton artifacts. Use task-9 APIs
above, durable native per-chunk raw evidence and exact frontier/source identity.
Preserve prior committed chunks on failure/cancellation; legacy complete reuse
must remain. Read existing authorities after preflight. No DB schema changes,
heuristic speaker/text seam merging, dependency upgrades, production jobs,
push/deploy or final archive. Keep baseline unrelated dirty paths untouched.
