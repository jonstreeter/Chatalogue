# Handoff: MOSS bounded recording batching

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Initial tasks 1–8 complete; owner-directed batching tasks 9–13 pending, 8/13.
Plan In progress; spec In Review. Owner reported a production job rejected by
the single-input guard and directed that model limits be accommodated with
chunks, rejecting another approval loop for that intent. This supersedes the
initial recording limit and whole-input retry; it does not authorize production
reprocessing. Initial proof is archived unchanged at
`archive/2026-10/2026-10-18-moss-engine-initial-complete.md`.

## Next step

Execute task 9 with a fresh Super Looper worker: pure bounded frame planner and
typed resource failures. Run preflight and read updated ADR/spec first. No
batching product code exists yet; do not claim the reported job is fixed.

## Verification state

Paired inline architect/planner/simplification/code review passed for updated
spec and appended plan: namespace local labels, preserve exact sample coverage,
retain raw per-chunk evidence, atomic durable frontier, legacy compatibility,
no heuristic text dedup, and measured request sizing. SDD validation and whitespace
checks pass. Initial backend 253/browser 19 and real CUDA results are historical
proof for unchanged behavior, not proof of batching.

## Diagnosis worth keeping

The app guard combines empty/format/duration checks and rejects the recording
before inference. The trace alone does not establish its sample count; no
production file or DB was inspected. The 5400-second request ceiling must not
be the recording ceiling. Initial chunk ceiling 120s is derived from the longest
tested CUDA app input. Installed faster-whisper supplies CPU Silero VAD and
bundled ONNX defaults; VAD suggests cuts only and never deletes source audio.
Forced contiguous cuts retain every native result without promising perfect
word recognition at seams. Only typed OOM/context failures bisect uncommitted
intervals; arbitrary malformed output still fails. Per-call moss-result-v1 stays
strict; recording-v2/partial changes belong to task 10, identities task 11,
UI task 12, real seam and >90-minute fixture proof task 13.

## Next worker prompt

Fresh task 9 only, resume:no. Write allowlist: moss_chunks.py, moss_worker.py,
moss_runtime.py, test_moss_chunks.py, test_moss_runtime.py, contracts.json and
plan/baton/archive artifacts. Read authority ADR-0002 and batching spec section;
read moss.py unchanged and installed VAD defaults as needed. Verify isolated
chunk/runtime/parser tests, scoped Ruff, SDD, whitespace and required hooks.
Do not begin pipeline/identity/UI changes early. No new framework, ML upgrades,
CPU/remote MOSS fallback or production settings/recordings. Preserve nine unrelated
dirty paths from session status. One worker at a time, scoped commits, no push,
deploy or final archive. Owner authorization covers execution of tasks 9–13.
