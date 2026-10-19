# MOSS managed worker — task 3 handoff

**Date:** 2026-10-19
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**State:** Task 3 complete; 3/8. Plan remains In progress; task 4 awaits owner Do next.

## Completed

Added `moss_runtime.py`, standalone `moss_worker.py`, exact tested dependency
wheel lock and isolated runtime tests. Private immutable model/helper/license
snapshots, verified wheel/model caches, interprocess setup serialization,
staged manifest publication and interrupted-install recovery. Setup and model
workers use unchanged `OwnedPhononProcess`; the actual interpreter handshakes
its PID before setup commands, with verified owned exit on cancellation.
No generic engine framework or shared Phonon edits.

One serialized versioned JSONL worker; async observable load, read-only status,
busy unload refusal, shutdown/setup race prevention, parent-exit cleanup,
sample-exact canonical WAV validation and 5400-second rejection. CUDA BF16 SDPA,
remaining-context generation budget, real completion metadata and independent
strict result validation. `transcribe(audio_bytes)` returns sibling `result`
and `runtime` objects: provenance stays outside `moss-result-v1`.

## Verification and evidence

- Post-pin SQLite-isolated `python -m pytest src/test_moss_runtime.py src/test_moss.py -q`
  from backend: **110 PASS** (25.66s). Includes real Windows venv child/parent
  exit and setup cancellation with fake model workers; no production data/jobs.
- Scoped Ruff PASS; SDD validation and diff whitespace checked at checkpoint.
- Actual managed installer and real GPU load/decode/unload PASS. Published
  version `214ed56931e0623b6d49672627c2336d0db877566235622c1b45a5e1d5f984ff`.
  Worker19972 / launcher48444, RTX5090, Torch2.10.0+cu128, CUDA12.8, BF16 SDPA.
  Retained 144000-sample fixture: four segments/three overlaps, 200/108 tokens,
  EOS151645, budget130872, exact index3 raw9.01→9 correction; independent
  backend validation PASS. Load3.575s / fixture decode6.922s.
- Actual interpreter exit verified; launcher0, handles cleared, temp audio empty.
  Global VRAM3659→6059→3647MiB has unrelated-desktop caveat. Local peak
  allocated/reserved1,874,400,256/1,952,448,512 bytes.
- Offline verified installed-file/67-wheel reuse PASS (5.115s), zero downloader
  or subprocess calls, no model load. This is cached reuse, not a second reinstall.
- Initial universal SoundFile wheel failed the native gate before publishing any
  manifest. Corrected same-version Windows wheel passed normal staged setup.
  Failed/corrected measured runs2009.542/1075.370s exclude preparatory cache copies.

Full task-3 evidence and commands: `docs/research/2026-10-18-moss-runtime.md`.
Ignored evidence/scripts remain under `backend/runtime/moss/task3/`, including
`managed-proof-final.json`, `prove-final.py`, `offline-proof.json`, setup/worker
logs and measured installer I/O snapshots. Original task-1 environment/model/source
were read only. Model/source caches were seeded from individually verified bytes;
Torch was actually downloaded. No accuracy claim or production integration.

## Important next-phase boundaries

Readiness is a real **complete-generation smoke probe**, not validated silence
transcription. Actual one-second silence produced a 23-token hallucinated apology
without a terminal clock. `runtime.readiness_probe` retains the raw output and
completion; future Load/Test API/UI must describe inference readiness and must
not claim a valid silence transcript. Real transcription still rejects malformed
output strictly. Preserve raw token/clock/correction evidence for task 5.

Task 4 is next: app manager/GPU lifecycle, settings actions and explicit multilingual
routing. No task-4 app/settings/pipeline/UI changes have started. Existing defaults,
engines, DB schema, voice identities and production media/jobs remain unchanged.
No push, deploy, installation refresh or archive closeout was performed.

## Workspace

Same main checkout; no new worktree or stashes. Phase commit contains only the
task-3 allowlist plus this baton and archived parser baton. Unrelated existing
dirty docs/tooling remain untouched: `docs/dev/README.md`, `.kilo/`,
`docs/CONSTITUTION.md`, `docs/dev/audits/`, the ASR handoff, semantic-search
plan/spec, tech-debt workflow and `docs/templates/`. Do not stage or revert them.

## Next worker prompt

Resume the MOSS plan at task 4 only after owner Do next. Run session status and
contract preflight for that phase; read its authorities. Reuse the verified
managed runtime and evidence above. Keep raw strict result separate from runtime
provenance; readiness smoke semantics and segment-only/90-minute capability
notices are explicit. Do not re-run the closed feasibility/timing benchmarks or
change pinned ML dependencies. Obtain a fresh task-4 context pack and allowlist.
