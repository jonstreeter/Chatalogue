# Handoff: optional MOSS engine — task 1 blocked (superseded by authorized EOF refinement)

**Date:** 2026-10-18
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`
**Checkpoint commit:** `HEAD` of this task-1 checkpoint; resolve with `git log -1`.
The phase result carries its exact resulting SHA; no self-hash/amend required.

## Status

Owner approved the reviewed spec and plan, segment-only timestamps, 90-minute
single-input limit and whole-input retries. Original approval is not pending.
The missing Super Looper workflow was found in the owner's GitHub repository
`jonstreeter/dev-process-kit` at `docs/dev/workflows/super-looper.md`; that earlier
blocker is resolved. Tasks 0/8 complete. Task 1 ran and is now blocked on a real
strict-validation failure, not on missing instructions or CUDA incompatibility.
No product code, production settings, DB, jobs or recording reprocessing changed.

Native private Windows CUDA BF16/SDPA load/decode works. Exact source/model and
all package pins, eight complete ordinary-fixture decode results, cold/warm
measurements, VRAM and process exit are in
`docs/research/2026-10-18-moss-runtime.md`. Two successful processes exited 0 and
released VRAM. The synthetic overlap probe generated native overlapping turns,
but its last end clock is 9.01 seconds for exactly 9.00 seconds of source audio.
The approved strict bounded-clock check rejects it. One diagnostic rerun changed
only raw-output capture and reproduced the failure; both exited 1 and released
VRAM. No tolerance, clamping, output rewriting, CPU/remote fallback or workaround.
No further inference after the diagnostic run. Tasks 2–8 remain unstarted.

## Blocker and owner decision

Root cause observed: model-generated end timestamp exceeds synthetic source
length by 0.01s. CUDA compatibility and EOS/full generation passed; the complete
native overlap result fails the approved output-bound contract. General accuracy
or failure on other recordings is not established. Changing timestamp policy
would change a contract depended on by downstream validation (escalation trigger).

Choose in plain words before more work:

- Spec+plan: explicitly review timestamp quantization/bounds policy and tests.
- Labeled stopgap: explicitly authorize a narrowly scoped temporary policy.
- Patch anyway: explicitly authorize proceeding despite the present gate.

No option is selected by this handoff. Original approval does not authorize a
clock tolerance or further attempts after this concrete failure. Do not start
product implementation while this choice is pending.

## Verification state

- Private Torch CUDA import: PASS, Torch 2.10.0+cu128 / CUDA 12.8,
  Transformers 5.6.0 / RTX 5090.
- Complete ordinary real-audio probes: PASS, eight outputs with EOS and full
  strict parsing; cold/warm 10/30/120s evidence in report.
- Native overlap with strict source bounds: FAIL, 9.01 > 9.00, reproduced.
- Process exit/VRAM release: PASS, successful and failed probe processes exit.
- `python scripts/check_sdd_docs.py validate`: PASS.
- `git diff --check`: PASS.
- Application suites omitted: research-only checkpoint, no application changes.

## Retained scratch / safety

`backend/runtime/moss/` is ignored scratch: private venv, pinned source/model,
`dependencies.txt`, runnable `probe.py`, `make_overlap.py`, 10/30/120s crops,
synthetic fixtures, per-decode/raw JSON, logs and `resume-state.txt`.
Source research WAV remains unchanged. Freeze and runnable scripts are embedded
in the committed research report so ignored artifacts are not needed to inspect
this failure. Both original uv installers completed; do not restart setup.
No owned inference remains running at handoff. Unrelated GPU/Python/Ollama
processes were never killed. No models/venvs or unrelated changes staged.

Baseline main HEAD before this checkpoint was 534a7c6c. Preserve unrelated dirt:
`docs/dev/README.md`, `.kilo/`, `docs/CONSTITUTION.md`, `docs/dev/audits/`,
`docs/dev/handoffs/2026-10-14-asr-benchmark.md`, semantic-search spec/plan,
approved MOSS spec, `docs/dev/workflows/tech-debt-audit.md`, `docs/templates/`.
Only this handoff, the MOSS plan and research report belong to this checkpoint.

## Next worker prompt

Resume **Native Windows NVIDIA CUDA feasibility**, task **1**, `resume: yes`,
after the owner answers the contract/gate decision. Read this handoff and
`docs/research/2026-10-18-moss-runtime.md`; reuse scratch and immutable pins.
Do not restart completed downloads, relax output validation, invent timestamps
or start task 2 without the explicit owner decision. If the owner chooses a
contract change, conductor must update authority/ownership before dispatch.
