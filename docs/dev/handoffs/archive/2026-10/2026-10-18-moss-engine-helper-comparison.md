# Handoff: optional MOSS engine — helper comparison complete

**Date:** 2026-10-18
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task 1 remains in progress; 0/8 complete. Only the owner-requested controlled
comparison is complete. Official helper and probe produced identical complete
inputs, effective generation settings, full returned IDs, generated IDs and raw
text on the retained exact 9.00s fixture. Both emitted raw terminal 9.01s.
No EOF policy, parsing, normalization or product work ran in this phase.
No causal conclusion about the 9.01s overrun follows from this experiment.

## Evidence

`docs/research/2026-10-18-moss-runtime.md` records the exact command, runnable
comparison source, full generated IDs/raw text, five tensor hashes/shapes/dtypes,
load and inference times. Ignored runtime retains `compare_helper.py`,
`helper-comparison.json`, both complete tensor/output `.pt` captures, log and
child exit evidence. Shared model process 36916 completed both serial paths,
108 tokens each with EOS 151645 under budget 4096; prompt widths and attention
lengths all 200. Normal-exit marker, shell exit 0 and subsequent PID absence
proved process completion. Global post-run memory 4452 MiB has no same-run
baseline and is not a measured release delta. Existing ordinary-audio evidence,
pins, source and model remain unchanged. Do not repeat setup or inference for
this comparison.

## Next step

Report comparison to owner, then evaluate the already approved moss-eof-v1 rule
as a separate fresh task-1 phase. Preserve old strict failure, raw outputs and
upstream processor/config. Exact end-only <=0.01s normalization requires its own
boundary/rejection proofs and targeted retained-fixture CUDA rerun. Do not mark
all task 1 complete until that evidence passes. Do not begin task 2 here.

## Verification

- Controlled comparison: PASS; child exit 0, complete matching output.
- `python scripts/check_sdd_docs.py validate`: PASS; `git diff --check`: PASS.
- No application suites: research-only scope.

## Ownership and next worker prompt

Nine unrelated baseline dirty paths remain outside this phase's ownership.
No production settings, DB/jobs/media, backend venv or pinned upstream changed.
Conductor rebuilds task-1 context pack with `resume: yes`. Scope the next phase
to EOF-policy proof in ignored runtime, research report, plan and handoff only.
Reuse retained environment and ordinary-audio proofs; task 1 stays in progress.
