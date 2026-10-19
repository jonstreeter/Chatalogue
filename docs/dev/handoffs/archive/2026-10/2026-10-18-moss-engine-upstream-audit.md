# Handoff: optional MOSS engine — EOF policy refined

**Date:** 2026-10-18
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Owner approved original MOSS design/plan and subsequently requested an explicit
tiny-boundary-overrun rule after the task-1 synthetic fixture ended at 9.01s for
9.00s audio. Existing spec now defines `moss-eof-v1`: only end timestamps at most
0.01s beyond the exact sample-derived duration may be normalized to EOF. Exact
comparisons, raw evidence, independent backend validation, invalid-start rejection
and rejection of collapsed corrections are required. No generic rounding or
word clocks. The value derives from the measured failure, not an upstream
precision guarantee. Paired refinement review passed inline with all four lenses.

Task 1 is reopened; 0/8 tasks complete. Owner subsequently requested a complete
timing-workflow audit before the rerun. That read-only CPU audit found no sample
duration/loader/parser discrepancy, but identified an upstream marker alignment
issue: configured five-second markers are placed every 62 tokens at 12.5 tokens/s
(nominal 4.96s), accumulating input-marker error. Its causal effect on generated
timestamps is unproven. Exact samples and raw output are preserved; no generation
or correction has been run since the prior blocked probe. No product code yet.
Ordinary CUDA/pinning/timing/exit evidence is retained from checkpoint `62da003e`;
do not repeat completed setup or ordinary-audio benchmarks. Previous strict
failure remains recorded in research report and archived baton.

## Next step

Upstream history/training follow-up is complete: marker arithmetic existed at
initial release and remains unchanged; published fine-tuning uses checkpoint
processor/config. Original pretraining intent and a safe marker correction are
not documented in the sources retrieved. Preserve pinned upstream behavior;
the arithmetic alone does not prove transcript drift or cause of `[9.01]`.
No inference or model/config change was made during these two investigations.

Next fresh task-1 worker compares unmodified official generation helper with
the probe on the retained fixture, capturing raw output before normalization.
Do not alter the fixture to obtain a pass. Then evaluate the approved EOF policy
separately, retaining the old failure and independent evidence. Do not silently
change marker config, widen tolerance or present normalization as a causal fix.

After that review, fresh Super Looper task-1 worker: update ignored probe to apply the spec's exact
EOF policy and record raw correction evidence, check boundary/rejection examples,
then rerun retained synthetic overlap WAV on pinned local CUDA runtime. Require
complete output, native overlap, bounded final clocks, preserved raw evidence and
process exit/VRAM evidence before task 2. Do not ask for original plan approval.

## Verification state

- Prior CUDA BF16/SDPA decodes, dependency pins and exit evidence: see
  `docs/research/2026-10-18-moss-runtime.md` (unchanged historical measurements).
- Original strict overlap test: FAIL; preserved, not retroactively passed.
- New-policy runtime rerun: not yet run.
- Refinement review: PASS, architect/planner/simplification/code-reviewer inline.
- CPU timing audit: PASS, exact file/loaded samples, fixture reconstruction,
  raw-before-parse clocks, matching pinned dynamic-code cache; CUDA uninitialized.
- Input marker alignment discrepancy: confirmed in code/config; effect on model
  predictions unresolved. EOF normalization is an application policy, not a
  demonstrated causal repair. No new runtime inference performed.
- Upstream history/training review: same arithmetic at initial release and in
  published fine-tuning path; no documented safe correction found. Original
  pretraining calibration unknown. Sources/limitations recorded in research.

## Blockers

The prior policy decision is resolved by the owner's refinement request. Audit
and upstream-source findings have been documented. No app-introduced marker
change or documented upstream fix was found; do not invent one. Product work
remains gated on the controlled task-1 runtime result and honest reporting of
timestamp accuracy limitations.

## Diagnosis worth keeping

Pinned helper README examples use two decimal places but `_parse_timestamp`
accepts arbitrary precision. Do not infer universal quantization from examples.
Keep complete-generation checks strict, and reject output outside the explicit
0.01s end-only policy. Preserve raw native labels and overlapping intervals.

Private environment/model/source/probe/fixtures are retained under ignored
`backend/runtime/moss/`. No reinstall is needed. No production settings, DB/jobs
or recordings changed. Nine unrelated dirty paths remain outside task ownership;
approved MOSS spec was initially untracked and is now included with this refinement.

## Next worker prompt

Conductor rebuilds context pack for task 1, `resume: yes`, against current HEAD.
Use spec section `Tiny EOF overruns (owner-authorized refinement, 2026-10-18)`;
keep task-1 write scope to ignored runtime and research report plus required plan
and handoff artifacts. Reuse the prior ordinary-audio proof. Fresh worker must
record new evidence before declaring the phase complete. No task 2 in that worker.
