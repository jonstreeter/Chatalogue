# Handoff: optional MOSS engine — Parakeet timing comparison complete

**Date:** 2026-10-18
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Task 1 remains in progress; 0/8 complete. The owner-requested timestamp
comparison is complete. No EOF-policy normalization or product work ran.

## Evidence

`docs/research/2026-10-18-moss-parakeet-timing.md` contains complete reproducible
comparison source, commands, all exact phrase matches, separate endpoint anchors,
ambiguities, unmatched text, engine/model provenance and limitations.

- Proved 120s WAV PCM identity before reusing retained actual-app Parakeet result:
  1,920,000 frames, mono PCM16/16kHz; SHA256
  `836fc11d4a1de01270096e65aebedfbe272596761c7fc9e308da16e26e55322c`.
- Conversation: 9/31 unique exact full phrases (97/392 MOSS normalized tokens),
  18 endpoints; median absolute delta 0.11s, mean 0.231667s, signed range
  -1.11 to +1.35s. Four repeated phrases ambiguous; 18 unmatched segments.
- Fresh retained 9s fixture processed by actual app Parakeet CUDA in isolated
  in-memory SQLite/temp directories, no production jobs/settings/media writes.
  Both runs requested/routed/used Parakeet; metadata route, no Whisper probe or
  transcription fallback. Cold/warm app phases 272.232/2.087s, model load 243.54s.
  Child 33104 and launcher 3628 absent afterward; shell exit 0 retained.
- Overlap: 0/4 exact full phrases (0/39 tokens); Parakeet has 23 words and lexical
  disagreement including 81 versus MOSS 91 and the competing S01 utterance.
  Three globally unique endpoint anchors give -0.16/+0.35/+0.13s deltas.
  No whole-phrase aggregate, speaker accuracy or audio ground truth claimed.
- Parakeet structural checks pass (both fresh runs and retained warm 120s).
  MOSS native9.01s terminal remains the previous strict-bound failure, unchanged.

Scratch evidence in `backend/runtime/moss/`: `compare_parakeet_timing.py`,
`parakeet-timing-comparison-20261018.{json,log}`,
`parakeet-overlap-9s-20261018.{json,log}` and
`parakeet-overlap-9s-20261018-exit.txt`. Prior artifacts were preserved.
Previous helper-comparison baton is archived as
`archive/2026-10/2026-10-18-moss-engine-helper-comparison.md`.

## Next step

Report comparison to owner, then resume pending task1 EOF-policy proof as a
separate phase. Preserve raw clocks/labels and pinned assets. Do not treat
Parakeet as timing ground truth or infer a uniform clock drift. Do not rerun
these completed comparison inferences. Tasks2–8 remain unstarted.

## Verification

- Comparison script: PASS; PCM equality, actual engine, Parakeet bounds and
  endpoint arithmetic assertions completed. Child session 83135 exit 0.
- `python scripts/check_sdd_docs.py validate`: PASS.
- `git diff --check`: PASS.
- Application suites omitted: research-only scope.

## Ownership and next worker prompt

Nine unrelated baseline dirty paths remain outside this phase's ownership.
No production settings, DB/jobs/media, backend venv or pinned upstream changed.
Conductor rebuilds task1 context pack with `resume: yes`; scope the next phase
to the approved moss-eof-v1 proof in ignored runtime, research, plan and baton.
Task1 remains incomplete until all required gate evidence passes.
