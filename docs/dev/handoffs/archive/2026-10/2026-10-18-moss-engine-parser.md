# Handoff: optional MOSS engine — task 2 parser boundary PASS

**Date:** 2026-10-18
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Tasks 1–2 complete, 2/8. Plan stays In progress; spec is not Shipped.
Task 3 awaits user Do next. No runtime/process/product integration in this phase.

## Evidence and interface

Accepted ADR-0002 records owner approval, measured pinned Windows feasibility,
limitations and the MOSS-only architecture. Registry maps only MOSS private files;
preflight resolved after staging the newly created authority, and both returned
contract/refines authorities were read before implementation.

`moss.py` is standard-library-only. `parse_generation(raw_text, sample_count,
sample_rate, completion)` returns JSON-safe moss-result-v1. Normalized start/end
and duration are exact rational strings; raw decimal tokens and full transcript
are retained. Every segment has native speaker/text/words=null. Corrections carry
index/raw_end/normalized_end/policy. Overlapping/nested/nonmonotonic ends keep order.
In-range zero spans follow raw start<=end; corrected collapsed spans fail.

Completion has exactly complete/reason/eos_token_id/terminal_token_id/input_tokens/
output_tokens/max_new_tokens/context_limit. Require complete=true, reason=eos,
pinned EOS 151645, terminal match, positive integer counts, output below budget,
budget within pinned 131072 context. Worker must derive this from real generation;
EOS does not prove recognition of every audible word. Whitespace-only complete
output is allowed; invalid output never counts as silence.

`validate_worker_result(result, sample_count, sample_rate)` is the backend trust
boundary: use manager's actual decoded sample dimensions, recompute from raw
transcript, compare the entire JSON schema/evidence, return fresh validated data.
Transport/model/device/timing provenance belongs outside this strict result object.
Errors are displayable and never echo transcript. Worker integration should reuse
this pure module without importing ingestion package/application dependencies.
Tests load it by file path to avoid package initializer/app/DB side effects.

## Verification

- DB_PROVIDER=sqlite and DATABASE_URL=sqlite:// before imports; backend `.venv/Scripts/python.exe -m pytest src/test_moss.py -q`: 63 PASS.
- Backend interpreter `-m ruff check backend/src/services/ingestion/moss.py backend/src/test_moss.py`: PASS.
- `python scripts/check_sdd_docs.py validate`: PASS.
- `git diff --check`: PASS.
- Required commit hooks run without bypass.

## Next worker prompt

Repack task 3 with its explicit ownership and fresh preflight. Reuse measured task-1
pins and this strict internal result boundary. Build isolated managed installer,
worker and owned process lifecycle only; exercise concurrent/interrupted setup,
offline cache, busy unload, worker death and parent exit. Enforce input duration in
worker/pipeline as planned; no arbitrary output cap, derive budget from available
context. Do not start settings/UI/checkpoint integration early. Existing engine
and backend dependencies stay unchanged. Nine unrelated baseline dirty paths remain
untouched. Previous baton: `archive/2026-10/2026-10-18-moss-engine-feasibility.md`.
