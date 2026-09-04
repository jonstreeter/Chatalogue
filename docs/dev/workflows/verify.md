# Workflow: verify

> **Harness-neutral workflow.** Run the full verification loop: static checks and tests. In Claude this is the `verify` skill; in other harnesses, read and follow this file directly. Prefer concise wrapper scripts that write full logs to disk and print only the useful summary, if this project has them.

## Instructions

Run `python scripts/check_sdd_docs.py status` first to confirm which plan
this work belongs to, then run all verification checks in order. Stop and report on the first BLOCKER. Summarize all warnings at the end. Adapt the exact commands to this project's stack from AGENTS.md "Canonical Commands"; the shape below is the contract.

### Static checks

```bash
npx tsc -b                  # frontend typecheck (from frontend/)
ruff check .                # python lint (repo root)
npm run lint                # frontend lint (from frontend/)
```
- Zero type errors required to pass — any error is a BLOCKER
- Zero lint errors required to pass; warnings are noted but do not block

### Tests

```bash
python -m pytest src -q     # backend tests (from backend/, venv active)
npm run test:e2e            # frontend e2e (from frontend/)
```
- All tests must pass
- Any test failure is a BLOCKER
- Report coverage if `--cov` flag is available

### Process checks

```bash
python scripts/check_sdd_docs.py validate
```
- SDD artifact and contract-registry structure. Any violation is a BLOCKER.
- If the change touched a mapped contract path, confirm the plan tracker and
  verification log were updated. `python scripts/check_sdd_docs.py progress`
  reports this for staged changes.

### Report format

```
VERIFICATION REPORT
===================
Typecheck:  PASS / FAIL (N errors)
Lint:       PASS / WARN (N warnings)
Tests:      PASS / FAIL (N failed, N passed)
SDD docs:   PASS / FAIL (N violations)

BLOCKERS:
- [list any blocking issues with file:line references]

WARNINGS:
- [list any non-blocking issues]

STATUS: READY TO COMMIT / BLOCKED
```

Do not commit or push if STATUS is BLOCKED. Fix blockers first.
