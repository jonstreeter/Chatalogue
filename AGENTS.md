# Chatalogue — Agent Working Guide

Default instruction layer for every AI coding agent and every human in this
repo. It is loaded constantly, so it holds only what is needed constantly:
facts about this project, and the two rules that decide when to stop.

Everything else is one hop away and listed at the bottom.

## Repo Map

- `backend/src/`: FastAPI backend — `main.py` (app entry), `routers/` (HTTP API),
  `services/` (ingestion, transcription, diarization, cloning), `db/` (storage).
  Run from `backend/` with its own venv (`.venv`).
- `frontend/`: React 19 + Vite + TypeScript + Tailwind UI. Run npm scripts from
  `frontend/`.
- `scripts/`: SDD process scripts (validation, preflight, commit gates).
- `docs/`: root docs plus `docs/dev/` — specs, plans, handoffs.

## First Places To Look

- `backend/src/main.py` — FastAPI app entry and startup wiring
- `backend/src/routers/` — the HTTP API surface
- `backend/src/services/` — business logic: ingest, transcribe, diarize, clone
- `backend/src/db/database.py` — storage of record (SQLite/Postgres)
- `frontend/src/` — UI; `frontend/package.json` for exact npm scripts

## Canonical Commands

- Session state: `python scripts/check_sdd_docs.py status`
- Backend dev server (from `backend/`, venv active):
  `python -m uvicorn src.main:app --reload`
- Frontend dev server (from `frontend/`): `npm run dev`
- Backend tests (from `backend/`): `python -m pytest src -q`
- Frontend tests (from `frontend/`): `npm run test:e2e`
- Lint: `ruff check .` (repo root); frontend `npm run lint` (from `frontend/`)
- Typecheck (from `frontend/`): `npx tsc -b`

## Search Boundaries

Avoid spending context on generated or third-party directories unless the task
directly targets them:

`backend/.venv*/`, `backend/data/`, `backend/logs/`, `frontend/node_modules/`,
`frontend/dist/`, `frontend/test-results/`, `checkpoints/`, `snapshots/`,
`REference Resources/`, and other reference-only vendored trees.

## Working Rules

- Prefer targeted reads over broad scans. Use `rg` and exact file paths.
- Keep output concise. Summarize logs; do not paste raw command output.
- For UI flow changes, prefer browser verification over reasoning alone.
- Run the smallest relevant check first; broaden only when a change crosses
  component boundaries.

## Spec-Driven Development

The process — spec → plan → build → verify → handoff — the review lenses, and
the workflow instructions live in [`docs/dev/`](docs/dev/README.md). That is the
single source of truth for **every** harness. Each harness carries a pointer
only; never duplicate process text into a harness directory.

Start any planning, review, or verification task by reading
`docs/dev/README.md`. Not every change needs a spec — `docs/dev/README.md`
defines the three tiers and which one applies.

### Contract Preflight

Before diagnosing, planning, or editing a subsystem:

1. Identify the likely code paths.
2. Run `python scripts/check_sdd_docs.py preflight <path> [<path> …]`.
3. Read every returned `contract` and `refines` authority in repository order.
4. Reconcile the current code with those contracts before proposing an edit.
5. If the change alters a mapped contract, update or refine the existing
   authority; do not create an overlapping spec.

This applies to every harness. Some enforce it before the edit, the rest at
commit time — the requirement is identical either way.

### Escalation Tripwire

A task that arrived as "just fix this" can turn out to need SDD. This usually
surfaces mid-testing, once the real cause is understood. **Stop before writing
the workaround** if any of these hold:

- The root cause sits in a different subsystem than the reported symptom.
- The fix changes a contract other code depends on — REST/WS message shape, DB
  schema, storage layout, file formats.
- The honest fix and the quick fix diverge: you are about to write something
  that would later have to be undone.
- It touches tenancy, auth, or the connected storage system of record.
- How to roll it back is not obvious.

None of these are about size. A one-line change to a WS payload trips the
tripwire; a hundred-line change inside one component does not.

When one fires, do not finish the patch and do not silently start a spec.
Report to the user in a few lines — the root cause, which trigger fired, and
the choice between spec+plan, a labeled stopgap, or patching anyway — then
wait. Procedure and the stopgap label are in
[`docs/dev/README.md`](docs/dev/README.md#escalation--when-a-fix-becomes-a-feature).

## Handhold Block

End every response that touches code, docs, tests, or git with the five-line
**Handhold** block defined in
[`docs/dev/workflows/handhold.md`](docs/dev/workflows/handhold.md). Read that
file before the first such response in a session; do not reconstruct the block
from memory.

It states where the user is in Track A (feature) or Track B (bug), the single
next action with its literal command, the step after, whether input is needed
or what is being monitored, and a worktree verdict.
That is what tells the user when to spec, plan, verify, commit, push, deploy,
and archive — and whether starting a second task now is safe. Read the real
state first (`python scripts/check_sdd_docs.py status`, `git status --short`,
`git log --oneline -3`, `git stash list`).

A reply of `d` means "proceed with the **Do next** action": execute the named
command or step and report back. If the block says input is needed, `d` is not
an answer to that input — answer the question instead.

## Deployment Reminder

Local-first desktop app — `install_windows.bat` / `install_mac.sh` clone the
repo into the install folder and run there. Nothing deploys on its own; after
significant changes, re-run the relevant installer or update script on a target
machine.

## Resuming Work

Run `python scripts/check_sdd_docs.py status`. It prints the active plans and
their tracker progress, the newest handoff notes, and the uncommitted surface.
Read the newest handoff before code, then the spec and plan it references.

Plan trackers are the source of truth for progress — update them as status
changes. If work stops mid-flight, write a handoff from
`docs/dev/handoffs/_TEMPLATE.md`.

Artifact priority when they disagree: current user request > handoff > spec >
plan > brainstorm notes. Never let an old plan override a newer user request.

## One Hop Away

| Read when | File |
| --- | --- |
| Planning, building, reviewing, or deciding whether to build at all | [`docs/dev/principles.md`](docs/dev/principles.md) |
| Running the process end to end | [`docs/dev/README.md`](docs/dev/README.md) |
| Doing a specific task | [`docs/dev/workflows/`](docs/dev/workflows/) |
| Reviewing anything | [`docs/dev/reviewers/`](docs/dev/reviewers/) |
| Wiring or questioning a tool's automation | [`docs/dev/harnesses/README.md`](docs/dev/harnesses/README.md) |
