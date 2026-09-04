# Handoffs

A handoff is a **baton, not a diary**: the note the next session reads to resume
exactly where the last one stopped. It is written for one hop and goes stale the
moment the next hop lands.

## Layout

```
docs/dev/handoffs/
  README.md                             ← this file
  _TEMPLATE.md                          ← the shape
  YYYY-MM-DD-<slug>.md                  ← active batons (at most one per open plan)
  archive/YYYY-MM/YYYY-MM-DD-<slug>.md  ← superseded and closed batons
```

Nothing else. Handoffs never live at `docs/handoff-*.md`, in `docs/archive/`, or
beside plans and specs. `python scripts/check_sdd_docs.py validate` fails on a
misplaced handoff.

- **Filename** — `YYYY-MM-DD-<slug>.md`, date of writing, slug matching the plan
  or topic. Phase batons for one plan take a suffix:
  `2026-08-24-widget-parity-p6.md`.
- **Archive month folder** — taken from the filename date, not the archive date.
  A file dated `2026-08-24` archives to `archive/2026-08/`.

## The one-baton rule

**At most one active handoff per plan.** A new baton for a plan supersedes the
previous one, and the same commit that writes the new one moves the old one to
`archive/YYYY-MM/`. That is what keeps the active folder readable: it lists
exactly the work genuinely in flight. `validate` enforces it by reading the plan
paths in each baton's header.

A handoff moves to `archive/` when any of these is true:

| Trigger | Who moves it |
| --- | --- |
| A newer handoff for the same plan is written | the worker writing the new one, in the same commit |
| The plan reaches completion / closeout | the closeout commit |
| The plan or spec is archived | the archive commit that moves them |
| No plan (one-off investigation) and its follow-up has landed | whoever lands the follow-up |

Never delete a handoff. Archived batons are the record of how a plan actually
proceeded, and `git log --follow` still works across the move.

## Contents

Keep it to what the next hop needs in order to act — the shape is in
[`_TEMPLATE.md`](_TEMPLATE.md):

- **Status**, and whether the same phase resumes.
- **Plan / spec** paths, repository-relative and in backticks (that is what the
  one-baton check reads).
- **Commit** hash or `WIP` — the HEAD the note describes.
- **Shipped paths** — exact files.
- **Next step** — what the next session does first.
- **Verification state** — what passed, what was skipped and why.

Do not restate the plan, the spec, or the diff. If a fact belongs to the plan
tracker, put it in the tracker.

## Finding the right one

- Active work: `python scripts/check_sdd_docs.py status` surfaces the newest
  batons; the active folder is short by construction.
- Resuming a specific plan: the plan's header links its handoffs; otherwise list
  `docs/dev/handoffs/` then `archive/YYYY-MM/`.

Artifact priority when documents disagree: **current user request > handoff >
spec > plan**. An archived handoff is history, not an instruction — verify any
path or command it names still exists before acting on it.
