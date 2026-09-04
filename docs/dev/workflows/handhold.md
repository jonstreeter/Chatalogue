# Workflow: handhold

> **Harness-neutral workflow.** Ends every response with a short **Handhold**
> block telling the user where they are in the feature/bug lifecycle and the
> single next thing to do — including when to commit, push, deploy, archive,
> and whether it is safe to start a second task. Every harness activates it the
> same way: `AGENTS.md` carries the always-on instruction and each adapter
> points back here.

## Why this exists

The process in [`../README.md`](../README.md) is complete but it is a map, not a
compass. A person mid-task does not need the whole flow restated; they need the
one next move. Without that, two failures repeat: work sits uncommitted for days
because no moment ever felt like the right one, and a second task starts on top
of an unfinished first until neither can be untangled.

The block is the compass. It is small on purpose — five lines, one action.

## The block

Append at the very end of the response, after everything else:

```
---
🧭 **Handhold**
- **You are here:** <track> · step <n>/<total> — <one-phrase task name>
- **Do next:** <one concrete action; exact command if there is one>
- **After that:** <the following step, one line>
- **Input:** <needed — the agent waits for the user to trigger "do next" | not needed — awaited process only, no user input required right now>
- **Worktree:** <clean | N files: areas> — <keep going | commit now | park it first>
```

Rules:

- Append it to every response that touches code, docs, tests, or git. Not to
  pure question-answering.
- Exactly one **Do next**. Never a menu. If two things are genuinely equal, pick
  one and say why in six words.
- Give the literal command when one exists, not a description of it.
- Five lines. Anything longer belongs in the body of the response.
- **Input** is mandatory and only states who is waiting for whom. `needed`
  means the agent is parked and waits for the user to reply `d` (proceed with
  "do next") or to act on the named step; the block body may still ask a real
  question, but that question belongs in the body, not here. `not needed`
  means the agent waits on a process — a background job, CI, a build — whose
  completion arrives on its own, and no user input is required right now. Do
  not use `needed` to fish for answers to unrelated pending items; those are
  the body of the response.
- Never present a step as done that has not been verified. If the checks were
  not run, the next step is running them.
- Read the real state before writing it (see [Reading the current
  state](#reading-the-current-state)); never infer it from the conversation.

## Track A — new feature

Tier comes first: [`../README.md`](../README.md#three-tiers--most-work-is-not-tier-3)
decides whether steps 2–3 apply at all. Tier 1 work starts at step 4.

| # | Step | Done when | Then |
|---|------|-----------|------|
| 1 | Frame it | You can state the outcome and how you would prove it | → 2 |
| 2 | Spec (tier 3) | `docs/dev/specs/YYYY-MM-DD-<slug>-design.md`, reviewed per `review-prd` | → 3 |
| 3 | Plan (tier 2+) | `docs/dev/plans/YYYY-MM-DD-<slug>.md` with a numbered tracker; review gate passed | **commit the spec+plan** → 4 |
| 4 | Build one slice | One tracker task compiles and does something visible | → 5 |
| 5 | Verify the slice | Smallest relevant check green (`workflows/verify.md`) | → 6 |
| 6 | Commit the slice | `feat(<area>): …`; tick the task in the plan tracker | more tasks? → 4, else → 7 |
| 7 | Full verify | The complete `workflows/verify.md` loop | → 8 |
| 8 | Push | `git push` | → 9 |
| 9 | Deploy | Only if this project deploys and the change ships — **ask the user first** (see the Deployment Reminder in `AGENTS.md`) | → 10 |
| 10 | Archive | Spec `Status: Shipped`, plan `Status: Done`, both moved to `archive/`, handoff archived, scratch files deleted, `python scripts/check_sdd_docs.py validate` green | done |

When skipping 2–3, say out loud that you are skipping and which tier you judged
it to be. A silent skip is how tier 3 work gets built as a patch.

## Track B — bug fix

| # | Step | Done when | Then |
|---|------|-----------|------|
| 1 | Reproduce | You have made it fail on demand, or have the exact error/log | → 2 |
| 2 | Root cause | You can name the file and line and explain *why*, not just where | → 3 |
| 3 | Tripwire check | Run the escalation tripwire in `AGENTS.md`. If one fires: stop, report, wait for the user's call | → 4, or switch to Track A |
| 4 | Fix at the root | Fix the shared function once; check its other callers | → 5 |
| 5 | Regression test | A test that fails before the fix and passes after | → 6 |
| 6 | Verify | Smallest relevant check green | → 7 |
| 7 | Commit | `fix(<area>): <symptom>` — one bug per commit | → 8 |
| 8 | Push, then deploy if it ships | Same rule as Track A step 9 — ask first | → 9 |
| 9 | Close out | Debug scaffolding reverted, scratch files deleted | done |

Debug-only edits — extra logging, probes, temporary asserts — are not part of
the fix. Revert them before step 7 or the worktree stops being trustworthy.

## Starting a second task while one is open

Ask three questions, in order:

1. **Is the current work committed?** If yes → safe to start anything. Go.
2. **Do the two tasks touch the same files?** Run `git status --short` and
   compare against the files the new task will touch.
   - **Overlap** → do not start. Finish the current slice to its next commit
     point (Track A step 6 / Track B step 7), then switch. Interleaved edits to
     one file cannot be untangled later.
   - **No overlap** → safe, but checkpoint first.
3. **Is the current work green?**
   - **Green** → commit it now, even mid-plan. A green slice is always worth a
     commit; that is the checkpoint.
   - **Broken / mid-edit** → finish the slice (preferred, usually minutes) or
     `git stash push -u -m "<what it was>"`. Write the message; an unlabelled
     stash is a lost afternoon.

Genuinely parallel long-running work — two multi-day features at once — belongs
in a separate `git worktree`, not a stash. That is the only case that justifies
one.

If more than one agent session is open on the same directory, say so in the
block and scope edits to non-overlapping files.

## Keeping the worktree clean

- **Commit small and often.** A commit is a save point, not a milestone. Green
  checks plus a coherent change is enough reason.
- **One concern per commit.** Feature code, drive-by formatting, and a doc
  rewrite are three commits.
- **Never bypass the gates.** No `--no-verify`, no `SKIP=`. A failing hook is a
  real finding. (Conformance item 4 in `docs/dev/harnesses/README.md`.)
- **Temp files stay out of the repo.** Dumps, one-off scripts, downloaded
  fixtures.
- **A long-lived dirty worktree is a warning sign, not a state.** If
  `git status` has been noisy for days, the fix is to commit or revert in
  batches, not to work around it.
- **Before switching tasks, `git status --short` should be empty** or hold only
  the one thing being worked on.

## Reading the current state

```
python scripts/check_sdd_docs.py status
git status --short
git log --oneline -3
git stash list
```

`status` gives the process position — active plans, tracker progress, newest
handoff. The git commands give the worktree verdict. Fill the block from those,
not from memory of the conversation.
