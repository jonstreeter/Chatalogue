# Spec-Driven Development (harness-agnostic)

This directory is the **single source of truth** for how features are designed,
reviewed, and built in this project — independent of which tool is driving.
Claude Code, Codex, Kilo Code, Roo Code, Cursor, Aider, any future harness, and
a human with an editor all point back to the files here rather than carrying
their own copies. How that layering is enforced is in
[`harnesses/README.md`](harnesses/README.md).

## Layout

```
docs/dev/
  README.md              ← this file: the workflow
  principles.md          ← how to work: necessity test + laziness ladder
  harnesses/             ← one thin adapter per tool; the layering rules
  reviewers/             ← adversarial + specialist review lenses
  workflows/             ← task workflows: handhold, plan-feature, review-prd, verify
  specs/                 ← design specs (dated), _TEMPLATE.md, archive/
  plans/                 ← implementation plans (dated), _TEMPLATE.md, archive/
  adr/                   ← architecture decision records, _TEMPLATE.md
  handoffs/              ← mid-flight batons, README.md, _TEMPLATE.md, archive/
  contracts.json         ← optional registry mapping code paths to authorities
```

## Start here, every session

```
python scripts/check_sdd_docs.py status
```

Prints active plans with tracker progress, the newest handoff notes, and the
uncommitted surface. It is the same command for every harness, and it replaces
re-reading the repo to work out where things stand.

## Three tiers — most work is not tier 3

The flow below is the full ceremony. Applying it to everything is how a process
gets abandoned. Pick the tier by **blast radius**, never by line count.

| Tier | When | What it requires |
| --- | --- | --- |
| **1. Patch** | Contained in one component, obvious rollback, no contract touched | Build → verify. No documents. |
| **2. Feature** | New user-facing behavior, several files, still inside known boundaries | Plan → build → verify. Spec optional. |
| **3. Durable** | Crosses subsystems, changes a contract, or touches tenancy/auth/storage-of-record | The full flow below, including the review gate. |

You do not choose the tier once and stop thinking. The **escalation tripwire**
(below) is the promotion rule: it fires mid-work, most often during testing,
and moves a tier-1 task to tier 3 when the real cause turns out to be
structural. Promotion is expected and is not a planning failure.

Default to the lowest tier that honestly fits. If two tiers seem to fit, the
tripwire conditions decide.

## The flow (tier 3)

1. **Spec** — write from [`specs/_TEMPLATE.md`](specs/_TEMPLATE.md) as
   `specs/YYYY-MM-DD-<slug>-design.md`. Captures motivation, tenets, data flow,
   contracts, and out-of-scope. Status starts `Draft`, and `**Plan:**` stays
   `_(not yet written)_` until a plan exists.
2. **Spec adversarial review** — run
   [`workflows/review-prd.md`](workflows/review-prd.md) against the spec.
   Resolve BLOCKERs before planning and record decisions in the spec. This does
   not complete a spec-and-plan request; the plan does not exist yet.
3. **Plan** — run [`workflows/plan-feature.md`](workflows/plan-feature.md) and
   write `plans/YYYY-MM-DD-<slug>.md` from
   [`plans/_TEMPLATE.md`](plans/_TEMPLATE.md). The task tracker is the source of
   truth for progress.
4. **Final review gate** — run `review-prd` against the completed spec and plan
   **together**. Resolve every finding required by the requested outcome, and
   record the review in both documents. A harness must not call, mark, or
   present a requested spec or plan as complete before this gate passes.
5. **Build** — execute plan tasks in order, updating the tracker as you go. Keep
   changes scoped to the current task; verify before starting the next.
6. **Verify** — run [`workflows/verify.md`](workflows/verify.md).
7. **Handoff** — if work stops mid-flight, write
   `handoffs/YYYY-MM-DD-<slug>.md` from
   [`handoffs/_TEMPLATE.md`](handoffs/_TEMPLATE.md) with status, next step,
   exact paths, and verification state. Handoffs live only in
   `handoffs/` (active) and `handoffs/archive/YYYY-MM/` (closed), at most one
   active baton per plan — writing a new one archives that plan's previous one
   in the same commit. Rules: [`handoffs/README.md`](handoffs/README.md).
8. **Archive** — on completion move the spec and plan into their `archive/`
   folders, set spec `Status: Shipped` and plan `Status: Done`, and fix both
   header cross-links in the same change.

`python scripts/check_sdd_docs.py validate` checks the mechanical parts of all
of this. CI runs the same validation on every push and pull request.

## Contract registry

`contracts.json` maps durable subsystem paths to the specs and ADRs that govern
them. Before working on likely code paths:

```
python scripts/check_sdd_docs.py preflight <paths>
```

Read every returned authority. The registry *discovers* authority; the
referenced documents still define behavior. See
[`contracts.example.json`](contracts.example.json) for the shape.

- Add a path only when it carries durable architecture and has a current
  accepted authority.
- Keep path ownership non-overlapping; leave uncertain paths unmapped rather
  than inventing authority. `validate` rejects overlapping globs.
- `**` is recursive: `src/widget/**` governs the whole subtree.
- Update an existing authority when a feature refines that domain. Create a new
  spec only for a genuinely distinct contract.
- Update `contracts.json` in the same change when an authority is superseded.

**How it is enforced.** Running `preflight` writes a receipt
(`.sdd/preflight-receipt.json`, gitignored) recording which authorities were
surfaced and their content hashes. Harnesses with a pre-edit hook consult it and
block mapped edits until it is current — that is why the receipt is written by
the ordinary CLI rather than a tool-specific mechanism. Editing an authority
invalidates it, so a changed spec must be re-read.

Harnesses without hooks are not exempt: `pre-commit` and the CI contract-impact
gate apply the same requirement to everyone, including human commits. Reproduce
the CI gate locally with:

```
python scripts/check_sdd_docs.py impact --base <ref> --ack-file <file>
```

The acknowledgement is a `Contract impact:` line in the PR body, or in a commit
message when pushing directly. Either `Contract impact: none — <why>` or a
comma-separated list of the affected authority paths.

## Escalation — when a fix becomes a feature

Small changes skip to build, and most should. But a bug fix sometimes reveals,
usually during testing, that the real cause is structural. The failure mode is
finishing the workaround anyway: the diagnosis lives only in a chat log, and the
stopgap becomes permanent because nothing records that it was one.

The trigger conditions live in `AGENTS.md` under **Escalation Tripwire**, so
every harness carries them without opening this file. They are about blast
radius, not size. When one fires:

1. **Stop before writing the workaround.** A half-written patch biases the
   decision toward keeping it.
2. **Report and wait.** Give the user the root cause, which trigger fired, and
   the three options below. Do not begin a spec unprompted — escalation is the
   user's call.
3. **Take the chosen path:**
   - **Spec + plan** — normal SDD from step 1. Write the spec while the
     diagnosis is fresh; that reasoning *is* the motivation section.
   - **Labeled stopgap** — land the workaround with a `# STOPGAP:` comment
     naming the root cause and linking the spec that will replace it, then write
     the spec. Use this when something has to work now.
   - **Patch it** — the user accepts the debt. Record the decision in one line
     so the next session does not rediscover it.
4. **Never discard the diagnosis.** Whichever path is chosen, the root-cause
   finding gets written down — in the spec, the stopgap comment, or a handoff
   note. Rediscovering the same root cause twice is the cost of skipping this.

A stopgap without a linked spec is indistinguishable from a bug. If the user
picks the stopgap path and declines the spec, it is a plain patch — label it
that way rather than implying a follow-up that will not come.

Artifact priority when they disagree: current user request > handoff > spec >
plan > brainstorm notes. Never let an old plan override a newer user request.

## Review lenses

The files in [`reviewers/`](reviewers/) are expert review lenses — adopt one to
shape analysis, findings, risks, and test plans. Any harness can load them
directly; a harness that supports subagents may spawn them in parallel **only
when the user explicitly asks**, otherwise apply the lens inline and say no
subagents were spawned.

| Lens | Use for |
| --- | --- |
| `architect` | subsystem design, ADRs, cross-module decisions |
| `planner` | pre-build implementation planning |
| `simplification-expert` | specs, plans, UI, and code that may be overcomplicated |
| `code-reviewer` | correctness/safety review after writing code |

These four are generic on purpose. The lenses that actually catch things are
the ones written for *this* project's failure modes — see
[`reviewers/writing-a-lens.md`](reviewers/writing-a-lens.md) and add them here.

## Wiring a harness

To make any tool spec-driven, its native rules file should contain **only a
pointer** to this directory — never a copy. `AGENTS.md` is the root entry point
that most tools read; `CLAUDE.md`, `.kilocode/`, `.roo/`, and `.cursor/` are
one-line pointers to it.

The full layering contract, what an adapter may and may not contain, and how to
add a new harness are in [`harnesses/README.md`](harnesses/README.md).
