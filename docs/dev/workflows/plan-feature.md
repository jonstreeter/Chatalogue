# Workflow: plan-feature

> **Harness-neutral workflow.** Plan the implementation of a new feature before writing any code. In Claude this is the `plan-feature` skill; in other harnesses, read and follow this file directly.

## Instructions

The user wants to implement a new feature. Do not write code yet. Produce a complete implementation plan and save it as a plan doc (see [../README.md](../README.md) for the spec→plan→build flow and [../plans/_TEMPLATE.md](../plans/_TEMPLATE.md)).

### Step 1 — Understand current state
- Read the relevant existing code in the affected modules
- Identify what already exists that the feature can build on
- Identify what is missing

### Step 2 — Define the feature precisely
- What is the exact user-facing behavior?
- What are the inputs and outputs?
- What are the edge cases and failure modes?
- What existing behavior must not break?

### Step 3 — Design the implementation
For each layer affected (UI / API / services / data store — adapt to this project's architecture):
- What changes are needed?
- What new files or modules are required?
- What existing files are modified?
- What is the interface contract between layers?

### Step 4 — Identify risks
Apply the relevant adversarial review lenses (see [../reviewers/](../reviewers/)) — spawn them as subagents if the harness supports it and the user asked, otherwise apply the lens inline:
- Architecture or cross-module decisions → `architect`
- Anything planned before build → `planner`
- Suspicion of overcomplication → `simplification-expert`
- Code already written → `code-reviewer`
- Add project-specific lenses to `../reviewers/` as the domain demands them.

### Step 5 — Produce the plan document

Write the plan to `docs/dev/plans/YYYY-MM-DD-<slug>.md` using
[`../plans/_TEMPLATE.md`](../plans/_TEMPLATE.md). Use the template as-is rather
than inventing a layout: its task tracker is what `check_sdd_docs.py status`
reads to report progress, and its verification log is what lets another session
(or another harness) resume without re-running everything.

Fill every section:

- **Summary** — matches the approved spec's scope.
- **Affected modules** — exact paths, with what changes in each.
- **Task tracker** — concrete steps, exact files, S/M/L, stable numbering. Each
  task must be independently verifiable; split anything that is not.
- **Test plan** — unit, integration, and the manual/browser flow.
- **Risks** — each tagged with the reviewer lens that covers it.
- **Open questions** — anything that must be resolved before coding.

### Step 6 — Pass the final adversarial review gate

Run [review-prd](review-prd.md) against the completed spec and plan together. Resolve every finding required by the requested outcome in the artifacts, then record the review result in both documents. Do not call, mark, or present either artifact as complete before this gate passes.

Do not begin implementation until this gate passes and the user approves the plan.
