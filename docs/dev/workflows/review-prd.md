# Workflow: review-prd

> **Harness-neutral workflow.** Run every adversarial specialist lens against a PRD or spec to surface risks before development begins. In Claude this is the `review-prd` skill (and can spawn subagents); in other harnesses, apply each lens inline unless the user explicitly asks for parallel agents.

## Instructions

The user wants adversarial review of a planning document or the final spec-and-plan pair. Apply every lens in [../reviewers/](../reviewers/) in sequence, each reading all artifacts under review and producing findings:

1. **architect** — subsystem boundaries, ADR-worthiness, cross-module coupling
2. **planner** — plan completeness, ordering, hidden dependencies
3. **simplification-expert** — accidental complexity, unjustified abstractions
4. **code-reviewer** (when code exists) — correctness, safety, error handling

Add domain-specific lenses to `reviewers/` as this project needs them; review with every lens present.

For each lens:
- Read the document at the path the user specified (or the spec under review)
- Produce a focused findings section: specific objections, unvalidated assumptions, and what needs to be proven before the decision or design can be trusted
- Label each finding with a severity: **BLOCKER**, **HIGH**, **MEDIUM**

After every lens has reported, produce a consolidated summary:
- Top 3 blockers that must be resolved before development starts
- Top 5 high-priority open questions
- Suggested order of resolution

When this is the final spec-and-plan gate, resolve every finding required by the requested outcome in the artifacts and record the review result in both. Repeat review only where a resolution materially changed the reviewed design. The gate passes when no necessary finding remains open; neither artifact may be presented as complete before then.

Be adversarial. Do not validate — find the gaps.
