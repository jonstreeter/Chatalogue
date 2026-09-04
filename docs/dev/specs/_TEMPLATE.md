# <Feature Name> — Design Spec

**Date:** YYYY-MM-DD
**Status:** Draft
<!-- Draft | In Review | Approved for implementation | Shipped | Superseded -->
**Owner:** <name>
**Plan:** _(not yet written)_
<!-- Replace with `docs/dev/plans/YYYY-MM-DD-<slug>.md` once the plan exists. -->
**Scope:** <one or two sentences: what this does and, explicitly, what it does not touch.>

---

## Motivation

Why does this exist? What can't be done today, and what does the user gain? Reference the existing behavior or code path it builds on.

## Design tenets

Numbered principles that constrain the design. Examples: "Reuse the existing X path — no changes to Y." "Match the canonical implementation." "Ad-hoc, not configurable." Keep each tenet falsifiable.

## Data flow

```
<ASCII diagram: source → store/field → request → backend function → pipeline → output>
```

Trace one request end to end. Name the concrete files, request fields, and functions at each hop.

## Interface contracts

For each boundary the feature crosses:

- **Client → API:** new request fields, types, validation.
- **Service → service / worker:** new payload fields, defaults, caps.
- **Persistence / filesystem:** new tables, columns, files, formats.

State what stays the same as explicitly as what changes.

## Edge cases & failure modes

- What happens on empty/invalid input?
- What existing behavior must not break?
- Concurrency, OOM, disconnect, partial failure.

## Adversarial review

Which lenses from `docs/dev/reviewers/` apply, and the BLOCKER/HIGH findings that must be resolved before build. Link the review output or summarize decisions here.

Record the final review of this spec and its implementation plan together, the resolutions applied, and whether the completion gate passed. Do not mark or present this spec as complete before it passes.

## Out of scope

Explicitly deferred items, so reviewers don't re-litigate them. Note which are candidate follow-ups.

## Open questions

Anything that must be decided before or during implementation. Do not paper over ambiguity.
