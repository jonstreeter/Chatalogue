# Working principles

Two lenses that shape *how* work gets done, kept out of `AGENTS.md` on purpose.

`AGENTS.md` is loaded on every turn in most harnesses, so it should hold facts
an agent needs constantly: the repo map, the commands, the boundaries, the
escalation tripwire. These principles are needed while **planning, building,
and reviewing** — not while answering "where do the tests live". Keeping them
here means the per-turn instruction layer stays short enough to actually be
read, and this file gets loaded when it bites.

Read this file when: writing a spec or plan, running a review lens, working
through a test/remediation loop, or deciding whether to build something.

The workflows in `docs/dev/workflows/` reference it at those points.

---

## MSW — the kernel

## program — complete

```
contract ← the requested outcome + the smallest criteria that prove it
while ∃ claim c : deleting c leaves contract unmet ∨ unproven
do c ; prove c
halt ; report
```

## definitions — no behavior lives here, only meaning

- **contract** — the requested outcome and the smallest set of acceptance criteria that would prove it, stated before any work. The sole source of necessity; a ceiling as much as a floor. If the request is ambiguous: attended → ask; unattended → bind the smallest reading consistent with stated intent and record the assumption.
- **claim** — anything petitioning to become work: a plan step, a change, a test, a reviewer's P1, a discovered edge case, your own instinct that one more pass would help. Everything enters as this type. Nothing enters as a verdict.
- **deleting c leaves contract unmet ∨ unproven** — the only test. A claim passes solely by breaking the contract — reproducibly, within the task's actual inputs and environment. Severity is derived from the contract, never inherited from whoever raised the claim. *Useful*, *thorough*, and *possible* are not aliases for *necessary*. A claim that fails receives one line in the report — never a fix, an investigation, or a deferred follow-up.
- **do ; prove** — the smallest reliable act that closes the gap, and evidence sized to the claim it settles. An unproven act keeps its claim alive; a proven one closes it — and re-proving a closed claim is itself an inadmissible claim.
- **halt** — the fixed point: contract proven, no remaining claim passes. Not reviewer silence; not exhausted imagination. Halting before the fixed point and looping past it are the same bug, mirrored.
- **report** — the outcome against the contract; the proof; rejected claims worth the user's attention, one line each. Nothing else.

## fuses — outside the program, for when its evaluator fails

- `rounds = 3` → halt anyway ; report open items, do not chase them
- `claim born in round n+1, visible in round n` → rejected

## No unauthoritative limits

Never invent a limit. A cap, threshold, quota, budget, timeout, retry or round count, file or line count, acceptance-criterion count, agent count, or similar constraint is admissible only when its exact value is:

- explicitly required by the requester;
- imposed by an applicable technical or platform contract;
- defined by authoritative project policy; or
- derived from measured evidence necessary to meet or prove the task contract.

State the authority or derivation whenever proposing or applying a limit. If no authority exists, omit the limit and use the MSW necessity test. Metrics may be reported as evidence, but they must not become gates, defaults, targets, or recommendations through agent intuition. Examples and representative proportions never become defaults. If a necessary limit is an unresolved owner choice, ask; do not manufacture a value.

## MSW Usage Note

Reinforce MSW whenever planning, implementing, or in a test/remediation loop by restating this line:

> Remember to follow the MSW deletion rule for all claims — no exceptions.

## Ponytail — lazy senior dev mode

Be a lazy senior dev: lazy means efficient, not careless. Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse it, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb. Bug fix = root cause, not symptom: grep every caller of the function you touch and fix the shared function once.

Never simplify away: input validation at trust boundaries, error handling that prevents data loss, security, accessibility, anything explicitly requested. Non-trivial logic leaves ONE runnable check behind.
