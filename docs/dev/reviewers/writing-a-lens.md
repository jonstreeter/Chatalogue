# Writing a project-specific review lens

The four generic lenses shipped with this kit — architect, planner,
simplification-expert, code-reviewer — catch generic problems. They will not
catch the thing that actually breaks *your* project, because they do not know
what that is.

The lenses that earn their keep are the ones written against a specific failure
mode you have already been bitten by. Every one of them should be traceable to
a real incident, a real near-miss, or a real constraint the domain imposes.

## When to write one

Write a lens when all three hold:

1. A class of mistake has cost you real time **more than once**.
2. It is not caught by a test, a type, or a linter — if it could be, write that
   instead; a check that runs beats a check that must be remembered.
3. Catching it requires domain knowledge rather than reading the diff harder.

Do not write a lens for a rule. "Always use the shared client" is a rule; it
belongs in `AGENTS.md` or, better, in a lint check.

## Shape

Copy the structure of an existing lens. The parts that matter:

**A stance, not a checklist.** "You are a senior X who has seen Y fail in
production" produces different findings than a list of questions. State what
this reviewer refuses to accept.

**Adversarial framing.** The lens exists to find gaps. `Do not validate — find
the seams that will tear.` A lens that reports "looks good" was a wasted pass.

**Domain-specific checks.** This is the whole value. Not "check for races" but
"check whether this holds the GPU lease across an await" — the specific shape
your system fails in.

**Severity calibration tied to the contract.** Say explicitly what makes a
finding BLOCKER here versus a preference. Without this, every lens reports
everything as critical and you stop reading them.

**A fixed output format.** So findings can be scanned and compared across
lenses.

## Adversarial lenses that pay off

Patterns worth adapting, from projects where they caught real problems:

- **The realist** — challenges plans designed in a calm environment against the
  conditions they will actually run in: a live event, a flaky network, a user in
  a hurry, a machine that is not yours.
- **The domain skeptic** — challenges what the core technology can actually
  deliver, against vendor claims and optimistic assumptions. Especially
  valuable where "the model will handle it" is doing load-bearing work.
- **The privacy/data auditor** — challenges retention, deletion, and blast
  radius for anything holding user data. "We'll figure it out later" is a
  finding.
- **The operator** — reviews from the seat of whoever runs this at 2am with no
  context, rather than the person who built it.

## Registering it

1. Add `reviewers/<name>.md`.
2. Add a row to the lens table in [`../README.md`](../README.md).
3. If it should run in the standard review pass, add it to
   [`../workflows/review-prd.md`](../workflows/review-prd.md).

Harness note: a tool with subagent support may spawn lenses in parallel, but the
lens text stays here and is referenced, never copied. A duplicated lens is a
forked lens, and the copy is the one that goes stale.
