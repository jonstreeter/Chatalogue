# Harness adapters

A **harness** is whatever is driving the work: Claude Code, Codex, Kilo, Roo,
Cursor, Aider, Zoo Coder, a future tool, or a human with an editor. This
directory holds one small adapter per harness, and defines what an adapter is
allowed to be.

## The three layers

The kit is deliberately split so that no harness owns any rule.

| Layer | Lives in | Nature | Who can use it |
| --- | --- | --- | --- |
| **1. Process** | `docs/dev/**.md` | Prose. The flow, the lenses, the escalation rules. | Anything that can read a file |
| **2. Capability** | `scripts/*.py` | Plain CLI. Validation, preflight, status, gates. | Anything that can run Python |
| **3. Adapter** | `.claude/`, `.kilocode/`, `.cursor/`, … | A pointer, plus optional automation that calls layer 2. | One harness each |

The rule that keeps this honest: **an adapter may not contain a rule that does
not exist in layer 1, and may not implement logic that belongs in layer 2.**
If you find yourself explaining the process inside a harness config file, that
content is in the wrong layer. Move it to `docs/dev/`, and leave a pointer.

Why it matters: every proprietary automation surface — Claude's hooks, Cursor's
rules, Kilo's modes — is a place where the process can quietly fork. Once two
harnesses disagree about the rules, the repo has no process, it has two. Layer 2
exists so that "run the gate" means the same thing everywhere, and layer 3 stays
too thin to drift.

## What every harness must do

This is the conformance contract. It is five items, all satisfiable by a tool
that can only read files and run commands.

1. **Read the process.** Start planning, review, and verification work from
   `docs/dev/README.md`.
2. **Load state before acting.** Run `python scripts/check_sdd_docs.py status`
   at the start of a session. It prints the active plans and their tracker
   progress, the newest handoff notes, and the uncommitted surface — the things
   a fresh session would otherwise rediscover by reading the repo.
3. **Preflight mapped code.** Before editing a path governed by
   `docs/dev/contracts.json`, run
   `python scripts/check_sdd_docs.py preflight <paths>` and read what it
   returns. This also writes the receipt that unblocks harnesses with a
   pre-edit gate.
4. **Never bypass the commit gates.** No `--no-verify`, no `SKIP=`, no
   disabling hooks. If a gate is wrong, change the gate in its own commit.
5. **End with the handhold block.** Every response that touches code, docs,
   tests, or git carries the five-line block from
   [`../workflows/handhold.md`](../workflows/handhold.md): position, one next
   action, the step after, whether input is needed or what is being monitored,
   and the worktree verdict. **Input** gates the `d` shortcut: append
   "— reply `d`" only when the agent is waiting for the user and `d` can
   trigger **Do next**; omit it while a process is being monitored. The block is what makes the
   process legible to the person driving, not only to the tool, and its
   one-key replies (`d`, `dd`, `p`) let them act without retyping.

## Automation is a convenience, never the guarantee

Some harnesses can enforce items 2–4 automatically; most cannot, and a human
committing by hand certainly cannot. So the kit never relies on that:

- **The real gates are `pre-commit` and CI.** Both run the same layer-2 scripts,
  both apply to every harness and every human equally, and CI cannot be skipped
  from a workstation.
- **Harness automation only moves a gate earlier.** Claude Code's pre-edit hook
  catches a contract violation before the edit rather than at commit time. That
  is a better experience, not a different rule. A harness without hooks reaches
  the same verdict a few minutes later.

Judge an adapter by that standard: if switching harnesses changes *when* you get
told, fine. If it changes *what you are told*, the adapter is wrong.

## Adapters in this repo

| Harness | Adapter file | What it does |
| --- | --- | --- |
| Any tool reading `AGENTS.md` (Codex, Cursor, Aider, Zed, …) | `AGENTS.md` | Pointer only. This is the default path. |
| Claude Code | `CLAUDE.md` → `AGENTS.md`, `.claude/settings.json` | Pointer, plus hooks that call layer-2 scripts early |
| Kilo Code | `.kilocode/rules/sdd.md` | Pointer only |
| Roo Code | `.roo/rules/sdd.md` | Pointer only |
| Cursor | `.cursor/rules/sdd.mdc` | Pointer only |
| Human / anything else | `docs/dev/README.md` | Read it; run the same commands |

See [`claude.md`](claude.md) for the one adapter with real automation, and what
it is and is not allowed to do.

## Adding a harness

1. Create its native rules file.
2. Put a pointer in it — the wording in the per-harness adapters here is a fine
   starting point. Nothing else.
3. If the harness has an automation surface, wire it to *existing* layer-2
   commands. If you need behavior that does not exist yet, add it to
   `scripts/` as a subcommand first, so every other harness gains it too.
4. Add a row to the table above.

If step 3 tempts you to write harness-specific logic, that is the signal that
layer 2 is missing something. Fix it there.
