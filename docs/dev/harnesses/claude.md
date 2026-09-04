# Adapter: Claude Code

Claude Code is the only harness in this kit with real automation, so it is the
one most at risk of quietly becoming the process instead of pointing at it.
Everything below is a **shim**: it calls `scripts/*.py` and adds no rules.

## Files

| File | Role |
| --- | --- |
| `CLAUDE.md` | One line: `@AGENTS.md`. No content of its own, ever. |
| `.claude/settings.json` | Hook wiring only |

## Hooks, and what each one is a shim for

| Hook | Calls | Layer-1 rule it front-loads | If absent |
| --- | --- | --- | --- |
| `SessionStart` | `check_sdd_docs.py status` | Conformance item 2 (load state) | Agent reads the repo to rediscover state, or misses a handoff |
| `PreToolUse` (Bash) | `sdd_bypass_guard.py` | Conformance item 4 (no bypass) | Pre-commit is skippable locally; CI still catches it |
| `PreToolUse` (edit tools) | `sdd_contract_hook.py` | Conformance item 3 (preflight) | Pre-commit and the CI contract-impact gate catch it at commit/PR time |
| `PostToolUse` (edit tools) | `sdd_tripwire.py` | The escalation tripwire in `AGENTS.md` | The agent still carries the tripwire in its instructions; the hook is a backstop for the mechanical cases |

Every row's last column is the point: nothing here is load-bearing. Delete the
whole hooks block and the repo still enforces the same process, later.

## The pre-edit gate and how to clear it

When an edit targets a path mapped in `docs/dev/contracts.json`, the hook denies
it until a current *preflight receipt* exists for every governing authority:

```
python scripts/check_sdd_docs.py preflight <paths>
```

That command prints the authorities and writes `.sdd/preflight-receipt.json`.
Design notes worth keeping in mind:

- **The unblock path is a documented command, not a hidden field.** The earlier
  version keyed off an undocumented `tool_input` field that appeared in no
  instruction file, so the first mapped edit deadlocked. Anything an agent must
  do to proceed has to be discoverable by reading the repo.
- **It is harness-neutral.** The receipt is written by the ordinary CLI, so a
  Codex or Kilo session clears the gate the same way — which matters because
  those sessions share the working tree with Claude sessions.
- **Receipts expire on content change.** The receipt stores a hash of each
  authority. Edit the spec, and everyone must re-read it.
- **It fails closed.** An unparseable registry denies the edit and says why. The
  earlier version returned "no mapped contract" on any error, silently
  disabling itself on a typo.

`.sdd/` is local state — it is gitignored, and never a source of truth.

## Skills and subagents

Do **not** copy workflow or reviewer text into `.claude/skills/` or
`.claude/agents/`. If you want slash-command ergonomics, a skill file should
contain a pointer and nothing else:

```markdown
---
name: verify
description: Run this project's verification loop.
---
Read `docs/dev/workflows/verify.md` and follow it.
```

The same applies to subagents: the lens content lives in `docs/dev/reviewers/`,
and an agent definition may point at it. A duplicated lens is a forked lens.
This is why the kit ships no skills or agents itself — generating them would
mean generating copies, and the copies are what rot.

## Permissions

`.claude/settings.json` is a shared, committed file that also holds permissions
and env vars. The bootstrap **merges** its hooks into an existing file rather
than overwriting it, so re-running the bootstrap to pick up kit improvements
does not destroy local configuration.
