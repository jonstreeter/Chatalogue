"""Refuse commands that disable this repo's commit-time gates.

Harness-neutral in two ways:

* As a CLI -- ``python scripts/sdd_bypass_guard.py "<command>"`` returns
  non-zero with an explanation. Any harness, wrapper, or human can call this;
  it needs nothing but Python.
* As a Claude Code PreToolUse Bash hook -- reads the payload on stdin and emits
  a deny decision.

Deliberately not implemented with ``jq``: the previous shell version failed
open on machines without jq, which is most fresh Windows installs -- exactly
the case where a bypass guard matters most.

Two false-positive classes are handled explicitly, because a guard that blocks
innocent commands gets disabled, and a disabled guard protects nothing:

* **Heredoc bodies.** ``git commit -F - <<'EOF' ... EOF`` carries prose that may
  discuss these very flags. Document text is not a command; bodies are stripped
  before matching. (This file's own commit tripped the earlier version.)
* **Unrelated tools.** ``grep -n`` and ``git log -n 5`` are not bypasses. A flag
  only counts when the command actually runs something that honors hooks.

This is a speed bump, not a security control. The real gate is CI, which
re-runs every commit-time check and cannot be skipped from a workstation.
"""

from __future__ import annotations

import json
import re
import sys

# Commands that run hooks. A bypass flag is only meaningful next to one.
HOOK_RUNNING = re.compile(r"(?:^|[\s;&|(])(?:git|hg|jj|pre-commit|husky|lefthook)\b")

HEREDOC = re.compile(
    r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1.*?^\s*\2\s*$",
    re.DOTALL | re.MULTILINE,
)

# (regex, what to do instead)
BYPASS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?:^|\s)--no-verify(?:\s|$)"),
        "Fix what the pre-commit hook reported instead of skipping it.",
    ),
    (
        re.compile(r"(?:^|[\s;&|(])git\s+commit\b[^;&|]*\s-[a-zA-Z]*n[a-zA-Z]*(?:\s|$)"),
        "`git commit -n` is --no-verify. Fix the hook failure instead.",
    ),
    (
        re.compile(r"(?:^|\s)SKIP=\S"),
        "SKIP= disables specific pre-commit hooks. Fix them instead.",
    ),
    (
        re.compile(r"(?:^|\s)PRE_COMMIT_ALLOW_NO_CONFIG=1"),
        "That runs commits with no hook config at all.",
    ),
    (
        re.compile(r"(?:^|\s)HUSKY=0"),
        "That disables husky hooks for the command.",
    ),
    (
        re.compile(r"core\.hooksPath\s*=\s*(?:/dev/null|''|\"\"|\s|$)"),
        "That unsets the repo's hook path.",
    ),
)


def strip_heredocs(command: str) -> str:
    """Remove heredoc bodies so document text is not read as a command."""
    stripped = HEREDOC.sub(lambda match: f"<<{match.group(2)}", command)
    if stripped != command:
        return stripped
    # Unterminated heredoc (the body continues past what we were given):
    # drop everything after the opening marker rather than scanning prose.
    opener = re.search(r"<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?", command)
    return command[: opener.end()] if opener else command


def check(command: str) -> str | None:
    """Return an explanation if *command* bypasses the gates, else None."""
    candidate = strip_heredocs(command)
    if not HOOK_RUNNING.search(candidate):
        return None
    for pattern, advice in BYPASS_PATTERNS:
        if pattern.search(candidate):
            return (
                "Blocked: this command bypasses the repository's commit gates.\n"
                f"{advice}\n"
                "If the hook itself is wrong, change the hook in a commit of its "
                "own so the fix is reviewable."
            )
    return None


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        reason = check(" ".join(argv[1:]))
        if reason:
            print(reason, file=sys.stderr)
            return 1
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001 - a hook bug must not break the session
        return 0
    if not isinstance(payload, dict):
        return 0
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    command = tool_input.get("command")
    if not isinstance(command, str):
        return 0
    reason = check(command)
    if not reason:
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
