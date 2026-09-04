"""SDD escalation tripwire — mechanical backstop for the instruction-layer rule.

Pairs with the "Escalation Tripwire" section in AGENTS.md and the "Escalation"
section in docs/dev/README.md. Runs as a Claude Code PostToolUse hook.

This hook does not judge how big a change is and enforces no threshold. It only
reminds the agent to run the tripwire test it already carries, at the moment an
edit lands on a surface that is expensive to unwind after it ships. Judgment
stays in the instruction layer, where it belongs.

EDIT ME: populate GUARDED below with this project's expensive-to-reverse
surfaces. Authority for each guarded path is that path's own rollback cost,
not a chosen limit. Typical candidates:

* database migrations directories — migrations apply during deploys and often
  cannot be un-applied cleanly.
* auth/tenancy code — a wrong default leaks data across users.
* the storage/system-of-record layer — holds user data.

Leave GUARDED empty if nothing qualifies yet; the hook then stays silent.

Reads the PostToolUse payload on stdin. Prints nothing unless a guarded path
was touched, and never fails the tool call.

Deliberately dependency-free: works on workstations without ``jq``.
"""

from __future__ import annotations

import json
import sys

# (path prefix, prefix-is-directory, description)
GUARDED: tuple[tuple[str, bool, str], ...] = (
    # Storage of record: schema and persistence for SQLite/Postgres.
    ("backend/src/db/", True, "the storage layer (DB schema and system of record)"),
    # Shared API/DB contracts: changing these ripples across routers and services.
    ("backend/src/schemas.py", False, "the shared API/DB schema contracts"),
)

GUIDANCE = (
    "Before continuing, apply the AGENTS.md Escalation Tripwire: if this change "
    "alters a contract other code depends on, if the root cause sits in a "
    "different subsystem than the reported symptom, or if you are writing a "
    "workaround you would later have to undo, stop and offer the user spec+plan "
    "instead of finishing the patch. If none of those hold, say so in one line "
    "and carry on."
)


def guarded_surface(file_path: str) -> str | None:
    """Return the guarded-surface description for *file_path*, else ``None``."""
    normalized = file_path.replace("\\", "/").lower().lstrip("/")
    for prefix, is_directory, surface in GUARDED:
        prefix = prefix.lower()
        if normalized.startswith(prefix) if is_directory else normalized == prefix:
            return surface
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return 0

    surface = guarded_surface(file_path)
    if surface is None:
        return 0

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"This edit touches {surface}. {GUIDANCE}",
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # never fail the tool call on a hook bug
        raise SystemExit(0) from None
