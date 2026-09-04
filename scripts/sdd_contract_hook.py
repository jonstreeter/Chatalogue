"""Claude Code adapter: block mapped edits until contract preflight has run.

This file is an ADAPTER, not policy. It translates one harness's hook payload
into a call to the harness-neutral engine in ``check_sdd_docs.py`` and back.
Every rule it enforces is defined there and is reachable from any tool:

    python scripts/check_sdd_docs.py preflight <path>

Running that command surfaces the governing authorities and writes the receipt
this hook checks. A harness with no hook support is not exempt -- the same
requirement is enforced for everyone by the pre-commit gate and by CI.

Fails closed: if the registry cannot be evaluated, the edit is denied with the
reason, because a gate that silently disappears on a typo is worse than none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Claude tools that write files. Bash is handled separately (see sdd_bypass_guard).
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "create_file", "edit_file"}

VALIDATOR_PATH = Path(__file__).with_name("check_sdd_docs.py")
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "sdd_contract_validator", VALIDATOR_PATH
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def _decision(decision: str, reason: str) -> dict:
    """PreToolUse response. `permissionDecisionReason` is the field Claude reads."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def _file_paths(tool_input: dict) -> list[str]:
    paths = []
    for key in ("file_path", "notebook_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict) and isinstance(edit.get("file_path"), str):
                paths.append(edit["file_path"])
    return list(dict.fromkeys(paths))


def evaluate(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("hook_event_name") != "PreToolUse":
        return {}
    if payload.get("tool_name") not in EDIT_TOOLS:
        return {}
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return {}
    paths = _file_paths(tool_input)
    cwd = payload.get("cwd")
    if not paths or not isinstance(cwd, str):
        return {}
    root = Path(cwd)

    try:
        domains = validator.preflight_paths(root, paths)
    except validator.ContractRegistryError as error:
        return _decision(
            "deny",
            "The contract registry (docs/dev/contracts.json) is invalid, so this "
            "edit cannot be checked against it. Fix the registry first:\n"
            + "\n".join(v.format() for v in error.violations),
        )
    except Exception as error:  # noqa: BLE001 - never crash the harness
        return _decision(
            "deny",
            f"Contract preflight could not run ({error!r}). Run "
            "`python scripts/check_sdd_docs.py validate` and fix what it reports.",
        )

    if not domains:
        return {}

    outstanding = validator.outstanding_authorities(root, domains)
    if not outstanding:
        return _decision(
            "allow",
            "Contract preflight receipt is current for every governing authority.",
        )

    listed = "\n".join(f"  - {path}" for path in outstanding)
    return _decision(
        "deny",
        "This path is governed by a contract you have not read yet.\n\n"
        "Run:\n"
        f"  python scripts/check_sdd_docs.py preflight {' '.join(paths)}\n\n"
        "then read these authorities before editing:\n"
        f"{listed}\n\n"
        "The command records the receipt that unblocks this edit. Re-read is "
        "required whenever an authority's content changes.",
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        response = evaluate(payload)
        if response:
            print(json.dumps(response))
    except Exception:  # noqa: BLE001 - a hook bug must not break the session
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
