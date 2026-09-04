"""Focused tests for the blocking contract pre-edit adapter.

The adapter holds no policy: it must delegate to the shared engine and it must
fail closed. Both properties are asserted here because both have silently
regressed before.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HOOK_PATH = Path(__file__).parents[1] / "scripts" / "sdd_contract_hook.py"
SPEC = importlib.util.spec_from_file_location("sdd_contract_hook", HOOK_PATH)
assert SPEC and SPEC.loader
hook = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hook
SPEC.loader.exec_module(hook)

AUTHORITY = "docs/dev/specs/2026-08-09-contract-design.md"


def _root(tmp_path: Path) -> Path:
    path = tmp_path / AUTHORITY
    path.parent.mkdir(parents=True)
    path.write_text(
        "# Contract\n\n**Status:** Draft\n**Plan:** _(not yet written)_\n",
        encoding="utf-8",
    )
    registry = {
        "version": 1,
        "domains": [
            {
                "id": "example-domain",
                "summary": "Example.",
                "paths": ["backend/example/**"],
                "authorities": [{"path": AUTHORITY, "role": "contract"}],
            }
        ],
    }
    (tmp_path / "docs/dev/contracts.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    return tmp_path


def _payload(root: Path, file_path: str, tool: str = "Edit") -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "cwd": str(root),
        "tool_input": {"file_path": file_path},
    }


def _run_preflight(root: Path, *paths: str) -> None:
    """Do what any harness does to clear the gate: run the documented command."""
    domains = hook.validator.preflight_paths(root, list(paths))
    hook.validator.record_preflight(root, domains)


def test_mapped_edit_is_denied_with_exact_authority_context(tmp_path: Path) -> None:
    root = _root(tmp_path)
    response = hook.evaluate(_payload(root, "backend/example/new.py"))
    output = response["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    # The reason must reach the model, not only the user.
    assert AUTHORITY in output["permissionDecisionReason"]
    assert "preflight" in output["permissionDecisionReason"]


def test_nested_mapped_edit_is_denied(tmp_path: Path) -> None:
    """`backend/example/**` governs the whole subtree, not just one level."""
    root = _root(tmp_path)
    response = hook.evaluate(_payload(root, "backend/example/deep/nested/new.py"))
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_preflight_receipt_unblocks_the_edit(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _run_preflight(root, "backend/example/new.py")
    response = hook.evaluate(_payload(root, "backend/example/new.py"))
    assert response["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_changed_authority_invalidates_the_receipt(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _run_preflight(root, "backend/example/new.py")
    (root / AUTHORITY).write_text(
        "# Contract\n\n**Status:** Draft\n**Plan:** _(not yet written)_\n\nNew rule.\n",
        encoding="utf-8",
    )
    response = hook.evaluate(_payload(root, "backend/example/new.py"))
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_unmapped_edit_is_untouched(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert hook.evaluate(_payload(root, "frontend/other.ts")) == {}


def test_no_registry_is_not_a_gate(tmp_path: Path) -> None:
    assert hook.evaluate(_payload(tmp_path, "backend/example/new.py")) == {}


def test_invalid_registry_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "docs/dev/contracts.json").write_text("{ not json", encoding="utf-8")
    response = hook.evaluate(_payload(root, "backend/example/new.py"))
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "registry" in response["hookSpecificOutput"]["permissionDecisionReason"]


def test_other_write_tools_are_covered(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for tool in ("Write", "MultiEdit", "NotebookEdit"):
        response = hook.evaluate(_payload(root, "backend/example/new.py", tool=tool))
        assert response["hookSpecificOutput"]["permissionDecision"] == "deny", tool


def test_multiedit_edits_list_is_inspected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    response = hook.evaluate(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "MultiEdit",
            "cwd": str(root),
            "tool_input": {"edits": [{"file_path": "backend/example/new.py"}]},
        }
    )
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_non_edit_events_are_ignored(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert hook.evaluate({"hook_event_name": "PostToolUse", "cwd": str(root)}) == {}
    assert hook.evaluate(_payload(root, "backend/example/new.py", tool="Read")) == {}
