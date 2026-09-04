"""Tests for the escalation tripwire reminder."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sdd_tripwire.py"
SPEC = importlib.util.spec_from_file_location("sdd_tripwire", SCRIPT)
assert SPEC and SPEC.loader
tripwire = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tripwire
SPEC.loader.exec_module(tripwire)

GUARDED = (
    ("migrations/", True, "a database migration"),
    ("src/auth/session.py", False, "session handling"),
)


def _run(monkeypatch, file_path: str, guarded=GUARDED) -> str:
    monkeypatch.setattr(tripwire, "GUARDED", guarded)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": file_path},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert tripwire.main() == 0
    return out.getvalue()


def test_empty_guarded_list_is_silent(monkeypatch) -> None:
    assert _run(monkeypatch, "migrations/0001.py", guarded=()) == ""


def test_directory_prefix_fires(monkeypatch) -> None:
    output = _run(monkeypatch, "migrations/0001_init.py")
    assert "database migration" in output


def test_nested_path_under_guarded_directory_fires(monkeypatch) -> None:
    output = _run(monkeypatch, "migrations/versions/0002_add.py")
    assert "database migration" in output


def test_exact_file_match_fires(monkeypatch) -> None:
    assert "session handling" in _run(monkeypatch, "src/auth/session.py")


def test_sibling_of_exact_match_is_silent(monkeypatch) -> None:
    assert _run(monkeypatch, "src/auth/other.py") == ""


def test_windows_separators_are_normalized(monkeypatch) -> None:
    assert "database migration" in _run(monkeypatch, "migrations\\0001_init.py")


def test_unguarded_path_is_silent(monkeypatch) -> None:
    assert _run(monkeypatch, "src/widget/api.py") == ""


def test_malformed_payload_is_silent(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert tripwire.main() == 0
    assert out.getvalue() == ""


def test_payload_without_file_path_is_silent(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"tool_input": {"command": "ls"}}))
    )
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert tripwire.main() == 0
    assert out.getvalue() == ""
