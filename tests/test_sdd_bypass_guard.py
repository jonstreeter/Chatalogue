"""Tests for the harness-neutral commit-gate bypass guard."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sdd_bypass_guard.py"
SPEC = importlib.util.spec_from_file_location("sdd_bypass_guard", SCRIPT)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

# Assembled so the literal never appears in this file as a bare token -- a
# guard that trips on its own test suite is not usable.
SKIP_FLAG = "--no" + "-verify"

HEREDOC_COMMIT = "\n".join(
    [
        "cat > msg.txt <<'EOF'",
        "Fixed the guard so {flag} is caught properly.",
        "EOF",
        "git commit {tail}",
    ]
)


def test_plain_commit_is_allowed() -> None:
    assert guard.check("git commit -m 'feat: thing'") is None


def test_long_flag_is_blocked() -> None:
    assert guard.check(f"git commit {SKIP_FLAG} -m x") is not None


def test_push_variant_is_blocked() -> None:
    assert guard.check(f"git push {SKIP_FLAG}") is not None


def test_short_flag_is_blocked() -> None:
    assert guard.check("git commit -n -m x") is not None


def test_bundled_short_flag_is_blocked() -> None:
    assert guard.check("git commit -nm x") is not None


def test_heredoc_body_is_not_a_command() -> None:
    """A commit message that discusses the flag is prose, not a bypass."""
    command = HEREDOC_COMMIT.format(flag=SKIP_FLAG, tail="-F msg.txt")
    assert guard.check(command) is None


def test_bypass_outside_the_heredoc_is_still_caught() -> None:
    command = HEREDOC_COMMIT.format(flag="nothing", tail=f"{SKIP_FLAG} -F msg.txt")
    assert guard.check(command) is not None


def test_non_vcs_command_with_the_flag_is_allowed() -> None:
    """Passing the literal to another tool is not an attempt to skip hooks."""
    assert guard.check(f"echo {SKIP_FLAG}") is None


def test_unrelated_dash_n_is_allowed() -> None:
    assert guard.check("grep -n pattern file.py") is None
    assert guard.check("git log -n 5") is None


def test_skip_env_is_blocked() -> None:
    assert guard.check("SKIP=ruff git commit -m x") is not None


def test_hooks_path_unset_is_blocked() -> None:
    assert guard.check("git -c core.hooksPath=/dev/null commit -m x") is not None


def test_hook_payload_emits_deny(monkeypatch) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": f"git commit {SKIP_FLAG} -m x"},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert guard.main(["sdd_bypass_guard.py"]) == 0
    decision = json.loads(out.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"]


def test_clean_payload_is_silent(monkeypatch) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -m x"},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert guard.main(["sdd_bypass_guard.py"]) == 0
    assert out.getvalue() == ""


def test_malformed_payload_is_silent(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert guard.main(["sdd_bypass_guard.py"]) == 0
    assert out.getvalue() == ""
