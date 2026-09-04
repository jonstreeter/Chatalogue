"""Focused contract tests for scripts/check_sdd_docs.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_sdd_docs.py"
SPEC = importlib.util.spec_from_file_location("check_sdd_docs", SCRIPT_PATH)
assert SPEC and SPEC.loader
check_sdd_docs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_sdd_docs
SPEC.loader.exec_module(check_sdd_docs)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_pair(root: Path, *, newline: str = "\n") -> None:
    spec_path = "docs/dev/specs/2026-08-09-example-design.md"
    plan_path = "docs/dev/plans/2026-08-09-example.md"
    _write(
        root / spec_path,
        newline.join(
            [
                "# Example",
                "",
                "**Status:** Approved for implementation",
                f"**Plan:** `{plan_path}`",
                "",
                "Body **Status:** ignored.",
            ]
        ),
    )
    _write(
        root / plan_path,
        newline.join(
            [
                "# Example Plan",
                "",
                "**Status:** In progress",
                f"**Spec:** `{spec_path}`",
            ]
        ),
    )


def _authority(
    root: Path, path: str, status: str = "Approved for implementation"
) -> None:
    if "/adr/" in path:
        text = f"# ADR\n\n- **Status:** {status}\n"
    else:
        text = f"# Contract\n\n**Status:** {status}\n**Plan:** _(not yet written)_\n"
    _write(root / path, text)


def _registry(root: Path, domains: list[dict] | None = None) -> None:
    authority = "docs/dev/specs/2026-08-09-contract-design.md"
    _authority(root, authority)
    payload = {
        "version": 1,
        "domains": domains
        or [
            {
                "id": "example-domain",
                "summary": "Example contract.",
                "paths": ["backend/example/**"],
                "authorities": [{"path": authority, "role": "contract"}],
            }
        ],
    }
    _write(root / "docs/dev/contracts.json", json.dumps(payload))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )


def _git_fixture(root: Path) -> str:
    _registry(root)
    _write(root / "backend/example/item.py", "before\n")
    _write(root / "README.md", "before\n")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_valid_pair_accepts_lf_and_crlf(tmp_path: Path, newline: str):
    _valid_pair(tmp_path, newline=newline)
    assert check_sdd_docs.validate_repository(tmp_path) == []


def test_reports_invalid_duplicate_and_missing_statuses_together(tmp_path: Path):
    _valid_pair(tmp_path)
    spec = tmp_path / "docs/dev/specs/2026-08-09-example-design.md"
    spec.write_text(
        spec.read_text(encoding="utf-8").replace(
            "**Status:** Approved for implementation",
            "**Status:** Approved\n**Status:** Shipped",
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "docs/dev/plans/2026-08-09-example.md"
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("**Status:** In progress\n", ""),
        encoding="utf-8",
    )
    codes = sorted(
        violation.code for violation in check_sdd_docs.validate_repository(tmp_path)
    )
    assert codes == ["duplicate-status", "invalid-status", "missing-status"]


def test_draft_spec_accepts_exact_no_plan_marker(tmp_path: Path):
    _write(
        tmp_path / "docs/dev/specs/2026-08-09-draft-design.md",
        "# Draft\n\n**Status:** Draft\n**Plan:** _(not yet written)_\n",
    )
    assert check_sdd_docs.validate_repository(tmp_path) == []


def test_link_requires_exact_repository_case(tmp_path: Path):
    _valid_pair(tmp_path)
    spec = tmp_path / "docs/dev/specs/2026-08-09-example-design.md"
    spec.write_text(
        spec.read_text(encoding="utf-8").replace("plans/2026", "Plans/2026"),
        encoding="utf-8",
    )
    violations = check_sdd_docs.validate_repository(tmp_path)
    assert [(item.code, item.line) for item in violations] == [("missing-target", 4)]


def test_archived_pair_must_link_to_archived_counterpart(tmp_path: Path):
    _valid_pair(tmp_path)
    active_spec = tmp_path / "docs/dev/specs/2026-08-09-example-design.md"
    archived_spec = tmp_path / "docs/dev/specs/archive/2026-08-09-example-design.md"
    archived_spec.parent.mkdir(parents=True)
    active_spec.replace(archived_spec)
    violations = check_sdd_docs.validate_repository(tmp_path)
    assert any(item.code == "archive-mismatch" for item in violations)


def test_diagnostics_are_sorted_and_include_path_line(tmp_path: Path):
    _write(tmp_path / "docs/dev/plans/z.md", "# Z\n")
    _write(tmp_path / "docs/dev/specs/a.md", "# A\n")
    formatted = [item.format() for item in check_sdd_docs.validate_repository(tmp_path)]
    assert formatted == sorted(formatted)
    assert formatted[0].startswith("docs/dev/plans/z.md:1:")


def test_copied_harness_references_are_rejected_but_pointer_is_allowed(tmp_path: Path):
    _write(
        tmp_path / ".kilo/command/pointer.md",
        "Read docs/dev/README.md and canonical reviewers directly.\n",
    )
    assert check_sdd_docs.validate_repository(tmp_path) == []

    _write(
        tmp_path / ".kilo/agents/architect.md",
        "copied reviewer\n",
    )
    violations = check_sdd_docs.validate_repository(tmp_path)
    assert [item.code for item in violations] == ["copied-harness-doc"]


def test_cli_accepts_untracked_active_pair(tmp_path: Path):
    _valid_pair(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "SDD documentation validation passed."


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("{", "invalid-contract-registry-json"),
        ({"version": 2, "domains": []}, "unsupported-contract-registry-version"),
        ({"version": 1}, "invalid-contract-registry-fields"),
        (
            {"version": 1, "domains": [], "extra": True},
            "invalid-contract-registry-fields",
        ),
    ],
)
def test_registry_rejects_malformed_version_and_fields(
    tmp_path: Path, payload, code: str
):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    _write(tmp_path / "docs/dev/contracts.json", text)
    assert code in {item.code for item in check_sdd_docs.validate_repository(tmp_path)}


def test_registry_rejects_duplicate_ids_globs_and_invalid_domain_values(tmp_path: Path):
    authority = "docs/dev/specs/2026-08-09-contract-design.md"
    _authority(tmp_path, authority)
    domains = [
        {
            "id": "Bad_ID",
            "summary": "",
            "paths": ["backend/shared/**", "backend/shared/**"],
            "authorities": [{"path": authority, "role": "unknown"}],
        },
        {
            "id": "Bad_ID",
            "summary": "Duplicate.",
            "paths": ["backend/shared/**"],
            "authorities": [],
        },
    ]
    _registry(tmp_path, domains)
    codes = {item.code for item in check_sdd_docs.validate_repository(tmp_path)}
    assert {
        "duplicate-contract-domain",
        "duplicate-contract-glob",
        "invalid-contract-domain-id",
        "invalid-contract-domain-summary",
        "invalid-contract-authorities",
        "invalid-contract-authority-role",
    } <= codes


def test_registry_requires_exact_case_authority_path(tmp_path: Path):
    """Only the mixed-case file exists; the lowercase reference must not resolve.

    The registry is written directly rather than through `_registry`, which
    always creates its own lowercase authority as a side effect. With that file
    present the lowercase path resolves legitimately and the test proves
    nothing -- it only passed on Windows, where the case-insensitive filesystem
    made the two names the same file.
    """
    path = "docs/dev/specs/2026-08-09-Contract-design.md"
    _authority(tmp_path, path)
    _write(
        tmp_path / "docs/dev/contracts.json",
        json.dumps(
            {
                "version": 1,
                "domains": [
                    {
                        "id": "example-domain",
                        "summary": "Example.",
                        "paths": ["backend/example/**"],
                        "authorities": [{"path": path.lower(), "role": "contract"}],
                    }
                ],
            }
        ),
    )
    assert "missing-contract-authority" in {
        item.code for item in check_sdd_docs.validate_repository(tmp_path)
    }


@pytest.mark.parametrize("role", ["contract", "refines"])
def test_registry_rejects_archived_or_terminal_current_authority(
    tmp_path: Path, role: str
):
    path = "docs/dev/specs/archive/2026-08-09-old-design.md"
    _authority(tmp_path, path, "Superseded")
    _registry(
        tmp_path,
        [
            {
                "id": "example-domain",
                "summary": "Example.",
                "paths": ["backend/example/**"],
                "authorities": [{"path": path, "role": role}],
            }
        ],
    )
    assert "invalid-current-contract-authority" in {
        item.code for item in check_sdd_docs.validate_repository(tmp_path)
    }


def test_registry_accepts_archived_context_and_accepted_adr(tmp_path: Path):
    archived = "docs/dev/specs/archive/2026-08-09-old-design.md"
    adr = "docs/dev/adr/ADR-0001-example.md"
    _authority(tmp_path, archived, "Superseded")
    _authority(tmp_path, adr, "ACCEPTED - 2026-08-09")
    _registry(
        tmp_path,
        [
            {
                "id": "example-domain",
                "summary": "Example.",
                "paths": ["backend/example/**"],
                "authorities": [
                    {"path": adr, "role": "refines"},
                    {"path": archived, "role": "context"},
                ],
            }
        ],
    )
    assert not {
        item.code
        for item in check_sdd_docs.validate_repository(tmp_path)
        if item.code.startswith("invalid-current-contract")
    }


def test_registry_overlap_diagnostic_names_both_domains_and_patterns(tmp_path: Path):
    authority = "docs/dev/specs/2026-08-09-contract-design.md"
    domains = [
        {
            "id": "broad-domain",
            "summary": "Broad.",
            "paths": ["backend/services/**"],
            "authorities": [{"path": authority, "role": "contract"}],
        },
        {
            "id": "specific-domain",
            "summary": "Specific.",
            "paths": ["backend/services/connected_*"],
            "authorities": [{"path": authority, "role": "contract"}],
        },
    ]
    _registry(tmp_path, domains)
    overlap = next(
        item
        for item in check_sdd_docs.validate_repository(tmp_path)
        if item.code == "overlapping-contract-glob"
    )
    assert all(
        value in overlap.message
        for value in (
            "broad-domain",
            "backend/services/**",
            "specific-domain",
            "backend/services/connected_*",
        )
    )


def test_preflight_normalizes_relative_absolute_windows_and_nonexistent_paths(
    tmp_path: Path,
):
    _registry(tmp_path)
    paths = [
        "backend/example/new.py",
        tmp_path / "backend/example/absolute.py",
        "backend\\example\\windows.py",
    ]
    result = check_sdd_docs.preflight_paths(tmp_path, paths)
    assert result[0]["matched_paths"] == [
        "backend/example/new.py",
        "backend/example/absolute.py",
        "backend/example/windows.py",
    ]


def test_preflight_deduplicates_domains_and_paths_and_ignores_unmatched(tmp_path: Path):
    _registry(tmp_path)
    result = check_sdd_docs.preflight_paths(
        tmp_path,
        ["backend/example/a.py", "backend/example/a.py", "README.md"],
    )
    assert len(result) == 1
    assert result[0]["id"] == "example-domain"
    assert result[0]["matched_paths"] == ["backend/example/a.py"]


def test_preflight_cli_text_json_and_unmatched_output(tmp_path: Path):
    _registry(tmp_path)
    text = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "preflight",
            "backend/example/new.py",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert text.returncode == 0
    assert "DOMAIN example-domain" in text.stdout
    assert "READ contract docs/dev/specs/2026-08-09-contract-design.md" in text.stdout

    machine = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "preflight",
            "backend/example/new.py",
            "--json",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert machine.returncode == 0
    assert json.loads(machine.stdout)["domains"][0]["id"] == "example-domain"

    unmatched = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "preflight",
            "README.md",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unmatched.returncode == 0
    assert unmatched.stdout.strip() == "No mapped architectural contract."


def test_impact_passes_when_no_mapped_paths_changed(tmp_path: Path):
    base = _git_fixture(tmp_path)
    _write(tmp_path / "README.md", "after\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "unmapped")
    result = check_sdd_docs.impact_repository(tmp_path, base, "HEAD", "")
    assert result == ([], [])


def test_impact_requires_ack_and_reports_each_domain_once(tmp_path: Path):
    base = _git_fixture(tmp_path)
    _write(tmp_path / "backend/example/item.py", "after\n")
    _write(tmp_path / "backend/example/second.py", "after\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "mapped")
    domains, violations = check_sdd_docs.impact_repository(tmp_path, base, "HEAD", "")
    assert [domain["id"] for domain in domains] == ["example-domain"]
    assert [item.code for item in violations] == ["missing-contract-impact-ack"]


def test_impact_accepts_none_reason_and_exact_authority_paths(tmp_path: Path):
    base = _git_fixture(tmp_path)
    _write(tmp_path / "backend/example/item.py", "after\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "mapped")
    none = "Contract impact: none — implementation preserves the mapped contract"
    assert check_sdd_docs.impact_repository(tmp_path, base, "HEAD", none)[1] == []
    exact = "Contract impact: docs/dev/specs/2026-08-09-contract-design.md"
    assert check_sdd_docs.impact_repository(tmp_path, base, "HEAD", exact)[1] == []


def test_impact_rejects_unknown_or_wrong_case_authority_path(tmp_path: Path):
    base = _git_fixture(tmp_path)
    _write(tmp_path / "backend/example/item.py", "after\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "mapped")
    ack = "Contract impact: docs/dev/specs/2026-08-09-Contract-design.md"
    violations = check_sdd_docs.impact_repository(tmp_path, base, "HEAD", ack)[1]
    assert [item.code for item in violations] == ["invalid-contract-impact-path"]


def test_unknown_harness_is_enforced_only_by_git_registry_and_pr_body(tmp_path: Path):
    base = _git_fixture(tmp_path)
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".codex").exists()
    _write(tmp_path / "backend/example/item.py", "unknown harness edit\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "unknown harness")

    missing = check_sdd_docs.impact_repository(tmp_path, base, "HEAD", "")[1]
    assert [item.code for item in missing] == ["missing-contract-impact-ack"]
    acknowledged = check_sdd_docs.impact_repository(
        tmp_path,
        base,
        "HEAD",
        "Contract impact: none — unknown harness preserved the authority",
    )[1]
    assert acknowledged == []


# ---------------------------------------------------------------------------
# Glob semantics, status, and the fail-closed registry contract.
# ---------------------------------------------------------------------------


def _registry_repo(tmp_path):
    authority = tmp_path / "docs/dev/specs/2026-08-23-widget-design.md"
    authority.parent.mkdir(parents=True)
    authority.write_text(
        "# Widget\n\n**Status:** Draft\n**Plan:** _(not yet written)_\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/dev/contracts.json").write_text(
        json.dumps(
            {
                "version": 1,
                "domains": [
                    {
                        "id": "widget",
                        "summary": "Widget service.",
                        "paths": ["src/widget/**"],
                        "authorities": [
                            {
                                "path": "docs/dev/specs/2026-08-23-widget-design.md",
                                "role": "contract",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_double_star_is_recursive():
    """`PurePosixPath.match` treats `**` as one segment; the registry must not."""
    assert check_sdd_docs.path_matches("src/widget/api.py", "src/widget/**")
    assert check_sdd_docs.path_matches("src/widget/routes/deep/api.py", "src/widget/**")
    assert not check_sdd_docs.path_matches("src/other/api.py", "src/widget/**")


def test_single_star_does_not_cross_directories():
    assert check_sdd_docs.path_matches("src/a/b.py", "src/*/b.py")
    assert not check_sdd_docs.path_matches("src/a/x/b.py", "src/*/b.py")


def test_leading_double_star_matches_at_any_depth():
    assert check_sdd_docs.path_matches("c.py", "**/c.py")
    assert check_sdd_docs.path_matches("a/b/c.py", "**/c.py")


def test_nested_path_preflights_to_its_domain(tmp_path):
    root = _registry_repo(tmp_path)
    domains = check_sdd_docs.preflight_paths(root, ["src/widget/routes/deep/api.py"])
    assert [domain["id"] for domain in domains] == ["widget"]


def test_absent_registry_is_not_an_error(tmp_path):
    assert check_sdd_docs.preflight_paths(tmp_path, ["src/widget/api.py"]) == []


def test_invalid_registry_raises_rather_than_returning_nothing(tmp_path):
    root = _registry_repo(tmp_path)
    (root / "docs/dev/contracts.json").write_text("{ broken", encoding="utf-8")
    with pytest.raises(check_sdd_docs.ContractRegistryError):
        check_sdd_docs.preflight_paths(root, ["src/widget/api.py"])


def test_structurally_invalid_registry_raises(tmp_path):
    root = _registry_repo(tmp_path)
    (root / "docs/dev/contracts.json").write_text(
        json.dumps({"version": 2, "domains": []}), encoding="utf-8"
    )
    with pytest.raises(check_sdd_docs.ContractRegistryError):
        check_sdd_docs.preflight_paths(root, ["src/widget/api.py"])


def test_receipt_round_trip(tmp_path):
    root = _registry_repo(tmp_path)
    domains = check_sdd_docs.preflight_paths(root, ["src/widget/api.py"])
    assert check_sdd_docs.outstanding_authorities(root, domains)
    check_sdd_docs.record_preflight(root, domains)
    assert check_sdd_docs.outstanding_authorities(root, domains) == []


def test_receipt_expires_when_the_authority_changes(tmp_path):
    root = _registry_repo(tmp_path)
    domains = check_sdd_docs.preflight_paths(root, ["src/widget/api.py"])
    check_sdd_docs.record_preflight(root, domains)
    authority = root / "docs/dev/specs/2026-08-23-widget-design.md"
    authority.write_text(
        authority.read_text(encoding="utf-8") + "\nNew rule.\n", encoding="utf-8"
    )
    assert check_sdd_docs.outstanding_authorities(root, domains) == [
        "docs/dev/specs/2026-08-23-widget-design.md"
    ]


def test_plan_progress_counts_tracker_rows(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        "| # | Task | Files | Size | Status |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 1 | a | `x` | S | ✅ done |\n"
        "| 2 | b | `y` | M | ▶ in progress |\n"
        "| 3 | c | `z` | L | ☐ todo |\n"
        "| 4 | d | `w` | S | ⛔ blocked |\n",
        encoding="utf-8",
    )
    assert check_sdd_docs.plan_progress(plan) == {
        "done": 1,
        "wip": 1,
        "todo": 1,
        "blocked": 1,
    }


def test_status_reports_active_artifacts(tmp_path):
    root = _registry_repo(tmp_path)
    plan = root / "docs/dev/plans/2026-08-23-widget.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        "# Widget\n\n**Status:** In progress\n"
        "**Spec:** `docs/dev/specs/2026-08-23-widget-design.md`\n",
        encoding="utf-8",
    )
    state = check_sdd_docs.repository_status(root)
    assert [item["path"] for item in state["plans"]] == [
        "docs/dev/plans/2026-08-23-widget.md"
    ]
    assert state["plans"][0]["status"] == "In progress"
    assert [item["path"] for item in state["specs"]] == [
        "docs/dev/specs/2026-08-23-widget-design.md"
    ]


def test_status_skips_archived_artifacts(tmp_path):
    root = _registry_repo(tmp_path)
    archived = root / "docs/dev/specs/archive/2026-01-01-old-design.md"
    archived.parent.mkdir(parents=True)
    archived.write_text(
        "# Old\n\n**Status:** Shipped\n**Plan:** _(not yet written)_\n",
        encoding="utf-8",
    )
    paths = [item["path"] for item in check_sdd_docs.repository_status(root)["specs"]]
    assert all("archive" not in path for path in paths)


def _handoff(root: Path, relative: str, plan: str = "") -> None:
    plan_line = f"**Plan:** `{plan}`\n" if plan else ""
    _write(
        root / relative,
        f"# Handoff: example\n\n**Date:** 2026-08-24\n{plan_line}\n## Status\n\nMid-flight.\n",
    )


def test_accepts_active_and_archived_handoffs(tmp_path: Path):
    _valid_pair(tmp_path)
    _handoff(tmp_path, "docs/dev/handoffs/2026-08-24-example.md", "docs/dev/plans/a.md")
    _handoff(
        tmp_path,
        "docs/dev/handoffs/archive/2026-07/2026-07-19-example.md",
        "docs/dev/plans/a.md",
    )
    assert check_sdd_docs.validate_repository(tmp_path) == []


def test_rejects_handoff_outside_the_handoff_directory(tmp_path: Path):
    _valid_pair(tmp_path)
    _write(tmp_path / "docs/handoff-example.md", "# Handoff\n")
    rules = {v.code for v in check_sdd_docs.validate_repository(tmp_path)}
    assert "handoff-location" in rules


def test_rejects_undated_handoff_filename(tmp_path: Path):
    _valid_pair(tmp_path)
    _handoff(tmp_path, "docs/dev/handoffs/example.md")
    rules = {v.code for v in check_sdd_docs.validate_repository(tmp_path)}
    assert "handoff-name" in rules


def test_rejects_archive_month_that_disagrees_with_the_filename(tmp_path: Path):
    _valid_pair(tmp_path)
    _handoff(tmp_path, "docs/dev/handoffs/archive/2026-08/2026-07-19-example.md")
    rules = {v.code for v in check_sdd_docs.validate_repository(tmp_path)}
    assert "handoff-archive-month" in rules


def test_rejects_two_active_handoffs_for_one_plan(tmp_path: Path):
    _valid_pair(tmp_path)
    _handoff(tmp_path, "docs/dev/handoffs/2026-08-20-first.md", "docs/dev/plans/a.md")
    _handoff(tmp_path, "docs/dev/handoffs/2026-08-24-second.md", "docs/dev/plans/a.md")
    rules = {v.code for v in check_sdd_docs.validate_repository(tmp_path)}
    assert "handoff-superseded" in rules


def test_readme_and_template_are_not_treated_as_handoffs(tmp_path: Path):
    _valid_pair(tmp_path)
    _write(tmp_path / "docs/dev/handoffs/README.md", "# Handoffs\n")
    _write(
        tmp_path / "docs/dev/handoffs/_TEMPLATE.md",
        "# Handoff\n\n**Plan:** `docs/dev/plans/YYYY-MM-DD-<slug>.md`\n",
    )
    assert check_sdd_docs.validate_repository(tmp_path) == []
