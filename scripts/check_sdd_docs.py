"""Validate the repository's canonical SDD documentation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SPEC_STATUSES = {
    "Draft",
    "In Review",
    "Approved for implementation",
    "Shipped",
    "Superseded",
}
PLAN_STATUSES = {"Not started", "In progress", "Blocked", "Done"}
NO_PLAN_MARKER = "_(not yet written)_"
STATUS_RE = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$")
LINK_RE = re.compile(r"^\*\*(Spec|Plan):\*\*\s*(.+?)\s*$")
BACKTICK_PATH_RE = re.compile(r"`(docs/dev/[^`]+\.md)`")
HEADER_LIMIT = 24
HANDOFF_DIR = "docs/dev/handoffs"
HANDOFF_NAME_RE = re.compile(r"^(\d{4})-(\d{2})-\d{2}-[a-z0-9][a-z0-9-]*\.md$")
MISPLACED_HANDOFF_GLOBS = ("docs/handoff-*.md", "docs/archive/handoff-*.md")
CONTRACT_REGISTRY = "docs/dev/contracts.json"
PREFLIGHT_RECEIPT = ".sdd/preflight-receipt.json"
DOMAIN_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CURRENT_SPEC_STATUSES = {"Draft", "In Review", "Approved for implementation"}
AUTHORITY_ROLES = {"contract", "refines", "context"}


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.message}"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _tracked_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _repository_files(root: Path) -> set[str]:
    try:
        tracked = _tracked_files(root)
        return tracked | {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root)
            .as_posix()
            .startswith(("docs/dev/specs/", "docs/dev/plans/", "docs/dev/handoffs/"))
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }


def _markdown_files(root: Path, category: str) -> list[Path]:
    base = root / "docs" / "dev" / category
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.md") if path.name != "_TEMPLATE.md")


def _metadata(
    lines: list[str], pattern: re.Pattern[str]
) -> list[tuple[int, re.Match[str]]]:
    matches: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines[:HEADER_LIMIT], start=1):
        match = pattern.fullmatch(line.rstrip("\r\n"))
        if match:
            matches.append((index, match))
    return matches


def _validate_status(
    root: Path,
    path: Path,
    allowed: set[str],
) -> tuple[list[Violation], str | None]:
    relative = _relative(path, root)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    matches = _metadata(lines, STATUS_RE)
    if not matches:
        return [
            Violation(relative, 1, "missing-status", "missing canonical status line")
        ], None
    violations: list[Violation] = []
    if len(matches) > 1:
        violations.append(
            Violation(
                relative, matches[1][0], "duplicate-status", "multiple status lines"
            )
        )
    line_number, match = matches[0]
    status = match.group(1)
    if status not in allowed:
        expected = ", ".join(sorted(allowed))
        violations.append(
            Violation(
                relative,
                line_number,
                "invalid-status",
                f"{status!r} is not one of: {expected}",
            )
        )
    return violations, status


def _validate_link(
    root: Path,
    path: Path,
    kind: str,
    repository_files: set[str],
) -> tuple[list[Violation], str | None]:
    relative = _relative(path, root)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    matches = [item for item in _metadata(lines, LINK_RE) if item[1].group(1) == kind]
    if not matches:
        return [
            Violation(relative, 1, "missing-link", f"missing **{kind}:** header")
        ], None
    violations: list[Violation] = []
    if len(matches) > 1:
        violations.append(
            Violation(
                relative, matches[1][0], "duplicate-link", f"multiple {kind} headers"
            )
        )
    line_number, match = matches[0]
    value = match.group(2)
    if kind == "Plan" and value == NO_PLAN_MARKER:
        return violations, None
    path_match = BACKTICK_PATH_RE.search(value)
    if not path_match:
        violations.append(
            Violation(
                relative,
                line_number,
                "invalid-link",
                f"**{kind}:** must contain one backticked docs/dev path",
            )
        )
        return violations, None
    target = path_match.group(1)
    if target not in repository_files:
        violations.append(
            Violation(
                relative,
                line_number,
                "missing-target",
                f"{target} does not resolve exactly",
            )
        )
    return violations, target


def _validate_archive_pair(
    root: Path,
    path: Path,
    target: str | None,
) -> list[Violation]:
    if target is None:
        return []
    relative = _relative(path, root)
    source_archived = "/archive/" in relative
    target_archived = "/archive/" in target
    if source_archived != target_archived:
        return [
            Violation(
                relative,
                1,
                "archive-mismatch",
                f"archive location does not match paired artifact {target}",
            )
        ]
    return []


COMPLETED_SPEC_STATUSES = {"Shipped", "Superseded"}
COMPLETED_PLAN_STATUSES = {"Done"}


def _validate_archive_required(
    root: Path,
    path: Path,
    status: str,
    completed_statuses: set[str],
) -> list[Violation]:
    if status not in completed_statuses:
        return []
    relative = _relative(path, root)
    if "/archive/" not in relative:
        expected = ", ".join(sorted(completed_statuses))
        return [
            Violation(
                relative,
                1,
                "archive-required",
                f"completed artifact ({status}) must live under archive/; expected status to be one of: {expected}",
            )
        ]
    return []


def _validate_handoffs(root: Path) -> list[Violation]:
    """Handoffs live only in docs/dev/handoffs/, one active baton per plan."""
    violations: list[Violation] = []

    for pattern in MISPLACED_HANDOFF_GLOBS:
        for path in sorted(root.glob(pattern)):
            violations.append(
                Violation(
                    _relative(path, root),
                    1,
                    "handoff-location",
                    f"handoff must live under {HANDOFF_DIR}/ "
                    f"(active) or {HANDOFF_DIR}/archive/YYYY-MM/ (closed)",
                )
            )

    base = root / "docs" / "dev" / "handoffs"
    if not base.exists():
        return violations

    active_plans: dict[str, str] = {}
    for path in sorted(base.rglob("*.md")):
        relative = _relative(path, root)
        if path.parent == base:
            if path.name in {"README.md", "_TEMPLATE.md"}:
                continue
            if not HANDOFF_NAME_RE.fullmatch(path.name):
                violations.append(
                    Violation(
                        relative,
                        1,
                        "handoff-name",
                        "handoff filename must be YYYY-MM-DD-<slug>.md",
                    )
                )
                continue
            for plan in _handoff_plan_paths(path):
                if plan in active_plans:
                    violations.append(
                        Violation(
                            relative,
                            1,
                            "handoff-superseded",
                            "second active handoff for "
                            f"{plan}; archive {active_plans[plan]} "
                            "- one active baton per plan",
                        )
                    )
                else:
                    active_plans[plan] = relative
            continue

        parts = path.relative_to(base).parts
        match = HANDOFF_NAME_RE.fullmatch(path.name)
        if parts[0] != "archive" or len(parts) != 3 or match is None:
            violations.append(
                Violation(
                    relative,
                    1,
                    "handoff-name",
                    "archived handoff must be archive/YYYY-MM/YYYY-MM-DD-<slug>.md",
                )
            )
            continue
        expected = f"{match.group(1)}-{match.group(2)}"
        if parts[1] != expected:
            violations.append(
                Violation(
                    relative,
                    1,
                    "handoff-archive-month",
                    f"archive month folder must match the filename date ({expected})",
                )
            )
    return violations


def _handoff_plan_paths(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()[:HEADER_LIMIT]
    except OSError:
        return set()
    return {
        candidate
        for line in lines
        for candidate in BACKTICK_PATH_RE.findall(line)
        if candidate.startswith("docs/dev/plans/")
    }


def _validate_harnesses(root: Path, repository_files: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    # Harness directories may hold pointers, never copies of the process.
    # See docs/dev/harnesses/README.md for the layering rule this enforces.
    forbidden_parts = {
        ".kilo/agents",
        ".kilocode/agents",
        ".roo/agents",
        ".claude/agents",
        ".claude/skills",
    }
    for relative in sorted(repository_files):
        if not (root / relative).is_file():
            continue
        if not relative.endswith(".md"):
            continue
        if any(
            relative == part or relative.startswith(f"{part}/")
            for part in forbidden_parts
        ):
            violations.append(
                Violation(
                    relative,
                    1,
                    "copied-harness-doc",
                    "harness must point to docs/dev instead of copying canonical content",
                )
            )
    return violations


def load_contract_registry(root: Path) -> tuple[dict[str, Any] | None, list[Violation]]:
    path = root / CONTRACT_REGISTRY
    if not path.exists():
        return None, []
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        line = getattr(error, "lineno", 1)
        return None, [
            Violation(
                CONTRACT_REGISTRY, line, "invalid-contract-registry-json", str(error)
            )
        ]
    if not isinstance(value, dict):
        return None, [
            Violation(
                CONTRACT_REGISTRY,
                1,
                "invalid-contract-registry-fields",
                "top level must be an object with exactly version and domains",
            )
        ]
    return value, []


def _authority_status(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8-sig").splitlines()[:HEADER_LIMIT]:
        match = re.match(r"^-?\s*\*\*Status:\*\*\s*(.+)$", line)
        if match:
            return match.group(1).strip().strip("*")
    return None


@lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a repo-relative glob into an anchored regex.

    ``**`` is recursive (crosses ``/``), ``*`` and ``?`` are not. Python's
    ``PurePosixPath.match`` treats ``**`` as a single segment, which silently
    unmapped every nested file under a governed domain; this does not.
    """
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    return re.compile(f"^{''.join(out)}$")


def path_matches(candidate: str, pattern: str) -> bool:
    """True when repo-relative *candidate* is governed by *pattern*."""
    return bool(_glob_regex(pattern).fullmatch(candidate.strip("/")))


def _pattern_samples(pattern: str) -> set[str]:
    """Concrete probe paths a pattern could plausibly govern."""
    samples = set()
    for deep in ("sample/item", "sample", ""):
        candidate = pattern.replace("**", deep).replace("*", "sample")
        candidate = re.sub(r"/{2,}", "/", candidate).strip("/")
        if candidate:
            samples.add(candidate)
    return samples


def _patterns_overlap(left: str, right: str) -> bool:
    """Heuristic: two globs overlap if either's probe paths match the other."""
    for candidate in _pattern_samples(left) | _pattern_samples(right):
        if path_matches(candidate, left) and path_matches(candidate, right):
            return True
    return False


def validate_contract_registry(
    root: Path, registry: dict[str, Any], repository_files: set[str] | None = None
) -> list[Violation]:
    violations: list[Violation] = []
    repository_files = repository_files or _repository_files(root)
    if set(registry) != {"version", "domains"}:
        violations.append(
            Violation(
                CONTRACT_REGISTRY,
                1,
                "invalid-contract-registry-fields",
                "top level must contain exactly version and domains",
            )
        )
    if registry.get("version") != 1:
        violations.append(
            Violation(
                CONTRACT_REGISTRY,
                1,
                "unsupported-contract-registry-version",
                f"expected version 1, got {registry.get('version')!r}",
            )
        )
    domains = registry.get("domains")
    if not isinstance(domains, list) or not domains:
        violations.append(
            Violation(
                CONTRACT_REGISTRY,
                1,
                "invalid-contract-domains",
                "domains must be a non-empty array",
            )
        )
        return violations

    seen_ids: set[str] = set()
    seen_globs: dict[str, str] = {}
    valid_globs: list[tuple[str, str]] = []
    for index, domain in enumerate(domains, start=1):
        if not isinstance(domain, dict) or set(domain) != {
            "id",
            "summary",
            "paths",
            "authorities",
        }:
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "invalid-contract-domain-fields",
                    "domain must contain exactly id, summary, paths, and authorities",
                )
            )
            continue
        domain_id = domain.get("id")
        if not isinstance(domain_id, str) or not DOMAIN_ID_RE.fullmatch(domain_id):
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "invalid-contract-domain-id",
                    f"invalid lowercase kebab-case id {domain_id!r}",
                )
            )
        if domain_id in seen_ids:
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "duplicate-contract-domain",
                    f"duplicate domain id {domain_id!r}",
                )
            )
        if isinstance(domain_id, str):
            seen_ids.add(domain_id)
        if not isinstance(domain.get("summary"), str) or not domain["summary"].strip():
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "invalid-contract-domain-summary",
                    f"domain {domain_id!r} summary must be non-empty",
                )
            )
        paths = domain.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(item, str) and item for item in paths)
        ):
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "invalid-contract-paths",
                    f"domain {domain_id!r} paths must be a non-empty string array",
                )
            )
        else:
            for pattern in paths:
                owner = seen_globs.get(pattern)
                if owner is not None:
                    violations.append(
                        Violation(
                            CONTRACT_REGISTRY,
                            index,
                            "duplicate-contract-glob",
                            f"glob {pattern!r} is repeated by {owner!r} and {domain_id!r}",
                        )
                    )
                else:
                    seen_globs[pattern] = str(domain_id)
                    valid_globs.append((str(domain_id), pattern))

        authorities = domain.get("authorities")
        if not isinstance(authorities, list) or not authorities:
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    index,
                    "invalid-contract-authorities",
                    f"domain {domain_id!r} authorities must be a non-empty array",
                )
            )
            continue
        for authority in authorities:
            if not isinstance(authority, dict) or set(authority) != {"path", "role"}:
                violations.append(
                    Violation(
                        CONTRACT_REGISTRY,
                        index,
                        "invalid-contract-authority-fields",
                        "authority must contain exactly path and role",
                    )
                )
                continue
            authority_path = authority.get("path")
            role = authority.get("role")
            if role not in AUTHORITY_ROLES:
                violations.append(
                    Violation(
                        CONTRACT_REGISTRY,
                        index,
                        "invalid-contract-authority-role",
                        f"unknown authority role {role!r}",
                    )
                )
                continue
            if (
                not isinstance(authority_path, str)
                or authority_path not in repository_files
            ):
                violations.append(
                    Violation(
                        CONTRACT_REGISTRY,
                        index,
                        "missing-contract-authority",
                        f"{authority_path!r} does not resolve exactly",
                    )
                )
                continue
            if role == "context":
                continue
            status = _authority_status(root / authority_path)
            active_spec = (
                authority_path.startswith("docs/dev/specs/")
                and "/archive/" not in authority_path
                and status in CURRENT_SPEC_STATUSES
            )
            accepted_adr = authority_path.startswith("docs/dev/adr/") and bool(
                status and status.upper().startswith("ACCEPTED")
            )
            if not (active_spec or accepted_adr):
                violations.append(
                    Violation(
                        CONTRACT_REGISTRY,
                        index,
                        "invalid-current-contract-authority",
                        f"{authority_path} is not an active spec or accepted ADR",
                    )
                )

    for left_index, (left_domain, left_pattern) in enumerate(valid_globs):
        for right_domain, right_pattern in valid_globs[left_index + 1 :]:
            if left_domain == right_domain or not _patterns_overlap(
                left_pattern, right_pattern
            ):
                continue
            violations.append(
                Violation(
                    CONTRACT_REGISTRY,
                    1,
                    "overlapping-contract-glob",
                    f"{left_domain}:{left_pattern} overlaps {right_domain}:{right_pattern}",
                )
            )
    return violations


def _normalize_candidate(root: Path, value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    root_text = root.resolve().as_posix().rstrip("/")
    if text.lower().startswith(f"{root_text.lower()}/"):
        text = text[len(root_text) + 1 :]
    return text.removeprefix("./")


class ContractRegistryError(RuntimeError):
    """The registry exists but cannot be trusted to answer a preflight query.

    Callers must fail closed on this. Returning "no mapped contract" for a
    malformed registry silently disables every gate built on top of it.
    """

    def __init__(self, violations: list[Violation]) -> None:
        super().__init__("; ".join(v.format() for v in violations))
        self.violations = violations


def preflight_paths(root: Path, paths: list[str | Path]) -> list[dict[str, Any]]:
    """Return the contract domains governing *paths*.

    Raises ContractRegistryError if the registry is present but invalid.
    An absent registry is a legitimate "not adopted yet" state and returns [].
    """
    registry, violations = load_contract_registry(root)
    if violations:
        raise ContractRegistryError(violations)
    if registry is None:
        return []
    structural = validate_contract_registry(root, registry)
    if structural:
        raise ContractRegistryError(structural)
    normalized = list(dict.fromkeys(_normalize_candidate(root, path) for path in paths))
    matches: list[dict[str, Any]] = []
    for domain in registry["domains"]:
        matched = [
            candidate
            for candidate in normalized
            if any(path_matches(candidate, pattern) for pattern in domain["paths"])
        ]
        if matched:
            matches.append(
                {
                    "id": domain["id"],
                    "summary": domain["summary"],
                    "matched_paths": matched,
                    "authorities": domain["authorities"],
                }
            )
    return matches


def _authority_digest(root: Path, authority_path: str) -> str:
    """Content hash of an authority, so edits to it invalidate old receipts."""
    try:
        data = (root / authority_path).read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(data).hexdigest()[:16]


def required_authorities(domains: list[dict[str, Any]]) -> list[str]:
    """Authority paths a mapped edit must have surfaced, in registry order."""
    seen: dict[str, None] = {}
    for domain in domains:
        for authority in domain["authorities"]:
            if authority["role"] in {"contract", "refines"}:
                seen.setdefault(authority["path"], None)
    return list(seen)


def record_preflight(root: Path, domains: list[dict[str, Any]]) -> None:
    """Write the receipt proving these authorities were surfaced to the agent.

    Harness-neutral by construction: the receipt is produced by running the
    preflight command, which every harness and every human can run.
    """
    if not domains:
        return
    path = root / PREFLIGHT_RECEIPT
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    for authority in required_authorities(domains):
        existing[authority] = _authority_digest(root, authority)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def outstanding_authorities(root: Path, domains: list[dict[str, Any]]) -> list[str]:
    """Required authorities with no current receipt (never read, or changed since)."""
    required = required_authorities(domains)
    if not required:
        return []
    try:
        receipt = json.loads((root / PREFLIGHT_RECEIPT).read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            receipt = {}
    except (OSError, json.JSONDecodeError):
        receipt = {}
    return [
        authority
        for authority in required
        if receipt.get(authority) != _authority_digest(root, authority)
    ]


def _print_preflight(domains: list[dict[str, Any]]) -> None:
    if not domains:
        print("No mapped architectural contract.")
        return
    for index, domain in enumerate(domains):
        if index:
            print()
        print(f"DOMAIN {domain['id']}")
        print(domain["summary"])
        for authority in domain["authorities"]:
            print(f"READ {authority['role']} {authority['path']}")


def changed_contract_domains(
    root: Path, base: str, head: str = "HEAD"
) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return preflight_paths(root, result.stdout.splitlines())


def impact_repository(
    root: Path, base: str, head: str, acknowledgement: str
) -> tuple[list[dict[str, Any]], list[Violation]]:
    domains = changed_contract_domains(root, base, head)
    if not domains:
        return [], []
    lines = [line.strip() for line in acknowledgement.splitlines()]
    values = [
        line.removeprefix("Contract impact:").strip()
        for line in lines
        if line.startswith("Contract impact:")
    ]
    if not values:
        return domains, [
            Violation(
                "<contract-impact>",
                1,
                "missing-contract-impact-ack",
                "mapped changes require a Contract impact acknowledgement",
            )
        ]
    value = values[0]
    if re.fullmatch(r"none\s+[—-]\s+\S.+", value):
        return domains, []
    valid_paths = {
        authority["path"]
        for domain in domains
        for authority in domain["authorities"]
        if authority["role"] in {"contract", "refines"}
    }
    referenced = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [path for path in referenced if path not in valid_paths]
    if not referenced or invalid:
        return domains, [
            Violation(
                "<contract-impact>",
                1,
                "invalid-contract-impact-path",
                f"unknown affected authority path(s): {', '.join(invalid or referenced)}",
            )
        ]
    return domains, []


# --------------------------------------------------------------------------
# status / progress — harness-neutral session state and tracker gate.
# Any harness (or a human) runs these; nothing here is tool-specific.
# --------------------------------------------------------------------------

TRACKER_ROW_RE = re.compile(r"^\|\s*\d+\s*\|.*\|\s*([^|]+?)\s*\|\s*$")
STATUS_SYMBOLS = {"todo": "☐", "wip": "▶", "done": "✅", "blocked": "⛔"}


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout


def plan_progress(path: Path) -> dict[str, int]:
    """Count task-tracker rows by status symbol."""
    counts = {key: 0 for key in STATUS_SYMBOLS}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = TRACKER_ROW_RE.match(line.rstrip())
        if not match:
            continue
        cell = match.group(1)
        for key, symbol in STATUS_SYMBOLS.items():
            if symbol in cell:
                counts[key] += 1
                break
    return counts


def _handoff_notes(root: Path) -> list[str]:
    notes = [
        _relative(path, root)
        for path in sorted((root / "docs" / "dev" / "handoffs").glob("*.md"))
        if path.name != "_TEMPLATE.md"
    ]
    notes += [
        _relative(path, root) for path in sorted((root / "docs").glob("handoff-*.md"))
    ]
    return sorted(notes)


def repository_status(root: Path) -> dict[str, Any]:
    """Everything a fresh session needs to resume without re-reading the repo."""
    root = root.resolve()
    active_specs = []
    for path in _markdown_files(root, "specs"):
        if "/archive/" in _relative(path, root):
            continue
        _, status = _validate_status(root, path, SPEC_STATUSES)
        active_specs.append({"path": _relative(path, root), "status": status})
    active_plans = []
    for path in _markdown_files(root, "plans"):
        if "/archive/" in _relative(path, root):
            continue
        _, status = _validate_status(root, path, PLAN_STATUSES)
        active_plans.append(
            {
                "path": _relative(path, root),
                "status": status,
                "tasks": plan_progress(path),
            }
        )
    dirty = [line[3:] for line in _git(root, "status", "--porcelain").splitlines()]
    return {
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "specs": active_specs,
        "plans": active_plans,
        "handoffs": _handoff_notes(root),
        "uncommitted": dirty,
    }


def _print_status(state: dict[str, Any]) -> None:
    print(f"BRANCH {state['branch'] or '(unknown)'}")
    if state["handoffs"]:
        print("\nHANDOFF NOTES (read the newest first)")
        for note in state["handoffs"][-3:]:
            print(f"  {note}")
    if state["plans"]:
        print("\nACTIVE PLANS (the tracker is the source of truth for progress)")
        for plan in state["plans"]:
            tasks = plan["tasks"]
            total = sum(tasks.values())
            summary = f"{tasks['done']}/{total} done" if total else "no tracker rows"
            extra = f", {tasks['blocked']} blocked" if tasks["blocked"] else ""
            print(f"  [{plan['status']}] {plan['path']} - {summary}{extra}")
    if state["specs"]:
        print("\nACTIVE SPECS")
        for spec in state["specs"]:
            print(f"  [{spec['status']}] {spec['path']}")
    if not (state["plans"] or state["specs"] or state["handoffs"]):
        print("\nNo active SDD artifacts. Small change? Build and verify directly.")
    if state["uncommitted"]:
        print(f"\nUNCOMMITTED {len(state['uncommitted'])} path(s)")
        for entry in state["uncommitted"][:10]:
            print(f"  {entry}")
        if len(state["uncommitted"]) > 10:
            print(f"  ... {len(state['uncommitted']) - 10} more")


def progress_gate(root: Path, strict: bool) -> tuple[list[str], list[Violation]]:
    """Remind (or require) that an in-progress plan is updated alongside code.

    Advisory by default: a blocking gate that misfires teaches people to bypass
    the hook, and bypassing is the thing this kit exists to prevent.
    """
    root = root.resolve()
    staged = [
        line.strip()
        for line in _git(root, "diff", "--cached", "--name-only").splitlines()
        if line.strip()
    ]
    if not staged:
        return [], []
    code_changes = [
        path
        for path in staged
        if not path.startswith("docs/") and not path.endswith(".md")
    ]
    if not code_changes:
        return [], []
    stale: list[str] = []
    for path in _markdown_files(root, "plans"):
        relative = _relative(path, root)
        if "/archive/" in relative:
            continue
        _, status = _validate_status(root, path, PLAN_STATUSES)
        if status == "In progress" and relative not in staged:
            stale.append(relative)
    if not stale or not strict:
        return stale, []
    return stale, [
        Violation(
            plan,
            1,
            "stale-plan-tracker",
            "in-progress plan not updated alongside staged code changes",
        )
        for plan in stale
    ]


def validate_repository(root: Path) -> list[Violation]:
    root = root.resolve()
    repository_files = _repository_files(root)
    violations: list[Violation] = []
    for path in _markdown_files(root, "specs"):
        status_violations, status = _validate_status(root, path, SPEC_STATUSES)
        violations.extend(status_violations)
        violations.extend(
            _validate_archive_required(root, path, status, COMPLETED_SPEC_STATUSES)
        )
        link_violations, target = _validate_link(root, path, "Plan", repository_files)
        violations.extend(link_violations)
        violations.extend(_validate_archive_pair(root, path, target))
    for path in _markdown_files(root, "plans"):
        status_violations, status = _validate_status(root, path, PLAN_STATUSES)
        violations.extend(status_violations)
        violations.extend(
            _validate_archive_required(root, path, status, COMPLETED_PLAN_STATUSES)
        )
        link_violations, target = _validate_link(root, path, "Spec", repository_files)
        violations.extend(link_violations)
        violations.extend(_validate_archive_pair(root, path, target))
    violations.extend(_validate_handoffs(root))
    violations.extend(_validate_harnesses(root, repository_files))
    registry, registry_violations = load_contract_registry(root)
    violations.extend(registry_violations)
    if registry is not None:
        violations.extend(validate_contract_registry(root, registry, repository_files))
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["validate", "preflight", "impact", "status", "progress"],
        default="validate",
    )
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--ack-file", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="progress: fail instead of reminding when a tracker looks stale",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.command == "status":
        state = repository_status(root)
        if args.json:
            print(json.dumps(state, sort_keys=True))
        else:
            _print_status(state)
        return 0

    if args.command == "progress":
        stale, violations = progress_gate(root, args.strict)
        for plan in stale:
            label = "STALE" if violations else "REMINDER"
            print(f"{label} {plan}: in progress, but not updated with this commit.")
        for violation in violations:
            print(violation.format())
        return 1 if violations else 0

    if args.command == "preflight":
        if not args.paths:
            parser.error("preflight requires at least one path")
        try:
            domains = preflight_paths(root, args.paths)
        except ContractRegistryError as error:
            for violation in error.violations:
                print(violation.format())
            print("Contract registry is unusable; fix it before editing mapped code.")
            return 1
        record_preflight(root, domains)
        if args.json:
            print(json.dumps({"domains": domains}, sort_keys=True))
        else:
            _print_preflight(domains)
        return 0

    if args.command == "impact":
        if not args.base:
            parser.error("impact requires --base REF")
        acknowledgement = ""
        if args.ack_file and args.ack_file.exists():
            acknowledgement = args.ack_file.read_text(encoding="utf-8-sig")
        try:
            domains, violations = impact_repository(
                root, args.base, args.head, acknowledgement
            )
        except ContractRegistryError as error:
            for violation in error.violations:
                print(violation.format())
            return 1
        _print_preflight(domains)
        for violation in violations:
            print(violation.format())
        return 1 if violations else 0

    violations = validate_repository(root)
    for violation in violations:
        print(violation.format())
    if violations:
        print(f"SDD documentation validation failed: {len(violations)} violation(s).")
        return 1
    print("SDD documentation validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
