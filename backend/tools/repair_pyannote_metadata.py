from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
import json
import sys


def normalize_pyannote_metadata() -> dict[str, object]:
    dist = importlib_metadata.distribution("pyannote-audio")
    meta_path = Path(getattr(dist, "_path", "")) / "METADATA"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not locate pyannote-audio METADATA at {meta_path}")

    original = meta_path.read_text(encoding="utf-8")
    updated_lines: list[str] = []
    removed: list[str] = []

    for line in original.splitlines():
        if line.strip() == "Requires-Dist: torchcodec>=0.7.0":
            removed.append(line)
            continue
        updated_lines.append(line)

    updated = "\n".join(updated_lines)
    if original.endswith("\n"):
        updated += "\n"

    if updated != original:
        meta_path.write_text(updated, encoding="utf-8")

    return {
        "metadata_path": str(meta_path),
        "removed": removed,
        "updated": bool(removed),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(normalize_pyannote_metadata(), indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
