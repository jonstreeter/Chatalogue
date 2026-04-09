from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
import json
import sys


def normalize_clearvoice_metadata() -> dict[str, object]:
    dist = importlib_metadata.distribution("clearvoice")
    meta_path = Path(getattr(dist, "_path", "")) / "METADATA"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not locate ClearVoice METADATA at {meta_path}")

    original = meta_path.read_text(encoding="utf-8")
    updated = original
    replacements = {
        "Requires-Dist: numpy<2.0,>=1.24.3": "Requires-Dist: numpy>=1.24.3",
        "Requires-Dist: soundfile==0.12.1": "Requires-Dist: soundfile>=0.12.1",
    }
    changed = []
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            changed.append({"from": old, "to": new})

    if updated != original:
        meta_path.write_text(updated, encoding="utf-8")

    return {
        "metadata_path": str(meta_path),
        "changed": changed,
        "updated": bool(changed),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(normalize_clearvoice_metadata(), indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
