"""Backend data/runtime directory constants shared by main.py and routers.

Importing this module ensures the directories exist.
"""
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.parent
BACKEND_DIR = _BACKEND_DIR

BACKEND_RUNTIME_DIR = _BACKEND_DIR / "runtime"
IMAGES_DIR = _BACKEND_DIR / "data" / "images"
THUMBNAILS_DIR = _BACKEND_DIR / "data" / "thumbnails"
MANUAL_MEDIA_DIR = _BACKEND_DIR / "data" / "manual_media"
AVATARS_DIR = _BACKEND_DIR / "data" / "avatars"

for _dir in (BACKEND_RUNTIME_DIR, IMAGES_DIR, THUMBNAILS_DIR, MANUAL_MEDIA_DIR, AVATARS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
