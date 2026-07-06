"""Persistent .env configuration helpers shared by main.py and routers."""
import os
from pathlib import Path

from dotenv import set_key

ENV_PATH = Path(__file__).parent.parent / ".env"


def _set_env_persist(key: str, value: str):
    set_key(ENV_PATH, key, value)
    os.environ[key] = value
