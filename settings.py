"""Loads config.yaml (tunable parameters) and .env (models, paths, secrets)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@lru_cache
def load_config(path: str | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config_path = Path(path or os.getenv("RFP_CONFIG", ROOT / "config.yaml"))
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def section(name: str) -> dict[str, Any]:
    return dict(load_config().get(name, {}))
