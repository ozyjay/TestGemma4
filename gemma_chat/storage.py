"""Persistence helpers for settings, prompts, and conversations."""

import json
from pathlib import Path

from .config import (
    CONVERSATION_FILE_NAME,
    SETTINGS_DIR,
    SETTINGS_FILE,
    SYSTEM_PROMPT_FILE_NAME,
    SYSTEM_PROMPT_HISTORY_FILE_NAME,
)


def read_settings() -> dict:
    try:
        settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return settings if isinstance(settings, dict) else {}


def write_settings(settings: dict):
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass


def system_prompt_path(log_dir: Path | None) -> Path | None:
    return log_dir / SYSTEM_PROMPT_FILE_NAME if log_dir else None


def system_prompt_history_path(log_dir: Path | None) -> Path | None:
    return log_dir / SYSTEM_PROMPT_HISTORY_FILE_NAME if log_dir else None


def conversation_path(log_dir: Path | None) -> Path | None:
    return log_dir / CONVERSATION_FILE_NAME if log_dir else None
