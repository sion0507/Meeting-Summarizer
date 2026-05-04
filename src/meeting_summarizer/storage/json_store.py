"""Lightweight JSON file storage helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    """Save JSON payload to path, converting dataclasses recursively."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(payload)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value
