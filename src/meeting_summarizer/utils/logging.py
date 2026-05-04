"""Project logging helpers."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a predictable format."""

    resolved_level = getattr(logging, level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Invalid logging level: {level!r}")

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(name)
