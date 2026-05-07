"""Standard JSON artifact storage for the local MVP pipeline.

The pipeline is intentionally file-based.  This module centralizes JSON
encoding, indentation, directory creation, artifact paths, and typed helpers for
early pipeline outputs so reruns leave a predictable, debuggable structure.

Canonical artifact layout::

    data/
      parsed/parsed_documents.json
      segments/segments.json
      candidates/event_candidates.json
      cases/event_cases.json
      vector_store/candidate_groups.json

Callers may still use :func:`save_json` and :func:`load_json` for ad-hoc JSON
files, but pipeline stages should prefer :class:`JsonStore` artifact helpers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

from meeting_summarizer.schemas import ParsedDocument, Segment

JSON_ENCODING = "utf-8"
JSON_INDENT = 2

ArtifactName = Literal[
    "parsed_documents",
    "segments",
    "event_candidates",
    "candidate_groups",
    "event_cases",
]

ARTIFACT_RELATIVE_PATHS: dict[ArtifactName, Path] = {
    "parsed_documents": Path("parsed") / "parsed_documents.json",
    "segments": Path("segments") / "segments.json",
    "event_candidates": Path("candidates") / "event_candidates.json",
    "candidate_groups": Path("vector_store") / "candidate_groups.json",
    "event_cases": Path("cases") / "event_cases.json",
}

_OUTPUT_DIRECTORIES = (
    Path("raw"),
    Path("parsed"),
    Path("segments"),
    Path("candidates"),
    Path("cases"),
    Path("reports"),
    Path("vector_store"),
)

T = TypeVar("T")


class JsonStoreError(RuntimeError):
    """Raised when JSON artifacts cannot be saved, loaded, or decoded."""


class JsonStore:
    """Read and write standardized JSON artifacts under one data directory."""

    def __init__(self, root_dir: str | Path = "data") -> None:
        self.root_dir = Path(root_dir)

    def ensure_structure(self) -> None:
        """Create the standard data directories used by the MVP pipeline."""
        for relative_dir in _OUTPUT_DIRECTORIES:
            (self.root_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    def artifact_path(self, artifact_name: ArtifactName) -> Path:
        """Return the canonical path for a known pipeline artifact."""
        return self.root_dir / ARTIFACT_RELATIVE_PATHS[artifact_name]

    def save_artifact(self, artifact_name: ArtifactName, payload: Any) -> Path:
        """Save payload to the canonical path for ``artifact_name``."""
        path = self.artifact_path(artifact_name)
        save_json(path, payload)
        return path

    def load_artifact(self, artifact_name: ArtifactName) -> Any:
        """Load JSON from the canonical path for ``artifact_name``."""
        return load_json(self.artifact_path(artifact_name))

    def save_parsed_documents(self, documents: list[ParsedDocument]) -> Path:
        """Save parsed document metadata/text for debugging after parsing."""
        return self.save_artifact("parsed_documents", documents)

    def load_parsed_documents(self) -> list[ParsedDocument]:
        """Load and validate parsed documents from the canonical JSON artifact."""
        payload = self.load_artifact("parsed_documents")
        return load_dataclass_list(payload, ParsedDocument)

    def save_segments(self, segments: list[Segment]) -> Path:
        """Save generated segment text for debugging before LLM extraction."""
        return self.save_artifact("segments", segments)

    def load_segments(self) -> list[Segment]:
        """Load and validate segments from the canonical JSON artifact."""
        payload = self.load_artifact("segments")
        return load_dataclass_list(payload, Segment)


def save_json(path: str | Path, payload: Any) -> None:
    """Save JSON payload using repository-wide formatting conventions.

    - Creates parent directories automatically.
    - Uses UTF-8 encoding.
    - Preserves non-ASCII text with ``ensure_ascii=False``.
    - Uses two-space indentation for readable debug artifacts.
    - Writes atomically through a temporary sibling file before replacing the
      target path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(payload)
    json_text = json.dumps(
        serializable,
        ensure_ascii=False,
        indent=JSON_INDENT,
        sort_keys=False,
    )
    tmp_path = target.with_name(f".{target.name}.tmp")
    try:
        tmp_path.write_text(f"{json_text}\n", encoding=JSON_ENCODING)
        tmp_path.replace(target)
    except OSError as exc:
        raise JsonStoreError(f"Failed to save JSON artifact: {target}") from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_json(path: str | Path) -> Any:
    """Load a JSON artifact with consistent errors and UTF-8 decoding."""

    target = Path(path)
    try:
        return json.loads(target.read_text(encoding=JSON_ENCODING))
    except FileNotFoundError as exc:
        raise JsonStoreError(f"JSON artifact not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise JsonStoreError(f"Invalid JSON artifact: {target} ({exc})") from exc
    except OSError as exc:
        raise JsonStoreError(f"Failed to load JSON artifact: {target}") from exc


def load_dataclass_list(payload: Any, model: type[T]) -> list[T]:
    """Convert a JSON list of dictionaries into validated dataclass objects."""

    if not isinstance(payload, list):
        raise JsonStoreError(
            f"Expected a list while loading {model.__name__} objects, got {type(payload).__name__}."
        )

    items: list[T] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise JsonStoreError(
                f"Expected item {index} to be an object for {model.__name__}, "
                f"got {type(item).__name__}."
            )
        try:
            items.append(_dataclass_from_dict(model, item))
        except (TypeError, ValueError) as exc:
            raise JsonStoreError(
                f"Invalid {model.__name__} item at index {index}: {exc}"
            ) from exc
    return items


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _dataclass_from_dict(model: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(model):
        raise TypeError(f"{model!r} is not a dataclass type.")

    model_fields = {field.name: field for field in fields(model)}
    unexpected_fields = sorted(set(data) - set(model_fields))
    if unexpected_fields:
        raise ValueError(f"unexpected fields: {unexpected_fields}")

    missing_fields = sorted(
        field_name for field_name in model_fields if field_name not in data
    )
    if missing_fields:
        raise ValueError(f"missing fields: {missing_fields}")

    kwargs = {
        field_name: _coerce_value(field.type, data[field_name])
        for field_name, field in model_fields.items()
    }
    return model(**kwargs)  # type: ignore[misc]


def _coerce_value(expected_type: Any, value: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"expected list, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item) for item in value]

    if origin is Literal:
        if value not in args:
            raise ValueError(f"expected one of {args}, got {value!r}")
        return value

    if origin is None:
        return value

    if origin is not None and type(None) in args:
        if value is None:
            return None
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_value(non_none_args[0], value)

    return value
