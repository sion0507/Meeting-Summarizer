"""FAISS-backed vector storage for event candidate embeddings.

The store persists two files together:

- ``candidates.faiss``: the FAISS index rows in vector order.
- ``candidate_metadata.json``: row-to-candidate metadata and index settings.

On load, the FAISS row count and dimension are validated against metadata so a
rerun cannot silently use mismatched vectors and candidate IDs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

METADATA_VERSION = 1
DEFAULT_INDEX_FILENAME = "candidates.faiss"
DEFAULT_METADATA_FILENAME = "candidate_metadata.json"


class FaissStoreError(RuntimeError):
    """Raised when FAISS index persistence or validation fails."""


@dataclass(frozen=True, slots=True)
class CandidateVectorRecord:
    """Metadata connecting one FAISS row to one event candidate."""

    vector_index: int
    candidate_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FaissStoreMetadata:
    """JSON-serializable metadata for a persisted candidate vector index."""

    version: int
    metric: str
    dimension: int
    count: int
    records: list[CandidateVectorRecord]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One similarity search hit from the FAISS store."""

    candidate_id: str
    score: float
    vector_index: int
    metadata: dict[str, Any]


class FaissCandidateStore:
    """Persist and load candidate vectors with a stable FAISS row mapping."""

    def __init__(
        self,
        vector_store_dir: str | Path = "data/vector_store",
        *,
        index_filename: str = DEFAULT_INDEX_FILENAME,
        metadata_filename: str = DEFAULT_METADATA_FILENAME,
        metric: str = "cosine",
    ) -> None:
        if metric != "cosine":
            raise ValueError("Only cosine metric is supported for MVP candidate grouping.")
        self.vector_store_dir = Path(vector_store_dir)
        self.index_path = self.vector_store_dir / index_filename
        self.metadata_path = self.vector_store_dir / metadata_filename
        self.metric = metric
        self._index: Any | None = None
        self._metadata: FaissStoreMetadata | None = None

    @property
    def metadata(self) -> FaissStoreMetadata:
        if self._metadata is None:
            raise FaissStoreError("FAISS metadata is not loaded. Call build() or load().")
        return self._metadata

    @property
    def index(self) -> Any:
        if self._index is None:
            raise FaissStoreError("FAISS index is not loaded. Call build() or load().")
        return self._index

    def build(
        self,
        candidate_vectors: dict[str, list[float]],
        *,
        candidate_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> FaissStoreMetadata:
        """Build an in-memory FAISS index from candidate vectors.

        ``dict`` insertion order is preserved, so callers can produce stable
        row numbers by passing candidates in deterministic order.
        """

        if not candidate_vectors:
            raise FaissStoreError("No candidate vectors were provided for FAISS build.")
        if len(candidate_vectors) != len(set(candidate_vectors)):
            raise FaissStoreError("candidate_id values must be unique.")

        faiss = _import_faiss()
        np = _import_numpy()

        candidate_ids = list(candidate_vectors)
        matrix = _vectors_to_numpy_matrix(candidate_vectors, np=np)
        dimension = int(matrix.shape[1])
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)

        extra_metadata = candidate_metadata or {}
        records = [
            CandidateVectorRecord(
                vector_index=vector_index,
                candidate_id=candidate_id,
                metadata=dict(extra_metadata.get(candidate_id, {})),
            )
            for vector_index, candidate_id in enumerate(candidate_ids)
        ]
        metadata = FaissStoreMetadata(
            version=METADATA_VERSION,
            metric=self.metric,
            dimension=dimension,
            count=len(records),
            records=records,
        )
        self._index = index
        self._metadata = metadata
        return metadata

    def save(self) -> None:
        """Atomically save the FAISS index and row metadata."""

        faiss = _import_faiss()
        metadata = self.metadata
        index = self.index
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        tmp_index_path = self.index_path.with_name(f".{self.index_path.name}.tmp")
        tmp_metadata_path = self.metadata_path.with_name(f".{self.metadata_path.name}.tmp")
        try:
            faiss.write_index(index, str(tmp_index_path))
            _write_metadata_json(tmp_metadata_path, metadata)
            tmp_index_path.replace(self.index_path)
            tmp_metadata_path.replace(self.metadata_path)
        except OSError as exc:
            raise FaissStoreError(
                f"Failed to save FAISS store under {self.vector_store_dir}."
            ) from exc
        finally:
            if tmp_index_path.exists():
                tmp_index_path.unlink()
            if tmp_metadata_path.exists():
                tmp_metadata_path.unlink()

    def build_and_save(
        self,
        candidate_vectors: dict[str, list[float]],
        *,
        candidate_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> FaissStoreMetadata:
        """Convenience method to rebuild and persist the candidate index."""

        metadata = self.build(
            candidate_vectors,
            candidate_metadata=candidate_metadata,
        )
        self.save()
        return metadata

    def load(self) -> FaissStoreMetadata:
        """Load a FAISS index and validate it against candidate metadata."""

        faiss = _import_faiss()
        if not self.index_path.exists():
            raise FaissStoreError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FaissStoreError(f"FAISS metadata not found: {self.metadata_path}")

        try:
            index = faiss.read_index(str(self.index_path))
            metadata = _read_metadata_json(self.metadata_path)
        except OSError as exc:
            raise FaissStoreError(
                f"Failed to load FAISS store under {self.vector_store_dir}."
            ) from exc
        self._validate_loaded(index, metadata)
        self._index = index
        self._metadata = metadata
        return metadata

    def search(self, query_vector: list[float], *, top_k: int = 5) -> list[SearchResult]:
        """Search the loaded index and return candidate metadata hits."""

        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        faiss = _import_faiss()
        np = _import_numpy()
        metadata = self.metadata
        if metadata.count == 0:
            return []

        matrix = _vectors_to_numpy_matrix({"query": query_vector}, np=np)
        if int(matrix.shape[1]) != metadata.dimension:
            raise FaissStoreError(
                "Query vector dimension mismatch: "
                f"expected {metadata.dimension}, got {int(matrix.shape[1])}."
            )
        faiss.normalize_L2(matrix)
        scores, indices = self.index.search(matrix, min(top_k, metadata.count))
        by_index = {record.vector_index: record for record in metadata.records}

        results: list[SearchResult] = []
        for score, vector_index in zip(scores[0], indices[0], strict=True):
            if int(vector_index) < 0:
                continue
            record = by_index[int(vector_index)]
            results.append(
                SearchResult(
                    candidate_id=record.candidate_id,
                    score=float(score),
                    vector_index=record.vector_index,
                    metadata=record.metadata,
                )
            )
        return results

    def candidate_id_for_index(self, vector_index: int) -> str:
        """Return the candidate ID stored at a FAISS row index."""

        for record in self.metadata.records:
            if record.vector_index == vector_index:
                return record.candidate_id
        raise KeyError(f"No candidate metadata for vector index {vector_index}.")

    def vector_index_for_candidate_id(self, candidate_id: str) -> int:
        """Return the FAISS row index for a candidate ID."""

        for record in self.metadata.records:
            if record.candidate_id == candidate_id:
                return record.vector_index
        raise KeyError(f"No vector index for candidate_id {candidate_id!r}.")

    def _validate_loaded(self, index: Any, metadata: FaissStoreMetadata) -> None:
        if metadata.version != METADATA_VERSION:
            raise FaissStoreError(
                f"Unsupported FAISS metadata version: {metadata.version}."
            )
        if metadata.metric != self.metric:
            raise FaissStoreError(
                f"FAISS metric mismatch: expected {self.metric}, got {metadata.metric}."
            )
        if metadata.dimension < 1:
            raise FaissStoreError("FAISS metadata dimension must be >= 1.")
        if int(index.d) != metadata.dimension:
            raise FaissStoreError(
                f"FAISS dimension mismatch: index={int(index.d)}, metadata={metadata.dimension}."
            )
        if int(index.ntotal) != metadata.count:
            raise FaissStoreError(
                f"FAISS count mismatch: index={int(index.ntotal)}, metadata={metadata.count}."
            )
        if len(metadata.records) != metadata.count:
            raise FaissStoreError(
                "FAISS metadata record count does not match declared count."
            )
        seen_candidate_ids: set[str] = set()
        seen_indices: set[int] = set()
        for expected_index, record in enumerate(metadata.records):
            if record.vector_index != expected_index:
                raise FaissStoreError(
                    "FAISS metadata vector_index values must be contiguous and ordered."
                )
            if not record.candidate_id.strip():
                raise FaissStoreError("FAISS metadata contains an empty candidate_id.")
            if record.candidate_id in seen_candidate_ids:
                raise FaissStoreError(
                    f"Duplicate candidate_id in FAISS metadata: {record.candidate_id!r}."
                )
            if record.vector_index in seen_indices:
                raise FaissStoreError(
                    f"Duplicate vector_index in FAISS metadata: {record.vector_index}."
                )
            seen_candidate_ids.add(record.candidate_id)
            seen_indices.add(record.vector_index)


def _vectors_to_numpy_matrix(candidate_vectors: dict[str, list[float]], *, np: Any) -> Any:
    dimension: int | None = None
    rows: list[list[float]] = []
    for candidate_id, vector in candidate_vectors.items():
        if not candidate_id.strip():
            raise FaissStoreError("candidate_id must be a non-empty string.")
        values = [float(value) for value in vector]
        if not values:
            raise FaissStoreError(f"Vector for {candidate_id!r} must not be empty.")
        if any(not math.isfinite(value) for value in values):
            raise FaissStoreError(
                f"Vector for {candidate_id!r} contains NaN or infinite values."
            )
        if math.sqrt(sum(value * value for value in values)) == 0.0:
            raise FaissStoreError(f"Vector for {candidate_id!r} has zero norm.")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise FaissStoreError(
                f"Vector dimension mismatch for {candidate_id!r}: "
                f"expected {dimension}, got {len(values)}."
            )
        rows.append(values)

    if dimension is None:
        raise FaissStoreError("No vectors were provided.")
    return np.asarray(rows, dtype="float32")


def _write_metadata_json(path: Path, metadata: FaissStoreMetadata) -> None:
    payload = asdict(metadata)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _read_metadata_json(path: Path) -> FaissStoreMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FaissStoreError(f"Invalid FAISS metadata JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise FaissStoreError("FAISS metadata JSON must be an object.")

    records_payload = payload.get("records")
    if not isinstance(records_payload, list):
        raise FaissStoreError("FAISS metadata records must be a list.")
    records = []
    for index, item in enumerate(records_payload):
        if not isinstance(item, dict):
            raise FaissStoreError(f"FAISS metadata record {index} must be an object.")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise FaissStoreError(
                f"FAISS metadata record {index}.metadata must be an object."
            )
        try:
            records.append(
                CandidateVectorRecord(
                    vector_index=int(item["vector_index"]),
                    candidate_id=str(item["candidate_id"]),
                    metadata=metadata,
                )
            )
        except KeyError as exc:
            raise FaissStoreError(
                f"FAISS metadata record {index} is missing {exc.args[0]!r}."
            ) from exc

    try:
        return FaissStoreMetadata(
            version=int(payload["version"]),
            metric=str(payload["metric"]),
            dimension=int(payload["dimension"]),
            count=int(payload["count"]),
            records=records,
        )
    except KeyError as exc:
        raise FaissStoreError(f"FAISS metadata is missing {exc.args[0]!r}.") from exc


def _import_faiss() -> Any:
    import faiss

    return faiss


def _import_numpy() -> Any:
    import numpy as np

    return np
