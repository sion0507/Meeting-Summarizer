"""Similarity-based grouping for event candidates.

This module implements the MVP grouping rule without LLM calls:

- candidate embeddings are compared with cosine similarity;
- candidates are linked only when ``similarity > threshold``;
- the default threshold is ``0.8``;
- unlinked candidates are emitted as single-candidate groups.

Groups are connected components over the pairwise similarity graph.  This keeps
candidate grouping deterministic and makes singleton handling explicit while
leaving semantic synthesis to the downstream LLM-based event merge stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meeting_summarizer.storage.json_store import JsonStore, save_json
from meeting_summarizer.utils.ids import make_case_id

DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_GROUPS_ARTIFACT_NAME = "candidate_groups"


class CandidateGroupingError(RuntimeError):
    """Raised when candidate grouping input is invalid or cannot be saved."""


@dataclass(frozen=True, slots=True)
class SimilarCandidatePair:
    """A pair of candidates whose cosine similarity passed the threshold."""

    candidate_id_a: str
    candidate_id_b: str
    similarity: float


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    """Candidate IDs that should be merged into one final event case."""

    group_id: str
    candidate_ids: list[str]
    is_singleton: bool
    threshold: float
    links: list[SimilarCandidatePair]


def group_candidates(
    candidate_vectors: dict[str, list[float]],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[CandidateGroup]:
    """Group candidate IDs by strict cosine similarity threshold.

    Args:
        candidate_vectors: Mapping of ``candidate_id`` to its embedding vector.
            Insertion order is preserved in the output for deterministic
            artifacts.
        similarity_threshold: Strict threshold for linking two candidates.
            The MVP rule is ``similarity > 0.8``; therefore a score exactly
            equal to ``0.8`` is not grouped.

    Returns:
        Deterministically ordered candidate groups, including singleton groups
        for candidates with no links.
    """

    _validate_threshold(similarity_threshold)
    _validate_candidate_vectors(candidate_vectors)

    candidate_ids = list(candidate_vectors)
    parent = {candidate_id: candidate_id for candidate_id in candidate_ids}
    links_by_root_pair: dict[frozenset[str], SimilarCandidatePair] = {}

    for left_index, left_id in enumerate(candidate_ids):
        for right_id in candidate_ids[left_index + 1 :]:
            similarity = cosine_similarity(
                candidate_vectors[left_id], candidate_vectors[right_id]
            )
            if similarity > similarity_threshold:
                union(parent, left_id, right_id)
                pair_key = frozenset((left_id, right_id))
                links_by_root_pair[pair_key] = SimilarCandidatePair(
                    candidate_id_a=left_id,
                    candidate_id_b=right_id,
                    similarity=similarity,
                )

    members_by_root: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        root = find(parent, candidate_id)
        members_by_root.setdefault(root, []).append(candidate_id)

    groups: list[CandidateGroup] = []
    for members in members_by_root.values():
        member_set = set(members)
        links = [
            link
            for link in links_by_root_pair.values()
            if link.candidate_id_a in member_set and link.candidate_id_b in member_set
        ]
        groups.append(
            CandidateGroup(
                group_id=make_group_id(members),
                candidate_ids=members,
                is_singleton=len(members) == 1,
                threshold=similarity_threshold,
                links=links,
            )
        )

    return groups


def save_candidate_groups(
    groups: list[CandidateGroup],
    path: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
) -> Path:
    """Save candidate groups to the canonical JSON artifact location.

    If ``path`` is omitted, groups are saved through :class:`JsonStore` at
    ``data/vector_store/candidate_groups.json``.  A direct path can be supplied
    by tests or custom scripts.
    """

    if path is None:
        store = JsonStore(data_dir)
        return store.save_artifact(DEFAULT_GROUPS_ARTIFACT_NAME, groups)

    target = Path(path)
    save_json(target, groups)
    return target


def group_and_save_candidates(
    candidate_vectors: dict[str, list[float]],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> list[CandidateGroup]:
    """Group candidates and persist ``candidate_groups.json`` in one call."""

    groups = group_candidates(
        candidate_vectors,
        similarity_threshold=similarity_threshold,
    )
    save_candidate_groups(groups, path, data_dir=data_dir)
    return groups


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two same-dimensional vectors."""

    if len(left) != len(right):
        raise CandidateGroupingError(
            f"Vector dimension mismatch: {len(left)} != {len(right)}."
        )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise CandidateGroupingError("Candidate vectors must not be zero vectors.")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def make_group_id(candidate_ids: list[str]) -> str:
    """Create a stable group ID from the group's candidate IDs."""

    return make_case_id(candidate_ids).replace("case_", "group_", 1)


def find(parent: dict[str, str], candidate_id: str) -> str:
    """Find the disjoint-set root for ``candidate_id`` with path compression."""

    root = parent[candidate_id]
    if root != candidate_id:
        parent[candidate_id] = find(parent, root)
    return parent[candidate_id]


def union(parent: dict[str, str], left_id: str, right_id: str) -> None:
    """Join two disjoint-set components using stable lexical root ordering."""

    left_root = find(parent, left_id)
    right_root = find(parent, right_id)
    if left_root == right_root:
        return
    first_root, second_root = sorted((left_root, right_root))
    parent[second_root] = first_root


def _validate_threshold(similarity_threshold: float) -> None:
    if not isinstance(similarity_threshold, int | float):
        raise CandidateGroupingError("similarity_threshold must be a number.")
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise CandidateGroupingError(
            f"similarity_threshold must be between 0 and 1, got {similarity_threshold}."
        )


def _validate_candidate_vectors(candidate_vectors: dict[str, list[float]]) -> None:
    if not candidate_vectors:
        raise CandidateGroupingError("No candidate vectors were provided for grouping.")

    dimension: int | None = None
    for candidate_id, vector in candidate_vectors.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise CandidateGroupingError("candidate_id values must be non-empty strings.")
        if not isinstance(vector, list) or not vector:
            raise CandidateGroupingError(
                f"Vector for {candidate_id!r} must be a non-empty list of numbers."
            )
        if not all(isinstance(value, int | float) for value in vector):
            raise CandidateGroupingError(
                f"Vector for {candidate_id!r} must contain only numbers."
            )
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            raise CandidateGroupingError(
                f"Vector for {candidate_id!r} must not be zero vectors."
            )
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise CandidateGroupingError(
                f"All vectors must have the same dimension; {candidate_id!r} has {len(vector)}, expected {dimension}."
            )


def groups_to_payload(groups: list[CandidateGroup]) -> list[dict[str, Any]]:
    """Return JSON-ready candidate-group dictionaries.

    ``save_json`` already handles dataclasses, but this helper is useful for
    callers that need to inspect or send the payload before saving.
    """

    return [
        {
            "group_id": group.group_id,
            "candidate_ids": list(group.candidate_ids),
            "is_singleton": group.is_singleton,
            "threshold": group.threshold,
            "links": [
                {
                    "candidate_id_a": link.candidate_id_a,
                    "candidate_id_b": link.candidate_id_b,
                    "similarity": link.similarity,
                }
                for link in group.links
            ],
        }
        for group in groups
    ]
