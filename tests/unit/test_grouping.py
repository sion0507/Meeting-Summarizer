import json
import math
from pathlib import Path

import pytest

from meeting_summarizer.linking.grouping import (
    CandidateGroupingError,
    cosine_similarity,
    group_and_save_candidates,
    group_candidates,
    save_candidate_groups,
)


def test_similarity_equal_to_threshold_is_not_grouped() -> None:
    groups = group_candidates(
        {
            "candidate_a": [1.0, 0.0],
            "candidate_b": [0.8, 0.6],
        },
        similarity_threshold=0.8,
    )

    assert len(groups) == 2
    assert all(group.is_singleton for group in groups)
    assert [group.candidate_ids for group in groups] == [["candidate_a"], ["candidate_b"]]


def test_similarity_greater_than_threshold_is_grouped() -> None:
    groups = group_candidates(
        {
            "candidate_a": [1.0, 0.0],
            "candidate_b": [0.81, math.sqrt(1.0 - 0.81**2)],
            "candidate_c": [0.0, 1.0],
        },
        similarity_threshold=0.8,
    )

    assert [group.candidate_ids for group in groups] == [
        ["candidate_a", "candidate_b"],
        ["candidate_c"],
    ]
    assert groups[0].is_singleton is False
    assert groups[0].group_id.startswith("group_")
    assert len(groups[0].links) == 1
    assert groups[0].links[0].similarity > 0.8
    assert groups[1].is_singleton is True


def test_grouping_uses_connected_components_for_candidate_pools() -> None:
    groups = group_candidates(
        {
            "candidate_a": [1.0, 0.0],
            "candidate_b": [0.81, math.sqrt(1.0 - 0.81**2)],
            "candidate_c": [0.31, math.sqrt(1.0 - 0.31**2)],
        },
        similarity_threshold=0.8,
    )

    assert [group.candidate_ids for group in groups] == [["candidate_a", "candidate_b", "candidate_c"]]
    assert cosine_similarity([1.0, 0.0], [0.31, math.sqrt(1.0 - 0.31**2)]) < 0.8


def test_candidate_groups_are_saved_to_canonical_artifact(tmp_path: Path) -> None:
    groups = group_and_save_candidates(
        {
            "candidate_a": [1.0, 0.0],
            "candidate_b": [0.81, math.sqrt(1.0 - 0.81**2)],
        },
        data_dir=tmp_path,
    )

    output_path = tmp_path / "vector_store" / "candidate_groups.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(groups) == 1
    assert payload[0]["group_id"] == groups[0].group_id
    assert payload[0]["candidate_ids"] == ["candidate_a", "candidate_b"]
    assert payload[0]["is_singleton"] is False
    assert payload[0]["threshold"] == 0.8
    assert payload[0]["links"][0]["similarity"] > 0.8


def test_candidate_groups_can_be_saved_to_custom_path(tmp_path: Path) -> None:
    groups = group_candidates({"candidate_a": [1.0, 0.0]})
    output_path = tmp_path / "custom_groups.json"

    saved_path = save_candidate_groups(groups, output_path)

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["is_singleton"] is True


def test_grouping_rejects_invalid_vectors() -> None:
    with pytest.raises(CandidateGroupingError, match="No candidate vectors"):
        group_candidates({})

    with pytest.raises(CandidateGroupingError, match="same dimension"):
        group_candidates({"candidate_a": [1.0], "candidate_b": [1.0, 0.0]})

    with pytest.raises(CandidateGroupingError, match="zero vectors"):
        group_candidates({"candidate_a": [0.0, 0.0]})
