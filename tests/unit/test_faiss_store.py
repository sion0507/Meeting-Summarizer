import json

import pytest

from meeting_summarizer.linking import FaissCandidateStore, FaissStoreError

pytest.importorskip("faiss")
pytest.importorskip("numpy")


def test_faiss_store_saves_loads_and_preserves_candidate_metadata(tmp_path) -> None:
    store = FaissCandidateStore(tmp_path)
    vectors = {
        "candidate_001": [1.0, 0.0, 0.0],
        "candidate_002": [0.0, 1.0, 0.0],
    }

    metadata = store.build_and_save(
        vectors,
        candidate_metadata={"candidate_001": {"source_file": "a.txt"}},
    )

    assert metadata.count == 2
    loaded_store = FaissCandidateStore(tmp_path)
    loaded_metadata = loaded_store.load()

    assert loaded_metadata == metadata
    assert loaded_store.candidate_id_for_index(0) == "candidate_001"
    assert loaded_store.vector_index_for_candidate_id("candidate_002") == 1
    assert (
        loaded_store.search([1.0, 0.0, 0.0], top_k=1)[0].candidate_id
        == "candidate_001"
    )


def test_faiss_store_detects_index_metadata_count_mismatch(tmp_path) -> None:
    store = FaissCandidateStore(tmp_path)
    store.build_and_save(
        {"candidate_001": [1.0, 0.0], "candidate_002": [0.0, 1.0]}
    )

    metadata_path = tmp_path / "candidate_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["count"] = 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FaissStoreError, match="count mismatch"):
        FaissCandidateStore(tmp_path).load()
