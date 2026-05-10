"""Embedding generation utilities for event candidates.

The MVP default embedding model is the locally stored ``nlpai-lab/KURE-v1``
SentenceTransformer model.  The model is loaded lazily on the first embedding
request so application startup and config loading do not eagerly allocate model
memory.  Korean text is tokenized with ``kiwipiepy`` before embedding according
to the project tokenization policy.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meeting_summarizer.schemas.event_candidate import EventCandidate

DEFAULT_EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
DEFAULT_EMBEDDING_MODEL_PATH = "./models/KURE-v1"
_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or returns invalid vectors."""


class BaseTextTokenizer(ABC):
    """Text tokenizer interface used before embedding generation."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return normalized tokens for one text."""

    def tokenize_to_text(self, text: str) -> str:
        """Return a whitespace-joined token string for embedding models."""

        tokens = self.tokenize(text)
        if not tokens:
            raise EmbeddingError("Cannot tokenize text with no tokenizable content.")
        return " ".join(tokens)


class KiwiMorphTokenizer(BaseTextTokenizer):
    """Korean morphological tokenizer backed by lazily loaded ``kiwipiepy``."""

    def __init__(self) -> None:
        self._kiwi: Any | None = None

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Tokenizer input must be a non-empty string.")
        kiwi = self._ensure_kiwi_loaded()
        tokens = [token.form for token in kiwi.tokenize(text) if token.form.strip()]
        if not tokens:
            raise EmbeddingError("kiwipiepy returned no tokens for embedding text.")
        return tokens

    def _ensure_kiwi_loaded(self) -> Any:
        if self._kiwi is not None:
            return self._kiwi
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()
        return self._kiwi


class RegexTokenizer(BaseTextTokenizer):
    """Small deterministic tokenizer used only by tests or explicit fallback runs."""

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Tokenizer input must be a non-empty string.")
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        if not tokens:
            raise EmbeddingError("RegexTokenizer returned no tokens for embedding text.")
        return tokens


class BaseEmbedder(ABC):
    """Abstract batch embedding interface used by the pipeline.

    Implementations must return one fixed-dimension vector for each input text
    in the same order.  Vectors are plain Python floats so callers are not tied
    to a specific numerical library.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the fixed embedding dimension produced by this embedder."""

    @abstractmethod
    def embed_texts(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        """Embed texts in batches while preserving input order."""

    def embed_candidates(
        self,
        candidates: Sequence[EventCandidate],
        *,
        batch_size: int | None = None,
    ) -> dict[str, list[float]]:
        """Return a ``candidate_id -> vector`` mapping for EventCandidate items."""

        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise EmbeddingError("EventCandidate.candidate_id values must be unique.")

        texts = [build_candidate_embedding_text(candidate) for candidate in candidates]
        vectors = self.embed_texts(texts, batch_size=batch_size)
        return dict(zip(candidate_ids, vectors, strict=True))


@dataclass(frozen=True, slots=True)
class HashingEmbedder(BaseEmbedder):
    """Deterministic local embedder for plumbing and repeatable tests.

    The production MVP path should use ``KureV1Embedder``.  This hash embedder
    remains useful for unit tests because it does not require a model file.
    """

    vector_size: int = 384
    tokenizer: BaseTextTokenizer | None = None

    def __post_init__(self) -> None:
        if self.vector_size < 1:
            raise ValueError("HashingEmbedder.vector_size must be >= 1.")
        if self.tokenizer is None:
            object.__setattr__(self, "tokenizer", RegexTokenizer())

    @property
    def dimension(self) -> int:
        return self.vector_size

    def embed_texts(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        _validate_batch_size(batch_size)
        text_list = _validate_texts(texts)
        vectors: list[list[float]] = []

        for batch in _batched(text_list, batch_size or len(text_list) or 1):
            vectors.extend(self._embed_one(text) for text in batch)
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokenizer = self.tokenizer
        if tokenizer is None:  # defensive; __post_init__ always sets it.
            raise EmbeddingError("HashingEmbedder tokenizer is not configured.")
        tokens = tokenizer.tokenize(text)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        return _l2_normalize(vector)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Lazy adapter for a locally stored SentenceTransformer embedding model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        tokenizer: BaseTextTokenizer | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        **model_kwargs: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.tokenizer = tokenizer or KiwiMorphTokenizer()
        self.model_kwargs = model_kwargs
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        self._ensure_model_loaded()
        if self._dimension is None:  # defensive; _ensure_model_loaded sets it.
            raise EmbeddingError("Embedding model dimension is unavailable.")
        return self._dimension

    def embed_texts(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        _validate_batch_size(batch_size)
        text_list = _validate_texts(texts)
        model = self._ensure_model_loaded()
        tokenized_texts = [self.tokenizer.tokenize_to_text(text) for text in text_list]
        encoded = model.encode(
            tokenized_texts,
            batch_size=batch_size or 32,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [_validate_vector(vector, self.dimension) for vector in encoded]

    def _ensure_model_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise EmbeddingError(
                f"Embedding model path does not exist for {self.model_name}: "
                f"{self.model_path}. Store the local model files there or set "
                "EMBEDDING_MODEL_PATH."
            )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self.model_path), **self.model_kwargs)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        if self._dimension < 1:
            raise EmbeddingError("Embedding model returned an invalid dimension.")
        return self._model


class KureV1Embedder(SentenceTransformerEmbedder):
    """Default local embedder for the ``nlpai-lab/KURE-v1`` model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
        *,
        tokenizer: BaseTextTokenizer | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            tokenizer=tokenizer,
            model_name=DEFAULT_EMBEDDING_MODEL,
            **model_kwargs,
        )


def build_candidate_embedding_text(candidate: EventCandidate) -> str:
    """Build the text used to embed an event candidate.

    ``embedding_text`` remains the primary source because upstream extraction is
    expected to prepare it from event-identity fields.  The fallback preserves
    robustness for hand-authored tests or partially migrated data.
    """

    fields = [
        candidate.embedding_text,
        candidate.title,
        candidate.summary,
        candidate.problem,
        candidate.action,
        candidate.result,
        " ".join(candidate.actors),
        " ".join(candidate.keywords),
    ]
    text = "\n".join(value.strip() for value in fields if value and value.strip())
    if not text:
        raise EmbeddingError(
            f"EventCandidate {candidate.candidate_id!r} has no text for embedding."
        )
    return text


def create_embedder(
    model_name: str | None = None,
    *,
    model_path: str | Path | None = None,
    tokenizer: BaseTextTokenizer | None = None,
) -> BaseEmbedder:
    """Create an embedder from configurable model name/path values.

    Supported values:
    - empty/``None``/``nlpai-lab/KURE-v1``/``kure-v1``: local KURE-v1 embedder
    - ``hashing`` or ``hashing:<dimension>``: deterministic test embedder
    - ``sentence-transformers:<local_path>``: generic local SentenceTransformer
    """

    raw_name = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    raw_path = Path(model_path or DEFAULT_EMBEDDING_MODEL_PATH)
    if raw_name in {DEFAULT_EMBEDDING_MODEL, "kure-v1", "KURE-v1"}:
        return KureV1Embedder(raw_path, tokenizer=tokenizer)
    if raw_name == "hashing":
        return HashingEmbedder(tokenizer=tokenizer or RegexTokenizer())
    if raw_name.startswith("hashing:"):
        _, raw_dimension = raw_name.split(":", 1)
        try:
            return HashingEmbedder(
                vector_size=int(raw_dimension),
                tokenizer=tokenizer or RegexTokenizer(),
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid hashing embedder dimension: {raw_dimension!r}"
            ) from exc
    if raw_name.startswith("sentence-transformers:"):
        _, transformer_path = raw_name.split(":", 1)
        return SentenceTransformerEmbedder(
            transformer_path,
            tokenizer=tokenizer or KiwiMorphTokenizer(),
            model_name=transformer_path,
        )
    raise ValueError(
        "Unsupported EMBEDDING_MODEL. Use 'nlpai-lab/KURE-v1', 'kure-v1', "
        "'hashing', 'hashing:<dim>', or 'sentence-transformers:<local_path>'."
    )


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _validate_batch_size(batch_size: int | None) -> None:
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1 when provided.")


def _validate_texts(texts: Sequence[str]) -> list[str]:
    text_list = list(texts)
    for index, text in enumerate(text_list):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(f"Text at index {index} must be a non-empty string.")
    return text_list


def _validate_vector(vector: Any, dimension: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != dimension:
        raise EmbeddingError(
            f"Embedding dimension mismatch: expected {dimension}, got {len(values)}."
        )
    if any(not math.isfinite(value) for value in values):
        raise EmbeddingError("Embedding vector contains NaN or infinite values.")
    return values


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise EmbeddingError("Embedding vector has zero norm.")
    return [value / norm for value in vector]
