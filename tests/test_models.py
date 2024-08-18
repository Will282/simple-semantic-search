import os
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.config import settings
from app.embedding import FakeTextEmbedder
from app.models import (
    Embedding,
    EmbeddingCreate,
    EmbeddingSearchQuery,
    EmbeddingSearchResult,
)


def test_custom_settings(monkeypatch):
    # Patch the environment variables
    monkeypatch.setenv("API_KEY", "testkey")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("VECTOR_DIMENSIONS", "128")

    # Reinitialize settings to apply patched environment variables
    settings.API_KEY = os.getenv("API_KEY")
    settings.EMBEDDING_MODEL_ID = os.getenv("MODEL_NAME")
    settings.VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS"))

    assert settings.API_KEY == "testkey"
    assert settings.EMBEDDING_MODEL_ID == "test-model"
    assert settings.VECTOR_DIMENSIONS == 128


def test_embedding_creation():
    # Test successful creation
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]
    metadata = {"key": "value"}

    embedding = Embedding(text=text, vector=vector, metadata=metadata)

    assert embedding.id is not None
    assert embedding.text == text
    assert embedding.vector == vector
    assert embedding.metadata == metadata
    assert isinstance(embedding.timestamp, datetime)


def test_embedding_default_values():
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]

    embedding = Embedding(text=text, vector=vector)

    assert embedding.id is not None
    assert embedding.metadata == {}
    assert embedding.timestamp <= datetime.now(timezone.utc)


def test_embedding_valid_shortuuid():
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]
    short_uuid = "CTEQoRU8ChxRtkxjUsSdpD"

    embedding = Embedding(id=short_uuid, text=text, vector=vector)
    assert embedding.id == short_uuid


def test_embedding_invalid_shortuuid():
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]

    with pytest.raises(ValidationError) as exc_info:
        Embedding(id="invalid_uuid", text=text, vector=vector)

    assert "Not a valid shortuuid" in str(exc_info.value)


def test_embedding_validate_length():
    # Test validate_length method
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]

    embedding = Embedding(text=text, vector=vector)

    # Valid length
    embedding.validate_length(3)

    # Invalid length
    with pytest.raises(ValueError) as exc_info:
        embedding.validate_length(4)

    assert "Vector length 3 does not match expected length of 4" in str(exc_info.value)


def test_embedding_search_result_creation():
    # Test successful creation of EmbeddingSearchResult
    text = "Sample text"
    vector = [0.1, 0.2, 0.3]

    embedding = Embedding(text=text, vector=vector)
    distance = 0.5

    result = EmbeddingSearchResult(embedding=embedding, distance=distance)

    assert result.embedding == embedding
    assert result.distance == distance


def test_embedding_search_result_invalid_distance():
    # Test invalid distance value (e.g., negative distance)
    text = "Sample text"
    vector = [0.1, 0.2, 0.3, 0.4]

    embedding = Embedding(text=text, vector=vector)

    with pytest.raises(ValidationError):
        EmbeddingSearchResult(embedding=embedding, distance=-1.0)


def test_embedding_create_creation():
    # Test successful creation of EmbeddingCreate and embedding generation
    text = "Test text for embedding"
    metadata = {"author": "test"}

    embedding_create = EmbeddingCreate(text=text, metadata=metadata)
    text_embedder = FakeTextEmbedder()

    embedding = embedding_create.create_embedding(text_embedder)

    assert embedding.text == text
    assert embedding.metadata == metadata
    assert isinstance(embedding.vector, list)
    assert len(embedding.vector) == settings.VECTOR_DIMENSIONS


# TODO: Make this test work
# def test_embedding_create_with_invalid_text_embedder():
#     # Test EmbeddingCreate with an invalid embedding vector length (if applicable)
#     class InvalidTextEmbedder(FakeTextEmbedder):
#         def embed_text(self, text: str) -> list[float]:
#             # Return an invalid vector length
#             return np.random.random(settings.VECTOR_DIMENSIONS + 1).tolist()

#     text = "Test text for embedding"
#     metadata = {"author": "test"}

#     embedding_create = EmbeddingCreate(text=text, metadata=metadata)
#     text_embedder = InvalidTextEmbedder()

#     with pytest.raises(ValueError) as exc_info:
#         embedding_create.create_embedding(text_embedder)

#     assert "Vector length" in str(exc_info.value)


def test_search_query_creation():
    # Test successful creation of SearchQuery with default top_k
    query = "Find similar embeddings"

    search_query = EmbeddingSearchQuery(query=query)

    assert search_query.query == query
    assert search_query.top_k == 5


def test_search_query_custom_top_k():
    # Test successful creation of SearchQuery with custom top_k
    query = "Find similar embeddings"
    top_k = 10

    search_query = EmbeddingSearchQuery(query=query, top_k=top_k)

    assert search_query.query == query
    assert search_query.top_k == top_k


def test_search_query_invalid_top_k():
    # Test invalid top_k value (e.g., negative or zero)
    query = "Find similar embeddings"

    with pytest.raises(ValidationError):
        EmbeddingSearchQuery(query=query, top_k=0)


def test_search_query_large_top_k():
    # Test invalid top_k value which is tool large
    query = "Find similar embeddings"

    with pytest.raises(ValidationError):
        EmbeddingSearchQuery(query=query, top_k=10000)
