import sqlite3

import numpy as np
import pytest

from app.database import VectorDatabase
from app.models import Embedding, EmbeddingSearchResult


@pytest.fixture
def db(request):
    """Fixture to create a VectorDatabase instance with a configurable in-memory SQLite database."""
    embedding_length = request.param  # Accessing the parameter passed by the parametrize decorator
    db_instance = VectorDatabase(":memory:", embedding_length=embedding_length)
    db_instance.connect()
    db_instance._create_tables()
    yield db_instance
    db_instance.close()


@pytest.mark.parametrize("db", [4], indirect=True)
def test_initialization(db):
    """Test that the VectorDatabase initializes correctly and connects."""
    assert db.conn is not None, "Database connection should be initialized."
    assert db.embedding_length == 4, "Embedding length should be set correctly."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_insert_vector(db):
    """Test inserting a vector into the database."""
    embedding = Embedding(text="Test sentence", vector=[0.1, 0.2, 0.3, 0.4], metadata={"key": "value"})
    db.insert_vector(embedding)

    retrieved_embedding = db.get_embedding_by_id(embedding.id)
    assert retrieved_embedding is not None, "Embedding should be inserted and retrievable."
    assert retrieved_embedding.id == embedding.id, "Retrieved embedding ID should match the inserted one."
    np.testing.assert_allclose(
        retrieved_embedding.vector, embedding.vector, err_msg="Retrieved vector should match the inserted vector."
    )


@pytest.mark.parametrize("db", [4], indirect=True)
def test_insert_vector_invalid_length(db):
    """Test inserting a vector with an invalid length."""
    embedding = Embedding(
        text="Test sentence with invalid vector length",
        vector=[0.1, 0.2],  # Invalid length, should be 4
        metadata={"key": "value"},
    )

    with pytest.raises(ValueError) as excinfo:
        db.insert_vector(embedding)
    assert "Vector length" in str(excinfo.value), "Should raise ValueError for invalid vector length."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_delete_embedding(db):
    """Test deleting a vector from the database."""
    embedding = Embedding(
        text="Test sentence to delete", vector=[0.5, 0.6, 0.7, 0.8], metadata={"delete_key": "delete_value"}
    )
    db.insert_vector(embedding)
    db.delete_embedding(embedding.id)

    retrieved_embedding = db.get_embedding_by_id(embedding.id)
    assert retrieved_embedding is None, "Embedding should be deleted and no longer retrievable."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_search_vectors(db):
    """Test searching for vectors in the database."""
    # Insert multiple embeddings for search
    embeddings = [
        Embedding(
            text=f"Test sentence {i}",
            vector=[i * 0.1, i * 0.2, i * 0.3, i * 0.4],
            metadata={"search_key": f"search_value_{i}"},
        )
        for i in range(5)
    ]
    for embedding in embeddings:
        db.insert_vector(embedding)

    search_vector = [0.2, 0.4, 0.6, 0.8]
    results = db.search_vectors(search_vector, top_k=3)
    assert len(results) == 3, "Should return top 3 closest embeddings."
    assert all(
        isinstance(result, EmbeddingSearchResult) for result in results
    ), "Results should be instances of EmbeddingSearchResult."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_search_vectors_invalid_length(db):
    """Test searching with an embedding vector of invalid length."""
    search_vector = [0.2, 0.4]  # Invalid length, should be 4
    with pytest.raises(ValueError) as excinfo:
        db.search_vectors(search_vector, top_k=3)
    assert "Search embedding must have the same length" in str(
        excinfo.value
    ), "Should raise ValueError for invalid search vector length."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_get_all_vectors(db):
    """Test retrieving all vectors from the database."""
    embeddings = [
        Embedding(
            text=f"Test sentence {i}", vector=[i * 0.1, i * 0.2, i * 0.3, i * 0.4], metadata={"key": f"value_{i}"}
        )
        for i in range(3)
    ]
    for embedding in embeddings:
        db.insert_vector(embedding)

    all_vectors = db.get_all_vectors()
    assert len(all_vectors) == 3, "Should retrieve all inserted vectors."
    assert all(
        isinstance(embedding, Embedding) for embedding in all_vectors
    ), "All retrieved items should be Embedding instances."


@pytest.mark.parametrize("db", [4], indirect=True)
def test_edge_cases(db):
    """Test various edge cases like inserting duplicates and handling missing IDs."""
    # Test inserting a duplicate ID
    embedding = Embedding(text="Test sentence", vector=[0.1, 0.2, 0.3, 0.4], metadata={"key": "value"})
    db.insert_vector(embedding)

    with pytest.raises(sqlite3.IntegrityError):
        db.insert_vector(embedding)  # Inserting the same embedding again should raise an error

    # Test retrieval of a non-existing ID
    result = db.get_embedding_by_id("non_existing_id")
    assert result is None, "Should return None for non-existing embedding ID."


@pytest.mark.parametrize("db", [4096], indirect=True)
def test_performance(db):
    """Test performance when inserting a large vector."""
    large_vector = [0.1] * 4096  # Assuming the VectorDatabase is initialized with 4096 embedding length

    embedding = Embedding(text="Test sentence with large vector", vector=large_vector, metadata={"key": "value"})
    db.insert_vector(embedding)

    retrieved_embedding = db.get_embedding_by_id(embedding.id)
    assert retrieved_embedding is not None, "Embedding with a large vector should be retrievable."
    np.testing.assert_allclose(
        retrieved_embedding.vector, embedding.vector, err_msg="Retrieved vector should match the inserted vector."
    )


@pytest.mark.parametrize("db", [4], indirect=True)
def test_create_and_drop_tables(db):
    """Test creating and dropping tables."""
    # Ensure tables exist after creation
    db._create_tables()
    db.conn.execute(f"SELECT 1 FROM {db.sentence_table_name} LIMIT 1")
    db.conn.execute(f"SELECT 1 FROM {db.embedding_table_name} LIMIT 1")

    # Drop tables and ensure they are removed
    db._drop_tables()
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute(f"SELECT 1 FROM {db.sentence_table_name} LIMIT 1")
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute(f"SELECT 1 FROM {db.embedding_table_name} LIMIT 1")


@pytest.mark.parametrize("db", [4], indirect=True)
def test_truncate_tables(db):
    """Test truncating (dropping and recreating) tables."""
    # Insert a vector before truncating
    embedding = Embedding(text="Test sentence before truncate", vector=[0.1, 0.2, 0.3, 0.4], metadata={"key": "value"})
    db.insert_vector(embedding)

    # Truncate tables and check that data is removed
    db._truncate_tables()
    result = db.get_embedding_by_id(embedding.id)
    assert result is None, "Truncate should remove all data from tables."

    # Ensure tables are still functional after truncation
    db.insert_vector(embedding)
    result = db.get_embedding_by_id(embedding.id)
    assert result is not None, "Tables should be functional after truncation."
