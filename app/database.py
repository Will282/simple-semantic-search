import json
import sqlite3
import struct
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import sqlite_vec

from app.models import Embedding, EmbeddingSearchResult

# Define a generic type variable
Func = TypeVar("Func", bound=Callable[..., Any])


class VectorSerializer:
    @staticmethod
    def serialize(vector: list[float]) -> bytes:
        """serializes a list of floats into a compact "raw bytes" format"""
        return struct.pack("%sf" % len(vector), *vector)

    @staticmethod
    def deserialize(byte_data: bytes) -> list[float]:
        """Deserializes a compact 'raw bytes' format back into a list of floats"""
        # Calculate the number of floats based on the length of the byte_data
        num_floats = len(byte_data) // struct.calcsize("f")
        # Unpack the byte_data into a tuple of floats
        return list(struct.unpack("%sf" % num_floats, byte_data))


class VectorDatabase:
    sentence_table_name = "sentences"
    embedding_table_name = "vec_sentences"
    vector_serializer = VectorSerializer

    _db_connection_error_message = "Database connection is not established. Use VectorDatabase.connect() to connect"

    def __init__(self, db_path: Union[Path, str], embedding_length: int):
        self.db_path = db_path
        self.conn = None

        self.embedding_length = embedding_length

    def connect(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    @staticmethod
    def ensure_connection(func: Func) -> Func:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check connection has been initialised
            if self.conn is None:
                raise RuntimeError(self._db_connection_error_message)

            # Check connection is not closed or dead.
            try:
                self.conn.cursor()
            except sqlite3.ProgrammingError:
                raise RuntimeError(self._db_connection_error_message)

            return func(self, *args, **kwargs)

        return wrapper  # type: ignore

    def close(self):
        """Close the SQLite database connection."""
        if self.conn:
            self.conn.close()

    @ensure_connection
    def insert_vector(self, embedding: Embedding) -> None:
        """Insert a new vector into the database."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        embedding.validate_length(self.embedding_length)

        with self.conn:
            self.conn.execute(
                f"INSERT INTO {self.sentence_table_name} (id, text, metadata, timestamp) VALUES(?, ?, ?, ?)",
                (
                    embedding.id,
                    embedding.text,
                    json.dumps(embedding.metadata),
                    embedding.timestamp.isoformat(),
                ),
            )

            self.conn.execute(
                f"INSERT INTO {self.embedding_table_name} (id, embedding) VALUES(?, ?)",
                [embedding.id, self.vector_serializer.serialize(embedding.vector)],
            )

    @ensure_connection
    def delete_embedding(self, embedding_id: str) -> None:
        """Delete a vector from the database."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        with self.conn:
            self.conn.execute(f"DELETE FROM {self.sentence_table_name} WHERE id = ?", (embedding_id,))
            self.conn.execute(f"DELETE FROM {self.embedding_table_name} WHERE id = ?", (embedding_id,))

    @ensure_connection
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Embedding]:
        """Retrieve a embedding by its ID."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        with self.conn:
            cursor = self.conn.execute(
                f"""
                SELECT
                    {self.embedding_table_name}.id,
                    text,
                    embedding,
                    metadata,
                    timestamp
                FROM {self.embedding_table_name}
                LEFT JOIN {self.sentence_table_name} ON {self.sentence_table_name}.id = {self.embedding_table_name}.id
                WHERE {self.embedding_table_name}.id = ?
                """,
                (embedding_id,),
            )
            row = cursor.fetchone()

            if row:
                return Embedding(
                    id=row[0],
                    text=row[1],
                    vector=self.vector_serializer.deserialize(row[2]),
                    metadata=json.loads(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                )

            return None

    @ensure_connection
    def search_vectors(self, search_embedding: list[float], top_k: int = 3) -> list[EmbeddingSearchResult]:
        """Search for the closest embedding vectors to the given embedding."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        if len(search_embedding) != self.embedding_length:
            raise ValueError(
                f"Search embedding must have the same length as the embeddings in the database {self.embedding_length}."
            )

        with self.conn:
            rows = self.conn.execute(
                f"""
                SELECT
                    {self.embedding_table_name}.id,
                    text,
                    embedding,
                    metadata,
                    timestamp,
                    distance
                FROM {self.embedding_table_name}
                LEFT JOIN {self.sentence_table_name} ON {self.sentence_table_name}.id = {self.embedding_table_name}.id
                WHERE embedding MATCH ?
                    AND k = {top_k}
                ORDER BY distance
                """,
                [self.vector_serializer.serialize(search_embedding)],
            ).fetchall()

        vectors = [
            EmbeddingSearchResult(
                embedding=Embedding(
                    id=row[0],
                    text=row[1],
                    vector=self.vector_serializer.deserialize(row[2]),
                    metadata=json.loads(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                ),
                distance=row[5],
            )
            for row in rows
        ]

        return vectors

    @ensure_connection
    def get_all_vectors(self) -> list[Embedding]:
        """Retrieve all vectors from the database."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        with self.conn:
            rows = self.conn.execute(
                f"""
                SELECT
                    {self.embedding_table_name}.id,
                    text,
                    embedding,
                    metadata,
                    timestamp
                FROM {self.embedding_table_name}
                LEFT JOIN {self.sentence_table_name} ON {self.sentence_table_name}.id = {self.embedding_table_name}.id
                """
            ).fetchall()

            embeddings_vectors = [
                Embedding(
                    id=row[0],
                    text=row[1],
                    vector=self.vector_serializer.deserialize(row[2]),
                    metadata=json.loads(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                )
                for row in rows
            ]
            return embeddings_vectors

    @ensure_connection
    def _create_tables(self):
        """Create the sentence and embeddings tables if it doesn't exist."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        with self.conn:
            # Metadata Table
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.sentence_table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata TEXT,
                    timestamp TEXT
                )
            """
            )

            # Vector Index
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.embedding_table_name} USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{self.embedding_length}]
                );
            """
            )

    @ensure_connection
    def _drop_tables(self):
        """Drop the sentence and embeddings table."""
        # To keep mypy happy.
        if self.conn is None:
            raise RuntimeError(self._db_connection_error_message)

        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.sentence_table_name}")
            self.conn.execute(f"DROP TABLE IF EXISTS {self.embedding_table_name}")

    @ensure_connection
    def _truncate_tables(self):
        """Truncate the sentence and embeddings table."""
        self._drop_tables()
        self._create_tables()
