from datetime import datetime, timezone
from typing import Annotated

import shortuuid
from pydantic import BaseModel, Field, field_validator

from app.embedding import TextEmbedder


class Embedding(BaseModel):
    id: str = Field(default_factory=lambda: shortuuid.uuid())
    text: str
    vector: list[float]
    metadata: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("id")
    @classmethod
    def name_must_be_valid_shortuuid(cls, value: str) -> str:

        alphabet = set(shortuuid.ShortUUID().get_alphabet())
        is_valid = len(value) == 22 and all(char in alphabet for char in value)
        if not is_valid:
            raise ValueError("Not a valid shortuuid")
        return value

    def validate_length(self, expected_length: int) -> None:
        vector_length = len(self.vector)
        if vector_length != expected_length:
            raise ValueError(f"Vector length {vector_length} does not match expected length of {expected_length}")


class EmbeddingSearchResult(BaseModel):
    embedding: Embedding
    distance: Annotated[float, Field(ge=0, le=1)]


class EmbeddingCreate(BaseModel):
    text: str
    metadata: dict[str, str]

    def create_embedding(self, text_embedder: TextEmbedder) -> Embedding:
        embedding = Embedding(
            text=self.text,
            vector=text_embedder.embed_text(self.text),
            metadata=self.metadata,
        )
        return embedding


class EmbeddingSearchQuery(BaseModel):
    query: str
    top_k: Annotated[int, Field(ge=0, le=100)]

    def get_search_embedding(self, text_embedder: TextEmbedder) -> list[float]:
        return text_embedder.embed_text(self.query)


class Status(BaseModel):
    status: str
