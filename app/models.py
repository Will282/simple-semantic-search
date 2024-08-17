from datetime import datetime, timezone

import shortuuid
from pydantic import BaseModel, Field


class Embedding(BaseModel):
    id: str = Field(default_factory=lambda: shortuuid.uuid())
    text: str
    vector: list[float]
    metadata: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def validate_length(self, expected_length: int) -> None:
        vector_length = len(self.vector)
        if vector_length != expected_length:
            raise ValueError(f"Vector length {vector_length} does not match expected length of {expected_length}")


class EmbeddingSearchResult(BaseModel):
    embedding: Embedding
    distance: float
