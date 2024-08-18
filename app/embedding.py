from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class TextEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @staticmethod
    def validate_embedding_length(embedding: list[float]) -> None:
        if len(embedding) != settings.VECTOR_DIMENSIONS:
            raise ValueError(f"Embedding must have {settings.VECTOR_DIMENSIONS} dimensions.")


class FakeTextEmbedder(TextEmbedder):
    def embed_text(self, text: str) -> list[float]:
        embedding_vector = np.random.random(settings.VECTOR_DIMENSIONS).tolist()
        self.validate_embedding_length(embedding_vector)
        return embedding_vector


class HuggingFaceTextEmbedder(TextEmbedder):
    def __init__(self, model_id: str = settings.EMBEDDING_MODEL_ID, local_files_only: bool = True):
        self.model = SentenceTransformer(model_id, local_files_only=local_files_only)

    def embed_text(self, text: str) -> list[float]:
        text_input = [text]
        embedding_vector = self.model.encode(text_input).squeeze().tolist()
        self.validate_embedding_length(embedding_vector)
        return embedding_vector
