# TODO: Placeholder..
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from app import auth
from app.config import settings
from app.database import VectorDatabase
from app.embedding import HuggingFaceTextEmbedder
from app.models import Embedding, EmbeddingCreate, EmbeddingSearchQuery, Status

app = FastAPI()

db = VectorDatabase(db_path=Path("vectors.db"), embedding_length=settings.VECTOR_DIMENSIONS)
db.connect()
# TODO: Do this better
db._create_tables()

hf_text_embedder = HuggingFaceTextEmbedder(model_id=settings.EMBEDDING_MODEL_ID, local_files_only=False)


@app.get("/", response_model=Status)
async def root():
    return Status(status="OK")


@app.post("/embeddings/", response_model=Embedding)
def create_embedding(embedding_create: EmbeddingCreate, api_key: str = Depends(auth.get_api_key)):
    embedding = embedding_create.create_embedding(text_embedder=hf_text_embedder)
    db.insert_embedding(embedding)
    return embedding


@app.get("/embeddings/", response_model=list[Embedding])
def read_vectors(limit: int = 100, api_key: str = Depends(auth.get_api_key)):
    embeddings = db.get_all_embeddings(limit=limit)
    return embeddings


@app.get("/embeddings/{embedding_id}", response_model=Embedding)
def get_vector(embedding_id: str, api_key: str = Depends(auth.get_api_key)):
    embedding = db.get_embedding_by_id(embedding_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


# TODO: Create this
# @app.post("/embeddings/{embedding_id}", response_model=Embedding)
# def update_vector(
#     embedding_id: str,
#     embedding_create: EmbeddingCreate,
#     api_key: str = Depends(auth.get_api_key),
# ):
#     vector_data = Embedding()
#     return vector_data


@app.post("/search/", response_model=list[Embedding])
def search_vectors(search_query: EmbeddingSearchQuery, api_key: str = Depends(auth.get_api_key)):
    search_embedding = search_query.get_search_embedding(hf_text_embedder)
    search_results = db.search_vectors(search_embedding, search_query.top_k)
    return search_results
