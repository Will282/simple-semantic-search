import os


class Settings:
    API_KEY = os.getenv("API_KEY", "mysecretkey")
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "infgrad/stella-base-en-v2")
    VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "768"))


settings = Settings()
