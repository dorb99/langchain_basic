from dataclasses import dataclass


# dataclass auto-generates boilerplate methods like __init__ and __repr__
# from the typed fields below, so config stays clean and explicit.
# We use frozen=True so settings are immutable after creation, which helps
# avoid accidental runtime changes to critical app configuration.
@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model_name: str = "llama3.2:3b"
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 100
    chunk_overlap: int = 20
    retriever_top_k: int = 3

    # embedding_provider: str = "ollama"
    # embedding_model: str = "nomic-embed-text"
    # chroma_collection: str = "assistant_v1_nomic"

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_collection: str = "assistant_v1_openai"


SETTINGS = Settings()
