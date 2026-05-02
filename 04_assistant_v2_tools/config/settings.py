from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model_name: str = "llama3.2:3b"
    chroma_persist_dir: str = "./chroma_db"

    # embedding_provider: str = "ollama"
    # embedding_model: str = "nomic-embed-text"
    # chroma_collection: str = "assistant_v2"
    
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_collection: str = "assistant_v2_openai"

    chunk_size: int = 100
    chunk_overlap: int = 20
    retriever_top_k: int = 6
    rerank_top_n: int = 3
    router_temperature: float = 0.0
    handler_temperature: float = 0.2


SETTINGS = Settings()
