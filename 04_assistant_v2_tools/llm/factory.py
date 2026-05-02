import os

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import SETTINGS


def get_llm(*, temperature: float | None = None) -> ChatOllama:
    temp = temperature if temperature is not None else SETTINGS.handler_temperature
    if SETTINGS.provider == "ollama":
        return ChatOllama(model=SETTINGS.model_name, temperature=temp)
    raise ValueError(f"Unsupported provider: {SETTINGS.provider}")


def get_router_llm() -> ChatOllama:
    return get_llm(temperature=SETTINGS.router_temperature)


def get_embeddings() -> Embeddings:
    if SETTINGS.embedding_provider == "ollama":
        return OllamaEmbeddings(model=SETTINGS.embedding_model)
    if SETTINGS.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=SETTINGS.embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    raise ValueError(f"Unsupported embedding provider: {SETTINGS.embedding_provider}")
