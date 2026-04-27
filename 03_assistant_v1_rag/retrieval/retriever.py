from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from config.settings import SETTINGS


def get_embedding_function():
    if SETTINGS.embedding_provider == "ollama":
        return OllamaEmbeddings(model=SETTINGS.embedding_model)
    if SETTINGS.embedding_provider == "openai":
        return OpenAIEmbeddings(model=SETTINGS.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {SETTINGS.embedding_provider}")


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=SETTINGS.chroma_persist_dir,
        collection_name=SETTINGS.chroma_collection,
        embedding_function=get_embedding_function(),
    )


def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": SETTINGS.retriever_top_k})
