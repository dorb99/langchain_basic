from langchain_ollama import ChatOllama

from config.settings import SETTINGS


def get_llm():
    if SETTINGS.provider == "ollama":
        return ChatOllama(model=SETTINGS.model_name, temperature=0)
    raise ValueError(f"Unsupported provider: {SETTINGS.provider}")
