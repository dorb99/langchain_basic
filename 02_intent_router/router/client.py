from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import HANDLER_TEMPERATURE, OLLAMA_MODEL, ROUTER_TEMPERATURE, OPENAI_MODEL


# def _build_client(*, temperature: float) -> ChatOllama:
#     return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)


def _build_client(*, temperature: float) -> ChatOpenAI:
    # return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)


def get_router_client() -> ChatOllama:
    return _build_client(temperature=ROUTER_TEMPERATURE)


def get_handler_client() -> ChatOllama:
    return _build_client(temperature=HANDLER_TEMPERATURE)

