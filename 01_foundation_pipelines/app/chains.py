from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama

from .prompts import (
    EXPLAIN_PROMPT,
    EXTRACT_PROMPT,
    STRUCTURED_PROMPT,
    SUMMARIZE_PROMPT,
)


class TaskPlan(BaseModel):
    title: str
    steps: list[str] = Field(default_factory=list)
    risk_level: str


def _model() -> ChatOllama:
    return ChatOllama(model="llama3.2:3b", temperature=0.2)


def explain_chain():
    return EXPLAIN_PROMPT | _model() | StrOutputParser()


def summarize_chain():
    return SUMMARIZE_PROMPT | _model() | StrOutputParser()


def extract_chain():
    return EXTRACT_PROMPT | _model() | StrOutputParser()


def structured_chain():
    parser = JsonOutputParser(pydantic_object=TaskPlan)
    prompt = STRUCTURED_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | _model() | parser
