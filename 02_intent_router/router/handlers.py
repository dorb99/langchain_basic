from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .client import get_handler_client


def _handler_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "{input}"),
        ]
    )


def explain_chain():
    system_instruction = "Explain the following concept clearly and thoroughly."
    prompt = _handler_prompt(system_instruction).with_config(
        run_name="explain_prompt",
        metadata={"system_instruction": system_instruction},
    )
    return prompt | get_handler_client() | StrOutputParser()


def summarize_chain():
    system_instruction = "Summarize the following text in 3 concise bullet points."
    prompt = _handler_prompt(system_instruction).with_config(
        run_name="summarize_prompt",
        metadata={"system_instruction": system_instruction},
    )
    return prompt | get_handler_client() | StrOutputParser()


def extract_chain():
    system_instruction = "Extract the key entities and facts from the following text."
    prompt = _handler_prompt(system_instruction).with_config(
        run_name="extract_prompt",
        metadata={"system_instruction": system_instruction},
    )
    return prompt | get_handler_client() | StrOutputParser()


def rewrite_chain():
    system_instruction = "Rewrite the following text to be professional and concise."
    prompt = _handler_prompt(system_instruction).with_config(
        run_name="rewrite_prompt",
        metadata={"system_instruction": system_instruction},
    )
    return prompt | get_handler_client() | StrOutputParser()


def translate_chain():
    system_instruction = "Translate the following text to Hebrew, preserving its meaning."
    prompt = _handler_prompt(system_instruction).with_config(
        run_name="translate_prompt",
        metadata={"system_instruction": system_instruction},
    )
    return prompt | get_handler_client() | StrOutputParser()
