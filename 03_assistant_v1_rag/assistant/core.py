from typing import Literal

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from llm.factory import get_llm
from prompts.templates import PROMPT
from retrieval.retriever import get_retriever


ChainMode = Literal["constructor", "custom"]


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_constructor_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def build_custom_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "input": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def answer(question: str, mode: ChainMode = "constructor") -> str:
    if mode == "constructor":
        result = build_constructor_rag_chain().invoke({"input": question})
        return result.get("answer", "I do not know.")
    if mode == "custom":
        return build_custom_rag_chain().invoke(question)
    raise ValueError(f"Unsupported chain mode: {mode}")
