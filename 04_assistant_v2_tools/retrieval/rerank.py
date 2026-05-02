"""LLM reranking of retrieved chunks """

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from llm.factory import get_llm
from prompts.templates import RERANK_PROMPT


class _RerankOrder(BaseModel):
    order: list[int] = Field(description="0..n-1 permutation, most relevant document index first")


def build_rerank_chain():
    """Prompt → LLM (native structured output → `_RerankOrder`)."""
    structured_llm = get_llm(temperature=0.0).with_structured_output(_RerankOrder)
    return (RERANK_PROMPT | structured_llm).with_config(
        {"run_name": "RetrievalRerank", "metadata": {"step": "rerank"}}
    )


def _apply_order(docs: list[Document], order: list[int], top_n: int) -> list[Document]:
    n = len(docs)
    seen: set[int] = set()
    out: list[Document] = []
    for idx in order:
        if isinstance(idx, int) and 0 <= idx < n and idx not in seen:
            seen.add(idx)
            out.append(docs[idx])
    for i, d in enumerate(docs):
        if i not in seen:
            out.append(d)
    return out[:top_n]


def rerank_documents(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """
    Reorder `docs` by LLM relevance to `query`, keep `top_n`.
    On parse errors or bad output, keep the vector store order (first `top_n` chunks).
    """
    if not docs:
        return []

    lines: list[str] = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        meta_s = ", ".join(f"{k}={v}" for k, v in meta.items())
        body = (d.page_content or "")[:500].replace("\n", " ")
        lines.append(f"### doc {i}\nmetadata: {meta_s}\n{body}")

    chain = build_rerank_chain()
    try:
        parsed = chain.invoke(
            {
                "question": query,
                "n": len(docs),
                "candidates": "\n\n".join(lines),
            }
        )
        order = parsed.order if isinstance(parsed, _RerankOrder) else None
        if not isinstance(order, list):
            return docs[:top_n]
        return _apply_order(docs, order, top_n)
    except Exception:
        return docs[:top_n]
