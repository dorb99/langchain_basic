"""
Query expansion / planning before vector search.

Produces a search string tuned for embeddings and an optional Chroma metadata
filter (e.g. single-file scope). This runs *before* retrieval — separate from
`rerank.py`, which reorders chunks *after* search.
"""

import os

from pydantic import BaseModel, Field

from llm.factory import get_llm
from prompts.templates import QUERY_PLAN_PROMPT


class RetrievalQueryPlan(BaseModel):
    """Structured output from the query-plan LLM."""

    search_query: str = Field(description="Short query tuned for similarity search.")
    metadata_filter: dict[str, str] | None = Field(
        default=None,
        description='Optional Chroma filter: only {"source": "<file basename>"} when the user names one file.',
    )


def chroma_source_filter(metadata_filter: dict | None) -> dict[str, str] | None:
    """Normalize optional `source` filter to a basename for Chroma (matches ingest)."""
    if not metadata_filter:
        return None
    raw = metadata_filter.get("source")
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or ".." in s or s.startswith(("/", "\\")):
        return None
    # Reject prompt-placeholder echoes like "<basename>" or "<filename>".
    if "<" in s or ">" in s:
        return None
    base = os.path.basename(s)
    return {"source": base} if base else None


def build_query_plan_chain():
    """Prompt → LLM (native structured output → `RetrievalQueryPlan`)."""
    structured_llm = get_llm(temperature=0.0).with_structured_output(RetrievalQueryPlan)
    return (QUERY_PLAN_PROMPT | structured_llm).with_config(
        {"run_name": "RetrievalQueryPlan", "metadata": {"step": "query_plan"}}
    )


def plan_from_llm(question: str) -> RetrievalQueryPlan:
    """Orchestrator: run the plan chain, normalize metadata filter, safe fallbacks."""
    try:
        plan = build_query_plan_chain().invoke({"question": question})
    except Exception:
        return RetrievalQueryPlan(search_query=question.strip(), metadata_filter=None)

    if not isinstance(plan, RetrievalQueryPlan):
        return RetrievalQueryPlan(search_query=question.strip(), metadata_filter=None)

    sq = (plan.search_query or "").strip() or question.strip()
    filt = chroma_source_filter(plan.metadata_filter)
    return RetrievalQueryPlan(search_query=sq, metadata_filter=filt)
