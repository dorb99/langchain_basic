from typing import Literal

from pydantic import BaseModel

from llm.factory import get_router_llm
from prompts.templates import ROUTER_PROMPT


class RouteDecision(BaseModel):
    route: Literal["direct_answer", "retrieval", "tool_use"]


def _router_chain():
    structured_llm = get_router_llm().with_structured_output(RouteDecision)
    return (ROUTER_PROMPT | structured_llm).with_config(
        {"run_name": "RouterChain", "metadata": {"step": "routing"}}
    )


def select_route(question: str) -> str:
    try:
        result = _router_chain().invoke({"input": question})
    except Exception:
        return "direct_answer"

    route = result.route if isinstance(result, RouteDecision) else None
    if route in {"direct_answer", "retrieval", "tool_use"}:
        return route
    return "direct_answer"
