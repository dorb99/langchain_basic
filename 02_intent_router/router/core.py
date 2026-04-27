from langchain_core.prompts import ChatPromptTemplate

from .client import get_router_client
from .handlers import explain_chain, extract_chain, rewrite_chain, summarize_chain, translate_chain
from .schema import RouteDecision

ROUTER_SYSTEM_INSTRUCTION = (
    "Classify into exactly one route: explain, summarize, extract, rewrite, translate.\n"
    "Rules:\n"
    "- explain: concept teaching\n"
    "- summarize: compress provided text\n"
    "- extract: pull fields/facts\n"
    "- rewrite: change tone/style\n"
    "- translate: convert language\n"
    "Return only the selected route."
)


def _extract_route(route_result: object) -> str | None:
    if isinstance(route_result, RouteDecision):
        return route_result.route
    if isinstance(route_result, dict):
        route = route_result.get("route")
        if isinstance(route, str):
            return route
    return None


def _router_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_INSTRUCTION),
            ("human", "{input}"),
        ]
    ).with_config(
        run_name="router_prompt",
        metadata={"system_instruction": ROUTER_SYSTEM_INSTRUCTION},
    )
    model = get_router_client()
    return prompt | model.with_structured_output(RouteDecision)


def run_workflow(user_input: str) -> str:
    route_result = _router_chain().invoke({"input": user_input})
    route = _extract_route(route_result)
    if route is None:
        return f"Failed to parse route decision: {route_result}"
    handlers = {
        "explain": explain_chain(),
        "summarize": summarize_chain(),
        "extract": extract_chain(),
        "rewrite": rewrite_chain(),
        "translate": translate_chain(),
    }
    handler = handlers.get(route)
    if handler is None:
        return f"Unknown route: {route}"
    return handler.invoke({"input": user_input})
