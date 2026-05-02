from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langsmith import trace

from assistant.router import select_route
from llm.factory import get_llm
from prompts.templates import DIRECT_ANSWER_PROMPT
from retrieval.pipeline import retrieval_chain
from tools.calculator import calculator
from tools.data_lookup import data_lookup

ALL_TOOLS = [calculator, data_lookup]
MAX_TOOL_ROUNDS = 3


def _direct_chain():
    return (DIRECT_ANSWER_PROMPT | get_llm() | StrOutputParser()).with_config(
        {"run_name": "DirectAnswerChain", "metadata": {"step": "direct_answer"}}
    )


def _tool_answer(question: str) -> str:
    with trace("ToolAnswerPipeline", metadata={"step": "tool_use"}):
        llm_with_tools = get_llm().bind_tools(ALL_TOOLS).with_config(
            {"run_name": "ToolSelection"}
        )
        messages = [HumanMessage(content=question)]
        tool_map = {t.name: t for t in ALL_TOOLS}

        for _ in range(MAX_TOOL_ROUNDS):
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
            if not ai_msg.tool_calls:
                return ai_msg.content or "No tool was selected."

            messages.append(ai_msg)
            for call in ai_msg.tool_calls:
                tool_fn = tool_map.get(call["name"])
                if tool_fn is None:
                    result = f"Unknown tool: {call['name']}"
                else:
                    try:
                        result = tool_fn.invoke(call["args"])
                    except Exception as exc:
                        result = f"Tool execution failed: {exc}"
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )

        final = get_llm().with_config(
            {"run_name": "ToolFinalAnswer"}
        ).invoke(messages)
        return final.content if hasattr(final, "content") else str(final)


def answer(question: str) -> str:
    with trace("AssistantAnswer", metadata={"question": question}):
        route = select_route(question)
        if route == "tool_use":
            return _tool_answer(question)
        if route == "retrieval":
            chain = retrieval_chain()
            return chain.invoke({"question": question})
        return _direct_chain().invoke({"question": question})
