from llm.factory import get_llm
from prompts.templates import PROMPT
from retrieval.retriever import get_retriever


def answer(question: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(question)
    if not docs:
        return "I could not find relevant information in the documents."

    context = "\n\n".join(d.page_content for d in docs)
    prompt_value = PROMPT.format_messages(context=context, question=question)
    response = get_llm().invoke(prompt_value)
    return response.content if hasattr(response, "content") else str(response)
