from langchain_core.prompts import ChatPromptTemplate


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant.\n"
    "Answer using ONLY the context below.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        ("human", "{question}"),
    ]
)
