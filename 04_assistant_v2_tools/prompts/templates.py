from langchain_core.prompts import ChatPromptTemplate


DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the question concisely and accurately."),
        ("human", "{question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "Answer using ONLY the context below.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

QUERY_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan retrieval for a vector database (Chroma).\n"
            "Produce a concise search_query string that captures what to look up, "
            "including important entities and synonyms if helpful.\n"
            "Set metadata_filter to null by default.\n"
            "Only set metadata_filter when the user explicitly names a real file "
            "(filename including an extension, e.g. sample.txt or notes.md). In that case, set "
            'metadata_filter to {{"source": "<the actual filename the user wrote>"}}. '
            "Use the real filename text, never the literal placeholder text.\n"
            "Never use angle brackets (< or >) in the filter value. "
            "The value must be a file basename, not a path.",
        ),
        ("human", "{question}"),
    ]
)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You reorder retrieved passages by relevance to the user's question.\n"
            "Return an order field: a list of every document index from 0 to n-1 exactly once, "
            "best match first. No extra fields.",
        ),
        (
            "human",
            "Question:\n{question}\n\nCandidates (n={n}):\n{candidates}",
        ),
    ]
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's request into exactly one route.\n\n"
            "Routes:\n"
            "- direct_answer: general knowledge questions, opinions, explanations\n"
            "- retrieval: questions about documents, policies, or that need lookup from a knowledge base\n"
            "- tool_use: math calculations, percentage computations, or data lookups requiring a tool",
        ),
        ("human", "{input}"),
    ]
)

EVAL_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluation judge. Given a question, a reference answer, and a candidate answer, "
            "decide if the candidate answer is correct.\n"
            "Respond with JSON: {{\"score\": 1}} if correct, {{\"score\": 0}} if incorrect.\n"
            "Be lenient: the candidate does not need to match word-for-word, just be factually consistent.\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Question: {question}\n"
            "Reference: {reference}\n"
            "Candidate: {candidate}",
        ),
    ]
)
