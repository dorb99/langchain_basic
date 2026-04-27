from langchain_core.prompts import ChatPromptTemplate

EXPLAIN_PROMPT = ChatPromptTemplate.from_template(
    "Explain {topic} for a {level} developer in under 120 words."
)

SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the text in 3 concise bullet points:\n\n{text}"
)

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "Read this text and extract the single key idea in one sentence:\n\n{text}"
)

STRUCTURED_PROMPT = ChatPromptTemplate.from_template(
    "Create a task plan for: {task}. "
    "Return JSON with fields title, steps (list of strings), and risk_level (low|medium|high).\n"
    "{format_instructions}"
)
