import os

from dotenv import load_dotenv


def init_tracing() -> None:
    """Load .env and print LangSmith tracing status."""
    load_dotenv()

    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    project = os.getenv("LANGSMITH_PROJECT", "(default)")

    if tracing_enabled:
        print(f"[LangSmith] Tracing ON  | project={project}")
    else:
        print("[LangSmith] Tracing OFF | set LANGSMITH_TRACING=true to enable")
