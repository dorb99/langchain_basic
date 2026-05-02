import json
from pathlib import Path

from langchain_core.tools import tool


DATA_PATH = Path(__file__).resolve().parents[1] / "eval" / "knowledge.json"


@tool
def data_lookup(topic: str) -> str:
    """Look up a topic in the local knowledge base.
    Provide a keyword like 'rag', 'langsmith', or 'chroma'."""
    if not DATA_PATH.exists():
        return "No data source configured."
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    for key, value in data.items():
        if key.lower() in topic.lower():
            return f"{key}: {value}"
    return "No matching data entry found."
