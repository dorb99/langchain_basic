# 01 Foundation Pipelines

Basic but professional LangChain pipeline project.

## What it shows

- Reusable prompt templates.
- Single-step and multi-step chains.
- Structured output with Pydantic.
- Optional LangSmith tracing.

## Run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run a chain:
   - `python main.py explain --topic "embeddings" --level "beginner"`
   - `python main.py summarize --text "Your long text here"`
   - `python main.py extract --text @"Your technical text here"@`
   - `python main.py structured --task "Create a release checklist"`
3. Show available commands:
   - `python main.py --help`

## Notes

- Defaults to local Ollama model `tinyllama`.
- CLI built with `typer` and output formatted with `rich`.
- Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=...` for trace visibility.

