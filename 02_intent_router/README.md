# 02 Intent Router

Structured request routing with focused handlers.

## Routes

- `explain`
- `summarize`
- `extract`
- `rewrite`
- `translate` (optional bonus)

## Run

1. `pip install -r requirements.txt`
2. `python main.py run --input "Summarize this text: ..."`
3. `python main.py --help`

## Configuration

- `ROUTER_OLLAMA_MODEL` overrides the default `tinyllama` model.
- `ROUTER_ROUTER_TEMPERATURE` controls classifier temperature.
- `ROUTER_HANDLER_TEMPERATURE` controls handler generation temperature.
- Standard LangSmith vars like `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGSMITH_TRACING_V2` can live in `.env`.

## Notes

- CLI built with `typer`.
- Output formatted with simple `rich` panels.
- Ollama client setup lives in `router/client.py`.

