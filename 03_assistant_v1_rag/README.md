# 03 Assistant v1 RAG

Local-first production-style RAG assistant.

## Features

- Config-driven settings.
- Separate ingestion and retrieval layers.
- Grounded prompt with safe fallback.
- CLI entry point.

## Run

1. `pip install -r requirements.txt`
2. Put source files in `docs/`
3. Ingest:
   - `python -m ingestion.ingest run --docs-path docs`
4. Ask:
   - `python main.py ask --question "Your question"`
5. Help:
   - `python main.py --help`

## Notes

- CLI built with `typer`.
- Output formatted with simple `rich` panels.

