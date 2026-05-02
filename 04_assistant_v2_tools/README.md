# 04 Assistant v2 Tools + Quality

Capstone project building on projects 2 and 3 with proper LangChain framework usage throughout.

## What's new over v1 (project 03)

- **LLM-based routing** via `JsonOutputParser` + Pydantic schema (pattern from project 02).
- **LangChain `@tool` decorator** for calculator and data lookup.
- **Tool calling** with `bind_tools` and full tool-call-result loop.
- **LCEL chain composition** for retrieval (retriever + rerank + RAG prompt as one chain).
- **Centralized prompts** in `prompts/templates.py`.
- **LLM factory** in `llm/factory.py` with config-driven provider/temperature.
- **LLM-judge evaluation pipeline** in `eval/evaluate.py`.
- **Expanded tests** for tool invocation.
- **LangSmith tracing** with labelled `run_name` and `metadata` on every chain step.

## Architecture

```
question
  -> Router chain (LLM + JsonOutputParser -> RouteDecision)
  -> direct_answer : LCEL chain (prompt | llm | parser)
  -> retrieval     : LCEL chain (retriever + rerank | RAG prompt | llm | parser)
  -> tool_use      : bind_tools + tool execution + final LLM answer
```

## Run

Prerequisites:
- Python 3.11+
- Ollama running locally with the model configured in `config/settings.py`

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your LangSmith API key (optional):
   - PowerShell: `Copy-Item .env.example .env`
   - Bash: `cp .env.example .env`
3. Optional ingestion:
   - `python -m retrieval.ingest run --docs-path docs`
4. Ask:
   - `python main.py --question "What is 15% of 4200?"`
   - `python main.py --question "What does RAG mean in these docs?"`
   - `python main.py --question "Explain embeddings"`
5. Evaluate:
   - `python -m eval.evaluate run`
6. Test:
   - `pytest tests/ -v`
7. Help:
   - `python main.py --help`

## LangSmith Tracing

Every chain step is tagged with a `run_name` and `metadata` so that traces in LangSmith are clearly labelled:

| Step              | `run_name`           | `metadata.step`  |
|-------------------|----------------------|-------------------|
| Router            | `RouterChain`        | `routing`         |
| Direct answer     | `DirectAnswerChain`  | `direct_answer`   |
| Tool selection    | `ToolSelection`      | `tool_use`        |
| Tool final answer | `ToolFinalAnswer`    | `tool_use`        |
| Retrieval RAG     | `RetrievalChain`     | `retrieval`       |
| Eval judge        | `EvalJudgeChain`     | `evaluation`      |

Top-level spans `AssistantAnswer` and `ToolAnswerPipeline` group the nested calls.

To enable, set the following in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your-api-key>
LANGSMITH_PROJECT="assistant-v2-tools"
```

## Configuration

All settings live in `config/settings.py`:
- `model_name` — Ollama model (default `qwen2.5:3b`).
- `router_temperature` / `handler_temperature` — separate temperatures for routing vs generation.
- `retriever_top_k` / `rerank_top_n` — retrieval and reranking depth.

## Notes

- CLI built with `typer`, output formatted with `rich`.
- Defaults to local Ollama.
- Routing is fail-safe: malformed router output falls back to direct answering.

## Security

- Never commit real secrets in `.env`.
- Keep `.env` local only and rotate any key that was accidentally exposed.

## Troubleshooting

- If you get model errors, verify Ollama is running and the configured model exists locally.
- If retrieval answers are weak, run ingestion again: `python -m retrieval.ingest run --docs-path docs`.
- If tracing is not visible, confirm `.env` has `LANGSMITH_TRACING=true`.
