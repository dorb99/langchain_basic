import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chains import explain_chain, extract_chain, structured_chain, summarize_chain

app = typer.Typer(help="Foundation LangChain pipelines")
console = Console()


def _show_text_result(title: str, result: str) -> None:
    console.print(Panel(result, title=title, border_style="cyan"))


def _show_structured_result(result) -> None:
    if isinstance(result, list):
        table = Table(title="Structured Output", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="bold")
        table.add_column("Value")
        for idx, item in enumerate(result):
            table.add_row(str(idx), json.dumps(item, ensure_ascii=True))
        console.print(table)
        console.print(Panel(json.dumps(result, indent=2), title="Raw JSON", border_style="green"))
        return

    if not isinstance(result, dict):
        console.print(Panel(str(result), title="Structured Output", border_style="yellow"))
        return

    table = Table(title="Structured Output", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    for key, value in result.items():
        table.add_row(str(key), json.dumps(value, ensure_ascii=True))

    console.print(table)
    console.print(Panel(json.dumps(result, indent=2), title="Raw JSON", border_style="green"))


@app.command()
def explain(
    topic: str = typer.Option(..., help="Topic to explain"),
    level: str = typer.Option("beginner", help="Audience level"),
) -> None:
    result = explain_chain().invoke({"topic": topic, "level": level})
    _show_text_result("Explain", result)


@app.command()
def summarize(
    text: str = typer.Option(..., help="Input text to summarize"),
) -> None:
    result = summarize_chain().invoke({"text": text})
    _show_text_result("Summarize", result)


@app.command()
def extract(
    text: str = typer.Option(..., help="Input text to extract key idea from"),
) -> None:
    result = extract_chain().invoke({"text": text})
    _show_text_result("Extract", result)


@app.command()
def structured(
    task: str = typer.Option(..., help="Task to convert into structured plan"),
) -> None:
    result = structured_chain().invoke({"task": task})
    _show_structured_result(result)


def main() -> None:
    app()
