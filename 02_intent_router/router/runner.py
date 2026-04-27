from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .core import run_workflow

app = typer.Typer(help="Intent router workflow")
console = Console()

@app.command()
def run(
    input_text: Optional[str] = typer.Argument(None, help="User input text"),
) -> None:
    if not input_text:
        raise typer.BadParameter("Provide input either as positional text or via --input.")
    result = run_workflow(input_text)
    console.print(Panel(result, title="Router Result", border_style="cyan"))


def main() -> None:
    app()
