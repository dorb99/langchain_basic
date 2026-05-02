import typer
from rich.console import Console
from rich.panel import Panel

from assistant.core import answer
from config.tracing import init_tracing

app = typer.Typer(help="Assistant v2 with tools")
console = Console()

@app.command()
def ask(question: str = typer.Option(..., help="Question to ask")) -> None:
    result = answer(question)
    console.print(Panel(result, title="Assistant v2", border_style="green"))


def main() -> None:
    init_tracing()
    app()


if __name__ == "__main__":
    main()
