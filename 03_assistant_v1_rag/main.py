import typer
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

from assistant.core import answer

load_dotenv()

app = typer.Typer(help="Assistant v1 RAG")
console = Console()

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    mode: str = typer.Option(
        "constructor",
        "--mode",
        help="Chain build style: constructor (LangChain helpers) or custom (your own runnable chain).",
    ),
) -> None:
    result = answer(question, mode=mode.lower())
    console.print(Panel(result, title="Assistant v1", border_style="green"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
