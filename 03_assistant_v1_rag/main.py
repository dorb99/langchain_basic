import typer
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

from assistant.core import answer

load_dotenv()

app = typer.Typer(help="Assistant v1 RAG")
console = Console()

@app.command()
def ask(question: str = typer.Argument(..., help="Question to ask")) -> None:
    result = answer(question)
    console.print(Panel(result, title="Assistant v1", border_style="green"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
