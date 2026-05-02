import json
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
import typer
from rich.console import Console
from rich.table import Table

from assistant.router import select_route
from assistant.core import answer
from llm.factory import get_llm
from prompts.templates import EVAL_JUDGE_PROMPT


class JudgeScore(BaseModel):
    score: int


DATASET_PATH = Path(__file__).parent / "dataset.json"

app = typer.Typer(help="Evaluate assistant quality")
console = Console()


def _judge_chain():
    parser = JsonOutputParser(pydantic_object=JudgeScore)
    prompt = EVAL_JUDGE_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return (prompt | get_llm(temperature=0.0) | parser).with_config(
        {"run_name": "EvalJudgeChain", "metadata": {"step": "evaluation"}}
    )


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate() -> list[dict]:
    dataset = load_dataset()
    judge = _judge_chain()
    results = []
    for item in dataset:
        question = item["question"]
        reference = item["reference"]
        expected_route = item.get("expected_route")

        actual_route = select_route(question)
        route_correct = actual_route == expected_route if expected_route else None

        candidate = answer(question)
        verdict = judge.invoke({
            "question": question,
            "reference": reference,
            "candidate": candidate,
        })
        score = verdict.get("score", 0)
        results.append({
            "question": question,
            "reference": reference,
            "candidate": candidate,
            "score": score,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
        })
    return results


@app.command()
def run() -> None:
    results = evaluate()
    table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Question", style="dim", max_width=40)
    table.add_column("Route", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Candidate", max_width=50)

    answer_total = 0
    route_total = 0
    route_count = 0
    for r in results:
        answer_total += r["score"]

        if r["route_correct"] is not None:
            route_count += 1
            route_total += int(r["route_correct"])
            route_icon = "Y" if r["route_correct"] else "N"
            route_display = f"{route_icon} ({r['actual_route']})"
        else:
            route_display = r["actual_route"]

        table.add_row(r["question"], route_display, str(r["score"]), r["candidate"][:50])

    console.print(table)

    ans_pct = 100 * answer_total / len(results) if results else 0
    console.print(f"\nAnswer quality: {answer_total}/{len(results)} correct ({ans_pct:.0f}%)")

    if route_count:
        rt_pct = 100 * route_total / route_count
        console.print(f"Route accuracy: {route_total}/{route_count} correct ({rt_pct:.0f}%)")


def main() -> None:
    from config.tracing import init_tracing
    init_tracing()
    app()


if __name__ == "__main__":
    main()
