import ast
import operator

from langchain_core.tools import tool


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node):
    if isinstance(node, ast.BinOp):
        op = SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported operator")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Unsupported expression")


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body))


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, / and parentheses.
    Examples: '2 + 3', '15/100 * 4200', '(10 + 5) * 3'."""
    try:
        result = _safe_eval(expression)
    except Exception as exc:
        return f"Calculator error: {exc}"
    return f"Result: {result}"
