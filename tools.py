import ast
import operator
import re
from pathlib import Path

DOCS_DIR = Path("docs")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos
}

def calculator(expression: str) -> str:
    """Безопасный калькулятор."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("слишком большая степень")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("допустимы только числа и операторы + - * / % **")

    try:
        expr = expression.strip().strip("`'\"").replace("^", "**")
        result = _eval(ast.parse(expr, mode="eval").body)
        return str(round(result, 6)) if isinstance(result, float) else str(result)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as e:
        return f"Ошибка: {e}"

def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

def text_search(query: str, top_k: int = 3) -> str:
    q = _tokenize(query)
    hits = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        paragraphs = content.split("\n\n")
        for i, para in enumerate(paragraphs):
            para_tokens = _tokenize(para)
            score = len(q.intersection(para_tokens))
            if score > 0:
                hits.append((score, path.name, i + 1, para.strip()))

    if not hits:
        return "Ничего не найдено. Попробуйте переформулировать запрос."

    hits.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, fname, idx, para in hits[:top_k]:
        short_para = para[:400] + "..." if len(para) > 400 else para
        results.append(f"[{fname}, абзац {idx}] {short_para}")

    return "\n\n".join(results)