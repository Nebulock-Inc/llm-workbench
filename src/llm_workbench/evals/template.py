"""Prompt-template helpers: find {placeholders} and fill them."""

from __future__ import annotations

import json
from typing import Any


def find_variables(template: str) -> tuple[str, ...]:
    """Return the {placeholders} a template references, in first-seen order.

    Ignores empty braces and any brace content containing a newline, so prose
    with stray braces doesn't get mistaken for a variable.
    """
    seen: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in template:
        if ch == "{":
            depth, current = depth + 1, []
        elif ch == "}" and depth:
            depth -= 1
            name = "".join(current).strip()
            if name and name not in seen and "\n" not in name:
                seen.append(name)
        elif depth:
            current.append(ch)
    return tuple(seen)


def fill_template(template: str, variables: dict[str, Any]) -> str:
    """Substitute {var} placeholders; dict/list values are JSON-encoded."""
    filled = template
    for key, value in variables.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        filled = filled.replace(f"{{{key}}}", str(value))
    return filled
