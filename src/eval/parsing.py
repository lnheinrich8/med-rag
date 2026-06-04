"""Extract a JSON object from a local LLM's free-form reply.

Small models don't reliably honor "respond with only JSON" — they wrap it in
prose, ```json fences, or trailing commentary. Rather than fight that with ever-
longer prompts, we pull the first balanced ``{...}`` object out of the text and
parse it. Returns ``None`` when nothing parseable is found so callers can skip
the item instead of crashing a whole eval run.
"""

from __future__ import annotations

import json


def extract_json(text: str) -> dict | None:
    """Return the first JSON object embedded in ``text``, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None
