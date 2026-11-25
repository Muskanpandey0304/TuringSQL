"""The one prompt template, used by training, evaluation and serving alike.

Everything that builds a prompt imports from here. If the wording changes, every model trained on the
old wording is evaluated with a different prompt than it was trained on, so record any change in
docs/DECISIONS.md and bump PROMPT_VERSION.
"""
from __future__ import annotations

import re

PROMPT_VERSION = "v1"

SYSTEM = "You write one SQLite query that answers the question. Output only SQL."


def user_content(schema: str, question: str, definitions: list[str] | None = None) -> str:
    """The user turn: schema, optional business definitions, then the question."""
    parts = ["### Schema", schema.strip()]
    if definitions:
        parts += ["### Definitions", *(d.strip() for d in definitions)]
    parts += ["### Question", question.strip()]
    return "\n".join(parts)


def build_messages(schema: str, question: str, definitions: list[str] | None = None,
                   sql: str | None = None) -> list[dict[str, str]]:
    """Chat messages for a model's chat template.

    Pass `sql` to get a training example (with the assistant turn); leave it out for inference.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content(schema, question, definitions)},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply.

    Handles a ```sql fenced block (an unclosed fence too), strips a trailing semicolon and keeps only
    the first statement. Returns "" when nothing is left.
    """
    text = text.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    elif text.lower().startswith(("```sql", "```sqlite", "```")):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    text = text.strip()
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()
