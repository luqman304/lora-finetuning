"""Chat prompts for the three settings (zero-shot, few-shot, fine-tuned) and parsing of model outputs."""

from __future__ import annotations

import re

from lora_ft.data import Example

INSTRUCTION = (
    "You are an intent classifier for a digital bank's customer support. "
    "Classify the customer's message into exactly one intent. "
    "Answer with the intent label only, with no other text."
)

PROMPT_STYLES = ("full", "short")


def system_prompt(labels: list[str], style: str) -> str:
    """`full` lists every allowed label (needed without fine-tuning); `short` relies on the label set being learned."""
    if style == "short":
        return INSTRUCTION
    if style == "full":
        return INSTRUCTION + "\n\nAllowed intents:\n" + "\n".join(labels)
    raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")


def build_messages(
    text: str, labels: list[str], style: str, shots: list[Example] | None = None
) -> list[dict[str, str]]:
    """Messages up to (not including) the assistant answer. Few-shot examples become prior chat turns."""
    messages = [{"role": "system", "content": system_prompt(labels, style)}]
    for shot in shots or []:
        messages.append({"role": "user", "content": shot.text})
        messages.append({"role": "assistant", "content": shot.label})
    messages.append({"role": "user", "content": text})
    return messages


_STRIP = re.compile(r"^[\s\"'`*.:]+|[\s\"'`*.:]+$")


def parse_prediction(output: str, labels: list[str]) -> str | None:
    """Map a generated string to a label, or None if it names no valid label.

    Exact match after light normalisation (case, surrounding quotes/punctuation, spaces vs underscores). If that
    fails, the longest label that appears verbatim in the output is used, so a base model that answers
    "Intent: card_arrival" is not penalised for formatting. Fine-tuned models almost always hit the exact path.
    """
    lines = output.strip().splitlines()
    first = lines[0] if lines else ""
    norm = _STRIP.sub("", first).lower().replace(" ", "_").replace("-", "_")
    label_set = set(labels)
    if norm in label_set:
        return norm
    lowered = output.lower()
    found = [lab for lab in labels if lab in lowered]
    return max(found, key=len) if found else None
