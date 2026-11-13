"""Supervised fine-tuning features: chat-templated inputs with the loss restricted to the answer tokens."""

from __future__ import annotations

from typing import Any

import torch

from lora_ft.data import Example
from lora_ft.prompts import build_messages

IGNORE_INDEX = -100


def _ids(out: Any) -> list[int]:
    """apply_chat_template returns a list in older transformers and a BatchEncoding in newer ones."""
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    return list(out)


def prompt_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    return _ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def encode_example(
    tokenizer, example: Example, labels: list[str], style: str, max_length: int = 1024
) -> dict[str, list[int]]:
    """Tokenise prompt + answer in one pass through the chat template, then mask the prompt.

    The answer (label plus the end-of-turn token) is the only part that contributes to the loss. Encoding the
    whole conversation at once, instead of tokenising inputs and targets separately, keeps input_ids and labels
    aligned position by position.
    """
    messages = build_messages(example.text, labels, style)
    prompt = prompt_ids(tokenizer, messages)
    full = _ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": example.label}], tokenize=True
        )
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template does not render the prompt as a prefix of the full conversation")
    full = full[:max_length]
    target = [IGNORE_INDEX] * len(prompt) + full[len(prompt) :]
    target = target[: len(full)]
    if all(t == IGNORE_INDEX for t in target):
        raise ValueError(f"example truncated away its answer (max_length={max_length})")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": target}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer, examples: list[Example], labels: list[str], style: str, max_length: int = 1024
    ):
        self.features = [encode_example(tokenizer, ex, labels, style, max_length) for ex in examples]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.features[i]


class PadCollator:
    """Pads a batch; padded positions get attention 0 and label -100 so they never reach the loss.

    Training uses left padding so every answer ends at the last position, which lets `answer_loss` compute
    logits for the final few positions only. Widths are rounded up to a multiple of `pad_to_multiple_of` so the
    accelerator sees a handful of tensor shapes instead of one per batch.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 1, padding_side: str = "left"):
        self.pad_token_id = pad_token_id
        self.multiple = pad_to_multiple_of
        self.left = padding_side == "left"

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in batch)
        width = -(-width // self.multiple) * self.multiple

        def pad(key: str, value: int) -> torch.Tensor:
            rows = [
                [value] * (width - len(f[key])) + f[key]
                if self.left
                else f[key] + [value] * (width - len(f[key]))
                for f in batch
            ]
            return torch.tensor(rows, dtype=torch.long)

        mask = pad("attention_mask", 0)
        return {
            "input_ids": pad("input_ids", self.pad_token_id),
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
            "labels": pad("labels", IGNORE_INDEX),
        }


def answer_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy over answer tokens, computing vocabulary logits only where they are needed.

    With left padding the supervised tokens occupy the last positions of each row. Asking the model for the
    logits of the last k+1 positions (k = longest answer in the batch) instead of all of them shrinks the
    largest activation, batch x length x 151k vocabulary, by roughly length / k (about 10x here).
    Returns the loss and the number of supervised tokens.
    """
    labels = batch["labels"]
    supervised = labels != IGNORE_INDEX
    first = int(supervised.float().argmax(dim=1).min())  # earliest supervised position in the batch
    k = labels.shape[1] - first
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        logits_to_keep=k + 1,
    )
    # logits at position t predict token t + 1
    logits = out.logits[:, :-1, :].float()
    targets = labels[:, first:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )
    return loss, int(supervised.sum())
