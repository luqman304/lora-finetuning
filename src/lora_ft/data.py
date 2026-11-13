"""Dataset preparation: Banking77 -> stratified train / val / test JSONL files."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATASET_ID = "PolyAI/banking77"


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def canonical_label(name: str) -> str:
    """Normalise the two irregular Banking77 label names ('Refund_not_showing_up', 'reverted_card_payment?')."""
    return name.strip().lower().rstrip("?")


def stratified_split(
    examples: list[Example], val_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split so every label keeps the same share in both parts (at least one validation example per label)."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"text": ex.text, "label": ex.label}, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]


def read_labels(data_dir: Path) -> list[str]:
    return json.loads((data_dir / "labels.json").read_text())


def load_split(data_dir: Path, split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    """Load a prepared split; `limit` takes a seeded random subset (stable across runs)."""
    examples = read_jsonl(data_dir / f"{split}.jsonl")
    if limit is not None and limit < len(examples):
        examples = random.Random(seed).sample(examples, limit)
    return examples


def prepare(out_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> dict[str, int]:
    """Download Banking77 from the Hugging Face Hub and write train/val/test JSONL plus the label list.

    The official test split is kept untouched; validation is carved out of the official train split.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    names = [canonical_label(n) for n in ds["train"].features["label"].names]

    def convert(split: str) -> list[Example]:
        return [Example(text=r["text"].strip(), label=names[r["label"]]) for r in ds[split]]

    train, val = stratified_split(convert("train"), val_fraction, seed)
    test = convert("test")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(names, indent=1))
    for split, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    return {"train": len(train), "val": len(val), "test": len(test), "labels": len(names)}
