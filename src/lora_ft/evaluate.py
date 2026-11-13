"""Batched greedy generation and scoring for zero-shot, few-shot and fine-tuned settings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lora_ft.data import Example, load_split, read_labels
from lora_ft.fewshot import TfidfRetriever
from lora_ft.metrics import encode, summarize
from lora_ft.modeling import ResourceMonitor, load_for_inference, load_tokenizer, pick_device
from lora_ft.prompts import build_messages, parse_prediction


def render_prompts(
    tokenizer,
    examples: list[Example],
    labels: list[str],
    style: str,
    shots: int = 0,
    pool: list[Example] | None = None,
) -> list[str]:
    retriever = TfidfRetriever(pool) if shots and pool else None
    prompts = []
    for ex in examples:
        demos = list(reversed(retriever.top_k(ex.text, shots))) if retriever else None
        messages = build_messages(ex.text, labels, style, demos)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


@torch.inference_mode()
def generate(
    model, tokenizer, prompts: list[str], batch_size: int = 16, max_new_tokens: int = 16
) -> list[str]:
    """Greedy decoding, batched by prompt length (left padding) to minimise wasted compute."""
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True), strict=True):
            outputs[i] = text
    return outputs


def run_eval(
    name: str,
    model_name: str,
    data_dir: Path,
    out_dir: Path,
    split: str = "test",
    adapter: Path | None = None,
    style: str = "full",
    shots: int = 0,
    limit: int | None = None,
    batch_size: int = 16,
    device: str = "auto",
    seed: int = 42,
) -> dict:
    device = pick_device(device)
    labels = read_labels(data_dir)
    examples = load_split(data_dir, split, limit, seed)
    tok = load_tokenizer(model_name)
    pool = load_split(data_dir, "train") if shots else None
    prompts = render_prompts(tok, examples, labels, style, shots, pool)
    prompt_tokens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    model = load_for_inference(model_name, adapter, device)
    with ResourceMonitor(device) as mon:
        raw = generate(model, tok, prompts, batch_size=batch_size)

    preds = [parse_prediction(r, labels) for r in raw]
    y_true = encode([ex.label for ex in examples], labels)
    y_pred = encode(preds, labels)
    metrics = summarize(y_true, y_pred, len(labels), seed=seed)
    result = {
        "name": name,
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "split": split,
        "prompt_style": style,
        "shots": shots,
        "device": device,
        "mean_prompt_tokens": float(np.mean(prompt_tokens)),
        "examples_per_second": round(len(examples) / mon.seconds, 2) if mon.seconds else None,
        "resources": mon.as_dict(),
        **metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    with (out_dir / f"{name}.predictions.jsonl").open("w", encoding="utf-8") as f:
        for ex, r, p in zip(examples, raw, preds, strict=True):
            f.write(json.dumps({"text": ex.text, "label": ex.label, "output": r, "pred": p}) + "\n")
    return result
