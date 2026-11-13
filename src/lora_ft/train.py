"""LoRA / QLoRA supervised fine-tuning."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lora_ft.data import load_split, read_labels
from lora_ft.modeling import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_MODULES,
    ResourceMonitor,
    add_lora,
    count_parameters,
    load_base_model,
    load_tokenizer,
    pick_device,
)
from lora_ft.sft import PadCollator, SFTDataset, answer_loss


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    data_dir: str = "data/processed"
    out_dir: str = "outputs/lora"
    prompt_style: str = "short"
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    qlora: bool = False
    train_limit: int | None = None
    val_limit: int | None = 300
    log_every: int = 20
    seed: int = 42
    device: str = "auto"


def train(cfg: TrainConfig, log=print) -> dict:
    """Plain PyTorch loop: AdamW on the LoRA parameters, linear warmup + cosine decay, gradient accumulation.

    A hand-written loop rather than `transformers.Trainer` so the loss can be computed on the answer positions
    only (see `answer_loss`). On Apple-silicon MPS the Trainer version of this run was several times slower per
    step and grew the allocator cache past 12 GB, mostly from full-vocabulary logits over every position.
    """
    from transformers import get_cosine_schedule_with_warmup, set_seed

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    data_dir, out_dir = Path(cfg.data_dir), Path(cfg.out_dir)
    labels = read_labels(data_dir)
    tok = load_tokenizer(cfg.model)

    train_ds = SFTDataset(
        tok,
        load_split(data_dir, "train", cfg.train_limit, cfg.seed),
        labels,
        cfg.prompt_style,
        cfg.max_length,
    )
    val_ds = SFTDataset(
        tok, load_split(data_dir, "val", cfg.val_limit, cfg.seed), labels, cfg.prompt_style, cfg.max_length
    )
    collate = PadCollator(tok.pad_token_id, pad_to_multiple_of=32)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, generator=gen
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)

    model = load_base_model(cfg.model, device, quantize_4bit=cfg.qlora)
    model = add_lora(
        model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules, quantized=cfg.qlora
    )
    params = count_parameters(model)

    steps_per_epoch = math.ceil(len(train_dl) / cfg.grad_accum)
    total_steps = max(1, math.ceil(steps_per_epoch * cfg.epochs))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(0.03 * total_steps), total_steps)

    history: list[dict] = []
    step, running, n_running = 0, 0.0, 0
    with ResourceMonitor(device) as mon:
        epoch = 0
        while step < total_steps:
            model.train()
            for i, batch in enumerate(train_dl):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = answer_loss(model, batch)
                (loss / cfg.grad_accum).backward()
                running += loss.item()
                n_running += 1
                if (i + 1) % cfg.grad_accum and i + 1 != len(train_dl):
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0 or step == total_steps:
                    entry = {
                        "step": step,
                        "epoch": round(step / steps_per_epoch, 3),
                        "loss": running / n_running,
                    }
                    entry["lr"] = scheduler.get_last_lr()[0]
                    history.append(entry)
                    log(json.dumps(entry))
                    running, n_running = 0.0, 0
                if step >= total_steps:
                    break
            epoch += 1
            val_loss = evaluate_loss(model, val_dl, device)
            history.append({"step": step, "epoch": epoch, "val_loss": val_loss})
            log(json.dumps(history[-1]))
            if device == "mps":
                torch.mps.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    lengths = [len(f["input_ids"]) for f in train_ds.features]
    train_losses = [h["loss"] for h in history if "loss" in h]
    summary = {
        "config": asdict(cfg),
        "device": device,
        "hardware": _hardware(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "optimizer_steps": step,
        "mean_sequence_tokens": round(sum(lengths) / len(lengths), 1),
        "parameters": params,
        "trainable_fraction": params["trainable"] / params["total"],
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": next((h["val_loss"] for h in reversed(history) if "val_loss" in h), None),
        "log_history": history,
        "resources": mon.as_dict(),
        "adapter_dir": str(adapter_dir),
        "versions": _versions(),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str) -> float:
    """Token-weighted mean cross-entropy over the answer tokens."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, n = answer_loss(model, batch)
        total += loss.item() * n
        tokens += n
    model.train()
    return total / max(tokens, 1)


def _hardware(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return f"{platform.machine()} {platform.platform()}"


def _versions() -> dict[str, str]:
    import peft
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
