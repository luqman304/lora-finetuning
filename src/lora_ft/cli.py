"""Command-line interface: prepare -> train -> eval -> report, plus predict / merge / push."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lora_ft.modeling import DEFAULT_MODEL

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


@app.command()
def prepare(
    out_dir: Path = typer.Option(DATA_DIR, help="Where to write train/val/test JSONL and labels.json."),
    val_fraction: float = typer.Option(
        0.1, help="Share of the official train split held out for validation."
    ),
    seed: int = 42,
) -> None:
    """Download Banking77 and write stratified train / val / test splits."""
    from lora_ft.data import prepare as _prepare

    typer.echo(json.dumps(_prepare(out_dir, val_fraction, seed)))


@app.command()
def train(
    model: str = DEFAULT_MODEL,
    data_dir: Path = DATA_DIR,
    out_dir: Path = Path("outputs/lora"),
    prompt_style: str = typer.Option(
        "short", help="'short' (instruction only) or 'full' (lists all labels)."
    ),
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    grad_accum: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    qlora: bool = typer.Option(False, help="Load the base model in 4-bit NF4 (CUDA + bitsandbytes only)."),
    train_limit: int | None = typer.Option(None, help="Train on a seeded random subset of this size."),
    val_limit: int | None = 300,
    seed: int = 42,
    device: str = "auto",
) -> None:
    """Fine-tune a LoRA (or QLoRA) adapter; loss is computed on the answer tokens only."""
    from lora_ft.train import TrainConfig
    from lora_ft.train import train as _train

    cfg = TrainConfig(
        model=model,
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        prompt_style=prompt_style,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        qlora=qlora,
        train_limit=train_limit,
        val_limit=val_limit,
        seed=seed,
        device=device,
    )
    s = _train(cfg)
    typer.echo(
        json.dumps(
            {
                k: s[k]
                for k in ("parameters", "trainable_fraction", "final_val_loss", "resources", "adapter_dir")
            },
            indent=2,
        )
    )


@app.command("eval")
def evaluate(
    name: str = typer.Option(..., help="Run name; results go to <results-dir>/<name>.json."),
    model: str = DEFAULT_MODEL,
    adapter: Path | None = typer.Option(
        None, help="LoRA adapter directory; omit to evaluate the base model."
    ),
    prompt_style: str = "full",
    shots: int = typer.Option(0, help="Number of retrieved in-context examples (few-shot)."),
    split: str = "test",
    limit: int | None = None,
    batch_size: int = 16,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    device: str = "auto",
    seed: int = 42,
) -> None:
    """Evaluate the base model (zero/few-shot) or a fine-tuned adapter on a prepared split."""
    from lora_ft.evaluate import run_eval

    r = run_eval(
        name,
        model,
        data_dir,
        results_dir,
        split,
        adapter,
        prompt_style,
        shots,
        limit,
        batch_size,
        device,
        seed,
    )
    typer.echo(
        json.dumps(
            {k: r[k] for k in ("name", "n", "accuracy", "accuracy_ci", "macro_f1", "invalid_rate")}, indent=2
        )
    )


@app.command()
def report(
    names: str = typer.Option(..., help="Comma-separated run names, in table order."),
    baseline: str | None = typer.Option(None, help="Run to compare the others against (paired bootstrap)."),
    results_dir: Path = RESULTS_DIR,
    data_dir: Path = DATA_DIR,
    out: Path | None = typer.Option(None, help="Also write the table to this file."),
) -> None:
    """Render evaluation results as a Markdown table."""
    from lora_ft.data import read_labels
    from lora_ft.report import build_report

    md = build_report(results_dir, read_labels(data_dir), [n.strip() for n in names.split(",")], baseline)
    if out:
        out.write_text(md)
    typer.echo(md)


@app.command()
def predict(
    texts: list[str] = typer.Argument(..., help="One or more customer messages."),
    model: str = DEFAULT_MODEL,
    adapter: Path | None = typer.Option(None, help="Local adapter directory or Hub repo id."),
    prompt_style: str | None = typer.Option(
        None, help="Defaults to 'short' with an adapter, 'full' without."
    ),
    labels_file: Path = DATA_DIR / "labels.json",
    device: str = "auto",
) -> None:
    """Classify messages with the base model or a fine-tuned adapter."""
    from lora_ft.evaluate import generate
    from lora_ft.modeling import load_for_inference, load_tokenizer, pick_device
    from lora_ft.prompts import build_messages, parse_prediction

    labels = _labels(labels_file)
    style = prompt_style or ("short" if adapter else "full")
    tok = load_tokenizer(model)
    prompts = [
        tok.apply_chat_template(build_messages(t, labels, style), tokenize=False, add_generation_prompt=True)
        for t in texts
    ]
    m = load_for_inference(model, adapter, pick_device(device))
    for text, raw in zip(texts, generate(m, tok, prompts), strict=True):
        typer.echo(json.dumps({"text": text, "intent": parse_prediction(raw, labels), "raw": raw}))


@app.command()
def merge(
    adapter: Path = typer.Option(...),
    out_dir: Path = Path("outputs/merged"),
    model: str = DEFAULT_MODEL,
) -> None:
    """Merge the adapter into the base weights and save a standalone model."""
    from lora_ft.modeling import merge_adapter

    typer.echo(str(merge_adapter(model, adapter, out_dir)))


@app.command()
def push(
    repo_id: str = typer.Option(..., help="Target Hub repo, e.g. user/qwen2.5-0.5b-banking77-lora."),
    adapter: Path = Path("outputs/lora/adapter"),
    model: str = DEFAULT_MODEL,
    eval_json: Path | None = typer.Option(RESULTS_DIR / "lora.json", help="Metrics for the model card."),
    private: bool = True,
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Upload the adapter (with a generated model card) to the Hugging Face Hub."""
    from lora_ft.hub import push_adapter

    if not yes:
        typer.confirm(f"Upload {adapter} to https://huggingface.co/{repo_id}?", abort=True)
    typer.echo(push_adapter(adapter, repo_id, model, eval_json, private))


def _labels(path: Path) -> list[str]:
    if path.exists():
        return json.loads(path.read_text())
    bundled = Path(__file__).with_name("labels.json")
    return json.loads(bundled.read_text())


if __name__ == "__main__":
    app()
