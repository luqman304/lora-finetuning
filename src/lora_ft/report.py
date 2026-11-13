"""Collect evaluation results into a Markdown table with paired significance tests."""

from __future__ import annotations

import json
from pathlib import Path

from lora_ft.metrics import encode, paired_bootstrap_diff


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(ci: list[float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def _load_preds(path: Path) -> tuple[list[str], list[str | None]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["label"] for r in rows], [r["pred"] for r in rows]


def build_report(results_dir: Path, labels: list[str], names: list[str], baseline: str | None = None) -> str:
    results = [json.loads((results_dir / f"{n}.json").read_text()) for n in names]
    lines = [
        "| System | Accuracy % (95% CI) | Macro-F1 % (95% CI) | Invalid % | Prompt tokens | Examples/s | Peak mem GB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {_pct(r['accuracy'])} {_ci(r['accuracy_ci'])} | {_pct(r['macro_f1'])} "
            f"{_ci(r['macro_f1_ci'])} | {_pct(r['invalid_rate'])} | {r['mean_prompt_tokens']:.0f} | "
            f"{r['examples_per_second']} | {r['resources']['peak_accelerator_gb']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"n = {results[0]['n']} examples ({results[0]['split']} split); CIs: 1,000 bootstrap resamples."
    )

    if baseline and baseline in names:
        gold, base_pred = _load_preds(results_dir / f"{baseline}.predictions.jsonl")
        y = encode(gold, labels)
        b = encode(base_pred, labels)
        lines += ["", f"Paired bootstrap, accuracy difference vs `{baseline}`:", ""]
        lines += ["| System | Δ accuracy (pp) | 95% CI |", "|---|---|---|"]
        for n in names:
            if n == baseline:
                continue
            g, p = _load_preds(results_dir / f"{n}.predictions.jsonl")
            if g != gold:
                continue
            d = paired_bootstrap_diff(y, encode(p, labels), b)
            lines.append(
                f"| {n} | {100 * d['accuracy_diff']:+.1f} | [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] |"
            )
    return "\n".join(lines) + "\n"
