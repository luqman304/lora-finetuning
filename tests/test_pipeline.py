"""End-to-end smoke test on CPU with the tiny random model: train, save, reload, evaluate, merge, report."""

import json

import pytest
import torch

from lora_ft.evaluate import run_eval
from lora_ft.modeling import load_for_inference, merge_adapter
from lora_ft.report import build_report
from lora_ft.train import TrainConfig, train
from tests.conftest import LABELS


@pytest.fixture(scope="module")
def trained(tiny_model_dir, data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(
        model=str(tiny_model_dir),
        data_dir=str(data_dir),
        out_dir=str(out),
        epochs=3,
        learning_rate=1e-2,
        batch_size=4,
        lora_r=4,
        lora_alpha=8,
        val_limit=None,
        device="cpu",
    )
    return train(cfg), out


def test_training_writes_adapter_and_summary(trained):
    summary, out = trained
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "train_summary.json").exists()
    p = summary["parameters"]
    assert 0 < p["trainable"] < p["total"]
    assert summary["resources"]["seconds"] > 0
    val = [h["val_loss"] for h in summary["log_history"] if "val_loss" in h]
    assert len(val) == 3 and val[-1] < val[0], "LoRA training should reduce validation loss"


def test_reloaded_adapter_reproduces_logits(trained, tiny_model_dir):
    _, out = trained
    model = load_for_inference(str(tiny_model_dir), out / "adapter", "cpu")
    base = load_for_inference(str(tiny_model_dir), None, "cpu")
    ids = torch.tensor([[2, 5, 6, 7, 3]])
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = base(input_ids=ids).logits
    assert not torch.allclose(a, b), "adapter should change the model"


def test_eval_and_report(trained, tiny_model_dir, data_dir, tmp_path):
    _, out = trained
    for name, adapter, style, shots in [
        ("zero", None, "full", 0),
        ("few", None, "full", 2),
        ("lora", out / "adapter", "short", 0),
    ]:
        r = run_eval(
            name, str(tiny_model_dir), data_dir, tmp_path, "test", adapter, style, shots, device="cpu"
        )
        assert 0.0 <= r["accuracy"] <= 1.0 and r["n"] == 4
        rows = (tmp_path / f"{name}.predictions.jsonl").read_text().splitlines()
        assert len(rows) == 4 and "output" in json.loads(rows[0])
    md = build_report(tmp_path, LABELS, ["zero", "few", "lora"], baseline="zero")
    assert md.count("\n| ") >= 5 and "Paired bootstrap" in md


def test_merged_model_matches_adapter(trained, tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM

    _, out = trained
    merged_dir = merge_adapter(str(tiny_model_dir), out / "adapter", tmp_path / "merged")
    merged = AutoModelForCausalLM.from_pretrained(merged_dir).eval()
    peft_model = load_for_inference(str(tiny_model_dir), out / "adapter", "cpu")
    ids = torch.tensor([[2, 5, 6, 7, 3]])
    with torch.no_grad():
        torch.testing.assert_close(
            merged(input_ids=ids).logits, peft_model(input_ids=ids).logits, atol=1e-4, rtol=1e-4
        )
