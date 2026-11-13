"""Model loading, LoRA / QLoRA setup, adapter I/O and resource measurement."""

from __future__ import annotations

import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# All linear projections in the attention and MLP blocks of Llama/Qwen-style decoders.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str, device: str, quantize_4bit: bool = False):
    """Load the base model in half precision, or in 4-bit NF4 for QLoRA (CUDA + bitsandbytes only)."""
    from transformers import AutoModelForCausalLM

    dtype = pick_dtype(device)
    if quantize_4bit:
        if device != "cuda":
            raise RuntimeError(
                "QLoRA needs a CUDA GPU: bitsandbytes 4-bit kernels are not available on MPS/CPU"
            )
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    return model.to(device)


def add_lora(model, r: int, alpha: int, dropout: float, target_modules: list[str], quantized: bool = False):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_for_inference(model_name: str, adapter: Path | str | None, device: str):
    """Base model, optionally with a saved LoRA adapter attached (not merged)."""
    model = load_base_model(model_name, device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


def count_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def merge_adapter(model_name: str, adapter: Path, out_dir: Path, device: str = "cpu") -> Path:
    """Fold the LoRA weights into the base weights and save a standalone model (no peft needed to load it)."""
    model = load_for_inference(model_name, adapter, device)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    load_tokenizer(model_name).save_pretrained(out_dir)
    return out_dir


def _accelerator_bytes(device: str) -> int:
    if device == "cuda":
        return torch.cuda.memory_allocated()
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    return 0


def _max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


@dataclass
class ResourceMonitor:
    """Wall time plus peak memory. CUDA has a native peak counter; MPS is sampled from a background thread."""

    device: str
    interval: float = 0.25
    peak_accelerator: int = 0
    seconds: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> ResourceMonitor:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps":
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_accelerator = max(self.peak_accelerator, _accelerator_bytes(self.device))
            self._stop.wait(self.interval)

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self.device == "cuda":
            self.peak_accelerator = torch.cuda.max_memory_allocated()
        elif self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.peak_accelerator = max(self.peak_accelerator, _accelerator_bytes(self.device))

    def as_dict(self) -> dict[str, float]:
        return {
            "seconds": round(self.seconds, 1),
            "peak_accelerator_gb": round(self.peak_accelerator / 1e9, 3),
            "peak_process_rss_gb": round(_max_rss_bytes() / 1e9, 3),
        }
