# lora-finetuning — Efficient LLM Fine‑Tuning (LoRA / QLoRA) 🔧🪶

Parameter‑efficient fine‑tuning starter for adapting open LLMs using **LoRA / QLoRA**.
Includes a minimal training script, toy instruction dataset, and an evaluation stub.

## Tech
Python 3.9+, PyTorch, Hugging Face `transformers`, optional `peft`

## Install
```bash
pip install -r requirements.txt
```

## Run (toy)
```bash
python finetune.py --base-model sshleifer/tiny-gpt2 --out-dir outputs/tiny-gpt2-lora --epochs 1 --batch-size 4
```

## Files
- `finetune.py` — LoRA pipeline + eval stub
- `data/toy.jsonl` — mini supervised pairs
- `requirements.txt`

## Notes
- Requires internet for first model download (or use a local model path)
- Upgrade with TRL (PPO/DPO), proper eval (MMLU/GSM8K), and W&B logging

© 2025 Luqman Abdulwahab. Proprietary.
