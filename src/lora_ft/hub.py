"""Publish a trained adapter to the Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path

CARD = """---
base_model: {base_model}
library_name: peft
license: apache-2.0
datasets:
- PolyAI/banking77
tags:
- lora
- intent-classification
---

# {repo_id}

LoRA adapter for `{base_model}` that classifies customer-support messages into the 77 Banking77 intents.
The model answers with the intent label only.

{metrics}

Training code, evaluation protocol and full results: https://github.com/luqman304/lora-finetuning

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base_model}"), "{repo_id}")
messages = [
    {{"role": "system", "content": "{instruction}"}},
    {{"role": "user", "content": "My card still hasn't arrived"}},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=16)[0, ids.shape[1]:], skip_special_tokens=True))
```
"""


def push_adapter(
    adapter_dir: Path, repo_id: str, base_model: str, eval_json: Path | None, private: bool
) -> str:
    from huggingface_hub import HfApi

    from lora_ft.prompts import INSTRUCTION

    metrics = ""
    if eval_json and eval_json.exists():
        r = json.loads(eval_json.read_text())
        metrics = (
            f"Banking77 test set (n={r['n']}): accuracy {100 * r['accuracy']:.1f}%, "
            f"macro-F1 {100 * r['macro_f1']:.1f}%."
        )
    card = CARD.format(repo_id=repo_id, base_model=base_model, metrics=metrics, instruction=INSTRUCTION)
    (adapter_dir / "README.md").write_text(card)

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, commit_message="Upload LoRA adapter")
    return f"https://huggingface.co/{repo_id}"
