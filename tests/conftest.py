"""Offline fixtures: a tiny random Qwen2-architecture model and a byte-level BPE tokenizer with a ChatML template."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_ft.data import Example, write_jsonl

LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed"]
TEXTS = {
    "card_arrival": ["where is my new card", "my card has not arrived yet", "when will my card come"],
    "lost_or_stolen_card": ["i lost my card", "someone stole my card", "my card was stolen today"],
    "exchange_rate": ["what is the exchange rate", "how much is the euro rate", "rate for dollars please"],
    "top_up_failed": ["my top up failed", "top up did not work", "why did my top up fail"],
}
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def examples() -> list[Example]:
    return [Example(text=t, label=lab) for lab, ts in TEXTS.items() for t in ts]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, examples) -> Path:
    d = tmp_path_factory.mktemp("data")
    (d / "labels.json").write_text(json.dumps(LABELS))
    write_jsonl(d / "train.jsonl", examples)
    write_jsonl(d / "val.jsonl", examples[::3])
    write_jsonl(d / "test.jsonl", examples[1::3])
    return d


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, examples) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from lora_ft.prompts import INSTRUCTION, system_prompt

    # Byte-level BPE like the real Qwen tokenizer, trained on the fixture text so it reloads identically
    # through AutoTokenizer (which maps the qwen2 model type to Qwen2Tokenizer).
    special = ["<pad>", "<|im_start|>", "<|im_end|>"]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    corpus = [INSTRUCTION, system_prompt(LABELS, "full"), "system user assistant"] + [
        ex.text for ex in examples
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    backend.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHAT_TEMPLATE
    vocab = tok.get_vocab()

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["<|im_end|>"],
        bos_token_id=None,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    out = tmp_path_factory.mktemp("tiny-qwen2")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
