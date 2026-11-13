import pytest

from lora_ft.data import Example
from lora_ft.sft import IGNORE_INDEX, PadCollator, encode_example
from tests.conftest import LABELS


@pytest.fixture(scope="module")
def tok(tiny_model_dir):
    from lora_ft.modeling import load_tokenizer

    return load_tokenizer(str(tiny_model_dir))


def test_loss_is_only_on_the_answer(tok):
    ex = Example("where is my new card", "card_arrival")
    f = encode_example(tok, ex, LABELS, "short")
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"])
    supervised = [t for t in f["labels"] if t != IGNORE_INDEX]
    assert tok.decode(supervised, skip_special_tokens=True).strip() == "card_arrival"
    assert tok.convert_tokens_to_ids("<|im_end|>") in supervised
    n_masked = f["labels"].index(next(t for t in f["labels"] if t != IGNORE_INDEX))
    # supervised positions carry the same token ids as the inputs (no shift / misalignment)
    assert f["labels"][n_masked:] == f["input_ids"][n_masked:]
    assert tok.decode(f["input_ids"][:n_masked]).rstrip().endswith("assistant")


def test_truncation_that_removes_the_answer_is_rejected(tok):
    with pytest.raises(ValueError):
        encode_example(tok, Example("where is my new card", "card_arrival"), LABELS, "full", max_length=5)


def test_collator_masks_padding(tok):
    a = encode_example(tok, Example("i lost my card", "lost_or_stolen_card"), LABELS, "short")
    b = encode_example(tok, Example("my top up failed and did not work", "top_up_failed"), LABELS, "short")
    batch = PadCollator(tok.pad_token_id)([a, b])
    short = 0 if len(a["input_ids"]) < len(b["input_ids"]) else 1
    assert batch["attention_mask"][short][-1] == 1, "left padding: real tokens end at the last position"
    pad = batch["attention_mask"][short] == 0
    assert pad.any()
    assert (batch["labels"][short][pad] == IGNORE_INDEX).all()
    assert (batch["input_ids"][short][pad] == tok.pad_token_id).all()


def test_answer_loss_matches_full_sequence_loss(tok, tiny_model_dir):
    import torch
    from transformers import AutoModelForCausalLM

    from lora_ft.sft import answer_loss

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir).eval()
    feats = [
        encode_example(tok, Example("i lost my card", "lost_or_stolen_card"), LABELS, "short"),
        encode_example(tok, Example("what is the exchange rate", "exchange_rate"), LABELS, "short"),
    ]
    batch = PadCollator(tok.pad_token_id, pad_to_multiple_of=8)(feats)
    with torch.no_grad():
        fast, n = answer_loss(model, batch)
        full = model(**batch).loss
    assert n == sum(t != IGNORE_INDEX for f in feats for t in f["labels"])
    torch.testing.assert_close(fast, full, atol=1e-5, rtol=1e-5)
