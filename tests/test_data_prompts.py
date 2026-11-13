from collections import Counter

from lora_ft.data import Example, canonical_label, load_split, read_jsonl, stratified_split, write_jsonl
from lora_ft.fewshot import TfidfRetriever
from lora_ft.prompts import build_messages, parse_prediction, system_prompt
from tests.conftest import LABELS


def test_canonical_label_fixes_irregular_names():
    assert canonical_label("Refund_not_showing_up") == "refund_not_showing_up"
    assert canonical_label("reverted_card_payment?") == "reverted_card_payment"


def test_stratified_split_keeps_every_label_in_both_parts():
    data = [Example(f"t{i}", lab) for lab in "abc" for i in range(20)]
    train, val = stratified_split(data, 0.1, seed=1)
    assert len(train) + len(val) == len(data)
    assert set(train).isdisjoint(val)
    assert Counter(e.label for e in val) == {"a": 2, "b": 2, "c": 2}


def test_jsonl_roundtrip_and_seeded_limit(tmp_path, examples):
    write_jsonl(tmp_path / "x.jsonl", examples)
    assert read_jsonl(tmp_path / "x.jsonl") == examples
    a = load_split(tmp_path, "x", limit=5, seed=3)
    assert a == load_split(tmp_path, "x", limit=5, seed=3)
    assert len(a) == 5


def test_full_prompt_lists_labels_and_short_does_not():
    assert all(lab in system_prompt(LABELS, "full") for lab in LABELS)
    assert not any(lab in system_prompt(LABELS, "short") for lab in LABELS)


def test_few_shot_examples_become_chat_turns():
    shots = [Example("i lost my card", "lost_or_stolen_card")]
    msgs = build_messages("where is my card", LABELS, "short", shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "lost_or_stolen_card"


def test_parse_prediction():
    assert parse_prediction("card_arrival", LABELS) == "card_arrival"
    assert parse_prediction("  'Card Arrival'.\nextra", LABELS) == "card_arrival"
    assert parse_prediction("Intent: exchange_rate", LABELS) == "exchange_rate"
    assert parse_prediction("no idea", LABELS) is None
    assert parse_prediction("", LABELS) is None


def test_retriever_ranks_lexically_similar_examples_first(examples):
    top = TfidfRetriever(examples).top_k("my top up failed again", 2)
    assert all(e.label == "top_up_failed" for e in top)
    assert TfidfRetriever(examples).top_k("anything", 0) == []
