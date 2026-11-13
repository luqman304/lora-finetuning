import numpy as np
import pytest

from lora_ft.metrics import INVALID, accuracy, encode, macro_f1, paired_bootstrap_diff, summarize


def test_encode_maps_unknown_and_none_to_invalid():
    assert encode(["b", None, "zzz", "a"], ["a", "b"]).tolist() == [1, INVALID, INVALID, 0]


def test_macro_f1_matches_hand_computation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, INVALID])
    # class 0: tp=1, support=2, predicted=1 -> 2/3; class 1: tp=1, support=2, predicted=2 -> 1/2
    assert macro_f1(y, p, 3) == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert accuracy(y, p) == 0.5


def test_perfect_predictions():
    y = np.arange(5)
    s = summarize(y, y.copy(), 5, n_resamples=50)
    assert s["accuracy"] == s["macro_f1"] == 1.0
    assert s["invalid_rate"] == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    p = np.where(rng.random(400) < 0.7, y, (y + 1) % 4)
    s = summarize(y, p, 4, n_resamples=300)
    assert s["accuracy_ci"][0] < s["accuracy"] < s["accuracy_ci"][1]
    assert s["macro_f1_ci"][0] < s["macro_f1"] < s["macro_f1_ci"][1]


def test_paired_diff_sign():
    y = np.zeros(100, dtype=int)
    good, bad = y.copy(), np.full(100, INVALID)
    d = paired_bootstrap_diff(y, good, bad, n_resamples=50)
    assert d["accuracy_diff"] == 1.0 and d["ci"][0] > 0
