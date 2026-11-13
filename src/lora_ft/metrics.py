"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

INVALID = -1


def encode(values: list[str | None], labels: list[str]) -> np.ndarray:
    """Label strings -> indices; None or unknown strings become INVALID (always counted as wrong)."""
    index = {lab: i for i, lab in enumerate(labels)}
    return np.array([index.get(v, INVALID) if v is not None else INVALID for v in values], dtype=np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> float:
    """Unweighted mean of per-class F1 over the classes present in y_true.

    Invalid predictions count as false negatives for the true class and as nobody's false positive.
    """
    if len(y_true) == 0:
        return 0.0
    correct = y_true == y_pred
    tp = np.bincount(y_true[correct], minlength=n_labels)
    support = np.bincount(y_true, minlength=n_labels)
    valid = y_pred != INVALID
    predicted = np.bincount(y_pred[valid], minlength=n_labels)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_labels, dtype=float), where=denom > 0)
    present = support > 0
    return float(f1[present].mean())


def bootstrap_ci(
    stat: Callable[[np.ndarray], float], n: int, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of a resampled index array."""
    rng = np.random.default_rng(seed)
    values = np.array([stat(rng.integers(0, n, n)) for _ in range(n_resamples)])
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def summarize(
    y_true: np.ndarray, y_pred: np.ndarray, n_labels: int, n_resamples: int = 1000, seed: int = 0
) -> dict[str, float]:
    n = len(y_true)
    acc_ci = bootstrap_ci(lambda i: accuracy(y_true[i], y_pred[i]), n, n_resamples, seed=seed)
    f1_ci = bootstrap_ci(lambda i: macro_f1(y_true[i], y_pred[i], n_labels), n, n_resamples, seed=seed)
    return {
        "n": n,
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci": list(acc_ci),
        "macro_f1": macro_f1(y_true, y_pred, n_labels),
        "macro_f1_ci": list(f1_ci),
        "invalid_rate": float(np.mean(y_pred == INVALID)) if n else 0.0,
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Accuracy difference (a - b) on the same examples, with a paired bootstrap CI."""
    diff = accuracy(y_true, pred_a) - accuracy(y_true, pred_b)
    lo, hi = bootstrap_ci(
        lambda i: accuracy(y_true[i], pred_a[i]) - accuracy(y_true[i], pred_b[i]),
        len(y_true),
        n_resamples,
        seed=seed,
    )
    return {"accuracy_diff": diff, "ci": [lo, hi]}
