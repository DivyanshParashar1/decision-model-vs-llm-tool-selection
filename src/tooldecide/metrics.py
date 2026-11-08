"""Metrics: accuracy, macro accuracy, multi-tool set match/F1, hard-distractor confusion, calibration,
bootstrap CIs and McNemar. All inputs are plain lists/arrays so the functions are easy to test."""
from __future__ import annotations

import numpy as np


def accuracy(pred, gold) -> np.ndarray:
    """Per-instance correctness for single-tool instances. pred: name, or None/NaN for no answer; gold: name."""
    return np.array([isinstance(p, str) and p == g for p, g in zip(pred, gold)], dtype=bool)


def macro_accuracy(correct, gold) -> float:
    """Mean over tools of per-tool accuracy (each tool counts once, however many queries it has)."""
    correct, gold = np.asarray(correct, bool), np.asarray(gold, object)
    return float(np.mean([correct[gold == t].mean() for t in np.unique(gold)]))


def set_scores(pred_sets, gold_sets) -> tuple[np.ndarray, np.ndarray]:
    """(exact set match, set F1) per instance."""
    exact, f1 = [], []
    for p, g in zip(pred_sets, gold_sets):
        p, g = set(p or []), set(g)
        tp = len(p & g)
        exact.append(p == g)
        f1.append(0.0 if tp == 0 else 2 * tp / (len(p) + len(g)))
    return np.array(exact, bool), np.array(f1, float)


def hard_confusion(pred, gold, hard_lists) -> float:
    """Among wrong (but parsed) answers, the fraction that picked one of the TF-IDF-similar distractors."""
    wrong = [(p, h) for p, g, h in zip(pred, gold, hard_lists) if isinstance(p, str) and p != g]
    return float(np.mean([p in h for p, h in wrong])) if wrong else float("nan")


def ece(conf, correct, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def acc_at_coverage(conf, correct, coverage: float = 0.9) -> float:
    """Accuracy on the `coverage` fraction of instances with the highest confidence (abstain on the rest)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = max(1, int(round(coverage * len(conf))))
    keep = np.argsort(-conf, kind="stable")[:n]
    return float(correct[keep].mean())


def bootstrap_ci(x, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def macro_bootstrap_ci(correct, gold, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """CI for macro accuracy by resampling tools (clusters), not instances."""
    correct, gold = np.asarray(correct, bool), np.asarray(gold, object)
    tools = np.unique(gold)
    per_tool = np.array([correct[gold == t].mean() for t in tools])
    rng = np.random.default_rng(seed)
    means = per_tool[rng.integers(0, len(tools), size=(n, len(tools)))].mean(axis=1)
    return float(per_tool.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a, b) -> float:
    """Exact two-sided McNemar p-value on paired booleans."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)
