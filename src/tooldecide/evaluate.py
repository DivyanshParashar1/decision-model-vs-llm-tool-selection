"""Score a predictions table against the instances, per split. Used by notebooks and the final report.

Predictions table (one row per instance id):
  id     : instance id
  pred   : JSON list of predicted tool names, best first ([] = unparseable / no answer)
  conf   : confidence of the top pick (NaN for the LLM)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import metrics as M


def _ci(x):
    m, lo, hi = M.bootstrap_ci(x)
    return round(m, 4), [round(lo, 4), round(hi, 4)]


def join(inst: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    d = inst.merge(preds[["id", "pred"] + (["conf"] if "conf" in preds else [])], on="id", how="inner")
    d["gold_l"] = d["gold"].map(json.loads)
    d["hard_l"] = d["hard"].map(json.loads)
    d["pred_l"] = d["pred"].map(lambda s: json.loads(s) if isinstance(s, str) else list(s))
    d["top1"] = d["pred_l"].map(lambda p: p[0] if p else None)
    if "conf" not in d:
        d["conf"] = np.nan
    return d


def score_split(d: pd.DataFrame) -> dict:
    """Metrics for one split's joined rows."""
    out = {"n": len(d), "parse_rate": round(float(d["pred_l"].map(bool).mean()), 4)}
    if d["gold_l"].map(len).max() > 1:  # multi-tool
        exact, f1 = M.set_scores([p[:2] for p in d["pred_l"]], d["gold_l"])
        out["exact"], out["exact_ci"] = _ci(exact)
        out["set_f1"], out["set_f1_ci"] = _ci(f1)
        out["any_gold_top1"] = round(float(np.mean([t in g for t, g in zip(d["top1"], d["gold_l"])])), 4)
        return out
    gold = d["gold_l"].map(lambda g: g[0]).to_numpy()
    correct = M.accuracy(d["top1"], gold)
    out["acc"], out["acc_ci"] = _ci(correct)
    m, lo, hi = M.macro_bootstrap_ci(correct, gold)
    out["macro_acc"], out["macro_acc_ci"] = round(m, 4), [round(lo, 4), round(hi, 4)]
    out["hard_confusion"] = round(M.hard_confusion(d["top1"], gold, d["hard_l"]), 4)
    leak = d["name_leak"].to_numpy(bool)
    out["acc_no_leak"] = round(float(correct[~leak].mean()), 4) if (~leak).any() else None
    out["acc_leak"] = round(float(correct[leak].mean()), 4) if leak.any() else None
    if (d["split"] == "test_official").all():
        clean = ~d["query_in_train"].to_numpy(bool)
        out["acc_query_not_in_train"], out["acc_query_not_in_train_ci"] = _ci(correct[clean])
        out["n_query_not_in_train"] = int(clean.sum())
        seen = ~d["gold_unseen"].to_numpy(bool)
        out["acc_gold_seen"] = round(float(correct[seen].mean()), 4) if seen.any() else None
        out["acc_gold_unseen"] = round(float(correct[~seen].mean()), 4) if (~seen).any() else None
    conf = d["conf"].to_numpy(float)
    if np.isfinite(conf).all():
        out["ece"] = round(M.ece(conf, correct), 4)
        out["acc_at_90_coverage"] = round(M.acc_at_coverage(conf, correct, 0.9), 4)
    return out


def score(inst: pd.DataFrame, preds: pd.DataFrame) -> dict:
    d = join(inst, preds)
    return {s: score_split(g) for s, g in d.groupby("split", sort=True)}


def correct_vector(inst: pd.DataFrame, preds: pd.DataFrame, split: str) -> pd.Series:
    """Per-instance correctness (top-1 for single, exact set for multi), indexed by id, for McNemar."""
    d = join(inst, preds)
    d = d[d["split"] == split].sort_values("id")
    if split == "test_multi":
        ok, _ = M.set_scores([p[:2] for p in d["pred_l"]], d["gold_l"])
    else:
        ok = M.accuracy(d["top1"], d["gold_l"].map(lambda g: g[0]))
    return pd.Series(ok, index=d["id"].to_numpy())
