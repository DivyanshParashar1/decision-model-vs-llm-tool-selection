import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tooldecide import decider as D  # noqa: E402
from tooldecide import evaluate as E  # noqa: E402
from tooldecide import metrics as M  # noqa: E402
from tooldecide import prompts as P  # noqa: E402

CANDS = ["WeatherTool", "NewsTool", "PDF&URLTool", "video_highlight", "Chess"]
DESC = {c: f"desc of {c}" for c in CANDS}


def test_messages_and_target():
    m = P.messages("rain?", CANDS, DESC, 1)
    assert m[0]["role"] == "system" and "1. WeatherTool: desc of WeatherTool" in m[1]["content"]
    assert '{"tool"' in m[1]["content"]
    assert '"tools"' in P.messages("x", CANDS, DESC, 2)[1]["content"]
    assert json.loads(P.target(["A"])) == {"tool": "A"}
    assert json.loads(P.target(["A", "B"])) == {"tools": ["A", "B"]}


def test_parse_tool_variants():
    assert P.parse_tool('{"tool": "WeatherTool"}', CANDS) == "WeatherTool"
    assert P.parse_tool('<think>hmm NewsTool</think>\n{"tool": "weathertool"}', CANDS) == "WeatherTool"
    assert P.parse_tool('```json\n{"tool": "Video Highlight"}\n```', CANDS) == "video_highlight"
    assert P.parse_tool('{"tool": 2}', CANDS) == "NewsTool"
    assert P.parse_tool('{"tool": "1. WeatherTool"}', CANDS) == "WeatherTool"
    assert P.parse_tool("I would use PDF&URLTool for this.", CANDS) == "PDF&URLTool"
    assert P.parse_tool('{"tool": "Nonexistent"}', CANDS) is None
    assert P.parse_tool("", CANDS) is None
    assert P.parse_tool("chessboard", CANDS) is None


def test_parse_tools_multi():
    assert P.parse_tools('{"tools": ["NewsTool", "WeatherTool"]}', CANDS, 2) == ["NewsTool", "WeatherTool"]
    assert P.parse_tools('{"tools": ["NewsTool", "NewsTool", "Chess", "x"]}', CANDS, 2) == ["NewsTool", "Chess"]
    assert P.parse_tools("Use Chess and then NewsTool", CANDS, 2) == ["Chess", "NewsTool"]


def test_metrics_basic():
    assert M.accuracy(["a", None, "b"], ["a", "a", "c"]).tolist() == [True, False, False]
    assert M.macro_accuracy([1, 1, 1, 0], ["a", "a", "a", "b"]) == 0.5
    ex, f1 = M.set_scores([["a", "b"], ["a", "c"], []], [["b", "a"], ["a", "b"], ["a", "b"]])
    assert ex.tolist() == [True, False, False] and f1.tolist() == [1.0, 0.5, 0.0]
    assert M.hard_confusion(["x", "a", "y", None], ["a", "a", "b", "b"], [["x"], ["x"], ["z"], ["z"]]) == 0.5
    assert math.isnan(M.hard_confusion(["a"], ["a"], [["x"]]))


def test_calibration():
    assert M.ece([1.0, 1.0], [1, 1]) == 0.0
    assert abs(M.ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9
    assert M.acc_at_coverage([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], 0.5) == 1.0


def test_bootstrap_and_mcnemar():
    m, lo, hi = M.bootstrap_ci([1, 0] * 50)
    assert m == 0.5 and lo < 0.5 < hi
    assert M.mcnemar([1, 1, 0], [1, 1, 0]) == 1.0
    assert M.mcnemar([1] * 30, [0] * 30) < 1e-6
    m, lo, hi = M.macro_bootstrap_ci([1, 1, 0, 0], ["a", "a", "b", "b"])
    assert m == 0.5 and lo <= m <= hi


def test_decider_helpers():
    raw = {"tool": [("A", 0.7), ("B", 0.2), ("C", 0.1)]}
    assert D.scores_from_raw(raw, ["A", "B", "C"]) == {"A": 0.7, "B": 0.2, "C": 0.1}
    raw2 = {"tool": [{"label": "B", "confidence": 0.6}]}
    assert D.scores_from_raw(raw2, ["A", "B"]) == {"B": 0.6}
    assert D.ranked({"A": 0.1, "B": 0.6}, ["A", "B", "C"]) == ["B", "A", "C"]
    cfg = D.task_config(["WeatherTool"], DESC)["tool"]
    assert cfg["multi_label"] and cfg["cls_threshold"] == 0.0 and cfg["class_act"] == "softmax"
    assert cfg["labels"] == {"WeatherTool": "desc of WeatherTool"}


def _inst():
    rows = [
        dict(id="a", split="test_seen", gold=["T1"], hard=["T2"], name_leak=False),
        dict(id="b", split="test_seen", gold=["T1"], hard=["T2"], name_leak=True),
        dict(id="c", split="test_seen", gold=["T3"], hard=["T2"], name_leak=False),
        dict(id="d", split="test_multi", gold=["T1", "T2"], hard=[], name_leak=False),
    ]
    df = pd.DataFrame(rows)
    df["gold"], df["hard"] = df["gold"].map(json.dumps), df["hard"].map(json.dumps)
    df["query_in_train"], df["gold_unseen"] = False, False
    return df


def test_evaluate_score():
    preds = pd.DataFrame(dict(id=list("abcd"), pred=[json.dumps(x) for x in [["T1"], ["T2"], [], ["T2", "T1"]]],
                              conf=[0.9, 0.8, 0.1, 0.5]))
    r = E.score(_inst(), preds)
    s = r["test_seen"]
    assert s["n"] == 3 and abs(s["acc"] - 1 / 3) < 1e-4 and abs(s["parse_rate"] - 2 / 3) < 1e-4
    assert s["macro_acc"] == 0.25 and s["hard_confusion"] == 1.0
    assert s["acc_leak"] == 0.0 and s["acc_no_leak"] == 0.5
    assert "ece" in s
    assert r["test_multi"]["exact"] == 1.0 and r["test_multi"]["set_f1"] == 1.0
    v = E.correct_vector(_inst(), preds, "test_seen")
    assert v.tolist() == [True, False, False]
    preds_llm = preds.drop(columns="conf")
    assert "ece" not in E.score(_inst(), preds_llm)["test_seen"]
    assert np.isnan(E.join(_inst(), preds_llm)["conf"]).all()
