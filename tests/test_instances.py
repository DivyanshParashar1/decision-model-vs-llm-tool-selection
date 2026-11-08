import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tooldecide import instances as I  # noqa: E402


def test_norm_and_name_words():
    assert I.norm_query("  Hello,   World!? ") == "hello world"
    assert I.name_words("ResearchHelper") == "research helper"
    assert I.name_words("video_highlight") == "video highlight"
    assert I.name_words("AI2sql") == "ai2sql"


def test_name_leak():
    assert I.name_leak("Can ResearchHelper find papers?", "ResearchHelper")
    assert I.name_leak("use the research helper please", "ResearchHelper")
    assert I.name_leak("Play chess with me", "Chess")
    assert not I.name_leak("chessboard art", "Chess")
    assert not I.name_leak("find papers", "ResearchHelper")


def test_dedupe_drops_ambiguous_and_duplicates():
    df = pd.DataFrame({"Query": ["Hi there", "hi there!", "same q", "same Q", "other"],
                       "Tool": ["A", "A", "A", "B", "B"]})
    out, stats = I.dedupe(df)
    assert sorted(out["nq"]) == ["hi there", "other"]
    assert stats["ambiguous_queries_dropped"] == 1


def test_pick_unseen_one_per_bin():
    counts = {f"t{i}": 100 - i for i in range(20)}
    u = I.pick_unseen(counts, 4, seed=0)
    assert len(u) == 4 and len(set(u)) == 4
    ranks = sorted(int(t[1:]) // 5 for t in u)
    assert ranks == [0, 1, 2, 3]  # one from each frequency bin
    assert u == I.pick_unseen(counts, 4, seed=0)


def test_build_candidates():
    tools = [f"t{i}" for i in range(30)]
    nbrs = {t: [x for x in tools if x != t] for t in tools}
    cands, hard = I.build_candidates(["t5"], tools[:20], nbrs, seed_int=7)
    assert len(cands) == 10 and len(set(cands)) == 10 and "t5" in cands
    assert hard == ["t0", "t1", "t2", "t3"]
    assert all(c in tools[:20] for c in cands)
    assert (cands, hard) == I.build_candidates(["t5"], tools[:20], nbrs, seed_int=7)
    c2, h2 = I.build_candidates(["t5", "t9"], tools, nbrs, seed_int=1)
    assert {"t5", "t9"} <= set(c2) and len(c2) == 10 and len(h2) == 4


def _toy():
    rng = __import__("random").Random(0)
    words = ["weather", "news", "stock", "recipe", "travel", "music", "movie", "code", "math", "game"]
    desc = {f"Tool{i}": f"helps with {words[i % 10]} and {words[(i * 3) % 10]} things {i}" for i in range(40)}
    rows = [(f"query {i} about {words[int(t[4:]) % 10]} {j}", t)
            for i, t in enumerate(desc) for j in range(rng.randint(10, 40))]
    raw = pd.DataFrame(rows, columns=["Query", "Tool"])
    multi = [{"query": "weather and news please", "tool": ["Tool0", "Tool1"]}]
    names = list(desc)[:10]
    prompt = "\n".join(f"{i + 1}. tool name: {n}, tool description: x" for i, n in enumerate(names))
    official = [{"action_prompt": prompt, "tool": "Tool3", "query": raw.Query[0], "index": 0}]
    return raw, desc, multi, official


def test_build_all_is_deterministic_and_leak_free():
    raw, desc, multi, official = _toy()
    inst, tools, meta = I.build_all(raw, desc, multi, official, cap=25, n_unseen=5)
    inst2, _, meta2 = I.build_all(raw, desc, multi, official, cap=25, n_unseen=5)
    assert meta["split_hashes"] == meta2["split_hashes"]
    assert set(inst["split"]) == {"train", "val", "test_seen", "test_unseen", "test_multi", "test_official"}
    unseen = set(meta["unseen_tools"])
    seen_splits = inst[inst["split"].isin(["train", "val", "test_seen"])]
    for cands in seen_splits["cands"]:
        assert not unseen & set(json.loads(cands)), "unseen tool used as a distractor in a seen split"
    assert all(json.loads(g)[0] in unseen for g in inst.loc[inst.split == "test_unseen", "gold"])
    for r in inst.itertuples():
        c = json.loads(r.cands)
        assert len(c) == 10 and set(json.loads(r.gold)) <= set(c)
    tr = inst[inst.split == "train"]
    assert sorted(tr.train_rank) == list(range(len(tr)))
    assert (inst.loc[inst.split != "train", "train_rank"] == -1).all()
    assert inst.loc[inst.split == "test_official", "query_in_train"].dtype == bool
    assert tools["n_capped"].max() <= 25


def test_official_candidates():
    p = "1. tool name: A_b, tool description: ['x, y']\n2. tool name: Cd, tool description: z"
    assert I.official_candidates(p) == ["A_b", "Cd"]
