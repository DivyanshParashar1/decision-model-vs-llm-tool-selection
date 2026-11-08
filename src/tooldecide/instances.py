"""Build k-candidate tool-selection instances from MetaTool (deterministic, no models).

One instance = (query, k candidate tools, gold tool(s)). Candidates = gold + 4 "hard" distractors (tools whose
descriptions are most TF-IDF-similar to the gold's) + random tools, shuffled with a per-instance seed.

Splits:
  train / val / test_seen : queries of the 169 seen tools, 80/10/10 per tool; distractors drawn from seen tools only
  test_unseen             : queries of 30 held-out tools (picked one per frequency bin); distractors from all tools
  test_multi              : MetaTool two-tool queries; gold = both tools; distractors from all tools
  test_official           : MetaTool Task 2 Subtask 1 prompts with their own 10 candidates, order kept

Everything is a pure function of the raw files + SEED, so two Kaggle accounts build identical splits
(checked via the hashes in meta.json).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

SEED = 0
K = 10
N_HARD = 4
CAP = 150
N_UNSEEN = 30
DATA_VERSION = "d1"

RAW_BASE = "https://raw.githubusercontent.com/HowieHwong/MetaTool/master/dataset"
RAW_FILES = {
    "queries": "data/all_clean_data.csv",
    "descriptions": "plugin_des.json",
    "multi": "data/multi_tool_query_golden.json",
    "official": "tmp_dataset/Task2-Subtask1.json",
}


# ------------------------------------------------------------------ text helpers
def norm_query(q: str) -> str:
    """Lowercase, keep letters/digits, collapse spaces. Used for dedupe and leak checks."""
    q = re.sub(r"[^a-z0-9]+", " ", str(q).lower())
    return " ".join(q.split())


def name_words(tool: str) -> str:
    """'ResearchHelper' -> 'research helper', 'video_highlight' -> 'video highlight'."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tool)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return norm_query(s)


def name_leak(query: str, tool: str) -> bool:
    """True if the tool's name appears in the query (as words, or glued, e.g. 'researchhelper')."""
    q = f" {norm_query(query)} "
    spaced = name_words(tool)
    glued = spaced.replace(" ", "")
    return f" {spaced} " in q or f" {glued} " in q


def stable_int(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def inst_id(prefix: str, *parts) -> str:
    return prefix + hashlib.sha1("\t".join(map(str, parts)).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ cleaning and splits
def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates (by normalized text) and queries labeled with more than one tool."""
    d = df.assign(nq=df["Query"].map(norm_query))
    d = d[d["nq"] != ""]
    n_tools = d.groupby("nq")["Tool"].nunique()
    ambiguous = set(n_tools[n_tools > 1].index)
    out = d[~d["nq"].isin(ambiguous)].drop_duplicates("nq", keep="first").reset_index(drop=True)
    return out, {"raw": len(df), "ambiguous_queries_dropped": len(ambiguous), "after_dedupe": len(out)}


def cap_per_tool(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    parts = []
    for tool, g in df.groupby("Tool", sort=True):
        g = g.sort_values("nq")
        if len(g) > cap:
            g = g.sample(n=cap, random_state=stable_int(seed, "cap", tool) % 2**32)
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def pick_unseen(counts: dict[str, int], n: int, seed: int) -> list[str]:
    """Sort tools by frequency, cut into n contiguous bins, pick one tool per bin (stratified by frequency)."""
    order = sorted(counts, key=lambda t: (-counts[t], t))
    rng = np.random.default_rng(seed)
    return sorted(str(b[rng.integers(len(b))]) for b in np.array_split(np.array(order), n))


def split_seen(df: pd.DataFrame, seed: int, frac_val=0.1, frac_test=0.1) -> pd.Series:
    """Per tool: shuffle, then val / test_seen / train (stratified by tool)."""
    split = pd.Series("train", index=df.index)
    for tool, g in df.groupby("Tool", sort=True):
        idx = g.sort_values("nq").index.to_numpy()
        idx = np.random.default_rng(stable_int(seed, "split", tool) % 2**32).permutation(idx)
        n_val, n_test = round(frac_val * len(idx)), round(frac_test * len(idx))
        split[idx[:n_val]] = "val"
        split[idx[n_val:n_val + n_test]] = "test_seen"
    return split


# ------------------------------------------------------------------ distractors
def tfidf_neighbors(desc: dict[str, str]) -> dict[str, list[str]]:
    """{tool: all other tools, most similar description first}. Text = name words + description."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tools = sorted(desc)
    docs = [f"{name_words(t)} {desc[t]}" for t in tools]
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    sim = np.round((X @ X.T).toarray(), 6)
    out = {}
    for i, t in enumerate(tools):
        others = [j for j in range(len(tools)) if j != i]
        others.sort(key=lambda j: (-sim[i, j], tools[j]))
        out[t] = [tools[j] for j in others]
    return out


def build_candidates(gold: list[str], pool: list[str], neighbors: dict[str, list[str]], seed_int: int,
                     k: int = K, n_hard: int = N_HARD) -> tuple[list[str], list[str]]:
    """Return (shuffled candidates, hard distractors). Hard ones are split evenly across the gold tools."""
    pool_set = set(pool) - set(gold)
    hard: list[str] = []
    per_gold = [n_hard // len(gold) + (i < n_hard % len(gold)) for i in range(len(gold))]
    for g, m in zip(gold, per_gold):
        hard += [t for t in neighbors[g] if t in pool_set and t not in hard][:m]
    rest = sorted(pool_set - set(hard))
    rng = np.random.default_rng(seed_int % 2**32)
    rand = [rest[i] for i in rng.choice(len(rest), size=k - len(gold) - len(hard), replace=False)]
    cands = list(gold) + hard + rand
    return [cands[i] for i in rng.permutation(len(cands))], hard


# ------------------------------------------------------------------ official prompts
_OFFICIAL_NAME = re.compile(r"tool name: (.*?), tool description")


def official_candidates(action_prompt: str) -> list[str]:
    return _OFFICIAL_NAME.findall(action_prompt)


# ------------------------------------------------------------------ main builder
def build_all(raw: pd.DataFrame, desc: dict[str, str], multi: list[dict], official: list[dict],
              seed: int = SEED, cap: int = CAP, n_unseen: int = N_UNSEEN, k: int = K, n_hard: int = N_HARD):
    """Returns (instances DataFrame, tools DataFrame, meta dict)."""
    clean, stats = dedupe(raw)
    capped = cap_per_tool(clean, cap, seed)
    counts = clean["Tool"].value_counts().to_dict()
    unseen = pick_unseen(counts, n_unseen, seed)
    seen_tools = sorted(set(desc) - set(unseen))
    all_tools = sorted(desc)
    nbrs = tfidf_neighbors(desc)

    is_unseen = capped["Tool"].isin(unseen)
    seen_df = capped[~is_unseen].copy()
    seen_df["split"] = split_seen(seen_df, seed)
    unseen_df = capped[is_unseen].copy()
    unseen_df["split"] = "test_unseen"

    rows = []
    for r in pd.concat([seen_df, unseen_df]).itertuples(index=False):
        iid = inst_id("q", r.Tool, r.nq)
        pool = all_tools if r.split == "test_unseen" else seen_tools
        cands, hard = build_candidates([r.Tool], pool, nbrs, stable_int(seed, "cands", iid), k, n_hard)
        rows.append(dict(id=iid, split=r.split, query=r.Query, gold=[r.Tool], cands=cands, hard=hard))

    seen_nq = set(clean["nq"])
    multi_seen = set()
    for m in multi:
        nq = norm_query(m["query"])
        gold = sorted(set(m["tool"]))
        if nq in multi_seen or len(gold) != 2 or any(g not in desc for g in gold):
            continue
        multi_seen.add(nq)
        iid = inst_id("m", *gold, nq)
        cands, hard = build_candidates(gold, all_tools, nbrs, stable_int(seed, "cands", iid), k, n_hard)
        rows.append(dict(id=iid, split="test_multi", query=m["query"], gold=gold, cands=cands, hard=hard))

    for o in official:
        cands = official_candidates(o["action_prompt"])
        if o["tool"] not in cands or any(c not in desc for c in cands):
            continue
        iid = inst_id("o", o["index"], o["tool"], norm_query(o["query"]))
        hard = [t for t in nbrs[o["tool"]] if t in cands][:n_hard]
        rows.append(dict(id=iid, split="test_official", query=o["query"], gold=[o["tool"]], cands=cands, hard=hard))

    inst = pd.DataFrame(rows)
    train_nq = set(seen_df.loc[seen_df["split"] == "train", "nq"])
    nqs = inst["query"].map(norm_query)
    inst["name_leak"] = [any(name_leak(q, g) for g in gs) for q, gs in zip(inst["query"], inst["gold"])]
    inst["gold_unseen"] = inst["gold"].map(lambda gs: any(g in unseen for g in gs))
    inst["query_in_train"] = nqs.isin(train_nq) & (inst["split"] != "train")
    inst["query_in_raw"] = nqs.isin(seen_nq)

    # fixed training order: sizes 1k / 4k / full are prefixes of the same permutation (same ids for both arms)
    inst = inst.sort_values(["split", "id"]).reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(inst.index[inst["split"] == "train"].to_numpy())
    inst["train_rank"] = -1
    inst.loc[order, "train_rank"] = np.arange(len(order))
    for c in ["gold", "cands", "hard"]:
        inst[c] = inst[c].map(json.dumps)

    tools = pd.DataFrame({"tool": all_tools, "description": [desc[t] for t in all_tools]})
    tools["unseen"] = tools["tool"].isin(unseen)
    tools["n_after_dedupe"] = tools["tool"].map(counts).fillna(0).astype(int)
    tools["n_capped"] = tools["tool"].map(capped["Tool"].value_counts()).fillna(0).astype(int)

    meta = dict(data_version=DATA_VERSION, seed=seed, k=k, n_hard=n_hard, cap=cap, n_unseen=n_unseen,
                cleaning=stats, after_cap=len(capped), unseen_tools=unseen,
                split_sizes=inst["split"].value_counts().sort_index().to_dict(),
                split_hashes=split_hashes(inst), tools_hash=_sha(tools.to_json(orient="records")),
                leak_rate=inst.groupby("split")["name_leak"].mean().round(4).to_dict(),
                official_query_in_train=float(inst.loc[inst.split == "test_official", "query_in_train"].mean()))
    return inst, tools, meta


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def split_hashes(inst: pd.DataFrame) -> dict[str, str]:
    cols = ["id", "query", "gold", "cands", "train_rank"]
    return {s: _sha(g.sort_values("id")[cols].to_json(orient="records"))
            for s, g in inst.groupby("split", sort=True)}


def load_raw(base: str = RAW_BASE):
    """Download the four MetaTool files (pinned to the repo's default branch at build time)."""
    import io
    import urllib.request

    def get(path):
        with urllib.request.urlopen(f"{base}/{path}", timeout=60) as r:
            return r.read()

    raw = pd.read_csv(io.BytesIO(get(RAW_FILES["queries"])))
    desc = json.loads(get(RAW_FILES["descriptions"]))
    multi = json.loads(get(RAW_FILES["multi"]))
    official = json.loads(get(RAW_FILES["official"]))
    return raw, desc, multi, official
