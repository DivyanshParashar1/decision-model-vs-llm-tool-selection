# %% [markdown]
# # 01 — Build tool-selection instances from MetaTool
#
# **Kaggle:** CPU · Internet **On** · Input: dataset `tooldecide-lib`.
# Runs on **both** accounts (private outputs are not shared across accounts); the split hashes in
# `meta.json` must be identical on both.
#
# Output: `instances.parquet` (one row per decision instance), `tools.parquet` (name, description, unseen flag),
# `meta.json` (sizes, hashes, unseen tools, leak rates).

# %%
import glob, json, os, sys

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern} (did you attach the inputs?)"
    return hits[0]

# the lib dataset mounts flat (files at root) -> expose it as a `tooldecide` package
_pkg = os.path.dirname(find("instances.py"))
LIB = "/kaggle/tmp/lib"
os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/tooldecide"):
    os.symlink(_pkg, f"{LIB}/tooldecide")
sys.path.insert(0, LIB)
WORK = "/kaggle/working"

# %%
import pandas as pd
from tooldecide import instances as I

raw, desc, multi, official = I.load_raw()
print(len(raw), "queries;", len(desc), "tools;", len(multi), "multi;", len(official), "official")
inst, tools, meta = I.build_all(raw, desc, multi, official)

# %%
inst.to_parquet(f"{WORK}/instances.parquet", index=False)
tools.to_parquet(f"{WORK}/tools.parquet", index=False)
json.dump(meta, open(f"{WORK}/meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
print(inst.groupby("split")[["name_leak", "gold_unseen", "query_in_train"]].mean().round(3))
print(inst.iloc[0].to_dict())
print(sorted(os.listdir(WORK)))
