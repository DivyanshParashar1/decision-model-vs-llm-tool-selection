# %% [markdown]
# # 03 — LLM\* (C3): QLoRA fine-tune Qwen3-8B, then predict on every test split
#
# **Kaggle:** GPU **T4 ×2** (training uses one, vLLM inference uses both) · Internet **On** ·
# Inputs: notebook output `tools-01-data`, dataset `tooldecide-lib`. Runs on the **second account**.
#
# One notebook run = one training size `N_TRAIN` (1k / 4k / full); push it once per size with a different slug.
# The train ids are the first `N_TRAIN` of a fixed permutation, the same ids D\* uses.
# Output: `preds_llm_star_<n>.parquet` (+ timing), `cost_llm_star_<n>.json`, `metrics_llm_star_<n>.json`,
# and the LoRA adapter.

# %%
import glob, json, os, shutil, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

QUICK = True
N_TRAIN = 1000                 # 1000 / 4000 / 1000000 (= full)
SPLITS = "val,test_seen,test_unseen,test_multi,test_official"
N_PER_SPLIT = 0                # 0 = every instance
MAX_STEPS = -1
TIME_BUDGET_H = 8.0
if QUICK:
    N_TRAIN, N_PER_SPLIT, MAX_STEPS = 64, 30, 6

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern} (did you attach the inputs?)"
    return hits[0]

INST, TOOLS = find("instances.parquet"), find("tools.parquet")
_pkg = os.path.dirname(find("prompts.py"))
LIB = "/kaggle/tmp/lib"
os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/tooldecide"):
    os.symlink(_pkg, f"{LIB}/tooldecide")
sys.path.insert(0, LIB)
os.environ["PYTHONPATH"] = LIB + ":" + os.environ.get("PYTHONPATH", "")
WORK = "/kaggle/working"

import pandas as pd
inst = pd.read_parquet(INST)
n = min(N_TRAIN, int((inst["split"] == "train").sum()))
TAG = f"{n}"
ADAPTER = f"{WORK}/llm_star_{TAG}"
print(f"training size {n}; adapter -> {ADAPTER}")

# %% [markdown]
# ## Train (subprocess, one T4)

# %%
sh(f"{sys.executable} -m pip install -q unsloth")
t0 = time.time()
sh(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} -m tooldecide.train_llm --instances {INST} --tools {TOOLS} "
   f"--n {n} --out {ADAPTER} --max-steps {MAX_STEPS} --time-budget-h {TIME_BUDGET_H}"
   + (" --warmup-steps 2" if QUICK else ""))
train_wall = time.time() - t0
shutil.rmtree(ADAPTER + "_ckpt", ignore_errors=True)
meta = json.load(open(f"{ADAPTER}/train_meta.json"))
print(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=1))

# %% [markdown]
# ## Predict with the adapter (vLLM, both T4s; subprocess)

# %%
sh(f"{sys.executable} -m pip install -q vllm==0.33.0")
PRED = f"{WORK}/preds_llm_star_{TAG}.parquet"
t0 = time.time()
sh(f"{sys.executable} -m tooldecide.gen_llm --instances {INST} --tools {TOOLS} --splits {SPLITS} "
   f"--n-per-split {N_PER_SPLIT} --out {PRED} --lora {ADAPTER}")
gen_wall = time.time() - t0

# %% [markdown]
# ## Metrics + cost

# %%
from tooldecide import evaluate as E

preds = pd.read_parquet(PRED)
res = E.score(inst, preds)
json.dump(dict(quick=QUICK, n_train=n, metrics=res), open(f"{WORK}/metrics_llm_star_{TAG}.json", "w"), indent=2)
timing = json.load(open(PRED + ".timing.json"))
cost = dict(meta, gpu_hours_train=round(meta["train_seconds"] / 3600, 3), train_wall_seconds=train_wall,
            gen_wall_seconds=gen_wall, inference=timing, quick=QUICK)
json.dump(cost, open(f"{WORK}/cost_llm_star_{TAG}.json", "w"), indent=2)
pd.set_option("display.width", 250)
print(pd.DataFrame({s: {m: v for m, v in d.items() if not isinstance(v, list)} for s, d in res.items()}).T)
print(preds.merge(inst[["id", "gold"]], on="id")[["raw", "pred", "gold"]].head(8).to_string())
print(sorted(os.listdir(WORK)))
