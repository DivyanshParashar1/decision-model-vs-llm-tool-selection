# %% [markdown]
# # 02 — Raw baselines: C1 (Qwen3-8B, no fine-tuning) and C2 (GLiNER2.5-Decide, no fine-tuning)
#
# **Kaggle:** GPU **T4 ×2** · Internet **On** · Inputs: notebook output `tools-01-data`, dataset `tooldecide-lib`.
#
# Both see the same instances: a query and 10 candidate tools with descriptions. C1 answers JSON; C2 scores
# every candidate. Output: `preds_c1.parquet`, `preds_c2.parquet` (+ `.timing.json`), `metrics_baselines.json`.

# %%
import glob, json, os, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

QUICK = True
N_PER_SPLIT = 40 if QUICK else 0          # 0 = every instance
SPLITS = "val,test_seen,test_unseen,test_multi,test_official"
LLM_ID = "Qwen/Qwen3-8B"
DECIDER = "fastino/GLiNER2.5-Decide"

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern} (did you attach the inputs?)"
    return hits[0]

INST, TOOLS = find("instances.parquet"), find("tools.parquet")
_pkg = os.path.dirname(find("prompts.py"))   # lib dataset mounts flat -> expose it as package `tooldecide`
LIB = "/kaggle/tmp/lib"
os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/tooldecide"):
    os.symlink(_pkg, f"{LIB}/tooldecide")
sys.path.insert(0, LIB)
os.environ["PYTHONPATH"] = LIB + ":" + os.environ.get("PYTHONPATH", "")
WORK = "/kaggle/working"
COMMON = f"--instances {INST} --tools {TOOLS} --splits {SPLITS} --n-per-split {N_PER_SPLIT}"
print(INST, TOOLS, _pkg, sep="\n")

# %% [markdown]
# ## C1: raw Qwen3-8B with vLLM (subprocess, so the GPUs are free afterwards)

# %%
sh(f"{sys.executable} -m pip install -q vllm==0.33.0")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
t0 = time.time()
sh(f"{sys.executable} -m tooldecide.gen_llm {COMMON} --out {WORK}/preds_c1.parquet --model {LLM_ID}")
print(f"C1 took {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## C2: raw GLiNER2.5-Decide

# %%
sh(f'{sys.executable} -m pip install -q "gliner2[local]"')
t0 = time.time()
sh(f"{sys.executable} -m tooldecide.run_decider {COMMON} --out {WORK}/preds_c2.parquet --model {DECIDER}")
print(f"C2 took {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## Metrics

# %%
import pandas as pd
from tooldecide import evaluate as E
from tooldecide import metrics as M

inst = pd.read_parquet(INST)
preds = {"C1": pd.read_parquet(f"{WORK}/preds_c1.parquet"), "C2": pd.read_parquet(f"{WORK}/preds_c2.parquet")}
res = {k: E.score(inst, p) for k, p in preds.items()}
mcn = {}
for s in SPLITS.split(","):
    a, b = E.correct_vector(inst, preds["C2"], s), E.correct_vector(inst, preds["C1"], s)
    mcn[s] = dict(c2_acc=float(a.mean()), c1_acc=float(b.mean()), p=M.mcnemar(a, b.loc[a.index]))
timing = {k: json.load(open(f"{WORK}/preds_{k.lower()}.parquet.timing.json")) for k in preds}
out = dict(quick=QUICK, n_per_split=N_PER_SPLIT, metrics=res, mcnemar_c2_vs_c1=mcn, timing=timing)
json.dump(out, open(f"{WORK}/metrics_baselines.json", "w"), indent=2)

rows = {(k, s): {m: v for m, v in d.items() if not isinstance(v, list)} for k, r in res.items() for s, d in r.items()}
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).T)
print(json.dumps(mcn, indent=1))
print(json.dumps(timing, indent=1))
c1 = preds["C1"].merge(inst[["id", "gold"]], on="id")
print("C1 raw samples:\n", c1[["raw", "pred", "gold"]].head(8).to_string())
print(sorted(os.listdir(WORK)))
