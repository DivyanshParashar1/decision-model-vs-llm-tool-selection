# %% [markdown]
# # 04 — D\* (C4): fine-tune GLiNER2.5-Decide at 1k / 4k / full × 3 seeds, predict on every test split
#
# **Kaggle:** GPU **T4 ×2** · Internet **On** · Inputs: notebook output `tools-01-data`, dataset `tooldecide-lib`.
#
# Every (size, seed) job runs in its own subprocesses (train, then predict). Two jobs run at once, one per T4.
# Same train ids as LLM\* (first n of the fixed permutation). Results are saved after every job.
# Output: `preds_dstar_<n>_s<seed>.parquet` (+ timing), `cost_dstar.json`, `metrics_dstar.json`,
# and the full-size seed-0 checkpoint `dstar_<full>_s0/` (for the optional C5).

# %%
import glob, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

QUICK = True
SIZES = [1_000_000, 4000, 1000]     # 1_000_000 = full; largest first so the long jobs start early
SEEDS = [0, 1, 2]
EPOCHS, BATCH = 3, 8
ENC_LR, TASK_LR = 1e-5, 5e-4
SPLITS = "val,test_seen,test_unseen,test_multi,test_official"
N_PER_SPLIT = 0
if QUICK:
    SIZES, SEEDS, EPOCHS, N_PER_SPLIT = [64], [0, 1], 1, 30

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
WORK, TMP = "/kaggle/working", "/kaggle/tmp"
subprocess.run(f'{sys.executable} -m pip install -q "gliner2[local]"', shell=True, check=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)

import pandas as pd
inst = pd.read_parquet(INST)
N_FULL = int((inst["split"] == "train").sum())
jobs = [(min(n, N_FULL), s) for n in SIZES for s in SEEDS]
KEEP = (min(max(SIZES), N_FULL), 0)   # checkpoint kept in the output
print(f"{len(jobs)} jobs: {jobs}; keep checkpoint {KEEP}")

# %%
gpus = Queue()
for g in ["0", "1"]:
    gpus.put(g)
lock = threading.Lock()
log = {}

def run(cmd, gpu, logfile):
    with open(logfile, "a") as f:
        f.write(f"$ {cmd}\n"); f.flush()
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)).returncode

def job(n, seed):
    gpu = gpus.get()
    tag = f"{n}_s{seed}"
    ckpt = f"{TMP}/dstar_{tag}"
    logfile = f"{WORK}/job_dstar_{tag}.log"
    rec = dict(n=n, seed=seed, gpu=gpu)
    try:
        t0 = time.time()
        for prec in ["fp16", "fp32"]:
            rc = run(f"{sys.executable} -m tooldecide.train_decider --instances {INST} --tools {TOOLS} --n {n} "
                     f"--seed {seed} --out {ckpt} --epochs {EPOCHS} --batch {BATCH} --encoder-lr {ENC_LR} "
                     f"--task-lr {TASK_LR} --precision {prec}", gpu, logfile)
            if rc == 0:
                break
        rec["train_rc"] = rc
        if rc != 0:
            return rec
        rec.update(json.load(open(f"{ckpt}/train_meta.json")), train_wall=time.time() - t0)
        pred = f"{WORK}/preds_dstar_{tag}.parquet"
        t1 = time.time()
        rec["pred_rc"] = run(f"{sys.executable} -m tooldecide.run_decider --instances {INST} --tools {TOOLS} "
                             f"--splits {SPLITS} --n-per-split {N_PER_SPLIT} --model {ckpt} --out {pred}", gpu, logfile)
        rec["pred_wall"] = time.time() - t1
        if rec["pred_rc"] == 0:
            rec["inference"] = json.load(open(pred + ".timing.json"))
        if (n, seed) == KEEP:
            shutil.copytree(ckpt, f"{WORK}/dstar_{tag}", dirs_exist_ok=True)
        shutil.rmtree(ckpt, ignore_errors=True)
        return rec
    finally:
        gpus.put(gpu)
        with lock:
            log[tag] = rec
            json.dump(dict(runs=log, quick=QUICK), open(f"{WORK}/cost_dstar.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] done {tag} on GPU {gpu}: "
              f"train_rc={rec.get('train_rc')} pred_rc={rec.get('pred_rc')} "
              f"train {rec.get('train_seconds', 0) / 60:.1f} min", flush=True)

t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(lambda j: job(*j), jobs))
print(f"all jobs: {(time.time() - t0) / 3600:.2f} h wall")

# %% [markdown]
# ## Metrics per (size, seed)

# %%
from tooldecide import evaluate as E

res = {}
for p in sorted(glob.glob(f"{WORK}/preds_dstar_*.parquet")):
    tag = os.path.basename(p)[len("preds_dstar_"):-len(".parquet")]
    res[tag] = E.score(inst, pd.read_parquet(p))
json.dump(dict(quick=QUICK, metrics=res), open(f"{WORK}/metrics_dstar.json", "w"), indent=2)
pd.set_option("display.width", 250)
print(pd.DataFrame({(t, s): {"acc": d.get("acc"), "macro": d.get("macro_acc"), "f1": d.get("set_f1"),
                             "ece": d.get("ece")} for t, r in res.items() for s, d in r.items()}).T)
failed = [t for t, r in log.items() if r.get("train_rc") != 0 or r.get("pred_rc") != 0]
print("failed jobs:", failed)
assert len(res) == len(jobs), f"only {len(res)}/{len(jobs)} jobs produced predictions"
assert glob.glob(f"{WORK}/dstar_*/*.safetensors") or glob.glob(f"{WORK}/dstar_*/*.bin"), "kept checkpoint missing"
print(sorted(os.listdir(WORK)))
