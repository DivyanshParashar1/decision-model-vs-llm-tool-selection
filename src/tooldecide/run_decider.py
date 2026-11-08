"""Run a GLiNER decider (base or fine-tuned) over instances, as a subprocess.

  python -m tooldecide.run_decider --instances instances.parquet --tools tools.parquet --out preds.parquet \
      [--model fastino/GLiNER2.5-Decide | path/to/dstar] [--splits ...] [--n-per-split 50]

Writes: id, split, pred (JSON list ranked best first: top-1 for single, top-2 for multi), conf (top-1 prob),
scores (JSON {tool: prob}), model; plus <out>.timing.json (batched ms/decision and unbatched latency).
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .decider import GlinerDecider, ranked
from .gen_llm import select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="fastino/GLiNER2.5-Decide")
    ap.add_argument("--splits", default="val,test_seen,test_unseen,test_multi,test_official")
    ap.add_argument("--n-per-split", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--latency-n", type=int, default=100, help="instances timed one at a time")
    a = ap.parse_args()

    import torch

    inst = select(pd.read_parquet(a.instances), a.splits.split(","), a.n_per_split or None)
    desc = dict(pd.read_parquet(a.tools)[["tool", "description"]].itertuples(index=False))
    cands = list(inst["cands"].map(json.loads))
    n_gold = list(inst["gold"].map(lambda s: len(json.loads(s))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = GlinerDecider(a.model, device=device)

    # unbatched latency (what one agent turn would pay)
    lat = []
    for q, c in list(zip(inst["query"], cands))[:a.latency_n]:
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        dec.score_batch([q], [c], desc, batch_size=1)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append(time.time() - t)

    t0 = time.time()
    scores = []
    for s in range(0, len(inst), 512):
        scores += dec.score_batch(list(inst["query"][s:s + 512]), cands[s:s + 512], desc, batch_size=a.batch)
        print(f"  {len(scores)}/{len(inst)}", flush=True)
    secs = time.time() - t0

    out = inst[["id", "split"]].copy()
    rk = [ranked(sc, c) for sc, c in zip(scores, cands)]
    out["pred"] = [json.dumps(r[:n]) for r, n in zip(rk, n_gold)]
    out["conf"] = [sc.get(r[0], float("nan")) for sc, r in zip(scores, rk)]
    out["scores"] = [json.dumps(sc) for sc in scores]
    out["model"] = a.model
    out.to_parquet(a.out, index=False)
    lat_sorted = sorted(lat)
    timing = dict(n=len(out), seconds=secs, ms_per_decision_batched=1000 * secs / max(1, len(out)),
                  latency_ms_median=1000 * lat_sorted[len(lat) // 2] if lat else None,
                  latency_ms_p90=1000 * lat_sorted[int(0.9 * (len(lat) - 1))] if lat else None,
                  device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu", batch=a.batch, model=a.model)
    json.dump(timing, open(a.out + ".timing.json", "w"), indent=2)
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
