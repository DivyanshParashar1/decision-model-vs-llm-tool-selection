"""Train one D* (GLiNER2.5-Decide) in a fresh process.

  python -m tooldecide.train_decider --instances instances.parquet --tools tools.parquet --n 4000 --seed 0 \
      --out /kaggle/tmp/dstar_4000_s0 [--epochs 3] [--precision fp16|fp32]

Train set = train instances with train_rank < n (the same ids the LLM arm uses). Runs in its own process so
AMP/optimizer state never leaks between runs. Verifies that weights changed; exits with code 2 if training was a
no-op (e.g. fp16 overflow) so the caller can retry in fp32. Writes <out>/train_meta.json.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import shutil
import sys
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="fastino/GLiNER2.5-Decide")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=5e-4)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=300)
    a = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.data import Classification, InputExample, TrainingDataset
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from .decider import TASK, render_labels, render_text

    inst = pd.read_parquet(a.instances)
    desc = dict(pd.read_parquet(a.tools)[["tool", "description"]].itertuples(index=False))
    train = inst[(inst["split"] == "train") & (inst["train_rank"] < a.n)].sort_values("train_rank")
    val = inst[inst["split"] == "val"].sort_values("id").head(a.n_val)

    def to_examples(df):
        exs = []
        for r in df.itertuples(index=False):
            cands = json.loads(r.cands)
            labs = render_labels(cands, desc)
            exs.append(InputExample(text=render_text(r.query), classifications=[
                Classification(task=TASK, labels=cands, true_label=json.loads(r.gold)[0], label_descriptions=labs)]))
        return exs

    torch.manual_seed(a.seed)
    model = AutoExtractor.from_pretrained(a.base).to("cuda")
    ckpt_dir = a.out + "_ckpt"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    want = dict(output_dir=ckpt_dir, experiment_name=f"dstar_{a.n}_s{a.seed}", num_epochs=a.epochs,
                batch_size=a.batch, eval_batch_size=16, encoder_lr=a.encoder_lr, task_lr=a.task_lr, seed=a.seed,
                fp16=a.precision == "fp16", bf16=False, logging_steps=50, report_to="none")
    params = inspect.signature(TrainingConfig).parameters
    cfg = TrainingConfig(**{k: v for k, v in want.items() if k in params})
    trainer = GLiNER2Trainer(model=model, config=cfg, train_data=TrainingDataset(to_examples(train)),
                             eval_data=TrainingDataset(to_examples(val)))

    name0, p0 = next((k, v.detach().float().cpu().clone()) for k, v in model.named_parameters() if v.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trained = getattr(trainer, "model", model)
    trained = getattr(trained, "module", trained)
    w = dict(trained.named_parameters())[name0].detach().float().cpu()
    delta = (w - p0).abs().max().item()
    finite = bool(torch.isfinite(w).all())
    print(f"[train_decider] n={len(train)} seed={a.seed} precision={a.precision} {secs / 60:.1f} min, "
          f"max|Δ|={delta:.3e}, finite={finite}", flush=True)
    if not finite or not delta > 0 or math.isnan(delta):
        print("[train_decider] training was a no-op or diverged", flush=True)
        sys.exit(2)

    os.makedirs(a.out, exist_ok=True)
    finals = sorted(glob.glob(f"{ckpt_dir}/**/final", recursive=True))
    if finals:
        shutil.copytree(finals[0], a.out, dirs_exist_ok=True)
    else:
        trained.save_pretrained(a.out)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    meta = dict(n_train=len(train), seed=a.seed, epochs=a.epochs, batch=a.batch, precision=a.precision,
                train_seconds=secs, peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9, weight_delta=delta,
                gpu=torch.cuda.get_device_name(0), params=sum(p.numel() for p in trained.parameters()),
                trainable_params=sum(p.numel() for p in trained.parameters() if p.requires_grad),
                used_final_checkpoint=bool(finals))
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=2)
    print(f"[train_decider] saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
