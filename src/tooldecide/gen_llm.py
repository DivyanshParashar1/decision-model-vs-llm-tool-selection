"""Batch-generate LLM tool choices for instances (raw Qwen3-8B, or with a LoRA adapter).

Run as a subprocess so the GPU is fully released afterwards:
  python -m tooldecide.gen_llm --instances instances.parquet --tools tools.parquet --out preds.parquet \
      --splits val,test_seen,test_unseen,test_multi,test_official [--lora adapter/] [--n-per-split 50]

Writes: id, split, raw, pred (JSON list, best first), conf (NaN), prompt_version, model, lora,
plus <out>.timing.json with generation seconds.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .prompts import PROMPT_VERSION, messages, parse_tools

MAX_TOK = {1: 48, 2: 80}


def select(inst: pd.DataFrame, splits: list[str], n_per_split: int | None) -> pd.DataFrame:
    d = inst[inst["split"].isin(splits)]
    if n_per_split:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sort_values("id").head(n_per_split))
    return d.reset_index(drop=True)


class VLLMBackend:
    def __init__(self, model, tp, lora, max_len):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="half", tensor_parallel_size=tp, max_model_len=max_len,
                       gpu_memory_utilization=0.90, seed=0, enable_lora=bool(lora), max_lora_rank=64,
                       enable_prefix_caching=True)
        self.lora = None
        if lora:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("ft", 1, lora)

    def generate(self, prompts, max_toks):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.0, max_tokens=m, seed=0) for m in max_toks]
        outs = self.llm.generate(prompts, sps, lora_request=self.lora, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    def __init__(self, model, lora, bs=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        self.bs = bs

    def generate(self, prompts, max_toks):
        from tqdm.auto import tqdm
        out = [None] * len(prompts)
        for mt in sorted(set(max_toks)):
            idx = sorted((i for i, m in enumerate(max_toks) if m == mt), key=lambda i: len(prompts[i]))
            for s in tqdm(range(0, len(idx), self.bs), desc=f"hf gen (max {mt})"):
                b = idx[s:s + self.bs]
                enc = self.tok([prompts[i] for i in b], return_tensors="pt", padding=True).to(self.model.device)
                with self.torch.no_grad():
                    g = self.model.generate(**enc, max_new_tokens=mt, do_sample=False,
                                            pad_token_id=self.tok.pad_token_id)
                for i, t in zip(b, self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    out[i] = t
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="val,test_seen,test_unseen,test_multi,test_official")
    ap.add_argument("--n-per-split", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()

    inst = select(pd.read_parquet(a.instances), a.splits.split(","), a.n_per_split or None)
    desc = dict(pd.read_parquet(a.tools)[["tool", "description"]].itertuples(index=False))
    print(f"{len(inst):,} instances: {inst['split'].value_counts().to_dict()}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    golds = inst["gold"].map(json.loads)
    cands = inst["cands"].map(json.loads)
    prompts = [tok.apply_chat_template(messages(q, c, desc, len(g)), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for q, c, g in zip(inst["query"], cands, golds)]
    print(prompts[0], flush=True)

    backend = None
    if a.backend in ("auto", "vllm"):
        try:
            backend = VLLMBackend(a.model, a.tp, a.lora, a.max_len)
            print("backend: vllm", flush=True)
        except Exception as e:  # noqa: BLE001
            if a.backend == "vllm":
                raise
            print(f"vLLM unavailable ({e!r}); falling back to HF transformers", flush=True)
    if backend is None:
        backend = HFBackend(a.model, a.lora)
        print("backend: hf", flush=True)

    t0 = time.time()
    texts = backend.generate(prompts, [MAX_TOK[min(len(g), 2)] for g in golds])
    secs = time.time() - t0
    out = inst[["id", "split"]].copy()
    out["raw"] = texts
    out["pred"] = [json.dumps(parse_tools(t, c, len(g))) for t, c, g in zip(texts, cands, golds)]
    out["conf"] = np.nan
    out["prompt_version"], out["model"], out["lora"] = PROMPT_VERSION, a.model, a.lora or ""
    out.to_parquet(a.out, index=False)
    timing = dict(n=len(out), gen_seconds=secs, ms_per_decision_batched=1000 * secs / max(1, len(out)),
                  backend=type(backend).__name__, tp=a.tp, model=a.model, lora=a.lora or "")
    json.dump(timing, open(a.out + ".timing.json", "w"), indent=2)
    print(f"{len(out)} generations in {secs:.0f}s; parse rate {out['pred'].map(lambda s: s != '[]').mean():.1%}",
          flush=True)


if __name__ == "__main__":
    main()
