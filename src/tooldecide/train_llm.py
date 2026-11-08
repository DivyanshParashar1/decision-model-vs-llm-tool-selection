"""QLoRA fine-tune Qwen3-8B (LLM*, the expensive arm) with Unsloth, in a fresh process.

  python -m tooldecide.train_llm --instances instances.parquet --tools tools.parquet --n 4000 \
      --out /kaggle/working/llm_star_4000 [--max-steps 20]

Train set = train instances with train_rank < n (same ids as D*). Prompt = prompts.single_messages, target =
{"tool": gold}; loss on assistant tokens only. Writes the LoRA adapter and <out>/train_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-h", type=float, default=10.0)
    a = ap.parse_args()

    import pandas as pd
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from .prompts import single_messages, target

    inst = pd.read_parquet(a.instances)
    desc = dict(pd.read_parquet(a.tools)[["tool", "description"]].itertuples(index=False))
    train = inst[(inst["split"] == "train") & (inst["train_rank"] < a.n)].sort_values("train_rank")

    model, tok = FastLanguageModel.from_pretrained(a.base, max_seq_length=a.max_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    texts = []
    for r in train.itertuples(index=False):
        msgs = single_messages(r.query, json.loads(r.cands), desc)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        texts.append(prompt + target(json.loads(r.gold)) + "<|im_end|>\n")
    lens = [len(tok(t)["input_ids"]) for t in texts[:500]]
    n_tokens = int(sum(lens) / len(lens) * len(texts))
    print(f"{len(texts)} examples, ~{n_tokens / 1e6:.2f}M tokens, max len (first 500) {max(lens)}", flush=True)
    print(texts[0][-500:], flush=True)

    class TimeBudget(TrainerCallback):
        def __init__(self, hours):
            self.t0, self.limit = time.time(), hours * 3600

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.t0 > self.limit:
                print("time budget hit: stopping early", flush=True)
                control.should_training_stop = True

    cfg = SFTConfig(
        output_dir=a.out + "_ckpt", dataset_text_field="text", max_seq_length=a.max_len,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=a.warmup_steps,
        logging_steps=10, fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.0, seed=a.seed, save_strategy="no", report_to="none", packing=False)
    ds = Dataset.from_dict({"text": texts})
    cb = [TimeBudget(a.time_budget_h)]
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg, callbacks=cb)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg, callbacks=cb)
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train()
    secs = time.time() - t0
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    meta = dict(arm="LLM*", base=a.base, method="QLoRA r16 a32 (Unsloth)", gpu=torch.cuda.get_device_name(0),
                n_gpus_used=1, trainable_params=trainable, n_train=len(train), approx_train_tokens=n_tokens,
                epochs=a.epochs, steps=trainer.state.global_step, batch=a.batch, accum=a.accum, lr=a.lr,
                train_seconds=secs, peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                final_loss=stats.training_loss, log_history=trainer.state.log_history[-20:])
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=2)
    print(f"[train_llm] {secs / 3600:.2f} h, loss {stats.training_loss:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
