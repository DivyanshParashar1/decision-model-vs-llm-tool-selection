# Agent brief: Decision Model vs LLM for Tool Selection

Read this whole file before doing anything. It is the hand-off from the session that ran the chess experiment and set up
this repo. The experiment design is in [SPEC.md](SPEC.md); this file covers status, the plan, the Kaggle workflow,
and the lessons already paid for.

## 1. What this project is

Second experiment in a series. Question: can a fine-tuned **decision model** (GLiNER2.5-Decide, advertised 340M,
486M parameters as loaded) pick the right tool for a user request from a list of 10 candidates as well as a
**QLoRA fine-tuned Qwen3-8B**, at a fraction of the cost, including on tools never seen in training?

**Previous experiment** (chess): repo `DivyanshParashar1/llm-vs-decision-model-chess`, local copy at
`../exp1 - LLM vs decision model for chess/`. Result: raw LLM + fine-tuned decider (C4) lost to the fine-tuned LLM
(26.8% vs 36.1% top-1). But the fine-tuned decider used *alone* over all legal moves matched the fine-tuned 8B on
win-prob loss (0.261 vs 0.261) and blunder rate, with about a quarter of the training GPU-hours. The chess loss came from
(a) needing lookahead and (b) the LLM's candidate list missing the best move 53% of the time. Tool selection removes
both: the options are given and the decision is shallow. See `../exp1 .../reports/RESULTS.md`.

## 2. The user: preferences and rules (follow strictly)

- **Commits:** many small, step-wise commits (one logical change each). **Never** add a Claude signature,
  `Co-Authored-By` or "Generated with" line to commits or PRs. Push after each group of commits.
- **Compute priority: Kaggle > Colab > Mac for every task.** The Mac is an M2 with 8 GB RAM and little free disk.
  Never run models on it; only light scripts, tests and report generation.
- **Free tiers only.** No paid compute.
- **You write all the code.** The user runs nothing by hand unless asked. Push notebooks to Kaggle yourself with the CLI.
- Ask questions and give suggestions before big design changes; give recommendations, not surveys.
- Report results honestly, including failures and negative results. Never inflate claims. The parameter count is
  "advertised 340M; 486M as loaded".
- The user wants to be **taught** the project afterwards and will put it on their resume. Keep the code readable and
  the reports accurate.
- Use they/them for the user.

## 3. Current status (as of 2026-11-09, 12:30 IST)

Done:
- [SPEC.md](SPEC.md): full design (MetaTool data, k = 10 candidates = 4 TF-IDF look-alikes + 5 random, 30 unseen tools,
  configs C1–C4 (+C5), scaling 1k/4k/full, metrics).
- `tools/`: Kaggle helpers with two-account support (see section 5).
- `.env` holds both Kaggle tokens (gitignored); `.env.example` has placeholders.
- GitHub repo `DivyanshParashar1/decision-model-vs-llm-tool-selection` (private).
- MetaTool data inspected: `all_clean_data.csv` = 20,614 (Query, Tool) rows, 199 tools, 12–1,122 queries per tool
  (16,082 after capping at 150 per tool); `plugin_des.json` = {tool: description}; `multi_tool_query_golden.json` =
  497 items `{query, tool: [a, b]}`; `tmp_dataset/Task2-Subtask1.json` = 995 official prompts.

- Account 2 verified (2026-11-08 22:08 IST): smoke notebook `divyanshparashar12/tools-acc2-smoke` saw 2× Tesla T4 and
  internet. (Its first run, before phone verification took effect, had neither.)

- Library `src/tooldecide/` written: `instances`, `prompts`, `metrics`, `evaluate` (shared per-split scorer),
  `decider`, `gen_llm`, `run_decider`, `train_decider`, `train_llm`. 15 unit tests pass.
- **Step 1 done** (2026-11-08): `tools-01-data` ran on both accounts; split hashes identical (and identical to a local
  build), frozen in `docs/data_manifest.json`. Sizes: train 10,897 / val 1,365 / test_seen 1,365 / test_unseen 2,407 /
  test_multi 497 / test_official 995. 64% of official queries also appear in train, so report official accuracy on the
  `query_in_train == False` subset too (the scorer does this).
- Notebooks 02, 03, 04 written; first QUICK smoke runs pushed (02 + 04 on main, 03 on acc2).

Decider scoring detail: a classification schema with `multi_label=True, cls_threshold=0, class_act="softmax"`
returns a softmax probability for every candidate (verified in gliner2 2.1.0 source), and `batch_extract` takes one
schema per text, so instances with different tool lists batch together.

- **Steps 2–5 done** (2026-11-09). Full runs: main `tools-02-baselines`, `tools-04-dstar` (9 jobs, all fp16 OK);
  acc2 `tools-03-llm-star-1k`, `-4k`, `-full`. Small outputs (json/log/parquet) are under `build/<slug>/out`
  (gitignored). `python tools/report.py` rebuilds instances locally (hash-checked), writes `reports/tables.md` and
  `reports/results.json`; the narrative is `reports/RESULTS.md`, and the findings are in the README.
- Headline: fine-tuned GLiNER beats QLoRA Qwen3-8B on seen tools (0.979 vs 0.951), ties on unseen (0.857 vs 0.857), loses
  on two-tool requests (0.645 vs 0.708), with ~3× fewer training GPU-hours.
- Not done (optional): C5, a second LLM\* seed, the manual label-noise check (SPEC §7), merged-LoRA latency for C3.
- Two failed slugs from a launch bug sit on acc2 (`tools-03-llm-star-1k-1000`, `tools-03-llm-star-full-1000000`); ignore them.
- Lessons from this run: zsh does not word-split `$var`, so never loop over "a b" pairs with `set -- $cfg`. vLLM with an
  unmerged LoRA is ~3× slower than the base model on T4. GLiNER training peaks at 15.4 GB on a T4 at batch 8, so don't
  raise the batch. `kernels logs` is empty while a notebook runs.

## 4. The plan (do in order; each step has a "done when")

Code layout to create, mirroring the chess repo (copy and adapt from `../exp1 .../src/chessdecide/` and `../exp1 .../kaggle/`):
- `src/tooldecide/`: `instances.py` (build k-candidate instances), `prompts.py` (LLM prompt + robust JSON parse),
  `decider.py` (GLiNER wrapper), `gen_llm.py` (vLLM/HF batch generation, LoRA support), `train_decider.py`
  (one D\* per process), `metrics.py` (accuracy, macro accuracy, set F1, ECE, bootstrap CI, McNemar).
- `kaggle/NN_*.py` with `# %%` cells; build `.ipynb` with `tools/py2nb.py`.
- `tests/`: unit tests for every pure function (run locally with `uv run --python 3.11 --with ... pytest`).

| Step | Notebook | Account | Accel | Done when |
|---|---|---|---|---|
| 1 | `01_data`: download MetaTool from GitHub raw URLs, dedupe, cap 150/tool, hold out 30 tools (stratified by frequency), build instances (seeded), splits, `test_multi`, `test_official`, tool-name-leak flag | **both** (deterministic) | CPU | split hashes in `meta.json` are **identical** on both accounts; written to `docs/data_manifest.json` |
| 2 | `02_baselines`: C1 (raw Qwen3-8B via vLLM) + C2 (raw GLiNER) on val, test_seen, test_unseen, test_multi, test_official | main | T4×2 | per-instance predictions parquet + metrics json |
| 3 | `03_llm_star`: QLoRA Qwen3-8B at 1k, 4k, full; after each size, generate predictions on all test splits with the adapter and save them | **acc2** | T4 | 3 adapters' predictions + `cost_llm_star_<n>.json` |
| 4 | `04_dstar`: GLiNER fine-tune at 1k, 4k, full × 3 seeds, each in its own subprocess; predictions on all test splits | main | T4 | predictions + costs per (size, seed) |
| 5 | Local (Mac, light): fetch only small prediction/metric files from both accounts, compute all metrics + stats, write `reports/RESULTS.md` and a README findings section | n/a | n/a | report committed and pushed |

Run steps 2 and 3 in parallel (different accounts). Use QUICK smoke runs (tiny sizes) first for every new notebook.
Private outputs are **not** visible across accounts, which is why step 1 runs on both and each account saves its own predictions.

**GPU budget** (quota resets 14 Nov 00:00 UTC): main has ~12.3 h left, acc2 has 30 h. Estimates: step 2 ≈ 0.7 h,
step 3 ≈ 7 h (full size ≈ 4.8 h), step 4 ≈ 2 h. Check with `kaggle quota` (add the account token for acc2, see below).

## 5. Kaggle workflow (verified in the chess project)

Accounts:
| Name in tools | Kaggle username | Token |
|---|---|---|
| `main` | `divyanshparashar1` | `KAGGLE_TOKEN_MAIN` in `.env` |
| `acc2` | `divyanshparashar12` (23cd3015@rgipt.ac.in) | `KAGGLE_TOKEN_ACC2` in `.env` |

Commands (run from the repo root):
```bash
python tools/push_lib.py "msg" [--account=acc2]      # publish src/tooldecide as private dataset <user>/tooldecide-lib
python tools/kpush.py kaggle/01_data.py --accel none --account main [--set QUICK=False] \
    [--input <user>/tools-01-data] [--dataset <user>/tooldecide-lib]
python tools/kwait.py tools-01-data [--account acc2] --fetch   # waits; exit 0 ok, 1 error, 3 login expired
```
- `kpush` builds the notebook, writes `kernel-metadata.json` under `build/<slug>/` and pushes. Slugs are `tools-<file stem>`.
  `--accel`: `none` | `t4x2` | `p100` (never use p100: vLLM does not support it).
- Raw CLI as acc2: `KAGGLE_API_TOKEN=$(grep KAGGLE_TOKEN_ACC2 .env | cut -d= -f2) kaggle quota`.
- Run `kwait` with `run_in_background` and wait for the notification; don't poll.

## 6. Secrets: hard rules

- Tokens live **only** in `.env` (gitignored) and `~/.kaggle/`. **Never** write a token value into any markdown, notebook,
  script, commit message or log you commit. Refer to tokens by variable name only.
- **Before every push** run `bash tools/check_secrets.sh` (also installed as a git pre-commit hook). It fails if anything
  staged or committed contains `KGAT_`. If it ever fires: unstage, remove the value, and tell the user.
- Notebooks pushed to Kaggle must not contain tokens either. Kaggle notebooks never need Kaggle tokens.
- Both tokens were pasted in chat once. Remind the user to revoke and regenerate them (Kaggle → Settings → API) when the
  project is finished, then update `.env`.

## 7. Lessons from the chess run (each cost real time; don't repeat them)

1. **Never download full kernel outputs to the Mac.** Always `--file-pattern '.*\.(json|log|csv|parquet)$'`
   (`kwait --fetch` already does json/log/csv). A GLiNER checkpoint is 1.9 GB.
2. **Re-run `push_lib.py` after any change in `src/`** before pushing a notebook that imports it.
3. The lib dataset mounts **flat** on Kaggle (files at root, no package folder). Notebooks must find `prompts.py` (or any module)
   with a recursive glob under `/kaggle/input` and symlink its directory to `/kaggle/tmp/lib/tooldecide`, then add
   `/kaggle/tmp/lib` to `sys.path` and `PYTHONPATH` (for subprocesses). Copy the pattern from `../exp1 .../kaggle/02_llm0.py`.
4. Input mounts land under `/kaggle/input/notebooks/<user>/<slug>/...`; always locate files with recursive glob.
5. **GLiNER training: one run per process** (`python -m tooldecide.train_decider ...`). A second run in the same process
   silently did nothing (fp16 overflow, loss 0.0000). Assert weights changed (max |Δ| > 0) and retry in fp32 if not.
   Copy the trainer's `final` checkpoint from its output_dir; don't trust `model.save_pretrained` on the in-memory model.
6. Cleanup code that deletes checkpoints must compare the **actual** train size (`min(n, len(train))`), not the requested one.
   Assert a checkpoint remains at the end.
7. vLLM: pin `vllm==0.33.0` (works on T4 with the TRITON_ATTN backend), `dtype="half"`, `tensor_parallel_size=2`. Run
   generation as a **subprocess** so the GPU is freed before GLiNER runs. LoRA via `enable_lora=True` + `LoRARequest`.
8. Unsloth QLoRA on one T4: ~20 s/step at batch 8 × accumulation 2 with ~400-token prompts. Use `warmup_steps`, not
   `warmup_ratio`. Base `unsloth/Qwen3-8B-unsloth-bnb-4bit`; apply the adapter to `Qwen/Qwen3-8B` at inference.
   Loss on assistant tokens only via `train_on_responses_only`. Qwen3: `enable_thinking=False` in `apply_chat_template`.
9. Two GPU notebooks can run at the same time on one account; each bills quota separately.
10. Kaggle OAuth logins expire after about 2 h; the API tokens in `.env` don't. `kwait` exits with code 3 on auth failure.
11. Save results after every sub-run so a crash keeps partial results.
12. Every new notebook gets a QUICK smoke run first. That caught real bugs every time.

## 8. Verified GLiNER2.5-Decide API (gliner2 2.1.0)

- Load: `from gliner2 import AutoExtractor; m = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")`.
- Classify: `m.classify_text(text, {"tool": {name: description, ...}}, include_confidence=True)` returns
  `{'tool': {'label': ..., 'confidence': ...}}`. `format_results=False` returns only `(label, score)` for the top label
  (no all-label scores). Batch: `batch_classify_text(texts, tasks, batch_size=8, ...)`. Multi-label: `{"labels": [...], "multi_label": True}`.
- Encoder: DeBERTa-v3-large, 512 positions with relative attention (longer inputs run). 44 labels per call worked.
- Train: `gliner2.training.trainer.GLiNER2Trainer(model, TrainingConfig(output_dir, num_epochs, batch_size, encoder_lr=1e-5,
  task_lr=5e-4, fp16, ...), train_data=TrainingDataset([...]), eval_data=...)`; examples are
  `InputExample(text, classifications=[Classification(task, labels, true_label, label_descriptions)])`.
  ~10 examples/s on a T4 at batch 8; peak ~14 GB.
- Speed: ~100–200 ms per decision on a T4.

## 9. Definition of done

`reports/RESULTS.md` with: headline table (C1–C4, seen vs unseen tools, CIs), McNemar for C4 vs C3 / C4 vs C1 / C2 vs C1,
scaling curves for both arms, cost table (GPU-hours, params, latency), hard-distractor and tool-name-leak analysis,
limitations. README findings section. Everything committed in small commits and pushed, with `check_secrets.sh` clean.
