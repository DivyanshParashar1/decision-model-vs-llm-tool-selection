# Decision Model vs LLM for Tool Selection

Can a small fine-tuned **decision model** pick the right tool for a user request as well as a fine-tuned **8B LLM**, at a
fraction of the cost, including on tools it never saw in training?

- **Decision model:** [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (advertised 340M parameters;
  486M as loaded, about 1/16 the size of the LLM). It scores each candidate label against the text and returns a
  probability for every candidate.
- **LLM:** [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), prompted to answer with a tool name in JSON.

This is the second experiment in a series. In the first ([llm-vs-decision-model-chess](https://github.com/DivyanshParashar1/llm-vs-decision-model-chess)),
the decision model lost when it had to choose among an LLM's candidate moves, because chess needs lookahead. Tool
selection is the opposite case: the options are given and each choice needs little reasoning.

Full results: [reports/RESULTS.md](reports/RESULTS.md). Design: [SPEC.md](SPEC.md).

## The task

Data: [MetaTool](https://github.com/HowieHwong/MetaTool) (about 20k user queries, each labelled with one of 199 tools,
each tool with a text description).

Each test item is a **query plus 10 candidate tools**, and the model must pick the correct one:

- 1 correct tool,
- 4 **look-alike distractors** (the tools whose names and descriptions are most similar, by TF-IDF),
- 5 random tools, in shuffled order.

| Split | Size | What it tests |
|---|---|---|
| train | 10,897 | used for fine-tuning (at most 150 queries per tool) |
| val | 1,365 | sanity check |
| test_seen | 1,365 | new queries for tools seen in training |
| test_unseen | 2,407 | queries for **30 tools held out of training entirely** (known only from their descriptions) |
| test_multi | 497 | queries that need **two** tools; the model must pick the exact pair |
| test_official | 995 | MetaTool's own benchmark prompts and candidate lists |

The splits are built deterministically and their hashes are frozen in [docs/data_manifest.json](docs/data_manifest.json).

## The experiments

| Config | Model | Training |
|---|---|---|
| C1 | Qwen3-8B | none (zero-shot prompt) |
| C2 | GLiNER2.5-Decide | none (zero-shot) |
| C3 | Qwen3-8B | QLoRA (4-bit, rank 16), 1 epoch, 1 seed |
| C4 | GLiNER2.5-Decide | full fine-tune, 3 epochs, 3 seeds |

- **Scaling:** C3 and C4 were each trained on 1k, 4k and all 10,897 examples, using the same training examples for both.
- **Statistics:** 95% bootstrap confidence intervals, plus paired McNemar tests on the same test items.
- **Cost:** training GPU-hours, peak memory and time per decision were recorded for every run.
- **Hardware:** everything ran on free Kaggle T4 GPUs.

## Findings

| Config | Seen tools | Unseen tools | Two-tool requests (exact pair) | Train T4-hours |
|---|---|---|---|---|
| C1 Qwen3-8B, no fine-tuning | 0.862 | 0.786 | 0.662 | 0 |
| C2 GLiNER, no fine-tuning | 0.773 | 0.676 | 0.592 | 0 |
| C3 Qwen3-8B, QLoRA | 0.951 | 0.857 | **0.708** | 2.62 |
| C4 GLiNER, fine-tuned (3 seeds) | **0.979 ± 0.002** | 0.857 ± 0.008 | 0.645 ± 0.014 | 0.86 |

1. **After fine-tuning, the decision model matches or beats the 8B LLM on picking one tool.**
   - On seen tools it wins with every seed (McNemar p ≤ 3e-5).
   - On unseen tools the two tie.
2. **Without fine-tuning, the result reverses.** Raw Qwen beats raw GLiNER by 9–11 points. Fine-tuning gains GLiNER 21
   points on seen tools and Qwen 9.
3. **The LLM wins on two-tool requests.** GLiNER's first pick is usually right (89–90%), but its second is weaker.
   Neither model saw two-tool examples in training.
4. **The decision model is cheaper.**
   - **Training:** about 3× fewer GPU-hours (0.86 vs 2.62).
   - **Inference:** about 110 ms per decision on one T4, against about 500 ms on two T4s for the LLM. Most of the LLM's
     time is the cost of serving an unmerged LoRA adapter; the raw LLM runs at about 170 ms, still on two GPUs.
5. **Most errors are look-alike tools.** For both fine-tuned models, most mistakes pick one of the 4 similar distractors,
   not a random tool. Queries that mention the tool's name do not explain the results.
6. **GLiNER's confidence is useful.** After fine-tuning it is well calibrated (ECE 0.02). Answering only its 90% most
   confident decisions gives 0.994–0.998 accuracy.

Taken with the chess experiment, this suggests a rule: **decision models do well when the options are given and judging
each one needs little reasoning.**

**Limitations:**
- Only one seed for the LLM.
- No hyperparameter search for either model.
- Label noise was not measured.
- Results come from one benchmark with one candidate format.
- 64% of the official prompts' queries also appear in training; RESULTS.md reports the overlap-free subset separately.

## Repository layout

| Path | Contents |
|---|---|
| `src/tooldecide/` | library: instance building, prompts and answer parsing, GLiNER wrapper, training and inference scripts, metrics |
| `kaggle/01–04_*.py` | Kaggle notebooks: data, baselines (C1, C2), LLM fine-tune (C3), GLiNER fine-tune (C4) |
| `tools/` | Kaggle CLI helpers (`kpush`, `kwait`, `push_lib`), `report.py` (builds all tables), secrets check |
| `tests/` | unit tests for the pure functions |
| `reports/` | `RESULTS.md` (write-up), `tables.md` and `results.json` (generated) |

Rebuild the tables from the prediction files:
```bash
python tools/report.py
```

Agents working on this repo: start with [CLAUDE.md](CLAUDE.md) (plan, Kaggle workflow, rules).

## Setup on a new machine

```bash
cp .env.example .env            # then fill in the two Kaggle tokens (never commit .env)
bash tools/install_hooks.sh     # pre-commit check that blocks tokens from being committed
pip install kaggle six          # Kaggle CLI (tools read tokens from .env)
```
