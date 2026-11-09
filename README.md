# Decision Model vs LLM for Tool Selection

Can a fine-tuned **decision model** (GLiNER2.5-Decide, advertised as 340M parameters) choose the right tool for a user
request as well as a **fine-tuned 8B LLM** (Qwen3-8B), at a fraction of the cost, including tools it never saw in training?

Follow-up to [llm-vs-decision-model-chess](https://github.com/DivyanshParashar1/llm-vs-decision-model-chess), where the
decision model alone nearly matched a fine-tuned 8B LLM on move quality. Design: [SPEC.md](SPEC.md).

## Findings

Full write-up: [reports/RESULTS.md](reports/RESULTS.md). MetaTool, 10 candidate tools per request, 30 tools held out of training.

| Config | Seen tools | Unseen tools | Two-tool requests (exact pair) | Train T4-hours |
|---|---|---|---|---|
| Qwen3-8B, no fine-tuning (C1) | 0.862 | 0.786 | 0.662 | 0 |
| GLiNER2.5-Decide, no fine-tuning (C2) | 0.773 | 0.676 | 0.592 | 0 |
| Qwen3-8B, QLoRA (C3) | 0.951 | 0.857 | **0.708** | 2.62 |
| GLiNER2.5-Decide, fine-tuned (C4, 3 seeds) | **0.979 ± 0.002** | 0.857 ± 0.008 | 0.645 ± 0.014 | 0.86 |

- After fine-tuning, the decision model (advertised 340M; 486M as loaded) **beats the fine-tuned 8B LLM on seen tools,
  ties on unseen tools, and loses on two-tool requests**. It used about 3× fewer training GPU-hours and is about 4×
  faster per decision on half the GPUs.
- Without fine-tuning, the LLM is clearly better. The decision model only becomes competitive after fine-tuning.
- Main limitations: one seed for the LLM arm, no hyperparameter search, label noise not measured.

**Agents: start with [CLAUDE.md](CLAUDE.md)** (plan, Kaggle workflow, rules).

## Setup on a new machine

```bash
cp .env.example .env            # then fill in the two Kaggle tokens (never commit .env)
bash tools/install_hooks.sh     # pre-commit check that blocks tokens from being committed
pip install kaggle six          # Kaggle CLI (tools read tokens from .env)
```
