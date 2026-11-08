# Decision Model vs LLM for Tool Selection: Spec

> Follow-up to the chess experiment (`llm-vs-decision-model-chess`). Audience: whoever runs or reviews this project.

## 1. Why this domain

In chess, the decision model lost to a fine-tuned LLM when it had to pick from an LLM's candidates, because
(a) good moves need lookahead and (b) the LLM's candidate list missed the best move 53% of the time. Used on its own over
all legal moves, the decider nearly matched the fine-tuned 8B LLM.

Tool selection removes both handicaps: **the options are given** (the tool list) and **the decision is shallow**
(understand the request and match it to a tool description). It is also exactly what decision models are marketed for,
and what production agents do on every turn.

**Hypothesis.** A fine-tuned GLiNER2.5-Decide (486M) picking a tool from a list matches or beats a QLoRA fine-tuned
Qwen3-8B on accuracy, at a fraction of training and inference cost, including on tools never seen in training.

## 2. Data: MetaTool (ICLR 2024, MIT license)

Source: `github.com/HowieHwong/MetaTool`
- `dataset/data/all_clean_data.csv`: 20,614 (query, tool) pairs over 199 tools.
- `dataset/plugin_des.json`: one description per tool.
- `dataset/data/multi_tool_query_golden.json`: 497 queries that need 2 tools.
- `dataset/tmp_dataset/Task2-Subtask*.json`: official tool-selection test prompts (for comparability with published numbers).

### 2.1 Building decision instances
Each instance = (query, k = 10 candidate tools with descriptions, gold tool). Distractors:
- 4 **hard**: tools whose descriptions are most similar to the gold tool's (TF-IDF cosine; deterministic, no model).
- 5 **random** from the remaining tools.
- Candidate order shuffled with a fixed seed (no position bias).

Per-tool cap: at most 150 queries per tool in train (the raw data ranges from 12 to 1,122 per tool), so frequent tools don't dominate.
Near-duplicate queries (same normalized text) are removed.

### 2.2 Splits (leakage rules)
- **Unseen tools**: 30 of the 199 tools (stratified by frequency) are held out entirely. Their queries appear only in
  `test_unseen`, and they are never used as distractors in train/val. This tests generalization from descriptions alone.
- The remaining 169 tools: queries split 80/10/10 into train/val/`test_seen`, stratified by tool.
- `test_multi`: the 497 two-tool queries (multi-label; gold = both tools present among 10 candidates).
- `test_official`: MetaTool Task 2 subtask files, scored as in the paper, for comparability.
- Split ids and hashes are frozen in `docs/data_manifest.json`.

## 3. Configurations

| ID | Config | Notes |
|---|---|---|
| C1 | Qwen3-8B raw | prompt = query + numbered tool list; answer JSON `{"tool": name}` |
| C2 | GLiNER2.5-Decide raw | labels = tool names, label descriptions = tool descriptions |
| C3 | Qwen3-8B QLoRA fine-tuned | same prompt; target = gold tool |
| C4 | GLiNER2.5-Decide fine-tuned | same instances as C3 |
| C5 (optional) | LLM shortlist (top-3) → D\* | only if C4 < C3 |

No proposer step is needed this time: the candidate list comes from the instance itself.

## 4. Metrics
- **Accuracy** (top-1 tool correct), per split; **macro accuracy** over tools (no frequency bias).
- **Hard-distractor confusion rate**: fraction of errors where the pick was one of the 4 similar tools.
- **Unseen-tool accuracy** (`test_unseen`): the generalization headline.
- **Multi-tool**: exact-set match and set F1 (LLM outputs a list; decider uses multi-label scores, top-2).
- **Calibration** (decider): expected calibration error (ECE) of the confidence; accuracy at 90% coverage (abstain on low confidence).
- **Latency** per decision (T4), and **training GPU-hours**.
- Stats: bootstrap 95% CIs; paired McNemar for C4 vs C3, C4 vs C1, C2 vs C1.

## 5. Training and the scaling curve
Both arms trained on {1k, 4k, full} train instances (same instance ids for both arms).
- D\*: full fine-tune, 3 epochs, 3 seeds (cheap).
- LLM\*: QLoRA r16 a32, 1 epoch, 1 seed per size (2nd Kaggle account for the full-size run if needed).
- Logged: GPU type, trainable params, wall-clock, peak VRAM.

## 6. Compute plan (free Kaggle; priority Kaggle > Colab > Mac)
| Notebook | Accel | Est. GPU-h |
|---|---|---|
| 01 data build + MetaTool official sets | CPU | 0 |
| 02 raw baselines C1 + C2 (all test splits) | T4×2 | ~0.7 |
| 03 LLM\* QLoRA at 1k / 4k / full | T4 | ~4–5 |
| 04 D\* at 1k / 4k / full × 3 seeds | T4 | ~2 |
| 05 final eval C3 / C4 (+C5) + report | T4×2 | ~1 |

## 7. Risks
- **Label noise / vague queries** in MetaTool (e.g. "How can I make this immersive?"): report an estimated noise floor
  by manually checking 100 random val errors made by *both* arms.
- **Description quality** drives unseen-tool accuracy for both arms; descriptions are used verbatim.
- **Tool-name leakage**: some queries contain the tool's name. Report accuracy with and without those queries.
