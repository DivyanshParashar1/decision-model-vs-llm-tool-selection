# Decision Model vs LLM for Tool Selection

Can a fine-tuned **decision model** (GLiNER2.5-Decide, advertised as 340M parameters) choose the right tool for a user
request as well as a **fine-tuned 8B LLM** (Qwen3-8B), at a fraction of the cost, including tools it never saw in training?

Follow-up to [llm-vs-decision-model-chess](https://github.com/DivyanshParashar1/llm-vs-decision-model-chess), where the
decision model alone nearly matched a fine-tuned 8B LLM on move quality. Design: [SPEC.md](SPEC.md).

Status: setting up.
