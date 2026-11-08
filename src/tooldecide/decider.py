"""GLiNER2.5-Decide wrapper: input rendering (shared by training and eval) and batched scoring.

Scoring uses a classification schema with multi_label=True, cls_threshold=0 and class_act="softmax":
gliner2 then returns (label, probability) for EVERY candidate, with probabilities identical to the
single-label softmax. That gives top-1, top-2 (multi-tool) and a confidence for calibration in one pass.
`batch_extract` accepts one schema per text, so instances with different tool lists batch together.
"""
from __future__ import annotations

RENDER_VERSION = "r1"
TASK = "tool"


def render_text(query: str) -> str:
    return query.strip()


def render_labels(cands: list[str], desc: dict[str, str]) -> dict[str, str]:
    """{tool name: description}; GLiNER encodes each label together with its description."""
    return {t: desc[t] for t in cands}


def task_config(cands: list[str], desc: dict[str, str]) -> dict:
    return {TASK: {"labels": render_labels(cands, desc), "multi_label": True, "cls_threshold": 0.0,
                   "class_act": "softmax"}}


def scores_from_raw(raw, labels: list[str]) -> dict[str, float]:
    """Pull {label: score} out of whatever structure gliner2 returns."""
    task = raw.get(TASK, raw) if isinstance(raw, dict) else raw
    scores: dict[str, float] = {}

    def visit(x):
        if isinstance(x, dict):
            lab = x.get("label") or x.get("name") or x.get("text")
            sc = x.get("confidence", x.get("score", x.get("probability")))
            if lab in labels and isinstance(sc, (int, float)):
                scores[lab] = max(scores.get(lab, 0.0), float(sc))
            for k, v in x.items():
                if k in labels and isinstance(v, (int, float)):
                    scores[k] = float(v)
                elif isinstance(v, (dict, list, tuple)):
                    visit(v)
        elif isinstance(x, (list, tuple)):
            if len(x) == 2 and x[0] in labels and isinstance(x[1], (int, float)):
                scores[x[0]] = float(x[1])
            else:
                for v in x:
                    visit(v)

    visit(task)
    return scores


def ranked(scores: dict[str, float], cands: list[str]) -> list[str]:
    """Candidates by score, best first; ties (and unscored) keep candidate order."""
    return sorted(cands, key=lambda c: -scores.get(c, -1.0))


class GlinerDecider:
    def __init__(self, model_name_or_path: str = "fastino/GLiNER2.5-Decide", device: str | None = None):
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained(model_name_or_path)
        if device:
            self.model = self.model.to(device)
        self.model.eval()

    def _schema(self, cands, desc):
        return self.model._classification_schema(task_config(cands, desc))

    def score_batch(self, queries: list[str], cand_lists: list[list[str]], desc: dict[str, str],
                    batch_size: int = 16) -> list[dict[str, float]]:
        """One {tool: probability} dict per instance."""
        texts = [render_text(q) for q in queries]
        try:
            raws = self.model.batch_extract(texts, [self._schema(c, desc) for c in cand_lists],
                                            batch_size=batch_size, format_results=False, include_confidence=True)
        except Exception as e:  # noqa: BLE001  (fall back to one call per instance)
            print(f"[decider] batch_extract failed ({e!r}); falling back to classify_text", flush=True)
            raws = [self.model.classify_text(t, task_config(c, desc), format_results=False, include_confidence=True)
                    for t, c in zip(texts, cand_lists)]
        out = [scores_from_raw(r, c) for r, c in zip(raws, cand_lists)]
        missing = sum(len(s) < len(c) for s, c in zip(out, cand_lists))
        if missing:
            print(f"[decider] {missing}/{len(out)} instances lack some label scores; raw sample: {raws[0]!r:.300}",
                  flush=True)
        return out
