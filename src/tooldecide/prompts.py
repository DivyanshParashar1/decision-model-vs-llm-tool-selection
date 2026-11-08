"""LLM prompts for tool selection + robust parsing of the answer.

The same prompt is used for the raw LLM (C1) and the fine-tuned LLM (C3); the SFT target is the JSON answer.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "p1"
SYSTEM = "You pick the right tool for a user's request from a list of tools. Answer with JSON only."


def tool_list(cands: list[str], desc: dict[str, str]) -> str:
    return "\n".join(f"{i + 1}. {t}: {desc[t]}" for i, t in enumerate(cands))


def single_messages(query: str, cands: list[str], desc: dict[str, str]) -> list[dict]:
    user = (f"User request: {query}\n\nTools:\n{tool_list(cands, desc)}\n\n"
            'Which one tool best handles this request? Reply with JSON only: {"tool": "<tool name>"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def multi_messages(query: str, cands: list[str], desc: dict[str, str]) -> list[dict]:
    user = (f"User request: {query}\n\nTools:\n{tool_list(cands, desc)}\n\n"
            "This request needs exactly two of these tools. Which two? "
            'Reply with JSON only: {"tools": ["<tool name>", "<tool name>"]}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def messages(query: str, cands: list[str], desc: dict[str, str], n_gold: int) -> list[dict]:
    return multi_messages(query, cands, desc) if n_gold > 1 else single_messages(query, cands, desc)


def target(gold: list[str]) -> str:
    """Assistant answer used as the SFT target."""
    return json.dumps({"tools": gold}) if len(gold) > 1 else json.dumps({"tool": gold[0]})


# ------------------------------------------------------------------ parsing
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _json_obj(text: str):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def _resolve(x, cands: list[str]) -> str | None:
    """Map one answer item (name, near-name, or 1-based list number) to a candidate."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()):
        i = int(x) - 1
        return cands[i] if 0 <= i < len(cands) else None
    if not isinstance(x, str):
        return None
    x = re.sub(r"^\s*\d+\.\s*", "", x).strip().strip("\"'`")
    if x in cands:
        return x
    by_key = {_key(c): c for c in cands}
    return by_key.get(_key(x))


def _mentions(text: str, cands: list[str]) -> list[str]:
    """Candidates named in free text, in order of first mention (longer names win at the same spot)."""
    low = text.lower()
    hits = []
    for c in cands:
        m = re.search(rf"(?<![a-z0-9_]){re.escape(c.lower())}(?![a-z0-9_])", low)
        if m:
            hits.append((m.start(), -len(c), c))
    return [c for _, _, c in sorted(hits)]


def parse_tools(text: str, cands: list[str], n: int) -> list[str]:
    """Up to n distinct candidates from an LLM answer ([] if nothing usable)."""
    text = _THINK.sub("", text or "")
    obj = _json_obj(text)
    items = []
    if isinstance(obj, dict):
        for k in ("tools", "tool", "answer", "name"):
            v = obj.get(k)
            if v is not None:
                items = v if isinstance(v, list) else [v]
                break
    out = []
    for x in items:
        r = _resolve(x, cands)
        if r and r not in out:
            out.append(r)
    if not out:
        out = _mentions(text, cands)
    return out[:n]


def parse_tool(text: str, cands: list[str]) -> str | None:
    got = parse_tools(text, cands, 1)
    return got[0] if got else None
