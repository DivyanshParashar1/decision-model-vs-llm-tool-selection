#!/usr/bin/env bash
# Fails if a Kaggle token (KGAT_...) is staged for commit or present in any tracked file.
# Run before every push; also installed as .git/hooks/pre-commit by tools/install_hooks.sh.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pattern='KGAT_[A-Za-z0-9]{8,}'
found=0
if git diff --cached -U0 | grep -nE "$pattern" >/dev/null; then
  echo "BLOCKED: a Kaggle token is staged for commit. Remove it (keep tokens only in .env)." >&2
  found=1
fi
if git grep -nIE "$pattern" -- . ':!.env.example' >/dev/null 2>&1; then
  echo "BLOCKED: a Kaggle token is present in a tracked file:" >&2
  git grep -lIE "$pattern" -- . ':!.env.example' >&2
  found=1
fi
if git check-ignore -q .env 2>/dev/null || [ ! -e .env ]; then :; else
  echo "BLOCKED: .env is not gitignored." >&2
  found=1
fi
[ "$found" -eq 0 ] && echo "secrets check: clean"
exit "$found"
