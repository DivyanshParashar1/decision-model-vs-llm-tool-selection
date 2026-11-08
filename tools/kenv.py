"""Pick which Kaggle account a CLI call runs as.

Accounts:
  main -> default CLI credentials (~/.kaggle/access_token)
  acc2 -> token in ~/.kaggle/access_token_acc2

Tokens are read from disk at call time and never printed or stored in the repo.
"""
import os
import pathlib
import re
import subprocess

TOKENS = {"acc2": pathlib.Path.home() / ".kaggle" / "access_token_acc2"}
ACCOUNTS = ["main", *TOKENS]


def env(account: str = "main") -> dict:
    e = dict(os.environ)
    if account != "main":
        e["KAGGLE_API_TOKEN"] = TOKENS[account].read_text().strip()
    return e


def run(args, account="main", check=False):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env(account))
    if check and p.returncode:
        raise SystemExit(p.stdout + p.stderr)
    return p.returncode, (p.stdout + p.stderr).strip()


def user(account="main"):
    _, out = run(["config", "view"], account)
    m = re.search(r"username: (\S+)", out)
    if not m:
        raise SystemExit(f"not logged in for account {account!r}")
    return m.group(1)
