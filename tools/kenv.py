"""Pick which Kaggle account a CLI call runs as.

Accounts and where their token is read from (first match wins):
  main -> KAGGLE_TOKEN_MAIN in <repo>/.env, else the default CLI credentials (~/.kaggle/access_token)
  acc2 -> KAGGLE_TOKEN_ACC2 in <repo>/.env, else ~/.kaggle/access_token_acc2

.env is gitignored. Tokens are read at call time and never printed or written anywhere else.
"""
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACCOUNTS = ["main", "acc2"]
ENV_KEYS = {"main": "KAGGLE_TOKEN_MAIN", "acc2": "KAGGLE_TOKEN_ACC2"}
FALLBACK_FILES = {"acc2": pathlib.Path.home() / ".kaggle" / "access_token_acc2"}


def _dotenv() -> dict:
    p = ROOT / ".env"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def token(account: str) -> str | None:
    v = _dotenv().get(ENV_KEYS[account]) or os.environ.get(ENV_KEYS[account])
    if v:
        return v
    f = FALLBACK_FILES.get(account)
    if f and f.exists():
        return f.read_text().strip()
    return None  # main: fall back to the CLI's own stored credentials


def env(account: str = "main") -> dict:
    e = dict(os.environ)
    t = token(account)
    if t:
        e["KAGGLE_API_TOKEN"] = t
    elif account != "main":
        raise SystemExit(f"no token for account {account!r}: set {ENV_KEYS[account]} in .env")
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
