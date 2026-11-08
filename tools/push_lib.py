"""Publish src/tooldecide as a private Kaggle dataset `<user>/tooldecide-lib`
so notebooks can `sys.path.append` it. Creates it the first time, adds a version after.

Usage: python tools/push_lib.py "message" [--account=acc2]
"""
import json
import pathlib
import shutil
import subprocess
import sys

import kenv
from kpush import ROOT

SLUG = "tooldecide-lib"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--account=")]
    account = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--account=")), "main")
    msg = args[0] if args else "update"
    user = kenv.user(account)
    build = ROOT / "build" / SLUG
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(ROOT / "src" / "tooldecide", build / "tooldecide",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (build / "dataset-metadata.json").write_text(json.dumps(
        {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = kenv.run(["datasets", "status", f"{user}/{SLUG}"], account)[0] == 0
    cmd = (["datasets", "version", "-p", str(build), "-m", msg, "--dir-mode", "zip"] if exists
           else ["datasets", "create", "-p", str(build), "--dir-mode", "zip"])
    print(kenv.run(cmd, account, check=True)[1][-300:])
    print(f"dataset: {user}/{SLUG}")


if __name__ == "__main__":
    main()
