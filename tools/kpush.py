"""Build a notebook from kaggle/NN_*.py and push it to Kaggle as a (background) run.

Usage:
  python tools/kpush.py kaggle/01_data.py --accel none
  python tools/kpush.py kaggle/02_baselines.py --accel t4x2 --input divyanshparashar1/tools-01-data
  python tools/kpush.py kaggle/01_data.py --set QUICK=False

Then: kaggle kernels status <user>/<slug>   |   kaggle kernels output <user>/<slug> -p out/
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from py2nb import convert  # noqa: E402
import kenv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACCEL = {  # kaggle machine_shape values
    "none": ("None", False),
    "t4x2": ("NvidiaTeslaT4", True),
    "p100": ("NvidiaTeslaP100", True),
}


def kaggle_user(account="main"):
    return kenv.user(account)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--accel", default="none", choices=ACCEL)
    ap.add_argument("--input", action="append", default=[], help="kernel output to mount (user/slug)")
    ap.add_argument("--dataset", action="append", default=[], help="dataset to mount (user/slug)")
    ap.add_argument("--set", action="append", default=[], help="override a top-level assignment, e.g. QUICK=False")
    ap.add_argument("--account", default="main", choices=kenv.ACCOUNTS)
    ap.add_argument("--slug", help="default: tools-<file stem with _ as ->")
    args = ap.parse_args()

    src = ROOT / args.src
    text = src.read_text()
    for kv in args.set:
        k, v = kv.split("=", 1)
        text, n = re.subn(rf"^{k} = .*$", f"{k} = {v}", text, count=1, flags=re.M)
        if not n:
            sys.exit(f"no top-level assignment `{k} = ...` in {src}")

    slug = args.slug or "tools-" + src.stem.replace("_", "-")
    build = ROOT / "build" / slug
    build.mkdir(parents=True, exist_ok=True)
    tmp = build / src.name
    tmp.write_text(text)
    nb = convert(tmp)
    tmp.unlink()

    shape, gpu = ACCEL[args.accel]
    user = kaggle_user(args.account)
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": nb.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": args.dataset, "kernel_sources": args.input,
        "competition_sources": [], "model_sources": [], "machine_shape": shape,
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    code, out = kenv.run(["kernels", "push", "-p", str(build)], args.account)
    print(out)
    if code:
        sys.exit(code)
    print(f"\naccount: {args.account}\nstatus: kaggle kernels status {user}/{slug}\n"
          f"logs:   kaggle kernels logs {user}/{slug}\n"
          f"output: kaggle kernels output {user}/{slug} -p build/{slug}/out")


if __name__ == "__main__":
    main()
