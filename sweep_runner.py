"""HP sweep runner for /home/nx/sd/movielens.

Usage: uv run python sweep_runner.py <configs.json>

Reads a JSON list of trials [{"name": "...", "env": {"LR": ..., ...}}, ...].
For each trial:
  - sets env vars (DATASET=ml-25m default, merged with trial env),
  - runs `uv run python train.py > run.log 2>&1`,
  - parses val_auc, peak_memory_mb, total_seconds from run.log,
  - appends a row to results.tsv,
  - prints a one-line progress update.

Idempotent: skips trials whose name already appears in results.tsv (in this
sweep's description column).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUN_LOG = REPO / "run.log"
RESULTS = REPO / "results.tsv"
TRAIN_CMD = ["uv", "run", "python", "train.py"]
BASELINE_AUC = 0.827224  # for delta display; trial 6 sanity check


def parse_run_log(path: Path):
    """Return (val_auc, peak_memory_mb, total_seconds) or raise ValueError."""
    text = path.read_text()
    def _grab(key):
        m = re.search(rf"^{re.escape(key)}:\s+([\d.]+)\s*$", text, flags=re.M)
        if not m:
            raise ValueError(f"could not parse {key} from run.log")
        return float(m.group(1))
    return _grab("val_auc"), _grab("peak_memory_mb"), _grab("total_seconds")


def git_short_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()


def already_done(trial_name: str) -> bool:
    """Check if results.tsv already has a 'done' row for this trial name."""
    if not RESULTS.exists():
        return False
    needle = f"{trial_name}:"
    for line in RESULTS.read_text().splitlines():
        cols = line.split("\t")
        if len(cols) >= 5 and cols[3] == "done" and needle in cols[4]:
            return True
    return False


def fmt_env_desc(env: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in env.items())


def run_trial(idx: int, total: int, trial: dict, sha: str):
    name = trial["name"]
    trial_env = trial.get("env", {})
    desc_env = fmt_env_desc(trial_env)

    if already_done(name):
        print(f"[{idx}/{total}] {name}: SKIP (already in results.tsv)")
        return None

    print(f"[{idx}/{total}] {name}: starting ({desc_env})", flush=True)

    env = os.environ.copy()
    env.setdefault("DATASET", "ml-25m")
    for k, v in trial_env.items():
        env[k] = str(v)

    t0 = time.time()
    try:
        with RUN_LOG.open("w") as f:
            proc = subprocess.run(
                TRAIN_CMD, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT
            )
    except Exception as e:
        wall = time.time() - t0
        msg = f"[{idx}/{total}] {name}: ERROR launching subprocess after {wall:.0f}s: {e!r}"
        print(msg, flush=True)
        with RESULTS.open("a") as f:
            f.write(f"{sha}\t-\t-\terror\t{name}: {desc_env}, exception={e!r}\n")
        return None
    wall = time.time() - t0

    if proc.returncode != 0:
        msg = f"[{idx}/{total}] {name}: NONZERO exit={proc.returncode} after {wall:.0f}s"
        print(msg, flush=True)
        with RESULTS.open("a") as f:
            f.write(
                f"{sha}\t-\t-\terror\t{name}: {desc_env}, exit={proc.returncode}, wall={wall:.0f}s\n"
            )
        return None

    try:
        val_auc, peak_mb, total_s = parse_run_log(RUN_LOG)
    except ValueError as e:
        print(
            f"[{idx}/{total}] {name}: PARSE FAIL after {wall:.0f}s ({e})",
            flush=True,
        )
        with RESULTS.open("a") as f:
            f.write(f"{sha}\t-\t-\terror\t{name}: {desc_env}, parse_fail\n")
        return None

    delta = val_auc - BASELINE_AUC
    desc = f"{name}: {desc_env}, total={total_s:.0f}s"
    with RESULTS.open("a") as f:
        f.write(f"{sha}\t{val_auc:.6f}\t{int(round(peak_mb))}\tdone\t{desc}\n")

    print(
        f"[{idx}/{total}] {name}: done val_auc={val_auc:.4f} "
        f"(Δ={delta:+.4f}) [{total_s:.0f}s]",
        flush=True,
    )
    return {
        "name": name,
        "env": trial_env,
        "val_auc": val_auc,
        "peak_mb": peak_mb,
        "total_s": total_s,
    }


def main():
    if len(sys.argv) != 2:
        print("usage: sweep_runner.py <configs.json>", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(sys.argv[1])
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    trials = json.loads(cfg_path.read_text())
    n = len(trials)
    print(f"Loaded {n} trials from {cfg_path}", flush=True)

    sha = git_short_sha()
    print(f"Commit: {sha}", flush=True)

    sweep_t0 = time.time()
    results = []
    for i, trial in enumerate(trials, start=1):
        r = run_trial(i, n, trial, sha)
        if r is not None:
            results.append(r)
    sweep_wall = time.time() - sweep_t0

    print("", flush=True)
    print(f"=== Sweep complete in {sweep_wall:.0f}s ({sweep_wall/60:.1f} min) ===", flush=True)
    if not results:
        print("No successful trials.", flush=True)
        return

    results_sorted = sorted(results, key=lambda r: r["val_auc"], reverse=True)
    print("", flush=True)
    print(f"Top-5 by val_auc:", flush=True)
    for r in results_sorted[:5]:
        env_s = fmt_env_desc(r["env"])
        print(
            f"  {r['name']:8s}  val_auc={r['val_auc']:.6f}  "
            f"({env_s})  [{r['total_s']:.0f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()
