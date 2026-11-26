"""
m3_reeval.py

Re-evaluate every saved M3 model on the SAME environment it was trained on
(heterogeneous fleet for (N, seed), real CAISO prices, 6-dim features) using
the same 20 episodes as the baselines (env.reset(seed=1000 + ep)).

Why: agent.evaluate() in the sweep builds its env with _make_fresh_env(), which
creates N identical default batteries, not the heterogeneous fleet used for
training and for the LP/threshold baselines. This script fixes the comparison
without retraining.

Run from the project root (<repo>):
    python m3_reeval.py                  # PPO and SAC, whatever cells exist
    python m3_reeval.py --algo ppo
    python m3_reeval.py --list           # only list the models it finds
    python m3_reeval.py --summary        # only print the comparison table

Needs m3_baselines_episodes.csv (made by m3_baselines.py) for the table.

Output:
    m3_reeval_episodes.csv    one row per (algo, N, seed, episode)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RE_CSV = os.path.join(ROOT, "m3_reeval_episodes.csv")
BASE_CSV = os.path.join(ROOT, "m3_baselines_episodes.csv")
RE_FIELDS = ["algo", "n", "seed", "episode", "per_battery_return", "total_profit", "model_path"]

SWEEP_DIRS = {
    "ppo": os.path.join(ROOT, "m3_logs", "ppo_sweep"),
    "sac": os.path.join(ROOT, "m3_logs", "sac_sweep"),
}
CONFIGS = {
    "ppo": os.path.join("m3_configs", "ppo_scaling.yaml"),
    "sac": os.path.join("m3_configs", "sac_scaling.yaml"),
}
N_EPISODES = 20
EVAL_SEED = 1000


def find_models(algo, root=None):
    """Return dict {(n, seed): path} for every final_model.zip under `root`
    (default: the standard sweep folder of `algo`; `algo` is also the label)."""
    found = {}
    if root is None:
        root = SWEEP_DIRS[algo]
    if not os.path.isdir(root):
        print(f"  [{algo}] folder not found: {root}")
        return found
    for dirpath, _dirs, files in os.walk(root):
        if "final_model.zip" not in files:
            continue
        parts = re.split(r"[\\/]", dirpath)
        n = seed = None
        for p in parts:
            m = re.fullmatch(r"N(\d+)", p)
            if m:
                n = int(m.group(1))
            m = re.fullmatch(r"seed_?(\d+)", p)
            if m:
                seed = int(m.group(1))
        path = os.path.join(dirpath, "final_model.zip")
        if n is None or seed is None:
            print(f"  [{algo}] cannot read N/seed from path, skipped: {path}")
            continue
        if (n, seed) in found:
            print(f"  [{algo}] duplicate for N={n} seed={seed}, keeping first: {found[(n, seed)]}")
            continue
        found[(n, seed)] = path
    return found


def done_keys():
    done = set()
    if os.path.exists(RE_CSV):
        with open(RE_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["algo"], int(r["n"]), int(r["seed"])))
    return done


def append_rows(rows):
    new = not os.path.exists(RE_CSV)
    with open(RE_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RE_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize():
    if not os.path.exists(RE_CSV):
        print("No m3_reeval_episodes.csv yet.")
        return
    if not os.path.exists(BASE_CSV):
        print("No m3_baselines_episodes.csv: run m3_baselines.py first.")
        return

    # per (n, seed): mean per-battery profit of each baseline policy
    base = {}
    for r in _read(BASE_CSV):
        base.setdefault((r["policy"], int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_profit"]))
    base = {k: float(np.mean(v)) for k, v in base.items()}

    rl = {}
    for r in _read(RE_CSV):
        rl.setdefault((r["algo"], int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_return"]))
    rl = {k: float(np.mean(v)) for k, v in rl.items()}

    algos = sorted({k[0] for k in rl})
    ns = sorted({k[1] for k in rl})
    print("\n" + "=" * 96)
    print("  $ per battery per episode, on each cell's own heterogeneous fleet, same 20 episodes")
    print("  %LP = mean over seeds of (RL / LP) for the same fleet.  d_thr = RL minus Threshold(q25/q75)")
    print("=" * 96)
    print(f"  {'algo':<4} {'N':>3} {'seeds':>5} {'RL mean':>9} {'RL std':>8} {'thr25/50':>9} {'thr-q':>8} {'LP':>8} {'%LP':>7} {'d_thr':>8}")
    for algo in algos:
        for n in ns:
            seeds = sorted(s for (a, nn, s) in rl if a == algo and nn == n
                           and ("LP", n, s) in base and ("Threshold(q25/q75)", n, s) in base)
            if not seeds:
                continue
            r_ = np.array([rl[(algo, n, s)] for s in seeds])
            lp = np.array([base[("LP", n, s)] for s in seeds])
            t1 = np.array([base[("Threshold(25/50)", n, s)] for s in seeds])
            tq = np.array([base[("Threshold(q25/q75)", n, s)] for s in seeds])
            print(f"  {algo:<4} {n:>3} {len(seeds):>5} {r_.mean():>9.2f} {r_.std():>8.2f} "
                  f"{t1.mean():>9.2f} {tq.mean():>8.2f} {lp.mean():>8.2f} "
                  f"{100*np.mean(r_/lp):>6.1f}% {np.mean(r_-tq):>8.2f}")


def run(jobs, list_only):
    """jobs: list of (label, algo_type, root_or_None). algo_type is 'sac' or 'ppo'."""
    import m3_baselines as mb          # sets sys.path and loads the env code
    from stable_baselines3 import SAC, PPO
    classes = {"sac": SAC, "ppo": PPO}

    prices = features = None
    done = done_keys()
    for label, atype, root in jobs:
        algo = label
        models = find_models(label, root)
        print(f"[{label}] found {len(models)} saved models")
        if list_only:
            for (n, s), p in sorted(models.items()):
                print(f"   N={n:>2} seed={s}: {p}")
            continue
        if not models:
            continue
        cfg = mb.load_config(os.path.join(ROOT, CONFIGS[atype]))
        if cfg.get("env", {}).get("price_source", "").lower() != "caiso":
            raise ValueError("config env.price_source must be 'caiso'")
        if prices is None:
            prices, timestamps = mb.load_prices(market="caiso", cache_dir=os.path.join(ROOT, "cache"))
            features = mb.make_features(timestamps)

        for (n, seed), path in sorted(models.items()):
            if (algo, n, seed) in done:
                print(f"  {algo} N={n} seed={seed}: already done, skipping")
                continue
            t0 = time.time()
            env = mb.build_env(n, seed, cfg, prices, features)
            model = classes[atype].load(path, device="cpu")
            rows = []
            rets = []
            for ep in range(N_EPISODES):
                obs, info = env.reset(seed=EVAL_SEED + ep)
                finished = False
                r_sum = 0.0
                while not finished:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(action)
                    r_sum += r
                    finished = term or trunc
                rets.append(r_sum)
                rows.append({"algo": algo, "n": n, "seed": seed, "episode": ep,
                             "per_battery_return": round(float(r_sum), 6),
                             "total_profit": round(float(info["cumulative_profit"]), 6),
                             "model_path": path})
            append_rows(rows)
            print(f"  {algo} N={n:>2} seed={seed}: per-battery ${np.mean(rets):7.2f}  ({time.time()-t0:.0f}s)")
    summarize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac", "both"], default="both")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--root", default=None,
                    help="evaluate the models under this folder instead of the standard sweep "
                         "(use with --label and --algo sac|ppo), e.g. m3_logs\\sac_n5_long_check")
    ap.add_argument("--label", default=None,
                    help="name for these models in the CSV and the table, e.g. sac_n5_long")
    args = ap.parse_args()
    if args.summary:
        summarize()
        return
    if args.root:
        if args.algo == "both" or not args.label:
            raise SystemExit("--root needs --algo sac (or ppo) and --label NAME")
        root = args.root if os.path.isabs(args.root) else os.path.join(ROOT, args.root)
        run([(args.label, args.algo, root)], args.list)
        return
    algos = ["ppo", "sac"] if args.algo == "both" else [args.algo]
    run([(a, a, None) for a in algos], args.list)


if __name__ == "__main__":
    main()