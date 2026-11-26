"""
m3_screen.py

Compare the first steps of several SAC runs (default vs variants) from their
TensorBoard logs. Built to judge whether a variant stops the critic blow-up
without a full 1M-step run.

For each run and each checkpoint step it prints the median over the previous
10K steps of:
  critic loss, |actor loss| (about the size of the Q-values when the entropy
  coefficient is small), and the training rollout reward ($ per battery).

Healthy means: critic loss below about 100 and |actor loss| below about 100
(a whole episode earns about $30 per battery, so Q-values cannot be much
larger than a few tens), with the reward rising.

Run from the project root (<repo>):
    python m3_screen.py
or with your own runs:
    python m3_screen.py label=folder label=folder ...
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = [
    ("default (lr 1e-3, 1K warm-up)", os.path.join("m3_logs", "sac_sweep", "N10", "SAC_seed3_1")),
    ("fixed ent_coef 0.01", os.path.join("m3_logs", "sac_n10_fixedent", "N10", "SAC_seed3_1")),
    ("lr 3e-4", os.path.join("m3_logs", "sac_n10_lr3e4", "N10", "SAC_seed3_1")),
    ("lr 1e-4", os.path.join("m3_logs", "sac_n10_lr1e4", "N10", "SAC_seed3_1")),
    ("warm-up 10K", os.path.join("m3_logs", "sac_n10_ls10k", "N10", "SAC_seed3_1")),
]
STEPS = [100_000, 200_000, 400_000, 600_000, 800_000, 1_000_000]


def load(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(folder, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    out = {}
    for key, tag in (("critic", "train/critic_loss"), ("actor", "train/actor_loss"), ("rew", "rollout/ep_rew_mean")):
        if tag in tags:
            ev = ea.Scalars(tag)
            out[key] = (np.array([e.step for e in ev], float), np.array([e.value for e in ev], float))
    return out


def med(xy, step, fn=lambda v: v):
    if xy is None:
        return float("nan")
    s, v = xy
    sel = (s <= step) & (s > step - 10_000)
    return float(np.median(fn(v[sel]))) if sel.any() else float("nan")


def fmt(x):
    return "     n/a" if not np.isfinite(x) else f"{x:8.3g}"


def main():
    runs = DEFAULTS
    if len(sys.argv) > 1:
        runs = [tuple(a.split("=", 1)) for a in sys.argv[1:]]
    for what, key, fn in (("critic loss (median of previous 10K steps)", "critic", lambda v: v),
                          ("|actor loss| (about the Q-value size)", "actor", np.abs),
                          ("training reward, $ per battery", "rew", lambda v: v)):
        print("\n" + "=" * 78)
        print("  " + what)
        print("=" * 78)
        print(f"  {'run':<32}" + "".join(f"{s//1000:>8}K" for s in STEPS))
        for label, folder in runs:
            path = folder if os.path.isabs(folder) else os.path.join(ROOT, folder)
            if not os.path.isdir(path):
                print(f"  {label:<32} folder not found: {folder}")
                continue
            d = load(path)
            print(f"  {label:<32}" + "".join(" " + fmt(med(d.get(key), s, fn)) for s in STEPS))
    print("\n  Healthy: critic and |actor loss| below about 100, reward rising. Compare with the default row.")


if __name__ == "__main__":
    main()
