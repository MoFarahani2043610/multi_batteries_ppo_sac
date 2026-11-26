"""
m3_paired.py

Paired SAC-vs-PPO comparison per fleet size N, using the corrected
re-evaluation (m3_reeval_episodes.csv) and the baselines
(m3_baselines_episodes.csv).

Pairing: the same seed = the same heterogeneous fleet and the same 20 eval
episodes (reset seed 1000+ep), so for each N we get 5 paired differences
(SAC - PPO), one per fleet. Per-fleet value = mean over the 20 episodes of
$ per battery per episode.

Reported per N:
  mean diff, std of diffs, paired t-test p (df = n_seeds-1), 95% t-interval,
  Cohen's dz, number of fleets where SAC > PPO.
Also: each algorithm vs Threshold(q25/q75) and vs LP, same pairing.

Note: with only 5 fleets the smallest possible two-sided sign-flip
(permutation) p-value is 2/32 = 0.0625, so it can never be < 0.05. The paired
t-test is the main test; the permutation p is printed as a check.

Run from the project root (D:\\Project 2026\\env):
    python m3_paired.py

Output: m3_paired_summary.csv
"""

import csv
import itertools
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RE_CSV = os.path.join(ROOT, "m3_reeval_episodes.csv")
BASE_CSV = os.path.join(ROOT, "m3_baselines_episodes.csv")
OUT_CSV = os.path.join(ROOT, "m3_paired_summary.csv")
NS = [1, 2, 5, 10, 20]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def per_fleet_means():
    """returns dict {(name, n, seed): mean per-battery $} for ppo, sac, thr, lp"""
    out = {}
    tmp = {}
    for r in read(RE_CSV):
        if r["algo"] in ("ppo", "sac"):
            tmp.setdefault((r["algo"], int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_return"]))
    for k, v in tmp.items():
        out[k] = float(np.mean(v))
    tmp = {}
    names = {"Threshold(q25/q75)": "thr", "LP": "lp"}
    for r in read(BASE_CSV):
        if r["policy"] in names:
            tmp.setdefault((names[r["policy"]], int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_profit"]))
    for k, v in tmp.items():
        out[k] = float(np.mean(v))
    return out


def paired_stats(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t, p = stats.ttest_1samp(d, 0.0)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        dz = mean / sd
    else:
        t, p, half, dz = float("nan"), float("nan"), float("nan"), float("nan")
    # exact sign-flip permutation test
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    perm = np.abs((signs * d).mean(axis=1))
    p_perm = float(np.mean(perm >= abs(mean) - 1e-12))
    return {"n": n, "mean": mean, "sd": sd, "p_t": float(p), "lo": mean - half, "hi": mean + half,
            "dz": dz, "wins": int((d > 0).sum()), "p_perm": p_perm}


def fmt_p(p):
    if np.isnan(p):
        return "  n/a"
    return "<0.001" if p < 0.001 else f"{p:6.3f}"


def main():
    for p in (RE_CSV, BASE_CSV):
        if not os.path.exists(p):
            raise SystemExit(f"missing file: {p}")
    m = per_fleet_means()
    rows = []

    def block(title, a, b):
        print("\n" + "=" * 92)
        print(f"  {title}")
        print("=" * 92)
        print(f"  {'N':>3} {'fleets':>6} {'mean diff':>10} {'sd':>7} {'95% interval':>20} {'p (t-test)':>11} "
              f"{'p (perm)':>9} {'dz':>6} {'A>B':>5}")
        for n in NS:
            seeds = sorted(s for (nm, nn, s) in m if nm == a and nn == n and (b, n, s) in m)
            if len(seeds) < 2:
                continue
            d = [m[(a, n, s)] - m[(b, n, s)] for s in seeds]
            r = paired_stats(d)
            print(f"  {n:>3} {r['n']:>6} {r['mean']:>10.2f} {r['sd']:>7.2f} "
                  f"[{r['lo']:>7.2f}, {r['hi']:>7.2f}] {fmt_p(r['p_t']):>11} {fmt_p(r['p_perm']):>9} "
                  f"{r['dz']:>6.2f} {r['wins']:>2}/{r['n']}")
            rows.append({"comparison": f"{a} - {b}", "N": n, "fleets": r["n"],
                         "mean_diff": round(r["mean"], 4), "sd": round(r["sd"], 4),
                         "ci_lo": round(r["lo"], 4), "ci_hi": round(r["hi"], 4),
                         "p_ttest": round(r["p_t"], 5), "p_perm": round(r["p_perm"], 5),
                         "dz": round(r["dz"], 3), "a_wins": r["wins"]})

    block("SAC minus PPO   ($ per battery per episode, paired by fleet)", "sac", "ppo")
    block("PPO minus Threshold(q25/q75)", "ppo", "thr")
    block("SAC minus Threshold(q25/q75)", "sac", "thr")
    block("PPO minus LP (negative = below the oracle)", "ppo", "lp")
    block("SAC minus LP (negative = below the oracle)", "sac", "lp")

    print("\n  Per-fleet values (mean $ per battery over 20 episodes):")
    print(f"  {'N':>3} {'seed':>4} {'PPO':>8} {'SAC':>8} {'thr-q':>8} {'LP':>8}")
    for n in NS:
        for s in range(5):
            if all((k, n, s) in m for k in ("ppo", "sac", "thr", "lp")):
                print(f"  {n:>3} {s:>4} {m[('ppo', n, s)]:>8.2f} {m[('sac', n, s)]:>8.2f} "
                      f"{m[('thr', n, s)]:>8.2f} {m[('lp', n, s)]:>8.2f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved: {OUT_CSV}")
    print("  Reminder: 5 fleets only; p_perm can never go below 0.0625. Seed = fleet AND RL randomness.")


if __name__ == "__main__":
    main()