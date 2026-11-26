"""
m3_paired_lr1e4.py

Paired comparison of SAC with learning rate 1e-4 against PPO, default SAC (lr 1e-3),
Threshold(q25/q75) and the LP oracle, per N, paired by fleet (same seed = same fleet,
same 20 evaluation episodes).

SAC lr 1e-4 values come from two re-evaluation labels:
    sac_lr1e4_scaling   (N = 1, 2, 5, 20, and N=5 seeds 1, 2)
    sac_n10_lr1e4_full  (N = 10, seeds 0-4)

Only N values with at least 2 common fleets are tested. With 3 fleets the t-test has
2 degrees of freedom, so intervals are wide; with 5 fleets the smallest possible
sign-flip permutation p is 0.0625 (3 fleets: 0.25). The paired t-test is the main test.
Bonferroni threshold over the 5 N values: 0.01.

Run from D:\\Project 2026\\env:
    python m3_paired_lr1e4.py

Output: m3_paired_lr1e4_summary.csv
"""

import csv
import itertools
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RE_CSV = os.path.join(ROOT, "m3_reeval_episodes.csv")
BASE_CSV = os.path.join(ROOT, "m3_baselines_episodes.csv")
OUT_CSV = os.path.join(ROOT, "m3_paired_lr1e4_summary.csv")
NS = [1, 2, 5, 10, 20]
LR_LABELS = ("sac_lr1e4_scaling", "sac_n10_lr1e4_full")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load():
    re_rows = read(RE_CSV)
    if not re_rows:
        raise SystemExit("m3_reeval_episodes.csv is empty")
    if "algo" not in re_rows[0] or "per_battery_return" not in re_rows[0]:
        raise SystemExit(f"unexpected columns in m3_reeval_episodes.csv: {list(re_rows[0].keys())}")
    labels = sorted({r["algo"] for r in re_rows})
    tmp = {}
    for r in re_rows:
        a = r["algo"]
        if a == "ppo":
            key = "ppo"
        elif a == "sac":
            key = "sac"
        elif a in LR_LABELS:
            key = "lr"
        else:
            continue
        tmp.setdefault((key, int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_return"]))
    names = {"Threshold(q25/q75)": "thr", "LP": "lp"}
    for r in read(BASE_CSV):
        if r["policy"] in names:
            tmp.setdefault((names[r["policy"]], int(r["n"]), int(r["seed"])), []).append(float(r["per_battery_profit"]))
    return {k: float(np.mean(v)) for k, v in tmp.items()}, labels


def paired_stats(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd > 0:
        t, p = stats.ttest_1samp(d, 0.0)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        dz = mean / sd
    else:
        p, half, dz = float("nan"), float("nan"), float("nan")
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    p_perm = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(mean) - 1e-12))
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
    m, labels = load()
    print("labels found in m3_reeval_episodes.csv:", ", ".join(labels))
    rows = []

    def block(title, b):
        print("\n" + "=" * 96)
        print(f"  {title}")
        print("=" * 96)
        print(f"  {'N':>3} {'fleets':>6} {'seeds':>9} {'mean diff':>10} {'sd':>7} {'95% interval':>20} "
              f"{'p (t)':>8} {'p (perm)':>9} {'dz':>6} {'wins':>5}")
        for n in NS:
            seeds = sorted(s for (k, nn, s) in m if k == "lr" and nn == n and (b, n, s) in m)
            if len(seeds) < 2:
                print(f"  {n:>3} {len(seeds):>6}   (fewer than 2 common fleets, not tested)")
                continue
            d = [m[("lr", n, s)] - m[(b, n, s)] for s in seeds]
            r = paired_stats(d)
            sd_txt = ",".join(str(s) for s in seeds)
            print(f"  {n:>3} {r['n']:>6} {sd_txt:>9} {r['mean']:>10.2f} {r['sd']:>7.2f} "
                  f"[{r['lo']:>7.2f}, {r['hi']:>7.2f}] {fmt_p(r['p_t']):>8} {fmt_p(r['p_perm']):>9} "
                  f"{r['dz']:>6.2f} {r['wins']:>2}/{r['n']}")
            rows.append({"comparison": f"SAC lr1e-4 - {b}", "N": n, "fleets": r["n"], "seeds": sd_txt,
                         "mean_diff": round(r["mean"], 4), "sd": round(r["sd"], 4),
                         "ci_lo": round(r["lo"], 4), "ci_hi": round(r["hi"], 4),
                         "p_ttest": round(r["p_t"], 5), "p_perm": round(r["p_perm"], 5),
                         "dz": round(r["dz"], 3), "wins": r["wins"]})

    block("SAC lr 1e-4 minus PPO        ($ per battery per episode, paired by fleet)", "ppo")
    block("SAC lr 1e-4 minus SAC lr 1e-3 (default)", "sac")
    block("SAC lr 1e-4 minus Threshold(q25/q75)", "thr")
    block("SAC lr 1e-4 minus LP oracle  (negative = below oracle)", "lp")

    print("\n  Per-fleet values (mean $ per battery over 20 episodes), only fleets with an lr 1e-4 run:")
    print(f"  {'N':>3} {'seed':>4} {'PPO':>8} {'SAC1e-3':>8} {'SAC1e-4':>8} {'thr-q':>8} {'LP':>8}")
    for n in NS:
        for s in range(5):
            if all((k, n, s) in m for k in ("ppo", "sac", "lr", "thr", "lp")):
                print(f"  {n:>3} {s:>4} {m[('ppo', n, s)]:>8.2f} {m[('sac', n, s)]:>8.2f} "
                      f"{m[('lr', n, s)]:>8.2f} {m[('thr', n, s)]:>8.2f} {m[('lp', n, s)]:>8.2f}")

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  saved: {OUT_CSV}")
    print("  Reminders: lr 1e-4 was chosen from the seed-3 fleet at N=10, so seeds 0, 1, 2, 4 did not take part in that choice. "
          "Only SAC was retuned. N=5 has 3 fleets (seeds 0-2); N=1, 2 and 20 have 1 fleet and are not tested.")


if __name__ == "__main__":
    main()