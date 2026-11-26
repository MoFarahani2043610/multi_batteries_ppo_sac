"""
m3_tb_diag.py

Read the TensorBoard logs of the M3 sweeps (no retraining) and answer:
is the SAC failure at large N (a) not enough steps still improving at 1M,
(b) a collapse / divergence during training, or (c) never learning?

Per run it reports, from the TRAINING logs (rollout reward is in $ per battery,
same unit for every N, measured on the training fleet with exploration noise,
so it is lower than the deterministic evaluation numbers):

  final_rew     mean rollout/ep_rew_mean over the last 100K steps
  slope_100k    least-squares slope of rollout reward over the last 300K steps,
                in $ per 100K steps  (clearly > 0 = still learning, ~0 = plateau)
  peak_drop     best smoothed reward during training minus final_rew
                (large = it learned something and then lost it)
  critic_med    median SAC critic loss over the last 100K steps (SAC only)
  critic_max    maximum SAC critic loss over the whole run
  ent_coef      mean SAC entropy coefficient over the last 100K steps
  PPO: value_loss, explained_variance, approx_kl over the last 100K steps

Critic loss (SAC) and value loss (PPO) are NOT comparable with each other
(different algorithms, different targets). Only compare each algorithm with
itself across N. Only the rollout reward is comparable between SAC and PPO.

Several run folders can exist for one (N, seed) (restarted runs, e.g. N1 seeds
0 and 4). The one that reaches the most training steps is used; the choice is
printed.

Run from the project root (<repo>):
    python m3_tb_diag.py
Output:
    m3_tb_diag_runs.csv     one row per (algo, N, seed)
    m3_tb_diag.png          reward, critic loss and entropy coefficient versus steps, per N
"""

import csv
import glob
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = {"sac": os.path.join(ROOT, "m3_logs", "sac_sweep"),
          "ppo": os.path.join(ROOT, "m3_logs", "ppo_sweep")}
NS = [1, 2, 5, 10, 20]
OUT_CSV = os.path.join(ROOT, "m3_tb_diag_runs.csv")
OUT_PNG = os.path.join(ROOT, "m3_tb_diag.png")

TAGS = {
    "rew": ["rollout/ep_rew_mean"],
    "critic": ["train/critic_loss"],
    "ent_coef": ["train/ent_coef"],
    "actor": ["train/actor_loss"],
    "vloss": ["train/value_loss"],
    "expl_var": ["train/explained_variance"],
    "kl": ["train/approx_kl"],
}


def load_run(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(folder, size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    out = {"_tags": avail}
    for key, cands in TAGS.items():
        for t in cands:
            if t in avail:
                ev = ea.Scalars(t)
                out[key] = (np.array([e.step for e in ev], float), np.array([e.value for e in ev], float))
                break
    return out


def find_runs(algo, n):
    """{seed: folder} using the run folder that reaches the most steps."""
    base = os.path.join(SWEEPS[algo], f"N{n}")
    best = {}
    for d in sorted(glob.glob(os.path.join(base, f"{algo.upper()}_seed*_*"))):
        m = re.search(r"_seed(\d+)_(\d+)$", d)
        if not m or not os.path.isdir(d):
            continue
        seed = int(m.group(1))
        try:
            run = load_run(d)
        except Exception as e:  # unreadable folder
            print(f"  cannot read {d}: {e}")
            continue
        if "rew" not in run:
            continue
        top = run["rew"][0].max()
        best.setdefault(seed, []).append((top, d, run))
    chosen = {}
    for seed, lst in best.items():
        lst.sort(key=lambda x: x[0])
        top, d, run = lst[-1]
        if len(lst) > 1:
            print(f"  [{algo}] N={n} seed={seed}: {len(lst)} run folders, using {os.path.basename(d)} "
                  f"(max step {top:,.0f}); others max steps {[f'{x[0]:,.0f}' for x in lst[:-1]]}")
        chosen[seed] = (d, run)
    return chosen


def tail_mean(xy, last=100_000, fn=np.mean):
    if xy is None:
        return float("nan")
    s, v = xy
    sel = v[s >= s.max() - last]
    return float(fn(sel)) if len(sel) else float("nan")


def slope_per_100k(xy, window=300_000):
    s, v = xy
    sel = s >= s.max() - window
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(s[sel] / 1e5, v[sel], 1)[0])


def summarize(algo, n, seed, folder, run):
    s, v = run["rew"]
    row = {"algo": algo, "n": n, "seed": seed, "max_step": int(s.max()), "folder": os.path.basename(folder),
           "final_rew": tail_mean(run["rew"]), "slope_100k": slope_per_100k(run["rew"]),
           "peak_rew": float(np.max(v)), "peak_step": int(s[int(np.argmax(v))])}
    row["peak_drop"] = row["peak_rew"] - row["final_rew"]
    row["critic_med"] = tail_mean(run.get("critic"), fn=np.median)
    row["critic_max"] = float(np.max(run["critic"][1])) if "critic" in run else float("nan")
    row["ent_coef"] = tail_mean(run.get("ent_coef"))
    row["actor_loss"] = tail_mean(run.get("actor"))
    row["value_loss"] = tail_mean(run.get("vloss"))
    row["expl_var"] = tail_mean(run.get("expl_var"))
    row["kl"] = tail_mean(run.get("kl"))
    return row


def nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def f(x, spec=".2f"):
    return "   n/a" if not np.isfinite(x) else format(x, spec)


def main():
    rows = []
    curves = {}   # (algo, n, key) -> list of (steps, values)
    for algo in ("sac", "ppo"):
        if not os.path.isdir(SWEEPS[algo]):
            print(f"[{algo}] folder not found: {SWEEPS[algo]}")
            continue
        for n in NS:
            runs = find_runs(algo, n)
            for seed, (d, run) in sorted(runs.items()):
                rows.append(summarize(algo, n, seed, d, run))
                for key in ("rew", "critic", "ent_coef"):
                    if key in run:
                        curves.setdefault((algo, n, key), []).append(run[key])
            if not runs:
                print(f"[{algo}] N={n}: no readable runs")
            else:
                missing = {k for k in ("rew", "critic", "ent_coef", "vloss") if all(k not in r for _, r in runs.values())}
                tags = sorted(next(iter(runs.values()))[1]["_tags"])
                if algo == "sac" and ({"critic", "ent_coef"} & missing):
                    print(f"  [{algo}] N={n}: missing SAC tags. Available: {tags}")
                if algo == "ppo" and "vloss" in missing:
                    print(f"  [{algo}] N={n}: missing PPO tags. Available: {tags}")

    if not rows:
        raise SystemExit("no runs found")

    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    print("\n" + "=" * 112)
    print("  SAC, mean over seeds. Reward = training rollout $ per battery (stochastic policy). Critic loss only comparable across N.")
    print("=" * 112)
    print(f"  {'N':>3} {'runs':>4} {'steps':>9} {'final_rew':>10} {'slope/100K':>11} {'peak_drop':>10} "
          f"{'critic_med':>11} {'critic_max':>11} {'ent_coef':>9}")
    for n in NS:
        rs = [r for r in rows if r["algo"] == "sac" and r["n"] == n]
        if not rs:
            continue
        print(f"  {n:>3} {len(rs):>4} {min(r['max_step'] for r in rs):>9,} "
              f"{f(nanmean([r['final_rew'] for r in rs])):>10} {f(nanmean([r['slope_100k'] for r in rs])):>11} "
              f"{f(nanmean([r['peak_drop'] for r in rs])):>10} {f(nanmean([r['critic_med'] for r in rs]), '.3g'):>11} "
              f"{f(nanmean([r['critic_max'] for r in rs]), '.3g'):>11} {f(nanmean([r['ent_coef'] for r in rs]), '.3g'):>9}")

    print("\n" + "=" * 112)
    print("  PPO, mean over seeds (value loss is NOT comparable with SAC critic loss)")
    print("=" * 112)
    print(f"  {'N':>3} {'runs':>4} {'steps':>9} {'final_rew':>10} {'slope/100K':>11} {'peak_drop':>10} "
          f"{'value_loss':>11} {'expl_var':>9} {'approx_kl':>10}")
    for n in NS:
        rs = [r for r in rows if r["algo"] == "ppo" and r["n"] == n]
        if not rs:
            continue
        print(f"  {n:>3} {len(rs):>4} {min(r['max_step'] for r in rs):>9,} "
              f"{f(nanmean([r['final_rew'] for r in rs])):>10} {f(nanmean([r['slope_100k'] for r in rs])):>11} "
              f"{f(nanmean([r['peak_drop'] for r in rs])):>10} {f(nanmean([r['value_loss'] for r in rs]), '.3g'):>11} "
              f"{f(nanmean([r['expl_var'] for r in rs]), '.3f'):>9} {f(nanmean([r['kl'] for r in rs]), '.3g'):>10}")

    print("\n  Per-seed SAC detail:")
    print(f"  {'N':>3} {'seed':>4} {'final_rew':>10} {'slope/100K':>11} {'peak_rew':>9} {'@step':>9} {'critic_med':>11} {'ent_coef':>9}")
    for r in rows:
        if r["algo"] == "sac":
            print(f"  {r['n']:>3} {r['seed']:>4} {f(r['final_rew']):>10} {f(r['slope_100k']):>11} {f(r['peak_rew']):>9} "
                  f"{r['peak_step']:>9,} {f(r['critic_med'], '.3g'):>11} {f(r['ent_coef'], '.3g'):>9}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.arange(0, 1_000_001, 10_000)
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / (len(NS) - 1)) for i, n in enumerate(NS)}

    def mean_curve(lst):
        ys = [np.interp(grid, s, v, left=np.nan, right=np.nan) for s, v in lst]
        return np.nanmean(np.vstack(ys), axis=0)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2), dpi=130)
    for n in NS:
        if ("sac", n, "rew") in curves:
            ax[0].plot(grid, mean_curve(curves[("sac", n, "rew")]), color=colors[n], label=f"N={n}")
        if ("ppo", n, "rew") in curves:
            ax[1].plot(grid, mean_curve(curves[("ppo", n, "rew")]), color=colors[n], label=f"N={n}")
        if ("sac", n, "critic") in curves:
            ax[2].plot(grid, mean_curve(curves[("sac", n, "critic")]), color=colors[n], label=f"N={n}")
        if ("sac", n, "ent_coef") in curves:
            ax[3].plot(grid, mean_curve(curves[("sac", n, "ent_coef")]), color=colors[n], label=f"N={n}")
    titles = ["SAC training reward ($/battery)", "PPO training reward ($/battery)",
              "SAC critic loss (log)", "SAC entropy coefficient (log)"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=10)
        a.set_xlabel("training steps")
        a.grid(alpha=0.25)
        a.legend(fontsize=8, frameon=False)
    ax[2].set_yscale("log")
    ax[3].set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"\n  saved: {OUT_CSV}\n  saved: {OUT_PNG}")
    print("  Reminder: ep_rew_mean averages the last 100 episodes, so it lags the real policy by about 29K steps.")


if __name__ == "__main__":
    main()