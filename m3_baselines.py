"""
m3_baselines.py

Recompute Random, Threshold and LP-oracle baselines on EXACTLY the M3 setup:
  - same heterogeneous fleets   make_heterogeneous_fleet(n, seed)
  - same real CAISO prices      cache/caiso_2024_2026_5min.parquet
  - same 6-dim features         make_features(timestamps)
  - same eval episodes          env.reset(seed=1000 + ep), ep = 0..19
    (this is what agent.evaluate(n_episodes=20) uses in the sweep)

Everything is reported in dollars PER BATTERY per 288-step episode, the same
unit as the SAC/PPO eval returns (the env divides the reward by N). Fleet
totals are saved too.

Run from the project root (D:\\Project 2026\\env):
    python m3_baselines.py
    python m3_baselines.py --n-values 1 2 --seeds 0 1      (subset)

Output:
    m3_baselines_episodes.csv   one row per (N, seed, policy, episode)
    m3_baselines_summary.csv    one row per (N, policy)
Re-running skips (N, seed) pairs that are already in the episodes CSV.
"""

import os
# keep this job light so it does not slow the SAC sweep running next to it
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "env", "data", "m3_utils", "baselines"):
    p = os.path.join(ROOT, sub) if sub else ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

# lower our CPU priority on Windows so the sweep keeps its speed
if sys.platform == "win32":
    try:
        import ctypes
        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    except Exception:
        pass

from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from make_heterogeneous_fleet import make_heterogeneous_fleet
from loader import load_prices, make_features
from baselines.perfect_foresight import PerfectForesightLP, _get_episode_prices
from baselines.random_policy import RandomPolicy, run_episodes
from baselines.threshold_policy import ThresholdPolicy

EP_CSV = os.path.join(ROOT, "m3_baselines_episodes.csv")
SUM_CSV = os.path.join(ROOT, "m3_baselines_summary.csv")
EP_FIELDS = ["n", "seed", "policy", "episode", "total_profit", "per_battery_profit"]

N_EPISODES = 20
EVAL_SEED = 1000


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(n, seed, cfg, prices, features):
    """Same construction as make_env_for_n in m3_scripts/scale_sweep.py."""
    env_cfg = cfg.get("env", {})
    fleet = make_heterogeneous_fleet(n, seed=seed)
    source = HistoricalPriceSource(prices=prices, features=features, episode_len=288)
    return StorageArbitrageEnv(
        batteries=fleet,
        dt_hours=float(env_cfg.get("dt_hours", 5 / 60)),
        degradation_penalty=float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs=bool(env_cfg.get("normalize_obs", True)),
        price_ref=float(env_cfg.get("price_ref", 52.0)),
        price_source=source,
    )


class _Wrapped:
    """Threshold policy needs the env attached at reset."""
    def __init__(self, p, e):
        self._p = p
        self._e = e

    def reset(self):
        self._p.reset(env=self._e)

    def act(self, obs):
        return self._p.act(obs)


def lp_profits(env, n_episodes, eval_seed):
    solver = PerfectForesightLP(env)
    out = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=eval_seed + ep)
        prices_ep = _get_episode_prices(env)
        res = solver.solve(prices_ep, info["soc_mwh"])
        out.append(float(res["profit"]))
    return out


def done_pairs():
    done = set()
    if os.path.exists(EP_CSV):
        with open(EP_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((int(r["n"]), int(r["seed"])))
    return done


def append_rows(rows):
    new = not os.path.exists(EP_CSV)
    with open(EP_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EP_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_summary():
    if not os.path.exists(EP_CSV):
        return
    with open(EP_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((int(r["n"]), r["policy"]), []).append(r)
    order = {"Random": 0, "Threshold(25/50)": 1, "Threshold(q25/q75)": 2, "LP": 3}
    out = []
    for (n, pol), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        pb = np.array([float(x["per_battery_profit"]) for x in rs])
        tot = np.array([float(x["total_profit"]) for x in rs])
        seeds = sorted({int(x["seed"]) for x in rs})
        out.append({
            "n": n, "policy": pol, "n_seeds": len(seeds), "n_rows": len(rs),
            "mean_per_battery": round(float(pb.mean()), 4),
            "std_per_battery": round(float(pb.std()), 4),
            "mean_total": round(float(tot.mean()), 4),
        })
    with open(SUM_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("\n" + "=" * 78)
    print("  Dollars PER BATTERY per episode (same unit as SAC/PPO eval returns)")
    print("=" * 78)
    print(f"  {'N':>3}  {'policy':<20} {'seeds':>5} {'mean':>10} {'std':>9} {'fleet total':>12}")
    for r in out:
        print(f"  {r['n']:>3}  {r['policy']:<20} {r['n_seeds']:>5} "
              f"{r['mean_per_battery']:>10.2f} {r['std_per_battery']:>9.2f} {r['mean_total']:>12.2f}")
    print(f"\n  saved: {SUM_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("m3_configs", "sac_scaling.yaml"))
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("env", {}).get("price_source", "").lower() != "caiso":
        raise ValueError("config env.price_source must be 'caiso'")

    prices, timestamps = load_prices(market="caiso", cache_dir=os.path.join(ROOT, "cache"))
    features = make_features(timestamps)
    q_lo = float(np.percentile(prices, 25))
    q_hi = float(np.percentile(prices, 75))
    print(f"CAISO prices: {len(prices)} points, mean {np.mean(prices):.2f}, "
          f"q25 {q_lo:.2f}, q75 {q_hi:.2f}")

    already = done_pairs()
    t_all = time.time()

    for n in args.n_values:
        for seed in args.seeds:
            if (n, seed) in already:
                print(f"N={n} seed={seed}: already done, skipping")
                continue
            t0 = time.time()
            env = build_env(n, seed, cfg, prices, features)
            rows = []

            def add(policy, profits):
                for ep, p in enumerate(profits):
                    rows.append({
                        "n": n, "seed": seed, "policy": policy, "episode": ep,
                        "total_profit": round(float(p), 6),
                        "per_battery_profit": round(float(p) / n, 6),
                    })

            rp = RandomPolicy(env, seed=EVAL_SEED)
            r = run_episodes(rp, env, n_episodes=args.episodes, seed=EVAL_SEED)
            add("Random", r["profits"])

            tp = ThresholdPolicy(25.0, 50.0)
            t = run_episodes(_Wrapped(tp, env), env, n_episodes=args.episodes, seed=EVAL_SEED)
            add("Threshold(25/50)", t["profits"])

            tq = ThresholdPolicy(q_lo, q_hi)
            tqr = run_episodes(_Wrapped(tq, env), env, n_episodes=args.episodes, seed=EVAL_SEED)
            add("Threshold(q25/q75)", tqr["profits"])

            lp = lp_profits(env, args.episodes, EVAL_SEED)
            add("LP", lp)

            append_rows(rows)
            print(f"N={n:>2} seed={seed}: per-battery  "
                  f"random {np.mean(r['profits'])/n:7.2f} | "
                  f"thr25/50 {np.mean(t['profits'])/n:7.2f} | "
                  f"thr-q {np.mean(tqr['profits'])/n:7.2f} | "
                  f"LP {np.mean(lp)/n:7.2f}   ({time.time()-t0:.0f}s)")

    write_summary()
    print(f"\nTotal time: {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()