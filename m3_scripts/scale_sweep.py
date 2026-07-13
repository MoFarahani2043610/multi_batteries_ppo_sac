"""
m3_scripts/scale_sweep.py
==========================
Milestone 3 — scalability sweep over N ∈ {1, 2, 5, 10, 20}.

Trains SAC and PPO at each N with 5 seeds each.
Logs results to m3_logs/ and saves a summary CSV.

Usage
-----
    # Full sweep (50 runs — run overnight)
    python m3_scripts/scale_sweep.py --config m3_configs/sac_scaling.yaml --algo sac
    python m3_scripts/scale_sweep.py --config m3_configs/ppo_scaling.yaml --algo ppo

    # Quick test — single N and seed
    python m3_scripts/scale_sweep.py --config m3_configs/sac_scaling.yaml --algo sac --n-values 1 2 --seeds 0

    # Both algorithms in one command
    python m3_scripts/scale_sweep.py --all

Note: this script always trains on real CAISO data. There is no synthetic
fallback — config.env.price_source must be explicitly "caiso", or the
script raises an error rather than silently substituting synthetic data.
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_ENV  = os.path.join(_ROOT, 'env')
_M3   = os.path.join(_ROOT, 'm3_utils')
for p in [_ROOT, _ENV, _M3]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json
import time
import numpy as np
import yaml
from pathlib import Path


# ── default N values and seeds from project spec ──────────────────
DEFAULT_N_VALUES = [1, 2, 5, 10, 20]
DEFAULT_SEEDS    = [0, 1, 2, 3, 4]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_for_n(n: int, config: dict, seed: int):
    """Build StorageArbitrageEnv with heterogeneous fleet for given N.

    Always uses real CAISO data. There is no synthetic-data fallback:
    config.env.price_source must be explicitly "caiso", or this raises
    an error rather than silently substituting synthetic data.
    """
    from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
    from make_heterogeneous_fleet import make_heterogeneous_fleet

    env_cfg      = config.get("env", {})
    fleet        = make_heterogeneous_fleet(n, seed=seed)
    price_source = env_cfg.get("price_source", "").lower()

    if price_source != "caiso":
        raise ValueError(
            f"config.env.price_source must be 'caiso', got {price_source!r}. "
            "Synthetic data is no longer supported — set price_source: caiso "
            "in the YAML config."
        )

    _DATA = os.path.join(_ROOT, 'data')
    if _DATA not in sys.path:
        sys.path.insert(0, _DATA)
    from loader import load_prices, make_features
    prices, timestamps = load_prices(
        market    = "caiso",
        cache_dir = os.path.join(_ROOT, "cache"),
    )
    features = make_features(timestamps)
    source   = HistoricalPriceSource(
        prices      = prices,
        features    = features,
        episode_len = 288,
    )

    return StorageArbitrageEnv(
        batteries           = fleet,
        dt_hours            = float(env_cfg.get("dt_hours", 5/60)),
        degradation_penalty = float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs       = bool(env_cfg.get("normalize_obs", True)),
        price_ref           = float(env_cfg.get("price_ref", 52.0)),
        price_source        = source,
    )


def get_lp_baseline(n: int, config: dict, seed: int, n_episodes: int = 5) -> float:
    """Compute LP upper bound for given N (used as denominator for LP attainment)."""
    try:
        from baselines.perfect_foresight import PerfectForesightLP, _get_episode_prices
    except ModuleNotFoundError:
        from perfect_foresight import PerfectForesightLP, _get_episode_prices

    env    = make_env_for_n(n, config, seed)
    solver = PerfectForesightLP(env)
    profits = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        prices    = _get_episode_prices(env)
        result    = solver.solve(prices, info["soc_mwh"])
        profits.append(result["profit"])

    return float(np.mean(profits))


def train_one_cell(algo: str, n: int, seed: int, config: dict) -> dict:
    """
    Train one (algo, N, seed) cell.
    Returns dict with all metrics for this cell.
    """
    env = make_env_for_n(n, config, seed)

    # ── log directory ─────────────────────────────────────────────
    log_base = config.get("logging", {}).get("log_dir", f"m3_logs/{algo}_sweep")
    log_dir  = Path(log_base) / f"N{n}" / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── override n_batteries in config for the agent ──────────────
    cfg = dict(config)
    cfg["env"] = dict(config.get("env", {}))
    cfg["env"]["n_batteries"] = n
    cfg["logging"] = dict(config.get("logging", {}))
    cfg["logging"]["log_dir"] = str(log_dir.parent)

    # ── build and train agent ──────────────────────────────────────
    t0 = time.perf_counter()

    if algo == "sac":
        from m2_agents.sac import SACAgent
        agent = SACAgent(env, cfg, seed=seed)
    elif algo == "ppo":
        from m2_agents.ppo import PPOAgent
        agent = PPOAgent(env, cfg, seed=seed)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    agent.train()
    train_time = time.perf_counter() - t0

    # ── evaluate ──────────────────────────────────────────────────
    eval_results = agent.evaluate(n_episodes=20, deterministic=True)

    # ── collect constraint violation metrics from eval episodes ───
    violation_rates = []
    soc_sat_rates   = []
    power_hit_rates = []
    eval_env = make_env_for_n(n, config, seed)
    for ep in range(20):
        obs, info = eval_env.reset(seed=2000 + ep)
        done = False
        while not done:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        violation_rates.append(info["violation_rate"])
        soc_sat_rates.append(info["soc_saturation_events"] / max(info["step_idx"], 1))
        power_hit_rates.append(info["power_limit_hits"] / max(info["step_idx"], 1))

    # ── LP baseline for this cell ─────────────────────────────────
    print(f"  Computing LP baseline for N={n} seed={seed}...")
    lp_return = get_lp_baseline(n, config, seed, n_episodes=5)

    # ── compute metrics ───────────────────────────────────────────
    mean_return = eval_results["mean_return"]
    lp_fraction = mean_return / lp_return if lp_return > 0 else 0.0

    result = {
        "algo":                algo,
        "n_batteries":         n,
        "seed":                seed,
        "mean_return":         mean_return,
        "std_return":          eval_results["std_return"],
        "min_return":          eval_results["min_return"],
        "max_return":          eval_results["max_return"],
        "lp_return":           lp_return,
        "lp_fraction":         lp_fraction,
        "train_time":          train_time,
        "log_dir":             str(log_dir),
        # Milestone 3 constraint violation metrics
        "violation_rate":      float(np.mean(violation_rates)),
        "soc_saturation_rate": float(np.mean(soc_sat_rates)),
        "power_limit_rate":    float(np.mean(power_hit_rates)),
    }

    # save cell result
    with open(log_dir / "cell_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  N={n} seed={seed}: ${mean_return:.2f} ({lp_fraction*100:.1f}% LP) "
          f"in {train_time/60:.1f} min")

    return result


def run_sweep(algo: str, config: dict, n_values: list, seeds: list) -> list:
    """Run the full sweep and return all cell results."""
    all_results = []
    total_cells = len(n_values) * len(seeds)
    cell_idx    = 0

    for n in n_values:
        for seed in seeds:
            cell_idx += 1
            print(f"\n{'='*56}")
            print(f"  {algo.upper()}  N={n}  seed={seed}  "
                  f"[{cell_idx}/{total_cells}]")
            print(f"{'='*56}")

            result = train_one_cell(algo, n, seed, config)
            all_results.append(result)

    return all_results


def save_sweep_results(results: list, algo: str, config: dict) -> None:
    """Save results to JSON and print summary table."""
    log_base = config.get("logging", {}).get("log_dir", f"m3_logs/{algo}_sweep")
    out_dir  = Path(log_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save full JSON
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # save CSV for easy analysis
    import csv
    csv_path = out_dir / "sweep_results.csv"
    fields   = ["algo", "n_batteries", "seed", "mean_return", "std_return",
                 "lp_return", "lp_fraction", "train_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # print summary table grouped by N
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'N':>4}  {'Mean Return':>14}  {'Std':>8}  {'LP %':>8}  {'LP $':>10}")
    print(f"  {'-'*56}")

    n_values = sorted(set(r["n_batteries"] for r in results))
    for n in n_values:
        cell_results = [r for r in results if r["n_batteries"] == n]
        returns  = [r["mean_return"] for r in cell_results]
        lp_frac  = [r["lp_fraction"] for r in cell_results]
        lp_base  = np.mean([r["lp_return"] for r in cell_results])
        print(f"  {n:>4}  "
              f"${np.mean(returns):>10.2f} ± {np.std(returns):<5.2f}  "
              f"{np.mean(lp_frac)*100:>6.1f}%  "
              f"${lp_base:>8.2f}")

    print(f"{'='*70}")
    print(f"\nResults saved to {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 3 — scalability sweep N ∈ {1,2,5,10,20}"
    )
    parser.add_argument("--algo", choices=["sac", "ppo"],
                        help="Algorithm to sweep")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--all", action="store_true",
                        help="Run both SAC and PPO (uses default configs)")
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=DEFAULT_N_VALUES,
                        help=f"Battery counts to sweep (default: {DEFAULT_N_VALUES})")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=DEFAULT_SEEDS,
                        help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    args = parser.parse_args()

    if args.all:
        for algo, cfg_path in [("sac", "m3_configs/sac_scaling.yaml"),
                                ("ppo", "m3_configs/ppo_scaling.yaml")]:
            config  = load_config(cfg_path)
            results = run_sweep(algo, config, args.n_values, args.seeds)
            save_sweep_results(results, algo, config)
    else:
        if not args.algo or not args.config:
            parser.error("--algo and --config are required (or use --all)")
        config  = load_config(args.config)
        results = run_sweep(args.algo, config, args.n_values, args.seeds)
        save_sweep_results(results, args.algo, config)