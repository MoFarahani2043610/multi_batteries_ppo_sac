"""
m2_scripts/train.py
====================
Milestone 2 — main training script.

Trains SAC or PPO on the storage arbitrage environment using a YAML config.
Runs multiple seeds, logs to TensorBoard, saves models and results.

Usage
-----
    # Train SAC with default config, 5 seeds
    python m2_scripts/train.py --algo sac --config m2_configs/sac_n1.yaml

    # Train PPO with a specific seed
    python m2_scripts/train.py --algo ppo --config m2_configs/ppo_n1.yaml --seed 2

    # Train both algorithms, all seeds, then compare
    python m2_scripts/train.py --algo sac --config m2_configs/sac_n1.yaml --all-seeds
    python m2_scripts/train.py --algo ppo --config m2_configs/ppo_n1.yaml --all-seeds

    # After training, view TensorBoard:
    tensorboard --logdir m2_logs/
"""

import sys, os

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_ENV  = os.path.join(_ROOT, 'env')
for p in [_ROOT, _ENV]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import time
import numpy as np
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(config: dict):
    """Build the environment from the config dict.

    Supports two price sources via the config 'env.price_source' key:
      - 'synthetic' (default) : SyntheticPriceSource — fast, no data needed
      - 'caiso'               : HistoricalPriceSource — real CAISO market data
    """
    from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource, SyntheticPriceSource
    env_cfg      = config.get("env", {})
    price_source = env_cfg.get("price_source", "synthetic").lower()

    if price_source == "caiso":
        # ── real CAISO data ──────────────────────────────────────────
        import sys, os
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        _DATA = os.path.join(_ROOT, 'data')
        if _DATA not in sys.path:
            sys.path.insert(0, _DATA)
        from loader import load_prices, make_features
        import numpy as np

        caiso_path = env_cfg.get("caiso_path", "cache/caiso_2024_2026_5min.parquet")
        print(f"  Loading real CAISO data from: {caiso_path}")
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
        print(f"  CAISO data loaded: {len(prices):,} timesteps, "
              f"mean=${prices.mean():.1f}/MWh")
    else:
        # ── synthetic prices (default) ───────────────────────────────
        source = SyntheticPriceSource()

    return StorageArbitrageEnv(
        n_batteries         = int(env_cfg.get("n_batteries", 1)),
        dt_hours            = float(env_cfg.get("dt_hours", 5/60)),
        degradation_penalty = float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs       = bool(env_cfg.get("normalize_obs", True)),
        price_ref           = float(env_cfg.get("price_ref", 52.0)),
        price_source        = source,
    )


def get_baselines(config: dict) -> dict:
    """
    Compute baseline returns (random, threshold, LP) for reference lines
    on the learning curve plot.
    Returns dict: {random: float, threshold: float, lp: float}
    """
    print("\nComputing baseline reference values ...")
    try:
        from baselines.random_policy    import RandomPolicy,    run_episodes
        from baselines.threshold_policy import ThresholdPolicy
        from baselines.perfect_foresight import PerfectForesightLP, _get_episode_prices
    except ModuleNotFoundError:
        from random_policy    import RandomPolicy,    run_episodes
        from threshold_policy import ThresholdPolicy
        from perfect_foresight import PerfectForesightLP, _get_episode_prices

    env  = make_env(config)
    N_EP = 10

    # Random
    rp = RandomPolicy(env, seed=0)
    r  = run_episodes(rp, env, n_episodes=N_EP, seed=0)
    random_ret = r["mean_profit"]

    # Threshold
    class _W:
        def __init__(self, p, e): self._p = p; self._e = e
        def reset(self):          self._p.reset(env=self._e)
        def act(self, obs):       return self._p.act(obs)
    tp = ThresholdPolicy()
    t  = run_episodes(_W(tp, env), env, n_episodes=N_EP, seed=0)
    threshold_ret = t["mean_profit"]

    # LP
    lp_solver  = PerfectForesightLP(env)
    lp_profits = []
    for ep in range(5):
        obs_ep, info_ep = env.reset(seed=ep)
        prices_ep = _get_episode_prices(env)
        res_ep    = lp_solver.solve(prices_ep, info_ep["soc_mwh"])
        lp_profits.append(res_ep["profit"])
    lp_ret = float(np.mean(lp_profits))

    print(f"  Random    : ${random_ret:.2f}")
    print(f"  Threshold : ${threshold_ret:.2f}")
    print(f"  LP        : ${lp_ret:.2f}")
    return {"random": random_ret, "threshold": threshold_ret, "lp": lp_ret}


def train_one_seed(algo: str, config: dict, seed: int) -> dict:
    """Train one algorithm for one seed. Returns evaluation results."""
    env = make_env(config)

    if algo == "sac":
        from m2_agents.sac import SACAgent
        agent = SACAgent(env, config, seed=seed)
    elif algo == "ppo":
        from m2_agents.ppo import PPOAgent
        agent = PPOAgent(env, config, seed=seed)
    else:
        raise ValueError(f"Unknown algo: {algo}. Choose 'sac' or 'ppo'.")

    t0 = time.perf_counter()
    agent.train()
    elapsed = time.perf_counter() - t0

    print(f"\nEvaluating seed {seed} ...")
    eval_results = agent.evaluate(n_episodes=20, deterministic=True)
    eval_results["seed"]       = seed
    eval_results["train_time"] = elapsed
    eval_results["algo"]       = algo

    print(f"  Mean return : ${eval_results['mean_return']:.2f}")
    print(f"  Std return  : ${eval_results['std_return']:.2f}")
    print(f"  Train time  : {elapsed/60:.1f} min")

    return eval_results


def run_all_seeds(algo: str, config: dict, seeds: list) -> list:
    """Train across all seeds and return list of eval result dicts."""
    all_results = []
    for seed in seeds:
        print(f"\n{'='*52}")
        print(f"  {algo.upper()}  seed {seed}/{seeds[-1]}")
        print(f"{'='*52}")
        res = train_one_seed(algo, config, seed)
        all_results.append(res)

    return all_results


def save_results(results: list, config: dict, algo: str, baselines: dict):
    """Save evaluation results and print summary table."""
    log_dir = Path(config.get("logging", {}).get("log_dir", f"m2_logs/{algo}_n1"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # save JSON
    output = {
        "algo":      algo,
        "config":    config,
        "baselines": baselines,
        "results":   results,
    }
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # print summary table
    returns = [r["mean_return"] for r in results]
    lp      = baselines.get("lp", 1.0)
    thr     = baselines.get("threshold", 0.0)

    print(f"\n{'='*52}")
    print(f"  {algo.upper()} — N=1 battery — Summary")
    print(f"{'='*52}")
    print(f"  Seeds run       : {len(results)}")
    print(f"  Mean return     : ${np.mean(returns):>8.2f} ± {np.std(returns):.2f}")
    print(f"  Min / Max       : ${np.min(returns):.2f} / ${np.max(returns):.2f}")
    print(f"  vs Random       : ${baselines.get('random', 0):.2f}")
    print(f"  vs Threshold    : ${thr:.2f}")
    print(f"  vs LP (ceiling) : ${lp:.2f}")
    pct = 100 * np.mean(returns) / lp if lp > 0 else 0
    print(f"  LP attainment   : {pct:.1f}%  (target >= 70%)")
    status = "PASS" if pct >= 70 else "below target"
    print(f"  Acceptance      : {status}")
    print(f"{'='*52}")

    return out_path


# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 2 — train SAC or PPO on the storage arbitrage env"
    )
    parser.add_argument(
        "--algo", required=True, choices=["sac", "ppo"],
        help="Algorithm to train"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. m2_configs/sac_n1.yaml)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="Train all 5 seeds from config (overrides --seed)"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline computation (faster, use if already computed)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps from config"
    )
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # override timesteps if given
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps
        print(f"Overriding total_timesteps = {args.timesteps:,}")

    # compute baselines
    if args.skip_baselines:
        baselines = {"random": 0.0, "threshold": 0.0, "lp": 1.0}
    else:
        baselines = get_baselines(config)

    # determine seeds
    if args.all_seeds:
        n_seeds = int(config.get("training", {}).get("n_seeds", 5))
        seeds   = list(range(n_seeds))
    else:
        seeds = [args.seed]

    # train
    if len(seeds) == 1:
        results = [train_one_seed(args.algo, config, seeds[0])]
    else:
        results = run_all_seeds(args.algo, config, seeds)

    # save and summarise
    save_results(results, config, args.algo, baselines)

    print("\nTo view TensorBoard:")
    log_parent = config.get("logging", {}).get("log_dir", f"m2_logs/{args.algo}_n1")
    print(f"  tensorboard --logdir {log_parent}")
    print(f"  then open http://localhost:6006 in your browser")