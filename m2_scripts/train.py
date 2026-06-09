"""Milestone 2 training script"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_ENV = os.path.join(_ROOT, 'env')
for p in [_ROOT, _ENV]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
    from storage_arbitrage_env import StorageArbitrageEnv
    env_cfg = config.get("env", {})
    return StorageArbitrageEnv(
        n_batteries=int(env_cfg.get("n_batteries", 1)),
        dt_hours=float(env_cfg.get("dt_hours", 5/60)),
        degradation_penalty=float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs=bool(env_cfg.get("normalize_obs", True)),
        price_ref=float(env_cfg.get("price_ref", 50.0)),
    )


def train_one_seed(algo: str, config: dict, seed: int) -> dict:
    env = make_env(config)
    
    if algo == "sac":
        from m2_agents.sac import SACAgent
        agent = SACAgent(env, config, seed=seed)
    elif algo == "ppo":
        from m2_agents.ppo import PPOAgent
        agent = PPOAgent(env, config, seed=seed)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    
    t0 = time.perf_counter()
    agent.train()
    elapsed = time.perf_counter() - t0
    
    print(f"\nEvaluating seed {seed} ...")
    eval_results = agent.evaluate(n_episodes=20, deterministic=True)
    eval_results["seed"] = seed
    eval_results["train_time"] = elapsed
    eval_results["algo"] = algo
    
    print(f"  Mean return : ${eval_results['mean_return']:.2f}")
    print(f"  Train time  : {elapsed/60:.1f} min")
    
    return eval_results


def run_all_seeds(algo: str, config: dict, seeds: list) -> list:
    all_results = []
    for seed in seeds:
        print(f"\n{'='*52}")
        print(f"  {algo.upper()}  seed {seed}")
        print(f"{'='*52}")
        res = train_one_seed(algo, config, seed)
        all_results.append(res)
    return all_results


def save_results(results: list, config: dict, algo: str):
    log_dir = Path(config.get("logging", {}).get("log_dir", f"m2_logs/{algo}_n1"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    output = {"algo": algo, "results": results}
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    returns = [r["mean_return"] for r in results]
    print(f"\n{'='*52}")
    print(f"  {algo.upper()} — Summary")
    print(f"{'='*52}")
    print(f"  Seeds run       : {len(results)}")
    print(f"  Mean return     : ${np.mean(returns):>8.2f} ± {np.std(returns):.2f}")
    print(f"  Min / Max       : ${np.min(returns):.2f} / ${np.max(returns):.2f}")
    print(f"  Target (70%LP)  : $354.99  (70% of $506.71)")
    status = "PASS" if np.mean(returns) >= 355 else "below target"
    print(f"  Acceptance      : {status}")
    print(f"{'='*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2 training")
    parser.add_argument("--algo", required=True, choices=["sac", "ppo"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.all_seeds:
        n_seeds = int(config.get("training", {}).get("n_seeds", 5))
        seeds = list(range(n_seeds))
    else:
        seeds = [args.seed]
    
    if len(seeds) == 1:
        results = [train_one_seed(args.algo, config, seeds[0])]
    else:
        results = run_all_seeds(args.algo, config, seeds)
    
    save_results(results, config, args.algo)
    
    print("\nTo view TensorBoard:")
    log_parent = config.get("logging", {}).get("log_dir", f"m2_logs/{args.algo}_n1")
    print(f"  tensorboard --logdir {log_parent}")