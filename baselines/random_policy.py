import sys, os
sys.path.insert(0, r'D:\Project 2026\env\env')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
baselines/random_policy.py
===========================
Random policy — the absolute lower bound.

Every timestep: sample a uniformly random action from
[−P_dis_max,i, +P_chg_max,i] for each battery.
Ignores price, SoC, time — completely blind.

Purpose
-------
If a trained SAC/PPO agent cannot beat this, training is broken.
This is the floor every other policy must clear.

Run
---
    python baselines/random_policy.py --n-batteries 3 --n-episodes 20
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


class RandomPolicy:
    """
    Samples uniformly random actions, ignoring all observations.

    Parameters
    ----------
    env  : StorageArbitrageEnv
    seed : int or None — fixed seed for reproducibility
    """

    name = "Random"

    def __init__(self, env, seed=None):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass   # nothing to reset

    def act(self, obs):
        """Return a random action uniformly sampled from the action space."""
        return self.rng.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)


# ─────────────────────────────────────────────────────
#  Shared evaluation loop (used by all three baselines)
# ─────────────────────────────────────────────────────

def run_episodes(policy, env, n_episodes=10, seed=0):
    """
    Roll out `policy` for `n_episodes` full episodes.

    Returns
    -------
    dict
        profits      : list[float]  — cumulative profit per episode ($)
        mean_profit  : float
        std_profit   : float
        min_profit   : float
        max_profit   : float
        all_soc      : list of SoC arrays  (last episode only, for plotting)
        all_prices   : list of price arrays (last episode only, for plotting)
        all_actions  : list of action arrays (last episode only, for plotting)
        all_rewards  : list of reward floats (last episode only, for plotting)
    """
    profits = []

    # storage for the last episode — used by Task 4 visualisations
    last_soc     = []
    last_prices  = []
    last_actions = []
    last_rewards = []

    # reseed policy rng so two calls with same seed are identical
    if hasattr(policy, 'rng'):
        policy.rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy, 'reset'):
            policy.reset()

        ep_soc     = [info["soc_mwh"].copy()]
        ep_prices  = [info["prices"].copy()]
        ep_actions = []
        ep_rewards = []

        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_soc.append(info["soc_mwh"].copy())
            ep_prices.append(info["prices"].copy())
            ep_actions.append(action.copy())
            ep_rewards.append(reward)

        profits.append(info["cumulative_profit"])

        # keep last episode for plotting
        if ep == n_episodes - 1:
            last_soc     = ep_soc
            last_prices  = ep_prices
            last_actions = ep_actions
            last_rewards = ep_rewards

    profits = np.array(profits)
    return {
        "profits":     profits.tolist(),
        "mean_profit": float(profits.mean()),
        "std_profit":  float(profits.std()),
        "min_profit":  float(profits.min()),
        "max_profit":  float(profits.max()),
        "all_soc":     last_soc,
        "all_prices":  last_prices,
        "all_actions": last_actions,
        "all_rewards": last_rewards,
    }


# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from storage_arbitrage_env import StorageArbitrageEnv

    parser = argparse.ArgumentParser(description="Random policy — lower bound")
    parser.add_argument("--n-batteries", type=int, default=1)
    parser.add_argument("--n-episodes",  type=int, default=10)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    env    = StorageArbitrageEnv(n_batteries=args.n_batteries)
    policy = RandomPolicy(env, seed=args.seed)
    res    = run_episodes(policy, env,
                          n_episodes=args.n_episodes, seed=args.seed)

    print(f"\nRandom Policy — N={args.n_batteries} batteries, "
          f"{args.n_episodes} episodes")
    print(f"  Mean profit : ${res['mean_profit']:>8.4f}")
    print(f"  Std         : ${res['std_profit']:>8.4f}")
    print(f"  Min / Max   : ${res['min_profit']:.4f} / ${res['max_profit']:.4f}")
    print(f"\n  → This is the LOWER BOUND. Every trained agent must beat this.")
