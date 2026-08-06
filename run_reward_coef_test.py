"""
Reward-coefficient experiment for N=1: divides reward by price_ref (52.0)
using a Gymnasium RewardWrapper, without modifying storage_arbitrage_env.py.
Tests whether reward scale affects the N=1 plateau/oscillation, per
Danial's suggestion to experiment with reward coefficients.
"""
import sys, os
sys.path.insert(0, 'env')
sys.path.insert(0, 'm2_agents')
sys.path.insert(0, 'data')

import gymnasium as gym
from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from loader import load_prices, make_features
from m2_agents.sac import SACAgent
import yaml


class RewardScaleWrapper(gym.RewardWrapper):
    """Divides the reward by a fixed scale factor."""
    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward / self.scale


# ── build env ──────────────────────────────────────────────────────
prices, timestamps = load_prices(market="caiso", cache_dir="cache")
features = make_features(timestamps)
source = HistoricalPriceSource(prices=prices, features=features, episode_len=288)

env = StorageArbitrageEnv(
    n_batteries=1,
    dt_hours=5/60,
    degradation_penalty=0.0,
    normalize_obs=True,
    price_ref=52.0,
    price_source=source,
)
env = RewardScaleWrapper(env, scale=52.0)

# ── build config (reuses the original N=1 SAC hyperparameters) ──────
config = {
    "env": {"n_batteries": 1, "dt_hours": 5/60, "degradation_penalty": 0.0,
            "normalize_obs": True, "price_ref": 52.0, "price_source": "caiso"},
    "training": {"total_timesteps": 1_000_000, "eval_freq": 10_000,
                 "eval_episodes": 10, "log_interval": 5},
    "sac": {"learning_rate": 0.001, "buffer_size": 500_000, "learning_starts": 1_000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "ent_coef": "auto", "target_entropy": "auto",
            "policy": "MlpPolicy", "policy_kwargs": {"net_arch": [400, 300]}},
    "logging": {"log_dir": "m3_logs/sac_n1_reward_coef_test", "tensorboard": True,
                "save_freq": 100_000, "verbose": 1},
}

agent = SACAgent(env, config, seed=0)
agent.train()

results = agent.evaluate(n_episodes=10, deterministic=True)
print(f"\nReward-coefficient test (N=1, seed=0, reward/52.0):")
print(f"  Mean return: {results['mean_return']:.2f}")
print(f"  Std return:  {results['std_return']:.2f}")
print(f"  (Note: this is the SCALED return -- multiply by 52.0 for real $)")
print(f"  Real $ equivalent mean: {results['mean_return']*52.0:.2f}")