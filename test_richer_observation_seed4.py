"""
Richer observation experiment for N=1, seed 4: adds two causal,
price-history-based features to the observation, on top of the existing
SoC/price/time features:
  - p_bar_recent: rolling mean of the past 12 steps (1 hour) of price
  - dp_recent: price change over the past 12 steps (p_t - p_{t-12})

Both features are computed using ONLY past/current price data (t and
earlier), never future prices -- strictly causal.

Uses the SAME evaluation protocol as the original N=1 baseline (NOT the
held-out train/test split from the separate leakage-verification
experiment), so results are directly comparable to the $96.20 baseline.

Everything else (algorithm, network, learning rate, reward, training
steps, evaluation procedure) is held identical to the N=1 baseline;
only the seed changes across this script and its seed 2/3/4 counterparts.
"""
import sys, os
sys.path.insert(0, 'env')
sys.path.insert(0, 'm2_agents')
sys.path.insert(0, 'data')

import numpy as np
from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from loader import load_prices
from m2_agents.sac import SACAgent

ROLLING_WINDOW = 12  # 12 steps x 5 min = 1 hour
PRICE_REF = 52.0
SEED = 4

prices, timestamps = load_prices(market="caiso", cache_dir="cache")

def make_time_features_2dim(T):
    step_of_day = np.arange(T) % 288
    angle = 2 * np.pi * step_of_day / 288
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

base_features = make_time_features_2dim(len(prices))

price_flat = prices.flatten()
T = len(price_flat)

p_bar = np.full(T, np.nan, dtype=np.float32)
cumsum = np.cumsum(np.insert(price_flat, 0, 0.0))
for t in range(T):
    start = max(0, t - ROLLING_WINDOW + 1)
    p_bar[t] = (cumsum[t + 1] - cumsum[start]) / (t - start + 1)

dp = np.zeros(T, dtype=np.float32)
dp[ROLLING_WINDOW:] = price_flat[ROLLING_WINDOW:] - price_flat[:-ROLLING_WINDOW]

p_bar_norm = (p_bar / PRICE_REF).astype(np.float32)
dp_norm = (dp / PRICE_REF).astype(np.float32)

richer_features = np.column_stack([base_features, p_bar_norm, dp_norm]).astype(np.float32)
print(f"Richer feature matrix shape: {richer_features.shape}  (expected (T, 4))")

richer_source = HistoricalPriceSource(prices, richer_features, episode_len=288)
env = StorageArbitrageEnv(
    n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
    normalize_obs=True, price_ref=PRICE_REF, price_source=richer_source,
)

obs, _ = env.reset(seed=SEED)
print(f"Observation shape with richer features: {obs.shape}  (expected (6,))")

config = {
    "env": {"n_batteries": 1, "dt_hours": 5/60, "degradation_penalty": 0.0,
            "normalize_obs": True, "price_ref": PRICE_REF, "price_source": "caiso"},
    "training": {"total_timesteps": 1_000_000, "eval_freq": 10_000,
                 "eval_episodes": 10, "log_interval": 5},
    "sac": {"learning_rate": 0.001, "buffer_size": 500_000, "learning_starts": 1_000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "ent_coef": "auto", "target_entropy": "auto",
            "policy": "MlpPolicy", "policy_kwargs": {"net_arch": [400, 300]}},
    "logging": {"log_dir": "m3_logs/sac_n1_richer_obs_test_seed4", "tensorboard": True,
                "save_freq": 100_000, "verbose": 1},
}

agent = SACAgent(env, config, seed=SEED)

def _make_fresh_env_richer(self):
    fresh_source = HistoricalPriceSource(prices, richer_features, episode_len=288)
    return StorageArbitrageEnv(
        n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
        normalize_obs=True, price_ref=PRICE_REF, price_source=fresh_source,
    )

import types
agent._make_fresh_env = types.MethodType(_make_fresh_env_richer, agent)

agent.train()

returns = []
for ep in range(10):
    eval_source = HistoricalPriceSource(prices, richer_features, episode_len=288)
    eval_env = StorageArbitrageEnv(
        n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
        normalize_obs=True, price_ref=PRICE_REF, price_source=eval_source,
    )
    obs, info = eval_env.reset(seed=1000 + ep)
    done = False
    ep_return = 0.0
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_return += reward
        done = terminated or truncated
    returns.append(ep_return)

returns = np.array(returns)
print(f"\n=== Richer observation experiment (N=1, seed {SEED}) ===")
print(f"Mean return: {returns.mean():.2f}")
print(f"Std return:  {returns.std():.2f}")
print(f"Individual returns: {returns.tolist()}")