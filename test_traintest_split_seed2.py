"""
Test: SAC N=1, seed 2, with a genuinely separated train/test split.
Training samples episodes ONLY from Jan 2022 - Sep 2023 (train window).
Evaluation samples episodes ONLY from Nov-Dec 2023 (held-out test window).
This directly tests whether current (leaked) results are inflated.
"""
import sys, os
sys.path.insert(0, 'env')
sys.path.insert(0, 'm2_agents')
sys.path.insert(0, 'data')

import numpy as np
import pandas as pd
from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from loader import load_prices, make_features
from m2_agents.sac import SACAgent

# ── load full dataset + compute split boundaries (same as dataset_analysis.py) ──
prices, timestamps = load_prices(market="caiso", cache_dir="cache")
features = make_features(timestamps)
ts = pd.to_datetime(timestamps)

last_date = ts[-1]
test_start = (last_date.replace(day=1) - pd.DateOffset(months=1))
val_start = (last_date.replace(day=1) - pd.DateOffset(months=2))

train_end_idx = int(np.searchsorted(ts, val_start))
test_start_idx = int(np.searchsorted(ts, test_start))

print(f"Train window: 0 to {train_end_idx} ({ts[0]} to {ts[train_end_idx-1]})")
print(f"Test window: {test_start_idx} to {len(prices)} ({ts[test_start_idx]} to {ts[-1]})")

# ── train env: restricted to train window only ──────────────────────
train_prices = prices[:train_end_idx]
train_features = features[:train_end_idx]
train_source = HistoricalPriceSource(train_prices, train_features, episode_len=288)

train_env = StorageArbitrageEnv(
    n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
    normalize_obs=True, price_ref=52.0, price_source=train_source,
)

# ── build config, train ───────────────────────────────────────────
config = {
    "env": {"n_batteries": 1, "dt_hours": 5/60, "degradation_penalty": 0.0,
            "normalize_obs": True, "price_ref": 52.0, "price_source": "caiso"},
    "training": {"total_timesteps": 1_000_000, "eval_freq": 10_000,
                 "eval_episodes": 10, "log_interval": 5},
    "sac": {"learning_rate": 0.001, "buffer_size": 500_000, "learning_starts": 1_000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "ent_coef": "auto", "target_entropy": "auto",
            "policy": "MlpPolicy", "policy_kwargs": {"net_arch": [400, 300]}},
    "logging": {"log_dir": "m3_logs/sac_n1_traintest_split_seed2", "tensorboard": True,
                "save_freq": 100_000, "verbose": 1},
}

agent = SACAgent(train_env, config, seed=2)
agent.train()

# ── eval env: restricted to held-out TEST window only ─────────────
test_prices = prices[test_start_idx:]
test_features = features[test_start_idx:]
test_source = HistoricalPriceSource(test_prices, test_features, episode_len=288)
test_env = StorageArbitrageEnv(
    n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
    normalize_obs=True, price_ref=52.0, price_source=test_source,
)

# manually evaluate on the TRUE held-out test window
returns = []
for ep in range(10):
    obs, info = test_env.reset(seed=2000 + ep)
    done = False
    ep_return = 0.0
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ep_return += reward
        done = terminated or truncated
    returns.append(ep_return)

returns = np.array(returns)
print(f"\n=== TRUE held-out test evaluation (no leakage) — SEED 2 ===")
print(f"Mean return: {returns.mean():.2f}")
print(f"Std return:  {returns.std():.2f}")
print(f"Individual returns: {returns.tolist()}")