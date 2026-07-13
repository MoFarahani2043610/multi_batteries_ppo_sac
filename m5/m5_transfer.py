"""
Milestone 5 — Cross-Market Generalization
Tests whether CAISO-trained policies transfer to ERCOT and PJM.
Three conditions: zero-shot, fine-tune 10K steps, train from scratch 10K steps.
"""

import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'env')
sys.path.insert(0, 'm2_agents')
sys.path.insert(0, 'data')

from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from stable_baselines3 import SAC, PPO
from loader import make_features   # FIX: use same 6-dim feature fn as M3 training

# ── Config ────────────────────────────────────────────────────────
N_BATTERIES    = 1
EVAL_EPISODES  = 10
FINETUNE_STEPS = 10_000
RESULTS_DIR    = "experiments/m5_transfer"
SEED           = 42
EPISODE_LEN    = 288   # 24 hours × 12 steps/hour

# Must match M2/M3 training config exactly (see THESIS key technical decisions)
DT_HOURS       = 5/60
NORMALIZE_OBS  = True
PRICE_REF      = 52.0
DEGRADATION    = 0.0

os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINTS = {
    'SAC': 'm3_logs/sac_sweep/N1/seed_2/final_model.zip',
    'PPO': 'm3_logs/ppo_sweep/N1/seed_2/final_model.zip',
}

CACHE = {
    'caiso': 'cache/caiso_2024_2026_5min.parquet',
    'ercot': 'cache/ercot_synthetic_5min.parquet',   # synthetic — real data corrupted
    'pjm':   'cache/pjm_synthetic_5min.parquet',     # synthetic — real data corrupted
}

SOURCE  = 'caiso'
# Note: ERCOT and PJM use synthetic prices calibrated to real market statistics
# (mean/std/spike_prob from 2022-2023 data) because real cached data was corrupted.
TARGETS = ['ercot', 'pjm']

# ── Load and clean price + timestamp data ─────────────────────────
print("Loading price data from cache...")
PRICE_DATA    = {}
FEATURE_DATA  = {}
for market, path in CACHE.items():
    df = pd.read_parquet(path)
    prices = df['lmp'].values.reshape(-1, 1).astype(np.float32)
    nan_count = np.isnan(prices).sum()
    if nan_count > 0:
        market_mean = float(np.nanmean(prices))
        prices = np.where(np.isnan(prices), market_mean, prices)
        print(f"  {market}: cleaned {nan_count} NaN values")
    print(f"  {market}: shape={prices.shape}, mean=${prices.mean():.2f}/MWh")
    PRICE_DATA[market] = prices

    # FIX: build the same 6-dim feature matrix used during M3 training
    # (sin/cos time-of-day, day-of-week, day-of-year), not a 2-dim stand-in.
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp']).to_numpy()
    else:
        # synthetic caches have no real calendar — synthesize one so
        # make_features() still produces a valid 6-dim matrix.
        timestamps = pd.date_range('2022-01-01', periods=len(prices), freq='5min').to_numpy()
    FEATURE_DATA[market] = make_features(timestamps)

# ── Exact battery config from M3 N=1 seed=2 ─────────────────────
# Recovered from make_heterogeneous_fleet(1, seed=2)
sys.path.insert(0, 'm3_scripts')
sys.path.insert(0, 'm3_utils')
from make_heterogeneous_fleet import make_heterogeneous_fleet
M3_FLEET = make_heterogeneous_fleet(1, seed=2)

# ── Make environment with matching feature dims ───────────────────
def make_env(market):
    prices   = PRICE_DATA[market]
    features = FEATURE_DATA[market]
    ps = HistoricalPriceSource(prices, features=features, episode_len=EPISODE_LEN)
    return StorageArbitrageEnv(
        batteries=M3_FLEET,
        price_source=ps,
        dt_hours=DT_HOURS,
        normalize_obs=NORMALIZE_OBS,
        price_ref=PRICE_REF,
        degradation_penalty=DEGRADATION,
    )

# Verify obs shape matches model (should be N + M + D = 1 + 1 + 6 = 8)
print("\nVerifying observation shapes...")
for market in ['caiso', 'ercot', 'pjm']:
    env = make_env(market)
    obs, _ = env.reset(seed=0)
    print(f"  {market}: obs shape={obs.shape}")

# ── Evaluation ────────────────────────────────────────────────────
def evaluate(model, market, n_episodes=EVAL_EPISODES):
    returns = []
    for ep in range(n_episodes):
        env = make_env(market)
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))

# ── Run one transfer experiment ───────────────────────────────────
def run_transfer(algo_name, target_market):
    result_path = os.path.join(RESULTS_DIR, f"{algo_name.lower()}_{target_market}.json")
    if os.path.exists(result_path):
        print(f"[SKIP] {algo_name} -> {target_market} already done")
        return

    print(f"\n{'='*60}")
    print(f"{algo_name}  source=CAISO  target={target_market.upper()}")
    print(f"{'='*60}")

    checkpoint = CHECKPOINTS[algo_name]
    AlgoClass  = SAC if algo_name == 'SAC' else PPO

    # Condition 1: Zero-shot
    print("  Zero-shot evaluation...")
    model_zs = AlgoClass.load(checkpoint)
    zs_mean, zs_std = evaluate(model_zs, target_market)
    print(f"  Zero-shot: ${zs_mean:.2f} +/- ${zs_std:.2f}")

    # Condition 2: Fine-tune 10K steps
    print("  Fine-tuning 10K steps...")
    model_ft = AlgoClass.load(checkpoint)
    ft_env = make_env(target_market)
    model_ft.set_env(ft_env)
    model_ft.learn(total_timesteps=FINETUNE_STEPS, reset_num_timesteps=False)
    ft_mean, ft_std = evaluate(model_ft, target_market)
    print(f"  Fine-tune: ${ft_mean:.2f} +/- ${ft_std:.2f}")

    # Condition 3: Train from scratch 10K steps
    print("  Training from scratch 10K steps...")
    scratch_env = make_env(target_market)
    if algo_name == 'SAC':
        model_sc = SAC('MlpPolicy', scratch_env,
                       learning_rate=1e-3, buffer_size=500_000,
                       learning_starts=1_000, batch_size=256,
                       policy_kwargs=dict(net_arch=[400,300]),
                       ent_coef='auto', seed=SEED, verbose=0)
    else:
        model_sc = PPO('MlpPolicy', scratch_env,
                       learning_rate=1e-3, n_steps=2048, batch_size=64,
                       ent_coef=0.001,
                       policy_kwargs=dict(net_arch=[400,300]),
                       seed=SEED, verbose=0)
    model_sc.learn(total_timesteps=FINETUNE_STEPS)
    sc_mean, sc_std = evaluate(model_sc, target_market)
    print(f"  Scratch:   ${sc_mean:.2f} +/- ${sc_std:.2f}")

    # CAISO reference
    print("  CAISO reference evaluation...")
    model_ref = AlgoClass.load(checkpoint)
    ref_mean, ref_std = evaluate(model_ref, SOURCE)
    print(f"  CAISO ref: ${ref_mean:.2f} +/- ${ref_std:.2f}")

    result = {
        "algo": algo_name,
        "source": SOURCE,
        "target": target_market,
        "n_batteries": N_BATTERIES,
        "finetune_steps": FINETUNE_STEPS,
        "eval_episodes": EVAL_EPISODES,
        "caiso_reference":            {"mean": ref_mean, "std": ref_std},
        "zero_shot":                  {"mean": zs_mean,  "std": zs_std},
        "fine_tune":                  {"mean": ft_mean,  "std": ft_std},
        "scratch":                    {"mean": sc_mean,  "std": sc_std},
        "transfer_gap_zs_vs_scratch": zs_mean - sc_mean,
        "transfer_gap_ft_vs_scratch": ft_mean - sc_mean,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Transfer gap (zero-shot vs scratch): ${zs_mean - sc_mean:.2f}")
    print(f"  Transfer gap (fine-tune vs scratch): ${ft_mean - sc_mean:.2f}")

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',   choices=['SAC','PPO','all'], default='all')
    parser.add_argument('--target', choices=['ercot','pjm','all'], default='all')
    args = parser.parse_args()

    algos   = ['SAC','PPO'] if args.algo   == 'all' else [args.algo]
    targets = TARGETS       if args.target == 'all' else [args.target]

    for algo in algos:
        for target in targets:
            run_transfer(algo, target)

    print("\nAll M5 runs complete.")
    print(f"Results saved to: {RESULTS_DIR}/")