import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.perfect_foresight import compare_all
from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from loader import load_prices, make_features

# FIX: previously this script created StorageArbitrageEnv(n_batteries=1)
# with NO price_source specified, which silently defaults to
# SyntheticPriceSource() -- meaning M1 baselines were being computed on
# fake data even when a real cache file was in place. This version
# explicitly loads and uses the real CAISO data.

def make_time_features_2dim(T):
    import numpy as np
    step_of_day = np.arange(T) % 288
    angle = 2 * 3.14159265358979 * step_of_day / 288
    return __import__('numpy').stack(
        [__import__('numpy').sin(angle), __import__('numpy').cos(angle)], axis=1
    ).astype('float32')

prices, timestamps = load_prices(market="caiso", cache_dir="cache")
features = make_time_features_2dim(len(prices))

price_source = HistoricalPriceSource(prices, features, episode_len=288)

env = StorageArbitrageEnv(
    n_batteries=1,
    dt_hours=5/60,
    degradation_penalty=0.0,
    normalize_obs=True,
    price_ref=float(prices.mean()),
    price_source=price_source,
)

print(f"Using REAL CAISO data: {len(prices):,} steps, mean=${prices.mean():.2f}/MWh")
print(f"Date range: {timestamps[0]} to {timestamps[-1]}")

compare_all(env, n_episodes=50)