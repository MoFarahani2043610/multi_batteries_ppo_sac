
"""
Sanity check for the richer-observation experiment: verify the rolling
mean and price-change features are computed correctly and causally
(only using past/current data) BEFORE launching the full 1M-step run.
"""
import sys
sys.path.insert(0, 'env')
sys.path.insert(0, 'data')

import numpy as np
from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
from loader import load_prices

ROLLING_WINDOW = 12
PRICE_REF = 52.0

prices, timestamps = load_prices(market="caiso", cache_dir="cache")

# Use the 2-dim sin/cos time-of-day feature (matching every other
# experiment in this thesis), NOT loader.py's 6-dim make_features()
# (which also includes day-of-week/day-of-year and was never used
# elsewhere in this thesis -- see Chapter 2's observation-space note).
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

print("="*70)
print("MANUAL VERIFICATION")
print("="*70)

# t=0: mean should equal p0, delta should be 0
print(f"\nt=0:")
print(f"  p_0 = {price_flat[0]:.4f}")
print(f"  computed p_bar[0] = {p_bar[0]:.4f}  (should equal p_0)")
print(f"  computed dp[0] = {dp[0]:.4f}  (should be 0)")
assert np.isclose(p_bar[0], price_flat[0]), "FAIL: p_bar[0] should equal p_0"
assert dp[0] == 0.0, "FAIL: dp[0] should be 0"

# t=1: mean should equal mean(p0,p1), delta should be 0
manual_mean_1 = np.mean(price_flat[0:2])
print(f"\nt=1:")
print(f"  manual mean(p0,p1) = {manual_mean_1:.4f}")
print(f"  computed p_bar[1] = {p_bar[1]:.4f}  (should match)")
print(f"  computed dp[1] = {dp[1]:.4f}  (should be 0)")
assert np.isclose(p_bar[1], manual_mean_1), "FAIL: p_bar[1] mismatch"
assert dp[1] == 0.0, "FAIL: dp[1] should be 0"

# t=11: mean should equal mean(p0..p11) (first full 12-window), delta still 0
manual_mean_11 = np.mean(price_flat[0:12])
print(f"\nt=11:")
print(f"  manual mean(p0..p11) = {manual_mean_11:.4f}")
print(f"  computed p_bar[11] = {p_bar[11]:.4f}  (should match)")
print(f"  computed dp[11] = {dp[11]:.4f}  (should be 0, window not yet 12 back)")
assert np.isclose(p_bar[11], manual_mean_11), "FAIL: p_bar[11] mismatch"
assert dp[11] == 0.0, "FAIL: dp[11] should be 0"

# t=12: mean should equal mean(p1..p12), delta should be p12 - p0
manual_mean_12 = np.mean(price_flat[1:13])
manual_dp_12 = price_flat[12] - price_flat[0]
print(f"\nt=12:")
print(f"  manual mean(p1..p12) = {manual_mean_12:.4f}")
print(f"  computed p_bar[12] = {p_bar[12]:.4f}  (should match)")
print(f"  manual delta p12-p0 = {manual_dp_12:.4f}")
print(f"  computed dp[12] = {dp[12]:.4f}  (should match)")
assert np.isclose(p_bar[12], manual_mean_12), "FAIL: p_bar[12] mismatch"
assert np.isclose(dp[12], manual_dp_12), "FAIL: dp[12] mismatch"

# t=100: general causality check -- p_bar[100] must not depend on any price after index 100
manual_mean_100 = np.mean(price_flat[89:101])
print(f"\nt=100 (general causality check):")
print(f"  manual mean(p89..p100) = {manual_mean_100:.4f}")
print(f"  computed p_bar[100] = {p_bar[100]:.4f}  (should match)")
assert np.isclose(p_bar[100], manual_mean_100), "FAIL: p_bar[100] mismatch"

print("\nAll manual checks PASSED.")

# ── build richer feature matrix and verify observation dimension ──
p_bar_norm = (p_bar / PRICE_REF).astype(np.float32)
dp_norm = (dp / PRICE_REF).astype(np.float32)
richer_features = np.column_stack([base_features, p_bar_norm, dp_norm]).astype(np.float32)
print(f"\nRicher feature matrix shape: {richer_features.shape}  (expected (T, 4): sin, cos, p_bar, dp)")

richer_source = HistoricalPriceSource(prices, richer_features, episode_len=288)
env = StorageArbitrageEnv(
    n_batteries=1, dt_hours=5/60, degradation_penalty=0.0,
    normalize_obs=True, price_ref=PRICE_REF, price_source=richer_source,
)
obs, _ = env.reset(seed=0)
print(f"Observation shape: {obs.shape}  (expected (6,): SoC(1)+price(1)+sin(1)+cos(1)+p_bar(1)+dp(1))")
assert obs.shape == (6,), f"FAIL: expected obs shape (6,), got {obs.shape}"

print("\nAll checks PASSED. Safe to launch the full training run.")
