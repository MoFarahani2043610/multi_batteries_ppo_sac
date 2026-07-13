"""
Conclusive test: is cache/caiso_2024_2026_5min.parquet real CAISO data,
or is it actually the synthetic fallback generator's output?

We regenerate synthetic prices using the exact same formula/seed as
loader.py's make_synthetic_prices(), and compare directly to the cached file.
If they match (even approximately, or with matching statistical fingerprints),
the cache is synthetic, not real.
"""
import numpy as np
import pandas as pd

CACHE_PATH = "cache/caiso_2024_2026_5min.parquet"

# ---- Load the cached file ----
df = pd.read_parquet(CACHE_PATH)
cached_prices = df['lmp'].values.astype(np.float64)
print(f"Cached file: {len(cached_prices):,} rows")
print(f"Cached first 10 values: {cached_prices[:10]}")

# ---- Regenerate using loader.py's exact synthetic formula (market='caiso', seed=42) ----
_MARKET_STATS = {
    "caiso": {"mean": 52.0, "std": 38.0, "spike_prob": 0.008, "floor": -50.0},
}
def make_synthetic_prices(market, n_steps=210_240, seed=42):
    s = _MARKET_STATS[market]
    rng = np.random.default_rng(seed)
    price = s["mean"]
    prices = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        noise = rng.normal(0, s["std"] * 0.05)
        price += 0.02 * (s["mean"] - price) + noise
        step_in_day = t % 288
        angle = 2 * np.pi * step_in_day / 288
        cycle = s["std"] * 0.3 * (np.sin(angle - np.pi / 3) + 0.5)
        price += cycle * 0.1
        if rng.random() < s["spike_prob"]:
            price *= rng.uniform(3, 10)
        prices[t] = np.clip(price, s["floor"], 3000.0)
    return prices

synthetic = make_synthetic_prices("caiso", n_steps=len(cached_prices), seed=42)
print(f"\nRegenerated synthetic first 10 values: {synthetic[:10]}")

# ---- Compare ----
exact_match = np.allclose(cached_prices, synthetic, atol=1e-6)
print(f"\nEXACT MATCH (seed=42): {exact_match}")

if not exact_match:
    # Try a few other common seeds in case a different one was used
    for test_seed in [0, 1, 2, 42, 123, 2024, 2025, 2026]:
        s = make_synthetic_prices("caiso", n_steps=len(cached_prices), seed=test_seed)
        match = np.allclose(cached_prices, s, atol=1e-6)
        corr = np.corrcoef(cached_prices, s)[0, 1]
        print(f"  seed={test_seed}: exact_match={match}, correlation={corr:.4f}")

# ---- Structural fingerprint check (works regardless of exact seed match) ----
# Real market data has irregular noise; this synthetic formula produces
# a very smooth, highly autocorrelated signal with a fixed 288-step cycle.
# Check lag-288 autocorrelation (one full day) -- synthetic data will show
# near-perfect periodicity; real data will be much noisier.
def lag_autocorr(x, lag):
    x1 = x[:-lag]
    x2 = x[lag:]
    return np.corrcoef(x1, x2)[0, 1]

print(f"\nLag-288 (1-day) autocorrelation of cached data: {lag_autocorr(cached_prices, 288):.4f}")
print(f"Lag-1 (5-min) autocorrelation of cached data: {lag_autocorr(cached_prices, 1):.4f}")
print("(Real 5-min LMP data usually has lag-1 autocorr > 0.9 but much weaker,")
print(" noisier lag-288 autocorrelation than a deterministic sine-cycle formula.)")