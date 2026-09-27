"""
Investigate Danial's claim: does the cached price file show negative
prices for close to half the day, in a suspiciously smooth sinusoidal
pattern (inconsistent with real market behavior)?
"""
import pandas as pd
import numpy as np

CACHE_PATH = "cache/caiso_2024_2026_5min.parquet"

df = pd.read_parquet(CACHE_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

prices = df['lmp'].values
timestamps = df['timestamp']

print("="*60)
print("NEGATIVE PRICE ANALYSIS")
print("="*60)
neg_frac = (prices < 0).mean()
print(f"Fraction of ALL 5-min steps with negative price: {neg_frac*100:.1f}%")

# Fraction of negative prices BY HOUR OF DAY (UTC, matching cache storage)
df['hour_utc'] = timestamps.dt.hour
neg_by_hour = df.groupby('hour_utc')['lmp'].apply(lambda x: (x < 0).mean() * 100)
print("\nFraction of negative-price steps, by UTC hour of day:")
print(neg_by_hour.round(1))

max_neg_hour_frac = neg_by_hour.max()
print(f"\nMax negative-price fraction in any single hour: {max_neg_hour_frac:.1f}%")

# Check smoothness: how much does the negative fraction vary between adjacent hours?
diffs = neg_by_hour.diff().dropna().abs()
print(f"\nMean absolute change between adjacent hours: {diffs.mean():.2f} percentage points")
print(f"Max absolute change between adjacent hours: {diffs.max():.2f} percentage points")
print("(A 'too smooth' / sinusoidal-looking pattern would have small, gradual")
print(" changes throughout; real market data usually has some hour-to-hour jaggedness.)")

# Print a few individual raw values for manual cross-check against public sources
print("\n" + "="*60)
print("SAMPLE RAW VALUES (for manual cross-check)")
print("="*60)
sample_idx = [0, 1000, 50000, 100000, 150000, 200000]
for i in sample_idx:
    print(f"  {timestamps.iloc[i]}  ->  ${prices[i]:.2f}/MWh")