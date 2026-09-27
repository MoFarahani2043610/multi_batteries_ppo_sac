"""
Verify the newly downloaded cache/caiso_REAL_verified_2024_2026_5min.parquet
is genuine real market data: check basic statistics, autocorrelation
structure, and compare against the old (fraudulent) cache file to confirm
they are meaningfully different.
"""
import pandas as pd
import numpy as np

NEW_PATH = "cache/caiso_REAL_verified_2024_2026_5min.parquet"
OLD_PATH = "cache/caiso_2024_2026_5min.parquet"

df_new = pd.read_parquet(NEW_PATH)
df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
df_new = df_new.sort_values('timestamp').reset_index(drop=True)
prices_new = df_new['lmp'].values.astype(np.float64)

print("="*60)
print("NEW (VERIFIED REAL) DATASET SUMMARY")
print("="*60)
print(f"Rows: {len(prices_new):,}")
print(f"Date range: {df_new['timestamp'].min()} to {df_new['timestamp'].max()}")
print(f"Mean: ${prices_new.mean():.2f}/MWh")
print(f"Median: ${np.median(prices_new):.2f}/MWh")
print(f"Std: ${prices_new.std():.2f}/MWh")
print(f"Min: ${prices_new.min():.2f}/MWh")
print(f"Max: ${prices_new.max():.2f}/MWh")
print(f"Negative price steps: {(prices_new < 0).sum():,} ({100*(prices_new<0).mean():.1f}%)")
print(f"NaN count: {df_new['lmp'].isna().sum()}")

def lag_autocorr(x, lag):
    x1 = x[:-lag]
    x2 = x[lag:]
    return np.corrcoef(x1, x2)[0, 1]

print(f"\nLag-1 (5-min) autocorrelation: {lag_autocorr(prices_new, 1):.4f}")
print(f"Lag-288 (1-day) autocorrelation: {lag_autocorr(prices_new, 288):.4f}")

# ── Compare against the OLD (fraudulent) cache for the SAME overlapping dates ──
print("\n" + "="*60)
print("COMPARISON AGAINST OLD (SUSPECT) CACHE FILE")
print("="*60)
try:
    df_old = pd.read_parquet(OLD_PATH)
    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'])

    overlap_start = max(df_new['timestamp'].min(), df_old['timestamp'].min())
    overlap_end = min(df_new['timestamp'].max(), df_old['timestamp'].max())
    print(f"Overlapping date range: {overlap_start} to {overlap_end}")

    if overlap_start < overlap_end:
        new_window = df_new[(df_new['timestamp'] >= overlap_start) & (df_new['timestamp'] <= overlap_end)]
        old_window = df_old[(df_old['timestamp'] >= overlap_start) & (df_old['timestamp'] <= overlap_end)]
        merged = pd.merge(new_window, old_window, on='timestamp', suffixes=('_new', '_old'))
        if len(merged) > 0:
            corr = merged['lmp_new'].corr(merged['lmp_old'])
            diff = (merged['lmp_new'] - merged['lmp_old']).abs()
            print(f"Matched timestamps: {len(merged)}")
            print(f"Correlation (new real vs old suspect): {corr:.4f}")
            print(f"Mean abs difference: ${diff.mean():.2f}/MWh")
            print("(Low correlation confirms the old cache was indeed different/fake)")
        else:
            print("No overlapping timestamps to compare directly.")
    else:
        print("No date overlap between old and new files.")
except FileNotFoundError:
    print(f"Old cache file not found at {OLD_PATH} -- skipping comparison.")

print("\nDone.")