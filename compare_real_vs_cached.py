"""
Compare real CAISO LMP data (downloaded directly from OASIS API) against
the cached file, for the exact same timestamps, to determine definitively
whether the cache is genuine.
"""
import pandas as pd
import numpy as np

# ── load real downloaded data, filter to total LMP only ──────────
df_real = pd.read_csv("real_data_check/caiso_oasis_raw.csv")
df_real = df_real[df_real['LMP_TYPE'] == 'LMP'].copy()
df_real['timestamp'] = pd.to_datetime(df_real['INTERVALSTARTTIME_GMT']).dt.tz_convert('UTC').dt.tz_localize(None)
df_real = df_real[['timestamp', 'VALUE']].rename(columns={'VALUE': 'real_lmp'})
df_real = df_real.sort_values('timestamp').reset_index(drop=True)

print(f"Real data: {len(df_real)} rows, {df_real['timestamp'].min()} to {df_real['timestamp'].max()}")

# ── load cached file for the same window ──────────────────────────
df_cache = pd.read_parquet("cache/caiso_2024_2026_5min.parquet")
df_cache['timestamp'] = pd.to_datetime(df_cache['timestamp'])
mask = (df_cache['timestamp'] >= df_real['timestamp'].min()) & (df_cache['timestamp'] <= df_real['timestamp'].max())
df_cache_window = df_cache[mask][['timestamp', 'lmp']].rename(columns={'lmp': 'cached_lmp'})

print(f"Cached data in same window: {len(df_cache_window)} rows")

# ── merge and compare ──────────────────────────────────────────────
merged = pd.merge(df_cache_window, df_real, on='timestamp', how='inner')
print(f"\nMatched {len(merged)} timestamps")

if len(merged) == 0:
    print("\nNO MATCHES FOUND. Printing both timestamp ranges for debugging:")
    print("Real timestamps sample:", df_real['timestamp'].head(5).tolist())
    print("Cache timestamps sample:", df_cache_window['timestamp'].head(5).tolist())
else:
    diff = (merged['cached_lmp'] - merged['real_lmp']).abs()
    corr = merged['cached_lmp'].corr(merged['real_lmp'])
    print(f"\nCorrelation (cached vs. real): {corr:.4f}")
    print(f"Mean absolute difference: ${diff.mean():.4f}/MWh")
    print(f"Max absolute difference: ${diff.max():.4f}/MWh")
    print(f"Exact matches (diff < $0.01): {(diff < 0.01).mean()*100:.1f}%")
    print(f"\nSide-by-side comparison (first 15 rows):")
    print(merged.head(15).to_string())