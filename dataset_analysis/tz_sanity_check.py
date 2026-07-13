"""
Sanity check for the CAISO timestamp / timezone handling.
Prints raw (assumed-UTC) timestamps alongside their Pacific-Time conversion
for a few specific dates, and shows the hourly average for one representative
summer day, to confirm whether the evening demand peak appears where expected.
"""
import pandas as pd
import numpy as np

CACHE_PATH = "cache/caiso_2024_2026_5min.parquet"

df = pd.read_parquet(CACHE_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Show raw timestamp dtype / tz info
print("Raw timestamp dtype:", df['timestamp'].dtype)
print("Raw timestamp tz:", df['timestamp'].dt.tz)
print()

# Convert assuming the stored timestamp is UTC (per loader.py's tz_convert('UTC').tz_localize(None))
ts_utc = df['timestamp'].dt.tz_localize('UTC')
ts_pacific = ts_utc.dt.tz_convert('US/Pacific')

print("="*70)
print("Sample rows: raw (assumed UTC) vs. converted Pacific Time")
print("="*70)
sample_idx = [0, 50000, 100000, 150000, 200000]
for i in sample_idx:
    print(f"row {i}:  raw={df['timestamp'].iloc[i]}  ->  "
          f"UTC={ts_utc.iloc[i]}  ->  Pacific={ts_pacific.iloc[i]}")

print()
print("="*70)
print("Hourly average price on a specific summer day (2023-07-15) — Pacific Time")
print("="*70)
day_mask = (ts_pacific.dt.date == pd.Timestamp('2023-07-15').date())
day_df = df[day_mask].copy()
day_df['hour_pacific'] = ts_pacific[day_mask].dt.hour
hourly_summer = day_df.groupby('hour_pacific')['lmp'].mean()
print(hourly_summer)

print()
print("="*70)
print("Hourly average price on a specific summer day (2023-07-15) — RAW (no tz conversion)")
print("="*70)
day_mask_raw = (df['timestamp'].dt.date == pd.Timestamp('2023-07-15').date())
day_df_raw = df[day_mask_raw].copy()
day_df_raw['hour_raw'] = df['timestamp'][day_mask_raw].dt.hour
hourly_summer_raw = day_df_raw.groupby('hour_raw')['lmp'].mean()
print(hourly_summer_raw)