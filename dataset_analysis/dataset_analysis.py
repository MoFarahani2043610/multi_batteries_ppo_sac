"""
Dataset description script for the thesis Environment & Data chapter.
Computes summary statistics and generates distribution/time-series figures
for the real CAISO price data, and proposes a temporal train/val/test split.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE_PATH = "cache/caiso_2024_2026_5min.parquet"

df = pd.read_parquet(CACHE_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

prices = df['lmp'].values
timestamps = df['timestamp']

# ---- Summary statistics ----
print("="*60)
print("CAISO PRICE DATASET SUMMARY")
print("="*60)
print(f"Date range: {timestamps.iloc[0]} to {timestamps.iloc[-1]}")
print(f"Total rows (5-min steps): {len(df):,}")
print(f"Total days: {(timestamps.iloc[-1] - timestamps.iloc[0]).days}")
print(f"Mean price: ${prices.mean():.2f}/MWh")
print(f"Median price: ${np.median(prices):.2f}/MWh")
print(f"Std dev: ${prices.std():.2f}/MWh")
print(f"Min price: ${prices.min():.2f}/MWh")
print(f"Max price: ${prices.max():.2f}/MWh")
print(f"5th percentile: ${np.percentile(prices, 5):.2f}/MWh")
print(f"95th percentile: ${np.percentile(prices, 95):.2f}/MWh")
print(f"Negative price steps: {(prices < 0).sum():,} ({100*(prices < 0).mean():.2f}%)")
print(f"NaN count: {df['lmp'].isna().sum()}")

# ---- Proposed temporal split ----
# Use last full month as test, second-to-last full month as validation,
# everything else as training.
last_date = timestamps.iloc[-1]
test_start = (last_date.replace(day=1) - pd.DateOffset(months=1))
val_start = (last_date.replace(day=1) - pd.DateOffset(months=2))

train_mask = timestamps < val_start
val_mask = (timestamps >= val_start) & (timestamps < test_start)
test_mask = timestamps >= test_start

print("\n" + "="*60)
print("PROPOSED TRAIN / VALIDATION / TEST SPLIT")
print("="*60)
print(f"Train: {timestamps[train_mask].iloc[0]} to {timestamps[train_mask].iloc[-1]}  "
      f"({train_mask.sum():,} steps, {100*train_mask.mean():.1f}%)")
print(f"Val:   {timestamps[val_mask].iloc[0]} to {timestamps[val_mask].iloc[-1]}  "
      f"({val_mask.sum():,} steps, {100*val_mask.mean():.1f}%)")
print(f"Test:  {timestamps[test_mask].iloc[0]} to {timestamps[test_mask].iloc[-1]}  "
      f"({test_mask.sum():,} steps, {100*test_mask.mean():.1f}%)")

# ---- Figure 1: Price distribution histogram ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(prices, bins=100, color='steelblue', edgecolor='none')
ax.axvline(prices.mean(), color='darkorange', linestyle='--', label=f'Mean (${prices.mean():.1f})')
ax.axvline(np.median(prices), color='green', linestyle=':', label=f'Median (${np.median(prices):.1f})')
ax.set_xlabel('Price ($/MWh)')
ax.set_ylabel('Frequency (5-min steps)')
ax.set_title('CAISO TH_NP15 Price Distribution (5-min LMP)')
ax.set_xlim(-50, 300)
ax.legend()
plt.tight_layout()
plt.savefig('plotting/fig_dataset_price_distribution.png', dpi=150)
plt.close()
print("\nSaved: plotting/fig_dataset_price_distribution.png")

# ---- Figure 2: Full time series with train/val/test regions shaded ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(timestamps, prices, linewidth=0.3, color='steelblue', alpha=0.7)
ax.axvspan(val_start, test_start, color='orange', alpha=0.2, label='Validation')
ax.axvspan(test_start, last_date, color='red', alpha=0.2, label='Test')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($/MWh)')
ax.set_title('CAISO Price Time Series with Train/Val/Test Split')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend()
plt.tight_layout()
plt.savefig('plotting/fig_dataset_timeseries_split.png', dpi=150)
plt.close()
print("Saved: plotting/fig_dataset_timeseries_split.png")

# ---- Figure 3: Average daily price profile (hour of day, Pacific Time) ----
# CAISO timestamps are stored in UTC; convert to US/Pacific so "hour of day"
# reflects real California local time (e.g. evening demand peak, duck curve).
timestamps_pacific = timestamps.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
df['hour'] = timestamps_pacific.dt.hour
hourly = df.groupby('hour')['lmp'].agg(['mean', 'std'])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hourly.index, hourly['mean'], marker='o', color='steelblue')
ax.fill_between(hourly.index, hourly['mean']-hourly['std'], hourly['mean']+hourly['std'],
                 alpha=0.2, color='steelblue')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Price ($/MWh)')
ax.set_title('Average Daily Price Profile — Pacific Time (± 1 std)')
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig('plotting/fig_dataset_daily_profile.png', dpi=150)
plt.close()
print("Saved: plotting/fig_dataset_daily_profile.png")

# ---- Figure 4: Seasonal daily price profiles (summer vs winter, Pacific Time) ----
# The all-season average above blends CAISO's seasonal duck-curve variation;
# splitting by season shows the pattern more clearly (e.g. stronger midday
# solar dip and evening ramp in summer vs. winter).
df['month'] = timestamps_pacific.dt.month
summer_mask = df['month'].isin([6, 7, 8])
winter_mask = df['month'].isin([12, 1, 2])

summer_hourly = df[summer_mask].groupby('hour')['lmp'].mean()
winter_hourly = df[winter_mask].groupby('hour')['lmp'].mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(summer_hourly.index, summer_hourly.values, marker='o', color='darkorange', label='Summer (Jun-Aug)')
ax.plot(winter_hourly.index, winter_hourly.values, marker='s', color='steelblue', label='Winter (Dec-Feb)')
ax.set_xlabel('Hour of Day (Pacific Time)')
ax.set_ylabel('Price ($/MWh)')
ax.set_title('Seasonal Daily Price Profiles')
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plotting/fig_dataset_seasonal_profile.png', dpi=150)
plt.close()
print("Saved: plotting/fig_dataset_seasonal_profile.png")

print("\nDone.")