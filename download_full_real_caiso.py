"""
Download the FULL real CAISO 5-min LMP dataset (Sept 2024 - Sept 2026) from
CAISO's public OASIS API, in 2-day chunks, and save as the new, verified
cache file, replacing the previous (fraudulent/wrong) cache file.
"""
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

NODE = "TH_NP15_GEN-APND"
START_DATE = datetime(2024, 9, 1)
END_DATE = datetime(2026, 9, 24)

url = "http://oasis.caiso.com/oasisapi/SingleZip"

all_chunks = []
current = START_DATE

while current < END_DATE:
    chunk_end = min(current + timedelta(days=2), END_DATE)
    start_str = current.strftime("%Y%m%dT00:00-0000")
    end_str = chunk_end.strftime("%Y%m%dT00:00-0000")

    print(f"Downloading {current.date()} to {chunk_end.date()}...")

    params = {
        "queryname": "PRC_INTVL_LMP",
        "version": "3",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "market_run_id": "RTM",
        "node": NODE,
        "resultformat": "6",
    }

    try:
        resp = requests.get(url, params=params, timeout=120)
        retry_count = 0
        while resp.status_code == 429 and retry_count < 5:
            wait = 30 * (retry_count + 1)
            print(f"  Rate limited (429), waiting {wait}s before retry {retry_count+1}/5...")
            time.sleep(wait)
            resp = requests.get(url, params=params, timeout=120)
            retry_count += 1

        if resp.status_code != 200:
            print(f"  FAILED after retries (status {resp.status_code})")
            current = chunk_end
            continue

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
        if not csv_files:
            first_file = zf.namelist()[0]
            with zf.open(first_file) as f:
                content = f.read().decode('utf-8', errors='replace')
            print(f"  No CSV. Error: {content[:300]}")
            current = chunk_end
            time.sleep(15)
            continue
        csv_name = csv_files[0]
        with zf.open(csv_name) as f:
            df_chunk = pd.read_csv(f)

        df_chunk = df_chunk[df_chunk['LMP_TYPE'] == 'LMP'].copy()
        df_chunk['timestamp'] = pd.to_datetime(df_chunk['INTERVALSTARTTIME_GMT']).dt.tz_convert('UTC').dt.tz_localize(None)
        df_chunk = df_chunk[['timestamp', 'VALUE']].rename(columns={'VALUE': 'lmp'})
        df_chunk = df_chunk.drop_duplicates('timestamp').sort_values('timestamp')

        print(f"  Got {len(df_chunk)} rows")
        all_chunks.append(df_chunk)

    except Exception as e:
        print(f"  ERROR: {e}")

    current = chunk_end
    time.sleep(15)

print("\nCombining all chunks...")
df_all = pd.concat(all_chunks, ignore_index=True)
df_all = df_all.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

print(f"\nTotal rows downloaded: {len(df_all):,}")
print(f"Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
expected_days = (END_DATE - START_DATE).days
expected_rows = expected_days * 288
print(f"Expected rows for {expected_days} days: ~{expected_rows:,}")
print(f"Missing rows: {expected_rows - len(df_all):,}")

df_all.to_parquet("cache/caiso_REAL_verified_2024_2026_5min.parquet", index=False)
print("\nSaved to: cache/caiso_REAL_verified_2024_2026_5min.parquet")