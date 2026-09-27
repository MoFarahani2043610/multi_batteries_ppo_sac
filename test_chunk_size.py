"""
Quick test: find the maximum date range CAISO's OASIS API accepts for
PRC_INTVL_LMP (5-min resolution) before committing to the full,
multi-hour download with the correct chunk size.
"""
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta

NODE = "TH_NP15_GEN-APND"
url = "http://oasis.caiso.com/oasisapi/SingleZip"

test_windows_days = [1, 2, 3, 5, 7, 10, 14]

for days in test_windows_days:
    start = datetime(2022, 6, 1)
    end = start + timedelta(days=days)
    params = {
        "queryname": "PRC_INTVL_LMP",
        "version": "3",
        "startdatetime": start.strftime("%Y%m%dT00:00-0000"),
        "enddatetime": end.strftime("%Y%m%dT00:00-0000"),
        "market_run_id": "RTM",
        "node": NODE,
        "resultformat": "6",
    }
    print(f"\nTesting {days}-day window ({start.date()} to {end.date()})...")
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429:
        print("  Rate limited, waiting 30s...")
        time.sleep(30)
        resp = requests.get(url, params=params, timeout=60)

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        has_csv = any(n.endswith('.csv') for n in zf.namelist())
        print(f"  Status: {resp.status_code}, Has CSV: {has_csv}, Files: {zf.namelist()}")
    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(10)

print("\nDone. Use the largest window size above that returned a CSV successfully.")