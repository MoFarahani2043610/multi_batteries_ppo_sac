"""
Find the earliest date for which CAISO's PRC_INTVL_LMP report actually
has data, by testing a range of dates from 2022 through 2023.
"""
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta

NODE = "TH_NP15_GEN-APND"
url = "http://oasis.caiso.com/oasisapi/SingleZip"

test_dates = [
    datetime(2022, 1, 1),
    datetime(2022, 3, 1),
    datetime(2022, 6, 1),
    datetime(2022, 9, 1),
    datetime(2022, 12, 1),
    datetime(2023, 1, 1),
    datetime(2023, 3, 1),
    datetime(2023, 6, 1),
    datetime(2023, 6, 19),  # known-good date from earlier test
    datetime(2023, 9, 1),
    datetime(2023, 12, 1),
]

for start in test_dates:
    end = start + timedelta(days=1)
    params = {
        "queryname": "PRC_INTVL_LMP",
        "version": "3",
        "startdatetime": start.strftime("%Y%m%dT00:00-0000"),
        "enddatetime": end.strftime("%Y%m%dT00:00-0000"),
        "market_run_id": "RTM",
        "node": NODE,
        "resultformat": "6",
    }
    print(f"\nTesting {start.date()}...")
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429:
        print("  Rate limited, waiting 30s...")
        time.sleep(30)
        resp = requests.get(url, params=params, timeout=60)

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        has_csv = any(n.endswith('.csv') for n in zf.namelist())
        print(f"  Status: {resp.status_code}, Has CSV: {has_csv}")
        if not has_csv:
            first_file = zf.namelist()[0]
            with zf.open(first_file) as f:
                content = f.read().decode('utf-8', errors='replace')
            if 'ERR_DESC' in content:
                err_start = content.find('ERR_DESC')
                print(f"  Error snippet: {content[err_start:err_start+80]}")
    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(10)

print("\nDone.")