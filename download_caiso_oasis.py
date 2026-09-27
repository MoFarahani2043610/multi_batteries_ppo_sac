"""
Download real CAISO real-time 5-min LMP data directly from CAISO's public
OASIS API (no gridstatus/lxml dependency needed), and compare it against
the existing cached file for the same date range.

NOTE: CAISO's OASIS API returns a ZIP file containing a CSV. Query
parameters below target the PRC_INTVL_LMP report (real-time 5-min LMP).
If CAISO changes their API or this exact report name/version, you may
need to adjust queryname/version -- check http://oasis.caiso.com/oasisapi/
if this errors out.
"""
import requests
import zipfile
import io
import pandas as pd
import numpy as np

NODE = "TH_NP15_GEN-APND"
START = "20230620T00:00-0000"
END = "20230622T00:00-0000"   # short 2-day window first, to test quickly before a longer pull

url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    "queryname": "PRC_INTVL_LMP",
    "version": "3",
    "startdatetime": START,
    "enddatetime": END,
    "market_run_id": "RTM",
    "node": NODE,
    "resultformat": "6",  # CSV
}

print("Requesting data from CAISO OASIS API...")
print(f"URL: {url}")
print(f"Params: {params}")

resp = requests.get(url, params=params, timeout=60)
print(f"Status code: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")

if resp.status_code != 200:
    print("ERROR: request failed.")
    print(resp.text[:500])
    raise SystemExit(1)

try:
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    print(f"Files in zip: {zf.namelist()}")
    csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
    with zf.open(csv_name) as f:
        df_real = pd.read_csv(f)
except Exception as e:
    print(f"ERROR unzipping/parsing response: {e}")
    print("Raw response (first 1000 chars):")
    print(resp.content[:1000])
    raise SystemExit(1)

print(f"\nDownloaded shape: {df_real.shape}")
print(f"Columns: {list(df_real.columns)}")
print(df_real.head(10).to_string())

df_real.to_csv("real_data_check/caiso_oasis_raw.csv", index=False)
print("\nSaved raw download to real_data_check/caiso_oasis_raw.csv")
print("\nIf this looks correct, inspect the columns above and let's write the")
print("comparison-against-cache step next using the correct timestamp/price column names.")