"""
audit_m3_data_provenance.py

Figures out which M3 scalability-sweep runs (N=1,2,5,10,20) could possibly
have used the REAL CAISO cache vs. the FAKE one, using filesystem timestamps
as forensic evidence (there is no other record, since the results-log system
didn't exist yet when these ran).

Logic:
  - cache/caiso_2024_2026_5min_FAKE_BACKUP.parquet is the renamed-away fake
    file. Its timestamp tells us roughly when the fake data was originally
    downloaded/created (rename preserves mtime on the same volume).
  - The real file (now at cache/caiso_2024_2026_5min.parquet, originally
    downloaded as cache/caiso_REAL_verified_2024_2026_5min.parquet before
    the rename) has a creation/modified time telling us when real data
    first existed on disk.
  - Any M3 run whose log folder was created BEFORE the real file existed
    could ONLY have used fake data — the real file didn't exist yet.
  - Any M3 run whose log folder was created AFTER the real file existed is
    PROBABLE real data, but not certain — only certain if you know you did
    the rename immediately after downloading and didn't retrain with the
    old fake file still active. Flagged as "check manually" either way.

Run from project root:
    python audit_m3_data_provenance.py
"""

import os
import glob
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_FAKE_BACKUP = [
    "cache/caiso_2024_2026_5min_FAKE_BACKUP.parquet",
]
CANDIDATE_REAL = [
    "cache/caiso_REAL_verified_2024_2026_5min.parquet",  # may no longer exist if renamed away
    "cache/caiso_2024_2026_5min.parquet",                  # current active file (should be real now)
]

M3_LOG_GLOBS = [
    "m3_logs/*/N*/*seed*",
    "m3_logs/*/N*/seed_*",
    "m3_logs/*_sweep/N*/*",
]


def fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def file_times(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    st = os.stat(full)
    return {
        "path": path,
        "size_mb": round(st.st_size / 1e6, 1),
        "modified": fmt(st.st_mtime),
        "created": fmt(st.st_ctime),  # on Windows, st_ctime IS creation time
    }


def main():
    print("=" * 70)
    print("CACHE FILE TIMESTAMPS")
    print("=" * 70)
    for p in CANDIDATE_FAKE_BACKUP + CANDIDATE_REAL:
        info = file_times(p)
        if info:
            print(f"  {info['path']}")
            print(f"    size: {info['size_mb']} MB  created: {info['created']}  modified: {info['modified']}")
        else:
            print(f"  {p}  -> NOT FOUND")
    print()

    real_info = file_times("cache/caiso_REAL_verified_2024_2026_5min.parquet") or file_times("cache/caiso_2024_2026_5min.parquet")
    if real_info is None:
        print("Could not find any real-data cache file to use as a cutoff. Stopping.")
        return
    cutoff_str = real_info["created"]
    cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d %H:%M:%S")
    print(f"Using real-data file creation time as cutoff: {cutoff_str}")
    print("(Any M3 run folder created before this could ONLY have used fake data.)\n")

    print("=" * 70)
    print("M3 SWEEP LOG FOLDERS")
    print("=" * 70)

    found_any = False
    rows = []
    for pattern in M3_LOG_GLOBS:
        for path in glob.glob(os.path.join(ROOT, pattern)):
            if not os.path.isdir(path):
                continue
            found_any = True
            st = os.stat(path)
            created = datetime.fromtimestamp(st.st_ctime)
            verdict = "FAKE (created before real data existed)" if created < cutoff else "CHECK MANUALLY (created after real data existed)"
            rows.append((path.replace(ROOT + os.sep, ""), created.strftime("%Y-%m-%d %H:%M:%S"), verdict))

    if not found_any:
        print("  No folders matched the expected m3_logs/*/N*/... patterns.")
        print("  Edit M3_LOG_GLOBS in this script to match your actual folder layout,")
        print("  e.g. run:  Get-ChildItem m3_logs -Recurse -Directory | Select FullName")
        print("  and tell me what the real structure looks like.")
        return

    rows.sort(key=lambda r: r[1])
    for path, created, verdict in rows:
        print(f"  {created}  {path}")
        print(f"      -> {verdict}")

    print()
    print("Rows marked FAKE are certain — do not use in the thesis, must be rerun.")
    print("Rows marked CHECK MANUALLY need you to confirm no stale fake file was")
    print("still active at that time (e.g. did you rerun anything between the")
    print("rename and re-verifying the swap worked?).")


if __name__ == "__main__":
    main()