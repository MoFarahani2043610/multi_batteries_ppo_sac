"""
data/loader.py
==============
Downloads 2 years of 5-minute Locational Marginal Prices (LMPs) from:
  - CAISO  : trading hub TH_NP15_GEN-APND
  - ERCOT  : trading hub HB_NORTH
  - PJM    : trading hub RTO

Uses the `gridstatus` library to query each ISO's public API.
Results are cached to local Parquet files so you never re-download.

Install:
    pip install gridstatus pandas pyarrow

Usage:
    from data.loader import load_prices, load_all_markets

    # single market, returns np.ndarray shape (T, 1)
    prices, timestamps = load_prices("caiso", years=2)

    # all three markets combined, shape (T, 3)
    prices, timestamps = load_all_markets(years=2)

    # plug straight into the environment
    from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPrices
    src = HistoricalPrices(prices, episode_len=288)
    env = StorageArbitrageEnv(n_batteries=5, price_source=src)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default cache directory — sits next to this file
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

# Market configuration — node names match gridstatus query keys
MARKET_CONFIG = {
    "caiso": {
        "iso":        "caiso",
        "location":   "TH_NP15_GEN-APND",
        "market":     "REAL_TIME_5_MIN",
        "label":      "CAISO TH_NP15",
        "price_col":  "lmp",               # column name in gridstatus response
    },
    "ercot": {
        "iso":        "ercot",
        "location":   "HB_NORTH",
        "market":     "REAL_TIME_SCED",
        "label":      "ERCOT HB_NORTH",
        "price_col":  "lmp",
    },
    "pjm": {
        "iso":        "pjm",
        "location":   "RTO",
        "market":     "REAL_TIME_5_MIN",
        "label":      "PJM RTO",
        "price_col":  "lmp",
    },
}

# Expected timestep in minutes
TIMESTEP_MIN = 5

# ─────────────────────────────────────────────────────────────────────────────
#  Low-level download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_chunk(
    iso_name: str,
    location: str,
    market: str,
    price_col: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Download one time chunk from gridstatus and return a clean DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp : UTC datetime64[ns], timezone-naive, 5-min aligned
        lmp       : float64, price in $/MWh
    """
    try:
        import gridstatus
    except ImportError:
        raise ImportError(
            "gridstatus is not installed. Run: pip install gridstatus"
        )

    iso_map = {
        "caiso": gridstatus.CAISO,
        "ercot": gridstatus.ERCOT,
        "pjm":   gridstatus.PJM,
    }
    iso = iso_map[iso_name]()

    logger.info(
        f"Downloading {iso_name.upper()} {location} "
        f"{start.date()} → {end.date()}"
    )

    df = iso.get_lmp(
        start=start,
        end=end,
        market=market,
        locations=[location],
    )

    # ── normalise columns ────────────────────────────────────────────────────
    df.columns = [c.lower().strip() for c in df.columns]

    # find timestamp column (gridstatus uses various names)
    ts_candidates = ["time", "interval_start", "datetime", "timestamp"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"Cannot find timestamp column in {iso_name} response. "
            f"Columns: {list(df.columns)}"
        )

    # find price column
    price_candidates = [price_col, "lmp", "price", "settlement_point_price"]
    p_col = next((c for c in price_candidates if c in df.columns), None)
    if p_col is None:
        raise ValueError(
            f"Cannot find price column in {iso_name} response. "
            f"Columns: {list(df.columns)}"
        )

    df = df[[ts_col, p_col]].copy()
    df.columns = ["timestamp", "lmp"]

    # ── clean timestamps ─────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    # drop duplicates and sort
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # ── clean prices ─────────────────────────────────────────────────────────
    df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")

    # clip extreme outliers (ERCOT 2021 crisis hit $9000/MWh — keep those,
    # but drop clear data errors above $10,000)
    df = df[df["lmp"].between(-500, 10_000)]
    df = df.dropna(subset=["lmp"])

    return df


def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to exact 5-minute grid, forward-filling small gaps (≤2 steps).

    Some ISOs return data at irregular intervals; this enforces the
    uniform 5-minute grid the environment expects.
    """
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="5min",
    )
    df = df.reindex(full_index)

    # forward-fill gaps of ≤2 steps (≤10 minutes), leave longer gaps as NaN
    df["lmp"] = df["lmp"].ffill(limit=2)

    # report how many NaNs remain
    n_nan = df["lmp"].isna().sum()
    if n_nan > 0:
        pct = 100 * n_nan / len(df)
        logger.warning(f"{n_nan} missing 5-min intervals after resampling ({pct:.1f}%)")

    df = df.dropna()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Parquet cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(market: str, years: int, cache_dir: Path) -> Path:
    """Return the Parquet file path for a given market and date range."""
    end_year  = datetime.now().year
    start_year = end_year - years
    return cache_dir / f"{market}_{start_year}_{end_year}_5min.parquet"


def _parquet_engine() -> str:
    """Return the best available parquet engine."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine.replace("fast", "fastparquet").replace("pyarrow", "pyarrow"))
            return engine
        except ImportError:
            continue
    # last resort — try anyway and let pandas raise a clear error
    return "auto"


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = _parquet_engine()
    try:
        df.to_parquet(path, index=False, engine=engine, compression="snappy")
    except Exception:
        # fallback: save as gzip-compressed CSV (no extra deps)
        csv_path = path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        path = csv_path
        logger.warning(f"Parquet unavailable — saved as CSV: {path}")
    size_mb = path.stat().st_size / 1e6
    logger.info(f"Saved cache → {path}  ({size_mb:.1f} MB, {len(df):,} rows)")


def _load_cache(path: Path) -> pd.DataFrame:
    # support both .parquet and .csv.gz fallback
    if not path.exists():
        csv_path = path.with_suffix(".csv.gz")
        if csv_path.exists():
            path = csv_path
    if str(path).endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip", parse_dates=["timestamp"])
    else:
        engine = _parquet_engine()
        df = pd.read_parquet(path, engine=engine)
    logger.info(f"Loaded cache ← {path}  ({len(df):,} rows)")
    return df


def _cache_is_valid(path: Path, min_rows: int = 100_000) -> bool:
    """
    A cache file is valid if it exists and has enough rows.
    2 years × 365 days × 288 steps/day ≈ 210,240 rows minimum.
    Checks both .parquet and .csv.gz fallback.
    """
    candidates = [path, path.with_suffix(".csv.gz")]
    for p in candidates:
        if not p.exists():
            continue
        try:
            if str(p).endswith(".csv.gz"):
                df = pd.read_csv(p, compression="gzip", usecols=["timestamp"])
            else:
                df = pd.read_parquet(p, columns=["timestamp"])
            return len(df) >= min_rows
        except Exception:
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main public API
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(
    market:    str,
    years:     int  = 2,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    fill_method: str = "ffill",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load 5-minute LMPs for one market.

    Downloads from gridstatus on first call; returns cached Parquet on
    subsequent calls. Cache is considered valid if it contains ≥100k rows.

    Parameters
    ----------
    market : str
        One of "caiso", "ercot", "pjm".
    years : int
        How many years of history to load (counting back from today).
        Default 2 (thesis requirement).
    cache_dir : Path
        Directory for Parquet cache files. Default: data/cache/.
    force_download : bool
        If True, ignore the cache and re-download. Default False.
    fill_method : str
        How to handle remaining NaNs after resampling.
        "ffill" (default) or "drop".

    Returns
    -------
    prices : np.ndarray, shape (T, 1), dtype float32
        LMP prices in $/MWh. Ready to pass to HistoricalPrices.
    timestamps : np.ndarray of datetime64[ns]
        UTC timestamps corresponding to each row.

    Example
    -------
    >>> prices, ts = load_prices("caiso", years=2)
    >>> print(prices.shape)   # (T, 1)
    >>> print(ts[0])          # 2023-01-01T00:00:00
    """
    market = market.lower().strip()
    if market not in MARKET_CONFIG:
        raise ValueError(
            f"Unknown market '{market}'. Choose from: {list(MARKET_CONFIG)}"
        )

    cfg   = MARKET_CONFIG[market]
    cache = _cache_path(market, years, Path(cache_dir))

    # ── try cache first ───────────────────────────────────────────────────────
    if not force_download and _cache_is_valid(cache):
        df = _load_cache(cache)
    else:
        # ── download in monthly chunks ────────────────────────────────────────
        end_dt   = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_dt = end_dt - timedelta(days=365 * years)

        chunks = []
        chunk_start = start_dt

        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=30), end_dt)
            try:
                chunk = _download_chunk(
                    iso_name  = cfg["iso"],
                    location  = cfg["location"],
                    market    = cfg["market"],
                    price_col = cfg["price_col"],
                    start     = chunk_start,
                    end       = chunk_end,
                )
                chunks.append(chunk)
            except Exception as e:
                logger.error(
                    f"Failed chunk {chunk_start.date()} → {chunk_end.date()}: {e}"
                )
            chunk_start = chunk_end

        if not chunks:
            raise RuntimeError(
                f"No data downloaded for {market}. "
                "Check your internet connection and gridstatus installation."
            )

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = _resample_to_5min(df)

        # fill any remaining gaps
        if fill_method == "ffill":
            df["lmp"] = df["lmp"].ffill().bfill()
        elif fill_method == "drop":
            df = df.dropna(subset=["lmp"])

        _save_cache(df, cache)

    # ── convert to numpy ──────────────────────────────────────────────────────
    prices     = df["lmp"].to_numpy(dtype=np.float32).reshape(-1, 1)
    timestamps = df["timestamp"].to_numpy()

    logger.info(
        f"{cfg['label']}: {len(prices):,} steps  "
        f"({timestamps[0]} → {timestamps[-1]})  "
        f"mean={prices.mean():.1f} $/MWh  "
        f"std={prices.std():.1f}  "
        f"min={prices.min():.1f}  max={prices.max():.1f}"
    )

    return prices, timestamps


def load_all_markets(
    years:     int  = 2,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    align: str = "inner",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CAISO, ERCOT, and PJM and align them to a common timestamp index.

    Parameters
    ----------
    years : int
        Years of history. Default 2.
    cache_dir : Path
        Cache directory.
    force_download : bool
        Re-download even if cache exists.
    align : str
        "inner" — keep only timesteps present in all three markets (default).
        "outer" — keep all timesteps, filling gaps with forward-fill.

    Returns
    -------
    prices : np.ndarray, shape (T, 3), dtype float32
        Columns: [CAISO, ERCOT, PJM] in $/MWh.
    timestamps : np.ndarray of datetime64[ns]
        Common UTC timestamps.

    Example
    -------
    >>> prices, ts = load_all_markets(years=2)
    >>> print(prices.shape)   # (T, 3)
    >>> # column 0 = CAISO, 1 = ERCOT, 2 = PJM
    """
    markets = ["caiso", "ercot", "pjm"]
    frames  = {}

    for mkt in markets:
        p, ts = load_prices(
            market         = mkt,
            years          = years,
            cache_dir      = cache_dir,
            force_download = force_download,
        )
        frames[mkt] = pd.Series(
            p.flatten(),
            index=pd.to_datetime(ts),
            name=mkt,
        )

    df = pd.concat(frames, axis=1)

    if align == "inner":
        df = df.dropna()
    else:
        df = df.ffill().bfill()

    prices     = df.to_numpy(dtype=np.float32)           # (T, 3)
    timestamps = df.index.to_numpy()

    logger.info(
        f"All markets aligned: {len(prices):,} common timesteps  "
        f"({timestamps[0]} → {timestamps[-1]})"
    )

    return prices, timestamps


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def make_features(timestamps: np.ndarray) -> np.ndarray:
    """
    Build a feature matrix f_t from timestamps alone.

    Returns float32 array shape (T, 6):
        col 0-1 : sin/cos of 5-min step within day   (time of day, circular)
        col 2-3 : sin/cos of day within week          (weekday pattern)
        col 4-5 : sin/cos of day within year          (seasonal pattern)

    All features are in [-1, 1] with no information leakage.

    Example
    -------
    >>> prices, ts = load_prices("caiso")
    >>> features   = make_features(ts)
    >>> print(features.shape)   # (T, 6)
    """
    ts = pd.to_datetime(timestamps)

    # steps within a day (288 per day at 5-min resolution)
    step_of_day  = (ts.hour * 60 + ts.minute) // 5
    angle_day    = 2 * np.pi * step_of_day / 288

    # day of week (0=Monday … 6=Sunday)
    angle_week   = 2 * np.pi * ts.dayofweek / 7

    # day of year
    angle_year   = 2 * np.pi * ts.dayofyear / 365

    features = np.column_stack([
        np.sin(angle_day),  np.cos(angle_day),
        np.sin(angle_week), np.cos(angle_week),
        np.sin(angle_year), np.cos(angle_year),
    ]).astype(np.float32)

    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Cache inspection utility
# ─────────────────────────────────────────────────────────────────────────────

def cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Print a summary of what is currently cached."""
    cache_dir = Path(cache_dir)
    files     = sorted(cache_dir.glob("*.parquet")) if cache_dir.exists() else []

    if not files:
        print(f"Cache empty — no Parquet files in {cache_dir}")
        return

    print(f"\nCache directory: {cache_dir}")
    print(f"{'File':<45} {'Rows':>10} {'Size MB':>9} {'From':<12} {'To':<12}")
    print("-" * 92)

    for f in files:
        try:
            df   = pd.read_parquet(f, columns=["timestamp", "lmp"])
            rows = len(df)
            mb   = f.stat().st_size / 1e6
            ts   = pd.to_datetime(df["timestamp"])
            print(
                f"{f.name:<45} {rows:>10,} {mb:>9.1f} "
                f"{str(ts.min().date()):<12} {str(ts.max().date()):<12}"
            )
        except Exception as e:
            print(f"{f.name:<45} ERROR: {e}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  —  python -m data.loader  or  python data/loader.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download and cache 5-min LMPs from CAISO, ERCOT, PJM"
    )
    parser.add_argument(
        "--market", default="all",
        choices=["caiso", "ercot", "pjm", "all"],
        help="Which market to download (default: all)",
    )
    parser.add_argument(
        "--years",  type=int, default=2,
        help="Years of history (default: 2)",
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR),
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if cache exists",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show cache contents and exit",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.info:
        cache_info(cache_dir)

    elif args.market == "all":
        prices, ts = load_all_markets(
            years          = args.years,
            cache_dir      = cache_dir,
            force_download = args.force,
        )
        print(f"\nAll markets: shape={prices.shape}  "
              f"({ts[0]} → {ts[-1]})")
        for i, mkt in enumerate(["CAISO", "ERCOT", "PJM"]):
            col = prices[:, i]
            print(f"  {mkt}: mean={col.mean():.1f}  std={col.std():.1f}  "
                  f"min={col.min():.1f}  max={col.max():.1f}")

    else:
        prices, ts = load_prices(
            market         = args.market,
            years          = args.years,
            cache_dir      = cache_dir,
            force_download = args.force,
        )
        print(f"\n{args.market.upper()}: shape={prices.shape}  "
              f"({ts[0]} → {ts[-1]})")
        print(f"  mean={prices.mean():.1f}  std={prices.std():.1f}  "
              f"min={prices.min():.1f}  max={prices.max():.1f}")