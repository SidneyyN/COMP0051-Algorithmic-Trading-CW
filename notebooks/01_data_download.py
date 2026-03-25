"""
COMP0051 Algorithmic Trading Coursework — Data Download
========================================================
Downloads raw 15-min kline data from data.binance.vision for BTC, ETH, DOGE
and the Effective Federal Funds Rate from FRED. Saves everything to disk.

Run from the project root:
    python notebooks/01_data_download.py
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================

ASSETS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
INTERVAL = "15m"
MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02",
]

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

FRED_URL_SIMPLE = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=EFFR&cosd=2025-08-01&coed=2026-03-01"
)

RAW_DIR = os.path.join("data", "raw")
RF_DIR = os.path.join("data", "risk_free")

for d in [RAW_DIR, RF_DIR]:
    os.makedirs(d, exist_ok=True)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

# ============================================================
# 1. Download raw kline data from Binance
# ============================================================

def download_binance_klines(asset: str, interval: str, month: str) -> pd.DataFrame:
    """Download one month of kline data. Returns a DataFrame or None on failure."""
    filename = f"{asset}-{interval}-{month}.zip"
    url = f"{BASE_URL}/{asset}/{interval}/{filename}"

    print(f"  Downloading {filename} ...", end=" ")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

    # Drop any header row that Binance sometimes includes
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"]).reset_index(drop=True)
    df["open_time"] = df["open_time"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"OK — {len(df)} rows")
    return df


def download_all_assets() -> None:
    """Download all assets and months, saving one raw CSV per asset."""
    for asset in ASSETS:
        print(f"\n{'='*50}")
        print(f"Downloading {asset} ({INTERVAL})")
        print(f"{'='*50}")

        monthly_dfs = []
        for month in MONTHS:
            df = download_binance_klines(asset, INTERVAL, month)
            if df is not None:
                monthly_dfs.append(df)

        if not monthly_dfs:
            print(f"WARNING: No data downloaded for {asset}")
            continue

        combined = pd.concat(monthly_dfs, ignore_index=True)
        raw_path = os.path.join(RAW_DIR, f"{asset}_{INTERVAL}_raw.csv")
        combined.to_csv(raw_path, index=False)
        print(f"Saved: {raw_path} ({len(combined)} rows)")


# ============================================================
# 2. Download risk-free rate from FRED
# ============================================================

def download_risk_free_rate() -> None:
    """Download EFFR from FRED and save as CSV."""
    print("\nDownloading risk-free rate (EFFR) from FRED ...")
    try:
        resp = requests.get(FRED_URL_SIMPLE, timeout=15)
        resp.raise_for_status()
        rf = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
        rf.columns = ["effr"]
    except Exception as e:
        print(f"  FRED download failed ({e}). Using 4.5% placeholder.")
        dates = pd.date_range("2025-08-01", "2026-03-01", freq="D")
        rf = pd.DataFrame({"effr": 4.50}, index=dates)
        rf.index.name = "DATE"

    rf["effr"] = pd.to_numeric(rf["effr"], errors="coerce")
    rf["effr"] = rf["effr"] / 100.0
    rf = rf["effr"].ffill().dropna()

    rf_path = os.path.join(RF_DIR, "effr_daily.csv")
    rf.to_csv(rf_path)
    print(f"  Saved: {rf_path} ({len(rf)} daily observations)")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("COMP0051 — Data Download")
    print("=" * 60)

    download_all_assets()
    download_risk_free_rate()

    print("\nDownload complete.")
    print(f"  Raw klines : {RAW_DIR}/")
    print(f"  Risk-free  : {RF_DIR}/")
    print("Run notebooks/02_data_clean.py next.")


if __name__ == "__main__":
    main()
