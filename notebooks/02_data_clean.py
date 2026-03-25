"""
COMP0051 Algorithmic Trading Coursework — Data Cleaning
========================================================
Reads raw CSVs saved by 01_data_download.py, cleans the OHLCV data,
computes excess returns, saves cleaned parquet files, and plots returns.

Run from the project root (after 01_data_download.py):
    python notebooks/02_data_clean.py
"""

import os
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Configuration
# ============================================================

ASSETS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
INTERVAL = "15m"
MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02",
]

RAW_DIR    = os.path.join("data", "raw")
CLEANED_DIR = os.path.join("data", "cleaned")
RF_DIR     = os.path.join("data", "risk_free")
FIGURES_DIR = os.path.join("report", "figures")

for d in [CLEANED_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Expected date bounds derived from MONTHS
DATE_MIN = pd.Timestamp(MONTHS[0]  + "-01", tz="UTC")
DATE_MAX = (pd.Timestamp(MONTHS[-1] + "-01", tz="UTC")
            + pd.offsets.MonthEnd(1)
            + pd.Timedelta(days=1))

# ============================================================
# 1. Clean raw kline data
# ============================================================

def clean_kline_data(asset: str) -> pd.DataFrame:
    """
    Load the raw CSV for one asset, clean it, and return a DataFrame
    indexed by UTC datetime with columns: open, high, low, close, volume.
    """
    raw_path = os.path.join(RAW_DIR, f"{asset}_{INTERVAL}_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}. Run 01_data_download.py first.")

    # Read CSV — open_time is saved as int64 by the download script
    df = pd.read_csv(raw_path, dtype={"open_time": np.int64})

    # --- Build datetime index from open_time (ms since epoch) ---
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # Keep only rows inside the expected date range (drops any corrupt timestamps)
    df = df[(df["datetime"] >= DATE_MIN) & (df["datetime"] <= DATE_MAX)].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"  [{asset}] Loaded {len(df)} rows after date filter "
          f"({df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()})")

    # --- Remove duplicates ---
    n_before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [{asset}] Removed {n_before - len(df)} duplicate rows")

    # --- Set datetime as index ---
    df = df.set_index("datetime")
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Reindex to complete 15-min grid and forward-fill small gaps ---
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min",
        tz="UTC",
    )
    n_missing = len(full_index) - len(df)
    if n_missing > 0:
        print(f"  [{asset}] {n_missing} missing bars — forward-filling (limit=4)")
    df = df.reindex(full_index)
    df.index.name = "datetime"

    df[price_cols] = df[price_cols].ffill(limit=4)
    df["volume"] = df["volume"].fillna(0)

    n_still_nan = df[price_cols].isna().any(axis=1).sum()
    if n_still_nan > 0:
        print(f"  [{asset}] Dropping {n_still_nan} rows with gaps > 1 hour")
        df = df.dropna(subset=price_cols)

    # --- Detect and repair price outliers (rolling MAD) ---
    rolling_median = df["close"].rolling(window=96, center=True, min_periods=20).median()
    rolling_mad = df["close"].rolling(window=96, center=True, min_periods=20).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    deviation = np.abs(df["close"] - rolling_median) / (rolling_mad + 1e-10)
    outlier_mask = deviation > 5
    if outlier_mask.sum() > 0:
        print(f"  [{asset}] Repairing {outlier_mask.sum()} price outliers")
        for col in price_cols:
            col_median = df[col].rolling(window=96, center=True, min_periods=20).median()
            df.loc[outlier_mask, col] = col_median[outlier_mask]

    # --- Enforce OHLC consistency ---
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    print(f"  [{asset}] Final: {len(df)} rows, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


# ============================================================
# 2. Compute excess returns
# ============================================================

def load_risk_free_rate() -> pd.Series:
    """Load the saved EFFR series (annual decimal) indexed by date."""
    rf_path = os.path.join(RF_DIR, "effr_daily.csv")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Risk-free file not found: {rf_path}. Run 01_data_download.py first.")
    rf = pd.read_csv(rf_path, index_col=0, parse_dates=True)
    return rf.iloc[:, 0].ffill().dropna()


def compute_excess_returns(price_df: pd.DataFrame, rf_daily: pd.Series,
                           bars_per_day: int = 96):
    """
    Returns (excess_returns, simple_returns, rf_per_bar) as pd.Series,
    all with the same DatetimeIndex (first row dropped due to pct_change).
    """
    simple_returns = price_df["close"].pct_change()

    bar_dates = price_df.index.normalize().tz_localize(None)
    rf_mapped = bar_dates.map(lambda d: rf_daily.get(d, np.nan))
    rf_per_bar = pd.Series(rf_mapped.values, index=price_df.index) / (bars_per_day * 365.25)
    rf_per_bar = rf_per_bar.ffill()

    excess_returns = simple_returns - rf_per_bar.shift(1)
    return excess_returns.iloc[1:], simple_returns.iloc[1:], rf_per_bar.iloc[1:]


# ============================================================
# 3. Plotting
# ============================================================

def plot_return_series(returns_dict: dict, title_suffix: str = "Excess Returns"):
    n = len(returns_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colours = {"BTCUSDT": "#F7931A", "ETHUSDT": "#627EEA", "DOGEUSDT": "#C2A633"}

    for ax, (asset, rets) in zip(axes, returns_dict.items()):
        ax.plot(rets.index, rets.values, linewidth=0.3, alpha=0.8,
                color=colours.get(asset, "steelblue"))
        ax.set_ylabel("Return")
        ax.set_title(f"{asset} — {title_suffix}")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.grid(True, alpha=0.3)
        stats = (f"Mean: {rets.mean():.6f}  |  Std: {rets.std():.4f}  |  "
                 f"Skew: {rets.skew():.2f}  |  Kurt: {rets.kurtosis():.2f}")
        ax.text(0.01, 0.95, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))

    plt.tight_layout()
    fig.autofmt_xdate()
    save_path = os.path.join(FIGURES_DIR, "return_time_series.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()


def plot_cumulative_returns(returns_dict: dict):
    fig, ax = plt.subplots(figsize=(14, 6))
    colours = {"BTCUSDT": "#F7931A", "ETHUSDT": "#627EEA", "DOGEUSDT": "#C2A633"}

    for asset, rets in returns_dict.items():
        cum = (1 + rets).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=asset,
                color=colours.get(asset, "steelblue"), linewidth=1)

    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Returns — All Assets (15-min)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    save_path = os.path.join(FIGURES_DIR, "cumulative_returns.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("COMP0051 — Data Cleaning")
    print("=" * 60)

    # --- Step 1: Clean ---
    print(f"\n{'='*60}")
    print("Cleaning OHLCV data")
    print(f"{'='*60}")

    cleaned_data = {}
    for asset in ASSETS:
        print(f"\nCleaning {asset}:")
        cleaned_data[asset] = clean_kline_data(asset)

    # --- Step 2: Risk-free rate ---
    rf_daily = load_risk_free_rate()
    print(f"\nLoaded risk-free rate: {len(rf_daily)} daily observations")

    # --- Step 3: Compute returns and save parquet ---
    print(f"\n{'='*60}")
    print("Computing excess returns & saving to parquet")
    print(f"{'='*60}")

    excess_returns_dict = {}
    simple_returns_dict = {}

    for asset, cdf in cleaned_data.items():
        excess_ret, simple_ret, rf_bars = compute_excess_returns(cdf, rf_daily)
        excess_returns_dict[asset] = excess_ret
        simple_returns_dict[asset] = simple_ret

        out = cdf.copy()
        out["simple_return"] = cdf["close"].pct_change()
        out["excess_return"] = np.nan
        out.loc[excess_ret.index, "excess_return"] = excess_ret.values
        out["rf_per_bar"] = np.nan
        out.loc[rf_bars.index, "rf_per_bar"] = rf_bars.values

        parquet_path = os.path.join(CLEANED_DIR, f"{asset}_{INTERVAL}_cleaned.parquet")
        out.to_parquet(parquet_path)
        print(f"  Saved: {parquet_path} ({len(out)} rows)")

        mean_rf = rf_bars.mean()
        mean_std = simple_ret.std()
        print(f"  [{asset}] rf/bar: {mean_rf:.8f} | return std/bar: {mean_std:.6f} "
              f"| ratio: {mean_std / (mean_rf + 1e-15):.0f}x")

    # --- Step 4: Plot ---
    print(f"\n{'='*60}")
    print("Generating plots")
    print(f"{'='*60}")

    plot_return_series(excess_returns_dict, title_suffix="15-min Excess Returns")
    plot_cumulative_returns(simple_returns_dict)

    print("\nCleaning complete.")
    print(f"  Parquet files : {CLEANED_DIR}/")
    print(f"  Figures       : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
