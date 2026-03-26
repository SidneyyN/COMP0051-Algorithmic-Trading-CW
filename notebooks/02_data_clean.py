"""
COMP0051 Algorithmic Trading Coursework — Data Cleaning
========================================================
Pipeline: Raw CSV → validate → align → returns → excess returns → final dataset

Steps:
  1.  Load raw data
  2.  Standardise structure
  3.  Remove duplicates
  4.  Check for missing timestamps (forward-fill small gaps)
  5.  Align assets (intersection of timestamps)
  6.  Basic sanity checks
  7.  Compute simple returns
  8.  Identify outliers (flag only — do NOT remove)
  9.  Load risk-free rate
  10. Align risk-free rate to 15-min frequency
  11. Compute excess returns
  12. Construct final wide dataset
  13. Final validation
  14. Save to parquet

Run from the project root:
    python notebooks/02_data_clean.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

ASSETS   = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
INTERVAL = "15m"

BARS_PER_YEAR = 35_064   # 365.25 × 24 × 4
FFILL_LIMIT   = 4        # forward-fill up to 4 consecutive bars (1 hour)

OUTLIER_ROLL_WINDOW = 100   # rolling window (bars) for z-score computation
OUTLIER_Z_THRESHOLD = 7     # flag |z| > 7

RAW_DIR     = os.path.join("data", "raw")
CLEANED_DIR = os.path.join("data", "cleaned")
RF_DIR      = os.path.join("data", "risk_free")
FIGURES_DIR = os.path.join("report", "figures")

for d in [CLEANED_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

SHORT = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "DOGEUSDT": "DOGE"}

# ============================================================
# Step 1 — Load Raw Data
# ============================================================

def load_raw(asset: str) -> pd.DataFrame:
    """
    Load raw CSV, keep OHLCV columns, parse timestamps.
    Auto-detects whether open_time is in microseconds or milliseconds.
    """
    path = os.path.join(RAW_DIR, f"{asset}_{INTERVAL}_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw file not found: {path}. Run 01_data_download.py first."
        )

    df = pd.read_csv(path)

    # Keep only relevant columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    # Convert OHLCV to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect timestamp unit: >1e14 = microseconds; otherwise milliseconds
    max_ts = df["open_time"].max()
    if max_ts > 1e14:
        print(f"  [{asset}] Timestamps in microseconds → converting to milliseconds")
        df["open_time"] = df["open_time"] // 1000
    else:
        print(f"  [{asset}] Timestamps confirmed in milliseconds")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])

    print(f"  [{asset}] Loaded {len(df)} rows")
    return df


# ============================================================
# Step 2 — Standardise Structure
# ============================================================

def standardise(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Set timestamp as index, sort chronologically."""
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df = df.sort_index()
    print(f"  [{asset}] Date range: {df.index.min()} → {df.index.max()}")
    return df


# ============================================================
# Step 3 — Remove Duplicates
# ============================================================

def remove_duplicates(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    removed = n_before - len(df)
    if removed:
        print(f"  [{asset}] Removed {removed} duplicate timestamps")
    else:
        print(f"  [{asset}] No duplicates found")
    return df


# ============================================================
# Step 4 — Check for Missing Timestamps
# ============================================================

def handle_missing_bars(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Reindex to complete 15-min grid.
    Small gaps (≤ FFILL_LIMIT bars): forward-fill prices, zero volume.
    Large gaps (> FFILL_LIMIT bars): drop affected rows.
    """
    price_cols = ["open", "high", "low", "close"]

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min",
        tz="UTC",
    )
    n_missing = len(full_index) - len(df)

    if n_missing == 0:
        print(f"  [{asset}] No missing bars")
        return df

    print(f"  [{asset}] {n_missing} missing bars detected")

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    df[price_cols] = df[price_cols].ffill(limit=FFILL_LIMIT)
    df["volume"] = df["volume"].fillna(0)

    n_large_gap = df[price_cols].isna().any(axis=1).sum()
    if n_large_gap:
        print(f"  [{asset}] Dropping {n_large_gap} rows in gaps > {FFILL_LIMIT} bars")
        df = df.dropna(subset=price_cols)

    return df


# ============================================================
# Step 5 — Align Assets
# ============================================================

def align_assets(asset_dfs: dict) -> dict:
    """Keep only timestamps present across all three assets."""
    common_index = asset_dfs[ASSETS[0]].index
    for asset in ASSETS[1:]:
        common_index = common_index.intersection(asset_dfs[asset].index)

    n_common = len(common_index)
    print(f"\n  Common timestamps across all assets: {n_common}")
    for asset in ASSETS:
        dropped = len(asset_dfs[asset]) - n_common
        if dropped:
            print(f"  [{asset}] Dropped {dropped} rows not in common index")

    return {asset: df.loc[common_index] for asset, df in asset_dfs.items()}


# ============================================================
# Step 6 — Basic Sanity Checks
# ============================================================

def sanity_checks(df: pd.DataFrame, asset: str) -> None:
    """Check for negative or zero prices; report zero-volume bars."""
    issues = False
    for col in ["open", "high", "low", "close"]:
        n_neg  = (df[col] <  0).sum()
        n_zero = (df[col] == 0).sum()
        if n_neg or n_zero:
            print(f"  [{asset}] WARNING: '{col}' has {n_neg} negative and {n_zero} zero values")
            issues = True
    n_zero_vol = (df["volume"] == 0).sum()
    if n_zero_vol:
        print(f"  [{asset}] Note: {n_zero_vol} bars with zero volume")
    if not issues:
        print(f"  [{asset}] Sanity checks passed — "
              f"close range {df['close'].min():.4f} – {df['close'].max():.4f}")


# ============================================================
# Step 7 — Compute Simple Returns
# ============================================================

def compute_simple_returns(df: pd.DataFrame) -> pd.Series:
    """r_t = (p_t − p_{t−1}) / p_{t−1}  (first row dropped)."""
    return df["close"].pct_change().iloc[1:]


# ============================================================
# Step 8 — Identify Outliers (flag only, do NOT remove)
# ============================================================

def flag_outliers(returns: pd.Series, asset: str) -> pd.Series:
    """
    Rolling z-score: z_t = r_t / σ_t  (σ from rolling std, window=OUTLIER_ROLL_WINDOW).
    Flags |z| > OUTLIER_Z_THRESHOLD.
    Returns a boolean mask — does NOT modify returns.
    """
    rolling_std = returns.rolling(window=OUTLIER_ROLL_WINDOW, min_periods=20).std()
    z_scores    = returns / (rolling_std + 1e-15)
    mask        = z_scores.abs() > OUTLIER_Z_THRESHOLD
    n_flagged   = mask.sum()

    if n_flagged:
        print(f"  [{asset}] {n_flagged} flagged outlier(s) (|z| > {OUTLIER_Z_THRESHOLD}):")
        for dt in returns[mask].index:
            print(f"    {dt}  return={returns[dt]:+.6f}  z={z_scores[dt]:+.1f}")
    else:
        print(f"  [{asset}] No outliers flagged")

    return mask


# ============================================================
# Step 9 — Load Risk-Free Rate
# ============================================================

def load_risk_free_rate() -> pd.Series:
    """Load daily EFFR (annual decimal) indexed by date."""
    path = os.path.join(RF_DIR, "effr_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Risk-free file not found: {path}. Run 01_data_download.py first."
        )
    rf = pd.read_csv(path, index_col=0, parse_dates=True)
    rf_series = rf.iloc[:, 0].ffill().dropna()
    print(f"  Loaded {len(rf_series)} daily RF observations "
          f"({rf_series.index.min().date()} – {rf_series.index.max().date()})")
    return rf_series


# ============================================================
# Step 10 — Align Risk-Free Rate to 15-Min Frequency
# ============================================================

def align_rf_to_bars(rf_daily: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill daily EFFR to 15-min bars, then convert annual → per-bar:
        r_f(15min) = r_f(annual) / (365.25 × 24 × 4)
    """
    bar_dates = bar_index.normalize().tz_localize(None)
    rf_mapped = bar_dates.map(lambda d: rf_daily.get(d, np.nan))
    rf_per_bar = pd.Series(rf_mapped.values, index=bar_index) / BARS_PER_YEAR
    rf_per_bar = rf_per_bar.ffill()
    return rf_per_bar


# ============================================================
# Step 11 — Compute Excess Returns
# ============================================================

def compute_excess_returns(simple_ret: pd.Series, rf_per_bar: pd.Series) -> pd.Series:
    """r_t^e = r_t − r_f(t−1)  (risk-free rate is lagged by one bar)."""
    rf_aligned = rf_per_bar.reindex(simple_ret.index).ffill()
    return simple_ret - rf_aligned.shift(1)


# ============================================================
# Step 12 — Construct Final Wide Dataset
# ============================================================

def build_final_dataset(
    asset_dfs:     dict,
    simple_ret:    dict,
    excess_ret:    dict,
    rf_per_bar:    pd.Series,
) -> pd.DataFrame:
    """
    Merge all assets into one wide DataFrame indexed by timestamp.

    Columns:
        BTC_open, BTC_high, BTC_low, BTC_close, BTC_volume,
        ETH_open, ..., DOGE_open, ...,
        BTC_ret, ETH_ret, DOGE_ret,
        BTC_excess, ETH_excess, DOGE_excess,
        rf_per_bar
    """
    # Return series start one row after price series; use returns index as base
    base_index = simple_ret[ASSETS[0]].index

    frames = {}
    for asset in ASSETS:
        tag = SHORT[asset]
        for col in ["open", "high", "low", "close", "volume"]:
            frames[f"{tag}_{col}"] = asset_dfs[asset][col].reindex(base_index)

    df = pd.DataFrame(frames, index=base_index)

    for asset in ASSETS:
        tag = SHORT[asset]
        df[f"{tag}_ret"]    = simple_ret[asset].reindex(base_index)
        df[f"{tag}_excess"] = excess_ret[asset].reindex(base_index)

    df["rf_per_bar"] = rf_per_bar.reindex(base_index)

    return df


# ============================================================
# Step 13 — Final Validation
# ============================================================

def final_validation(df: pd.DataFrame) -> None:
    print(f"  Shape      : {df.shape}")
    print(f"  Date range : {df.index.min()} → {df.index.max()}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing values: none")
    else:
        print(f"  Missing values:\n{missing}")

    for asset in ASSETS:
        tag = SHORT[asset]
        ret = df[f"{tag}_ret"].dropna()
        print(f"  [{tag}] return — mean={ret.mean():.6f} | std={ret.std():.6f} "
              f"| min={ret.min():.4f} | max={ret.max():.4f}")


# ============================================================
# Plots
# ============================================================

def plot_returns(df: pd.DataFrame) -> None:
    tags    = ["BTC", "ETH", "DOGE"]
    colours = {"BTC": "#F7931A", "ETH": "#627EEA", "DOGE": "#C2A633"}

    # Return time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, tag in zip(axes, tags):
        ret = df[f"{tag}_ret"].dropna()
        ax.plot(ret.index, ret.values, linewidth=0.3, alpha=0.8, color=colours[tag])
        ax.set_ylabel("Return")
        ax.set_title(f"{tag}/USDT — Simple Returns (15-min)")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.autofmt_xdate()
    path = os.path.join(FIGURES_DIR, "return_time_series.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # Cumulative returns
    fig, ax = plt.subplots(figsize=(14, 6))
    for tag in tags:
        ret = df[f"{tag}_ret"].dropna()
        cum = (1 + ret).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=f"{tag}/USDT",
                color=colours[tag], linewidth=1)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Returns — All Assets (15-min)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    fig.autofmt_xdate()
    path = os.path.join(FIGURES_DIR, "cumulative_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # Return histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, tag in zip(axes, tags):
        ret = df[f"{tag}_ret"].dropna()
        ax.hist(ret, bins=200, color=colours[tag], alpha=0.7, edgecolor="none")
        stats = (f"mean={ret.mean():.5f}\nstd={ret.std():.4f}\n"
                 f"skew={ret.skew():.2f}  kurt={ret.kurtosis():.2f}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", bbox=dict(boxstyle="round", alpha=0.1))
        ax.set_title(f"{tag}/USDT Return Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "return_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("COMP0051 — Data Cleaning Pipeline")
    print("=" * 60)

    # Steps 1–3: load, standardise, deduplicate
    print("\n[Steps 1–3] Load, standardise, remove duplicates")
    asset_dfs = {}
    for asset in ASSETS:
        print(f"\n  {asset}:")
        df = load_raw(asset)
        df = standardise(df, asset)
        df = remove_duplicates(df, asset)
        asset_dfs[asset] = df

    # Step 4: missing bars
    print("\n[Step 4] Check for missing timestamps")
    for asset in ASSETS:
        asset_dfs[asset] = handle_missing_bars(asset_dfs[asset], asset)

    # Step 5: align assets
    print("\n[Step 5] Align assets to common timestamps")
    asset_dfs = align_assets(asset_dfs)

    # Step 6: sanity checks
    print("\n[Step 6] Basic sanity checks")
    for asset in ASSETS:
        sanity_checks(asset_dfs[asset], asset)

    # Step 7: simple returns
    print("\n[Step 7] Compute simple returns")
    simple_returns = {}
    for asset in ASSETS:
        simple_returns[asset] = compute_simple_returns(asset_dfs[asset])
        print(f"  [{asset}] {len(simple_returns[asset])} return observations")

    # Step 8: flag outliers (do NOT remove)
    print("\n[Step 8] Identify outliers (flag only — retaining all observations)")
    outlier_flags = {}
    for asset in ASSETS:
        outlier_flags[asset] = flag_outliers(simple_returns[asset], asset)

    # Step 9: risk-free rate
    print("\n[Step 9] Load risk-free rate")
    rf_daily = load_risk_free_rate()

    # Step 10: align RF to 15-min bars
    print("\n[Step 10] Align risk-free rate to 15-min frequency")
    bar_index  = simple_returns[ASSETS[0]].index
    rf_per_bar = align_rf_to_bars(rf_daily, bar_index)
    mean_rf    = rf_per_bar.mean()
    print(f"  RF per bar (mean): {mean_rf:.3e}  "
          f"(≈ {mean_rf * BARS_PER_YEAR:.4f} annualised)")

    # Step 11: excess returns
    print("\n[Step 11] Compute excess returns")
    excess_returns = {}
    for asset in ASSETS:
        excess_returns[asset] = compute_excess_returns(
            simple_returns[asset], rf_per_bar
        )
        print(f"  [{asset}] excess return mean: {excess_returns[asset].mean():.8f}")

    # Step 12: build wide dataset
    print("\n[Step 12] Construct final wide dataset")
    final_df = build_final_dataset(
        asset_dfs, simple_returns, excess_returns, rf_per_bar
    )
    print(f"  Columns: {list(final_df.columns)}")

    # Step 13: validate
    print("\n[Step 13] Final validation")
    final_validation(final_df)

    # Step 14: save
    print("\n[Step 14] Save cleaned data")
    out_path = os.path.join(CLEANED_DIR, "cleaned_data.parquet")
    final_df.to_parquet(out_path)
    print(f"  Saved: {out_path}  ({len(final_df)} rows × {len(final_df.columns)} columns)")

    # Plots
    print("\n[Plots] Generating visualisations")
    plot_returns(final_df)

    print("\nData cleaning complete.")
    print(f"  Parquet : {out_path}")
    print(f"  Figures : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()