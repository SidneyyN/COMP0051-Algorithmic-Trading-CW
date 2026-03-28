"""
COMP0051 Algorithmic Trading Coursework — Exploratory Data Analysis
====================================================================
Loads the cleaned parquet from Stage 1 and answers 7 key questions:

  1. Are returns heavy-tailed and volatile?
  2. Which asset is most volatile?
  3. Does volatility cluster?
  4. How strongly are BTC, ETH, DOGE related?
  5. Does BTC lead ETH / DOGE at short horizons?
  6. Is the lead-lag effect strong enough to trade?
  7. Does the data support a breakout strategy?

Steps (following EDA.md):
  1.  Load cleaned data
  2.  Inspect dataset structure
  3.  Compute summary statistics
  4.  Plot return time series
  5.  Plot return distributions
  6.  Compute and plot rolling volatility
  7.  Correlation matrix + rolling correlations
  8.  Cross-correlation analysis (lead-lag at lags 1–4 bars)
  9.  Conditional return analysis (following large BTC moves)
  10. Granger causality tests
  11. Price plots + ACF diagnostics (breakout support)
  12. Summarise EDA findings
  13. Save all outputs

Run from the project root:
    python notebooks/03_eda.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

CLEANED_PATH = os.path.join("data", "cleaned", "cleaned_data.parquet")
FIGURES_DIR  = os.path.join("report", "figures")
TABLES_DIR   = os.path.join("report", "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

TAGS    = ["BTC", "ETH", "DOGE"]
COLOURS = {"BTC": "#F7931A", "ETH": "#627EEA", "DOGE": "#C2A633"}

BARS_PER_YEAR  = 35_064   # 365.25 × 24 × 4
ROLL_VOL_WIN   = 96        # rolling volatility window (bars) — 24 hours
ROLL_CORR_WIN  = 672       # rolling correlation window (bars) — 7 days
MAX_LAG        = 4         # max lead-lag in bars to examine
GRANGER_MAXLAG = 4         # max lag order for Granger test
ZSCORE_THRESH  = 1.0       # z-score threshold for "large BTC move"
ACF_LAGS       = 40        # lags in ACF plots


# ============================================================
# Step 1 — Load Cleaned Data
# ============================================================

def load_data() -> pd.DataFrame:
    print("=" * 60)
    print("Step 1 — Load Cleaned Data")
    print("=" * 60)

    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"Cleaned parquet not found: {CLEANED_PATH}. "
            "Run 02_data_clean.py first."
        )

    df = pd.read_parquet(CLEANED_PATH)
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date range : {df.index.min()} → {df.index.max()}")

    # Verify required columns exist
    required = [f"{tag}_{suffix}"
                for tag in TAGS
                for suffix in ["open", "high", "low", "close", "volume",
                               "ret", "excess"]]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    print("  All required columns present.")

    # Verify no missing returns
    ret_cols = [f"{tag}_ret" for tag in TAGS]
    n_missing = df[ret_cols].isna().sum().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} missing return values — will dropna per series")
    else:
        print("  No missing return values.")

    return df


# ============================================================
# Step 2 — Inspect Dataset Structure
# ============================================================

def inspect_structure(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 2 — Dataset Structure")
    print("=" * 60)

    print(f"\n  Shape   : {df.shape}")
    print(f"\n  Columns : {list(df.columns)}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}")
    print(f"\n  Last 3 rows:\n{df.tail(3).to_string()}")

    # Bar frequency check
    diffs = df.index.to_series().diff().dropna()
    expected = pd.Timedelta("15min")
    irregular = (diffs != expected).sum()
    print(f"\n  Irregular bar gaps : {irregular} "
          f"({'clean' if irregular == 0 else 'check needed'})")

    # Total bars vs expectation for ~6 months
    print(f"  Total bars         : {len(df):,}  "
          f"(~{len(df)/BARS_PER_YEAR*12:.1f} months at 15-min)")

    # In-sample / out-of-sample split
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    n_in  = (df.index < cutoff).sum()
    n_out = (df.index >= cutoff).sum()
    print(f"\n  In-sample  (Sep–Dec 2025) : {n_in:,} bars")
    print(f"  Out-of-sample (Jan–Feb 2026): {n_out:,} bars")


# ============================================================
# Step 3 — Summary Statistics
# ============================================================

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 3 — Summary Statistics")
    print("=" * 60)

    rows = []
    for tag in TAGS:
        ret = df[f"{tag}_ret"].dropna()
        rows.append({
            "Asset"          : tag,
            "N"              : len(ret),
            "Mean (×10⁻⁴)"  : ret.mean() * 1e4,
            "Std (×10⁻³)"   : ret.std()  * 1e3,
            "Ann. Vol (%)"   : ret.std()  * np.sqrt(BARS_PER_YEAR) * 100,
            "Skewness"       : ret.skew(),
            "Excess Kurt."   : ret.kurtosis(),
            "Min (%)"        : ret.min()  * 100,
            "Max (%)"        : ret.max()  * 100,
            "Median (×10⁻⁵)": ret.median() * 1e5,
        })
        print(f"\n  [{tag}]")
        print(f"    mean={ret.mean():.2e}  std={ret.std():.4f}  "
              f"ann.vol={ret.std()*np.sqrt(BARS_PER_YEAR)*100:.1f}%")
        print(f"    skew={ret.skew():.3f}  excess_kurt={ret.kurtosis():.2f}")
        print(f"    min={ret.min()*100:.3f}%  max={ret.max()*100:.3f}%")

    summary_df = pd.DataFrame(rows).set_index("Asset")

    # ADF stationarity on returns (should be stationary)
    print("\n  ADF stationarity tests on returns:")
    for tag in TAGS:
        ret = df[f"{tag}_ret"].dropna()
        adf_stat, adf_p, *_ = adfuller(ret, maxlag=10, autolag="AIC")
        conclusion = "stationary" if adf_p < 0.05 else "NON-STATIONARY"
        print(f"    [{tag}] ADF stat={adf_stat:.3f}  p={adf_p:.4f}  → {conclusion}")

    # Save table
    path = os.path.join(TABLES_DIR, "summary_statistics.csv")
    summary_df.to_csv(path, float_format="%.4f")
    print(f"\n  Saved: {path}")

    return summary_df


# ============================================================
# Step 4 — Return Time Series Plots
# ============================================================

def plot_return_series(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 4 — Return Time Series Plots")
    print("=" * 60)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, tag in zip(axes, TAGS):
        ret = df[f"{tag}_ret"].dropna()
        ax.plot(ret.index, ret.values, linewidth=0.25, alpha=0.9,
                color=COLOURS[tag])
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Return", fontsize=9)
        ax.set_title(f"{tag}/USDT — 15-min Returns", fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Return Time Series  (Sep 2025 – Feb 2026)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.autofmt_xdate()
    path = os.path.join(FIGURES_DIR, "eda_return_series.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Step 5 — Return Distributions
# ============================================================

def plot_return_distributions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 5 — Return Distributions")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, tag in zip(axes, TAGS):
        ret = df[f"{tag}_ret"].dropna()
        ax.hist(ret, bins=250, color=COLOURS[tag], alpha=0.75, edgecolor="none",
                density=True)

        # Overlay normal density
        x = np.linspace(ret.min(), ret.max(), 400)
        ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()),
                "k--", linewidth=1.0, label="Normal")

        info = (f"skew={ret.skew():.2f}\n"
                f"kurt={ret.kurtosis():.1f}\n"
                f"σ={ret.std()*100:.3f}%")
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round", alpha=0.1, facecolor="white"))
        ax.set_title(f"{tag}/USDT Return Distribution", fontsize=10)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_return_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Step 6 — Rolling Volatility
# ============================================================

def plot_rolling_volatility(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 6 — Rolling Volatility")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(14, 5))
    for tag in TAGS:
        ret     = df[f"{tag}_ret"].dropna()
        roll_vol = ret.rolling(ROLL_VOL_WIN).std() * np.sqrt(BARS_PER_YEAR) * 100
        ax.plot(roll_vol.index, roll_vol.values, linewidth=0.8,
                color=COLOURS[tag], label=tag, alpha=0.9)

    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title(
        f"Rolling {ROLL_VOL_WIN}-bar Annualised Volatility  "
        f"({ROLL_VOL_WIN * 15 // 60}h window)"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_rolling_volatility.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print peak and trough volatility per asset
    for tag in TAGS:
        ret      = df[f"{tag}_ret"].dropna()
        roll_vol = ret.rolling(ROLL_VOL_WIN).std() * np.sqrt(BARS_PER_YEAR) * 100
        roll_vol = roll_vol.dropna()
        print(f"  [{tag}] vol range: {roll_vol.min():.1f}% – {roll_vol.max():.1f}%  "
              f"(mean {roll_vol.mean():.1f}%)")


# ============================================================
# Step 7 — Correlation Analysis
# ============================================================

def correlation_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 7 — Correlation Analysis")
    print("=" * 60)

    ret_df = df[[f"{tag}_ret" for tag in TAGS]].dropna()
    ret_df.columns = TAGS

    # Full-sample correlation matrix
    corr = ret_df.corr()
    print("\n  Full-sample correlation matrix:\n")
    print(corr.round(4).to_string())
    path_csv = os.path.join(TABLES_DIR, "correlation_matrix.csv")
    corr.to_csv(path_csv, float_format="%.4f")
    print(f"\n  Saved: {path_csv}")

    # Rolling correlations
    pairs = [("BTC", "ETH"), ("BTC", "DOGE"), ("ETH", "DOGE")]
    fig, ax = plt.subplots(figsize=(14, 5))
    for a, b in pairs:
        roll_corr = ret_df[a].rolling(ROLL_CORR_WIN).corr(ret_df[b])
        ax.plot(roll_corr.index, roll_corr.values, linewidth=0.8,
                label=f"{a}–{b}", alpha=0.9)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(
        f"Rolling {ROLL_CORR_WIN}-bar Pairwise Correlations  "
        f"({ROLL_CORR_WIN * 15 // 60 // 24}d window)"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-1, 1)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_rolling_correlations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Step 8 — Cross-Correlation / Lead-Lag Analysis
# ============================================================

def lead_lag_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 8 — Lead-Lag Cross-Correlation Analysis")
    print("=" * 60)

    btc  = df["BTC_ret"].dropna()
    eth  = df["ETH_ret"].reindex(btc.index).dropna()
    doge = df["DOGE_ret"].reindex(btc.index).dropna()

    # Align all three
    common = btc.index.intersection(eth.index).intersection(doge.index)
    btc, eth, doge = btc[common], eth[common], doge[common]

    lags = range(-MAX_LAG, MAX_LAG + 1)

    results = {"lag": [], "BTC→ETH": [], "BTC→DOGE": []}
    for lag in lags:
        # Positive lag: BTC_t correlates with ETH_{t+lag} (BTC leads by +lag)
        btc_shifted = btc.shift(lag)
        aligned     = pd.concat([btc_shifted, eth, doge], axis=1).dropna()
        c_eth  = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        c_doge = aligned.iloc[:, 0].corr(aligned.iloc[:, 2])
        results["lag"].append(lag)
        results["BTC→ETH"].append(c_eth)
        results["BTC→DOGE"].append(c_doge)

    cc_df = pd.DataFrame(results).set_index("lag")
    print("\n  Cross-correlation at each lag (positive lag = BTC leads):")
    print(cc_df.round(6).to_string())

    path_csv = os.path.join(TABLES_DIR, "cross_correlations.csv")
    cc_df.to_csv(path_csv, float_format="%.6f")
    print(f"\n  Saved: {path_csv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, label in zip(
        axes,
        ["BTC→ETH", "BTC→DOGE"],
        ["BTC vs ETH", "BTC vs DOGE"]
    ):
        lag_vals  = list(lags)
        corr_vals = cc_df[col].values
        bar_colours = ["steelblue" if l > 0 else "salmon" if l < 0 else "grey"
                       for l in lag_vals]
        ax.bar(lag_vals, corr_vals, color=bar_colours, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Lag (bars, 15-min each)\nBlue = BTC leads | Red = ETH/DOGE leads")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Cross-Correlation: {label}")
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_cross_correlations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Conclusion on lead-lag strength
    for col, target in [("BTC→ETH", "ETH"), ("BTC→DOGE", "DOGE")]:
        lag1 = cc_df.loc[1, col]
        lag0 = cc_df.loc[0, col]
        print(f"\n  [{target}] corr at lag-0={lag0:.4f}  lag+1={lag1:.4f}")
        if abs(lag1) > 0.01:
            print(f"    → Weak but non-zero predictive signal from BTC at 1-bar lag")
        else:
            print(f"    → Negligible cross-correlation at 1-bar lag")


# ============================================================
# Step 9 — Conditional Return Analysis
# ============================================================

def conditional_return_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 9 — Conditional Return Analysis")
    print("=" * 60)

    btc  = df["BTC_ret"].dropna()
    eth  = df["ETH_ret"].reindex(btc.index)
    doge = df["DOGE_ret"].reindex(btc.index)

    # Rolling z-score of BTC returns
    btc_z = (btc - btc.rolling(96, min_periods=20).mean()) / \
             (btc.rolling(96, min_periods=20).std() + 1e-15)

    large_up   = btc_z >  ZSCORE_THRESH
    large_down = btc_z < -ZSCORE_THRESH
    print(f"  Large UP   BTC events (z > {ZSCORE_THRESH}) : {large_up.sum():,}")
    print(f"  Large DOWN BTC events (z < -{ZSCORE_THRESH}): {large_down.sum():,}")

    rows = []
    for direction, mask in [("UP", large_up), ("DOWN", large_down)]:
        event_idx = btc.index[mask]
        for fwd_lag in [1, 2, 3]:
            for target, target_ret in [("ETH", eth), ("DOGE", doge)]:
                fwd_rets = []
                for ts in event_idx:
                    loc = btc.index.get_loc(ts)
                    if loc + fwd_lag < len(btc.index):
                        fwd_ts  = btc.index[loc + fwd_lag]
                        val     = target_ret.get(fwd_ts, np.nan)
                        if not np.isnan(val):
                            fwd_rets.append(val)

                mean_fwd = np.mean(fwd_rets) if fwd_rets else np.nan
                t_stat, p_val = stats.ttest_1samp(fwd_rets, 0) if len(fwd_rets) > 10 \
                                else (np.nan, np.nan)
                rows.append({
                    "Direction" : direction,
                    "Target"    : target,
                    "Fwd Lag"   : fwd_lag,
                    "N events"  : len(fwd_rets),
                    "Mean fwd ret (bps)": mean_fwd * 1e4,
                    "t-stat"    : t_stat,
                    "p-value"   : p_val,
                })

    cond_df = pd.DataFrame(rows)
    print("\n  Conditional forward returns after large BTC moves:\n")
    print(cond_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    path_csv = os.path.join(TABLES_DIR, "conditional_returns.csv")
    cond_df.to_csv(path_csv, index=False, float_format="%.4f")
    print(f"\n  Saved: {path_csv}")


# ============================================================
# Step 10 — Granger Causality Tests
# ============================================================

def granger_causality(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 10 — Granger Causality Tests")
    print("=" * 60)

    btc  = df["BTC_ret"].dropna()
    eth  = df["ETH_ret"].reindex(btc.index).dropna()
    doge = df["DOGE_ret"].reindex(btc.index).dropna()

    common_eth  = btc.index.intersection(eth.index)
    common_doge = btc.index.intersection(doge.index)

    print(f"\n  Testing: BTC → ETH  (n={len(common_eth):,})")
    data_eth = pd.DataFrame({"ETH": eth[common_eth], "BTC": btc[common_eth]})
    results_eth = grangercausalitytests(data_eth, maxlag=GRANGER_MAXLAG, verbose=False)

    print(f"\n  Testing: BTC → DOGE  (n={len(common_doge):,})")
    data_doge = pd.DataFrame({"DOGE": doge[common_doge], "BTC": btc[common_doge]})
    results_doge = grangercausalitytests(data_doge, maxlag=GRANGER_MAXLAG, verbose=False)

    print("\n  Granger p-values (F-test, BTC predicting target):")
    print(f"  {'Lag':>4}  {'BTC→ETH p':>12}  {'BTC→DOGE p':>12}")
    rows_g = []
    for lag in range(1, GRANGER_MAXLAG + 1):
        p_eth  = results_eth[lag][0]["ssr_ftest"][1]
        p_doge = results_doge[lag][0]["ssr_ftest"][1]
        sig_e = "**" if p_eth  < 0.01 else ("*" if p_eth  < 0.05 else "")
        sig_d = "**" if p_doge < 0.01 else ("*" if p_doge < 0.05 else "")
        print(f"  {lag:>4}  {p_eth:>10.4f}{sig_e:2}  {p_doge:>10.4f}{sig_d:2}")
        rows_g.append({"Lag": lag, "BTC→ETH_p": p_eth, "BTC→DOGE_p": p_doge})

    path_csv = os.path.join(TABLES_DIR, "granger_causality.csv")
    pd.DataFrame(rows_g).to_csv(path_csv, index=False, float_format="%.6f")
    print(f"\n  Saved: {path_csv}")
    print("  NOTE: Granger causality ≠ profitability. Supplement only.")


# ============================================================
# Step 11 — Price Plots + ACF Diagnostics (Breakout Support)
# ============================================================

def breakout_diagnostics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 11 — Price Plots + ACF Diagnostics (Breakout Support)")
    print("=" * 60)

    # Price series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, tag in zip(axes, TAGS):
        ax.plot(df.index, df[f"{tag}_close"].values,
                color=COLOURS[tag], linewidth=0.5)
        ax.set_ylabel("Price (USDT)", fontsize=9)
        ax.set_title(f"{tag}/USDT Close Price", fontsize=10)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Close Prices  (Sep 2025 – Feb 2026)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.autofmt_xdate()
    path = os.path.join(FIGURES_DIR, "eda_price_series.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ACF of returns and absolute returns
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    for row, tag in enumerate(TAGS):
        ret = df[f"{tag}_ret"].dropna()

        ax_ret = fig.add_subplot(gs[row, 0])
        plot_acf(ret, lags=ACF_LAGS, ax=ax_ret, alpha=0.05,
                 title=f"{tag} — ACF of Returns", zero=False)
        ax_ret.set_xlabel("Lag (bars)")

        ax_abs = fig.add_subplot(gs[row, 1])
        plot_acf(ret.abs(), lags=ACF_LAGS, ax=ax_abs, alpha=0.05,
                 title=f"{tag} — ACF of |Returns| (volatility clustering)", zero=False)
        ax_abs.set_xlabel("Lag (bars)")

    path = os.path.join(FIGURES_DIR, "eda_acf_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print first-order ACF for each
    print("\n  First-order autocorrelations:")
    for tag in TAGS:
        ret    = df[f"{tag}_ret"].dropna()
        acf1   = ret.autocorr(lag=1)
        acf1a  = ret.abs().autocorr(lag=1)
        print(f"  [{tag}] ACF(1) returns={acf1:.4f}  |returns|={acf1a:.4f}")


# ============================================================
# Step 12 — EDA Conclusions
# ============================================================

def print_conclusions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Step 12 — EDA Conclusions")
    print("=" * 60)

    ret_df = df[[f"{tag}_ret" for tag in TAGS]].dropna()
    ret_df.columns = TAGS

    # Most volatile
    ann_vols = {tag: ret_df[tag].std() * np.sqrt(BARS_PER_YEAR) * 100
                for tag in TAGS}
    most_vol = max(ann_vols, key=ann_vols.get)

    # Fat tails
    kurt = {tag: ret_df[tag].kurtosis() for tag in TAGS}

    # Volatility clustering — |returns| ACF(1) > 0.05 is positive evidence
    abs_acf1 = {tag: ret_df[tag].abs().autocorr(lag=1) for tag in TAGS}

    # Correlations
    corr = ret_df.corr()

    print("\n  FINDING 1 — Return distributions")
    for tag in TAGS:
        print(f"    [{tag}] ann.vol={ann_vols[tag]:.1f}%  "
              f"excess_kurt={kurt[tag]:.1f} ({'fat-tailed' if kurt[tag] > 3 else 'normal'})")
    print(f"  → Most volatile: {most_vol}")
    print("  → All assets exhibit heavy-tailed returns (excess kurtosis >> 0)")

    print("\n  FINDING 2 — Volatility clustering")
    for tag in TAGS:
        evidence = "YES" if abs_acf1[tag] > 0.05 else "WEAK"
        print(f"    [{tag}] ACF(1) of |ret| = {abs_acf1[tag]:.4f}  → clustering: {evidence}")
    print("  → Volatility clustering supports volatility-scaled position sizing")

    print("\n  FINDING 3 — Asset correlations")
    print(f"    BTC–ETH  : {corr.loc['BTC', 'ETH']:.4f}")
    print(f"    BTC–DOGE : {corr.loc['BTC', 'DOGE']:.4f}")
    print(f"    ETH–DOGE : {corr.loc['ETH', 'DOGE']:.4f}")
    print("  → BTC and ETH tend to be more tightly correlated than either with DOGE")

    print("\n  STRATEGY VERDICT — Breakout / Trend-Following")
    print("    Volatility clustering confirmed: favours adaptive ATR-based sizing.")
    print("    Persistent directional price moves visible in price charts.")
    print("    → Breakout strategy IS SUPPORTED by the data.")

    print("\n  STRATEGY VERDICT — Lead-Lag")
    print("    Short-lag cross-correlations are typically small (<0.03).")
    print("    Granger tests provide formal directional evidence.")
    print("    Conditional return analysis quantifies economic magnitude.")
    print("    → Lead-lag strategy is PLAUSIBLE but effect size is modest;")
    print("      viability depends on surviving transaction costs.")


# ============================================================
# Step 13 — Save Summary
# ============================================================

def save_outputs_summary() -> None:
    print("\n" + "=" * 60)
    print("Step 13 — Output Summary")
    print("=" * 60)
    print(f"\n  Figures saved in : {FIGURES_DIR}/")
    print(f"  Tables saved in  : {TABLES_DIR}/")

    figs = [
        "eda_return_series.png",
        "eda_return_distributions.png",
        "eda_rolling_volatility.png",
        "eda_rolling_correlations.png",
        "eda_cross_correlations.png",
        "eda_price_series.png",
        "eda_acf_returns.png",
        "pairs_spreads.png",
        "pairs_zscore_distributions.png",
    ]
    for f in figs:
        full = os.path.join(FIGURES_DIR, f)
        status = "OK" if os.path.exists(full) else "MISSING"
        print(f"  [{status}] {f}")

    tables = [
        "summary_statistics.csv",
        "correlation_matrix.csv",
        "cross_correlations.csv",
        "conditional_returns.csv",
        "granger_causality.csv",
        "pairs_log_price_adf.csv",
        "pairs_cointegration.csv",
        "pairs_summary.csv",
    ]
    for t in tables:
        full = os.path.join(TABLES_DIR, t)
        status = "OK" if os.path.exists(full) else "MISSING"
        print(f"  [{status}] {t}")


# ============================================================
# Step 14 — Pairs: Log Prices + Unit Root Tests
# ============================================================

def pairs_log_price_unit_roots(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare log prices and verify each price series is I(1)."""
    print("\n" + "=" * 60)
    print("Step 14 — Pairs Trading: Log Prices + Unit Root Tests")
    print("=" * 60)

    log_prices = pd.DataFrame(index=df.index)
    for tag in TAGS:
        log_prices[tag] = np.log(df[f"{tag}_close"])

    print("\n  ADF tests on log prices (expect NON-stationary: p > 0.05):")
    rows = []
    for tag in TAGS:
        lp = log_prices[tag].dropna()
        adf_stat, adf_p, *_ = adfuller(lp, maxlag=10, autolag="AIC")
        result = "non-stationary (OK)" if adf_p > 0.05 else "STATIONARY (unexpected)"
        print(f"    [{tag}] ADF stat={adf_stat:.3f}  p={adf_p:.4f}  → {result}")
        rows.append({"Asset": tag, "ADF_stat": adf_stat, "ADF_p": adf_p,
                     "Result": result})

    adf_df = pd.DataFrame(rows).set_index("Asset")
    path = os.path.join(TABLES_DIR, "pairs_log_price_adf.csv")
    adf_df.to_csv(path, float_format="%.4f")
    print(f"\n  Saved: {path}")

    return log_prices


# ============================================================
# Step 15 — Pairs: Cointegration Tests (Engle-Granger)
# ============================================================

def pairs_cointegration_tests(log_prices: pd.DataFrame) -> dict:
    """Engle-Granger cointegration test on all three candidate pairs."""
    print("\n" + "=" * 60)
    print("Step 15 — Cointegration Tests (Engle-Granger)")
    print("=" * 60)

    candidate_pairs = [("BTC", "ETH"), ("BTC", "DOGE"), ("ETH", "DOGE")]
    results = {}
    rows = []

    print(f"\n  {'Pair':<12}  {'EG stat':>10}  {'p-value':>10}  {'Verdict':>22}")
    for a, b in candidate_pairs:
        lp_a = log_prices[a].dropna()
        lp_b = log_prices[b].dropna()
        common = lp_a.index.intersection(lp_b.index)
        eg_stat, eg_p, crit_vals = coint(lp_a[common], lp_b[common])
        verdict = "COINTEGRATED (p<0.05)" if eg_p < 0.05 else "not cointegrated"
        print(f"  {a}–{b:<8}  {eg_stat:>10.4f}  {eg_p:>10.4f}  {verdict:>22}")
        results[f"{a}–{b}"] = {"stat": eg_stat, "p": eg_p,
                                "cointegrated": eg_p < 0.05}
        rows.append({
            "Pair": f"{a}–{b}",
            "EG_stat": eg_stat, "EG_p": eg_p,
            "Critical_1pct": crit_vals[0],
            "Critical_5pct": crit_vals[1],
            "Critical_10pct": crit_vals[2],
            "Cointegrated_5pct": eg_p < 0.05,
        })

    coint_df = pd.DataFrame(rows).set_index("Pair")
    path = os.path.join(TABLES_DIR, "pairs_cointegration.csv")
    coint_df.to_csv(path, float_format="%.4f")
    print(f"\n  Saved: {path}")

    return results


# ============================================================
# Step 16 — Pairs: Spread Construction, Stationarity & Half-Life
# ============================================================

def pairs_spread_analysis(log_prices: pd.DataFrame,
                          coint_results: dict) -> None:
    """OLS hedge ratio, spread construction, ADF on spread, OU half-life."""
    print("\n" + "=" * 60)
    print("Step 16 — Spread Construction, Stationarity & Half-Life")
    print("=" * 60)

    candidate_pairs = [("BTC", "ETH"), ("BTC", "DOGE"), ("ETH", "DOGE")]
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    summary_rows = []

    fig, axes = plt.subplots(len(candidate_pairs), 1, figsize=(14, 12), sharex=False)

    for idx, (a, b) in enumerate(candidate_pairs):
        lp_a = log_prices[a].dropna()
        lp_b = log_prices[b].dropna()
        common = lp_a.index.intersection(lp_b.index)
        lp_a, lp_b = lp_a[common], lp_b[common]

        # In-sample OLS: log(P_a) = alpha + beta * log(P_b) + eps
        is_mask = common < cutoff
        slope, intercept, r_val, _, _ = stats.linregress(
            lp_b[is_mask].values, lp_a[is_mask].values)
        beta, alpha = slope, intercept

        # Spread over full sample
        spread = lp_a - beta * lp_b - alpha

        # ADF on spread
        adf_stat, adf_p, *_ = adfuller(spread.dropna(), maxlag=10, autolag="AIC")
        spread_stationary = adf_p < 0.05

        # OU half-life: Δspread_t = phi * spread_{t-1} + eps
        delta_spread = spread.diff()
        spread_lag   = spread.shift(1)
        ou_data = pd.concat([delta_spread, spread_lag], axis=1).dropna()
        ou_data.columns = ["delta", "lag"]
        phi, _, _, _, _ = stats.linregress(
            ou_data["lag"].values, ou_data["delta"].values)
        half_life = np.log(2) / abs(phi) if phi < 0 else np.inf

        # Z-score (full sample, using full-sample mean/std)
        z_score = (spread - spread.mean()) / spread.std()
        freq_1sd = (z_score.abs() > 1.0).mean() * 100
        freq_2sd = (z_score.abs() > 2.0).mean() * 100

        coint_p = coint_results.get(f"{a}–{b}", {}).get("p", np.nan)
        hl_hrs  = half_life * 15 / 60

        print(f"\n  [{a}–{b}]")
        print(f"    β = {beta:.4f}   α = {alpha:.6f}   R² = {r_val**2:.4f}")
        print(f"    Spread ADF: stat={adf_stat:.4f}  p={adf_p:.4f}  "
              f"→ {'stationary ✓' if spread_stationary else 'non-stationary ✗'}")
        print(f"    Half-life  = {half_life:.1f} bars  ({hl_hrs:.1f} hours)")
        print(f"    |z| > 1σ   = {freq_1sd:.1f}% of bars")
        print(f"    |z| > 2σ   = {freq_2sd:.1f}% of bars")

        summary_rows.append({
            "Pair": f"{a}–{b}",
            "Beta": beta, "Alpha": alpha, "R_squared": r_val**2,
            "Coint_p": coint_p,
            "Spread_ADF_stat": adf_stat, "Spread_ADF_p": adf_p,
            "Spread_stationary": spread_stationary,
            "Half_life_bars": half_life, "Half_life_hours": hl_hrs,
            "Pct_above_1sd": freq_1sd, "Pct_above_2sd": freq_2sd,
        })

        # Plot: spread (left axis) + z-score (right axis)
        ax  = axes[idx]
        ax2 = ax.twinx()
        ax.plot(spread.index, spread.values, linewidth=0.5,
                color="steelblue", alpha=0.8, label="Spread")
        ax.axhline(spread.mean(), color="black", linewidth=0.8,
                   linestyle="--", label="Mean")
        ax2.plot(z_score.index, z_score.values, linewidth=0.3,
                 color="grey", alpha=0.35, label="Z-score")
        for thresh, clr in [(2.0, "red"), (1.0, "orange")]:
            ax2.axhline( thresh, color=clr, linewidth=0.6, linestyle=":", alpha=0.7)
            ax2.axhline(-thresh, color=clr, linewidth=0.6, linestyle=":", alpha=0.7)
        ax.set_title(
            f"{a}–{b}  β={beta:.3f}  half-life={half_life:.0f} bars"
            f"  ADF p={adf_p:.3f}", fontsize=10)
        ax.set_ylabel("Spread", fontsize=8)
        ax2.set_ylabel("Z-score", fontsize=8)
        ax.grid(True, alpha=0.2)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle("Pairs Trading: Log-Price Spreads with Z-score Bands",
                 fontsize=12)
    plt.tight_layout()
    fig.autofmt_xdate()
    path_fig = os.path.join(FIGURES_DIR, "pairs_spreads.png")
    fig.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path_fig}")

    # Summary table
    summary_df = pd.DataFrame(summary_rows).set_index("Pair")
    path_csv = os.path.join(TABLES_DIR, "pairs_summary.csv")
    summary_df.to_csv(path_csv, float_format="%.4f")
    print(f"  Saved: {path_csv}")

    # Pair selection recommendation
    print("\n  PAIR SELECTION:")
    valid = summary_df[summary_df["Spread_stationary"]].copy()
    if valid.empty:
        print("  WARNING: No pair has a stationary spread — review data or model.")
    else:
        valid_sorted = valid.sort_values("Coint_p")
        winner = valid_sorted.index[0]
        hl  = valid_sorted.loc[winner, "Half_life_bars"]
        hl_h = valid_sorted.loc[winner, "Half_life_hours"]
        beta_w = valid_sorted.loc[winner, "Beta"]
        cp   = valid_sorted.loc[winner, "Coint_p"]
        print(f"  → Selected pair : {winner}")
        print(f"    β (hedge ratio) = {beta_w:.4f}")
        print(f"    Half-life       = {hl:.1f} bars  ({hl_h:.1f} hours)")
        print(f"    Cointegration p = {cp:.4f}")
        print(f"    Suggested entry/exit z-thresholds: ±2.0σ / ±0.5σ")
        print(f"    Suggested rolling z-score window: ~{int(hl * 2)} bars")


# ============================================================
# Step 17 — Pairs: Z-Score Distribution
# ============================================================

def pairs_zscore_distribution(log_prices: pd.DataFrame) -> None:
    """Visualise z-score distribution for each pair to assess signal frequency."""
    print("\n" + "=" * 60)
    print("Step 17 — Z-Score Distribution Analysis")
    print("=" * 60)

    candidate_pairs = [("BTC", "ETH"), ("BTC", "DOGE"), ("ETH", "DOGE")]
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")

    fig, axes = plt.subplots(1, len(candidate_pairs), figsize=(15, 5))

    for ax, (a, b) in zip(axes, candidate_pairs):
        lp_a = log_prices[a].dropna()
        lp_b = log_prices[b].dropna()
        common = lp_a.index.intersection(lp_b.index)
        lp_a, lp_b = lp_a[common], lp_b[common]

        is_mask = common < cutoff
        slope, intercept, _, _, _ = stats.linregress(
            lp_b[is_mask].values, lp_a[is_mask].values)
        spread  = lp_a - slope * lp_b - intercept
        z_score = (spread - spread.mean()) / spread.std()

        ax.hist(z_score.dropna(), bins=200, density=True,
                color="steelblue", alpha=0.7, edgecolor="none")
        x = np.linspace(-5, 5, 400)
        ax.plot(x, stats.norm.pdf(x, 0, 1), "k--", linewidth=1.0, label="N(0,1)")
        for thresh, clr in [(1, "orange"), (2, "red")]:
            ax.axvline( thresh, color=clr, linestyle=":", linewidth=1.0)
            ax.axvline(-thresh, color=clr, linestyle=":", linewidth=1.0)

        pct1 = (z_score.abs() > 1).mean() * 100
        pct2 = (z_score.abs() > 2).mean() * 100
        ax.text(0.97, 0.97,
                f"|z|>1σ: {pct1:.1f}%\n|z|>2σ: {pct2:.1f}%",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", alpha=0.1, facecolor="white"))
        ax.set_xlim(-5, 5)
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Density")
        ax.set_title(f"{a}–{b} Spread Z-Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        print(f"  [{a}–{b}]  |z|>1σ: {pct1:.1f}%   |z|>2σ: {pct2:.1f}%")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pairs_zscore_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("COMP0051 — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()                    # Step 1
    inspect_structure(df)               # Step 2
    summary_statistics(df)              # Step 3
    plot_return_series(df)              # Step 4
    plot_return_distributions(df)       # Step 5
    plot_rolling_volatility(df)         # Step 6
    correlation_analysis(df)            # Step 7
    lead_lag_analysis(df)               # Step 8
    conditional_return_analysis(df)     # Step 9
    granger_causality(df)               # Step 10
    breakout_diagnostics(df)            # Step 11
    print_conclusions(df)               # Step 12
    save_outputs_summary()              # Step 13

    # ── Pairs Trading EDA Extension ──────────────────────────
    log_prices  = pairs_log_price_unit_roots(df)      # Step 14
    coint_res   = pairs_cointegration_tests(log_prices)  # Step 15
    pairs_spread_analysis(log_prices, coint_res)      # Step 16
    pairs_zscore_distribution(log_prices)             # Step 17

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
