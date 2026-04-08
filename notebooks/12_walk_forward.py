"""
COMP0051 Algorithmic Trading Coursework — Walk-Forward Validation
=================================================================
Expanding-window WFV with 3 monthly test folds across both strategies.

Folds (expanding train, fixed monthly test):
  Fold 1: Train Sep 1 – Nov 30 2025  |  Test Dec 2025
  Fold 2: Train Sep 1 – Dec 31 2025  |  Test Jan 2026  ← matches IS/OOS split
  Fold 3: Train Sep 1 – Jan 31 2026  |  Test Feb 2026

Per-fold calibration:
  - Breakout: vol_threshold computed on train window only (not full IS)
  - Pairs:    OLS beta recomputed on train window only

Transaction costs: Roll model scalars (BTC 0.000624, ETH 0.000524).

Outputs:
  report/tables/wfv_results.csv
  report/figures/wfv_sharpe_by_fold.png
  report/figures/wfv_cumulative_pnl.png

Run from project root:
    python notebooks/12_walk_forward.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Constants
# ============================================================

CAPITAL       = 100_000.0
BARS_PER_YEAR = 35_064        # 365.25 × 24 × 4

# Roll model spread scalars from notebook 08
ROLL_BTC = 0.000624
ROLL_ETH = 0.000524

DATA_PATH   = os.path.join("data", "cleaned", "cleaned_data.parquet")
FIGURES_DIR = os.path.join("report", "figures")
TABLES_DIR  = os.path.join("report", "tables")

for d in [FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Walk-Forward Fold Definitions (expanding window)
# ============================================================
# Each fold: (name, test_start_inclusive, test_end_exclusive)
# Train window = all data before test_start_inclusive
# Fold 2 is identical to the standard IS/OOS split — serves as sanity check.

FOLDS = [
    ("Fold 1 (Dec 2025)", "2025-12-01", "2026-01-01"),
    ("Fold 2 (Jan 2026)", "2026-01-01", "2026-02-01"),
    ("Fold 3 (Feb 2026)", "2026-02-01", None),          # None → run to end of data
]

# ============================================================
# Parameter Configs
# ============================================================

BREAKOUT_CONFIGS = {
    "baseline":   dict(N=200, vol_window=50,  vol_quantile=0.6, max_hold=40),
    "sha_winner": dict(N=150, vol_window=30,  vol_quantile=0.6, max_hold=60),
}

PAIRS_CONFIGS = {
    "baseline":   dict(zscore_window=100, entry_z=3.0, exit_z=0.0, stop_z=3.0,
                       vol_mult=1.2, min_hold=24, max_hold=384, cooldown=20),
    "sha_winner": dict(zscore_window=75,  entry_z=3.5, exit_z=0.0, stop_z=3.0,
                       vol_mult=1.5, min_hold=24, max_hold=384, cooldown=20),
}

# ============================================================
# Data Loading
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df[["BTC_close", "ETH_close", "BTC_ret", "ETH_ret"]].dropna()
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    return df

# ============================================================
# Transaction Costs (Roll model, inlined)
# ============================================================

def _cost_roll_breakout(position: pd.Series, btc_price: pd.Series) -> pd.Series:
    """Half-spread cost for single-asset (BTC) position changes."""
    delta = position.diff().abs().fillna(0)
    return delta * ROLL_BTC * btc_price / 2


def _cost_roll_pairs(position: pd.Series,
                     btc_price: pd.Series,
                     eth_price: pd.Series) -> pd.Series:
    """Half-spread cost for both legs of the pairs trade."""
    delta = position.diff().abs().fillna(0)
    return delta * ROLL_BTC * btc_price / 2 + delta * ROLL_ETH * eth_price / 2

# ============================================================
# Breakout Strategy (self-contained, per-fold vol calibration)
# ============================================================

def run_breakout(df: pd.DataFrame, train_mask: pd.Series,
                 N: int, vol_window: int, vol_quantile: float,
                 max_hold: int) -> pd.DataFrame:
    """
    Contrarian Donchian breakout strategy.
    vol_threshold calibrated on train_mask only — no future leakage.

    Returns df with columns: position, gross_pnl, cost, net_pnl.
    """
    df = df.copy()

    # Donchian bands (shift(1) prevents lookahead)
    df["upper_band"] = df["BTC_close"].rolling(N).max().shift(1)
    df["lower_band"] = df["BTC_close"].rolling(N).min().shift(1)

    # Raw entry signals (contrarian: fade the breakout)
    df["long_entry"]  = df["BTC_close"] < df["lower_band"]   # buy the dip
    df["short_entry"] = df["BTC_close"] > df["upper_band"]   # sell the spike

    # Volatility filter — threshold calibrated on train window only
    df["rolling_vol"] = df["BTC_ret"].rolling(vol_window).std()
    vol_threshold = df.loc[train_mask, "rolling_vol"].quantile(vol_quantile)
    df["vol_filter"]  = df["rolling_vol"] > vol_threshold

    df["long_entry"]  = df["long_entry"]  & df["vol_filter"]
    df["short_entry"] = df["short_entry"] & df["vol_filter"]

    # State machine (identical to notebook 04)
    n = len(df)
    position    = np.zeros(n, dtype=int)
    hold_bars   = 0
    current_pos = 0

    for t in range(1, n):
        long_sig  = df["long_entry"].iloc[t]
        short_sig = df["short_entry"].iloc[t]

        if current_pos != 0:
            hold_bars += 1
            if (current_pos == 1 and short_sig) or (current_pos == -1 and long_sig):
                current_pos = 0
                hold_bars   = 0
            elif hold_bars >= max_hold:
                current_pos = 0
                hold_bars   = 0
        else:
            if long_sig:
                current_pos = 1
                hold_bars   = 0
            elif short_sig:
                current_pos = -1
                hold_bars   = 0

        position[t] = current_pos

    df["position"] = position

    # PnL: position at bar t earns return at bar t+1
    pos_lag         = df["position"].shift(1).fillna(0)
    df["gross_pnl"] = CAPITAL * pos_lag * df["BTC_ret"]
    df["cost"]      = _cost_roll_breakout(df["position"], df["BTC_close"])
    df["net_pnl"]   = df["gross_pnl"] - df["cost"]

    return df

# ============================================================
# Pairs Strategy (self-contained, per-fold beta calibration)
# ============================================================

def run_pairs(df: pd.DataFrame, train_mask: pd.Series,
              zscore_window: int, entry_z: float, exit_z: float,
              stop_z: float, vol_mult: float, min_hold: int,
              max_hold: int, cooldown: int):
    """
    BTC–ETH mean-reversion pairs strategy with volatility filter.
    OLS beta recomputed on train_mask only — no future leakage.

    Returns (df_with_pnl, beta_used).
    """
    df = df.copy()

    # Log prices
    df["log_btc"] = np.log(df["BTC_close"])
    df["log_eth"] = np.log(df["ETH_close"])

    # OLS beta on train window: log(BTC) = beta * log(ETH) + const
    log_btc_tr = df.loc[train_mask, "log_btc"].values
    log_eth_tr = df.loc[train_mask, "log_eth"].values
    beta = np.polyfit(log_eth_tr, log_btc_tr, 1)[0]

    # Spread and rolling z-score
    df["spread"]      = df["log_btc"] - beta * df["log_eth"]
    roll              = df["spread"].rolling(zscore_window, min_periods=zscore_window)
    df["spread_mean"] = roll.mean()
    df["spread_std"]  = roll.std()
    df["zscore"]      = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    # Volatility filter (fully rolling — no IS calibration)
    spread_diff      = df["spread"].diff()
    df["spread_vol"] = spread_diff.rolling(100, min_periods=100).std()
    df["vol_ref"]    = df["spread_vol"].rolling(200, min_periods=200).median()
    df["allow_trade"] = df["spread_vol"] < vol_mult * df["vol_ref"]

    # State machine (identical to notebook 07)
    n           = len(df)
    z           = df["zscore"].values
    allow_arr   = df["allow_trade"].values

    position      = np.zeros(n, dtype=np.int8)
    hold_count    = 0
    cooldown_left = 0
    state         = 0

    for i in range(1, n):
        zi      = z[i - 1]          # lagged z-score (no lookahead)
        allowed = allow_arr[i - 1]  # lagged vol-filter flag

        if np.isnan(zi):
            position[i] = 0
            state       = 0
            hold_count  = 0
            continue

        if state != 0:
            hold_count += 1
            exiting = False
            if hold_count >= min_hold:
                if state == 1 and zi >= -exit_z:
                    exiting = True
                elif state == -1 and zi <= exit_z:
                    exiting = True
                if abs(zi) > stop_z:
                    exiting = True
            if hold_count >= max_hold:
                exiting = True
            if exiting:
                state         = 0
                hold_count    = 0
                cooldown_left = cooldown

        if state == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
            elif allowed:
                if zi < -entry_z:
                    state      = 1
                    hold_count = 0
                elif zi > entry_z:
                    state      = -1
                    hold_count = 0

        position[i] = state

    df["position"] = position

    # Dollar-neutral PnL
    btc_notional    = CAPITAL / (1.0 + beta)
    eth_notional    = beta * btc_notional
    pos_lag         = df["position"].shift(1).fillna(0)
    df["btc_pnl"]   = pos_lag *  btc_notional * df["BTC_ret"]
    df["eth_pnl"]   = pos_lag * -eth_notional * df["ETH_ret"]
    df["gross_pnl"] = df["btc_pnl"] + df["eth_pnl"]
    df["cost"]      = _cost_roll_pairs(df["position"], df["BTC_close"], df["ETH_close"])
    df["net_pnl"]   = df["gross_pnl"] - df["cost"]

    return df, beta

# ============================================================
# Metrics (evaluated on test window slice only)
# ============================================================

def compute_metrics(df_test: pd.DataFrame) -> dict:
    """Annualised performance metrics on the supplied slice."""
    ret = df_test["net_pnl"] / CAPITAL

    total_gross = df_test["gross_pnl"].sum()
    total_net   = df_test["net_pnl"].sum()
    total_cost  = df_test["cost"].sum()

    mean_r = ret.mean()
    std_r  = ret.std(ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(BARS_PER_YEAR)) if std_r > 0 else np.nan

    neg_r         = ret[ret < 0]
    sortino_denom = neg_r.std(ddof=1)
    sortino = (mean_r / sortino_denom * np.sqrt(BARS_PER_YEAR)) if sortino_denom > 0 else np.nan

    cum_pnl     = df_test["net_pnl"].cumsum()
    rolling_max = cum_pnl.cummax()
    max_dd      = (cum_pnl - rolling_max).min()
    calmar      = (total_net / abs(max_dd)) if abs(max_dd) > 0 else np.nan

    # Count complete trades within the test window
    df_t = df_test.copy()
    trade_entry = (df_t["position"] != 0) & (df_t["position"].shift(1).fillna(0) == 0)
    df_t["trade_id"] = trade_entry.cumsum()
    df_t.loc[df_t["position"] == 0, "trade_id"] = 0
    trade_pnl = df_t[df_t["trade_id"] > 0].groupby("trade_id")["net_pnl"].sum()
    n_trades = len(trade_pnl)
    win_rate = (trade_pnl > 0).mean() if n_trades > 0 else np.nan

    return {
        "sharpe":         round(sharpe, 4),
        "sortino":        round(sortino, 4),
        "calmar":         round(calmar, 4),
        "total_net_$":    round(total_net, 2),
        "total_gross_$":  round(total_gross, 2),
        "total_cost_$":   round(total_cost, 2),
        "max_drawdown_$": round(max_dd, 2),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
    }

# ============================================================
# Plots
# ============================================================

def plot_sharpe_by_fold(results_df: pd.DataFrame) -> None:
    fold_names = [f[0] for f in FOLDS]
    x     = np.arange(len(fold_names))
    width = 0.35
    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for ax, strategy in zip(axes, ["breakout", "pairs"]):
        for i, cfg_name in enumerate(["baseline", "sha_winner"]):
            sub = results_df[(results_df["strategy"] == strategy) &
                             (results_df["config"] == cfg_name)]
            sharpes = []
            for fn in fold_names:
                row = sub[sub["fold"] == fn]
                sharpes.append(row["sharpe"].values[0] if len(row) else np.nan)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, sharpes, width, label=cfg_name,
                          color=colors[i], alpha=0.85)
            for bar, val in zip(bars, sharpes):
                if not np.isnan(val):
                    ypos = bar.get_height() + (0.05 if val >= 0 else -0.25)
                    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Sharpe = 1")
        ax.axhline(0.0, color="gray",  linestyle="-",  linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_names, fontsize=9)
        ax.set_ylabel("Sharpe Ratio (Roll cost)")
        ax.set_title(f"{strategy.capitalize()} — OOS Sharpe by Fold")
        ax.legend(fontsize=9)

    fig.suptitle("Walk-Forward Validation: Baseline vs SHA Winner", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "wfv_sharpe_by_fold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cumulative_pnl(pnl_series: dict) -> None:
    """
    Concatenate test-window PnL series across folds and plot cumulative curves.
    Vertical dashed lines mark fold boundaries.
    """
    fold_names  = [f[0] for f in FOLDS]
    boundaries  = [pd.Timestamp(f[1], tz="UTC") for f in FOLDS[1:]]
    colors      = {"baseline": "steelblue", "sha_winner": "darkorange"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, strategy in zip(axes, ["breakout", "pairs"]):
        for cfg_name in ["baseline", "sha_winner"]:
            key = (strategy, cfg_name)
            parts = [pnl_series[key][fn] for fn in fold_names
                     if fn in pnl_series[key]]
            if not parts:
                continue
            combined = pd.concat(parts).sort_index()
            cum      = combined.cumsum()
            ax.plot(cum.index, cum.values, label=cfg_name,
                    color=colors[cfg_name], linewidth=1.2)

        for ts in boundaries:
            ax.axvline(ts, color="gray", linestyle=":", linewidth=0.9,
                       label="_fold boundary")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylabel("Cumulative Net PnL ($)")
        ax.set_title(f"{strategy.capitalize()} — Cumulative OOS PnL (WFV folds)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Annotate fold labels at top of chart
    for ax in axes:
        ylim = ax.get_ylim()
        for i, (fn, ts_str, _) in enumerate(FOLDS):
            ts = pd.Timestamp(ts_str, tz="UTC")
            ax.text(ts, ylim[1] * 0.98, fn.split("(")[1].rstrip(")"),
                    fontsize=7, color="gray", va="top", ha="left")

    fig.suptitle("Walk-Forward Validation: Cumulative OOS PnL (Roll costs)", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "wfv_cumulative_pnl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

# ============================================================
# Main
# ============================================================

def main():
    df = load_data(DATA_PATH)

    all_results = []
    pnl_series  = {}   # (strategy, config_name) → {fold_name: net_pnl series}

    print("\n" + "=" * 72)
    print("  WALK-FORWARD VALIDATION  (expanding window, Roll costs)")
    print("=" * 72)

    for fold_name, test_start, test_end in FOLDS:
        ts_start = pd.Timestamp(test_start, tz="UTC")
        ts_end   = pd.Timestamp(test_end,   tz="UTC") if test_end else None

        train_mask = df.index < ts_start
        test_mask  = (df.index >= ts_start) if ts_end is None else \
                     (df.index >= ts_start) & (df.index < ts_end)

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())
        print(f"\n{'─' * 72}")
        print(f"  {fold_name}  |  train {n_train:,} bars  |  test {n_test:,} bars")
        print(f"{'─' * 72}")

        # ── Breakout ─────────────────────────────────────────────────────────
        for cfg_name, cfg in BREAKOUT_CONFIGS.items():
            df_run   = run_breakout(df, train_mask, **cfg)
            df_test  = df_run[test_mask]
            m        = compute_metrics(df_test)

            row = {"strategy": "breakout", "config": cfg_name, "fold": fold_name, **m}
            all_results.append(row)

            key = ("breakout", cfg_name)
            pnl_series.setdefault(key, {})[fold_name] = df_test["net_pnl"]

            trades_str = f"{m['n_trades']:>3}" if m['n_trades'] > 0 else "  0"
            print(f"  Breakout  {cfg_name:<12}  Sharpe {m['sharpe']:>7.4f}  "
                  f"Net PnL ${m['total_net_$']:>9,.0f}  Trades {trades_str}")

        # ── Pairs ─────────────────────────────────────────────────────────────
        for cfg_name, cfg in PAIRS_CONFIGS.items():
            df_run, beta = run_pairs(df, train_mask, **cfg)
            df_test      = df_run[test_mask]
            m            = compute_metrics(df_test)

            row = {"strategy": "pairs", "config": cfg_name, "fold": fold_name,
                   "beta_used": round(beta, 4), **m}
            all_results.append(row)

            key = ("pairs", cfg_name)
            pnl_series.setdefault(key, {})[fold_name] = df_test["net_pnl"]

            trades_str = f"{m['n_trades']:>3}" if m['n_trades'] > 0 else "  0"
            print(f"  Pairs     {cfg_name:<12}  Sharpe {m['sharpe']:>7.4f}  "
                  f"Net PnL ${m['total_net_$']:>9,.0f}  Trades {trades_str}  "
                  f"β={beta:.4f}")

    results_df = pd.DataFrame(all_results)

    # ── Summary: mean ± std Sharpe across folds ───────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY — MEAN ± STD SHARPE ACROSS 3 FOLDS  (Roll costs)")
    print("=" * 72)
    print(f"  {'Strategy':<10} {'Config':<14} {'Mean Sharpe':>12} {'Std Sharpe':>11} "
          f"{'Avg Net PnL':>13}")
    print(f"  {'─'*10} {'─'*14} {'─'*12} {'─'*11} {'─'*13}")

    for strategy in ["breakout", "pairs"]:
        for cfg_name in ["baseline", "sha_winner"]:
            sub = results_df[(results_df["strategy"] == strategy) &
                             (results_df["config"] == cfg_name)]
            mean_s = sub["sharpe"].mean()
            std_s  = sub["sharpe"].std(ddof=1)
            avg_pnl = sub["total_net_$"].mean()
            print(f"  {strategy:<10} {cfg_name:<14} {mean_s:>12.4f} {std_s:>11.4f} "
                  f"${avg_pnl:>12,.0f}")

    # Fold-2 sanity check reminder
    fold2_bl = results_df[(results_df["strategy"] == "breakout") &
                          (results_df["config"]   == "baseline") &
                          (results_df["fold"]     == "Fold 2 (Jan 2026)")]["sharpe"].values
    if len(fold2_bl):
        print(f"\n  [Sanity] Breakout baseline Fold-2 Sharpe: {fold2_bl[0]:.4f}  "
              f"(expected ≈ 1.82 from notebook 09 OOS Roll)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    path_csv = os.path.join(TABLES_DIR, "wfv_results.csv")
    results_df.to_csv(path_csv, index=False)
    print(f"\n  Saved: {path_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_sharpe_by_fold(results_df)
    plot_cumulative_pnl(pnl_series)

    print("\nDone.")


if __name__ == "__main__":
    main()
