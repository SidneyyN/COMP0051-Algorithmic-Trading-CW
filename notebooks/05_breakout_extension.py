import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration — shared with baseline (04_breakout_strategy.py)
# ============================================================

N            = 200         # Donchian channel lookback (bars)
VOL_WINDOW   = 50          # rolling vol window
MAX_HOLD     = 40          # max holding period (bars)
CAPITAL      = 100_000.0   # gross notional ($)
COST_BPS     = 5.0         # default cost in basis points

IS_END        = "2025-12-31"
BARS_PER_YEAR = 35_064     # 365.25 × 24 × 4

# Extension parameters
THRESHOLD  = 0.001   # Test A/B: price must exceed band by 10 bps before entry
VOL_Q_LOW  = 0.2     # Test B: lower vol quantile — do not trade below
VOL_Q_HIGH = 0.8     # Test B: upper vol quantile — do not trade above

DATA_PATH   = os.path.join("data", "cleaned", "cleaned_data.parquet")
FIGURES_DIR = os.path.join("report", "figures")
TABLES_DIR  = os.path.join("report", "tables")

for d in [FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Step 1 — Load Data
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df[["BTC_close", "BTC_ret"]].dropna()
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    return df

# ============================================================
# Step 2 — Compute signals
#
#  version="baseline"   → single-sided vol filter (60th pct), no threshold
#  version="threshold"  → single-sided vol filter + 10 bps threshold on breach
#  version="full"       → banded vol filter (20th–80th pct) + 10 bps threshold
# ============================================================

def compute_signals(df, N, vol_window, threshold, vol_q_low, vol_q_high, version):
    df = df.copy()

    # Donchian bands — lagged one bar (no look-ahead)
    df['upper_band'] = df['BTC_close'].rolling(N).max().shift(1)
    df['lower_band']  = df['BTC_close'].rolling(N).min().shift(1)

    # Rolling volatility
    df['rolling_vol'] = df['BTC_ret'].rolling(vol_window).std()

    is_mask = df.index <= IS_END

    if version == "baseline":
        # Raw band breach — contrarian fade
        df['long_entry']  = df['BTC_close'] < df['lower_band']
        df['short_entry'] = df['BTC_close'] > df['upper_band']
        # Single-sided vol filter (matching 04_breakout_strategy.py exactly)
        vol_threshold = df.loc[is_mask, 'rolling_vol'].quantile(0.6)
        df['vol_filter'] = df['rolling_vol'] > vol_threshold

    elif version == "threshold":
        # Only enter after a more extreme overextension
        df['long_entry']  = df['BTC_close'] < df['lower_band']  * (1 - threshold)
        df['short_entry'] = df['BTC_close'] > df['upper_band'] * (1 + threshold)
        # Same single-sided vol filter as baseline
        vol_threshold = df.loc[is_mask, 'rolling_vol'].quantile(0.6)
        df['vol_filter'] = df['rolling_vol'] > vol_threshold

    elif version == "full":
        # Threshold-adjusted breach
        df['long_entry']  = df['BTC_close'] < df['lower_band']  * (1 - threshold)
        df['short_entry'] = df['BTC_close'] > df['upper_band'] * (1 + threshold)
        # Middle-range vol filter: skip both quiet and explosive regimes
        low_vol  = df.loc[is_mask, 'rolling_vol'].quantile(vol_q_low)
        high_vol = df.loc[is_mask, 'rolling_vol'].quantile(vol_q_high)
        df['vol_filter'] = (
            (df['rolling_vol'] > low_vol) &
            (df['rolling_vol'] < high_vol)
        )

    else:
        raise ValueError(f"Unknown version: {version!r}")

    df['long_entry']  = df['long_entry']  & df['vol_filter']
    df['short_entry'] = df['short_entry'] & df['vol_filter']

    return df

# ============================================================
# Step 3 — Run backtest
# ============================================================

def run_backtest(df, max_hold):
    df = df.copy()
    n = len(df)
    position    = np.zeros(n, dtype=int)
    hold_bars   = 0
    current_pos = 0

    for t in range(1, n):
        long_sig  = df['long_entry'].iloc[t]
        short_sig = df['short_entry'].iloc[t]

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

    df['position'] = position
    return df

# ============================================================
# Step 4 — Compute PnL
# ============================================================

def compute_pnl(df, capital, cost_bps):
    df = df.copy()
    df['position_lag'] = df['position'].shift(1).fillna(0)
    df['gross_return'] = df['position_lag'] * df['BTC_ret']

    # Count +1→-1 reversals as 2 units of trading
    df['trade']     = df['position'].diff().abs().fillna(0)
    cost_per_unit   = capital * cost_bps * 0.0001
    df['cost']      = df['trade'] * cost_per_unit
    df['gross_pnl'] = capital * df['gross_return']
    df['net_pnl']   = df['gross_pnl'] - df['cost']

    return df

# ============================================================
# Step 5 — Performance metrics
# ============================================================

def compute_metrics(df, bars_per_year, capital):
    df  = df.copy()
    ret = df['net_pnl'] / capital

    total_net   = df['net_pnl'].sum()
    total_gross = df['gross_pnl'].sum()
    total_cost  = df['cost'].sum()

    mean_r = ret.mean()
    std_r  = ret.std(ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0 else np.nan

    neg_r         = ret[ret < 0]
    sortino_denom = neg_r.std(ddof=1)
    sortino = (mean_r / sortino_denom * np.sqrt(bars_per_year)) if sortino_denom > 0 else np.nan

    cum_pnl     = df['net_pnl'].cumsum()
    rolling_max = cum_pnl.cummax()
    max_dd      = (cum_pnl - rolling_max).min()
    calmar      = (total_net / abs(max_dd)) if abs(max_dd) > 0 else np.nan

    # Trade-level win rate
    trade_entry = (df['position'] != 0) & (df['position'].shift(1) == 0)
    df['trade_id'] = trade_entry.cumsum()
    df.loc[df['position'] == 0, 'trade_id'] = 0
    trade_pnl = df[df['trade_id'] > 0].groupby('trade_id')['net_pnl'].sum()
    n_trades  = len(trade_pnl)
    win_rate  = (trade_pnl > 0).mean() if n_trades > 0 else np.nan

    return {
        "total_gross_$":  round(total_gross, 2),
        "total_net_$":    round(total_net, 2),
        "total_cost_$":   round(total_cost, 2),
        "sharpe":         round(sharpe,  4),
        "sortino":        round(sortino, 4),
        "max_drawdown_$": round(max_dd,  2),
        "calmar":         round(calmar,  4),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
    }

# ============================================================
# Print helpers
# ============================================================

def print_metrics(label: str, m: dict) -> None:
    print(f"\n{'='*44}")
    print(f"  {label}")
    print(f"{'='*44}")
    rows = [
        ("Total gross PnL",  f"${m['total_gross_$']:>12,.2f}"),
        ("Total net PnL",    f"${m['total_net_$']:>12,.2f}"),
        ("Total costs",      f"${m['total_cost_$']:>12,.2f}"),
        ("Sharpe ratio",     f"{m['sharpe']:>13.4f}"),
        ("Sortino ratio",    f"{m['sortino']:>13.4f}"),
        ("Max drawdown",     f"${m['max_drawdown_$']:>12,.2f}"),
        ("Calmar ratio",     f"{m['calmar']:>13.4f}"),
        ("No. of trades",    f"{m['n_trades']:>13}"),
        ("Win rate",         f"{m['win_rate']:>13.2%}" if not np.isnan(m['win_rate']) else f"{'N/A':>13}"),
    ]
    for name, value in rows:
        print(f"  {name:<20} {value}")
    print(f"{'='*44}")

# ============================================================
# Comparison table
# ============================================================

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: {version_label: {"IS": metrics_dict, "OOS": metrics_dict}}
    Returns one row per version with IS and OOS metrics side by side.
    """
    records = {}
    for label, split in results.items():
        is_m  = split["IS"]
        oos_m = split["OOS"]
        records[label] = {
            "IS net PnL ($)":     is_m["total_net_$"],
            "OOS net PnL ($)":    oos_m["total_net_$"],
            "IS Sharpe":          is_m["sharpe"],
            "OOS Sharpe":         oos_m["sharpe"],
            "IS Sortino":         is_m["sortino"],
            "OOS Sortino":        oos_m["sortino"],
            "IS trades":          is_m["n_trades"],
            "OOS trades":         oos_m["n_trades"],
            "IS total cost ($)":  is_m["total_cost_$"],
            "OOS total cost ($)": oos_m["total_cost_$"],
            "IS win rate":        is_m["win_rate"],
            "OOS win rate":       oos_m["win_rate"],
            "IS max DD ($)":      is_m["max_drawdown_$"],
            "OOS max DD ($)":     oos_m["max_drawdown_$"],
        }
    tbl = pd.DataFrame(records).T
    tbl.index.name = "version"
    return tbl

# ============================================================
# Plots
# ============================================================

def plot_equity_curves(version_dfs: dict):
    """Cumulative net PnL for all three versions, IS and OOS side by side."""
    colours = {
        "Baseline (04)":                  "steelblue",
        "Test A — Threshold":             "darkorange",
        "Test B — Threshold + Vol Band":  "seagreen",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for label, df in version_dfs.items():
        c = colours.get(label, "black")
        is_df  = df[df.index <= IS_END]
        oos_df = df[df.index >  IS_END]
        axes[0].plot(is_df.index,  is_df['net_pnl'].cumsum(),  label=label, color=c, linewidth=1.2)
        axes[1].plot(oos_df.index, oos_df['net_pnl'].cumsum(), label=label, color=c, linewidth=1.2)

    for ax, title in zip(axes, ["In-Sample (Sep–Dec 2025)", "Out-of-Sample (Jan–Feb 2026)"]):
        ax.set_title(title)
        ax.set_ylabel("Cumulative Net PnL ($)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Breakout Extension — Net PnL Comparison", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "breakout_extension_equity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_signals(df, label):
    """Price + bands + entry signals for the given version."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['BTC_close'],  color='black',  linewidth=0.6, label='BTC close')
    ax.plot(df.index, df['upper_band'], color='blue',   linewidth=0.7, linestyle='--', label='Upper band')
    ax.plot(df.index, df['lower_band'], color='orange', linewidth=0.7, linestyle='--', label='Lower band')

    longs  = df[df['long_entry']]
    shorts = df[df['short_entry']]
    ax.scatter(longs.index,  longs['BTC_close'],  marker='^', color='green', s=15, zorder=5, label='Long entry')
    ax.scatter(shorts.index, shorts['BTC_close'], marker='v', color='red',   s=15, zorder=5, label='Short entry')
    ax.axvline(pd.Timestamp(IS_END, tz='UTC'), color='black', linestyle=':', label='IS/OOS split')

    ax.set_title(f'Breakout Extension ({label}) — Price, Bands, Signals')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()

    slug = label.lower().replace(" ", "_").replace("—", "").replace("+", "").replace("(", "").replace(")", "")
    slug = "_".join(slug.split())
    path = os.path.join(FIGURES_DIR, f"breakout_extension_signals_{slug}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ============================================================
# Main
# ============================================================

def main():
    raw = load_data(DATA_PATH)

    is_mask  = raw.index <= IS_END
    oos_mask = raw.index >  IS_END

    version_configs = [
        ("Baseline (04)",                "baseline"),
        ("Test A — Threshold",           "threshold"),
        ("Test B — Threshold + Vol Band", "full"),
    ]

    results     = {}
    version_dfs = {}

    for label, ver in version_configs:
        print(f"\n{'#'*52}")
        print(f"  Running: {label}")
        print(f"{'#'*52}")

        df = compute_signals(
            raw, N, VOL_WINDOW,
            threshold=THRESHOLD,
            vol_q_low=VOL_Q_LOW,
            vol_q_high=VOL_Q_HIGH,
            version=ver,
        )
        df = run_backtest(df, MAX_HOLD)
        df = compute_pnl(df, CAPITAL, COST_BPS)

        is_m  = compute_metrics(df[is_mask].copy(),  BARS_PER_YEAR, CAPITAL)
        oos_m = compute_metrics(df[oos_mask].copy(), BARS_PER_YEAR, CAPITAL)

        print_metrics(f"{label}  |  IN-SAMPLE",     is_m)
        print_metrics(f"{label}  |  OUT-OF-SAMPLE", oos_m)

        results[label]     = {"IS": is_m, "OOS": oos_m}
        version_dfs[label] = df

    # Comparison table
    tbl = build_comparison_table(results)
    print("\n\n" + "="*80)
    print("  COMPARISON TABLE")
    print("="*80)
    print(tbl.to_string())
    tbl_path = os.path.join(TABLES_DIR, "breakout_extension_comparison.csv")
    tbl.to_csv(tbl_path)
    print(f"\nSaved: {tbl_path}")

    # Plots
    plot_equity_curves(version_dfs)
    for label, df in version_dfs.items():
        plot_signals(df, label)


if __name__ == "__main__":
    main()
