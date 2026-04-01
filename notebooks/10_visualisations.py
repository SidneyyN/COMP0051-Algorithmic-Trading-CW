"""
COMP0051 Algorithmic Trading Coursework — Combined Visualisations
=================================================================
Generates four publication-quality figures combining both strategies:

  1. Cumulative PnL curves (gross vs net) for breakout and pairs side by side
  2. Drawdown chart for both strategies under each cost model
  3. Position / exposure over time (breakout + pairs on same timeline)
  4. Return distribution histograms (IS vs OOS, gross vs net)

Run from project root:
    python notebooks/10_visualisations.py
"""

import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Configuration
# ============================================================

# Breakout
N            = 200
VOL_WINDOW   = 50
VOL_QUANTILE = 0.6
MAX_HOLD     = 40

# Pairs
BETA           = 0.6780
ROLL_WINDOW    = 100
ENTRY_Z        = 3.0
EXIT_Z         = 0.0
STOP_Z         = 3.0
MIN_HOLD_PAIRS = 24
COOLDOWN       = 20
MAX_HOLD_PAIRS = 384
CAPITAL        = 100_000.0
COST_BPS       = 5.0

VOL_WINDOW_PAIRS = 100
VOL_REF_WINDOW   = 200
VOL_MULT         = 1.2

IS_END        = "2025-12-31"
BARS_PER_YEAR = 35_064

ASSETS    = ["BTC", "ETH", "DOGE"]
DATA_PATH = os.path.join("data", "cleaned", "cleaned_data.parquet")
BREAKOUT_PATH     = os.path.join("notebooks", "04_breakout_strategy.py")
PAIRS_PATH        = os.path.join("notebooks", "07_pairs_strategy_vol_filter.py")
COSTS_PATH        = os.path.join("notebooks", "08_transaction_costs.py")
FIGURES_DIR       = os.path.join("report", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
COLORS = {
    "gross":   "#2196F3",   # blue
    "fixed":   "#4CAF50",   # green
    "roll":    "#FF9800",   # orange
    "cs":      "#F44336",   # red
    "breakout":"#1565C0",
    "pairs":   "#6A1B9A",
    "is_shade":"#E3F2FD",
    "oos_shade":"#FFF3E0",
}

# ============================================================
# Module loading helpers
# ============================================================

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# Regenerate positions + gross PnL
# ============================================================

def recover_breakout(path, breakout_mod):
    df = breakout_mod.load_data(path)
    price_df = pd.read_parquet(path).sort_index()[["BTC_high", "BTC_low"]]
    df = df.join(price_df)
    df = breakout_mod.compute_signals(df, N, VOL_WINDOW, VOL_QUANTILE)
    df = breakout_mod.run_backtest(df, MAX_HOLD)

    df["position_lag"] = df["position"].shift(1).fillna(0)
    df["gross_return"] = df["position_lag"] * df["BTC_ret"]
    df["gross_pnl"]    = CAPITAL * df["gross_return"]
    df["delta_pos"]    = df["position"].diff().abs().fillna(0)
    return df


def recover_pairs(path, pairs_mod):
    df = pairs_mod.load_data(path)
    df = pairs_mod.compute_spread(df, BETA)
    df = pairs_mod.compute_zscore(df, ROLL_WINDOW)
    df = pairs_mod.compute_vol_filter(df)
    df = pairs_mod.generate_positions(df)
    df = pairs_mod.compute_leg_pnl(df, BETA, CAPITAL)
    df["delta_pos"] = df["position"].diff().abs().fillna(0)
    return df

# ============================================================
# Attach net PnL columns for each cost model
# ============================================================

def attach_costs(df_b, df_p, tc_mod, roll_spreads, cs_df):
    """
    Returns (df_b, df_p) each with columns:
      net_pnl_fixed, net_pnl_roll, net_pnl_cs, cost_fixed, cost_roll, cost_cs
    """
    # --- Breakout ---
    for model, spread in [
        ("fixed", COST_BPS / 10_000),
        ("roll",  roll_spreads["BTC"]),
        ("cs",    cs_df["BTC"]),
    ]:
        cost = tc_mod.compute_cost(df_b["position"], spread, df_b["BTC_close"])
        df_b[f"cost_{model}"]    = cost
        df_b[f"net_pnl_{model}"] = df_b["gross_pnl"] - cost

    # --- Pairs ---
    for model, s_btc, s_eth in [
        ("fixed", COST_BPS / 10_000,    COST_BPS / 10_000),
        ("roll",  roll_spreads["BTC"],   roll_spreads["ETH"]),
        ("cs",    cs_df["BTC"],          cs_df["ETH"]),
    ]:
        cost = (
            tc_mod.compute_cost(df_p["position"], s_btc, df_p["BTC_close"]) +
            tc_mod.compute_cost(df_p["position"], s_eth, df_p["ETH_close"])
        )
        df_p[f"cost_{model}"]    = cost
        df_p[f"net_pnl_{model}"] = df_p["gross_pnl"] - cost

    return df_b, df_p

# ============================================================
# Helpers: cumulative PnL and drawdown series
# ============================================================

def cum_pnl(series):
    return series.cumsum()


def drawdown(series):
    c = series.cumsum()
    return c - c.cummax()


def _ts(s, tz):
    """Parse date string, matching the timezone of the index."""
    ts = pd.Timestamp(s)
    return ts.tz_localize(tz) if tz is not None else ts


def shade_is_oos(ax, df):
    """Shade IS and OOS regions."""
    tz        = df.index.tz
    is_start  = df.index[0]
    is_end    = _ts(IS_END, tz)
    oos_end   = df.index[-1]
    ax.axvspan(is_start, is_end,  alpha=0.08, color=COLORS["is_shade"],  zorder=0, label="_IS")
    ax.axvspan(is_end,   oos_end, alpha=0.08, color=COLORS["oos_shade"], zorder=0, label="_OOS")
    ax.axvline(is_end, color="grey", linewidth=0.8, linestyle="--", zorder=1)


def label_is_oos(ax, df):
    """Add IS / OOS text annotations."""
    tz      = df.index.tz
    is_end  = _ts(IS_END, tz)
    is_mid  = df.index[0] + (is_end - df.index[0]) / 2
    oos_mid = is_end + (df.index[-1] - is_end) / 2
    y_top = ax.get_ylim()[1]
    ax.text(is_mid,  y_top * 0.96, "In-sample",     ha="center", va="top", fontsize=7, color="grey")
    ax.text(oos_mid, y_top * 0.96, "Out-of-sample", ha="center", va="top", fontsize=7, color="grey")

# ============================================================
# Figure 1 — Cumulative PnL: gross vs net (IS + OOS)
# ============================================================

def plot_cumulative_pnl(df_b, df_p):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, df, title in [
        (axes[0], df_b, "Breakout Strategy"),
        (axes[1], df_p, "Pairs Trading Strategy"),
    ]:
        shade_is_oos(ax, df)

        ax.plot(df.index, cum_pnl(df["gross_pnl"]),    color=COLORS["gross"],
                linewidth=1.2, label="Gross",       zorder=3)
        ax.plot(df.index, cum_pnl(df["net_pnl_fixed"]), color=COLORS["fixed"],
                linewidth=1.0, label="Net (Fixed)",  zorder=3)
        ax.plot(df.index, cum_pnl(df["net_pnl_roll"]),  color=COLORS["roll"],
                linewidth=1.0, label="Net (Roll)",   zorder=3)
        ax.plot(df.index, cum_pnl(df["net_pnl_cs"]),    color=COLORS["cs"],
                linewidth=1.0, label="Net (CS)",     zorder=3)

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative PnL ($)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        label_is_oos(ax, df)

    fig.suptitle("Cumulative PnL — Gross vs Net (Three Cost Models)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vis_cumulative_pnl.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ============================================================
# Figure 2 — Drawdown
# ============================================================

def plot_drawdown(df_b, df_p):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    for ax, df, title in [
        (axes[0], df_b, "Breakout Strategy — Drawdown"),
        (axes[1], df_p, "Pairs Trading Strategy — Drawdown"),
    ]:
        shade_is_oos(ax, df)

        ax.fill_between(df.index, drawdown(df["net_pnl_fixed"]), 0,
                        color=COLORS["fixed"], alpha=0.4, label="Net (Fixed)", zorder=2)
        ax.fill_between(df.index, drawdown(df["net_pnl_roll"]),  0,
                        color=COLORS["roll"],  alpha=0.4, label="Net (Roll)",  zorder=2)
        ax.fill_between(df.index, drawdown(df["net_pnl_cs"]),    0,
                        color=COLORS["cs"],    alpha=0.35, label="Net (CS)",   zorder=2)
        ax.plot(df.index, drawdown(df["gross_pnl"]),
                color=COLORS["gross"], linewidth=1.2, label="Gross", zorder=3)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Drawdown ($)", fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Drawdown — Gross vs Net (Three Cost Models)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vis_drawdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ============================================================
# Figure 3 — Position / Exposure over time
# ============================================================

def plot_positions(df_b, df_p):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    # Breakout: position ∈ {-1, 0, +1}; exposure = |position| × CAPITAL
    ax = axes[0]
    shade_is_oos(ax, df_b)
    ax.fill_between(df_b.index, df_b["position"] * CAPITAL, 0,
                    where=df_b["position"] > 0,
                    color=COLORS["breakout"], alpha=0.6, label="Long  (+$100k)", zorder=2)
    ax.fill_between(df_b.index, df_b["position"] * CAPITAL, 0,
                    where=df_b["position"] < 0,
                    color=COLORS["cs"],       alpha=0.6, label="Short (−$100k)", zorder=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Notional ($)", fontsize=9)
    ax.set_title("Breakout Strategy — Position / Exposure", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    label_is_oos(ax, df_b)

    # Pairs: position ∈ {-1, 0, +1} (represents spread direction)
    ax = axes[1]
    shade_is_oos(ax, df_p)
    ax.fill_between(df_p.index, df_p["position"], 0,
                    where=df_p["position"] > 0,
                    color=COLORS["pairs"], alpha=0.6, label="Long spread",  zorder=2)
    ax.fill_between(df_p.index, df_p["position"], 0,
                    where=df_p["position"] < 0,
                    color=COLORS["roll"],  alpha=0.6, label="Short spread", zorder=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Position (spread units)", fontsize=9)
    ax.set_title("Pairs Trading Strategy — Position (BTC–ETH Spread Direction)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    label_is_oos(ax, df_p)

    fig.suptitle("Position / Exposure Over Time", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vis_positions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ============================================================
# Figure 4 — Return distribution histograms
# ============================================================

def _bar_returns(df, pnl_col):
    """Bar-level return = PnL / CAPITAL."""
    return df[pnl_col] / CAPITAL


def plot_return_distributions(df_b, df_p):
    is_mask_b  = df_b.index <= IS_END
    oos_mask_b = df_b.index >  IS_END
    is_mask_p  = df_p.index <= IS_END
    oos_mask_p = df_p.index >  IS_END

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    def _hist(ax, series, title, color, bins=60):
        series = series.dropna()
        # Clip extreme tails for display clarity
        p1, p99 = series.quantile(0.01), series.quantile(0.99)
        series_clipped = series.clip(p1, p99)
        ax.hist(series_clipped, bins=bins, color=color, alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(series.mean(), color="darkred", linewidth=1.0, linestyle="-",
                   label=f"Mean: {series.mean()*100:.3f}%")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Bar return (%)", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
        ax.legend(fontsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Breakout IS — gross vs net (Roll)
    ax = axes[0][0]
    r_gross_is = _bar_returns(df_b[is_mask_b],  "gross_pnl")
    r_net_is   = _bar_returns(df_b[is_mask_b],  "net_pnl_roll")
    p1 = min(r_gross_is.quantile(0.01), r_net_is.quantile(0.01))
    p99 = max(r_gross_is.quantile(0.99), r_net_is.quantile(0.99))
    bins = np.linspace(p1, p99, 61)
    ax.hist(r_gross_is.clip(p1, p99), bins=bins, color=COLORS["gross"],
            alpha=0.6, label="Gross", edgecolor="none")
    ax.hist(r_net_is.clip(p1, p99),   bins=bins, color=COLORS["roll"],
            alpha=0.6, label="Net (Roll)", edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Breakout — IS Return Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Bar return (%)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Breakout OOS — gross vs net (Roll)
    ax = axes[0][1]
    r_gross_oos = _bar_returns(df_b[oos_mask_b], "gross_pnl")
    r_net_oos   = _bar_returns(df_b[oos_mask_b], "net_pnl_roll")
    p1 = min(r_gross_oos.quantile(0.01), r_net_oos.quantile(0.01))
    p99 = max(r_gross_oos.quantile(0.99), r_net_oos.quantile(0.99))
    bins = np.linspace(p1, p99, 61)
    ax.hist(r_gross_oos.clip(p1, p99), bins=bins, color=COLORS["gross"],
            alpha=0.6, label="Gross", edgecolor="none")
    ax.hist(r_net_oos.clip(p1, p99),   bins=bins, color=COLORS["roll"],
            alpha=0.6, label="Net (Roll)", edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Breakout — OOS Return Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Bar return (%)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Pairs IS — gross vs net (Roll)
    ax = axes[1][0]
    r_gross_is = _bar_returns(df_p[is_mask_p],  "gross_pnl")
    r_net_is   = _bar_returns(df_p[is_mask_p],  "net_pnl_roll")
    p1 = min(r_gross_is.quantile(0.01), r_net_is.quantile(0.01))
    p99 = max(r_gross_is.quantile(0.99), r_net_is.quantile(0.99))
    bins = np.linspace(p1, p99, 61)
    ax.hist(r_gross_is.clip(p1, p99), bins=bins, color=COLORS["gross"],
            alpha=0.6, label="Gross", edgecolor="none")
    ax.hist(r_net_is.clip(p1, p99),   bins=bins, color=COLORS["roll"],
            alpha=0.6, label="Net (Roll)", edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Pairs Trading — IS Return Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Bar return (%)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Pairs OOS — gross vs net (Roll)
    ax = axes[1][1]
    r_gross_oos = _bar_returns(df_p[oos_mask_p], "gross_pnl")
    r_net_oos   = _bar_returns(df_p[oos_mask_p], "net_pnl_roll")
    p1 = min(r_gross_oos.quantile(0.01), r_net_oos.quantile(0.01))
    p99 = max(r_gross_oos.quantile(0.99), r_net_oos.quantile(0.99))
    bins = np.linspace(p1, p99, 61)
    ax.hist(r_gross_oos.clip(p1, p99), bins=bins, color=COLORS["gross"],
            alpha=0.6, label="Gross", edgecolor="none")
    ax.hist(r_net_oos.clip(p1, p99),   bins=bins, color=COLORS["roll"],
            alpha=0.6, label="Net (Roll)", edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Pairs Trading — OOS Return Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Bar return (%)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Return Distributions — IS vs OOS, Gross vs Net (Roll Cost)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vis_return_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ============================================================
# Main
# ============================================================

def main():
    print("Loading modules...")
    breakout_mod = load_module("breakout", BREAKOUT_PATH)
    pairs_mod    = load_module("pairs",    PAIRS_PATH)
    tc_mod       = load_module("tc",       COSTS_PATH)

    print("Recovering positions...")
    df_b = recover_breakout(DATA_PATH, breakout_mod)
    df_p = recover_pairs(DATA_PATH, pairs_mod)

    print("Computing transaction costs...")
    raw = tc_mod.load_data(DATA_PATH)
    roll_spreads = {a: tc_mod.roll_spread(raw[f"{a}_close"]) for a in ASSETS}
    cs_df        = pd.DataFrame(
        {a: tc_mod.cs_spread(raw[f"{a}_high"], raw[f"{a}_low"]) for a in ASSETS},
        index=raw.index,
    )

    print("Attaching cost columns...")
    df_b, df_p = attach_costs(df_b, df_p, tc_mod, roll_spreads, cs_df)

    print("Plotting Figure 1 — Cumulative PnL...")
    plot_cumulative_pnl(df_b, df_p)

    print("Plotting Figure 2 — Drawdown...")
    plot_drawdown(df_b, df_p)

    print("Plotting Figure 3 — Positions...")
    plot_positions(df_b, df_p)

    print("Plotting Figure 4 — Return distributions...")
    plot_return_distributions(df_b, df_p)

    print("\nDone. All figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
