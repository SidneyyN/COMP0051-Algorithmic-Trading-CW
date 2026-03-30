"""
COMP0051 Algorithmic Trading Coursework — Pairs Trading Strategy
=================================================================
BTC–ETH mean-reversion pairs trading using Engle–Granger cointegration.

Strategy design:
  - Spread:      spread_t = log(BTC_t) - beta * log(ETH_t),  beta = 0.6780
  - Signal:      rolling z-score over 100-bar window
  - Entry:       |z| > 3 (long spread if z < -3, short spread if z > +3)
  - Exit:        z crosses 0 (full mean reversion)
  - Stop-loss:   |z| > 3 after min-hold (no-progress check)
  - Min-hold:    24 bars (~6 hours, prevents premature exit)
  - Cooldown:    20 bars after exit before re-entering
  - Time-stop:   384 bars (~4 days, close to half-life of 392 bars)
  - Capital:     $100,000 gross notional, dollar-neutral leg sizing
  - Costs:       proportional to traded notional (sensitivity sweep)

Run from project root:
    python notebooks/05_pairs_strategy.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Configuration
# ============================================================

BETA          = 0.6780          # OLS hedge ratio from EDA cointegration
ROLL_WINDOW   = 100             # bars for rolling z-score
ENTRY_Z       = 3.0             # enter when |z| > ENTRY_Z
EXIT_Z        = 0.0             # exit when z crosses 0 (full mean reversion)
STOP_Z        = 3.0             # no-progress stop after min-hold: exit if |z| still > STOP_Z
MIN_HOLD      = 24              # minimum holding period (bars, ~6 hours)
COOLDOWN      = 20              # bars to wait after exit before re-entering
MAX_HOLD      = 384             # max holding period (bars, ~4 days)
CAPITAL       = 100_000.0       # gross notional ($)
COST_BPS      = 5.0             # transaction cost in basis points (default)

IS_END        = "2025-12-31"    # in-sample / out-of-sample boundary
BARS_PER_YEAR = 35_064          # 365.25 × 24 × 4

DATA_PATH     = os.path.join("data", "cleaned", "cleaned_data.parquet")
FIGURES_DIR   = os.path.join("report", "figures")
TABLES_DIR    = os.path.join("report", "tables")

for d in [FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Step 1 — Load Data
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df[["BTC_close", "ETH_close", "BTC_ret", "ETH_ret"]].dropna()
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    return df

# ============================================================
# Step 2 — Compute Spread
# ============================================================

def compute_spread(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    df = df.copy()
    df["log_btc"] = np.log(df["BTC_close"])
    df["log_eth"] = np.log(df["ETH_close"])
    df["spread"]  = df["log_btc"] - beta * df["log_eth"]
    return df

# ============================================================
# Step 3 — Rolling Z-Score
# ============================================================

def compute_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    roll_mean = df["spread"].rolling(window, min_periods=window)
    df["spread_mean"] = roll_mean.mean()
    df["spread_std"]  = roll_mean.std()
    df["zscore"]      = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    return df

# ============================================================
# Step 4-8 — State Machine: Generate Positions
# ============================================================

def generate_positions(df: pd.DataFrame,
                       entry_z:  float = ENTRY_Z,
                       exit_z:   float = EXIT_Z,
                       stop_z:   float = STOP_Z,
                       min_hold: int   = MIN_HOLD,
                       max_hold: int   = MAX_HOLD,
                       cooldown: int   = COOLDOWN) -> pd.DataFrame:
    """
    Iterate bar-by-bar to build position states without lookahead bias.

    State:
        0  = flat
       +1  = long spread  (long BTC, short ETH)
       -1  = short spread (short BTC, long ETH)

    Controls:
        min_hold  — exits blocked for first min_hold bars after entry
        cooldown  — re-entry blocked for cooldown bars after any exit
        stop_z    — after min_hold, exit if |z| > stop_z (no-progress check)
        max_hold  — force exit after max_hold bars regardless
    """
    n = len(df)
    z = df["zscore"].values

    position      = np.zeros(n, dtype=np.int8)
    hold_count    = 0
    cooldown_left = 0
    state         = 0

    for i in range(1, n):
        zi = z[i - 1]   # signal uses previous bar's z-score (no lookahead)

        if np.isnan(zi):
            position[i] = 0
            state        = 0
            hold_count   = 0
            continue

        # --- already in a trade ---
        if state != 0:
            hold_count += 1
            exiting = False

            # only allow exits after minimum holding period
            if hold_count >= min_hold:
                # normal exit: spread has reverted through 0
                if state == 1 and zi >= -exit_z:
                    exiting = True
                elif state == -1 and zi <= exit_z:
                    exiting = True

                # no-progress stop: spread still hasn't moved toward 0
                if abs(zi) > stop_z:
                    exiting = True

            # time-stop: always applies regardless of min_hold
            if hold_count >= max_hold:
                exiting = True

            if exiting:
                state         = 0
                hold_count    = 0
                cooldown_left = cooldown

        # --- flat: decrement cooldown, then check for entry ---
        if state == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                if zi < -entry_z:
                    state      = 1
                    hold_count = 0
                elif zi > entry_z:
                    state      = -1
                    hold_count = 0

        position[i] = state

    df = df.copy()
    df["position"] = position
    return df

# ============================================================
# Step 9-10 — Leg-Level PnL
# ============================================================

def compute_leg_pnl(df: pd.DataFrame,
                    beta: float   = BETA,
                    capital: float = CAPITAL) -> pd.DataFrame:
    """
    Dollar-neutral notional allocation:
        btc_notional = capital / (1 + beta)
        eth_notional = beta * btc_notional

    PnL uses position LAGGED by one bar (position set at close of bar i,
    applied to return of bar i+1).
    """
    df = df.copy()

    btc_notional = capital / (1.0 + beta)
    eth_notional = beta * btc_notional

    pos = df["position"].shift(1).fillna(0)

    df["btc_pnl"]   = pos * btc_notional  * df["BTC_ret"]
    df["eth_pnl"]   = pos * (-eth_notional) * df["ETH_ret"]
    df["gross_pnl"] = df["btc_pnl"] + df["eth_pnl"]
    return df

# ============================================================
# Step 11 — Transaction Costs
# ============================================================

def apply_transaction_costs(df: pd.DataFrame,
                             beta: float   = BETA,
                             capital: float = CAPITAL,
                             cost_bps: float = COST_BPS) -> pd.DataFrame:
    """
    Cost = cost_bps/10000 × (btc_notional + eth_notional) × |position change|
    Costs are incurred whenever position changes (entry, exit, flip).
    """
    df = df.copy()
    btc_notional = capital / (1.0 + beta)
    eth_notional = beta * btc_notional
    gross_notional = btc_notional + eth_notional

    pos_change  = df["position"].diff().abs().fillna(0)
    # A flip (1→-1) generates change of 2; cost for two legs is correct as-is
    df["cost"]    = (cost_bps / 10_000.0) * gross_notional * pos_change
    df["net_pnl"] = df["gross_pnl"] - df["cost"]
    return df

# ============================================================
# Step 12 — Extract Trade Log
# ============================================================

def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Build a row-per-trade summary dataframe."""
    trades = []
    pos    = df["position"].values
    times  = df.index
    z      = df["zscore"].values
    gp     = df["gross_pnl"].values
    np_    = df["net_pnl"].values

    in_trade  = False
    entry_idx = None
    direction = 0

    for i in range(len(df)):
        if not in_trade and pos[i] != 0:
            in_trade  = True
            entry_idx = i
            direction = pos[i]
            cumgross  = 0.0
            cumnet    = 0.0

        if in_trade:
            cumgross += gp[i]
            cumnet   += np_[i]

            exiting = (i > entry_idx) and (pos[i] == 0 or
                        (pos[i] != direction))

            if exiting or i == len(df) - 1:
                exit_idx = i
                # determine exit reason
                zi_prev = z[exit_idx - 1] if exit_idx > 0 else np.nan
                if abs(zi_prev) > STOP_Z:
                    reason = "stop-loss"
                elif (exit_idx - entry_idx) >= MAX_HOLD:
                    reason = "time-stop"
                else:
                    reason = "mean-reversion"

                trades.append({
                    "entry_time":   times[entry_idx],
                    "exit_time":    times[exit_idx],
                    "direction":    "long_spread" if direction == 1 else "short_spread",
                    "entry_z":      round(z[entry_idx], 4),
                    "exit_z":       round(z[exit_idx], 4) if not np.isnan(z[exit_idx]) else np.nan,
                    "holding_bars": exit_idx - entry_idx,
                    "gross_pnl":    round(cumgross, 2),
                    "net_pnl":      round(cumnet, 2),
                    "exit_reason":  reason,
                })
                in_trade  = False
                direction = 0

    return pd.DataFrame(trades)

# ============================================================
# Step 13 — Performance Metrics
# ============================================================

def compute_metrics(df: pd.DataFrame, trades: pd.DataFrame,
                    label: str = "full") -> dict:
    ret = df["net_pnl"] / CAPITAL   # normalised returns

    total_gross = df["gross_pnl"].sum()
    total_net   = df["net_pnl"].sum()
    total_cost  = df["cost"].sum()

    mean_r = ret.mean()
    std_r  = ret.std(ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(BARS_PER_YEAR)) if std_r > 0 else np.nan

    neg_r   = ret[ret < 0]
    sortino_denom = neg_r.std(ddof=1)
    sortino = (mean_r / sortino_denom * np.sqrt(BARS_PER_YEAR)) if sortino_denom > 0 else np.nan

    cum_pnl = df["net_pnl"].cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown    = cum_pnl - rolling_max
    max_dd      = drawdown.min()
    calmar = (total_net / CAPITAL * BARS_PER_YEAR / len(df)) / (-max_dd / CAPITAL + 1e-10)

    n_trades     = len(trades)
    win_rate     = (trades["net_pnl"] > 0).mean() if n_trades > 0 else np.nan
    avg_hold     = trades["holding_bars"].mean() if n_trades > 0 else np.nan
    avg_pnl      = trades["net_pnl"].mean() if n_trades > 0 else np.nan

    pos_diff = df["position"].diff().abs()
    turnover = pos_diff.sum() / (2 * len(df))

    metrics = {
        "label":          label,
        "total_gross_$":  round(total_gross, 2),
        "total_net_$":    round(total_net, 2),
        "total_cost_$":   round(total_cost, 2),
        "sharpe":         round(sharpe, 4),
        "sortino":        round(sortino, 4),
        "max_drawdown_$": round(max_dd, 2),
        "calmar":         round(calmar, 4),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        "avg_hold_bars":  round(avg_hold, 1) if not np.isnan(avg_hold) else np.nan,
        "avg_pnl_$":      round(avg_pnl, 2) if not np.isnan(avg_pnl) else np.nan,
        "turnover":       round(turnover, 6),
    }
    return metrics

# ============================================================
# Step 14 — Plots
# ============================================================

def plot_spread_zscore(df: pd.DataFrame, is_end: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(df.index, df["spread"], lw=0.7, color="steelblue", label="Spread")
    ax1.plot(df.index, df["spread_mean"], lw=1.0, color="orange", linestyle="--", label="Rolling mean")
    ax1.plot(df.index, df["spread_mean"] + 2 * df["spread_std"], lw=0.8,
             color="red", linestyle=":", label="+2σ entry")
    ax1.plot(df.index, df["spread_mean"] - 2 * df["spread_std"], lw=0.8,
             color="green", linestyle=":", label="−2σ entry")
    ax1.axvline(pd.Timestamp(is_end, tz="UTC"), color="black",
                linestyle="--", lw=1.2, label="IS/OOS split")
    ax1.set_ylabel("Spread (log units)")
    ax1.set_title("BTC–ETH Spread with Rolling Bands")
    ax1.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    ax2.plot(df.index, df["zscore"], lw=0.6, color="purple", label="Z-score")
    for level, col, ls in [(ENTRY_Z, "red", "--"), (-ENTRY_Z, "green", "--"),
                            (STOP_Z, "darkred", ":"), (-STOP_Z, "darkgreen", ":"),
                            (EXIT_Z, "orange", "-."), (-EXIT_Z, "orange", "-.")]:
        ax2.axhline(level, color=col, linestyle=ls, lw=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axvline(pd.Timestamp(is_end, tz="UTC"), color="black",
                linestyle="--", lw=1.2)
    ax2.set_ylabel("Z-score")
    ax2.set_title("Rolling Z-Score with Entry/Exit/Stop Thresholds")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pairs_spread_zscore.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_equity_curve(df: pd.DataFrame, is_end: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    gross_cum = df["gross_pnl"].cumsum()
    net_cum   = df["net_pnl"].cumsum()

    ax1 = axes[0]
    ax1.plot(df.index, gross_cum, lw=1.0, color="steelblue", label="Gross PnL")
    ax1.plot(df.index, net_cum,   lw=1.0, color="darkorange", label="Net PnL")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.axvline(pd.Timestamp(is_end, tz="UTC"), color="black",
                linestyle="--", lw=1.2, label="IS/OOS split")
    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.set_title("Pairs Strategy — Cumulative PnL")
    ax1.legend(fontsize=9)

    rolling_max = net_cum.cummax()
    drawdown    = net_cum - rolling_max

    ax2 = axes[1]
    ax2.fill_between(df.index, drawdown, 0, color="red", alpha=0.4, label="Drawdown")
    ax2.axvline(pd.Timestamp(is_end, tz="UTC"), color="black",
                linestyle="--", lw=1.2)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_title("Drawdown")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pairs_equity_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_positions(df: pd.DataFrame, is_end: str):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(df.index, df["position"], 0,
                    where=df["position"] > 0, color="green", alpha=0.6, label="Long spread")
    ax.fill_between(df.index, df["position"], 0,
                    where=df["position"] < 0, color="red", alpha=0.6, label="Short spread")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(pd.Timestamp(is_end, tz="UTC"), color="black",
                linestyle="--", lw=1.2, label="IS/OOS split")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short spread", "Flat", "Long spread"])
    ax.set_title("BTC–ETH Pairs Strategy — Position Over Time")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pairs_positions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_cost_sensitivity(df_base: pd.DataFrame, cost_range_bps: list):
    """Sharpe and net PnL across a sweep of transaction cost assumptions."""
    results = []
    for c in cost_range_bps:
        df_c = apply_transaction_costs(df_base, cost_bps=c)
        net_pnl = df_c["net_pnl"].sum()
        ret     = df_c["net_pnl"] / CAPITAL
        std_r   = ret.std(ddof=1)
        mean_r  = ret.mean()
        sharpe  = (mean_r / std_r * np.sqrt(BARS_PER_YEAR)) if std_r > 0 else np.nan
        results.append({"cost_bps": c, "net_pnl": net_pnl, "sharpe": sharpe})

    res_df = pd.DataFrame(results)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(res_df["cost_bps"], res_df["net_pnl"], marker="o", color="darkorange")
    ax1.axhline(0, color="black", lw=0.5, linestyle="--")
    ax1.set_xlabel("Transaction cost (bps)")
    ax1.set_ylabel("Net PnL ($)")
    ax1.set_title("Net PnL vs Transaction Cost")
    ax2.plot(res_df["cost_bps"], res_df["sharpe"], marker="o", color="steelblue")
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_xlabel("Transaction cost (bps)")
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_title("Sharpe vs Transaction Cost")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pairs_cost_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return res_df

# ============================================================
# Step 15 — Save Outputs
# ============================================================

def save_outputs(df: pd.DataFrame, trades: pd.DataFrame,
                 metrics_list: list, cost_df: pd.DataFrame):
    # Backtest bar-level results
    cols_out = ["spread", "spread_mean", "spread_std", "zscore",
                "position", "gross_pnl", "cost", "net_pnl"]
    path_bt = os.path.join(TABLES_DIR, "pairs_backtest.csv")
    df[cols_out].to_csv(path_bt)
    print(f"  Saved: {path_bt}")

    # Trade log
    path_tl = os.path.join(TABLES_DIR, "pairs_trade_log.csv")
    trades.to_csv(path_tl, index=False)
    print(f"  Saved: {path_tl}")

    # Performance metrics
    path_pm = os.path.join(TABLES_DIR, "pairs_performance.csv")
    pd.DataFrame(metrics_list).to_csv(path_pm, index=False)
    print(f"  Saved: {path_pm}")

    # Cost sensitivity
    path_cs = os.path.join(TABLES_DIR, "pairs_cost_sensitivity.csv")
    cost_df.to_csv(path_cs, index=False)
    print(f"  Saved: {path_cs}")

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Pairs Trading Strategy — BTC/ETH")
    print("=" * 60)

    # --- Load ---
    df = load_data(DATA_PATH)

    # --- Compute spread and z-score ---
    df = compute_spread(df, beta=BETA)
    df = compute_zscore(df, window=ROLL_WINDOW)

    # --- Generate positions ---
    df = generate_positions(df)

    # --- PnL ---
    df = compute_leg_pnl(df, beta=BETA, capital=CAPITAL)
    df = apply_transaction_costs(df, beta=BETA, capital=CAPITAL, cost_bps=COST_BPS)

    # --- Trade log ---
    trades = extract_trades(df)

    # --- Split in-sample / out-of-sample ---
    is_mask  = df.index <= pd.Timestamp(IS_END, tz="UTC")
    oos_mask = df.index >  pd.Timestamp(IS_END, tz="UTC")
    df_is    = df[is_mask]
    df_oos   = df[oos_mask]
    trades_is  = trades[trades["entry_time"] <= pd.Timestamp(IS_END, tz="UTC")]
    trades_oos = trades[trades["entry_time"] >  pd.Timestamp(IS_END, tz="UTC")]

    # --- Metrics ---
    m_full = compute_metrics(df,      trades,      label="full")
    m_is   = compute_metrics(df_is,   trades_is,   label="in-sample")
    m_oos  = compute_metrics(df_oos,  trades_oos,  label="out-of-sample")

    print("\n--- Performance Summary ---")
    for m in [m_full, m_is, m_oos]:
        print(f"\n[{m['label']}]")
        for k, v in m.items():
            if k != "label":
                print(f"  {k:<20s}: {v}")

    # --- Cost sensitivity sweep ---
    print("\n--- Transaction Cost Sensitivity ---")
    cost_bps_range = [1, 5, 10, 15, 20]
    cost_df = plot_cost_sensitivity(df, cost_bps_range)
    print(cost_df.to_string(index=False))

    # --- Plots ---
    print("\n--- Generating plots ---")
    plot_spread_zscore(df, IS_END)
    plot_equity_curve(df, IS_END)
    plot_positions(df, IS_END)

    # --- Save ---
    print("\n--- Saving outputs ---")
    save_outputs(df, trades, [m_full, m_is, m_oos], cost_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
