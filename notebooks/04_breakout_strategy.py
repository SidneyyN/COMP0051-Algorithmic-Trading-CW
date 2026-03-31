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

N = 200                     # Donchian channel lookback (bars)
VOL_WINDOW = 50             # rolling vol window 
VOL_QUANTILE = 0.6          # vol filter: trade only above 60th percentile 
MAX_HOLD = 40               # max holding period (bars)
CAPITAL = 100_000.0         # gross notional ($)
COST_BPS = 5.0              # default cost in basis points 

IS_END = "2025-12-31"       # in-sample / out-of-sample boundary
BARS_PER_YEAR = 35_064      # 365.25 × 24 × 4

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
    df = df[["BTC_close", "BTC_ret"]].dropna()
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    return df

# ============================================================
# Step 2 — Compute Donchian bands 
# ============================================================

def compute_signals(df, N, vol_window, vol_quantile):
    # Donchian bands 
    # without the shift(1), we are just looking at today and past 99 day prices max
    # with the shift(1), we are looking at yesterday's upper band that does not include today's
    df['upper_band'] = df['BTC_close'].rolling(N).max().shift(1)
    df['lower_band'] = df['BTC_close'].rolling(N).min().shift(1)

    # Raw entry signals 
    df['long_entry'] = df['BTC_close'] < df['lower_band'] # buy dips below range
    df['short_entry'] = df['BTC_close'] > df['upper_band'] # sell spikes above range 

    # Volatility filter 
    df['rolling_vol'] = df['BTC_ret'].rolling(vol_window).std()
    is_mask = df.index <= IS_END
    vol_threshold = df.loc[is_mask, 'rolling_vol'].quantile(vol_quantile)
    df['vol_filter'] = df['rolling_vol'] > vol_threshold 

    # Applying the filter to entries 
    df['long_entry'] = df['long_entry'] & df['vol_filter']
    df['short_entry'] = df['short_entry'] & df['vol_filter']

    return df

# ============================================================
# Step 3 — Run backtest 
# ============================================================

def run_backtest(df, max_hold):
    n = len(df)
    position  = np.zeros(n, dtype=int)
    hold_bars = 0
    current_pos = 0

    for t in range(1, n):
        long_sig  = df['long_entry'].iloc[t]
        short_sig = df['short_entry'].iloc[t]

        # --- manage existing position ---
        if current_pos != 0:
            hold_bars += 1
            # exit conditions: (1) opposite signal OR (2) time stop
            # YOUR CODE HERE
            if (current_pos == 1 and short_sig) or (current_pos == -1 and long_sig):
                current_pos = 0
                hold_bars = 0
            
            elif hold_bars >= max_hold:
                current_pos = 0
                hold_bars = 0

        # --- enter if flat ---
        elif current_pos == 0:
            # YOUR CODE HERE
            if long_sig:
                current_pos = 1
                hold_bars = 0
            elif short_sig:
                current_pos = -1 
                hold_bars = 0

        position[t] = current_pos

    df['position'] = position
    return df

# ============================================================
# Step 4 — Compute returns and pnl
# ============================================================

def compute_pnl(df, capital, cost_bps):
    # Next-bar return convention 
    # the position at bar t earns the return at bar t + 1
    # so at bar t, we have the position of t - 1 and then we use that to cal the return from that position
    df['position_lag'] = df['position'].shift(1).fillna(0)
    df['gross_return'] = df['position_lag'] * df['BTC_ret']

    # Detect when a trade happens 
    # df['trade'] = (df['position'] != df['position'].shift(1)).astype(int)
    df['trade'] = abs(df['position'] - df['position'].shift(1))

    # Transaction cost per bar 
    cost_per_trade = capital * cost_bps * 0.0001
    df['cost'] = df['trade'] * cost_per_trade

    # Net PnL
    df['gross_pnl'] = capital * df['gross_return']
    df['net_pnl'] = df['gross_pnl'] - df['cost']

    return df

# ============================================================
# Step 5 — Performance metrics 
# ============================================================

def compute_metrics(df, bars_per_year):
    ret = df['net_pnl'] / CAPITAL
    total_net = df['net_pnl'].sum()
    total_gross = df['gross_pnl'].sum()
    total_cost = df['cost'].sum()

    # Sharpe ratio
    mean_r = ret.mean()
    std_r = ret.std(ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0 else np.nan

    # Sortino ratio 
    neg_r = ret[ret < 0]
    sortino_denom = neg_r.std(ddof=1)
    sortino = (mean_r / sortino_denom * np.sqrt(bars_per_year)) if sortino_denom > 0 else np.nan

    # Cum PnL and max drawdown 
    cum_pnl = df['net_pnl'].cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown = cum_pnl - rolling_max
    max_dd = drawdown.min()
    calmar = (total_net / abs(max_dd)) if abs(max_dd) > 0 else np.nan

    # Trade stats: 
    n_trades = df['trade'].sum()
    # win_rate = (df['net_pnl'][df['trade'] == 1] > 0).mean()

    # assign a trade ID: increment every time a new trade opens (position goes from 0 to non-zero)
    trade_entry = (df['position'] != 0) & (df['position'].shift(1) == 0)
    df['trade_id'] = trade_entry.cumsum()

    # zero out bars where we're flat (trade_id 0 = no trade)
    df.loc[df['position'] == 0, 'trade_id'] = 0

    # sum net PnL per trade, then check which are positive
    trade_pnl = df[df['trade_id'] > 0].groupby('trade_id')['net_pnl'].sum()
    win_rate = (trade_pnl > 0).mean()

    metrics = {
        "total_gross_$":  round(total_gross, 2),
        "total_net_$":    round(total_net, 2),
        "total_cost_$":   round(total_cost, 2),
        "sharpe":         round(sharpe, 4),
        "sortino":        round(sortino, 4),
        "max_drawdown_$": round(max_dd, 2),
        "calmar":         round(calmar, 4),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
    }

    return metrics 

# ============================================================
# Print helpers
# ============================================================

def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'='*44}")
    print(f"  {label}")
    print(f"{'='*44}")
    rows = [
        ("Total gross PnL",  f"${metrics['total_gross_$']:>12,.2f}"),
        ("Total net PnL",    f"${metrics['total_net_$']:>12,.2f}"),
        ("Total costs",      f"${metrics['total_cost_$']:>12,.2f}"),
        ("Sharpe ratio",     f"{metrics['sharpe']:>13.4f}"),
        ("Sortino ratio",    f"{metrics['sortino']:>13.4f}"),
        ("Max drawdown",     f"${metrics['max_drawdown_$']:>12,.2f}"),
        ("Calmar ratio",     f"{metrics['calmar']:>13.4f}"),
        ("No. of trades",    f"{metrics['n_trades']:>13}"),
        ("Win rate",         f"{metrics['win_rate']:>13.2%}" if not np.isnan(metrics['win_rate']) else f"{'N/A':>13}"),
    ]
    for name, value in rows:
        print(f"  {name:<20} {value}")
    print(f"{'='*44}")

# ============================================================
# Plot results
# ============================================================

def plot_results(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    # plot: BTC_close, upper_band, lower_band
    ax.plot(df.index, df['BTC_close'], color='black', linewidth=0.8, label='BTC close')
    ax.plot(df.index, df['upper_band'], color='blue', linewidth=0.8, linestyle='--', label='Upper band')
    ax.plot(df.index, df['lower_band'], color='orange', linewidth=0.8, linestyle='--', label='Lower band')
        
    longs = df[df["long_entry"]]
    shorts = df[df["short_entry"]]
    ax.scatter(longs.index, longs['BTC_close'], marker='^', color='green', s=20, zorder=5)
    ax.scatter(shorts.index, shorts['BTC_close'], marker='v', color='red', s=20, zorder=5)
    ax.axvline(pd.Timestamp(IS_END, tz='UTC'), color='black', linestyle=':', label='IS/OOS split')

    ax.set_title('BTC Breakout - Price, Bands, and Signals')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'breakout_price_bands.png'), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(df.index, df['gross_pnl'].cumsum(), label='Gross PnL', color='steelblue')
    ax2.plot(df.index, df['net_pnl'].cumsum(),   label='Net PnL',   color='crimson')
    ax2.axvline(pd.Timestamp(IS_END, tz='UTC'), color='black', linestyle=':', label='IS/OOS split')
    ax2.set_title('Breakout Strategy — Cumulative PnL')
    ax2.set_ylabel('Cumulative PnL ($)')
    ax2.legend()
    ax2.axhline(0, color='gray', linewidth=0.5)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, 'breakout_equity_curve.png'), dpi=150)
    plt.close(fig2)


# ============================================================
# Main
# ============================================================

def main():
    df = load_data(DATA_PATH)

    df = compute_signals(df, N, VOL_WINDOW, VOL_QUANTILE)

    df = run_backtest(df, MAX_HOLD)

    df = compute_pnl(df, CAPITAL, COST_BPS)

    is_mask  = df.index <= IS_END
    oos_mask = df.index >  IS_END

    is_metrics  = compute_metrics(df[is_mask],  BARS_PER_YEAR)
    oos_metrics = compute_metrics(df[oos_mask], BARS_PER_YEAR)

    print_metrics("IN-SAMPLE  (Sep–Dec 2025)", is_metrics)
    print_metrics("OUT-OF-SAMPLE  (Jan–Feb 2026)", oos_metrics)

    plot_results(df)


if __name__ == "__main__":
    main()