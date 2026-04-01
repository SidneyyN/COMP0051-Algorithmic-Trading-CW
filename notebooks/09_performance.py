# ============================================================
# Section 1: Imports and Setups 
# ============================================================

import pandas as pd
import numpy as np
import importlib.util, sys, os

# Breakout config
N = 200                     # Donchian channel lookback (bars)
VOL_WINDOW = 50             # rolling vol window 
VOL_QUANTILE = 0.6          # vol filter: trade only above 60th percentile 
MAX_HOLD     = 40           # max holding period (bars)

# Pairs conifg 
BETA          = 0.6780          # OLS hedge ratio from EDA cointegration
ROLL_WINDOW   = 100             # bars for rolling z-score
ENTRY_Z       = 3.0             # enter when |z| > ENTRY_Z
EXIT_Z        = 0.0             # exit when z crosses 0 (full mean reversion)
STOP_Z        = 3.0             # no-progress stop after min-hold: exit if |z| still > STOP_Z
MIN_HOLD_PAIRS      = 24              # minimum holding period (bars, ~6 hours)
COOLDOWN      = 20              # bars to wait after exit before re-entering
MAX_HOLD_PAIRS     = 384             # max holding period (bars, ~4 days)
CAPITAL       = 100_000.0       # gross notional ($)
COST_BPS      = 5.0             # transaction cost in basis points (default)

# Volatility filter for pairs
VOL_WINDOW_PAIRS     = 100            # rolling window for spread volatility (bars)
VOL_REF_WINDOW = 200            # rolling window for reference volatility median (bars)
VOL_MULT       = 1.2            # allow entry only when spread_vol < VOL_MULT * vol_ref

IS_END = "2025-12-31"
BARS_PER_YEAR = 35_064      # 365.25 × 24 × 4

ASSETS = ["BTC", "ETH", "DOGE"]

DATA_PATH   = os.path.join("data", "cleaned", "cleaned_data.parquet")
BREAKOUT_PATH = os.path.join("notebooks", "04_breakout_strategy.py")
PAIRS_PATH = os.path.join("notebooks", "07_pairs_strategy_vol_filter.py")
TRANSACTION_COSTS_PATH = os.path.join("notebooks", "08_transaction_costs.py")
FIGURES_DIR = os.path.join("report", "figures")
TABLES_DIR  = os.path.join("report", "tables")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

for d in [FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

breakout = load_module("breakout", BREAKOUT_PATH)
pairs = load_module("pairs", PAIRS_PATH)
transaction_costs = load_module("transaction_costs", TRANSACTION_COSTS_PATH)

# ============================================================
# Section 2: Regenerate Positions  
# ============================================================

def recover_breakout_signal(path, N, vol_window, vol_quantile, max_hold):
    df = breakout.load_data(path)
    price_df = pd.read_parquet(path).sort_index()[["BTC_high", "BTC_low"]]
    df = df.join(price_df)

    df = breakout.compute_signals(df, N, vol_window, vol_quantile)
    df = breakout.run_backtest(df, max_hold)

    # normalise so that the df from both strat shares the same name 
    df["delta_pos"] = df["position"].diff().abs().fillna(0)

    return df 

def recover_pairs_signal(path, beta=BETA, window=ROLL_WINDOW, capital=CAPITAL):
    df = pairs.load_data(path)
    df = pairs.compute_spread(df, beta)
    df = pairs.compute_zscore(df, window)
    df = pairs.compute_vol_filter(df)
    pct_allowed = df["allow_trade"].mean() * 100
    print(f"  Vol filter: {pct_allowed:.1f}% of bars allow entry")

    # --- Generate positions ---
    df = pairs.generate_positions(df)
    df = pairs.compute_leg_pnl(df, beta, capital)

    # normalise so that the df from both strat shares the same name 
    df["delta_pos"] = df["position"].diff().abs().fillna(0)

    return df 

# ============================================================
# Section 3: Compute Costs 
# ============================================================

def compute_transaction_costs(path):
    df = transaction_costs.load_data(path)
    
    roll_spreads = {} # scalar per asset 
    cs_spreads = {} # series per asset 
    
    for asset in ASSETS:
        roll_spreads[asset] = transaction_costs.roll_spread(df[f"{asset}_close"])
        cs_spreads[asset] = transaction_costs.cs_spread(df[f"{asset}_high"], df[f"{asset}_low"])

    cs_df = pd.DataFrame(cs_spreads, index=df.index) # time series, one col per asset 
    return roll_spreads, cs_df

# ============================================================
# Section 4: Compute Metrics 
# ============================================================

# one small thing, our breakout strat does not have gross_pnl col, so we gotta include that
def compute_pnl(df, capital):
    # Next-bar return convention 
    # the position at bar t earns the return at bar t + 1
    # so at bar t, we have the position of t - 1 and then we use that to cal the return from that position
    df['position_lag'] = df['position'].shift(1).fillna(0)
    df['gross_return'] = df['position_lag'] * df['BTC_ret']

    # Detect when a trade happens 
    # df['trade'] = (df['position'] != df['position'].shift(1)).astype(int)
    df['trade'] = abs(df['position'] - df['position'].shift(1))

    # Net PnL
    df['gross_pnl'] = capital * df['gross_return']

    return df 

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
    n_trades = df['delta_pos'].sum()
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

def run_breakout_metrics(df, roll_spreads, cs_df, capital):
    metrics = {}

    df = compute_pnl(df, capital)

    # Fixed 
    cost = transaction_costs.compute_cost(df['position'], COST_BPS / 10_000, df['BTC_close'])
    df['cost'] = cost
    df['net_pnl'] = df['gross_pnl'] - cost
    fixed_metrics = compute_metrics(df, BARS_PER_YEAR)

    # Roll 
    cost = transaction_costs.compute_cost(df['position'], roll_spreads['BTC'], df['BTC_close'])
    df['cost'] = cost
    df['net_pnl'] = df['gross_pnl'] - cost
    roll_metrics = compute_metrics(df, BARS_PER_YEAR)

    # CS
    cost = transaction_costs.compute_cost(df['position'], cs_df['BTC'], df['BTC_close'])
    df['cost'] = cost
    df['net_pnl'] = df['gross_pnl'] - cost
    cs_metrics = compute_metrics(df, BARS_PER_YEAR)

    metrics["fixed"] = fixed_metrics
    metrics["roll"] = roll_metrics
    metrics["cs"] = cs_metrics
    return metrics 

def run_pairs_metrics(df, roll_spreads, cs_df, capital=CAPITAL):
    metrics = {}

    # Fixed 
    btc_cost = transaction_costs.compute_cost(df['position'], COST_BPS / 10_000, df['BTC_close'])
    eth_cost = transaction_costs.compute_cost(df['position'], COST_BPS / 10_000, df['ETH_close'])
    total_cost = btc_cost + eth_cost
    df['cost'] = total_cost
    df['net_pnl'] = df['gross_pnl'] - total_cost
    fixed_metrics = compute_metrics(df, BARS_PER_YEAR)

    # Roll
    btc_cost = transaction_costs.compute_cost(df['position'], roll_spreads['BTC'], df['BTC_close'])
    eth_cost = transaction_costs.compute_cost(df['position'], roll_spreads['ETH'], df['ETH_close'])
    total_cost = btc_cost + eth_cost
    df['cost'] = total_cost
    df['net_pnl'] = df['gross_pnl'] - total_cost
    roll_metrics = compute_metrics(df, BARS_PER_YEAR)

    # CS
    btc_cost = transaction_costs.compute_cost(df['position'], cs_df['BTC'], df['BTC_close'])
    eth_cost = transaction_costs.compute_cost(df['position'], cs_df['ETH'], df['ETH_close'])
    total_cost = btc_cost + eth_cost
    df['cost'] = total_cost
    df['net_pnl'] = df['gross_pnl'] - total_cost
    cs_metrics = compute_metrics(df, BARS_PER_YEAR)

    metrics["fixed"] = fixed_metrics
    metrics["roll"] = roll_metrics
    metrics["cs"] = cs_metrics
    return metrics 

# ============================================================
# Section 5: Compute Turnover
# ============================================================

def compute_turnover(df):
    total_turnover = np.sum(abs(df['delta_pos']))
    annualised_turnover_rate = total_turnover / len(df) * BARS_PER_YEAR
    return total_turnover, annualised_turnover_rate

# ============================================================
# Section 6: Cost Sensitivity 
# ============================================================

def cost_sensitivity(df, base_spread, position_col, price_col, capital, bars_per_year):
    multipliers = [0.5, 1.0, 1.5, 2.0]
    cost_sens = {}
    for mult in multipliers:
        cost = transaction_costs.compute_cost(position_col, mult * base_spread, price_col)
        df['cost'] = cost
        df['net_pnl'] = df['gross_pnl'] - cost

        ret = df['net_pnl'] / capital
        mean_r = ret.mean()
        std_r = ret.std(ddof=1)
        sharpe = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0 else np.nan
        cost_sens[mult] = sharpe    
    sens_df = pd.Series(cost_sens)    
    return sens_df

# ============================================================
# Helper function
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
# Main
# ============================================================

def main():
    roll_spreads, cs_df = compute_transaction_costs(DATA_PATH)
    
    df_breakout = recover_breakout_signal(DATA_PATH, N, VOL_WINDOW, VOL_QUANTILE, MAX_HOLD)

    df_pairs = recover_pairs_signal(DATA_PATH)

    is_mask  = df_breakout.index <= IS_END
    oos_mask = df_breakout.index >  IS_END

    # IS/OOS masks for pairs (index may differ from breakout)
    is_pairs_mask  = df_pairs.index <= IS_END
    oos_pairs_mask = df_pairs.index >  IS_END

    is_breakout_metrics  = run_breakout_metrics(df_breakout[is_mask].copy(),  roll_spreads, cs_df, CAPITAL)
    oos_breakout_metrics = run_breakout_metrics(df_breakout[oos_mask].copy(), roll_spreads, cs_df, CAPITAL)

    is_pairs_metrics  = run_pairs_metrics(df_pairs[is_pairs_mask].copy(),  roll_spreads, cs_df)
    oos_pairs_metrics = run_pairs_metrics(df_pairs[oos_pairs_mask].copy(), roll_spreads, cs_df)

    # Print metrics — one block per strategy × split × cost model
    for label, metrics in [
        ("BREAKOUT  |  IN-SAMPLE  (Sep–Dec 2025)",     is_breakout_metrics),
        ("BREAKOUT  |  OUT-OF-SAMPLE  (Jan–Feb 2026)", oos_breakout_metrics),
        ("PAIRS     |  IN-SAMPLE  (Sep–Dec 2025)",     is_pairs_metrics),
        ("PAIRS     |  OUT-OF-SAMPLE  (Jan–Feb 2026)", oos_pairs_metrics),
    ]:
        print(f"\n{'#'*52}")
        print(f"  {label}")
        print(f"{'#'*52}")
        for model in ["fixed", "roll", "cs"]:
            print_metrics(f"Cost model: {model.upper()}", metrics[model])

    # Turnover
    total_turnover_breakout, ann_turnover_breakout = compute_turnover(df_breakout)
    total_turnover_pairs,    ann_turnover_pairs    = compute_turnover(df_pairs)
    print(f"\n{'='*52}")
    print(f"  TURNOVER ANALYSIS")
    print(f"{'='*52}")
    print(f"  {'Strategy':<20} {'Total':>10}  {'Annualised':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Breakout':<20} {total_turnover_breakout:>10.1f}  {ann_turnover_breakout:>12.1f}")
    print(f"  {'Pairs':<20} {total_turnover_pairs:>10.1f}  {ann_turnover_pairs:>12.1f}")
    print(f"{'='*52}")

    # Ensure gross_pnl exists on full breakout df before sensitivity
    df_breakout = compute_pnl(df_breakout, CAPITAL)

    # Cost sensitivity — breakout uses BTC Roll spread directly
    breakout_sens = cost_sensitivity(
        df_breakout.copy(), roll_spreads["BTC"],
        df_breakout["position"], df_breakout["BTC_close"],
        CAPITAL, BARS_PER_YEAR
    )

    # Pairs sensitivity — scale the 1x Roll cost series rather than recomputing per leg
    pairs_base_cost = (
        transaction_costs.compute_cost(df_pairs["position"], roll_spreads["BTC"], df_pairs["BTC_close"]) +
        transaction_costs.compute_cost(df_pairs["position"], roll_spreads["ETH"], df_pairs["ETH_close"])
    )
    pairs_sens_rows = []
    for mult in [0.5, 1.0, 1.5, 2.0]:
        net_pnl = df_pairs["gross_pnl"] - mult * pairs_base_cost
        ret = net_pnl / CAPITAL
        sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(BARS_PER_YEAR) if ret.std(ddof=1) > 0 else np.nan
        pairs_sens_rows.append({"multiplier": mult, "sharpe": round(sharpe, 4)})
    pairs_sens = pd.DataFrame(pairs_sens_rows).set_index("multiplier")

    print(f"\n{'='*44}")
    print(f"  COST SENSITIVITY — BREAKOUT (Roll base)")
    print(f"{'='*44}")
    print(breakout_sens.to_string())
    print(f"\n{'='*44}")
    print(f"  COST SENSITIVITY — PAIRS (Roll base)")
    print(f"{'='*44}")
    print(pairs_sens.to_string())


if __name__ == "__main__":
    main()