import os
import sys
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

IS_END        = "2025-12-31"
BARS_PER_YEAR = 35_064
CAPITAL       = 100_000.0
COST_BPS      = 5.0
ETA           = 3          # SHA elimination ratio
N_CONFIGS     = 27         # 3^3 initial random configs
MIN_TRADES    = 5          # reject configs with fewer trades
RANDOM_SEED   = 42

DATA_PATH               = os.path.join("data", "cleaned", "cleaned_data.parquet")
BREAKOUT_PATH           = os.path.join("notebooks", "04_breakout_strategy.py")
PAIRS_PATH              = os.path.join("notebooks", "07_pairs_strategy_vol_filter.py")
TRANSACTION_COSTS_PATH  = os.path.join("notebooks", "08_transaction_costs.py")
FIGURES_DIR             = os.path.join("report", "figures")
TABLES_DIR              = os.path.join("report", "tables")

# Baseline configs (from 09_performance.py defaults)
BASELINE_BREAKOUT_CFG = {
    "N": 200, "vol_window": 50, "vol_quantile": 0.6, "max_hold": 40,
}
BASELINE_PAIRS_CFG = {
    "coint_window": None, "zscore_window": 100,
    "entry_z": 3.0, "exit_z": 0.0, "stop_z": 3.0, "vol_mult": 1.2,
}

# ============================================================
# Search spaces
# ============================================================

BREAKOUT_SPACE = {
    "N":            [100, 150, 200, 300, 400],
    "vol_window":   [30, 50, 75, 100],
    "vol_quantile": [0.4, 0.5, 0.6, 0.7, 0.8],
    "max_hold":     [20, 40, 60, 100],
}

PAIRS_SPACE = {
    "coint_window":  [500, 1000, 2000, None],  # None = static OLS on budget slice
    "zscore_window": [50, 75, 100, 150, 200],
    "entry_z":       [2.0, 2.5, 3.0, 3.5],
    "exit_z":        [0.0, 0.5, 1.0],
    "stop_z":        [2.5, 3.0, 4.0, 5.0],
    "vol_mult":      [1.0, 1.2, 1.5, 2.0],
}

# ============================================================
# Module loading (same pattern as 09_performance.py)
# ============================================================

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# Transaction cost spread estimates
# ============================================================

def compute_spread_estimates(tc_mod) -> tuple[dict, pd.DataFrame]:
    """
    Compute Roll (scalar per asset) and Corwin-Schultz (time series per asset)
    spread estimates from the full dataset, matching 09_performance.py.

    Returns:
        roll_spreads : dict  {"BTC": float, "ETH": float}
        cs_df        : pd.DataFrame  columns ["BTC", "ETH"], index = bar timestamps
    """
    df_tc = tc_mod.load_data(DATA_PATH)
    roll_spreads = {
        "BTC": tc_mod.roll_spread(df_tc["BTC_close"]),
        "ETH": tc_mod.roll_spread(df_tc["ETH_close"]),
    }
    cs_df = pd.DataFrame({
        "BTC": tc_mod.cs_spread(df_tc["BTC_high"], df_tc["BTC_low"]),
        "ETH": tc_mod.cs_spread(df_tc["ETH_high"], df_tc["ETH_low"]),
    })
    return roll_spreads, cs_df


# ============================================================
# Rolling OLS beta
# ============================================================

def compute_rolling_beta(df: pd.DataFrame, coint_window: int) -> pd.Series:
    """
    Rolling OLS: log(BTC) ~ alpha + beta * log(ETH)
    Returns a Series of time-varying beta values.
    NaN for first (coint_window - 1) bars.
    """
    log_btc = np.log(df["BTC_close"].values)
    log_eth = np.log(df["ETH_close"].values)
    n = len(df)
    betas = np.full(n, np.nan)

    for i in range(coint_window - 1, n):
        y = log_btc[i - coint_window + 1 : i + 1]
        X = np.column_stack([
            np.ones(coint_window),
            log_eth[i - coint_window + 1 : i + 1]
        ])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        betas[i] = coeffs[1]

    return pd.Series(betas, index=df.index)


def compute_static_beta(df: pd.DataFrame) -> float:
    """Single OLS beta on the entire passed-in DataFrame."""
    log_btc = np.log(df["BTC_close"].values)
    log_eth = np.log(df["ETH_close"].values)
    X = np.column_stack([np.ones(len(df)), log_eth])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_btc, rcond=None)
    return float(coeffs[1])

# ============================================================
# Breakout evaluator
# ============================================================

def evaluate_breakout(params: dict, df_full: pd.DataFrame,
                      budget_end_idx: int, breakout_mod) -> float:
    N            = params["N"]
    vol_window   = params["vol_window"]
    vol_quantile = params["vol_quantile"]
    max_hold     = params["max_hold"]

    df = df_full.iloc[:budget_end_idx].copy()

    try:
        df = breakout_mod.compute_signals(df, N, vol_window, vol_quantile)
        df = breakout_mod.run_backtest(df, max_hold)
        df = breakout_mod.compute_pnl(df, CAPITAL, COST_BPS)

        # Count completed trades
        n_trades = int(df["trade"].sum())
        if n_trades < MIN_TRADES:
            return -np.inf

        metrics = breakout_mod.compute_metrics(df, BARS_PER_YEAR)
        sharpe  = metrics["sharpe"]
        return float(sharpe) if not np.isnan(sharpe) else -np.inf

    except Exception:
        return -np.inf

# ============================================================
# Pairs evaluator
# ============================================================

def evaluate_pairs(params: dict, df_full: pd.DataFrame,
                   budget_end_idx: int, pairs_mod) -> float:
    coint_window  = params["coint_window"]   # int or None
    zscore_window = params["zscore_window"]
    entry_z       = params["entry_z"]
    exit_z        = params["exit_z"]
    stop_z        = params["stop_z"]
    vol_mult      = params["vol_mult"]

    df = df_full.iloc[:budget_end_idx].copy()

    try:
        # Step 1: compute beta and spread
        if coint_window is None:
            beta_val = compute_static_beta(df)
            df = pairs_mod.compute_spread(df, beta_val)
        else:
            beta_series = compute_rolling_beta(df, coint_window)
            df = df.copy()
            df["log_btc"] = np.log(df["BTC_close"])
            df["log_eth"] = np.log(df["ETH_close"])
            df["spread"]  = df["log_btc"] - beta_series * df["log_eth"]
            beta_val = float(beta_series.dropna().mean())

        # Step 2: z-score
        df = pairs_mod.compute_zscore(df, zscore_window)

        # Step 3: vol filter (only vol_mult is searched; windows fixed at module defaults)
        df = pairs_mod.compute_vol_filter(df, vol_mult=vol_mult)

        # Step 4: positions (min_hold, max_hold, cooldown are fixed)
        df = pairs_mod.generate_positions(
            df,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            min_hold=24,
            max_hold=384,
            cooldown=20,
        )

        # Step 5: PnL
        df = pairs_mod.compute_leg_pnl(df, beta_val, CAPITAL)
        df = pairs_mod.apply_transaction_costs(df, beta_val, CAPITAL, COST_BPS)

        # Count position changes as a proxy for n_trades
        n_trades = int((df["position"].diff().abs() > 0).sum())
        if n_trades < MIN_TRADES:
            return -np.inf

        # Inline Sharpe (avoids expensive extract_trades during search)
        ret   = df["net_pnl"] / CAPITAL
        mean_r = ret.mean()
        std_r  = ret.std(ddof=1)
        if std_r <= 0:
            return -np.inf
        sharpe = mean_r / std_r * np.sqrt(BARS_PER_YEAR)
        return float(sharpe) if not np.isnan(sharpe) else -np.inf

    except Exception:
        return -np.inf

# ============================================================
# Random config sampling
# ============================================================

def sample_configs(space: dict, n: int, rng) -> list[dict]:
    configs = []
    for _ in range(n):
        cfg = {
            param: rng.choice(values)
            for param, values in space.items()
        }
        configs.append(cfg)
    return configs

# ============================================================
# Successive Halving
# ============================================================

def run_sha(evaluate_fn, df_full: pd.DataFrame, space: dict,
            n_configs: int, eta: int, is_bars: int,
            label: str = "") -> tuple[dict, float, pd.DataFrame]:
    """
    Successive Halving Algorithm.

    Rounds (eta=3, n_configs=27):
      Round 1: 27 configs on 1/3 IS → keep top 9
      Round 2:  9 configs on 2/3 IS → keep top 3
      Round 3:  3 configs on full IS → pick best

    Returns (winner_cfg, winner_sharpe, results_df).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    configs = sample_configs(space, n_configs, rng)

    budget_fractions = [1 / eta, 2 / eta, 1.0]
    all_round_rows   = []

    for round_idx, budget_frac in enumerate(budget_fractions):
        budget_end_idx = int(is_bars * budget_frac)
        round_num      = round_idx + 1

        print(f"  [{label}] Round {round_num}/3 — {len(configs)} configs on "
              f"{budget_end_idx:,} IS bars ({budget_frac:.0%} IS) ...")

        sharpes = [evaluate_fn(cfg, df_full, budget_end_idx) for cfg in configs]

        for cfg, sh in zip(configs, sharpes):
            row = {k: (str(v) if k == "coint_window" else v)
                   for k, v in cfg.items()}
            row["round"]       = round_num
            row["budget_frac"] = round(budget_frac, 4)
            row["sharpe"]      = round(sh, 6)
            all_round_rows.append(row)

        # Filter to top configs for next round (skip on final round)
        if round_idx < len(budget_fractions) - 1:
            top_k  = max(1, len(configs) // eta)
            ranked = sorted(zip(sharpes, configs), key=lambda x: x[0], reverse=True)
            top_sharpes = [s for s, _ in ranked[:top_k]]
            configs     = [c for _, c in ranked[:top_k]]

            if all(s == -np.inf for s in top_sharpes):
                print(f"  WARNING: all configs scored -inf in round {round_num}. "
                      f"Retaining all configs for next round.")
                configs = [c for _, c in ranked]

    results_df   = pd.DataFrame(all_round_rows)
    final_round  = results_df[results_df["round"] == len(budget_fractions)]
    winner_idx   = final_round["sharpe"].idxmax()

    param_cols = list(space.keys())
    winner_cfg = {}
    for col in param_cols:
        val = final_round.loc[winner_idx, col]
        # restore None from string for coint_window
        if col == "coint_window" and str(val) == "None":
            winner_cfg[col] = None
        elif col == "coint_window":
            winner_cfg[col] = int(val)
        else:
            winner_cfg[col] = val
    winner_sharpe = float(final_round.loc[winner_idx, "sharpe"])

    return winner_cfg, winner_sharpe, results_df

# ============================================================
# OOS validation for the winning config
# ============================================================

def validate_winner_breakout(winner_cfg: dict, df_all: pd.DataFrame,
                              breakout_mod, tc_mod,
                              roll_spreads: dict,
                              cs_df: pd.DataFrame) -> dict:
    """
    Run winner on full IS+OOS dataset using all three cost models.
    Returns {"fixed": (is_m, oos_m), "roll": ..., "cs": ...}.
    """
    is_end_ts = pd.Timestamp(IS_END, tz="UTC")
    is_mask   = df_all.index <= is_end_ts
    oos_mask  = df_all.index >  is_end_ts

    df = df_all.copy()
    df = breakout_mod.compute_signals(
        df, winner_cfg["N"], winner_cfg["vol_window"], winner_cfg["vol_quantile"]
    )
    df = breakout_mod.run_backtest(df, winner_cfg["max_hold"])
    # cost_bps=0 gives us gross_pnl and trade column with zero cost baked in
    df = breakout_mod.compute_pnl(df, CAPITAL, 0)

    cost_models = {
        "fixed": COST_BPS / 10_000,
        "roll":  roll_spreads["BTC"],
        "cs":    cs_df["BTC"].reindex(df.index),
    }

    results = {}
    for name, spread in cost_models.items():
        df["cost"]    = tc_mod.compute_cost(df["position"], spread, df["BTC_close"])
        df["net_pnl"] = df["gross_pnl"] - df["cost"]
        is_m   = breakout_mod.compute_metrics(df[is_mask],  BARS_PER_YEAR)
        oos_m  = breakout_mod.compute_metrics(df[oos_mask], BARS_PER_YEAR)
        full_m = breakout_mod.compute_metrics(df,           BARS_PER_YEAR)
        results[name] = (is_m, oos_m, full_m)

    return results


def validate_winner_pairs(winner_cfg: dict, df_all: pd.DataFrame,
                          pairs_mod, tc_mod,
                          roll_spreads: dict,
                          cs_df: pd.DataFrame) -> dict:
    """
    Run winner on full IS+OOS dataset using all three cost models.
    Returns {"fixed": (is_m, oos_m), "roll": ..., "cs": ...}.
    """
    is_end_ts = pd.Timestamp(IS_END, tz="UTC")
    is_mask   = df_all.index <= is_end_ts
    oos_mask  = df_all.index >  is_end_ts
    is_bars   = int(is_mask.sum())

    coint_window  = winner_cfg["coint_window"]
    zscore_window = winner_cfg["zscore_window"]

    df = df_all.copy()

    if coint_window is None:
        log_btc_is = np.log(df.loc[is_mask, "BTC_close"].values)
        log_eth_is = np.log(df.loc[is_mask, "ETH_close"].values)
        X = np.column_stack([np.ones(is_bars), log_eth_is])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_btc_is, rcond=None)
        beta_val = float(coeffs[1])
        df = pairs_mod.compute_spread(df, beta_val)
    else:
        beta_series = compute_rolling_beta(df, coint_window)
        df["log_btc"] = np.log(df["BTC_close"])
        df["log_eth"] = np.log(df["ETH_close"])
        df["spread"]  = df["log_btc"] - beta_series * df["log_eth"]
        beta_val = float(beta_series.iloc[:is_bars].dropna().mean())

    df = pairs_mod.compute_zscore(df, zscore_window)
    df = pairs_mod.compute_vol_filter(df, vol_mult=winner_cfg["vol_mult"])
    df = pairs_mod.generate_positions(
        df,
        entry_z=winner_cfg["entry_z"],
        exit_z=winner_cfg["exit_z"],
        stop_z=winner_cfg["stop_z"],
        min_hold=24,
        max_hold=384,
        cooldown=20,
    )
    # gross_pnl only — no costs yet
    df = pairs_mod.compute_leg_pnl(df, beta_val, CAPITAL)

    cost_models = {
        "fixed": (COST_BPS / 10_000,    COST_BPS / 10_000),
        "roll":  (roll_spreads["BTC"],   roll_spreads["ETH"]),
        "cs":    (cs_df["BTC"].reindex(df.index), cs_df["ETH"].reindex(df.index)),
    }

    results = {}
    for name, (btc_spread, eth_spread) in cost_models.items():
        btc_cost       = tc_mod.compute_cost(df["position"], btc_spread, df["BTC_close"])
        eth_cost       = tc_mod.compute_cost(df["position"], eth_spread, df["ETH_close"])
        df["cost"]     = btc_cost + eth_cost
        df["net_pnl"]  = df["gross_pnl"] - df["cost"]

        trades     = pairs_mod.extract_trades(df)
        is_trades  = trades[trades["entry_time"] <= is_end_ts]
        oos_trades = trades[trades["entry_time"] >  is_end_ts]

        is_m   = pairs_mod.compute_metrics(df[is_mask],  is_trades,  label="in-sample")
        oos_m  = pairs_mod.compute_metrics(df[oos_mask], oos_trades, label="out-of-sample")
        full_m = pairs_mod.compute_metrics(df,           trades,     label="full")
        results[name] = (is_m, oos_m, full_m)

    return results

# ============================================================
# Build winner DataFrames with all cost-model net_pnl columns
# ============================================================

def build_winner_df_breakout(winner_cfg: dict, df_all: pd.DataFrame,
                              breakout_mod, tc_mod,
                              roll_spreads: dict, cs_df: pd.DataFrame) -> pd.DataFrame:
    """Return full df with gross_pnl and net_pnl_{fixed,roll,cs} for the winner config."""
    df = df_all.copy()
    df = breakout_mod.compute_signals(
        df, winner_cfg["N"], winner_cfg["vol_window"], winner_cfg["vol_quantile"]
    )
    df = breakout_mod.run_backtest(df, winner_cfg["max_hold"])
    df = breakout_mod.compute_pnl(df, CAPITAL, 0)  # cost_bps=0 → gross_pnl only

    for model, spread in [
        ("fixed", COST_BPS / 10_000),
        ("roll",  roll_spreads["BTC"]),
        ("cs",    cs_df["BTC"].reindex(df.index)),
    ]:
        cost = tc_mod.compute_cost(df["position"], spread, df["BTC_close"])
        df[f"net_pnl_{model}"] = df["gross_pnl"] - cost

    return df


def build_winner_df_pairs(winner_cfg: dict, df_all: pd.DataFrame,
                          pairs_mod, tc_mod,
                          roll_spreads: dict, cs_df: pd.DataFrame) -> pd.DataFrame:
    """Return full df with gross_pnl and net_pnl_{fixed,roll,cs} for the winner config."""
    is_end_ts = pd.Timestamp(IS_END, tz="UTC")
    is_mask   = df_all.index <= is_end_ts
    is_bars   = int(is_mask.sum())

    coint_window = winner_cfg["coint_window"]
    df = df_all.copy()

    if coint_window is None:
        log_btc_is = np.log(df.loc[is_mask, "BTC_close"].values)
        log_eth_is = np.log(df.loc[is_mask, "ETH_close"].values)
        X = np.column_stack([np.ones(is_bars), log_eth_is])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_btc_is, rcond=None)
        beta_val = float(coeffs[1])
        df = pairs_mod.compute_spread(df, beta_val)
    else:
        beta_series = compute_rolling_beta(df, coint_window)
        df["log_btc"] = np.log(df["BTC_close"])
        df["log_eth"] = np.log(df["ETH_close"])
        df["spread"]  = df["log_btc"] - beta_series * df["log_eth"]
        beta_val = float(beta_series.iloc[:is_bars].dropna().mean())

    df = pairs_mod.compute_zscore(df, winner_cfg["zscore_window"])
    df = pairs_mod.compute_vol_filter(df, vol_mult=winner_cfg["vol_mult"])
    df = pairs_mod.generate_positions(
        df,
        entry_z=winner_cfg["entry_z"],
        exit_z=winner_cfg["exit_z"],
        stop_z=winner_cfg["stop_z"],
        min_hold=24,
        max_hold=384,
        cooldown=20,
    )
    df = pairs_mod.compute_leg_pnl(df, beta_val, CAPITAL)

    for model, s_btc, s_eth in [
        ("fixed", COST_BPS / 10_000,                  COST_BPS / 10_000),
        ("roll",  roll_spreads["BTC"],                 roll_spreads["ETH"]),
        ("cs",    cs_df["BTC"].reindex(df.index),      cs_df["ETH"].reindex(df.index)),
    ]:
        cost = (tc_mod.compute_cost(df["position"], s_btc, df["BTC_close"]) +
                tc_mod.compute_cost(df["position"], s_eth, df["ETH_close"]))
        df[f"net_pnl_{model}"] = df["gross_pnl"] - cost

    return df


# ============================================================
# Plot helpers (matching 10_visualisations.py style)
# ============================================================

COLORS = {
    "gross": "#2196F3",
    "fixed": "#4CAF50",
    "roll":  "#FF9800",
    "cs":    "#F44336",
}


def _ts(s, tz):
    ts = pd.Timestamp(s)
    return ts.tz_localize(tz) if tz is not None else ts


def _shade_is_oos(ax, df):
    tz       = df.index.tz
    is_end   = _ts(IS_END, tz)
    ax.axvspan(df.index[0], is_end,       alpha=0.06, color="#E3F2FD", zorder=0)
    ax.axvspan(is_end,       df.index[-1], alpha=0.06, color="#FFF3E0", zorder=0)
    ax.axvline(is_end, color="grey", linewidth=0.8, linestyle="--", zorder=1)


def _label_is_oos(ax, df):
    tz      = df.index.tz
    is_end  = _ts(IS_END, tz)
    is_mid  = df.index[0] + (is_end - df.index[0]) / 2
    oos_mid = is_end + (df.index[-1] - is_end) / 2
    y_top   = ax.get_ylim()[1]
    ax.text(is_mid,  y_top * 0.96, "In-sample",     ha="center", va="top", fontsize=7, color="grey")
    ax.text(oos_mid, y_top * 0.96, "Out-of-sample", ha="center", va="top", fontsize=7, color="grey")


# ============================================================
# Plots
# ============================================================

def plot_winner_cumulative_pnl(df_b: pd.DataFrame, df_p: pd.DataFrame) -> None:
    """
    Two-panel cumulative PnL plot for the SHA winner configs.
    Matches the style of 10_visualisations.py vis_cumulative_pnl.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, df, title in [
        (axes[0], df_b, "Breakout (SHA Winner)"),
        (axes[1], df_p, "Pairs Trading (SHA Winner)"),
    ]:
        _shade_is_oos(ax, df)

        ax.plot(df.index, df["gross_pnl"].cumsum(),        color=COLORS["gross"],
                linewidth=1.2, label="Gross",       zorder=3)
        ax.plot(df.index, df["net_pnl_fixed"].cumsum(),    color=COLORS["fixed"],
                linewidth=1.0, label="Net (Fixed)",  zorder=3)
        ax.plot(df.index, df["net_pnl_roll"].cumsum(),     color=COLORS["roll"],
                linewidth=1.0, label="Net (Roll)",   zorder=3)
        ax.plot(df.index, df["net_pnl_cs"].cumsum(),       color=COLORS["cs"],
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
        _label_is_oos(ax, df)

    fig.suptitle("Cumulative PnL — SHA Winner Configs (Gross vs Net, Three Cost Models)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "hparam_winner_cumulative_pnl.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_sha_sharpe_distribution(breakout_results: pd.DataFrame,
                                  pairs_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    round_labels = ["Round 1\n(1/3 IS)", "Round 2\n(2/3 IS)", "Round 3\n(Full IS)"]

    for ax, results, title in zip(
        axes,
        [breakout_results, pairs_results],
        ["Breakout: Sharpe per SHA Round", "Pairs: Sharpe per SHA Round"],
    ):
        data = []
        for r in [1, 2, 3]:
            vals = results.loc[results["round"] == r, "sharpe"].replace(-np.inf, np.nan).dropna()
            data.append(vals.values)

        bp = ax.boxplot(data, tick_labels=round_labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        colours = ["#90caf9", "#42a5f5", "#1565c0"]
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.axhline(0, color="red", linestyle="--", lw=0.9, label="Sharpe = 0")
        ax.set_ylabel("IS Sharpe (net, fixed 5 bps)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Successive Halving: Sharpe Distribution per Round", fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "hparam_sha_sharpe_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_is_oos_comparison(b_results: dict, p_results: dict) -> None:
    """
    Grouped bar chart: IS vs OOS Sharpe for each cost model × strategy.
    b_results / p_results: {"fixed": (is_m, oos_m), "roll": ..., "cs": ...}
    """
    cost_labels   = ["Fixed\n(5 bps)", "Roll", "Corwin-\nSchultz"]
    cost_keys     = ["fixed", "roll", "cs"]
    is_colours    = ["#1a6faf", "#2ca02c", "#d62728"]
    oos_colours   = ["#6baed6", "#74c476", "#fc8d59"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, res, title in zip(
        axes,
        [b_results, p_results],
        ["Breakout winner", "Pairs winner"],
    ):
        x     = np.arange(len(cost_keys))
        width = 0.35
        is_sharpes  = [res[k][0]["sharpe"] for k in cost_keys]
        oos_sharpes = [res[k][1]["sharpe"] for k in cost_keys]

        bars_is  = ax.bar(x - width / 2, is_sharpes,  width, label="IS (Sep–Dec 2025)",
                          color=is_colours,  alpha=0.85)
        bars_oos = ax.bar(x + width / 2, oos_sharpes, width, label="OOS (Jan–Feb 2026)",
                          color=oos_colours, alpha=0.85)

        for bar, val in zip(bars_is,  is_sharpes):
            ypos = val + 0.03 if val >= 0 else val - 0.12
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_oos, oos_sharpes):
            ypos = val + 0.03 if val >= 0 else val - 0.12
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(cost_labels)
        ax.set_ylabel("Annualised Sharpe Ratio")
        ax.set_title(f"{title}: IS vs OOS Sharpe by Cost Model")
        ax.axhline(0, color="black", lw=0.6, linestyle="--")
        ax.legend(fontsize=8)

    plt.suptitle("Winner Config: IS vs OOS Sharpe across Cost Models (SHA Search)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "hparam_is_vs_oos_sharpe.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ============================================================
# Full evaluation: print, plot, save (baseline vs winner)
# ============================================================

_METRIC_LABELS = [
    ("sharpe",         "Sharpe ratio",    "{:>10.4f}",  "{:>+10.4f}"),
    ("sortino",        "Sortino ratio",   "{:>10.4f}",  "{:>+10.4f}"),
    ("calmar",         "Calmar ratio",    "{:>10.4f}",  "{:>+10.4f}"),
    ("total_net_$",    "Net PnL ($)",     "{:>10,.0f}", "{:>+10,.0f}"),
    ("total_gross_$",  "Gross PnL ($)",   "{:>10,.0f}", "{:>+10,.0f}"),
    ("total_cost_$",   "Total costs ($)", "{:>10,.0f}", "{:>+10,.0f}"),
    ("max_drawdown_$", "Max DD ($)",      "{:>10,.0f}", "{:>+10,.0f}"),
    ("n_trades",       "N trades",        "{:>10}",     "{:>+10}"),
    ("win_rate",       "Win rate",        "{:>10.2%}",  "{:>+10.2%}"),
]


def _fmt(val, fmt_str):
    if isinstance(val, float) and np.isnan(val):
        return "       N/A"
    try:
        return fmt_str.format(val)
    except (ValueError, TypeError):
        return f"{val:>10}"


def print_full_evaluation(strategy: str, base_val: dict, win_val: dict) -> None:
    """
    Print full metrics for baseline vs winner, matching 09_performance.py style.
    base_val / win_val: {"fixed": (is_m, oos_m), "roll": ..., "cs": ...}
    """
    cost_names = {"fixed": "Fixed 5 bps", "roll": "Roll", "cs": "Corwin-Schultz"}
    split_names = [
        ("IS (Sep–Dec 2025)",            0),
        ("OOS (Jan–Feb 2026)",           1),
        ("Full period (Sep 2025–Feb 2026)", 2),
    ]

    print(f"\n{'#'*70}")
    print(f"  {strategy.upper()}  —  Baseline vs SHA Winner")
    print(f"{'#'*70}")

    for split_label, split_idx in split_names:
        print(f"\n  ── {split_label} ──")
        for cm_key, cm_label in cost_names.items():
            base_m = base_val[cm_key][split_idx]
            win_m  = win_val[cm_key][split_idx]

            print(f"\n  Cost model: {cm_label}")
            print(f"  {'Metric':<20} {'Baseline':>12}  {'Winner':>12}  {'Delta':>10}")
            print(f"  {'-'*58}")

            for key, label, fmt_v, fmt_d in _METRIC_LABELS:
                bv = base_m.get(key, float("nan"))
                wv = win_m.get(key,  float("nan"))
                try:
                    delta = wv - bv
                except TypeError:
                    delta = float("nan")
                print(f"  {label:<20} {_fmt(bv, fmt_v)}  {_fmt(wv, fmt_v)}  {_fmt(delta, fmt_d)}")


def plot_sharpe_comparison(b_base_val: dict, b_win_val: dict,
                            p_base_val: dict, p_win_val: dict) -> None:
    """
    Grouped bar chart: Sharpe for baseline vs winner across IS/OOS × cost models.
    2 panels (breakout, pairs); x-axis = 6 conditions (3 IS + 3 OOS).
    """
    cost_keys   = ["fixed", "roll", "cs"]
    cost_labels = ["Fixed", "Roll", "CS"]
    conditions  = [f"IS\n{c}" for c in cost_labels] + [f"OOS\n{c}" for c in cost_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, base_val, win_val, title in [
        (axes[0], b_base_val, b_win_val, "Breakout Strategy"),
        (axes[1], p_base_val, p_win_val, "Pairs Trading Strategy"),
    ]:
        base_sharpes = (
            [base_val[cm][0]["sharpe"] for cm in cost_keys] +  # IS
            [base_val[cm][1]["sharpe"] for cm in cost_keys]    # OOS
        )
        win_sharpes  = (
            [win_val[cm][0]["sharpe"]  for cm in cost_keys] +
            [win_val[cm][1]["sharpe"]  for cm in cost_keys]
        )

        x     = np.arange(len(conditions))
        width = 0.35
        bars_b = ax.bar(x - width / 2, base_sharpes, width,
                        label="Baseline", color="#90caf9", edgecolor="white", alpha=0.9)
        bars_w = ax.bar(x + width / 2, win_sharpes,  width,
                        label="SHA Winner", color="#1565c0", edgecolor="white", alpha=0.9)

        for bar, val in zip(list(bars_b) + list(bars_w),
                            base_sharpes + win_sharpes):
            yoff = 0.04 if val >= 0 else -0.15
            ax.text(bar.get_x() + bar.get_width() / 2, val + yoff,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        # Vertical separator between IS and OOS
        ax.axvline(2.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(0.5 / 6, 0.97, "In-sample", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="grey")
        ax.text(4.5 / 6, 0.97, "Out-of-sample", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="grey")

        ax.axhline(0, color="black", lw=0.6, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=8)
        ax.set_ylabel("Annualised Sharpe Ratio", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Baseline vs SHA Winner: Sharpe Ratio across Cost Models & Periods",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "hparam_baseline_vs_winner_sharpe.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_comparison_csv(b_base_val: dict, b_win_val: dict,
                         p_base_val: dict, p_win_val: dict,
                         b_winner_cfg: dict, p_winner_cfg: dict) -> None:
    """Save a flat CSV with baseline vs winner metrics for all conditions."""
    rows = []
    cost_keys = ["fixed", "roll", "cs"]
    split_labels = ["is", "oos", "full"]

    for strategy, base_val, win_val, win_cfg in [
        ("breakout", b_base_val, b_win_val, b_winner_cfg),
        ("pairs",    p_base_val, p_win_val, p_winner_cfg),
    ]:
        for cm in cost_keys:
            for si, split in enumerate(split_labels):
                base_m = base_val[cm][si]
                win_m  = win_val[cm][si]
                row = {
                    "strategy":     strategy,
                    "cost_model":   cm,
                    "split":        split,
                    "winner_params": str(win_cfg),
                }
                for key, *_ in _METRIC_LABELS:
                    bv = base_m.get(key, float("nan"))
                    wv = win_m.get(key, float("nan"))
                    row[f"baseline_{key}"] = bv
                    row[f"winner_{key}"]   = wv
                    try:
                        row[f"delta_{key}"] = wv - bv
                    except TypeError:
                        row[f"delta_{key}"] = float("nan")
                rows.append(row)

    df_out = pd.DataFrame(rows)
    out = os.path.join(TABLES_DIR, "baseline_vs_winner_comparison.csv")
    df_out.to_csv(out, index=False)
    print(f"Saved {out}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR,  exist_ok=True)

    print("=" * 60)
    print("Hyperparameter Search — Random Search + Successive Halving")
    print(f"  eta={ETA}, n_configs={N_CONFIGS}, seed={RANDOM_SEED}")
    print("=" * 60)

    # Load strategy modules
    breakout_mod = load_module("breakout",          BREAKOUT_PATH)
    pairs_mod    = load_module("pairs",             PAIRS_PATH)
    tc_mod       = load_module("transaction_costs", TRANSACTION_COSTS_PATH)

    # Load data (each module returns the columns it needs)
    print("\nLoading data ...")
    df_breakout = breakout_mod.load_data(DATA_PATH)
    df_pairs    = pairs_mod.load_data(DATA_PATH)

    # Compute Roll and CS spread estimates (full dataset, matching 09_performance.py)
    print("Computing Roll and Corwin-Schultz spread estimates ...")
    roll_spreads, cs_df = compute_spread_estimates(tc_mod)
    print(f"  Roll spreads — BTC: {roll_spreads['BTC']:.6f}  ETH: {roll_spreads['ETH']:.6f}")
    print(f"  CS spreads (mean) — BTC: {cs_df['BTC'].mean()*1e4:.2f} bps  "
          f"ETH: {cs_df['ETH'].mean()*1e4:.2f} bps")

    # Compute IS bar counts
    is_end_ts  = pd.Timestamp(IS_END, tz="UTC")
    is_bars_b  = int((df_breakout.index <= is_end_ts).sum())
    is_bars_p  = int((df_pairs.index    <= is_end_ts).sum())
    print(f"IS bars — Breakout: {is_bars_b:,}  |  Pairs: {is_bars_p:,}")

    # ---- Breakout SHA ----
    print("\n" + "-" * 60)
    print("BREAKOUT: Successive Halving Random Search")
    print("-" * 60)

    def eval_b(params, df, budget_end_idx):
        return evaluate_breakout(params, df, budget_end_idx, breakout_mod)

    b_winner, b_sha_sharpe, b_results = run_sha(
        eval_b, df_breakout, BREAKOUT_SPACE,
        N_CONFIGS, ETA, is_bars_b, label="Breakout"
    )
    print(f"\nBreakout winner: {b_winner}")
    print(f"SHA Round-3 IS Sharpe: {b_sha_sharpe:.4f}")

    # ---- Pairs SHA ----
    print("\n" + "-" * 60)
    print("PAIRS: Successive Halving Random Search")
    print("-" * 60)

    def eval_p(params, df, budget_end_idx):
        return evaluate_pairs(params, df, budget_end_idx, pairs_mod)

    p_winner, p_sha_sharpe, p_results = run_sha(
        eval_p, df_pairs, PAIRS_SPACE,
        N_CONFIGS, ETA, is_bars_p, label="Pairs"
    )
    print(f"\nPairs winner: {p_winner}")
    print(f"SHA Round-3 IS Sharpe: {p_sha_sharpe:.4f}")

    # ---- OOS validation ----
    print("\n" + "-" * 60)
    print("OOS Validation for winning configs (all three cost models)")
    print("-" * 60)

    b_val = validate_winner_breakout(b_winner, df_breakout, breakout_mod,
                                     tc_mod, roll_spreads, cs_df)
    p_val = validate_winner_pairs(p_winner, df_pairs, pairs_mod,
                                  tc_mod, roll_spreads, cs_df)

    print(f"\n{'Cost model':<12} {'Breakout IS':>12} {'Breakout OOS':>13} "
          f"{'Pairs IS':>10} {'Pairs OOS':>11}")
    print("-" * 62)
    for cm in ["fixed", "roll", "cs"]:
        b_is_s  = b_val[cm][0]["sharpe"]
        b_oos_s = b_val[cm][1]["sharpe"]
        p_is_s  = p_val[cm][0]["sharpe"]
        p_oos_s = p_val[cm][1]["sharpe"]
        print(f"{cm:<12} {b_is_s:>12.4f} {b_oos_s:>13.4f} "
              f"{p_is_s:>10.4f} {p_oos_s:>11.4f}")

    # ---- Save tables ----
    b_out = os.path.join(TABLES_DIR, "breakout_hparam_results.csv")
    p_out = os.path.join(TABLES_DIR, "pairs_hparam_results.csv")
    b_results.to_csv(b_out, index=False)
    p_results.to_csv(p_out, index=False)
    print(f"\nSaved {b_out}")
    print(f"Saved {p_out}")

    # Summary — one row per strategy × cost model
    summary_rows = []
    for strategy, val, winner in [("breakout", b_val, b_winner),
                                   ("pairs",    p_val, p_winner)]:
        for cm in ["fixed", "roll", "cs"]:
            is_m, oos_m, full_m = val[cm]
            summary_rows.append({
                "strategy":       strategy,
                "cost_model":     cm,
                "winner_params":  str(winner),
                "is_sharpe":      is_m["sharpe"],
                "oos_sharpe":     oos_m["sharpe"],
                "full_sharpe":    full_m["sharpe"],
                "is_net_pnl":     is_m["total_net_$"],
                "oos_net_pnl":    oos_m["total_net_$"],
                "full_net_pnl":   full_m["total_net_$"],
                "is_n_trades":    is_m["n_trades"],
                "oos_n_trades":   oos_m["n_trades"],
                "full_n_trades":  full_m["n_trades"],
            })
    summary_df = pd.DataFrame(summary_rows)
    s_out = os.path.join(TABLES_DIR, "hparam_summary.csv")
    summary_df.to_csv(s_out, index=False)
    print(f"Saved {s_out}")

    # ---- Plots ----
    plot_sha_sharpe_distribution(b_results, p_results)
    plot_is_oos_comparison(b_val, p_val)

    print("\nBuilding winner DataFrames for cumulative PnL plot ...")
    df_b_winner = build_winner_df_breakout(b_winner, df_breakout, breakout_mod,
                                           tc_mod, roll_spreads, cs_df)
    df_p_winner = build_winner_df_pairs(p_winner, df_pairs, pairs_mod,
                                        tc_mod, roll_spreads, cs_df)
    plot_winner_cumulative_pnl(df_b_winner, df_p_winner)

    # ---- Baseline validation (reuse same pipeline) ----
    print("\n" + "-" * 60)
    print("Baseline validation (same cost models, for comparison)")
    print("-" * 60)
    b_base_val = validate_winner_breakout(BASELINE_BREAKOUT_CFG, df_breakout,
                                          breakout_mod, tc_mod, roll_spreads, cs_df)
    p_base_val = validate_winner_pairs(BASELINE_PAIRS_CFG, df_pairs,
                                       pairs_mod, tc_mod, roll_spreads, cs_df)

    # ---- Full side-by-side evaluation ----
    print_full_evaluation("Breakout", b_base_val, b_val)
    print_full_evaluation("Pairs Trading", p_base_val, p_val)

    # ---- Comparison plot and CSV ----
    plot_sharpe_comparison(b_base_val, b_val, p_base_val, p_val)
    save_comparison_csv(b_base_val, b_val, p_base_val, p_val, b_winner, p_winner)

    print("\nDone.")


if __name__ == "__main__":
    main()
