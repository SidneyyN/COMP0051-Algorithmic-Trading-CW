# CLAUDE.md — COMP0051 Algorithmic Trading Coursework

## Instructions for Claude

This file is the single source of truth for project state across sessions.
**At the end of every session, update this file before closing:**
- Tick off any completed checklist items
- Update the progress table and overall percentage
- Update "Next Actions" to reflect what should be done next session
- Note any bugs, design decisions, or blockers discovered during the session

When starting a new session, read this file first to restore full context.

---

## Project Overview

Two cryptocurrency trading strategies backtested on Binance 15-min OHLCV data.
- **Assets**: BTCUSDT, ETHUSDT, DOGEUSDT
- **Period**: Sep 1 2025 – Feb 28 2026 (in-sample: Sep–Dec 2025, out-of-sample: Jan–Feb 2026)
- **Capital**: $10,000 USDT starting, max $100,000 gross exposure (10x leverage)
- **Annualisation factor**: 365.25 × 24 × 4 = 35,064 bars/year

---

## Work Plan by Section

| Section | Topic | Points |
|---------|-------|--------|
| 1 | Data pipeline (download, clean, returns) | 10 |
| 2a | Breakout / Trend-Following strategy | 20 |
| 2b | Pairs Trading strategy (replaces Lead-Lag) | 20 |
| 3 | Transaction costs (Roll model, Corwin-Schultz) | 10 |
| 4 | Performance evaluation (Sharpe, Sortino, Calmar, PnL) | 10 |
| 5 | Discussion / Next steps | 10 |
| — | Report (5 pages) + 60-sec video | remaining |

---

## Implementation Stages & Task Checklist

### Stage 1 — Data Pipeline (`notebooks/`)
- [x] `01_data_download.py` — Download 15-min klines from data.binance.vision + EFFR from FRED
- [x] `02_data_clean.py` — Clean OHLCV, compute excess returns, save parquet
  - Timestamp bug fixed: raw CSVs use microseconds (16-digit), not milliseconds.
    Auto-detection added (>1e14 → divide by 1000 → unit="ms").
  - Output: single wide `data/cleaned/cleaned_data.parquet` (17,375 × 22 cols).
  - Outliers flagged (log only, not removed): 7 BTC, 9 ETH, 12 DOGE.
  - Notable: DOGE Oct 10 2025 flash crash (-6%, -8%, -16% over 3 bars) retained as genuine.

### Stage 2 — Exploratory Data Analysis
- [x] `03_eda.py` — Stationarity tests (ADF), cross-correlations, Granger causality, volatility regimes
  - 13-step pipeline: summary stats, rolling vol, correlations, lead-lag CCF, conditional returns, Granger, ACF/price diagnostics
  - Outputs to `report/figures/` (7 plots) and `report/tables/` (5 CSV tables)
  - Requires scipy + statsmodels in the active virtual environment
  - EDA results showed lead-lag predictive structure is too weak to trade; strategy replaced with pairs trading
- [x] `03_eda.py` extension — Pairs trading diagnostics (see `EDA_EXTENSION.md`)
  - Steps 14–17 appended to `03_eda.py` (runs after core EDA steps 1–13)
  - Step 14: log prices + ADF unit root tests on prices (expect non-stationary)
  - Step 15: Engle-Granger cointegration test on BTC–ETH, BTC–DOGE, ETH–DOGE
  - Step 16: OLS hedge ratio (in-sample), spread construction, ADF on spread, OU half-life; pair selection recommendation
  - Step 17: Z-score distribution plots (signal frequency at ±1σ, ±2σ)
  - Outputs: `pairs_log_price_adf.csv`, `pairs_cointegration.csv`, `pairs_summary.csv`, `pairs_spreads.png`, `pairs_zscore_distributions.png`

### Stage 3 — Breakout Strategy
- [x] `04_breakout_strategy.py` — contrarian Donchian channel breakout (BTC, baseline)
  - **Key finding**: BTC at 15-min frequency is mean-reverting around breakouts, not trending
  - Signals: short when close > upper band (fade spike), long when close < lower band (buy dip)
  - Lookback N=200 bars; vol filter: trade only when rolling 50-bar vol > 60th percentile (IS-computed)
  - Max holding period 40 bars; position: +1 long, -1 short, 0 flat; wait one bar after exit before re-entry
  - Reversal cost fix: `trade = abs(position - position.shift(1))` counts +1→-1 as 2 units
  - Per-trade win rate via trade_id groupby (not per-bar)
  - IS Sharpe ~1.05, OOS Sharpe ~1.51 (OOS > IS: no overfitting); costs ~38% IS, ~28% OOS of gross
  - Outputs: `breakout_price_bands.png`, `breakout_equity_curve.png`
- [x] `05_breakout_extension.py` — breakout extension: threshold filter + banded vol filter
  - Three versions compared: Baseline, Test A (threshold only), Test B (threshold + 20–80th pct vol band)
  - THRESHOLD=0.001 (10 bps), VOL_Q_LOW=0.2, VOL_Q_HIGH=0.8
  - Test A: IS Sharpe 0.45, OOS Sharpe 1.67 (slightly ↑ vs baseline 1.51); fewer trades, lower costs
  - Test B: IS Sharpe 0.87 but OOS Sharpe -2.24 — banded vol filter destabilises OOS (overfits IS regime)
  - Outputs: `breakout_extension_comparison.csv`, 4 plots

### Stage 4 — Pairs Trading Strategy (replaces Lead-Lag)
- [x] `06_pairs_strategy.py` — baseline pairs strategy
  - BTC–ETH pair, β = 0.6780 (OLS in-sample)
  - Spread = log(BTC) − 0.6780·log(ETH); rolling 100-bar z-score
  - Entry ±3σ | exit z=0 | stop-loss ±3σ after min-hold | min-hold 24 bars | cooldown 20 bars | time-stop 384 bars
  - Dollar-neutral: btc_notional = $100k/(1+β), eth_notional = β·btc_notional
  - Leg-level PnL; proportional cost model; sensitivity sweep 1–20 bps
  - Outputs: `pairs_backtest.csv`, `pairs_trade_log.csv`, `pairs_performance.csv`, `pairs_cost_sensitivity.csv`, 4 plots
- [x] `07_pairs_strategy_vol_filter.py` — pairs strategy + volatility filter
  - All parameters identical to `06_pairs_strategy.py`
  - Added volatility filter: `spread_vol = spread.diff().rolling(100).std()`, `vol_ref = spread_vol.rolling(200).median()`, entry blocked when `spread_vol ≥ 1.2 × vol_ref`
  - Filter applied to entry only; exits (mean-reversion, stop, time-stop) are unaffected
  - Outputs: `vol_filter_backtest.csv`, `vol_filter_trade_log.csv`, `vol_filter_performance.csv`, `vol_filter_cost_sensitivity.csv`, 4 plots (spread plot has 3 panels incl. vol regime panel)

### Stage 5 — Transaction Costs
- [ ] `08_transaction_costs.py`
  - Roll model: `s = sqrt(-Cov(Δp_t, Δp_{t-1}))`
  - Corwin-Schultz high-low spread estimator
  - Sensitivity analysis: Sharpe/PnL across 1bp, 5bp, 10bp, 15bp, 20bp slippage
  - Break-even slippage per strategy

### Stage 6 — Performance Evaluation
- [ ] `09_performance.py`
  - PnL: `ΔV_t = Σ(θ_i * r_i) - Cost_t`
  - Metrics: Sharpe, Sortino, Calmar, total PnL, turnover, avg holding horizon
  - Reinvestment logic (compound profits, scale down if portfolio < $10k)
  - In-sample vs out-of-sample comparison

### Stage 7 — Visualisations
- [ ] `10_visualisations.py`
  - Cumulative PnL curve (gross and net)
  - Drawdown chart
  - Position/exposure over time
  - Return distribution histogram

### Stage 8 — Report & Video
- [ ] `report/report.pdf` — 5-page report
- [ ] 60-second video presentation

---

## Current Progress

**Overall: ~70%**

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Data download | Done | `01_data_download.py` working |
| 1 — Data cleaning | Done | `02_data_clean.py` complete; 17,375 rows, Sep 2025–Feb 2026 |
| 2 — EDA (core) | Done | `03_eda.py` complete; lead-lag evidence too weak → pivot to pairs trading |
| 2 — EDA (extension) | Done | Steps 14–17 appended to `03_eda.py`; cointegration, spread, half-life, z-score |
| 3 — Breakout (baseline) | Done | `04_breakout_strategy.py` complete; contrarian fade strategy; IS Sharpe 1.05, OOS Sharpe 1.51 |
| 3 — Breakout (extension) | Done | `05_breakout_extension.py`; threshold+vol-band comparison; Test A mild OOS ↑, Test B OOS Sharpe -2.24 |
| 4 — Pairs Trading | Done | `06_pairs_strategy.py` baseline + `07_pairs_strategy_vol_filter.py` with vol regime filter on entry |
| 5 — Costs | Not started | |
| 6 — Performance | Not started | |
| 7 — Visualisations | Not started | |
| 8 — Report/Video | Not started | |

---

## Next Actions

1. **Write `08_transaction_costs.py`** — Roll model + Corwin-Schultz spread estimation
   - Roll model: `s = 2 × sqrt(-Cov(Δp_t, Δp_{t-1}))` on close returns
   - Corwin-Schultz: high-low spread estimator; compute per asset
   - Compare estimated spreads to the cost sweep assumptions used in strategies
   - Break-even slippage per strategy

3. **Write `09_performance.py`** — combined performance evaluation
   - Combine pairs + breakout PnL into portfolio-level metrics
   - Metrics: Sharpe, Sortino, Calmar, total PnL, turnover, avg holding horizon
   - In-sample vs out-of-sample comparison table

---

## Key Design Decisions (for report)

- 15-min bars chosen for intraday signal resolution
- In/out-of-sample split: 67/33 on clean calendar boundaries
- Lead-lag strategy abandoned: EDA showed cross-correlations and conditional returns too small to survive costs
- Replaced with pairs trading: cointegration-based mean reversion is less sensitive to short-horizon noise
- Risk-free rate negligible at 15-min frequency (~1.4×10⁻⁵% per bar vs ~0.1–1% crypto moves)
- Breakout strategy is **contrarian** (fade the breakout), not trend-following: BTC at 15-min frequency mean-reverts after band breaks; confirmed by flipping signals and observing strongly positive gross PnL
- Breakout extension (Test A): 10 bps threshold marginally improves OOS Sharpe (1.51→1.67) with fewer trades; IS Sharpe drops (1.06→0.45) — suggests baseline was partly trading marginal signals that happen to be profitable IS
- Breakout extension (Test B): banded vol filter (20th–80th pct) catastrophically harms OOS (Sharpe -2.24); the IS vol-regime structure does not persist OOS — strong evidence against over-filtering
- Roll model used for spread estimation; Corwin-Schultz as robustness check
- Gross exposure capped at min($100k, 10 × portfolio_value) at all times

---

## File Map

```
notebooks/
  01_data_download.py     — download klines + FRED (DONE)
  02_data_clean.py        — clean + returns + parquet (IN PROGRESS)
  03_eda.py               — EDA (core done; extension pending — see EDA_EXTENSION.md)
  04_breakout_strategy.py          — contrarian Donchian breakout, BTC baseline (DONE)
  05_breakout_extension.py         — breakout extension: threshold + banded vol filter (DONE)
  06_pairs_strategy.py             — pairs trading BTC–ETH baseline (DONE)
  07_pairs_strategy_vol_filter.py  — pairs trading + volatility filter on entry (DONE)
  08_transaction_costs.py          — Roll model + Corwin-Schultz (not started)
  09_performance.py                — combined performance evaluation (not started)
  10_visualisations.py             — combined visualisations (not started)

data/
  raw/        — raw CSVs from Binance (one per asset)
  cleaned/    — cleaned parquet files with returns
  risk_free/  — effr_daily.csv from FRED

src/          — reusable modules (to populate as strategies are built)
report/       — final PDF + figures
```

