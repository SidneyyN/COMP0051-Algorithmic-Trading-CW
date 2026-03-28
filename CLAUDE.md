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
- [ ] `03_breakout_strategy.py`
  - Donchian Channel breakout signal (rolling N-bar high / M-bar low)
  - ATR filter (only trade when ATR above threshold)
  - Optional volume confirmation
  - Volatility-targeted position sizing
  - MVO allocation across BTC/ETH/DOGE
  - Respect $100k gross exposure cap

### Stage 4 — Pairs Trading Strategy (replaces Lead-Lag)
- [ ] `04_pairs_strategy.py`
  - Decision: lead-lag effect too weak to survive costs; replaced with pairs trading
  - Asset pair(s) selected based on EDA extension cointegration results
  - Spread = log(P_A) − β·log(P_B), β estimated via OLS in-sample
  - Signal: enter when spread z-score > threshold, exit at mean-reversion
  - Rolling z-score with Ornstein-Uhlenbeck half-life for window sizing
  - Dollar-neutral positioning; gross exposure capped at $100k

### Stage 5 — Transaction Costs
- [ ] `05_transaction_costs.py`
  - Roll model: `s = sqrt(-Cov(Δp_t, Δp_{t-1}))`
  - Corwin-Schultz high-low spread estimator
  - Sensitivity analysis: Sharpe/PnL across 1bp, 5bp, 10bp, 15bp, 20bp slippage
  - Break-even slippage per strategy

### Stage 6 — Performance Evaluation
- [ ] `06_performance.py`
  - PnL: `ΔV_t = Σ(θ_i * r_i) - Cost_t`
  - Metrics: Sharpe, Sortino, Calmar, total PnL, turnover, avg holding horizon
  - Reinvestment logic (compound profits, scale down if portfolio < $10k)
  - In-sample vs out-of-sample comparison

### Stage 7 — Visualisations
- [ ] `07_visualisations.py`
  - Cumulative PnL curve (gross and net)
  - Drawdown chart
  - Position/exposure over time
  - Return distribution histogram

### Stage 8 — Report & Video
- [ ] `report/report.pdf` — 5-page report
- [ ] 60-second video presentation

---

## Current Progress

**Overall: ~40%**

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Data download | Done | `01_data_download.py` working |
| 1 — Data cleaning | Done | `02_data_clean.py` complete; 17,375 rows, Sep 2025–Feb 2026 |
| 2 — EDA (core) | Done | `03_eda.py` complete; lead-lag evidence too weak → pivot to pairs trading |
| 2 — EDA (extension) | Done | Steps 14–17 appended to `03_eda.py`; cointegration, spread, half-life, z-score |
| 3 — Breakout | Not started | |
| 4 — Pairs Trading | Not started | Replaces lead-lag strategy |
| 5 — Costs | Not started | |
| 6 — Performance | Not started | |
| 7 — Visualisations | Not started | |
| 8 — Report/Video | Not started | |

---

## Next Actions

1. **Read `PAIRS_STRATEGY_PLAN.md`** — understand the full design spec before implementing
   - EDA extension outputs (β, half-life, pair selection) are now available in `report/tables/pairs_summary.csv`

2. **Write `05_pairs_strategy.py`** — implement pairs trading strategy per plan
   - Parameters (β, entry ±2σ, exit ±0.5σ, lookback ≈ 2×half-life) from `pairs_summary.csv`
   - Dollar-neutral positioning; gross exposure capped at $100k

3. **Write `04_breakout_strategy.py`** — implement after pairs strategy is done
   - Donchian channel + ATR filter + vol-scaled sizing
   - MVO allocation across BTC/ETH/DOGE; respect $100k gross cap

---

## Key Design Decisions (for report)

- 15-min bars chosen for intraday signal resolution
- In/out-of-sample split: 67/33 on clean calendar boundaries
- Lead-lag strategy abandoned: EDA showed cross-correlations and conditional returns too small to survive costs
- Replaced with pairs trading: cointegration-based mean reversion is less sensitive to short-horizon noise
- Risk-free rate negligible at 15-min frequency (~1.4×10⁻⁵% per bar vs ~0.1–1% crypto moves)
- Roll model used for spread estimation; Corwin-Schultz as robustness check
- Gross exposure capped at min($100k, 10 × portfolio_value) at all times

---

## File Map

```
notebooks/
  01_data_download.py     — download klines + FRED (DONE)
  02_data_clean.py        — clean + returns + parquet (IN PROGRESS)
  03_eda.py               — EDA (core done; extension pending — see EDA_EXTENSION.md)
  04_breakout_strategy.py
  05_pairs_strategy.py    — pairs trading (replaces leadlag)
  06_transaction_costs.py
  07_performance.py
  08_visualisations.py

data/
  raw/        — raw CSVs from Binance (one per asset)
  cleaned/    — cleaned parquet files with returns
  risk_free/  — effr_daily.csv from FRED

src/          — reusable modules (to populate as strategies are built)
report/       — final PDF + figures
```

