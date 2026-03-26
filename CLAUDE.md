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
| 2b | Lead-Lag / Causal strategy | 20 |
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

### Stage 3 — Breakout Strategy
- [ ] `03_breakout_strategy.py`
  - Donchian Channel breakout signal (rolling N-bar high / M-bar low)
  - ATR filter (only trade when ATR above threshold)
  - Optional volume confirmation
  - Volatility-targeted position sizing
  - MVO allocation across BTC/ETH/DOGE
  - Respect $100k gross exposure cap

### Stage 4 — Lead-Lag Strategy
- [ ] `04_leadlag_strategy.py`
  - Rolling cross-correlation (BTC vs ETH/DOGE, lags 1–4 bars)
  - Granger causality test over rolling windows
  - Signal: BTC return > 1 z-score → trade lagging asset in same direction
  - Exit after expected lag period (2–3 bars) or catch-up
  - Signal-proportional position sizing

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

**Overall: ~30%**

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Data download | Done | `01_data_download.py` working |
| 1 — Data cleaning | Done | `02_data_clean.py` complete; 17,375 rows, Sep 2025–Feb 2026 |
| 2 — EDA | Done | `03_eda.py` written; run after activating venv with scipy/statsmodels |
| 3 — Breakout | Not started | |
| 4 — Lead-Lag | Not started | |
| 5 — Costs | Not started | |
| 6 — Performance | Not started | |
| 7 — Visualisations | Not started | |
| 8 — Report/Video | Not started | |

---

## Next Actions

1. **Activate venv, install scipy + statsmodels, then run `03_eda.py`**
   - Inspect the cross-correlation and Granger output to confirm lead-lag evidence
   - Review conditional returns table — if mean fwd return at lag-1 is < 1 bps, reconsider lead-lag strategy aggressiveness

2. **Write `04_breakout_strategy.py`** — Donchian channel + ATR filter + vol-scaled sizing
   - Use ATR threshold calibrated from EDA rolling-vol output
   - MVO allocation across BTC/ETH/DOGE; respect $100k gross cap

3. **Write `05_leadlag_strategy.py`** — BTC signal → ETH/DOGE trade
   - Use Granger + CCF results from EDA to set lag and z-score threshold

---

## Key Design Decisions (for report)

- 15-min bars chosen for lead-lag detection (1–4 bar lag ≈ 15–60 min)
- In/out-of-sample split: 67/33 on clean calendar boundaries
- Risk-free rate negligible at 15-min frequency (~1.4×10⁻⁵% per bar vs ~0.1–1% crypto moves)
- Roll model used for spread estimation; Corwin-Schultz as robustness check
- Gross exposure capped at min($100k, 10 × portfolio_value) at all times

---

## File Map

```
notebooks/
  01_data_download.py     — download klines + FRED (DONE)
  02_data_clean.py        — clean + returns + parquet (IN PROGRESS)
  03_eda.py               — EDA
  04_breakout_strategy.py
  05_leadlag_strategy.py
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

