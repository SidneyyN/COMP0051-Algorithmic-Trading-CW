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
- [x] `08_transaction_costs.py`
  - Roll model: BTC 0.000624, ETH 0.000524, DOGE 0.000993 (scalars, fraction of price)
  - Corwin-Schultz mean: BTC 7.77 bps, ETH 11.74 bps, DOGE 14.58 bps (time series)
  - `compute_cost(position, spread, price)` → `delta_pos × spread × price / 2`; flip (+1→-1) costs 2 units correctly
  - Functions: `load_data`, `roll_spread`, `cs_spread`, `compute_cost`

### Stage 6 — Performance Evaluation
- [x] `09_performance.py`
  - Loads spread estimates, regenerates positions for both strategies via `importlib`
  - Breakout IS: Fixed Sharpe 1.38, Roll 1.30, CS 1.18 | OOS: Fixed 1.88, Roll 1.82, CS 1.58
  - Pairs IS: Fixed Sharpe 0.08, Roll -0.12, CS -0.95 — barely survives fixed costs, unprofitable under Roll/CS
  - Pairs OOS: Fixed 1.24, Roll 1.04, CS 0.54 — better OOS but highly cost-sensitive
  - Turnover: breakout 182 total (367 ann.), pairs 116 total (234 ann.)
  - Cost sensitivity (Roll base): breakout Sharpe stays positive at 2× spread; pairs goes negative at 1.5×
  - Key insight: pairs trading has too small a gross edge to survive microstructure costs IS

### Stage 7 — Visualisations
- [x] `10_visualisations.py`
  - Cumulative PnL curves (gross vs net, three cost models) — `vis_cumulative_pnl.png`
  - Drawdown chart (gross + three net cost models) — `vis_drawdown.png`
  - Position/exposure over time (breakout notional + pairs spread direction) — `vis_positions.png`
  - Return distribution histograms (IS vs OOS, gross vs net Roll) — `vis_return_distributions.png`
  - Timezone fix needed: tz-aware UTC index requires `IS_END` localized before use in `axvspan`/`axvline`

### Stage 8 — Hyperparameter Search
- [x] `11_hyperparameter_search.py` — Random Search + Successive Halving (SHA) for both strategies
  - Algorithm: η=3, 27 initial configs → SHA rounds (1/3 IS → 2/3 IS → full IS), 39 evaluations per strategy
  - **Breakout search space** (4 params): N ∈ [100,150,200,300,400], vol_window ∈ [30,50,75,100], vol_quantile ∈ [0.4–0.8], max_hold ∈ [20,40,60,100]
  - **Pairs search space** (6 params): coint_window ∈ [500,1000,2000,None], zscore_window ∈ [50–200], entry_z ∈ [2.0–3.5], exit_z ∈ [0.0–1.0], stop_z ∈ [2.5–5.0], vol_mult ∈ [1.0–2.0]
  - `coint_window` is a new parameter: rolling OLS beta estimation window (None = static OLS on IS, same semantics as baseline)
  - SHA evaluation uses fixed 5 bps cost for speed; final validation uses all three cost models (Fixed, Roll, CS)
  - **Breakout winner**: N=150, vol_window=30, vol_quantile=0.6, max_hold=60
  - **Pairs winner**: coint_window=None, zscore_window=75, entry_z=3.5, exit_z=0.0, stop_z=3.0, vol_mult=1.5
  - **Key finding**: SHA winner improves IS Sharpe for both strategies but OOS collapses — baseline hand-picked params are more robust (breakout baseline full-period Roll Sharpe 1.49 vs winner 1.01; pairs baseline OOS Roll Sharpe 1.19 vs winner -0.32)
  - Full side-by-side comparison printed (IS / OOS / full period × 3 cost models × 9 metrics) with delta column
  - Outputs:
    - `report/tables/breakout_hparam_results.csv` — 39-row SHA search log
    - `report/tables/pairs_hparam_results.csv` — 39-row SHA search log
    - `report/tables/hparam_summary.csv` — winner metrics per cost model
    - `report/tables/baseline_vs_winner_comparison.csv` — full flat comparison table
    - `report/figures/hparam_sha_sharpe_distribution.png` — boxplot of Sharpe per SHA round
    - `report/figures/hparam_is_vs_oos_sharpe.png` — winner IS vs OOS by cost model
    - `report/figures/hparam_winner_cumulative_pnl.png` — winner cumulative PnL (matches `vis_cumulative_pnl.png` style)
    - `report/figures/hparam_baseline_vs_winner_sharpe.png` — grouped bar chart baseline vs winner

### Stage 9 — Walk-Forward Validation
- [x] `12_walk_forward.py` — Expanding-window WFV, 3 monthly test folds, baseline vs SHA winner
  - Fold structure: Train Sep–Nov → Test Dec | Train Sep–Dec → Test Jan | Train Sep–Jan → Test Feb
  - Per-fold calibration: vol_threshold (breakout) and OLS beta (pairs) fit on train window only
  - Transaction costs: Roll model scalars (BTC 0.000624, ETH 0.000524), same as notebook 09
  - **Fold 2 sanity check passes**: Breakout baseline Sharpe 1.8148 ≈ notebook 09 OOS Roll 1.82
  - **Breakout results**: Baseline mean Sharpe 3.72 ± 3.32 (all folds positive: 7.56, 1.81, 1.80); SHA winner 1.82 ± 3.37 (goes negative fold 3: -0.97)
  - **Pairs results**: Baseline mean Sharpe 1.55 ± 1.33 (all positive: 3.03, 1.17, 0.46); SHA winner -0.84 ± 1.15 (negative in 2/3 folds)
  - **Beta stability confirmed**: fold betas (0.617, 0.678, 0.688) all close to static 0.6780, supporting coint_window=None
  - **Key finding**: WFV multi-fold evidence is stronger than single IS/OOS split — baseline consistently outperforms SHA winner across all folds; SHA overfitting to IS period is systematic, not period-specific
  - Outputs:
    - `report/tables/wfv_results.csv` — per-fold × config × metric table
    - `report/figures/wfv_sharpe_by_fold.png` — grouped bar chart by fold
    - `report/figures/wfv_cumulative_pnl.png` — cumulative OOS PnL with fold boundaries

### Stage 10 — Report & Video
- [ ] `report/report.pdf` — 5-page report
- [ ] 60-second video presentation

---

## Current Progress

**Overall: ~97%**

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Data download | Done | `01_data_download.py` working |
| 1 — Data cleaning | Done | `02_data_clean.py` complete; 17,375 rows, Sep 2025–Feb 2026 |
| 2 — EDA (core) | Done | `03_eda.py` complete; lead-lag evidence too weak → pivot to pairs trading |
| 2 — EDA (extension) | Done | Steps 14–17 appended to `03_eda.py`; cointegration, spread, half-life, z-score |
| 3 — Breakout (baseline) | Done | `04_breakout_strategy.py` complete; contrarian fade strategy; IS Sharpe 1.05, OOS Sharpe 1.51 |
| 3 — Breakout (extension) | Done | `05_breakout_extension.py`; threshold+vol-band comparison; Test A mild OOS ↑, Test B OOS Sharpe -2.24 |
| 4 — Pairs Trading | Done | `06_pairs_strategy.py` baseline + `07_pairs_strategy_vol_filter.py` with vol regime filter on entry |
| 5 — Costs | Done | `08_transaction_costs.py`; Roll + Corwin-Schultz spreads; `compute_cost` function |
| 6 — Performance | Done | `09_performance.py`; all three cost models × both strategies × IS/OOS; turnover + sensitivity |
| 7 — Visualisations | Done | `10_visualisations.py`; 4 figures saved to `report/figures/` |
| 8 — Hyperparameter Search | Done | `11_hyperparameter_search.py`; SHA random search; winner IS Sharpe ↑ but baseline more robust OOS |
| 9 — Walk-Forward Validation | Done | `12_walk_forward.py`; 3 expanding folds; baseline beats SHA winner in all folds |
| 10 — Report/Video | Not started | |

---

## Next Actions

1. **Write report** — 5 pages covering all sections
   - Key narrative: breakout survives costs (robust edge), pairs is cost-sensitive (marginal IS edge)
   - Hyperparameter search narrative: SHA winner improves IS Sharpe but baseline parameters are more OOS-robust — illustrates overfitting risk of IS-only optimisation
   - WFV narrative: 3-fold expanding window confirms baseline robustness; SHA winner goes negative in multiple folds; beta stability (0.617–0.688) validates static OLS choice
   - Include all figures from `report/figures/`; tables from `report/tables/`

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
- Breakout is cost-robust: gross edge large enough that even CS spreads leave Sharpe >1.18 IS and >1.58 OOS
- Pairs IS edge too small: gross PnL $1,924 barely exceeds fixed cost $1,759; negative net PnL under Roll/CS IS
- Pairs OOS looks better (fixed Sharpe 1.24) but sensitivity sweep shows it breaks at ~1.5× Roll spread
- Pairs cost sensitivity used pre-computed base cost series scaled by multiplier (avoids two-leg price indexing issue)
- SHA hyperparameter search confirms IS overfitting risk: winner configs improve IS Sharpe substantially but collapse OOS; hand-picked baseline params act as implicit regularisation
- `coint_window=None` (static OLS on IS) outperforms rolling OLS windows in the pairs search — suggests the BTC–ETH cointegration relationship is stable enough that re-estimation adds noise rather than adaptability
- WFV (3 expanding folds) provides stronger evidence than single IS/OOS split: breakout baseline Sharpe positive across all folds (7.56, 1.81, 1.80); SHA winner goes negative in fold 3 (-0.97); pairs baseline positive all folds, SHA winner negative in 2/3. The Dec fold Sharpe of 7.56 is an outlier (strong trending month) — mean ± std (3.72 ± 3.32) gives a more honest picture than single OOS (1.82)

---

## File Map

```
notebooks/
  01_data_download.py     — download klines + FRED (DONE)
  02_data_clean.py        — clean + returns + parquet (DONE)
  03_eda.py               — EDA core + pairs trading diagnostics extension, steps 1–17 (DONE)
  04_breakout_strategy.py          — contrarian Donchian breakout, BTC baseline (DONE)
  05_breakout_extension.py         — breakout extension: threshold + banded vol filter (DONE)
  06_pairs_strategy.py             — pairs trading BTC–ETH baseline (DONE)
  07_pairs_strategy_vol_filter.py  — pairs trading + volatility filter on entry (DONE)
  08_transaction_costs.py          — Roll model + Corwin-Schultz (DONE)
  09_performance.py                — combined performance evaluation (DONE)
  10_visualisations.py             — combined visualisations (DONE)
  11_hyperparameter_search.py      — SHA random search + baseline vs winner comparison (DONE)
  12_walk_forward.py               — expanding-window WFV, 3 folds, baseline vs SHA winner (DONE)

data/
  raw/        — raw CSVs from Binance (one per asset)
  cleaned/    — cleaned parquet files with returns
  risk_free/  — effr_daily.csv from FRED

src/          — reusable modules (to populate as strategies are built)
report/       — final PDF + figures
```

