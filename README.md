# COMP0051 Algorithmic Trading Coursework 2025/26

## Overview

This project implements two cryptocurrency trading strategies — **Breakout/Trend-Following** and **Lead-Lag/Causal Trading** — backtested on Binance 15-minute OHLCV data. Starting capital is $10,000 USDT with a maximum gross exposure of $100,000 (10x leverage).

---

## Project Structure

```
comp0051-coursework/
│
├── README.md                    # This file — project plan and documentation
├── data/
│   ├── raw/                     # Raw CSVs downloaded from Binance
│   ├── cleaned/                 # Cleaned data in parquet format
│   └── risk_free/               # Risk-free rate data (Fed Funds rate)
│
├── notebooks/
│   ├── 01_data_download.py      # Data acquisition (Binance klines + FRED risk-free rate)
│   ├── 02_data_clean.py         # Data cleaning, excess returns, save to parquet
│   ├── 03_eda.py                # Exploratory data analysis
│   ├── 04_breakout_strategy.py  # Breakout/trend-following strategy
│   ├── 05_leadlag_strategy.py   # Lead-lag/causal strategy
│   ├── 06_transaction_costs.py  # Roll model and slippage estimation
│   ├── 07_performance.py        # PnL, metrics, and final evaluation
│   └── 08_visualisations.py     # Charts and plots for the report
│
├── src/
│   ├── data_utils.py            # Data loading, cleaning, return computation
│   ├── strategy_base.py         # Base class for strategy interface
│   ├── breakout.py              # Breakout strategy implementation
│   ├── leadlag.py               # Lead-lag strategy implementation
│   ├── costs.py                 # Transaction cost models (Roll, etc.)
│   ├── performance.py           # Sharpe, Sortino, Calmar, PnL computation
│   └── portfolio.py             # MVO / position sizing and exposure management
│
├── report/
│   ├── report.pdf               # Final 5-page report
│   └── figures/                 # Exported charts for the report
│
└── requirements.txt             # Python dependencies
```

---

## Data (Section 1 — 10 pts)

### Assets
- **BTC/USDT** — primary asset, market leader, used as the "leading" asset
- **ETH/USDT** — secondary asset, often lags BTC
- **DOGE/USDT** — altcoin with high volatility, strong lag relationship with BTC

### Frequency
- **15-minute bars** — chosen because:
  - Lead-lag effects between BTC and alts typically play out over 15–60 minutes
  - Enough granularity to capture intraday breakouts
  - ~17,500 bars per asset over 6 months (well above the 1,000-bar minimum)

### Time Period
- **September 1, 2025 – February 28, 2026** (6 complete months)
- ~17,500 bars per asset (6 × ~30 × 24 × 4)
- **In-sample**: Sep 1, 2025 – Dec 31, 2025 (~4 months, ~67%)
- **Out-of-sample**: Jan 1, 2026 – Feb 28, 2026 (~2 months, ~33%)
- Rationale: clean calendar boundaries, fully complete data, recent enough to reflect current market microstructure, avoids cherry-picking

### Data Pipeline
1. Download raw 15-min klines from `data.binance.vision` (public, no API key needed)
2. Columns: Open time, Open, High, Low, Close, Volume (+ additional fields we may discard)
3. Clean data: check for missing bars, handle NaN/zero volume, detect outliers (read section below for more details)
4. Store cleaned data as `.parquet`
5. Download Effective Federal Funds Rate from FRED (Federal Reserve Economic Data)
6. Compute excess simple returns: `r_e_t = (p_t - p_{t-1}) / p_{t-1} - r_f_{t-1}`

### Data Cleaning

Raw data is checked for missing timestamps, duplicate entries, and basic integrity issues (e.g., zero or negative prices). All timestamps are aligned across assets.

Outliers are identified using extreme return thresholds (e.g., returns exceeding a multiple of rolling standard deviation). However, cryptocurrency markets are known to exhibit genuine large price movements, particularly during periods of high volatility.

Therefore, **outliers are not automatically removed or adjusted**. Instead:
- Each flagged observation is inspected in context (neighbouring bars, volume, and market behaviour)
- Corrections are only applied if the anomaly is clearly attributable to a data error (e.g., isolated spikes inconsistent with surrounding data)
- Otherwise, extreme observations are retained as valid market events

This ensures that the dataset preserves true market dynamics while avoiding distortions from erroneous data.

### Risk-Free Rate Discussion
At 15-minute frequency, the per-bar risk-free rate is negligible. With an annual rate of ~5%, the 15-min rate is approximately 5% / (365.25 × 24 × 4) ≈ 0.000014% per bar. This is orders of magnitude smaller than typical crypto returns (which can move 0.1–1% in 15 minutes). We compute it for completeness but argue it can be safely ignored.

---

## Strategy 1: Breakout / Trend-Following (Section 2a — 20 pts)

### Economic Rationale
Cryptocurrency markets exhibit momentum and trend persistence due to:
- Herding behaviour and retail FOMO (fear of missing out)
- Cascading liquidations that amplify directional moves
- Low institutional market-making relative to traditional markets

### Signal Construction
- **Donchian Channel Breakout**: track the rolling N-bar high and M-bar low
  - Long entry: price breaks above N-bar high
  - Short entry: price breaks below M-bar low
  - Exit: price crosses the opposite channel or a middle band
- **ATR (Average True Range) filter**: only take breakout signals when ATR is above a threshold (filters out low-volatility, choppy periods that cause whipsaws)
- **Volume confirmation** (optional): require above-average volume on the breakout bar

### Parameters to Optimise
- Lookback window for channel (N, M) — e.g., 96 bars = 24 hours, 192 bars = 48 hours
- ATR lookback and threshold
- Exit rules: trailing stop vs opposite channel vs time-based exit

### Position Sizing

We adopt a **volatility scaling and exposure cap** approach to ensure robust and interpretable position sizing.

- Positions are scaled inversely with recent volatility (measured using ATR or rolling standard deviation of returns), such that more volatile assets receive smaller allocations.
- Signal strength (e.g., breakout strength or magnitude of BTC move in lead-lag) can be incorporated as a multiplier to increase exposure when signals are stronger.
- At each time step, raw position sizes are normalised to satisfy the coursework constraint:
  \[
  \sum_i |\theta_t^i| \le 100{,}000
  \]
  ensuring total gross exposure does not exceed \$100,000.

This approach avoids the instability of full mean-variance optimisation while remaining cost-aware and suitable for high-frequency data. It also reflects practical trading systems, where risk is controlled via volatility targeting and strict exposure limits.

### Assets Traded
- All three: BTC, ETH, DOGE (breakout signals generated independently per asset)

---

## Strategy 2: Lead-Lag / Causal Trading (Section 2b — 20 pts)

### Economic Rationale
BTC is the dominant cryptocurrency and tends to be the first to react to market-wide information (macroeconomic news, regulatory events, sentiment shifts). Altcoins like ETH and DOGE often follow with a delay of 1–4 bars at 15-minute frequency. This is due to:
- BTC's deeper liquidity and tighter spreads attracting informed traders first
- Algorithmic arbitrageurs not fully closing the gap instantly
- Retail traders reacting to BTC moves and rotating into alts

### Signal Construction
1. **Cross-correlation analysis**: compute rolling cross-correlations between BTC returns and ETH/DOGE returns at lags 1, 2, 3, 4 bars to identify optimal lag
2. **Granger causality test**: formally test whether BTC returns Granger-cause ETH/DOGE returns (and vice versa) over rolling windows
3. **Trading signal**: when BTC has a significant return (above a threshold, e.g., > 1 z-score), take a position in the lagging asset in the same direction
4. **Signal decay**: exit after the expected lag period (e.g., 2–3 bars) or when the lagging asset has "caught up"

### Parameters to Optimise
- Lag order (1–4 bars)
- BTC return threshold for triggering a trade
- Rolling window for cross-correlation / Granger test estimation
- Exit timing (fixed hold period vs signal-based exit)

### Position Sizing

We adopt a **volatility scaling and exposure cap** approach to ensure robust and interpretable position sizing.

- Positions are scaled inversely with recent volatility (measured using ATR or rolling standard deviation of returns), such that more volatile assets receive smaller allocations.
- Signal strength (e.g., breakout strength or magnitude of BTC move in lead-lag) can be incorporated as a multiplier to increase exposure when signals are stronger.
- At each time step, raw position sizes are normalised to satisfy the coursework constraint:
  \[
  \sum_i |\theta_t^i| \le 100{,}000
  \]
  ensuring total gross exposure does not exceed \$100,000.

This approach avoids the instability of full mean-variance optimisation while remaining cost-aware and suitable for high-frequency data. It also reflects practical trading systems, where risk is controlled via volatility targeting and strict exposure limits.

### Assets Traded
- Signal asset: BTC (observe only, may or may not trade)
- Trade assets: ETH, DOGE (take positions based on BTC signal)

---

## Transaction Costs (Section 3 — 10 pts)

### Roll Model
Estimate the effective bid-ask spread (slippage) using:
```
s = sqrt(-Cov(Δp_t, Δp_{t-1}))
```
where Δp_t = p_t - p_{t-1} are price changes. If the autocovariance is positive (which can happen), note this and consider alternative estimators.

### Alternative Estimators to Consider
- **Corwin-Schultz (2012)** high-low spread estimator — uses the ratio of daily high-low ranges to estimate the spread. May be more robust for crypto.
- Compare Roll vs Corwin-Schultz estimates and discuss which is more appropriate.

### Sensitivity Analysis
- Compute strategy performance (Sharpe, total PnL) across a range of slippage values: 1bp, 5bp, 10bp, 15bp, 20bp
- Show at what slippage level each strategy breaks even (Sharpe ≈ 0)
- This is critical for assessing real-world viability

---

## Performance Evaluation (Section 4 — 10 pts)

### PnL Computation
```
ΔV_t = Σ_i (θ_i_t × r_i_t) - Cost_t
Cost_t = s × Σ_i |θ_i_t - θ_i_{t-1} × (1 + r_i_{t-1})|
```
- Track both gross (before costs) and net (after costs) PnL
- Unallocated capital remains as USDT cash (no interest assumed)

### Metrics
- **Sharpe Ratio**: annualised, net of slippage
- **Sortino Ratio**: using downside deviation only
- **Calmar Ratio**: annualised return / maximum drawdown
- **Total PnL** in USDT and percentage return on $10,000
- **Total turnover**: sum of absolute position changes
- **Average holding horizon**: mean duration of each trade

### Reinvestment Logic
- Profits are reinvested (portfolio value compounds)
- Gross exposure is capped at min($100,000, 10 × current_portfolio_value)
- If portfolio drops below $10,000, exposure scales down proportionally

### Visualisations
- Cumulative PnL curve (gross and net) for each strategy
- Drawdown chart
- Position/exposure over time
- Return distribution histogram

---

## Next Steps (Section 5 — 10 pts)

### Topics to Discuss
- **Regime dependence**: how do the strategies perform in bull, bear, and sideways markets? Segment backtest by market regime.
- **Parameter stability**: do optimal parameters change over time? Walk-forward optimisation.
- **Live trading considerations**: API latency, order book depth, execution slippage beyond the Roll estimate, exchange downtime, funding rates for perpetual futures.
- **Improvements**:
  - Adaptive parameters (e.g., adjust breakout window based on volatility regime)
  - Machine learning for signal combination (e.g., random forest on breakout + lead-lag features)
  - Adding more assets to the lead-lag universe
  - Incorporating order flow / volume imbalance data
- **Market conditions**: breakout works best in volatile trending markets; lead-lag works best when BTC dominance is high and cross-asset correlations are strong but lagged.

---

## Dependencies

```
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
pyarrow          # for parquet I/O
scikit-learn     # for any ML-based extensions
requests         # for downloading data / risk-free rate from FRED
```

---

## Workflow

1. **Data download** → `01_data_download.py`
2. **Data cleaning** → `02_data_clean.py`
3. **Exploratory analysis** → `03_eda.py` (stationarity, cross-correlations, cointegration)
4. **Strategy 1** → `04_breakout_strategy.py`
5. **Strategy 2** → `05_leadlag_strategy.py`
6. **Transaction costs** → `06_transaction_costs.py`
7. **Performance** → `07_performance.py`
8. **Visualisations** → `08_visualisations.py`
9. **Report writing** → compile into 5-page PDF
10. **Video** → 60-second presentation to camera

---

## Notes

- All times in UTC
- All prices in USDT
- Annualisation factor for 15-min data: 365.25 × 24 × 4 = 35,064 bars/year
- In-sample / out-of-sample split: ~70/30 by time
