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
│   ├── 01_data_download.py      # Data acquisition and cleaning
│   ├── 02_eda.py                # Exploratory data analysis
│   ├── 03_breakout_strategy.py  # Breakout/trend-following strategy
│   ├── 04_leadlag_strategy.py   # Lead-lag/causal strategy
│   ├── 05_transaction_costs.py  # Roll model and slippage estimation
│   ├── 06_performance.py        # PnL, metrics, and final evaluation
│   └── 07_visualisations.py     # Charts and plots for the report
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
  - ~35,000 bars per asset over ~3 months (well above the 1,000-bar minimum)

### Time Period
- Target: **3–6 months** of recent data
- Split: ~70% in-sample (parameter fitting), ~30% out-of-sample (validation)

### Data Pipeline
1. Download raw 15-min klines from `data.binance.vision` (public, no API key needed)
2. Columns: Open time, Open, High, Low, Close, Volume (+ additional fields we may discard)
3. Clean data: check for missing bars, handle NaN/zero volume, detect and repair outliers (e.g., wicks > 5 standard deviations from rolling mean)
4. Store cleaned data as `.parquet`
5. Download Effective Federal Funds Rate from FRED (Federal Reserve Economic Data)
6. Compute excess simple returns: `r_e_t = (p_t - p_{t-1}) / p_{t-1} - r_f_{t-1}`

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
- Use volatility targeting: size positions inversely proportional to recent ATR
- Apply MVO (Mean-Variance Optimisation) or a simplified cost-aware method to allocate across BTC/ETH/DOGE
- Respect the $100,000 gross exposure cap at all times

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
- Size proportional to signal strength (magnitude of BTC move) and inversely proportional to lagging asset's volatility
- Apply MVO or cost-aware optimisation
- Respect $100,000 gross exposure cap

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

1. **Data pipeline** → `01_data_download.py`
2. **Exploratory analysis** → `02_eda.py` (stationarity, cross-correlations, cointegration)
3. **Strategy 1** → `03_breakout_strategy.py`
4. **Strategy 2** → `04_leadlag_strategy.py`
5. **Transaction costs** → `05_transaction_costs.py`
6. **Performance** → `06_performance.py`
7. **Visualisations** → `07_visualisations.py`
8. **Report writing** → compile into 5-page PDF
9. **Video** → 60-second presentation to camera

---

## Notes

- All times in UTC
- All prices in USDT
- Annualisation factor for 15-min data: 365.25 × 24 × 4 = 35,064 bars/year
- In-sample / out-of-sample split: ~70/30 by time
