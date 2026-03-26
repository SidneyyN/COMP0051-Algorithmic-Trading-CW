# Coursework Progress Notes (Draft for Report)

This document summarises the current progress of the COMP0051 coursework and the key insights obtained so far. It is intended as a quick reference when writing the final report.

---

## 1. Data and Preprocessing

### Dataset
- Assets: BTC/USDT, ETH/USDT, DOGE/USDT
- Frequency: 15-minute bars
- Period: 1 Sep 2025 – 28 Feb 2026 (~6 months)
- Total observations: ~17,375 bars per asset

### Data Pipeline
- Raw OHLCV data downloaded from Binance
- Cleaned and stored in `.parquet` format
- Timestamps aligned across all assets
- No missing or irregular bars detected

### Returns
- Simple returns computed using close prices
- Excess returns computed using Fed Funds rate (FRED)
- Risk-free rate is negligible at 15-minute frequency but included for completeness

### Key Takeaway
- Dataset is clean, high-frequency, and suitable for backtesting

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Distributional Properties
- All assets exhibit:
  - High volatility
  - Heavy tails (excess kurtosis >> 0)
- DOGE is the most volatile and extreme:
  - Very high kurtosis and large tail events

👉 Implication:
- Risk is highly non-Gaussian
- Extreme moves must be accounted for

---

### 2.2 Volatility Behaviour
- Strong volatility clustering observed:
  - ACF of |returns| ≈ 0.3+
- Volatility varies significantly over time

👉 Implication:
- Volatility scaling is justified
- Strategy performance likely regime-dependent

---

### 2.3 Correlation Structure
- Strong contemporaneous correlation:
  - BTC–ETH ≈ 0.87
  - BTC–DOGE ≈ 0.77
- Correlations are high but not perfect

👉 Implication:
- Assets share common drivers
- Potential for relative-value / spread-based strategies

---

### 2.4 Lead-Lag Analysis

#### Cross-Correlation
- Peak correlation occurs at lag 0
- Near-zero correlation at short positive lags

#### Conditional Returns
- ETH and DOGE do not consistently follow BTC moves
- Forward returns are small and statistically insignificant

#### Granger Causality
- Some statistical significance at higher lags
- However, economic magnitude is weak

👉 Final Conclusion:
- No meaningful short-term lead-lag structure
- Effect size too small to be tradable after costs

---

### 2.5 Breakout / Momentum Evidence
- Price series show sustained directional moves
- Volatility clustering supports regime persistence
- Returns exhibit heavy tails and clustering

👉 Conclusion:
- Breakout / trend-following strategy is supported by the data

---

## 3. Strategy Decisions

### Strategy 1 — Breakout / Trend-Following
- Strongly supported by EDA
- Will use:
  - Donchian-style breakout signals
  - ATR / volatility filters
  - Volatility-scaled position sizing

---

### Strategy 2 — Initial Idea: Lead-Lag (Rejected)

- Hypothesis: BTC leads ETH/DOGE
- EDA findings:
  - No meaningful lagged correlation
  - No economically significant predictive power
- Conclusion:
  - Not viable after transaction costs
  - Strategy will NOT be implemented

---

### Strategy 2 — Revised Plan: Pairs Trading (Cointegration)

- Motivation:
  - Strong correlation between BTC and ETH
  - Potential for mean-reverting spread
- Approach:
  - Test for cointegration
  - Construct spread and trade deviations from equilibrium

👉 This provides contrast:
- Breakout = momentum
- Pairs = mean reversion

---

## 4. Position Sizing Approach

- Use volatility scaling:
  - Lower weight for more volatile assets
  - Higher weight for more stable assets
- Incorporate signal strength where applicable
- Enforce gross exposure constraint:
  - Total exposure ≤ $100,000

👉 Avoids instability of full MVO while remaining realistic

---

## 5. Transaction Costs Plan

- Estimate slippage using:
  - Roll model
- Perform sensitivity analysis across:
  - 1–20 basis points
- Evaluate:
  - Impact on Sharpe ratio
  - Strategy robustness

👉 Key goal:
- Ensure strategies remain profitable after costs

---

## 6. Performance Evaluation Plan

Metrics to compute:
- Gross and net PnL
- Sharpe ratio (annualised)
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Turnover
- Average holding horizon

👉 Both strategies will be evaluated consistently

---

## 7. Key Insights So Far

- Crypto returns are highly volatile and heavy-tailed
- Volatility clustering is strong → supports dynamic risk control
- Assets are highly correlated but not predictively linked
- Lead-lag effects are statistically weak and economically insignificant
- Breakout strategy is strongly supported
- Pairs trading is a promising alternative second strategy

---

## 8. Next Steps

1. Implement breakout strategy (`04_breakout_strategy.py`)
2. Implement pairs trading strategy (`05_pairs_strategy.py`)
3. Estimate transaction costs (`06_transaction_costs.py`)
4. Compute performance metrics (`07_performance.py`)
5. Generate final plots and tables for report
6. Write 5-page report and record video

---

## 9. Overall Direction

The project focuses on comparing two distinct trading paradigms:

- **Momentum (Breakout Strategy)**
- **Mean Reversion (Pairs Trading)**

This allows for a clear and well-motivated analysis of how different strategies perform under varying market conditions.