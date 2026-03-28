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

## 10. Pairs Trading Validation (EDA Extension)

### 10.1 Unit Root Tests on Prices
- Augmented Dickey-Fuller (ADF) tests applied to log prices:
  - BTC: p = 0.9500 → non-stationary
  - ETH: p = 0.9162 → non-stationary
  - DOGE: p = 0.8997 → non-stationary

👉 Conclusion:
- All price series are non-stationary (I(1)), satisfying the prerequisite for cointegration analysis

---

### 10.2 Cointegration Tests (Engle-Granger)

- BTC–ETH: p = 0.0065 → cointegrated ✅
- BTC–DOGE: p = 0.6202 → not cointegrated ❌
- ETH–DOGE: p = 0.6190 → not cointegrated ❌

👉 Conclusion:
- Only BTC–ETH exhibits a statistically significant long-run equilibrium relationship
- This pair is selected as the candidate for pairs trading

---

### 10.3 Spread Construction

- Spread defined as:
  spread_t = log(P_BTC) − β · log(P_ETH)

- Estimated hedge ratio:
  - β = 0.6780
  - R² = 0.9451 (strong fit)

👉 Interpretation:
- BTC and ETH share a strong linear relationship
- The spread represents deviations from this equilibrium

---

### 10.4 Stationarity of Spread

- ADF test on BTC–ETH spread:
  - p = 0.0015 → stationary ✅

👉 Conclusion:
- Spread is mean-reverting
- Valid signal for pairs trading

---

### 10.5 Mean Reversion Speed (Half-Life)

- Estimated half-life:
  - 392.4 bars ≈ 98.1 hours (~4 days)

👉 Interpretation:
- Mean reversion exists but is slow
- Strategy should be treated as a **low-frequency / multi-day trading strategy**
- Not suitable for short-horizon trading

---

### 10.6 Z-Score Behaviour

- BTC–ETH:
  - |z| > 1σ: 31.0%
  - |z| > 2σ: 5.4%

👉 Interpretation:
- Sufficient number of trading opportunities
- Signals are not overly frequent, consistent with slow mean reversion

---

### 10.7 Final Pairs Trading Decision

- Selected pair: BTC–ETH
- Cointegration confirmed
- Spread is stationary and mean-reverting
- Half-life indicates slow adjustment dynamics

👉 Final Conclusion:
- BTC–ETH is suitable for pairs trading
- Strategy must be designed with:
  - wider entry thresholds (e.g. ±2σ)
  - longer holding periods
  - lower turnover to mitigate transaction costs

---

## 11. Strategy Update

### Strategy 1 — Breakout / Trend-Following
- Remains unchanged
- Supported by:
  - volatility clustering
  - heavy tails
  - observed price trends

---

### Strategy 2 — Pairs Trading (Final Choice)

- Replaces initial lead-lag idea
- Based on:
  - BTC–ETH cointegration
  - mean-reverting spread

### Key Characteristics
- Mean-reversion strategy
- Lower frequency than breakout
- Longer holding periods (~days)
- Lower turnover expected

---

## 12. Key Insight (Very Important)

The two strategies now capture fundamentally different market behaviours:

- **Breakout Strategy**
  - Exploits momentum and trend persistence
  - Performs better in trending markets

- **Pairs Trading Strategy**
  - Exploits mean reversion between correlated assets
  - Performs better in stable or range-bound markets

👉 This contrast provides a strong conceptual framework for the report.

---

## 13. Updated Direction

The coursework now focuses on comparing:

- Momentum (Breakout)
- Mean Reversion (Pairs Trading)

This allows:
- richer analysis
- stronger interpretation of results
- better discussion of market regimes and strategy suitability