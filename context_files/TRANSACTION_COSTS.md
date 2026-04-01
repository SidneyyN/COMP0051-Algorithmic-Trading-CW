# Transaction Costs Implementation Plan (COMP0051)

This document provides a **step-by-step implementation guide** for incorporating realistic transaction costs into the trading strategies.

---

# 1. Objective

Replace the current **fixed cost model** with more realistic microstructure-based models:

* Roll Model (1984)
* Corwin–Schultz Spread Estimator (2012)

Then evaluate their impact on strategy performance.

---

# 2. Overall Workflow

1. Prepare price data (close, high, low)
2. Compute spreads:

   * Roll model (constant spread)
   * Corwin–Schultz (time-varying spread)
3. Convert spreads into trading costs
4. Integrate costs into backtest
5. Compare results across models

---

# 3. Step 1 — Data Preparation

## Required Inputs

* Close prices → for returns (Roll model)
* High and Low prices → for Corwin–Schultz

## Data Structure

Ensure DataFrame contains:

* `close`
* `high`
* `low`

Also ensure:

* No missing values
* Time index is aligned with positions

---

# 4. Step 2 — Roll Model Implementation

## Idea

Uses negative autocovariance of returns to estimate effective spread.

## Steps

1. Compute log returns
2. Compute covariance between consecutive returns
3. Apply formula:

   spread = 2 * sqrt(-covariance)

## Edge Case

* If covariance ≥ 0 → set spread = 0

## Output

* Single scalar spread per asset

---

# 5. Step 3 — Corwin–Schultz Implementation

## Idea

Uses high-low price ranges to estimate bid-ask spread dynamically.

## Steps

1. Compute:

   * log(high / low)
2. Compute beta:

   * current + lagged squared ranges
3. Compute gamma:

   * two-period high-low range
4. Compute alpha:

   * sqrt(beta) − sqrt(gamma)
5. Convert to spread:

   * nonlinear transformation using exponential

## Output

* Time series of spreads

---

# 6. Step 4 — Convert Spread to Cost

## Key Principle

Each trade pays half the spread per side.

## Steps

1. Compute position change:

   * delta_pos = |position_t − position_{t-1}|

2. Compute cost per time step:

   cost_t = delta_pos × spread_t × price_t / 2

## Important Cases

* Entry (0 → 1): cost = half spread
* Exit (1 → 0): cost = half spread
* Flip (1 → -1): cost = full spread × 2

---

# 7. Step 5 — Integration into Backtest

## Replace Fixed Cost

Old:

* cost = constant per trade

New:

* cost computed dynamically per time step

## Steps

1. Compute spread series
2. Compute delta positions
3. Compute cost per bar
4. Sum total cost
5. Compute net PnL:

   net_pnl = gross_pnl − total_cost

---

# 8. Step 6 — Model Comparison

Run backtest under three cost models:

## Model A — Fixed Cost

* Baseline for comparison

## Model B — Roll Model

* Constant spread
* Microstructure-based estimate

## Model C — Corwin–Schultz

* Time-varying spread
* More realistic

---

# 9. Step 7 — Performance Evaluation

## Metrics to Compare

* Net PnL
* Sharpe ratio
* Sortino ratio
* Max drawdown
* Number of trades
* Total cost

## Analysis Questions

* Does the strategy remain profitable?
* How sensitive is performance to costs?
* Does dynamic spread worsen results?

---

# 10. Step 8 — Turnover Analysis

## Compute

* Total turnover = sum(delta_pos)

## Insight

High turnover strategies:

* More sensitive to transaction costs
* Likely to lose profitability after costs

---

# 11. Step 9 — Cost Sensitivity Analysis (Optional)

## Idea

Scale spreads to test robustness:

* 0.5× spread
* 1× spread
* 1.5× spread
* 2× spread

## Goal

Understand how performance changes under different cost environments.

---

# 12. Step 10 — Reporting Insights

## Key Points to Discuss

* Difference between gross and net performance
* Impact of realistic cost models
* Role of turnover in profitability
* Comparison across strategies (e.g. pairs vs breakout)

## Expected Insight

Strategies with small edges and high frequency:

→ Highly vulnerable to transaction costs

---

# 13. Implementation Checklist

## Core Implementation

* [ ] Compute Roll spread
* [ ] Compute Corwin–Schultz spread
* [ ] Compute delta positions
* [ ] Compute cost per bar
* [ ] Integrate into backtest

## Analysis

* [ ] Compare 3 cost models
* [ ] Evaluate net performance
* [ ] Analyse turnover impact
* [ ] Perform sensitivity analysis (optional)

---

# 14. Final Takeaway

The purpose of this section is not just implementation, but demonstrating that:

> Profitability of trading strategies depends critically on transaction costs and market microstructure effects.

---

**Ready to implement in `transaction_costs.py` and integrate into backtest.**
