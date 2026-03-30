# Pairs Trading Validation Plan (EDA Extension)

This document outlines the additional steps required to validate whether a pairs trading (cointegration-based) strategy is suitable for the selected assets.

The goal is to determine whether a stable, mean-reverting relationship exists between asset pairs, which can be exploited for trading.

---

## 1. Objective

The pairs trading validation stage aims to answer:

1. Are asset prices non-stationary (I(1))?
2. Are any asset pairs cointegrated?
3. Does the constructed spread exhibit mean reversion?
4. Is the speed of mean reversion suitable for trading?

---

## 2. Candidate Pairs

Test the following pairs:
- BTC–ETH (primary candidate)
- BTC–DOGE (secondary)
- ETH–DOGE (optional)

---

## 3. Validation Workflow

Pipeline:

Log Prices → Unit Root Tests → Cointegration Test → Spread Construction → Stationarity Check → Mean Reversion Analysis

---

## Step 1 — Prepare Log Prices

### Objective
Transform prices to log scale for stability and linear modelling.

### Actions
- Compute:
  - log(BTC_close)
  - log(ETH_close)
  - log(DOGE_close)

### Notes
- Log transformation stabilises variance
- Required for cointegration testing

---

## Step 2 — Unit Root Tests on Prices

### Objective
Verify that price series are non-stationary.

### Method
- Apply Augmented Dickey-Fuller (ADF) test to each log price series

### Interpretation

| Result | Meaning |
|------|--------|
| p > 0.05 | Non-stationary (desired) |
| p < 0.05 | Stationary (unexpected) |

### Expected Outcome
- All price series should be non-stationary

---

## Step 3 — Cointegration Test (Engle-Granger)

### Objective
Determine whether a linear combination of two assets is stationary.

### Method
For each pair:
1. Regress:
   log(P₁) = α + β · log(P₂) + ε
2. Apply ADF test on residuals ε

### Interpretation

| Result | Meaning |
|------|--------|
| p < 0.05 | Cointegrated (valid pair) |
| p > 0.05 | Not cointegrated |

### Notes
- This is the most important test
- Only proceed with pairs that pass this test

---

## Step 4 — Hedge Ratio (β) Estimation

### Objective
Estimate the long-run relationship between the two assets.

### Method
- Extract β from the regression in Step 3

### Interpretation
- β determines relative position sizing
- Used to construct the spread

---

## Step 5 — Spread Construction

### Objective
Construct the mean-reverting spread to be traded.

### Formula

spread_t = log(P₁)_t − β · log(P₂)_t

### Actions
- Compute spread for each candidate pair
- Plot spread over time

### What to look for
- Oscillation around a stable mean
- No strong trend

---

## Step 6 — Stationarity Test on Spread

### Objective
Confirm that the spread is stationary.

### Method
- Apply ADF test to the spread

### Interpretation

| Result | Meaning |
|------|--------|
| p < 0.05 | Stationary (mean-reverting) |
| p > 0.05 | Not suitable |

### Expected Outcome
- Spread should be stationary if pair is valid

---

## Step 7 — Mean Reversion Speed (Half-Life)

### Objective
Estimate how quickly the spread reverts to its mean.

### Method
1. Regress:
   Δspread_t = φ · spread_{t−1} + ε
2. Compute half-life:

half_life = ln(2) / |φ|

### Interpretation

| Half-life | Meaning |
|----------|--------|
| Small (e.g. < 20 bars) | Fast reversion (good) |
| Moderate (20–50 bars) | Usable |
| Large (> 100 bars) | Too slow |

---

## Step 8 — Z-Score Behaviour (Optional)

### Objective
Understand trading signal frequency.

### Method
- Standardise spread:
  z_t = (spread_t − mean) / std
- Examine:
  - frequency of |z| > 1
  - frequency of |z| > 2

### Interpretation
- Frequent deviations → more trading opportunities
- Rare deviations → fewer trades

---

## Step 9 — Final Pair Selection

### Objective
Choose the best pair for the trading strategy.

### Selection Criteria
- Strong cointegration (low p-value)
- Stationary spread
- Reasonable half-life
- Stable spread behaviour

### Decision Rule
- Select the pair that best satisfies all criteria
- BTC–ETH is expected to be the strongest candidate

---

## Step 10 — Final Conclusion

### Objective
Summarise findings for report inclusion.

### Output
A concise conclusion stating:
- whether cointegration exists
- which pair is selected
- why it is suitable for trading

### Example Structure
- Price series are non-stationary, consistent with financial time series
- BTC–ETH pair is cointegrated at the 5% level
- Constructed spread is stationary and mean-reverting
- Estimated half-life indicates feasible trading horizon
- Therefore, BTC–ETH is selected for pairs trading

---

## 4. Expected Output Files

### Tables
- cointegration test results
- ADF results
- half-life estimates

### Figures
- spread time series
- optional z-score distribution

---

## 5. Common Pitfalls

- Using raw prices instead of log prices
- Misinterpreting correlation as cointegration
- Ignoring stationarity of the spread
- Selecting a pair without statistical validation
- Using a spread that trends instead of mean-reverts

---

## 6. Deliverable of This Stage

At the end of this validation:

- One pair is selected (likely BTC–ETH)
- A stationary spread is confirmed
- Mean reversion speed is quantified
- The strategy foundation is fully justified

This output feeds directly into:
- `05_pairs_strategy.py`
- Report Section 2 (Strategy description)