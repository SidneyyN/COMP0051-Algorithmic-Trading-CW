# Volatility Filter — Implementation Plan

This document outlines the step-by-step implementation plan for adding a volatility filter to the pairs trading strategy.

---

## 🎯 Objective

Introduce a volatility filter to:

- avoid trading during unstable regimes
- reduce exposure to structural breaks
- improve robustness of the mean-reversion strategy

---

## 🧩 Recommended Implementation Order

### Step 1 — Compute the Spread

```python
spread = log_btc - beta * log_eth
```

---

### Step 2 — Compute Spread Changes

```python
spread_diff = spread.diff()
```

---

### Step 3 — Compute Rolling Spread Volatility

Use a rolling window (e.g. 100 bars):

```python
spread_vol = spread_diff.rolling(100).std()
```

---

### Step 4 — Compute Reference Volatility

Use a longer window (e.g. 200 bars):

```python
vol_ref = spread_vol.rolling(200).median()
```

---

### Step 5 — Define Volatility Filter Condition

```python
allow_trade = spread_vol < 1.2 * vol_ref
```

Interpretation:

- only trade when current volatility is within a “normal” range

---

### Step 6 — Apply Filter to Entry Logic Only

```python
if position == 0:
    if z > entry_threshold and allow_trade:
        position = -1
    elif z < -entry_threshold and allow_trade:
        position = 1
```

---

### Step 7 — Keep Exit Logic Unchanged

Do NOT apply volatility filter to exits:

- normal exit (z → 0)
- stop-loss
- time stop

---

### Step 8 — Rerun Backtest

Evaluate performance with the filter enabled.

---

### Step 9 — Compare Against Baseline

Check changes in:

- number of trades
- gross PnL
- net PnL
- Sharpe ratio
- max drawdown
- turnover
- win rate

---

### Step 10 — Visual Diagnostics

Plot:

- spread and z-score
- highlight filtered-out entry points

Goal:

- verify whether the filter removes unstable periods

---

## 🚀 Recommended First Configuration

- entry threshold = 3  
- exit threshold = 0  
- min holding = 24 bars  
- max holding = 384 bars  

Volatility filter:

```python
spread_vol = spread.diff().rolling(100).std()
vol_ref = spread_vol.rolling(200).median()
allow_trade = spread_vol < 1.2 * vol_ref
```

---

## 🧠 Expected Effects

- fewer trades
- reduced turnover
- lower drawdown
- potentially improved stability

---

## 📝 Notes

- Keep the first version simple
- Avoid overfitting thresholds
- Use this as a robustness enhancement, not a guarantee of higher returns

---

## 📌 Next Steps

- implement filter
- rerun strategy
- interpret results carefully

