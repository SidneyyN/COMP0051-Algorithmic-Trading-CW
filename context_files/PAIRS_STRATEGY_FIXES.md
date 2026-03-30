# Pairs Trading Strategy — Step-by-Step Fixes & Improvements

This document summarises the key fixes and improvements applied to the BTC–ETH pairs trading strategy, based on backtest results and diagnostic analysis.

---

## 🎯 Objective

Improve the profitability and robustness of the pairs trading strategy by:

- aligning trading horizon with half-life
- reducing unnecessary trades
- lowering transaction cost impact
- improving signal quality

---

## 🚨 Key Problems Identified

From initial results:

- Holding period too short vs half-life (~4 days)
- Excessive number of trades
- High transaction costs destroying profitability
- Strategy trading noise instead of true mean reversion
- Weak out-of-sample performance

---

## ✅ Step 1 — Increase Exit Threshold (Reduce Early Exit)

### Fix
```python
exit when z crosses 0
```

---

## ✅ Step 2 — Increase Entry Threshold (Stronger Signals)

```python
entry = 3
```

---

## ✅ Step 3 — Align Holding Period with Half-Life

- Target holding: multi-day horizon (~100–200 bars minimum)

---

## ✅ Step 4 — Enforce Time Stop

```python
max_holding = 384
```

---

## ✅ Step 5 — Add Minimum Holding Period

```python
min_hold = 24
```

---

## ✅ Step 6 — Reduce Trading Frequency

Optional:
```python
cooldown = 20
```

---

## ✅ Step 7 — Account for Transaction Costs

- Strategy profitable at 1 bps
- Not profitable at 5 bps+

---

## ✅ Step 8 — Improve Signal Quality

- Only trade extreme deviations (z ≥ 3)

---

## ✅ Step 9 — Scale Position by Signal Strength (Optional)

```python
position_size ∝ |z|
position_size = base_size * min(abs(z) / entry_threshold, 2)
```

---

## ✅ Final Recommended Configuration

```python
entry = 3
exit = 0
stop = 3
min_hold = 24
max_holding = 384
```

---

## 🧠 Key Takeaways

- Mean reversion works (gross positive)
- Costs are the main issue
- Better alignment with half-life improves performance
- Fewer, stronger trades is the key

---

## 🚀 Next Steps

- Run final backtest
- Generate plots
- Write results section

