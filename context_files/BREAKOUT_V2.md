# Breakout Strategy Refinements — Threshold and Volatility Filter

This note lists the **step-by-step implementation** of the next two fixes for the BTC range-reversion strategy:

1. **Threshold filter**
2. **Improved volatility filter**

The goal is to apply these changes on a **copy** of the current code so the original working version remains intact.

---

# 1. Make a Safe Copy First

Before changing anything:

- duplicate the current script
- rename it clearly, for example:
  - `btc_breakout_reversion_v2.py`
  - or `btc_breakout_threshold_volfilter.py`

This keeps the current working version available for comparison.

---

# 2. Fix 1 — Add a Threshold to Entry Signals

## Purpose

Right now, a signal triggers as soon as price moves just beyond the Donchian band.

That can create trades from very small deviations that may just be noise.

A threshold makes the rule stricter:

- only **buy** if price falls sufficiently below the lower band
- only **short** if price rises sufficiently above the upper band

This helps target **more extreme overextensions**.

---

## Step 2.1 — Add a New Configuration Parameter

In the configuration section, add a threshold parameter.

Suggested starting value:

```python
THRESHOLD = 0.001   # 0.1% = 10 bps
```

You can later test alternatives such as:

- `0.0005` = 5 bps
- `0.0010` = 10 bps
- `0.0020` = 20 bps

---

## Step 2.2 — Pass the Threshold into `compute_signals`

Update the function signature so the threshold is an explicit input.

Example:

```python
def compute_signals(df, N, vol_window, vol_quantile, threshold):
```

Then update the call in `main()` accordingly.

---

## Step 2.3 — Replace the Current Raw Entry Rules

Current logic:

```python
df['long_entry'] = df['BTC_close'] < df['lower_band']
df['short_entry'] = df['BTC_close'] > df['upper_band']
```

Replace with threshold-adjusted rules:

```python
df['long_entry'] = df['BTC_close'] < df['lower_band'] * (1 - threshold)
df['short_entry'] = df['BTC_close'] > df['upper_band'] * (1 + threshold)
```

---

## Step 2.4 — Keep Everything Else the Same

For this test, do **not** change:

- holding logic
- transaction cost logic
- capital
- lookback window `N`
- max holding
- in-sample / out-of-sample split

This way, any performance difference can be attributed mainly to the threshold.

---

## Step 2.5 — Run and Compare

After applying the threshold fix, compare against the current baseline on:

- total net PnL
- Sharpe ratio
- max drawdown
- number of trades
- win rate

### What to look for

Expected effects:

- fewer trades
- lower costs
- potentially better signal quality

Possible downside:

- threshold too large may remove profitable reversals

---

# 3. Fix 2 — Refine the Volatility Filter

## Purpose

The current code only trades when rolling volatility is **above** a threshold:

```python
rolling_vol > threshold
```

That may be too crude for a mean-reversion strategy.

For range reversion, the best regime is often not:

- very low volatility, where moves are too small
- nor extreme volatility, where trends can overpower reversion

A better idea is to trade only in a **middle volatility range**.

---

## Step 3.1 — Decide on the New Volatility Logic

Instead of one cutoff, use two:

- a **lower volatility bound**
- an **upper volatility bound**

So the strategy only trades when volatility is inside a band.

Conceptually:

```python
low_vol_threshold < rolling_vol < high_vol_threshold
```

---

## Step 3.2 — Add Two New Configuration Parameters

In the configuration section, replace the single quantile with two quantiles.

Suggested starting point:

```python
VOL_Q_LOW = 0.2
VOL_Q_HIGH = 0.8
```

This means:

- do not trade in the lowest 20% of volatility regimes
- do not trade in the highest 20% of volatility regimes
- trade only in the middle 60%

You can later test nearby choices if needed.

---

## Step 3.3 — Update the `compute_signals` Function Signature

Modify the function so it accepts both volatility quantiles.

Example:

```python
def compute_signals(df, N, vol_window, vol_q_low, vol_q_high, threshold):
```

---

## Step 3.4 — Compute the Two Volatility Thresholds Using In-Sample Data Only

Keep using the in-sample section to estimate thresholds.

That avoids leaking out-of-sample information into the filter.

Conceptually:

```python
is_mask = df.index <= IS_END
low_vol_threshold = df.loc[is_mask, 'rolling_vol'].quantile(vol_q_low)
high_vol_threshold = df.loc[is_mask, 'rolling_vol'].quantile(vol_q_high)
```

---

## Step 3.5 — Replace the Old Volatility Filter Rule

Current rule:

```python
df['vol_filter'] = df['rolling_vol'] > vol_threshold
```

Replace with a bounded rule:

```python
df['vol_filter'] = (
    (df['rolling_vol'] > low_vol_threshold) &
    (df['rolling_vol'] < high_vol_threshold)
)
```

This keeps trades only in the middle-volatility region.

---

## Step 3.6 — Apply the New Filter to Both Long and Short Entries

Keep the current structure:

```python
df['long_entry'] = df['long_entry'] & df['vol_filter']
df['short_entry'] = df['short_entry'] & df['vol_filter']
```

No further logic change is needed at this stage.

---

# 4. Recommended Order of Testing

To keep the experiment clean, apply the fixes in this order:

## Test A — Threshold only

Change only:

- `THRESHOLD`

Keep volatility filter unchanged.

This tells you whether the threshold alone improves the strategy.

---

## Test B — Threshold + refined volatility filter

Then apply:

- threshold
- middle-range volatility filter

This tells you whether the additional volatility refinement improves robustness.

---

# 5. What to Record After Each Test

For each version, save:

- parameter values used
- in-sample metrics
- out-of-sample metrics
- number of trades
- total costs

A simple comparison table is enough.

Suggested columns:

- version name
- threshold
- vol filter rule
- IS net PnL
- OOS net PnL
- IS Sharpe
- OOS Sharpe
- trades
- total costs

---

# 6. Interpretation Guide

## If threshold improves performance

Interpretation:

> weak band breaks were mostly noise, while stronger deviations contain better mean-reversion signals.

## If the refined volatility filter improves performance

Interpretation:

> the strategy works best in moderate-volatility regimes, while both quiet and extreme markets reduce reliability.

## If neither fix improves performance

Interpretation:

> the current baseline may already be close to the best trade-off between frequency and signal quality.

That is still a useful finding.

---

# 7. Practical Advice

While testing these fixes:

- change one thing at a time
- keep notes of each run
- do not tune excessively to the out-of-sample period
- prefer simple, economically defensible rules

The goal is not to find the most overfit specification, but to show a **reasonable and robust refinement process**.

---

# 8. Summary Checklist

## Threshold fix
- [ ] copy the current script
- [ ] add `THRESHOLD` to config
- [ ] pass threshold into `compute_signals`
- [ ] replace entry rules with threshold-adjusted rules
- [ ] rerun and compare results

## Volatility filter fix
- [ ] add `VOL_Q_LOW` and `VOL_Q_HIGH`
- [ ] compute low and high volatility cutoffs from in-sample data
- [ ] replace one-sided filter with middle-range filter
- [ ] rerun and compare results

---

# 9. Suggested First Parameters

A sensible first test is:

```python
THRESHOLD = 0.001
VOL_Q_LOW = 0.2
VOL_Q_HIGH = 0.8
```

This is strict enough to remove some noise, but not so aggressive that it kills most signals.
