# Breakout Strategy — Step-by-Step Implementation Guide

This document sets out a clean implementation plan for the breakout strategy, so it can be built, tested, and compared against the existing pairs trading strategy.

It follows directly from the current project status: the pairs strategy captured the **mean-reversion** side of the story, while the breakout strategy will capture the **momentum** side. As noted in the project progress summary, the breakout strategy is expected to be shorter-horizon, momentum-based, and more likely to benefit from volatility. fileciteturn0file0L66-L72

---

## 1. Strategy Objective

The goal of the breakout strategy is to trade sustained price moves after the market breaks out of a recent trading range.

This gives a useful contrast with the pairs trading strategy:

- **Pairs trading**: mean reversion, lower-frequency, regime-sensitive, improved by volatility filtering
- **Breakout trading**: momentum, shorter-horizon, expected to work better in stronger directional moves

This comparison supports the overall project narrative of **momentum versus mean reversion under different market regimes**. fileciteturn0file0L74-L82

---

## 2. Choose the Market and Input Series

### Step 2.1 — Select asset(s)

Start with one asset at a time.

Recommended order:

1. BTC
2. ETH
3. DOGE (optional later, if time allows)

BTC is usually the best first choice because it is the most liquid and tends to produce cleaner trend behaviour.

### Step 2.2 — Choose the price series

Use the same 15-minute cleaned dataset already prepared for the project.

For each asset, create a dataframe containing at least:

- timestamp
- close price
- log return

Example columns:

```python
['timestamp', 'close', 'log_return']
```

---

## 3. Define the Breakout Signal

The simplest and most defensible implementation is a **Donchian channel breakout**.

### Step 3.1 — Choose a lookback window

Let `N` be the breakout lookback window.

Suggested values to test:

- `N = 50`
- `N = 100`
- `N = 200`

Interpretation:

- `50`: faster, more sensitive, more trades
- `100`: balanced starting point
- `200`: slower, more robust, fewer trades

A sensible first baseline is:

```python
N = 100
```

### Step 3.2 — Compute breakout bands

For each bar, calculate:

- **Upper band** = rolling maximum of close over past `N` bars
- **Lower band** = rolling minimum of close over past `N` bars

Important: use **lagged bands** when generating signals to avoid lookahead bias.

Example:

```python
df['upper_band'] = df['close'].rolling(N).max()
df['lower_band'] = df['close'].rolling(N).min()

df['upper_band_lag'] = df['upper_band'].shift(1)
df['lower_band_lag'] = df['lower_band'].shift(1)
```

---

## 4. Generate Entry Signals

### Step 4.1 — Long breakout

Enter a **long** position when the current close breaks above the previous upper band:

```python
df['long_entry'] = df['close'] > df['upper_band_lag']
```

### Step 4.2 — Short breakout

Enter a **short** position when the current close breaks below the previous lower band:

```python
df['short_entry'] = df['close'] < df['lower_band_lag']
```

### Step 4.3 — Avoid lookahead bias

Do **not** compare price with the same-bar rolling max/min if that max/min includes the current bar.

Correct logic:

- use `.rolling(N)` to compute the range
- then `.shift(1)` before generating the signal

---

## 5. Add a Volatility Filter

This is strongly recommended because volatility filtering already improved robustness in the pairs trading strategy by reducing bad trades and drawdown. fileciteturn0file0L40-L48

For breakout trading, the filter should be used in the **opposite direction** from mean reversion:

- pairs trading preferred calmer, more normal regimes
- breakout trading should only trade when volatility is sufficiently high

### Step 5.1 — Compute rolling volatility

Example using rolling standard deviation of returns:

```python
vol_window = 50
df['rolling_vol'] = df['log_return'].rolling(vol_window).std()
```

### Step 5.2 — Define the threshold

A simple choice is to use a percentile threshold:

```python
vol_threshold = df['rolling_vol'].quantile(0.6)
```

### Step 5.3 — Only allow entries in high-volatility regimes

```python
df['vol_filter'] = df['rolling_vol'] > vol_threshold
```

Then update the entry conditions:

```python
df['long_entry'] = (df['close'] > df['upper_band_lag']) & df['vol_filter']
df['short_entry'] = (df['close'] < df['lower_band_lag']) & df['vol_filter']
```

---

## 6. Define Position Management Rules

Keep this part simple and fixed-size. The earlier pairs trading experiments showed that scaling position size can amplify costs and worsen robustness, so the breakout baseline should avoid signal-based scaling. fileciteturn0file0L31-L39

### Step 6.1 — Position states

Use:

- `+1` for long
- `-1` for short
- `0` for flat

### Step 6.2 — Entry rules

- If flat and `long_entry == True`, enter long
- If flat and `short_entry == True`, enter short

### Step 6.3 — Exit rules

Recommended baseline:

- Exit when the **opposite breakout signal** appears
- Also impose a **maximum holding period** to prevent stale trades

This gives both responsiveness and protection against being trapped in non-trending conditions.

Suggested first version:

- Opposite signal exit
- `max_hold_bars = 40`

### Step 6.4 — Optional time stop

If long:

- exit if held for more than `max_hold_bars`

If short:

- exit if held for more than `max_hold_bars`

This is helpful because 15-minute strategies can otherwise remain stuck through sideways price action.

---

## 7. Implement the Trading Loop

A simple loop-based implementation is easiest to debug.

### Step 7.1 — Initialise variables

Track:

- current position
- entry price
- holding duration
- trade list
- per-bar strategy return

Example:

```python
position = 0
hold_bars = 0
entry_price = None
strategy_returns = []
trades = []
```

### Step 7.2 — Iterate through the dataframe row by row

For each bar:

1. Read the signals (`long_entry`, `short_entry`)
2. Update existing position logic
3. Check exit rules
4. Check entry rules if flat
5. Store the resulting position

Pseudo-logic:

```python
for t in range(1, len(df)):
    long_signal = df.loc[t, 'long_entry']
    short_signal = df.loc[t, 'short_entry']

    if position == 1:
        hold_bars += 1
        if short_signal or hold_bars >= max_hold_bars:
            position = 0
            hold_bars = 0

    elif position == -1:
        hold_bars += 1
        if long_signal or hold_bars >= max_hold_bars:
            position = 0
            hold_bars = 0

    if position == 0:
        if long_signal:
            position = 1
            hold_bars = 0
        elif short_signal:
            position = -1
            hold_bars = 0
```

---

## 8. Compute Strategy Returns

### Step 8.1 — Use next-bar return convention

To avoid lookahead bias, the position taken at time `t` should earn the return from `t` to `t+1`.

Example:

```python
df['position_lag'] = df['position'].shift(1).fillna(0)
df['strategy_return_gross'] = df['position_lag'] * df['log_return']
```

### Step 8.2 — Convert to PnL

If using dollar notional `capital_per_trade`:

```python
df['strategy_pnl_gross'] = capital_per_trade * df['strategy_return_gross']
```

---

## 9. Add Transaction Costs

This is essential. One of the strongest findings so far is that positive gross performance can disappear once realistic costs are included. fileciteturn0file0L15-L29

### Step 9.1 — Detect turnover

A trade happens when position changes.

```python
df['turnover'] = (df['position'] != df['position'].shift(1)).astype(int)
```

If you want more detail:

- `0 → 1` = entry
- `1 → 0` = exit
- `1 → -1` = reversal, which may effectively cost as two trades

### Step 9.2 — Apply per-trade cost

If `cost_per_trade` is fixed:

```python
df['transaction_cost'] = df['turnover'] * cost_per_trade
```

Then:

```python
df['strategy_pnl_net'] = df['strategy_pnl_gross'] - df['transaction_cost']
```

### Step 9.3 — Be consistent with the pairs trading backtest

Use the **same cost assumptions** as in the pairs strategy wherever possible, so the comparison is fair.

---

## 10. Compute Performance Metrics

Once the net PnL series is available, compute the same metrics used for pairs trading.

### Step 10.1 — Core metrics

At minimum:

- total gross PnL
- total net PnL
- total transaction cost
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- number of trades
- win rate
- average holding period

### Step 10.2 — Why this matters

You want the breakout results to be directly comparable with the pairs trading results, so keep the reporting format consistent.

---

## 11. Run a Baseline Experiment First

Before tuning, run one clean baseline.

Recommended first baseline:

- Asset: BTC
- Lookback: `N = 100`
- Volatility window: `50`
- Volatility filter: trade only when rolling volatility is above the 60th percentile
- Position size: fixed
- Exit: opposite signal or `max_hold_bars = 40`
- Costs: same as pairs trading

This first run is not about perfection. It is about establishing:

1. whether the strategy trades too often
2. whether costs dominate
3. whether the volatility filter helps
4. whether breakout behaviour is meaningfully different from pairs trading

---

## 12. Then Tune Systematically

After the baseline works, vary one thing at a time.

### Step 12.1 — Tune lookback window

Test:

- `N = 50`
- `N = 100`
- `N = 200`

Expected effect:

- smaller `N` → more trades, more noise, higher cost
- larger `N` → fewer trades, cleaner signals, slower response

### Step 12.2 — Tune volatility threshold

Test thresholds such as:

- 50th percentile
- 60th percentile
- 70th percentile

Expected effect:

- lower threshold → more trades
- higher threshold → fewer but more selective trades

### Step 12.3 — Tune holding limit

Test:

- 20 bars
- 40 bars
- 80 bars

Expected effect:

- shorter holding → faster turnover, possibly higher costs
- longer holding → more trend capture, but more exposure to reversals

---

## 13. Produce Diagnostic Plots

These will help both debugging and report writing.

### Step 13.1 — Price with breakout bands

Plot:

- close price
- upper band
- lower band
- long/short entry markers

This checks whether the signal is behaving logically.

### Step 13.2 — Equity curve

Plot cumulative:

- gross PnL
- net PnL

This shows whether transaction costs destroy the edge.

### Step 13.3 — Trade distribution

Summarise:

- number of longs vs shorts
- holding period histogram
- PnL per trade

### Step 13.4 — Filter comparison

Compare:

- breakout without vol filter
- breakout with vol filter

This will likely become one of the most useful report figures.

---

## 14. Compare Against Pairs Trading

Once breakout results are ready, structure the comparison clearly.

### Step 14.1 — Strategy comparison table

Include:

- strategy type
- trading frequency
- average holding period
- net PnL
- Sharpe
- drawdown
- cost sensitivity
- best regime

### Step 14.2 — Expected qualitative contrast

Likely conclusion:

- **Pairs trading** works only when mean reversion is stable and costs are controlled
- **Breakout trading** may benefit more from directional and volatile periods
- Neither strategy is universally best; effectiveness depends on regime

That would match the current project narrative very well. fileciteturn0file0L74-L97

---

## 15. Recommended Build Order

To keep implementation manageable, follow this exact order:

### Phase 1 — Minimal working strategy

1. Load one asset dataframe
2. Compute rolling breakout bands
3. Generate long/short signals
4. Build basic position loop
5. Compute gross returns

### Phase 2 — Make it realistic

6. Add transaction costs
7. Add volatility filter
8. Add holding limit
9. Compute metrics

### Phase 3 — Analyse and compare

10. Generate diagnostic plots
11. Run parameter sensitivity tests
12. Compare against pairs trading
13. Write up the conclusions

---

## 16. Recommended First Coding Target

A good first coding target is:

> Write a single backtest function that takes a dataframe and parameters, and returns both a results dataframe and a summary dictionary.

For example:

```python
def backtest_breakout(df, lookback=100, vol_window=50, vol_quantile=0.6,
                      max_hold_bars=40, cost_per_trade=50):
    ...
    return results_df, summary
```

This will make later tuning much easier.

---

## 17. Final Implementation Goal

By the end of this stage, the project should have:

- one completed mean-reversion strategy (pairs trading)
- one completed momentum strategy (breakout trading)
- consistent backtesting assumptions
- performance comparison under realistic transaction costs
- a clear regime-based interpretation

That would give the coursework a strong and balanced final story.
