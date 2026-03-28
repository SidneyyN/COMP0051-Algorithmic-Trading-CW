# Pairs Trading Strategy — Step-by-Step Plan

This file lays out a clear implementation plan for `pairs_strategy`, based on the validated BTC–ETH cointegration result.

---

## Objective

Build a **mean-reversion pairs trading strategy** on **BTC–ETH** using the stationary spread identified from the Engle–Granger test.

The strategy should:

- compute the spread using the estimated hedge ratio
- generate rolling z-score signals
- apply entry / exit / stop-loss rules
- size positions using the hedge ratio
- backtest the strategy without lookahead bias
- produce plots and performance metrics

---

## Pair and Spread Definition

Use:

- **BTC** as asset 1
- **ETH** as asset 2
- **hedge ratio**: \( \beta = 0.6780 \)

Define the spread as:

\[
\text{spread}_t = \log(P^{BTC}_t) - \beta \log(P^{ETH}_t)
\]

Why this works:

- BTC and ETH log prices are individually non-stationary
- their linear combination is stationary
- deviations in the spread should tend to revert toward equilibrium

---

## High-Level Strategy Logic

The strategy is based on the idea that when the spread moves too far away from its recent mean, it is likely to revert.

### Signal intuition

- If spread is **too high**, BTC is rich relative to ETH  
  → short BTC, long ETH
- If spread is **too low**, BTC is cheap relative to ETH  
  → long BTC, short ETH

We detect this using a rolling z-score.

---

## Step 1 — Prepare Inputs

The script should begin by loading aligned BTC and ETH data.

Required inputs:

- timestamp
- BTC close
- ETH close

Preprocessing steps:

1. sort by timestamp
2. align both series on common timestamps
3. compute log prices:
   - `log_btc = np.log(btc_close)`
   - `log_eth = np.log(eth_close)`
4. verify no missing values remain

---

## Step 2 — Compute the Spread

Using the fixed hedge ratio from the cointegration regression:

\[
\text{spread}_t = \log(BTC_t) - 0.6780 \cdot \log(ETH_t)
\]

Implementation note:

- first version should use a **fixed beta**
- rolling beta can be considered later as an optional enhancement

---

## Step 3 — Compute Rolling Z-Score

The trading signal should be based on a **rolling mean** and **rolling standard deviation** of the spread.

### Formula

\[
z_t = \frac{\text{spread}_t - \mu_t}{\sigma_t}
\]

where:

- \(\mu_t\) = rolling mean of spread
- \(\sigma_t\) = rolling standard deviation of spread

### Recommended first choice

- rolling window = **100 bars**

Since the data uses 15-minute bars:

- 100 bars = 25 hours

This gives an adaptive signal while avoiding full-sample lookahead bias.

### Implementation details

For each time \(t\):

- compute rolling mean using only data up to time \(t\)
- compute rolling std using only data up to time \(t\)
- compute z-score only when enough observations exist

---

## Step 4 — Define Trading Rules

Because the half-life is quite long, this should be treated as a **lower-frequency, multi-day mean reversion strategy**.

Use **wider thresholds** and avoid excessive turnover.

### Entry rules

- enter **short spread** when `z > +2`
- enter **long spread** when `z < -2`

Interpretation:

#### Short spread
If `z > 2`, spread is high:

- short BTC
- long ETH

#### Long spread
If `z < -2`, spread is low:

- long BTC
- short ETH

---

## Step 5 — Define Exit Rules

A trade should close once the spread has materially reverted.

### Recommended exit rule

- close long spread when `z > -0.5`
- close short spread when `z < +0.5`

This is slightly more conservative than waiting for an exact zero crossing.

### Why this is sensible

- reduces the chance of over-waiting
- locks in mean-reversion profits earlier
- helps reduce round-trip risk in volatile markets

---

## Step 6 — Add Risk Controls

Pairs trades can still fail badly in crypto, so risk controls are necessary.

### Stop-loss rule

Force exit if:

- `|z| > 3`

This handles situations where the spread keeps diverging instead of reverting.

### Time stop

Since half-life is about 392 bars, set a maximum holding period around that scale.

Recommended first choice:

- `max_holding = 384 bars`

This is approximately:

- 4 days
- close to the estimated half-life

If a trade has not reverted by then, exit and wait for a cleaner setup.

---

## Step 7 — Position Sizing

The strategy should be **hedged** using the estimated beta.

### Core idea

For a long spread position:

- long 1 unit of BTC exposure
- short \(\beta\) units of ETH exposure

For a short spread position:

- short 1 unit of BTC exposure
- long \(\beta\) units of ETH exposure

### Dollar-neutral implementation

Let total gross capital be \(C\), for example:

- `capital = 100000`

Then allocate:

\[
\text{btc_notional} = \frac{C}{1 + \beta}
\]

\[
\text{eth_notional} = \beta \cdot \text{btc_notional}
\]

This keeps gross exposure controlled and makes the pair approximately neutral.

### Position directions

#### Long spread
- BTC position = `+btc_notional`
- ETH position = `-eth_notional`

#### Short spread
- BTC position = `-btc_notional`
- ETH position = `+eth_notional`

---

## Step 8 — Build the Signal State Machine

Track a position state variable:

- `0` = flat
- `+1` = long spread
- `-1` = short spread

At each bar:

1. check current z-score
2. if flat, evaluate entry
3. if in trade, evaluate:
   - normal exit
   - stop-loss
   - time stop
4. store the current state

This should be implemented carefully so the signal is easy to audit and plot.

---

## Step 9 — Avoid Lookahead Bias

This is critical.

Trades must be based only on information known at the time.

### Rules to enforce

- use rolling statistics computed from current and past data only
- positions should be applied from the **next bar onward**
- PnL must use `position.shift(1)` or equivalent

This ensures the backtest is realistic.

---

## Step 10 — Compute Returns and PnL

There are two good implementation routes.

### Option A — Spread-return approximation

Compute:

\[
\Delta \text{spread}_t = \Delta \log(BTC_t) - \beta \Delta \log(ETH_t)
\]

Then:

\[
\text{strategy return}_t = \text{position}_{t-1} \cdot \Delta \text{spread}_t
\]

This is simple and good for a first backtest.

### Option B — Leg-level PnL

Compute PnL separately for BTC and ETH using their allocated notionals.

This is more realistic and preferred for the final version.

#### Leg-level outline

At time \(t\):

- BTC PnL = previous BTC notional × BTC return
- ETH PnL = previous ETH notional × ETH return
- total PnL = BTC PnL + ETH PnL

This is the better final implementation because it directly reflects the two-leg trade.

---

## Step 11 — Include Transaction Costs

Transaction costs should not be ignored.

Costs should be applied whenever the position changes.

### When costs occur

- opening a trade
- closing a trade
- flipping from long spread to short spread or vice versa

### First implementation idea

Use a simple cost model such as:

- cost in basis points × traded notional

Later, this can be linked to the Roll model estimates developed elsewhere in the project.

---

## Step 12 — Store Trade Information

The strategy should record useful diagnostics for analysis.

Suggested trade-level fields:

- entry time
- exit time
- trade direction
- entry z-score
- exit z-score
- holding period
- gross PnL
- net PnL
- exit reason:
  - mean reversion
  - stop-loss
  - time stop

This will make the report much stronger.

---

## Step 13 — Produce Core Plots

The script should generate at least these plots:

### 1. Spread plot
- spread over time
- rolling mean
- upper and lower entry bands

### 2. Z-score plot
- z-score over time
- entry thresholds at ±2
- exit thresholds at ±0.5
- stop-loss thresholds at ±3

### 3. Equity curve
- cumulative gross PnL
- cumulative net PnL

### 4. Optional trade markers
- mark entries and exits on the z-score or spread chart

These will make it much easier to explain the strategy in the report and video.

---

## Step 14 — Compute Performance Metrics

The script should report:

- total gross PnL
- total net PnL
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- turnover
- number of trades
- average holding period
- win rate
- average trade PnL

These metrics should match the broader project evaluation framework.

---

## Step 15 — Suggested File Structure

A clean structure for `05_pairs_strategy.py` could be:

```python
load_data()
compute_spread()
compute_zscore()
generate_positions()
compute_leg_pnl()
apply_transaction_costs()
extract_trades()
compute_performance_metrics()
make_plots()
save_outputs()
```

Recommended outputs:

- CSV of backtest results
- CSV of trade log
- plots saved to report folder
- summary metrics table

---

## Step 16 — Recommended First Version

To keep the first implementation manageable, start with:

- fixed beta = 0.6780
- rolling window = 100
- entry at ±2
- exit at ±0.5
- stop-loss at ±3
- max holding = 384 bars
- fixed capital = 100000
- simple proportional transaction cost model

This version is already strong enough for coursework.

---

## Step 17 — Optional Enhancements

If time allows, improve the strategy with:

### 1. Rolling beta
Estimate the hedge ratio dynamically over time.

### 2. Volatility scaling
Reduce position size when spread volatility is high.

### 3. Volatility filter
Only enter trades when the spread is not in an extreme volatility regime.

### 4. Signal-strength weighting
Scale exposure by how far the z-score is from the entry threshold.

### 5. Robustness checks
Compare different parameter choices:

- rolling window: 100 vs 200
- entry threshold: 1.5 vs 2 vs 2.5
- exit threshold: 0 vs 0.5
- max holding: 2 days vs 4 days

These would strengthen the analysis section.

---

## Step 18 — Interpretation for the Report

The final strategy description can be summarised as follows:

> We implement a mean-reversion pairs trading strategy using the cointegrated BTC–ETH pair. The spread is defined using the OLS hedge ratio estimated from log prices. Trading signals are generated from rolling z-scores of the spread. Positions are entered when the spread deviates beyond ±2 standard deviations and closed when it reverts toward equilibrium. Additional stop-loss and time-stop rules are applied to control divergence risk. Positions are sized according to the hedge ratio to maintain a hedged two-leg exposure.

---

## Immediate Coding Plan

Implement in this order:

1. load BTC and ETH data
2. compute log prices
3. compute spread using fixed beta
4. compute rolling mean / std / z-score
5. generate position states
6. add stop-loss and time-stop
7. compute leg-level PnL
8. subtract transaction costs
9. extract trade log
10. compute metrics
11. generate plots
12. save outputs

---

## Final Note

This strategy should be presented as a **low-frequency mean-reversion complement** to the faster breakout strategy.

That contrast is one of the strongest parts of the overall coursework narrative:

- **Breakout strategy** captures trend continuation
- **Pairs strategy** captures relative-value mean reversion

Together, they provide a balanced and defensible trading framework.
