# Exploratory Data Analysis (EDA) Plan  
## File: `03_eda.py`

This document sets out the objective, workflow, and step-by-step implementation plan for the exploratory data analysis stage of the coursework.

The purpose of this file is not just to “look at the data”, but to test whether the chosen strategies are actually supported by the behaviour of the market data.

---

## 1. Objective

The EDA stage should answer the following questions:

1. What are the main statistical properties of BTC, ETH, and DOGE returns?
2. Do the assets exhibit volatility clustering and changing market regimes?
3. Are the assets strongly related to one another, and does that relationship vary over time?
4. Is there evidence of lead-lag behaviour from BTC to ETH and DOGE?
5. Is there enough evidence to justify:
   - a breakout / trend-following strategy
   - a lead-lag strategy

---

## 2. Input and Output

### Input
The file should use the cleaned dataset produced in `02_data_clean.py`, containing:
- aligned timestamps
- close prices
- simple returns
- excess returns
- volume
- risk-free rate if stored

### Output
The file should produce:
- summary statistics tables
- diagnostic plots
- cross-asset relationship analysis
- lead-lag validation results
- clear conclusions on whether each strategy is justified

All figures and tables that may later be used in the report should be saved to the `report/figures/` folder or another chosen output directory.

---

## 3. High-Level Workflow

The EDA process should follow this order:

1. Load cleaned data
2. Inspect the structure of the dataset
3. Compute summary statistics
4. Analyse return behaviour
5. Analyse volatility behaviour
6. Analyse cross-asset relationships
7. Test for lead-lag structure
8. Assess whether breakout and lead-lag strategies are justified
9. Save figures and tables

---

## 4. Step-by-Step Implementation Plan

---

## Step 1 — Load the Cleaned Dataset

### Objective
Load the cleaned parquet file and confirm that the dataset is ready for analysis.

### Actions
- Read the cleaned data file
- Confirm the date range is correct
- Confirm all three assets are present
- Confirm timestamps are aligned
- Check column names and datatypes

### Things to verify
- No missing values in the key columns
- Returns are already computed
- Time index is sorted
- Data frequency is consistently 15 minutes

### Expected outcome
A clean dataframe ready for EDA.

---

## Step 2 — Inspect the Dataset Structure

### Objective
Understand exactly what columns and series are available before analysis begins.

### Actions
- Print the first few rows
- Print the last few rows
- Print column names
- Print dataset shape
- Check the number of observations
- Confirm the number of observations matches expectation for 6 months of 15-minute bars

### Expected outcome
Confidence that the correct data has been loaded and the sample size is sensible.

---

## Step 3 — Compute Basic Summary Statistics

### Objective
Understand the overall distributional properties of returns for BTC, ETH, and DOGE.

### Actions
For each asset, compute:
- mean return
- standard deviation
- skewness
- kurtosis
- minimum return
- maximum return
- median
- selected quantiles if useful

### Why this matters
This helps establish the stylised facts of crypto returns:
- high volatility
- fat tails
- potentially skewed distributions

### Expected interpretation
- Mean return is likely close to zero at 15-minute frequency
- Standard deviation should be highest for DOGE and lower for BTC
- Kurtosis should be greater than 3, indicating heavy tails

### Output
A summary statistics table that can potentially be used in the report.

---

## Step 4 — Plot Return Time Series

### Objective
Visualise how returns evolve through time.

### Actions
- Plot BTC returns over the sample period
- Plot ETH returns over the sample period
- Plot DOGE returns over the sample period
- Either use separate plots or a vertically stacked layout

### What to look for
- Volatility clustering
- Sudden spikes
- Calm periods versus turbulent periods
- Whether DOGE appears more erratic than BTC and ETH

### Why this matters
This supports discussion of:
- changing market conditions
- the need for volatility scaling
- the difficulty of forecasting during high-volatility periods

### Output
Return time series plots for each asset.

---

## Step 5 — Plot Return Distributions

### Objective
Examine the shape of the return distributions.

### Actions
- Plot a histogram for each asset’s returns
- Optionally overlay a normal density for comparison
- Optionally compare all three return distributions on the same scale

### What to look for
- Fat tails
- High peak around zero
- Non-normality
- Differences in dispersion between BTC, ETH, and DOGE

### Why this matters
This helps justify:
- why simple Gaussian assumptions may be poor
- why extreme moves are important in crypto markets
- why robust risk control is needed

### Output
Return distribution histograms.

---

## Step 6 — Analyse Volatility

### Objective
Understand whether volatility is stable or clustered through time.

### Actions
- Compute rolling volatility for each asset using a fixed lookback window
- Example windows could be:
  - 50 bars
  - 96 bars
  - 100 bars
- Plot rolling volatility over time

### What to look for
- Persistent periods of high volatility
- Persistent periods of low volatility
- Whether DOGE has more extreme volatility regimes than BTC and ETH

### Why this matters
This provides direct support for:
- volatility-scaled position sizing
- regime dependence in performance
- the idea that strategy performance may change across the sample

### Output
Rolling volatility plots.

---

## Step 7 — Correlation Analysis

### Objective
Measure how closely the three assets move together.

### Actions
- Compute the full-sample correlation matrix of returns
- Compute rolling correlations for:
  - BTC vs ETH
  - BTC vs DOGE
  - ETH vs DOGE
- Plot rolling correlations through time

### What to look for
- BTC and ETH should likely show stronger correlation
- DOGE may be more unstable and less tightly linked
- Correlations may change through time, especially during volatile periods

### Why this matters
This helps with:
- understanding whether cross-asset strategies are plausible
- assessing whether BTC can realistically act as a “leading” asset
- supporting the discussion of market interconnectedness

### Output
- Correlation matrix table
- Rolling correlation plots

---

## Step 8 — Lead-Lag Analysis Using Cross-Correlation

### Objective
Test whether BTC appears to lead ETH and DOGE at short lags.

### Actions
- Compute cross-correlations between BTC returns and ETH returns
- Compute cross-correlations between BTC returns and DOGE returns
- Evaluate short positive lags such as:
  - 1 bar
  - 2 bars
  - 3 bars
  - 4 bars
- Optionally evaluate negative lags too, to confirm directionality

### How to interpret
- If the strongest relationship appears when BTC is shifted earlier and ETH/DOGE later, this suggests BTC leads
- If the cross-correlation is weak or flat across lags, the evidence for lead-lag is poor

### Why this matters
This is a direct validation step for the lead-lag strategy.

### Output
- Cross-correlation values by lag
- A plot of correlation versus lag for BTC–ETH and BTC–DOGE

---

## Step 9 — Conditional Return Analysis

### Objective
Test whether large BTC moves are followed by moves in ETH and DOGE in the same direction.

### Actions
- Define a “large BTC move” rule, for example:
  - BTC return greater than 1 standard deviation in absolute value
  - or BTC return in the top decile of absolute moves
- Split events into:
  - large positive BTC moves
  - large negative BTC moves
- For each event, compute the average ETH and DOGE returns over the next:
  - 1 bar
  - 2 bars
  - 3 bars
- Compare the average forward returns to zero

### What to look for
- If ETH and DOGE consistently move in the same direction after BTC, this supports the lead-lag idea
- If results are noisy, inconsistent, or tiny, the strategy may not be tradable after costs

### Why this matters
A strategy needs more than statistical dependence. It needs economically meaningful predictability.

### Output
A small results table showing forward ETH/DOGE returns following large BTC moves.

---

## Step 10 — Optional Granger Causality Tests

### Objective
Formally test whether BTC returns help predict ETH and DOGE returns.

### Actions
- Run Granger causality tests:
  - BTC → ETH
  - BTC → DOGE
- Use short lag orders consistent with 15-minute trading logic
- Keep the interpretation cautious

### Important note
Granger causality should not be treated as sufficient evidence on its own.
It should only supplement the cross-correlation and conditional return analysis.

### Why this matters
It adds statistical depth, but should not become the main justification.

### Output
P-values and a short conclusion on whether predictive structure exists.

---

## Step 11 — Trend / Momentum Diagnostics for Breakout Strategy

### Objective
Check whether there is evidence supporting a breakout or trend-following strategy.

### Actions
- Plot close prices for BTC, ETH, and DOGE
- Optionally overlay moving averages
- Visually inspect whether there are sustained directional moves
- Compute autocorrelation of returns
- Compute autocorrelation of absolute returns

### What to look for
- Returns themselves may show weak autocorrelation
- Absolute returns may show stronger autocorrelation due to volatility clustering
- Price series may still exhibit persistent trends even if short-horizon return autocorrelation is weak

### Why this matters
Breakout strategies rely more on persistent price movement and regime behaviour than on simple return autocorrelation.

### Output
- Price plots
- ACF of returns
- ACF of absolute returns

---

## Step 12 — Estimate Expected Strategy Difficulty

### Objective
Get an early sense of whether the strategies are likely to be robust or cost-sensitive.

### Actions
- Observe how noisy the series are
- Assess how often large moves occur
- Assess how persistent the directional moves appear
- Assess whether lead-lag effects look strong enough to survive trading costs

### Expected interpretation
- Breakout may be supported if trends are visible and volatility regimes are persistent
- Lead-lag may only be viable if predictive structure is clear and not too weak

### Why this matters
This step helps you decide whether to proceed with both strategies or replace one before wasting time building it.

---

## Step 13 — Summarise EDA Findings

### Objective
Translate the analysis into direct conclusions for strategy development.

### Actions
Write a short set of conclusions covering:
- which asset is most volatile
- whether returns are heavy-tailed
- whether volatility clustering exists
- whether BTC appears to lead ETH and/or DOGE
- whether breakout is justified
- whether lead-lag is justified

### Example conclusion structure
- BTC, ETH, and DOGE all display heavy-tailed return distributions and time-varying volatility
- DOGE is the most volatile asset in the sample
- BTC and ETH show stronger correlation than BTC and DOGE
- Short-lag predictive structure from BTC to ETH/DOGE is either present or weak
- Breakout strategy is supported by persistent volatility regimes and sustained price moves
- Lead-lag strategy should be pursued only if short-horizon predictive effects are clear enough to justify trading after costs

### Output
A concise EDA conclusion section that can later feed directly into the report.

---

## 5. Suggested Figures and Tables

The following outputs are the most useful candidates for the final report:

### Tables
- Summary statistics table
- Correlation matrix
- Conditional return results after large BTC moves

### Figures
- Return time series
- Rolling volatility
- Rolling correlation
- Cross-correlation vs lag
- Price plots showing trends

Since the report is only 5 pages, only the strongest 2 to 3 plots should eventually be selected.

---

## 6. Order of Execution in `03_eda.py`

A clean order for the file is:

1. Load data
2. Inspect dataset structure
3. Compute summary statistics
4. Plot return series
5. Plot return distributions
6. Compute and plot rolling volatility
7. Compute correlation matrix and rolling correlations
8. Run cross-correlation analysis
9. Run conditional return analysis
10. Optionally run Granger causality tests
11. Plot prices and basic momentum diagnostics
12. Write final EDA conclusions
13. Save outputs

---

## 7. Key Questions This File Must Answer

By the end of `03_eda.py`, you should be able to answer:

1. Are the returns heavy-tailed and volatile?
2. Which asset is the most volatile?
3. Does volatility cluster?
4. How strongly are BTC, ETH, and DOGE related?
5. Does BTC appear to lead ETH and DOGE at short horizons?
6. Is the lead-lag effect strong enough to trade?
7. Does the data support using a breakout strategy?

If these questions are not answered clearly, the EDA has not done its job.

---

## 8. Common Mistakes to Avoid

- Producing many plots without linking them to strategy decisions
- Looking only at full-sample correlation and calling it predictive
- Treating Granger causality as proof of profitability
- Ignoring the size of the effect relative to trading costs
- Doing EDA without writing down conclusions
- Keeping weak strategy ideas alive despite poor evidence

---

## 9. Final Deliverable of This Stage

The final output of the EDA stage should be:

- a cleaned set of diagnostic tables and figures
- a clear conclusion on whether breakout is justified
- a clear conclusion on whether lead-lag is justified
- a set of report-ready visualisations
- direction for the next implementation stage

Once this file is completed, the next step is to decide whether to proceed with:
- `04_breakout_strategy.py`
- `05_leadlag_strategy.py`

or whether the second strategy should be revised based on the EDA evidence.