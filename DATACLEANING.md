# Data Cleaning Action Plan (02_data_clean.py)

This document outlines the step-by-step pipeline for transforming raw Binance OHLCV data into a clean, aligned dataset ready for strategy backtesting.

---

## Overview

Pipeline flow:

Raw Data → Validation → Alignment → Returns → Excess Returns → Final Dataset

Assets:
- BTC/USDT
- ETH/USDT
- DOGE/USDT

Frequency:
- 15-minute bars

---

## Step 1 — Load Raw Data

### Objective
Load raw CSV files and standardise formats.

### Actions
- Load BTC, ETH, DOGE datasets
- Keep relevant columns:
  - timestamp (open time)
  - open, high, low, close, volume
- Convert:
  - timestamp → datetime (UTC)
  - prices → float

### Checks
- Confirm timestamps are in milliseconds
- Ensure consistent column naming across assets

---

## Step 2 — Standardise Structure

### Objective
Ensure all datasets share identical structure.

### Actions
- Rename columns to:
  timestamp, open, high, low, close, volume
- Set timestamp as index
- Sort data chronologically

---

## Step 3 — Remove Duplicates

### Objective
Eliminate duplicate timestamps.

### Actions
- Identify duplicate timestamps
- Keep first occurrence
- Drop remaining duplicates

---

## Step 4 — Check for Missing Timestamps

### Objective
Ensure continuous 15-minute intervals.

### Actions
- Generate full timestamp range (15-min intervals)
- Compare against actual timestamps
- Identify missing bars

### Handling Strategy
- Small gaps → optionally forward-fill prices
- Large gaps → remove affected periods

---

## Step 5 — Align Assets

### Objective
Ensure all assets share identical timestamps.

### Actions
- Take intersection of timestamps across BTC, ETH, DOGE
- Filter datasets to common timestamps only

---

## Step 6 — Basic Data Sanity Checks

### Objective
Validate data integrity.

### Checks
- No negative prices
- No zero prices
- Volume values reasonable

### Action
- Investigate and fix anomalies if necessary

---

## Step 7 — Compute Returns

### Objective
Compute simple returns for each asset.

### Formula
r_t = (p_t - p_{t-1}) / p_{t-1}

### Actions
- Use close prices
- Compute returns
- Drop first row (NaN)

---

## Step 8 — Identify Outliers (Do NOT Automatically Remove)

### Objective
Detect extreme observations without distorting true market behaviour.

### Actions
- Compute rolling standard deviation (e.g., 100 bars)
- Compute z-score of returns:
  z_t = r_t / sigma_t
- Flag observations where:
  - |z| > 6–8

### Handling Strategy
- Inspect flagged points:
  - Check neighbouring bars
  - Check volume
  - Compare across assets
- Only correct if clearly a data error
- Otherwise retain as valid market movement

---

## Step 9 — Load Risk-Free Rate

### Objective
Prepare risk-free rate series.

### Actions
- Load daily Effective Federal Funds Rate (FRED)
- Convert percentage → decimal

---

## Step 10 — Align Risk-Free Rate to 15-Min Frequency

### Objective
Match risk-free rate to intraday data.

### Actions
- Forward-fill daily rate across intraday timestamps
- Convert annual rate to per-bar rate:

r_f(15min) = r_f(annual) / (365.25 × 24 × 4)

---

## Step 11 — Compute Excess Returns

### Formula
r_t^e = r_t - r_f(t-1)

### Actions
- Subtract lagged risk-free rate
- Store both:
  - raw returns
  - excess returns

---

## Step 12 — Construct Final Dataset

### Objective
Create unified dataset for all assets.

### Structure Example

| timestamp | BTC_close | ETH_close | DOGE_close | BTC_ret | ETH_ret | DOGE_ret | BTC_excess | ... |

### Actions
- Merge all assets into one dataset
- Ensure perfect alignment

---

## Step 13 — Final Validation

### Objective
Ensure dataset is clean and usable.

### Checks
- No missing values
- All timestamps aligned
- Returns look reasonable:
  - mean ≈ 0
  - heavy tails expected
- Visual inspection:
  - histogram of returns
  - time series plots

---

## Step 14 — Save Cleaned Data

### Objective
Persist final dataset.

### Actions
- Save as .parquet file
- Location:
  data/cleaned/cleaned_data.parquet

---

## Notes

- All timestamps are in UTC
- Risk-free rate is negligible at 15-minute frequency but included for completeness
- Extreme returns are preserved unless clearly erroneous
- No look-ahead bias is introduced at any stage

---

## Common Pitfalls (Avoid)

- Computing returns before sorting timestamps
- Misaligned timestamps across assets
- Removing genuine volatility spikes
- Using future data in preprocessing
- Forgetting to lag the risk-free rate