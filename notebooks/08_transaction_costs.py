import pandas as pd
import numpy as np
import os 

# ====================================
# Configuration 
# ====================================

DATA_PATH = os.path.join("data", "cleaned", "cleaned_data.parquet")
ASSETS = ["BTC", "ETH", "DOGE"]

# ====================================
# Step 1 - Data preparation 
# ====================================

def load_data(path):
    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df[["BTC_close", "ETH_close", "DOGE_close", "BTC_high", "ETH_high", "DOGE_high", "BTC_low", "ETH_low", "DOGE_low"]].copy()
    n_dropped = df.isna().any(axis=1).sum()
    if n_dropped:
        print(f"WARNING: dropped {n_dropped} rows with NaN")
    df = df.dropna()
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    return df 

# ====================================
# Step 2 - Roll Model
# ====================================

def roll_spread(close):
    returns = np.log(close).diff().dropna()
    # np.cov returns a matrix
    # we only want [0, 1], which is row 0, column 1, and that is the Cov(a, b)
    # [0, 0] and [1, 1] are variances, we don't need them
    # [1, 0] also works, as Cov(0, 1) == Cov(1, 0)
    cov = np.cov(returns[1:], returns[:-1])[0, 1]
    if cov < 0:
        return 2 * np.sqrt(-cov)
    else:
        return 0
    
# ====================================
# Step 3 - Corwin-Schultz 
# ====================================

def cs_spread(high, low):
    h_t = np.log(high / low) 

    beta = np.square(h_t) + np.square(h_t.shift(1))

    two_period_high = high.rolling(2).max()
    two_period_low = low.rolling(2).min()
    gamma = np.square(np.log(two_period_high / two_period_low))

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = np.clip(spread, a_min=0, a_max=None)
    return spread 

# ====================================
# Step 4 - Convert spread to cost 
# ====================================

def compute_cost(position, spread, price):
    delta_pos = abs(position - position.shift(1))
    cost = delta_pos * spread * price / 2
    return cost 

# ====================================
# Main
# ====================================

def main():
    df = load_data(DATA_PATH)

    for asset in ASSETS:
        spread = roll_spread(df[f"{asset}_close"])
        print(f"{asset} Roll spread: {spread: .6f}")

        cs = cs_spread(df[f"{asset}_high"], df[f"{asset}_low"])
        print(f"{asset} CS spread (mean): {cs.mean():.6f}  ({cs.mean()*1e4:.2f} bps) \n")

    # Demonstrate compute_cost with a toy example
    test_pos   = pd.Series([0, 1, 1, 0, -1, 0])
    test_price = pd.Series([100.0] * 6)
    test_spread = 0.001  # 10 bps

    cost = compute_cost(test_pos, test_spread, test_price)
    print("\nToy cost demo:")
    print(cost)


if __name__ == "__main__":
    main()