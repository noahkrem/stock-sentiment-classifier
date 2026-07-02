"""
label_data.py
-------------
Merges index and AMZN price data, constructs the aggregate volatility
index, and labels each week by next-week AMZN return.

Output: data/processed/labeled.csv
"""

import pandas as pd
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────

INDEXES_CSV   = "data/raw/indexes_raw.csv"
AMZN_CSV      = "data/raw/amzn_prices_raw.csv"
OUTPUT_CSV    = "data/processed/labeled.csv"

INDEX_COLS    = ["vxn", "vix", "vix3m", "rvx", "vvix"]

# Threshold for labeling: next-week AMZN return above/below this → Positive/Negative
POSITIVE_THRESHOLD =  0.02   # +2%
NEGATIVE_THRESHOLD = -0.02   # -2%

# Score ranges for display (used by predict.py, not by the classifier)
SCORE_MAP = {"Positive": "7-9", "Neutral": "4-6", "Negative": "1-3"}


# ── Step 1: Load and merge ────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    indexes = pd.read_csv(INDEXES_CSV, parse_dates=["date"])
    amzn    = pd.read_csv(AMZN_CSV,    parse_dates=["date"])

    # Rename AMZN close column defensively in case Erik names it differently
    amzn = amzn.rename(columns={amzn.columns[1]: "amzn_close"})

    # Inner join on date — drops any week where either source has no data
    df = pd.merge(indexes, amzn, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Merged: {len(df)} weekly rows, {df['date'].min().date()} → {df['date'].max().date()}")

    missing = df[INDEX_COLS + ["amzn_close"]].isnull().sum()
    if missing.any():
        print(f"WARNING — missing values after merge:\n{missing[missing > 0]}")

    return df


# ── Step 2: Aggregate volatility index ───────────────────────────────────────

def compute_aggregate_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes each of the 5 indexes to [0, 1] using the full series
    min/max, then averages across them to produce a single
    aggregate_vol_index per week.

    Note: this is a global min-max normalization over the full 10-year
    window. Noah's StandardScaler in features.py will re-scale for
    training — these normalized values are for the aggregate index
    construction only, and are kept as separate columns so Noah can
    use the raw values too.
    """
    for col in INDEX_COLS:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)

    norm_cols = [f"{col}_norm" for col in INDEX_COLS]
    df["aggregate_vol_index"] = df[norm_cols].mean(axis=1)

    # Drop the intermediate norm columns — Noah doesn't need them,
    # they'd just add noise to the feature matrix
    df = df.drop(columns=norm_cols)

    return df


# ── Step 3: Label by next-week AMZN return ───────────────────────────────────

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    The label for week T is based on AMZN's return in week T+1.
    This means: given the volatility indexes at the END of week T,
    predict whether AMZN goes up or down NEXT week.

    The last row is always dropped — there's no T+1 return for it.
    """
    # Next-week return: (close[T+1] - close[T]) / close[T]
    df["amzn_next_return"] = df["amzn_close"].shift(-1) / df["amzn_close"] - 1

    def assign_label(ret):
        if ret > POSITIVE_THRESHOLD:
            return "Positive"
        elif ret < NEGATIVE_THRESHOLD:
            return "Negative"
        else:
            return "Neutral"

    df["label"] = df["amzn_next_return"].apply(assign_label)
    df["score_range"] = df["label"].map(SCORE_MAP)

    # Drop last row — NaN next-week return, can't label it
    df = df.dropna(subset=["amzn_next_return"]).reset_index(drop=True)

    return df


# ── Step 4: Write output ──────────────────────────────────────────────────────

def write_output(df: pd.DataFrame):
    # Column order: what Noah's features.py expects
    out_cols = ["date"] + INDEX_COLS + ["aggregate_vol_index",
                "amzn_close", "amzn_next_return", "label", "score_range"]
    df = df[out_cols]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
    print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nAggregate index stats:")
    print(df["aggregate_vol_index"].describe().round(4))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_merge()
    df = compute_aggregate_index(df)
    df = compute_labels(df)
    write_output(df)