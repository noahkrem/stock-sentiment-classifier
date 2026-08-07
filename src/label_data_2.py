"""
label_data_daily.py
-------------------
Merges 7 daily raw market CSV files, computes configurable lag & momentum 
(delta) features, and generates daily next-day return quantile labels.

Output: data/processed/labeled.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ── Config & Directory Setup ──────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "AMZN-Index-VIX-data"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "labeled_2.csv"

# Percentile boundaries for daily labeling (~33/34/33 split)
NEGATIVE_PERCENTILE = 0.33
POSITIVE_PERCENTILE = 0.67

# List of all 7 raw files to merge
RAW_FILES = [
    "AMZN-Historical-Data-2011-to-2026.csv",
    "IXIC_or_NASDAQCOM_NASDAQ Composite Historical Data 2011-to-2026.csv",
    "NASDAQNDXT_NASDAQ-100 Technology Sector Index_2011_2026.csv",
    "SP500-25_S&P 500 Consumer Discretionary Historical Data-2011-to-2026.csv",
    "SPX_S&P 500 Historical Data_2011-to-2026.csv",
    "VIXCLS-Cboe Volatility Index (VIX) Daily Closing Values_2011_2026.csv",
    "VXAZN_Cboe Equity VIX on Amazon Index-2011-to-2026.csv"
]


# ── Argument Parsing (Configurable Lags) ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily labeled dataset.")
    parser.add_argument(
        "--lag-days", "-l",
        type=int,
        default=1,
        help="Number of days for historical memory lag and delta features (default: 1)."
    )
    return parser.parse_args()


# ── Step 1: Clean & Merge ─────────────────────────────────────────────────────

def clean_numeric_value(val) -> float:
    """Strips commas, percentage signs, and converts strings to numeric float."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("%", "").strip()
    return float(s)


def load_and_merge_daily() -> pd.DataFrame:
    dfs = []
    
    for filename in RAW_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Could not find required dataset file at: {filepath}")

        df = pd.read_csv(filepath)
        
        # Standardize date column
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col is None:
            raise ValueError(f"No date column found in {filename}")
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Clean numeric string columns (volume, prices, change percentages)
        for col in df.columns:
            if col != "date":
                df[col] = df[col].apply(clean_numeric_value)

        dfs.append(df)

    # Perform sequential inner joins across all daily files
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="inner")

    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    print(f"✓ Successfully merged {len(merged_df)} daily rows across all 7 datasets.")
    return merged_df


# ── Step 2: Feature Engineering (Lags & Deltas) ──────────────────────────────

def compute_features(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Computes configurable N-day lags and deltas for all numerical features.
    Change `lag_days` via command line argument `--lag-days`.
    """
    print(f"⚡ Generating features with a {lag_days}-day historical lag...")

    feature_cols = [c for c in df.columns if c != "date"]
    
    for col in feature_cols:
        # Create configurable lag feature
        df[f"{col}_lag{lag_days}"] = df[col].shift(lag_days)
        # Create momentum (delta) feature: Current - Lag
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag{lag_days}"]

    return df


# ── Step 3: Compute Daily Targets & Quantile Labels ───────────────────────────

def compute_daily_labels(df: pd.DataFrame, threshold: float = 0.015) -> pd.DataFrame:
    """
    Calculates two-stage targets:
    1. stage1_significant: 1 if |return| > threshold, else 0 (Volatility/Movement filter)
    2. stage2_direction: 1 if return > 0, else 0 (Up vs Down direction)
    """
    # Calculate next-day return
    df["amzn_next_return"] = df["amzn"].shift(-1) / df["amzn"] - 1

    # Stage 1: Is the move significant?
    df["stage1_significant"] = (df["amzn_next_return"].abs() > threshold).astype(int)

    # Stage 2: Up vs Down Direction (Binary)
    df["stage2_direction"] = (df["amzn_next_return"] > 0).astype(int)

    # Keep original 3-class label if needed for legacy comparison
    def assign_legacy_label(ret):
        if pd.isna(ret): return np.nan
        if ret > threshold: return "Positive"
        if ret < -threshold: return "Negative"
        return "Neutral"

    df["label"] = df["amzn_next_return"].apply(assign_legacy_label)

    # Clean missing boundary values
    df = df.dropna().reset_index(drop=True)
    return df


# ── Step 4: Save Labeled Output ───────────────────────────────────────────────

def main():
    args = parse_args()
    
    df = load_and_merge_daily()
    df = compute_features(df, lag_days=args.lag_days)
    df = compute_daily_labels(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")

    print(f"\n✅ Output successfully written to: {OUTPUT_CSV}")
    print(f"Total labeled daily rows: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
