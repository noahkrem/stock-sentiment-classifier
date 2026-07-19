"""
label_data.py
-------------
Merges index and AMZN price data, constructs the aggregate volatility
index, and labels each week by next-week AMZN return.

Output: data/processed/labeled.csv

Usage:
    # Use all indexes (default)
    python3 src/label_data.py

    # Use a subset of indexes
    python3 src/label_data.py --indexes vix vxn vix3m

Available indexes: nasdaqndxt, vix, vxazn, vxn, vix3m
Best results so far (accuracy 44.72%): nasdaqndxt, vxazn, vix3m
"""

from pathlib import Path
import argparse
import math
import os

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
INDEXES_CSV = DATA_DIR / "indexes_raw.csv"
AMZN_CSV = DATA_DIR / "amzn_weekly_returns.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "labeled.csv"

ALL_INDEX_COLS = ["nasdaqndxt", "vix", "vxazn", "vxn", "vix3m"]

# Percentile boundaries for labeling.
NEGATIVE_PERCENTILE = 0.33
POSITIVE_PERCENTILE = 0.67

# Score ranges for display (used by predict.py, not by the classifier)
SCORE_MAP = {"Positive": "3", "Neutral": "2", "Negative": "1"}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build labeled.csv from volatility index + AMZN price data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Use all indexes (default):
    python3 src/label_data.py

  Use only VIX-family indexes:
    python3 src/label_data.py --indexes vix vxn vix3m

  Use a single index:
    python3 src/label_data.py --indexes vxazn

  Exclude nasdaqndxt (suspected noise source):
    python3 src/label_data.py --indexes vix vxazn vxn vix3m

Available indexes: nasdaqndxt, vix, vxazn, vxn, vix3m
        """
    )
    parser.add_argument(
        "--indexes", "-i",
        nargs="+",
        choices=ALL_INDEX_COLS,
        default=ALL_INDEX_COLS,
        metavar="INDEX",
        help=(
            "Which indexes to include as features. "
            f"Choose from: {', '.join(ALL_INDEX_COLS)}. "
            "Defaults to all five."
        ),
    )
    return parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────

def resolve_input_path(default_path: Path, alternates=None) -> Path:
    candidates = [Path(default_path)]
    if alternates:
        candidates.extend([Path(p) for p in alternates])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find any of: {', '.join(str(p) for p in candidates)}")


# ── Step 1: Load and merge ────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    index_path = resolve_input_path(INDEXES_CSV, [REPO_ROOT / "data" / "raw" / "indexes_raw.csv"])
    amzn_path = resolve_input_path(AMZN_CSV, [REPO_ROOT / "data" / "raw" / "amzn_prices_raw.csv"])

    indexes = pd.read_csv(index_path)
    amzn = pd.read_csv(amzn_path)

    index_date_col = next((col for col in indexes.columns if col.lower() == "date"), None)
    if index_date_col is None:
        raise ValueError(f"Could not find a date column in index file {index_path}")
    indexes = indexes.rename(columns={index_date_col: "date"})
    indexes["date"] = pd.to_datetime(indexes["date"], errors="coerce")
    indexes = indexes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    amzn_date_col = next((col for col in amzn.columns if col.lower() == "date"), None)
    if amzn_date_col is None:
        raise ValueError(f"Could not find a date column in AMZN file {amzn_path}")
    amzn = amzn.rename(columns={amzn_date_col: "date"})
    amzn["date"] = pd.to_datetime(amzn["date"], errors="coerce")
    amzn = amzn.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    close_candidates = [
        col for col in amzn.columns
        if col.lower() in {"amzn_close", "amzn_close_price", "close", "adj close", "adjusted close"}
    ]
    if not close_candidates:
        close_candidates = [
            col for col in amzn.columns
            if "close" in col.lower() and "return" not in col.lower()
        ]

    if not close_candidates:
        raise ValueError(f"Could not find an AMZN close-price column in {amzn_path}")

    amzn = amzn.rename(columns={close_candidates[0]: "amzn_close"})
    amzn["amzn_close"] = pd.to_numeric(amzn["amzn_close"], errors="coerce")
    amzn = amzn.dropna(subset=["amzn_close"]).reset_index(drop=True)

    rename_map = {}
    for source_col, target_col in {
        "NASDAQNDXT": "nasdaqndxt",
        "VIXCLS": "vix",
        "VXAZN": "vxazn",
        "^VXN": "vxn",
        "^VIX3M": "vix3m",
    }.items():
        if source_col in indexes.columns:
            rename_map[source_col] = target_col
    indexes = indexes.rename(columns=rename_map)

    index_cols = [col for col in ALL_INDEX_COLS if col in indexes.columns]
    if not index_cols:
        raise ValueError(f"No index columns were found in {index_path}")

    for col in index_cols:
        indexes[col] = pd.to_numeric(indexes[col], errors="coerce")

    df = pd.merge(indexes, amzn, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Merged: {len(df)} weekly rows, {df['date'].min().date()} → {df['date'].max().date()}")

    missing = df[index_cols + ["amzn_close"]].isnull().sum()
    if missing.any():
        print(f"WARNING — missing values after merge:\n{missing[missing > 0]}")

    return df


# ── Step 2: Aggregate volatility index ───────────────────────────────────────

def compute_aggregate_index(df: pd.DataFrame, index_cols: list) -> pd.DataFrame:
    """
    Converts annualized index values to weekly expected moves, then
    normalizes each to [0, 1] and averages them into aggregate_vol_index.
    Only uses the indexes specified by the --indexes flag.
    """
    print(f"\nUsing indexes : {index_cols}")

    # Convert annualized index values to weekly expected moves.
    # VIX-style indexes are quoted as annualized percentages (e.g. 16.0 = 16%),
    # dividing by sqrt(52) gives the weekly expected percentage move.
    for col in index_cols:
        df[col] = df[col] / math.sqrt(52)

    # Engineered cross-index features — only added when both source
    # indexes are present in the active index_cols set.
    if "vix" in index_cols and "vix3m" in index_cols:
        df["vix_term_spread"] = df["vix"] - df["vix3m"]
    if "vxazn" in index_cols and "vix" in index_cols:
        df["vxazn_vix_ratio"] = df["vxazn"] / df["vix"]

    for col in index_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)

    norm_cols = [f"{col}_norm" for col in index_cols]
    df["aggregate_vol_index"] = df[norm_cols].mean(axis=1)
    df = df.drop(columns=norm_cols)

    return df


# ── Step 3: Label by next-week AMZN return ───────────────────────────────────

def compute_labels(df: pd.DataFrame, index_cols: list) -> pd.DataFrame:
    """
    The label for week T is based on AMZN's return in week T+1.
    This means: given the volatility indexes at the END of week T,
    predict whether AMZN goes up or down NEXT week.

    Thresholds are derived from the data using percentiles rather than
    fixed values. This guarantees a balanced ~33/34/33 class split
    regardless of the return distribution over the training window.

    Thresholds are computed on the full dataset (before any train/test
    split) so that labels are consistent across all rows.

    The last row is always dropped — there's no T+1 return for it.
    """
    df["amzn_next_return"] = df["amzn_close"].shift(-1) / df["amzn_close"] - 1

    # Lag and delta features — only for the active index set
    for col in index_cols:
        df[f"{col}_lag1"]  = df[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[col].shift(1)
    df["amzn_return_lag1"] = df["amzn_next_return"].shift(1)

    # Drop the last row (NaN next return) before computing percentiles
    # so the thresholds aren't skewed by the NaN row.
    valid_returns = df["amzn_next_return"].dropna()
    neg_threshold = valid_returns.quantile(NEGATIVE_PERCENTILE)
    pos_threshold = valid_returns.quantile(POSITIVE_PERCENTILE)

    print(f"\nLabel thresholds (data-driven):")
    print(f"  Negative : return < {neg_threshold:.4f} ({NEGATIVE_PERCENTILE*100:.0f}th percentile)")
    print(f"  Neutral  : {neg_threshold:.4f} ≤ return ≤ {pos_threshold:.4f}")
    print(f"  Positive : return > {pos_threshold:.4f} ({POSITIVE_PERCENTILE*100:.0f}th percentile)")

    def assign_label(ret):
        if ret > pos_threshold:
            return "Positive"
        if ret < neg_threshold:
            return "Negative"
        return "Neutral"

    df["label"] = df["amzn_next_return"].apply(assign_label)
    df["score_range"] = df["label"].map(SCORE_MAP)

    # Drop rows with NaN across lag/delta/return columns
    lag_cols = ([f"{col}_lag1"  for col in index_cols] +
                [f"{col}_delta" for col in index_cols] +
                ["amzn_return_lag1", "amzn_next_return"])
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


# ── Step 4: Write output ──────────────────────────────────────────────────────

def write_output(df: pd.DataFrame, index_cols: list):
    lag_delta_cols  = [col for col in df.columns
                       if col.endswith("_lag1") or col.endswith("_delta")]
    engineered_cols = [col for col in ["vix_term_spread", "vxazn_vix_ratio"]
                       if col in df.columns]

    out_cols = (["date"] + index_cols + engineered_cols + lag_delta_cols +
                ["aggregate_vol_index", "amzn_close", "amzn_next_return",
                 "label", "score_range"])
    df = df[out_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")

    feature_cols = [c for c in out_cols
                    if c not in {"date", "amzn_close", "amzn_next_return", "label", "score_range"}]
    print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")
    print(f"Feature columns written ({len(feature_cols)}): {feature_cols}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nAggregate index stats:")
    print(df["aggregate_vol_index"].describe().round(4))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Preserve the canonical column order from ALL_INDEX_COLS
    # regardless of the order the user typed them in
    index_cols = [col for col in ALL_INDEX_COLS if col in args.indexes]

    df = load_and_merge()
    df = compute_aggregate_index(df, index_cols)
    df = compute_labels(df, index_cols)
    write_output(df, index_cols)