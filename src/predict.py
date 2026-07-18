"""
predict.py
----------
CLI inference script for the AMZN Stock Sentiment Classifier.

Preferred usage (live yfinance pull):
    python3 src/predict.py

Fallback usage (manual values, in the order saved by train.py):
    python3 src/predict.py --manual 18500 18.5 35.2 20.1 19.3

    Run with --list-features first to see the exact order train.py expects.

Sentiment score: 1-3
    1 = Negative  (AMZN expected to underperform next week)
    2 = Neutral   (No clear directional signal)
    3 = Positive  (AMZN expected to outperform next week)
"""

import argparse
import pickle
import sys
import os
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/model.pkl"

# yfinance ticker symbols keyed by the column name used in labeled.csv / features.py.
# These must match the column names that train.py's get_feature_columns() selected.
# If a feature column has no live ticker (e.g. a derived column like aggregate_vol_index),
# it is filled with None and handled in build_feature_vector().
INDEX_TICKERS = {
    "nasdaqndxt": "^NDXT",
    "vix":        "^VIX",
    "vxazn":      "^VXAZN",
    "vxn":        "^VXN",
    "vix3m":      "^VIX3M",
}

# Sentiment display config keyed by the string class labels train.py uses.
SCORE_LABELS = {
    "Negative": ("1", "Negative",
        "Moderate-to-strong expectation that AMZN will underperform over the "
        "next week. Likelihood of a price decline."),
    "Neutral":  ("2", "Neutral",
        "No clear directional signal. AMZN expected to trade relatively flat "
        "or close to broader market performance next week."),
    "Positive": ("3", "Positive",
        "Moderate-to-strong expectation that AMZN will outperform over the "
        "next week. Likelihood of a price increase."),
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_bundle(model_path: str) -> dict:
    """
    Load the model bundle saved by train.py.

    Expected keys inside the pickle:
        model           — fitted sklearn classifier
        scaler          — StandardScaler fitted on X_train
        feature_columns — ordered list of column names the model was trained on
        classes         — class label order (e.g. ['Negative', 'Neutral', 'Positive'])
        model_name      — 'LogisticRegression' or 'RandomForest'
    """
    if not os.path.exists(model_path):
        print(f"ERROR: model bundle not found at '{model_path}'.")
        print("Run train.py first to generate it:")
        print("  python src/train.py --data data/processed/labeled.csv --out models/model.pkl")
        sys.exit(1)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    required = {"model", "scaler", "feature_columns", "classes"}
    missing = required - bundle.keys()
    if missing:
        print(f"ERROR: model bundle is missing keys: {missing}")
        print("Re-run train.py to regenerate a valid bundle.")
        sys.exit(1)

    return bundle


# ── Live data pull ────────────────────────────────────────────────────────────

def fetch_live_values(feature_columns: list) -> dict:
    """
    Pull the most recent close value for each raw index via yfinance.
    Only fetches tickers for columns that appear in INDEX_TICKERS —
    derived columns (e.g. aggregate_vol_index) are handled separately.

    Returns {col_name: float} for every column in INDEX_TICKERS that
    is also in feature_columns.

    Raises RuntimeError if any required ticker fetch fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    # Only fetch tickers whose column appears in the trained feature set
    to_fetch = {col: sym for col, sym in INDEX_TICKERS.items() if col in feature_columns}
    values = {}
    failed = []

    print("Fetching latest index values from Yahoo Finance...")
    for col_name, ticker_sym in to_fetch.items():
        try:
            hist = yf.Ticker(ticker_sym).history(period="5d", interval="1d")
            if hist.empty or "Close" not in hist.columns:
                raise ValueError("empty response")
            latest = float(hist["Close"].dropna().iloc[-1])
            values[col_name] = latest
            print(f"  {ticker_sym:<10} ({col_name}): {latest:.2f}")
        except Exception as e:
            failed.append((col_name, ticker_sym, str(e)))

    if failed:
        lines = "\n".join(f"  {sym} ({col}): {err}" for col, sym, err in failed)
        raise RuntimeError(
            f"Failed to fetch the following tickers:\n{lines}\n"
            "Use --manual to provide all values directly."
        )

    return values


# ── Feature vector construction ───────────────────────────────────────────────

def build_feature_vector(raw_values: dict, bundle: dict) -> np.ndarray:
    """
    Build and scale the feature vector in the exact column order train.py used.

    train.py derives feature_columns dynamically from the labeled CSV, so the
    order is whatever get_feature_columns() returned at training time — and that
    order is saved inside the bundle. We strictly follow it here.

    Columns not in INDEX_TICKERS (e.g. aggregate_vol_index, or any delta/lag
    columns Jag's label_data.py computed) cannot be fetched live. These are
    filled with the training-set column mean (retrieved from the fitted scaler)
    so the vector length stays correct.

    A warning is printed for every filled column so the limitation is visible.
    """
    feature_columns = bundle["feature_columns"]
    scaler          = bundle["scaler"]

    # scaler.mean_ holds the per-feature training mean in feature_columns order
    train_means = scaler.mean_

    vector = []
    filled = []

    for i, col in enumerate(feature_columns):
        if col in raw_values:
            vector.append(raw_values[col])
        else:
            # Derived / unresolvable column — substitute training mean
            vector.append(float(train_means[i]))
            filled.append(col)

    if filled:
        print(f"\n  NOTE: {len(filled)} feature(s) not available from live pull; "
              f"substituted training-set mean:")
        for col in filled:
            print(f"    {col}")

    X = np.array(vector, dtype=float).reshape(1, -1)
    return scaler.transform(X)


# ── Output formatting ─────────────────────────────────────────────────────────

def print_prediction(raw_values: dict, X_scaled: np.ndarray,
                     bundle: dict, source: str, show_proba: bool):
    model       = bundle["model"]
    classes     = bundle["classes"]   # e.g. ['Negative', 'Neutral', 'Positive']
    model_name  = bundle.get("model_name", "Unknown")

    pred_label = model.predict(X_scaled)[0]   # string: 'Negative'/'Neutral'/'Positive'

    probabilities = None
    if show_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[0]

    score, sentiment, description = SCORE_LABELS[pred_label]

    print("\n" + "─" * 54)
    print("   AMZN SENTIMENT CLASSIFIER — NEXT WEEK OUTLOOK")
    print("─" * 54)
    print(f"  Score      : {score} / 3")
    print(f"  Sentiment  : {sentiment.upper()}")
    print(f"  Summary    : {description}")
    print(f"  Model used : {model_name}")
    print()

    if probabilities is not None:
        print("  Confidence breakdown:")
        for cls_label, prob in zip(classes, probabilities):
            s, lbl, _ = SCORE_LABELS[cls_label]
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"    {lbl:<10} [{bar}] {prob * 100:5.1f}%")
        print()

    print(f"  Index values used ({source}):")
    for col, val in raw_values.items():
        print(f"    {col:<16}: {val:.2f}")
    print("─" * 54)
    print("  ⚠  Not financial advice. For educational use only.")
    print("─" * 54 + "\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AMZN Stock Sentiment Classifier — CLI inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live pull (default):
    python3 src/predict.py

  See which features the model expects and in what order:
    python3 src/predict.py --list-features

  Manual fallback (values must match --list-features order):
    python3 src/predict.py --manual 18500 18.5 35.2 20.1 19.3 0.31

  Suppress confidence breakdown:
    python3 src/predict.py --no-proba
        """
    )
    parser.add_argument(
        "--manual", "-m",
        nargs="+",
        type=float,
        metavar="VALUE",
        help=(
            "Skip live fetch and provide raw feature values manually. "
            "Values must be in the order shown by --list-features."
        ),
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print the feature columns the loaded model was trained on, then exit.",
    )
    parser.add_argument(
        "--no-proba",
        action="store_true",
        help="Suppress the confidence breakdown.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to the model bundle pkl (default: {MODEL_PATH}).",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    bundle = load_bundle(args.model)

    # --list-features: show what train.py saved, then exit
    if args.list_features:
        print(f"\nModel      : {bundle.get('model_name', 'Unknown')}")
        print(f"Classes    : {bundle['classes']}")
        print(f"\nFeature columns ({len(bundle['feature_columns'])}) in order:")
        for i, col in enumerate(bundle["feature_columns"]):
            ticker = INDEX_TICKERS.get(col, "(derived — not live-fetchable)")
            print(f"  [{i:02d}] {col:<30} {ticker}")
        print()
        sys.exit(0)

    feature_columns = bundle["feature_columns"]

    # Manual mode: user supplies every feature value in order
    if args.manual:
        if len(args.manual) != len(feature_columns):
            print(f"ERROR: --manual expects {len(feature_columns)} values "
                  f"(got {len(args.manual)}).")
            print("Run with --list-features to see the required order.")
            sys.exit(1)
        raw_values = dict(zip(feature_columns, args.manual))
        source = "manual input"
        print("\nUsing manually provided values:")
        for col, val in raw_values.items():
            print(f"  {col}: {val}")

    # Live mode (default): fetch what we can, fill derived columns from training mean
    else:
        try:
            raw_values = fetch_live_values(feature_columns)
            source = "live yfinance"
        except RuntimeError as e:
            print(f"\n{e}")
            print("\nTip: run with --list-features to see the feature order, then use --manual.")
            sys.exit(1)

    X_scaled = build_feature_vector(raw_values, bundle)
    print_prediction(raw_values, X_scaled, bundle, source, show_proba=not args.no_proba)


if __name__ == "__main__":
    main()