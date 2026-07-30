"""
predict.py
----------
CLI inference script for the AMZN Stock Sentiment Classifier.

Preferred usage (live yfinance pull):
    python3 src/predict.py

Fallback usage (manual values, in the order saved by train.py):
    python3 src/predict.py --manual 18500 18.5 35.2 20.1 19.3

    Run with --list-features first to see the exact order train.py expects.

Override the confidence threshold at runtime:
    python3 src/predict.py --threshold 0.50

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

# yfinance ticker symbols keyed by the column name used in labeled.csv.
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
        model              — fitted sklearn classifier
        scaler             — StandardScaler fitted on X_train
        feature_columns    — ordered list of column names the model was trained on
        classes            — class label order (e.g. ['Negative', 'Neutral', 'Positive'])
        model_name         — 'LogisticRegression' or 'RandomForest'
        confidence_threshold — optimal threshold found by train.py's tuning loop
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
    derived columns (e.g. aggregate_vol_index, lag/delta columns) are
    handled in build_feature_vector() using training-set means.

    Raises RuntimeError if any required ticker fetch fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

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

    Columns not fetchable from live data (lag/delta/derived columns) are filled
    with the training-set column mean from the fitted scaler so the vector
    length and scale stay correct.
    """
    feature_columns = bundle["feature_columns"]
    scaler          = bundle["scaler"]
    train_means     = scaler.mean_

    vector = []
    filled = []

    for i, col in enumerate(feature_columns):
        if col in raw_values:
            vector.append(raw_values[col])
        else:
            vector.append(float(train_means[i]))
            filled.append(col)

    if filled:
        print(f"\n  NOTE: {len(filled)} feature(s) not available from live pull; "
              f"substituted training-set mean:")
        for col in filled:
            print(f"    {col}")

    X = np.array(vector, dtype=float).reshape(1, -1)
    return scaler.transform(X)


# ── Confidence-gated prediction ───────────────────────────────────────────────

def predict_with_threshold(model, X_scaled: np.ndarray,
                           classes: list, threshold: float) -> tuple:
    """
    Apply the confidence-gated prediction policy.

    For Negative and Positive predictions, the model must clear `threshold`
    confidence to commit to a directional call. Below threshold, the
    prediction falls back to Neutral — equivalent to "no position" in a
    trading context, avoiding costly wrong-direction errors.

    Neutral predictions are never gated — abstaining when the model already
    says Neutral would be redundant.

    Returns:
        pred_label  — final label after gating ('Negative'/'Neutral'/'Positive')
        raw_label   — what the model predicted before gating
        confidence  — probability of the raw predicted class
        proba       — full probability array aligned to classes
        gated       — True if the threshold caused a fallback to Neutral
    """
    proba      = model.predict_proba(X_scaled)[0]
    raw_label  = classes[int(np.argmax(proba))]
    confidence = float(proba.max())

    gated = False
    if raw_label in ("Negative", "Positive") and confidence < threshold:
        pred_label = "Neutral"
        gated = True
    else:
        pred_label = raw_label

    return pred_label, raw_label, confidence, proba, gated


# ── Output formatting ─────────────────────────────────────────────────────────

def print_prediction(raw_values: dict, X_scaled: np.ndarray,
                     bundle: dict, source: str,
                     show_proba: bool, threshold: float):

    model      = bundle["model"]
    classes    = bundle["classes"]
    model_name = bundle.get("model_name", "Unknown")

    pred_label, raw_label, confidence, proba, gated = predict_with_threshold(
        model, X_scaled, classes, threshold
    )

    score, sentiment, description = SCORE_LABELS[pred_label]

    print("\n" + "─" * 54)
    print("   AMZN SENTIMENT CLASSIFIER — NEXT WEEK OUTLOOK")
    print("─" * 54)
    print(f"  Score      : {score} / 3")
    print(f"  Sentiment  : {sentiment.upper()}")

    # Show gating note inline when a fallback occurred
    if gated:
        print(f"  ⚑ Gated    : model predicted {raw_label} ({confidence*100:.1f}% confidence)")
        print(f"               below threshold ({threshold:.2f}) → fell back to Neutral")

    print(f"  Summary    : {description}")
    print(f"  Model used : {model_name}")
    print(f"  Threshold  : {threshold:.2f}")
    print()

    if show_proba:
        print("  Confidence breakdown:")
        for cls_label, prob in zip(classes, proba):
            s, lbl, _ = SCORE_LABELS[cls_label]
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            # Mark the threshold line on the bar for the predicted class
            marker = " ◀ gated" if (gated and cls_label == raw_label) else ""
            print(f"    {lbl:<10} [{bar}] {prob * 100:5.1f}%{marker}")
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

  Override the confidence threshold (default loaded from model bundle):
    python3 src/predict.py --threshold 0.50

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
        "--threshold", "-t",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Confidence threshold for directional predictions (0.0–1.0). "
            "Predictions below this fall back to Neutral. "
            "Defaults to the value optimised by train.py and saved in the model bundle."
        ),
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
        print(f"Threshold  : {bundle.get('confidence_threshold', 'not set — re-run train.py')}")
        print(f"\nFeature columns ({len(bundle['feature_columns'])}) in order:")
        for i, col in enumerate(bundle["feature_columns"]):
            ticker = INDEX_TICKERS.get(col, "(derived — not live-fetchable)")
            print(f"  [{i:02d}] {col:<30} {ticker}")
        print()
        sys.exit(0)

    # Resolve threshold: CLI arg > bundle value > hard fallback
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using threshold from --threshold flag: {threshold:.2f}")
    elif "confidence_threshold" in bundle:
        threshold = bundle["confidence_threshold"]
        print(f"Using threshold from model bundle: {threshold:.2f}")
    else:
        threshold = 0.40
        print(f"WARNING: no threshold in bundle — using fallback {threshold:.2f}. "
              f"Re-run train.py to generate an optimised value.")

    feature_columns = bundle["feature_columns"]

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
    else:
        try:
            raw_values = fetch_live_values(feature_columns)
            source = "live yfinance"
        except RuntimeError as e:
            print(f"\n{e}")
            print("\nTip: run with --list-features to see the feature order, then use --manual.")
            sys.exit(1)

    X_scaled = build_feature_vector(raw_values, bundle)
    print_prediction(raw_values, X_scaled, bundle, source,
                     show_proba=not args.no_proba, threshold=threshold)


if __name__ == "__main__":
    main()