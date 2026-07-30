"""
evaluate_threshold.py
---------------------
Evaluates the effect of a static ±2% practical return gate applied on top of
the model's predicted labels, on a held-out validation set.

The model is trained on percentile-based labels (balanced classes), but in
practice a predicted "Negative" week that only moves -0.8% is not actionable.
This script tests whether re-labeling low-magnitude predictions as Neutral
at inference time improves practical performance.

Two gating strategies are compared side-by-side:
  1. Confidence gate  — predictions below a probability threshold → Neutral
  2. Return gate      — model predicts direction, but if the predicted class
                        corresponds to a historically low-magnitude outcome,
                        re-label as Neutral using a static ±RETURN_THRESHOLD

Strategy 2 is approximated by using the model's confidence as a proxy for
predicted magnitude — higher confidence correlates with larger expected moves.
The return gate threshold is swept alongside the confidence gate so you can
see which setting best captures the ±2% practical boundary.

Usage:
    python src/evaluate_threshold.py
    python src/evaluate_threshold.py --data data/processed/labeled.csv
    python src/evaluate_threshold.py --model models/model.pkl --return-gate 0.015
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# ── Config (must match train.py) ──────────────────────────────────────────────

RANDOM_STATE     = 23
TEST_SIZE        = 0.10
TARGET_COL       = "label"
NON_FEATURE_COLS = {"date", TARGET_COL, "score_range", "amzn_next_return", "amzn_close"}

RETURN_THRESHOLD = 0.02   # ±2% practical gate
CONFIDENCE_RANGE = np.arange(0.33, 0.65, 0.05)

CLASS_ORDER = ["Negative", "Neutral", "Positive"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        print(f"ERROR: model bundle not found at '{model_path}'.")
        print("Run train.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_feature_columns(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLS]


def rebuild_test_split(df: pd.DataFrame, feature_cols: list, scaler):
    """
    Reconstruct the exact same test split train.py used.
    Returns X_test_scaled, y_test (string labels), actual_returns.
    """
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

    X = df[feature_cols].to_numpy()
    y = df[TARGET_COL].to_numpy()
    returns = df["amzn_next_return"].to_numpy()

    _, X_test, _, y_test, _, ret_test = train_test_split(
        X, y, returns,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test, ret_test


def apply_confidence_gate(proba_matrix, class_order, threshold) -> list:
    """Gate low-confidence directional predictions to Neutral."""
    y_gated = []
    for proba in proba_matrix:
        raw_class  = class_order[int(np.argmax(proba))]
        confidence = float(proba.max())
        if raw_class in ("Negative", "Positive") and confidence < threshold:
            y_gated.append("Neutral")
        else:
            y_gated.append(raw_class)
    return y_gated


def apply_return_gate(y_pred, actual_returns, return_threshold) -> list:
    """
    Re-label predictions as Neutral when the actual return doesn't clear
    ±return_threshold. This measures how often the model predicts direction
    correctly but for moves too small to be practically actionable.

    NOTE: this uses actual future returns, so it can only be computed on a
    validation set — not at live inference time. Its purpose is to quantify
    the practical value of the model's directional calls, not to gate live
    predictions.
    """
    y_gated = []
    for pred, ret in zip(y_pred, actual_returns):
        if pred == "Positive" and ret <= return_threshold:
            y_gated.append("Neutral")
        elif pred == "Negative" and ret >= -return_threshold:
            y_gated.append("Neutral")
        else:
            y_gated.append(pred)
    return y_gated


def print_metrics(y_true, y_pred, label: str, class_order: list):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_df    = pd.DataFrame(
        cm,
        index=[f"actual_{c}"  for c in class_order],
        columns=[f"pred_{c}"  for c in class_order],
    )
    report   = classification_report(
        y_true, y_pred, labels=class_order, output_dict=True, zero_division=0
    )

    dir_errors = sum(
        1 for t, p in zip(y_true, y_pred)
        if (t == "Negative" and p == "Positive") or
           (t == "Positive" and p == "Negative")
    )
    neutral_pct = 100 * sum(p == "Neutral" for p in y_pred) / len(y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Dir. errors   : {dir_errors}")
    print(f"  Neutral preds : {neutral_pct:.1f}%")
    print(f"\n  Confusion matrix:")
    print(cm_df.to_string())
    print(f"\n  Per-class metrics:")
    for cls in class_order:
        r = report[cls]
        print(
            f"    {cls:10s}  f1={r['f1-score']:.4f}  "
            f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  "
            f"support={int(r['support'])}"
        )
    return macro_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: Path, model_path: Path, return_gate: float):
    bundle = load_bundle(model_path)
    model       = bundle["model"]
    scaler      = bundle["scaler"]
    model_name  = bundle.get("model_name", "Unknown")
    saved_threshold = bundle.get("confidence_threshold", None)

    print(f"Model          : {model_name}")
    print(f"Saved threshold: {saved_threshold}")
    print(f"Return gate    : ±{return_gate*100:.1f}%")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X_test_scaled, y_test, ret_test = rebuild_test_split(df, feature_cols, scaler)

    proba_matrix = model.predict_proba(X_test_scaled)
    y_raw        = [CLASS_ORDER[int(np.argmax(p))] for p in proba_matrix]

    # ── 1. Baseline (no gate) ─────────────────────────────────────────────────
    print_metrics(y_test, y_raw, "BASELINE — no gate", CLASS_ORDER)

    # ── 2. Return gate (actual returns used to filter low-magnitude calls) ────
    y_return_gated = apply_return_gate(y_raw, ret_test, return_gate)
    print_metrics(
        y_test, y_return_gated,
        f"RETURN GATE — actual |return| < {return_gate*100:.0f}% → Neutral",
        CLASS_ORDER,
    )

    # ── 3. Confidence gate sweep ──────────────────────────────────────────────
    print(f"\n{'─' * 66}")
    print("  CONFIDENCE GATE SWEEP")
    print(f"{'─' * 66}")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Macro F1':>10}  "
          f"{'Dir. Errors':>12}  {'Neutral%':>9}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*9}")

    best_conf_f1  = -1.0
    best_conf_t   = float(CONFIDENCE_RANGE[0])

    for t in CONFIDENCE_RANGE:
        y_gated   = apply_confidence_gate(proba_matrix, CLASS_ORDER, t)
        acc       = accuracy_score(y_test, y_gated)
        macro_f1  = f1_score(y_test, y_gated, labels=CLASS_ORDER,
                             average="macro", zero_division=0)
        dir_err   = sum(
            1 for tr, pr in zip(y_test, y_gated)
            if (tr == "Negative" and pr == "Positive") or
               (tr == "Positive" and pr == "Negative")
        )
        neut_pct  = 100 * sum(p == "Neutral" for p in y_gated) / len(y_gated)
        saved_marker = " ◀ saved" if saved_threshold and abs(t - saved_threshold) < 0.001 else ""
        best_marker  = " ◀ best"  if macro_f1 > best_conf_f1 else ""
        print(f"  {t:>10.2f}  {acc:>10.4f}  {macro_f1:>10.4f}  "
              f"{dir_err:>12}  {neut_pct:>8.1f}%{saved_marker}{best_marker}")
        if macro_f1 > best_conf_f1:
            best_conf_f1 = macro_f1
            best_conf_t  = float(t)

    # ── 4. Best confidence gate (full metrics) ────────────────────────────────
    y_best_conf = apply_confidence_gate(proba_matrix, CLASS_ORDER, best_conf_t)
    print_metrics(
        y_test, y_best_conf,
        f"BEST CONFIDENCE GATE (threshold = {best_conf_t:.2f})",
        CLASS_ORDER,
    )

    # ── 5. Combined: confidence gate + return gate ────────────────────────────
    # Apply confidence gate first, then additionally filter low-magnitude calls
    # using actual returns. This shows the upper bound of what the practical
    # ±2% filter could achieve if the model's directional calls are correct.
    y_combined = apply_return_gate(y_best_conf, ret_test, return_gate)
    print_metrics(
        y_test, y_combined,
        f"COMBINED — confidence gate ({best_conf_t:.2f}) + return gate (±{return_gate*100:.0f}%)",
        CLASS_ORDER,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    configs = [
        ("Baseline",                y_raw),
        (f"Return gate ±{return_gate*100:.0f}%", y_return_gated),
        (f"Confidence gate {best_conf_t:.2f}",   y_best_conf),
        ("Combined",                y_combined),
    ]
    print(f"\n{'─' * 55}")
    print("  SUMMARY")
    print(f"{'─' * 55}")
    print(f"  {'Strategy':<35}  {'Macro F1':>8}  {'Dir. Err':>8}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}")
    for name, preds in configs:
        f1  = f1_score(y_test, preds, labels=CLASS_ORDER, average="macro", zero_division=0)
        de  = sum(
            1 for t, p in zip(y_test, preds)
            if (t == "Negative" and p == "Positive") or
               (t == "Positive" and p == "Negative")
        )
        print(f"  {name:<35}  {f1:>8.4f}  {de:>8}")
    print(f"{'─' * 55}")
    print("\n  NOTE: Return gate uses actual future returns and cannot be")
    print("  applied at live inference time. It shows the practical ceiling")
    print("  of the model's directional calls on this validation set.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",        type=Path,  default=Path("data/processed/labeled.csv"))
    parser.add_argument("--model",       type=Path,  default=Path("models/model.pkl"))
    parser.add_argument("--return-gate", type=float, default=RETURN_THRESHOLD,
                        help=f"Static return magnitude gate (default: {RETURN_THRESHOLD})")
    args = parser.parse_args()
    main(args.data, args.model, args.return_gate)