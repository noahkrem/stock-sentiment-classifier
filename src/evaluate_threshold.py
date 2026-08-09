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

ENHANCEMENTS:
  - Now generates PNG visualizations of all metrics and saves to results/plots/
  - Text outputs are preserved alongside graphical outputs

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

import matplotlib.pyplot as plt
import seaborn as sns

# ── Config (must match train.py) ──────────────────────────────────────────────

RANDOM_STATE     = 23
TEST_SIZE        = 0.10
TARGET_COL       = "label"
NON_FEATURE_COLS = {"date", TARGET_COL, "score_range", "amzn_next_return", "amzn_close"}

RETURN_THRESHOLD = 0.02   # ±2% practical gate
CONFIDENCE_RANGE = np.arange(0.33, 0.65, 0.05)

CLASS_ORDER = ["Negative", "Neutral", "Positive"]

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ── Visualization Helpers ─────────────────────────────────────────────────────

def ensure_plots_dir(plots_dir: Path):
    """Create results/plots/ directory if it doesn't exist."""
    plots_dir.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, class_order: list, title: str, save_path: Path):
    """Plot and save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_order, 
        yticklabels=class_order,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved confusion matrix: {save_path}")


def plot_confidence_sweep(thresholds: list, accuracies: list, f1_scores: list, 
                          dir_errors: list, neutral_pcts: list, 
                          saved_threshold, best_threshold, save_path: Path):
    """Plot confidence gate sweep results as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Confidence Gate Sweep Analysis', fontsize=16, fontweight='bold', y=1.00)
    
    # Accuracy
    axes[0, 0].plot(thresholds, accuracies, 'o-', color='steelblue', linewidth=2, markersize=6)
    if saved_threshold:
        saved_idx = np.argmin(np.abs(np.array(thresholds) - saved_threshold))
        axes[0, 0].axvline(saved_threshold, color='green', linestyle='--', 
                          linewidth=2, label=f'Saved ({saved_threshold:.2f})', alpha=0.7)
        axes[0, 0].plot(thresholds[saved_idx], accuracies[saved_idx], 'g^', markersize=10)
    best_idx = np.argmax(f1_scores)
    axes[0, 0].axvline(best_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Best F1 ({best_threshold:.2f})', alpha=0.7)
    axes[0, 0].plot(best_threshold, accuracies[best_idx], 'r^', markersize=10)
    axes[0, 0].set_xlabel('Confidence Threshold', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Accuracy vs Confidence Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Macro F1
    axes[0, 1].plot(thresholds, f1_scores, 'o-', color='darkgreen', linewidth=2, markersize=6)
    if saved_threshold:
        saved_idx = np.argmin(np.abs(np.array(thresholds) - saved_threshold))
        axes[0, 1].axvline(saved_threshold, color='green', linestyle='--', 
                          linewidth=2, label=f'Saved ({saved_threshold:.2f})', alpha=0.7)
        axes[0, 1].plot(thresholds[saved_idx], f1_scores[saved_idx], 'g^', markersize=10)
    axes[0, 1].axvline(best_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Best F1 ({best_threshold:.2f})', alpha=0.7)
    axes[0, 1].plot(best_threshold, max(f1_scores), 'r^', markersize=10)
    axes[0, 1].set_xlabel('Confidence Threshold', fontweight='bold')
    axes[0, 1].set_ylabel('Macro F1', fontweight='bold')
    axes[0, 1].set_title('Macro F1 vs Confidence Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Directional Errors
    axes[1, 0].plot(thresholds, dir_errors, 'o-', color='darkred', linewidth=2, markersize=6)
    if saved_threshold:
        saved_idx = np.argmin(np.abs(np.array(thresholds) - saved_threshold))
        axes[1, 0].axvline(saved_threshold, color='green', linestyle='--', 
                          linewidth=2, label=f'Saved ({saved_threshold:.2f})', alpha=0.7)
        axes[1, 0].plot(thresholds[saved_idx], dir_errors[saved_idx], 'g^', markersize=10)
    axes[1, 0].axvline(best_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Best F1 ({best_threshold:.2f})', alpha=0.7)
    axes[1, 0].plot(best_threshold, dir_errors[best_idx], 'r^', markersize=10)
    axes[1, 0].set_xlabel('Confidence Threshold', fontweight='bold')
    axes[1, 0].set_ylabel('Directional Errors', fontweight='bold')
    axes[1, 0].set_title('Directional Errors vs Confidence Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Neutral Percentage
    axes[1, 1].plot(thresholds, neutral_pcts, 'o-', color='purple', linewidth=2, markersize=6)
    if saved_threshold:
        saved_idx = np.argmin(np.abs(np.array(thresholds) - saved_threshold))
        axes[1, 1].axvline(saved_threshold, color='green', linestyle='--', 
                          linewidth=2, label=f'Saved ({saved_threshold:.2f})', alpha=0.7)
        axes[1, 1].plot(thresholds[saved_idx], neutral_pcts[saved_idx], 'g^', markersize=10)
    axes[1, 1].axvline(best_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Best F1 ({best_threshold:.2f})', alpha=0.7)
    axes[1, 1].plot(best_threshold, neutral_pcts[best_idx], 'r^', markersize=10)
    axes[1, 1].set_xlabel('Confidence Threshold', fontweight='bold')
    axes[1, 1].set_ylabel('Neutral Predictions (%)', fontweight='bold')
    axes[1, 1].set_title('Neutral Predictions vs Confidence Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved confidence sweep plot: {save_path}")


def plot_strategy_comparison(strategies: list, f1_scores: list, accuracies: list, 
                            dir_errors: list, save_path: Path):
    """Plot comparison of all strategies as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x_pos = np.arange(len(strategies))
    
    # Macro F1 Comparison
    bars1 = axes[0].bar(x_pos, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    axes[0].set_title('Macro F1 Score by Strategy', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars1, f1_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy Comparison
    bars2 = axes[1].bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy by Strategy', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(strategies, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars2, accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Directional Errors Comparison
    bars3 = axes[2].bar(x_pos, dir_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Directional Errors', fontsize=12, fontweight='bold')
    axes[2].set_title('Directional Errors by Strategy', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(strategies, rotation=45, ha='right')
    axes[2].grid(True, axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars3, dir_errors):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved strategy comparison plot: {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_order: list, title: str, save_path: Path):
    """Plot per-class precision, recall, and F1 as PNG."""
    report = classification_report(
        y_true, y_pred, labels=class_order, output_dict=True, zero_division=0
    )
    
    precisions = [report[cls]['precision'] for cls in class_order]
    recalls = [report[cls]['recall'] for cls in class_order]
    f1s = [report[cls]['f1-score'] for cls in class_order]
    
    x_pos = np.arange(len(class_order))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x_pos - width, precisions, width, label='Precision', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos, recalls, width, label='Recall', 
                   color='darkorange', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x_pos + width, f1s, width, label='F1-Score', 
                   color='darkgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_order)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved per-class metrics plot: {save_path}")


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
    return acc, macro_f1, dir_errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: Path, model_path: Path, return_gate: float):
    plots_dir = Path("results/plots")
    ensure_plots_dir(plots_dir)
    
    bundle = load_bundle(model_path)
    model       = bundle["model"]
    scaler      = bundle["scaler"]
    model_name  = bundle.get("model_name", "Unknown")
    saved_threshold = bundle.get("confidence_threshold", None)

    print(f"Model          : {model_name}")
    print(f"Saved threshold: {saved_threshold}")
    print(f"Return gate    : ±{return_gate*100:.1f}%")
    print(f"Plots dir      : {plots_dir}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X_test_scaled, y_test, ret_test = rebuild_test_split(df, feature_cols, scaler)

    proba_matrix = model.predict_proba(X_test_scaled)
    y_raw        = [CLASS_ORDER[int(np.argmax(p))] for p in proba_matrix]

    # ── 1. Baseline (no gate) ─────────────────────────────────────────────────
    print_metrics(y_test, y_raw, "BASELINE — no gate", CLASS_ORDER)
    plot_confusion_matrix(y_test, y_raw, CLASS_ORDER, 
                         "Baseline (No Gate)", plots_dir / "01_baseline_confusion.png")
    plot_per_class_metrics(y_test, y_raw, CLASS_ORDER,
                          "Baseline (No Gate) - Per-Class Metrics",
                          plots_dir / "01_baseline_per_class.png")

    # ── 2. Return gate (actual returns used to filter low-magnitude calls) ────
    y_return_gated = apply_return_gate(y_raw, ret_test, return_gate)
    print_metrics(
        y_test, y_return_gated,
        f"RETURN GATE — actual |return| < {return_gate*100:.0f}% → Neutral",
        CLASS_ORDER,
    )
    plot_confusion_matrix(y_test, y_return_gated, CLASS_ORDER,
                         f"Return Gate (±{return_gate*100:.0f}%)",
                         plots_dir / "02_return_gate_confusion.png")
    plot_per_class_metrics(y_test, y_return_gated, CLASS_ORDER,
                          f"Return Gate (±{return_gate*100:.0f}%) - Per-Class Metrics",
                          plots_dir / "02_return_gate_per_class.png")

    # ── 3. Confidence gate sweep ──────────────────────────────────────────────
    print(f"\n{'─' * 66}")
    print("  CONFIDENCE GATE SWEEP")
    print(f"{'─' * 66}")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Macro F1':>10}  "
          f"{'Dir. Errors':>12}  {'Neutral%':>9}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*9}")

    best_conf_f1  = -1.0
    best_conf_t   = float(CONFIDENCE_RANGE[0])
    
    # Track sweep metrics for plotting
    sweep_thresholds = []
    sweep_accuracies = []
    sweep_f1s = []
    sweep_dir_errors = []
    sweep_neutral_pcts = []

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
        
        sweep_thresholds.append(t)
        sweep_accuracies.append(acc)
        sweep_f1s.append(macro_f1)
        sweep_dir_errors.append(dir_err)
        sweep_neutral_pcts.append(neut_pct)
        
        saved_marker = " ◀ saved" if saved_threshold and abs(t - saved_threshold) < 0.001 else ""
        best_marker  = " ◀ best"  if macro_f1 > best_conf_f1 else ""
        print(f"  {t:>10.2f}  {acc:>10.4f}  {macro_f1:>10.4f}  "
              f"{dir_err:>12}  {neut_pct:>8.1f}%{saved_marker}{best_marker}")
        if macro_f1 > best_conf_f1:
            best_conf_f1 = macro_f1
            best_conf_t  = float(t)

    # Plot confidence sweep
    plot_confidence_sweep(sweep_thresholds, sweep_accuracies, sweep_f1s,
                         sweep_dir_errors, sweep_neutral_pcts,
                         saved_threshold, best_conf_t,
                         plots_dir / "03_confidence_sweep.png")

    # ── 4. Best confidence gate (full metrics) ────────────────────────────────
    y_best_conf = apply_confidence_gate(proba_matrix, CLASS_ORDER, best_conf_t)
    print_metrics(
        y_test, y_best_conf,
        f"BEST CONFIDENCE GATE (threshold = {best_conf_t:.2f})",
        CLASS_ORDER,
    )
    plot_confusion_matrix(y_test, y_best_conf, CLASS_ORDER,
                         f"Best Confidence Gate (threshold={best_conf_t:.2f})",
                         plots_dir / "04_best_confidence_confusion.png")
    plot_per_class_metrics(y_test, y_best_conf, CLASS_ORDER,
                          f"Best Confidence Gate (threshold={best_conf_t:.2f}) - Per-Class Metrics",
                          plots_dir / "04_best_confidence_per_class.png")

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
    plot_confusion_matrix(y_test, y_combined, CLASS_ORDER,
                         f"Combined (Confidence {best_conf_t:.2f} + Return Gate ±{return_gate*100:.0f}%)",
                         plots_dir / "05_combined_confusion.png")
    plot_per_class_metrics(y_test, y_combined, CLASS_ORDER,
                          f"Combined (Confidence {best_conf_t:.2f} + Return Gate ±{return_gate*100:.0f}%) - Per-Class Metrics",
                          plots_dir / "05_combined_per_class.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    configs = [
        ("Baseline", y_raw),
        (f"Return gate ±{return_gate*100:.0f}%", y_return_gated),
        (f"Confidence gate {best_conf_t:.2f}", y_best_conf),
        ("Combined", y_combined),
    ]
    print(f"\n{'─' * 55}")
    print("  SUMMARY")
    print(f"{'─' * 55}")
    print(f"  {'Strategy':<35}  {'Macro F1':>8}  {'Dir. Err':>8}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}")
    
    summary_strategies = []
    summary_f1s = []
    summary_accuracies = []
    summary_dir_errors = []
    
    for name, preds in configs:
        f1  = f1_score(y_test, preds, labels=CLASS_ORDER, average="macro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        de  = sum(
            1 for t, p in zip(y_test, preds)
            if (t == "Negative" and p == "Positive") or
               (t == "Positive" and p == "Negative")
        )
        summary_strategies.append(name)
        summary_f1s.append(f1)
        summary_accuracies.append(acc)
        summary_dir_errors.append(de)
        print(f"  {name:<35}  {f1:>8.4f}  {de:>8}")
    print(f"{'─' * 55}")
    
    # Plot strategy comparison
    plot_strategy_comparison(summary_strategies, summary_f1s, summary_accuracies,
                            summary_dir_errors, plots_dir / "06_summary_comparison.png")
    
    print("\n  NOTE: Return gate uses actual future returns and cannot be")
    print("  applied at live inference time. It shows the practical ceiling")
    print("  of the model's directional calls on this validation set.\n")
    print(f"  All plots saved to: {plots_dir.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",        type=Path,  default=Path("data/processed/labeled.csv"))
    parser.add_argument("--model",       type=Path,  default=Path("models/model.pkl"))
    parser.add_argument("--return-gate", type=float, default=RETURN_THRESHOLD,
                        help=f"Static return magnitude gate (default: {RETURN_THRESHOLD})")
    args = parser.parse_args()
    main(args.data, args.model, args.return_gate)