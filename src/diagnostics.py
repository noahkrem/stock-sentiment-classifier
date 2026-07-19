import json
from pathlib import Path

import pandas as pd


def get_confusion_summary(data):
    """Parses raw confusion matrix into a human-readable list of errors."""
    matrix = data['confusion_matrix']
    labels = data['confusion_matrix_labels']
    summary = []
    
    for i, actual_row in enumerate(matrix):
        for j, count in enumerate(actual_row):
            if i != j and count > 0: # Only report errors
                summary.append(f"- {labels[i]} was misclassified as {labels[j]} ({count} times)")
    return summary if summary else ["No misclassifications found (Perfect!)"]

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "models" / "metrics.json"
DATA_PATH = REPO_ROOT / "data" / "processed" / "labeled.csv"

with open(METRICS_PATH) as f:
    metrics = json.load(f)

df = pd.read_csv(DATA_PATH)

print("# Model Diagnostics Report\n")
print("### Return Distribution")
print("#### Describe")
print(df["amzn_next_return"].describe().to_string())
print("\n#### Quantiles")
print(df["amzn_next_return"].quantile([0.25, 0.33, 0.50, 0.67, 0.75]).to_string())
print("\n")

for model_name, m in metrics.items():
    print(f"## Model: {m['model_name']}")
    
    # 1. Performance Overview
    print("### Summary Metrics")
    print(f"| Metric | Value |")
    print(f"| :--- | :--- |")
    print(f"| Accuracy | {m['accuracy']:.4f} |")
    print(f"| Macro F1 | {m['macro_f1']:.4f} |")
    print("\n")

    # 2. Precision/Recall/F1 per class
    print("### Per-Class Performance")
    print("| Class | Precision | Recall | F1 | Support |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for cls, stats in m['classification_report'].items():
        if cls not in ["accuracy", "macro avg", "weighted avg"]:
            print(f"| {cls} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1-score']:.3f} | {stats['support']} |")
    print("\n")

    # 3. Error Analysis
    print("### Misclassification Patterns")
    for error in get_confusion_summary(m):
        print(error)
    print("\n---\n")