"""
create_plots.py
------------------
Generates and saves model evaluation and data analysis figures:
1. Class Distribution Bar Chart
2. Confusion Matrix Heatmaps (for evaluated models)
3. Feature Importance Bar Chart (Random Forest / Model Weights)
4. Aggregate Volatility Index vs. AMZN Price Overlay Chart

Output directory: results/plots/
Usage:
    python3 src/create_plots.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Config ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "labeled.csv"
METRICS_PATH = REPO_ROOT / "models" / "metrics.json"
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"
OUTPUT_DIR = REPO_ROOT / "results" / "plots"


# ── Class Distribution ────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out_dir: Path):
    """Plots the distribution of target labels (Negative, Neutral, Positive)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    order = ["Negative", "Neutral", "Positive"]
    counts = df["label"].value_counts().reindex(order).fillna(0)
    colors = ["#e74c3c", "#7f8c8d", "#2ecc71"]
    
    bars = ax.bar(order, counts, color=colors, edgecolor="black", alpha=0.85)
    
    total = len(df)
    for bar in bars:
        yval = bar.get_height()
        percentage = (yval / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + (total * 0.01), 
            f"{int(yval)} ({percentage:.1f}%)", 
            ha="center", 
            fontweight="bold"
        )
        
    ax.set_title("Target Label Distribution (AMZN Next-Week Movement)")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Weeks")
    ax.set_ylim(0, counts.max() * 1.15)
    
    out_file = out_dir / "class_distribution.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


# ── Heatmaps ──────────────────────────────────────────────────────────────

def plot_confusion_matrices(metrics: dict, out_dir: Path):
    """Generates heatmaps using pure Matplotlib's plt.imshow."""
    for model_key, data in metrics.items():
        model_name = data.get("model_name", model_key)
        conf_matrix = np.array(data["confusion_matrix"])
        labels = data["confusion_matrix_labels"]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        cax = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                val = conf_matrix[i, j]
                text_color = "white" if val > thresh else "black"
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    color=text_color, fontweight="bold", fontsize=12
                )
        
        ax.grid(False)
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        
        filename = f"confusion_matrix_{model_key.lower()}.png"
        out_file = out_dir / filename
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved: {out_file}")


# ── Feature Importance ────────────────────────────────────────────────

def plot_feature_importance(model_path: Path, out_dir: Path):        
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
        
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle.get("model_name", "Trained Model")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ylabel = "Relative Importance"
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
        ylabel = "Mean Absolute Coefficient Weight"
    else:
        return

    feat_df = pd.DataFrame({"Feature": feature_columns,"Importance": importances}).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#3498db", edgecolor="black", alpha=0.85)
    
    ax.set_title(f"Feature Importance ({model_name})")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Feature Name")
    plt.grid(True, axis="x")
    fig.tight_layout()
    
    out_file = out_dir / "feature_importance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


# ── Aggregate Volatility vs AMZN Price Overlay ────────────────────────

def plot_aggregate_vs_price(df: pd.DataFrame, out_dir: Path):
    """Plots a dual-axis line chart overlaying Aggregate Volatility against AMZN Price."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Date")
    ax1.set_ylabel("AMZN Close Price ($USD)", color="#27ae60", fontweight="bold")
    line1 = ax1.plot(df["date"], df["amzn_close"], color="#27ae60", linewidth=1.8, label="AMZN Close Price")
    ax1.tick_params(axis="y", labelcolor="#27ae60")

    ax2 = ax1.twinx()  
    ax2.set_ylabel("Aggregate Volatility Index", color="#e67e22", fontweight="bold")
    line2 = ax2.plot(df["date"], df["aggregate_vol_index"], color="#e67e22", linewidth=1.2, alpha=0.8, label="Aggregate Vol Index")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax2.grid(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("AMZN Closing Price vs. Aggregate Volatility Index Over Time")
    plt.grid(True)
    
    out_file = out_dir / "aggregate_vol_vs_amzn_price.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at '{DATA_PATH}'. Run label_data.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    
    print("Generating figures (pure Matplotlib)...")
    plot_class_distribution(df, OUTPUT_DIR)
    
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        plot_confusion_matrices(metrics, OUTPUT_DIR)
    else:
        print(f"Skipping confusion matrices: {METRICS_PATH} not found.")

    plot_feature_importance(MODEL_PATH, OUTPUT_DIR)
    plot_aggregate_vs_price(df, OUTPUT_DIR)
    
    print(f"\nAll plots successfully exported to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()