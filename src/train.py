
# train.py

# Trains and compares two classifiers (Logistic Regression, Random Forest) to
# predict next-week AMZN price movement direction (Negative / Neutral /
# Positive) from volatility-index features.

# Expected input: labeled data CSV
# Usage:
# python src/train.py --data data/processed/labeled.csv --out models/model.pkl <- specify paths depending on where you run, I ran this in the root directory


import argparse # allow for --data and --out for easier development
import json 
from pathlib import Path # path objects easy to deal with
import pickle # for saving model into .pkl file
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42 # random seed so train and test can be reproduced
TEST_SIZE = 0.20 # 20% of data left out for testing
TARGET_COL = "label" # target column to predict

# Excluded columns:
# adding in score_range yields 1.0 accuracy, adding in amzn_close yields worse f1 by about 0.1 for logistic regression but slightly better performance for random forest
# adding in amzn_next_return yields ~0.96 for accuracy and f1, adding in date yields marginally worse results
NON_FEATURE_COLS = {"date", TARGET_COL, "score_range",  "amzn_next_return", "amzn_close"} 


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]) # change dates from text to datetime objs
    df = df.sort_values("date").reset_index(drop=True) # make sure rows are in order and renumber
    return df


def get_feature_columns(df: pd.DataFrame) -> list:

    # Every numeric column that isn't in NON_FEATURE_COLS becomes a feature.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLS]


def to_jsonable(obj): # had a bug with this in development seems fixed now ty claude
    """Recursively convert numpy types to plain python so json.dump doesn't choke."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate(model, X_test, y_test, class_order, model_name: str) -> dict:
    y_pred = model.predict(X_test) # ask model to predict 

    acc = accuracy_score(y_test, y_pred) # get accuracy

    # get confusion matrix, rows = true class and cols = predicted class
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in class_order],
        columns=[f"pred_{c}" for c in class_order],
    )

    report = classification_report(
        y_test, y_pred, labels=class_order, output_dict=True, zero_division=0
    )

    # compute f1 seperately for each class, then average the three scores with equal weight
    macro_f1 = f1_score(y_test, y_pred, labels=class_order, average="macro", zero_division=0)

    # printing results got claude to make these clean and easy to read in terminal
    print(f"\n{'=' * 60}")
    print(model_name)
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nConfusion matrix:")
    print(cm_df.to_string())
    print("\nPer-class F1 / precision / recall:")
    for cls in class_order:
        r = report[cls]
        print(
            f"  {cls:10s} f1={r['f1-score']:.4f}  "
            f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  support={int(r['support'])}"
        )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_order,
        "per_class_f1": {cls: report[cls]["f1-score"] for cls in class_order},
        "classification_report": report,
    }


def main(data_path: Path, out_path: Path):
    df = load_data(data_path) # load labeled csv
    feature_cols = get_feature_columns(df) # get feature columns

    before = len(df)
    # drop rows where a feature or label is missing then renumber
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing values in features/target.")

    print(f"Dataset shape: {df.shape}") # shape is rows, cols
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print("\nClass distribution:")
    print(df[TARGET_COL].value_counts())

    expected = ["Negative", "Neutral", "Positive"]
    class_order = expected if set(df[TARGET_COL].unique()) == set(expected) else sorted(
        df[TARGET_COL].unique()
    )

    X = df[feature_cols].to_numpy() # feature matrix as a numpy array
    y = df[TARGET_COL].to_numpy() # target vector is the string labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y  # stratify keeps porportions roughly equal for positive, negative, neutral
    ) 

    scaler = StandardScaler() # rescale every feature to mean 0 and std dev. 1
    X_train_scaled = scaler.fit_transform(X_train) # learn mean,std from thje trainging data then apply scaling
    X_test_scaled = scaler.transform(X_test) # apply same learned scaling to test data

    # Penalize the model for defaulting to the majority class (Neutral) by weighting classes inversely to their frequency in the training data. 
    # This helps the model learn to predict the minority classes (Positive and Negative) better.
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE) # max iterations set to 1000 (can set lower if its taking too long however it may not converge)
    logreg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE) # n_estimators=300 is the number of individual decision trees in the forest and predictions are the majority vote across all 300 trees
    rf.fit(X_train_scaled, y_train) # setting n_estimators higher gets marginally better results for instance 600 (need to experiment with this more)

    # evaluate results
    results = {
        "LogisticRegression": evaluate(logreg, X_test_scaled, y_test, class_order, "Logistic Regression"),
        "RandomForest": evaluate(rf, X_test_scaled, y_test, class_order, "Random Forest"),
    }

    # pick whichever model performs better
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = logreg if best_name == "LogisticRegression" else rf
    print(f"\nBest model: {best_name}  (macro F1 = {results[best_name]['macro_f1']:.4f})") 

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "scaler": scaler,
                "feature_columns": feature_cols,
                "classes": class_order,
            },
            f,
        )
    print(f"Saved best model + scaler -> {out_path}")

    # save metrics json with model.pkl
    metrics_path = out_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f"Saved full metrics -> {metrics_path}")


# script for running
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/labeled.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/model.pkl"))
    args = parser.parse_args()
    main(args.data, args.out)
