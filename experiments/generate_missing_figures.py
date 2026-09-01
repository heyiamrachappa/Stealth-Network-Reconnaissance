#!/usr/bin/env python3
import json, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from joblib import load
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models" / "v2"
SPLITS_DIR = ROOT / "dataset" / "splits" / "cross_dataset"

model_order = ["random_forest", "xgboost", "svm", "isolation_forest", "mlp"]
model_labels = {"random_forest": "RF", "xgboost": "XGB", "svm": "SVM",
                "isolation_forest": "IF", "mlp": "MLP"}

# 1. Confusion Matrices for cross-dataset
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Confusion Matrices — Cross-Dataset Split (Aggregated across 5 seeds)", fontsize=14, fontweight="bold")

for i, m in enumerate(model_order):
    ax = axes[i]
    pattern = str(METRICS_DIR / f"cross_dataset_{m}_seed*_metrics.json")
    files = glob.glob(pattern)
    if files:
        cm_sum = np.zeros((2, 2), dtype=int)
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            cm_sum += np.array(d["test_metrics"]["confusion_matrix"])
        
        sns.heatmap(cm_sum, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Recon"], yticklabels=["Benign", "Recon"])
        ax.set_title(model_labels[m], fontsize=11)
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Actual")
    else:
        ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
cm_path = FIGURES_DIR / "confusion_matrices_cross_dataset.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved confusion matrices → {cm_path}")

# 2. ROC/PR curves
# Load the test dataset
test_path = SPLITS_DIR / "test.parquet"
if not test_path.exists():
    print("Cannot find test.parquet for cross_dataset. Skipping ROC/PR.")
else:
    test_df = pd.read_parquet(test_path)
    with open(ROOT / "models" / "feature_names.json", "r") as f:
        features = json.load(f)
    X_test = test_df[features].fillna(0)
    y_test = test_df["label"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC and Precision-Recall Curves (Cross-Dataset, Seed 42 Models)", fontsize=14, fontweight="bold")
    
    train_path = SPLITS_DIR / "train.parquet"
    if not train_path.exists():
        print("Train split not found. Cannot retrain models.")
    else:
        train_df = pd.read_parquet(train_path)
        X_train = train_df[features].fillna(0)
        y_train = train_df["label"]

        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import IsolationForest
        from sklearn.neural_network import MLPClassifier

        def get_model(m_name):
            if m_name == "random_forest": return RandomForestClassifier(random_state=42)
            if m_name == "xgboost": return XGBClassifier(random_state=42)
            if m_name == "svm": return SVC(probability=True, random_state=42)
            if m_name == "isolation_forest": return IsolationForest(random_state=42)
            if m_name == "mlp": return MLPClassifier(random_state=42)

        for m in model_order:
            model = get_model(m)
            print(f"Retraining {m}...")
            model.fit(X_train, y_train if m != "isolation_forest" else X_train)
            
            if m == "isolation_forest":
                scores = model.decision_function(X_test)
                probs = 1 / (1 + np.exp(scores)) # Invert scores for IF
            elif hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)
                probs = 1 / (1 + np.exp(-scores))
            
        # ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{model_labels[m]} (AUC = {roc_auc:.3f})")
        
        # PR
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(rec, prec)
        ax2.plot(rec, prec, label=f"{model_labels[m]} (AUC = {pr_auc:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    roc_pr_path = FIGURES_DIR / "roc_pr_curves_cross_dataset.png"
    plt.savefig(roc_pr_path, dpi=150)
    plt.close()
    print(f"Saved ROC/PR curves → {roc_pr_path}")

print("Figure generation script complete.")
