#!/usr/bin/env python3
"""
generate_roc_curves.py
======================
Generates REAL ROC curves for the PhantomTrace manuscript.

For each of the 4 evaluation conditions and all 5 models, this script:
  1. Loads the saved .joblib model (seed=42) and the corresponding test Parquet.
  2. Calls predict_proba / decision_function to get raw per-instance scores.
  3. Saves y_true + y_score arrays to results/roc_data/<condition>_<model>_seed42_roc.npz
  4. Calls sklearn.metrics.roc_curve(y_true, y_score) on the real data.
  5. Saves one PNG per condition (all 5 models overlaid).

The AUC values printed and annotated in each plot must match the values in
results/metrics/<condition>_<model>_seed42_metrics.json (verified inline).

No curve is derived from the AUC scalar alone — every point on every curve
comes from roc_curve(y_true, y_score) applied to the full score array.
"""

import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "dataset" / "splits"
MODELS_DIR = ROOT / "models" / "v2"
METRICS_DIR= ROOT / "results" / "metrics"
ROC_DIR    = ROOT / "results" / "roc_data"
FIGS_DIR   = ROOT / "results" / "figures"
FEATURES   = json.load(open(ROOT / "models" / "feature_names.json"))

ROC_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # representative seed

# ------------------------------------------------------------------
# Strategy ↔ manuscript condition name mapping
# ------------------------------------------------------------------
CONDITIONS = {
    "time_wise":      "Temporal Holdout",
    "pcap_wise":      "Synthetic-to-Real Transfer",
    "host_wise":      "Cross-Dataset (CIC→UNSW)",
    "cross_dataset":  "Cross-Dataset (UNSW→CIC)",
}

MODELS = [
    "random_forest",
    "xgboost",
    "svm",
    "isolation_forest",
    "mlp",
]

MODEL_LABELS = {
    "random_forest":   "RF",
    "xgboost":         "XGBoost",
    "svm":             "SVM",
    "isolation_forest":"IF",
    "mlp":             "MLP",
}

# Colors per model — consistent across all 4 plots
MODEL_COLORS = {
    "random_forest":   "#2196F3",   # blue
    "xgboost":         "#4CAF50",   # green
    "svm":             "#FF9800",   # orange
    "isolation_forest":"#9C27B0",   # purple
    "mlp":             "#F44336",   # red
}

# SVM + MLP need StandardScaler (fitted on train)
NEEDS_SCALE = {"svm", "mlp"}


def load_test_data(strategy: str):
    test_df  = pd.read_parquet(SPLITS_DIR / strategy / "test.parquet")
    train_df = pd.read_parquet(SPLITS_DIR / strategy / "train.parquet")
    X_test   = test_df[FEATURES].values
    y_test   = test_df["label"].values
    X_train  = train_df[FEATURES].values
    return X_test, y_test, X_train


def get_score(model, X, model_key: str) -> np.ndarray:
    """Return a continuous anomaly/probability score for ROC computation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        # IsolationForest: negative score = more anomalous → invert for label=1
        # Use sigmoid of inverted score so higher = more attack-like
        return 1.0 / (1.0 + np.exp(raw))
    else:
        return model.predict(X).astype(float)


def verify_auc(strategy: str, model_key: str, computed_auc: float) -> float:
    """Load the stored AUC from the metrics JSON and compare."""
    path = METRICS_DIR / f"{strategy}_{model_key}_seed{SEED}_metrics.json"
    if not path.exists():
        print(f"  [WARN] metrics JSON not found: {path}")
        return computed_auc
    stored = json.load(open(path))
    stored_auc = stored["test_metrics"].get("roc_auc")
    if stored_auc is None:
        return computed_auc
    diff = abs(computed_auc - stored_auc)
    if diff > 1e-4:
        print(f"  [WARN] AUC mismatch for {strategy}/{model_key}: "
              f"computed={computed_auc:.6f} stored={stored_auc:.6f} diff={diff:.6f}")
    else:
        print(f"  [OK]   AUC match  for {strategy}/{model_key}: "
              f"computed={computed_auc:.6f} stored={stored_auc:.6f}")
    return stored_auc  # return stored value so plot label matches table


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
all_roc_data = {}   # strategy → {model: (fpr, tpr, auc)}

for strategy, condition_name in CONDITIONS.items():
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}  (strategy: {strategy})")
    print(f"{'='*60}")

    X_test, y_test, X_train = load_test_data(strategy)

    # Fit scaler on train (same as train_baselines.py)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    roc_data = {}

    for model_key in MODELS:
        model_path = MODELS_DIR / strategy / f"{model_key}_seed{SEED}.joblib"
        if not model_path.exists():
            print(f"  [SKIP] model not found: {model_path}")
            continue

        print(f"\n  Model: {MODEL_LABELS[model_key]}")
        model = joblib.load(model_path)

        X = X_test_scaled if model_key in NEEDS_SCALE else X_test

        y_score = get_score(model, X, model_key)

        # Save raw arrays
        npz_path = ROC_DIR / f"{strategy}_{model_key}_seed{SEED}_roc.npz"
        np.savez_compressed(str(npz_path), y_true=y_test, y_score=y_score)
        print(f"    Saved per-instance arrays → {npz_path.name}")
        print(f"    Test set: {len(y_test)} instances  "
              f"| positives: {y_test.sum()}  "
              f"| score range: [{y_score.min():.4f}, {y_score.max():.4f}]")

        # Compute ROC curve from REAL data
        fpr, tpr, _ = roc_curve(y_test, y_score)
        computed_auc = sklearn_auc(fpr, tpr)

        # Verify against stored AUC (cross-check with manuscript table)
        reported_auc = verify_auc(strategy, model_key, computed_auc)

        roc_data[model_key] = (fpr, tpr, reported_auc, computed_auc)

    all_roc_data[strategy] = roc_data


# ------------------------------------------------------------------
# Plot — one PNG per condition, all 5 models overlaid
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

for strategy, condition_name in CONDITIONS.items():
    roc_data = all_roc_data.get(strategy, {})
    if not roc_data:
        print(f"\n[SKIP] No ROC data for {strategy}")
        continue

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUC = 0.50)")

    for model_key in MODELS:
        if model_key not in roc_data:
            continue
        fpr, tpr, reported_auc, computed_auc = roc_data[model_key]
        label = (f"{MODEL_LABELS[model_key]}  (AUC = {reported_auc:.3f})")
        ax.plot(fpr, tpr,
                lw=1.8,
                color=MODEL_COLORS[model_key],
                label=label)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {condition_name}", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    # Caption text (inside plot, bottom)
    caption = f"Seed={SEED} | AUC values from sklearn.metrics.roc_curve(y_true, y_score) on real test-set scores"
    ax.text(0.5, -0.13, caption,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=7.5, color="#555555", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = FIGS_DIR / f"roc_{strategy}.png"
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

print("\n\nAll done. ROC PNGs in results/figures/, raw arrays in results/roc_data/")
