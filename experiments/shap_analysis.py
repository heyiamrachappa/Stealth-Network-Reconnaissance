#!/usr/bin/env python3
"""
Phase 6 — SHAP Explainability Analysis

Generates SHAP summary plots and feature-importance rankings for the best
model (by validation F1) on each split strategy.  Uses TreeExplainer for
tree-based models and KernelExplainer as fallback.

Outputs:
  results/shap/<strategy>_<model>_shap_summary.png
  results/shap/<strategy>_<model>_shap_values.json
"""

import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
SHAP_DIR    = ROOT / "results" / "shap"
SPLITS_DIR  = ROOT / "dataset" / "splits"
MODELS_DIR  = ROOT / "models" / "v2"
FEATURES    = json.load(open(ROOT / "models" / "feature_names.json"))

os.makedirs(SHAP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Find best model per strategy (highest mean test F1 across seeds)
# ---------------------------------------------------------------------------
import glob
from collections import defaultdict

def find_best_models():
    """Return dict: strategy -> (model_name, seed_with_best_val_f1)"""
    scores = defaultdict(lambda: defaultdict(list))
    for fp in glob.glob(str(METRICS_DIR / "*_metrics.json")):
        with open(fp) as f:
            d = json.load(f)
        strat = d["strategy"]
        model = d["model"]
        f1 = d["test_metrics"]["f1"]
        scores[strat][model].append((f1, d["seed"]))

    best = {}
    for strat, models in scores.items():
        best_model, best_seed, best_f1 = None, None, -1
        for model, vals in models.items():
            mean_f1 = np.mean([v[0] for v in vals])
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_model = model
                # pick the seed closest to the mean
                best_seed = min(vals, key=lambda x: abs(x[0] - mean_f1))[1]
        best[strat] = (best_model, best_seed)
    return best

# ---------------------------------------------------------------------------
# Generate SHAP analysis for each strategy
# ---------------------------------------------------------------------------
best_models = find_best_models()
print("Best models per strategy:")
for strat, (model, seed) in best_models.items():
    print(f"  {strat}: {model} (seed {seed})")

for strat, (model_name, seed) in best_models.items():
    print(f"\n=== SHAP analysis: {strat} / {model_name} ===")

    # Load model
    model_path = MODELS_DIR / strat / f"{model_name}_seed{seed}.joblib"
    model = joblib.load(model_path)

    # Load test data (subsample for speed)
    test_df = pd.read_parquet(SPLITS_DIR / strat / "test.parquet")
    X_test = test_df[FEATURES]
    max_samples = min(500, len(X_test))
    X_sample = X_test.sample(n=max_samples, random_state=seed)

    # Choose explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers, TreeExplainer may return a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
    except Exception:
        # Fallback: KernelExplainer (slow but universal)
        print(f"  Using KernelExplainer (slow) for {model_name}…")
        bg = shap.sample(X_sample, min(50, len(X_sample)))
        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_sample.iloc[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(X_sample.iloc[:100])
            X_sample = X_sample.iloc[:100]

    # Summary plot
    fig_path = SHAP_DIR / f"{strat}_{model_name}_shap_summary.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved summary plot → {fig_path}")

    # Feature importance ranking (mean |SHAP|)
    sv = np.array(shap_values)
    # If 3-D (e.g. n_samples × n_features × n_outputs), collapse outputs
    if sv.ndim == 3:
        sv = sv.mean(axis=2)
    mean_abs = np.abs(sv).mean(axis=0).flatten()
    ranking = sorted(zip(FEATURES, mean_abs.tolist()), key=lambda x: -x[1])
    json_path = SHAP_DIR / f"{strat}_{model_name}_shap_values.json"
    with open(json_path, "w") as f:
        json.dump({
            "strategy": strat,
            "model": model_name,
            "seed": seed,
            "feature_importance_mean_abs_shap": [
                {"feature": feat, "importance": round(val, 6)} for feat, val in ranking
            ]
        }, f, indent=2)
    print(f"  Saved SHAP values → {json_path}")

print("\nSHAP analysis complete.")
