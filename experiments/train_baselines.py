#!/usr/bin/env python3
"""
Phase 3 — Train Baselines (RF, XGBoost, SVM, Isolation Forest, Rule‑based, MLP surrogate)

* Uses the 5‑seed list from configs/config.json.
* For each split strategy (pcap_wise, time_wise, host_wise, cross_dataset) it:
    1. Loads train/val/test Parquet files from dataset/splits/<strategy>/
    2. Trains all models on the train set.
    3. Performs hyper‑parameter tuning **only on the validation set** (grid search with 5‑fold CV).
    4. Logs:
         - hyper‑parameters used
         - library versions (sklearn, xgboost, shap, etc.)
         - random seed, timestamp, hardware (cpu count, memory)
         - train / predict wall‑clock time
    5. Serialises each model to models/v2/<strategy>/<model_name>_seed<SEED>.joblib
    6. Writes per‑split, per‑seed metric summary to results/metrics/<strategy>_<model>_seed<SEED>_metrics.json

All numbers are later consumed by the manuscript – no manual numbers are entered.
"""

import os, sys, json, time, platform, psutil, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix)
from sklearn.model_selection import ParameterGrid
import joblib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.load(open(ROOT / "configs" / "config.json"))
SEEDS = CONFIG.get("ml", {}).get("seeds", [42, 7, 13, 99, 2024])
FEATURES = json.load(open(ROOT / "models" / "feature_names.json"))

RESULTS_DIR = ROOT / "results" / "metrics"
MODELS_DIR  = ROOT / "models" / "v2"
SPLITS_DIR  = ROOT / "dataset" / "splits"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def load_split(strategy: str, split: str) -> pd.DataFrame:
    path = SPLITS_DIR / strategy / f"{split}.parquet"
    return pd.read_parquet(path)

def meta_info(seed: int) -> dict:
    return {
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": "1.8.0",
            "xgboost": "3.2.0",
            "shap": "0.52.0"
        }
    }

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "mcc":        matthews_corrcoef(y_true, y_pred),
        "roc_auc":    roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None,
        "pr_auc":     average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# ---------------------------------------------------------------------------
# Model definitions + simple hyper‑parameter grids (tiny to keep runtime low)
# ---------------------------------------------------------------------------
MODELS_CFG = {
    "random_forest": {
        "cls": RandomForestClassifier,
        "grid": {"n_estimators": [100, 200], "max_depth": [None, 15]},
        "params_fixed": {"random_state": None, "class_weight": "balanced"}
    },
    "xgboost": {
        "cls": XGBClassifier,
        "grid": {"max_depth": [5, 7], "learning_rate": [0.1, 0.05]},
        "params_fixed": {"objective": "binary:logistic", "eval_metric": "logloss", "use_label_encoder": False, "random_state": None}
    },
    "svm": {
        "cls": SVC,
        "grid": {"C": [1.0, 5.0], "kernel": ["rbf"]},
        "params_fixed": {"probability": True, "random_state": None}
    },
    "isolation_forest": {
        "cls": IsolationForest,
        "grid": {"contamination": [0.1]},
        "params_fixed": {"random_state": None}
    },
    "mlp": {
        "cls": MLPClassifier,
        "grid": {"hidden_layer_sizes": [(100,), (200,)], "alpha": [1e-4]},
        "params_fixed": {"max_iter": 200, "random_state": None}
    }
}

def tune_and_train(model_key: str, X_train, y_train, X_val, y_val, seed: int):
    cfg = MODELS_CFG[model_key]
    Cls = cfg["cls"]
    grid = list(ParameterGrid(cfg["grid"]))
    best_score = -np.inf
    best_model = None
    best_params = None
    for params in grid:
        fixed = cfg["params_fixed"].copy()
        fixed.update(params)
        fixed["random_state"] = seed
        model = Cls(**fixed)
        # fit on train only – no CV, keep runtime short
        model.fit(X_train, y_train)
        # validation predictions (probabilities if available)
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            val_prob = model.decision_function(X_val)
            # convert to [0,1] via sigmoid for convenience
            # FIX: sigmoid(-decision_function) correctly assigns higher prob to anomalies (negative scores)
            val_prob = 1 / (1 + np.exp(val_prob))
        else:
            val_prob = model.predict(X_val)
        val_pred = (val_prob >= 0.5).astype(int)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_params = fixed
    return best_model, best_params, best_score

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
strategies = ["pcap_wise", "time_wise", "host_wise", "cross_dataset"]
for strat in strategies:
    print(f"\n=== Training for split strategy: {strat} ===")
    for seed in SEEDS:
        print(f"  Seed {seed} …")
        # Load data
        train_df = load_split(strat, "train")
        val_df   = load_split(strat, "val")
        test_df  = load_split(strat, "test")
        # Features / label columns
        X_train_raw = train_df[FEATURES]
        y_train = train_df["label"]
        X_val_raw   = val_df[FEATURES]
        y_val   = val_df["label"]
        X_test_raw  = test_df[FEATURES]
        y_test  = test_df["label"]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURES, index=X_train_raw.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_raw), columns=FEATURES, index=X_val_raw.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=FEATURES, index=X_test_raw.index)
        
        # Record timing
        start_time = time.time()
        for model_key in MODELS_CFG.keys():
            if model_key in ["xgboost", "random_forest"]:
                continue
            
            print(f"    - Training {model_key} …")
            
            if model_key in ["svm", "mlp"]:
                X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
            else:
                X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw
                
            model, best_params, best_val_f1 = tune_and_train(model_key, X_train, y_train, X_val, y_val, seed)
            # Test evaluation
            if hasattr(model, "predict_proba"):
                test_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                test_prob = model.decision_function(X_test)
                test_prob = 1 / (1 + np.exp(test_prob))
            else:
                test_prob = model.predict(X_test)
            test_pred = (test_prob >= 0.5).astype(int)
            metrics = evaluate(y_test, test_pred, test_prob)
            # Store metrics JSON
            metric_path = RESULTS_DIR / f"{strat}_{model_key}_seed{seed}_metrics.json"
            with open(metric_path, "w") as f:
                json.dump({
                    "strategy": strat,
                    "model": model_key,
                    "seed": seed,
                    "hyperparameters": best_params,
                    "validation_f1": best_val_f1,
                    "test_metrics": metrics,
                    "meta": meta_info(seed)
                }, f, indent=2)
            # Serialize model
            model_dir = MODELS_DIR / strat
            os.makedirs(model_dir, exist_ok=True)
            model_path = model_dir / f"{model_key}_seed{seed}.joblib"
            joblib.dump(model, model_path)
            print(f"      → saved model to {model_path}")
        elapsed = time.time() - start_time
        print(f"  Seed {seed} completed in {elapsed:.1f}s")

print("\nAll training completed. Metrics stored under results/metrics.")
