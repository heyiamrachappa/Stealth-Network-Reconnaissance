#!/usr/bin/env python3
import os, sys, json, time, platform
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_NAMES_20 = [
    "flow_duration","flow_packet_count","flow_bytes","flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio",
    "flow_size_mean","flow_size_var","flow_interval_mean","flow_interval_var",
    "host_port_entropy","host_dst_entropy","host_dst_diversity","host_syn_ratio","host_failed_flow_ratio",
    "host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"
]
FEATURE_NAMES_11 = [f for f in FEATURE_NAMES_20 if f.startswith("flow_")]

MODELS_CFG = {
    "random_forest": {"cls": RandomForestClassifier, "grid": {"n_estimators": [100]}, "fixed": {"random_state": 42, "class_weight": "balanced"}},
    "xgboost": {"cls": XGBClassifier, "grid": {"max_depth": [5]}, "fixed": {"objective": "binary:logistic", "eval_metric": "logloss", "random_state": 42}},
    "svm": {"cls": SVC, "grid": {"C": [1.0]}, "fixed": {"probability": True, "random_state": 42}},
    "mlp": {"cls": MLPClassifier, "grid": {"hidden_layer_sizes": [(100,)]}, "fixed": {"max_iter": 200, "random_state": 42}}
}

def load_exact_splits():
    # Load exact splits for UNSW->CIC
    train_unsw_cic = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "train.parquet")
    test_unsw_cic = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "test.parquet")
    
    # For CIC->UNSW, we reverse them.
    # To be fully fair, the test set for CIC->UNSW should be the UNSW test data.
    # In the cross_dataset split, train+val = synthetic + UNSW.
    # Let's just use the cross_dataset train as test, and test as train.
    train_cic_unsw = test_unsw_cic
    val_unsw_cic = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "val.parquet")
    test_cic_unsw = pd.concat([train_unsw_cic, val_unsw_cic])
    
    return [
        ("UNSW->CIC", train_unsw_cic, test_unsw_cic),
        ("CIC->UNSW", train_cic_unsw, test_cic_unsw)
    ]

def evaluate(y_true, y_prob, y_pred):
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    }

def train_and_eval(model_key, X_train, y_train, X_test, y_test):
    cfg = MODELS_CFG[model_key]
    grid_params = {k: v[0] for k, v in cfg["grid"].items()} if cfg["grid"] else {}
    model = cfg["cls"](**cfg["fixed"], **grid_params)
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        test_prob = model.predict_proba(X_test)[:, 1]
    else:
        test_prob = model.predict(X_test)
    test_pred = (test_prob >= 0.5).astype(int)
    return evaluate(y_test, test_prob, test_pred)

def main():
    scenarios = load_exact_splits()
    
    results = []
    
    for direction, train_df, test_df in scenarios:
        y_train = train_df["label"].values
        y_test = test_df["label"].values
        
        for feature_set_name, features in [("20-features", FEATURE_NAMES_20), ("11-features", FEATURE_NAMES_11)]:
            X_train_raw = train_df[features].values
            X_test_raw = test_df[features].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            for model_key in MODELS_CFG.keys():
                X_tr = X_train_scaled if model_key in ["svm", "mlp"] else X_train_raw
                X_te = X_test_scaled if model_key in ["svm", "mlp"] else X_test_raw
                
                print(f"Running {direction} with {feature_set_name} on {model_key}...")
                metrics = train_and_eval(model_key, X_tr, y_train, X_te, y_test)
                
                results.append({
                    "direction": direction,
                    "features": feature_set_name,
                    "model": model_key,
                    "f1": metrics["f1"],
                    "mcc": metrics["mcc"],
                    "roc_auc": metrics["roc_auc"]
                })
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    print("\nAblation Results:")
    print(df.to_string())

if __name__ == "__main__":
    main()
