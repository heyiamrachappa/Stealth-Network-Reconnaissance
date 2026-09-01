#!/usr/bin/env python3
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

FEATURE_NAMES = [
    "flow_duration","flow_packet_count","flow_bytes","flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio",
    "flow_size_mean","flow_size_var","flow_interval_mean","flow_interval_var",
    "host_port_entropy","host_dst_entropy","host_dst_diversity","host_syn_ratio","host_failed_flow_ratio",
    "host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"
]

def limit_duplicates(df: pd.DataFrame, subset_cols: list, max_duplicates: int):
    if max_duplicates is None:
        return df
    # Group by feature subset, take up to max_duplicates
    return df.groupby(subset_cols).head(max_duplicates).reset_index(drop=True)

def evaluate(y_true, y_prob, y_pred):
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    }

def main():
    print("Loading cross_dataset split...")
    train_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "train.parquet")
    test_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "test.parquet")
    
    thresholds = [1, 2, 5, 10, None]
    results = {}
    
    for thresh in thresholds:
        print(f"Evaluating deduplication threshold: {thresh}")
        # Apply deduplication only to the test set to measure sensitivity, or both?
        # Standard approach for deduplication evaluation is to deduplicate the test set
        test_dedup = limit_duplicates(test_df, FEATURE_NAMES, thresh)
        
        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["label"].values
        X_test = test_dedup[FEATURE_NAMES].values
        y_test = test_dedup["label"].values
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        
        metrics = evaluate(y_test, test_prob, test_pred)
        results[str(thresh)] = {
            "test_size": len(test_dedup),
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"]
        }
    
    with open(RESULTS_DIR / "dedup_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Deduplication evaluation complete. Results written to results/dedup_evaluation.json")

if __name__ == "__main__":
    main()
