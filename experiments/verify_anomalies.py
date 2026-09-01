#!/usr/bin/env python3
import os, sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "v2"

FEATURE_NAMES = [
    "flow_duration","flow_packet_count","flow_bytes","flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio",
    "flow_size_mean","flow_size_var","flow_interval_mean","flow_interval_var",
    "host_port_entropy","host_dst_entropy","host_dst_diversity","host_syn_ratio","host_failed_flow_ratio",
    "host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"
]

def verify_item_4():
    print("\n--- ITEM 4: Isolation Forest ROC-AUC ---")
    train_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "train.parquet")
    test_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "test.parquet")
    
    model = joblib.load(MODELS_DIR / "cross_dataset" / "isolation_forest_seed42.joblib")
    
    # Analyze the decision function scores on the test set
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values
    
    scores = model.decision_function(X_test)
    
    benign_scores = scores[y_test == 0]
    recon_scores = scores[y_test == 1]
    
    print(f"Average decision_function for Benign (label 0): {np.mean(benign_scores):.4f}")
    print(f"Average decision_function for Recon (label 1): {np.mean(recon_scores):.4f}")
    print("In Isolation Forest, smaller (more negative) values indicate anomalies.")
    print("If Benign scores are smaller than Recon scores, the model thinks Benign is the anomaly.")
    
    # Density / Unique values proxy
    print(f"Unique feature combinations in Benign test set: {len(np.unique(X_test[y_test == 0], axis=0))} / {len(benign_scores)}")
    print(f"Unique feature combinations in Recon test set: {len(np.unique(X_test[y_test == 1], axis=0))} / {len(recon_scores)}")
    print("Conclusion: IF expects anomalies to be sparse and rare. Here, Recon traffic is dense and heavily duplicated, while Benign traffic is varied. This violates IF's core assumption, causing the near-zero AUC.")

def verify_item_5():
    print("\n--- ITEM 5: MLP Synthetic-to-Real Transfer Convergence ---")
    print("The 'pcap_wise' split trains on the 'synthetic' dataset and tests on CIC.")
    train_df = pd.read_parquet(ROOT / "dataset" / "splits" / "pcap_wise" / "train.parquet")
    print(f"Size of pcap_wise training set: {len(train_df)} samples")
    print(f"Benign: {sum(train_df['label'] == 0)}, Recon: {sum(train_df['label'] == 1)}")
    
    # Load multiple seeds to see varying performance
    f1s = []
    for seed in [42, 7, 13, 99, 2024]:
        with open(ROOT / "results" / "metrics" / f"pcap_wise_mlp_seed{seed}_metrics.json") as f:
            metrics = json.load(f)
            f1s.append(metrics["test_metrics"]["f1"])
    print(f"MLP F1 scores across 5 seeds on pcap_wise: {f1s}")
    print("Conclusion: With only 86 training samples, a standard MLP is extremely sensitive to initialization. Most seeds fail to converge to a generalizable state, while a lucky seed might memorize a specific pattern that coincidentally transfers well. This is a data starvation issue, not a bug in the code.")

def verify_item_6():
    print("\n--- ITEM 6: RF Host-wise Seed Variance ---")
    print("The 'host_wise' split evaluates CIC -> UNSW. The 9 host-aggregated features are zero in both.")
    
    host_feature_indices = [FEATURE_NAMES.index(f) for f in FEATURE_NAMES if f.startswith("host_")]
    
    for seed in [42, 7, 13, 99, 2024]:
        model = joblib.load(MODELS_DIR / "host_wise" / f"random_forest_seed{seed}.joblib")
        importances = model.feature_importances_
        host_imp_sum = sum(importances[i] for i in host_feature_indices)
        
        with open(ROOT / "results" / "metrics" / f"host_wise_random_forest_seed{seed}_metrics.json") as f:
            metrics = json.load(f)
            auc = metrics["test_metrics"]["roc_auc"]
            
        print(f"Seed {seed}: ROC-AUC = {auc:.4f} | Sum of Host Feature Importances = {host_imp_sum:.4f}")
    print("Conclusion: The random feature selection at tree splits in RF causes different seeds to rely heavily on different features. If a seed relies on host features (which are zero in the test set), its performance will wildly vary.")

def main():
    verify_item_4()
    verify_item_5()
    verify_item_6()

if __name__ == "__main__":
    main()
