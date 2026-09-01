#!/usr/bin/env python3
import os, sys, joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "shap"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "flow_duration","flow_packet_count","flow_bytes","flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio",
    "flow_size_mean","flow_size_var","flow_interval_mean","flow_interval_var",
    "host_port_entropy","host_dst_entropy","host_dst_diversity","host_syn_ratio","host_failed_flow_ratio",
    "host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"
]

def main():
    print("Loading data for SHAP analysis (UNSW->CIC cross-dataset)...")
    train_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "train.parquet")
    test_df = pd.read_parquet(ROOT / "dataset" / "splits" / "cross_dataset" / "test.parquet")
    
    # SHAP for MLP requires scaled features
    scaler = StandardScaler()
    X_train_raw = train_df[FEATURE_NAMES].values
    X_test_raw = test_df[FEATURE_NAMES].values
    
    scaler.fit(X_train_raw)
    
    # We sample background data from train to reduce computation
    X_background = shap.sample(scaler.transform(X_train_raw), 100, random_state=42)
    # We sample test data to explain
    X_explain_raw = shap.sample(X_test_raw, 500, random_state=42)
    X_explain = scaler.transform(X_explain_raw)
    
    print("Loading MLP model...")
    model_path = ROOT / "models" / "v2" / "cross_dataset" / "mlp_seed42.joblib"
    model = joblib.load(model_path)
    
    print("Running SHAP KernelExplainer (MLP is not tree-based)...")
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=100)
    
    # For binary classification, shap_values is a list of two arrays (one per class). We take class 1 (recon).
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # In newer shap versions, it might just return the values
        shap_values_class1 = shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values
    
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_class1, X_explain_raw, feature_names=FEATURE_NAMES, show=False)
    out_path = RESULTS_DIR / "mlp_shap_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"SHAP summary plot saved to {out_path}")

if __name__ == "__main__":
    main()
