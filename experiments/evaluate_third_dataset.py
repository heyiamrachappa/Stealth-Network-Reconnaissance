#!/usr/bin/env python3
import json, sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MAPPED_DIR = ROOT / "dataset" / "mapped"
MAPPED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models" / "v2" / "cross_dataset"

with open(ROOT / "models" / "feature_names.json", "r") as f:
    FEATURE_NAMES = json.load(f)

def load_and_map_ton_iot():
    print("Loading NF-ToN-IoT-v2 from HuggingFace...")
    # Load a subset of 100k rows
    ds = load_dataset('Nora9029/NF-ToN-IoT-v2', split='train[:200000]')
    df = ds.to_pandas()
    
    print("Filtering for 'scanning' and 'Benign'...")
    df = df[df['Attack'].isin(['scanning', 'Benign'])].copy()
    
    # Balance it a bit if needed, or just take what we have
    n_benign = (df['Attack'] == 'Benign').sum()
    n_recon = (df['Attack'] == 'scanning').sum()
    print(f"Found {n_benign} benign and {n_recon} scanning samples.")
    
    print("Mapping features to 20 canonical features...")
    out = pd.DataFrame(index=df.index)
    
    out["flow_duration"] = pd.to_numeric(df["FLOW_DURATION_MILLISECONDS"], errors="coerce").fillna(0) / 1000.0
    out["flow_packet_count"] = pd.to_numeric(df["IN_PKTS"], errors="coerce").fillna(0) + pd.to_numeric(df["OUT_PKTS"], errors="coerce").fillna(0)
    out["flow_bytes"] = pd.to_numeric(df["IN_BYTES"], errors="coerce").fillna(0) + pd.to_numeric(df["OUT_BYTES"], errors="coerce").fillna(0)
    
    pkt_count_safe = out["flow_packet_count"].replace(0, np.nan)
    out["flow_size_mean"] = out["flow_bytes"] / pkt_count_safe
    out["flow_interval_mean"] = out["flow_duration"] / pkt_count_safe
    
    out["flow_syn_ratio"] = 0.0
    out["flow_ack_ratio"] = 0.0
    out["flow_rst_ratio"] = 0.0
    out["flow_fin_ratio"] = 0.0
    out["flow_size_var"] = 0.0
    out["flow_interval_var"] = 0.0
    
    for col in ["host_port_entropy","host_dst_entropy","host_dst_diversity",
                "host_syn_ratio","host_failed_flow_ratio","host_packet_rate",
                "host_interval_mean","host_interval_var","host_packet_size_var"]:
        out[col] = 0.0

    out["label"] = (df["Attack"] == 'scanning').astype(int)
    
    out = out.fillna(0.0)
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    
    mapped_path = MAPPED_DIR / "ton_iot_mapped.parquet"
    out.to_parquet(mapped_path, index=False)
    print(f"Mapped dataset saved to {mapped_path}")
    return out

def evaluate_models(df):
    X = df[FEATURE_NAMES].fillna(0)
    y_true = df["label"]
    
    train_path = ROOT / "dataset" / "splits" / "cross_dataset" / "train.parquet"
    train_df = pd.read_parquet(train_path)
    X_train = train_df[FEATURE_NAMES].fillna(0)
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

    results = {}
    model_order = ["random_forest", "xgboost", "svm", "isolation_forest", "mlp"]
    for m in model_order:
        model = get_model(m)
        print(f"Retraining {m}...")
        model.fit(X_train, y_train if m != "isolation_forest" else X_train)
        print(f"Evaluating {m}...")
        
        if m == "isolation_forest":
            scores = model.decision_function(X)
            probs = 1 / (1 + np.exp(scores))
            preds = (probs >= 0.5).astype(int)
        else:
            preds = model.predict(X)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            else:
                scores = model.decision_function(X)
                probs = 1 / (1 + np.exp(-scores))
                
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        mcc = matthews_corrcoef(y_true, preds)
        try:
            roc_auc = roc_auc_score(y_true, probs)
        except:
            roc_auc = 0.0
            
        results[m] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mcc": mcc,
            "roc_auc": roc_auc
        }
        print(f"  F1: {f1:.4f}, MCC: {mcc:.4f}")
        
    return results

def main():
    df = load_and_map_ton_iot()
    results = evaluate_models(df)
    
    report_path = ROOT / "report.json"
    with open(report_path, "r") as f:
        report = json.load(f)
        
    report["third_dataset"] = {
        "dataset_chosen": "TON_IoT (NF-ToN-IoT-v2)",
        "attempted": ["TON_IoT"],
        "blocked_reason": None,
        "results": results,
        "if_resilience_holds": results.get("isolation_forest", {}).get("f1", 0) > 0.8
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Updated report.json with third-dataset results!")

if __name__ == "__main__":
    main()
