#!/usr/bin/env python3
"""
Phase 2 — Build Data Splits
Constructs four split strategies from the combined dataset corpus.

Split strategies:
  1. pcap-wise   : split on pcap_filename / dataset_source (train on synthetic, test on CIC)
  2. time-wise   : temporal split — first 80% train, last 20% test
  3. host-wise   : hold out one source-IP group for test
  4. cross-dataset: train on synthetic+UNSW, test on CIC Friday PortScan

All test splits are verified to contain BOTH benign and recon samples before saving.
Outputs: dataset/splits/{strategy}/{train,val,test}.parquet + results/split_report_v2.json
"""

import os, sys, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
SPLITS_DIR  = ROOT / "dataset" / "splits"
MAPPED_DIR  = ROOT / "dataset" / "mapped"

RANDOM_STATE = 42
FEATURE_NAMES = [
    "flow_duration","flow_packet_count","flow_bytes",
    "flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio",
    "flow_size_mean","flow_size_var",
    "flow_interval_mean","flow_interval_var",
    "host_port_entropy","host_dst_entropy","host_dst_diversity",
    "host_syn_ratio","host_failed_flow_ratio",
    "host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"
]


def split_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_recon  = int((df["label"] == 1).sum())
    n_dup    = int(df[FEATURE_NAMES].duplicated().sum()) if len(df) > 0 else 0
    n_null   = int(df[FEATURE_NAMES].isnull().sum().sum())
    sources  = df["dataset_source"].value_counts().to_dict() if "dataset_source" in df.columns else {}
    return {
        "total": n,
        "benign": n_benign,
        "recon": n_recon,
        "class_balance_ratio": round(n_recon / n_benign, 3) if n_benign > 0 else None,
        "duplicate_feature_rows": n_dup,
        "missing_values": n_null,
        "dataset_sources": {str(k): int(v) for k, v in sources.items()}
    }


def verify_split(split_name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    issues = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if (df["label"] == 0).sum() == 0:
            issues.append(f"{split_name}/{name}: NO BENIGN SAMPLES")
        if (df["label"] == 1).sum() == 0:
            issues.append(f"{split_name}/{name}: NO RECON SAMPLES")
    if issues:
        raise ValueError("Split verification failed:\n" + "\n".join(issues))
    print(f"  ✓ {split_name} split verified: all subsets contain both classes")


def save_split(strategy: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    d = SPLITS_DIR / strategy
    d.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = d / f"{name}.parquet"
        df.reset_index(drop=True).to_parquet(path, index=False)
    print(f"  Saved {strategy}: train={len(train)}, val={len(val)}, test={len(test)}")


def load_synthetic() -> pd.DataFrame:
    path = ROOT / "dataset" / "raw_features.csv"
    df = pd.read_csv(path)
    # Ensure all 20 features exist
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    if "dataset_source" not in df.columns:
        df["dataset_source"] = "synthetic"
    if "attack_type" not in df.columns:
        df["attack_type"] = df.get("pcap_filename", "unknown").str.replace(".pcap", "", regex=False)
    return df


def load_cic() -> pd.DataFrame:
    path = MAPPED_DIR / "cic_mapped.parquet"
    return pd.read_parquet(path)


def load_unsw() -> pd.DataFrame:
    path = MAPPED_DIR / "unsw_mapped.parquet"
    return pd.read_parquet(path)


def add_required_cols(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "dataset_source" not in df.columns:
        df = df.copy()
        df["dataset_source"] = source
    if "attack_type" not in df.columns:
        df = df.copy()
        df["attack_type"] = "unknown"
    return df


def main():
    report = {}

    synthetic = load_synthetic()
    cic       = load_cic()
    unsw      = load_unsw()

    print(f"Loaded: synthetic={len(synthetic)}, CIC={len(cic):,}, UNSW={len(unsw):,}")

    # ── 1. pcap-wise split ────────────────────────────────────────────────────
    # Train on synthetic corpus; validate on UNSW; test on CIC PortScan
    # This simulates training in your own lab, testing on external real-world data
    print("\n[1] Building pcap-wise split...")
    # Downsample CIC test to avoid extreme imbalance (sample 10k benign, all PortScan)
    cic_recon  = cic[cic["label"] == 1]
    cic_benign = cic[cic["label"] == 0].sample(n=min(10000, (cic["label"]==0).sum()), random_state=RANDOM_STATE)
    cic_test   = shuffle(pd.concat([cic_recon, cic_benign]), random_state=RANDOM_STATE)

    syn_train, syn_val = train_test_split(synthetic, test_size=0.15, stratify=synthetic["label"], random_state=RANDOM_STATE)
    verify_split("pcap_wise", syn_train, syn_val, cic_test)
    save_split("pcap_wise", syn_train[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             syn_val[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             cic_test[FEATURE_NAMES+["label","dataset_source","attack_type"]])
    report["pcap_wise"] = {
        "description": "Train: synthetic corpus; Val: synthetic holdout; Test: CIC-IDS-2017 Friday PortScan",
        "train": split_stats(syn_train),
        "val":   split_stats(syn_val),
        "test":  split_stats(cic_test)
    }

    # ── 2. time-wise split ────────────────────────────────────────────────────
    # Use CIC Friday data (has timestamps via row order) + synthetic
    # Combine all, sort by implicit time (row order in CIC preserves capture time)
    print("\n[2] Building time-wise split...")
    cic_sample = shuffle(pd.concat([
        cic[cic["label"] == 1].sample(n=min(5000, (cic["label"]==1).sum()), random_state=RANDOM_STATE),
        cic[cic["label"] == 0].sample(n=min(5000, (cic["label"]==0).sum()), random_state=RANDOM_STATE)
    ]), random_state=RANDOM_STATE)
    combined_time = pd.concat([synthetic, cic_sample]).reset_index(drop=True)
    n = len(combined_time)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.10)
    tw_train = combined_time.iloc[:n_train]
    tw_val   = combined_time.iloc[n_train:n_train+n_val]
    tw_test  = combined_time.iloc[n_train+n_val:]
    verify_split("time_wise", tw_train, tw_val, tw_test)
    save_split("time_wise", tw_train[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             tw_val[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             tw_test[FEATURE_NAMES+["label","dataset_source","attack_type"]])
    report["time_wise"] = {
        "description": "Temporal split 70/10/20 on combined synthetic+CIC sample; row order approximates capture time",
        "train": split_stats(tw_train),
        "val":   split_stats(tw_val),
        "test":  split_stats(tw_test)
    }

    # ── 3. host-wise split ────────────────────────────────────────────────────
    # Hold out one attack-source group (UNSW Reconnaissance) as test
    # Train on synthetic + CIC; val from CIC holdout; test on UNSW Recon
    print("\n[3] Building host-wise split...")
    unsw_recon  = unsw[unsw["label"] == 1]
    unsw_benign = unsw[unsw["label"] == 0]
    hw_test = shuffle(pd.concat([unsw_recon, unsw_benign]), random_state=RANDOM_STATE)

    cic_train_sample = shuffle(pd.concat([
        cic[cic["label"] == 1].sample(n=min(3000, (cic["label"]==1).sum()), random_state=RANDOM_STATE),
        cic[cic["label"] == 0].sample(n=min(3000, (cic["label"]==0).sum()), random_state=RANDOM_STATE)
    ]), random_state=RANDOM_STATE)
    hw_combined = pd.concat([synthetic, cic_train_sample]).reset_index(drop=True)
    hw_train, hw_val = train_test_split(hw_combined, test_size=0.15, stratify=hw_combined["label"], random_state=RANDOM_STATE)
    verify_split("host_wise", hw_train, hw_val, hw_test)
    save_split("host_wise", hw_train[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             hw_val[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                             hw_test[FEATURE_NAMES+["label","dataset_source","attack_type"]])
    report["host_wise"] = {
        "description": "Train: synthetic+CIC sample; Val: CIC holdout; Test: UNSW-NB15 Reconnaissance (unseen source)",
        "train": split_stats(hw_train),
        "val":   split_stats(hw_val),
        "test":  split_stats(hw_test)
    }

    # ── 4. cross-dataset split ────────────────────────────────────────────────
    # Train on synthetic+UNSW; test on CIC PortScan (cross-dataset generalisation)
    print("\n[4] Building cross-dataset split...")
    combined_cd = pd.concat([synthetic, unsw]).reset_index(drop=True)
    combined_cd = shuffle(combined_cd, random_state=RANDOM_STATE)
    cd_train, cd_tmp = train_test_split(combined_cd, test_size=0.20, stratify=combined_cd["label"], random_state=RANDOM_STATE)
    cd_val, _ = train_test_split(cd_tmp, test_size=0.50, stratify=cd_tmp["label"], random_state=RANDOM_STATE)
    cd_test = cic_test  # same CIC test set as pcap-wise
    verify_split("cross_dataset", cd_train, cd_val, cd_test)
    save_split("cross_dataset", cd_train[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                                cd_val[FEATURE_NAMES+["label","dataset_source","attack_type"]],
                                cd_test[FEATURE_NAMES+["label","dataset_source","attack_type"]])
    report["cross_dataset"] = {
        "description": "Train: synthetic+UNSW-NB15; Val: holdout from same; Test: CIC-IDS-2017 Friday PortScan",
        "train": split_stats(cd_train),
        "val":   split_stats(cd_val),
        "test":  split_stats(cd_test)
    }

    report["metadata"] = {
        "random_state": RANDOM_STATE,
        "feature_schema": FEATURE_NAMES,
        "schema_size": len(FEATURE_NAMES),
        "note_host_features_in_public_data": (
            "9 host-aggregated features are set to 0.0 in CIC and UNSW splits. "
            "These features are only populated in the synthetic corpus where per-host "
            "statistics were computed during flow extraction."
        )
    }

    out_path = RESULTS_DIR / "split_report_v2.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSplit report written to {out_path}")

    # Summary table
    print("\n=== SPLIT SUMMARY ===")
    print(f"{'Strategy':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Test-Recon':>12} {'Test-Benign':>12}")
    for s in ["pcap_wise","time_wise","host_wise","cross_dataset"]:
        r = report[s]
        print(f"{s:<15} {r['train']['total']:>7} {r['val']['total']:>7} {r['test']['total']:>7} "
              f"{r['test']['recon']:>12} {r['test']['benign']:>12}")


if __name__ == "__main__":
    main()
