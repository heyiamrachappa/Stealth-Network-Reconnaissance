#!/usr/bin/env python3
# ==============================================================================
# Phase 5 - Dataset Pipeline Module
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import os
import sys
import json
import logging
import argparse
import joblib
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.parser import PCAPParser
from src.features import FeatureExtractor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("DatasetPipeline")


class DatasetPipeline:
    """
    Automates dataset building, heuristic labeling, normalization,
    and train/test splitting.
    """
    def __init__(self, config_path: str = "/home/yi/Stealth System/configs/config.json"):
        self.config = self._load_config(config_path)
        self.project_root = self.config.get("project_root", "/home/yi/Stealth System")
        self.dataset_dir = self.config.get("directories", {}).get("dataset", f"{self.project_root}/dataset")
        self.models_dir = self.config.get("directories", {}).get("models", f"{self.project_root}/models")
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        return {}

    def heuristic_labeler(self, df: pd.DataFrame, scanner_ips: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Labels network traffic flows as Normal (0) or Stealth Reconnaissance (1)
        using both explicit IP matching and a multi-factor expert heuristic.
        """
        df = df.copy()
        
        # Default label is Normal (0)
        df["label"] = 0
        
        # 1. IP-based Labeling (Known attacker/scanner IPs)
        if scanner_ips:
            ip_mask = df["src_ip"].isin(scanner_ips)
            df.loc[ip_mask, "label"] = 1
            logger.info(f"Labeled {ip_mask.sum()} flows based on attacker IP list.")
            
        # 2. Heuristic Rule-Based Labeling for Stealth Reconnaissance:
        # High host port entropy (scanning multiple ports) AND high failed flow ratio 
        # OR High host destination diversity (sweeping subnets) with high SYN ratios
        heuristic_mask = (
            # Port scanning heuristic
            ((df["host_port_entropy"] > 2.0) & (df["host_failed_flow_ratio"] > 0.7) & (df["flow_syn_ratio"] > 0.5)) |
            # Subnet sweep / Ping sweep heuristic
            ((df["host_dst_diversity"] > 4) & (df["host_dst_entropy"] > 1.5) & (df["host_syn_ratio"] > 0.7)) |
            # Individual half-open scan flow behavior
            ((df["flow_packet_count"] <= 2) & (df["flow_syn_ratio"] == 1.0) & (df["flow_ack_ratio"] == 0.0) & (df["host_port_entropy"] > 1.5))
        )
        
        # Mark heuristic matches
        df.loc[heuristic_mask, "label"] = 1
        
        total_labeled = df["label"].sum()
        logger.info(f"Total labeled flows in dataset: {total_labeled} scans / {len(df)} total flows.")
        return df

    def process_pcap_to_dataset(self, pcap_path: str, scanner_ips: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Full feature-extraction + labeling pipeline for a single PCAP file.
        """
        logger.info(f"Starting pipeline for PCAP: {pcap_path}")
        
        # 1. Parse PCAP to flows
        parser = PCAPParser()
        flows = parser.aggregate_flows(pcap_path)
        
        if not flows:
            logger.warning(f"No valid IP flows found in PCAP: {pcap_path}")
            return pd.DataFrame()
            
        # 2. Extract Features
        extractor = FeatureExtractor()
        df = extractor.extract_features(flows)
        
        # 3. Label Flows
        df_labeled = self.heuristic_labeler(df, scanner_ips)
        
        return df_labeled

    def build_and_split_dataset(self, 
                                pcap_files: List[str], 
                                scanner_ips: Optional[List[str]] = None,
                                test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes multiple PCAP files, aggregates them, normalizes features,
        saves the fitted StandardScaler, and outputs train/test splits.
        """
        all_dfs = []
        for pcap in pcap_files:
            df = self.process_pcap_to_dataset(pcap, scanner_ips)
            if not df.empty:
                all_dfs.append(df)
                
        if not all_dfs:
            raise ValueError("No feature data could be extracted from the provided PCAP files.")
            
        # Aggregate all data
        dataset = pd.concat(all_dfs, ignore_index=True)
        
        # Save raw dataset
        raw_path = f"{self.dataset_dir}/raw_features.csv"
        dataset.to_csv(raw_path, index=False)
        logger.info(f"Saved raw feature dataset to {raw_path}")
        
        # Define features to drop for training (non-numeric, IP identifiers, etc.)
        # This prevents overfitting on specific network topologies
        cols_to_drop = ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "label"]
        
        X = dataset.drop(columns=cols_to_drop, errors="ignore")
        y = dataset["label"]
        
        # Save feature column names for reference during real-time detection
        feature_names = list(X.columns)
        names_path = f"{self.models_dir}/feature_names.json"
        with open(names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Saved feature list to {names_path}")
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
        # Preprocessing: Fit Standard Scaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save fitted scaler
        scaler_path = f"{self.models_dir}/scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Fitted Standard Scaler saved to {scaler_path}")
        
        # Re-convert to DataFrames to keep column names in output
        train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        train_df["label"] = y_train.values
        
        test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_df["label"] = y_test.values
        
        # Save processed splits
        train_path = f"{self.dataset_dir}/train_features.csv"
        test_path = f"{self.dataset_dir}/test_features.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved preprocessed train set to {train_path} ({train_df.shape[0]} samples)")
        logger.info(f"Saved preprocessed test set to {test_path} ({test_df.shape[0]} samples)")
        
        return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stealth IDS - Dataset Pipeline CLI")
    parser.add_argument("pcaps", nargs="+", help="Paths to PCAP files for processing")
    parser.add_argument("-s", "--scanners", help="Comma-separated list of known scanner IPs")
    parser.add_argument("-t", "--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    
    args = parser.parse_args()
    
    scanner_ips = args.scanners.split(",") if args.scanners else None
    
    pipeline = DatasetPipeline()
    try:
        pipeline.build_and_split_dataset(
            pcap_files=args.pcaps, 
            scanner_ips=scanner_ips, 
            test_size=args.test_size
        )
    except Exception as e:
        logger.error(f"Dataset Pipeline build failed: {e}")
