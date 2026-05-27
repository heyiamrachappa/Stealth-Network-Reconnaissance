#!/usr/bin/env python3
import os
import sys
import json
import joblib
import argparse
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add root folder to python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logger, load_config, get_project_root
from capture.sniffer import PacketSniffer
from flows.tracker import FlowTracker
from features.extractor import FeatureExtractor

logger = setup_logger("DatasetPipeline")

class DatasetPipeline:
    """
    Automates dataset building, heuristic labeling, standardization,
    and train/test splitting from PCAP files.
    """
    def __init__(self):
        self.config = load_config()
        self.project_root = self.config.get("project_root", get_project_root())
        self.dataset_dir = self.config.get("directories", {}).get("dataset", os.path.join(self.project_root, "dataset"))
        self.models_dir = self.config.get("directories", {}).get("models", os.path.join(self.project_root, "models"))
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def heuristic_labeler(self, df: pd.DataFrame, scanner_ips: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies a multi-factor cybersecurity scanning heuristic to label traffic flows.
        """
        df = df.copy()
        df["label"] = 0
        
        if scanner_ips:
            ip_mask = df["src_ip"].isin(scanner_ips)
            df.loc[ip_mask, "label"] = 1
            logger.info(f"Labeled {ip_mask.sum()} flows based on attacker IP list.")
            
        # Core stealth reconnaissance detection heuristics
        heuristic_mask = (
            # Port scanning indicator: high host port entropy + high failed flows
            ((df["host_port_entropy"] > 2.0) & (df["host_failed_flow_ratio"] > 0.7) & (df["flow_syn_ratio"] > 0.5)) |
            # Subnet sweep / enumeration: high destination IP diversity + syn dominance
            ((df["host_dst_diversity"] > 4) & (df["host_dst_entropy"] > 1.5) & (df["host_syn_ratio"] > 0.7)) |
            # TCP flags half-open scan signature
            ((df["flow_packet_count"] <= 2) & (df["flow_syn_ratio"] == 1.0) & (df["flow_ack_ratio"] == 0.0) & (df["host_port_entropy"] > 1.5))
        )
        
        df.loc[heuristic_mask, "label"] = 1
        total_labeled = df["label"].sum()
        logger.info(f"Total labeled flows: {total_labeled} anomalies / {len(df)} total flows.")
        return df

    def process_pcap_to_dataset(self, pcap_path: str, scanner_ips: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Processes a single offline PCAP capture file into a labeled DataFrame.
        """
        logger.info(f"Processing offline PCAP capture: {pcap_path}")
        from scapy.all import PcapReader
        
        tracker = FlowTracker(flow_timeout_seconds=99999) # Infinite timeout for batch parsing
        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    parsed = PacketSniffer.parse_packet(pkt)
                    if parsed:
                        tracker.handle_packet(parsed)
        except Exception as e:
            logger.error(f"Failed to parse target PCAP {pcap_path}: {e}")
            return pd.DataFrame()
            
        flows = tracker.get_active_sessions()
        if not flows:
            logger.warning(f"No valid IP flows found in PCAP: {pcap_path}")
            return pd.DataFrame()
            
        extractor = FeatureExtractor()
        df = extractor.extract_features(flows)
        
        return self.heuristic_labeler(df, scanner_ips)

    def build_and_split_dataset(self, 
                                pcap_files: List[str], 
                                scanner_ips: Optional[List[str]] = None,
                                test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assembles all PCAP captures, extracts features, standardizes variables,
        and saves training/testing dataset splits.
        """
        all_dfs = []
        for pcap in pcap_files:
            df = self.process_pcap_to_dataset(pcap, scanner_ips)
            if not df.empty:
                all_dfs.append(df)
                
        if not all_dfs:
            raise ValueError("No feature data could be extracted from PCAP files.")
            
        dataset = pd.concat(all_dfs, ignore_index=True)
        
        # Save raw dataset for research reference
        raw_path = os.path.join(self.dataset_dir, "raw_features.csv")
        dataset.to_csv(raw_path, index=False)
        logger.info(f"Saved raw feature dataset to {raw_path}")
        
        # Strip identifiers before training to prevent topology overfitting
        cols_to_drop = ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "label"]
        X = dataset.drop(columns=cols_to_drop, errors="ignore")
        y = dataset["label"]
        
        # Save feature columns lists
        feature_names = list(X.columns)
        names_path = os.path.join(self.models_dir, "feature_names.json")
        with open(names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Saved feature list schema to {names_path}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Serialize fitted scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Standard scaler successfully saved to {scaler_path}")
        
        # Output train/test frames
        train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        train_df["label"] = y_train.values
        
        test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_df["label"] = y_test.values
        
        train_path = os.path.join(self.dataset_dir, "train_features.csv")
        test_path = os.path.join(self.dataset_dir, "test_features.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved preprocessed train split to {train_path} ({train_df.shape[0]} samples)")
        logger.info(f"Saved preprocessed test split to {test_path} ({test_df.shape[0]} samples)")
        
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
