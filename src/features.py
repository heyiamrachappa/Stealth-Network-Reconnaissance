#!/usr/bin/env python3
# ==============================================================================
# Phase 4 - Feature Extraction Module
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import sys
import logging
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from src.parser import FlowRecord, PCAPParser, PacketRecord

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("FeatureExtractor")


class FeatureExtractor:
    """
    Computes packet-level, flow-level, and host-level statistical features
    from parsed network flows.
    """
    @staticmethod
    def calculate_entropy(labels: List[Any]) -> float:
        """
        Calculates Shannon entropy of a list of labels.
        High entropy indicates high randomness/diversity.
        """
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        probs = [count / total for count in counts.values()]
        return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    def compute_host_features(self, flows: List[FlowRecord]) -> Dict[str, Dict[str, float]]:
        """
        Groups flows by Source IP to extract behavioral profiles.
        """
        host_data: Dict[str, Dict[str, Any]] = {}
        
        # Aggregate packets and flows by source IP
        for flow in flows:
            # We track flows bidirectional. Since we don't know who started the session 
            # with 100% certainty from bidirectional key alone, we look at the flow's packets
            # to attribute initiator. Usually, packet 0 is the initiator.
            if not flow.packets:
                continue
                
            initiator_ip = flow.packets[0].src_ip
            
            if initiator_ip not in host_data:
                host_data[initiator_ip] = {
                    "dst_ips": [],
                    "dst_ports": [],
                    "timestamps": [],
                    "packet_sizes": [],
                    "flows_count": 0,
                    "total_syn": 0,
                    "total_ack": 0,
                    "failed_flows": 0,
                    "total_packets": 0
                }
                
            host = host_data[initiator_ip]
            host["flows_count"] += 1
            
            # Record destinations accessed by this host
            for pkt in flow.packets:
                if pkt.src_ip == initiator_ip:
                    host["total_packets"] += 1
                    host["dst_ips"].append(pkt.dst_ip)
                    if pkt.proto == 6 or pkt.proto == 17:
                        host["dst_ports"].append(pkt.dst_port)
                    host["timestamps"].append(pkt.timestamp)
                    host["packet_sizes"].append(pkt.payload_len)
                    
                    if "S" in pkt.flags.upper():
                        host["total_syn"] += 1
                    if "A" in pkt.flags.upper():
                        host["total_ack"] += 1
            
            # Track failed connection (flow level) from this initiator:
            # e.g., SYN sent but no ACK received in entire bidirectional flow
            if flow.syn_count > 0 and flow.ack_count == 0:
                host["failed_flows"] += 1
                
        # Calculate Host profiles
        host_profiles: Dict[str, Dict[str, float]] = {}
        for ip, data in host_data.items():
            timestamps = sorted(data["timestamps"])
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
            
            # Port & Destination Entropies
            port_entropy = self.calculate_entropy(data["dst_ports"])
            dst_entropy = self.calculate_entropy(data["dst_ips"])
            
            syn_ratio = data["total_syn"] / max(1, data["total_packets"])
            failed_ratio = data["failed_flows"] / max(1, data["flows_count"])
            
            host_profiles[ip] = {
                "host_port_entropy": port_entropy,
                "host_dst_entropy": dst_entropy,
                "host_dst_diversity": float(len(set(data["dst_ips"]))),
                "host_syn_ratio": float(syn_ratio),
                "host_failed_flow_ratio": float(failed_ratio),
                "host_packet_rate": float(data["total_packets"] / max(0.1, (timestamps[-1] - timestamps[0]))) if timestamps else 0.0,
                "host_interval_mean": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
                "host_interval_var": float(np.var(intervals)) if len(intervals) > 0 else 0.0,
                "host_packet_size_var": float(np.var(data["packet_sizes"])) if data["packet_sizes"] else 0.0
            }
            
        return host_profiles

    def extract_features(self, flows_dict: Dict[Tuple[str, str, int, int, int], FlowRecord]) -> pd.DataFrame:
        """
        Transforms parsed flows and host characteristics into a structured DataFrame.
        """
        logger.info(f"Extracting features from {len(flows_dict)} connections...")
        
        flows = list(flows_dict.values())
        host_profiles = self.compute_host_features(flows)
        
        feature_rows = []
        
        for flow in flows:
            if not flow.packets:
                continue
                
            # Flow initiator
            initiator_ip = flow.packets[0].src_ip
            host_features = host_profiles.get(initiator_ip, {
                "host_port_entropy": 0.0,
                "host_dst_entropy": 0.0,
                "host_dst_diversity": 1.0,
                "host_syn_ratio": 0.0,
                "host_failed_flow_ratio": 0.0,
                "host_packet_rate": 0.0,
                "host_interval_mean": 0.0,
                "host_interval_var": 0.0,
                "host_packet_size_var": 0.0
            })
            
            # Flow-level timestamps and intervals
            timestamps = [pkt.timestamp for pkt in flow.packets]
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
            
            flow_interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
            flow_interval_var = float(np.var(intervals)) if len(intervals) > 0 else 0.0
            
            # Flow-level packet size statistics
            payload_lens = [pkt.payload_len for pkt in flow.packets]
            flow_size_mean = float(np.mean(payload_lens)) if payload_lens else 0.0
            flow_size_var = float(np.var(payload_lens)) if payload_lens else 0.0
            
            # TCP flag ratios
            total_pkts = len(flow.packets)
            syn_ratio = flow.syn_count / total_pkts
            ack_ratio = flow.ack_count / total_pkts
            rst_ratio = flow.rst_count / total_pkts
            fin_ratio = flow.fin_count / total_pkts
            
            # Combine Flow & Host behavioral properties
            row = {
                # Identifiers
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "proto": flow.proto,
                
                # Flow characteristics
                "flow_duration": flow.duration,
                "flow_packet_count": total_pkts,
                "flow_bytes": flow.forward_bytes + flow.backward_bytes,
                "flow_syn_ratio": syn_ratio,
                "flow_ack_ratio": ack_ratio,
                "flow_rst_ratio": rst_ratio,
                "flow_fin_ratio": fin_ratio,
                "flow_size_mean": flow_size_mean,
                "flow_size_var": flow_size_var,
                "flow_interval_mean": flow_interval_mean,
                "flow_interval_var": flow_interval_var,
                
                # Host contextual behavioral features
                "host_port_entropy": host_features["host_port_entropy"],
                "host_dst_entropy": host_features["host_dst_entropy"],
                "host_dst_diversity": host_features["host_dst_diversity"],
                "host_syn_ratio": host_features["host_syn_ratio"],
                "host_failed_flow_ratio": host_features["host_failed_flow_ratio"],
                "host_packet_rate": host_features["host_packet_rate"],
                "host_interval_mean": host_features["host_interval_mean"],
                "host_interval_var": host_features["host_interval_var"],
                "host_packet_size_var": host_features["host_packet_size_var"]
            }
            
            feature_rows.append(row)
            
        df = pd.DataFrame(feature_rows)
        logger.info(f"Feature matrix generated successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/features.py <path_to_pcap>")
        sys.exit(1)
        
    pcap_path = sys.argv[1]
    
    # Run Parser
    parser = PCAPParser()
    flows = parser.aggregate_flows(pcap_path)
    
    # Run Extractor
    extractor = FeatureExtractor()
    df = extractor.extract_features(flows)
    
    # Output quick inspect
    print("\n--- SAMPLE EXTRACTED FEATURES ---")
    print(df[["src_ip", "dst_ip", "dst_port", "flow_syn_ratio", "host_port_entropy", "host_dst_diversity"]].head())
