#!/usr/bin/env python3
import sys
import logging
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from capture.sniffer import PacketRecord
from flows.tracker import FlowSession
from utils.helpers import setup_logger, load_config

logger = setup_logger("FeatureExtractor")

class FeatureExtractor:
    """
    Extracts statistical, structural, and temporal connection features from live flow session states.
    Employs sliding windows for real-time host-level reconnaissance behavior profiling.
    """
    def __init__(self):
        self.config = load_config()
        self.sliding_window = self.config.get("features", {}).get("sliding_window_seconds", 30)

    @staticmethod
    def calculate_entropy(labels: List[Any]) -> float:
        """
        Calculates Shannon entropy of a list of labels (e.g. destination ports).
        High entropy denotes high randomness/diversity, standard for reconnaissance scans.
        """
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        probs = [count / total for count in counts.values()]
        return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    def compute_host_features(self, 
                              all_flows: List[FlowSession], 
                              sliding_window_seconds: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Groups flows by Source IP inside a dynamic sliding temporal window to extract active profiles.
        """
        window = sliding_window_seconds if sliding_window_seconds is not None else self.sliding_window
        current_time = max([f.end_time for f in all_flows]) if all_flows else 0.0
        
        host_data: Dict[str, Dict[str, Any]] = {}
        
        for flow in all_flows:
            # Skip flows outside the sliding window
            if current_time > 0.0 and (current_time - flow.end_time) > window:
                continue
                
            if not flow.packets:
                continue
                
            # Attribute connection starting host
            initiator_ip = flow.src_ip
            
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
            
            # Record packet specifics inside the session
            for pkt in flow.packets:
                if pkt.src_ip == initiator_ip:
                    host["total_packets"] += 1
                    host["dst_ips"].append(pkt.dst_ip)
                    if pkt.proto in [6, 17]:
                        host["dst_ports"].append(pkt.dst_port)
                    host["timestamps"].append(pkt.timestamp)
                    host["packet_sizes"].append(pkt.payload_len)
                    
                    if "S" in pkt.flags.upper():
                        host["total_syn"] += 1
                    if "A" in pkt.flags.upper():
                        host["total_ack"] += 1
            
            # Reconnaissance metric: connection failed (sent SYN, never completed handshake)
            if flow.syn_count > 0 and flow.ack_count == 0:
                host["failed_flows"] += 1
                
        # Calculate behavioral feature profiles for active hosts
        host_profiles: Dict[str, Dict[str, float]] = {}
        for ip, data in host_data.items():
            timestamps = sorted(data["timestamps"])
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
            
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

    def extract_single_flow_vector(self, flow: FlowSession, host_profiles: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Computes a complete real-time feature vector for an active connection session.
        """
        timestamps = [pkt.timestamp for pkt in flow.packets]
        intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
        
        flow_interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        flow_interval_var = float(np.var(intervals)) if len(intervals) > 0 else 0.0
        
        payload_lens = [pkt.payload_len for pkt in flow.packets]
        flow_size_mean = float(np.mean(payload_lens)) if payload_lens else 0.0
        flow_size_var = float(np.var(payload_lens)) if payload_lens else 0.0
        
        total_pkts = len(flow.packets)
        syn_ratio = flow.syn_count / total_pkts if total_pkts > 0 else 0.0
        ack_ratio = flow.ack_count / total_pkts if total_pkts > 0 else 0.0
        rst_ratio = flow.rst_count / total_pkts if total_pkts > 0 else 0.0
        fin_ratio = flow.fin_count / total_pkts if total_pkts > 0 else 0.0
        
        # Retrieve context of initiator
        initiator_ip = flow.src_ip
        host_feat = host_profiles.get(initiator_ip, {
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
        
        return {
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
            
            "host_port_entropy": host_feat["host_port_entropy"],
            "host_dst_entropy": host_feat["host_dst_entropy"],
            "host_dst_diversity": host_feat["host_dst_diversity"],
            "host_syn_ratio": host_feat["host_syn_ratio"],
            "host_failed_flow_ratio": host_feat["host_failed_flow_ratio"],
            "host_packet_rate": host_feat["host_packet_rate"],
            "host_interval_mean": host_feat["host_interval_mean"],
            "host_interval_var": host_feat["host_interval_var"],
            "host_packet_size_var": host_feat["host_packet_size_var"]
        }

    def extract_features(self, flows: List[FlowSession]) -> pd.DataFrame:
        """
        Converts a list of flow sessions into a pandas DataFrame representing feature matrices.
        """
        host_profiles = self.compute_host_features(flows)
        feature_rows = []
        
        for flow in flows:
            if not flow.packets:
                continue
                
            vector = self.extract_single_flow_vector(flow, host_profiles)
            
            # Append indexing metadata
            vector["src_ip"] = flow.src_ip
            vector["dst_ip"] = flow.dst_ip
            vector["src_port"] = flow.src_port
            vector["dst_port"] = flow.dst_port
            vector["proto"] = flow.proto
            
            feature_rows.append(vector)
            
        return pd.DataFrame(feature_rows)
