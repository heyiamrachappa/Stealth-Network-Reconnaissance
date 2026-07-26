#!/usr/bin/env python3
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pcap_processing.parser import PacketRecord
from utils.helpers import setup_logger, load_config

logger = setup_logger("FeatureExtractor")

@dataclass
class FlowSession:
    """
    Tracks state and aggregates rolling packet-level stats for a single network flow.
    """
    flow_key: Tuple[str, str, int, int, int]  # (IP_low, IP_high, port_low, port_high, proto)
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    
    packets: List[PacketRecord] = field(default_factory=list)
    forward_packets: int = 0
    backward_packets: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0
    
    syn_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    psh_count: int = 0
    
    start_time: float = 0.0
    end_time: float = 0.0

    def add_packet(self, pkt: PacketRecord) -> None:
        if not self.packets:
            self.start_time = pkt.timestamp
            
        self.packets.append(pkt)
        self.end_time = pkt.timestamp
        
        is_forward = (pkt.src_ip == self.src_ip and pkt.src_port == self.src_port)
        
        if is_forward:
            self.forward_packets += 1
            self.forward_bytes += pkt.payload_len
        else:
            self.backward_packets += 1
            self.backward_bytes += pkt.payload_len
            
        if pkt.proto == 6:  # TCP
            flags = pkt.flags.upper()
            if "S" in flags:
                self.syn_count += 1
            if "A" in flags:
                self.ack_count += 1
            if "R" in flags:
                self.rst_count += 1
            if "F" in flags:
                self.fin_count += 1
            if "P" in flags:
                self.psh_count += 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class StaticFlowTracker:
    """
    Statically groups a chronological sequence of parsed PacketRecord objects
    into bidirectional flow sessions without any thread locks or timeout pruning loops.
    """
    @staticmethod
    def get_canonical_key(src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: int) -> Tuple[str, str, int, int, int]:
        if (src_ip < dst_ip) or (src_ip == dst_ip and src_port <= dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    @classmethod
    def track_flows(cls, packets: List[PacketRecord]) -> List[FlowSession]:
        sessions: Dict[Tuple[str, str, int, int, int], FlowSession] = {}
        
        for pkt in packets:
            key = cls.get_canonical_key(pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.proto)
            if key not in sessions:
                sessions[key] = FlowSession(
                    flow_key=key,
                    src_ip=pkt.src_ip,
                    dst_ip=pkt.dst_ip,
                    src_port=pkt.src_port,
                    dst_port=pkt.dst_port,
                    proto=pkt.proto
                )
            sessions[key].add_packet(pkt)
            
        return list(sessions.values())


class FeatureExtractor:
    """
    Extracts statistical features and sliding temporal window host aggregates from sessions.
    Identical schema outputs to ensure pre-trained ML weights remain compatible.
    """
    def __init__(self):
        self.config = load_config()
        self.sliding_window = self.config.get("features", {}).get("sliding_window_seconds", 30)

    @staticmethod
    def calculate_entropy(labels: List[Any]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        probs = [count / total for count in counts.values()]
        return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    def compute_host_features(self, 
                              all_flows: List[FlowSession], 
                              sliding_window_seconds: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        window = sliding_window_seconds if sliding_window_seconds is not None else self.sliding_window
        current_time = max([f.end_time for f in all_flows]) if all_flows else 0.0
        
        host_data: Dict[str, Dict[str, Any]] = {}
        
        for flow in all_flows:
            if current_time > 0.0 and (current_time - flow.end_time) > window:
                continue
                
            if not flow.packets:
                continue
                
            initiator_ip = flow.src_ip
            
            if initiator_ip not in host_data:
                host_data[initiator_ip] = {
                    "dst_ips": [],
                    "dst_ports": [],
                    "protos": [],
                    "timestamps": [],
                    "packet_sizes": [],
                    "flows_count": 0,
                    "total_syn": 0,
                    "total_ack": 0,
                    "dns_packets": 0,
                    "icmp_packets": 0,
                    "failed_flows": 0,
                    "total_packets": 0
                }
                
            host = host_data[initiator_ip]
            host["flows_count"] += 1
            
            for pkt in flow.packets:
                if pkt.src_ip == initiator_ip:
                    host["total_packets"] += 1
                    host["dst_ips"].append(pkt.dst_ip)
                    if pkt.proto in [6, 17]:
                        host["dst_ports"].append(pkt.dst_port)
                    host["protos"].append(pkt.proto)
                    
                    if pkt.dst_port == 53:
                        host["dns_packets"] += 1
                    if pkt.proto == 1:
                        host["icmp_packets"] += 1
                        
                    host["timestamps"].append(pkt.timestamp)
                    host["packet_sizes"].append(pkt.payload_len)
                    
                    if "S" in pkt.flags.upper():
                        host["total_syn"] += 1
                    if "A" in pkt.flags.upper():
                        host["total_ack"] += 1
            
            if flow.syn_count > 0 and flow.ack_count == 0:
                host["failed_flows"] += 1
                
        host_profiles: Dict[str, Dict[str, float]] = {}
        for ip, data in host_data.items():
            timestamps = sorted(data["timestamps"])
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
            
            port_entropy = self.calculate_entropy(data["dst_ports"])
            dst_entropy = self.calculate_entropy(data["dst_ips"])
            proto_entropy = self.calculate_entropy(data["protos"])
            
            syn_ratio = data["total_syn"] / max(1, data["total_packets"])
            failed_ratio = data["failed_flows"] / max(1, data["flows_count"])
            dns_ratio = data["dns_packets"] / max(1, data["total_packets"])
            icmp_ratio = data["icmp_packets"] / max(1, data["total_packets"])
            
            std_intervals = np.std(intervals) if len(intervals) > 0 else 0.0
            mean_intervals = np.mean(intervals) if len(intervals) > 0 else 0.0
            burstiness = (std_intervals - mean_intervals) / max(0.0001, std_intervals + mean_intervals)
            
            session_duration = timestamps[-1] - timestamps[0] if timestamps else 0.0
            
            host_profiles[ip] = {
                "host_port_entropy": port_entropy,
                "host_dst_entropy": dst_entropy,
                "host_dst_diversity": float(len(set(data["dst_ips"]))),
                "host_port_diversity": float(len(set(data["dst_ports"]))),
                "host_proto_entropy": proto_entropy,
                "host_syn_ratio": float(syn_ratio),
                "host_dns_ratio": float(dns_ratio),
                "host_icmp_ratio": float(icmp_ratio),
                "host_failed_flow_ratio": float(failed_ratio),
                "host_packet_rate": float(data["total_packets"] / max(0.0001, session_duration)),
                "host_interval_mean": float(mean_intervals),
                "host_interval_var": float(np.var(intervals)) if len(intervals) > 0 else 0.0,
                "host_burstiness": float(burstiness),
                "host_session_duration": float(session_duration),
                "host_packet_size_var": float(np.var(data["packet_sizes"])) if data["packet_sizes"] else 0.0
            }
            
        return host_profiles

    def extract_single_flow_vector(self, flow: FlowSession, host_profiles: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        timestamps = [pkt.timestamp for pkt in flow.packets]
        intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
        
        flow_interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        flow_interval_var = float(np.var(intervals)) if len(intervals) > 0 else 0.0
        
        payload_lens = [pkt.payload_len for pkt in flow.packets]
        flow_size_mean = float(np.mean(payload_lens)) if payload_lens else 0.0
        flow_size_var = float(np.var(payload_lens)) if payload_lens else 0.0
        
        total_pkts = len(flow.packets)
        dur = max(0.0001, flow.duration)
        flow_bytes = flow.forward_bytes + flow.backward_bytes
        
        syn_ratio = flow.syn_count / total_pkts if total_pkts > 0 else 0.0
        ack_ratio = flow.ack_count / total_pkts if total_pkts > 0 else 0.0
        rst_ratio = flow.rst_count / total_pkts if total_pkts > 0 else 0.0
        fin_ratio = flow.fin_count / total_pkts if total_pkts > 0 else 0.0
        
        flow_packets_per_sec = total_pkts / dur
        flow_bytes_per_sec = flow_bytes / dur
        flow_syn_ack_ratio = flow.syn_count / max(1, flow.ack_count)
        
        initiator_ip = flow.src_ip
        host_feat = host_profiles.get(initiator_ip, {
            "host_port_entropy": 0.0,
            "host_dst_entropy": 0.0,
            "host_dst_diversity": 1.0,
            "host_port_diversity": 0.0,
            "host_proto_entropy": 0.0,
            "host_syn_ratio": 0.0,
            "host_dns_ratio": 0.0,
            "host_icmp_ratio": 0.0,
            "host_failed_flow_ratio": 0.0,
            "host_packet_rate": 0.0,
            "host_interval_mean": 0.0,
            "host_interval_var": 0.0,
            "host_burstiness": 0.0,
            "host_session_duration": 0.0,
            "host_packet_size_var": 0.0
        })
        
        return {
            "flow_duration": flow.duration,
            "flow_packet_count": total_pkts,
            "flow_bytes": flow_bytes,
            "flow_packets_per_sec": flow_packets_per_sec,
            "flow_bytes_per_sec": flow_bytes_per_sec,
            "flow_syn_ratio": syn_ratio,
            "flow_ack_ratio": ack_ratio,
            "flow_rst_ratio": rst_ratio,
            "flow_fin_ratio": fin_ratio,
            "flow_syn_ack_ratio": flow_syn_ack_ratio,
            "flow_tcp_syn": flow.syn_count,
            "flow_tcp_ack": flow.ack_count,
            "flow_tcp_rst": flow.rst_count,
            "flow_tcp_fin": flow.fin_count,
            "flow_tcp_psh": flow.psh_count,
            "flow_size_mean": flow_size_mean,
            "flow_size_var": flow_size_var,
            "flow_interval_mean": flow_interval_mean,
            "flow_interval_var": flow_interval_var,
            
            "host_port_entropy": host_feat["host_port_entropy"],
            "host_dst_entropy": host_feat["host_dst_entropy"],
            "host_dst_diversity": host_feat["host_dst_diversity"],
            "host_port_diversity": host_feat["host_port_diversity"],
            "host_proto_entropy": host_feat["host_proto_entropy"],
            "host_syn_ratio": host_feat["host_syn_ratio"],
            "host_dns_ratio": host_feat["host_dns_ratio"],
            "host_icmp_ratio": host_feat["host_icmp_ratio"],
            "host_failed_flow_ratio": host_feat["host_failed_flow_ratio"],
            "host_packet_rate": host_feat["host_packet_rate"],
            "host_interval_mean": host_feat["host_interval_mean"],
            "host_interval_var": host_feat["host_interval_var"],
            "host_burstiness": host_feat["host_burstiness"],
            "host_session_duration": host_feat["host_session_duration"],
            "host_packet_size_var": host_feat["host_packet_size_var"]
        }

    def extract_features(self, flows: List[FlowSession]) -> pd.DataFrame:
        host_profiles = self.compute_host_features(flows)
        feature_rows = []
        
        for flow in flows:
            if not flow.packets:
                continue
                
            vector = self.extract_single_flow_vector(flow, host_profiles)
            
            vector["src_ip"] = flow.src_ip
            vector["dst_ip"] = flow.dst_ip
            vector["src_port"] = flow.src_port
            vector["dst_port"] = flow.dst_port
            vector["proto"] = flow.proto
            
            feature_rows.append(vector)
            
        return pd.DataFrame(feature_rows)
