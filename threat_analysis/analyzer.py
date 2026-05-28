#!/usr/bin/env python3
import time
from typing import Dict, List, Any, Optional, Tuple
from feature_extraction.extractor import FlowSession
from ml_engine.engine import MLInferenceEngine
from utils.helpers import setup_logger

logger = setup_logger("ThreatAnalyzer")

class ForensicThreatReport:
    """
    Structured outcome of a static PCAP threat analysis.
    """
    def __init__(self, 
                 flow: FlowSession, 
                 features: Dict[str, float], 
                 prediction: int, 
                 confidence: float, 
                 model_name: str):
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(flow.start_time))
        self.start_epoch = flow.start_time
        self.src_ip = flow.src_ip
        self.dst_ip = flow.dst_ip
        self.src_port = flow.src_port
        self.dst_port = flow.dst_port
        self.proto = "TCP" if flow.proto == 6 else ("UDP" if flow.proto == 17 else "Other")
        
        self.ml_prediction = prediction
        self.ml_confidence = confidence
        self.model_name = model_name
        self.features = features
        
        self.scan_category = "Normal Traffic"
        self.severity = "LOW"
        self.threat_score = 0.0
        self.mitre_mappings = []
        self.evidence = []
        
        self._analyze_behavior(flow)

    def _analyze_behavior(self, flow: FlowSession) -> None:
        """
        Inspects TCP flags, feature values, and ML outputs to categorize and score threats.
        """
        # Determine scan category based on TCP flag combinations
        is_syn = flow.syn_count > 0 and flow.ack_count == 0
        is_fin = flow.fin_count > 0 and flow.syn_count == 0 and flow.ack_count == 0
        
        # Check for NULL scan (TCP packets with no flags set)
        is_null = False
        if flow.proto == 6:
            has_tcp_flags = any(len(p.flags) > 0 for p in flow.packets)
            if not has_tcp_flags and len(flow.packets) > 0:
                is_null = True
                
        # 1. Identify Stealth Reconnaissance Category
        if self.ml_prediction == 1 or self.features.get("host_port_entropy", 0) > 2.0:
            if is_syn:
                self.scan_category = "TCP SYN Stealth Reconnaissance"
                self.mitre_mappings = ["T1595 (Active Scanning)", "T1046 (Network Service Scanning)"]
            elif is_fin:
                self.scan_category = "TCP FIN Stealth Reconnaissance"
                self.mitre_mappings = ["T1595.002 (Active Scanning: Ports)", "T1046"]
            elif is_null:
                self.scan_category = "TCP NULL Stealth Reconnaissance"
                self.mitre_mappings = ["T1595.002", "T1046"]
            elif self.features.get("host_dst_diversity", 0) > 4:
                self.scan_category = "Subnet Discovery Sweep"
                self.mitre_mappings = ["T1595.001 (Active Scanning: IP Addresses)"]
            else:
                self.scan_category = "Stealth Port Sweep"
                self.mitre_mappings = ["T1046 (Network Service Scanning)"]
        else:
            self.scan_category = "Standard Session"
            self.severity = "LOW"
            self.threat_score = 0.0
            return

        # 2. Compute Multi-Factor Threat Score (0.0 to 100.0)
        # Combines ML confidence (50%), Failed Flows (30%), and target port entropy (20%)
        pe = min(100.0, self.features.get("host_port_entropy", 0.0) * 22.0)
        ff = self.features.get("host_failed_flow_ratio", 0.0) * 100.0
        ml_conf = self.ml_confidence * 100.0
        
        self.threat_score = (0.5 * ml_conf) + (0.3 * ff) + (0.2 * pe)
        
        # Adjust score for explicit stealth indicators
        if is_fin or is_null:
            self.threat_score = min(100.0, self.threat_score + 10.0)

        # 3. Assign Severity Levels
        if self.threat_score >= 85.0:
            self.severity = "CRITICAL"
        elif self.threat_score >= 60.0:
            self.severity = "HIGH"
        elif self.threat_score >= 35.0:
            self.severity = "MEDIUM"
        else:
            self.severity = "LOW"

        # 4. Generate Explanatory Forensic Evidence
        if self.ml_prediction == 1:
            self.evidence.append(f"ML Model Classification: '{self.model_name}' predicted anomalous scanning with {self.ml_confidence*100:.1f}% confidence")
        
        pe_val = self.features.get("host_port_entropy", 0.0)
        if pe_val > 2.0:
            self.evidence.append(f"High Target Port Entropy ({pe_val:.3f}): Denotes randomized multi-port reconnaissance probe")
            
        ff_val = self.features.get("host_failed_flow_ratio", 0.0)
        if ff_val > 0.7:
            self.evidence.append(f"Host Connection Failure Ratio ({ff_val*100:.1f}%): Elevated connection half-opens or rejected SYN requests")
            
        dst_div = self.features.get("host_dst_diversity", 1.0)
        if dst_div > 3:
            self.evidence.append(f"Host Target Diversity ({dst_div:.0f} unique IPs): Indicates horizontal sweep across multiple hosts")
            
        if is_syn:
            self.evidence.append("Protocol Signature: Half-open TCP SYN scan sequence detected (no completed 3-way handshakes)")
        elif is_fin:
            self.evidence.append("Protocol Signature: TCP FIN scan bypass attempt detected")
        elif is_null:
            self.evidence.append("Protocol Signature: TCP NULL scan (completely blank TCP flag headers)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.proto,
            "scan_category": self.scan_category,
            "severity": self.severity,
            "threat_score": float(self.threat_score),
            "mitre_mappings": self.mitre_mappings,
            "evidence": self.evidence,
            "ml_confidence": float(self.ml_confidence)
        }


class ThreatAnalyzer:
    """
    Coordinates threat scoring and behavioral forensic classifications.
    """
    def __init__(self, model_name: str = "random_forest"):
        self.engine = MLInferenceEngine(model_name=model_name)

    def analyze_flows(self, flows: List[FlowSession], features_df: Any) -> List[ForensicThreatReport]:
        """
        Takes raw flow sessions and their extracted feature vectors, runs ML models,
        and returns structured ForensicThreatReport instances.
        """
        reports = []
        
        # Pre-group host profiles to fast-resolve contextual lookups
        extractor = features_df.copy() if hasattr(features_df, "copy") else features_df
        
        for flow in flows:
            # Skip flows with too few packets to prevent noise overfitting
            if len(flow.packets) < 2:
                continue
                
            # Locate corresponding row in features dataframe
            mask = (
                (features_df["src_ip"] == flow.src_ip) &
                (features_df["dst_ip"] == flow.dst_ip) &
                (features_df["dst_port"] == flow.dst_port) &
                (features_df["proto"] == flow.proto)
            )
            
            rows = features_df[mask]
            if rows.empty:
                continue
                
            raw_features = rows.iloc[0].drop(labels=["src_ip", "dst_ip", "src_port", "dst_port", "proto", "label"], errors="ignore").to_dict()
            
            # Run machine learning model prediction
            prediction, confidence = self.engine.predict(raw_features)
            
            # Instantiate analysis report
            report = ForensicThreatReport(
                flow=flow,
                features=raw_features,
                prediction=prediction,
                confidence=confidence,
                model_name=self.engine.model_name
            )
            
            reports.append(report)
            
        return reports
