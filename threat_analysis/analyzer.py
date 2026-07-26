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
        self.severity = "SAFE"
        self.threat_score = 0.0
        
        # Threat Scoring Sub-components
        self.behavior_score = 0.0
        self.protocol_score = 0.0
        self.flow_score = 0.0
        self.historical_score = 0.0
        self.rule_confidence = 0.0
        
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
                self.rule_confidence = 90.0
            elif is_fin:
                self.scan_category = "TCP FIN Stealth Reconnaissance"
                self.mitre_mappings = ["T1595.002 (Active Scanning: Ports)", "T1046"]
                self.rule_confidence = 100.0
            elif is_null:
                self.scan_category = "TCP NULL Stealth Reconnaissance"
                self.mitre_mappings = ["T1595.002", "T1046"]
                self.rule_confidence = 100.0
            elif self.features.get("host_dst_diversity", 0) > 4:
                self.scan_category = "Subnet Discovery Sweep"
                self.mitre_mappings = ["T1595.001 (Active Scanning: IP Addresses)"]
                self.rule_confidence = 80.0
            else:
                self.scan_category = "Stealth Port Sweep"
                self.mitre_mappings = ["T1046 (Network Service Scanning)"]
                self.rule_confidence = 70.0
        else:
            self.scan_category = "Standard Session"
            self.rule_confidence = 0.0

        # 2. Compute Multi-Factor Threat Score (0.0 to 100.0)
        # Behavior Score
        pe = self.features.get("host_port_entropy", 0.0)
        hd = self.features.get("host_dst_diversity", 0.0)
        ff = self.features.get("host_failed_flow_ratio", 0.0)
        self.behavior_score = min(100.0, (pe * 15.0) + (hd * 5.0) + (ff * 50.0))
        
        # Protocol Score
        proto_ent = self.features.get("host_proto_entropy", 0.0)
        syn_ratio = self.features.get("host_syn_ratio", 0.0)
        ps = 0.0
        if is_fin or is_null: ps += 80.0
        if is_syn: ps += 50.0
        ps += (proto_ent * 20.0) + (syn_ratio * 30.0)
        self.protocol_score = min(100.0, ps)
        
        # Flow Score
        fs_syn_ack = self.features.get("flow_syn_ack_ratio", 0.0)
        self.flow_score = min(100.0, fs_syn_ack * 50.0)
        if len(flow.packets) <= 3 and is_syn:
            self.flow_score = max(self.flow_score, 85.0)
            
        # Historical Deviation Score (using burstiness as proxy)
        burst = self.features.get("host_burstiness", 0.0)
        self.historical_score = min(100.0, abs(burst) * 50.0)
        
        # Fusion
        ml_score = self.ml_confidence * 100.0
        self.threat_score = (
            (0.30 * ml_score) +
            (0.20 * self.behavior_score) +
            (0.20 * self.protocol_score) +
            (0.15 * self.rule_confidence) +
            (0.10 * self.flow_score) +
            (0.05 * self.historical_score)
        )
        
        if self.scan_category == "Standard Session":
            self.threat_score = min(self.threat_score, 25.0)

        # 3. Assign Severity Levels
        if self.threat_score >= 91.0:
            self.severity = "CRITICAL"
        elif self.threat_score >= 71.0:
            self.severity = "HIGH"
        elif self.threat_score >= 51.0:
            self.severity = "MEDIUM"
        elif self.threat_score >= 26.0:
            self.severity = "LOW"
        else:
            self.severity = "SAFE"

        # 4. Generate Explanatory Forensic Evidence
        
        # A. Detection Rationale & ML
        if self.ml_prediction == 1:
            self.evidence.append(f"[Detection Rationale] ML engine '{self.model_name}' classified flow as anomalous with {self.ml_confidence*100:.1f}% confidence. Final Threat Score: {self.threat_score:.1f}/100.")
        elif self.threat_score > 25:
            self.evidence.append(f"[Detection Rationale] Heuristic threat score reached {self.threat_score:.1f}/100, exceeding standard safe traffic thresholds.")
        else:
            self.evidence.append(f"[Detection Rationale] Traffic profile matches standard baseline (Threat Score: {self.threat_score:.1f}/100).")
            
        # B. MITRE ATT&CK
        if self.mitre_mappings:
            self.evidence.append(f"[MITRE ATT&CK Mapping] Matches adversarial techniques: {', '.join(self.mitre_mappings)}.")

        # C. Feature Contributions
        feature_factors = []
        pe_val = self.features.get("host_port_entropy", 0.0)
        if pe_val > 2.0:
            feature_factors.append(f"High Port Entropy ({pe_val:.2f})")
        
        dst_div = self.features.get("host_dst_diversity", 1.0)
        if dst_div > 3:
            feature_factors.append(f"High Target Diversity ({dst_div:.0f} IPs)")
            
        ff_val = self.features.get("host_failed_flow_ratio", 0.0)
        if ff_val > 0.5:
            feature_factors.append(f"Elevated Connection Failure Ratio ({ff_val*100:.1f}%)")
            
        if feature_factors:
            self.evidence.append(f"[Contributing Features] Anomalous behavioral indicators triggered: {', '.join(feature_factors)}.")

        # D. Protocol Evidence
        if is_syn:
            self.evidence.append("[Protocol Evidence] TCP SYN stealth scan signature detected (incomplete 3-way handshakes).")
        elif is_fin:
            self.evidence.append("[Protocol Evidence] TCP FIN packet detected without prior connection state (firewall bypass attempt).")
        elif is_null:
            self.evidence.append("[Protocol Evidence] TCP NULL packet detected (zero flags set; potential OS fingerprinting probe).")
            
        proto_ent = self.features.get("host_proto_entropy", 0.0)
        if proto_ent > 1.0:
            self.evidence.append(f"[Protocol Evidence] Protocol multiplexing or abnormal protocol usage detected (Entropy: {proto_ent:.2f}).")
            
        # E. Flow Statistics
        fs_syn_ack = self.features.get("flow_syn_ack_ratio", 0.0)
        self.evidence.append(f"[Flow Statistics] SYN/ACK Ratio: {fs_syn_ack:.2f}. Total Packets in Flow: {len(flow.packets)}.")

        # F. Historical Deviation
        burst = self.features.get("host_burstiness", 0.0)
        if abs(burst) > 0.5:
            self.evidence.append(f"[Historical Deviation] Abnormal traffic burstiness detected (Index: {abs(burst):.2f}). Indicates sharp deviation from historical baseline flow rates.")
        else:
            self.evidence.append(f"[Historical Deviation] Traffic burstiness (Index: {abs(burst):.2f}) remains within standard historical bounds.")

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
            "behavior_score": float(self.behavior_score),
            "protocol_score": float(self.protocol_score),
            "flow_score": float(self.flow_score),
            "historical_score": float(self.historical_score),
            "rule_confidence": float(self.rule_confidence),
            "ml_confidence": float(self.ml_confidence),
            "mitre_mappings": self.mitre_mappings,
            "evidence": self.evidence
        }


class IncidentReport:
    """
    Summarized aggregation of multiple ForensicThreatReport flows grouped into a single incident.
    """
    def __init__(self, src_ip: str, dst_ip: str, scan_category: str):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.scan_category = scan_category
        
        self.start_epoch = float('inf')
        self.end_epoch = 0.0
        
        self.packet_count = 0
        self.flow_count = 0
        self.target_ports = set()
        self.protocols = set()
        
        self._mitre_mappings_set = set()
        self._evidence_set = set()
        
        self.threat_scores = []
        self.ml_confidences = []
        self.behavior_scores = []
        self.protocol_scores = []
        self.flow_scores = []
        self.historical_scores = []
        self.rule_confidences = []
        
        # UI accessible fields (populated on finalize)
        self.timestamp = ""
        self.dst_port = ""
        self.proto = ""
        self.severity = "SAFE"
        self.threat_score = 0.0
        self.ml_confidence = 0.0

    def add_report(self, r: ForensicThreatReport, flow: FlowSession):
        self.start_epoch = min(self.start_epoch, r.start_epoch)
        self.end_epoch = max(self.end_epoch, r.start_epoch)
        
        self.packet_count += len(flow.packets)
        self.flow_count += 1
        
        self.target_ports.add(r.dst_port)
        self.protocols.add(r.proto)
        
        for m in r.mitre_mappings:
            self._mitre_mappings_set.add(m)
        for e in r.evidence:
            self._evidence_set.add(e)
            
        self.threat_scores.append(r.threat_score)
        self.ml_confidences.append(r.ml_confidence)
        self.behavior_scores.append(r.behavior_score)
        self.protocol_scores.append(r.protocol_score)
        self.flow_scores.append(r.flow_score)
        self.historical_scores.append(r.historical_score)
        self.rule_confidences.append(r.rule_confidence)

    def finalize(self):
        """
        Calculates aggregates and prepares strings for dashboard rendering.
        """
        self.threat_score = max(self.threat_scores) if self.threat_scores else 0.0
        self.ml_confidence = sum(self.ml_confidences) / len(self.ml_confidences) if self.ml_confidences else 0.0
        self.ml_prediction = 1 if self.ml_confidence >= 0.5 else 0
        
        self.behavior_score = sum(self.behavior_scores) / len(self.behavior_scores) if self.behavior_scores else 0.0
        self.protocol_score = sum(self.protocol_scores) / len(self.protocol_scores) if self.protocol_scores else 0.0
        self.flow_score = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0.0
        self.historical_score = sum(self.historical_scores) / len(self.historical_scores) if self.historical_scores else 0.0
        self.rule_confidence = max(self.rule_confidences) if self.rule_confidences else 0.0
        
        sorted_ports = sorted(list(self.target_ports))
        if len(sorted_ports) > 5:
            self.dst_port = ", ".join(map(str, sorted_ports[:5])) + f" (+{len(sorted_ports)-5} more)"
        else:
            self.dst_port = ", ".join(map(str, sorted_ports))
            
        self.proto = ", ".join(self.protocols)
        self._mitre_mappings_list = list(self._mitre_mappings_set)
        self._evidence_list = list(self._evidence_set)
        
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_epoch))
        
        if self.threat_score >= 91.0:
            self.severity = "CRITICAL"
        elif self.threat_score >= 71.0:
            self.severity = "HIGH"
        elif self.threat_score >= 51.0:
            self.severity = "MEDIUM"
        elif self.threat_score >= 26.0:
            self.severity = "LOW"
        else:
            self.severity = "SAFE"

    # Support compatibility with the dashboard attribute access
    @property
    def mitre_mappings(self):
        return self._mitre_mappings_list

    @property
    def evidence(self):
        return self._evidence_list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": self.proto,
            "scan_category": self.scan_category,
            "severity": self.severity,
            "threat_score": float(self.threat_score),
            "ml_prediction": int(self.ml_prediction),
            "ml_confidence": float(self.ml_confidence),
            "behavior_score": float(self.behavior_score),
            "protocol_score": float(self.protocol_score),
            "flow_score": float(self.flow_score),
            "historical_score": float(self.historical_score),
            "rule_confidence": float(self.rule_confidence),
            "packet_count": self.packet_count,
            "flow_count": self.flow_count,
            "mitre_mappings": self._mitre_mappings_list,
            "evidence": self._evidence_list
        }


class ThreatAnalyzer:
    """
    Coordinates threat scoring and behavioral forensic classifications.
    """
    def __init__(self, model_name: str = "random_forest"):
        self.engine = MLInferenceEngine(model_name=model_name)

    def analyze_flows(self, flows: List[FlowSession], features_df: Any) -> List[IncidentReport]:
        """
        Takes raw flow sessions and their extracted feature vectors, runs ML models,
        groups them into incidents based on source, destination, category, and time window,
        and returns structured IncidentReport instances.
        """
        flow_reports = []
        
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
            
            flow_reports.append((report, flow))
            
        # Group into Incidents
        flow_reports.sort(key=lambda x: x[0].start_epoch)
        incidents = {}
        time_window = 60.0
        
        for r, flow in flow_reports:
            key = (r.src_ip, r.dst_ip, r.scan_category)
            if key not in incidents:
                inc = IncidentReport(r.src_ip, r.dst_ip, r.scan_category)
                inc.add_report(r, flow)
                incidents[key] = [inc]
            else:
                latest_inc = incidents[key][-1]
                if r.start_epoch - latest_inc.start_epoch <= time_window:
                    latest_inc.add_report(r, flow)
                else:
                    new_inc = IncidentReport(r.src_ip, r.dst_ip, r.scan_category)
                    new_inc.add_report(r, flow)
                    incidents[key].append(new_inc)
                    
        final_incidents = []
        for inc_list in incidents.values():
            for inc in inc_list:
                inc.finalize()
                final_incidents.append(inc)
                
        final_incidents.sort(key=lambda x: x.start_epoch)
        return final_incidents
