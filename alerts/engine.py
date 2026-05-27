#!/usr/bin/env python3
import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from flows.tracker import FlowSession
from utils.helpers import setup_logger, load_config, get_project_root

logger = setup_logger("AlertEngine")

class AlertEngine:
    """
    Computes hybrid threat scores by correlating ML inferences and temporal heuristics.
    Logs structured JSON incidents to alerts.log and raises formatted console warnings.
    """
    def __init__(self):
        self.config = load_config()
        self.project_root = self.config.get("project_root", get_project_root())
        self.alert_log = self.config.get("alerts", {}).get("alert_log_file", os.path.join(self.project_root, "logs", "alerts.log"))
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.alert_log), exist_ok=True)

    def calculate_hybrid_threat_score(self, 
                                      ml_prediction: int, 
                                      ml_confidence: float, 
                                      raw_features: Dict[str, float]) -> Tuple[float, List[str], str]:
        """
        Correlates machine learning inferences with behavioral evidence to produce
        an explainable, weighted threat score and severity rating.
        """
        evidence = []
        score = 0.0
        
        # 1. Base ML Contribution
        if ml_prediction == 1:
            score += ml_confidence * 0.45
            evidence.append(f"ML Model Classification (Confidence: {ml_confidence*100:.1f}%)")
        else:
            # Minor penalty reduction if ML predicts normal but heuristics are suspicious
            score += ml_confidence * 0.1
            
        # 2. Port Entropy Contribution (Targeting high diversity/random ports)
        port_entropy = raw_features.get("host_port_entropy", 0.0)
        if port_entropy > 3.0:
            score += 0.20
            evidence.append(f"Critical Destination Port Entropy ({port_entropy:.3f})")
        elif port_entropy > 2.0:
            score += 0.10
            evidence.append(f"Moderate Destination Port Entropy ({port_entropy:.3f})")
            
        # 3. Connection Failures (Scanning signature of half-open scans)
        failed_ratio = raw_features.get("host_failed_flow_ratio", 0.0)
        if failed_ratio > 0.8:
            score += 0.20
            evidence.append(f"Severe Failed Connection Rate ({failed_ratio*100:.1f}%)")
        elif failed_ratio > 0.5:
            score += 0.10
            evidence.append(f"Elevated Failed Connection Rate ({failed_ratio*100:.1f}%)")
            
        # 4. Destination Host Diversity (Subnet sweep signature)
        dst_diversity = raw_features.get("host_dst_diversity", 1.0)
        if dst_diversity > 10:
            score += 0.15
            evidence.append(f"Critical Subnet Target Diversity ({int(dst_diversity)} hosts)")
        elif dst_diversity > 4:
            score += 0.08
            evidence.append(f"Moderate Subnet Target Diversity ({int(dst_diversity)} hosts)")
            
        # Bound score between 0.0 and 1.0
        final_score = float(min(1.0, max(0.0, score)))
        
        # Severity Classification
        if final_score >= 0.75:
            severity = "HIGH"
        elif final_score >= 0.40:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        return final_score, evidence, severity

    def generate_alert(self, 
                       flow: FlowSession, 
                       raw_features: Dict[str, float], 
                       ml_prediction: int, 
                       ml_confidence: float, 
                       detection_method: str) -> Optional[Dict[str, Any]]:
        """
        Analyzes connection flow, generates hybrid scores, writes structured JSON logs,
        and outputs high-severity incidents to the stderr console stream.
        """
        threat_score, evidence, severity = self.calculate_hybrid_threat_score(
            ml_prediction, ml_confidence, raw_features
        )
        
        # Suppress alerts for negligible/low-score events to keep alert noise clean
        if severity == "LOW":
            return None

        # Build highly structured threat record
        alert_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(flow.end_time)),
            "alert_id": f"IDS-ALERT-{int(time.time() * 1000)}",
            "threat_category": "Stealth Reconnaissance Scan",
            "severity": severity,
            "detection_method": f"ML Inference ({detection_method})" if ml_prediction == 1 else "Behavioral Heuristics",
            "source_ip": flow.src_ip,
            "destination_ip": flow.dst_ip,
            "target_port": flow.dst_port,
            "protocol": "TCP" if flow.proto == 6 else "UDP" if flow.proto == 17 else str(flow.proto),
            "flow_statistics": {
                "packets": len(flow.packets),
                "duration_seconds": round(flow.duration, 4),
                "syn_ratio": round(flow.syn_count / max(1, len(flow.packets)), 2),
                "rst_ratio": round(flow.rst_count / max(1, len(flow.packets)), 2)
            },
            "host_context": {
                "port_entropy": round(raw_features.get("host_port_entropy", 0.0), 3),
                "destination_diversity": int(raw_features.get("host_dst_diversity", 1.0)),
                "failed_flow_ratio": round(raw_features.get("host_failed_flow_ratio", 0.0), 2)
            },
            "threat_confidence": round(threat_score, 4),
            "trigger_evidence": evidence
        }

        # Write to structured JSON lines alerts log
        try:
            with open(self.alert_log, 'a') as f:
                f.write(json.dumps(alert_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write structured JSON alert: {e}")

        # Console Alert Printing with decorative severity tags
        color = "\033[91m" if severity == "HIGH" else "\033[93m"  # Red for High, Yellow for Medium
        reset = "\033[0m"
        
        logger.warning(
            f"\n{color}🚨 [THREAT DETECTED - {severity} SEVERITY]{reset}"
            f"\n   Source IP:  {alert_record['source_ip']} -> Target: {alert_record['destination_ip']}:{alert_record['target_port']} ({alert_record['protocol']})"
            f"\n   Confidence: {alert_record['threat_confidence']*100:.2f}% | Method: {alert_record['detection_method']}"
            f"\n   Evidence:   {', '.join(evidence)}"
            f"\n   Port Entropy: {alert_record['host_context']['port_entropy']} | Active Targets: {alert_record['host_context']['destination_diversity']}\n"
        )
        
        return alert_record
