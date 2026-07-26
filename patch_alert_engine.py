import re

with open('alerts/engine.py', 'r') as f:
    content = f.read()

# 1. Update calculate_hybrid_threat_score signature
content = content.replace(
    "raw_features: Dict[str, float]) -> Tuple[float, List[str], str]:",
    "raw_features: Dict[str, float],\n                                      drift_score: float = 0.0) -> Tuple[float, List[str], str]:"
)

# 2. Add Behavioral Drift heuristic inside calculate_hybrid_threat_score
drift_logic = """        # 4. Destination Host Diversity (Subnet sweep signature)
        dst_diversity = raw_features.get("host_dst_diversity", 1.0)
        if dst_diversity > 10:
            score += 0.15
            evidence.append(f"Critical Subnet Target Diversity ({int(dst_diversity)} hosts)")
        elif dst_diversity > 4:
            score += 0.08
            evidence.append(f"Moderate Subnet Target Diversity ({int(dst_diversity)} hosts)")
            
        # 5. Behavioral Drift Contribution
        if drift_score > 75.0:
            score += 0.20
            evidence.append(f"Severe Behavioral Drift Score ({drift_score:.1f}/100)")
        elif drift_score > 40.0:
            score += 0.10
            evidence.append(f"Moderate Behavioral Drift Score ({drift_score:.1f}/100)")"""
content = content.replace(
"""        # 4. Destination Host Diversity (Subnet sweep signature)
        dst_diversity = raw_features.get("host_dst_diversity", 1.0)
        if dst_diversity > 10:
            score += 0.15
            evidence.append(f"Critical Subnet Target Diversity ({int(dst_diversity)} hosts)")
        elif dst_diversity > 4:
            score += 0.08
            evidence.append(f"Moderate Subnet Target Diversity ({int(dst_diversity)} hosts)")""",
drift_logic)


# 3. Update generate_alert signature
content = content.replace(
    "detection_method: str) -> Optional[Dict[str, Any]]:",
    "detection_method: str,\n                       drift_score: float = 0.0) -> Optional[Dict[str, Any]]:"
)

# 4. Pass drift_score
content = content.replace(
    "ml_prediction, ml_confidence, raw_features\n        )",
    "ml_prediction, ml_confidence, raw_features, drift_score\n        )"
)

# 5. Add drift_score to alert_record
content = content.replace(
    '"threat_confidence": round(threat_score, 4),',
    '"threat_confidence": round(threat_score, 4),\n            "behavioral_drift_score": round(drift_score, 2),'
)

# 6. Add drift_score to console
old_console = """            f"\\n   Evidence:   {', '.join(evidence)}"
            f"\\n   Port Entropy: {alert_record['host_context']['port_entropy']} | Active Targets: {alert_record['host_context']['destination_diversity']}\\n\""""
new_console = """            f"\\n   Evidence:   {', '.join(evidence)}"
            f"\\n   Port Entropy: {alert_record['host_context']['port_entropy']} | Active Targets: {alert_record['host_context']['destination_diversity']} | Drift: {alert_record['behavioral_drift_score']}/100\\n\""""
content = content.replace(old_console, new_console)


with open('alerts/engine.py', 'w') as f:
    f.write(content)

