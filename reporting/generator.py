#!/usr/bin/env python3
import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
from threat_analysis.analyzer import ForensicThreatReport

class ForensicReporter:
    """
    Serializes forensic logs, generates markdown summaries,
    and handles exporting to CSV/JSON format.
    """
    @staticmethod
    def export_csv(reports: List[ForensicThreatReport], output_path: str) -> None:
        """
        Exports detailed threat metrics to a CSV spreadsheet.
        """
        rows = []
        for r in reports:
            data = r.to_dict()
            # Flatten lists to comma-separated strings for CSV compatibility
            data["mitre_mappings"] = ", ".join(data["mitre_mappings"])
            data["evidence"] = "; ".join(data["evidence"])
            rows.append(data)
            
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    @staticmethod
    def export_json(reports: List[ForensicThreatReport], output_path: str) -> None:
        """
        Appends or writes alerts in structured JSON-line format.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for r in reports:
                f.write(json.dumps(r.to_dict()) + "\n")

    @staticmethod
    def generate_markdown_summary(reports: List[ForensicThreatReport], 
                                  pcap_metadata: Dict[str, Any], 
                                  model_name: str) -> str:
        """
        Compiles a professional high-tech cyber forensic assessment summary.
        """
        total_sessions = len(reports)
        anomalies = [r for r in reports if r.ml_prediction == 1 or r.severity in ["CRITICAL", "HIGH", "MEDIUM"]]
        anom_count = len(anomalies)
        
        # Calculate severity counts
        sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in anomalies:
            sev_counts[a.severity] += 1
            
        unique_scanners = set(a.src_ip for a in anomalies)
        
        # MITRE Mapping aggregates
        mitre_counts = {}
        for a in anomalies:
            for m in a.mitre_mappings:
                mitre_counts[m] = mitre_counts.get(m, 0) + 1
                
        # Top scanner IP breakdown
        scanner_summary = {}
        for a in anomalies:
            if a.src_ip not in scanner_summary:
                scanner_summary[a.src_ip] = {
                    "total_probes": 0,
                    "max_score": 0.0,
                    "ports_targeted": set(),
                    "category": a.scan_category
                }
            s = scanner_summary[a.src_ip]
            s["total_probes"] += 1
            s["max_score"] = max(s["max_score"], a.threat_score)
            s["ports_targeted"].add(a.dst_port)

        summary_md = f"""# PhantomTrace // DIGITAL PCAP FORENSIC REPORT
**Timestamp of Assessment:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
**Classifier Inference Model:** {model_name.upper()}

---

## 1. CAPTURE DATA ASSESSMENT METADATA
- **Target Filename:** `{pcap_metadata.get('file_name', 'N/A')}`
- **Capture File Size:** `{pcap_metadata.get('file_size_mb', 0.0):.3f} MB`
- **Total Capture Packets:** `{pcap_metadata.get('total_packets', 0)}`
- **Total Decoded Flows:** `{pcap_metadata.get('total_packets', 0)}`
- **Unique Scaped IP Entities:** `{pcap_metadata.get('unique_ips_count', 0)}`
- **Capture Absolute Duration:** `{pcap_metadata.get('duration_seconds', 0.0):.2f} seconds`
- **Assessment Span:** `{pcap_metadata.get('start_time_utc', 'N/A')} — {pcap_metadata.get('end_time_utc', 'N/A')}`

---

## 2. HIGH-LEVEL EXECUTIVE SUMMARY
Within the analyzed Wireshark packet capture, PhantomTrace has scanned and reconstructed connection session states.

- **Analyzed Session Conversions:** `{total_sessions}`
- **Anomalous Reconnaissance Flags:** `{anom_count} ({ (anom_count / total_sessions * 100) if total_sessions > 0 else 0:.1f}%)`
- **Identified Threat Host Entities:** `{len(unique_scanners)}`
- **Forensic Severity Classification:**
  - **CRITICAL HAZARDS:** `{sev_counts['CRITICAL']}`
  - **HIGH HAZARDS:** `{sev_counts['HIGH']}`
  - **MEDIUM HAZARDS:** `{sev_counts['MEDIUM']}`
  - **LOW RISK SESSIONS:** `{sev_counts['LOW']}`

---

## 3. MITRE ATT&CK TECHNIQUES COVERAGE
Reconnaissance behaviors matched to standard MITRE ATT&CK enterprise catalog:
"""
        for tech, count in mitre_counts.items():
            summary_md += f"- **{tech}**: Observed in `{count}` sessions\n"
            
        summary_md += """
---

## 4. DETECTED HOST PROFILES
Details of active entities flagged during temporal sliding-window analysis:
"""
        for ip, stats in scanner_summary.items():
            summary_md += f"""
### Attacker IP Profile: `{ip}`
- **Primary Recon Category:** `{stats['category']}`
- **Total Anomaly Flows:** `{stats['total_probes']}`
- **Composite Risk Rating:** `{stats['max_score']:.1f}%`
- **Target Ports Count:** `{len(stats['ports_targeted'])}`
- **Target Ports Scanned:** `{sorted(list(stats['ports_targeted']))[:15]}`
"""
        return summary_md
