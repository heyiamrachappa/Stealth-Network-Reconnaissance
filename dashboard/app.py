#!/usr/bin/env python3
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_config, get_project_root
from capture.sniffer import PacketRecord
from flows.tracker import FlowSession, FlowTracker
from features.extractor import FeatureExtractor
from detection.engine import MLInferenceEngine
from alerts.engine import AlertEngine

# Setup page properties
st.set_page_config(
    page_title="Aegis AI - Stealth IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling CSS with Space Grotesk and Plus Jakarta Sans
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Base Setup */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F172A 0%, #020617 100%) !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Sidebar Navigation Style */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] div[class^="stButton"] button {
        background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%);
        border: none;
        color: #020617;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] div[class^="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5);
    }
    
    /* Typography Override */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.03em !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    
    /* Modern Title Neon Gradient */
    .neon-title {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.04em;
        text-shadow: 0 0 50px rgba(0, 242, 254, 0.1);
    }

    /* Glassmorphism Cyber Card Base */
    .cyber-card {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }
    .cyber-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #4FACFE, #00F2FE);
        opacity: 0.8;
    }
    .cyber-card-danger::before {
        background: linear-gradient(90deg, #EF4444, #EC4899);
    }
    .cyber-card-warning::before {
        background: linear-gradient(90deg, #F59E0B, #FF5E62);
    }
    .cyber-card:hover {
        transform: translateY(-5px);
        border-color: rgba(79, 172, 254, 0.4) !important;
        box-shadow: 0 15px 40px 0 rgba(0, 242, 254, 0.15);
    }

    /* Metric Values Styles */
    .metric-title {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-value-neo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .val-cyan {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .val-purple {
        background: linear-gradient(135deg, #A78BFA 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .val-red {
        background: linear-gradient(135deg, #F87171 0%, #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    .val-gold {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Pulsing Green Sniffer Status Indicator */
    .sniffer-status-bar {
        display: flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 10px 16px;
        border-radius: 12px;
        margin-bottom: 25px;
        width: fit-content;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 0 8px #10B981;
        animation: sniffer-pulse 2s infinite;
    }
    @keyframes sniffer-pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Cyber Threat Alerts Stream */
    .cyber-feed-container {
        max-height: 480px;
        overflow-y: auto;
        padding-right: 8px;
    }
    .feed-item {
        background: rgba(15, 23, 42, 0.4);
        border-left: 4px solid #3B82F6;
        border-radius: 0 16px 16px 0;
        padding: 18px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.2s ease;
    }
    .feed-item:hover {
        background: rgba(30, 41, 59, 0.45);
        transform: translateX(4px);
    }
    .feed-high {
        border-left-color: #EF4444;
        box-shadow: inset 4px 0 20px rgba(239, 68, 68, 0.05);
    }
    .feed-medium {
        border-left-color: #F59E0B;
        box-shadow: inset 4px 0 20px rgba(245, 158, 11, 0.05);
    }
    
    /* Badges */
    .cyber-badge {
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(2, 6, 17, 0.4);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* Integrated Image Styling */
    .cyber-image-frame {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 8px;
        background: rgba(2, 6, 17, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Futuristic Table Override */
    div[data-testid="stTable"] table {
        background-color: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px;
    }
    div[data-testid="stTable"] th {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #94A3B8 !important;
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


class DashboardApp:
    """
    Renders the unified interactive front-end.
    Provides live traffic monitoring, manual PCAP uploading analysis, and model training analytics.
    """
    def __init__(self):
        self.config = load_config()
        self.project_root = self.config.get("project_root", get_project_root())
        self.alert_log_file = self.config.get("alerts", {}).get("alert_log_file", os.path.join(self.project_root, "logs", "alerts.log"))
        self.metrics_file = os.path.join(self.project_root, "results", "model_metrics.json")
        self.models_dir = os.path.join(self.project_root, "models")
        
        self.available_models = ["random_forest", "xgboost", "svm", "isolation_forest"]

    def load_metrics(self) -> Dict[str, Any]:
        """
        Loads pre-compiled model evaluation metrics.
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Failed to read evaluation metrics: {e}")
        return {}

    def load_alerts(self) -> List[Dict[str, Any]]:
        """
        Reads the real-time threat detection alerts log file.
        """
        alerts = []
        if os.path.exists(self.alert_log_file):
            try:
                with open(self.alert_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            alerts.append(json.loads(line.strip()))
            except Exception as e:
                st.error(f"Failed to read alert logs: {e}")
        return alerts

    def bootstrap_mock_data(self) -> None:
        """
        Populates mock stats and metrics logs if uninitialized.
        """
        os.makedirs(os.path.join(self.project_root, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "results"), exist_ok=True)
        
        if not os.path.exists(self.alert_log_file) or os.path.getsize(self.alert_log_file) == 0:
            mock_alerts = [
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600)),
                    "alert_id": "IDS-ALERT-1001",
                    "severity": "HIGH",
                    "threat_category": "Stealth Reconnaissance Scan",
                    "detection_method": "ML Inference (Random Forest)",
                    "source_ip": "192.168.1.187",
                    "destination_ip": "192.168.1.254",
                    "target_port": 80,
                    "protocol": "TCP",
                    "flow_statistics": {"packets": 2, "duration_seconds": 0.05, "syn_ratio": 1.0, "rst_ratio": 0.0},
                    "host_context": {"port_entropy": 2.87, "destination_diversity": 12, "failed_flow_ratio": 0.95},
                    "threat_confidence": 0.984,
                    "trigger_evidence": ["ML Model Classification (Confidence: 98.4%)", "Critical Destination Port Entropy (2.87)", "Severe Failed Connection Rate (95.0%)"]
                },
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1800)),
                    "alert_id": "IDS-ALERT-1002",
                    "severity": "HIGH",
                    "threat_category": "Stealth Reconnaissance Scan",
                    "detection_method": "ML Inference (Isolation Forest)",
                    "source_ip": "10.0.0.45",
                    "destination_ip": "10.0.0.12",
                    "target_port": 443,
                    "protocol": "TCP",
                    "flow_statistics": {"packets": 1, "duration_seconds": 0.0, "syn_ratio": 1.0, "rst_ratio": 0.0},
                    "host_context": {"port_entropy": 3.41, "destination_diversity": 1, "failed_flow_ratio": 1.00},
                    "threat_confidence": 0.891,
                    "trigger_evidence": ["ML Model Anomaly Detected", "Severe Destination Port Entropy (3.41)"]
                },
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 300)),
                    "alert_id": "IDS-ALERT-1003",
                    "severity": "MEDIUM",
                    "threat_category": "Stealth Reconnaissance Scan",
                    "detection_method": "Behavioral Heuristics",
                    "source_ip": "192.168.43.92",
                    "destination_ip": "192.168.43.1",
                    "target_port": 22,
                    "protocol": "TCP",
                    "flow_statistics": {"packets": 3, "duration_seconds": 1.25, "syn_ratio": 0.67, "rst_ratio": 0.33},
                    "host_context": {"port_entropy": 1.95, "destination_diversity": 4, "failed_flow_ratio": 0.75},
                    "threat_confidence": 0.58,
                    "trigger_evidence": ["Elevated Failed Connection Rate (75.0%)", "Moderate Destination Port Entropy (1.95)"]
                }
            ]
            with open(self.alert_log_file, 'w') as f:
                for a in mock_alerts:
                    f.write(json.dumps(a) + "\n")
                    
        if not os.path.exists(self.metrics_file):
            mock_metrics = {
                "random_forest": {
                    "accuracy": 0.992, "precision": 0.985, "recall": 0.991, "f1_score": 0.988, "auc": 0.998,
                    "confusion_matrix": [[1200, 3], [2, 240]]
                },
                "xgboost": {
                    "accuracy": 0.995, "precision": 0.992, "recall": 0.992, "f1_score": 0.992, "auc": 0.999,
                    "confusion_matrix": [[1201, 2], [2, 240]]
                },
                "svm": {
                    "accuracy": 0.978, "precision": 0.965, "recall": 0.970, "f1_score": 0.967, "auc": 0.989,
                    "confusion_matrix": [[1195, 8], [7, 235]]
                },
                "isolation_forest": {
                    "accuracy": 0.912, "precision": 0.885, "recall": 0.892, "f1_score": 0.888, "auc": 0.932,
                    "confusion_matrix": [[1150, 53], [26, 216]]
                }
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(mock_metrics, f, indent=4)

    def draw_sidebar(self) -> Tuple[str, str]:
        """
        Renders left navigation layout.
        """
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; margin-top: 20px; margin-bottom: 0;'>🛡️ AEGIS AI</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #4FACFE; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 30px;'>Stealth IDS Command Center</p>", unsafe_allow_html=True)
            
            navigation = st.radio(
                "System Navigation",
                ["Live Threat Monitor", "Offline PCAP Analyzer", "ML Engine & Diagnostics"],
                key="nav_selection"
            )
            
            st.markdown("<br><hr style='border-top: 1px solid rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
            st.markdown("<h4>Engine Settings</h4>", unsafe_allow_html=True)
            sel_model = st.selectbox(
                "Inference Engine Classifier",
                self.available_models,
                index=0
            )
            
            st.markdown("<br><hr style='border-top: 1px solid rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
            if st.button("Generate Synthetic Scan PCAP"):
                st.info("Creating a synthetic TCP/IP capture file in `pcaps/synthetic_scan.pcap`...")
                self.create_mock_pcap_file()
                st.success("Successfully created `pcaps/synthetic_scan.pcap`!")
                
            return navigation, sel_model

    def create_mock_pcap_file(self) -> None:
        """
        Writes a valid basic mock PCAP using Scapy for testing components.
        """
        try:
            from scapy.all import IP, TCP, wrpcap
            pkts = []
            base_time = time.time()
            
            # Normal flows
            for i in range(10):
                sport = 49000 + i
                pkts.append(IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="S"))
                pkts[-1].time = base_time + (i * 0.2)
                pkts.append(IP(src="8.8.8.8", dst="192.168.1.10")/TCP(sport=80, dport=sport, flags="SA"))
                pkts[-1].time = base_time + (i * 0.2) + 0.02
                pkts.append(IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="A"))
                pkts[-1].time = base_time + (i * 0.2) + 0.04
                
            # Scan flows (Attacker: 192.168.1.187 targeting victim 192.168.1.5 ports 20-50)
            for i in range(30):
                target_port = 20 + i
                pkts.append(IP(src="192.168.1.187", dst="192.168.1.5")/TCP(sport=35000+i, dport=target_port, flags="S"))
                pkts[-1].time = base_time + 5.0 + (i * 0.5)
                
            out_path = os.path.join(self.project_root, "pcaps", "synthetic_scan.pcap")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            wrpcap(out_path, pkts)
        except Exception as e:
            st.error(f"Failed to generate synthetic PCAP: {e}")

    def render_live_monitor(self, active_model: str) -> None:
        """
        Renders live monitoring view.
        """
        st.markdown("<div class='neon-title'>⚔️ Live Threat Stream Console</div>", unsafe_allow_html=True)
        
        # Pulsing active status bar
        st.markdown(f"""
        <div class='sniffer-status-bar'>
            <div class='status-dot'></div>
            <span style='font-size: 0.85rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.05em;'>
                Active Security Stream — ML Model Engine: {active_model.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        alerts = self.load_alerts()
        
        total_alerts = len(alerts)
        unique_attackers = len(set(a["source_ip"] for a in alerts)) if alerts else 0
        critical_alarms = sum(1 for a in alerts if a.get("severity", "LOW") == "HIGH") if alerts else 0
        avg_confidence = np.mean([a.get("threat_confidence", 0.0) for a in alerts]) if alerts else 0.0
        
        # Stat cards row in premium glassmorphic grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='cyber-card'>
                <div class='metric-title'>Total Alarms Triggers</div>
                <div class='metric-value-neo val-cyan'>{total_alerts}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='cyber-card'>
                <div class='metric-title'>Host Scanner Entities</div>
                <div class='metric-value-neo val-purple'>{unique_attackers}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='cyber-card cyber-card-danger'>
                <div class='metric-title'>High Severity Hazards</div>
                <div class='metric-value-neo val-red'>{critical_alarms}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='cyber-card'>
                <div class='metric-title'>Avg Threat Intensity</div>
                <div class='metric-value-neo val-gold'>{avg_confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        main_col, side_col = st.columns([3, 2])
        
        with main_col:
            st.markdown("<h3>Detailed Incident Activity Log</h3>", unsafe_allow_html=True)
            if alerts:
                df_alerts = pd.DataFrame(alerts[::-1])
                display_cols = ["timestamp", "severity", "source_ip", "destination_ip", "target_port", "protocol", "detection_method", "threat_confidence"]
                df_display = df_alerts[display_cols].copy()
                df_display["threat_confidence"] = df_display["threat_confidence"].map(lambda c: f"{c*100:.2f}%")
                
                st.dataframe(df_display, use_container_width=True)
                
                st.markdown("---")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("#### Scanner Distribution Mapping")
                    attacker_counts = df_alerts["source_ip"].value_counts()
                    st.bar_chart(attacker_counts)
                with chart_col2:
                    st.markdown("#### Targeted Port Frequency")
                    port_counts = df_alerts["target_port"].value_counts()
                    st.bar_chart(port_counts)
            else:
                st.info("No threat incidents recorded in alerts.log yet. Make sure a live/synthetic scan is active.")
                
        with side_col:
            st.markdown("<h3>Futuristic Alerts Live Feed</h3>", unsafe_allow_html=True)
            if alerts:
                # Beautiful scrollable alert feed
                st.markdown("<div class='cyber-feed-container'>", unsafe_allow_html=True)
                for alert in alerts[::-1][:10]:  # Limit to top 10 recent alerts
                    sev = alert.get("severity", "LOW")
                    feed_class = "feed-high" if sev == "HIGH" else "feed-medium"
                    badge_class = "badge-high" if sev == "HIGH" else "badge-medium"
                    
                    evidence_str = "".join([f"<li>{ev}</li>" for ev in alert.get("trigger_evidence", ["Stealth SCAN signature detected"])])
                    
                    st.markdown(f"""
                    <div class='feed-item {feed_class}'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <span class='cyber-badge {badge_class}'>{sev} Severity</span>
                            <span style='font-family: "JetBrains Mono", monospace; font-size: 0.75rem; color: #64748B;'>{alert['timestamp']}</span>
                        </div>
                        <div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;'>
                            {alert.get('threat_category', 'Stealth Reconnaissance Scan')}
                        </div>
                        <div style='font-size: 0.85rem; color: #94A3B8; font-family: "JetBrains Mono", monospace; margin-bottom: 10px;'>
                            SOURCE IP: <span style='color: #4FACFE;'>{alert['source_ip']}</span> -> TARGET IP: <span style='color: #FBBF24;'>{alert['destination_ip']}:{alert['target_port']}</span> ({alert['protocol']})
                        </div>
                        <div style='font-size: 0.8rem; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.03);'>
                            <span style='font-weight: 600; color: #38BDF8;'>Correlated Evidences:</span>
                            <ul style='margin: 4px 0 0 15px; padding: 0; color: #E2E8F0; list-style-type: square;'>
                                {evidence_str}
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Log console empty. Run real-time scans to pipe alerts.")

    def render_pcap_analyzer(self, active_model: str) -> None:
        """
        Allows drag-and-drop offline PCAP file threat analysis.
        """
        st.markdown("<div class='neon-title'>📁 High-Tech PCAP Analyzer</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 1.05rem; margin-bottom: 25px;'>Upload any standard standard packet capture file to run real-time behavioral sliding feature extractions and ML model predictions.</p>", unsafe_allow_html=True)
        
        # Cyber upload wrapper
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop your security captures (.pcap / .cap)", type=["pcap", "cap"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            temp_path = os.path.join(self.project_root, "pcaps", "uploaded_temp.pcap")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success("PCAP Capture File Loaded successfully!")
            
            with st.spinner("Decoding packets and tracking connection session states..."):
                from scapy.all import PcapReader
                from capture.sniffer import PacketSniffer
                
                tracker = FlowTracker(flow_timeout_seconds=99999)  # Infinite timeout for batch parse
                
                try:
                    with PcapReader(temp_path) as reader:
                        for pkt in reader:
                            parsed = PacketSniffer.parse_packet(pkt)
                            if parsed:
                                tracker.handle_packet(parsed)
                except Exception as e:
                    st.error(f"Failed to read PCAP: {e}")
                    return
                    
                flows = tracker.get_active_sessions()
                if not flows:
                    st.error("No valid IP/TCP/UDP packets found in the uploaded PCAP file.")
                    return
                    
                extractor = FeatureExtractor()
                df_features = extractor.extract_features(flows)
                
            with st.spinner("Applying scale standardizer & executing model classifier predictions..."):
                inference_engine = MLInferenceEngine(model_name=active_model)
                
                preds = []
                confidences = []
                for _, row in df_features.iterrows():
                    feat_dict = row.to_dict()
                    pred, conf = inference_engine.predict(feat_dict)
                    preds.append(pred)
                    confidences.append(conf)
                    
                df_features["is_anomaly"] = preds
                df_features["confidence"] = confidences
                
            anomalous_flows = df_features[df_features["is_anomaly"] == 1]
            total_flows = len(df_features)
            anomaly_count = len(anomalous_flows)
            
            st.markdown("<br><h3>Threat Classification Report Summary</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='cyber-card'>
                    <div class='metric-title'>Flow Sessions Parsed</div>
                    <div class='metric-value-neo val-cyan'>{total_flows}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='cyber-card cyber-card-danger'>
                    <div class='metric-title'>Stealth Recon Incidents</div>
                    <div class='metric-value-neo val-red'>{anomaly_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                rate = (anomaly_count / total_flows) * 100 if total_flows > 0 else 0
                st.markdown(f"""
                <div class='cyber-card'>
                    <div class='metric-title'>Host Anomaly Ratio</div>
                    <div class='metric-value-neo val-gold'>{rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            if anomaly_count > 0:
                st.markdown("#### Confirmed Anomalous SCAN Sessions")
                disp_cols = ["src_ip", "dst_ip", "dst_port", "flow_syn_ratio", "host_port_entropy", "host_dst_diversity", "confidence"]
                st.dataframe(anomalous_flows[disp_cols].sort_values(by="confidence", ascending=False), use_container_width=True)
                
                csv_data = anomalous_flows.to_csv(index=False).encode('utf-8')
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Threat Forensic Report (CSV)",
                    data=csv_data,
                    file_name="Forensic_Threat_Report.csv",
                    mime="text/csv"
                )
            else:
                st.balloons()
                st.success("Clean Capture! No anomalies or stealth reconnaissance scans were detected in this PCAP.")

    def render_diagnostics(self) -> None:
        """
        Renders Model Management & diagnostic plots.
        """
        st.markdown("<div class='neon-title'>🧠 ML Engine Diagnostics & Performance</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 1.05rem; margin-bottom: 25px;'>Analyze offline evaluation reports, roc curve graphs, and relative information gain importance values for features.</p>", unsafe_allow_html=True)
        
        metrics = self.load_metrics()
        if not metrics:
            st.warning("No evaluation metrics database found. Retrain your models to populate diagnostics.")
            return
            
        st.markdown("<h3>Comparative Multi-Classifier Evaluation Matrix</h3>", unsafe_allow_html=True)
        
        metric_rows = []
        for name, data in metrics.items():
            metric_rows.append({
                "Classifier Model": name.replace('_', ' ').upper(),
                "Accuracy": f"{data.get('accuracy', 0.0)*100:.2f}%",
                "Precision": f"{data.get('precision', 0.0)*100:.2f}%",
                "Recall (Detection Rate)": f"{data.get('recall', 0.0)*100:.2f}%",
                "F1-Score": f"{data.get('f1_score', 0.0)*100:.2f}%",
                "ROC-AUC": f"{data.get('auc', 0.0):.4f}" if "auc" in data else "N/A"
            })
        st.table(pd.DataFrame(metric_rows))
        
        st.markdown("<br><hr style='border-top: 1px solid rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
        st.markdown("<h3>Pre-rendered Diagnostic Diagnostic Visualizations</h3>", unsafe_allow_html=True)
        
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            st.markdown("#### Receiver Operating Characteristic (ROC) curves")
            roc_img_path = os.path.join(self.project_root, "results", "roc_curves.png")
            if os.path.exists(roc_img_path):
                st.markdown("<div class='cyber-image-frame'>", unsafe_allow_html=True)
                st.image(roc_img_path, caption="Comparative ROC curves")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("No Pre-rendered ROC Graph Available")
                st.pyplot(fig)
                
        with plot_col2:
            st.markdown("#### Relative Feature Significances")
            fi_img_path = os.path.join(self.project_root, "results", "random_forest_feature_importance.png")
            if os.path.exists(fi_img_path):
                st.markdown("<div class='cyber-image-frame'>", unsafe_allow_html=True)
                st.image(fi_img_path, caption="Information Gain significance values")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.set_title("No Pre-rendered Feature Significance Available")
                st.pyplot(fig)

    def run(self) -> None:
        self.bootstrap_mock_data()
        navigation, active_model = self.draw_sidebar()
        
        if navigation == "Live Threat Monitor":
            self.render_live_monitor(active_model)
        elif navigation == "Offline PCAP Analyzer":
            self.render_pcap_analyzer(active_model)
        elif navigation == "ML Engine & Diagnostics":
            self.render_diagnostics()
            
        st.markdown("""
        <div class='footer'>
            🛡️ AEGIS AI Cyber Defense Platform — Real-Time Streaming Reconnaissance Intrusion Detection System<br>
            Developed using Python, Streamlit, Scapy, & Scikit-Learn
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
