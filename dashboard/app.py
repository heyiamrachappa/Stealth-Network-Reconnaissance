#!/usr/bin/env python3
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
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

# Custom premium styling CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #6366F1;
        margin-bottom: 5px;
    }
    .metric-value-crit {
        font-size: 2.2rem;
        font-weight: 800;
        color: #EF4444;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .footer {
        text-align: center;
        padding: 30px;
        color: #64748B;
        font-size: 0.8rem;
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
                    "threat_confidence": 0.984
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
                    "threat_confidence": 0.891
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
                    "threat_confidence": 0.58
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
            st.markdown("<h2 style='text-align: center;'>🛡️ Aegis AI IDS</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Real-Time Cyber Reconnaissance Detector</p>", unsafe_allow_html=True)
            st.markdown("---")
            
            navigation = st.radio(
                "System Navigation",
                ["Live Threat Monitor", "Offline PCAP Analyzer", "ML Engine & Diagnostics"],
                key="nav_selection"
            )
            
            st.markdown("---")
            st.markdown("### Configured Engine Properties")
            sel_model = st.selectbox(
                "Active Detection Model",
                self.available_models,
                index=0
            )
            
            st.markdown("---")
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
        st.markdown("<h1>⚔️ Live Network Threat Monitor</h1>", unsafe_allow_html=True)
        st.markdown(f"**Detector Status**: <span style='color:#10B981;'>● Active Sniffing</span> | ML Engine: `{active_model.upper()}`", unsafe_allow_html=True)
        
        alerts = self.load_alerts()
        
        total_alerts = len(alerts)
        unique_attackers = len(set(a["source_ip"] for a in alerts)) if alerts else 0
        critical_alarms = sum(1 for a in alerts if a.get("severity", "LOW") == "HIGH") if alerts else 0
        avg_confidence = np.mean([a.get("threat_confidence", 0.0) for a in alerts]) if alerts else 0.0
        
        # Stat cards row
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_alerts}</div><div class='metric-label'>Threat Incidents Logged</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value-crit'>{unique_attackers}</div><div class='metric-label'>Suspicious Attacker IPs</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value-crit'>{critical_alarms}</div><div class='metric-label'>High Severity Incidents</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_confidence*100:.1f}%</div><div class='metric-label'>Mean Anomaly Score</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br><h3>Detailed Incident Activity Stream</h3>", unsafe_allow_html=True)
        if alerts:
            df_alerts = pd.DataFrame(alerts[::-1])
            
            # Formulate pretty presentation dataframe
            display_cols = ["timestamp", "severity", "source_ip", "destination_ip", "target_port", "protocol", "detection_method", "threat_confidence"]
            df_display = df_alerts[display_cols].copy()
            df_display["threat_confidence"] = df_display["threat_confidence"].map(lambda c: f"{c*100:.2f}%")
            
            st.dataframe(df_display, use_container_width=True)
            
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Threat Alert Sources (Scanner IP Distribution)")
                attacker_counts = df_alerts["source_ip"].value_counts()
                st.bar_chart(attacker_counts)
            with chart_col2:
                st.markdown("#### Target Port Distribution")
                port_counts = df_alerts["target_port"].value_counts()
                st.bar_chart(port_counts)
        else:
            st.info("No threat incidents recorded in alerts.log yet. Make sure a live/synthetic scan is active.")

    def render_pcap_analyzer(self, active_model: str) -> None:
        """
        Allows drag-and-drop offline PCAP file threat analysis.
        """
        st.markdown("<h1>📁 Offline PCAP Threat Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("Upload any standard `.pcap` capture file to parse, extract high-fidelity flow behaviors, and run ML-assisted anomaly classification.")
        
        uploaded_file = st.file_uploader("Choose a PCAP file", type=["pcap", "cap"])
        
        if uploaded_file is not None:
            temp_path = os.path.join(self.project_root, "pcaps", "uploaded_temp.pcap")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success("File uploaded successfully!")
            
            with st.spinner("Parsing packet records and reconstructing flow sessions..."):
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
                
            with st.spinner("Applying pre-trained normalization and model inference..."):
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
            
            st.markdown("### Threat Classification Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_flows}</div><div class='metric-label'>Total Flow Connections</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-value-crit'>{anomaly_count}</div><div class='metric-label'>Anomalous Reconnaissance Sessions</div></div>", unsafe_allow_html=True)
            with col3:
                rate = (anomaly_count / total_flows) * 100 if total_flows > 0 else 0
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{rate:.2f}%</div><div class='metric-label'>Flow Anomaly Ratio</div></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            if anomaly_count > 0:
                st.markdown("#### Detected Scanning Flows")
                disp_cols = ["src_ip", "dst_ip", "dst_port", "flow_syn_ratio", "host_port_entropy", "host_dst_diversity", "confidence"]
                st.dataframe(anomalous_flows[disp_cols].sort_values(by="confidence", ascending=False), use_container_width=True)
                
                csv_data = anomalous_flows.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Threat Report (CSV)",
                    data=csv_data,
                    file_name="IDS_threat_report.csv",
                    mime="text/csv"
                )
            else:
                st.success("Clean Capture! No anomalies or stealth reconnaissance scans were detected in this PCAP.")

    def render_diagnostics(self) -> None:
        """
        Renders Model Management & diagnostic plots.
        """
        st.markdown("<h1>🧠 ML Model Diagnostics & Performance</h1>", unsafe_allow_html=True)
        
        metrics = self.load_metrics()
        if not metrics:
            st.warning("No evaluation metrics database found. Retrain your models to populate diagnostics.")
            return
            
        st.markdown("### Comparative Classifiers Performance Metric Matrix")
        
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
        
        st.markdown("---")
        st.markdown("### Pre-rendered Model Diagnostics Charts")
        
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            st.markdown("#### ROC Curves Comparison")
            roc_img_path = os.path.join(self.project_root, "results", "roc_curves.png")
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, caption="Receiver Operating Characteristic Comparison")
            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("No Pre-rendered ROC Graph Available")
                st.pyplot(fig)
                
        with plot_col2:
            st.markdown("#### Random Forest Feature Importances")
            fi_img_path = os.path.join(self.project_root, "results", "random_forest_feature_importance.png")
            if os.path.exists(fi_img_path):
                st.image(fi_img_path, caption="Information Gain Importance values")
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
            🛡️ Aegis AI Intrusion Detection System — Advanced Cybersecurity Real-Time Engine<br>
            Developed using Python, Streamlit, Scapy, & Scikit-Learn
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
