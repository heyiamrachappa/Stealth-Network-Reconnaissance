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
from pcap_processing.parser import PCAPParser, PacketRecord
from feature_extraction.extractor import FeatureExtractor, StaticFlowTracker, FlowSession
from ml_engine.engine import MLInferenceEngine
from threat_analysis.analyzer import ThreatAnalyzer, ForensicThreatReport
from reporting.generator import ForensicReporter

# Setup page properties
st.set_page_config(
    page_title="PhantomTrace // OFFLINE PCAP FORENSIC WORKSTATION",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modernistic Forensic CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global styling and Cyber grid overlay */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #060913 0%, #010204 100%) !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 242, 254, 0.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 242, 254, 0.012) 1px, transparent 1px);
        background-size: 35px 35px;
        pointer-events: none;
        z-index: 0;
    }

    /* Left Sidebar Navigation override */
    section[data-testid="stSidebar"] {
        background: rgba(4, 7, 14, 0.95) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
        backdrop-filter: blur(25px);
        box-shadow: 5px 0 35px rgba(0,0,0,0.85);
    }
    
    /* Interactive Sidebar menu buttons */
    div[data-testid="stSidebarUserContent"] button {
        background: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #94A3B8 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stSidebarUserContent"] button:hover {
        background: rgba(0, 242, 254, 0.1) !important;
        border-color: #00F2FE !important;
        color: #00F2FE !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.25);
    }

    /* Headings and Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: -0.01em !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.05);
    }

    /* Forensic Lab HUD Bar */
    .forensic-hud-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(8, 14, 27, 0.85);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-bottom: 2px solid rgba(0, 242, 254, 0.35);
        padding: 14px 20px;
        border-radius: 4px;
        margin-bottom: 25px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.05);
        position: relative;
    }
    .lbl { color: #64748B; font-weight: 600; }
    .val { font-weight: 700; margin-left: 6px; }
    .cyan-glow { color: #00F2FE; text-shadow: 0 0 8px rgba(0, 242, 254, 0.5); }
    .purple-glow { color: #A78BFA; text-shadow: 0 0 8px rgba(167, 139, 250, 0.5); }
    .red-glow { color: #EF4444; text-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }
    .green-glow { color: #10B981; text-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }

    /* Tactical Title Headers */
    .lab-header {
        border-left: 4px solid #00F2FE;
        padding-left: 12px;
        margin-bottom: 20px;
    }
    .lab-subtitle {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        color: #4FACFE;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: -8px;
    }

    /* Cyber Lab Panels */
    .cyber-panel {
        background: rgba(8, 12, 21, 0.75) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.15) !important;
        border-radius: 4px !important;
        padding: 20px !important;
        position: relative;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8), inset 0 0 20px rgba(0, 242, 254, 0.01);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .cyber-panel::before {
        content: '';
        position: absolute;
        top: -1px;
        left: -1px;
        width: 12px;
        height: 12px;
        border-top: 2px solid #00F2FE;
        border-left: 2px solid #00F2FE;
    }
    .cyber-panel::after {
        content: '';
        position: absolute;
        bottom: -1px;
        right: -1px;
        width: 12px;
        height: 12px;
        border-bottom: 2px solid #8B5CF6;
        border-right: 2px solid #8B5CF6;
    }
    
    .panel-critical { border-color: rgba(239, 68, 68, 0.4) !important; }
    .panel-critical::before { border-top-color: #EF4444; border-left-color: #EF4444; }
    .panel-critical::after { border-bottom-color: #EF4444; border-right-color: #EF4444; }

    /* Lab Telemetry HUD Labels */
    .hud-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 5px;
    }
    .hud-value-large {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 0.9;
        margin-bottom: 8px;
    }

    /* Radar Sweep animation */
    .radar-container {
        width: 180px;
        height: 180px;
        border: 2px solid rgba(0, 242, 254, 0.18);
        border-radius: 50%;
        position: relative;
        background: radial-gradient(circle, rgba(0,242,254,0.03) 0%, rgba(4,7,15,0.95) 100%);
        overflow: hidden;
        margin: 0 auto;
        box-shadow: 0 0 30px rgba(0,0,0,0.85);
    }
    .radar-sweep {
        width: 100%;
        height: 100%;
        background: conic-gradient(from 0deg, rgba(0, 242, 254, 0.35) 0deg, transparent 70deg, transparent 360deg);
        border-radius: 50%;
        position: absolute;
        top: 0;
        left: 0;
        animation: rotate-radar 3.2s linear infinite;
        transform-origin: center;
    }
    .radar-cross-h {
        width: 100%;
        height: 1px;
        background: rgba(0, 242, 254, 0.12);
        position: absolute;
        top: 50%;
        left: 0;
    }
    .radar-cross-v {
        width: 1px;
        height: 100%;
        background: rgba(0, 242, 254, 0.12);
        position: absolute;
        top: 0;
        left: 50%;
    }
    @keyframes rotate-radar {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Tactical Alerts stream */
    .forensic-alerts-feed {
        max-height: 550px;
        overflow-y: auto;
        padding-right: 6px;
    }
    .alert-card {
        background: rgba(10, 15, 27, 0.55);
        border-left: 4px solid #3B82F6;
        border-top: 1px solid rgba(255, 255, 255, 0.02);
        border-right: 1px solid rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
        transition: all 0.2s ease-in-out;
    }
    .alert-card:hover {
        background: rgba(15, 23, 42, 0.65);
        transform: translateX(4px);
    }
    
    .card-crit { border-left-color: #EF4444; box-shadow: inset 4px 0 20px rgba(239, 68, 68, 0.05); }
    .card-high { border-left-color: #FF5E62; box-shadow: inset 4px 0 20px rgba(255, 94, 98, 0.04); }
    .card-med { border-left-color: #FBBF24; box-shadow: inset 4px 0 20px rgba(251, 191, 36, 0.04); }
    .card-low { border-left-color: #06B6D4; box-shadow: inset 4px 0 20px rgba(6, 182, 212, 0.04); }

    .tag-sec {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        border: 1px solid currentColor;
    }
    .tag-crit { color: #EF4444; background: rgba(239, 68, 68, 0.08); }
    .tag-high { color: #FF9900; background: rgba(255, 153, 0, 0.08); }
    .tag-med { color: #FBBF24; background: rgba(251, 191, 36, 0.08); }
    .tag-low { color: #06B6D4; background: rgba(6, 182, 212, 0.08); }

    /* MITRE ATT&CK visual grids */
    .mitre-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 15px;
    }
    .mitre-card {
        background: rgba(14, 22, 40, 0.5);
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 4px;
        padding: 12px;
        text-align: center;
    }
    .mitre-id {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        color: #00F2FE;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .mitre-name {
        font-size: 0.8rem;
        color: #F8FAFC;
        font-weight: 600;
    }

    /* Custom technical dividers */
    .tech-line {
        height: 1px;
        background: linear-gradient(90deg, rgba(0,242,254,0.05) 0%, rgba(0,242,254,0.4) 50%, rgba(139,92,246,0.05) 100%);
        margin: 25px 0;
        position: relative;
    }
    .tech-line::after {
        content: '[ LAB FORENSIC PERIMETER SECURE ]';
        position: absolute;
        top: -7px;
        left: 50%;
        transform: translateX(-50%);
        background: #020306;
        padding: 0 12px;
        color: #64748B;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.62rem;
        letter-spacing: 0.15em;
    }

    /* Drag & Drop File Loader Override */
    div[data-testid="stFileUploader"] {
        background: rgba(10, 15, 27, 0.6) !important;
        border: 1.5px dashed rgba(0, 242, 254, 0.3) !important;
        border-radius: 6px !important;
        padding: 15px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #00F2FE !important;
        background: rgba(0, 242, 254, 0.03) !important;
    }

    /* Forensic Table Override */
    div[data-testid="stTable"] table {
        background-color: rgba(4, 7, 15, 0.6) !important;
        border: 1px solid rgba(0, 242, 254, 0.15) !important;
        border-collapse: collapse;
    }
    div[data-testid="stTable"] th {
        background: rgba(8, 14, 27, 0.9) !important;
        border-bottom: 2px solid rgba(0, 242, 254, 0.3) !important;
        color: #F8FAFC !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


class ForensicWorkstationApp:
    """
    Unified Offline Cybersecurity PCAP Reconnaissance Forensics Workstation.
    Statically analyzes Wireshark capture packages.
    """
    def __init__(self):
        self.config = load_config()
        self.project_root = self.config.get("project_root", get_project_root())
        self.available_models = ["random_forest", "xgboost", "svm", "isolation_forest"]
        self.metrics_file = os.path.join(self.project_root, "results", "model_metrics.json")

    def load_metrics(self) -> Dict[str, Any]:
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Failed to read metrics database: {e}")
        return {}

    def draw_hud_bar(self, filename: str, packets: int, anomalies: int, duration: float) -> None:
        st.markdown(f"""
        <div class="forensic-hud-bar">
            <div class="hud-item"><span class="lbl">WORKSTATION:</span> <span class="val cyan-glow">PhantomTrace v3.0</span></div>
            <div class="hud-item"><span class="lbl">PCAP LOADED:</span> <span class="val purple-glow">{filename}</span></div>
            <div class="hud-item"><span class="lbl">PACKETS:</span> <span class="val cyan-glow">{packets} pkts</span></div>
            <div class="hud-item"><span class="lbl">DURATION:</span> <span class="val cyan-glow">{duration:.2f}s</span></div>
            <div class="hud-item"><span class="lbl">ANOMALIES:</span> <span class="val red-glow">{anomalies} DETECTED</span></div>
            <div class="hud-item"><span class="lbl">RISK:</span> <span class="val red-glow">{"HIGH" if anomalies > 0 else "SECURE"}</span></div>
        </div>
        """, unsafe_allow_html=True)

    def generate_radar_sweep_plot(self, reports: List[ForensicThreatReport]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        ax.spines['polar'].set_color((0.0, 0.949, 0.996, 0.25))
        ax.grid(True, color=(0.0, 0.949, 0.996, 0.12), linestyle='dashed')
        ax.tick_params(colors=(0.0, 0.949, 0.996, 0.55), labelsize=7)
        
        theta = np.linspace(0, 2*np.pi, 100)
        ax.fill_between(theta, 0, 95, color=(0.0, 0.949, 0.996, 0.025))
        
        # Plot parsed threat reports as blips
        r_blips = []
        theta_blips = []
        color_blips = []
        
        # Compile threat anomalies
        anomalies = [r for r in reports if r.severity in ["CRITICAL", "HIGH", "MEDIUM"]]
        for i, a in enumerate(anomalies[:40]):  # Limit to 40 polar elements
            # Generate polar coordinates dynamically based on target ports
            r = min(90.0, 15.0 + (a.dst_port % 75.0))
            t = (hash(a.src_ip) % 360) * (np.pi / 180.0)
            
            r_blips.append(r)
            theta_blips.append(t)
            
            if a.severity == "CRITICAL":
                color_blips.append("#EF4444")
            elif a.severity == "HIGH":
                color_blips.append("#FF5E62")
            else:
                color_blips.append("#FBBF24")
                
        if r_blips:
            ax.scatter(theta_blips, r_blips, c=color_blips, s=80, alpha=0.9, edgecolors='#F8FAFC', linewidths=1.2, zorder=5)
            
        ax.set_title("Reconnaissance Coordinate Radar Map", color='#F8FAFC', fontname='Orbitron', fontsize=10, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig

    def generate_anomaly_timeline_plot(self, reports: List[ForensicThreatReport]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.15))
        ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#94A3B8', labelsize=8)
        ax.grid(True, color=(1.0, 1.0, 1.0, 0.04), linestyle='solid')
        
        if reports:
            # Group anomalies chronologically in chunks
            df = pd.DataFrame([r.to_dict() for r in reports])
            df["time"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("time")
            # Build rolling count
            df["anomaly_cum"] = (df["ml_confidence"] > 0.5).astype(int).cumsum()
            
            ax.plot(df["time"], df["anomaly_cum"], color='#8B5CF6', linewidth=2.5, label='Recon Infiltration Vector')
            ax.fill_between(df["time"], 0, df["anomaly_cum"], color=(0.545, 0.361, 0.965, 0.12))
            
        ax.legend(facecolor='#060913', edgecolor=(1.0, 1.0, 1.0, 0.1), labelcolor='#E2E8F0', fontsize=8)
        ax.set_title("Chronological Behavioral Escalation", color='#F8FAFC', fontname='Orbitron', fontsize=10, fontweight='bold')
        plt.tight_layout()
        return fig

    def draw_sidebar(self) -> Tuple[str, str]:
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; margin-top: 15px; margin-bottom: 0;'>👻 PhantomTrace</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #00F2FE; font-size: 0.65rem; font-family:\"Orbitron\",sans-serif; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 25px;'>Stealth Forensics Platform</p>", unsafe_allow_html=True)
            
            navigation = st.radio(
                "DECK SECTIONS",
                [
                    "PCAP Upload Center", 
                    "Forensic Overview", 
                    "MITRE ATT&CK Mapping", 
                    "ML Analysis Diagnostics",
                    "Settings & Utilities"
                ],
                key="nav_selection"
            )
            
            st.markdown("<br><hr style='border-top: 1px solid rgba(0,242,254,0.15);'><br>", unsafe_allow_html=True)
            st.markdown("<h4>FORENSIC CLASSIFIER</h4>", unsafe_allow_html=True)
            sel_model = st.selectbox(
                "Analysis Inference Engine",
                self.available_models,
                index=0
            )
            
            st.markdown("<br><hr style='border-top: 1px solid rgba(0,242,254,0.15);'><br>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.75rem; color:#64748B; font-family:\"JetBrains Mono\"; text-align:center;'>LAB OPERATOR LEVEL 4 // CLASSIFIED SESSIONS</p>", unsafe_allow_html=True)
            
            return navigation, sel_model

    def create_mock_pcap_file(self) -> None:
        try:
            from scapy.all import IP, TCP, wrpcap
            pkts = []
            base_time = time.time()

            # Normal flows: 12 bidirectional TCP handshakes
            for i in range(12):
                sport = 49100 + i
                pkt_syn = IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="S")
                pkt_syn.time = base_time + i * 0.15
                pkts.append(pkt_syn)
                pkt_synack = IP(src="8.8.8.8", dst="192.168.1.10")/TCP(sport=80, dport=sport, flags="SA")
                pkt_synack.time = base_time + i * 0.15 + 0.02
                pkts.append(pkt_synack)
                pkt_ack = IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="A")
                pkt_ack.time = base_time + i * 0.15 + 0.04
                pkts.append(pkt_ack)

            # TCP SYN Scan attacker: 25 half-open probes
            for i in range(25):
                target_port = 20 + i
                pkt = IP(src="192.168.1.187", dst="192.168.1.5")/TCP(sport=38200 + i, dport=target_port, flags="S")
                pkt.time = base_time + 4.0 + i * 0.4
                pkts.append(pkt)

            out_path = os.path.join(self.project_root, "pcaps", "synthetic_scan.pcap")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            wrpcap(out_path, pkts)
        except Exception as e:
            st.error(f"Failed to generate synthetic attack capture: {e}")


    def run(self) -> None:
        navigation, active_model = self.draw_sidebar()
        
        st.markdown("""
        <div class="lab-header">
            <h2>🔬 OFFLINE AI-POWERED PCAP RECONNAISSANCE FORENSICS</h2>
            <div class="lab-subtitle">Post-capture Wireshark forensic investigation & anomalous behavioral intelligence laboratory</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. PCAP Upload Center
        if navigation == "PCAP Upload Center":
            st.markdown("<div class='cyber-panel'><h4>1. WIRESHARK PCAP FORENSIC LOAD PANEL</h4>", unsafe_allow_html=True)
            st.markdown("<p style='color:#94A3B8;'>Drag and drop any standard packet capture file (.pcap or .pcapng) below to execute feature preprocessing, scale standardizations, and multi-model machine learning scans.</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Drop your security captures (.pcap / .pcapng / .cap)", type=["pcap", "pcapng", "cap"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_file is not None:
                temp_pcap_path = os.path.join(self.project_root, "pcaps", "uploaded_temp.pcap")
                os.makedirs(os.path.dirname(temp_pcap_path), exist_ok=True)
                with open(temp_pcap_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.success("PCAP Capture File Loaded successfully! Switch to the 'Forensic Overview' or 'MITRE ATT&CK Mapping' tabs in the sidebar to review details.")
                st.session_state["loaded_pcap_path"] = temp_pcap_path
                st.session_state["loaded_filename"] = uploaded_file.name
            else:
                st.info("Drag and load a network capture to launch forensic analytics.")
                
        # Load and Cache PCAP analysis
        pcap_path = st.session_state.get("loaded_pcap_path")
        filename = st.session_state.get("loaded_filename", "No PCAP Loaded")
        
        if pcap_path and os.path.exists(pcap_path):
            with st.spinner("Decoding packets and extracting connection sessions..."):
                # Static load & cached analysis
                if "cached_reports" not in st.session_state or st.session_state.get("cached_pcap_path") != pcap_path or st.session_state.get("cached_model") != active_model:
                    try:
                        packets, metadata = PCAPParser.load_pcap(pcap_path)
                        flows = StaticFlowTracker.track_flows(packets)
                        
                        extractor = FeatureExtractor()
                        df_features = extractor.extract_features(flows)
                        
                        analyzer = ThreatAnalyzer(model_name=active_model)
                        reports = analyzer.analyze_flows(flows, df_features)
                        
                        st.session_state["cached_reports"] = reports
                        st.session_state["cached_metadata"] = metadata
                        st.session_state["cached_pcap_path"] = pcap_path
                        st.session_state["cached_model"] = active_model
                    except Exception as e:
                        st.error(f"Failed parsing connection sessions: {e}")
                        return
                        
            reports = st.session_state["cached_reports"]
            metadata = st.session_state["cached_metadata"]
            
            anomalies = [r for r in reports if r.ml_prediction == 1 or r.severity in ["CRITICAL", "HIGH", "MEDIUM"]]
            anom_count = len(anomalies)
            
            # Draw HUD HUD bar
            self.draw_hud_bar(filename, metadata.get("total_packets", 0), anom_count, metadata.get("duration_seconds", 0.0))
            
            if navigation == "Forensic Overview":
                # High-level statistics counts
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="cyber-panel">
                        <div class="hud-label">TOTAL PACKETS PARSED</div>
                        <div class="hud-value-large val-cyan">{metadata.get('total_packets', 0)}</div>
                        <div style="font-size:0.75rem; color:#64748B;">Reconstructed offline packets</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="cyber-panel">
                        <div class="hud-label">SESSIONS EXTRAS</div>
                        <div class="hud-value-large val-purple">{len(reports)}</div>
                        <div style="font-size:0.75rem; color:#64748B;">Bidirectional flow sessions</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="cyber-panel panel-critical">
                        <div class="hud-label">RECON ANOMALIES</div>
                        <div class="hud-value-large val-red">{anom_count}</div>
                        <div style="font-size:0.75rem; color:#EF4444;">Host scan events resolved</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    rate = (anom_count / len(reports)) * 100 if len(reports) > 0 else 0.0
                    st.markdown(f"""
                    <div class="cyber-panel">
                        <div class="hud-label">ANOMALOUS RATIO</div>
                        <div class="hud-value-large val-gold">{rate:.1f}%</div>
                        <div style="font-size:0.75rem; color:#64748B;">Proportion of scanning profiles</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
                
                # Visual charts
                g1, g2, g3 = st.columns([2, 2, 3])
                with g1:
                    st.markdown("<div class='cyber-panel'><h4 style='margin-bottom:15px; text-align:center;'>RECON VECTORS</h4>", unsafe_allow_html=True)
                    st.pyplot(self.generate_radar_sweep_plot(reports))
                    st.markdown("</div>", unsafe_allow_html=True)
                with g2:
                    st.markdown("<div class='cyber-panel'><h4>PROTOCOL RATIO BREAKDOWN</h4>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 4.3))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    p_data = metadata.get("protocols", {"TCP": 0, "UDP": 0, "Other": 0})
                    ax.pie(p_data.values(), labels=p_data.keys(), colors=["#EF4444", "#00F2FE", "#8B5CF6"], autopct='%1.1f%%', textprops={'color': '#F8FAFC', 'fontsize': 8}, wedgeprops={'edgecolor': 'none'})
                    ax.set_title("Decoded Protocol Volume", color='#F8FAFC', fontname='Orbitron', fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)
                with g3:
                    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                    st.pyplot(self.generate_anomaly_timeline_plot(reports))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Threat Feed Details
                st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
                mc1, mc2 = st.columns([3, 1])
                with mc1:
                    st.markdown("<h3>Detailed Forensic Threat Feed</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='forensic-alerts-feed'>", unsafe_allow_html=True)
                    if anomalies:
                        for a in anomalies:
                            sev = a.severity
                            card_class = "card-crit" if sev == "CRITICAL" else ("card-high" if sev == "HIGH" else "card-med")
                            tag_class = "tag-crit" if sev == "CRITICAL" else ("tag-high" if sev == "HIGH" else "tag-med")
                            
                            ev_list = "".join([f"<li style='margin-bottom:2px;'>{ev}</li>" for ev in a.evidence])
                            
                            st.markdown(f"""
                            <div class="alert-card {card_class}">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <div>
                                        <span class="tag-sec {tag_class}">{sev} RISK</span>
                                        <span style="font-family:'Orbitron'; font-size:0.95rem; font-weight:700; color:#F8FAFC; margin-left:12px;">
                                            {a.scan_category}
                                        </span>
                                    </div>
                                    <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748B;">{a.timestamp}</span>
                                </div>
                                <div style="font-family:'JetBrains Mono'; font-size:0.82rem; color:#E2E8F0; margin-bottom:10px; background:rgba(0,0,0,0.22); padding:8px 12px; border-radius:4px;">
                                    SOURCE IP: <span style="color:#00F2FE; font-weight:700;">{a.src_ip}</span> // 
                                    DESTINATION IP: <span style="color:#FBBF24; font-weight:700;">{a.dst_ip}:{a.dst_port}</span> // 
                                    PROTO: <span style="color:#A78BFA; font-weight:700;">{a.proto}</span>
                                </div>
                                <div style="font-size:0.8rem;">
                                    <span style="color:#EF4444; font-weight:700; font-family:'Orbitron';">FORENSIC TELEMETRY EVIDENCE:</span>
                                    <ul style="margin:4px 0 0 16px; padding:0; color:#94A3B8; list-style-type:square;">
                                        {ev_list}
                                    </ul>
                                </div>
                                <div style="margin-top:8px; text-align:right; font-family:'Orbitron'; font-size:0.72rem; color:#38BDF8; letter-spacing:0.05em;">
                                    SCORING METHOD: <span style="font-weight:700;">ML CLASSIFICATION [CONF: {a.ml_confidence*100:.1f}%] // COMPOSITE THREAT SCORE: {a.threat_score:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("Perimeter Secure! No suspicious stealth reconnaissance behaviors identified in this capture.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with mc2:
                    st.markdown("<div class='cyber-panel'><h4>SUSPICIOUS HOST TRACKING</h4>", unsafe_allow_html=True)
                    if anomalies:
                        df_anom = pd.DataFrame([a.to_dict() for a in anomalies])
                        st.markdown("📈 **Top Scan Vectors**")
                        st.dataframe(df_anom["scan_category"].value_counts(), use_container_width=True)
                        
                        st.markdown("🌐 **Top Suspect IPs**")
                        st.dataframe(df_anom["src_ip"].value_counts(), use_container_width=True)
                    else:
                        st.info("No host anomalies detected.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif navigation == "MITRE ATT&CK Mapping":
                st.markdown("<h3>Forensic MITRE ATT&CK Matrix Mapping</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#94A3B8; margin-bottom:20px;'>Reconstructed scan events mapped directly to standard enterprise MITRE ATT&CK tactics.</p>", unsafe_allow_html=True)
                
                # Dynamic compilation of MITRE counts
                mitre_counts = {}
                for a in anomalies:
                    for m in a.mitre_mappings:
                        mitre_counts[m] = mitre_counts.get(m, 0) + 1
                        
                st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                if mitre_counts:
                    st.markdown("<div class='mitre-grid'>", unsafe_allow_html=True)
                    for tech, count in mitre_counts.items():
                        tech_id = tech.split(" ")[0]
                        tech_name = tech.replace(tech_id, "").strip()
                        st.markdown(f"""
                        <div class="mitre-card">
                            <div class="mitre-id">{tech_id}</div>
                            <div class="mitre-name">{tech_name}</div>
                            <div style="font-size:0.75rem; color:#64748B; margin-top:8px;">Observed in <b>{count}</b> sessions</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No network reconnaissance sessions flagged to map to MITRE ATT&CK matrices.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Interactive export panel
                st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
                st.markdown("<h3>Forensic dossier summary center</h3>", unsafe_allow_html=True)
                
                ec1, ec2, ec3 = st.columns(3)
                
                report_md = ForensicReporter.generate_markdown_summary(reports, metadata, active_model)
                
                with ec1:
                    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                    st.markdown("📥 **FORENSIC DOSSIER**")
                    st.download_button(
                        label="Download Forensic dossier (TXT)",
                        data=report_md,
                        file_name="Forensic_Dossier_Report.txt",
                        mime="text/plain"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with ec2:
                    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                    st.markdown("📊 **FLOW FEATURES (CSV)**")
                    # Build CSV payload
                    csv_rows = []
                    for r in reports:
                        row = r.to_dict()
                        row["mitre_mappings"] = ", ".join(row["mitre_mappings"])
                        row["evidence"] = "; ".join(row["evidence"])
                        csv_rows.append(row)
                    df_csv = pd.DataFrame(csv_rows)
                    st.download_button(
                        label="Download Threat CSV Data",
                        data=df_csv.to_csv(index=False).encode('utf-8'),
                        file_name="Forensic_Threat_Metrics.csv",
                        mime="text/csv"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with ec3:
                    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                    st.markdown("🔌 **ALERTS LOG (JSON-L)**")
                    json_lines = "\n".join([json.dumps(r.to_dict()) for r in reports])
                    st.download_button(
                        label="Download JSON Alerts Log",
                        data=json_lines,
                        file_name="Forensic_Alerts.json",
                        mime="application/json"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif navigation == "ML Analysis Diagnostics":
                st.markdown("<h3>ML Multi-Classifier Forensic Diagnostics</h3>", unsafe_allow_html=True)
                
                metrics = self.load_metrics()
                if not metrics:
                    st.warning("No model evaluation metrics database found. Run validate.py first to train models.")
                else:
                    st.markdown("<div class='cyber-panel'><h4>MULTI-MODEL CONFUSION SCORING MATRIX</h4>", unsafe_allow_html=True)
                    metric_rows = []
                    for name, data in metrics.items():
                        metric_rows.append({
                            "Classifier Model": name.replace('_', ' ').upper(),
                            "Accuracy Score": f"{data.get('accuracy', 0.0)*100:.2f}%",
                            "Precision Index": f"{data.get('precision', 0.0)*100:.2f}%",
                            "Recall (Detection Rate)": f"{data.get('recall', 0.0)*100:.2f}%",
                            "F1-Score Profile": f"{data.get('f1_score', 0.0)*100:.2f}%",
                            "ROC-AUC Parameter": f"{data.get('auc', 0.0):.4f}" if "auc" in data else "N/A"
                        })
                    st.table(pd.DataFrame(metric_rows))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='cyber-panel'><h4>RECEIVER OPERATING CHARACTERISTIC (ROC) SHAPES</h4>", unsafe_allow_html=True)
                    roc_img_path = os.path.join(self.project_root, "results", "roc_curves.png")
                    if os.path.exists(roc_img_path):
                        st.markdown("<div class='cyber-image-frame'>", unsafe_allow_html=True)
                        st.image(roc_img_path, caption="Comparative ROC curves")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No ROC diagrams generated yet.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='cyber-panel'><h4>INFORMATION GAIN SIGNIFICANCE RATIOS</h4>", unsafe_allow_html=True)
                    fi_img_path = os.path.join(self.project_root, "results", "random_forest_feature_importance.png")
                    if os.path.exists(fi_img_path):
                        st.markdown("<div class='cyber-image-frame'>", unsafe_allow_html=True)
                        st.image(fi_img_path, caption="Ensemble information gain metrics")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No feature importance diagrams generated yet.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif navigation == "Settings & Utilities":
                st.markdown("<h3>Forensic Laboratory Utilities</h3>", unsafe_allow_html=True)
                
                st.markdown("<div class='cyber-panel'><h4>ACCUMULATION WINDOW ADJUSTMENT</h4>", unsafe_allow_html=True)
                st.slider("Flow temporal window (seconds)", 5, 120, 30)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='cyber-panel'><h4>SYNTHETIC PCAP GENERATOR</h4>", unsafe_allow_html=True)
                st.markdown("<p style='color:#94A3B8;'>Generate a sample stealth TCP scan capture file containing 12 normal TCP handshake sessions and 25 half-open stealth reconnaissance probes targetingports 20-45.</p>", unsafe_allow_html=True)
                if st.button("CREATE SYNTHETIC ATTACK PCAP"):
                    self.create_mock_pcap_file()
                    st.success("Successfully generated synthetic capture file inside 'pcaps/synthetic_scan.pcap'! You can now load this file in the PCAP Upload Center to visualize anomalies.")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("🔬 Please navigate to the 'PCAP Upload Center' in the sidebar to drop your Wireshark capture package and begin forensic analysis.")
            
        st.markdown("""
        <div style="text-align: center; padding: 25px; color: #64748B; font-family: 'Orbitron', sans-serif; font-size: 0.75rem; border-top: 1px dashed rgba(255,255,255,0.05); margin-top:40px;">
            👻 PhantomTrace Digital Forensics Command Deck // SECURED OPERATOR SYSTEM // CLASSIFIED INTEL
        </div>
        """, unsafe_allow_html=True)

def main():
    app = ForensicWorkstationApp()
    app.run()

if __name__ == "__main__":
    main()
