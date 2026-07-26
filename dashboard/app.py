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

# Imports for live capture and real-time analysis
import queue
import threading
from scapy.all import sniff, wrpcap, PcapReader
from flows.tracker import FlowTracker
from capture.sniffer import PacketSniffer
from threat_analysis.intel import ThreatIntelEngine

def run_simulation_thread(pcap_path, packet_queue, stop_event):
    """
    Reads from a static PCAP file and streams parsed packet records
    into the packet queue to emulate live capture timing.
    """
    try:
        if not os.path.exists(pcap_path):
            return
            
        last_pkt_time = None
        while not stop_event.is_set():
            with PcapReader(pcap_path) as reader:
                # Setup a fresh wall clock baseline for this simulation pass
                time_offset = time.time()
                first_pkt_time = None
                
                for pkt in reader:
                    if stop_event.is_set():
                        break
                        
                    parsed = PacketSniffer.parse_packet(pkt)
                    if parsed:
                        if first_pkt_time is None:
                            first_pkt_time = parsed.timestamp
                            
                        # Shift the epoch to current time to align with flow pruner and real-time HUD
                        relative_offset = parsed.timestamp - first_pkt_time
                        normalized_time = time_offset + relative_offset
                        
                        parsed.timestamp = normalized_time
                        pkt.time = normalized_time
                        
                        try:
                            packet_queue.put((parsed, pkt), block=True, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        # Replicate interval sleeps to emulate live flows
                        if last_pkt_time is not None:
                            delta = min(0.1, max(0.0, parsed.timestamp - last_pkt_time))
                            time.sleep(delta)
                        last_pkt_time = parsed.timestamp
            # Loop the simulation PCAP so the live capture dashboard runs indefinitely
            time.sleep(0.5)
    except Exception:
        pass

def run_live_sniffer_thread(interface, packet_queue, stop_event, error_list):
    """
    Sniffs packets live using Scapy sniff loop inside a background thread.
    Uses short timeouts to periodically check for exit triggers.
    """
    def packet_callback(pkt):
        parsed = PacketSniffer.parse_packet(pkt)
        if parsed:
            # Force packet timestamp to match current real-time
            parsed.timestamp = time.time()
            pkt.time = parsed.timestamp
            try:
                packet_queue.put((parsed, pkt), block=False)
            except queue.Full:
                pass
                
    sniff_args = {
        "prn": packet_callback,
        "filter": "ip and (tcp or udp)",
        "store": 0,
        "timeout": 1.0  # Periodic timeout to allow checking stop_event
    }
    if interface and interface != "any":
        sniff_args["iface"] = interface
        
    while not stop_event.is_set():
        try:
            sniff(**sniff_args)
        except PermissionError:
            error_list.append("Permission Denied: Live sniffing requires raw socket privileges. Please run with sudo ('sudo ./venv/bin/streamlit run app.py') or use the SIMULATED option.")
            break
        except Exception as e:
            error_list.append(f"Physical Sniffer Error: {str(e)}")
            break

def consumer_thread_loop(packet_queue, tracker, captured_packets_list, raw_packets_list, state_lock, stop_event):
    """
    Pulls packet records from the thread-safe queue, updates flow states in FlowTracker,
    and populates memory arrays for the live dashboard view.
    """
    while not stop_event.is_set() or not packet_queue.empty():
        try:
            pkt, raw_pkt = packet_queue.get(timeout=0.2)
            tracker.handle_packet(pkt)
            with state_lock:
                captured_packets_list.append(pkt)
                raw_packets_list.append(raw_pkt)
                
                # Prevent memory exhaustion by clipping arrays to the last 5,000 packets
                if len(captured_packets_list) > 5000:
                    captured_packets_list.pop(0)
                if len(raw_packets_list) > 5000:
                    raw_packets_list.pop(0)
        except queue.Empty:
            continue
        except Exception:
            pass

def detection_thread_loop(tracker, active_model, realtime_alerts, triggered_alerts, state_lock, stop_event):
    """
    Performs periodic feature extraction and multi-model machine learning inference 
    on active network flows every 5 seconds to raise security alert vectors.
    """
    from features.extractor import FeatureExtractor as RealtimeFeatureExtractor
    from threat_analysis.analyzer import ThreatAnalyzer as RealtimeThreatAnalyzer
    
    extractor = RealtimeFeatureExtractor()
    analyzer = RealtimeThreatAnalyzer(model_name=active_model)
    
    while not stop_event.is_set():
        # Dynamic check interval totaling 5 seconds
        for _ in range(50):
            if stop_event.is_set():
                break
            time.sleep(0.1)
            
        try:
            active_sessions = tracker.get_active_sessions()
            if not active_sessions:
                continue
                
            # Run same feature extraction preprocessing as offline
            df_features = extractor.extract_features(active_sessions)
            if df_features.empty:
                continue
                
            # Perform ML classification and scoring
            reports = analyzer.analyze_flows(active_sessions, df_features)
            
            with state_lock:
                for r in reports:
                    if r.ml_prediction == 1 or r.severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                        # Prevent duplicate incident rendering within a rolling window
                        alert_key = (r.src_ip, r.dst_ip, r.scan_category)
                        
                        if alert_key not in triggered_alerts:
                            triggered_alerts[alert_key] = time.time()
                            realtime_alerts.append(r)
                        else:
                            if time.time() - triggered_alerts[alert_key] > 60.0:
                                triggered_alerts[alert_key] = time.time()
                                realtime_alerts.append(r)
                                
                        if len(realtime_alerts) > 1000:
                            realtime_alerts.pop(0)
        except Exception:
            pass

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

    /* Live capture pulsing indicators */
    .pulse-live {
        width: 12px;
        height: 12px;
        background-color: #EF4444;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-animation 1.5s infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    @keyframes pulse-animation {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .pulse-idle {
        width: 12px;
        height: 12px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        vertical-align: middle;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
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
    .card-safe { border-left-color: #10B981; box-shadow: inset 4px 0 20px rgba(16, 185, 129, 0.04); }

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
    .tag-safe { color: #10B981; background: rgba(16, 185, 129, 0.08); }

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
        
        # Threat intelligence engine setup
        abuse_k = st.session_state.get("abuseipdb_api_key")
        vt_k = st.session_state.get("virustotal_api_key")
        shodan_k = st.session_state.get("shodan_api_key")
        self.intel_engine = ThreatIntelEngine(abuseipdb_key=abuse_k, virustotal_key=vt_k, shodan_key=shodan_k)

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
        import re
        for i, a in enumerate(anomalies[:40]):  # Limit to 40 polar elements
            # Extract numerical port safely
            port_match = re.search(r'\d+', str(a.dst_port))
            first_port = int(port_match.group()) if port_match else 0
            
            # Logarithmic distribution for ports (0-65535) -> (15-90) radius
            port_scaled = (np.log10(first_port + 1) / 4.8) * 75.0 if first_port > 0 else 0.0
            r = min(90.0, 15.0 + port_scaled)
            
            # Jitter angle based on time to prevent perfectly overlapping dots
            jitter = (hash(a.timestamp) % 20) - 10
            base_angle = hash(a.src_ip) % 360
            t = ((base_angle + jitter) % 360) * (np.pi / 180.0)
            
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

    def generate_anomaly_timeline_plot(self, reports: List[Any], figsize=(6, 3.5)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.15))
        ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#94A3B8', labelsize=8)
        ax.grid(True, color=(1.0, 1.0, 1.0, 0.04), linestyle='solid')
        
        if reports:
            df = pd.DataFrame([r.to_dict() for r in reports])
            df["time"] = pd.to_datetime(df["timestamp"])
            
            # Group by exact time and compute max threat score
            trend_df = df.groupby("time")["threat_score"].max().reset_index()
            trend_df = trend_df.sort_values("time")
            
            ax.plot(trend_df["time"], trend_df["threat_score"], color='#8B5CF6', linewidth=2.5, label='Max Threat Score')
            ax.fill_between(trend_df["time"], 0, trend_df["threat_score"], color=(0.545, 0.361, 0.965, 0.12))
            
            # Set Y-axis for accurate 0-100 severity scaling
            ax.set_ylim(0, 100)
            
            # Add severity horizontal threshold lines
            ax.axhline(y=91, color='#EF4444', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=71, color='#FF5E62', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=51, color='#FBBF24', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=26, color='#06B6D4', linestyle='--', alpha=0.3, linewidth=1)
            
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
                    "Live SOC Console",
                    "PCAP Upload Center", 
                    "Forensic Overview", 
                    "MITRE ATT&CK Mapping", 
                    "ML Analysis Diagnostics",
                    "Behavioral Intelligence",
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


    def get_linux_interfaces(self) -> List[str]:
        """
        Gathers a list of local active Linux network interfaces,
        falling back to filesystem scanners if Scapy module fails.
        """
        interfaces = ["SIMULATED TRAFFIC STREAM (Synthetic PCAP)"]
        try:
            from scapy.all import get_if_list
            interfaces.extend([i for i in get_if_list() if i != "any"])
        except Exception:
            if os.path.exists("/sys/class/net"):
                interfaces.extend(os.listdir("/sys/class/net"))
        return sorted(list(set(interfaces)))

    def draw_live_soc_console(self, active_model: str) -> None:
        """
        Renders the complete Real-Time SOC AI Intrusion Detection and Forensics deck.
        Tracks active packets, draws metrics, computes timelines/radar sweeps,
        performs reputation lookup, and exports PCAP summaries.
        """
        # Render any background thread capture errors
        if st.session_state.get("capture_errors"):
            st.error(f"⚠️ **SENSOR CAPTURE ERROR:** {st.session_state.capture_errors[-1]}")
            st.session_state.is_capturing = False
            st.session_state.stop_event.set()
            
        # 1. CONTROL PANEL PANEL
        st.markdown("<div class='cyber-panel'><h4>LIVE DECK SENSORS CONTROL PANEL</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            interfaces = self.get_linux_interfaces()
            selected_if = st.selectbox(
                "Sniffing Interface",
                interfaces,
                index=0,
                disabled=st.session_state.is_capturing,
                key="active_interface_select"
            )
        
        with col2:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            start_btn = st.button(
                "🚀 START CAPTURE",
                use_container_width=True,
                disabled=st.session_state.is_capturing
            )
            
        with col3:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            stop_btn = st.button(
                "🛑 STOP CAPTURE",
                use_container_width=True,
                disabled=not st.session_state.is_capturing
            )
            
        with col4:
            st.markdown("<div style='margin-top: 15px; text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<span style='font-size:0.65rem; color:#64748B; font-family:\"Orbitron\"; letter-spacing:0.1em; display:block;'>STATUS</span>", unsafe_allow_html=True)
            if st.session_state.is_capturing:
                st.markdown("<span class='pulse-live'></span><span style='color:#EF4444; font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:bold;'>ACTIVE SNIFFING</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pulse-idle'></span><span style='color:#10B981; font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:bold;'>IDLE STANDBY</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # START CAPTURE HANDLER
        if start_btn:
            # Recreate simulated scanner PCAP if missing
            if "SIMULATED" in selected_if:
                pcap_sim_path = os.path.join(self.project_root, "pcaps", "synthetic_scan.pcap")
                if not os.path.exists(pcap_sim_path):
                    self.create_mock_pcap_file()
                    
            st.session_state.captured_packets = []
            st.session_state.raw_packets = []
            st.session_state.realtime_alerts = []
            st.session_state.tracker = FlowTracker(flow_timeout_seconds=60)
            st.session_state.start_time = time.time()
            st.session_state.stop_event.clear()
            st.session_state.packet_queue = queue.Queue(maxsize=10000)
            st.session_state.triggered_alerts = {}
            st.session_state.capture_errors = []
            st.session_state.selected_interface = selected_if
            st.session_state.saved_pcap_path = None
            st.session_state.saved_pcap_name = None
            st.session_state.is_capturing = True
            
            # 1. Start Consumer Thread
            c_thread = threading.Thread(
                target=consumer_thread_loop,
                args=(
                    st.session_state.packet_queue,
                    st.session_state.tracker,
                    st.session_state.captured_packets,
                    st.session_state.raw_packets,
                    st.session_state.state_lock,
                    st.session_state.stop_event
                ),
                daemon=True
            )
            c_thread.start()
            
            # 2. Start ML Detection Thread
            d_thread = threading.Thread(
                target=detection_thread_loop,
                args=(
                    st.session_state.tracker,
                    active_model,
                    st.session_state.realtime_alerts,
                    st.session_state.triggered_alerts,
                    st.session_state.state_lock,
                    st.session_state.stop_event
                ),
                daemon=True
            )
            d_thread.start()
            
            # 3. Start Sniffer/Simulation Thread
            if "SIMULATED" in selected_if:
                pcap_sim_path = os.path.join(self.project_root, "pcaps", "synthetic_scan.pcap")
                s_thread = threading.Thread(
                    target=run_simulation_thread,
                    args=(pcap_sim_path, st.session_state.packet_queue, st.session_state.stop_event),
                    daemon=True
                )
            else:
                s_thread = threading.Thread(
                    target=run_live_sniffer_thread,
                    args=(selected_if, st.session_state.packet_queue, st.session_state.stop_event, st.session_state.capture_errors),
                    daemon=True
                )
            s_thread.start()
            
            st.rerun()

        # STOP CAPTURE HANDLER
        if stop_btn:
            st.session_state.stop_event.set()
            st.session_state.is_capturing = False
            st.session_state.capture_duration = time.time() - st.session_state.start_time
            
            # Write standard PCAP file from raw packet dump
            with st.session_state.state_lock:
                raw_dumps = list(st.session_state.raw_packets)
                
            if raw_dumps:
                pcap_name = f"live_capture_{int(time.time())}.pcap"
                pcap_path = os.path.join(self.project_root, "pcaps", pcap_name)
                os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
                try:
                    wrpcap(pcap_path, raw_dumps)
                    st.session_state.saved_pcap_path = pcap_path
                    st.session_state.saved_pcap_name = pcap_name
                except Exception as e:
                    st.error(f"Failed to compile PCAP file: {e}")
            st.rerun()

        # 2. DYNAMIC TELEMETRY HUD BAR
        with st.session_state.state_lock:
            pkts_count = len(st.session_state.captured_packets)
            alerts_copy = list(st.session_state.realtime_alerts)
            flows_count = len(st.session_state.tracker.active_sessions) if st.session_state.tracker else 0

        duration_sec = 0.0
        if st.session_state.is_capturing:
            duration_sec = time.time() - st.session_state.start_time
        else:
            duration_sec = st.session_state.get("capture_duration", 0.0)

        anom_flag_count = len(alerts_copy)
        risk_label = "CLEAN SECURE"
        risk_class = "cyan-glow"
        if any(r.severity == "CRITICAL" for r in alerts_copy):
            risk_label = "CRITICAL HAZARD"
            risk_class = "red-glow"
        elif any(r.severity in ["HIGH", "MEDIUM"] for r in alerts_copy):
            risk_label = "HIGH RISK WARNING"
            risk_class = "orange-glow"

        st.markdown(f"""
        <div class="forensic-hud-bar">
            <div class="hud-item"><span class="lbl">WORKSTATION:</span> <span class="val cyan-glow">PhantomTrace LIVE SOC</span></div>
            <div class="hud-item"><span class="lbl">INTERFACE:</span> <span class="val purple-glow">{st.session_state.selected_interface}</span></div>
            <div class="hud-item"><span class="lbl">PACKETS DETECTED:</span> <span class="val cyan-glow">{pkts_count} pkts</span></div>
            <div class="hud-item"><span class="lbl">ACTIVE TRACKS:</span> <span class="val cyan-glow">{flows_count} flows</span></div>
            <div class="hud-item"><span class="lbl">ELAPSED TIME:</span> <span class="val cyan-glow">{duration_sec:.1f}s</span></div>
            <div class="hud-item"><span class="lbl">AI IDS ALERTS:</span> <span class="val red-glow">{anom_flag_count} INCIDENTS</span></div>
            <div class="hud-item"><span class="lbl">THREAT INDEX:</span> <span class="val {risk_class}">{risk_label}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # 3. SOC ANALYTICS PLOTS GRID
        st.markdown("<h3 style='font-family:\"Orbitron\"; letter-spacing:0.05em; color:#00F2FE;'>📊 LIVE INTRUSION SECURITY REPORT</h3>", unsafe_allow_html=True)
        col_plot1, col_plot2, col_plot3 = st.columns(3)
        
        with col_plot1:
            st.markdown("<div class='cyber-panel'><h4>RECONNAISSANCE POLAR RADAR SCAN</h4>", unsafe_allow_html=True)
            fig_rad = self.generate_radar_sweep_plot(alerts_copy)
            st.pyplot(fig_rad)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_plot2:
            st.markdown("<div class='cyber-panel'><h4>CHRONOLOGICAL ANOMALY TIMELINE</h4>", unsafe_allow_html=True)
            fig_line = self.generate_anomaly_timeline_plot(alerts_copy)
            st.pyplot(fig_line)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_plot3:
            st.markdown("<div class='cyber-panel'><h4>ROLLING TRAFFIC DENSITY (PPS)</h4>", unsafe_allow_html=True)
            
            # Calculate rolling Packets Per Second (PPS) for the last 30 seconds
            import collections
            now = time.time()
            time_bins = collections.defaultdict(int)
            with st.session_state.state_lock:
                snapshot_packets = list(st.session_state.captured_packets)
                
            for p in snapshot_packets:
                age = now - p.timestamp
                if age <= 30:
                    bin_sec = int(p.timestamp)
                    time_bins[bin_sec] += 1
            
            # Ensure complete 30 second window without broken gaps
            now_sec = int(now)
            sec_keys = list(range(now_sec - 30, now_sec + 1))
            pps_vals = [time_bins.get(s, 0) for s in sec_keys]
            
            if len(pps_vals) < 5:
                pps_vals = [0] * 30
                labels = ["00:00:00"] * 30
            else:
                pps_vals = pps_vals[-30:]
                labels = [time.strftime("%H:%M:%S", time.localtime(s)) for s in sec_keys[-30:]]
                
            fig_pps, ax_pps = plt.subplots(figsize=(5, 3.8))
            fig_pps.patch.set_facecolor('none')
            ax_pps.set_facecolor('none')
            ax_pps.plot(range(len(pps_vals)), pps_vals, color='#00F2FE', linewidth=2, marker='o', markersize=3, label="PPS")
            ax_pps.fill_between(range(len(pps_vals)), pps_vals, color='#00F2FE', alpha=0.15)
            ax_pps.spines['bottom'].set_color((0.0, 0.949, 0.996, 0.25))
            ax_pps.spines['left'].set_color((0.0, 0.949, 0.996, 0.25))
            ax_pps.spines['top'].set_visible(False)
            ax_pps.spines['right'].set_visible(False)
            ax_pps.tick_params(colors='#64748B', labelsize=7)
            ax_pps.set_ylabel("Packets / Sec", color='#00F2FE', fontsize=8)
            ax_pps.grid(True, color=(1.0, 1.0, 1.0, 0.05), linestyle='dashed')
            plt.xticks(range(0, len(pps_vals), max(1, len(pps_vals)//5)), [labels[i] for i in range(0, len(labels), max(1, len(labels)//5))], rotation=30)
            plt.tight_layout()
            st.pyplot(fig_pps)
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. WIRESHARK-STYLE LIVE PACKET VIEWER
        st.markdown("<h3 style='font-family:\"Orbitron\"; letter-spacing:0.05em; color:#00F2FE;'>🔬 WIRESHARK LIVE PACKET CAPTURE VIEW</h3>", unsafe_allow_html=True)
        st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
        
        v_col1, v_col2 = st.columns([3, 1])
        with v_col1:
            search_query = st.text_input("🔍 Dynamic Header Search (Filter by IP, Port, or Flag)", value="", key="search_query")
        with v_col2:
            proto_filter = st.selectbox("Protocol Filter", ["All Protocols", "TCP", "UDP"], index=0, key="proto_filter")

        with st.session_state.state_lock:
            snap_packets = list(st.session_state.captured_packets)

        # Filter packets
        filtered_pkts = []
        search_q = search_query.strip().lower()
        for p in snap_packets:
            if proto_filter == "TCP" and p.proto != 6:
                continue
            if proto_filter == "UDP" and p.proto != 17:
                continue
            if search_q:
                match = (
                    search_q in p.src_ip.lower() or
                    search_q in p.dst_ip.lower() or
                    search_q in str(p.src_port) or
                    search_q in str(p.dst_port) or
                    search_q in p.flags.lower()
                )
                if not match:
                    continue
            filtered_pkts.append(p)

        # Display packet table (Newest first)
        table_rows = []
        display_limit = 100
        for idx, p in enumerate(reversed(filtered_pkts[-display_limit:])):
            t_str = time.strftime("%H:%M:%S", time.localtime(p.timestamp)) + f".{int((p.timestamp % 1) * 1000):03d}"
            proto_str = "TCP" if p.proto == 6 else ("UDP" if p.proto == 17 else "IP")
            sum_str = f"{proto_str} Flow: {p.src_ip}:{p.src_port} -> {p.dst_ip}:{p.dst_port}"
            if p.proto == 6:
                sum_str += f" [Flags: {p.flags}]"
                
            table_rows.append({
                "Index": len(filtered_pkts) - idx,
                "Time": t_str,
                "Source Address": p.src_ip,
                "Destination Address": p.dst_ip,
                "Proto": proto_str,
                "S-Port": p.src_port,
                "D-Port": p.dst_port,
                "Payload (B)": p.payload_len,
                "TCP Flags": p.flags if p.flags else "N/A",
                "Summary Info": sum_str
            })

        if table_rows:
            import pandas as pd
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            st.markdown(f"<p style='color:#64748B; font-size:0.75rem; text-align:right;'>Showing last {len(table_rows)} matches out of {len(filtered_pkts)} captured packets.</p>", unsafe_allow_html=True)
        else:
            st.info("Awaiting incoming packets from sensor bindings...")
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. LIVE INCIDENT ALERTS FEED & MITRE ATT&CK COUNTER
        st.markdown("<h3 style='font-family:\"Orbitron\"; letter-spacing:0.05em; color:#00F2FE;'>🚨 REAL-TIME AI INCIDENT FEED</h3>", unsafe_allow_html=True)
        feed_col1, feed_col2 = st.columns([2, 1])
        
        with feed_col1:
            st.markdown("<div class='cyber-panel'><h4 style='color:#EF4444;'>ACTIVE INTEL STREAM</h4>", unsafe_allow_html=True)
            if alerts_copy:
                st.markdown("<div class='forensic-alerts-feed'>", unsafe_allow_html=True)
                for alert in reversed(alerts_copy):
                    sev_class = "card-safe"
                    sev_badge = "tag-safe"
                    if alert.severity == "CRITICAL":
                        sev_class = "card-crit"
                        sev_badge = "tag-crit"
                    elif alert.severity == "HIGH":
                        sev_class = "card-high"
                        sev_badge = "tag-high"
                    elif alert.severity == "MEDIUM":
                        sev_class = "card-med"
                        sev_badge = "tag-med"
                    elif alert.severity == "LOW":
                        sev_class = "card-low"
                        sev_badge = "tag-low"
                        
                    t_str = time.strftime("%H:%M:%S", time.localtime(alert.start_epoch))
                    
                    st.markdown(f"""
                    <div class="alert-card {sev_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span class="tag-sec {sev_badge}">{alert.severity}</span>
                            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748B;">Observed at {t_str}</span>
                        </div>
                        <h5 style="color:#F8FAFC; margin:0 0 6px 0; font-family:'Orbitron';">{alert.scan_category}</h5>
                        <p style="margin:0 0 8px 0; font-size:0.85rem; color:#94A3B8;">
                            Source <b>{alert.src_ip}</b> triggered alert targeting <b>{alert.dst_ip}</b> (Ports: {alert.dst_port}) over {alert.proto}.
                            <br><span style="color:#00F2FE;">Aggregated Incident: {alert.flow_count} flows / {alert.packet_count} packets.</span>
                        </p>
                        <div style="margin-top:6px; padding:6px; background:rgba(0,0,0,0.25); border-radius:3px;">
                            <span style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#00F2FE;">MITRE MAPPINGS: {", ".join(alert.mitre_mappings)}</span>
                        </div>
                        <div style="margin-top:4px;">
                            <span style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#A78BFA;">AI Confidence: {alert.ml_confidence*100:.1f}% // Sub-scores: Flow={alert.flow_score:.1f}, Behavior={alert.behavior_score:.1f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Standby. Machine learning pipelines are listening for anomalous scanning connections...")
            st.markdown("</div>", unsafe_allow_html=True)

        with feed_col2:
            st.markdown("<div class='cyber-panel'><h4>MITRE ATT&CK TACTICAL COVERAGE</h4>", unsafe_allow_html=True)
            # Count tactical techniques observed
            mitre_counts = {}
            for a in alerts_copy:
                for mapping in a.mitre_mappings:
                    mitre_counts[mapping] = mitre_counts.get(mapping, 0) + 1

            if mitre_counts:
                st.markdown("<div class='mitre-grid'>", unsafe_allow_html=True)
                for tech, cnt in mitre_counts.items():
                    tech_id = tech.split(" ")[0]
                    tech_name = tech.replace(tech_id, "").strip()
                    st.markdown(f"""
                    <div class="mitre-card">
                        <div class="mitre-id">{tech_id}</div>
                        <div class="mitre-name" style="font-size:0.7rem;">{tech_name}</div>
                        <div style="font-family:'Orbitron'; font-size:0.9rem; font-weight:bold; color:#FF9900; margin-top:5px;">{cnt} HITS</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No adversarial reconnaissance tactics mapped yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        # 6. FORENSIC DRAWER / HOST REPUTATION LOOKUP
        st.markdown("<h3 style='font-family:\"Orbitron\"; letter-spacing:0.05em; color:#00F2FE;'>🔬 DYNAMIC SOURCE IP FORENSIC INSPECTOR</h3>", unsafe_allow_html=True)
        st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
        
        suspicious_ips = sorted(list(set(a.src_ip for a in alerts_copy)))
        if suspicious_ips:
            sel_suspect = st.selectbox("Select anomalous IP from capture logs to inspect", suspicious_ips, index=0)
            if sel_suspect:
                # Query Threat Intelligence Engine
                with st.spinner("Aggregating threat intelligence feeds..."):
                    intel = self.intel_engine.lookup_ip(sel_suspect)
                    
                sub1, sub2 = st.columns([1, 2])
                with sub1:
                    score = intel["abuse_score"]
                    color_h = "#10B981"
                    if score > 75:
                        color_h = "#EF4444"
                    elif score > 35:
                        color_h = "#FF9900"
                        
                    st.markdown(f"""
                    <div style="text-align:center; padding:15px; background:rgba(0,0,0,0.3); border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
                        <h4 style="color:#00F2FE; margin:0 0 10px 0; font-family:'Orbitron';">REPUTATION</h4>
                        <div style="font-size:3rem; font-family:'Orbitron'; font-weight:bold; color:{color_h};">{score}%</div>
                        <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748B;">Abuse Confidence Index</span>
                        <div style="margin-top:10px; font-weight:bold; color:{color_h}; font-size:0.8rem;">{intel["blacklist_status"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with sub2:
                    st.markdown(f"""
                    <table style="width:100%; border:none; color:#F8FAFC; font-size:0.85rem;">
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">ENTITY IP:</td><td style="font-family:'JetBrains Mono';">{intel["ip"]}</td></tr>
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">ISP PROVIDER:</td><td>{intel["isp"]}</td></tr>
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">GEO-LOCATION:</td><td>{intel["country"]}</td></tr>
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">VT POSITIVES:</td><td><b style="color:#EF4444;">{intel["vt_positives"]}</b> flags</td></tr>
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">SHODAN PORTS:</td><td style="color:#00F2FE; font-family:'JetBrains Mono';">{", ".join(map(str, intel["shodan_ports"])) if intel["shodan_ports"] else "No public active ports detected"}</td></tr>
                        <tr><td style="color:#64748B; font-family:'Orbitron'; padding:4px;">INTEL SCOPE:</td><td><i>{intel["enrichment_source"]}</i></td></tr>
                    </table>
                    <div style="margin-top:10px; padding:8px; background:rgba(0,0,255,0.08); border-radius:4px; font-size:0.8rem; border-left:3px solid #8B5CF6;">
                        {intel["details"]}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Awaiting alerts to select a suspicious host for dossier inspection.")
        st.markdown("</div>", unsafe_allow_html=True)

        # 7. CAPTURE MANAGEMENT EXPORT DECK
        if not st.session_state.is_capturing and st.session_state.saved_pcap_path:
            st.markdown("<h3 style='font-family:\"Orbitron\"; letter-spacing:0.05em; color:#00F2FE;'>📥 SESSION EXPORT DECK</h3>", unsafe_allow_html=True)
            st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
            st.success(f"Capture finalized and PCAP file serialized: <b>pcaps/{st.session_state.saved_pcap_name}</b>", icon="✅")
            
            ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
            
            # Load PCAP bytes
            pcap_bytes = b""
            if os.path.exists(st.session_state.saved_pcap_path):
                with open(st.session_state.saved_pcap_path, "rb") as f:
                    pcap_bytes = f.read()
                    
            with ex_col1:
                st.download_button(
                    "💾 DOWNLOAD PCAP CAPTURE",
                    data=pcap_bytes,
                    file_name=st.session_state.saved_pcap_name,
                    mime="application/vnd.tcpdump.pcap",
                    use_container_width=True
                )
                
            # Build CSV features bytes
            csv_bytes = b""
            live_flows = st.session_state.tracker.get_active_sessions() if st.session_state.tracker else []
            if live_flows:
                from feature_extraction.extractor import FeatureExtractor as LiveExtractor
                ext = LiveExtractor()
                df_f = ext.extract_features(live_flows)
                csv_bytes = df_f.to_csv(index=False).encode('utf-8')
                
            with ex_col2:
                st.download_button(
                    "📊 DOWNLOAD CSV FEATURES",
                    data=csv_bytes,
                    file_name=f"features_capture_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=len(csv_bytes) == 0
                )
                
            # Build JSON alerts bytes
            alert_dicts = [a.to_dict() for a in alerts_copy]
            json_bytes = json.dumps(alert_dicts, indent=2).encode('utf-8')
            
            with ex_col3:
                st.download_button(
                    "🚨 DOWNLOAD JSON ALERTS",
                    data=json_bytes,
                    file_name=f"alerts_capture_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                    disabled=len(json_bytes) == 0
                )
                
            # Build Markdown report bytes
            metadata_dict = {
                "file_name": st.session_state.saved_pcap_name,
                "file_size_mb": os.path.getsize(st.session_state.saved_pcap_path) / (1024 * 1024) if os.path.exists(st.session_state.saved_pcap_path) else 0.0,
                "total_packets": len(snapshot_packets),
                "duration_seconds": duration_sec,
                "unique_ips_count": len(set(p.src_ip for p in snapshot_packets)),
                "protocols": {
                    "TCP": sum(1 for p in snapshot_packets if p.proto == 6),
                    "UDP": sum(1 for p in snapshot_packets if p.proto == 17),
                    "Other": sum(1 for p in snapshot_packets if p.proto not in [6, 17])
                },
                "start_time_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(st.session_state.start_time)),
                "end_time_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }
            
            report_md = ForensicReporter.generate_markdown_summary(alerts_copy, metadata_dict, active_model)
            report_bytes = report_md.encode('utf-8')
            
            with ex_col4:
                st.download_button(
                    "📄 DOWNLOAD FORENSIC REPORT",
                    data=report_bytes,
                    file_name=f"report_capture_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    disabled=len(report_bytes) == 0
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # 8. AUTO-REFRESH RERUNNER
        if st.session_state.is_capturing:
            time.sleep(1.0)
            st.rerun()

    def run(self) -> None:
        """
        Unified Forensic Workstation orchestrator.
        Manages state initializations and branches routing to targeted deck pages.
        """
        # Ensure thread-safe session states are active
        if "captured_packets" not in st.session_state:
            st.session_state.captured_packets = []
        if "raw_packets" not in st.session_state:
            st.session_state.raw_packets = []
        if "realtime_alerts" not in st.session_state:
            st.session_state.realtime_alerts = []
        if "tracker" not in st.session_state:
            st.session_state.tracker = None
        if "is_capturing" not in st.session_state:
            st.session_state.is_capturing = False
        if "stop_event" not in st.session_state:
            st.session_state.stop_event = threading.Event()
        if "state_lock" not in st.session_state:
            st.session_state.state_lock = threading.Lock()
        if "packet_queue" not in st.session_state:
            st.session_state.packet_queue = None
        if "start_time" not in st.session_state:
            st.session_state.start_time = 0.0
        if "capture_duration" not in st.session_state:
            st.session_state.capture_duration = 0.0
        if "selected_interface" not in st.session_state:
            st.session_state.selected_interface = "any"
        if "saved_pcap_path" not in st.session_state:
            st.session_state.saved_pcap_path = None
        if "saved_pcap_name" not in st.session_state:
            st.session_state.saved_pcap_name = None
        if "triggered_alerts" not in st.session_state:
            st.session_state.triggered_alerts = {}
        if "capture_errors" not in st.session_state:
            st.session_state.capture_errors = []

        navigation, active_model = self.draw_sidebar()

        # ROUTER DISPATCHING
        if navigation == "Live SOC Console":
            st.markdown("""
            <div class="lab-header">
                <h2>🔬 LIVE AI-POWERED SOC MONITORING DECK</h2>
                <div class="lab-subtitle">Real-Time Intrusion Detection System, active sniffer & behavioral network threat forensics console</div>
            </div>
            """, unsafe_allow_html=True)
            self.draw_live_soc_console(active_model)
            
        else:
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

            if navigation != "PCAP Upload Center" and (not pcap_path or not os.path.exists(pcap_path)):
                st.info("🔬 Please navigate to the 'PCAP Upload Center' in the sidebar to drop your Wireshark capture package and begin forensic analysis.")
                
            elif pcap_path and os.path.exists(pcap_path):
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
                        
                            st.session_state["cached_packets"] = packets
                            st.session_state["cached_reports"] = reports
                            st.session_state["cached_metadata"] = metadata
                            st.session_state["cached_pcap_path"] = pcap_path
                            st.session_state["cached_model"] = active_model
                        except Exception as e:
                            st.error(f"Failed parsing connection sessions: {e}")
                            return
                        
                reports = st.session_state["cached_reports"]
                metadata = st.session_state["cached_metadata"]
            
                anomalies = [r for r in reports if r.ml_prediction == 1 or r.severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]]
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
                    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
                    st.pyplot(self.generate_anomaly_timeline_plot(reports, figsize=(12, 4)))
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
                                card_class = "card-crit" if sev == "CRITICAL" else ("card-high" if sev == "HIGH" else ("card-med" if sev == "MEDIUM" else ("card-low" if sev == "LOW" else "card-safe")))
                                tag_class = "tag-crit" if sev == "CRITICAL" else ("tag-high" if sev == "HIGH" else ("tag-med" if sev == "MEDIUM" else ("tag-low" if sev == "LOW" else "tag-safe")))
                            
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
                                        DESTINATION IP: <span style="color:#FBBF24; font-weight:700;">{a.dst_ip}</span> // 
                                        TARGET PORTS: <span style="color:#F8FAFC;">{a.dst_port}</span> // 
                                        PROTO: <span style="color:#A78BFA; font-weight:700;">{a.proto}</span>
                                        <br><span style="color:#10B981;">INCIDENT VOLUME: {a.flow_count} flows / {a.packet_count} pkts</span>
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
                    
                elif navigation == "Behavioral Intelligence":
                    st.markdown("### 🧠 BEHAVIORAL INTELLIGENCE & DRIFT DETECTION")
                    
                    if "cached_packets" not in st.session_state:
                        st.error("Raw packets not found in cache. Please re-upload the PCAP.")
                    else:
                        with st.spinner("Compiling behavioral profile timelines..."):
                            from threat_analysis.behavioral_profile_engine import BehavioralProfileEngine
                            engine = BehavioralProfileEngine()
                            
                            packets = st.session_state["cached_packets"]
                            timeline_records = []
                            last_timeline_save = 0.0
                            
                            # Process chronologically
                            for pkt in sorted(packets, key=lambda p: p.timestamp):
                                engine.update_profile(
                                    src_ip=pkt.src_ip,
                                    dst_ip=pkt.dst_ip,
                                    dst_port=pkt.dst_port,
                                    protocol=str(pkt.proto),
                                    bytes_len=pkt.payload_len,
                                    timestamp=pkt.timestamp,
                                    session_id=str((pkt.src_ip, pkt.dst_ip, pkt.dst_port, pkt.proto))
                                )
                                if (pkt.timestamp - last_timeline_save) > 5.0:
                                    last_timeline_save = pkt.timestamp
                                    for host_ip, profile in engine._profiles.items():
                                        ds = profile.calculate_drift(pkt.timestamp)
                                        if ds > 0:
                                            timeline_records.append({"time": pkt.timestamp, "host": host_ip, "drift": ds})
                        
                        # Get all profiles
                        all_profiles = engine.retrieve_all_profiles()
                        ranked_hosts = sorted(all_profiles.values(), key=lambda p: p.get("drift_score", 0.0), reverse=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>TOP ABNORMAL HOSTS</h4>", unsafe_allow_html=True)
                        if ranked_hosts:
                            top_host = ranked_hosts[0]
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                st.metric("Top Abnormal Host", top_host["src_ip"], f"Drift: {top_host.get('drift_score', 0.0):.1f}/100", delta_color="inverse")
                            with c2:
                                df_top = pd.DataFrame([{
                                    "Host IP": p["src_ip"], 
                                    "Drift Score": round(p.get("drift_score", 0.0), 2),
                                    "Sessions": p["session_count"],
                                    "Dest Diversity": p["destination_ip_count"]
                                } for p in ranked_hosts[:5]])
                                st.dataframe(df_top, use_container_width=True, hide_index=True)
                        else:
                            st.info("No host profiles generated.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>BEHAVIORAL DRIFT TIMELINE</h4>", unsafe_allow_html=True)
                        if timeline_records:
                            df_tl = pd.DataFrame(timeline_records)
                            min_time = df_tl["time"].min()
                            df_tl["time"] = df_tl["time"] - min_time
                            
                            tl_pivot = df_tl.pivot(index="time", columns="host", values="drift").ffill().fillna(0)
                            
                            fig, ax = plt.subplots(figsize=(10, 3.5))
                            fig.patch.set_facecolor('none')
                            ax.set_facecolor('none')
                            ax.tick_params(colors='#94A3B8')
                            ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.15))
                            ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.set_ylabel("Drift Score (0-100)", color="#94A3B8")
                            ax.set_xlabel("Seconds from Start", color="#94A3B8")
                            
                            # Limit to top 5 hosts
                            top_hosts_list = [p["src_ip"] for p in ranked_hosts[:5]]
                            for col in tl_pivot.columns:
                                if col in top_hosts_list:
                                    ax.plot(tl_pivot.index, tl_pivot[col], label=col, linewidth=2)
                            ax.legend(facecolor='#060913', edgecolor=(1.0, 1.0, 1.0, 0.1), labelcolor='#E2E8F0', fontsize=8, loc='upper left')
                            st.pyplot(fig)
                        else:
                            st.info("Insufficient timeline data. PCAP duration must exceed 5 periods of 60 seconds (total > 300s) to establish baseline.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>CURRENT VS HISTORICAL BEHAVIOR</h4>", unsafe_allow_html=True)
                        if ranked_hosts:
                            selected_host = st.selectbox("Select Host to Inspect", [p["src_ip"] for p in ranked_hosts])
                            if selected_host:
                                host_data = next((p for p in ranked_hosts if p["src_ip"] == selected_host), None)
                                if host_data:
                                    w5m = host_data.get("windows", {}).get("5m", {})
                                    base = host_data.get("baselines", {})
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Destination Diversity", w5m.get("destination_ip_count", 0), f"Baseline: {base.get('destination_diversity', 0.0):.1f}", delta_color="inverse")
                                    c2.metric("Port Diversity", w5m.get("destination_port_count", 0), f"Baseline: {base.get('port_diversity', 0.0):.1f}", delta_color="inverse")
                                    c3.metric("Pkt Rate (pkts/sec)", f"{w5m.get('packet_count', 0)/300.0:.2f}", f"Baseline: {base.get('packet_rate', 0.0):.2f}", delta_color="inverse")
                                    c4.metric("Mean Session Duration", f"{w5m.get('mean_session_duration', 0.0):.2f}s", f"Baseline: {base.get('session_duration', 0.0):.2f}s", delta_color="inverse")
                        st.markdown("</div>", unsafe_allow_html=True)

                elif navigation == "Settings & Utilities":
                    st.markdown("<h3>Forensic Laboratory Utilities</h3>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='cyber-panel'><h4>CYBER THREAT INTELLIGENCE API CREDENTIALS</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#94A3B8;'>Provide external API credentials to enrich IP addresses and scanning hosts with real-time reputation analysis. Leaving them blank activates local mock databases for safety and simulation.</p>", unsafe_allow_html=True)
                    
                    a_key = st.text_input("AbuseIPDB API Secret Key", value=st.session_state.get("abuseipdb_api_key", ""), type="password", help="Enables active scanning reputation lookups.")
                    v_key = st.text_input("VirusTotal Public API Key", value=st.session_state.get("virustotal_api_key", ""), type="password", help="Provides malware scan positive detections context.")
                    s_key = st.text_input("Shodan Search Engine API Key", value=st.session_state.get("shodan_api_key", ""), type="password", help="Resolves open public ports and hosts profiling.")
                    
                    if st.button("💾 SAVE API CREDENTIALS"):
                        st.session_state["abuseipdb_api_key"] = a_key
                        st.session_state["virustotal_api_key"] = v_key
                        st.session_state["shodan_api_key"] = s_key
                        
                        # Re-instantiate the threat intel engine dynamically
                        self.intel_engine = ThreatIntelEngine(abuseipdb_key=a_key, virustotal_key=v_key, shodan_key=s_key)
                        st.success("Cyber Threat Intelligence API Credentials updated and securely bound successfully!")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='cyber-panel'><h4>ACCUMULATION WINDOW ADJUSTMENT</h4>", unsafe_allow_html=True)
                    st.slider("Flow temporal window (seconds)", 5, 120, 30)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='cyber-panel'><h4>SYNTHETIC PCAP GENERATOR</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#94A3B8;'>Generate a sample stealth TCP scan capture file containing 12 normal TCP handshake sessions and 25 half-open stealth reconnaissance probes targeting ports 20-45.</p>", unsafe_allow_html=True)
                    if st.button("CREATE SYNTHETIC ATTACK PCAP"):
                        self.create_mock_pcap_file()
                        st.success("Successfully generated synthetic capture file inside 'pcaps/synthetic_scan.pcap'! You can now load this file in the PCAP Upload Center to visualize anomalies.")
                    st.markdown("</div>", unsafe_allow_html=True)

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
