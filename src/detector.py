#!/usr/bin/env python3
# ==============================================================================
# Phase 8 - Real-Time Stream Detection Engine
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import os
import sys
import json
import time
import queue
import logging
import argparse
import threading
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scapy.all import Packet
from src.capture import PacketCapturer
from src.parser import PCAPParser, PacketRecord, FlowRecord
from src.features import FeatureExtractor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("RealTimeDetector")


class RealTimeDetector:
    """
    Multithreaded, real-time stream packet processor, flow tracker,
    and online machine learning prediction engine.
    """
    def __init__(self, 
                 model_name: str = "random_forest",
                 interface: str = "any",
                 pcap_simulation: Optional[str] = None,
                 config_path: str = "/home/yi/Stealth System/configs/config.json"):
        self.config = self._load_config(config_path)
        self.project_root = self.config.get("project_root", "/home/yi/Stealth System")
        self.models_dir = self.config.get("directories", {}).get("models", f"{self.project_root}/models")
        self.alert_log_file = self.config.get("alerts", {}).get("alert_log_file", f"{self.project_root}/logs/alerts.log")
        
        self.interface = interface
        self.model_name = model_name
        self.pcap_simulation = pcap_simulation
        self.running = False
        
        # Load pipeline artifacts
        self.scaler = self._load_scaler()
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        
        # Thread safe queue for incoming parsed packets
        self.packet_queue: queue.Queue[PacketRecord] = queue.Queue(maxsize=5000)
        
        # In-memory sliding window of flows
        self.active_flows: Dict[Tuple[str, str, int, int, int], FlowRecord] = {}
        
        # Track locks for multi-threaded access
        self.flow_lock = threading.Lock()
        
        # Timers and limits from config
        self.sliding_window = self.config.get("features", {}).get("sliding_window_seconds", 30)
        self.flow_timeout = self.config.get("features", {}).get("flow_timeout_seconds", 60)
        self.min_packets = self.config.get("features", {}).get("min_packets_per_flow", 3)
        
        # Ensure log directories exist
        os.makedirs(os.path.dirname(self.alert_log_file), exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        return {}

    def _load_scaler(self) -> Optional[Any]:
        scaler_path = f"{self.models_dir}/scaler.joblib"
        try:
            if os.path.exists(scaler_path):
                logger.info(f"Loaded Standard Scaler from {scaler_path}")
                return joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Failed to load Standard Scaler: {e}")
        logger.warning("No standard scaler loaded. Live inference will operate unscaled!")
        return None

    def _load_model(self) -> Optional[Any]:
        model_path = f"{self.models_dir}/{self.model_name}_model.joblib"
        try:
            if os.path.exists(model_path):
                logger.info(f"Loaded ML model: {self.model_name} from {model_path}")
                return joblib.load(model_path)
        except Exception as e:
            logger.critical(f"Failed to load model {self.model_name}: {e}")
        logger.warning("No machine learning model loaded! Real-time engine will run in heuristic mode only.")
        return None

    def _load_feature_names(self) -> List[str]:
        names_path = f"{self.models_dir}/feature_names.json"
        try:
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature list: {e}")
        return []

    def packet_producer_callback(self, scapy_pkt: Packet) -> None:
        """
        Scapy sniffer callback (Producer Thread).
        Lightweight parsing and push to thread-safe queue.
        """
        parsed = PCAPParser.parse_packet(scapy_pkt)
        if parsed:
            try:
                self.packet_queue.put(parsed, block=False)
            except queue.Full:
                logger.debug("Packet Queue Full! Dropping packet.")

    def run_sniffer_thread(self, capture_timeout: int) -> None:
        """
        Sniff wrapper designed to run in a dedicated thread.
        Supports live sniffing or PCAP simulation streaming.
        """
        if self.pcap_simulation:
            logger.info(f"PCAP Simulation Mode: Streaming parsed packets from {self.pcap_simulation}...")
            parser = PCAPParser()
            try:
                last_pkt_time = None
                for pkt in parser.parse_pcap_generator(self.pcap_simulation):
                    if not self.running:
                        break
                        
                    # Queue the parsed packet record directly
                    try:
                        self.packet_queue.put(pkt, block=True, timeout=1.0)
                    except queue.Full:
                        pass
                        
                    # Introduce dynamic sleep to emulate live capture timing
                    if last_pkt_time is not None:
                        delta = min(0.15, max(0.0, pkt.timestamp - last_pkt_time))
                        time.sleep(delta)
                    last_pkt_time = pkt.timestamp
                    
                logger.info("PCAP Simulation Streaming complete.")
            except Exception as e:
                logger.error(f"Error in simulation sniffer thread: {e}")
            finally:
                self.running = False
        else:
            capturer = PacketCapturer(interface=self.interface)
            try:
                capturer.start_sniffing(
                    timeout=capture_timeout,
                    filter_str="ip and (tcp or udp)",
                    external_callback=self.packet_producer_callback
                )
            except Exception as e:
                logger.critical(f"Packet Capturer thread crashed: {e}")
            finally:
                self.running = False

    def trigger_alert(self, flow: FlowRecord, host_profile: Dict[str, float], confidence: float, source: str) -> None:
        """
        Writes a standardized threat alert log in JSON and prints to stdout.
        """
        alert = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(flow.end_time)),
            "alert_id": f"IDS-ALERT-{int(time.time() * 1000)}",
            "threat_category": "Stealth Reconnaissance Scan",
            "detection_method": f"ML Inference ({source})" if "ML" in source else "Heuristic Signature Match",
            "source_ip": flow.src_ip,
            "destination_ip": flow.dst_ip,
            "target_port": flow.dst_port,
            "protocol": "TCP" if flow.proto == 6 else "UDP" if flow.proto == 17 else str(flow.proto),
            "flow_statistics": {
                "packets": len(flow.packets),
                "duration_seconds": round(flow.duration, 4),
                "syn_ratio": round(flow.syn_count / len(flow.packets), 2),
                "rst_ratio": round(flow.rst_count / len(flow.packets), 2)
            },
            "host_context": {
                "port_entropy": round(host_profile.get("host_port_entropy", 0.0), 3),
                "destination_diversity": int(host_profile.get("host_dst_diversity", 0.0)),
                "failed_flow_ratio": round(host_profile.get("host_failed_flow_ratio", 0.0), 2)
            },
            "threat_confidence": round(confidence, 4)
        }
        
        # Append alert to JSON lines logs file
        try:
            with open(self.alert_log_file, 'a') as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert log: {e}")
            
        logger.warning(
            f"\n🚨 [THREAT DETECTED] {alert['threat_category']}!"
            f"\n   Source: {alert['source_ip']} -> Target: {alert['destination_ip']}:{alert['target_port']} "
            f"({alert['protocol']})"
            f"\n   Confidence: {alert['threat_confidence']*100:.2f}% | Method: {alert['detection_method']}"
            f"\n   Host Port Entropy: {alert['host_context']['port_entropy']} (High Randomness!)"
            f"\n   Target IPs Contacted: {alert['host_context']['destination_diversity']}\n"
        )

    def analyze_active_flows(self) -> None:
        """
        Performs feature extraction and ML prediction on active flows.
        """
        with self.flow_lock:
            if not self.active_flows:
                return
                
            flows_list = list(self.active_flows.values())
            
            # Prune/Clean old idle flows to avoid memory issues
            current_time = time.time()
            self.active_flows = {
                k: f for k, f in self.active_flows.items()
                if (current_time - f.end_time) < self.flow_timeout
            }
            
        # 1. Compute Host behavioral profile context
        extractor = FeatureExtractor()
        host_profiles = extractor.compute_host_features(flows_list)
        
        # 2. Extract features for active flows exceeding packet minimums
        for flow in flows_list:
            if len(flow.packets) < self.min_packets:
                continue  # Skip premature flows to reduce false positives
                
            # Extract features for this flow
            timestamps = [pkt.timestamp for pkt in flow.packets]
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
            
            payload_lens = [pkt.payload_len for pkt in flow.packets]
            
            # Compute raw metrics
            flow_duration = flow.duration
            total_pkts = len(flow.packets)
            flow_bytes = flow.forward_bytes + flow.backward_bytes
            syn_ratio = flow.syn_count / total_pkts
            ack_ratio = flow.ack_count / total_pkts
            rst_ratio = flow.rst_count / total_pkts
            fin_ratio = flow.fin_count / total_pkts
            
            flow_interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
            flow_interval_var = float(np.var(intervals)) if len(intervals) > 0 else 0.0
            flow_size_mean = float(np.mean(payload_lens)) if payload_lens else 0.0
            flow_size_var = float(np.var(payload_lens)) if payload_lens else 0.0
            
            # Attributed initiator host profile
            initiator_ip = flow.packets[0].src_ip
            host_feat = host_profiles.get(initiator_ip, {
                "host_port_entropy": 0.0, "host_dst_entropy": 0.0, "host_dst_diversity": 1.0,
                "host_syn_ratio": 0.0, "host_failed_flow_ratio": 0.0, "host_packet_rate": 0.0,
                "host_interval_mean": 0.0, "host_interval_var": 0.0, "host_packet_size_var": 0.0
            })
            
            # Build flow feature vector dictionary
            raw_features = {
                "flow_duration": flow_duration,
                "flow_packet_count": total_pkts,
                "flow_bytes": flow_bytes,
                "flow_syn_ratio": syn_ratio,
                "flow_ack_ratio": ack_ratio,
                "flow_rst_ratio": rst_ratio,
                "flow_fin_ratio": fin_ratio,
                "flow_size_mean": flow_size_mean,
                "flow_size_var": flow_size_var,
                "flow_interval_mean": flow_interval_mean,
                "flow_interval_var": flow_interval_var,
                
                "host_port_entropy": host_feat["host_port_entropy"],
                "host_dst_entropy": host_feat["host_dst_entropy"],
                "host_dst_diversity": host_feat["host_dst_diversity"],
                "host_syn_ratio": host_feat["host_syn_ratio"],
                "host_failed_flow_ratio": host_feat["host_failed_flow_ratio"],
                "host_packet_rate": host_feat["host_packet_rate"],
                "host_interval_mean": host_feat["host_interval_mean"],
                "host_interval_var": host_feat["host_interval_var"],
                "host_packet_size_var": host_feat["host_packet_size_var"]
            }
            
            # Predict using ML model if loaded
            if self.model and self.scaler and self.feature_names:
                try:
                    # Construct matching column DataFrame
                    df_inf = pd.DataFrame([raw_features])[self.feature_names]
                    df_scaled = pd.DataFrame(self.scaler.transform(df_inf), columns=self.feature_names)
                    
                    if self.model_name == "isolation_forest":
                        raw_pred = self.model.predict(df_scaled)[0]
                        prediction = 1 if raw_pred == -1 else 0
                        confidence = -self.model.decision_function(df_scaled)[0]
                    else:
                        prediction = self.model.predict(df_scaled)[0]
                        confidence = self.model.predict_proba(df_scaled)[0][1]
                        
                    if prediction == 1:
                        self.trigger_alert(flow, host_feat, confidence, self.model_name)
                        
                except Exception as e:
                    logger.error(f"Inference error on flow {flow.flow_key}: {e}")
            else:
                # HEURISTIC ALERT BACKUP
                # If no model is trained yet, fallback to heuristic rules to flag active scans
                if (host_feat["host_port_entropy"] > 2.2 and host_feat["host_failed_flow_ratio"] > 0.8) or \
                   (host_feat["host_dst_diversity"] > 4 and host_feat["host_syn_ratio"] > 0.8):
                    
                    self.trigger_alert(flow, host_feat, 0.99, "Fallback Expert Rules")

    def run_analyzer_thread(self) -> None:
        """
        Dequeues packets, groups them, and runs evaluation periodically.
        """
        logger.info("Analyzer Engine Thread Started.")
        last_eval_time = time.time()
        
        while self.running or not self.packet_queue.empty():
            try:
                # Retrieve parsed packet from queue
                pkt = self.packet_queue.get(timeout=1.0)
            except queue.Empty:
                # Perform periodic evaluations even if no packets are arriving
                if (time.time() - last_eval_time) > 3.0:
                    self.analyze_active_flows()
                    last_eval_time = time.time()
                continue
                
            # Aggregate into bidirectional flow
            key = PCAPParser.get_canonical_key(pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.proto)
            
            with self.flow_lock:
                if key not in self.active_flows:
                    self.active_flows[key] = FlowRecord(key)
                self.active_flows[key].add_packet(pkt)
                
            # Perform periodic ML assessment every 3 seconds
            if (time.time() - last_eval_time) > 3.0:
                self.analyze_active_flows()
                last_eval_time = time.time()
                
        logger.info("Analyzer Engine Thread Stopped.")

    def start(self, duration: int = 60) -> None:
        """
        Starts the multithreaded detector execution.
        """
        logger.info("==========================================================")
        logger.info("Initializing Real-Time AI Detection Engine...")
        logger.info(f"Target Interface: {self.interface} | Model: {self.model_name}")
        logger.info("==========================================================")
        
        self.running = True
        
        # Start Sniffer thread (Producer)
        sniffer_thread = threading.Thread(target=self.run_sniffer_thread, args=(duration,))
        # Start Analyzer thread (Consumer)
        analyzer_thread = threading.Thread(target=self.run_analyzer_thread)
        
        sniffer_thread.start()
        analyzer_thread.start()
        
        logger.info("Real-Time Engine is active. Sniffing and evaluating...")
        
        try:
            sniffer_thread.join()
            analyzer_thread.join()
        except KeyboardInterrupt:
            logger.info("Termination signal received. Stopping engine gracefully...")
            self.running = False
            sniffer_thread.join(timeout=2.0)
            analyzer_thread.join(timeout=2.0)
            
        logger.info("Real-Time AI Detection Engine closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stealth IDS - Real-time Detection Engine CLI")
    parser.add_argument("-i", "--interface", default="any", help="Network interface to sniff (default: any)")
    parser.add_argument("-m", "--model", default="random_forest", choices=["random_forest", "xgboost", "svm", "isolation_forest"], help="ML Model to load (default: random_forest)")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Run duration in seconds (default: 60)")
    parser.add_argument("-s", "--sim-pcap", default=None, help="Simulate real-time by streaming from PCAP file")
    
    args = parser.parse_args()
    
    # Correct argparse attribute conversion for hyphenated args
    pcap_sim = getattr(args, 'sim_pcap', None)
    detector = RealTimeDetector(model_name=args.model, interface=args.interface, pcap_simulation=pcap_sim)
    detector.start(duration=args.duration)
