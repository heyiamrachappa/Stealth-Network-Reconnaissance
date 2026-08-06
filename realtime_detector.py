#!/usr/bin/env python3
import os
import sys
import time
import queue
import argparse
import threading
from typing import Optional

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logger, load_config
from capture.sniffer import PacketSniffer, PacketRecord
from flows.tracker import FlowTracker, FlowSession
from features.extractor import FeatureExtractor
from detection.engine import MLInferenceEngine
from alerts.engine import AlertEngine
from threat_analysis.behavioral_profile_engine import BehavioralProfileEngine

logger = setup_logger("RealTimeDetectorSystem")

class RealTimeDetectorSystem:
    """
    Multithreaded network security orchestrator.
    Manages capturing packet streams, mapping connection states, extracting features,
    performing machine learning predictions, and raising explainable alerts.
    """
    def __init__(self, 
                 model_name: str = "random_forest", 
                 interface: str = "any", 
                 pcap_simulation: Optional[str] = None):
        self.config = load_config()
        self.interface = interface
        self.model_name = model_name
        self.pcap_simulation = pcap_simulation
        self.running = False
        
        # Core data structures
        self.packet_queue = queue.Queue(maxsize=10000)
        
        # Modular engine instances
        self.sniffer: Optional[PacketSniffer] = None
        self.tracker = FlowTracker(flow_timeout_seconds=self.config.get("features", {}).get("flow_timeout_seconds", 60))
        self.extractor = FeatureExtractor()
        self.inference_engine = MLInferenceEngine(model_name=self.model_name)
        self.alert_engine = AlertEngine()
        self.behavioral_engine = BehavioralProfileEngine()
        
        # Configuration limits
        self.min_packets = self.config.get("features", {}).get("min_packets_per_flow", 3)
        self.eval_interval = 3.0  # Periodic evaluation interval in seconds

    def run_simulation(self) -> None:
        """
        Reads from a static PCAP file and streams parsed packet records
        into the packet queue to emulate live capture timing.
        """
        logger.info(f"PCAP Simulation Mode active: Streaming from {self.pcap_simulation}...")
        from capture.sniffer import PacketSniffer
        
        # We temporarily load PCAP using Scapy reader to stream packets
        try:
            from scapy.all import PcapReader
            if not os.path.exists(self.pcap_simulation):
                logger.error(f"Target simulation PCAP file not found: {self.pcap_simulation}")
                self.running = False
                return
                
            last_pkt_time = None
            with PcapReader(self.pcap_simulation) as reader:
                for pkt in reader:
                    if not self.running:
                        break
                        
                    parsed = PacketSniffer.parse_packet(pkt)
                    if parsed:
                        try:
                            self.packet_queue.put(parsed, block=True, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        # Replicate interval sleeps to emulate live flows
                        if last_pkt_time is not None:
                            delta = min(0.2, max(0.0, parsed.timestamp - last_pkt_time))
                            time.sleep(delta)
                        last_pkt_time = parsed.timestamp
                        
            logger.info("PCAP Simulation Stream complete.")
        except Exception as e:
            logger.error(f"Error in simulation engine: {e}")
        finally:
            self.running = False

    def run_analyzer_thread(self) -> None:
        """
        Pulls packets from the queue, aggregates flows, and triggers model scoring.
        """
        logger.info("Analyzer Engine Consumer thread started.")
        last_eval_time = time.time()
        
        while self.running or not self.packet_queue.empty():
            try:
                # Retrieve parsed packet from queue
                pkt = self.packet_queue.get(timeout=1.0)
                session = self.tracker.handle_packet(pkt)
                self.behavioral_engine.update_profile(
                    src_ip=pkt.src_ip,
                    dst_ip=pkt.dst_ip,
                    dst_port=pkt.dst_port,
                    protocol=str(pkt.proto),
                    bytes_len=pkt.payload_len,
                    session_id=str(session.flow_key),
                    timestamp=pkt.timestamp,
                    session_duration=session.duration
                )
            except queue.Empty:
                # Perform periodic evaluations when no packets arrive
                if (time.time() - last_eval_time) > self.eval_interval:
                    self.evaluate_active_flows()
                    last_eval_time = time.time()
                continue
                
            # Perform periodic evaluation every 3 seconds
            if (time.time() - last_eval_time) > self.eval_interval:
                self.evaluate_active_flows()
                last_eval_time = time.time()
                
        logger.info("Analyzer Engine Consumer thread stopped.")

    def evaluate_active_flows(self) -> None:
        """
        Computes features on all active connection flows and performs ML/heuristic evaluation.
        """
        active_sessions = self.tracker.get_active_sessions()
        if not active_sessions:
            return
            
        import psutil
        start_eval_time = time.perf_counter()
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Compute sliding temporal behavioral profiles for initiating hosts
        host_profiles = self.extractor.compute_host_features(active_sessions)
        
        # Check active flows exceeding min packet limits
        for flow in active_sessions:
            if len(flow.packets) < self.min_packets:
                continue
                
            # Compute real-time rolling feature vector
            raw_features = self.extractor.extract_single_flow_vector(flow, host_profiles)
            
            # Fetch drift score and behavioral explanations
            profile = self.behavioral_engine.retrieve_profile(flow.src_ip)
            drift_score = profile.get("drift_score", 0.0) if profile else 0.0
            
            # Augment raw_features with behavioral stats for the AI pipeline
            if profile:
                baselines = profile.get("baselines", {})
                raw_features["behavioral_drift_score"] = drift_score
                raw_features["historical_packet_rate"] = baselines.get("packet_rate", 0.0)
                raw_features["historical_destination_diversity"] = baselines.get("destination_diversity", 0.0)
                raw_features["historical_protocol_usage"] = float(len(baselines.get("protocol_usage", {})))
                raw_features["historical_session_count"] = baselines.get("session_count", 0.0)
                raw_features["historical_threat_score"] = profile.get("historical_threat_score", 0.0)
            else:
                raw_features["behavioral_drift_score"] = 0.0
                raw_features["historical_packet_rate"] = 0.0
                raw_features["historical_destination_diversity"] = 0.0
                raw_features["historical_protocol_usage"] = 0.0
                raw_features["historical_session_count"] = 0.0
                raw_features["historical_threat_score"] = 0.0
            
            # Predict scanning behavior using machine learning / fallback heuristics
            prediction, confidence = self.inference_engine.predict(raw_features)
            
            # Adjust confidence based on behavioral profile
            if prediction == 1:
                if drift_score < 30.0:  # ML predicts attack but behavioral profile is normal
                    confidence = max(0.0, confidence - 0.15)
                elif drift_score > 60.0:  # Both ML and behavioral profile indicate suspicious activity
                    confidence = min(1.0, confidence + 0.15)
                    
            drift_explanations = self.behavioral_engine.retrieve_drift_explanation(flow.src_ip, raw_features=raw_features)
            
            # Formulate structured threat incident alerts
            alert_record = self.alert_engine.generate_alert(
                flow=flow,
                raw_features=raw_features,
                ml_prediction=prediction,
                ml_confidence=confidence,
                detection_method=self.model_name,
                drift_score=drift_score,
                drift_explanations=drift_explanations
            )
            
            if alert_record:
                threat_score = alert_record.get("threat_confidence", 0.0)
                self.behavioral_engine.update_threat_score(flow.src_ip, threat_score)

        end_eval_time = time.perf_counter()
        latency_ms = (end_eval_time - start_eval_time) * 1000
        logger.info(f"[Perf] Flows={len(active_sessions)} | Latency={latency_ms:.2f}ms | CPU={cpu_usage}% | RAM={mem_usage:.2f}MB")

    def start(self, duration: int = 0) -> None:
        """
        Coordinates and starts the packet sniffers and model evaluation threads.
        """
        logger.info("==========================================================")
        logger.info("👻 PhantomTrace - Running Real-Time Intrusion Detection System")
        logger.info(f"   Listening Interface: {self.interface} | Model: {self.model_name}")
        logger.info("==========================================================")
        
        self.running = True
        
        # Start capture sniffer (Producer)
        if self.pcap_simulation:
            sniffer_thread = threading.Thread(target=self.run_simulation, daemon=True)
        else:
            self.sniffer = PacketSniffer(interface=self.interface, packet_queue=self.packet_queue)
            sniffer_thread = threading.Thread(
                target=self.sniffer.start, 
                args=(duration, "ip and (tcp or udp)"),
                daemon=True
            )
            
        # Start model inference engine (Consumer)
        analyzer_thread = threading.Thread(target=self.run_analyzer_thread, daemon=True)
        
        sniffer_thread.start()
        analyzer_thread.start()
        
        start_time = time.time()
        try:
            while self.running:
                time.sleep(1.0)
                # Check run-duration limits
                if duration > 0 and (time.time() - start_time) >= duration:
                    logger.info(f"Execution duration limit ({duration}s) reached.")
                    break
        except KeyboardInterrupt:
            logger.info("Termination signal received. Shutting down system gracefully...")
        finally:
            self.running = False
            if self.sniffer:
                self.sniffer.stop()
            self.tracker.stop()
            
            # Wait for consumer thread to flush pending packet queue elements
            analyzer_thread.join(timeout=2.0)
            logger.info("PhantomTrace System closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhantomTrace - Real-time Network Intrusion Detection System CLI Launcher")
    parser.add_argument("-i", "--interface", default="any", help="Network interface to capture live traffic (default: any)")
    parser.add_argument("-m", "--model", default="random_forest", choices=["random_forest", "xgboost", "svm", "isolation_forest"], help="Machine learning model to use (default: random_forest)")
    parser.add_argument("-d", "--duration", type=int, default=0, help="Run duration limit in seconds (default: 0 for continuous execution)")
    parser.add_argument("-s", "--sim-pcap", default=None, help="Simulate real-time stream by streaming from an offline PCAP capture file")
    
    args = parser.parse_args()
    
    # Argparse attribute parsing for hyphenated arguments
    pcap_sim = getattr(args, 'sim_pcap', None)
    
    detector = RealTimeDetectorSystem(
        model_name=args.model,
        interface=args.interface,
        pcap_simulation=pcap_sim
    )
    detector.start(duration=args.duration)
