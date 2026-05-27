#!/usr/bin/env python3
import os
import sys
import time
import shutil
import logging

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logger, load_config, get_project_root
from pipeline import DatasetPipeline
from models.engine import ModelEngine
from realtime_detector import RealTimeDetectorSystem

logger = setup_logger("SystemValidation")

def run_pipeline_validation():
    logger.info("==========================================================")
    logger.info("🛡️ STARTING MODULAR REAL-TIME SYSTEM INTEGRATION VALIDATION")
    logger.info("==========================================================")
    
    project_root = get_project_root()
    pcap_path = os.path.join(project_root, "pcaps", "synthetic_scan.pcap")
    alert_file = os.path.join(project_root, "logs", "alerts.log")
    
    # 1. Clean previous run logs and results for absolute verification
    logger.info("Step 1: Cleaning previous logs and serialized weights...")
    for folder in ["dataset", "models", "results", "logs"]:
        path = os.path.join(project_root, folder)
        if os.path.exists(path):
            if folder == "models":
                # Only clean serialized weights and configurations, preserve our python code files
                for item in os.listdir(path):
                    if not item.endswith(".py"):
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
            else:
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    
    # 2. Generate Synthetic Attack PCAP
    logger.info("Step 2: Generating synthetic stealth scanning PCAP...")
    try:
        from scapy.all import IP, TCP, wrpcap
        pkts = []
        base_time = time.time()
        
        # Simulating normal TCP handshake flows (10 sessions, 3 packets each)
        for i in range(10):
            sport = 45000 + i
            pkts.append(IP(src="192.168.1.100", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="S"))
            pkts[-1].time = base_time + (i * 0.2)
            pkts.append(IP(src="8.8.8.8", dst="192.168.1.100")/TCP(sport=80, dport=sport, flags="SA"))
            pkts[-1].time = base_time + (i * 0.2) + 0.05
            pkts.append(IP(src="192.168.1.100", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="A"))
            pkts[-1].time = base_time + (i * 0.2) + 0.1
            
        # Simulating stealth SYN scan attacker (1 attacker IP -> targets 25 victim ports)
        # Slow intervals (0.5s between SYNs), only SYN sent (half-open scan behavior)
        for i in range(25):
            target_port = 20 + i
            pkts.append(IP(src="192.168.1.187", dst="192.168.1.5")/TCP(sport=38200+i, dport=target_port, flags="S"))
            pkts[-1].time = base_time + 5.0 + (i * 0.5)
            
        os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
        wrpcap(pcap_path, pkts)
        logger.info(f"Successfully created synthetic scan PCAP: {pcap_path} ({len(pkts)} packets)")
    except Exception as e:
        logger.error(f"Failed to generate synthetic scan PCAP: {e}")
        sys.exit(1)

    # 3. Process Dataset Pipeline
    logger.info("Step 3: Executing Feature Preprocessing & Scale Normalization Pipeline...")
    try:
        pipeline = DatasetPipeline()
        train_df, test_df = pipeline.build_and_split_dataset(
            pcap_files=[pcap_path],
            scanner_ips=["192.168.1.187"],
            test_size=0.2
        )
        logger.info("Dataset Pipeline executed successfully.")
        logger.info(f"  Training Split Size: {train_df.shape}")
        logger.info(f"  Testing Split Size: {test_df.shape}")
    except Exception as e:
        logger.error(f"Dataset Pipeline validation failed: {e}")
        sys.exit(1)

    # 4. Train Models Suite
    logger.info("Step 4: Executing Machine Learning Classifier Training & Evaluation...")
    try:
        engine = ModelEngine()
        X_train, y_train, X_test, y_test = engine.load_data()
        models = engine.train_models(X_train, y_train)
        metrics = engine.evaluate_models(models, X_test, y_test)
        
        logger.info("Model suite trained and evaluated successfully.")
        for name, data in metrics.items():
            logger.info(f"  -> Model '{name}': Accuracy = {data['accuracy']*100:.2f}% | F1-Score = {data['f1_score']*100:.2f}%")
    except Exception as e:
        logger.error(f"Model Training validation failed: {e}")
        sys.exit(1)

    # 5. Run Real-Time Stream Detector Simulation
    logger.info("Step 5: Executing Real-Time Streaming Detection Engine Simulation...")
    try:
        # Load detector with pre-trained Random Forest and PCAP simulation file
        detector = RealTimeDetectorSystem(
            model_name="random_forest",
            interface="none",
            pcap_simulation=pcap_path
        )
        # Run simulation
        detector.start(duration=15)
        
        # Verify alert output
        if os.path.exists(alert_file) and os.path.getsize(alert_file) > 0:
            with open(alert_file, 'r') as f:
                alerts_count = sum(1 for line in f if line.strip())
            logger.info(f"🟢 SUCCESS! Real-time simulation complete. Triggered {alerts_count} threat alerts in alerts.log.")
        else:
            logger.error("🔴 FAILURE: Real-time simulation completed but no threat alarms were triggered.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Real-time detector validation failed: {e}")
        sys.exit(1)

    logger.info("==========================================================")
    logger.info("🎉 CONGRATULATIONS! ALL RESTRENGTHENED MODULAR REAL-TIME")
    logger.info("   DETECTION SYSTEM PHASES INTEGRATED & PASSED SUCCESSFULLY!")
    logger.info("==========================================================")


if __name__ == "__main__":
    run_pipeline_validation()
