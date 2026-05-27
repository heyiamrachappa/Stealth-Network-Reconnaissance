#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, Any

def get_project_root() -> str:
    """
    Returns the absolute path to the root of the project.
    Resolves dynamically based on this file's location.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_name: str = "config.json") -> Dict[str, Any]:
    """
    Loads JSON configuration from the configs directory.
    """
    root = get_project_root()
    config_path = os.path.join(root, "configs", config_name)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Overwrite standard keys dynamically to keep paths correct
                config["project_root"] = root
                if "directories" in config:
                    for key, val in config["directories"].items():
                        config["directories"][key] = val.replace("/home/yi/Stealth System", root)
                if "alerts" in config:
                    if "alert_log_file" in config["alerts"]:
                        config["alerts"]["alert_log_file"] = config["alerts"]["alert_log_file"].replace("/home/yi/Stealth System", root)
                    if "log_file" in config["alerts"]:
                        config["alerts"]["log_file"] = config["alerts"]["log_file"].replace("/home/yi/Stealth System", root)
                return config
        except Exception as e:
            print(f"[ERROR] Failed to load config from {config_path}: {e}")
    
    # Fallback default configuration
    return {
        "project_root": root,
        "directories": {
            "pcaps": os.path.join(root, "pcaps"),
            "dataset": os.path.join(root, "dataset"),
            "models": os.path.join(root, "models"),
            "dashboard": os.path.join(root, "dashboard"),
            "results": os.path.join(root, "results"),
            "logs": os.path.join(root, "logs"),
            "configs": os.path.join(root, "configs")
        },
        "capture": {
            "default_interface": "any",
            "pcap_filename": "recon_capture.pcap",
            "sniff_timeout": 30,
            "buffer_size": 1000
        },
        "features": {
            "flow_timeout_seconds": 60,
            "sliding_window_seconds": 30,
            "min_packets_per_flow": 3
        },
        "ml": {
            "random_state": 42,
            "test_size": 0.2,
            "rf_n_estimators": 100,
            "xgboost_max_depth": 5,
            "models_to_train": ["random_forest", "isolation_forest", "xgboost"]
        },
        "alerts": {
            "alert_log_file": os.path.join(root, "logs", "alerts.log"),
            "log_file": os.path.join(root, "logs", "system.log"),
            "entropy_threshold": 3.0,
            "syn_ratio_threshold": 5.0
        }
    }

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with format, streams to stdout, and logs to a system file.
    """
    config = load_config()
    log_file = config.get("alerts", {}).get("log_file", os.path.join(get_project_root(), "logs", "system.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s")
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
