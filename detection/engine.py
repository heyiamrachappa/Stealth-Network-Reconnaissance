#!/usr/bin/env python3
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils.helpers import setup_logger, load_config, get_project_root

logger = setup_logger("MLInferenceEngine")

class MLInferenceEngine:
    """
    Manages low-latency real-time machine learning predictions.
    Loads models dynamically, scales feature vectors, and computes anomaly confidence scores.
    """
    def __init__(self, model_name: str = "random_forest"):
        self.config = load_config()
        self.project_root = self.config.get("project_root", get_project_root())
        self.models_dir = self.config.get("directories", {}).get("models", os.path.join(self.project_root, "models"))
        
        self.model_name = model_name
        self.scaler = self._load_scaler()
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()

    def _load_scaler(self) -> Optional[Any]:
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                logger.info(f"Loading StandardScaler from {scaler_path}")
                return joblib.load(scaler_path)
            except Exception as e:
                logger.error(f"Failed to load StandardScaler: {e}")
        logger.warning("No pre-trained StandardScaler found. Live inference will operate unscaled!")
        return None

    def _load_model(self) -> Optional[Any]:
        model_path = os.path.join(self.models_dir, f"{self.model_name}_model.joblib")
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading ML classifier: '{self.model_name}' from {model_path}")
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load ML model {self.model_name}: {e}")
        logger.warning(f"ML Model '{self.model_name}' not found. Falling back to heuristic expert rule classification.")
        return None

    def _load_feature_names(self) -> List[str]:
        names_path = os.path.join(self.models_dir, "feature_names.json")
        if os.path.exists(names_path):
            try:
                with open(names_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load feature schema names: {e}")
        return []

    def predict(self, raw_features: Dict[str, float]) -> Tuple[int, float]:
        """
        Processes a raw flow feature dictionary and outputs class prediction and confidence.
        :param raw_features: Dictionary of calculated connection metrics.
        :return: Tuple of (prediction [0/1], confidence_score [0.0 - 1.0])
        """
        # Fallback to heuristics if no ML models are loaded
        if self.model is None or self.scaler is None or not self.feature_names:
            return self._heuristic_fallback(raw_features)

        try:
            # Structurize features to match the exact schema columns
            df_inf = pd.DataFrame([raw_features])[self.feature_names]
            
            # Standardize feature matrix
            df_scaled = pd.DataFrame(self.scaler.transform(df_inf), columns=self.feature_names)
            
            if self.model_name == "isolation_forest":
                raw_pred = self.model.predict(df_scaled)[0]
                # Isolation Forest outputs -1 for anomalies, 1 for normal
                prediction = 1 if raw_pred == -1 else 0
                confidence = -self.model.decision_function(df_scaled)[0]
                # Scale Isolation Forest decision function between 0 and 1
                confidence = float(min(1.0, max(0.0, (confidence + 0.5) / 1.0)))
            else:
                prediction = int(self.model.predict(df_scaled)[0])
                confidence = float(self.model.predict_proba(df_scaled)[0][1])
                
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Real-time ML inference failed: {e}. Using heuristic fallback.")
            return self._heuristic_fallback(raw_features)

    def _heuristic_fallback(self, raw_features: Dict[str, float]) -> Tuple[int, float]:
        """
        Expert heuristic threat rules used when machine learning components are uninitialized.
        """
        port_entropy = raw_features.get("host_port_entropy", 0.0)
        failed_flow_ratio = raw_features.get("host_failed_flow_ratio", 0.0)
        dst_diversity = raw_features.get("host_dst_diversity", 1.0)
        syn_ratio= raw_features.get("host_syn_ratio", 0.0)
        
        # Rule 1: High port entropy + failed flows (Horizontal/Vertical Port Scan)
        if port_entropy > 2.2 and failed_flow_ratio > 0.8:
            return 1, 0.95
            
        # Rule 2: High IP destination diversity + SYN flag dominance (Subnet Enumeration)
        if dst_diversity > 4 and syn_ratio > 0.8:
            return 1, 0.90
            
        return 0, 0.0
