#!/usr/bin/env python3
# ==============================================================================
# Phase 6 & 7 - Machine Learning & Model Evaluation Module
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import os
import sys
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("ModelEngine")


class ModelEngine:
    """
    Manages training, evaluation, validation, plotting, and serialization 
    of cybersecurity detection classifiers.
    """
    def __init__(self, config_path: str = "/home/yi/Stealth System/configs/config.json"):
        self.config = self._load_config(config_path)
        self.project_root = self.config.get("project_root", "/home/yi/Stealth System")
        self.dataset_dir = self.config.get("directories", {}).get("dataset", f"{self.project_root}/dataset")
        self.models_dir = self.config.get("directories", {}).get("models", f"{self.project_root}/models")
        self.results_dir = self.config.get("directories", {}).get("results", f"{self.project_root}/results")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        return {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads the preprocessed train and test splits from the dataset directory.
        """
        train_path = f"{self.dataset_dir}/train_features.csv"
        test_path = f"{self.dataset_dir}/test_features.csv"
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Preprocessed train/test features not found. Run dataset pipeline first.")
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        X_train = train_df.drop(columns=["label"])
        y_train = train_df["label"]
        
        X_test = test_df.drop(columns=["label"])
        y_test = test_df["label"]
        
        logger.info(f"Loaded train set size: {X_train.shape[0]} samples, test set size: {X_test.shape[0]} samples")
        return X_train, y_train, X_test, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Instantiates and trains Random Forest, Isolation Forest, XGBoost, and SVM.
        """
        logger.info("Initializing model training suites...")
        models = {}
        
        # 1. Random Forest Classifier
        rf_est = self.config.get("ml", {}).get("rf_n_estimators", 100)
        logger.info(f"Training Random Forest (n_estimators={rf_est})...")
        rf = RandomForestClassifier(n_estimators=rf_est, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        models["random_forest"] = rf
        
        # 2. XGBoost Classifier
        logger.info("Training XGBoost Classifier...")
        xgb = XGBClassifier(
            max_depth=self.config.get("ml", {}).get("xgboost_max_depth", 5),
            eval_metric="logloss",
            random_state=42
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
        
        # 3. Support Vector Machine (SVM)
        logger.info("Training Support Vector Machine (RBF kernel)...")
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(X_train, y_train)
        models["svm"] = svm
        
        # 4. Isolation Forest (Unsupervised Anomaly Detection)
        # Typically trained mostly on normal instances, but can be fit on all training data
        logger.info("Training Unsupervised Isolation Forest...")
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X_train)  # Unsupervised fitting
        models["isolation_forest"] = iso
        
        # Save all trained models
        for name, model in models.items():
            model_path = f"{self.models_dir}/{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Successfully serialized {name} to {model_path}")
            
        return models

    def evaluate_models(self, 
                        models: Dict[str, Any], 
                        X_test: pd.DataFrame, 
                        y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates each model, produces classification metrics, and exports metrics.
        """
        logger.info("Evaluating trained models on holdout test set...")
        metrics_summary = {}
        
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Predict
            if name == "isolation_forest":
                # Isolation Forest predicts -1 for anomalies, 1 for normal
                raw_preds = model.predict(X_test)
                y_pred = np.where(raw_preds == -1, 1, 0)
                y_prob = -model.decision_function(X_test)  # Anomaly score as probability surrogate
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            metrics_summary[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }
            
            logger.info(f"\n[{name.upper()} RESULTS]\n" + classification_report(y_test, y_pred, zero_division=0))
            
            # ROC Curve Calculation
            if y_test.nunique() > 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
                metrics_summary[name]["auc"] = roc_auc
            else:
                logger.warning(f"Only one class present in y_test. Skipping ROC curve plot for {name}.")
                
        # Finalize and Save ROC Plot
        if y_test.nunique() > 1:
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("Receiver Operating Characteristic (ROC) Comparison")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            roc_path = f"{self.results_dir}/roc_curves.png"
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved model ROC curve comparison to {roc_path}")
            
        # Export metrics JSON
        metrics_path = f"{self.results_dir}/model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Plot Feature Importance for Random Forest & XGBoost
        self.plot_feature_importances(models, X_test.columns)
        
        return metrics_summary

    def plot_feature_importances(self, models: Dict[str, Any], feature_names: List[str]) -> None:
        """
        Plots and saves feature importance graphs for Random Forest and XGBoost.
        """
        for name in ["random_forest", "xgboost"]:
            model = models.get(name)
            if not model:
                continue
                
            logger.info(f"Generating feature importance plot for {name}...")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importance Profile ({name.replace('_', ' ').title()})")
            plt.bar(range(len(importances)), importances[indices], align="center", color="#3F51B5")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
            plt.xlim([-1, len(importances)])
            plt.ylabel("Relative Significance")
            plt.tight_layout()
            
            imp_path = f"{self.results_dir}/{name}_feature_importance.png"
            plt.savefig(imp_path, dpi=300)
            plt.close()
            logger.info(f"Saved feature importance plot to {imp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stealth IDS - Model Suite CLI")
    parser.add_argument("--train", action="store_true", help="Train, serialize, and evaluate models")
    
    args = parser.parse_args()
    
    engine = ModelEngine()
    if args.train:
        try:
            X_train, y_train, X_test, y_test = engine.load_data()
            models = engine.train_models(X_train, y_train)
            engine.evaluate_models(models, X_test, y_test)
        except Exception as e:
            logger.error(f"Failed to execute training pipeline: {e}")
            raise e
    else:
        parser.print_help()
