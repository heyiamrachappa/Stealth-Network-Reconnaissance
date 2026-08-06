import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from utils.helpers import get_project_root
from ml_engine.engine import MLInferenceEngine
from features.extractor import FeatureExtractor
from scapy.all import rdpcap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

print("Writing evaluation data to outputs.csv")
