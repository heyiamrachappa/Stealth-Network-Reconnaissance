# AI-Assisted Detection of Stealth Network Reconnaissance

An intelligent cybersecurity research system for detecting stealth reconnaissance, low-and-slow scanning, and anomalous network behavior using packet analysis, behavioral feature engineering, and machine learning.

---

## Overview

Traditional Intrusion Detection Systems (IDS) often fail to detect:
- low-and-slow scans
- stealth reconnaissance
- fragmented probing
- distributed enumeration
- TCP flag manipulation

This project focuses on identifying reconnaissance intent through:
- packet metadata analysis
- behavioral anomaly detection
- feature engineering
- machine learning classification
- stealth scan correlation

The system analyzes `.pcap` network captures, extracts suspicious behavioral indicators, and classifies reconnaissance activity using AI-assisted detection models.

---

# Features

- PCAP traffic analysis
- TCP/IP packet inspection
- SYN/FIN/NULL/XMAS scan detection
- Feature extraction pipeline
- Machine learning-based threat detection
- XGBoost classification engine
- Real-time detection architecture
- Behavioral anomaly analysis
- Threat scoring system
- Streamlit dashboard integration
- Explainable detection workflow

---

# Project Architecture

```text
PCAP / Live Traffic
          ↓
Packet Parsing Engine
          ↓
Feature Extraction Pipeline
          ↓
Behavioral Analysis Layer
          ↓
Machine Learning Models
          ↓
Threat Classification
          ↓
Dashboard & Alerts
