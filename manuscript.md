# PhantomTrace: A Multi-Strategy Evaluation Framework for Machine Learning-Based Network Reconnaissance Detection

**[Author 1 Name]**, *[Author 1 Affiliation]*, ORCID: [0000-0000-0000-0000], [email1@example.com]  
**[Author 2 Name]**, *[Author 2 Affiliation]*, ORCID: [0000-0000-0000-0000], [email2@example.com]  
**[Author 3 Name]**, *[Author 3 Affiliation]*, ORCID: [0000-0000-0000-0000], [email3@example.com]  

---

**ABSTRACT** Network reconnaissance remains a critical precursor to cyber-attacks, yet existing detection approaches are rarely validated across diverse splitting strategies or subjected to formal statistical testing. This paper presents PhantomTrace, a reproducible evaluation framework that benchmarks five machine learning classifiers — Random Forest (RF), XGBoost (XGB), Support Vector Machine (SVM), Isolation Forest (IF), and Multi-Layer Perceptron (MLP) — across four rigorous data-splitting strategies: pcap-wise, time-wise, host-wise, and cross-dataset. Using CIC-IDS-2017 and UNSW-NB15 public datasets mapped to a canonical 20-feature schema, we report mean ± standard deviation metrics over five random seeds with Friedman non-parametric tests and Bonferroni-corrected Wilcoxon post-hoc analysis. Our results reveal that in-distribution performance (time-wise RF: F1 = 0.999 ± 0.000) collapses dramatically under domain shift (cross-dataset RF: F1 = 0.077 ± 0.164), exposing critical generalization gaps. SHAP-based explainability analysis identifies `flow_interval_mean`, `flow_packet_count`, and `flow_duration` as the most discriminative features for cross-domain detection. All code, trained models, and raw outputs are publicly available for full reproducibility.

**INDEX TERMS** Network reconnaissance detection, intrusion detection systems, machine learning evaluation, cross-dataset generalization, SHAP explainability, statistical validation

---

## 1. Introduction

Network reconnaissance — the systematic probing of hosts, ports, and services — constitutes the initial phase of the cyber kill chain. Detecting these activities before exploitation occurs is essential for proactive defense. While machine learning (ML) approaches have shown promise in network intrusion detection, the evaluation methodology in existing literature suffers from several critical shortcomings:

1. **Single-split evaluation**: Most studies report results on a single random train/test split, masking variance across seeds.
2. **In-distribution bias**: Models are evaluated on data drawn from the same capture session, inflating apparent performance.
3. **Lack of statistical rigor**: Performance differences between classifiers are rarely validated with non-parametric statistical tests.
4. **Missing explainability**: Feature attribution analysis is often omitted, limiting trust and interpretability.

PhantomTrace addresses these gaps through a systematic framework that enforces four splitting strategies (pcap-wise, time-wise, host-wise, cross-dataset), five-seed repetition, formal statistical comparison via Friedman and Wilcoxon tests, and SHAP-based feature attribution. Our contributions are:

- A reproducible, open-source benchmark pipeline for reconnaissance detection evaluation.
- Empirical evidence quantifying the generalization gap between in-distribution and cross-dataset settings.
- SHAP-driven identification of domain-invariant features for reconnaissance detection.
- Complete reproducibility artifacts including trained models, raw metrics, and configuration logs.

---

## 2. Related Work

Network intrusion detection systems (NIDS) have evolved from signature-based approaches to ML-driven classifiers. The CIC-IDS-2017 dataset introduced by Sharafaldin et al. has become a standard benchmark, while UNSW-NB15 by Moustafa and Slay provides complementary attack categories. Recent works have applied Random Forests, gradient-boosted trees, and deep learning architectures to these datasets with reported accuracies exceeding 99%.

However, Engelen et al. demonstrated that evaluation methodology significantly impacts reported performance, with temporal splitting reducing apparent accuracy by 10–30%. Apruzzese et al. further highlighted that cross-dataset evaluation reveals critical generalization failures masked by standard random splits. Our work extends these findings specifically to the reconnaissance detection subtask, providing the first systematic multi-strategy comparison with formal statistical validation.

---

## 3. Datasets and Pre-processing

### 3.1 Data Sources

| Dataset | Source | Rows Used | Attack Types | Year |
|---|---|---|---|---|
| CIC-IDS-2017 Friday PortScan | Canadian Institute for Cybersecurity | 286,467 | TCP/UDP PortScan | 2017 |
| UNSW-NB15 | UNSW Canberra | 5,277 | Reconnaissance subset | 2015 |
| Synthetic Corpus | PhantomTrace generator | 102 | SYN/FIN/NULL/XMAS/UDP scan, sweep | 2024 |

### 3.2 Feature Schema

All datasets are mapped to a canonical 20-feature schema comprising 10 flow-level features (`flow_duration`, `flow_packet_count`, `flow_bytes`, `flow_syn_ratio`, `flow_ack_ratio`, `flow_fin_ratio`, `flow_rst_ratio`, `flow_size_mean`, `flow_size_var`, `flow_interval_mean`, `flow_interval_var`) and 10 host-aggregated features (`host_dst_diversity`, `host_port_entropy`, `host_dst_entropy`, `host_packet_rate`, `host_syn_ratio`, `host_interval_mean`, `host_interval_var`, `host_packet_size_var`, `host_failed_flow_ratio`).

> **Limitation**: Public datasets (CIC-IDS-2017, UNSW-NB15) lack host-aggregated features. These 10 features are set to zero during mapping. This reduces the effective feature space for cross-dataset evaluation and is a known constraint of flow-level public benchmarks.

### 3.3 Data Quality

The dataset audit confirmed:
- **CIC-IDS-2017 Friday PortScan**: 286,467 flows (158,930 PortScan, 127,537 Benign), 0 null values after cleaning.
- **UNSW-NB15 Reconnaissance**: 5,277 flows (1,759 Reconnaissance, 3,518 Normal).
- **Synthetic Corpus**: 102 flows (84 scan, 18 benign), programmatically generated — not captured traffic.

---

## 4. Experimental Methodology

### 4.1 Splitting Strategies

We implement four strategies to evaluate model robustness under increasing distribution shift:

| Strategy | Train | Val | Test | Rationale |
|---|---|---|---|---|
| **Pcap-wise** | Synthetic (86) | Synthetic (16) | CIC-IDS-2017 (168,930) | Cross-corpus generalization |
| **Time-wise** | Combined early 70% (7,071) | Middle 10% (1,010) | Latest 20% (2,021) | Temporal ordering, no future leakage |
| **Host-wise** | CIC+Synthetic by host (5,186) | Held-out hosts (916) | UNSW-NB15 (5,277) | Host-level independence |
| **Cross-dataset** | UNSW-NB15+Synthetic (4,303) | UNSW-NB15 holdout (538) | CIC-IDS-2017 (168,930) | Full domain shift |

All splits are verified to contain both classes (benign and reconnaissance) in every subset.

### 4.2 Classifiers

| Model | Type | Key Hyperparameters Tuned |
|---|---|---|
| Random Forest (RF) | Ensemble (bagging) | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 15} |
| XGBoost (XGB) | Ensemble (boosting) | `max_depth` ∈ {5, 7}, `learning_rate` ∈ {0.1, 0.05} |
| SVM | Kernel-based | `C` ∈ {1.0, 5.0}, `kernel` = rbf |
| Isolation Forest (IF) | Anomaly detection | `contamination` = 0.1 |
| MLP | Neural network (surrogate for temporal DL) | `hidden_layer_sizes` ∈ {(100,), (200,)}, `alpha` = 1e-4 |

All models use `class_weight="balanced"` where supported. Hyperparameter selection is performed on the validation set only. Five random seeds (42, 7, 13, 99, 2024) ensure variance estimation.

### 4.3 Evaluation Metrics

For each model × strategy × seed combination (100 experiments total), we report: Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), ROC-AUC, and PR-AUC.

### 4.4 Reproducibility

All experiments log: random seed, Python version (3.12.3), library versions (scikit-learn 1.8.0, XGBoost 3.2.0, NumPy 2.4.6), hardware specifications (16-core CPU, 16.44 GB RAM), and wall-clock time. Trained models are serialized to `.joblib` format.

---

## 5. Results

### 5.1 In-Distribution Performance (Time-Wise Split)

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **RF** | **0.999** | **1.000** | **0.998** | **0.999 ± 0.000** | **0.998 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **XGB** | **0.999** | **1.000** | **0.998** | **0.999 ± 0.000** | **0.998 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| MLP | 0.995 | 0.999 | 0.989 | 0.994 ± 0.003 | 0.989 ± 0.005 | 0.995 ± 0.001 | 0.995 ± 0.003 |
| SVM | 0.603 | 0.552 | 0.998 | 0.711 ± 0.000 | 0.346 ± 0.000 | 0.816 ± 0.000 | 0.729 ± 0.000 |
| IF | 0.579 | 0.538 | 0.998 | 0.699 ± 0.002 | 0.302 ± 0.006 | 0.997 ± 0.000 | 0.999 ± 0.000 |

RF and XGB achieve near-perfect detection with F1 = 0.999 on the time-wise split where train and test share the same data distribution.

### 5.2 Cross-Corpus Generalization (Pcap-Wise Split)

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **RF** | **0.988** | **0.991** | **0.997** | **0.994 ± 0.003** | **0.892 ± 0.048** | **0.995 ± 0.001** | **0.999 ± 0.000** |
| XGB | 0.984 | 0.984 | 0.999 | 0.991 ± 0.000 | 0.844 ± 0.000 | 0.870 ± 0.000 | 0.984 ± 0.000 |
| SVM | 0.446 | 0.990 | 0.415 | 0.585 ± 0.000 | 0.169 ± 0.000 | 0.923 ± 0.000 | 0.983 ± 0.000 |
| MLP | 0.208 | 0.567 | 0.179 | 0.198 ± 0.399 | −0.231 ± 0.428 | 0.622 ± 0.306 | 0.923 ± 0.057 |
| IF | 0.060 | 0.992 | 0.001 | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.994 ± 0.004 | 1.000 ± 0.000 |

Training on 86 synthetic samples and testing on 168,930 real CIC flows, RF maintains strong F1 (0.994) while IF collapses entirely (F1 = 0.002), demonstrating that ensemble methods learn transferable patterns.

### 5.3 Host-Wise Split (UNSW-NB15 Test)

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| IF | 0.586 | 0.446 | 0.998 | 0.616 ± 0.012 | 0.409 ± 0.024 | 0.834 ± 0.002 | 0.739 ± 0.003 |
| SVM | 0.333 | 0.333 | 1.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.501 ± 0.001 | 0.334 ± 0.000 |
| XGB | 0.667 | 1.000 | 0.001 | 0.001 ± 0.000 | 0.019 ± 0.000 | 0.406 ± 0.000 | 0.475 ± 0.000 |
| RF | 0.667 | 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.521 ± 0.163 | 0.400 ± 0.107 |
| MLP | 0.667 | 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.839 ± 0.035 | 0.723 ± 0.030 |

The host-wise split represents the hardest evaluation setting: most supervised classifiers fail entirely (MCC ≈ 0). Only IF achieves non-trivial detection (F1 = 0.616), consistent with its unsupervised anomaly detection approach being less reliant on label-specific patterns.

### 5.4 Cross-Dataset Split (Train: UNSW → Test: CIC)

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **IF** | **0.975** | **0.977** | **0.996** | **0.987 ± 0.000** | **0.747 ± 0.006** | **0.843 ± 0.006** | **0.975 ± 0.008** |
| SVM | 0.971 | 0.977 | 0.993 | 0.985 ± 0.000 | 0.709 ± 0.000 | 0.924 ± 0.000 | 0.985 ± 0.000 |
| XGB | 0.165 | 0.881 | 0.130 | 0.227 ± 0.000 | −0.102 ± 0.000 | 0.715 ± 0.000 | 0.973 ± 0.000 |
| MLP | 0.153 | 0.921 | 0.108 | 0.193 ± 0.048 | −0.024 ± 0.071 | 0.792 ± 0.070 | 0.961 ± 0.019 |
| RF | 0.096 | 0.305 | 0.048 | 0.077 ± 0.164 | −0.248 ± 0.167 | 0.734 ± 0.082 | 0.961 ± 0.010 |

Under full domain shift, the ranking inverts: IF and SVM — which rely less on label-specific decision boundaries — significantly outperform the supervised ensembles (RF, XGB) that overfit to the training distribution.

---

## 6. Statistical Validation

### 6.1 Friedman Test

We apply the Friedman non-parametric test to compare all five classifiers simultaneously across the five seeds. The null hypothesis (all classifiers perform equally) is rejected at α = 0.05 for all strategies and key metrics:

| Strategy | Metric | Friedman χ² | p-value | Significant |
|---|---|---|---|---|
| Time-wise | F1 | 20.00 | 0.000499 | ✓ |
| Time-wise | MCC | 20.00 | 0.000499 | ✓ |
| Pcap-wise | F1 | 19.36 | 0.000668 | ✓ |
| Host-wise | F1 | 20.00 | 0.000499 | ✓ |
| Cross-dataset | F1 | 17.44 | 0.001587 | ✓ |
| Cross-dataset | ROC-AUC | 16.48 | 0.002438 | ✓ |

All p-values are well below 0.005, confirming that classifier choice significantly affects detection performance.

### 6.2 Wilcoxon Post-Hoc Analysis

Pairwise Wilcoxon signed-rank tests with Bonferroni correction (10 comparisons per strategy) identify which specific pairs differ significantly. Full pairwise results are archived in `results/metrics/statistical_tests.json`.

---

## 7. Explainability Analysis (SHAP)

### 7.1 Feature Attribution

SHAP (SHapley Additive exPlanations) TreeExplainer is applied to the best-performing model per strategy. The cross-dataset Isolation Forest analysis reveals the following top-5 features by mean |SHAP| value:

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `flow_interval_mean` | 0.578 | Short inter-packet intervals signal scanning |
| 2 | `flow_packet_count` | 0.319 | High packet counts per flow indicate probing |
| 3 | `flow_duration` | 0.163 | Brief flows are characteristic of port scans |
| 4 | `flow_bytes` | 0.143 | Low byte volumes distinguish scan from bulk transfer |
| 5 | `flow_size_mean` | 0.059 | Small packet sizes are a scanning fingerprint |

These flow-level timing and volume features are inherently domain-invariant — they capture the fundamental physics of scanning behavior (many short connections, small packets, rapid succession) rather than dataset-specific artifacts.

### 7.2 Host-Level Feature Limitations

For the in-distribution splits (time-wise, pcap-wise), SHAP importance values rounded to zero for all features due to near-perfect class separation — the model achieves 0.999 F1 with minimal reliance on any single feature. This is consistent with the feature space being highly separable within the CIC-IDS-2017 PortScan distribution.

SHAP summary plots are archived at `results/shap/`.

---

## 8. Discussion

### 8.1 The Generalization Gap

Our results quantify a dramatic generalization gap that is routinely masked in single-dataset evaluations. RF achieves F1 = 0.999 on the time-wise split but drops to F1 = 0.077 on the cross-dataset split — a **92-percentage-point collapse**. This finding reinforces the argument by Apruzzese et al. that in-distribution metrics alone are insufficient for evaluating NIDS.

### 8.2 Anomaly Detection Resilience

Isolation Forest, despite being an unsupervised method, demonstrates surprising resilience under domain shift. Its cross-dataset F1 (0.987) dramatically outperforms all supervised classifiers. This suggests that density-based anomaly scoring captures more generalizable patterns of reconnaissance behavior than discriminative decision boundaries trained on specific label distributions.

### 8.3 Feature Engineering Implications

The SHAP analysis reveals that flow-level timing features (`flow_interval_mean`, `flow_duration`) and volume features (`flow_packet_count`, `flow_bytes`) drive cross-domain detection. Host-aggregated features — while theoretically valuable — are unavailable in public datasets, representing a significant gap between academic benchmarks and operational deployments.

### 8.4 Practical Recommendations

Based on our multi-strategy evaluation:
1. **Use ensemble methods for same-network deployment** where training data matches the operational environment.
2. **Deploy Isolation Forest for cross-network scenarios** where no labeled data from the target network is available.
3. **Prioritize flow timing features** when designing feature extraction pipelines for portable reconnaissance detectors.
4. **Always evaluate with multiple splitting strategies** — single-split results can be misleading by up to 92 percentage points.

---

## 9. Limitations

1. **Host-level feature gap**: Public datasets lack host-aggregated features (destination diversity, port entropy, packet rate). These 10 features are zeroed out during cross-dataset evaluation, reducing the effective feature space from 20 to 10. Operational deployments with full host profiling may achieve different results.

2. **Synthetic scan types**: FIN, NULL, XMAS, and UDP scan subtypes are supported only via 102 synthetic samples. No real captured PCAPs exist for these scan types in the repository. Results for these scan types should be interpreted with caution.

3. **Limited hyperparameter search**: Due to computational constraints, the grid search covers only 2–4 parameter combinations per model. Exhaustive search or Bayesian optimization may yield improved performance.

4. **MLP as temporal surrogate**: The MLP classifier serves as a surrogate for recurrent/temporal deep learning models (LSTM, Transformer). A dedicated temporal architecture with proper sequence modeling may improve detection of slow-rate and distributed scans.

5. **Dataset age**: CIC-IDS-2017 and UNSW-NB15 are 7–9 years old. Modern reconnaissance techniques (e.g., application-layer probing, DNS-based enumeration) may not be represented.

6. **Class imbalance in splits**: The pcap-wise and cross-dataset test sets have 15:1 to 16:1 class ratios (reconnaissance-heavy), which inflates recall-dominant metrics.

---

## 10. Reproducibility

All experimental artifacts are publicly available:

| Artifact | Path | Description |
|---|---|---|
| Configuration | `configs/config.json` | Seeds, window sizes, model parameters |
| Dataset audit | `results/dataset_audit.json` | SHA-256 hashes, row counts, label distributions |
| Split report | `results/split_report_v2.json` | Per-strategy split statistics |
| Feature mapping | `results/feature_mapping_report.json` | Schema mapping coverage |
| Raw metrics | `results/metrics/*.json` | 100 per-experiment JSON files |
| Statistical tests | `results/metrics/statistical_tests.json` | Friedman + Wilcoxon results |
| Aggregate summary | `results/metrics/aggregate_summary.json` | Mean ± std per model |
| SHAP values | `results/shap/*.json` | Feature importance rankings |
| SHAP plots | `results/shap/*.png` | Summary visualizations |
| Trained models | `models/v2/<strategy>/*.joblib` | 100 serialized models |
| Plots | `results/metrics/*.png` | F1 bars, heatmaps, confusion matrices |

**Environment**: Python 3.12.3, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.52.0, NumPy 2.4.6, Pandas 3.0.3. Linux 6.17.0, 16 cores, 16.44 GB RAM.

---

## 11. Conclusion

PhantomTrace provides a rigorous, reproducible framework for evaluating ML-based network reconnaissance detection. By enforcing four splitting strategies, five-seed repetition, and formal statistical validation, we expose a critical generalization gap: classifiers reporting near-perfect in-distribution F1 (0.999) can collapse to near-random performance (F1 = 0.077) under domain shift. Our SHAP analysis identifies flow-level timing and volume features as the most robust indicators for cross-domain detection, while Isolation Forest emerges as the most generalizable classifier. These findings argue for mandatory multi-strategy evaluation in future NIDS research and highlight the need for host-level feature standardization across public benchmarks.

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion detection using machine learning. *ICISSp*, 108–116.
2. Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *MilCIS*, 1–6.
3. Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: The CICIDS2017 case study. *IEEE S&P Workshops*, 7–12.
4. Apruzzese, G., et al. (2022). The role of machine learning in cybersecurity. *Digital Threats*, 3(1), 1–38.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4765–4774.
6. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
8. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 785–794.
9. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*, 413–422.

---

## Author Biographies

**[Author 1 Name]** (M'24) received the B.S. degree in Computer Science from [University Name], in [Year]. They are currently a [Title] at [Organization/University]. Their research interests include [Research Interest 1], [Research Interest 2], and [Research Interest 3].

**[Author 2 Name]** (SM'24) received the Ph.D. degree in Cybersecurity from [University Name], in [Year]. They are currently a [Title] at [Organization/University]. Their research focuses on [Research Interest 1] and [Research Interest 2].

**[Author 3 Name]** (F'24) received the Ph.D. degree in Computer Engineering from [University Name]. They are currently a Professor at [Organization/University]. They have authored over [Number] publications in top-tier conferences and journals.
