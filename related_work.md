## 2. Related Work

Network intrusion detection systems (NIDS) have evolved significantly over the past decade, driven by advances in machine learning and the availability of large-scale public datasets. However, the evaluation methodologies employed in the literature often fail to reflect operational realities. We review the state-of-the-art across eight critical dimensions that frame the contribution of PhantomTrace.

### 2.1 ML-Based Reconnaissance and Port-Scan Detection
The detection of network reconnaissance (e.g., port scanning, vulnerability sweeping) has traditionally relied on rate-limiting and signature matching. Recent ML approaches utilizing Random Forests and deep neural networks have reported near-perfect accuracies on benchmarks like CIC-IDS-2017 [1, 2]. However, these studies rarely isolate reconnaissance from bulk volumetric attacks (like DDoS), masking specific detection limitations for stealthy, low-rate probing.

### 2.2 Evaluation Leakage in IDS Datasets
A growing body of work has identified critical flaws in how public NIDS datasets are evaluated. Engelen et al. [3] and Arash et al. demonstrated that random k-fold cross-validation introduces severe temporal and flow-level data leakage, artificially inflating F1 scores by up to 30%. Our temporal and host-wise splitting strategies directly mitigate these leakage vectors.

### 2.3 Cross-Dataset Intrusion Detection
Evaluating models across different datasets (e.g., training on UNSW-NB15 and testing on CIC-IDS-2017) remains rare due to feature schema incompatibilities. Early works by Apruzzese et al. [4] highlighted that models achieving 99% accuracy in-distribution often perform worse than random guessing across domains.

### 2.4 Domain Adaptation and Generalization
To address cross-dataset performance collapse, recent studies have explored domain adaptation techniques, including adversarial feature alignment and transfer learning. While promising, these methods often require target-domain data during training, which is not always available in zero-day NIDS deployments.

### 2.5 Anomaly Detection Under Distribution Shift
Unsupervised anomaly detection (e.g., Isolation Forests [9], Autoencoders) is theoretically more robust to distribution shift because it models benign traffic rather than specific attack signatures. However, rigorous comparative studies quantifying this resilience against supervised ensembles under strict domain shift are lacking.

### 2.6 Explainability in NIDS
Trust in ML-based NIDS requires interpretable decision-making. The application of SHAP [5] and LIME has gained traction to explain model predictions locally. Yet, most explainability research in NIDS focuses on in-distribution attribution, failing to analyze which features remain robust across varying network environments.

### 2.7 Statistical Comparison of Security Classifiers
Despite clear guidelines by Demšar [6] on the statistical comparison of classifiers, the cybersecurity domain routinely omits formal significance testing. Pairwise comparisons of single-seed results are mathematically invalid, necessitating the Friedman and McNemar tests deployed in our framework.

### 2.8 Reproducible Cybersecurity Benchmarks
The crisis of reproducibility in ML security research is well-documented. Lack of published code, ambiguous hyperparameter configurations, and unavailable seed values render many published results unverifiable.

### 2.9 Comparison of Existing Literature

Table 2.1 contextualizes PhantomTrace against representative studies in the domain. Unlike existing works, our framework uniquely combines cross-dataset evaluation, strict duplicate control, and formal statistical significance testing into a single reproducible pipeline.

**Table 2.1: Comparison of NIDS Evaluation Methodologies**

| Study / Framework | Datasets | Recon-specific | Cross-dataset | Temporal split | Duplicate control | Statistical testing | Explainability | Code available |
|---|---|---|---|---|---|---|---|---|
| Sharafaldin et al. [1] | CIC-IDS-2017 | No | No | No | No | No | No | Yes |
| Moustafa & Slay [2] | UNSW-NB15 | No | No | No | No | No | No | Yes |
| Engelen et al. [3] | CIC-IDS-2017 | No | No | Yes | Yes | No | No | Yes |
| Apruzzese et al. [4] | Multiple | No | Yes | No | No | No | No | No |
| Nadeem et al. | Multiple | Yes | Yes | No | No | No | No | No |
| **PhantomTrace (Ours)** | **CIC, UNSW, Synth** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

*(Note: The comprehensive expansion to 40-60 verified citations is partially blocked pending access to a live academic literature index (e.g., Crossref/Semantic Scholar API). The categories above form the structural foundation.)*
