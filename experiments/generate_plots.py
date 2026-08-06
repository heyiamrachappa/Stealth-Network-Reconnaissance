#!/usr/bin/env python3
"""
Phase 4 — Generate ROC Curves & Comparison Plots

Produces per-strategy ROC curve comparison plots and a consolidated
performance heatmap across all strategies.

Outputs:
  results/metrics/roc_<strategy>.png
  results/metrics/performance_heatmap.png
"""

import json, glob, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------
def load_all():
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fp in glob.glob(str(METRICS_DIR / "*_metrics.json")):
        bn = os.path.basename(fp)
        if "statistical" in bn or "aggregate" in bn:
            continue
        with open(fp) as f:
            d = json.load(f)
        strat = d["strategy"]
        model = d["model"]
        for metric, val in d["test_metrics"].items():
            if isinstance(val, (int, float)) and val is not None:
                data[strat][model][metric].append(val)
    return data

data = load_all()

# ---------------------------------------------------------------------------
# 1. Bar charts: mean F1 per model per strategy
# ---------------------------------------------------------------------------
strategies = ["pcap_wise", "time_wise", "host_wise", "cross_dataset"]
model_order = ["random_forest", "xgboost", "svm", "isolation_forest", "mlp"]
model_labels = {"random_forest": "RF", "xgboost": "XGB", "svm": "SVM",
                "isolation_forest": "IF", "mlp": "MLP"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Test F1 Score by Model and Split Strategy (mean ± std, 5 seeds)",
             fontsize=14, fontweight="bold")
colors = sns.color_palette("viridis", len(model_order))

for idx, strat in enumerate(strategies):
    ax = axes[idx // 2][idx % 2]
    means, stds, labels = [], [], []
    for m in model_order:
        vals = data[strat][m].get("f1", [0])
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        labels.append(model_labels[m])
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(strat.replace("_", " ").title(), fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    # value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
f1_path = METRICS_DIR / "f1_comparison_barplot.png"
plt.savefig(f1_path, dpi=150)
plt.close()
print(f"Saved F1 bar chart → {f1_path}")

# ---------------------------------------------------------------------------
# 2. Performance heatmap (F1, MCC, ROC-AUC across strategies)
# ---------------------------------------------------------------------------
key_metrics = ["f1", "mcc", "roc_auc"]
metric_labels = {"f1": "F1", "mcc": "MCC", "roc_auc": "ROC-AUC"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Performance Heatmap Across Split Strategies (mean, 5 seeds)",
             fontsize=14, fontweight="bold")

for i, metric in enumerate(key_metrics):
    ax = axes[i]
    matrix = []
    for strat in strategies:
        row = []
        for m in model_order:
            vals = data[strat][m].get(metric, [0])
            row.append(np.mean(vals))
        matrix.append(row)
    matrix = np.array(matrix)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[model_labels[m] for m in model_order],
                yticklabels=[s.replace("_", "\n") for s in strategies],
                ax=ax, vmin=0, vmax=1)
    ax.set_title(metric_labels[metric], fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.92])
heatmap_path = METRICS_DIR / "performance_heatmap.png"
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved heatmap → {heatmap_path}")

# ---------------------------------------------------------------------------
# 3. Confusion matrix grid for time_wise (best in-distribution strategy)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Confusion Matrices — Time-Wise Split (seed 42)", fontsize=14, fontweight="bold")

for i, m in enumerate(model_order):
    ax = axes[i]
    fp = METRICS_DIR / f"time_wise_{m}_seed42_metrics.json"
    if fp.exists():
        with open(fp) as f:
            d = json.load(f)
        cm = np.array(d["test_metrics"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Recon"], yticklabels=["Benign", "Recon"])
        ax.set_title(model_labels[m], fontsize=11)
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Actual")
    else:
        ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
cm_path = METRICS_DIR / "confusion_matrices_timewise.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved confusion matrices → {cm_path}")

print("\nAll plots generated.")
