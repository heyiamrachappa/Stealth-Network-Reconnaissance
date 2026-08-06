#!/usr/bin/env python3
"""
Phase 5 — Statistical Validation (Friedman test + Wilcoxon post-hoc)

For each split strategy, compare all 5 models across the 5 seeds using:
  1. Friedman test (non-parametric repeated-measures ANOVA)
  2. Wilcoxon signed-rank post-hoc pairwise tests (Bonferroni-corrected)

Outputs:
  results/metrics/statistical_tests.json
  results/metrics/aggregate_summary.json  (mean ± std per model per strategy)
"""

import json, glob, os
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"


class NpEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------------------
# Load all per-seed metrics
# ---------------------------------------------------------------------------
def load_all():
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fp in glob.glob(str(METRICS_DIR / "*_metrics.json")):
        if "statistical" in fp or "aggregate" in fp:
            continue
        with open(fp) as f:
            d = json.load(f)
        strat = d["strategy"]
        model = d["model"]
        for metric, val in d["test_metrics"].items():
            if isinstance(val, (int, float)) and val is not None:
                data[strat][model][metric].append(val)
    return data

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def run_tests(data):
    results = {}
    aggregate = {}
    for strat, models in sorted(data.items()):
        results[strat] = {}
        aggregate[strat] = {}
        model_names = sorted(models.keys())

        # Per-model aggregate
        for model in model_names:
            mets = models[model]
            aggregate[strat][model] = {
                m: {
                    "mean": round(np.mean(vals), 4),
                    "std": round(np.std(vals, ddof=1), 4) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals)
                }
                for m, vals in mets.items()
            }

        # Friedman test per metric
        key_metrics = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]
        for metric in key_metrics:
            vectors = []
            valid_models = []
            for model in model_names:
                vals = models[model].get(metric, [])
                if len(vals) >= 3:  # need at least 3 observations
                    vectors.append(vals[:5])  # take first 5 seeds
                    valid_models.append(model)

            if len(vectors) < 3:
                results[strat][metric] = {"note": "Too few models with enough seeds for Friedman"}
                continue

            # Pad to same length
            min_len = min(len(v) for v in vectors)
            vectors = [v[:min_len] for v in vectors]

            try:
                stat, p_value = stats.friedmanchisquare(*vectors)
                friedman = {
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 6),
                    "significant_005": p_value < 0.05,
                    "models_compared": valid_models
                }
            except Exception as e:
                friedman = {"error": str(e)}

            # Wilcoxon post-hoc (pairwise)
            posthoc = {}
            n_comparisons = len(list(combinations(range(len(valid_models)), 2)))
            bonferroni = n_comparisons if n_comparisons > 0 else 1
            for i, j in combinations(range(len(valid_models)), 2):
                pair = f"{valid_models[i]} vs {valid_models[j]}"
                try:
                    w_stat, w_p = stats.wilcoxon(vectors[i], vectors[j])
                    adjusted_p = min(float(w_p) * bonferroni, 1.0)
                    posthoc[pair] = {
                        "statistic": round(float(w_stat), 4),
                        "p_value_raw": round(float(w_p), 6),
                        "p_value_bonferroni": round(adjusted_p, 6),
                        "significant_005": adjusted_p < 0.05
                    }
                except Exception as e:
                    posthoc[pair] = {"error": str(e)}

            results[strat][metric] = {
                "friedman": friedman,
                "wilcoxon_posthoc": posthoc
            }

    return results, aggregate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_all()
    results, aggregate = run_tests(data)

    out_stat = METRICS_DIR / "statistical_tests.json"
    with open(out_stat, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"Statistical tests → {out_stat}")

    out_agg = METRICS_DIR / "aggregate_summary.json"
    with open(out_agg, "w") as f:
        json.dump(aggregate, f, indent=2, cls=NpEncoder)
    print(f"Aggregate summary → {out_agg}")

    # Print key findings
    for strat in sorted(results.keys()):
        print(f"\n=== {strat} ===")
        for metric in ["f1", "mcc", "roc_auc"]:
            if metric in results[strat] and "friedman" in results[strat][metric]:
                fr = results[strat][metric]["friedman"]
                sig = "YES" if fr.get("significant_005") else "no"
                print(f"  {metric}: Friedman χ²={fr.get('statistic','N/A')}, "
                      f"p={fr.get('p_value','N/A')}, significant={sig}")
