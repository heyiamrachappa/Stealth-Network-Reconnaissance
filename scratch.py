import json, glob
for f in sorted(glob.glob("results/metrics/host_wise_random_forest_seed*_metrics.json")):
    with open(f) as fp: d = json.load(fp)
    print(f, "AUC:", d["test_metrics"]["roc_auc"], "CM:", d["test_metrics"]["confusion_matrix"])
