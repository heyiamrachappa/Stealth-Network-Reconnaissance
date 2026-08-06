# PhantomTrace

A reproducible cross-dataset and temporal evaluation framework for machine learning-based network reconnaissance detection.

## Quickstart Reproduction

To completely reproduce the results reported in the PhantomTrace manuscript (including dataset downloading, preprocessing, training 100 models, and running statistical tests):

```bash
# 1. Clone the repository
git clone https://github.com/PhantomTrace-Project/Stealth-Network-Reconnaissance.git
cd Stealth-Network-Reconnaissance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline (Downloads datasets, trains models, evaluates metrics)
bash run_evals.sh
```

All figures, statistical test results (`report.json`), and raw metrics will be generated in the `results/` directory. Serialized model weights will be saved to `models/v2/`.
