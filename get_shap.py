import joblib, shap, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
train = pd.read_parquet("dataset/splits/cross_dataset/train.parquet")
test = pd.read_parquet("dataset/splits/cross_dataset/test.parquet")
features = ["flow_duration","flow_packet_count","flow_bytes","flow_syn_ratio","flow_ack_ratio","flow_rst_ratio","flow_fin_ratio","flow_size_mean","flow_size_var","flow_interval_mean","flow_interval_var","host_port_entropy","host_dst_entropy","host_dst_diversity","host_syn_ratio","host_failed_flow_ratio","host_packet_rate","host_interval_mean","host_interval_var","host_packet_size_var"]
scaler = StandardScaler().fit(train[features].values)
X_bg = shap.sample(scaler.transform(train[features].values), 100, random_state=42)
X_ex = scaler.transform(shap.sample(test[features].values, 100, random_state=42))
model = joblib.load("models/v2/cross_dataset/mlp_seed42.joblib")
exp = shap.KernelExplainer(model.predict_proba, X_bg)
sv = exp.shap_values(X_ex, nsamples=100)
sv1 = sv[1] if isinstance(sv, list) else (sv[:,:,1] if len(sv.shape)>2 else sv)
mean_abs = np.mean(np.abs(sv1), axis=0)
sorted_idx = np.argsort(mean_abs)[::-1]
for i in range(5):
    print(f"{features[sorted_idx[i]]}: {mean_abs[sorted_idx[i]]:.4f}")
