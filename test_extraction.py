import sys
import pandas as pd
from feature_extraction.extractor import FlowSession, FeatureExtractor
from pcap_processing.parser import PacketRecord

# Create some dummy packets
p1 = PacketRecord(timestamp=1.0, src_ip="192.168.1.1", dst_ip="10.0.0.1", src_port=1234, dst_port=80, proto=6, payload_len=100, flags="S")
p2 = PacketRecord(timestamp=1.1, src_ip="10.0.0.1", dst_ip="192.168.1.1", src_port=80, dst_port=1234, proto=6, payload_len=200, flags="SA")
p3 = PacketRecord(timestamp=1.2, src_ip="192.168.1.1", dst_ip="10.0.0.1", src_port=1234, dst_port=80, proto=6, payload_len=100, flags="A")

flow = FlowSession(flow_key=("192.168.1.1", "10.0.0.1", 1234, 80, 6), src_ip="192.168.1.1", dst_ip="10.0.0.1", src_port=1234, dst_port=80, proto=6)
flow.add_packet(p1)
flow.add_packet(p2)
flow.add_packet(p3)

extractor = FeatureExtractor()
df = extractor.extract_features([flow])

print(f"Extracted {len(df.columns)} features. Shape: {df.shape}")
expected_columns = [
    "flow_duration", "flow_packet_count", "flow_bytes", "flow_packets_per_sec", "flow_bytes_per_sec", 
    "flow_syn_ratio", "flow_ack_ratio", "flow_rst_ratio", "flow_fin_ratio", "flow_syn_ack_ratio", 
    "flow_tcp_syn", "flow_tcp_ack", "flow_tcp_rst", "flow_tcp_fin", "flow_tcp_psh",
    "host_port_entropy", "host_dst_entropy", "host_dst_diversity", "host_port_diversity",
    "host_proto_entropy", "host_syn_ratio", "host_dns_ratio", "host_icmp_ratio", 
    "host_failed_flow_ratio", "host_packet_rate", "host_interval_mean", "host_interval_var", 
    "host_burstiness", "host_session_duration", "host_packet_size_var"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    print("MISSING:", missing)
else:
    print("All expected columns are present.")

# Now test engine compatibility
from ml_engine.engine import MLInferenceEngine
engine = MLInferenceEngine()
raw_features = df.iloc[0].to_dict()
pred, conf = engine.predict(raw_features)
print(f"Prediction: {pred}, Conf: {conf}")

