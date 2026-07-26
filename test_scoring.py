from threat_analysis.analyzer import ThreatAnalyzer
from feature_extraction.extractor import FeatureExtractor, StaticFlowTracker
from pcap_processing.parser import PCAPParser
import os
import time

def create_mock_pcap_file():
    from scapy.all import IP, TCP, wrpcap
    pkts = []
    base_time = time.time()
    for i in range(12):
        sport = 49100 + i
        pkt_syn = IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="S")
        pkt_syn.time = base_time + i * 0.15
        pkts.append(pkt_syn)
        pkt_synack = IP(src="8.8.8.8", dst="192.168.1.10")/TCP(sport=80, dport=sport, flags="SA")
        pkt_synack.time = base_time + i * 0.15 + 0.02
        pkts.append(pkt_synack)
        pkt_ack = IP(src="192.168.1.10", dst="8.8.8.8")/TCP(sport=sport, dport=80, flags="A")
        pkt_ack.time = base_time + i * 0.15 + 0.04
        pkts.append(pkt_ack)

    for i in range(25):
        target_port = 20 + i
        pkt = IP(src="192.168.1.187", dst="192.168.1.5")/TCP(sport=38200 + i, dport=target_port, flags="S")
        pkt.time = base_time + 4.0 + i * 0.4
        pkts.append(pkt)

    out_path = "pcaps/synthetic_scan.pcap"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wrpcap(out_path, pkts)

create_mock_pcap_file()

packets, meta = PCAPParser.load_pcap("pcaps/synthetic_scan.pcap")
flows = StaticFlowTracker.track_flows(packets)
features = FeatureExtractor().extract_features(flows)
    
analyzer = ThreatAnalyzer(model_name="random_forest")
reports = analyzer.analyze_flows(flows, features)

for r in reports:
    print(f"[{r.severity}] {r.scan_category} from {r.src_ip} - Threat Score: {r.threat_score:.1f}")
    for ev in r.evidence:
        print(f"  - {ev}")
