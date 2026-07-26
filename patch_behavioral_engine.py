import re

with open('threat_analysis/behavioral_profile_engine.py', 'r') as f:
    content = f.read()

# 1. Update PacketEvent
content = content.replace(
    "__slots__ = ['timestamp', 'bytes_len', 'protocol', 'dst_ip', 'dst_port', 'session_id']",
    "__slots__ = ['timestamp', 'bytes_len', 'protocol', 'dst_ip', 'dst_port', 'session_id', 'session_duration']"
)
content = content.replace(
    "def __init__(self, timestamp: float, bytes_len: int, protocol: str, dst_ip: Optional[str], dst_port: Optional[int], session_id: Any):",
    "def __init__(self, timestamp: float, bytes_len: int, protocol: str, dst_ip: Optional[str], dst_port: Optional[int], session_id: Any, session_duration: float = 0.0):"
)
content = content.replace(
    "self.session_id = session_id",
    "self.session_id = session_id\n        self.session_duration = session_duration"
)

# 2. Add to baselines
content = content.replace(
    '"session_count": 0.0',
    '"session_count": 0.0,\n            "session_duration": 0.0,\n            "protocol_usage": {}'
)

# 3. Update method signature
content = content.replace(
    "session_id: Optional[str] = None, timestamp: Optional[float] = None",
    "session_id: Optional[str] = None, timestamp: Optional[float] = None, session_duration: float = 0.0"
)

# 4. Update event append
content = content.replace(
    "event = PacketEvent(timestamp, bytes_len, protocol, dst_ip, dst_port, session_id)",
    "event = PacketEvent(timestamp, bytes_len, protocol, dst_ip, dst_port, session_id, session_duration)"
)

# 5. Update _update_baseline
# Need to replace the inside of _update_baseline to track session duration and protocol usage
old_baseline_update = """            packet_rate = stats["packet_count"] / elapsed
            bytes_sec = stats["bytes_transferred"] / elapsed
            dst_diversity = stats["destination_ip_count"]
            port_diversity = stats["destination_port_count"]
            session_count = stats["session_count"]
            
            # Incorporate into running average (Cumulative Moving Average)
            self.baseline_samples += 1
            
            # We limit the divisor to prevent the baseline from becoming completely rigid
            # over a very long time, essentially giving it a slight EMA characteristic
            # after 1000 samples.
            alpha = 1.0 / min(self.baseline_samples, 1000)
            
            self.baselines["packet_rate"] += alpha * (packet_rate - self.baselines["packet_rate"])
            self.baselines["bytes_sec"] += alpha * (bytes_sec - self.baselines["bytes_sec"])
            self.baselines["destination_diversity"] += alpha * (dst_diversity - self.baselines["destination_diversity"])
            self.baselines["port_diversity"] += alpha * (port_diversity - self.baselines["port_diversity"])
            self.baselines["session_count"] += alpha * (session_count - self.baselines["session_count"])"""

new_baseline_update = """            packet_rate = stats["packet_count"] / elapsed
            bytes_sec = stats["bytes_transferred"] / elapsed
            dst_diversity = stats["destination_ip_count"]
            port_diversity = stats["destination_port_count"]
            session_count = stats["session_count"]
            session_duration = stats.get("mean_session_duration", 0.0)
            
            self.baseline_samples += 1
            alpha = 1.0 / min(self.baseline_samples, 1000)
            
            self.baselines["packet_rate"] += alpha * (packet_rate - self.baselines["packet_rate"])
            self.baselines["bytes_sec"] += alpha * (bytes_sec - self.baselines["bytes_sec"])
            self.baselines["destination_diversity"] += alpha * (dst_diversity - self.baselines["destination_diversity"])
            self.baselines["port_diversity"] += alpha * (port_diversity - self.baselines["port_diversity"])
            self.baselines["session_count"] += alpha * (session_count - self.baselines["session_count"])
            self.baselines["session_duration"] += alpha * (session_duration - self.baselines["session_duration"])
            
            # Update protocol baseline distribution
            total_protocols = sum(stats["protocol_usage"].values())
            if total_protocols > 0:
                for proto, count in stats["protocol_usage"].items():
                    current_ratio = count / total_protocols
                    existing_ratio = self.baselines["protocol_usage"].get(proto, 0.0)
                    self.baselines["protocol_usage"][proto] = existing_ratio + alpha * (current_ratio - existing_ratio)
                
                # Normalize just in case
                proto_sum = sum(self.baselines["protocol_usage"].values())
                if proto_sum > 0:
                    for proto in self.baselines["protocol_usage"]:
                        self.baselines["protocol_usage"][proto] /= proto_sum"""

content = content.replace(old_baseline_update, new_baseline_update)

# 6. Update get_window_stats to include mean_session_duration
old_window_stats_return = """        return {
            "packet_count": packet_count,
            "bytes_transferred": bytes_transferred,
            "protocol_usage": dict(protocols),
            "destination_ip_count": len(dst_ips),
            "destination_port_count": len(dst_ports),
            "session_count": len(sessions)
        }"""

new_window_stats_return = """        session_duration_sum = 0.0
        session_duration_count = 0
        for event in reversed(self.events):
            if event.timestamp < cutoff:
                break
            if event.session_duration > 0:
                session_duration_sum += event.session_duration
                session_duration_count += 1
                
        mean_duration = (session_duration_sum / session_duration_count) if session_duration_count > 0 else 0.0
        
        return {
            "packet_count": packet_count,
            "bytes_transferred": bytes_transferred,
            "protocol_usage": dict(protocols),
            "destination_ip_count": len(dst_ips),
            "destination_port_count": len(dst_ports),
            "session_count": len(sessions),
            "mean_session_duration": mean_duration
        }"""

content = content.replace(old_window_stats_return, new_window_stats_return)

# 7. Add calculate_drift method
drift_logic = """    def calculate_drift(self, current_time: float) -> float:
        \"\"\"
        Calculates a Behavioral Drift Score between 0.0 and 100.0 based on current 5m window vs baseline.
        \"\"\"
        if self.baseline_samples < 5:  # Not enough baseline data
            return 0.0
            
        current = self.get_window_stats(current_time, 300) # 5-minute window for current behavior
        elapsed = 300.0
        
        # Calculate current rates
        curr_packet_rate = current["packet_count"] / elapsed
        curr_dst_div = current["destination_ip_count"]
        curr_port_div = current["destination_port_count"]
        curr_sess_dur = current["mean_session_duration"]
        
        # Helper to compute normalized difference bounded to 1.0
        def norm_diff(curr, base):
            if base == 0:
                return 1.0 if curr > 0 else 0.0
            diff = abs(curr - base) / base
            return min(1.0, diff)
            
        drift_packet_rate = norm_diff(curr_packet_rate, self.baselines["packet_rate"])
        drift_dst_div = norm_diff(curr_dst_div, self.baselines["destination_diversity"])
        drift_port_div = norm_diff(curr_port_div, self.baselines["port_diversity"])
        drift_sess_dur = norm_diff(curr_sess_dur, self.baselines["session_duration"])
        
        # Protocol drift using Total Variation Distance
        curr_total_proto = sum(current["protocol_usage"].values())
        curr_proto_dist = {}
        if curr_total_proto > 0:
            for p, c in current["protocol_usage"].items():
                curr_proto_dist[p] = c / curr_total_proto
                
        all_protos = set(curr_proto_dist.keys()).union(set(self.baselines["protocol_usage"].keys()))
        tvd = 0.5 * sum(abs(curr_proto_dist.get(p, 0.0) - self.baselines["protocol_usage"].get(p, 0.0)) for p in all_protos)
        drift_protocol = min(1.0, tvd)
        
        # Weighted aggregate
        # Packet rate (20%), Dst Div (25%), Port Div (25%), Protocol (15%), Session Dur (15%)
        score = (
            drift_packet_rate * 20.0 +
            drift_dst_div * 25.0 +
            drift_port_div * 25.0 +
            drift_protocol * 15.0 +
            drift_sess_dur * 15.0
        )
        
        return float(min(100.0, score))

    def to_dict"""

content = content.replace("    def to_dict", drift_logic)

# 8. Add drift_score to to_dict
to_dict_add = """            "baselines": self.baselines,
            "drift_score": self.calculate_drift(current_time)
        }"""
content = content.replace('            "baselines": self.baselines\n        }', to_dict_add)


with open('threat_analysis/behavioral_profile_engine.py', 'w') as f:
    f.write(content)

