import collections
import time
from typing import Dict, Any, Optional, Deque

class PacketEvent:
    """
    Represents a single packet/flow event for rolling window calculations.
    """
    __slots__ = ['timestamp', 'bytes_len', 'protocol', 'dst_ip', 'dst_port', 'session_id', 'session_duration']
    
    def __init__(self, timestamp: float, bytes_len: int, protocol: str, dst_ip: Optional[str], dst_port: Optional[int], session_id: Any, session_duration: float = 0.0):
        self.timestamp = timestamp
        self.bytes_len = bytes_len
        self.protocol = protocol
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.session_id = session_id
        self.session_duration = session_duration

class HostBehavioralProfile:
    """
    Maintains the behavioral profile for a single source IP, including rolling windows.
    """
    def __init__(self, src_ip: str):
        self.src_ip = src_ip
        
        # Overall (unbounded) stats
        self.packet_count: int = 0
        self.bytes_transferred: int = 0
        self.protocol_usage = collections.Counter()
        self.destination_ips = set()
        self.destination_ports = set()
        self.sessions = set()
        self.session_count: int = 0

        # Rolling window state
        self.events: Deque[PacketEvent] = collections.deque()
        self.max_window = 3600  # 1 hour in seconds
        
        # Baseline state
        self.baseline_interval = 60.0  # Calculate baseline stats every 60 seconds
        self.last_baseline_time: Optional[float] = None
        self.baseline_samples: int = 0
        self.baselines = {
            "packet_rate": 0.0,
            "bytes_sec": 0.0,
            "destination_diversity": 0.0,
            "port_diversity": 0.0,
            "session_count": 0.0,
            "session_duration": 0.0,
            "protocol_usage": {}
        }

    def update(self, dst_ip: Optional[str], dst_port: Optional[int], protocol: str, bytes_len: int, session_id: Optional[str] = None, timestamp: Optional[float] = None, session_duration: float = 0.0):
        """
        Updates the profile with a new packet/flow record.
        """
        if timestamp is None:
            timestamp = time.time()

        # Update unbounded overall stats
        self.packet_count += 1
        self.bytes_transferred += bytes_len
        
        if protocol:
            self.protocol_usage[protocol] += 1
            
        if dst_ip:
            self.destination_ips.add(dst_ip)
            
        if dst_port is not None:
            self.destination_ports.add(dst_port)
            
        # Session tracking logic
        if not session_id:
            # Fallback for session if no explicit session_id is provided
            session_id = (self.src_ip, dst_ip, dst_port, protocol)
            
        if session_id not in self.sessions:
            self.sessions.add(session_id)
            self.session_count += 1

        # Add to rolling window events
        event = PacketEvent(timestamp, bytes_len, protocol, dst_ip, dst_port, session_id, session_duration)
        self.events.append(event)
        
        # Prune old events
        self._prune(timestamp)
        
        # Update baselines periodically
        self._update_baseline(timestamp)

    def _update_baseline(self, current_time: float):
        """
        Periodically calculates and updates the historical averages (baseline) for this host.
        """
        if self.last_baseline_time is None:
            self.last_baseline_time = current_time
            return
            
        elapsed = current_time - self.last_baseline_time
        if elapsed >= self.baseline_interval:
            # Calculate metrics over the elapsed interval
            stats = self.get_window_stats(current_time, int(elapsed))
            
            packet_rate = stats["packet_count"] / elapsed
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
                        self.baselines["protocol_usage"][proto] /= proto_sum
            
            # Update last baseline time
            self.last_baseline_time = current_time

    def _prune(self, current_time: float):
        """
        Removes events older than the maximum window (1 hour).
        """
        cutoff = current_time - self.max_window
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    def get_window_stats(self, current_time: float, window_seconds: int) -> Dict[str, Any]:
        """
        Calculates statistics for a specific time window looking backwards from current_time.
        """
        cutoff = current_time - window_seconds
        
        packet_count = 0
        bytes_transferred = 0
        protocols = collections.Counter()
        dst_ips = set()
        dst_ports = set()
        sessions = set()
        
        # Iterate backwards from newest to oldest event
        for event in reversed(self.events):
            if event.timestamp < cutoff:
                break
            packet_count += 1
            bytes_transferred += event.bytes_len
            if event.protocol:
                protocols[event.protocol] += 1
            if event.dst_ip:
                dst_ips.add(event.dst_ip)
            if event.dst_port is not None:
                dst_ports.add(event.dst_port)
            if event.session_id:
                sessions.add(event.session_id)
                
        session_duration_sum = 0.0
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
        }

    def calculate_drift(self, current_time: float) -> float:
        """
        Calculates a Behavioral Drift Score between 0.0 and 100.0 based on current 5m window vs baseline.
        """
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

    def generate_drift_explanation(self, current_time: float) -> list:
        """
        Produces analyst-friendly, plain-English explanations of how the host's
        current behavior deviates from its historical baseline.
        Returns a list of human-readable explanation strings.
        """
        explanations = []

        if self.baseline_samples < 5:
            return explanations

        current = self.get_window_stats(current_time, 300)
        elapsed = 300.0

        curr_packet_rate = current["packet_count"] / elapsed
        curr_dst_div = current["destination_ip_count"]
        curr_port_div = current["destination_port_count"]
        curr_sess_dur = current["mean_session_duration"]

        # --- Helper: describe a metric change in plain language ---
        def describe_change(metric_name: str, current_val: float, baseline_val: float, unit: str = ""):
            if baseline_val == 0:
                if current_val > 0:
                    explanations.append(
                        f"{metric_name} appeared for the first time (current: {current_val:.1f}{unit}, no prior baseline)."
                    )
                return
            ratio = current_val / baseline_val
            pct = abs(ratio - 1.0) * 100.0
            if pct < 20.0:
                return  # Within normal range, nothing to report

            if ratio > 1.0:
                if ratio >= 10.0:
                    explanations.append(f"{metric_name} surged by {ratio:.0f}x above baseline ({current_val:.1f}{unit} vs baseline {baseline_val:.1f}{unit}).")
                elif ratio >= 2.0:
                    explanations.append(f"{metric_name} increased by {pct:.0f}% ({current_val:.1f}{unit} vs baseline {baseline_val:.1f}{unit}).")
                else:
                    explanations.append(f"{metric_name} elevated by {pct:.0f}% above normal ({current_val:.1f}{unit} vs baseline {baseline_val:.1f}{unit}).")
            else:
                if ratio <= 0.1:
                    explanations.append(f"{metric_name} dropped to near zero ({current_val:.1f}{unit} vs baseline {baseline_val:.1f}{unit}).")
                else:
                    explanations.append(f"{metric_name} decreased by {pct:.0f}% ({current_val:.1f}{unit} vs baseline {baseline_val:.1f}{unit}).")

        # 1. Packet rate
        describe_change("Packet rate", curr_packet_rate, self.baselines["packet_rate"], " pkts/sec")

        # 2. Destination diversity
        describe_change("Destination diversity", curr_dst_div, self.baselines["destination_diversity"], " hosts")

        # 3. Port diversity
        describe_change("Port diversity", curr_port_div, self.baselines["port_diversity"], " ports")

        # 4. Session duration
        describe_change("Average session duration", curr_sess_dur, self.baselines["session_duration"], "s")

        # 5. Protocol mix shift
        curr_total_proto = sum(current["protocol_usage"].values())
        if curr_total_proto > 0 and self.baselines["protocol_usage"]:
            curr_proto_dist = {p: c / curr_total_proto for p, c in current["protocol_usage"].items()}
            all_protos = set(curr_proto_dist.keys()).union(set(self.baselines["protocol_usage"].keys()))
            tvd = 0.5 * sum(
                abs(curr_proto_dist.get(p, 0.0) - self.baselines["protocol_usage"].get(p, 0.0))
                for p in all_protos
            )
            if tvd > 0.15:
                # Build a short description of what changed
                new_protos = [p for p in curr_proto_dist if p not in self.baselines["protocol_usage"]]
                vanished = [p for p in self.baselines["protocol_usage"] if p not in curr_proto_dist]
                parts = []
                if new_protos:
                    parts.append(f"new protocol(s) {', '.join(new_protos)} appeared")
                if vanished:
                    parts.append(f"protocol(s) {', '.join(vanished)} disappeared")
                if not parts:
                    parts.append("protocol distribution shifted significantly")
                explanations.append(f"Protocol usage anomaly — {'; '.join(parts)}.")

        # 6. Overall summary sentence
        drift = self.calculate_drift(current_time)
        if drift > 60.0:
            explanations.append("Host behavior deviates significantly from its historical baseline.")
        elif drift > 30.0:
            explanations.append("Host behavior shows moderate deviation from its historical baseline.")

        return explanations

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the profile including rolling windows.
        """
        current_time = time.time()
        self._prune(current_time)
        
        return {
            "src_ip": self.src_ip,
            "packet_count": self.packet_count,
            "bytes_transferred": self.bytes_transferred,
            "protocol_usage": dict(self.protocol_usage),
            "destination_ip_count": len(self.destination_ips),
            "destination_port_count": len(self.destination_ports),
            "session_count": self.session_count,
            
            # Rolling window statistics
            "windows": {
                "1m": self.get_window_stats(current_time, 60),
                "5m": self.get_window_stats(current_time, 300),
                "15m": self.get_window_stats(current_time, 900),
                "30m": self.get_window_stats(current_time, 1800),
                "1h": self.get_window_stats(current_time, 3600)
            },
            
            # Baseline statistics
            "baseline_samples": self.baseline_samples,
            "baselines": self.baselines,
            "drift_score": self.calculate_drift(current_time)
        }


class BehavioralProfileEngine:
    """
    Engine to manage behavioral profiles for multiple source IPs.
    """
    def __init__(self):
        self._profiles: Dict[str, HostBehavioralProfile] = {}

    def get_or_create_profile(self, src_ip: str) -> HostBehavioralProfile:
        """
        Retrieves the profile for a given source IP, creating one if it doesn't exist.
        """
        if src_ip not in self._profiles:
            self._profiles[src_ip] = HostBehavioralProfile(src_ip)
        return self._profiles[src_ip]

    def update_profile(self, src_ip: str, dst_ip: Optional[str] = None, dst_port: Optional[int] = None, 
                       protocol: str = "UNKNOWN", bytes_len: int = 0, session_id: Optional[str] = None,
                       timestamp: Optional[float] = None, session_duration: float = 0.0):
        """
        Updates the behavioral profile for a given source IP.
        
        :param src_ip: The source IP address.
        :param dst_ip: The destination IP address.
        :param dst_port: The destination port.
        :param protocol: The protocol used (e.g., 'TCP', 'UDP').
        :param bytes_len: Number of bytes transferred in the packet/flow.
        :param session_id: Optional unique identifier for the session.
        :param timestamp: Optional timestamp of the packet (defaults to current time).
        :param session_duration: Optional duration of the session for tracking session length drift.
        """
        profile = self.get_or_create_profile(src_ip)
        profile.update(
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            bytes_len=bytes_len,
            session_id=session_id,
            timestamp=timestamp,
            session_duration=session_duration
        )

    def retrieve_profile(self, src_ip: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the profile statistics for a given source IP.
        """
        if src_ip in self._profiles:
            return self._profiles[src_ip].to_dict()
        return None

    def retrieve_drift_explanation(self, src_ip: str, current_time: Optional[float] = None) -> list:
        """
        Returns a list of analyst-friendly explanation strings describing how
        the given host's current behavior deviates from its historical baseline.
        """
        if src_ip not in self._profiles:
            return []
        if current_time is None:
            current_time = time.time()
        return self._profiles[src_ip].generate_drift_explanation(current_time)

    def retrieve_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all profiles managed by the engine.
        """
        return {ip: profile.to_dict() for ip, profile in self._profiles.items()}
