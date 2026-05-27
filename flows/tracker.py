#!/usr/bin/env python3
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from capture.sniffer import PacketRecord
from utils.helpers import setup_logger, load_config

logger = setup_logger("FlowTracker")

@dataclass
class FlowSession:
    """
    Tracks state and aggregates rolling packet-level stats for a single network flow.
    """
    flow_key: Tuple[str, str, int, int, int]  # Canonical: (IP_low, IP_high, port_low, port_high, proto)
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    
    packets: List[PacketRecord] = field(default_factory=list)
    forward_packets: int = 0
    backward_packets: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0
    
    syn_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    psh_count: int = 0
    
    start_time: float = 0.0
    end_time: float = 0.0

    def add_packet(self, pkt: PacketRecord) -> None:
        """
        Updates internal flow counters and lists with a parsed packet record.
        """
        if not self.packets:
            self.start_time = pkt.timestamp
            
        self.packets.append(pkt)
        self.end_time = pkt.timestamp
        
        # Check direction of connection flow
        is_forward = (pkt.src_ip == self.src_ip and pkt.src_port == self.src_port)
        
        if is_forward:
            self.forward_packets += 1
            self.forward_bytes += pkt.payload_len
        else:
            self.backward_packets += 1
            self.backward_bytes += pkt.payload_len
            
        # Parse TCP flags
        if pkt.proto == 6:  # TCP
            flags = pkt.flags.upper()
            if "S" in flags:
                self.syn_count += 1
            if "A" in flags:
                self.ack_count += 1
            if "R" in flags:
                self.rst_count += 1
            if "F" in flags:
                self.fin_count += 1
            if "P" in flags:
                self.psh_count += 1

    @property
    def duration(self) -> float:
        """
        Flow lifetime duration in seconds.
        """
        return max(0.0, self.end_time - self.start_time)


class FlowTracker:
    """
    Maintains thread-safe active flows in memory, maps bidirectional packet traffic,
    and automatically manages stale session timeouts using a daemon background thread.
    """
    def __init__(self, flow_timeout_seconds: Optional[int] = None):
        self.config = load_config()
        self.flow_timeout = flow_timeout_seconds if flow_timeout_seconds is not None else self.config.get("features", {}).get("flow_timeout_seconds", 60)
        
        self.active_sessions: Dict[Tuple[str, str, int, int, int], FlowSession] = {}
        self.lock = threading.Lock()
        
        # Start background pruner thread
        self.running = True
        self.pruner_thread = threading.Thread(target=self._pruning_loop, daemon=True)
        self.pruner_thread.start()

    @staticmethod
    def get_canonical_key(src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: int) -> Tuple[str, str, int, int, int]:
        """
        Standardizes the connection key so bidirectional packets map to the same session.
        """
        if (src_ip < dst_ip) or (src_ip == dst_ip and src_port <= dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    def handle_packet(self, pkt: PacketRecord) -> FlowSession:
        """
        Places a parsed packet into its canonical bidirectional flow session.
        """
        key = self.get_canonical_key(pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.proto)
        
        with self.lock:
            if key not in self.active_sessions:
                # Store the original packet's source and destination as the flow's primary direction
                self.active_sessions[key] = FlowSession(
                    flow_key=key,
                    src_ip=pkt.src_ip,
                    dst_ip=pkt.dst_ip,
                    src_port=pkt.src_port,
                    dst_port=pkt.dst_port,
                    proto=pkt.proto
                )
            
            session = self.active_sessions[key]
            session.add_packet(pkt)
            return session

    def get_active_sessions(self) -> List[FlowSession]:
        """
        Returns a snapshot copy of all active flows.
        """
        with self.lock:
            return list(self.active_sessions.values())

    def _pruning_loop(self) -> None:
        """
        Prunes timed-out/expired flows periodically to control memory consumption.
        """
        logger.info("Session pruner thread started.")
        while self.running:
            time.sleep(5.0)  # Prune every 5 seconds
            
            current_time = time.time()
            expired_keys = []
            
            with self.lock:
                for key, session in self.active_sessions.items():
                    # If flow has had no new packet for longer than the timeout threshold
                    if (current_time - session.end_time) > self.flow_timeout:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.active_sessions[key]
            
            if expired_keys:
                logger.debug(f"Session pruner expired {len(expired_keys)} idle network flows.")

    def stop(self) -> None:
        """
        Stops the pruner thread.
        """
        self.running = False
