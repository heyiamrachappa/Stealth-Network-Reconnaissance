#!/usr/bin/env python3
# ==============================================================================
# Phase 3 - PCAP Parsing & Connection Tracking Module
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Iterator, Optional
from scapy.all import PcapReader, Packet, IP, TCP, UDP

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("PCAPParser")


class PacketRecord:
    """
    Lightweight, parsed representations of essential packet headers.
    Avoids keeping heavy Scapy packet objects in memory for feature extraction.
    """
    def __init__(self, 
                 timestamp: float, 
                 src_ip: str, 
                 dst_ip: str, 
                 src_port: int, 
                 dst_port: int, 
                 proto: int, 
                 payload_len: int, 
                 flags: str = ""):
        self.timestamp = timestamp
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto  # 6 = TCP, 17 = UDP
        self.payload_len = payload_len
        self.flags = flags


class FlowRecord:
    """
    Tracks and aggregates packet statistics for a bidirectional connection flow.
    """
    def __init__(self, flow_key: Tuple[str, str, int, int, int]):
        self.flow_key = flow_key  # Canonical: (low_ip, high_ip, low_port, high_port, proto)
        self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto = flow_key
        
        self.packets: List[PacketRecord] = []
        self.forward_packets = 0
        self.backward_packets = 0
        self.forward_bytes = 0
        self.backward_bytes = 0
        
        self.syn_count = 0
        self.ack_count = 0
        self.rst_count = 0
        self.fin_count = 0
        self.psh_count = 0
        
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def add_packet(self, pkt: PacketRecord) -> None:
        if not self.packets:
            self.start_time = pkt.timestamp
            
        self.packets.append(pkt)
        self.end_time = pkt.timestamp
        
        # Check direction (compare with canonical key setup)
        # Standard flow key is ordered by IPs, let's check actual direction
        is_forward = (pkt.src_ip == self.src_ip and pkt.src_port == self.src_port)
        
        if is_forward:
            self.forward_packets += 1
            self.forward_bytes += pkt.payload_len
        else:
            self.backward_packets += 1
            self.backward_bytes += pkt.payload_len
            
        # Extract TCP flags statistics
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
        return max(0.0, self.end_time - self.start_time)


class PCAPParser:
    """
    Parses PCAPs or live packet lists and aggregates them into bidirectional flows.
    """
    @staticmethod
    def get_canonical_key(src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: int) -> Tuple[str, str, int, int, int]:
        """
        Generates a canonical key to represent bidirectional flows.
        """
        if (src_ip < dst_ip) or (src_ip == dst_ip and src_port <= dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    @classmethod
    def parse_packet(cls, packet: Packet) -> Optional[PacketRecord]:
        """
        Parses a single Scapy packet and extracts clean fields.
        """
        try:
            if not packet.haslayer(IP):
                return None
                
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto = ip_layer.proto
            
            payload_len = len(ip_layer.payload)
            timestamp = float(packet.time)
            
            src_port = 0
            dst_port = 0
            flags = ""
            
            if proto == 6 and packet.haslayer(TCP):  # TCP
                tcp_layer = packet[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                flags = tcp_layer.sprintf("%TCP.flags%")
            elif proto == 17 and packet.haslayer(UDP):  # UDP
                udp_layer = packet[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                
            return PacketRecord(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                proto=proto,
                payload_len=payload_len,
                flags=flags
            )
        except Exception as e:
            logger.debug(f"Failed to parse packet layer: {e}")
            return None

    def parse_pcap_generator(self, pcap_path: str) -> Iterator[PacketRecord]:
        """
        Memory-efficient PCAP generator. Yields parsed PacketRecord structures.
        """
        if not os.path.exists(pcap_path):
            logger.error(f"PCAP file not found: {pcap_path}")
            return
            
        try:
            with PcapReader(pcap_path) as reader:
                for i, pkt in enumerate(reader):
                    parsed = self.parse_packet(pkt)
                    if parsed:
                        yield parsed
        except Exception as e:
            logger.error(f"Error reading PCAP {pcap_path}: {e}")

    def aggregate_flows(self, pcap_path: str) -> Dict[Tuple[str, str, int, int, int], FlowRecord]:
        """
        Parses a PCAP file and aggregates packets into bidirectional flows.
        """
        flows: Dict[Tuple[str, str, int, int, int], FlowRecord] = {}
        packet_count = 0
        
        logger.info(f"Aggregating flows from PCAP: {pcap_path}...")
        
        for pkt in self.parse_pcap_generator(pcap_path):
            packet_count += 1
            key = self.get_canonical_key(pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.proto)
            
            if key not in flows:
                flows[key] = FlowRecord(key)
                
            flows[key].add_packet(pkt)
            
            if packet_count % 10000 == 0:
                logger.info(f"Processed {packet_count} packets...")
                
        logger.info(f"Flow aggregation complete. Total packets processed: {packet_count}, Unique flows: {len(flows)}")
        return flows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/parser.py <path_to_pcap>")
        sys.exit(1)
        
    pcap_file = sys.argv[1]
    parser = PCAPParser()
    try:
        flows = parser.aggregate_flows(pcap_file)
        # Print a small summary of top 5 flows by packet count
        top_flows = sorted(flows.values(), key=lambda f: len(f.packets), reverse=True)[:5]
        print("\n--- TOP FLOWS SUMMARY ---")
        for f in top_flows:
            proto_str = "TCP" if f.proto == 6 else "UDP" if f.proto == 17 else str(f.proto)
            print(f"Flow: {f.src_ip}:{f.src_port} <-> {f.dst_ip}:{f.dst_port} ({proto_str})")
            print(f"  Packets: {len(f.packets)} (Fwd: {f.forward_packets}, Bwd: {f.backward_packets})")
            print(f"  TCP Flags: SYN={f.syn_count}, ACK={f.ack_count}, RST={f.rst_count}, FIN={f.fin_count}")
            print(f"  Duration: {f.duration:.4f} seconds\n")
    except Exception as e:
        logger.error(f"Failed to parse target PCAP: {e}")
