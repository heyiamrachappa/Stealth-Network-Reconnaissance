#!/usr/bin/env python3
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from scapy.all import PcapReader, IP, TCP, UDP, Packet
from utils.helpers import setup_logger

logger = setup_logger("PCAPParser")

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

class PCAPParser:
    """
    Statically processes offline PCAP and PCAPNG files.
    Reads packet records and compiles forensic metadata.
    """
    @staticmethod
    def parse_scapy_packet(packet: Packet) -> Optional[PacketRecord]:
        """
        Parses Scapy layers into lightweight PacketRecord structures.
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
            logger.debug(f"Failed parsing scapy packet layer: {e}")
            return None

    @classmethod
    def load_pcap(cls, pcap_path: str) -> Tuple[List[PacketRecord], Dict[str, Any]]:
        """
        Loads all packets from a target capture file, gathers file and packet
        telemetry, and returns list of PacketRecord objects.
        """
        logger.info(f"Opening and parsing capture file: {pcap_path}")
        
        file_size_bytes = os.path.getsize(pcap_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        packets = []
        protocols_count = {"TCP": 0, "UDP": 0, "Other": 0}
        unique_ips = set()
        
        start_time = None
        end_time = None
        
        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    parsed = cls.parse_scapy_packet(pkt)
                    if parsed:
                        packets.append(parsed)
                        unique_ips.add(parsed.src_ip)
                        unique_ips.add(parsed.dst_ip)
                        
                        if parsed.proto == 6:
                            protocols_count["TCP"] += 1
                        elif parsed.proto == 17:
                            protocols_count["UDP"] += 1
                        else:
                            protocols_count["Other"] += 1
                            
                        if start_time is None:
                            start_time = parsed.timestamp
                        end_time = parsed.timestamp
        except Exception as e:
            logger.error(f"Error statically parsing PCAP: {e}")
            raise e
            
        total_packets = len(packets)
        duration = (end_time - start_time) if (start_time and end_time) else 0.0
        
        metadata = {
            "file_name": os.path.basename(pcap_path),
            "file_size_mb": file_size_mb,
            "total_packets": total_packets,
            "duration_seconds": duration,
            "unique_ips_count": len(unique_ips),
            "protocols": protocols_count,
            "start_time_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(start_time)) if start_time else "N/A",
            "end_time_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(end_time)) if end_time else "N/A",
        }
        
        logger.info(f"Successfully processed PCAP: {total_packets} packets loaded across {duration:.2f} seconds.")
        return packets, metadata
