#!/usr/bin/env python3
# ==============================================================================
# Phase 2 - Packet Capture Module
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

import os
import sys
import json
import logging
import argparse
from typing import Callable, Optional
from scapy.all import sniff, wrpcap, Packet, IP, TCP, UDP

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/yi/Stealth System/logs/system.log", mode="a")
    ]
)
logger = logging.getLogger("PacketCapturer")


class PacketCapturer:
    """
    Handles live packet sniffing, filtering, and writing to PCAP files.
    """
    def __init__(self, interface: str = "any", config_path: str = "/home/yi/Stealth System/configs/config.json"):
        self.interface = interface
        self.config = self._load_config(config_path)
        self.captured_packets = []
        
        # Override interface if defaults specify and not set by user
        if self.interface == "any" and self.config:
            self.interface = self.config.get("capture", {}).get("default_interface", "any")

    def _load_config(self, config_path: str) -> Optional[dict]:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
        return None

    def packet_callback(self, packet: Packet, external_callback: Optional[Callable[[Packet], None]] = None) -> None:
        """
        Callback executed for every sniffed packet.
        Perform initial inspection and log basic details.
        """
        try:
            self.captured_packets.append(packet)
            
            # Simple summary logging
            if packet.haslayer(IP):
                ip_src = packet[IP].src
                ip_dst = packet[IP].dst
                proto = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "Other"
                
                info = f"Flow: {ip_src} -> {ip_dst} [{proto}]"
                
                if packet.haslayer(TCP):
                    flags = packet[TCP].sprintf("%TCP.flags%")
                    sport = packet[TCP].sport
                    dport = packet[TCP].dport
                    info += f" Ports: {sport}->{dport} Flags: {flags}"
                elif packet.haslayer(UDP):
                    info += f" Ports: {packet[UDP].sport}->{packet[UDP].dport}"
                    
                logger.debug(info)
                
            # If an external pipeline callback (e.g. real-time predictor) is registered, pass the packet
            if external_callback:
                external_callback(packet)
                
        except Exception as e:
            logger.error(f"Error handling packet: {e}")

    def start_sniffing(self, 
                       timeout: int = 30, 
                       packet_count: int = 0, 
                       filter_str: str = "ip and (tcp or udp)",
                       external_callback: Optional[Callable[[Packet], None]] = None) -> int:
        """
        Starts packet capture.
        :param timeout: Sniffing duration in seconds.
        :param packet_count: Number of packets to capture (0 for infinite/until timeout).
        :param filter_str: BPF filter expression.
        :param external_callback: Custom processor for packets.
        :return: Count of captured packets.
        """
        logger.info(f"Starting packet sniffing on interface '{self.interface}'...")
        logger.info(f"Filter: '{filter_str}', Timeout: {timeout}s, Count: {packet_count if packet_count > 0 else 'Unbounded'}")
        
        self.captured_packets.clear()
        
        # Scapy sniff parameters. 'store=0' is useful for pure real-time, 
        # but we store here to write PCAP at the end.
        try:
            # Under linux, "any" interface requires scapy to run with socket capabilities.
            # Scapy converts "any" to appropriate Linux socket types.
            sniff_args = {
                "prn": lambda pkt: self.packet_callback(pkt, external_callback),
                "filter": filter_str,
                "store": 1
            }
            if self.interface != "any":
                sniff_args["iface"] = self.interface
            if timeout > 0:
                sniff_args["timeout"] = timeout
            if packet_count > 0:
                sniff_args["count"] = packet_count
                
            sniff(**sniff_args)
            
            captured_len = len(self.captured_packets)
            logger.info(f"Sniffing stopped. Captured {captured_len} packets.")
            return captured_len
            
        except PermissionError:
            logger.critical("Permission Denied: Run with elevated privileges or apply 'setcap' on the Python binary.")
            raise
        except Exception as e:
            logger.error(f"Sniffing error encountered: {e}")
            return 0

    def save_pcap(self, filename: str) -> bool:
        """
        Saves currently captured packets to a PCAP file.
        """
        if not self.captured_packets:
            logger.warning("No packets captured to save.")
            return False
            
        try:
            # Ensure folder exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            wrpcap(filename, self.captured_packets)
            logger.info(f"Successfully saved {len(self.captured_packets)} packets to PCAP: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save PCAP to {filename}: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stealth IDS - Packet Capture CLI Utility")
    parser.add_argument("-i", "--interface", default="any", help="Network interface to sniff on (default: any)")
    parser.add_argument("-t", "--timeout", type=int, default=15, help="Capture timeout in seconds (default: 15)")
    parser.add_argument("-c", "--count", type=int, default=0, help="Number of packets to capture (default: 0 for unlimited)")
    parser.add_argument("-f", "--filter", default="ip and (tcp or udp)", help="BPF filter string (default: ip and (tcp or udp))")
    parser.add_argument("-o", "--output", default="/home/yi/Stealth System/pcaps/capture.pcap", help="Output PCAP file path")
    
    args = parser.parse_args()
    
    capturer = PacketCapturer(interface=args.interface)
    try:
        count = capturer.start_sniffing(timeout=args.timeout, packet_count=args.count, filter_str=args.filter)
        if count > 0:
            capturer.save_pcap(args.output)
    except KeyboardInterrupt:
        logger.info("Sniffing interrupted by user.")
        if capturer.captured_packets:
            capturer.save_pcap(args.output)
    except Exception as e:
        logger.error(f"CLI Capture Execution failed: {e}")
