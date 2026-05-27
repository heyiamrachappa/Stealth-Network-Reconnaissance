#!/usr/bin/env python3
import os
import sys
import queue
import threading
from typing import Callable, Optional
from scapy.all import sniff, Packet, IP, TCP, UDP
from utils.helpers import setup_logger, load_config

logger = setup_logger("PacketSniffer")

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

class PacketSniffer:
    """
    Handles live, asynchronous network packet capture on selected interfaces.
    Parses packets to lightweight PacketRecord objects and queues them asynchronously.
    """
    def __init__(self, 
                 interface: str = "any", 
                 packet_queue: Optional[queue.Queue] = None):
        self.config = load_config()
        self.interface = interface
        if self.interface == "any":
            self.interface = self.config.get("capture", {}).get("default_interface", "any")
            
        self.packet_queue = packet_queue if packet_queue is not None else queue.Queue(maxsize=5000)
        self.sniff_thread: Optional[threading.Thread] = None
        self.running = False

    @staticmethod
    def parse_packet(packet: Packet) -> Optional[PacketRecord]:
        """
        Extracts key IP, TCP, and UDP layer metrics into a lightweight PacketRecord.
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

    def _packet_callback(self, packet: Packet) -> None:
        """
        Callback executed for every sniffed packet. Parsed and queued.
        """
        parsed = self.parse_packet(packet)
        if parsed:
            try:
                # Put to queue non-blocking to prevent packet loss backpressure
                self.packet_queue.put(parsed, block=False)
            except queue.Full:
                logger.warning("Packet Queue Full! Dropping parsed packet to maintain capture rate.")

    def _sniff_loop(self, timeout: int, filter_str: str) -> None:
        """
        Blocking sniff execution. Runs inside a background thread.
        """
        logger.info(f"Asynchronous sniffing started on interface: '{self.interface}'")
        logger.info(f"BPF Filter: '{filter_str}' | Session Timeout: {timeout}s")
        
        sniff_args = {
            "prn": self._packet_callback,
            "filter": filter_str,
            "store": 0  # Crucial for live sniffing: do not buffer raw packets in Scapy
        }
        
        if self.interface != "any":
            sniff_args["iface"] = self.interface
        if timeout > 0:
            sniff_args["timeout"] = timeout

        try:
            sniff(**sniff_args)
        except PermissionError:
            logger.critical("Permission Denied: Sniffing requires raw socket privileges. Try applying 'setcap' or running as sudo.")
        except Exception as e:
            logger.error(f"Sniffer encountered an error: {e}")
        finally:
            self.running = False
            logger.info("Sniffer thread stopped.")

    def start(self, timeout: int = 0, filter_str: str = "ip and (tcp or udp)") -> None:
        """
        Launches the packet capture in a dedicated daemon thread.
        """
        if self.running:
            logger.warning("Sniffer is already running.")
            return

        self.running = True
        self.sniff_thread = threading.Thread(
            target=self._sniff_loop, 
            args=(timeout, filter_str),
            daemon=True
        )
        self.sniff_thread.start()

    def stop(self) -> None:
        """
        Gracefully terminates sniffing.
        """
        if not self.running:
            return
        self.running = False
        logger.info("Stopping sniffer capture...")
