#!/usr/bin/env python3
import time
import requests
from typing import Dict, Any, Optional
from utils.helpers import setup_logger, load_config

logger = setup_logger("ThreatIntelEngine")

class ThreatIntelEngine:
    """
    Enriches forensic IP analysis with real-time reputations from AbuseIPDB, VirusTotal, and Shodan.
    Supports in-memory caching to avoid rate-limiting and API overhead.
    """
    def __init__(self, 
                 abuseipdb_key: Optional[str] = None, 
                 virustotal_key: Optional[str] = None, 
                 shodan_key: Optional[str] = None):
        self.config = load_config()
        
        # Load API keys prioritising constructor arguments then falling back to configs
        self.abuseipdb_key = abuseipdb_key or self.config.get("intel", {}).get("abuseipdb_api_key")
        self.virustotal_key = virustotal_key or self.config.get("intel", {}).get("virustotal_api_key")
        self.shodan_key = shodan_key or self.config.get("intel", {}).get("shodan_api_key")
        
        # In-memory lookup cache to prevent excessive API requests
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_private_ip(self, ip: str) -> bool:
        """
        Determines if an IP address belongs to RFC 1918 private scopes.
        """
        if ip.startswith("127.") or ip == "localhost":
            return True
        parts = list(map(int, ip.split("."))) if "." in ip else []
        if len(parts) != 4:
            return False
        
        # 10.0.0.0/8
        if parts[0] == 10:
            return True
        # 172.16.0.0/12
        if parts[0] == 172 and (16 <= parts[1] <= 31):
            return True
        # 192.168.0.0/16
        if parts[0] == 192 and parts[1] == 168:
            return True
        # 169.254.0.0/16 (Link-Local)
        if parts[0] == 169 and parts[1] == 254:
            return True
        return False

    def lookup_ip(self, ip: str) -> Dict[str, Any]:
        """
        Aggregates reputation across configured Threat Intel feeds for a given IP.
        """
        if not ip or ip == "any":
            return self._empty_response(ip)
            
        if ip in self._cache:
            return self._cache[ip]

        if self.is_private_ip(ip):
            res = {
                "ip": ip,
                "is_private": True,
                "abuse_score": 0,
                "vt_positives": 0,
                "shodan_ports": [],
                "isp": "Internal LAN Scope",
                "country": "Local Intranet",
                "blacklist_status": "SECURE",
                "enrichment_source": "RFC 1918 Address Lookup",
                "details": "Local private network address. Excluded from global blacklisting lookup."
            }
            self._cache[ip] = res
            return res

        # Run lookups using real endpoints or detailed, informative simulation fallbacks
        result = self._fetch_reputation(ip)
        self._cache[ip] = result
        return result

    def _fetch_reputation(self, ip: str) -> Dict[str, Any]:
        """
        Queries live endpoints if keys are set, otherwise falls back to a detailed simulation model.
        """
        reputation = {
            "ip": ip,
            "is_private": False,
            "abuse_score": 0,
            "vt_positives": 0,
            "shodan_ports": [],
            "isp": "Unknown ISP",
            "country": "Unknown Location",
            "blacklist_status": "CLEAN",
            "enrichment_source": "Simulated Cyber-Intel Feeds",
            "details": "No active API keys found. Displaying realistic simulated telemetry."
        }

        # 1. AbuseIPDB Lookup
        if self.abuseipdb_key:
            try:
                url = "https://api.abuseipdb.com/api/v2/check"
                headers = {"Accept": "application/json", "Key": self.abuseipdb_key}
                params = {"ipAddress": ip, "maxAgeInDays": "90"}
                resp = requests.get(url, headers=headers, params=params, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    reputation["abuse_score"] = data.get("abuseConfidenceScore", 0)
                    reputation["isp"] = data.get("isp", reputation["isp"])
                    reputation["country"] = data.get("countryName", reputation["country"])
                    reputation["enrichment_source"] = "AbuseIPDB Live API"
            except Exception as e:
                logger.error(f"AbuseIPDB Lookup error for {ip}: {e}")

        # 2. VirusTotal Lookup
        if self.virustotal_key:
            try:
                url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
                headers = {"x-apikey": self.virustotal_key}
                resp = requests.get(url, headers=headers, timeout=3.0)
                if resp.status_code == 200:
                    stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                    pos = stats.get("malicious", 0) + stats.get("suspicious", 0)
                    reputation["vt_positives"] = pos
                    reputation["enrichment_source"] += " + VirusTotal Live API"
            except Exception as e:
                logger.error(f"VirusTotal Lookup error for {ip}: {e}")

        # 3. Shodan Lookup
        if self.shodan_key:
            try:
                url = f"https://api.shodan.io/shodan/host/{ip}?key={self.shodan_key}"
                resp = requests.get(url, timeout=3.0)
                if resp.status_code == 200:
                    ports = resp.json().get("ports", [])
                    reputation["shodan_ports"] = ports
                    reputation["enrichment_source"] += " + Shodan Live API"
            except Exception as e:
                logger.error(f"Shodan Lookup error for {ip}: {e}")

        # Fallback simulated response generating high-fidelity mock data if no live API calls succeeded
        if reputation["enrichment_source"] == "Simulated Cyber-Intel Feeds":
            # Generate realistic deterministic values based on IP hash to make simulation feel alive
            ip_hash = hash(ip)
            
            # Map malicious indicators deterministically
            abuse_score = abs(ip_hash) % 100
            vt_positives = abs(ip_hash) % 8
            
            # Simulated open ports
            ports_pool = [22, 80, 443, 8080, 3389, 21, 23]
            shodan_ports = [p for p in ports_pool if (ip_hash + p) % 3 == 0]
            
            isps = ["Cloudflare CDN Anchor", "Amazon Web Services Hosting", "DigitalOcean Datacenter", "Chinanet Backbone", "OVH SAS Hosting"]
            countries = ["United States", "Germany", "China", "Netherlands", "Singapore"]
            
            reputation["abuse_score"] = abuse_score
            reputation["vt_positives"] = vt_positives
            reputation["shodan_ports"] = shodan_ports
            reputation["isp"] = isps[abs(ip_hash) % len(isps)]
            reputation["country"] = countries[abs(ip_hash) % len(countries)]
            
            if abuse_score > 75 or vt_positives >= 4:
                reputation["blacklist_status"] = "CRITICAL BLACKLISTED ATTACKER"
            elif abuse_score > 35 or vt_positives >= 1:
                reputation["blacklist_status"] = "SUSPICIOUS HOST SCANNER"
            else:
                reputation["blacklist_status"] = "CLEAN ENTITY"
                
            reputation["details"] = (
                f"Intel feed simulation compiled. Entity is mapped to {reputation['isp']} in {reputation['country']}. "
                f"Abuse score is calculated at {reputation['abuse_score']}% with {reputation['vt_positives']} positive scanner flags on VirusTotal."
            )
            
        else:
            # Map status for real API results
            asc = reputation["abuse_score"]
            vtp = reputation["vt_positives"]
            if asc > 75 or vtp >= 4:
                reputation["blacklist_status"] = "CRITICAL BLACKLISTED ATTACKER"
            elif asc > 35 or vtp >= 1:
                reputation["blacklist_status"] = "SUSPICIOUS HOST SCANNER"
            else:
                reputation["blacklist_status"] = "CLEAN ENTITY"
            reputation["details"] = f"Live aggregated intelligence resolved: Abuse Confidence {asc}%. VirusTotal flag count: {vtp} positive hits."

        return reputation

    def _empty_response(self, ip: str) -> Dict[str, Any]:
        return {
            "ip": ip,
            "is_private": False,
            "abuse_score": 0,
            "vt_positives": 0,
            "shodan_ports": [],
            "isp": "N/A",
            "country": "N/A",
            "blacklist_status": "CLEAN",
            "enrichment_source": "N/A",
            "details": "Invalid target query."
        }
