#!/usr/bin/env python3
"""Capture reproducibility metadata for every experimental run."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _pip_versions() -> Dict[str, str]:
    packages = [
        "scikit-learn", "xgboost", "pandas", "numpy", "scipy",
        "matplotlib", "joblib", "scapy", "streamlit",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", pkg],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if line.startswith("Version:"):
                    versions[pkg] = line.split(":", 1)[1].strip()
                    break
        except subprocess.CalledProcessError:
            versions[pkg] = "not installed"
    return versions


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "ram_gb": None,
        "gpu": None,
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL, text=True)
        info["gpu"] = out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["gpu"] = "none detected"
    return info


def build_run_metadata(seed: int, config_path: Path, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "config_path": str(config_path),
        "config": config,
        "library_versions": _pip_versions(),
        "hardware": hardware_info(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_run_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
