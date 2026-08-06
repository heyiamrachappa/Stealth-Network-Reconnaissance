#!/usr/bin/env python3
"""Extract flow features from PCAP files using feature_extraction/ pipeline."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_extraction.extractor import StaticFlowTracker, FeatureExtractor
from pcap_processing.parser import PCAPParser
from utils.helpers import load_config, setup_logger

logger = setup_logger("PcapFeatureExtract")

METADATA_COLS = {"src_ip", "dst_ip", "src_port", "dst_port", "proto", "timestamp", "label",
                 "pcap_filename", "attack_type", "dataset", "flow_id"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_pcap(
    pcap_path: Path,
    label: int | None = None,
    attack_type: str = "unknown",
    dataset: str = "lab",
    max_packets: int | None = None,
    sliding_window: int | None = None,
) -> pd.DataFrame:
    packets, _ = PCAPParser.load_pcap(str(pcap_path))
    if max_packets is not None:
        packets = packets[:max_packets]

    if not packets:
        logger.warning("No packets in %s", pcap_path)
        return pd.DataFrame()

    flows = StaticFlowTracker.track_flows(packets)
    extractor = FeatureExtractor()
    if sliding_window is not None:
        extractor.sliding_window = sliding_window

    df = extractor.extract_features(flows)
    if df.empty:
        return df

    df["timestamp"] = df.apply(
        lambda r: packets[0].timestamp if not packets else packets[-1].timestamp, axis=1
    )
    # Per-flow start time from first packet in each flow
    flow_starts = {}
    for flow in flows:
        if flow.packets:
            key = (flow.src_ip, flow.dst_ip, flow.src_port, flow.dst_port, flow.proto)
            flow_starts[key] = flow.packets[0].timestamp

    timestamps = []
    for _, row in df.iterrows():
        key = (row["src_ip"], row["dst_ip"], row["src_port"], row["dst_port"], row["proto"])
        timestamps.append(flow_starts.get(key, packets[0].timestamp))
    df["timestamp"] = timestamps

    df["pcap_filename"] = pcap_path.name
    df["attack_type"] = attack_type
    df["dataset"] = dataset
    if label is not None:
        df["label"] = label
    df["flow_id"] = df.apply(
        lambda r: f"{r['src_ip']}:{r['src_port']}-{r['dst_ip']}:{r['dst_port']}/{r['proto']}",
        axis=1,
    )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap-dir", type=Path, required=True)
    parser.add_argument("--experiment-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--dataset-name", default="lab")
    args = parser.parse_args()

    config = load_config()
    if args.sliding_window is None:
        args.sliding_window = config.get("features", {}).get("sliding_window_seconds", 30)

    label_map = {}
    attack_map = {}
    if args.experiment_log and args.experiment_log.exists():
        log = pd.read_csv(args.experiment_log)
        for _, row in log.iterrows():
            label_map[row["pcap_filename"]] = int(row["label"])
            attack_map[row["pcap_filename"]] = row["attack_type"]

    frames = []
    for pcap in sorted(args.pcap_dir.glob("*.pcap")):
        label = label_map.get(pcap.name)
        attack = attack_map.get(pcap.name, "unknown")
        logger.info("Extracting %s (label=%s)", pcap.name, label)
        df = extract_from_pcap(
            pcap, label=label, attack_type=attack, dataset=args.dataset_name,
            max_packets=args.max_packets, sliding_window=args.sliding_window,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("No features extracted")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    schema_path = ROOT / "models" / "feature_names.json"
    feat_cols = feature_columns(combined)
    schema_path.write_text(json.dumps(feat_cols, indent=2))
    logger.info("Saved %d flows to %s (%d features)", len(combined), args.output, len(feat_cols))


if __name__ == "__main__":
    main()
