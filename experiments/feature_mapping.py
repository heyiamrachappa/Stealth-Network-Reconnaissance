#!/usr/bin/env python3
"""
Phase 1B — Feature Mapping
Maps CIC-IDS-2017 (79-col) and UNSW-NB15 (49-col) flow features
to the project's canonical 20-feature schema defined in models/feature_names.json.

Every mapping decision is documented explicitly. Where a field is unavailable,
0.0 is substituted and the limitation is recorded in results/feature_mapping_report.json.

Output:
  - results/feature_mapping_report.json  (explicit mapping table)
  - dataset/mapped/cic_mapped.parquet    (CIC Friday PortScan → 20 features + label)
  - dataset/mapped/unsw_mapped.parquet   (UNSW-NB15 Recon subset → 20 features + label)
"""

import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
MAPPED_DIR  = ROOT / "dataset" / "mapped"
MAPPED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "flow_duration", "flow_packet_count", "flow_bytes",
    "flow_syn_ratio", "flow_ack_ratio", "flow_rst_ratio", "flow_fin_ratio",
    "flow_size_mean", "flow_size_var",
    "flow_interval_mean", "flow_interval_var",
    "host_port_entropy", "host_dst_entropy", "host_dst_diversity",
    "host_syn_ratio", "host_failed_flow_ratio",
    "host_packet_rate", "host_interval_mean", "host_interval_var", "host_packet_size_var"
]

# ── CIC-IDS-2017 mapping ──────────────────────────────────────────────────────
CIC_MAPPING = {
    "flow_duration":       ("Flow Duration",                   "direct",       "microseconds → seconds: divide by 1e6"),
    "flow_packet_count":   ("Total Fwd Packets+Total Backward Packets", "computed", "sum fwd+bwd packet counts"),
    "flow_bytes":          ("Total Length of Fwd Packets+Total Length of Bwd Packets", "computed", "sum fwd+bwd byte counts"),
    "flow_syn_ratio":      ("SYN Flag Count/flow_packet_count","computed",     "flag count / total packets"),
    "flow_ack_ratio":      ("ACK Flag Count/flow_packet_count","computed",     "flag count / total packets"),
    "flow_rst_ratio":      ("RST Flag Count/flow_packet_count","computed",     "flag count / total packets"),
    "flow_fin_ratio":      ("FIN Flag Count/flow_packet_count","computed",     "flag count / total packets"),
    "flow_size_mean":      ("Packet Length Mean",              "direct",       "mean packet size"),
    "flow_size_var":       ("Packet Length Variance",          "direct",       "variance of packet sizes"),
    "flow_interval_mean":  ("Flow IAT Mean",                   "direct",       "mean inter-arrival time in microseconds"),
    "flow_interval_var":   ("Flow IAT Std",                    "approximated", "using std^2; CIC reports std not variance"),
    "host_port_entropy":   ("N/A",                             "unavailable",  "per-flow dataset; host aggregation requires raw pcap → set to 0.0"),
    "host_dst_entropy":    ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_dst_diversity":  ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_syn_ratio":      ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_failed_flow_ratio":("N/A",                           "unavailable",  "per-flow dataset → set to 0.0"),
    "host_packet_rate":    ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_interval_mean":  ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_interval_var":   ("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
    "host_packet_size_var":("N/A",                             "unavailable",  "per-flow dataset → set to 0.0"),
}

# ── UNSW-NB15 mapping ─────────────────────────────────────────────────────────
UNSW_MAPPING = {
    "flow_duration":        ("dur",                            "direct",       "seconds"),
    "flow_packet_count":    ("spkts+dpkts",                   "computed",     "sum of source and destination packet counts"),
    "flow_bytes":           ("sbytes+dbytes",                  "computed",     "sum of source and destination bytes"),
    "flow_syn_ratio":       ("state=='SYN' or state=='S0'",   "approximated", "binary: 1.0 if state indicates SYN-only flow, else 0.0"),
    "flow_ack_ratio":       ("N/A",                            "unavailable",  "UNSW-NB15 provides state string, not per-flag counts → set to 0.0"),
    "flow_rst_ratio":       ("state=='RST'",                   "approximated", "binary: 1.0 if state=RST, else 0.0"),
    "flow_fin_ratio":       ("N/A",                            "unavailable",  "not distinguishable from state field → set to 0.0"),
    "flow_size_mean":       ("(sbytes+dbytes)/(spkts+dpkts)", "computed",     "total bytes / total packets; approximate mean"),
    "flow_size_var":        ("N/A",                            "unavailable",  "UNSW-NB15 has no per-packet size distribution → set to 0.0"),
    "flow_interval_mean":   ("(sintpkt+dintpkt)/2",            "approximated", "average of mean source and destination inter-packet times"),
    "flow_interval_var":    ("N/A",                            "unavailable",  "UNSW-NB15 provides mean inter-packet time only → set to 0.0"),
    "host_port_entropy":    ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_dst_entropy":     ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_dst_diversity":   ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_syn_ratio":       ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_failed_flow_ratio":("N/A",                           "unavailable",  "per-flow dataset → set to 0.0"),
    "host_packet_rate":     ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_interval_mean":   ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_interval_var":    ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
    "host_packet_size_var": ("N/A",                            "unavailable",  "per-flow dataset → set to 0.0"),
}


def map_cic_friday(path: Path) -> pd.DataFrame:
    print("Loading CIC-IDS-2017 Friday PortScan parquet...")
    df = pd.read_parquet(path)

    # Strip whitespace from column names (CIC datasets often have leading spaces)
    df.columns = df.columns.str.strip()

    out = pd.DataFrame(index=df.index)
    out["flow_duration"]       = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0) / 1e6
    out["flow_packet_count"]   = (pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0) +
                                   pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0))
    out["flow_bytes"]          = (pd.to_numeric(df["Total Length of Fwd Packets"], errors="coerce").fillna(0) +
                                   pd.to_numeric(df["Total Length of Bwd Packets"], errors="coerce").fillna(0))

    pkt_count_safe = out["flow_packet_count"].replace(0, np.nan)
    out["flow_syn_ratio"]      = pd.to_numeric(df["SYN Flag Count"], errors="coerce").fillna(0) / pkt_count_safe
    out["flow_ack_ratio"]      = pd.to_numeric(df["ACK Flag Count"], errors="coerce").fillna(0) / pkt_count_safe
    out["flow_rst_ratio"]      = pd.to_numeric(df["RST Flag Count"], errors="coerce").fillna(0) / pkt_count_safe
    out["flow_fin_ratio"]      = pd.to_numeric(df["FIN Flag Count"], errors="coerce").fillna(0) / pkt_count_safe

    out["flow_size_mean"]      = pd.to_numeric(df["Packet Length Mean"], errors="coerce").fillna(0)
    out["flow_size_var"]       = pd.to_numeric(df["Packet Length Variance"], errors="coerce").fillna(0)
    out["flow_interval_mean"]  = pd.to_numeric(df["Flow IAT Mean"], errors="coerce").fillna(0)
    iat_std = pd.to_numeric(df["Flow IAT Std"], errors="coerce").fillna(0)
    out["flow_interval_var"]   = iat_std ** 2

    # Host-level features — unavailable in per-flow CIC data
    for col in ["host_port_entropy","host_dst_entropy","host_dst_diversity",
                "host_syn_ratio","host_failed_flow_ratio","host_packet_rate",
                "host_interval_mean","host_interval_var","host_packet_size_var"]:
        out[col] = 0.0

    # Label: PortScan → 1, BENIGN → 0
    out["label"]       = (df["Label"].str.strip() != "BENIGN").astype(int)
    out["attack_type"] = df["Label"].str.strip()
    out["dataset_source"] = "cic_ids_2017_friday_portscan"

    out = out.fillna(0.0)
    # Replace inf
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    return out


def map_unsw(path: Path) -> pd.DataFrame:
    UNSW_COLS = [
        "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes",
        "sttl","dttl","sloss","dloss","service","sload","dload","spkts","dpkts",
        "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len",
        "sjit","djit","stime","ltime","sintpkt","dintpkt","tcprtt","synack","ackdat",
        "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
        "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm",
        "ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","label"
    ]
    print("Loading UNSW-NB15 (700k rows)...")
    df = pd.read_csv(path, header=None, names=UNSW_COLS, low_memory=False)

    # Keep only Reconnaissance + benign flows to limit scope and class imbalance
    recon_mask = df["attack_cat"].str.strip().str.lower() == "reconnaissance"
    benign_mask = df["label"] == 0
    # Sample benign to 2× recon count for balance
    n_recon = recon_mask.sum()
    benign_idx = df[benign_mask].sample(n=min(n_recon * 2, benign_mask.sum()), random_state=42).index
    df_sub = pd.concat([df[recon_mask], df.loc[benign_idx]])
    print(f"  UNSW subset: {len(df_sub)} rows ({n_recon} recon + {len(benign_idx)} benign)")

    out = pd.DataFrame(index=df_sub.index)
    out["flow_duration"]      = pd.to_numeric(df_sub["dur"], errors="coerce").fillna(0)
    spkts = pd.to_numeric(df_sub["spkts"], errors="coerce").fillna(0)
    dpkts = pd.to_numeric(df_sub["dpkts"], errors="coerce").fillna(0)
    sbytes = pd.to_numeric(df_sub["sbytes"], errors="coerce").fillna(0)
    dbytes = pd.to_numeric(df_sub["dbytes"], errors="coerce").fillna(0)

    out["flow_packet_count"]  = spkts + dpkts
    out["flow_bytes"]         = sbytes + dbytes

    total_pkts_safe = out["flow_packet_count"].replace(0, np.nan)
    state = df_sub["state"].str.strip().str.upper().fillna("")
    out["flow_syn_ratio"]     = state.isin(["SYN","S0","S1"]).astype(float)
    out["flow_ack_ratio"]     = 0.0   # unavailable
    out["flow_rst_ratio"]     = state.isin(["RST","RSTO","RSTR","RSTRH"]).astype(float)
    out["flow_fin_ratio"]     = 0.0   # unavailable

    out["flow_size_mean"]     = out["flow_bytes"] / total_pkts_safe
    out["flow_size_var"]      = 0.0   # unavailable
    sintpkt = pd.to_numeric(df_sub["sintpkt"], errors="coerce").fillna(0)
    dintpkt = pd.to_numeric(df_sub["dintpkt"], errors="coerce").fillna(0)
    out["flow_interval_mean"] = (sintpkt + dintpkt) / 2.0
    out["flow_interval_var"]  = 0.0   # unavailable

    for col in ["host_port_entropy","host_dst_entropy","host_dst_diversity",
                "host_syn_ratio","host_failed_flow_ratio","host_packet_rate",
                "host_interval_mean","host_interval_var","host_packet_size_var"]:
        out[col] = 0.0

    out["label"]          = pd.to_numeric(df_sub["label"], errors="coerce").fillna(0).astype(int)
    out["attack_type"]    = df_sub["attack_cat"].str.strip().fillna("benign")
    out["dataset_source"] = "unsw_nb15"

    out = out.fillna(0.0)
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    return out


def main():
    report = {
        "canonical_schema": FEATURE_NAMES,
        "schema_size": len(FEATURE_NAMES),
        "unavailable_in_public_datasets": [
            "host_port_entropy","host_dst_entropy","host_dst_diversity",
            "host_syn_ratio","host_failed_flow_ratio","host_packet_rate",
            "host_interval_mean","host_interval_var","host_packet_size_var"
        ],
        "cic_ids_2017_mapping": {
            feat: {"source_col": m[0], "type": m[1], "note": m[2]}
            for feat, m in CIC_MAPPING.items()
        },
        "unsw_nb15_mapping": {
            feat: {"source_col": m[0], "type": m[1], "note": m[2]}
            for feat, m in UNSW_MAPPING.items()
        },
        "mapping_stats": {
            "cic_direct_mappings": sum(1 for m in CIC_MAPPING.values() if m[1] == "direct"),
            "cic_computed_mappings": sum(1 for m in CIC_MAPPING.values() if m[1] == "computed"),
            "cic_approximated_mappings": sum(1 for m in CIC_MAPPING.values() if m[1] == "approximated"),
            "cic_unavailable_mappings": sum(1 for m in CIC_MAPPING.values() if m[1] == "unavailable"),
            "unsw_direct_mappings": sum(1 for m in UNSW_MAPPING.values() if m[1] == "direct"),
            "unsw_computed_mappings": sum(1 for m in UNSW_MAPPING.values() if m[1] == "computed"),
            "unsw_approximated_mappings": sum(1 for m in UNSW_MAPPING.values() if m[1] == "approximated"),
            "unsw_unavailable_mappings": sum(1 for m in UNSW_MAPPING.values() if m[1] == "unavailable"),
        }
    }

    # Map CIC Friday PortScan
    cic_fri = ROOT / "dataset/public/cic-ids-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet"
    cic_mapped = map_cic_friday(cic_fri)
    cic_out = MAPPED_DIR / "cic_mapped.parquet"
    cic_mapped.to_parquet(cic_out, index=False)
    report["cic_mapped_output"] = {
        "path": str(cic_out),
        "rows": int(len(cic_mapped)),
        "label_distribution": cic_mapped["label"].value_counts().to_dict(),
        "attack_type_distribution": cic_mapped["attack_type"].value_counts().to_dict()
    }
    print(f"CIC mapped: {len(cic_mapped):,} rows → {cic_out}")

    # Map UNSW-NB15
    unsw_path = ROOT / "dataset/public/unsw-nb15/UNSW-NB15_1.csv"
    unsw_mapped = map_unsw(unsw_path)
    unsw_out = MAPPED_DIR / "unsw_mapped.parquet"
    unsw_mapped.to_parquet(unsw_out, index=False)
    report["unsw_mapped_output"] = {
        "path": str(unsw_out),
        "rows": int(len(unsw_mapped)),
        "label_distribution": {str(k): int(v) for k, v in unsw_mapped["label"].value_counts().items()},
        "attack_type_distribution": {str(k): int(v) for k, v in unsw_mapped["attack_type"].value_counts().items()}
    }
    print(f"UNSW mapped: {len(unsw_mapped):,} rows → {unsw_out}")

    out_path = RESULTS_DIR / "feature_mapping_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFeature mapping report written to {out_path}")
    print(f"Mapping stats: {report['mapping_stats']}")


if __name__ == "__main__":
    main()
