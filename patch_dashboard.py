import re

with open('dashboard/app.py', 'r') as f:
    content = f.read()

# 1. Update Sidebar
sidebar_old = """                    "MITRE ATT&CK Mapping", 
                    "ML Analysis Diagnostics",
                    "Settings & Utilities"
                ],"""
sidebar_new = """                    "MITRE ATT&CK Mapping", 
                    "ML Analysis Diagnostics",
                    "Behavioral Intelligence",
                    "Settings & Utilities"
                ],"""
content = content.replace(sidebar_old, sidebar_new)

# 2. Add cached_packets
cached_old = """                            st.session_state["cached_reports"] = reports
                            st.session_state["cached_metadata"] = metadata
                            st.session_state["cached_pcap_path"] = pcap_path
                            st.session_state["cached_model"] = active_model
                        except Exception as e:"""
cached_new = """                            st.session_state["cached_packets"] = packets
                            st.session_state["cached_reports"] = reports
                            st.session_state["cached_metadata"] = metadata
                            st.session_state["cached_pcap_path"] = pcap_path
                            st.session_state["cached_model"] = active_model
                        except Exception as e:"""
content = content.replace(cached_old, cached_new)


# 3. Add Behavioral Intelligence section right before "Settings & Utilities"
settings_old = """                elif navigation == "Settings & Utilities":"""
behavioral_section = """                elif navigation == "Behavioral Intelligence":
                    st.markdown("### 🧠 BEHAVIORAL INTELLIGENCE & DRIFT DETECTION")
                    
                    if "cached_packets" not in st.session_state:
                        st.error("Raw packets not found in cache. Please re-upload the PCAP.")
                    else:
                        with st.spinner("Compiling behavioral profile timelines..."):
                            from threat_analysis.behavioral_profile_engine import BehavioralProfileEngine
                            engine = BehavioralProfileEngine()
                            
                            packets = st.session_state["cached_packets"]
                            timeline_records = []
                            last_timeline_save = 0.0
                            
                            # Process chronologically
                            for pkt in sorted(packets, key=lambda p: p.timestamp):
                                engine.update_profile(
                                    src_ip=pkt.src_ip,
                                    dst_ip=pkt.dst_ip,
                                    dst_port=pkt.dst_port,
                                    protocol=str(pkt.proto),
                                    bytes_len=pkt.payload_len,
                                    timestamp=pkt.timestamp,
                                    session_id=str((pkt.src_ip, pkt.dst_ip, pkt.dst_port, pkt.proto))
                                )
                                if (pkt.timestamp - last_timeline_save) > 5.0:
                                    last_timeline_save = pkt.timestamp
                                    for host_ip, profile in engine._profiles.items():
                                        ds = profile.calculate_drift(pkt.timestamp)
                                        if ds > 0:
                                            timeline_records.append({"time": pkt.timestamp, "host": host_ip, "drift": ds})
                        
                        # Get all profiles
                        all_profiles = engine.retrieve_all_profiles()
                        ranked_hosts = sorted(all_profiles.values(), key=lambda p: p.get("drift_score", 0.0), reverse=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>TOP ABNORMAL HOSTS</h4>", unsafe_allow_html=True)
                        if ranked_hosts:
                            top_host = ranked_hosts[0]
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                st.metric("Top Abnormal Host", top_host["src_ip"], f"Drift: {top_host.get('drift_score', 0.0):.1f}/100", delta_color="inverse")
                            with c2:
                                df_top = pd.DataFrame([{
                                    "Host IP": p["src_ip"], 
                                    "Drift Score": round(p.get("drift_score", 0.0), 2),
                                    "Sessions": p["session_count"],
                                    "Dest Diversity": p["destination_ip_count"]
                                } for p in ranked_hosts[:5]])
                                st.dataframe(df_top, use_container_width=True, hide_index=True)
                        else:
                            st.info("No host profiles generated.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>BEHAVIORAL DRIFT TIMELINE</h4>", unsafe_allow_html=True)
                        if timeline_records:
                            df_tl = pd.DataFrame(timeline_records)
                            min_time = df_tl["time"].min()
                            df_tl["time"] = df_tl["time"] - min_time
                            
                            tl_pivot = df_tl.pivot(index="time", columns="host", values="drift").ffill().fillna(0)
                            
                            fig, ax = plt.subplots(figsize=(10, 3.5))
                            fig.patch.set_facecolor('none')
                            ax.set_facecolor('none')
                            ax.tick_params(colors='#94A3B8')
                            ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.15))
                            ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.set_ylabel("Drift Score (0-100)", color="#94A3B8")
                            ax.set_xlabel("Seconds from Start", color="#94A3B8")
                            
                            # Limit to top 5 hosts
                            top_hosts_list = [p["src_ip"] for p in ranked_hosts[:5]]
                            for col in tl_pivot.columns:
                                if col in top_hosts_list:
                                    ax.plot(tl_pivot.index, tl_pivot[col], label=col, linewidth=2)
                            ax.legend(facecolor='#060913', edgecolor=(1.0, 1.0, 1.0, 0.1), labelcolor='#E2E8F0', fontsize=8, loc='upper left')
                            st.pyplot(fig)
                        else:
                            st.info("Insufficient timeline data. PCAP duration must exceed 5 periods of 60 seconds (total > 300s) to establish baseline.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='cyber-panel'><h4>CURRENT VS HISTORICAL BEHAVIOR</h4>", unsafe_allow_html=True)
                        if ranked_hosts:
                            selected_host = st.selectbox("Select Host to Inspect", [p["src_ip"] for p in ranked_hosts])
                            if selected_host:
                                host_data = next((p for p in ranked_hosts if p["src_ip"] == selected_host), None)
                                if host_data:
                                    w5m = host_data.get("windows", {}).get("5m", {})
                                    base = host_data.get("baselines", {})
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Destination Diversity", w5m.get("destination_ip_count", 0), f"Baseline: {base.get('destination_diversity', 0.0):.1f}", delta_color="inverse")
                                    c2.metric("Port Diversity", w5m.get("destination_port_count", 0), f"Baseline: {base.get('port_diversity', 0.0):.1f}", delta_color="inverse")
                                    c3.metric("Pkt Rate (pkts/sec)", f"{w5m.get('packet_count', 0)/300.0:.2f}", f"Baseline: {base.get('packet_rate', 0.0):.2f}", delta_color="inverse")
                                    c4.metric("Mean Session Duration", f"{w5m.get('mean_session_duration', 0.0):.2f}s", f"Baseline: {base.get('session_duration', 0.0):.2f}s", delta_color="inverse")
                        st.markdown("</div>", unsafe_allow_html=True)

                elif navigation == "Settings & Utilities":"""
content = content.replace(settings_old, behavioral_section)

with open('dashboard/app.py', 'w') as f:
    f.write(content)

