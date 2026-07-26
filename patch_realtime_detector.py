import re

with open('realtime_detector.py', 'r') as f:
    content = f.read()

# 1. Import engine
content = content.replace(
    "from alerts.engine import AlertEngine",
    "from alerts.engine import AlertEngine\nfrom threat_analysis.behavioral_profile_engine import BehavioralProfileEngine"
)

# 2. Instantiate engine
content = content.replace(
    "self.alert_engine = AlertEngine()",
    "self.alert_engine = AlertEngine()\n        self.behavioral_engine = BehavioralProfileEngine()"
)

# 3. Update run_analyzer_thread
old_analyzer = """                pkt = self.packet_queue.get(timeout=1.0)
                self.tracker.handle_packet(pkt)"""
new_analyzer = """                pkt = self.packet_queue.get(timeout=1.0)
                session = self.tracker.handle_packet(pkt)
                self.behavioral_engine.update_profile(
                    src_ip=pkt.src_ip,
                    dst_ip=pkt.dst_ip,
                    dst_port=pkt.dst_port,
                    protocol=str(pkt.proto),
                    bytes_len=pkt.payload_len,
                    session_id=str(session.flow_key),
                    timestamp=pkt.timestamp,
                    session_duration=session.duration
                )"""
content = content.replace(old_analyzer, new_analyzer)

# 4. Fetch drift score and pass to generate_alert
old_eval = """            # Formulate structured threat incident alerts
            self.alert_engine.generate_alert(
                flow=flow,
                raw_features=raw_features,
                ml_prediction=prediction,
                ml_confidence=confidence,
                detection_method=self.model_name
            )"""
new_eval = """            # Fetch drift score
            profile = self.behavioral_engine.retrieve_profile(flow.src_ip)
            drift_score = profile.get("drift_score", 0.0) if profile else 0.0
            
            # Formulate structured threat incident alerts
            self.alert_engine.generate_alert(
                flow=flow,
                raw_features=raw_features,
                ml_prediction=prediction,
                ml_confidence=confidence,
                detection_method=self.model_name,
                drift_score=drift_score
            )"""
content = content.replace(old_eval, new_eval)

with open('realtime_detector.py', 'w') as f:
    f.write(content)

