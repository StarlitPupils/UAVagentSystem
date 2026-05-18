import json, time, os
from core.data_logger import DataLogger
from config.settings import config

class ReportingAgent:
    def __init__(self, logger: DataLogger): self.logger = logger
    def generate_report(self):
        events = self.logger.get_recent_events(100)
        report = {"total": len(events), "events": events, "generated": time.time()}
        path = os.path.join(config.REPORT_DIR, f"report_{int(time.time())}.json")
        with open(path, 'w') as f: json.dump(report, f, indent=2)
        print(f"[Report] 报告已保存: {path}")
        return report