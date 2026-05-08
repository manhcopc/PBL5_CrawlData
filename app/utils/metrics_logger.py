import csv
import os
from datetime import datetime
from app.core.config import settings

METRICS_FILE = os.path.join(settings.ROOT_DIR, "metrics_log.csv")

def log_metric(api_name: str, latency_sec: float, vram_gb: float = 0.0, extra_info: str = ""):
    file_exists = os.path.isfile(METRICS_FILE)
    
    with open(METRICS_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Ghi header nếu file mới tinh
        if not file_exists:
            writer.writerow(["timestamp", "api_name", "latency_seconds", "peak_vram_gb", "extra_info"])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            api_name,
            round(latency_sec, 4),
            round(vram_gb, 2),
            extra_info
        ])