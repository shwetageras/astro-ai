from datetime import datetime
from zoneinfo import ZoneInfo
import requests

def notify_embedding_status(file_id, job_id, created_at, file_name):

    ist = ZoneInfo("Asia/Kolkata")

    created_at_str = datetime.fromtimestamp(created_at, ist).strftime("%Y-%m-%d %H:%M:%S")
    completed_at_str = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "job_id": job_id,
        "file_id": file_id,
        "file_name": file_name,
        "status": "completed",
        "created_at": created_at_str,
        "completed_at": completed_at_str,
        "error": None
    }

    response = requests.post(
        "https://api.xtrology.ai/kb/kb_status.php",
        json=payload
    )

    print("Callback response:", response.text)