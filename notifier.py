from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import time

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

    try:
        response = requests.post(
            "https://api.xtrology.ai/kb/kb_status.php",
            json=payload,
            timeout=10
        )

        print("Callback response:", response.text)

        if response.status_code != 200:
            print(f"[WARNING] KB callback failed with status code {response.status_code}")

    except Exception as e:
        print(f"[ERROR] KB callback failed: {str(e)}")


def notify_chart_status(job_id, chart_id, file_id):

    url = "https://api.xtrology.ai/charts/charts_status.php"

    payload = {
        "job_id": job_id,
        "chart_id": chart_id,
        "status": "completed",
        "timestamp": int(time.time()),
        "file_id": file_id
    }

    print("Chart Callback Payload:", payload)

    try:
        response = requests.post(url, json=payload, timeout=10)

        print("Chart Callback Response:", response.text)

        if response.status_code != 200:
            print(f"[WARNING] Callback failed with status code {response.status_code}")

    except Exception as e:
        print(f"[ERROR] Chart callback failed: {str(e)}")