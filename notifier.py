import requests
import time
from datetime import datetime
import pytz

ist = pytz.timezone("Asia/Kolkata")

created_at_str = datetime.fromtimestamp(created_at, ist).strftime("%Y-%m-%d %H:%M:%S")
completed_at_str = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

def notify_embedding_status(file_id, job_id, created_at, file_name):

    url = "https://api.xtrology.ai/kb/kb_status.php"

    payload = {
        "job_id": job_id,
        "file_id": file_id,
        "file_name": file_name,
        "status": "completed",
        "created_at": created_at_str,
        "completed_at": completed_at_str,
        "error": None,
        "s3_url": f"https://your-bucket.s3.amazonaws.com/{file_id}"  # optional
    }

    print("Sending payload:", payload)

    # 🔥 FIXED LINE
    response = requests.post(url, json=payload)

    print("Callback response:", response.text)