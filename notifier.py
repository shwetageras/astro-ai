import requests
import time
from datetime import datetime

def notify_embedding_status(file_id, job_id, created_at, file_name):

    url = "https://api.xtrology.ai/kb/kb_status.php"

    payload = {
        "job_id": job_id,
        "file_id": file_name,
        "file_name": file_name,
        "status": "completed",
        "created_at": datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": datetime.fromtimestamp(int(time.time())).strftime("%Y-%m-%d %H:%M:%S"),
        "error": None
    }

    response = requests.post(url, data=payload)

    print("Callback response:", response.text)