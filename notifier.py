import requests
import time

def notify_embedding_status(file_id, job_id, file_name, created_at):
    url = "https://api.xtrology.ai/kb/kb_status.php"   
    
    payload = {
        "job_id": str(job_id),
        "file_id": str(file_id),
        "file_name": str(file_id),
        "status": "completed",
        "created_at": str(created_at),
        "completed_at": str(int(time.time())),
        "error": ""
    }

    try:
        print("Sending payload:", payload)
        response = requests.post(url, data=payload)
        print("Status pushed:", response.status_code, response.text)
    except Exception as e:
        print("Failed to notify:", str(e))