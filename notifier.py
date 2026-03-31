import requests

def notify_embedding_status(file_id, job_id, file_name, created_at):

    url = "https://api.xtrology.ai/kb/kb_status.php"

    payload = {
        "job_id": job_id,
        "file_id": file_id,
        "file_name": file_id,   # ✅ FIXED
        "status": "completed",
        "created_at": created_at,
        "completed_at": int(time.time()),
        "error": ""
    }

    response = requests.post(url, data=payload)

    print("Callback response:", response.text)