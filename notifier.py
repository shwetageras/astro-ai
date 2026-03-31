import requests
from datetime import datetime

def notify_embedding_status(file_id, embedding_size):
    url = "https://api.xtrology.ai/kb/kb_status.php"   
    
    payload = {
        "file_id": file_id,
        "status": "embedded",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  
        "embedded_size": embedding_size
    }

    try:
        print("Sending payload:", payload)
        response = requests.post(url, data=payload)
        print("Status pushed:", response.status_code, response.text)
    except Exception as e:
        print("Failed to notify:", str(e))