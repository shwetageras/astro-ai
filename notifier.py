import requests

def notify_completion(file_id):
    
    url = "https://example.com/api/notify"  # dummy for now
    
    payload = {
        "file_id": file_id,
        "status": "completed"
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Notification sent:", response.status_code)
    except Exception as e:
        print("Notification failed:", str(e))


def notify_embedding_status(file_id, embedding_size):
    url = "https://api.xtrology.ai/charts/charts_status.php"
    
    payload = {
        "file_id": file_id,
        "status": "embedded",
        "embedded_size": embedding_size
    }

    try:
        response = requests.post(url, json=payload)
        print("Status pushed:", response.status_code, response.text)
    except Exception as e:
        print("Failed to notify:", str(e))