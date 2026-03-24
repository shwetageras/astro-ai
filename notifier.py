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