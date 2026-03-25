import json
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = "xtrology-genai-data"

def save_metadata(file_id, file_name, upload_time):
    
    metadata_file = "metadata.json"
    
    # If file exists, load existing data
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []   # file exists but empty/corrupt
    else:
        data = []
    
    # Append new entry
    data.append({
        "file_id": file_id,
        "file_name": file_name,
        "upload_time": upload_time
    })
    
    # Save back
    with open(metadata_file, "w") as f:
        json.dump(data, f, indent=2)


def save_file(file_bytes, file_id):
    
    import io
    
    file_obj = io.BytesIO(file_bytes)
    
    s3.upload_fileobj(
        file_obj,
        BUCKET_NAME,
        file_id
    )
    
    print(f"Uploaded to S3: {file_id}")
    
    return file_id

def save_kb_to_s3(kb_data, file_id):
    import io

    # convert JSON to bytes
    json_bytes = json.dumps(kb_data).encode("utf-8")

    file_obj = io.BytesIO(json_bytes)

    s3.upload_fileobj(
        file_obj,
        BUCKET_NAME,
        f"kb/{file_id}.json"   # store inside kb/ folder in S3
    )

    print(f"KB uploaded to S3: kb/{file_id}.json")