import time
from fastapi import FastAPI, UploadFile, File
from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb
from storage import save_metadata, save_file
from notifier import notify_completion

app = FastAPI()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    
    # Create unique file_id
    timestamp = int(time.time())
    file_id = f"{timestamp}_{file.filename}"

    # 🔥 Read file ONCE
    file_bytes = await file.read()

    # Upload to S3
    storage_key = save_file(file_bytes, file_id)

    # Create temp local file for processing
    temp_file_path = f"temp_{file_id}"

    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)
    
    # Run pipeline
    text = read_pdf(temp_file_path)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    kb = build_kb(chunks, embeddings)
    
    save_kb(kb, file_id)
    save_metadata(file_id, file.filename, timestamp)

    # Notify external system
    notify_completion(file_id)

    # Delete temp file
    import os
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return {
        "file_id": file_id,
        "message": "KB created successfully"
    }