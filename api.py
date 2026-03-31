# import time
# from fastapi import FastAPI, UploadFile, File
# from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb
# from storage import save_metadata, save_file
# from notifier import notify_completion

# app = FastAPI()

# @app.post("/upload_pdf")
# async def upload_pdf(file: UploadFile = File(...)):
    
#     # Create unique file_id
#     timestamp = int(time.time())
#     file_id = f"{timestamp}_{file.filename}"

#     # 🔥 Read file ONCE
#     file_bytes = await file.read()

#     # Upload to S3
#     storage_key = save_file(file_bytes, file_id)

#     # Create temp local file for processing
#     temp_file_path = f"temp_{file_id}"

#     with open(temp_file_path, "wb") as f:
#         f.write(file_bytes)
    
#     # Run pipeline
#     text = read_pdf(temp_file_path)
#     chunks = chunk_text(text)
#     embeddings = create_embeddings(chunks)
#     kb = build_kb(chunks, embeddings)
    
#     save_kb(kb, file_id)
#     save_metadata(file_id, file.filename, timestamp)

#     # Notify external system
#     notify_completion(file_id)

#     # Delete temp file
#     import os
#     if os.path.exists(temp_file_path):
#         os.remove(temp_file_path)

#     return {
#         "file_id": file_id,
#         "message": "KB created successfully"
#     }

import time
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks

from storage import save_file, save_metadata
from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb
from notifier import notify_embedding_status

from db import insert_job, get_job, update_job
from vector_db import upsert_embeddings
from vector_db import query_embeddings
from kb_builder import client

app = FastAPI()

# 🔥 BACKGROUND FUNCTION (Step 2B)
def process_pdf(file_bytes, file_id, file_name, job_id, timestamp):
    try:
        # Create temp file
        temp_file_path = f"temp_{file_id}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # Run pipeline
        text = read_pdf(temp_file_path)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        upsert_embeddings(file_id, chunks, embeddings)
        
        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)
        save_metadata(file_id, file_name, int(time.time()))

        # Update job first
        update_job(job_id, "completed", int(time.time()))

        # 🔥 THEN notify
        notify_embedding_status(file_id, job_id, file_id, timestamp)

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        update_job(job_id, "failed", int(time.time()), str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# 🔥 UPDATED API ENDPOINT (Step 2C)
from fastapi import Form

@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_id: str = Form(...),
    file_id: str = Form(...)
):
    timestamp = int(time.time())

    # Read file
    file_bytes = await file.read()

    # Upload to S3
    save_file(file_bytes, file_id)

    # Store job info
    insert_job(job_id, file_id, file.filename, "processing", timestamp)

    # 🔥 Run processing in background
    background_tasks.add_task(process_pdf, file_bytes, file_id, file.filename, job_id, timestamp)

    # Immediate response
    return {
        "job_id": job_id,
        "status": "processing"
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    return job

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_docs(request: QueryRequest):
    
    # Create embedding for query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    )
    
    query_embedding = response.data[0].embedding

    # Search Pinecone
    results = query_embeddings(query_embedding)

    # Extract texts
    matches = []
    for match in results["matches"]:
        matches.append({
            "score": match["score"],
            "text": match["metadata"]["text"]
        })

    return {
        "query": request.query,
        "results": matches
    }