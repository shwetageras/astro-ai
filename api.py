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
from fastapi import Form
from notifier import notify_chart_status
from db import insert_chart_job, update_chart_job
from vector_db import query_chart_embeddings, query_kb_embeddings
from db import insert_qna, update_qna_answer

app = FastAPI()

# BACKGROUND FUNCTION FOR upload_pdf
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
        notify_embedding_status(file_id, job_id, timestamp, file_name)

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        update_job(job_id, "failed", int(time.time()), str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# BACKGROUND FUNCTION FOR upload_chart
def process_chart(file_bytes, file_id, file_name, job_id, chart_id, user_id, profile_id, timestamp):
    try:
        temp_file_path = f"temp_{file_id}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # SAME pipeline as PDF
        text = read_pdf(temp_file_path)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)

        # Add metadata (IMPORTANT)
        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "chart_id": chart_id,
                "user_id": user_id,
                "profile_id": profile_id
            }
        )

        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)

        save_metadata(file_id, file_name, int(time.time()))

        # Update DB
        update_chart_job(job_id, "completed", int(time.time()))

        # 🔥 CALLBACK
        notify_chart_status(job_id, chart_id, file_id)

    except Exception as e:
        print(f"Error in chart job {job_id}: {e}")
        update_chart_job(job_id, "failed", int(time.time()), str(e))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def build_context(chart_results, kb_results):
    context = ""

    # Chart results
    if chart_results.matches:
        for match in chart_results.matches:
            context += match.metadata["text"] + "\n"

    # KB results
    if kb_results.matches:
        for match in kb_results.matches:
            context += match.metadata["text"] + "\n"

    # Limit context
    context = context[:3000]

    return context


def generate_answer(question, context):
    prompt = f"""
You are an expert astrologer.

Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful astrologer."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Create API → /upload_pdf
@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    import time

    timestamp = int(time.time())

    file_id = f"{timestamp}_{file.filename}"
    job_id = f"job_{timestamp}"

    # Read file
    file_bytes = await file.read()

    # Upload to S3
    save_file(file_bytes, file_id)

    # Store job info
    insert_job(job_id, file_id, file.filename, "processing", timestamp)

    # Run processing in background
    background_tasks.add_task(
        process_pdf,
        file_bytes,
        file_id,
        file.filename,
        job_id,
        timestamp
    )

    # Response
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

# Create NEW API → /upload_chart
@app.post("/upload_chart")
async def upload_chart(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Form(...),
    profile_id: int = Form(...),
    chart_id: int = Form(...)
):

    timestamp = int(time.time())

    file_id = f"{timestamp}_{file.filename}"
    job_id = f"chart_{timestamp}"

    file_bytes = await file.read()

    # Save file to S3
    save_file(file_bytes, file_id)

    # Insert into DB (create new function)
    insert_chart_job(job_id, chart_id, user_id, profile_id, file.filename, "processing", timestamp)

    # Background processing
    background_tasks.add_task(
        process_chart,
        file_bytes,
        file_id,
        file.filename,
        job_id,
        chart_id,
        user_id,
        profile_id,
        timestamp
    )

    return {
        "job_id": job_id,
        "status": "processing"
    }


from pydantic import BaseModel

class QuestionRequest(BaseModel):
    user_id: int
    profile_id: int
    chart_id: int
    question: str


@app.post("/ask_question")
def ask_question(request: QuestionRequest):

    # 1. Store question
    qna_id = insert_qna(
        request.user_id,
        request.profile_id,
        request.chart_id,
        request.question
    )

    # 2. Embed question
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = response.data[0].embedding

    # 3. Retrieve context
    chart_results = query_chart_embeddings(
        query_embedding,
        request.user_id,
        request.profile_id,
        request.chart_id
    )

    kb_results = query_kb_embeddings(query_embedding)

    # 4. Build context
    context = build_context(chart_results, kb_results)

    # 5. Generate answer
    answer = generate_answer(request.question, context)

    # 6. Store answer
    update_qna_answer(qna_id, answer)

    # 7. Return answer
    return {
        "answer": answer
    }