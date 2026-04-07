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
from fastapi import HTTPException
from vector_db import delete_embeddings
from prompts import build_prompt

app = FastAPI()

def make_safe_filename(name: str):
    return name.replace(" ", "_").replace("/", "_")

# BACKGROUND FUNCTION FOR upload_pdf
def process_pdf(file_bytes, file_id, file_name, job_id, timestamp):
    try:
        # Create temp file
        temp_file_path = f"temp_{file_id}.{file_name.split('.')[-1]}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # Run pipeline
        if "." in file_name:
            file_ext = file_name.split(".")[-1].lower()
        else:
            raise Exception("File has no extension")

        if file_ext == "pdf":
            text = read_pdf(temp_file_path)

        elif file_ext in ["md", "txt"]:
            from kb_builder import read_text_file
            text = read_text_file(temp_file_path)

        else:
            raise Exception(f"Unsupported file type: {file_ext}")
        
        chunks = chunk_text(text)
        print(f"📊 Total chunks created: {len(chunks)}")

        embeddings = create_embeddings(chunks)
        print(f"📊 Total embeddings generated: {len(embeddings)}")
        
        upsert_embeddings(file_id, chunks, embeddings)    # Upsert = Update + Insert (record already exists → UPDATE it, else INSERT it)
        
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
        temp_file_path = f"temp_{file_id}.{file_name.split('.')[-1]}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # SAME pipeline as PDF
        file_ext = file_name.split(".")[-1].lower()

        if file_ext == "pdf":
            text = read_pdf(temp_file_path)

        elif file_ext in ["md", "txt"]:
            from kb_builder import read_text_file
            text = read_text_file(temp_file_path)

        else:
            raise Exception(f"Unsupported file type: {file_ext}")

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

def is_similar(text1, text2, threshold=0.8):
    # Simple similarity using overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    overlap = len(words1 & words2) / max(len(words1), 1)

    return overlap > threshold


def build_context(chart_results, kb_results):

    all_chunks = []

    # -------- Step 1: Collect all --------
    for match in chart_results.matches:
        all_chunks.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": "chart"
        })

    for match in kb_results.matches:
        all_chunks.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": "kb"
        })

    # -------- Step 2: Sort globally --------
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    all_chunks_backup = all_chunks.copy()

    # -------- Step 3: Filter by threshold --------
    SCORE_THRESHOLD = 0.70

    filtered_chunks = [c for c in all_chunks if c["score"] >= SCORE_THRESHOLD]

    # Fallback if nothing passes threshold
    if not filtered_chunks:
        filtered_chunks = all_chunks_backup[:5]   # take top 5 anyway

    all_chunks = filtered_chunks

    # -------- Step 4: De-duplicate --------
    selected = []

    for chunk in all_chunks:
        if not any(is_similar(chunk["text"], s["text"]) for s in selected):
            selected.append(chunk)

        if len(selected) >= 6:   # total context size cap
            break

    # -------- Step 5: Separate again (for structure) --------
    chart_data = [c for c in selected if c["source"] == "chart"]
    kb_data = [c for c in selected if c["source"] == "kb"]

    if not chart_data:
        chart_data = [c for c in all_chunks if c["source"] == "chart"][:2]

    # -------- Step 6: Build structured context --------
    context = ""

    if chart_data:
        context += "CHART DATA:\n"
        for c in chart_data:
            context += f"- {c['text']}\n"

    if kb_data:
        context += "\nKNOWLEDGE BASE:\n"
        for c in kb_data:
            context += f"- {c['text']}\n"

    return context[:3000]


def generate_answer(question, context):

    prompt = build_prompt(question, context)
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a careful and reasoning-based astrologer."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def process_text(text, file_id, file_name, job_id, timestamp):
    try:
        chunks = chunk_text(text)
        print(f"📊 Total chunks created: {len(chunks)}")

        embeddings = create_embeddings(chunks)
        print(f"📊 Total embeddings generated: {len(embeddings)}")

        upsert_embeddings(file_id, chunks, embeddings)

        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)

        save_metadata(file_id, file_name, int(time.time()))

        update_job(job_id, "completed", int(time.time()))

        notify_embedding_status(file_id, job_id, timestamp, file_name)

    except Exception as e:
        print(f"Error processing text job {job_id}: {e}")
        update_job(job_id, "failed", int(time.time()), str(e))

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


class QuestionRequest(BaseModel):
    user_id: int
    profile_id: int
    chart_id: int
    question: str


class DeleteKBRequest(BaseModel):   
    job_id: str


# Create API → /upload_kb
@app.post("/upload_kb")
async def upload_kb(
    background_tasks: BackgroundTasks,
    isKbtype: str = Form(...),
    name: str = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None),
):
    import time

    timestamp = int(time.time())

    safe_name = make_safe_filename(name)
    file_id = f"{timestamp}_{safe_name}"
    job_id = f"job_{timestamp}"

    # Store job info
    insert_job(job_id, file_id, name, "processing", timestamp)

    # 🔥 CASE 1: TEXT INPUT
    if isKbtype == "article":

        if not content:
            raise HTTPException(status_code=400, detail="Content is required for article type")

        background_tasks.add_task(
            process_text,
            content,
            file_id,
            name,
            job_id,
            timestamp
        )

    # 🔥 CASE 2: FILE INPUT
    elif isKbtype == "file":

        if not file:
            raise HTTPException(status_code=400, detail="File is required for file type")

        file_bytes = await file.read()

        # Upload to S3
        save_file(file_bytes, file_id)

        background_tasks.add_task(
            process_pdf,
            file_bytes,
            file_id,
            file.filename,   # USE REAL FILE NAME
            job_id,
            timestamp
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid isKbtype")

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

    safe_name = make_safe_filename(file.filename)
    file_id = f"{timestamp}_{safe_name}"
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
        request.chart_id,
        top_k=10   # 🔥 IMPORTANT
    )

    kb_results = query_kb_embeddings(
        query_embedding,
        top_k=10   # 🔥 IMPORTANT
    )

    # 🔥 DEBUGGING
    print("\n================ RETRIEVAL DEBUG ================")

    print("\n--- CHART RESULTS (Raw) ---")
    for match in chart_results.matches:
        print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    print("\n--- KB RESULTS (Raw) ---")
    for match in kb_results.matches:
        print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # 4. Build context
    context = build_context(chart_results, kb_results)

    print("\n--- FINAL CONTEXT (After Filtering & Selection) ---")
    print(context[:1000])

    # Count Summary
    print(f"\nSelected Context Length: {len(context)} chars")

    # 5. Generate answer
    answer = generate_answer(request.question, context)

    # 6. Store answer
    update_qna_answer(qna_id, answer)

    # 7. Return answer
    return {
        "answer": answer
    }


@app.post("/delete_kb")
def delete_kb(request: DeleteKBRequest):

    job_id = request.job_id

    # 1. Get job
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_id = job["file_id"]

    try:
        # 2. Delete embeddings (Pinecone)
        print(f"🧹 Deleting embeddings for file_id: {file_id}")
        delete_embeddings(file_id)
        print(f"✅ Embeddings deleted for file_id: {file_id}")

        # 3. Delete file from S3
        from storage import delete_file
        delete_file(file_id)

    except Exception as e:
        print(f"⚠️ Delete error (continuing): {e}")

    # 🔥 ALWAYS update DB (no matter what)
    update_job(job_id, "deleted", int(time.time()))

    return {
        "status": "success",
        "message": f"KB deleted for job_id: {job_id}"
    }