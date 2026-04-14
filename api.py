import time
import os
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from openai import OpenAI
from google import genai
from storage import save_file, save_metadata
from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb
from notifier import notify_embedding_status
from db import get_chart_details_bulk, soft_delete_chart_job, get_chart_job
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
from typing import List
from vector_db import query_kb_embeddings_filtered
from dotenv import load_dotenv

load_dotenv()

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

    
    # Handle both cases: list or Pinecone object
    chart_matches = chart_results.matches if hasattr(chart_results, "matches") else chart_results

    # -------- Step 1: Collect all --------
    for match in chart_matches:
        all_chunks.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": "chart"
        })

    if kb_results:
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


def process_chart_text(content, file_id, job_id, chart_id, user_id, profile_id, timestamp):

    try:
        print("PROCESS START:", job_id)

        # Step 1: Chunk
        chunks = chunk_text(content)

        # Step 2: Embeddings
        embeddings = create_embeddings(chunks)

        # Step 3: Store in Pinecone
        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "user_id": user_id,
                "profile_id": profile_id,
                "chart_id": chart_id   
            }
        )

        # Step 4: Build KB (IMPORTANT)
        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)

        # Step 5: Save metadata
        save_metadata(file_id, "chart_text", int(time.time()))

        # Step 6: Update correct DB
        update_chart_job(job_id, "completed", int(time.time()))

        # Step 7: Notify UI
        notify_chart_status(job_id, chart_id, file_id)

        print("PROCESS COMPLETE:", job_id)

    except Exception as e:
        print("❌ ERROR:", str(e))
        update_chart_job(job_id, "failed", int(time.time()), str(e))


from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


class QuestionRequest(BaseModel):
    chart_ids: List[str]
    kb_id: List[str]
    question: str


class DeleteKBRequest(BaseModel):   
    job_id: str


class DeleteChartRequest(BaseModel):
    job_id: str


class GPTChartRequest(BaseModel):
    name: str
    dob: str   # keep string for flexibility (YYYY-MM-DD expected)
    tob: str   # time of birth (HH:MM or HH:MM:SS)
    pob: str   # place of birth
    gender: str


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

        # Store job info
        insert_job(job_id, file_id, name, "processing", timestamp)
        
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
    isCharttype: str = Form(...),
    name: str = Form(...),
    user_id: int = Form(...),
    profile_id: int = Form(...),
    chart_id: int = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None)
):

    import time

    timestamp = int(time.time())

    safe_name = make_safe_filename("chart")
    file_id = f"{timestamp}_{safe_name}"
    job_id = f"job_{timestamp}"

    insert_chart_job(
        job_id,
        chart_id,
        user_id,
        profile_id,
        name,
        "processing",
        timestamp
    )

    # 🔥 CASE 1: TEXT INPUT
    if isCharttype == "article":

        if not content:
            raise HTTPException(status_code=400, detail="Content required for text chart")

        # Start background processing
        background_tasks.add_task(
            process_chart_text,
            content,
            file_id,
            job_id,
            chart_id,   
            user_id,
            profile_id,
            timestamp
        )

    # 🔥 CASE 2: FILE INPUT
    elif isCharttype == "file":

        if not file:
            raise HTTPException(status_code=400, detail="File required for chart upload")
        
        file_bytes = await file.read()

        # Save to S3
        save_file(file_bytes, file_id)

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

    else:
        raise HTTPException(status_code=400, detail="Invalid isCharttype")

    return {
        "job_id": job_id,
        "status": "processing"
    }


@app.post("/ask_question")
def ask_question(request: QuestionRequest):

    chart_ids = request.chart_ids
    kb_ids = request.kb_id

    use_chart = chart_ids and chart_ids != ["0"] and chart_ids != [""]
    use_kb = kb_ids and kb_ids != ["0"] and kb_ids != [""]

    # -------------------------------
    # STEP 1: INIT
    # -------------------------------
    qna_id = None
    chart_details = []
    all_chart_matches = []

    # -------------------------------
    # STEP 2: FETCH CHART + STORE QNA
    # -------------------------------
    if use_chart:
        chart_details = get_chart_details_bulk(chart_ids)

        if chart_details:
            primary_chart = chart_details[0]

            qna_id = insert_qna(
                primary_chart["user_id"],
                primary_chart["profile_id"],
                primary_chart["chart_id"],
                request.question
            )

    # -------------------------------
    # STEP 3: EMBEDDING
    # -------------------------------
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = response.data[0].embedding

    # -------------------------------
    # STEP 4: CHART RETRIEVAL
    # -------------------------------
    if use_chart and chart_details:
        for chart in chart_details:
            results = query_chart_embeddings(
                query_embedding,
                chart["user_id"],
                chart["profile_id"],
                chart["chart_id"],
                top_k=5
            )
            all_chart_matches.extend(results.matches)

    # -------------------------------
    # STEP 5: KB RETRIEVAL
    # -------------------------------
    kb_results = None

    if use_kb:
        if "kbn" in kb_ids:
            kb_results = query_kb_embeddings(query_embedding, top_k=10)
        else:
            kb_results = query_kb_embeddings_filtered(query_embedding, kb_ids, top_k=10)

    # -------------------------------
    # STEP 6: DEBUG
    # -------------------------------
    print("\n================ RETRIEVAL DEBUG ================")

    print("\n--- CHART RESULTS (Merged) ---")
    for match in all_chart_matches:
        print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    print("\n--- KB RESULTS ---")
    if kb_results:
        for match in kb_results.matches:
            print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # -------------------------------
    # STEP 7: CONTEXT BUILDING
    # -------------------------------
    if not use_chart and not use_kb:
        context = ""   # 🔥 PURE LLM MODE
    else:
        context = build_context(all_chart_matches, kb_results)

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])

    # -------------------------------
    # STEP 8: GENERATE ANSWER
    # -------------------------------
    answer = generate_answer(request.question, context)

    # -------------------------------
    # STEP 9: STORE ANSWER
    # -------------------------------
    if qna_id:
        update_qna_answer(qna_id, answer)

    # -------------------------------
    # STEP 10: RESPONSE
    # -------------------------------
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

    if job["status"] == "processing":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete while processing is in progress"
        )

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


@app.post("/delete_chart")
def delete_chart(request: DeleteChartRequest):

    job_id = request.job_id

    # 1. Get chart job
    chart_job = get_chart_job(job_id)

    if not chart_job:
        raise HTTPException(status_code=404, detail="Chart not found")

    if chart_job["status"] == "processing":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete while processing"
        )

    try:
        # 2. Delete embeddings
        delete_embeddings(job_id)

        # 3. Delete file from S3
        from storage import delete_file
        delete_file(job_id)

    except Exception as e:
        print(f"⚠️ Delete error: {e}")

    # 🔥 4. SOFT DELETE (THIS WAS MISSING)
    soft_delete_chart_job(job_id)

    return {
        "status": "success",
        "message": f"Chart deleted for job_id: {job_id}"
    }


client = OpenAI()

@app.post("/create_chart_gpt")
async def create_chart_gpt(
    user_id: int = Form(...),
    profile_id: int = Form(...),
    chart_id: int = Form(...),
    name: str = Form(...),
    dob: str = Form(...),
    tob: str = Form(...),
    pob: str = Form(...),
    country: str = Form(...)
):
    timestamp = int(time.time())
    job_id = f"job_{timestamp}"

    prompt = f"""
    Generate a natal chart interpretation.

    Return ONLY valid JSON in this format:

    {{
    "planets": {{
        "sun": "...",
        "moon": "...",
        "ascendant": "...",
        "mercury": "...",
        "venus": "...",
        "mars": "...",
        "jupiter": "...",
        "saturn": "..."
    }},
    "aspects": [
        "...",
        "...",
        "...",
        "..."
    ],
    "summary": "..."
    }}

    Rules:
    - No explanation
    - No markdown
    - No extra text
    - Keep it concise

    Input:
    Name: {name}
    DOB: {dob}
    TOB: {tob}
    POB: {pob}, {country}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        try:
            content = response.choices[0].message.content
            chart_by_gpt = json.loads(content)

        except Exception as e:
            chart_by_gpt = {
                "error": f"Parsing failed: {str(e)}"
            }

    except Exception as e:
        chart_by_gpt = {
            "error": f"Generation failed: {str(e)}"
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "chart_content": chart_by_gpt,
        "chart_size_chars": len(str(chart_by_gpt)),   # ✅ fixed
        "chart_size_words": len(str(chart_by_gpt).split()),  # ✅ fixed
        "source": {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    }

@app.post("/create_chart_gemini")
async def create_chart_gemini(
    user_id: int = Form(...),
    profile_id: int = Form(...),
    chart_id: int = Form(...),
    name: str = Form(...),
    dob: str = Form(...),
    tob: str = Form(...),
    pob: str = Form(...),
    country: str = Form(...)
):

    timestamp = int(time.time())
    job_id = f"job_{timestamp}"

    prompt = (
        f"Create a natal chart for {name} born on {dob} at {tob} in {pob}, {country}. "
        "Provide at least 4 chart aspects and interpretations."
    )

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        # 🔥 TEMP CHANGE HERE
        chart_by_gemini = "🔥 NEW SERVER WORKING 🔥"

    except Exception as e:
        chart_by_gemini = f"❌ Error generating chart: {str(e)}"
    # try:
    #     # NEW SDK STARTS HERE
    #     client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    #     response = client.models.generate_content(
    #         model="gemini-1.5-flash",
    #         contents=prompt
    #     )

    #     content = response.text

    #     if not content or content.strip() == "":
    #         chart_by_gemini = "⚠️ Empty response from Gemini. Please try again."
    #     else:
    #         chart_by_gemini = content.strip()

    # except Exception as e:
    #     chart_by_gemini = f"❌ Error generating chart: {str(e)}"

    return {
        "job_id": job_id,
        "status": "completed",
        "chart_content": chart_by_gemini,
        "chart_size_chars": len(chart_by_gemini),
        "chart_size_words": len(chart_by_gemini.split()),
        "source": {
            "provider": "google",
            "model": "gemini-1.5-flash"
        }
    }