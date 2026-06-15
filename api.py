import time
import uuid
import os
import json
import requests
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from settings import OPENAI_MODEL, OPENAI_MINI_MODEL
# from google import genai
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
from typing import List, Optional
from vector_db import query_kb_embeddings_filtered
from dotenv import load_dotenv
import google.generativeai as genai
from db import insert_qna_sl
from kb_builder import client
from prompts import build_prompt
from db import update_qna_sl_validation, get_qna_sl
from db import mark_qna_ml_ready
from pydantic import BaseModel
from vector_db import query_qna_sl_embeddings
from qna_generator import generate_qnas
from db import delete_qna_record
from vector_db import delete_qna_embeddings
from vector_db import get_all_kb_chunks
# from prompts import build_welcome_prompt
from vector_db import index


# load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

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
# def process_chart(file_bytes, file_id, file_name, job_id, chart_id, user_id, profile_id, timestamp):
    
#     print("🚀 PROCESS_CHART STARTED", flush=True)

#     try:
#         temp_file_path = f"temp_{file_id}.{file_name.split('.')[-1]}"

#         with open(temp_file_path, "wb") as f:
#             f.write(file_bytes)

#         # SAME pipeline as PDF
#         file_ext = file_name.split(".")[-1].lower()

#         if file_ext == "pdf":
#             text = read_pdf(temp_file_path)

#         elif file_ext in ["md", "txt"]:
#             from kb_builder import read_text_file
#             text = read_text_file(temp_file_path)

#         else:
#             raise Exception(f"Unsupported file type: {file_ext}")

#         chunks = chunk_text(text)
#         print("✅ CHUNKS:", len(chunks))

#         embeddings = create_embeddings(chunks)
#         print("✅ EMBEDDINGS:", len(embeddings))

#         # Add metadata (IMPORTANT)
#         upsert_embeddings(
#             file_id,
#             chunks,
#             embeddings,
#             metadata={
#                 "chart_id": str(chart_id),
#                 "user_id": str(user_id),
#                 "profile_id": str(profile_id)
#             }
#         )

#         print("✅ UPSERT DONE")

#         kb = build_kb(chunks, embeddings)
#         save_kb(kb, file_id)

#         save_metadata(file_id, file_name, int(time.time()))

#         # Update DB
#         update_chart_job(job_id, "completed", int(time.time()))

#         # 🔥 CALLBACK
#         notify_chart_status(job_id, chart_id, file_id)

#     except Exception as e:
#         print(f"Error in chart job {job_id}: {e}")
#         update_chart_job(job_id, "failed", int(time.time()), str(e))

#     finally:
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

def process_chart(file_bytes, file_id, file_name, job_id, chart_id, user_id, profile_id, timestamp):

    print("🚀 PROCESS_CHART STARTED", flush=True)

    try:
        temp_file_path = f"temp_{file_id}.{file_name.split('.')[-1]}"
        print("FILE SAVING START")

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        print("FILE SAVED")

        file_ext = file_name.split(".")[-1].lower()
        print("FILE TYPE:", file_ext)

        if file_ext == "pdf":

            text = read_pdf(temp_file_path)

        elif file_ext in ["md", "txt"]:

            from kb_builder import read_text_file
            text = read_text_file(temp_file_path)

        elif file_ext == "json":

            with open(temp_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            print("TOP LEVEL KEYS:")
            print(list(json_data.keys()))

            print("TOTAL TOP LEVEL KEYS:", len(json_data.keys()))

            text = json.dumps(
                json_data,
                ensure_ascii=False,
                indent=2
            )

        else:

            raise Exception(
                f"Unsupported file type: {file_ext}"
            )

        print("TEXT EXTRACTED")
        print("TEXT LENGTH:", len(text))

        chunks = chunk_text(text)

        print("TOTAL CHUNKS:", len(chunks))

        for i in range(min(5, len(chunks))):
            print(f"\n========== CHUNK {i+1} ==========")
            print(chunks[i][:1000])

        embeddings = create_embeddings(chunks)
        print("EMBEDDINGS:", len(embeddings))

        print("UPSERT METADATA:")
        print({
            "chart_id": str(chart_id),
            "user_id": str(user_id),
            "profile_id": str(profile_id)
        })

        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "chart_id": str(chart_id),
                "user_id": str(user_id),
                "profile_id": str(profile_id)
            }
        )

        print("UPSERT DONE")

        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)
        print("KB SAVED")

        save_metadata(file_id, file_name, int(time.time()))
        print("METADATA SAVED")

        update_chart_job(job_id, "completed", int(time.time()))
        print("DB UPDATED")

        notify_chart_status(job_id, chart_id, file_id)
        print("CALLBACK SENT")

    except Exception as e:
        print(f"❌ ERROR in chart job {job_id}: {e}")

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

    # if kb_results:
    #     for match in kb_results.matches:
    #         all_chunks.append({
    #             "score": match.score,
    #             "text": match.metadata.get("text", ""),
    #             "source": "kb"
    #         })

    if kb_results:

        kb_matches = (
            kb_results.matches
            if hasattr(kb_results, "matches")
            else kb_results
        )

        for match in kb_matches:
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
        filtered_chunks = all_chunks_backup[:15]   # take top 5 anyway

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

    print("\n===== FINAL CONTEXT SENT TO GPT =====")
    print(context)
    print("===== END =====")

    return context[:3000]

def get_max_tokens(question):

    q = question.lower()

    # Detailed analysis requests
    if any(word in q for word in [
        "detailed", "deep", "analyze", "complete", "full"
    ]):
        return 450

    # Default conversational mode
    return 220


def generate_answer(question, context):

    prompt = build_prompt(question, context)
    
    max_tokens_value = get_max_tokens(question)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=max_tokens_value,
        messages=[
            {"role": "system", "content": "You are a careful and reasoning-based astrologer."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def generate_answer_gpt_mini(question, context):

    prompt = build_prompt(question, context)

    max_tokens_value = get_max_tokens(question)

    response = client.chat.completions.create(
        model=OPENAI_MINI_MODEL,
        temperature=0.2,
        max_tokens=max_tokens_value,
        messages=[
            {
                "role": "system",
                "content": "You are a careful and reasoning-based astrologer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# def generate_welcome_message(context, user_name="User"):

#     prompt = build_welcome_prompt(context, user_name)

#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         temperature=0.7,
#         max_tokens=180,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a warm and thoughtful Vedic astrologer."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     return (response.choices[0].message.content or "").strip()


def generate_answer_gemini(question, context):
    prompt = build_prompt(question, context)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        from typing import cast

        res_text = getattr(response, "text", "")

        if res_text:
            return cast(str, res_text).strip()

        if hasattr(response, "candidates") and response.candidates:
            try:
                candidate_text = response.candidates[0].content.parts[0].text
                if isinstance(candidate_text, str):
                    return candidate_text.strip()
            except (AttributeError, IndexError):
                pass

        return "⚠️ Empty Gemini response"

    except Exception as e:
        print("❌ GEMINI ERROR:", str(e))
        return f"Gemini error: {str(e)}"


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


# def process_chart_text(content, file_id, job_id, chart_id, user_id, profile_id, timestamp):

#     print("PROCESS_CHART STARTED", flush=True)

#     try:
#         print("PROCESS START:", job_id)

#         # Step 1: Chunk
#         chunks = chunk_text(content)
#         print("✅ CHUNKS:", len(chunks))

#         # Step 2: Embeddings
#         embeddings = create_embeddings(chunks)
#         print("✅ EMBEDDINGS:", len(embeddings))

#         # Step 3: Store in Pinecone
#         upsert_embeddings(
#             file_id,
#             chunks,
#             embeddings,
#             metadata={
#                 "chart_id": str(chart_id),
#                 "user_id": str(user_id),
#                 "profile_id": str(profile_id)
#             }
#         )

#         print("✅ UPSERT DONE")

#         # Step 4: Build KB (IMPORTANT)
#         kb = build_kb(chunks, embeddings)
#         save_kb(kb, file_id)

#         # Step 5: Save metadata
#         save_metadata(file_id, "chart_text", int(time.time()))

#         # Step 6: Update correct DB
#         update_chart_job(job_id, "completed", int(time.time()))

#         # Step 7: Notify UI
#         notify_chart_status(job_id, chart_id, file_id)

#         print("PROCESS COMPLETE:", job_id)

#     except Exception as e:
#         print("❌ ERROR:", str(e))
#         update_chart_job(job_id, "failed", int(time.time()), str(e))


def process_chart_text(content, file_id, job_id, chart_id, user_id, profile_id, timestamp):

    print("PROCESS_CHART_TEXT STARTED", flush=True)

    try:
        print("CONTENT RECEIVED")

        print("\n========== RAW CHART CONTENT ==========")
        print(content[:10000])
        print("=======================================")

        # Step 1: Chunk
        chunks = chunk_text(content)
        print("CHUNKS:", len(chunks))

        # Step 2: Embeddings
        embeddings = create_embeddings(chunks)
        print("EMBEDDINGS:", len(embeddings))

        # Step 3: Store in Pinecone
        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "user_id": str(user_id),
                "profile_id": str(profile_id),
                "chart_id": str(chart_id)
            }
        )
        print("UPSERT DONE")

        # Step 4: Build KB
        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)
        print("KB SAVED")

        # Step 5: Metadata
        save_metadata(file_id, "chart_text", int(time.time()))
        print("METADATA SAVED")

        # Step 6: DB update
        update_chart_job(job_id, "completed", int(time.time()))
        print("DB UPDATED")

        # Step 7: Callback
        notify_chart_status(job_id, chart_id, file_id)
        print("CALLBACK SENT")

        print("PROCESS COMPLETE:", job_id)

    except Exception as e:
        print(f"❌ ERROR in chart text job {job_id}: {e}")
        update_chart_job(job_id, "failed", int(time.time()), str(e))

# -----------------------------------
# EXACT KB MATCH HELPER
# -----------------------------------

def find_exact_kb_match(kb_id, question):

    print("========== NEW FIND_EXACT_KB_MATCH RUNNING ==========")
    if not question.lower().startswith("what is"):
        return None

    chunks = get_all_kb_chunks(kb_id)

    search_term = (
        question.lower()
        .replace("what is", "")
        .replace("?", "")
        .strip()
    )

    print("SEARCH TERM:", search_term)

    best_match = None
    best_score = -1

    for chunk in chunks:

        text = chunk.metadata.get("text", "")
        text_lower = text.lower()

        if search_term not in text_lower:
            continue

        position = text_lower.find(search_term)

        print("\n====================")
        print("POSITION:", position)

        start = max(0, position - 100)
        end = min(len(text), position + 300)

        print(text[start:end])

        score = 0

        score += max(0, 1000 - position)
        
        definition_pos = text_lower.find("definition")

        if definition_pos >= 0 and abs(definition_pos - position) < 200:
            score += 1000

        if position >= 0:
            score += max(0, 500 - position)

        print(
            f"CANDIDATE score={score} pos={position} def={definition_pos}"
        )
        print(text[:250])

        if score > best_score:
            best_score = score
            best_match = chunk

    if best_match:
        print("BEST SCORE:", best_score)
        print("BEST MATCH:", best_match.metadata.get("text", "")[:300])

    return best_match

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


class QuestionRequest(BaseModel):
    chart_ids: List[str]
    kb_id: List[str]
    sl_id: List[str] = []
    question: str

    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None


class WelcomeRequest(BaseModel):
    question: str
    chart_ids: List[str] = []
    profile_id: int | None = None


class DeleteKBRequest(BaseModel):   
    job_id: str


class DeleteChartRequest(BaseModel):
    job_id: str


class QnaSLRequest(BaseModel):
    kb_id: str
    question: str


class QnaSLValidationRequest(BaseModel):
    source_type: str
    qna_id: int
    is_valid: bool
    corrected_answer: Optional[str] = None


class QnaMLRequest(BaseModel):
    qna_ids: List[int]


class QnaSearchRequest(BaseModel):
    question: str
    kb_id: str


class QnaGenerateRequest(BaseModel):
    kb_id: str


class DeleteQnaSLRequest(BaseModel):
    qna_id: int


# class RetrievalTestRequest(BaseModel):
#     question: str

#     kb_ids: Optional[list[str]] = None

#     user_id: Optional[str] = None
#     profile_id: Optional[str] = None
#     chart_id: Optional[str] = None

#     include_kb: bool = True
#     include_chart: bool = False
#     include_sl: bool = False

#     top_k: int = 50




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
    
    job_id = f"job_{timestamp}_{uuid.uuid4().hex}"

    # 🔥 CASE 1: TEXT INPUT
    if isKbtype == "article":

        if not content:
            raise HTTPException(status_code=400, detail="Content is required for article type")

        # STEP 1
        insert_job(job_id, file_id, name, "processing", timestamp)

        # STEP 2
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
    print("========== UPLOAD CHART HIT ==========")


    timestamp = int(time.time())

    safe_name = make_safe_filename("chart")
    file_id = f"{timestamp}_{uuid.uuid4().hex}_{safe_name}"

    job_id = f"job_{timestamp}_{uuid.uuid4().hex}"

    insert_chart_job(
        job_id,
        file_id,
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

        print("FILE UPLOAD MODE")

        if not file:
            raise HTTPException(status_code=400, detail="File required for chart upload")

        print("FILE NAME:", file.filename)

        file_bytes = await file.read()

        print(
            "FILE SIZE MB:",
            round(len(file_bytes) / (1024 * 1024), 2)
        )

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
    
    start_time = time.time()

    chart_ids = request.chart_ids
    
    kb_ids = request.kb_id

    # normalize safely
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    elif not isinstance(kb_ids, list):
        kb_ids = []

    kb_id = kb_ids[0] if kb_ids else ""

    print("KB IDS:", kb_ids)
    print("KB ID USED:", kb_id)

    use_chart = chart_ids and chart_ids != ["0"] and chart_ids != [""]
    use_kb = kb_ids and kb_ids != ["0"] and kb_ids != [""]
    use_sl = request.sl_id and request.sl_id != ["0"] and request.sl_id != [""]

    print("SL IDS:", request.sl_id)
    print("USE SL:", use_sl)

    # -------------------------------
    # STEP 0: Initialize SL vars
    # -------------------------------
    use_sl_as_context = False
    
    ttl_q_embedding = 0
    ttl_sl_match = 0
    ttl_kb_context = 0
    ttl_chart_context = 0
    ttl_prompt = 0
    ttl_reasoning = 0
    ttl_delivery = 0

    # -------------------------------
    # STEP 0.1: Check SL memory
    # -------------------------------
    if use_sl:

        sl_start = time.perf_counter()

        sl_result = qna_sl_search(
            QnaSearchRequest(
                question=request.question,
                kb_id=kb_id
            )
        )

        ttl_sl_match = round(
            (time.perf_counter() - sl_start) * 1000,
            2
        )

        sl_found = sl_result.get("found")
        matches = sl_result.get("matches", [])

        if not matches:
            sl_found = False
            sl_score = None
            second = None
        else:
            best = matches[0]
            second = matches[1] if len(matches) > 1 else None
            sl_score = best["score"]

    else:

        print("SL DISABLED")

        sl_found = False
        matches = []
        sl_score = None
        second = None
        
    # 🔍 DEBUG PRINTS
    print("SL FOUND:", sl_found)
    print("FIRST MATCH SCORE:", sl_score)
    print("SECOND MATCH SCORE:", second["score"] if second else None)
    print("MATCH COUNT:", len(matches))
    print("MATCH SCORES (top 5):", [round(m["score"], 2) for m in matches[:5]])

    # -------------------------
    # STEP 0.2: Decision logic 
    # -------------------------
    use_sl_as_context = False
    sl_context = None

    if sl_found and sl_score is not None:

        # 🟢 STRONG + 🟡 MEDIUM → BOTH go to context (NO DIRECT REUSE)
        if sl_score >= 0.60:

            selected_matches = [
                m for m in sorted(matches, key=lambda x: x["score"], reverse=True)
                if m["score"] >= 0.60 and m.get("answer")
            ][:3]

            formatted_matches = []

            for m in selected_matches:

                answer = m.get("answer") or ""
                question = m.get("question") or ""

                short_answer = (
                    answer[:300].rsplit(' ', 1)[0] + "..."
                    if len(answer) > 300
                    else answer
                )

                formatted_matches.append(
                    f"Previous QnA (score {round(m['score'],2)}):\n"
                    f"Q: {question}\n"
                    f"A: {short_answer}"
                )

            sl_context = "\n\n".join(formatted_matches)

            use_sl_as_context = True

        # 🔴 LOW → ignore
        else:
            use_sl_as_context = False
            sl_context = None

    # 🔍 DEBUG
    print("USING SL CONTEXT:", use_sl_as_context)

    # chart_ids = request.chart_ids
    # kb_ids = request.kb_id

    # # Safety: ensure list
    # if isinstance(chart_ids, str):
    #     chart_ids = [chart_ids]

    # if isinstance(kb_ids, str):
    #     kb_ids = [kb_ids]

    # def is_valid_ids(ids):
    #     return ids and ids != ["0"] and ids != [""]

    # use_chart = is_valid_ids(chart_ids)
    # use_kb = is_valid_ids(kb_ids)


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
    embedding_start = time.perf_counter()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )

    ttl_q_embedding = round(
        (time.perf_counter() - embedding_start) * 1000,
        2
    )
    query_embedding = response.data[0].embedding

    # -------------------------------
    # STEP 4: CHART RETRIEVAL
    # -------------------------------
    chart_start = time.perf_counter()

    if use_chart and chart_details:
        for chart in chart_details:
            results = query_chart_embeddings(
                query_embedding,
                chart["user_id"],
                chart["profile_id"],
                chart["chart_id"],
                top_k=10
            )
            all_chart_matches.extend(results.matches)

    ttl_chart_context = round(
        (time.perf_counter() - chart_start) * 1000,
        2
    )

    # -------------------------------
    # TEST EXACT MATCH
    # -------------------------------

    if kb_ids:

        test_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        print("TEST MATCH FOUND:", test_match is not None)

    # -------------------------------
    # STEP 5: KB RETRIEVAL
    # -------------------------------
    # kb_results = None

    # if use_kb:
    #     if "job_n" in kb_ids:
    #         kb_results = query_kb_embeddings(query_embedding, top_k=10)
    #     else:
    #         kb_results = query_kb_embeddings_filtered(query_embedding, kb_ids, top_k=10)

    kb_results = None

    kb_start = time.perf_counter()

    if use_kb:

        exact_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        if exact_match:

            print("USING EXACT MATCH")

            kb_results = [exact_match]

        else:

            print("USING VECTOR SEARCH")

            kb_results = query_kb_embeddings_filtered(
                query_embedding,
                kb_ids,
                top_k=15
            )

    ttl_kb_context = round(
        (time.perf_counter() - kb_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 6: DEBUG
    # -------------------------------
    # print("\n================ RETRIEVAL DEBUG ================")

    # print("\n--- CHART RESULTS (Merged) ---")
    # for match in all_chart_matches:
    #     print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # print("\n--- KB RESULTS ---")
    # if kb_results:
    #     for match in kb_results.matches:
    #         print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    print("\n--- KB RESULTS ---")

    if kb_results:

        if isinstance(kb_results, list):
            for match in kb_results:
                print(f"Score: EXACT_MATCH | {match.metadata.get('text', '')[:100]}")
        else:
            for match in kb_results.matches:
                print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # -------------------------------
    # STEP 7: CONTEXT BUILDING
    # -------------------------------
    prompt_start = time.perf_counter()

    if not use_chart and not use_kb:
        context = ""   # 🔥 PURE LLM MODE
    else:
        context = build_context(all_chart_matches, kb_results)

    # -------------------------------
    # Inject SL context (if medium confidence)
    # -------------------------------
    print("INJECTING SL INTO CONTEXT:", use_sl_as_context)

    if use_sl_as_context and sl_context:

        context = (
            "Relevant past learned answers (use as base, refine and synthesize):\n"
            f"{sl_context}\n\n"
            f"{context}"
        )

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])


    # -------------------------------
    # Inject previous conversation
    # -------------------------------
    if request.previous_question and request.previous_answer:

        conversation_context = f"""
    PREVIOUS CONVERSATION:

    User: {request.previous_question}

    Astrologer: {request.previous_answer}
    """

        context = f"{conversation_context}\n\n{context}"

    ttl_context_build = round(
        (time.perf_counter() - prompt_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 8: GENERATE ANSWER
    # -------------------------------
    reasoning_start = time.perf_counter()

    answer = generate_answer(
        request.question,
        context
    )

    ttl_reasoning = round(
        (time.perf_counter() - reasoning_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 9: STORE ANSWER
    # -------------------------------
    delivery_start = time.perf_counter()

    if qna_id:
        update_qna_answer(qna_id, answer)

    ttl_delivery = round(
        (time.perf_counter() - delivery_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 10: RESPONSE
    # -------------------------------
    c_rttl = round(
        (time.time() - start_time) * 1000,
        2
    )

    ttl_breakdown = [
        {
            "stage": "ttl_q_embedding",
            "time_ms": ttl_q_embedding
        },
        {
            "stage": "ttl_sl_match",
            "time_ms": ttl_sl_match
        },
        {
            "stage": "ttl_kb_context",
            "time_ms": ttl_kb_context
        },
        {
            "stage": "ttl_chart_context",
            "time_ms": ttl_chart_context
        },
        {
            "stage": "ttl_context_build",
            "time_ms": ttl_prompt
        },
        {
            "stage": "ttl_reasoning",
            "time_ms": ttl_reasoning
        },
        {
            "stage": "ttl_delivery",
            "time_ms": ttl_delivery
        }
    ]

    print("TOTAL RTTL (ms):", c_rttl)

    print("TTL BREAKDOWN:")
    print(ttl_breakdown)

    return {
        "source": "gpt-4.1",

        "used_sl": use_sl_as_context,
        "used_kb": use_kb,
        "used_chart": use_chart,

        "rttl": c_rttl,

        "c_ttl": ttl_breakdown,

        "answer": answer
    }

@app.post("/qna_gpt_mini")
def qna_gpt_mini(request: QuestionRequest):

    start_time = time.time()

    chart_ids = request.chart_ids
    
    kb_ids = request.kb_id

    # normalize safely
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    elif not isinstance(kb_ids, list):
        kb_ids = []

    kb_id = kb_ids[0] if kb_ids else ""

    print("KB IDS:", kb_ids)
    print("KB ID USED:", kb_id)

    use_chart = chart_ids and chart_ids != ["0"] and chart_ids != [""]
    use_kb = kb_ids and kb_ids != ["0"] and kb_ids != [""]
    use_sl = request.sl_id and request.sl_id != ["0"] and request.sl_id != [""]

    print("SL IDS:", request.sl_id)
    print("USE SL:", use_sl)

    # -------------------------------
    # STEP 0: Initialize SL vars
    # -------------------------------
    use_sl_as_context = False
    
    # -------------------------------
    # STEP 0.1: Check SL memory
    # -------------------------------
    if use_sl:

        sl_result = qna_sl_search(
            QnaSearchRequest(
                question=request.question,
                kb_id=kb_id
            )
        )

        sl_found = sl_result.get("found")
        matches = sl_result.get("matches", [])

        if not matches:
            sl_found = False
            sl_score = None
            second = None
        else:
            best = matches[0]
            second = matches[1] if len(matches) > 1 else None
            sl_score = best["score"]

    else:

        print("SL DISABLED")

        sl_found = False
        matches = []
        sl_score = None
        second = None
        
    # 🔍 DEBUG PRINTS
    print("SL FOUND:", sl_found)
    print("FIRST MATCH SCORE:", sl_score)
    print("SECOND MATCH SCORE:", second["score"] if second else None)
    print("MATCH COUNT:", len(matches))
    print("MATCH SCORES (top 5):", [round(m["score"], 2) for m in matches[:5]])

    # -------------------------
    # STEP 0.2: Decision logic 
    # -------------------------
    use_sl_as_context = False
    sl_context = None

    if sl_found and sl_score is not None:

        # 🟢 STRONG + 🟡 MEDIUM → BOTH go to context (NO DIRECT REUSE)
        if sl_score >= 0.60:

            selected_matches = [
                m for m in sorted(matches, key=lambda x: x["score"], reverse=True)
                if m["score"] >= 0.60 and m.get("answer")
            ][:3]

            formatted_matches = []

            for m in selected_matches:

                answer = m.get("answer") or ""
                question = m.get("question") or ""

                short_answer = (
                    answer[:300].rsplit(' ', 1)[0] + "..."
                    if len(answer) > 300
                    else answer
                )

                formatted_matches.append(
                    f"Previous QnA (score {round(m['score'],2)}):\n"
                    f"Q: {question}\n"
                    f"A: {short_answer}"
                )

            sl_context = "\n\n".join(formatted_matches)

            use_sl_as_context = True

        # 🔴 LOW → ignore
        else:
            use_sl_as_context = False
            sl_context = None

    # 🔍 DEBUG
    print("USING SL CONTEXT:", use_sl_as_context)

    # chart_ids = request.chart_ids
    # kb_ids = request.kb_id

    # # Safety: ensure list
    # if isinstance(chart_ids, str):
    #     chart_ids = [chart_ids]

    # if isinstance(kb_ids, str):
    #     kb_ids = [kb_ids]

    # def is_valid_ids(ids):
    #     return ids and ids != ["0"] and ids != [""]

    # use_chart = is_valid_ids(chart_ids)
    # use_kb = is_valid_ids(kb_ids)


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
                top_k=10
            )
            all_chart_matches.extend(results.matches)

    # -------------------------------
    # TEST EXACT MATCH
    # -------------------------------

    if kb_ids:

        test_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        print("TEST MATCH FOUND:", test_match is not None)

    # -------------------------------
    # STEP 5: KB RETRIEVAL
    # -------------------------------
    # kb_results = None

    # if use_kb:
    #     if "job_n" in kb_ids:
    #         kb_results = query_kb_embeddings(query_embedding, top_k=10)
    #     else:
    #         kb_results = query_kb_embeddings_filtered(query_embedding, kb_ids, top_k=10)

    kb_results = None

    if use_kb:

        exact_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        if exact_match:

            print("USING EXACT MATCH")

            kb_results = [exact_match]

        else:

            print("USING VECTOR SEARCH")

            kb_results = query_kb_embeddings_filtered(
                query_embedding,
                kb_ids,
                top_k=15
            )

    # -------------------------------
    # STEP 6: DEBUG
    # -------------------------------
    # print("\n================ RETRIEVAL DEBUG ================")

    # print("\n--- CHART RESULTS (Merged) ---")
    # for match in all_chart_matches:
    #     print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # print("\n--- KB RESULTS ---")
    # if kb_results:
    #     for match in kb_results.matches:
    #         print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    print("\n--- KB RESULTS ---")

    if kb_results:

        if isinstance(kb_results, list):
            for match in kb_results:
                print(f"Score: EXACT_MATCH | {match.metadata.get('text', '')[:100]}")
        else:
            for match in kb_results.matches:
                print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # -------------------------------
    # STEP 7: CONTEXT BUILDING
    # -------------------------------
    if not use_chart and not use_kb:
        context = ""   # 🔥 PURE LLM MODE
    else:
        context = build_context(all_chart_matches, kb_results)

    # -------------------------------
    # Inject SL context (if medium confidence)
    # -------------------------------
    print("INJECTING SL INTO CONTEXT:", use_sl_as_context)

    if use_sl_as_context and sl_context:

        context = (
            "Relevant past learned answers (use as base, refine and synthesize):\n"
            f"{sl_context}\n\n"
            f"{context}"
        )

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])


    # -------------------------------
    # Inject previous conversation
    # -------------------------------
    if request.previous_question and request.previous_answer:

        conversation_context = f"""
    PREVIOUS CONVERSATION:

    User: {request.previous_question}

    Astrologer: {request.previous_answer}
    """

        context = f"{conversation_context}\n\n{context}"

    # -------------------------------
    # STEP 8: GENERATE ANSWER
    # -------------------------------
    print("MODEL USED: GPT-4.1-MINI")

    answer = generate_answer_gpt_mini(
        request.question,
        context
    )

    # -------------------------------
    # STEP 9: STORE ANSWER
    # -------------------------------
    if qna_id:
        update_qna_answer(qna_id, answer)

    # -------------------------------
    # STEP 10: RESPONSE
    # -------------------------------
    c_rttl = round(
        (time.time() - start_time) * 1000,
        2
    )

    print("TOTAL RTTL (ms):", c_rttl)

    return {
        "source": "gpt-4.1-mini",
        "used_sl": use_sl_as_context,
        "used_kb": use_kb,
        "used_chart": use_chart,
        "rttl": c_rttl,
        "answer": answer
    }


@app.post("/qna_gemini")
def qna_gemini(request: QuestionRequest):

    start_time = time.time()

    chart_ids = request.chart_ids
    
    kb_ids = request.kb_id

    # normalize safely
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    elif not isinstance(kb_ids, list):
        kb_ids = []

    kb_id = kb_ids[0] if kb_ids else ""

    print("KB IDS:", kb_ids)
    print("KB ID USED:", kb_id)

    use_chart = chart_ids and chart_ids != ["0"] and chart_ids != [""]
    use_kb = kb_ids and kb_ids != ["0"] and kb_ids != [""]
    use_sl = request.sl_id and request.sl_id != ["0"] and request.sl_id != [""]

    print("SL IDS:", request.sl_id)
    print("USE SL:", use_sl)

    # -------------------------------
    # STEP 0: Initialize SL vars
    # -------------------------------
    use_sl_as_context = False

    ttl_q_embedding = 0
    ttl_sl_match = 0
    ttl_kb_context = 0
    ttl_chart_context = 0
    ttl_prompt = 0
    ttl_reasoning = 0
    ttl_delivery = 0
    
    # -------------------------------
    # STEP 0.1: Check SL memory
    # -------------------------------
    if use_sl:

        sl_start = time.perf_counter()

        sl_result = qna_sl_search(
            QnaSearchRequest(
                question=request.question,
                kb_id=kb_id
            )
        )

        print("SL RESULT RAW:", sl_result)

        ttl_sl_match = round(
            (time.perf_counter() - sl_start) * 1000,
            2
        )

        sl_found = sl_result.get("found")
        matches = sl_result.get("matches", [])

        if not matches:
            sl_found = False
            sl_score = None
            second = None
        else:
            best = matches[0]
            second = matches[1] if len(matches) > 1 else None
            sl_score = best["score"]

    else:

        print("SL DISABLED")

        sl_found = False
        matches = []
        sl_score = None
        second = None
        
    # 🔍 DEBUG PRINTS
    print("SL FOUND:", sl_found)
    print("FIRST MATCH SCORE:", sl_score)
    print("SECOND MATCH SCORE:", second["score"] if second else None)
    print("MATCH COUNT:", len(matches))
    print("MATCH SCORES (top 5):", [round(m["score"], 2) for m in matches[:5]])

    # -------------------------
    # STEP 0.2: Decision logic 
    # -------------------------
    use_sl_as_context = False
    sl_context = None

    if sl_found and sl_score is not None:

        # 🟢 STRONG + 🟡 MEDIUM → BOTH go to context (NO DIRECT REUSE)
        if sl_score >= 0.60:

            selected_matches = [
                m for m in sorted(matches, key=lambda x: x["score"], reverse=True)
                if m["score"] >= 0.60 and m.get("answer")
            ][:3]

            formatted_matches = []

            for m in selected_matches:

                answer = m.get("answer") or ""
                question = m.get("question") or ""

                short_answer = (
                    answer[:300].rsplit(' ', 1)[0] + "..."
                    if len(answer) > 300
                    else answer
                )

                formatted_matches.append(
                    f"Previous QnA (score {round(m['score'],2)}):\n"
                    f"Q: {question}\n"
                    f"A: {short_answer}"
                )

            sl_context = "\n\n".join(formatted_matches)

            use_sl_as_context = True

        # 🔴 LOW → ignore
        else:
            use_sl_as_context = False
            sl_context = None

    # 🔍 DEBUG
    print("USING SL CONTEXT:", use_sl_as_context)

    # chart_ids = request.chart_ids
    # kb_ids = request.kb_id

    # # Safety: ensure list
    # if isinstance(chart_ids, str):
    #     chart_ids = [chart_ids]

    # if isinstance(kb_ids, str):
    #     kb_ids = [kb_ids]

    # def is_valid_ids(ids):
    #     return ids and ids != ["0"] and ids != [""]

    # use_chart = is_valid_ids(chart_ids)
    # use_kb = is_valid_ids(kb_ids)


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
    embedding_start = time.perf_counter()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )

    ttl_q_embedding = round(
        (time.perf_counter() - embedding_start) * 1000,
        2
    )
    query_embedding = response.data[0].embedding

    # -------------------------------
    # STEP 4: CHART RETRIEVAL
    # -------------------------------
    chart_start = time.perf_counter()
    
    if use_chart and chart_details:
        for chart in chart_details:
            results = query_chart_embeddings(
                query_embedding,
                chart["user_id"],
                chart["profile_id"],
                chart["chart_id"],
                top_k=10
            )
            all_chart_matches.extend(results.matches)

    ttl_chart_context = round(
        (time.perf_counter() - chart_start) * 1000,
        2
    )

    # -------------------------------
    # TEST EXACT MATCH
    # -------------------------------

    if kb_ids:

        test_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        print("TEST MATCH FOUND:", test_match is not None)

    # -------------------------------
    # STEP 5: KB RETRIEVAL
    # -------------------------------
    # kb_results = None

    # if use_kb:
    #     if "job_n" in kb_ids:
    #         kb_results = query_kb_embeddings(query_embedding, top_k=10)
    #     else:
    #         kb_results = query_kb_embeddings_filtered(query_embedding, kb_ids, top_k=10)

    kb_results = None

    kb_start = time.perf_counter()

    if use_kb:

        exact_match = find_exact_kb_match(
            kb_ids[0],
            request.question
        )

        if exact_match:

            print("USING EXACT MATCH")

            kb_results = [exact_match]

        else:

            print("USING VECTOR SEARCH")

            kb_results = query_kb_embeddings_filtered(
                query_embedding,
                kb_ids,
                top_k=15
            )

    ttl_kb_context = round(
        (time.perf_counter() - kb_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 6: DEBUG
    # -------------------------------
    # print("\n================ RETRIEVAL DEBUG ================")

    # print("\n--- CHART RESULTS (Merged) ---")
    # for match in all_chart_matches:
    #     print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # print("\n--- KB RESULTS ---")
    # if kb_results:
    #     for match in kb_results.matches:
    #         print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    print("\n--- KB RESULTS ---")

    if kb_results:

        if isinstance(kb_results, list):
            for match in kb_results:
                print(f"Score: EXACT_MATCH | {match.metadata.get('text', '')[:100]}")
        else:
            for match in kb_results.matches:
                print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

    # -------------------------------
    # STEP 7: CONTEXT BUILDING
    # -------------------------------
    prompt_start = time.perf_counter()
    
    if not use_chart and not use_kb:
        context = ""   # 🔥 PURE LLM MODE
    else:
        context = build_context(all_chart_matches, kb_results)

    # -------------------------------
    # Inject SL context (if medium confidence)
    # -------------------------------
    print("INJECTING SL INTO CONTEXT:", use_sl_as_context)

    if use_sl_as_context and sl_context:

        context = (
            "Relevant past learned answers (use as base, refine and synthesize):\n"
            f"{sl_context}\n\n"
            f"{context}"
        )

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])


    # -------------------------------
    # Inject previous conversation
    # -------------------------------
    if request.previous_question and request.previous_answer:

        conversation_context = f"""
    PREVIOUS CONVERSATION:

    User: {request.previous_question}

    Astrologer: {request.previous_answer}
    """

        context = f"{conversation_context}\n\n{context}"

    ttl_prompt = round(
        (time.perf_counter() - prompt_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 8: GENERATE ANSWER
    # -------------------------------
    reasoning_start = time.perf_counter()

    answer = generate_answer_gemini(
        request.question,
        context
    )

    ttl_reasoning = round(
        (time.perf_counter() - reasoning_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 9: STORE ANSWER
    # -------------------------------
    delivery_start = time.perf_counter()
    
    if qna_id:
        update_qna_answer(qna_id, answer)

    ttl_delivery = round(
        (time.perf_counter() - delivery_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 10: RESPONSE
    # -------------------------------
    c_rttl = round(
        (time.time() - start_time) * 1000,
        2
    )

    ttl_breakdown = [
        {
            "stage": "ttl_q_embedding",
            "time_ms": ttl_q_embedding
        },
        {
            "stage": "ttl_sl_match",
            "time_ms": ttl_sl_match
        },
        {
            "stage": "ttl_kb_context",
            "time_ms": ttl_kb_context
        },
        {
            "stage": "ttl_chart_context",
            "time_ms": ttl_chart_context
        },
        {
            "stage": "ttl_context_build",
            "time_ms": ttl_prompt
        },
        {
            "stage": "ttl_reasoning",
            "time_ms": ttl_reasoning
        },
        {
            "stage": "ttl_delivery",
            "time_ms": ttl_delivery
        }
    ]

    print("TOTAL RTTL (ms):", c_rttl)

    print("TTL BREAKDOWN:")
    print(ttl_breakdown)

    return {
        "source": "gemini-2.5-flash",

        "used_sl": use_sl_as_context,
        "used_kb": use_kb,
        "used_chart": use_chart,

        "rttl": c_rttl,

        "c_ttl": ttl_breakdown,

        "answer": answer
    }


# @app.post("/welcome_message")
# def welcome_message(request: WelcomeRequest):

#     chart_details = get_chart_details_bulk(request.chart_ids)

#     if not chart_details:
#         return {
#             "messages": [
#                 "Hello ✨",
#                 "I’m here whenever you’d like guidance."
#             ]
#         }

#     all_chart_matches = []

#     # Create generic onboarding query embedding
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input="personality life overview current dasha strengths challenges"
#     )

#     query_embedding = response.data[0].embedding

#     # Retrieve chart context
#     for chart in chart_details:

#         results = query_chart_embeddings(
#             query_embedding,
#             chart["user_id"],
#             chart["profile_id"],
#             chart["chart_id"],
#             top_k=5
#         )

#         all_chart_matches.extend(results.matches)

#     # Build context
#     context = build_context(all_chart_matches, None)

#     # Generate welcome message
#     welcome_text = generate_welcome_message(
#         context=context,
#         user_name=request.user_name
#     )

#     # Convert paragraphs into separate chat messages
#     welcome_lines = [
#         line.strip()
#         for line in welcome_text.split("\n")
#         if line.strip()
#     ]

#     return {
#         "messages": welcome_lines
#     }

@app.post("/welcome_message")
def welcome_message(request: WelcomeRequest):

    q = request.question.lower()

    # -------------------------------
    # CASE 1: Greeting
    # -------------------------------
    if "my name is" in q:

        name = q.split("my name is")[-1].strip().replace(".", "").title()

        return {
            "answer": f"Hello {name}"
        }

    # -------------------------------
    # CASE 2: Generic onboarding
    # -------------------------------
    if "ask questions about myself" in q:

        return {
            "answer": "Feel free to ask anything you would like guidance about."
        }

    # -------------------------------
    # CASE 2.5: Simple Hello
    # -------------------------------
    if q.strip() in ["hello", "hi", "hey"]:

        return {
            "answer": "Hello ✨"
        }

    # -------------------------------
    # CASE 3: Ascendant / Dasha
    # -------------------------------
    chart_details = get_chart_details_bulk(request.chart_ids)

    if not chart_details:
        return {
            "answer": "I could not find your chart details."
        }

    all_chart_matches = []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )

    query_embedding = response.data[0].embedding

    for chart in chart_details:

        results = query_chart_embeddings(
            query_embedding,
            chart["user_id"],
            chart["profile_id"],
            chart["chart_id"],
            top_k=20
        )

        all_chart_matches.extend(results.matches)

    # DEBUG HERE
    print("\n===== TOP RETRIEVED CHUNKS =====")

    for i, m in enumerate(all_chart_matches[:20], 1):
        print(
            f"{i}. SCORE={round(m.score,3)} | "
            f"{m.metadata.get('text','')[:200]}"
        )

    context = build_context(all_chart_matches, None)

    print("\n===== FINAL CONTEXT =====")
    print(context[:5000])

    answer = generate_answer(request.question, context)

    return {
        "answer": answer
    }

# @app.post("/qna_gemini")
# def qna_gemini(request: QuestionRequest):

#     chart_ids = request.chart_ids
#     kb_ids = request.kb_id

#     use_chart = chart_ids and chart_ids != ["0"] and chart_ids != [""]
#     use_kb = kb_ids and kb_ids != ["0"] and kb_ids != [""]

#     # -------------------------------
#     # STEP 1: INIT
#     # -------------------------------
#     qna_id = None
#     chart_details = []
#     all_chart_matches = []

#     # -------------------------------
#     # STEP 2: FETCH CHART + STORE QNA
#     # -------------------------------
#     if use_chart:
#         chart_details = get_chart_details_bulk(chart_ids)

#         if chart_details:
#             primary_chart = chart_details[0]

#             qna_id = insert_qna(
#                 primary_chart["user_id"],
#                 primary_chart["profile_id"],
#                 primary_chart["chart_id"],
#                 request.question
#             )

#     # -------------------------------
#     # STEP 3: EMBEDDING
#     # -------------------------------
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=request.question
#     )
#     query_embedding = response.data[0].embedding

#     # -------------------------------
#     # STEP 4: CHART RETRIEVAL
#     # -------------------------------
#     if use_chart and chart_details:
#         for chart in chart_details:
#             results = query_chart_embeddings(
#                 query_embedding,
#                 chart["user_id"],
#                 chart["profile_id"],
#                 chart["chart_id"],
#                 top_k=5
#             )
#             all_chart_matches.extend(results.matches)

#     # -------------------------------
#     # STEP 5: KB RETRIEVAL
#     # -------------------------------
#     kb_results = None

#     if use_kb:
#         if "job_n" in kb_ids:
#             kb_results = query_kb_embeddings(query_embedding, top_k=10)
#         else:
#             kb_results = query_kb_embeddings_filtered(query_embedding, kb_ids, top_k=10)

#     # -------------------------------
#     # STEP 6: DEBUG
#     # -------------------------------
#     print("\n================ RETRIEVAL DEBUG ================")

#     print("\n--- CHART RESULTS (Merged) ---")
#     for match in all_chart_matches:
#         print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

#     print("\n--- KB RESULTS ---")
#     if kb_results:
#         for match in kb_results.matches:
#             print(f"Score: {round(match.score, 3)} | {match.metadata.get('text', '')[:100]}")

#     # -------------------------------
#     # STEP 7: CONTEXT BUILDING
#     # -------------------------------
#     if not use_chart and not use_kb:
#         context = ""
#     else:
#         context = build_context(all_chart_matches, kb_results)

#     # -------------------------------
#     # STEP 8: GENERATE ANSWER
#     # -------------------------------
#     # answer = generate_answer_gemini(request.question, context)

#     print("MODEL USED: GEMINI")

#     answer = generate_answer_gemini(
#         request.question,
#         context
#     )

#     # -------------------------------
#     # STEP 9: STORE ANSWER
#     # -------------------------------
#     if qna_id:
#         update_qna_answer(qna_id, answer)

#     # -------------------------------
#     # STEP 10: RESPONSE
#     # -------------------------------
#     return {
#         "answer": answer
#     }



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

    # DEBUG (IMPORTANT)
    print("🧾 DELETE JOB:", job)
    print("📁 FILE_ID:", file_id)

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

    file_id = chart_job["file_id"]

    if not file_id:
        raise HTTPException(
            status_code=400,
            detail="file_id missing for this chart job"
        )    

    print("🧾 DELETE CHART JOB:", chart_job)
    print("📁 FILE_ID:", file_id)

    try:
        # 2. Delete embeddings
        print(f"🧹 Deleting embeddings for file_id: {file_id}")
        delete_embeddings(file_id)
        print(f"✅ Embeddings deleted for file_id: {file_id}")

        # 3. Delete files from S3
        from storage import delete_file

        delete_file(file_id)

        print(f"✅ S3 cleanup completed for file_id: {file_id}")

    except Exception as e:
        print(f"⚠️ Delete error: {e}")

    # 🔥 4. SOFT DELETE (THIS WAS MISSING)
    soft_delete_chart_job(job_id)

    return {
        "status": "success",
        "message": f"Chart deleted for job_id: {job_id}"
    }


@app.post("/qna_sl")
def qna_sl(request: QnaSLRequest):

    # -------------------------------
    # STEP 1: CREATE EMBEDDING
    # -------------------------------
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = response.data[0].embedding

    # -------------------------------
    # STEP 2: RETRIEVE KB ONLY
    # -------------------------------
    kb_results = query_kb_embeddings_filtered(
        query_embedding,
        [request.kb_id],
        top_k=5
    )

    # -------------------------------
    # STEP 3: BUILD CONTEXT
    # -------------------------------
    context = ""
    for match in kb_results.matches:
        context += match.metadata.get("text", "") + "\n"

    context = context[:3000]

    # -------------------------------
    # STEP 4: GENERATE ANSWER
    # -------------------------------
    prompt = build_prompt(request.question, context)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        max_tokens=220,
        messages=[
            {"role": "system", "content": "You are a careful domain expert."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    # -------------------------------
    # STEP 5: STORE
    # -------------------------------
    qna_id = insert_qna_sl(
        request.kb_id,
        request.question,
        answer
    )

    # -------------------------------
    # STEP 6: RETURN
    # -------------------------------
    return {
        "qna_id": qna_id,
        "answer": answer
    }


@app.post("/qna_sl_validation")
def qna_sl_validation(request: QnaSLValidationRequest):

    table_map = {
        "sl": "qna_sl_logs",
        "generated": "generated_qnas"
    }

    if request.source_type not in table_map:
        raise HTTPException(
            status_code=400,
            detail="Invalid source_type"
        )

    table_name = table_map[request.source_type]

    # -------------------------------
    # STEP 1: FETCH ORIGINAL QNA
    # -------------------------------
    record = get_qna_sl(
        table_name,
        request.qna_id
    )

    if not record:
        raise HTTPException(status_code=404, detail="QnA not found")

    # -------------------------------
    # STEP 2: DECIDE FINAL ANSWER
    # -------------------------------
    original_answer = (
        record.get("llm_answer")
        or record.get("answer")
    )

    if request.is_valid:

        final_answer = original_answer

    else:

        if not request.corrected_answer:
            raise HTTPException(
                status_code=400,
                detail="Corrected answer required when invalid"
            )

        final_answer = request.corrected_answer

    # -------------------------------
    # STEP 3: UPDATE DB
    # -------------------------------
    update_qna_sl_validation(
        table_name=table_name,
        qna_id=request.qna_id,
        is_valid=request.is_valid,
        llm_answer=original_answer,
        corrected_answer=request.corrected_answer
    )

    # -------------------------------
    # STEP 4: CREATE EMBEDDING
    # -------------------------------
    text = record['question']

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    print("KB_ID SAVED:", record["kb_id"])
    print("QUESTION SAVED:", record["question"])

    # -------------------------------
    # STEP 5: STORE IN VECTOR DB
    # -------------------------------
    upsert_embeddings(
        file_id=f"{request.source_type}_{request.qna_id}",
        chunks=[text],
        embeddings=[embedding],
        metadata={
            "type": "qna_sl",
            "kb_id": str(record["kb_id"]).strip(),
            "question": record["question"],
            "answer": final_answer,
            "source_type": request.source_type
        }
    )

    # -------------------------------
    # STEP 6: RESPONSE
    # -------------------------------
    return {
        "status": "validated and learned"
    }


@app.post("/qna_ml_submit")
def qna_ml_submit(request: QnaMLRequest):

    results = []

    for qna_id in request.qna_ids:

        record = get_qna_sl(
            "qna_sl_logs",
            qna_id
        )

        if not record:
            results.append({
                "qna_id": qna_id,
                "status": "not_found"
            })
            continue

        final_answer = record["corrected_answer"]

        payload = {
            "question": record["question"],
            "answer": final_answer,
            "kb_id": record["kb_id"]
        }

        # -------------------------------
        # Step 2: Call ML pipeline (TEMP MOCK)
        # -------------------------------
        success = True

        # -------------------------------
        # Step 3: Update flag
        # -------------------------------
        if success:
            mark_qna_ml_ready(qna_id)

        results.append({
            "qna_id": qna_id,
            "ml_status": success
        })

    return {
        "results": results
    }


@app.post("/qna_sl_search")
def qna_sl_search(request: QnaSearchRequest):

    print("INSIDE QNA SL SEARCH")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = response.data[0].embedding

    results = query_qna_sl_embeddings(
        query_embedding,
        request.kb_id
    )

    if not results.matches:
        return {
            "found": False,
            "matches": []
        }

    # NEW CODE STARTS HERE
    TOP_K = 3
    matches = results.matches[:TOP_K]

    return {
        "found": True,
        "matches": [
            {
                "score": m.score,
                "question": m.metadata.get("question"),
                "answer": m.metadata.get("answer")
            }
            for m in matches
        ]
    }

@app.post("/qna_generate")
def qna_generate(request: QnaGenerateRequest):

    qnas = generate_qnas(request.kb_id)

    return {
        "status": "success",
        "message": "5 qnas created",
        "data": qnas
    }


@app.post("/delete_qna_sl")
def delete_qna_sl(request: DeleteQnaSLRequest):

    qna_id = request.qna_id

    # -------------------------------
    # STEP 1: FETCH RECORD
    # -------------------------------
    record = get_qna_sl(
        "generated_qnas",
        qna_id
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="QnA not found"
        )

    try:

        # -------------------------------
        # STEP 2: DELETE EMBEDDINGS
        # -------------------------------
        delete_qna_embeddings(
            "generated",
            qna_id
        )

        # -------------------------------
        # STEP 3: DELETE DB RECORD
        # -------------------------------
        delete_qna_record(
            "generated_qnas",
            qna_id
        )

    except Exception as e:

        print(f"⚠️ QnA delete error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to delete QnA"
        )

    return {
        "status": "success",
        "message": "QnA deleted successfully"
    }


@app.get("/find_yoga")
def find_yoga(kb_id: str, yoga: str):

    chunks = get_all_kb_chunks(kb_id)

    matches = []

    for chunk in chunks:

        text = chunk.metadata.get("text", "")

        if yoga.lower() in text.lower():

            matches.append(text[:1000])

    return matches[:5]


class RetrievalTestRequest(BaseModel):
    question: str

    kb_ids: Optional[list[str]] = None

    user_id: Optional[str] = None
    profile_id: Optional[str] = None
    chart_id: Optional[str] = None

    include_kb: bool = True
    include_chart: bool = False
    include_sl: bool = False

    top_k: int = 50

# @app.post("/retrieval_test")
# async def retrieval_test(request: RetrievalTestRequest):
#     try:

#         # --------------------------------
#         # EMBEDDING
#         # --------------------------------
#         response = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=request.question
#         )

#         query_embedding = response.data[0].embedding

#         # --------------------------------
#         # EXACT MATCH
#         # --------------------------------
#         exact_match = None

#         if (
#             request.include_kb
#             and request.kb_ids
#             and len(request.kb_ids) > 0
#         ):
#             exact_match = find_exact_kb_match(
#                 request.kb_ids[0],
#                 request.question
#             )

#         # --------------------------------
#         # VECTOR SEARCH
#         # --------------------------------
#         semantic_results = None

#         # -----------------------------
#         # KB Retrieval
#         # -----------------------------
#         if (
#             request.include_kb
#             and request.kb_ids
#         ):
#             semantic_results = query_kb_embeddings_filtered(
#                 query_embedding,
#                 request.kb_ids,
#                 top_k=request.top_k
#             )

#         # -----------------------------
#         # Chart Retrieval
#         # -----------------------------
#         elif (
#             request.include_chart
#             and request.user_id
#             and request.profile_id
#             and request.chart_id
#         ):
#             semantic_results = query_chart_embeddings(
#                 query_embedding,
#                 request.user_id,
#                 request.profile_id,
#                 request.chart_id,
#                 top_k=request.top_k
#             )

#         # -----------------------------
#         # SL Retrieval
#         # -----------------------------
#         elif (
#             request.include_sl
#             and request.kb_ids
#         ):
#             semantic_results = query_qna_sl_embeddings(
#                 query_embedding,
#                 request.kb_ids[0],
#                 top_k=request.top_k
#             )

#         # --------------------------------
#         # NOISE FILTER
#         # --------------------------------

#         import re

#         def is_noise_chunk(text):

#             text_lower = text.lower()

#             # obvious PDF garbage
#             if "appendix" in text_lower:
#                 return True

#             if "contents" in text_lower:
#                 return True

#             # count numbers
#             numbers = re.findall(r"\d+", text)

#             # chunks full of page numbers/index entries
#             if len(numbers) > 20:
#                 return True

#             return False

#         filtered_matches = []

#         if semantic_results:

#             for match in semantic_results.matches:

#                 text = match.metadata.get("text", "")

#                 if not is_noise_chunk(text):
#                     filtered_matches.append(match)


#         # ==========================================
#         # DEBUG RAW PINECONE RESULTS
#         # ==========================================
#         if semantic_results:

#             print("\n===== ORIGINAL PINECONE =====")

#             for idx, match in enumerate(
#                 semantic_results.matches[:20]
#             ):

#                 print(
#                     idx + 1,
#                     round(match.score, 4),
#                     match.metadata.get("text", "")[:150]
#                 )

#             print("\n===== AFTER NOISE FILTER =====")

#             for idx, match in enumerate(
#                 filtered_matches[:20]
#             ):

#                 print(
#                     idx + 1,
#                     round(match.score, 4),
#                     match.metadata.get("text", "")[:150]
#                 )

#         # --------------------------------
#         # RERANKING
#         # --------------------------------

#         query_lower = request.question.lower()

#         query_phrase = None

#         if "yoga" in query_lower:

#             words = query_lower.split()

#             for i in range(len(words) - 1):

#                 if words[i + 1] == "yoga":

#                     query_phrase = words[i] + " yoga"
#                     break

#         query_words = set(
#             re.findall(
#                 r"[a-zA-Z]+",
#                 request.question.lower()
#             )
#         )

#         reranked = []

#         for match in filtered_matches:

#             text = match.metadata.get(
#                 "text",
#                 ""
#             ).lower()

#             overlap_score = sum(
#                 1
#                 for word in query_words
#                 if len(word) > 3 and word in text
#             )

#             final_score = match.score

#             final_score += overlap_score * 0.05

#             reranked.append(
#                 (final_score, match)
#             )

#         reranked.sort(
#             key=lambda x: x[0],
#             reverse=True
#         )

#         filtered_matches = [
#             item[1]
#             for item in reranked
#         ]

#         # --------------------------------
#         # FOUND RANKS
#         # --------------------------------

#         search_terms = [
#             "yupa yoga",
#             "vasumathi yoga"
#         ]

#         found_ranks = {}

#         for term in search_terms:

#             found_ranks[term] = None

#             for idx, match in enumerate(filtered_matches):

#                 text = match.metadata.get(
#                     "text",
#                     ""
#                 ).lower()

#                 if term in text:

#                     found_ranks[term] = idx + 1
#                     break

#         # --------------------------------
#         # RESPONSE
#         # --------------------------------

#         return {
#             "status": "success",

#             "found_ranks": found_ranks,

#             "question": request.question,

#             "embedding_dimension": len(
#                 query_embedding
#             ),

#             "exact_match_found":
#                 exact_match is not None,

#             "exact_match_text":
#                 exact_match.metadata.get(
#                     "text",
#                     ""
#                 )[:500]
#                 if exact_match
#                 else None,

#             "semantic_match_count":
#                 len(filtered_matches),

#             "semantic_matches": [
#                 {
#                     "rank": idx + 1,

#                     "score": round(
#                         reranked[idx][0],
#                         4
#                     ),

#                     "file_id":
#                         match.metadata.get(
#                             "file_id"
#                         ),

#                     "user_id":
#                         match.metadata.get(
#                             "user_id"
#                         ),

#                     "profile_id":
#                         match.metadata.get(
#                             "profile_id"
#                         ),

#                     "chart_id":
#                         match.metadata.get(
#                             "chart_id"
#                         ),

#                     "text":
#                         match.metadata.get(
#                             "text",
#                             ""
#                         )[:500]
#                 }
#                 for idx, match in enumerate(
#                     filtered_matches[:10]
#                 )
#             ]
#         }

#     except Exception as e:

#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )

@app.post("/retrieval_test")
async def retrieval_test(request: RetrievalTestRequest):
    try:

        # --------------------------------
        # EMBEDDING
        # --------------------------------
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.question
        )

        query_embedding = response.data[0].embedding

        # --------------------------------
        # CHART RETRIEVAL ONLY (HARDCODED)
        # --------------------------------
        print("\n========== CHART RETRIEVAL TEST ==========")

        semantic_results = query_chart_embeddings(
            query_embedding,
            user_id="1",
            profile_id="38",
            chart_id="96",
            top_k=20
        )

        # --------------------------------
        # NOISE FILTER
        # --------------------------------
        import re

        def is_noise_chunk(text):

            text_lower = text.lower()

            if "appendix" in text_lower:
                return True

            if "contents" in text_lower:
                return True

            numbers = re.findall(r"\d+", text)

            if len(numbers) > 20:
                return True

            return False

        filtered_matches = []

        if semantic_results:

            filtered_matches = semantic_results.matches

        # --------------------------------
        # DEBUG
        # --------------------------------
        if semantic_results:

            print("\n===== ORIGINAL PINECONE =====")

            for idx, match in enumerate(
                semantic_results.matches[:20]
            ):

                print(
                    idx + 1,
                    round(match.score, 4),
                    match.metadata.get(
                        "text",
                        ""
                    )[:150]
                )

            print("\n===== AFTER NOISE FILTER =====")

            for idx, match in enumerate(
                filtered_matches[:20]
            ):

                print(
                    idx + 1,
                    round(match.score, 4),
                    match.metadata.get(
                        "text",
                        ""
                    )[:150]
                )

        # --------------------------------
        # SIMPLE RERANKING
        # --------------------------------

        query_words = set(
            re.findall(
                r"[a-zA-Z]+",
                request.question.lower()
            )
        )

        reranked = []

        for match in filtered_matches:

            text = match.metadata.get(
                "text",
                ""
            ).lower()

            overlap_score = sum(
                1
                for word in query_words
                if len(word) > 3 and word in text
            )

            final_score = match.score

            final_score += overlap_score * 0.05

            reranked.append(
                (final_score, match)
            )

        reranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        filtered_matches = [
            item[1]
            for item in reranked
        ]

        # --------------------------------
        # RESPONSE
        # --------------------------------

        return {

            "status": "success",

            "question": request.question,

            "hardcoded_chart": {
                "user_id": "1",
                "profile_id": "38",
                "chart_id": "96"
            },

            "embedding_dimension": len(
                query_embedding
            ),

            "semantic_match_count":
                len(filtered_matches),

            "semantic_matches": [

                {
                    "rank": idx + 1,

                    "score": round(
                        reranked[idx][0],
                        4
                    ),

                    "file_id":
                        match.metadata.get(
                            "file_id"
                        ),

                    "user_id":
                        match.metadata.get(
                            "user_id"
                        ),

                    "profile_id":
                        match.metadata.get(
                            "profile_id"
                        ),

                    "chart_id":
                        match.metadata.get(
                            "chart_id"
                        ),

                    "text":
                        match.metadata.get(
                            "text",
                            ""
                        )[:500]
                }

                for idx, match in enumerate(
                    filtered_matches[:10]
                )
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

# @app.get("/debug_chart_vectors")
# def debug_chart_vectors():

#     print("DEBUG ENDPOINT HIT")

#     results = index.query(
#         vector=[0.0] * 1536,
#         top_k=10,
#         include_metadata=True
#     )

#     print("MATCH COUNT:", len(results.matches))

#     for i, match in enumerate(results.matches):
#         print(f"MATCH {i+1}")
#         print(match.metadata)

#     return {
#         "match_count": len(results.matches)
#     }