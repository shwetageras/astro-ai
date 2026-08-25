import time
import uuid
import os
import re
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from settings import OPENAI_MODEL
# from google import genai
from storage import save_file, save_metadata
from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb
from notifier import notify_embedding_status
from db import get_chart_details_bulk, soft_delete_chart_job, get_chart_job
from db import insert_job, get_job, update_job
from vector_db import upsert_embeddings
from openai_client import client
from vector_db import query_chart_embeddings
from fastapi import HTTPException
from vector_db import delete_embeddings
from prompts import build_prompt
from typing import List, Optional
from vector_db import query_kb_embeddings_filtered
import google.generativeai as genai
from db import insert_qna_sl
from db import update_qna_sl_validation, get_qna_sl
from db import mark_qna_ml_ready
from pydantic import BaseModel
from qna_generator import generate_qnas
from db import delete_qna_record
from vector_db import delete_qna_embeddings
from vector_db import get_all_kb_chunks


genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()

def make_safe_filename(name: str):
    return name.replace(" ", "_").replace("/", "_")

# BACKGROUND FUNCTION FOR upload_pdf
def process_pdf(file_bytes, file_id, file_name, job_id, timestamp):
    
    temp_file_path = None

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
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def is_similar(text1, text2, threshold=0.8):
    # Simple similarity using overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    overlap = len(words1 & words2) / max(len(words1), 1)

    return overlap > threshold


def build_context(chart_results, kb_results):

    all_chart_chunks = []
    all_kb_chunks = []

    # --------------------------------
    # HANDLE CHART RESULTS
    # --------------------------------

    chart_matches = (
        chart_results.matches
        if hasattr(chart_results, "matches")
        else chart_results
    )

    for match in chart_matches:

        all_chart_chunks.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": "chart"
        })


    # --------------------------------
    # HANDLE KB RESULTS
    # --------------------------------

    if kb_results:

        kb_matches = (
            kb_results.matches
            if hasattr(kb_results, "matches")
            else kb_results
        )

        for match in kb_matches:

            all_kb_chunks.append({
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": "kb"
            })


    # --------------------------------
    # SORT EACH SOURCE INDEPENDENTLY
    # --------------------------------

    all_chart_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    all_kb_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    print("\n===== CHART RETRIEVAL =====")

    for idx, chunk in enumerate(
        all_chart_chunks[:10]
    ):

        print(
            "CHART",
            idx + 1,
            round(chunk["score"], 4),
            chunk["text"][:150]
        )


    print("\n===== KB RETRIEVAL =====")

    for idx, chunk in enumerate(
        all_kb_chunks[:10]
    ):

        print(
            "KB",
            idx + 1,
            round(chunk["score"], 4),
            chunk["text"][:150]
        )


    # --------------------------------
    # FINAL CONTEXT SELECTION
    # --------------------------------

    MAX_CONTEXT_CHUNKS = 6
    MAX_CONTEXT_CHARS = 3000

    selected_chart = []
    selected_kb = []


    # --------------------------------
    # BOTH SOURCES AVAILABLE
    # --------------------------------

    if all_chart_chunks and all_kb_chunks:

        # Always preserve at least one
        # user-specific Chart chunk.
        selected_chart.append(
            all_chart_chunks[0]
        )

        # Always preserve at least one
        # Knowledge Base chunk.
        selected_kb.append(
            all_kb_chunks[0]
        )


        chart_candidates = all_chart_chunks[1:]
        kb_candidates = all_kb_chunks[1:]


        remaining_slots = (
            MAX_CONTEXT_CHUNKS
            - len(selected_chart)
            - len(selected_kb)
        )


        # --------------------------------
        # DYNAMIC SOURCE ALLOCATION
        # --------------------------------

        for _ in range(remaining_slots):

            if not chart_candidates and not kb_candidates:
                break


            if not chart_candidates:

                selected_kb.append(
                    kb_candidates.pop(0)
                )

                continue


            if not kb_candidates:

                selected_chart.append(
                    chart_candidates.pop(0)
                )

                continue


            # --------------------------------
            # NORMALIZE SCORES WITHIN SOURCE
            # --------------------------------

            chart_max_score = all_chart_chunks[0]["score"]
            kb_max_score = all_kb_chunks[0]["score"]


            chart_score = chart_candidates[0]["score"]
            kb_score = kb_candidates[0]["score"]


            chart_normalized = (
                chart_score / chart_max_score
                if chart_max_score > 0
                else 0
            )


            kb_normalized = (
                kb_score / kb_max_score
                if kb_max_score > 0
                else 0
            )


            # --------------------------------
            # CHART PRIORITY
            # --------------------------------
            #
            # If both candidates are reasonably
            # close, prefer Chart because Chart
            # is the user-specific source.
            #
            # If KB is clearly stronger, allow
            # KB to take the slot.
            # --------------------------------

            CHART_PRIORITY_MARGIN = 0.05


            if (
                chart_normalized
                >= kb_normalized - CHART_PRIORITY_MARGIN
            ):

                selected_chart.append(
                    chart_candidates.pop(0)
                )

            else:

                selected_kb.append(
                    kb_candidates.pop(0)
                )


    # --------------------------------
    # ONLY CHART AVAILABLE
    # --------------------------------

    elif all_chart_chunks:

        selected_chart = all_chart_chunks[
            :MAX_CONTEXT_CHUNKS
        ]


    # --------------------------------
    # ONLY KB AVAILABLE
    # --------------------------------

    elif all_kb_chunks:

        selected_kb = all_kb_chunks[
            :MAX_CONTEXT_CHUNKS
        ]


    # --------------------------------
    # FINAL BUCKET
    # --------------------------------

    selected = (
        selected_chart +
        selected_kb
    )


    print("\n===== FINAL CONTEXT BUCKET =====")

    for idx, chunk in enumerate(
        selected
    ):

        print(
            idx + 1,
            chunk["source"].upper(),
            round(chunk["score"], 4),
            chunk["text"][:150]
        )


    # --------------------------------
    # BUILD STRUCTURED CONTEXT
    # --------------------------------

    context = ""


    if selected_chart:

        context += "CHART DATA:\n"

        for chunk in selected_chart:

            context += (
                f"- {chunk['text'].strip()}\n"
            )


    if selected_kb:

        context += "\nKNOWLEDGE BASE:\n"

        for chunk in selected_kb:

            context += (
                f"- {chunk['text'].strip()}\n"
            )


    print("\n===== FINAL CONTEXT SENT TO GPT =====")
    print(context)
    print("===== END =====")

    print(
        "FINAL CHART CHUNKS:",
        len(selected_chart)
    )

    print(
        "FINAL KB CHUNKS:",
        len(selected_kb)
    )

    return context[:MAX_CONTEXT_CHARS], selected


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


class RetrievalTestRequest(BaseModel):

    question: str

    chart_ids: List[str] = []
    kb_id: List[str] = []
    sl_id: List[str] = []

    user_id: Optional[int] = None
    profile_id: Optional[int] = None
    chart_id: Optional[str] = None

    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None

    include_chart: bool = True
    include_kb: bool = True
    include_sl: bool = True

    top_k: int = 20



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

        # Payload structure for the future ML pipeline integration.
        # Currently unused because the ML call is mocked.

        _payload = {
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


@app.post("/retrieval_test")
async def retrieval_test(request: RetrievalTestRequest):
    try:

        print("\n========== RETRIEVAL TEST REQUEST ==========")
        print("QUESTION:", request.question)

        print("REQUEST DICT:")
        print(request.dict())

        print("===========================================")


        # --------------------------------
        # EMBEDDING
        # --------------------------------
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.question
        )

        query_embedding = response.data[0].embedding


        # --------------------------------
        # RETRIEVAL PIPELINE
        # --------------------------------
        print("\n========== RETRIEVAL PIPELINE ==========")

        # -------------------------------
        # Chart Retrieval
        # -------------------------------
        chart_details = []
        all_chart_matches = []

        if request.chart_id:

            chart_details = get_chart_details_bulk(
                [request.chart_id]
            )

            for chart in chart_details:

                results = query_chart_embeddings(
                    query_embedding,
                    chart["user_id"],
                    chart["profile_id"],
                    chart["chart_id"],
                    top_k=request.top_k
                )

                all_chart_matches.extend(results.matches)

        elif request.chart_ids:

            # Backward compatibility with old payload
            chart_details = get_chart_details_bulk(
                request.chart_ids
            )

            for chart in chart_details:

                results = query_chart_embeddings(
                    query_embedding,
                    chart["user_id"],
                    chart["profile_id"],
                    chart["chart_id"],
                    top_k=request.top_k
                )

                all_chart_matches.extend(results.matches)

        semantic_results = all_chart_matches

        # -------------------------------
        # Knowledge Base Retrieval
        # -------------------------------
        exact_match = None
        kb_results = None

        if request.kb_id:

            exact_match = find_exact_kb_match(
                request.kb_id[0],
                request.question
            )

            if exact_match:

                print("USING EXACT MATCH")

                kb_results = [exact_match]

            else:

                print("USING VECTOR SEARCH")

                kb_results = query_kb_embeddings_filtered(
                    query_embedding,
                    request.kb_id,
                    top_k=request.top_k
                )

        else:

            print("SEARCHING ALL KBs")

            kb_results = query_kb_embeddings_filtered(
                query_embedding,
                [],
                top_k=request.top_k
            )

        # --------------------------------
        # NOISE FILTER
        # --------------------------------

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

            filtered_matches = [
                match
                for match in semantic_results
                if not is_noise_chunk(
                    match.metadata.get("text", "")
                )
            ]

        # --------------------------------
        # DEBUG
        # --------------------------------
        if semantic_results:

            print("\n===== ORIGINAL PINECONE =====")

            for idx, match in enumerate(
                semantic_results[:20]
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

        chart_matches = filtered_matches

        context, final_bucket = build_context(
            chart_matches,
            kb_results
        )

        print("\n========== CONTEXT DEBUG ==========")
        print("Context length:", len(context))
        print("Retrieved chart chunks:", len(filtered_matches))
        print(
            "KB chunks:",
            len(kb_results)
            if isinstance(kb_results, list)
            else len(kb_results.matches)
            if kb_results
            else 0
        )

        # --------------------------------
        # RESPONSE
        # --------------------------------

        return {

            "status": "success",

            "question": request.question,

            "chart_ids": request.chart_ids,

            "kb_id": request.kb_id,

            "top_k": request.top_k,

            "embedding_dimension": len(query_embedding),

            "filtered_chart_match_count": len(filtered_matches),

            "exact_match_used": isinstance(kb_results, list),

            "kb_match_count": (
                len(kb_results)
                if isinstance(kb_results, list)
                else len(kb_results.matches)
                if kb_results
                else 0
            ),

            "semantic_matches": [

                {
                    "rank": idx + 1,
                    "source": chunk["source"],
                    "score": round(chunk["score"], 4),
                    "text": chunk["text"][:500]
                }

                for idx, chunk in enumerate(
                    final_bucket
                )
            ],

            "context": context
        }
        
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )