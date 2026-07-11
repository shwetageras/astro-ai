from fastapi import FastAPI
from db import get_job
from typing import List, Optional
from pydantic import BaseModel
from chat_service import (
    generate_answer,
    generate_answer_gpt_mini,
    generate_answer_gemini,
    query_docs_service,
    welcome_message_service,
    qna_sl_search_service,
    process_question,
)


app = FastAPI()


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


class RetrievalTestRequest(BaseModel):

    question: str

    chart_ids: List[str] = []
    kb_id: List[str] = []
    sl_id: List[str] = []

    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None

    include_chart: bool = True
    include_kb: bool = True
    include_sl: bool = True

    top_k: int = 20




# Create API → /upload_kb
# @app.post("/upload_kb")
# async def upload_kb(
#     background_tasks: BackgroundTasks,
#     isKbtype: str = Form(...),
#     name: str = Form(...),
#     content: str = Form(None),
#     file: UploadFile = File(None),
# ):
#     import time

#     timestamp = int(time.time())

#     safe_name = make_safe_filename(name)
#     file_id = f"{timestamp}_{safe_name}"
    
#     job_id = f"job_{timestamp}_{uuid.uuid4().hex}"

#     # 🔥 CASE 1: TEXT INPUT
#     if isKbtype == "article":

#         if not content:
#             raise HTTPException(status_code=400, detail="Content is required for article type")

#         # STEP 1
#         insert_job(job_id, file_id, name, "processing", timestamp)

#         # STEP 2
#         background_tasks.add_task(
#             process_text,
#             content,
#             file_id,
#             name,
#             job_id,
#             timestamp
#         )

#     # 🔥 CASE 2: FILE INPUT
#     elif isKbtype == "file":

#         if not file:
#             raise HTTPException(status_code=400, detail="File is required for file type")

#         # Store job info
#         insert_job(job_id, file_id, name, "processing", timestamp)
        
#         file_bytes = await file.read()

#         # Upload to S3
#         save_file(file_bytes, file_id)

#         background_tasks.add_task(
#             process_pdf,
#             file_bytes,
#             file_id,
#             file.filename,   # USE REAL FILE NAME
#             job_id,
#             timestamp
#         )

#     else:
#         raise HTTPException(status_code=400, detail="Invalid isKbtype")

#     return {
#         "job_id": job_id,
#         "status": "processing"
#     }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    return job

@app.post("/query")
def query_docs(request: QueryRequest):
    return query_docs_service(request)

# Create NEW API → /upload_chart
# @app.post("/upload_chart")
# async def upload_chart(
#     background_tasks: BackgroundTasks,
#     isCharttype: str = Form(...),
#     name: str = Form(...),
#     user_id: int = Form(...),
#     profile_id: int = Form(...),
#     chart_id: int = Form(...),
#     content: str = Form(None),
#     file: UploadFile = File(None)
# ):
#     print("========== UPLOAD CHART HIT ==========")


#     timestamp = int(time.time())

#     safe_name = make_safe_filename("chart")
#     file_id = f"{timestamp}_{uuid.uuid4().hex}_{safe_name}"

#     job_id = f"job_{timestamp}_{uuid.uuid4().hex}"

#     insert_chart_job(
#         job_id,
#         file_id,
#         chart_id,
#         user_id,
#         profile_id,
#         name,
#         "processing",
#         timestamp
#     )

#     # 🔥 CASE 1: TEXT INPUT
#     if isCharttype == "article":

#         if not content:
#             raise HTTPException(status_code=400, detail="Content required for text chart")

#         # Start background processing
#         background_tasks.add_task(
#             process_chart_text,
#             content,
#             file_id,
#             job_id,
#             chart_id,   
#             user_id,
#             profile_id,
#             timestamp
#         )

#     # 🔥 CASE 2: FILE INPUT
#     elif isCharttype == "file":

#         print("FILE UPLOAD MODE")

#         if not file:
#             raise HTTPException(status_code=400, detail="File required for chart upload")

#         print("FILE NAME:", file.filename)

#         file_bytes = await file.read()

#         print(
#             "FILE SIZE MB:",
#             round(len(file_bytes) / (1024 * 1024), 2)
#         )

#         # Save to S3
#         save_file(file_bytes, file_id)

#         print("ADDING BACKGROUND TASK", flush=True)

#         background_tasks.add_task(
#             process_chart,
#             file_bytes,
#             file_id,
#             file.filename,
#             job_id,
#             chart_id,
#             user_id,
#             profile_id,
#             timestamp
#         )

#         print("BACKGROUND TASK ADDED", flush=True)

#     else:
#         raise HTTPException(status_code=400, detail="Invalid isCharttype")

#     return {
#         "job_id": job_id,
#         "status": "processing"
#     }


@app.post("/ask_question")
def ask_question(request: QuestionRequest):
    
    return process_question(
        request=request,
        answer_generator=generate_answer,
        source_name="gpt-4.1",
    )

@app.post("/qna_gpt_mini")
def qna_gpt_mini(request: QuestionRequest):

    return process_question(
        request=request,
        answer_generator=generate_answer_gpt_mini,
        source_name="gpt-4.1-mini",
    )


@app.post("/qna_gemini")
def qna_gemini(request: QuestionRequest):

    return process_question(
        request=request,
        answer_generator=generate_answer_gemini,
        source_name="gemini-2.5-flash",
    )


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
    return welcome_message_service(request)


# @app.post("/delete_kb")
# def delete_kb(request: DeleteKBRequest):

#     job_id = request.job_id

#     # 1. Get job
#     job = get_job(job_id)

#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     if job["status"] == "processing":
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot delete while processing is in progress"
#         )

#     file_id = job["file_id"]

#     # DEBUG (IMPORTANT)
#     print("🧾 DELETE JOB:", job)
#     print("📁 FILE_ID:", file_id)

#     try:
#         # 2. Delete embeddings (Pinecone)
#         print(f"🧹 Deleting embeddings for file_id: {file_id}")
#         delete_embeddings(file_id)
#         print(f"✅ Embeddings deleted for file_id: {file_id}")

#         # 3. Delete file from S3
#         from storage import delete_file
#         delete_file(file_id)

#     except Exception as e:
#         print(f"⚠️ Delete error (continuing): {e}")

#     # 🔥 ALWAYS update DB (no matter what)
#     update_job(job_id, "deleted", int(time.time()))

#     return {
#         "status": "success",
#         "message": f"KB deleted for job_id: {job_id}"
#     }


# @app.post("/delete_chart")
# def delete_chart(request: DeleteChartRequest):

#     job_id = request.job_id

#     # 1. Get chart job
#     chart_job = get_chart_job(job_id)

#     if not chart_job:
#         raise HTTPException(status_code=404, detail="Chart not found")

#     if chart_job["status"] == "processing":
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot delete while processing"
#         )

#     file_id = chart_job["file_id"]

#     if not file_id:
#         raise HTTPException(
#             status_code=400,
#             detail="file_id missing for this chart job"
#         )    

#     print("🧾 DELETE CHART JOB:", chart_job)
#     print("📁 FILE_ID:", file_id)

#     try:
#         # 2. Delete embeddings
#         print(f"🧹 Deleting embeddings for file_id: {file_id}")
#         delete_embeddings(file_id)
#         print(f"✅ Embeddings deleted for file_id: {file_id}")

#         # 3. Delete files from S3
#         from storage import delete_file

#         delete_file(file_id)

#         print(f"✅ S3 cleanup completed for file_id: {file_id}")

#     except Exception as e:
#         print(f"⚠️ Delete error: {e}")

#     # 🔥 4. SOFT DELETE (THIS WAS MISSING)
#     soft_delete_chart_job(job_id)

#     return {
#         "status": "success",
#         "message": f"Chart deleted for job_id: {job_id}"
#     }


# @app.post("/qna_sl")
# def qna_sl(request: QnaSLRequest):

#     # -------------------------------
#     # STEP 1: CREATE EMBEDDING
#     # -------------------------------
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=request.question
#     )
#     query_embedding = response.data[0].embedding

#     # -------------------------------
#     # STEP 2: RETRIEVE KB ONLY
#     # -------------------------------
#     kb_results = query_kb_embeddings_filtered(
#         query_embedding,
#         [request.kb_id],
#         top_k=5
#     )

#     # -------------------------------
#     # STEP 3: BUILD CONTEXT
#     # -------------------------------
#     context = ""
#     for match in kb_results.matches:
#         context += match.metadata.get("text", "") + "\n"

#     context = context[:3000]

#     # -------------------------------
#     # STEP 4: GENERATE ANSWER
#     # -------------------------------
#     prompt = build_prompt(request.question, context)

#     response = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         temperature=0.7,
#         max_tokens=220,
#         messages=[
#             {"role": "system", "content": "You are a careful domain expert."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     answer = response.choices[0].message.content

#     # -------------------------------
#     # STEP 5: STORE
#     # -------------------------------
#     qna_id = insert_qna_sl(
#         request.kb_id,
#         request.question,
#         answer
#     )

#     # -------------------------------
#     # STEP 6: RETURN
#     # -------------------------------
#     return {
#         "qna_id": qna_id,
#         "answer": answer
#     }


# @app.post("/qna_sl_validation")
# def qna_sl_validation(request: QnaSLValidationRequest):

#     table_map = {
#         "sl": "qna_sl_logs",
#         "generated": "generated_qnas"
#     }

#     if request.source_type not in table_map:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid source_type"
#         )

#     table_name = table_map[request.source_type]

#     # -------------------------------
#     # STEP 1: FETCH ORIGINAL QNA
#     # -------------------------------
#     record = get_qna_sl(
#         table_name,
#         request.qna_id
#     )

#     if not record:
#         raise HTTPException(status_code=404, detail="QnA not found")

#     # -------------------------------
#     # STEP 2: DECIDE FINAL ANSWER
#     # -------------------------------
#     original_answer = (
#         record.get("llm_answer")
#         or record.get("answer")
#     )

#     if request.is_valid:

#         final_answer = original_answer

#     else:

#         if not request.corrected_answer:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Corrected answer required when invalid"
#             )

#         final_answer = request.corrected_answer

#     # -------------------------------
#     # STEP 3: UPDATE DB
#     # -------------------------------
#     update_qna_sl_validation(
#         table_name=table_name,
#         qna_id=request.qna_id,
#         is_valid=request.is_valid,
#         llm_answer=original_answer,
#         corrected_answer=request.corrected_answer
#     )

#     # -------------------------------
#     # STEP 4: CREATE EMBEDDING
#     # -------------------------------
#     text = record['question']

#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     embedding = response.data[0].embedding

#     print("KB_ID SAVED:", record["kb_id"])
#     print("QUESTION SAVED:", record["question"])

#     # -------------------------------
#     # STEP 5: STORE IN VECTOR DB
#     # -------------------------------
#     upsert_embeddings(
#         file_id=f"{request.source_type}_{request.qna_id}",
#         chunks=[text],
#         embeddings=[embedding],
#         metadata={
#             "type": "qna_sl",
#             "kb_id": str(record["kb_id"]).strip(),
#             "question": record["question"],
#             "answer": final_answer,
#             "source_type": request.source_type
#         }
#     )

#     # -------------------------------
#     # STEP 6: RESPONSE
#     # -------------------------------
#     return {
#         "status": "validated and learned"
#     }


# @app.post("/qna_ml_submit")
# def qna_ml_submit(request: QnaMLRequest):

#     results = []

#     for qna_id in request.qna_ids:

#         record = get_qna_sl(
#             "qna_sl_logs",
#             qna_id
#         )

#         if not record:
#             results.append({
#                 "qna_id": qna_id,
#                 "status": "not_found"
#             })
#             continue

#         final_answer = record["corrected_answer"]

#         payload = {
#             "question": record["question"],
#             "answer": final_answer,
#             "kb_id": record["kb_id"]
#         }

#         # -------------------------------
#         # Step 2: Call ML pipeline (TEMP MOCK)
#         # -------------------------------
#         success = True

#         # -------------------------------
#         # Step 3: Update flag
#         # -------------------------------
#         if success:
#             mark_qna_ml_ready(qna_id)

#         results.append({
#             "qna_id": qna_id,
#             "ml_status": success
#         })

#     return {
#         "results": results
#     }


@app.post("/qna_sl_search")
def qna_sl_search(request: QnaSearchRequest):

    return qna_sl_search_service(
        request.question,
        request.kb_id
    )

# @app.post("/qna_generate")
# def qna_generate(request: QnaGenerateRequest):

#     qnas = generate_qnas(request.kb_id)

#     return {
#         "status": "success",
#         "message": "5 qnas created",
#         "data": qnas
#     }


# @app.post("/delete_qna_sl")
# def delete_qna_sl(request: DeleteQnaSLRequest):

#     qna_id = request.qna_id

#     # -------------------------------
#     # STEP 1: FETCH RECORD
#     # -------------------------------
#     record = get_qna_sl(
#         "generated_qnas",
#         qna_id
#     )

#     if not record:
#         raise HTTPException(
#             status_code=404,
#             detail="QnA not found"
#         )

#     try:

#         # -------------------------------
#         # STEP 2: DELETE EMBEDDINGS
#         # -------------------------------
#         delete_qna_embeddings(
#             "generated",
#             qna_id
#         )

#         # -------------------------------
#         # STEP 3: DELETE DB RECORD
#         # -------------------------------
#         delete_qna_record(
#             "generated_qnas",
#             qna_id
#         )

#     except Exception as e:

#         print(f"⚠️ QnA delete error: {e}")

#         raise HTTPException(
#             status_code=500,
#             detail="Failed to delete QnA"
#         )

#     return {
#         "status": "success",
#         "message": "QnA deleted successfully"
#     }


# -------------------------------------------------
# DEBUG ENDPOINT - Search KB for a specific yoga
# Used during retrieval development.
# Not used by ask_question or retrieval_test.
# -------------------------------------------------

# @app.get("/find_yoga")
# def find_yoga(kb_id: str, yoga: str):

#     chunks = get_all_kb_chunks(kb_id)

#     matches = []

#     for chunk in chunks:

#         text = chunk.metadata.get("text", "")

#         if yoga.lower() in text.lower():

#             matches.append(text[:1000])

#     return matches[:5]


# @app.post("/retrieval_test")
# async def retrieval_test(request: RetrievalTestRequest):
#     try:

#         print("\n========== RETRIEVAL TEST REQUEST ==========")
#         print("QUESTION:", request.question)

#         print("REQUEST DICT:")
#         print(request.dict())

#         print("===========================================")


#         # --------------------------------
#         # EMBEDDING
#         # --------------------------------
#         response = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=request.question
#         )

#         query_embedding = response.data[0].embedding


#         # --------------------------------
#         # RETRIEVAL PIPELINE
#         # --------------------------------
#         print("\n========== RETRIEVAL PIPELINE ==========")

#         # -------------------------------
#         # Chart Retrieval
#         # -------------------------------
#         chart_details = []
#         all_chart_matches = []

#         if request.chart_ids:

#             chart_details = get_chart_details_bulk(request.chart_ids)

#             for chart in chart_details:

#                 results = query_chart_embeddings(
#                     query_embedding,
#                     chart["user_id"],
#                     chart["profile_id"],
#                     chart["chart_id"],
#                     top_k=request.top_k
#                 )

#                 all_chart_matches.extend(results.matches)

#         semantic_results = all_chart_matches

#         # -------------------------------
#         # Knowledge Base Retrieval
#         # -------------------------------
#         exact_match = None
#         kb_results = None

#         if request.kb_id:

#             exact_match = find_exact_kb_match(
#                 request.kb_id[0],
#                 request.question
#             )

#             if exact_match:

#                 print("USING EXACT MATCH")

#                 kb_results = [exact_match]

#             else:

#                 print("USING VECTOR SEARCH")

#                 kb_results = query_kb_embeddings_filtered(
#                     query_embedding,
#                     request.kb_id,
#                     top_k=request.top_k
#                 )

#         # --------------------------------
#         # NOISE FILTER
#         # --------------------------------

#         def is_noise_chunk(text):

#             text_lower = text.lower()

#             if "appendix" in text_lower:
#                 return True

#             if "contents" in text_lower:
#                 return True

#             numbers = re.findall(r"\d+", text)

#             if len(numbers) > 20:
#                 return True

#             return False

#         filtered_matches = []

#         if semantic_results:

#             filtered_matches = [
#                 match
#                 for match in semantic_results
#                 if not is_noise_chunk(
#                     match.metadata.get("text", "")
#                 )
#             ]

#         # --------------------------------
#         # DEBUG
#         # --------------------------------
#         if semantic_results:

#             print("\n===== ORIGINAL PINECONE =====")

#             for idx, match in enumerate(
#                 semantic_results[:20]
#             ):

#                 print(
#                     idx + 1,
#                     round(match.score, 4),
#                     match.metadata.get(
#                         "text",
#                         ""
#                     )[:150]
#                 )

#             print("\n===== AFTER NOISE FILTER =====")

#             for idx, match in enumerate(
#                 filtered_matches[:20]
#             ):

#                 print(
#                     idx + 1,
#                     round(match.score, 4),
#                     match.metadata.get(
#                         "text",
#                         ""
#                     )[:150]
#                 )

#         # --------------------------------
#         # SIMPLE RERANKING
#         # --------------------------------

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

#         chart_matches = filtered_matches

#         context = build_context(
#             chart_matches,
#             kb_results
#         )

#         print("\n========== CONTEXT DEBUG ==========")
#         print("Context length:", len(context))
#         print("Retrieved chart chunks:", len(filtered_matches))
#         print(
#             "KB chunks:",
#             len(kb_results)
#             if isinstance(kb_results, list)
#             else len(kb_results.matches)
#             if kb_results
#             else 0
#         )

#         # --------------------------------
#         # RESPONSE
#         # --------------------------------

#         return {

#             "status": "success",

#             "question": request.question,

#             "chart_ids": request.chart_ids,

#             "kb_id": request.kb_id,

#             "top_k": request.top_k,

#             "embedding_dimension": len(query_embedding),

#             "filtered_chart_match_count": len(filtered_matches),

#             "exact_match_used": isinstance(kb_results, list),

#             "kb_match_count": (
#                 len(kb_results)
#                 if isinstance(kb_results, list)
#                 else len(kb_results.matches)
#                 if kb_results
#                 else 0
#             ),

#             "semantic_matches": [

#                 {
#                     "rank": idx + 1,
#                     "score": round(reranked[idx][0], 4),
#                     "file_id": match.metadata.get("file_id"),
#                     "text": match.metadata.get("text", "")[:500]
#                 }

#                 for idx, match in enumerate(
#                     filtered_matches[:10]
#                 )
#             ],

#             "context": context
#         }
        
#     except Exception as e:

#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )