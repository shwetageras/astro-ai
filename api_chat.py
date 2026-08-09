from fastapi import FastAPI
from db import get_job
from typing import List, Optional
from pydantic import BaseModel

from chat_service import (
    query_docs_service,
    welcome_message_service,
    qna_sl_search_service,
    process_question,
)

from llm_client import (
    generate_answer,
    generate_answer_gpt_mini,
    generate_answer_gemini,
)

app = FastAPI()


# -----------------------------------
# REQUEST MODELS
# -----------------------------------

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


# ----------------------------------
# ACTIVE ENDPOINTS
# ----------------------------------

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    return job

@app.post("/query")
def query_docs(request: QueryRequest):
    return query_docs_service(request)

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

@app.post("/welcome_message")
def welcome_message(request: WelcomeRequest):
    return welcome_message_service(request)


# -----------------------------------
# FUTURE / INACTIVE APIS
# -----------------------------------

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
