from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from retrieval_service import retrieve_context


app = FastAPI()


# -----------------------------------
# REQUEST MODELS
# -----------------------------------

class ChartDetails(BaseModel):
    user_id: str
    profile_id: int
    chart_id: str


class RetrievalRequest(BaseModel):
    question: str

    chart_details: List[ChartDetails] = []
    kb_ids: List[str] = []
    sl_ids: List[str] = []

    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None


# -----------------------------------
# RETRIEVAL ENDPOINT
# -----------------------------------

@app.post("/retrieve")
def retrieve(request: RetrievalRequest):

    result = retrieve_context(
        question=request.question,
        chart_details=[
            chart.model_dump()
            for chart in request.chart_details
        ],
        kb_ids=request.kb_ids,
        sl_ids=request.sl_ids,
        previous_question=request.previous_question,
        previous_answer=request.previous_answer,
    )

    return result