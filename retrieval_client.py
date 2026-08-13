import requests
import os

RETRIEVAL_URL = os.getenv(
    "RETRIEVAL_URL",
    "http://127.0.0.1:8006"
)


def retrieve_context(
    question,
    chart_details=None,
    kb_ids=None,
    sl_ids=None,
    previous_question=None,
    previous_answer=None,
):
    response = requests.post(
        f"{RETRIEVAL_URL}/retrieve",
        json={
            "question": question,
            "chart_details": chart_details or [],
            "kb_ids": kb_ids or [],
            "sl_ids": sl_ids or [],
            "previous_question": previous_question,
            "previous_answer": previous_answer,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()