import os
import requests

LLM_URL = os.getenv(
    "LLM_URL",
    "http://127.0.0.1:8005"
)


def generate_answer(question: str, context: str):

    response = requests.post(
        f"{LLM_URL}/generate",
        json={
            "question": question,
            "context": context,
            "provider": "gpt"
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["answer"]


def generate_answer_gpt_mini(question: str, context: str):

    response = requests.post(
        f"{LLM_URL}/generate",
        json={
            "question": question,
            "context": context,
            "provider": "gpt-mini"
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["answer"]


def generate_answer_gemini(question: str, context: str):

    response = requests.post(
        f"{LLM_URL}/generate",
        json={
            "question": question,
            "context": context,
            "provider": "gemini"
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["answer"]