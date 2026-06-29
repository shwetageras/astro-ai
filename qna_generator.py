import json
import google.generativeai as genai

from kb_builder import client
from vector_db import get_all_kb_chunks
from prompts import build_qna_generation_prompt
from db import insert_generated_qna
from db import get_file_id_from_job
from settings import OPENAI_MODEL, GEMINI_MODEL, QNA_GENERATION_MODEL

import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_qnas_gpt(prompt):

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate educational QnAs."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty response from GPT")

    return content


def generate_qnas_gemini(prompt):

    print(f"GENERATING QNAS USING: {GEMINI_MODEL}")

    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(prompt)

    content = getattr(response, "text", "")

    if not content:
        raise ValueError("Empty response from Gemini")

    return content


def generate_qnas(kb_id):

    print("GENERATING QNAS FOR JOB:", kb_id)

    # -----------------------------------
    # STEP 0: CONVERT JOB_ID → FILE_ID
    # -----------------------------------
    file_id = get_file_id_from_job(kb_id)

    if not file_id:
        raise ValueError("Invalid job_id")

    print("FILE ID:", file_id)

    # -----------------------------------
    # STEP 1: RETRIEVE KB CHUNKS
    # -----------------------------------
    results = get_all_kb_chunks(file_id)

    print("TOTAL MATCHES:", len(results))

    # -----------------------------------
    # STEP 2: BUILD CONTEXT
    # -----------------------------------
    context = ""

    for match in results:
        text = match.metadata.get("text", "")
        context += text + "\n"

    print("CONTEXT LENGTH:", len(context))

    # -----------------------------------
    # SAFETY CHECK
    # -----------------------------------
    if not context.strip():
        print("NO KB CONTEXT FOUND")
        return []

    # Safety limit
    context = context[:12000]

    # -----------------------------------
    # STEP 3: BUILD PROMPT
    # -----------------------------------
    prompt = build_qna_generation_prompt(context)

    # -----------------------------------
    # STEP 4: QNA GENERATION
    # -----------------------------------

    print(f"MODEL USED: {QNA_GENERATION_MODEL.upper()}")

    if QNA_GENERATION_MODEL == "gemini":

        content = generate_qnas_gemini(prompt)

    elif QNA_GENERATION_MODEL == "gpt":

        content = generate_qnas_gpt(prompt)

    else:

        raise ValueError(
            f"Unsupported QNA_GENERATION_MODEL: {QNA_GENERATION_MODEL}"
        )

    # -----------------------------------
    # STEP 5: PARSE RESPONSE
    # -----------------------------------

    print("RAW GPT RESPONSE:")
    print(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by GPT: {e}")

    if "qnas" not in parsed:
        raise ValueError("Invalid GPT response format")
    
    # -----------------------------------
    # STEP 6: RETURN QNAS
    # -----------------------------------
    final_qnas = []

    for item in parsed["qnas"]:

        qna_id = insert_generated_qna(
            kb_id,
            item["question"],
            item["answer"]
        )

        final_qnas.append({
            "qna_id": qna_id,
            "question": item["question"],
            "answer": item["answer"]
        })

    return final_qnas