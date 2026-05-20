import json

from kb_builder import client
from vector_db import get_all_kb_chunks
from prompts import build_qna_generation_prompt
from db import insert_generated_qna
from db import get_file_id_from_job

def generate_qnas(kb_id):

    print("GENERATING QNAS FOR JOB:", kb_id)

    # -----------------------------------
    # STEP 0: CONVERT JOB_ID → FILE_ID
    # -----------------------------------
    file_id = get_file_id_from_job(kb_id)

    if not file_id:
        raise Exception("Invalid job_id")

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
    # STEP 4: GPT GENERATION
    # -----------------------------------
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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

    # -----------------------------------
    # STEP 5: PARSE RESPONSE
    # -----------------------------------
    content = response.choices[0].message.content

    if not content:
        raise Exception("Empty response from GPT")

    print("RAW GPT RESPONSE:")
    print(content)

    parsed = json.loads(content)

    if "qnas" not in parsed:
        raise Exception("Invalid GPT response format")
    
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