import os

from settings import OPENAI_MODEL, OPENAI_MINI_MODEL, GEMINI_MODEL
from kb_builder import client
from prompts import build_prompt
import google.generativeai as genai

from vector_db import query_embeddings

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Chat business logic

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

    print("\n===== TOP CHUNKS AFTER GLOBAL SORT =====")

    for c in all_chunks[:15]:
        print(
            c["source"],
            round(c["score"], 4),
            c["text"][:100]
        )

    all_chunks_backup = all_chunks.copy()

    # -------- Step 3: Filter by threshold --------
    SCORE_THRESHOLD = 0.45
    MAX_CONTEXT_CHUNKS = 6
    MAX_CONTEXT_CHARS = 3000

    filtered_chunks = [c for c in all_chunks if c["score"] >= SCORE_THRESHOLD]

    # Fallback if nothing passes threshold
    if not filtered_chunks:
        filtered_chunks = all_chunks_backup[:15]   # take top 15 anyway

    all_chunks = filtered_chunks

    # -------- Step 4: De-duplicate --------
    selected = []

    for chunk in all_chunks:
        if not any(is_similar(chunk["text"], s["text"]) for s in selected):
            selected.append(chunk)

        if len(selected) >= MAX_CONTEXT_CHUNKS:
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
            context += f"- {c['text'].strip()}\n"

    if kb_data:
        context += "\nKNOWLEDGE BASE:\n"
        for c in kb_data:
            context += f"- {c['text'].strip()}\n"

    print("\n===== FINAL CONTEXT SENT TO GPT =====")
    print(context)
    print("===== END =====")

    print(
        "FINAL CHART CHUNKS:",
        len(chart_data)
    )

    print(
        "FINAL KB CHUNKS:",
        len(kb_data)
    )

    return context[:MAX_CONTEXT_CHARS]


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

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
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

    except Exception as e:

        print("❌ OPENAI ERROR:", str(e))
        raise

def generate_answer_gpt_mini(question, context):

    prompt = build_prompt(question, context)

    max_tokens_value = get_max_tokens(question)

    try:

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

    except Exception as e:

        print("❌ GPT MINI ERROR:", str(e))
        raise


def generate_answer_gemini(question, context):
    prompt = build_prompt(question, context)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
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
            except (AttributeError, IndexError, TypeError):
                pass

        raise ValueError("Gemini returned an empty response.")

    except Exception as e:

        print("❌ GEMINI ERROR:", str(e))
        raise


def query_docs_service(request):

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