import os
import time

from settings import OPENAI_MODEL, OPENAI_MINI_MODEL, GEMINI_MODEL
from kb_builder import client
from prompts import build_prompt
import google.generativeai as genai

from vector_db import query_embeddings

from vector_db import (
    query_chart_embeddings,
    query_kb_embeddings_filtered,
    get_all_kb_chunks,
    query_qna_sl_embeddings,
)

from db import (
    get_chart_details_bulk,
    insert_qna,
    update_qna_answer,
)

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


def welcome_message_service(request):

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


def qna_sl_search_service(question, kb_id):

    print("INSIDE QNA SL SEARCH")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = response.data[0].embedding

    results = query_qna_sl_embeddings(
        query_embedding,
        kb_id
    )

    if not results.matches:
        return {
            "found": False,
            "matches": []
        }

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


def process_question(
    request,
    answer_generator,
    source_name,
    collect_metrics=True,
):

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
    ttl_reasoning = 0
    ttl_delivery = 0

    # -------------------------------
    # STEP 0.1: Check SL memory
    # -------------------------------
    if use_sl:

        sl_start = time.perf_counter()

        sl_result = qna_sl_search_service(
            request.question,
            kb_id
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
    # STEP 5: KB RETRIEVAL
    # -------------------------------

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

    answer = answer_generator(
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
            "time_ms": ttl_context_build
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
        "source": source_name,

        "used_sl": use_sl_as_context,
        "used_kb": use_kb,
        "used_chart": use_chart,

        "rttl": c_rttl,

        "c_ttl": ttl_breakdown,

        "answer": answer
    }