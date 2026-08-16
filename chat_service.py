import time

from vector_db import query_embeddings

from llm_client import generate_answer

from embedder_client import generate_query_embedding

from retrieval_client import retrieve_context

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


def query_docs_service(request):

    # Create embedding for query
    query_embedding = generate_query_embedding(
        request.query
    )

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

    query_embedding = generate_query_embedding(
        request.question
    )

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

    query_embedding = generate_query_embedding(
        question
    )

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
    # STEP 1: INIT
    # -------------------------------
    qna_id = None
    chart_details = []

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
    # STEP 3: RETRIEVAL SERVICE
    # -------------------------------
    retrieval_start = time.perf_counter()

    retrieval_result = retrieve_context(
        question=request.question,
        chart_details=chart_details,
        kb_ids=kb_ids,
        sl_ids=request.sl_id,
        previous_question=request.previous_question,
        previous_answer=request.previous_answer,
    )

    ttl_retrieval = round(
        (time.perf_counter() - retrieval_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 4: RETRIEVAL RESPONSE
    # -------------------------------
    context = retrieval_result.get(
        "context",
        ""
    )

    use_sl_as_context = retrieval_result.get(
        "used_sl",
        False
    )

    used_kb = retrieval_result.get(
        "used_kb",
        use_kb
    )

    used_chart = retrieval_result.get(
        "used_chart",
        bool(use_chart)
    )

    retrieval_rttl = retrieval_result.get(
        "rttl",
        ttl_retrieval
    )

    retrieval_c_ttl = retrieval_result.get(
        "c_ttl",
        []
    )

    print("\n--- RETRIEVAL RESULT ---")
    print("USED SL:", use_sl_as_context)
    print("USED KB:", used_kb)
    print("USED CHART:", used_chart)
    print("RETRIEVAL RTTL:", retrieval_rttl)

    print("\n--- FINAL CONTEXT ---")
    print(context[:1000])

    # -------------------------------
    # STEP 5: GENERATE ANSWER
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
    # STEP 6: STORE ANSWER
    # -------------------------------
    delivery_start = time.perf_counter()

    if qna_id:
        update_qna_answer(
            qna_id,
            answer
        )

    ttl_delivery = round(
        (time.perf_counter() - delivery_start) * 1000,
        2
    )

    # -------------------------------
    # STEP 7: RESPONSE
    # -------------------------------
    c_rttl = round(
        (time.time() - start_time) * 1000,
        2
    )

    # Retrieval service already provides
    # its internal stage-level metrics.
    ttl_breakdown = retrieval_c_ttl.copy()

    # Add Chat-side stages.
    ttl_breakdown.extend([
        {
            "stage": "ttl_retrieval_total",
            "time_ms": ttl_retrieval
        },
        {
            "stage": "ttl_reasoning",
            "time_ms": ttl_reasoning
        },
        {
            "stage": "ttl_delivery",
            "time_ms": ttl_delivery
        }
    ])

    print("TOTAL RTTL (ms):", c_rttl)

    print("TTL BREAKDOWN:")
    print(ttl_breakdown)

    print("RETURNING RESPONSE FROM process_question")

    print({
        "source": source_name,
        "used_sl": use_sl_as_context,
        "used_kb": used_kb,
        "used_chart": used_chart,
        "rttl": c_rttl,
        "c_ttl": ttl_breakdown,
        "answer": answer[:50]
    })

    response = {
        "source": source_name,
        "used_sl": use_sl_as_context,
        "used_kb": used_kb,
        "used_chart": used_chart,
        "rttl": c_rttl,
        "c_ttl": ttl_breakdown,
        "answer": answer,
    }

    print("RETURN RESPONSE KEYS:", response.keys())
    print("RETURN RESPONSE:", response)

    return response