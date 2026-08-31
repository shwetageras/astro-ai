import time

from embedder_client import generate_query_embedding

from vector_db import (
    query_embeddings,
    query_chart_embeddings,
    query_kb_embeddings_filtered,
    get_all_kb_chunks,
    query_qna_sl_embeddings,
)


def is_similar(text1, text2, threshold=0.8):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    overlap = len(words1 & words2) / max(len(words1), 1)

    return overlap > threshold


def build_context(chart_results, kb_results):

    all_chunks = []

    # Handle both cases: list or Pinecone object
    chart_matches = (
        chart_results.matches
        if hasattr(chart_results, "matches")
        else chart_results
    )

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
    all_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

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
    MAX_CONTEXT_CHARS = 16000

    filtered_chunks = [
        c for c in all_chunks
        if c["score"] >= SCORE_THRESHOLD
    ]

    # Fallback if nothing passes threshold
    if not filtered_chunks:
        filtered_chunks = all_chunks_backup[:15]

    all_chunks = filtered_chunks

    # -------- Step 4: De-duplicate --------
    selected = []

    for chunk in all_chunks:

        if not any(
            is_similar(chunk["text"], s["text"])
            for s in selected
        ):
            selected.append(chunk)

        if len(selected) >= MAX_CONTEXT_CHUNKS:
            break

    # -------- Step 5: Separate again --------
    chart_data = [
        c for c in selected
        if c["source"] == "chart"
    ]

    kb_data = [
        c for c in selected
        if c["source"] == "kb"
    ]

    if not chart_data:
        chart_data = [
            c for c in all_chunks
            if c["source"] == "chart"
        ][:2]

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

    print("\n===== FINAL CONTEXT SENT TO LLM =====")
    print(context)
    print("===== END =====")

    print("FINAL CHART CHUNKS:", len(chart_data))
    print("FINAL KB CHUNKS:", len(kb_data))

    return context[:MAX_CONTEXT_CHARS]


def find_exact_kb_match(kb_id, question):

    print(
        "========== NEW FIND_EXACT_KB_MATCH RUNNING =========="
    )

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

        if (
            definition_pos >= 0
            and abs(definition_pos - position) < 200
        ):
            score += 1000

        if position >= 0:
            score += max(0, 500 - position)

        print(
            f"CANDIDATE score={score} "
            f"pos={position} "
            f"def={definition_pos}"
        )

        print(text[:250])

        if score > best_score:
            best_score = score
            best_match = chunk

    if best_match:

        print("BEST SCORE:", best_score)

        print(
            "BEST MATCH:",
            best_match.metadata.get("text", "")[:300]
        )

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


def retrieve_context(
    question,
    chart_details=None,
    kb_ids=None,
    sl_ids=None,
    previous_question=None,
    previous_answer=None,
):
    """
    Execute the retrieval and context-building pipeline.

    This preserves the current retrieval behavior from process_question()
    without handling QnA database writes or LLM generation.
    """

    start_time = time.time()

    chart_details = chart_details or []
    kb_ids = kb_ids or []
    sl_ids = sl_ids or []

    # --------------------------------
    # FLAGS
    # --------------------------------
    use_chart = bool(
        chart_details
    )

    use_kb = bool(
        kb_ids
        and kb_ids != ["0"]
        and kb_ids != [""]
    )

    use_sl = bool(
        sl_ids
        and sl_ids != ["0"]
        and sl_ids != [""]
    )

    print("KB IDS:", kb_ids)
    print("USE CHART:", use_chart)
    print("USE KB:", use_kb)
    print("SL IDS:", sl_ids)
    print("USE SL:", use_sl)

    # --------------------------------
    # TIMINGS
    # --------------------------------
    ttl_q_embedding = 0
    ttl_sl_match = 0
    ttl_kb_context = 0
    ttl_chart_context = 0
    ttl_context_build = 0

    # --------------------------------
    # STEP 0: SL SEARCH
    # --------------------------------
    use_sl_as_context = False
    sl_context = None

    if use_sl:

        sl_start = time.perf_counter()

        # Preserve current behavior:
        # qna_sl_search_service() uses kb_ids[0]
        kb_id = kb_ids[0] if kb_ids else ""

        sl_result = qna_sl_search_service(
            question,
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

    # --------------------------------
    # SL DEBUG
    # --------------------------------
    print("SL FOUND:", sl_found)
    print("FIRST MATCH SCORE:", sl_score)
    print(
        "SECOND MATCH SCORE:",
        second["score"] if second else None
    )
    print("MATCH COUNT:", len(matches))
    print(
        "MATCH SCORES (top 5):",
        [round(m["score"], 2) for m in matches[:5]]
    )

    # --------------------------------
    # SL DECISION LOGIC
    # --------------------------------
    if sl_found and sl_score is not None:

        # Preserve existing threshold:
        # >= 0.60 goes into context
        if sl_score >= 0.60:

            selected_matches = [
                m
                for m in sorted(
                    matches,
                    key=lambda x: x["score"],
                    reverse=True
                )
                if m["score"] >= 0.60
                and m.get("answer")
            ][:3]

            formatted_matches = []

            for m in selected_matches:

                answer = m.get("answer") or ""
                qna_question = m.get("question") or ""

                short_answer = (
                    answer[:300].rsplit(" ", 1)[0] + "..."
                    if len(answer) > 300
                    else answer
                )

                formatted_matches.append(
                    f"Previous QnA (score {round(m['score'], 2)}):\n"
                    f"Q: {qna_question}\n"
                    f"A: {short_answer}"
                )

            sl_context = "\n\n".join(
                formatted_matches
            )

            use_sl_as_context = True

        else:

            use_sl_as_context = False
            sl_context = None

    print(
        "USING SL CONTEXT:",
        use_sl_as_context
    )

    # --------------------------------
    # STEP 1: QUERY EMBEDDING
    # --------------------------------
    embedding_start = time.perf_counter()

    query_embedding = generate_query_embedding(
        question
    )

    ttl_q_embedding = round(
        (time.perf_counter() - embedding_start) * 1000,
        2
    )

    # --------------------------------
    # STEP 2: CHART RETRIEVAL
    # --------------------------------
    chart_start = time.perf_counter()

    all_chart_matches = []

    if use_chart:

        for chart in chart_details:

            results = query_chart_embeddings(
                query_embedding,
                chart["user_id"],
                chart["profile_id"],
                chart["chart_id"],
                top_k=10
            )

            all_chart_matches.extend(
                results.matches
            )

    ttl_chart_context = round(
        (time.perf_counter() - chart_start) * 1000,
        2
    )

    # --------------------------------
    # STEP 3: KB RETRIEVAL
    # --------------------------------
    kb_results = None

    kb_start = time.perf_counter()

    if use_kb:

        exact_match = find_exact_kb_match(
            kb_ids[0],
            question
        )

        if exact_match:

            print("USING EXACT MATCH")

            kb_results = [
                exact_match
            ]

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

    # --------------------------------
    # STEP 4: KB DEBUG
    # --------------------------------
    print("\n--- KB RESULTS ---")

    if kb_results:

        if isinstance(kb_results, list):

            for match in kb_results:

                print(
                    "Score: EXACT_MATCH |",
                    match.metadata.get(
                        "text",
                        ""
                    )[:100]
                )

        else:

            for match in kb_results.matches:

                print(
                    f"Score: {round(match.score, 3)} | "
                    f"{match.metadata.get('text', '')[:100]}"
                )

    # --------------------------------
    # STEP 5: CONTEXT BUILDING
    # --------------------------------
    prompt_start = time.perf_counter()

    if not use_chart and not use_kb:

        context = ""

    else:

        context = build_context(
            all_chart_matches,
            kb_results
        )

    # --------------------------------
    # STEP 6: INJECT SL CONTEXT
    # --------------------------------
    print(
        "INJECTING SL INTO CONTEXT:",
        use_sl_as_context
    )

    if use_sl_as_context and sl_context:

        context = (
            "Relevant past learned answers "
            "(use as base, refine and synthesize):\n"
            f"{sl_context}\n\n"
            f"{context}"
        )

    print("\n--- CONTEXT BEFORE CONVERSATION ---")
    print(context[:1000])

    # --------------------------------
    # STEP 7: PREVIOUS CONVERSATION
    # --------------------------------
    if previous_question and previous_answer:

        conversation_context = f"""
PREVIOUS CONVERSATION:

User: {previous_question}

Astrologer: {previous_answer}
"""

        context = (
            f"{conversation_context}\n\n"
            f"{context}"
        )

    ttl_context_build = round(
        (time.perf_counter() - prompt_start) * 1000,
        2
    )

    # --------------------------------
    # TOTAL RETRIEVAL TIME
    # --------------------------------
    retrieval_rttl = round(
        (time.time() - start_time) * 1000,
        2
    )

    retrieval_metrics = [
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
        }
    ]

    print(
        "RETRIEVAL RTTL (ms):",
        retrieval_rttl
    )

    print(
        "RETRIEVAL METRICS:",
        retrieval_metrics
    )

    return {
        "context": context,
        "used_sl": use_sl_as_context,
        "used_kb": use_kb,
        "used_chart": use_chart,
        "rttl": retrieval_rttl,
        "c_ttl": retrieval_metrics
    }
