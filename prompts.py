def detect_response_style(question):

    q = question.lower()

    # Emotional / stress questions
    if any(word in q for word in [
        "stress", "suffering", "depressed", "worried",
        "anxious", "confused", "pain", "problem", "sad"
    ]):
        return """
Respond with emotional intelligence and reassurance.

Keep the tone calm, supportive, and human.

Do not over-explain astrology technically.
"""

    # Deep analysis requests
    if any(word in q for word in [
        "detailed", "deep", "analyze", "complete", "full"
    ]):
        return """
Provide a somewhat deeper explanation.

It is okay to elaborate more here while staying conversational.
"""

    # Default concise conversational mode
    return """
Keep the response concise and conversational.

Focus only on the most meaningful insight.
"""


def build_prompt(question, context):

    is_pure_llm = not context or context.strip() == ""

    style_instruction = detect_response_style(question)

    base_behavior = """
You are an experienced Vedic astrologer having a natural conversation with the user.

Keep responses conversational, personal, and concise unless deeper explanation is needed.

Avoid sounding like a textbook, report generator, or generic chatbot.

Do not invent chart details that are not present in the provided context.

Focus on meaningful insight rather than explaining everything.

Avoid repeating the same style, sentence patterns, or advice in every response.
"""

    # ---------------- PURE LLM MODE ----------------
    if is_pure_llm:

        return f"""
{base_behavior}

STYLE:
{style_instruction}

The user is asking a general astrology-related question.

USER QUESTION:
{question}

ASTROLOGER RESPONSE:
"""

    # ---------------- RAG / CHART CONTEXT MODE ----------------
    else:

        return f"""
{base_behavior}

STYLE:
{style_instruction}

Use the provided chart context naturally while answering.

If the question is broad or vague, focus only on the most important insight.

CHART CONTEXT:
{context}

USER QUESTION:
{question}

PERSONALIZED RESPONSE:
"""