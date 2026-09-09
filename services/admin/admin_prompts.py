def build_qna_generation_prompt(context):

    return f"""
You are an expert educator.

Based on the provided knowledge base,
generate 5 meaningful questions and answers.

IMPORTANT RULES:
- Questions should cover important concepts.
- Answers should be concise and clear.
- Avoid duplicate questions.
- Use ONLY the provided context.
- Do not hallucinate outside the KB.

Return STRICT JSON ONLY.

FORMAT:
{{
  "qnas": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}

KNOWLEDGE BASE:
{context}
"""