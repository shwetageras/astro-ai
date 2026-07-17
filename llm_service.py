import os

from settings import (
    OPENAI_MODEL,
    OPENAI_MINI_MODEL,
    GEMINI_MODEL,
)

from openai_client import client
from prompts import build_prompt

import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


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