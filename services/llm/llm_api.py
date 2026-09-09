from fastapi import FastAPI
from pydantic import BaseModel

from llm_service import (
    generate_answer,
    generate_answer_gemini,
    generate_answer_gpt_mini
)

app = FastAPI()


class LLMRequest(BaseModel):
    question: str
    context: str
    provider: str = "gemini"


@app.post("/generate")
async def generate(request: LLMRequest):

    provider = request.provider.lower()

    try:

        if provider == "gemini":

            answer = generate_answer_gemini(
                request.question,
                request.context
            )

        elif provider == "gpt":

            answer = generate_answer(
                request.question,
                request.context
            )

        elif provider == "gpt-mini":

            answer = generate_answer_gpt_mini(
                request.question,
                request.context
            )

        else:

            return {
                "status": "error",
                "message": f"Unsupported provider: {provider}"
            }

        return {
            "status": "success",
            "provider": provider,
            "answer": answer
        }

    except Exception as e:

        return {
            "status": "error",
            "provider": provider,
            "message": str(e)
        }