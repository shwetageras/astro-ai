import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4.1-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

QNA_GENERATION_MODEL = os.getenv("QNA_GENERATION_MODEL", "gemini").lower()


EMBEDDER_URL = os.getenv(
    "EMBEDDER_URL",
    "http://127.0.0.1:8002"
)