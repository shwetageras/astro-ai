import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4.1-mini")