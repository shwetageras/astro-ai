from pypdf import PdfReader
from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv
from storage import save_kb_to_s3

# -------------------------------
# PDF READING
# -------------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
# TEXT CHUNKING
# -------------------------------
def chunk_text(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence stays within limit → add
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------------
# OPENAI SETUP
# -------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------
# EMBEDDINGS
# -------------------------------
def create_embeddings(chunks):
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


# -------------------------------
# BUILD KB
# -------------------------------
def build_kb(chunks, embeddings):
    kb = []
    
    for i in range(len(chunks)):
        kb.append({
            "text": chunks[i],
            "embedding": embeddings[i]
        })
    
    return kb


# -------------------------------
# SAVE KB (UPDATED WITH METADATA)
# -------------------------------
def save_kb(kb, file_id):
    
    # ensure kb folder exists
    os.makedirs("kb", exist_ok=True)
    
    file_path = f"kb/{file_id}.json"

    # 🔥 NEW: metadata calculation
    num_chunks = len(kb)

    # each float ~ 4 bytes (approximation)
    embedding_size_bytes = sum(len(item["embedding"]) * 4 for item in kb)

    # 🔥 NEW: wrap with metadata
    kb_with_metadata = {
        "metadata": {
            "num_chunks": num_chunks,
            "embedding_size_bytes": embedding_size_bytes
        },
        "data": kb
    }

    # Save locally (optional)
    with open(file_path, "w") as f:
        json.dump(kb_with_metadata, f, indent=2)

    # 🔥 Upload to S3
    save_kb_to_s3(kb_with_metadata, file_id)