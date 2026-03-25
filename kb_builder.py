from pypdf import PdfReader
from openai import OpenAI
import os
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


# -------------------------------
# TEXT CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
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
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        
        embeddings.append(response.data[0].embedding)
    
    return embeddings


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