from pypdf import PdfReader
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_embeddings(chunks):
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        
        embeddings.append(response.data[0].embedding)
    
    return embeddings


def build_kb(chunks, embeddings):
    kb = []
    
    for i in range(len(chunks)):
        kb.append({
            "text": chunks[i],
            "embedding": embeddings[i]
        })
    
    return kb

def save_kb(kb, file_id):
    
    # ensure kb folder exists
    os.makedirs("kb", exist_ok=True)
    
    file_path = f"kb/{file_id}.json"
    
    with open(file_path, "w") as f:
        json.dump(kb, f, indent=2)

# if __name__ == "__main__":
#     text = read_pdf("sample.pdf")
#     chunks = chunk_text(text)
#     embeddings = create_embeddings(chunks)
    
#     kb = build_kb(chunks, embeddings)

#     save_kb(kb)

#     print("KB saved successfully!")