from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "astro-ai-index"

# Create index if not exists
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # same as embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

def upsert_embeddings(file_id, chunks, embeddings, metadata=None):
    vectors = []

    for i in range(len(chunks)):
        vectors.append({
            "id": f"{file_id}_{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunks[i],
                "file_id": file_id,
                **(metadata or {})   
            }
        })

    index.upsert(vectors=vectors)

def query_embeddings(query_embedding, top_k=5):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results


def query_chart_embeddings(query_embedding, user_id, profile_id, chart_id, top_k=3):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={
            "user_id": user_id,
            "profile_id": profile_id,
            "chart_id": chart_id
        }
    )
    return results


def query_kb_embeddings(query_embedding, top_k=2):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={
            "user_id": {"$exists": False}
        }
    )
    return results