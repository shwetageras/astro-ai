from fastapi import FastAPI
from pydantic import BaseModel

from embedding_service import (
    generate_query_embedding,
    generate_embeddings,
)

app = FastAPI(title="Embedding Service")


class EmbedRequest(BaseModel):
    text: str


class BatchEmbedRequest(BaseModel):
    texts: list[str]


@app.get("/")
def root():
    return {
        "service": "Embedding Service",
        "status": "running"
    }


@app.post("/embed")
def embed(request: EmbedRequest):
    embedding = generate_query_embedding(request.text)

    return {
        "embedding": embedding
    }


@app.post("/embed-batch")
def embed_batch(request: BatchEmbedRequest):
    embeddings = generate_embeddings(request.texts)

    return {
        "embeddings": embeddings
    }