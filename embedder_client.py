import requests

EMBEDDER_URL = "http://127.0.0.1:8002"


def generate_query_embedding(text: str):
    raise Exception("EMBEDDER CLIENT EXECUTED")
    
    response = requests.post(
        f"{EMBEDDER_URL}/embed",
        json={"text": text},
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["embedding"]


def generate_embeddings(texts: list[str]):
    response = requests.post(
        f"{EMBEDDER_URL}/embed-batch",
        json={"texts": texts},
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embeddings"]