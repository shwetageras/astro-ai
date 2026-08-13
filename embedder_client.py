import requests

from settings import EMBEDDER_URL


def generate_query_embedding(text: str):
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