from openai_client import client


def generate_query_embedding(question):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    return response.data[0].embedding


def generate_embeddings(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]