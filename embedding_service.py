from kb_builder import client


def generate_query_embedding(question):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    return response.data[0].embedding