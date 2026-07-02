from fastembed import TextEmbedding

_model = TextEmbedding("intfloat/multilingual-e5-large")


def embed_query(text: str) -> list[float]:
    # multilingual-e5 requires "query: " prefix at query time
    vec = next(_model.embed([f"query: {text}"]))
    return vec.tolist()
