from fastembed import SparseTextEmbedding
from qdrant_client import models

_model = SparseTextEmbedding("Qdrant/bm25")


def sparse_query(text: str) -> models.SparseVector:
    sv = next(_model.embed([text]))
    return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
