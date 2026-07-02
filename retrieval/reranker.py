from sentence_transformers import CrossEncoder
from retrieval.retriever import SearchResult

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
    if not candidates:
        return []
    pairs = [(query, c.text) for c in candidates]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
