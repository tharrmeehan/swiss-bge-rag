import os
from dataclasses import dataclass
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SearchResult:
    text: str
    case_number: str
    year: int
    division: str
    section: str
    score: float


def hybrid_search(
    client: QdrantClient,
    collection: str,
    query_dense: list[float],
    query_sparse: models.SparseVector,
    top_k: int = 20,
    division: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[SearchResult]:
    conditions = []
    if division:
        conditions.append(models.FieldCondition(key="division", match=models.MatchValue(value=division)))
    if year_min is not None:
        conditions.append(models.FieldCondition(key="year", range=models.Range(gte=year_min)))
    if year_max is not None:
        conditions.append(models.FieldCondition(key="year", range=models.Range(lte=year_max)))
    query_filter = models.Filter(must=conditions) if conditions else None

    hits = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=top_k, filter=query_filter),
            models.Prefetch(query=query_sparse, using="sparse", limit=top_k, filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        SearchResult(
            text=h.payload["text"],
            case_number=h.payload["case_number"],
            year=h.payload["year"],
            division=h.payload["division"],
            section=h.payload["section"],
            score=h.score,
        )
        for h in hits
    ]


def get_client() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
