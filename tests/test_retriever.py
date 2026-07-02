import pytest
from qdrant_client import QdrantClient, models
from retrieval.retriever import hybrid_search, SearchResult


@pytest.fixture
def client_with_data(tmp_path):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.upsert(
        collection_name="test",
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [1.0, 0.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[0], values=[1.0])},
                payload={"text": "Verwaltungsgericht Zürich", "case_number": "BGE 148 I 1", "year": 2022, "division": "I", "section": "Erwägungen"},
            ),
            models.PointStruct(
                id=2,
                vector={"dense": [0.0, 1.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[1], values=[1.0])},
                payload={"text": "Strafrecht Bundesgericht", "case_number": "BGE 147 IV 5", "year": 2021, "division": "IV", "section": "Erwägungen"},
            ),
        ],
    )
    return client


def test_hybrid_search_returns_results(client_with_data):
    results = hybrid_search(
        client=client_with_data,
        collection="test",
        query_dense=[1.0, 0.0, 0.0, 0.0],
        query_sparse=models.SparseVector(indices=[0], values=[1.0]),
        top_k=2,
    )
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert results[0].text != ""


def test_hybrid_search_division_filter(client_with_data):
    results = hybrid_search(
        client=client_with_data,
        collection="test",
        query_dense=[1.0, 0.0, 0.0, 0.0],
        query_sparse=models.SparseVector(indices=[0], values=[1.0]),
        top_k=2,
        division="I",
    )
    assert all(r.division == "I" for r in results)
