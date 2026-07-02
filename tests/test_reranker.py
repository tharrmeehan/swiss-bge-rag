from retrieval.reranker import rerank
from retrieval.retriever import SearchResult


def _make_result(text: str) -> SearchResult:
    return SearchResult(text=text, case_number="BGE 148 I 1", year=2022, division="I", section="Erwägungen", score=0.5)


def test_rerank_returns_top_k():
    candidates = [_make_result(f"Text {i}") for i in range(10)]
    results = rerank(query="Was hat das Gericht entschieden?", candidates=candidates, top_k=5)
    assert len(results) == 5


def test_rerank_returns_search_results():
    candidates = [_make_result("Verwaltungsrecht"), _make_result("Strafrecht")]
    results = rerank(query="Verwaltungsrecht Kanton", candidates=candidates, top_k=2)
    assert all(isinstance(r, SearchResult) for r in results)


def test_rerank_orders_by_relevance():
    candidates = [
        _make_result("Das Bundesgericht hat die Beschwerde in Strafsachen abgewiesen."),
        _make_result("Gemüseanbau in der Schweiz."),
    ]
    results = rerank(query="Beschwerde Strafsachen Bundesgericht", candidates=candidates, top_k=2)
    # The first result should be more relevant to the query
    assert "Beschwerde" in results[0].text or "Bundesgericht" in results[0].text
