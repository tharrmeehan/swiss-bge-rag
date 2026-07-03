---
title: Swiss BGE RAG
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Swiss BGE RAG

Retrieval-augmented Q&A over German/French/Italian-language Swiss Federal Supreme Court rulings (BGE). Hybrid BM25 + dense retrieval (multilingual-e5-large) fused via Qdrant's native RRF, cross-encoder re-ranking, GPT-4o generation with enforced citations, and RAGAS evaluation.

## Live demo

[huggingface.co/spaces/tharrmeehan/swiss-bge-rag](https://huggingface.co/spaces/tharrmeehan/swiss-bge-rag) — 15 rulings indexed, `gpt-4o-mini` generation.

## Evaluation

A 26-question grounded test set (`evaluation/test_set.json`) covers the 15 BGE rulings currently indexed: 24 answerable questions drawn from each ruling's Regeste (official headnote) plus 2 designed to be unanswerable from the indexed corpus, to check the model refuses to hallucinate. Run it with:

```bash
python -m evaluation.evaluate
```

`generation/chain.py` and `evaluation/evaluate.py` both read `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LLM_MODEL`, so the whole pipeline — generation and the RAGAS judge — can also point at a local [Ollama](https://ollama.com) instance instead of paying for OpenAI (see `.env.example`). Ollama works but a small local model is a noticeably weaker judge and generator than GPT; the numbers below are the real ones.

```markdown
## Evaluation (RAGAS, n=26, judge: gpt-4o-mini)

| Metric             | Score  |
|--------------------|--------|
| Faithfulness       | 0.5808 |
| Answer Relevancy   | 0.5442 |
| Context Precision  | 0.7480 |
| Context Recall     | 0.7692 |
```

Retrieval (context precision/recall) is solid. Faithfulness and answer relevancy sitting lower is mostly the corpus, not the pipeline: 15 rulings spanning five divisions means the reranker sometimes hands the LLM passages from the right ruling but the wrong paragraph, or a mix of one relevant chunk and four unrelated ones — and a citation-enforced prompt correctly refuses to answer from a page it isn't confident about rather than guessing. Cost for a full run: under $0.10 in OpenAI credits.

Expand `test_set.json` further as more rulings get indexed — every grounded pair in it is tied to a ruling actually indexed; adding untested questions against uncrawled content would make the eval numbers meaningless.

## Architecture

```
scrape BGE HTML (bger.ch)
  → parse into sections (Sachverhalt / Erwägungen / Dispositiv)
  → chunk by paragraph, token-bounded
  → embed dense (multilingual-e5-large) + sparse (BM25) → Qdrant
query
  → optional metadata filter (division, year range)
  → hybrid retrieval, top 20, RRF fusion
  → cross-encoder re-rank → top 5
  → GPT-4o, citation-enforced prompt
  → Streamlit UI, Sources panel
```

## Design decisions

- **Hybrid over dense-only retrieval**: BGE rulings are dense with statute citations, article numbers, and Latin legal terms that dense embeddings under-weight; BM25 catches exact-term matches (e.g. "Art. 426 ZGB") that a pure embedding search can miss.
- **Cross-encoder re-ranking**: RRF fusion over two independent retrievers produces noisy top-20 results; a cross-encoder scores query/passage pairs jointly and reliably surfaces the 5 most relevant before they hit the LLM context.
- **Section-aware chunking**: Sachverhalt (facts) and Erwägungen (legal reasoning) serve different query types — keeping them as separate labeled sections (rather than chunking the raw page) lets citations and filters stay meaningful.
- **Citation-enforced prompting**: legal Q&A is worthless without traceable sources; the system prompt forces every factual claim to cite a case number, and forces an explicit "not found" response rather than a hallucinated answer.

## Run locally

```bash
docker compose up -d          # starts Qdrant on :6333
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY

# fetch a few rulings (adjust volume/division/page in ingestion/scraper.py)
python -c "
import ingestion.scraper as scraper
scraper.fetch_ruling(scraper.ruling_url(148, 'I', 1))
"

python -m ingestion.indexer   # embed + index into Qdrant
streamlit run app.py          # open http://localhost:8501
```

## Deploy to Hugging Face Spaces

Streamlit isn't a first-class Spaces SDK anymore, so this deploys as a Docker Space (`Dockerfile` included, runs `streamlit run app.py` on port 7860).

```bash
# Create a Space (or use the web UI at huggingface.co/spaces -> Docker)
python -c "
from huggingface_hub import HfApi
HfApi().create_repo(repo_id='<your-username>/swiss-bge-rag', repo_type='space', space_sdk='docker')
"

git remote add hf https://huggingface.co/spaces/<your-username>/swiss-bge-rag
git push hf main
```

Add secrets in the Space settings: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (Qdrant Cloud free tier works for this corpus size), and optionally `LLM_MODEL` (defaults to `gpt-4o`; set to `gpt-4o-mini` to control cost on a public demo).
