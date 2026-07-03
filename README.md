# Swiss BGE RAG

Retrieval-augmented Q&A over German/French/Italian-language Swiss Federal Supreme Court rulings (BGE). Hybrid BM25 + dense retrieval (multilingual-e5-large) fused via Qdrant's native RRF, cross-encoder re-ranking, GPT-4o generation with enforced citations, and RAGAS evaluation.

## Live demo

Not yet deployed — see [Deploy to Hugging Face Spaces](#deploy-to-hugging-face-spaces) below.

## Evaluation

A 12-question grounded test set (`evaluation/test_set.json`) covers the 2 BGE rulings indexed so far (BGE 148 I 1, BGE 148 IV 1): 10 answerable questions plus 2 designed to be unanswerable from the indexed corpus. Run it with:

```bash
python -m evaluation.evaluate
```

RAGAS scoring (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) requires `OPENAI_API_KEY` — it has not been run in this environment, so no scores are published here yet. Paste your results below after running it:

```markdown
## Evaluation (RAGAS, n=12)

| Metric             | Score |
|--------------------|-------|
| Faithfulness       | X.XX  |
| Answer Relevancy   | X.XX  |
| Context Precision  | X.XX  |
| Context Recall     | X.XX  |
```

Expand `test_set.json` toward 50 questions as more rulings get indexed — the 12 grounded pairs currently in it are tied to the two rulings actually indexed; adding untested questions against uncrawled content would make the eval numbers meaningless.

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

```bash
# Create a new Space at huggingface.co/spaces -> Streamlit runtime
git remote add hf https://huggingface.co/spaces/<your-username>/swiss-bge-rag
git push hf main
```

Add secrets in the Space settings: `OPENAI_API_KEY`, `QDRANT_URL` (use Qdrant Cloud free tier for persistence).
