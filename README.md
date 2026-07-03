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

Not yet deployed — see [Deploy to Hugging Face Spaces](#deploy-to-hugging-face-spaces) below.

## Evaluation

A 26-question grounded test set (`evaluation/test_set.json`) covers the 15 BGE rulings currently indexed: 24 answerable questions drawn from each ruling's Regeste (official headnote) plus 2 designed to be unanswerable from the indexed corpus, to check the model refuses to hallucinate. Run it with:

```bash
python -m evaluation.evaluate
```

`generation/chain.py` and `evaluation/evaluate.py` both read `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LLM_MODEL`, so the whole pipeline — generation and the RAGAS judge — can point at a local [Ollama](https://ollama.com) instance instead of paying for OpenAI (see `.env.example`).

```markdown
## Evaluation (RAGAS, n=26, judge: llama3.1 via local Ollama)

| Metric             | Score  |
|--------------------|--------|
| Faithfulness       | 0.4872 |
| Answer Relevancy   | 0.3968 |
| Context Precision  | n/a — see note below |
| Context Recall     | n/a — see note below |
```

**Note on these numbers**: this run used a free local 8B model (`llama3.1` via Ollama) as both the answer generator *and* the RAGAS judge, on a memory-constrained laptop — not GPT-4o. `context_precision`/`context_recall` never completed a clean pass locally: running the full generation + 4-metric judge pipeline concurrently pushed the machine into heavy memory pressure (embedding model + reranker + Qdrant client + a resident 4.9GB LLM all competing for RAM), which degraded response latency until judge calls started timing out. `faithfulness` and `answer_relevancy` completed with zero errors across all 26 questions, so those two are trustworthy as a *relative, same-model-family* baseline — not as an absolute quality bar. A real deployment would use `gpt-4o` (fast, hosted, no local memory contention) for both generation and judging; expect materially different absolute scores, and rerun context_precision/context_recall with that setup for real numbers on those two metrics.

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

```bash
# Create a new Space at huggingface.co/spaces -> Streamlit runtime
git remote add hf https://huggingface.co/spaces/<your-username>/swiss-bge-rag
git push hf main
```

Add secrets in the Space settings: `OPENAI_API_KEY`, `QDRANT_URL` (use Qdrant Cloud free tier for persistence).
