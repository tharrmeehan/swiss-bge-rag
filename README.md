# Swiss BGE RAG

Retrieval-augmented Q&A over Swiss Federal Supreme Court rulings (BGE — *Bundesgerichtsentscheide*), published in German, French, and Italian. Ask a legal question, get an answer grounded in the actual ruling text with an enforced case-number citation for every claim — or an explicit refusal if the indexed corpus doesn't cover it.

Built to explore a specific failure mode of naive RAG: legal Q&A is worthless if the system confidently cites the wrong case or fabricates a holding. Every generation call is constrained to answer only from retrieved passages and to cite the source case number, with a hard-coded refusal string when the retrieved context doesn't support an answer.

## Live demo

[huggingface.co/spaces/tharrmeehan/swiss-bge-rag](https://huggingface.co/spaces/tharrmeehan/swiss-bge-rag) — small demo corpus, `gpt-4o-mini` generation.

## Key Features

- **Hybrid retrieval**: dense (multilingual-e5-large) + sparse (BM25) search fused server-side via Qdrant's native Reciprocal Rank Fusion — catches both semantic matches and exact statute/article citations (e.g. "Art. 426 ZGB") that dense embeddings alone tend to under-weight.
- **Cross-encoder re-ranking**: a `ms-marco-MiniLM` cross-encoder re-scores the top-20 fused candidates before the top 5 reach the LLM, trading a small latency cost for materially cleaner context.
- **Section-aware chunking**: rulings are parsed into their native *Sachverhalt* (facts) / *Erwägungen* (legal reasoning) / *Dispositiv* (ruling) sections before token-bounded chunking, so retrieval and citations stay meaningful instead of chunking across unrelated content.
- **Citation-enforced generation**: the system prompt requires a `(BGE ... )` citation on every factual claim and a fixed refusal string when the context doesn't contain the answer — verified empirically against a set of mismatched-citation questions (see [Evaluation](#evaluation)).
- **Metadata filtering**: filter by division (I–V) and BGE volume/year range from the Streamlit sidebar, applied as a Qdrant payload filter before retrieval.
- **RAGAS evaluation harness**: faithfulness, answer relevancy, context precision, and context recall, computed against a hand-built grounded test set; swappable to a local Ollama judge for zero-cost runs.
- **Bulk ingestion pipeline**: scraper walks bger.ch's per-volume/division index pages, samples across a configurable BGE volume range, and skips already-fetched rulings on re-run.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit (chat interface, source inspector, sidebar filters) |
| Vector DB | Qdrant (Cloud or local via Docker) — hybrid dense + sparse vectors, RRF fusion, payload filtering |
| Dense embeddings | `intfloat/multilingual-e5-large` via `fastembed` |
| Sparse embeddings | `Qdrant/bm25` via `fastembed` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| Generation | OpenAI `gpt-4o` (or any OpenAI-compatible endpoint, e.g. local Ollama) |
| Evaluation | RAGAS + LangChain (judge LLM wrapper only) |
| Scraping/parsing | `httpx`, `BeautifulSoup4` against bger.ch/search.bger.ch |
| Testing | `pytest` |
| Deployment | Docker, Docker Compose, Hugging Face Spaces (Docker SDK) |

## Architecture

```
scrape BGE HTML (search.bger.ch / bger.ch)
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

## Getting Started

### Prerequisites

- Python 3.12+
- Docker (for local Qdrant) or a [Qdrant Cloud](https://cloud.qdrant.io) cluster
- An OpenAI API key, or a local [Ollama](https://ollama.com) install for a free (weaker) alternative

### Installation

```bash
git clone https://github.com/tharrmeehan/swiss-bge-rag.git
cd swiss-bge-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # fill in OPENAI_API_KEY (and QDRANT_URL/KEY if using Cloud)
```

### Run Qdrant locally

```bash
docker compose up -d               # starts Qdrant on localhost:6333
```

Skip this step if `.env` points at a Qdrant Cloud cluster instead.

### Ingest some rulings

```bash
# fetch a single known ruling
python -c "
import ingestion.scraper as scraper
scraper.fetch_ruling(scraper.ruling_url(148, 'I', 1))
"

# or sample across a BGE volume range (politely rate-limited, ~1.5s/request)
python -c "
from ingestion.scraper import scrape_volume_range
scrape_volume_range(vol_min=147, vol_max=148, cap_per_division=10)
"

python -m ingestion.indexer        # parse, chunk, embed, and upsert into Qdrant
```

### Run the app

```bash
streamlit run app.py               # open http://localhost:8501
```

### Run tests

```bash
pytest
```

## Evaluation

Two separate test sets probe two different things:

**`evaluation/test_set.json`** — grounded coverage: questions drawn from the Regeste (official headnote) of each indexed ruling, plus a few deliberately unanswerable ones, to check both retrieval quality and refusal behavior on in-scope content.

```bash
python -m evaluation.evaluate
```

**`evaluation/citation_robustness_test_set.json`** — a stress test for the opposite failure mode: questions that cite a real-looking but *wrong* BGE number. All 7 were pulled from an external benchmark whose citations, on manual verification against the actual ruling text, turned out to be mismatched (e.g. it claims BGE 139 I 218 is a face-veiling-ban ruling; the real BGE 139 I 218 is a welfare job-placement dispute). The system correctly refused on all 7 rather than fabricating an answer from the real ruling's unrelated content, and correctly answered the 2 questions in that same set whose citations happened to be accurate.

```bash
python -c "from evaluation.evaluate import run_evaluation; from pathlib import Path; print(run_evaluation(Path('evaluation/citation_robustness_test_set.json')))"
```

`generation/chain.py` and `evaluation/evaluate.py` both read `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LLM_MODEL`, so the whole pipeline — generation and the RAGAS judge — can point at a local Ollama instance instead of paying for OpenAI (see `.env.example`). Ollama works but is a noticeably weaker judge and generator than GPT.

RAGAS results (n=26, judge: `gpt-4o-mini`):

| Metric             | Score  |
|--------------------|--------|
| Faithfulness       | 0.5808 |
| Answer Relevancy   | 0.5442 |
| Context Precision  | 0.7480 |
| Context Recall     | 0.7692 |

Retrieval (context precision/recall) is solid. Faithfulness and answer relevancy sitting lower is mostly a corpus-size effect, not a pipeline bug: a small corpus spanning five divisions means the reranker sometimes hands the LLM passages from the right ruling but the wrong paragraph — and the citation-enforced prompt correctly refuses to answer from a page it isn't confident about rather than guessing.

## Limitations

- **Small indexed corpus**: only a couple dozen rulings are indexed in the live/default setup, all from BGE volumes 147–148 plus a handful of manually verified older rulings. A broader sample (~2,000 rulings across BGE 100–150) has been scraped to `data/raw/` but is **not yet indexed** — local CPU embedding via `fastembed` processes one ruling at a time and would take roughly 8–9 hours for the full set on a laptop CPU. Query results outside the indexed volumes will correctly refuse rather than answer.
- **Qdrant free-tier ceiling**: dense vectors are 1024-dim; a full 100–150 volume corpus (~500k+ chunks) is estimated at 3GB+ RAM for the HNSW index, likely exceeding a free-tier Qdrant Cloud cluster. Scaling the corpus up requires either a paid tier or switching vectors to on-disk mode (slower queries).
- **No incremental/scheduled ingestion**: adding new rulings is a manual scrape + reindex step; there's no scheduled job to pull newly published BGE volumes.
- **Scraper is coupled to bger.ch's current HTML structure**: section detection relies on specific `<span class="big bold" id="...">` and `<div class="paraatf">` markup; a site redesign would silently break parsing.
- **No source-side citation verification**: the system will correctly *refuse* to answer when a cited case doesn't support the claimed topic, but it doesn't flag "the citation you gave appears to be wrong" — it just reports the retrieved content doesn't answer the question. Distinguishing "wrong citation" from "topic not covered by our corpus" would need a separate check.
- **Single collection, no multi-tenancy or auth**: the Streamlit app and Qdrant collection are single-user/single-corpus; there's no access control on the public demo.
- **Language handling is implicit**: rulings are indexed as-is in whatever language they were published (de/fr/it); there's no language-aware routing or cross-lingual re-ranking tuning beyond what the multilingual embedding model provides natively.
- **Generation cost**: OpenAI `gpt-4o` calls cost real money at scale; the Ollama fallback avoids that but is a measurably weaker generator and judge (see Evaluation).

## Deploy to Hugging Face Spaces

Streamlit isn't a first-class Spaces SDK anymore, so this deploys as a Docker Space (`Dockerfile` included, runs `streamlit run app.py` on port 7860).

Spaces reads its config from `README.md`'s frontmatter, which GitHub doesn't need — `README-HF.md` carries that frontmatter separately. Swap it in only for the push to the `hf` remote, never commit it over the GitHub-facing `README.md`:

```bash
# Create a Space (or use the web UI at huggingface.co/spaces -> Docker)
python -c "
from huggingface_hub import HfApi
HfApi().create_repo(repo_id='<your-username>/swiss-bge-rag', repo_type='space', space_sdk='docker')
"

git remote add hf https://huggingface.co/spaces/<your-username>/swiss-bge-rag

cp README.md /tmp/README.github.md
cp README-HF.md README.md
git add README.md && git commit -m "chore: swap in HF Spaces README"
git push hf main
git reset --hard HEAD~1              # drop that commit locally, restore GitHub README
```

Add secrets in the Space settings: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (Qdrant Cloud free tier works for a small corpus), and optionally `LLM_MODEL` (defaults to `gpt-4o`; set to `gpt-4o-mini` to control cost on a public demo).
