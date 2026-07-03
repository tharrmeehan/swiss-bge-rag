import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from retrieval.embedder import embed_query
from retrieval.sparse import sparse_query
from retrieval.retriever import hybrid_search, get_client
from retrieval.reranker import rerank
from generation.chain import answer

load_dotenv()
COLLECTION = os.getenv("COLLECTION_NAME", "bge_rulings")
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def ragas_judge_llm():
    """RAGAS's evaluate() defaults to real OpenAI regardless of our chain.py env override,
    so the judge LLM/embeddings need to be passed explicitly. Reuses OPENAI_BASE_URL/
    OPENAI_API_KEY/LLM_MODEL to point at Ollama for free local runs."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    embed_model = os.getenv("RAGAS_EMBED_MODEL", "nomic-embed-text" if "localhost" in base_url else "text-embedding-3-small")
    llm = LangchainLLMWrapper(ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        base_url=base_url, api_key=api_key, model=embed_model,
        check_embedding_ctx_length=False,  # Ollama's endpoint wants raw strings, not tiktoken-encoded token arrays
    ))
    return llm, embeddings


def run_evaluation(test_set_path: Path = Path("evaluation/test_set.json")):
    test_set = json.loads(test_set_path.read_text())
    client = get_client()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_set:
        q = item["question"]
        gt = item["ground_truth"]
        results = hybrid_search(client, COLLECTION, embed_query(q), sparse_query(q), top_k=20)
        top5 = rerank(q, results, top_k=5)
        ans = answer(q, top5)
        questions.append(q)
        answers.append(ans)
        contexts.append([r.text for r in top5])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    llm, embeddings = ragas_judge_llm()
    # low max_workers: a single local Ollama instance serves one request at a time,
    # RAGAS's default of 16 concurrent workers just queues until they all time out
    run_config = RunConfig(max_workers=2, timeout=300)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )


if __name__ == "__main__":
    result = run_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for name in METRIC_NAMES:
        values = [v for v in result[name] if v == v]  # drop NaN (failed judge calls, e.g. timeouts)
        avg = sum(values) / len(values) if values else float("nan")
        print(f"{name:<25} {avg:.4f}  (n={len(values)}/{len(result[name])})")
