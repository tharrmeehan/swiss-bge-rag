import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from retrieval.embedder import embed_query
from retrieval.sparse import sparse_query
from retrieval.retriever import hybrid_search, get_client
from retrieval.reranker import rerank
from generation.chain import answer

load_dotenv()
COLLECTION = os.getenv("COLLECTION_NAME", "bge_rulings")
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


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

    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])


if __name__ == "__main__":
    result = run_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for name in METRIC_NAMES:
        values = result[name]
        avg = sum(values) / len(values)
        print(f"{name:<25} {avg:.4f}")
