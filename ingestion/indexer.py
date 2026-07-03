import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from ingestion.parser import parse_ruling
from ingestion.chunker import chunk_section, Chunk

load_dotenv()

COLLECTION = os.getenv("COLLECTION_NAME", "bge_rulings")
DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 1024

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)


def get_client() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY"))


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def _embed_dense(texts: list[str]) -> list[list[float]]:
    # multilingual-e5 requires "passage: " prefix at index time
    prefixed = [f"passage: {t}" for t in texts]
    return [v.tolist() for v in dense_model.embed(prefixed)]


def _embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    results = []
    for sv in sparse_model.embed(texts):
        results.append(models.SparseVector(
            indices=sv.indices.tolist(),
            values=sv.values.tolist(),
        ))
    return results


def index_chunks(chunks: list[Chunk], client: QdrantClient, batch_size: int = 32) -> None:
    texts = [c.text for c in chunks]
    dense_vecs = _embed_dense(texts)
    sparse_vecs = _embed_sparse(texts)

    points = []
    for chunk, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        points.append(models.PointStruct(
            id=abs(hash(chunk.id)) % (2**63),
            vector={"dense": dv, "sparse": sv},
            payload={
                "text": chunk.text,
                "case_number": chunk.case_number,
                "year": chunk.year,
                "division": chunk.division,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            },
        ))

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i:i + batch_size])


def index_raw_dir(raw_dir: Path = Path("data/raw")) -> None:
    client = get_client()
    ensure_collection(client)
    html_files = list(raw_dir.glob("*.html"))
    print(f"Indexing {len(html_files)} rulings...")
    for html_file in tqdm(html_files):
        case_number = html_file.stem.replace("_", " ")
        html = html_file.read_text(encoding="utf-8")
        sections = parse_ruling(html, case_number)
        chunks = [c for s in sections for c in chunk_section(s)]
        index_chunks(chunks, client)
    print(f"Done. Collection: {COLLECTION}")


if __name__ == "__main__":
    index_raw_dir()
