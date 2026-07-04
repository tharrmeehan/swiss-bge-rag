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

Retrieval-augmented Q&A over Swiss Federal Supreme Court rulings (BGE), in German/French/Italian. Hybrid BM25 + dense retrieval, cross-encoder re-ranking, citation-enforced GPT-4o generation.

Ask a legal question and get an answer grounded in the actual ruling text, with an enforced `(BGE ...)` citation for every claim — or an explicit refusal if the indexed corpus doesn't cover it.

Full source, architecture notes, evaluation results, and setup instructions: [github.com/tharrmeehan/swiss-bge-rag](https://github.com/tharrmeehan/swiss-bge-rag)
