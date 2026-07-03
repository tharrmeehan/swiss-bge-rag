import os
import streamlit as st
from dotenv import load_dotenv
from retrieval.embedder import embed_query
from retrieval.sparse import sparse_query
from retrieval.retriever import hybrid_search, get_client, SearchResult
from retrieval.reranker import rerank
from generation.chain import answer

load_dotenv()
COLLECTION = os.getenv("COLLECTION_NAME", "bge_rulings")

st.set_page_config(page_title="BGE RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ Schweizer Bundesgerichts-Assistent")
st.caption("Retrieval-augmented Q&A über BGE-Entscheide · Hybrid Search + Re-Ranking")

# Sidebar filters
with st.sidebar:
    st.header("Filter")
    division = st.selectbox("Abteilung", ["Alle", "I", "II", "III", "IV"])
    year_range = st.slider("Jahrgang (BGE-Band)", min_value=100, max_value=150, value=(130, 150))
    st.divider()
    st.markdown("**Modell:** GPT-4o  \n**Retrieval:** Hybrid BM25 + dense, RRF, Cross-Encoder Re-Ranking")

div_filter = None if division == "Alle" else division


@st.cache_resource
def get_qdrant_client():
    return get_client()


qdrant = get_qdrant_client()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Stellen Sie eine Frage zu Bundesgerichtsentscheiden...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in BGE-Entscheiden..."):
            results = hybrid_search(
                qdrant, COLLECTION, embed_query(query), sparse_query(query),
                top_k=20, division=div_filter,
                year_min=year_range[0], year_max=year_range[1],
            )
            top5 = rerank(query, results, top_k=5)
            response = answer(query, top5)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.sources = top5

if st.session_state.sources:
    with st.expander(f"Quellen ({len(st.session_state.sources)} Entscheide)", expanded=False):
        for i, src in enumerate(st.session_state.sources, 1):
            st.markdown(f"**{i}. {src.case_number}** — {src.section} ({src.year})")
            st.markdown(f"> {src.text[:400]}{'...' if len(src.text) > 400 else ''}")
            st.divider()
