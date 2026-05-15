import pickle
from functools import lru_cache
from dotenv import load_dotenv
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from ingest_data import ingest_new_data

load_dotenv()

@lru_cache(maxsize=1)
def get_retriever() -> ContextualCompressionRetriever:
    vectorstore = ingest_new_data()
    ##semantic search
    vector_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 10}
    )

    ##keyword search
    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)

    ##hybrid search
    hybrid_retriever = EnsembleRetriever(
        retrievers = [vector_retriever, bm25_retriever],
        weights = [0.5, 0.5],
        c = 60
    )

    ##Reranking
    compressor = CohereRerank(
        model = "rerank-v4.0-pro",
        top_n = 4
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever = hybrid_retriever,
        base_compressor = compressor
    )

    return compression_retriever






