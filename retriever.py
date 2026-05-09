from dotenv import load_dotenv
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from ingest_data import ingest_data_and_get_vectorstore

load_dotenv()

def preprocess_text(text: str) -> list[str]:
    return text.lower().split()

def get_retriever() -> ContextualCompressionRetriever:
    vectorstore = ingest_data_and_get_vectorstore()
    ##semantic search
    vector_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 10}
    )

    raw_data = vectorstore._collection.get()
    docs = [
        Document(page_content = text, metadata = meta)
        for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]

    ##keyword search
    bm25_retriever = BM25Retriever.from_documents(
        documents = docs,
        bm25_params={"k1" : 1.2, "b" : 0.75},
        k = 10,
        preprocess_func = preprocess_text
    )

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






