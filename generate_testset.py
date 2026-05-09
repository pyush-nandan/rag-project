from dotenv import load_dotenv
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from ingest_data import ingest_data_and_get_vectorstore

load_dotenv()

llm = ChatGroq(
    model_name = "llama-3.1-8b-instant",
    temperature = 0.2,
    max_tokens = 1000,
    timeout = None,
    max_retries = 3
)

generator_llm = llm_factory(
    model_name= "llama-3.1-8b-instant", 
    client = llm
)

generator_embedding = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vectorstore = ingest_data_and_get_vectorstore()

chunks = vectorstore._collection.get()
docs = [
        Document(page_content = text, metadata = meta)
        for text, meta in zip(chunks["documents"], chunks["metadatas"])
    ]

generator = TestsetGenerator(llm = generator_llm, embedding_model = generator_embedding)

testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=120,
       
)

