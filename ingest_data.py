import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def ingest_data_and_get_vectorstore() -> Chroma:

    embedding_model = SentenceTransformerEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name = "embeddings",
        embedding_function = embedding_model,
        persist_directory = "./chroma_db"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800, 
        chunk_overlap = 100
    )

    for file_name in os.listdir("data"):
        existing = vectorstore._collection.get(where={"source": os.path.join("data", file_name)})
        if existing["ids"]:
            print(f"{file_name} already ingested. Skipping!")
            continue
        else:
            loader = PyPDFLoader(os.path.join("data", file_name))
            document = loader.load()
            print(f"Loaded {len(document)} pages")

            chunks = splitter.split_documents(document)
            print(f"Split into {len(chunks)} chunks")

            vectorstore.add_documents(chunks)
            print("Documents added to the vectorstore")   
            
    return vectorstore

